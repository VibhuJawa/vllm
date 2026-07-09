# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm import envs
from vllm.config import VllmConfig
from vllm.model_executor.models.nemotron_ocr import tensor_to_json
from vllm.multimodal.media import MediaConnector, MediaWithBytes
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer

OCRInput = Image.Image | np.ndarray | torch.Tensor | bytes | str | Path
_URL_PREFIXES = ("data:", "http://", "https://", "file://")
_BASE64_JPEG_PREFIXES = (
    "data:image/jpeg;base64,",
    "data:image/jpg;base64,",
)


def _decode_image_bytes(data: bytes) -> np.ndarray | Image.Image:
    """Decode wire images without making vLLM treat pixels as embeddings."""
    from torchvision.io import ImageReadMode, decode_image

    try:
        encoded = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        tensor = decode_image(encoded, mode=ImageReadMode.RGB)
    except (RuntimeError, ValueError):
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")

    if (
        envs.VLLM_MAX_IMAGE_PIXELS > 0
        and tensor.shape[-2] * tensor.shape[-1] > envs.VLLM_MAX_IMAGE_PIXELS
    ):
        raise ValueError(
            f"Image has {tensor.shape[-2] * tensor.shape[-1]} pixels, which "
            f"exceeds the limit of {envs.VLLM_MAX_IMAGE_PIXELS}. Set "
            "VLLM_MAX_IMAGE_PIXELS to increase this limit."
        )
    # The HTTP multimodal parser reserves torch.Tensor for precomputed image
    # embeddings. Return HWC NumPy pixels so this remains ordinary image data.
    return tensor.permute(1, 2, 0).contiguous().numpy()


class NemotronOCRV2IOProcessor(IOProcessor[OCRInput | list[OCRInput], Any]):
    def __init__(
        self,
        vllm_config: VllmConfig | None,
        renderer: BaseRenderer | None,
    ):
        super().__init__(vllm_config, renderer)

        model_config = getattr(vllm_config, "model_config", None)
        mm_config = getattr(model_config, "multimodal_config", None)
        media_io_kwargs = getattr(mm_config, "media_io_kwargs", None)
        if not isinstance(media_io_kwargs, dict):
            media_io_kwargs = None
        allowed_local_media_path = getattr(
            model_config, "allowed_local_media_path", ""
        )
        if not isinstance(allowed_local_media_path, str):
            allowed_local_media_path = ""
        allowed_media_domains = getattr(model_config, "allowed_media_domains", None)
        if not isinstance(allowed_media_domains, list):
            allowed_media_domains = None

        self.media_connector = MediaConnector(
            media_io_kwargs=media_io_kwargs,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
        )

    def parse_data(self, data: object) -> OCRInput | list[OCRInput]:
        if isinstance(data, Mapping):
            if "image" in data:
                return self.parse_data(data["image"])
            if "images" in data:
                return self.parse_data(data["images"])
            if "image_url" in data:
                image_url = data["image_url"]
                if isinstance(image_url, Mapping):
                    image_url = image_url.get("url")
                return self.parse_data(image_url)
            if "url" in data:
                return self.parse_data(data["url"])
            raise TypeError(
                "Nemotron OCR request dictionaries must contain `image`, "
                "`images`, or `image_url`."
            )

        if isinstance(data, Sequence) and not isinstance(
            data, (str, bytes, bytearray, np.ndarray, torch.Tensor)
        ):
            return [self._parse_one(item) for item in data]

        return self._parse_one(data)

    def _parse_one(self, data: object) -> OCRInput:
        if isinstance(data, Image.Image):
            return data.convert("RGB")
        if isinstance(data, np.ndarray | torch.Tensor):
            return data
        if isinstance(data, (bytes, bytearray)):
            return _decode_image_bytes(bytes(data))
        if isinstance(data, (str, Path)):
            data_str = str(data)
            data_str_lower = data_str.lower()
            if data_str_lower.startswith(_BASE64_JPEG_PREFIXES):
                try:
                    encoded = data_str.split(",", 1)[1]
                    return _decode_image_bytes(
                        base64.b64decode(encoded, validate=True)
                    )
                except (IndexError, ValueError, binascii.Error) as exc:
                    raise ValueError("Invalid base64 JPEG data URI.") from exc
            if data_str_lower.startswith(_URL_PREFIXES):
                fetched = self.media_connector.fetch_image(data_str)
                if isinstance(fetched, MediaWithBytes):
                    fetched = fetched.media
                return fetched.convert("RGB")

            path = Path(data_str).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Nemotron OCR image path does not exist: {path}"
                )
            # torchvision decodes directly to the CHW uint8 tensor consumed by
            # the multimodal processor. Keep PIL as a fallback because
            # torchvision does not support every format accepted by Pillow
            # (notably TIFF and BMP in common builds).
            from torchvision.io import ImageReadMode, read_image

            try:
                tensor = read_image(str(path), mode=ImageReadMode.RGB)
                return tensor.permute(1, 2, 0).contiguous().numpy()
            except (RuntimeError, ValueError):
                with Image.open(path) as image:
                    return image.convert("RGB")

        raise TypeError(
            "Nemotron OCR data must be an image, image path, bytes, "
            "or a list of those values."
        )

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        merged = params or PoolingParams()
        merged.task = "plugin"
        return merged

    def pre_process(
        self,
        prompt: OCRInput | list[OCRInput],
        request_id: str | None = None,
        **kwargs,
    ):
        images = prompt if isinstance(prompt, list) else [prompt]
        prompts = [
            {
                "prompt": "",
                "prompt_token_ids": [1],
                "multi_modal_data": {"image": image},
            }
            for image in images
        ]
        return prompts[0] if len(prompts) == 1 else prompts

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> Any:
        payloads = []
        for output_item in model_output:
            output = output_item.outputs
            if not isinstance(output, PoolingOutput):
                raise TypeError(f"Expected PoolingOutput, got {type(output).__name__}.")
            payloads.append(tensor_to_json(output.data))

        return payloads[0] if len(payloads) == 1 else payloads


def get_io_processor_class() -> str:
    return "vllm.plugins.io_processors.nemotron_ocr.NemotronOCRV2IOProcessor"
