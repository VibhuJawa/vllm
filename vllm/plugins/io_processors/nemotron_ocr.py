# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm.model_executor.models.nemotron_ocr import tensor_to_json
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams

OCRInput = Image.Image | np.ndarray | torch.Tensor | bytes | str | Path


class NemotronOCRV2IOProcessor(IOProcessor[OCRInput | list[OCRInput], Any]):
    def parse_data(self, data: object) -> OCRInput | list[OCRInput]:
        if isinstance(data, Mapping):
            if "image" in data:
                return self.parse_data(data["image"])
            if "images" in data:
                return self.parse_data(data["images"])
            raise TypeError(
                "Nemotron OCR request dictionaries must contain `image` or `images`."
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
            with Image.open(BytesIO(data)) as image:
                return image.convert("RGB")
        if isinstance(data, (str, Path)):
            path = Path(data).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Nemotron OCR image path does not exist: {path}"
                )
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
