# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch
import torch.nn as nn
from PIL import Image
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces import IsAttentionFree, MultiModalEmbeddings, SupportsMultiModal
from .interfaces_base import attn_type

_CHECKPOINT_FILES = (
    "detector.pth",
    "recognizer.pth",
    "relational.pth",
    "charset.txt",
    "model_config.json",
)
_MAX_OUTPUT_BYTES = 1024 * 1024


def _enable_ocr_profile_logging() -> None:
    logger = logging.getLogger("nemotron_ocr.inference.pipeline_v2")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = False


def _json_to_bytes(payload: Any) -> bytes:
    raw = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
    if len(raw) > _MAX_OUTPUT_BYTES:
        raise ValueError(
            f"Nemotron OCR payload is {len(raw)} bytes; "
            f"max supported is {_MAX_OUTPUT_BYTES}."
        )
    return len(raw).to_bytes(4, "little") + raw


def _json_to_tensor(payload: Any, *, device: torch.device) -> torch.Tensor:
    encoded = _json_to_bytes(payload)

    host_tensor = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
    return host_tensor.to(device=device)


def tensor_to_json(data: torch.Tensor) -> Any:
    flat = (
        data.detach()
        .to("cpu", dtype=torch.uint8)
        .contiguous()
        .flatten()
        .numpy()
        .tobytes()
    )
    if len(flat) < 4:
        raise ValueError("Nemotron OCR output tensor is too short.")

    size = int.from_bytes(flat[:4], "little")
    raw = flat[4 : 4 + size]
    return orjson.loads(raw)


def _image_to_chw_uint8(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).expand(3, -1, -1)
        elif tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3]
            if tensor.shape[0] == 1:
                tensor = tensor.expand(3, -1, -1)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
            if tensor.shape[0] == 1:
                tensor = tensor.expand(3, -1, -1)
        else:
            raise ValueError(f"Unsupported image tensor shape: {tuple(image.shape)}")

        if tensor.dtype != torch.uint8:
            if tensor.is_floating_point():
                tensor = tensor.clamp(0, 1).mul(255)
            tensor = tensor.clamp(0, 255).to(torch.uint8)
        return tensor.contiguous()

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unsupported image array shape: {array.shape}")
        array = array[..., :3]
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, 0, 1) * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        array = np.ascontiguousarray(array)
        if not array.flags.writeable:
            array = array.copy()
        return torch.from_numpy(array).permute(2, 0, 1)

    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), copy=True)
        return torch.from_numpy(array).permute(2, 0, 1)

    if isinstance(image, (bytes, bytearray)):
        with Image.open(BytesIO(image)) as pil_image:
            return _image_to_chw_uint8(pil_image)

    raise TypeError(f"Unsupported Nemotron OCR image type: {type(image).__name__}")


def _images_from_mm_kwargs(ocr_images: object) -> list[torch.Tensor]:
    if ocr_images is None:
        return []

    if isinstance(ocr_images, torch.Tensor):
        if ocr_images.ndim == 4:
            return [_image_to_chw_uint8(image) for image in ocr_images]
        return [_image_to_chw_uint8(ocr_images)]

    if isinstance(ocr_images, Sequence) and not isinstance(
        ocr_images, (str, bytes, bytearray)
    ):
        return [_image_to_chw_uint8(image) for image in ocr_images]

    return [_image_to_chw_uint8(ocr_images)]


def _complete_checkpoint_dir(path: Path) -> bool:
    for filename in _CHECKPOINT_FILES:
        checkpoint = path / filename
        if not checkpoint.is_file():
            return False
        if checkpoint.suffix == ".pth" and checkpoint.stat().st_size < 1024:
            header = checkpoint.read_bytes()[:64]
            if header.startswith(b"version https://git-lfs.github.com"):
                return False
    return True


class NemotronOCRV2ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class NemotronOCRV2DummyInputsBuilder(
    BaseDummyInputsBuilder[NemotronOCRV2ProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        image_count = mm_counts.get("image", 1)
        image_count = max(image_count, 1)
        return {
            "image": [
                Image.new("RGB", (64, 64), color="white") for _ in range(image_count)
            ]
        }


class NemotronOCRV2MultiModalProcessor(
    BaseMultiModalProcessor[NemotronOCRV2ProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
        *,
        is_shared: bool = True,
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {"ocr_images": MultiModalFieldConfig.batched("image", keep_on_cpu=True)}

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        mm_items = inputs.mm_data_items
        images = mm_items["image"].get_all() if "image" in mm_items else []

        with timing_ctx.record("apply_hf_processor"):
            image_tensors = [_image_to_chw_uint8(image) for image in images]
            mm_processed_data = BatchFeature(
                {"ocr_images": image_tensors},
                tensor_type=None,
            )

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_processed_data,
            self._get_mm_fields_config(
                mm_processed_data,
                inputs.hf_processor_mm_kwargs,
                is_shared=False,
            ),
        )

        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        mm_placeholders = {
            "image": [
                PlaceholderRange(offset=0, length=0) for _ in range(len(image_tensors))
            ]
        }

        return mm_input(
            prompt_token_ids=[1],
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


class NemotronOCRV2PayloadPooler(Pooler):
    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        num_requests = len(pooling_metadata.prompt_lens)
        if hidden_states.dtype == torch.uint8 and hidden_states.ndim == 2:
            if hidden_states.shape[0] >= num_requests:
                return [hidden_states[index] for index in range(num_requests)]
            if hidden_states.shape[0] == 1:
                return [hidden_states[0] for _ in range(num_requests)]

        return [
            torch.zeros(4, dtype=torch.uint8, device=hidden_states.device)
            for _ in range(num_requests)
        ]


@attn_type("attention_free")
@MULTIMODAL_REGISTRY.register_processor(
    NemotronOCRV2MultiModalProcessor,
    info=NemotronOCRV2ProcessingInfo,
    dummy_inputs=NemotronOCRV2DummyInputsBuilder,
)
class NemotronOCRV2ForImageToText(nn.Module, IsAttentionFree, SupportsMultiModal):
    supports_multimodal_raw_input_only = True
    is_pooling_model = True
    allow_patterns_overrides = ["__vllm_no_primary_weights__"]

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Nemotron OCR v2 only supports image inputs.")

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.vllm_config = vllm_config
        self.pooler = NemotronOCRV2PayloadPooler()
        self._ocr = None

        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        self.model_name_or_path = model_config.model
        self.revision = model_config.revision

        self.lang = getattr(hf_config, "nemotron_ocr_lang", "multi")
        self.merge_level = getattr(hf_config, "nemotron_ocr_merge_level", "paragraph")
        self.model_dir = getattr(hf_config, "nemotron_ocr_model_dir", None)
        self.model_subdir = getattr(hf_config, "nemotron_ocr_model_subdir", None)
        self.repo_id = getattr(hf_config, "nemotron_ocr_repo_id", None)
        self.include_invalid = getattr(hf_config, "nemotron_ocr_include_invalid", False)
        self.detector_max_batch_size = getattr(
            hf_config, "nemotron_ocr_detector_max_batch_size", 8
        )
        self.recognizer_chunk_size = getattr(
            hf_config, "nemotron_ocr_recognizer_chunk_size", 128
        )
        self.relational_chunk_size = getattr(
            hf_config, "nemotron_ocr_relational_chunk_size", 128
        )
        self.infer_length = getattr(hf_config, "nemotron_ocr_infer_length", None)
        self.verbose_post = getattr(hf_config, "nemotron_ocr_verbose_post", False)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty((input_ids.shape[0], 0), device=input_ids.device)

    def get_language_model(self) -> nn.Module:
        return self

    def _resolve_relative_model_dir(self, value: str) -> str | None:
        path = Path(value).expanduser()
        if path.is_dir() and _complete_checkpoint_dir(path):
            return str(path)

        base = Path(str(self.model_name_or_path)).expanduser()
        candidate = base / value
        if candidate.is_dir() and _complete_checkpoint_dir(candidate):
            return str(candidate)

        return None

    def _download_model_subdir(self, subdir: str) -> str:
        if local_dir := self._resolve_relative_model_dir(subdir):
            return local_dir

        from huggingface_hub import hf_hub_download

        repo_id = self.repo_id or str(self.model_name_or_path)
        downloaded_path = None
        for filename in _CHECKPOINT_FILES:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{subdir.rstrip('/')}/{filename}",
                revision=self.revision,
            )

        assert downloaded_path is not None
        return str(Path(downloaded_path).parent)

    def _get_model_dir_for_pipeline(self) -> str | None:
        if self.model_dir:
            resolved = self._resolve_relative_model_dir(self.model_dir)
            return resolved or self.model_dir

        if self.model_subdir:
            return self._download_model_subdir(self.model_subdir)

        return None

    def _load_ocr(self):
        if self._ocr is not None:
            return self._ocr

        source_dir = os.environ.get("NEMOTRON_OCR_SOURCE")
        if source_dir and source_dir not in sys.path:
            sys.path.insert(0, source_dir)

        try:
            from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
        except ImportError as exc:
            raise ImportError(
                "Nemotron OCR v2 support requires the optional "
                "`nemotron_ocr` package and its CUDA extension. Install the "
                "package from the nvidia/nemotron-ocr-v2 repository, or set "
                "`NEMOTRON_OCR_SOURCE` to its `nemotron-ocr/src` directory."
            ) from exc

        if self.verbose_post:
            _enable_ocr_profile_logging()

        kwargs: dict[str, Any] = {
            "detector_max_batch_size": self.detector_max_batch_size,
            "recognizer_chunk_size": self.recognizer_chunk_size,
            "relational_chunk_size": self.relational_chunk_size,
            "verbose_post": self.verbose_post,
        }
        if self.infer_length is not None:
            kwargs["infer_length"] = self.infer_length

        model_dir = self._get_model_dir_for_pipeline()
        if model_dir is not None:
            kwargs["model_dir"] = model_dir
        else:
            kwargs["lang"] = self.lang

        self._ocr = NemotronOCRV2(**kwargs)
        return self._ocr

    def _payload_from_predictions(
        self,
        image_index: int,
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "backend": "vllm",
            "model": "nemotron-ocr-v2",
            "image_index": image_index,
            "merge_level": self.merge_level,
            "regions": predictions,
        }

    def _run_ocr_batch(
        self,
        images: list[torch.Tensor],
    ) -> list[dict[str, Any]]:
        if not images:
            return []

        ocr = self._load_ocr()
        raw_predictions = ocr(
            images,
            merge_level=self.merge_level,
            include_invalid=self.include_invalid,
        )

        return [
            self._payload_from_predictions(index, predictions)
            for index, predictions in enumerate(raw_predictions)
        ]

    def _encode_payloads(
        self,
        payloads: list[dict[str, Any]],
        device: torch.device,
    ) -> torch.Tensor:
        if not payloads:
            payloads = [
                {
                    "backend": "vllm",
                    "model": "nemotron-ocr-v2",
                    "regions": [],
                }
            ]

        encoded = [_json_to_bytes(payload) for payload in payloads]

        max_len = max(map(len, encoded))
        output = np.zeros((len(encoded), max_len), dtype=np.uint8)
        for index, row in enumerate(encoded):
            output[index, : len(row)] = np.frombuffer(row, dtype=np.uint8)
        return torch.from_numpy(output).to(device=device)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        images = _images_from_mm_kwargs(kwargs.get("ocr_images"))
        if not images:
            raise ValueError(
                "Nemotron OCR v2 requires image input via multi_modal_data."
            )

        device = positions.device
        if input_ids is not None:
            device = input_ids.device

        payloads = self._run_ocr_batch(images)
        return self._encode_payloads(payloads, device)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return set()
