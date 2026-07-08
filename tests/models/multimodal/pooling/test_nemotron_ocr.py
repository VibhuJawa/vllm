# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from vllm.config.model import ModelConfig
from vllm.model_executor.models.nemotron_ocr import (
    NemotronOCRV2ForImageToText,
    _image_to_chw_uint8,
    _json_to_tensor,
    tensor_to_json,
)
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.plugins.io_processors.nemotron_ocr import NemotronOCRV2IOProcessor
from vllm.renderers import BaseRenderer
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.nemotron_ocr import NemotronOCRV2Config


def test_empty_config_can_use_nemotron_ocr_hf_overrides(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    config = get_config(
        tmp_path,
        trust_remote_code=False,
        hf_overrides_kw={"model_type": "nemotron_ocr_v2"},
    )

    assert config.model_type == "nemotron_ocr_v2"
    assert config.architectures == ["NemotronOCRV2ForImageToText"]
    assert config.io_processor_plugin == "nemotron_ocr_v2"


def test_nemotron_ocr_config_can_carry_model_fork_subdir(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_ocr_v2",
                "nemotron_ocr_model_subdir": "v2_multilingual",
                "nemotron_ocr_lang": "multi",
            }
        ),
        encoding="utf-8",
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert config.nemotron_ocr_model_subdir == "v2_multilingual"
    assert config.architectures == ["NemotronOCRV2ForImageToText"]


def test_nemotron_ocr_generation_config_falls_back_to_loaded_config(monkeypatch):
    def raise_value_error(*args, **kwargs):
        raise ValueError("empty upstream config")

    monkeypatch.setattr(
        "vllm.config.model.try_get_generation_config", raise_value_error
    )

    model_config = SimpleNamespace(
        generation_config="auto",
        hf_config_path=None,
        model="nvidia/nemotron-ocr-v2",
        trust_remote_code=False,
        revision=None,
        config_format="auto",
        hf_token=None,
        hf_config=NemotronOCRV2Config(),
    )

    config = ModelConfig.try_get_generation_config(model_config)

    assert config["_from_model_config"] is True


def test_image_to_chw_uint8_accepts_common_image_types():
    pil_image = Image.new("RGB", (5, 3), color="white")
    np_image = np.ones((3, 5, 3), dtype=np.float32)
    tensor_image = torch.ones(3, 3, 5)

    for image in (pil_image, np_image, tensor_image):
        tensor = _image_to_chw_uint8(image)
        assert tensor.shape == (3, 3, 5)
        assert tensor.dtype == torch.uint8


def test_nemotron_ocr_payload_codec_round_trip():
    payload = {
        "backend": "vllm",
        "regions": [{"text": "hello", "confidence": 0.99}],
    }

    tensor = _json_to_tensor(payload, device=torch.device("cpu"))

    assert tensor_to_json(tensor) == payload


def test_nemotron_ocr_model_loader_does_not_consume_hf_weight_iterator():
    hf_config = NemotronOCRV2Config()
    model_config = SimpleNamespace(
        model="nvidia/nemotron-ocr-v2",
        revision=None,
        hf_config=hf_config,
    )
    model = NemotronOCRV2ForImageToText(
        SimpleNamespace(model_config=model_config),
    )

    def raise_if_iterated():
        raise AssertionError("Nemotron OCR should not consume top-level HF weights")
        yield

    assert model.load_weights(raise_if_iterated()) == set()
    assert model.get_language_model() is model


def test_nemotron_ocr_io_processor_is_builtin_without_entry_point():
    vllm_config = MagicMock()
    renderer = MagicMock(spec=BaseRenderer)

    with patch("importlib.metadata.entry_points", return_value=[]):
        processor = get_io_processor(
            vllm_config,
            renderer=renderer,
            plugin_from_init="nemotron_ocr_v2",
        )

    assert isinstance(processor, NemotronOCRV2IOProcessor)


def test_nemotron_ocr_io_processor_accepts_image_url_data_uri():
    image = Image.new("RGB", (4, 3), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode()
    processor = NemotronOCRV2IOProcessor(vllm_config=None, renderer=None)

    parsed = processor.parse_data({"image_url": {"url": image_url}})

    assert isinstance(parsed, Image.Image)
    assert parsed.size == (4, 3)


def test_nemotron_ocr_io_processor_round_trip(tmp_path: Path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 4), color="white").save(image_path)
    processor = NemotronOCRV2IOProcessor(vllm_config=None, renderer=None)

    parsed = processor.parse_data(str(image_path))
    prompt = processor.pre_process(parsed)

    assert prompt["prompt_token_ids"] == [1]
    assert "image" in prompt["multi_modal_data"]

    payload = {"regions": [{"text": "hello"}]}
    output = PoolingRequestOutput(
        request_id="0",
        outputs=PoolingOutput(_json_to_tensor(payload, device=torch.device("cpu"))),
        prompt_token_ids=[],
        num_cached_tokens=0,
        finished=True,
    )

    assert processor.post_process([output]) == payload
