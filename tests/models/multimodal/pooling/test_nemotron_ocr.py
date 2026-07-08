# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vllm.model_executor.models.nemotron_ocr import (
    _image_to_chw_uint8,
    _json_to_tensor,
    tensor_to_json,
)
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.plugins.io_processors.nemotron_ocr import NemotronOCRV2IOProcessor
from vllm.transformers_utils.config import get_config


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
