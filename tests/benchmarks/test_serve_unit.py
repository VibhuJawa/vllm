# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import time
from pathlib import Path

import pytest

from vllm.benchmarks.datasets import SampleRequest, get_samples
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncOutput,
)
from vllm.benchmarks.serve import (
    TaskType,
    _make_request_func_input,
    _partition_setup_requests,
    _validate_tokenizerless_benchmark,
    benchmark,
)


def _sample(name: str, **kwargs) -> SampleRequest:
    return SampleRequest(
        prompt={"data": name},
        prompt_len=1,
        expected_output_len=1,
        request_id=name,
        **kwargs,
    )


def test_make_request_func_input_preserves_sample_fields():
    messages = [{"role": "user", "content": "hello"}]
    request = _sample(
        "sample-id",
        multi_modal_data={"image": "image-value"},
        request_overrides={"shared": "sample", "sample_only": 2},
        chat_messages=messages,
    )

    request_input = _make_request_func_input(
        request,
        model_id="model-id",
        model_name="served-name",
        api_url="http://localhost/pooling",
        logprobs=None,
        ignore_eos=False,
        extra_headers={"header": "value"},
        extra_body={"shared": "default", "default_only": 1},
    )

    assert request_input.prompt == {"data": "sample-id"}
    assert request_input.request_id == "sample-id"
    assert request_input.multi_modal_content == {"image": "image-value"}
    assert request_input.chat_messages is messages
    assert request_input.extra_body == {
        "shared": "sample",
        "default_only": 1,
        "sample_only": 2,
    }


def test_partition_setup_requests_holds_them_out_of_timed_workload():
    requests = [_sample(str(index)) for index in range(5)]

    setup, timed = _partition_setup_requests(
        requests,
        num_setup_requests=2,
        num_timed_requests=3,
    )

    assert setup == requests[3:]
    assert timed == requests[:3]
    assert not set(request.request_id for request in setup or []).intersection(
        request.request_id for request in timed
    )


@pytest.mark.parametrize(
    ("dataset_name", "backend"),
    [
        ("custom", "openai-embeddings"),
        ("custom_image", "vllm-pooling"),
    ],
)
def test_tokenizerless_benchmark_is_limited_to_native_custom_pooling(
    dataset_name,
    backend,
):
    with pytest.raises(ValueError, match="custom.*vllm-pooling"):
        _validate_tokenizerless_benchmark(
            None,
            dataset_name=dataset_name,
            backend=backend,
        )

    _validate_tokenizerless_benchmark(
        None,
        dataset_name="custom",
        backend="vllm-pooling",
    )


def test_custom_pooling_dataset_can_skip_tokenizer(tmp_path: Path):
    dataset_path = tmp_path / "pooling.jsonl"
    prompt = {"data": "data:image/jpeg;base64,AA=="}
    dataset_path.write_text(
        json.dumps({"prompt": prompt, "output_tokens": 1}) + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        dataset_name="custom",
        dataset_path=str(dataset_path),
        disable_shuffle=True,
        num_prompts=1,
        custom_output_len=None,
        skip_chat_template=True,
        chat_template_kwargs=None,
        no_oversample=False,
        seed=0,
        request_id_prefix="bench-",
    )

    samples = get_samples(args, None)

    assert samples == [
        SampleRequest(
            prompt=prompt,
            prompt_len=1,
            expected_output_len=1,
            request_id="bench-0",
        )
    ]


def test_pooling_benchmark_reports_failures_and_uses_held_out_warmups(
    monkeypatch,
):
    seen_inputs = []

    async def fake_request(request_func_input, session, pbar=None):
        seen_inputs.append(request_func_input)
        success = request_func_input.prompt != {"data": "timed-failure"}
        return RequestFuncOutput(
            success=success,
            error="expected failure" if not success else "",
            latency=0.001,
            prompt_len=request_func_input.prompt_len,
            start_time=time.perf_counter(),
        )

    async def no_server_metrics(base_url, session):
        return None

    monkeypatch.setitem(ASYNC_REQUEST_FUNCS, "vllm-pooling", fake_request)
    monkeypatch.setattr(
        "vllm.benchmarks.serve.fetch_spec_decode_metrics", no_server_metrics
    )
    monkeypatch.setattr(
        "vllm.benchmarks.serve.fetch_diffusion_metrics", no_server_metrics
    )

    result = asyncio.run(
        benchmark(
            task_type=TaskType.POOLING,
            endpoint_type="vllm-pooling",
            api_url="http://localhost/pooling",
            base_url="http://localhost",
            model_id="model-id",
            model_name="served-name",
            tokenizer=None,
            input_requests=[_sample("timed-success"), _sample("timed-failure")],
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            num_warmups=2,
            profile=False,
            selected_percentile_metrics=["e2el"],
            selected_percentiles=[99.0],
            ignore_eos=False,
            goodput_config_dict={},
            max_concurrency=2,
            lora_modules=None,
            extra_headers=None,
            extra_body={"default": True},
            ready_check_timeout_sec=0,
            warmup_input_requests=[
                _sample("warmup-one", request_overrides={"warmup": 1}),
                _sample("warmup-two", request_overrides={"warmup": 2}),
            ],
        )
    )

    assert result["completed"] == 1
    assert result["failed"] == 1
    assert [request.request_id for request in seen_inputs] == [
        "warmup-one",
        "warmup-two",
        "timed-success",
        "timed-failure",
    ]
    assert seen_inputs[0].extra_body == {"default": True, "warmup": 1}
    assert seen_inputs[1].extra_body == {"default": True, "warmup": 2}
