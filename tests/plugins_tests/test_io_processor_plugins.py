# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import VllmConfig
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.pooling.io_processor import PluginWithIOProcessorPlugins
from vllm.entrypoints.pooling.typing import OfflineInputsContext, OfflineOutputsContext
from vllm.inputs import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer


class DummyIOProcessor(IOProcessor):
    """Minimal IOProcessor used as the target of the mocked plugin entry point."""

    def pre_process(
        self,
        prompt: object,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> object:
        raise NotImplementedError


@pytest.fixture
def my_plugin_entry_points():
    """Patch importlib.metadata.entry_points to expose a single 'my_plugin'
    entry point backed by DummyIOProcessor, exercising the full plugin-loading
    code path: entry_points → plugin.load() → func() →
    resolve_obj_by_qualname → IOProcessor.__init__."""
    qualname = f"{DummyIOProcessor.__module__}.{DummyIOProcessor.__qualname__}"
    ep = MagicMock()
    ep.name = "my_plugin"
    ep.value = qualname
    ep.load.return_value = lambda: qualname
    with patch("importlib.metadata.entry_points", return_value=[ep]):
        yield


def test_loading_missing_plugin():
    vllm_config = VllmConfig()
    renderer = MagicMock(spec=BaseRenderer)
    with pytest.raises(ValueError):
        get_io_processor(
            vllm_config, renderer=renderer, plugin_from_init="wrong_plugin"
        )


def test_loading_plugin(my_plugin_entry_points):
    # Plugin name supplied via plugin_from_init.
    vllm_config = MagicMock(spec=VllmConfig)
    renderer = MagicMock(spec=BaseRenderer)

    result = get_io_processor(
        vllm_config, renderer=renderer, plugin_from_init="my_plugin"
    )

    assert isinstance(result, DummyIOProcessor)


def test_loading_missing_plugin_from_model_config():
    # Build a mock VllmConfig whose hf_config advertises a plugin name,
    # exercising the model-config code path without loading a real model.
    mock_hf_config = MagicMock()
    mock_hf_config.to_dict.return_value = {"io_processor_plugin": "wrong_plugin"}

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config.hf_config = mock_hf_config

    renderer = MagicMock(spec=BaseRenderer)
    with pytest.raises(ValueError):
        get_io_processor(vllm_config, renderer=renderer)


def test_loading_plugin_from_model_config(my_plugin_entry_points):
    # Plugin name supplied via the model's hf_config.
    mock_hf_config = MagicMock()
    mock_hf_config.to_dict.return_value = {"io_processor_plugin": "my_plugin"}

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config.hf_config = mock_hf_config

    renderer = MagicMock(spec=BaseRenderer)

    result = get_io_processor(vllm_config, renderer=renderer)

    assert isinstance(result, DummyIOProcessor)


def test_offline_plugin_supports_batched_data_prompts(monkeypatch):
    class EchoIOProcessor:
        def parse_data(self, data):
            return data

        def pre_process(self, prompt, request_id=None, **kwargs):
            prompts = prompt if isinstance(prompt, list) else [prompt]
            return [{"prompt_token_ids": [1], "multi_modal_data": {"image": item}}
                    for item in prompts]

        def merge_pooling_params(self, params=None):
            params = params or PoolingParams()
            params.task = "plugin"
            return params

        def post_process(self, model_output, request_id=None, **kwargs):
            return [output.request_id for output in model_output]

    monkeypatch.setattr(
        PoolingIOProcessor,
        "pre_process_offline",
        lambda self, ctx: ctx.prompts,
    )

    processor = object.__new__(PluginWithIOProcessorPlugins)
    processor.io_processor = EchoIOProcessor()

    ctx = OfflineInputsContext(
        prompts=[{"data": "a"}, {"data": "b"}],
        pooling_params=PoolingParams(),
    )

    engine_inputs = processor.pre_process_offline(ctx)

    assert list(engine_inputs) == [
        {"prompt_token_ids": [1], "multi_modal_data": {"image": "a"}},
        {"prompt_token_ids": [1], "multi_modal_data": {"image": "b"}},
    ]
    assert ctx.plugin_output_sizes == [1, 1]

    outputs = [
        PoolingRequestOutput(
            request_id="0",
            outputs=MagicMock(),
            prompt_token_ids=[],
            num_cached_tokens=0,
            finished=True,
        ),
        PoolingRequestOutput(
            request_id="1",
            outputs=MagicMock(),
            prompt_token_ids=[],
            num_cached_tokens=0,
            finished=True,
        ),
    ]
    processed = processor.post_process_offline(
        OfflineOutputsContext(
            outputs=outputs,
            plugin_output_sizes=ctx.plugin_output_sizes,
        )
    )

    assert [item.outputs for item in processed] == [["0"], ["1"]]


def test_offline_plugin_preserves_single_prompt_expansion(monkeypatch):
    class EchoIOProcessor:
        def parse_data(self, data):
            return data

        def pre_process(self, prompt, request_id=None, **kwargs):
            return [{"prompt_token_ids": [1], "multi_modal_data": {"image": item}}
                    for item in prompt]

        def merge_pooling_params(self, params=None):
            params = params or PoolingParams()
            params.task = "plugin"
            return params

        def post_process(self, model_output, request_id=None, **kwargs):
            return [output.request_id for output in model_output]

    monkeypatch.setattr(
        PoolingIOProcessor,
        "pre_process_offline",
        lambda self, ctx: ctx.prompts,
    )

    processor = object.__new__(PluginWithIOProcessorPlugins)
    processor.io_processor = EchoIOProcessor()

    ctx = OfflineInputsContext(
        prompts={"data": ["a", "b"]},
        pooling_params=PoolingParams(),
    )

    processor.pre_process_offline(ctx)

    assert ctx.plugin_output_sizes == [2]

    outputs = [
        PoolingRequestOutput(
            request_id="0",
            outputs=MagicMock(),
            prompt_token_ids=[],
            num_cached_tokens=0,
            finished=True,
        ),
        PoolingRequestOutput(
            request_id="1",
            outputs=MagicMock(),
            prompt_token_ids=[],
            num_cached_tokens=0,
            finished=True,
        ),
    ]
    processed = processor.post_process_offline(
        OfflineOutputsContext(
            outputs=outputs,
            plugin_output_sizes=ctx.plugin_output_sizes,
        )
    )

    assert len(processed) == 1
    assert processed[0].outputs == ["0", "1"]
