# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from transformers import PretrainedConfig


class NemotronOCRV2Config(PretrainedConfig):
    model_type = "nemotron_ocr_v2"

    def __init__(
        self,
        architectures: list[str] | None = None,
        nemotron_ocr_lang: str = "multi",
        nemotron_ocr_merge_level: str = "paragraph",
        nemotron_ocr_model_dir: str | None = None,
        nemotron_ocr_model_subdir: str | None = None,
        nemotron_ocr_repo_id: str | None = None,
        nemotron_ocr_include_invalid: bool = False,
        nemotron_ocr_detector_max_batch_size: int = 8,
        nemotron_ocr_recognizer_chunk_size: int = 128,
        nemotron_ocr_relational_chunk_size: int = 128,
        nemotron_ocr_infer_length: int | None = None,
        nemotron_ocr_verbose_post: bool = False,
        io_processor_plugin: str = "nemotron_ocr_v2",
        **kwargs: Any,
    ):
        if architectures is None:
            architectures = ["NemotronOCRV2ForImageToText"]

        super().__init__(architectures=architectures, **kwargs)

        self.nemotron_ocr_lang = nemotron_ocr_lang
        self.nemotron_ocr_merge_level = nemotron_ocr_merge_level
        self.nemotron_ocr_model_dir = nemotron_ocr_model_dir
        self.nemotron_ocr_model_subdir = nemotron_ocr_model_subdir
        self.nemotron_ocr_repo_id = nemotron_ocr_repo_id
        self.nemotron_ocr_include_invalid = nemotron_ocr_include_invalid
        self.nemotron_ocr_detector_max_batch_size = nemotron_ocr_detector_max_batch_size
        self.nemotron_ocr_recognizer_chunk_size = nemotron_ocr_recognizer_chunk_size
        self.nemotron_ocr_relational_chunk_size = nemotron_ocr_relational_chunk_size
        self.nemotron_ocr_infer_length = nemotron_ocr_infer_length
        self.nemotron_ocr_verbose_post = nemotron_ocr_verbose_post
        self.io_processor_plugin = io_processor_plugin
