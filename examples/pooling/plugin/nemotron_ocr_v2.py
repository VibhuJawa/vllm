# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm import LLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default="nvidia/nemotron-ocr-v2")
    parser.add_argument("--model-subdir", default="v2_multilingual")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        skip_tokenizer_init=True,
        enforce_eager=True,
        io_processor_plugin="nemotron_ocr_v2",
        hf_overrides={
            "model_type": "nemotron_ocr_v2",
            "architectures": ["NemotronOCRV2ForImageToText"],
            "nemotron_ocr_model_subdir": args.model_subdir,
            "io_processor_plugin": "nemotron_ocr_v2",
        },
    )

    output = llm.encode({"data": args.image}, pooling_task="plugin")[0].outputs
    print(output)


if __name__ == "__main__":
    main()
