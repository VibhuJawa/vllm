# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import base64
import json
import mimetypes
from pathlib import Path

import requests


def image_to_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--endpoint", default="http://localhost:8000/pooling")
    parser.add_argument("--model", default="nvidia/nemotron-ocr-v2")
    args = parser.parse_args()

    response = requests.post(
        args.endpoint,
        json={
            "model": args.model,
            "data": {
                "image_url": {
                    "url": image_to_data_url(args.image.expanduser().resolve())
                }
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
