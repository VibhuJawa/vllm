#!/usr/bin/env python3
"""Build a vLLM pooling benchmark JSONL from document images."""

from __future__ import annotations

import argparse
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--payload-mode",
        choices=("path", "jpeg-bytes"),
        default="jpeg-bytes",
    )
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--jpeg-subsampling", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def jpeg_data_uri(path: Path, *, quality: int, subsampling: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=subsampling,
            optimize=True,
        )
    payload = base64.b64encode(buffer.getbuffer()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    if args.jpeg_subsampling not in (0, 1, 2):
        raise ValueError("--jpeg-subsampling must be 0, 1, or 2")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    images = [
        path.resolve()
        for path in sorted(args.image_dir.expanduser().resolve().iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(images) < args.limit:
        raise RuntimeError(f"found {len(images)} images; need {args.limit}")

    selected = images[: args.limit]
    if args.payload_mode == "path":
        payloads = (str(image) for image in selected)
    else:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        payloads = executor.map(
            lambda path: jpeg_data_uri(
                path,
                quality=args.jpeg_quality,
                subsampling=args.jpeg_subsampling,
            ),
            selected,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for payload in payloads:
            output.write(
                json.dumps(
                    {"prompt": {"data": payload}, "output_tokens": 1},
                    separators=(",", ":"),
                )
                + "\n"
            )

    if args.payload_mode == "jpeg-bytes":
        executor.shutdown()

    print(
        f"wrote {args.limit} {args.payload_mode} pooling requests to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
