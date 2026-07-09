#!/usr/bin/env python3
"""Benchmark the official NVIDIA Nemotron OCR v2 pipeline in-process.

This deliberately bypasses vLLM and its model wrapper. It imports
``NemotronOCRV2`` directly from a clean model-repository checkout and feeds it
the same JPEG payloads used by the vLLM pooling benchmark.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

JPEG_SUFFIXES = {".jpeg", ".jpg"}
GPU_TRACE_QUERY = (
    "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate"
)


@dataclass(frozen=True)
class BenchmarkImage:
    """A compressed JPEG payload ready for the official pipeline loader."""

    source: str
    payload: bytes
    representation: Literal["base64", "raw"]
    decoded_nbytes: int

    def pipeline_input(self) -> bytes | io.BytesIO:
        # The official pipeline treats bytes as base64 text and file-like
        # objects as raw compressed image bytes.
        if self.representation == "base64":
            return self.payload
        return io.BytesIO(self.payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the official NVIDIA Nemotron OCR v2 PyTorch pipeline "
            "without vLLM."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-jsonl",
        type=Path,
        help=(
            "vLLM custom-dataset JSONL containing prompt.data JPEG data URIs "
            "or JPEG paths."
        ),
    )
    source.add_argument(
        "--jpeg-dir",
        type=Path,
        help="Directory of JPEG files; compressed bytes are preloaded before timing.",
    )

    parser.add_argument(
        "--model-repo",
        type=Path,
        required=True,
        help="Path to the nvidia/nemotron-ocr-v2 model repository checkout.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Checkpoint directory; defaults to MODEL_REPO/v2_multilingual.",
    )
    parser.add_argument(
        "--allow-dirty-model-repo",
        action="store_true",
        help=(
            "Permit a modified model source tree instead of requiring a clean "
            "checkout."
        ),
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--replay-count", type=int, default=1)
    parser.add_argument(
        "--merge-level",
        choices=("word", "sentence", "paragraph"),
        default="paragraph",
    )
    parser.add_argument("--infer-length", type=int, default=1024)
    parser.add_argument("--detector-max-batch-size", type=int, default=8)
    parser.add_argument("--recognizer-chunk-size", type=int, default=128)
    parser.add_argument("--relational-chunk-size", type=int, default=128)
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include low-confidence regions (off in the accuracy baseline).",
    )
    parser.add_argument(
        "--profile-ocr",
        action="store_true",
        help="Enable CUDA-synchronized pipeline phase logs; distorts throughput.",
    )
    parser.add_argument("--predictions-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--gpu-trace-csv", type=Path)
    parser.add_argument(
        "--gpu-trace-device",
        help="Physical nvidia-smi GPU index or UUID for an unambiguous trace.",
    )
    parser.add_argument("--gpu-trace-interval-ms", type=int, default=250)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.replay_count <= 0:
        raise ValueError("--replay-count must be positive")
    if args.infer_length <= 0:
        raise ValueError("--infer-length must be positive")
    if min(
        args.detector_max_batch_size,
        args.recognizer_chunk_size,
        args.relational_chunk_size,
    ) <= 0:
        raise ValueError("OCR batch/chunk sizes must be positive")
    if args.gpu_trace_interval_ms <= 0:
        raise ValueError("--gpu-trace-interval-ms must be positive")


def decoded_base64_size(payload: bytes) -> int:
    padding = len(payload) - len(payload.rstrip(b"="))
    return len(payload) * 3 // 4 - padding


def check_jpeg_magic(payload: bytes, *, source: str) -> None:
    if not payload.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"Expected JPEG bytes from {source}")


def make_base64_image(payload: str, *, source: str) -> BenchmarkImage:
    encoded = payload.strip().encode("ascii")
    if len(encoded) % 4:
        raise ValueError(f"Invalid base64 JPEG payload length from {source}")
    # Decode only a short, complete base64 prefix to validate the media type;
    # the full base64 decode remains inside the timed official pipeline path.
    prefix_length = min(len(encoded) - len(encoded) % 4, 64)
    if prefix_length < 4:
        raise ValueError(f"Invalid base64 JPEG payload from {source}")
    try:
        decoded_prefix = base64.b64decode(encoded[:prefix_length], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(f"Invalid base64 JPEG payload from {source}") from exc
    check_jpeg_magic(decoded_prefix, source=source)
    return BenchmarkImage(
        source=source,
        payload=encoded,
        representation="base64",
        decoded_nbytes=decoded_base64_size(encoded),
    )


def make_raw_image(path: Path, *, source: str) -> BenchmarkImage:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"JPEG path does not exist: {resolved}")
    payload = resolved.read_bytes()
    check_jpeg_magic(payload, source=source)
    return BenchmarkImage(
        source=str(resolved),
        payload=payload,
        representation="raw",
        decoded_nbytes=len(payload),
    )


def unwrap_payloads(value: Any) -> list[Any]:
    """Extract image payloads from vLLM custom-dataset/plugin envelopes."""

    if isinstance(value, Mapping):
        for key in ("prompt", "data", "image", "images", "image_url", "url"):
            if key in value:
                return unwrap_payloads(value[key])
        raise TypeError(
            "JSONL image mapping must contain prompt, data, image, images, "
            "image_url, or url"
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [item for nested in value for item in unwrap_payloads(nested)]
    return [value]


def image_from_json_value(
    value: Any,
    *,
    base_dir: Path,
    source: str,
) -> BenchmarkImage:
    if not isinstance(value, str):
        raise TypeError(
            f"Expected a JPEG string payload at {source}, got {type(value)}"
        )

    if value.startswith("data:"):
        try:
            header, payload = value.split(",", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed data URI at {source}") from exc
        media_type = header[5:].split(";", 1)[0].lower()
        if media_type not in {"image/jpeg", "image/jpg"} or ";base64" not in header:
            raise ValueError(f"Expected a base64 JPEG data URI at {source}")
        return make_base64_image(payload, source=source)

    if value.startswith(("http://", "https://")):
        raise ValueError(
            f"Remote image URLs are not fetched by this standalone baseline: {source}"
        )

    if value.startswith("file://"):
        parsed = urlparse(value)
        return make_raw_image(Path(unquote(parsed.path)), source=source)

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    # Avoid calling stat() on a long unprefixed base64 payload.
    try:
        is_file = len(value) < 4096 and candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        return make_raw_image(candidate, source=source)

    return make_base64_image(value, source=source)


def load_jsonl_images(
    path: Path,
    *,
    offset: int,
    limit: int,
) -> list[BenchmarkImage]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset JSONL does not exist: {resolved}")

    images: list[BenchmarkImage] = []
    seen = 0
    with resolved.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {resolved} at line {line_number}"
                ) from exc
            for payload_index, value in enumerate(unwrap_payloads(row)):
                if seen >= offset and len(images) < limit:
                    images.append(
                        image_from_json_value(
                            value,
                            base_dir=resolved.parent,
                            source=(
                                f"{resolved}:line {line_number}:image "
                                f"{payload_index}"
                            ),
                        )
                    )
                seen += 1
                if len(images) == limit:
                    return images

    raise RuntimeError(
        f"Dataset has {seen} images; need at least {offset + limit} for "
        f"--offset {offset} --limit {limit}"
    )


def load_jpeg_dir_images(
    path: Path,
    *,
    offset: int,
    limit: int,
) -> list[BenchmarkImage]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"JPEG directory does not exist: {resolved}")
    paths = sorted(
        image for image in resolved.iterdir() if image.suffix.lower() in JPEG_SUFFIXES
    )
    needed = offset + limit
    if len(paths) < needed:
        raise RuntimeError(f"Found {len(paths)} JPEGs in {resolved}; need {needed}")
    return [
        make_raw_image(image, source=str(image)) for image in paths[offset:needed]
    ]


def batched(
    items: Sequence[BenchmarkImage], batch_size: int
) -> Iterable[Sequence[BenchmarkImage]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def git_provenance(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        process = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()

    try:
        status = run("status", "--porcelain", "--untracked-files=all")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current") or None,
            "origin": run("remote", "get-url", "origin"),
            "dirty": bool(status),
            "status": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not inspect model repository {repo}") from exc


def start_gpu_trace(
    path: Path,
    *,
    interval_ms: int,
    device: str | None,
) -> tuple[subprocess.Popen[Any], Any]:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    output = resolved.open("w", encoding="utf-8")
    command = ["nvidia-smi"]
    if device is not None:
        command.extend(("-i", device))
    command.extend(
        (
            f"--query-gpu={GPU_TRACE_QUERY}",
            "--format=csv",
            "-lms",
            str(interval_ms),
        )
    )
    try:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except Exception:
        output.close()
        raise

    time.sleep(max(interval_ms / 1000, 0.1))
    if process.poll() is not None:
        stderr = process.stderr.read() if process.stderr is not None else ""
        output.close()
        raise RuntimeError(f"nvidia-smi trace failed to start: {stderr.strip()}")
    return process, output


def stop_gpu_trace(process: subprocess.Popen[Any], output: Any) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if process.stderr is not None:
        process.stderr.close()
    output.close()


def parse_metric(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped in {"", "[Not Supported]", "N/A"}:
        return None
    try:
        return float(stripped.split()[0])
    except (IndexError, ValueError):
        return None


def parse_trace_timestamp(value: str) -> float | None:
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def summarize_gpu_trace(
    path: Path,
    *,
    start_epoch_s: float | None = None,
    finish_epoch_s: float | None = None,
) -> dict[str, float | int]:
    with path.expanduser().resolve().open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is not None:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = []
        for raw in reader:
            row = {key.strip(): value for key, value in raw.items()}
            row["_epoch_s"] = parse_trace_timestamp(row.get("timestamp", ""))
            rows.append(row)

    if start_epoch_s is not None:
        rows = [
            row
            for row in rows
            if row["_epoch_s"] is not None and row["_epoch_s"] >= start_epoch_s
        ]
    if finish_epoch_s is not None:
        rows = [
            row
            for row in rows
            if row["_epoch_s"] is not None and row["_epoch_s"] <= finish_epoch_s
        ]

    summary: dict[str, float | int] = {"samples": len(rows)}
    for column, prefix in (
        ("utilization.gpu [%]", "gpu_util_pct"),
        ("utilization.memory [%]", "gpu_mem_util_pct"),
        ("memory.used [MiB]", "gpu_memory_used_mib"),
        ("temperature.gpu", "gpu_temp_c"),
        ("power.draw [W]", "gpu_power_w"),
        ("clocks.current.sm [MHz]", "gpu_clocks_sm_mhz"),
        ("clocks.current.memory [MHz]", "gpu_clocks_mem_mhz"),
    ):
        values = [
            metric
            for row in rows
            if (metric := parse_metric(row.get(column))) is not None
        ]
        if values:
            summary[f"{prefix}_avg"] = sum(values) / len(values)
            summary[f"{prefix}_min"] = min(values)
            summary[f"{prefix}_max"] = max(values)
    return summary


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    model_repo = args.model_repo.expanduser().resolve()
    model_source = model_repo / "nemotron-ocr/src"
    model_dir = (
        args.model_dir.expanduser().resolve()
        if args.model_dir is not None
        else model_repo / "v2_multilingual"
    )
    if not model_source.is_dir():
        raise NotADirectoryError(
            f"Nemotron OCR source directory missing: {model_source}"
        )
    if not model_dir.is_dir():
        raise NotADirectoryError(
            f"Nemotron OCR checkpoint directory missing: {model_dir}"
        )

    provenance = git_provenance(model_repo)
    if provenance["dirty"] and not args.allow_dirty_model_repo:
        raise RuntimeError(
            f"Model repository is modified: {model_repo}. Use the clean baseline "
            "checkout or pass --allow-dirty-model-repo explicitly."
        )

    dataset_load_started = time.perf_counter()
    if args.dataset_jsonl is not None:
        images = load_jsonl_images(
            args.dataset_jsonl,
            offset=args.offset,
            limit=args.limit,
        )
        dataset_source = str(args.dataset_jsonl.expanduser().resolve())
        dataset_kind = "jpeg_byte_jsonl"
    else:
        assert args.jpeg_dir is not None
        images = load_jpeg_dir_images(
            args.jpeg_dir,
            offset=args.offset,
            limit=args.limit,
        )
        dataset_source = str(args.jpeg_dir.expanduser().resolve())
        dataset_kind = "jpeg_directory_preloaded_bytes"
    dataset_load_s = time.perf_counter() - dataset_load_started

    # Put the clean source tree first even if the caller's PYTHONPATH points at
    # the optimized checkout.
    sys.path.insert(0, str(model_source))
    import torch
    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

    if not torch.cuda.is_available():
        raise RuntimeError("The official Nemotron OCR pipeline requires CUDA")
    if args.profile_ocr:
        logging.basicConfig(level=logging.INFO)

    model_init_started = time.perf_counter()
    ocr = NemotronOCRV2(
        model_dir=str(model_dir),
        detector_max_batch_size=args.detector_max_batch_size,
        recognizer_chunk_size=args.recognizer_chunk_size,
        relational_chunk_size=args.relational_chunk_size,
        infer_length=args.infer_length,
        verbose_post=args.profile_ocr,
    )
    torch.cuda.synchronize()
    model_init_s = time.perf_counter() - model_init_started

    def run_batch(batch: Sequence[BenchmarkImage]) -> list[list[dict[str, Any]]]:
        inputs = [image.pipeline_input() for image in batch]
        with torch.inference_mode():
            outputs = ocr(
                inputs,
                merge_level=args.merge_level,
                include_invalid=args.include_invalid,
            )
        torch.cuda.synchronize()
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"Official pipeline returned {len(outputs)} outputs for "
                f"a batch of {len(batch)} images"
            )
        return outputs

    warmup_images = [images[index % len(images)] for index in range(args.warmup)]
    warmup_started = time.perf_counter()
    for batch in batched(warmup_images, args.batch_size):
        run_batch(batch)
    warmup_s = time.perf_counter() - warmup_started

    trace_process = None
    trace_output = None
    if args.gpu_trace_csv is not None:
        trace_process, trace_output = start_gpu_trace(
            args.gpu_trace_csv,
            interval_ms=args.gpu_trace_interval_ms,
            device=args.gpu_trace_device,
        )

    predictions: list[dict[str, Any]] | None = (
        [] if args.predictions_json is not None else None
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    timed_started_at_epoch_s = time.time()
    timed_started_at_monotonic_s = time.perf_counter()
    completed = 0
    try:
        for replay_index in range(args.replay_count):
            for batch_start, batch in enumerate(batched(images, args.batch_size)):
                outputs = run_batch(batch)
                if predictions is not None:
                    first_index = batch_start * args.batch_size
                    for local_index, (image, regions) in enumerate(
                        zip(batch, outputs, strict=True)
                    ):
                        predictions.append(
                            {
                                "backend": "official_nvidia_pipeline",
                                "model": "nvidia/nemotron-ocr-v2",
                                "image_index": first_index + local_index,
                                "replay_index": replay_index,
                                "merge_level": args.merge_level,
                                "source": image.source,
                                "regions": regions,
                            }
                        )
                completed += len(batch)
    finally:
        elapsed_s = time.perf_counter() - timed_started_at_monotonic_s
        timed_finished_at_epoch_s = time.time()
        if trace_process is not None and trace_output is not None:
            stop_gpu_trace(trace_process, trace_output)

    if completed != len(images) * args.replay_count:
        raise RuntimeError(
            f"Completed {completed} images; expected "
            f"{len(images) * args.replay_count}"
        )

    if args.predictions_json is not None:
        assert predictions is not None
        write_json(args.predictions_json, predictions)

    representation_counts = {
        representation: sum(
            image.representation == representation for image in images
        )
        for representation in ("base64", "raw")
    }
    result: dict[str, Any] = {
        "backend": "official_nvidia_pipeline",
        "pipeline_class": "nemotron_ocr.inference.pipeline_v2.NemotronOCRV2",
        "uses_vllm": False,
        "model": "nvidia/nemotron-ocr-v2",
        "official_upstream": "https://huggingface.co/nvidia/nemotron-ocr-v2",
        "model_repo": str(model_repo),
        "model_dir": str(model_dir),
        "model_repo_provenance": provenance,
        "dataset_kind": dataset_kind,
        "dataset_source": dataset_source,
        "dataset_load_s": dataset_load_s,
        "dataset_compressed_bytes": sum(image.decoded_nbytes for image in images),
        "input_representation_counts": representation_counts,
        "base64_decode_in_timed_pipeline": representation_counts["base64"] > 0,
        "jpeg_decode_in_timed_pipeline": True,
        "offset": args.offset,
        "unique_image_count": len(images),
        "replay_count": args.replay_count,
        "timed_workload_image_count": completed,
        "timed_repeated_image_count": len(images) * (args.replay_count - 1),
        "batch_size": args.batch_size,
        "warmup_image_count": len(warmup_images),
        "warmup_s": warmup_s,
        "model_init_s": model_init_s,
        "elapsed_s": elapsed_s,
        "images_per_second": completed / elapsed_s,
        "ms_per_image": elapsed_s * 1000 / completed,
        "timed_started_at_epoch_s": timed_started_at_epoch_s,
        "timed_finished_at_epoch_s": timed_finished_at_epoch_s,
        "timed_started_at_monotonic_s": timed_started_at_monotonic_s,
        "timed_finished_at_monotonic_s": timed_started_at_monotonic_s + elapsed_s,
        "infer_length": args.infer_length,
        "merge_level": args.merge_level,
        "include_invalid": args.include_invalid,
        "detector_max_batch_size": args.detector_max_batch_size,
        "recognizer_chunk_size": args.recognizer_chunk_size,
        "relational_chunk_size": args.relational_chunk_size,
        "profile_ocr": args.profile_ocr,
        "peak_gpu_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_gpu_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_index": torch.cuda.current_device(),
        "cuda_device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "predictions_json": (
            str(args.predictions_json.expanduser().resolve())
            if args.predictions_json is not None
            else None
        ),
        "output_json": (
            str(args.output_json.expanduser().resolve())
            if args.output_json is not None
            else None
        ),
        "gpu_trace_csv": (
            str(args.gpu_trace_csv.expanduser().resolve())
            if args.gpu_trace_csv is not None
            else None
        ),
        "gpu_trace_device": args.gpu_trace_device,
        "gpu_trace_interval_ms": args.gpu_trace_interval_ms,
    }
    if args.gpu_trace_csv is not None:
        result["gpu_trace"] = summarize_gpu_trace(args.gpu_trace_csv)
        result["gpu_trace_timed"] = summarize_gpu_trace(
            args.gpu_trace_csv,
            start_epoch_s=timed_started_at_epoch_s,
            finish_epoch_s=timed_finished_at_epoch_s,
        )

    rendered = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=json_default,
    )
    print(rendered)
    if args.output_json is not None:
        write_json(args.output_json, result)


if __name__ == "__main__":
    main()
