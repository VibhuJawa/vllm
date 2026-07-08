# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import urllib.request
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

SAFEDOCS_1K_URL = (
    "https://downloads.digitalcorpora.org/corpora/files/"
    "CC-MAIN-2021-31-PDF-UNTRUNCATED/zipfiles/0000-0999/0000.zip"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
GPU_TRACE_QUERY = (
    "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Nemotron OCR v2 on real document images."
    )
    parser.add_argument("--backend", choices=["vllm", "direct"], default="vllm")
    parser.add_argument("--model", default="nvidia/nemotron-ocr-v2")
    parser.add_argument("--model-subdir", default="v2_multilingual")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--image-offset",
        type=int,
        default=0,
        help="Skip this many sorted images before applying --limit.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=0,
        help=(
            "Number of prompts per LLM.encode call for the vLLM backend. "
            "Set 0 to submit all timed prompts in one call and let the "
            "engine schedule up to --batch-size requests at a time."
        ),
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--merge-level", default="paragraph")
    parser.add_argument(
        "--infer-length",
        type=int,
        help="Override the OCR detector input resolution.",
    )
    parser.add_argument(
        "--profile-ocr",
        action="store_true",
        help=(
            "Enable Nemotron OCR per-phase timing logs. This adds CUDA "
            "synchronization and should not be used for throughput numbers."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--gpu-trace-csv",
        type=Path,
        help="Write a sampled nvidia-smi GPU utilization trace around the run.",
    )
    parser.add_argument(
        "--gpu-trace-interval-ms",
        type=int,
        default=500,
        help="Sampling interval for --gpu-trace-csv.",
    )
    parser.add_argument(
        "--disable-io-processor",
        action="store_true",
        help="Use raw multimodal prompts instead of the IO processor plugin.",
    )
    parser.add_argument(
        "--plugin-prompt-mode",
        choices=["independent", "bundled"],
        default="independent",
        help=(
            "For the IO processor path, submit each image as its own plugin "
            "request or bundle a batch of images inside one plugin request."
        ),
    )

    parser.add_argument("--prepare-safedocs", action="store_true")
    parser.add_argument(
        "--safedocs-root",
        type=Path,
        default=Path("bench-data/safedocs"),
    )
    parser.add_argument("--safedocs-url", default=SAFEDOCS_1K_URL)
    parser.add_argument("--render-dpi", type=int, default=144)
    parser.add_argument("--render-workers", type=int, default=8)

    parser.add_argument("--detector-max-batch-size", type=int, default=8)
    parser.add_argument("--recognizer-chunk-size", type=int, default=128)
    parser.add_argument("--relational-chunk-size", type=int, default=128)
    return parser.parse_args()


def download_file(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def extract_zip(zip_path: Path, pdf_dir: Path):
    if pdf_dir.exists() and any(pdf_dir.glob("*.pdf")):
        return
    pdf_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(pdf_dir)


def render_pdf_page(
    pdf: Path,
    out_dir: Path,
    page: int,
    dpi: int,
    suffix: str = "",
) -> Path | None:
    output_prefix = out_dir / f"{pdf.stem}{suffix}"
    output_path = output_prefix.with_suffix(".png")
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    cmd = [
        "pdftoppm",
        "-r",
        str(dpi),
        "-f",
        str(page),
        "-l",
        str(page),
        "-singlefile",
        "-png",
        str(pdf),
        str(output_prefix),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    return None


def render_safedocs_images(
    pdf_dir: Path,
    out_dir: Path,
    *,
    dpi: int,
    limit: int,
    workers: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))

    def render_first_page(pdf: Path):
        return render_pdf_page(pdf, out_dir, page=1, dpi=dpi)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(render_first_page, pdf) for pdf in pdfs]
        for _ in as_completed(futures):
            pass

    images = sorted(out_dir.glob("*.png"))
    page = 2
    while len(images) < limit and page <= 10:
        for pdf in pdfs:
            if len(images) >= limit:
                break
            image = render_pdf_page(
                pdf, out_dir, page=page, dpi=dpi, suffix=f"_p{page}"
            )
            if image is not None:
                images.append(image)
        page += 1

    if len(images) < limit:
        raise RuntimeError(
            f"Only rendered {len(images)} images from {len(pdfs)} PDFs; "
            f"requested {limit}."
        )


def prepare_safedocs(args) -> Path:
    root = args.safedocs_root.expanduser().resolve()
    zip_path = root / "0000.zip"
    pdf_dir = root / "pdfs"
    image_dir = root / "page_images"
    download_file(args.safedocs_url, zip_path)
    extract_zip(zip_path, pdf_dir)
    render_safedocs_images(
        pdf_dir,
        image_dir,
        dpi=args.render_dpi,
        limit=args.limit,
        workers=args.render_workers,
    )
    return image_dir


def list_images(image_dir: Path, *, limit: int, offset: int) -> list[Path]:
    if offset < 0:
        raise ValueError("--image-offset must be non-negative.")

    images = [
        path
        for path in sorted(image_dir.expanduser().resolve().iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    needed = offset + limit
    if len(images) < needed:
        raise RuntimeError(
            f"Found {len(images)} images in {image_dir}; need {needed} "
            f"for offset {offset} and limit {limit}."
        )
    return images[offset:needed]


def batches(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    if batch_size <= 0:
        yield items
        return

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def start_gpu_trace(path: Path, interval_ms: int) -> tuple[subprocess.Popen, Any]:
    if interval_ms <= 0:
        raise ValueError("--gpu-trace-interval-ms must be positive.")

    path.parent.mkdir(parents=True, exist_ok=True)
    output = path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_TRACE_QUERY}",
            "--format=csv",
            "-lms",
            str(interval_ms),
        ],
        stdout=output,
        stderr=subprocess.DEVNULL,
    )
    return proc, output


def stop_gpu_trace(proc: subprocess.Popen, output: Any):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    output.close()


def parse_metric(value: str) -> float | None:
    stripped = value.strip()
    if stripped in {"", "[Not Supported]"}:
        return None
    token = stripped.split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def summarize_gpu_trace(path: Path) -> dict[str, float | int] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is not None:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = [{key.strip(): value for key, value in row.items()} for row in reader]
    if not rows:
        return None

    def values(column: str) -> list[float]:
        parsed = [parse_metric(row.get(column, "")) for row in rows]
        return [value for value in parsed if value is not None]

    summary: dict[str, float | int] = {"gpu_trace_samples": len(rows)}
    for column, prefix in (
        ("utilization.gpu [%]", "gpu_util_pct"),
        ("utilization.memory [%]", "gpu_mem_util_pct"),
        ("memory.used [MiB]", "gpu_memory_used_mib"),
        ("temperature.gpu", "gpu_temp_c"),
        ("power.draw [W]", "gpu_power_w"),
        ("clocks.current.sm [MHz]", "gpu_clocks_sm_mhz"),
        ("clocks.current.memory [MHz]", "gpu_clocks_mem_mhz"),
    ):
        vals = values(column)
        if vals:
            summary[f"{prefix}_avg"] = sum(vals) / len(vals)
            summary[f"{prefix}_max"] = max(vals)

    return summary


def hf_overrides(args, *, use_io_processor: bool = True) -> dict[str, Any]:
    overrides = {
        "model_type": "nemotron_ocr_v2",
        "architectures": ["NemotronOCRV2ForImageToText"],
        "nemotron_ocr_model_subdir": args.model_subdir,
        "nemotron_ocr_merge_level": args.merge_level,
        "nemotron_ocr_detector_max_batch_size": args.detector_max_batch_size,
        "nemotron_ocr_recognizer_chunk_size": args.recognizer_chunk_size,
        "nemotron_ocr_relational_chunk_size": args.relational_chunk_size,
        "nemotron_ocr_verbose_post": args.profile_ocr,
        "io_processor_plugin": "nemotron_ocr_v2" if use_io_processor else None,
    }
    if args.infer_length is not None:
        overrides["nemotron_ocr_infer_length"] = args.infer_length
    return overrides


def run_vllm_backend(args, images: list[Path]) -> dict[str, Any]:
    from vllm import LLM

    llm = LLM(
        model=args.model,
        skip_tokenizer_init=True,
        enforce_eager=True,
        io_processor_plugin=None if args.disable_io_processor else "nemotron_ocr_v2",
        max_num_seqs=max(args.batch_size, 1),
        hf_overrides=hf_overrides(
            args,
            use_io_processor=not args.disable_io_processor,
        ),
    )

    def prompts_for(batch: list[Path]):
        if not args.disable_io_processor:
            images = [str(path) for path in batch]
            if args.plugin_prompt_mode == "independent":
                return [{"data": image} for image in images]
            return {"data": {"images": images} if len(images) != 1 else images[0]}
        return [
            {
                "prompt_token_ids": [1],
                "multi_modal_data": {"image": Image.open(path).convert("RGB")},
            }
            for path in batch
        ]

    warmup = images[: args.warmup]
    if warmup:
        llm.encode(
            prompts_for(warmup),
            pooling_task="plugin",
            use_tqdm=False,
        )
        torch.cuda.synchronize()

    timed = images[: args.limit]
    start = time.perf_counter()
    request_batch_size = args.request_batch_size or len(timed)
    for batch in batches(timed, request_batch_size):
        llm.encode(
            prompts_for(batch),
            pooling_task="plugin",
            use_tqdm=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "backend": "vllm",
        "count": len(timed),
        "elapsed_s": elapsed,
        "request_batch_size": request_batch_size,
        "max_num_seqs": args.batch_size,
        "plugin_prompt_mode": args.plugin_prompt_mode,
    }


def run_direct_backend(args, images: list[Path]) -> dict[str, Any]:
    from vllm.model_executor.models.nemotron_ocr import NemotronOCRV2ForImageToText
    from vllm.transformers_utils.configs.nemotron_ocr import NemotronOCRV2Config

    hf_config = NemotronOCRV2Config(**hf_overrides(args))
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=args.model,
            revision=None,
            hf_config=hf_config,
        )
    )
    model = NemotronOCRV2ForImageToText(vllm_config)

    def run_batch(batch_paths: list[Path]):
        images_pil = [Image.open(path).convert("RGB") for path in batch_paths]
        input_ids = torch.ones((len(images_pil), 1), dtype=torch.long, device="cuda")
        positions = torch.arange(len(images_pil), dtype=torch.long, device="cuda")
        with torch.inference_mode():
            output = model(
                input_ids=input_ids, positions=positions, ocr_images=images_pil
            )
        torch.cuda.synchronize()
        return output

    warmup = images[: args.warmup]
    if warmup:
        run_batch(warmup)

    timed = images[: args.limit]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for batch in batches(timed, args.batch_size):
        run_batch(batch)
    elapsed = time.perf_counter() - start

    result = {"backend": "direct", "count": len(timed), "elapsed_s": elapsed}
    if torch.cuda.is_available():
        result["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return result


def main():
    args = parse_args()
    image_dir = prepare_safedocs(args) if args.prepare_safedocs else args.image_dir
    if image_dir is None:
        raise ValueError("Pass --image-dir or --prepare-safedocs.")

    images = list_images(image_dir, limit=args.limit, offset=args.image_offset)
    trace_proc = None
    trace_output = None
    try:
        if args.gpu_trace_csv is not None:
            trace_proc, trace_output = start_gpu_trace(
                args.gpu_trace_csv,
                args.gpu_trace_interval_ms,
            )

        if args.backend == "vllm":
            result = run_vllm_backend(args, images)
        else:
            result = run_direct_backend(args, images)
    finally:
        if trace_proc is not None and trace_output is not None:
            stop_gpu_trace(trace_proc, trace_output)

    result.update(
        {
            "model": args.model,
            "model_subdir": args.model_subdir,
            "batch_size": args.batch_size,
            "image_offset": args.image_offset,
            "infer_length": args.infer_length,
            "profile_ocr": args.profile_ocr,
            "gpu_trace_csv": (
                str(args.gpu_trace_csv) if args.gpu_trace_csv is not None else None
            ),
            "images_per_second": result["count"] / result["elapsed_s"],
            "ms_per_image": result["elapsed_s"] * 1000 / result["count"],
        }
    )
    if args.gpu_trace_csv is not None:
        summary = summarize_gpu_trace(args.gpu_trace_csv)
        if summary is not None:
            result.update(summary)
    print(json.dumps(result, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
