#!/usr/bin/env python3
"""Benchmark a work-conserving queue across native vLLM pooling servers."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp
import orjson

GPU_TRACE_QUERY = (
    "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--replay-count", type=int, default=1)
    parser.add_argument("--infer-length", type=int, default=1024)
    parser.add_argument("--concurrency-per-endpoint", type=int, default=128)
    parser.add_argument("--warmups-per-endpoint", type=int, default=8)
    parser.add_argument(
        "--gpu",
        required=True,
        help="Physical nvidia-smi GPU index or UUID to trace.",
    )
    parser.add_argument("--trace-interval-ms", type=int, default=250)
    return parser.parse_args()


def load_request_bodies(path: Path, model: str) -> list[bytes]:
    bodies: list[bytes] = []
    with path.open("rb") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = orjson.loads(line)
                prompt = row["prompt"]
            except (orjson.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid custom dataset row {line_number}") from exc
            if not isinstance(prompt, dict):
                raise TypeError(f"row {line_number} prompt must be a mapping")
            bodies.append(
                orjson.dumps(
                    {
                        "model": model,
                        "truncate_prompt_tokens": -1,
                        **prompt,
                    }
                )
            )
    if not bodies:
        raise ValueError("dataset is empty")
    return bodies


def parse_metric(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.strip().split()[0])
    except (IndexError, ValueError):
        return None


def summarize_trace(path: Path) -> dict[str, float | int]:
    with path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is not None:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = [{key.strip(): value for key, value in row.items()} for row in reader]
    summary: dict[str, float | int] = {"samples": len(rows)}
    for column, prefix in (
        ("utilization.gpu [%]", "gpu_util_pct"),
        ("utilization.memory [%]", "gpu_mem_util_pct"),
        ("memory.used [MiB]", "gpu_memory_used_mib"),
        ("power.draw [W]", "gpu_power_w"),
    ):
        values = [parse_metric(row.get(column)) for row in rows]
        valid = [value for value in values if value is not None]
        if valid:
            summary[f"{prefix}_avg"] = sum(valid) / len(valid)
            summary[f"{prefix}_max"] = max(valid)
    return summary


async def benchmark(args: argparse.Namespace, bodies: list[bytes]) -> dict[str, Any]:
    endpoints = [endpoint.rstrip("/") + "/pooling" for endpoint in args.endpoint]
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=0)
    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def request(endpoint: str, body: bytes) -> tuple[bool, float, str]:
            started = time.perf_counter()
            try:
                async with session.post(
                    endpoint, data=body, headers=headers
                ) as response:
                    payload = await response.read()
                    elapsed = time.perf_counter() - started
                    if response.status != 200:
                        error_body = payload[:500].decode(errors="replace")
                        return False, elapsed, (
                            f"HTTP {response.status}: {error_body}"
                        )
                    return True, elapsed, ""
            except Exception as exc:  # noqa: BLE001 - benchmark records failures
                return False, time.perf_counter() - started, repr(exc)

        warmup_tasks = []
        for endpoint_index, endpoint in enumerate(endpoints):
            for warmup_index in range(args.warmups_per_endpoint):
                body = bodies[(endpoint_index + warmup_index) % len(bodies)]
                warmup_tasks.append(request(endpoint, body))
        warmup_results = await asyncio.gather(*warmup_tasks)
        warmup_failures = [error for ok, _, error in warmup_results if not ok]
        if warmup_failures:
            raise RuntimeError(f"warmup failures: {warmup_failures[:3]}")

        queue: asyncio.Queue[int] = asyncio.Queue()
        for request_index in range(args.num_prompts):
            queue.put_nowait(request_index)

        completed_by_endpoint: dict[int, int] = defaultdict(int)
        latencies_by_endpoint: dict[int, list[float]] = defaultdict(list)
        errors: list[str] = []

        async def worker(endpoint_index: int) -> None:
            endpoint = endpoints[endpoint_index]
            while True:
                try:
                    request_index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                ok, latency, error = await request(
                    endpoint, bodies[request_index % len(bodies)]
                )
                latencies_by_endpoint[endpoint_index].append(latency)
                if ok:
                    completed_by_endpoint[endpoint_index] += 1
                elif len(errors) < 100:
                    errors.append(f"endpoint {endpoint_index}: {error}")
                queue.task_done()

        timed_started_at_epoch_s = time.time()
        started = time.perf_counter()
        workers = [
            asyncio.create_task(worker(endpoint_index))
            for endpoint_index in range(len(endpoints))
            for _ in range(args.concurrency_per_endpoint)
        ]
        await asyncio.gather(*workers)
        elapsed = time.perf_counter() - started
        timed_finished_at_epoch_s = time.time()

    completed = sum(completed_by_endpoint.values())
    endpoint_results = []
    for index, endpoint in enumerate(endpoints):
        latencies = latencies_by_endpoint[index]
        endpoint_results.append(
            {
                "endpoint": endpoint,
                "completed": completed_by_endpoint[index],
                "mean_e2el_ms": statistics.fmean(latencies) * 1000,
                "median_e2el_ms": statistics.median(latencies) * 1000,
            }
        )
    return {
        "backend": "native_vllm_pooling_multi_endpoint",
        "dispatcher": "work_conserving_client_side_queue",
        "replicas": len(endpoints),
        "unique_image_count": len(bodies),
        "replay_count": args.replay_count,
        "infer_length": args.infer_length,
        "timed_workload_image_count": args.num_prompts,
        "timed_repeated_image_count": args.num_prompts - len(bodies),
        "completed": completed,
        "failed": args.num_prompts - completed,
        "elapsed_s": elapsed,
        "aggregate_images_per_second": completed / elapsed,
        "concurrency_per_endpoint": args.concurrency_per_endpoint,
        "total_concurrency": len(endpoints) * args.concurrency_per_endpoint,
        "warmups_per_endpoint": args.warmups_per_endpoint,
        "timed_started_at_epoch_s": timed_started_at_epoch_s,
        "timed_finished_at_epoch_s": timed_finished_at_epoch_s,
        "errors": errors,
        "endpoints": endpoint_results,
    }


def main() -> None:
    args = parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    if args.replay_count <= 0:
        raise ValueError("--replay-count must be positive")
    if args.infer_length <= 0:
        raise ValueError("--infer-length must be positive")
    if args.concurrency_per_endpoint <= 0:
        raise ValueError("--concurrency-per-endpoint must be positive")
    if args.warmups_per_endpoint < 0:
        raise ValueError("--warmups-per-endpoint must be non-negative")
    if args.trace_interval_ms <= 0:
        raise ValueError("--trace-interval-ms must be positive")

    bodies = load_request_bodies(args.dataset, args.model)
    if args.num_prompts != args.replay_count * len(bodies):
        raise ValueError(
            "--num-prompts must equal dataset rows times --replay-count"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "gpu_trace.csv"
    with trace_path.open("w", encoding="utf-8") as trace_output:
        trace_process = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                args.gpu,
                f"--query-gpu={GPU_TRACE_QUERY}",
                "--format=csv",
                "-lms",
                str(args.trace_interval_ms),
            ],
            stdout=trace_output,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            result = asyncio.run(benchmark(args, bodies))
        finally:
            trace_process.terminate()
            trace_process.wait(timeout=10)

    result["dataset"] = str(args.dataset.resolve())
    result["model"] = args.model
    result["gpu_trace_csv"] = str(trace_path.resolve())
    result["gpu_trace"] = summarize_trace(trace_path)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
