# Nemotron OCR v2 A100 throughput study

This page records a workload-specific optimization study for
[`nvidia/nemotron-ocr-v2`](https://huggingface.co/nvidia/nemotron-ocr-v2) on one
NVIDIA A100-SXM4-80GB. The final native vLLM deployment processed a matched
30,000-image workload at **85.3941 images/s**: **2.73x** the official
NVIDIA/Hugging Face in-process pipeline and **1.89x** a separately tuned,
clean-model vLLM baseline.

## Matched 30K result

![Matched 30K Nemotron OCR v2 throughput](../assets/benchmarking/nemotron_ocr_v2_a100/matched_30k_speedup.png)

[Vector version](../assets/benchmarking/nemotron_ocr_v2_a100/matched_30k_speedup.svg)

| System | Completed | Throughput | Versus official HF | Versus clean vLLM |
| --- | ---: | ---: | ---: | ---: |
| Official NVIDIA/HF in-process | 30,000 / 30,000 | 31.2456869 images/s | 1.000x | 0.692x |
| Tuned clean-model vLLM, one replica | 30,000 / 30,000 | 45.1633991 images/s | 1.445x | 1.000x |
| Optimized native vLLM, eight replicas | 30,000 / 30,000 | **85.3941303 images/s** | **2.733x** | **1.891x** |

The native vLLM summaries report zero failed requests. The official HF result
records all 30,000 completed images but its schema has no separate failure
field. Model initialization and warmup were excluded from every timed span.
The optimized result was also reproduced in two isolated 10,000-image runs at
85.3153 and 85.2091 images/s, a 0.12% difference.

## Corrected clean vLLM baseline

The earlier 43.8927 images/s control was a valid isolated run, but it was not a
tuned vLLM baseline. It used detector batch 8 and one renderer worker. A new
21-configuration, clean-model sweep varied detector batching, renderer workers,
`max_num_seqs`, and client concurrency. Every 10,000-image sweep point completed
without a request failure.

| Clean-model 10K configuration | Throughput |
| --- | ---: |
| detector 16, renderers 4, seqs 64, concurrency 128 | **44.3450 images/s** |
| detector 16, renderers 4, seqs 96, concurrency 192 | 44.2288 images/s |
| detector 32, renderers 4, seqs 64, concurrency 128 | 44.0366 images/s |
| detector 16, renderers 4, seqs 64, concurrency 192 | 43.9811 images/s |
| detector 8, renderers 4, seqs 32, concurrency 128 | 43.9639 images/s |

The winning shape was then rerun over the full matched 30K workload and reached
45.1634 images/s. That full run, rather than either the original control or a
short sweep point, is the baseline used in the headline speedup.

## Matched workload contract

| Control | Recorded value |
| --- | --- |
| Hardware | One NVIDIA A100-SXM4-80GB, UUID `GPU-242d3c90-db9c-a49e-e2b9-ddb0b36f1ba3` |
| Corpus | The same ordered 1,000 real document pages for every system |
| Encoding | JPEG quality 100, 4:4:4, carried as base64 JPEG bytes in JSONL |
| Timed workload | 1,000 unique pages x 30 replays = 30,000 images |
| OCR contract | Multilingual checkpoint, paragraph merge, `infer_length=1024` |
| Metric | Completed timed images divided by the measured timed span |
| Telemetry | `nvidia-smi` at a target 250 ms interval around each timed run |
| Isolation | Systems ran sequentially; overlapping GPU work invalidated a run |

The JSONL corpus is `bo767_1k_pooling_jpeg_q100_444.jsonl`, contains
642,407,686 compressed JPEG bytes, and has SHA-256
`139c96ef75a85da440350722a95d9eb3bd21dd4155d43f7281253f63c07eaa16`.
Base64 and JPEG decoding are part of the timed processing path. Thirty replays
were used deliberately so every system remained in sustained execution long
enough to expose queue starvation and utilization gaps.

## GPU-active behavior

![GPU-active aligned utilization and power](../assets/benchmarking/nemotron_ocr_v2_a100/gpu_active_comparison.png)

[Vector version](../assets/benchmarking/nemotron_ocr_v2_a100/gpu_active_comparison.svg)

| System | Active duration | Mean active GPU | Mean active power |
| --- | ---: | ---: | ---: |
| Official NVIDIA/HF | 15.98 min | 50.79% | 235.23 W |
| Tuned clean vLLM | 11.15 min | 73.80% | 300.80 W |
| Optimized native vLLM | 5.91 min | **99.93%** | 395.92 W |

Each trace is aligned independently at its first sustained GPU-active sample.
The plotted lines are sample-based trailing 15-second means and retain the raw
tail so completion remains visible. The optimized run's complete trace averaged
99.724% GPU utilization, peaked at 61,668 MiB observed memory, and averaged
395.293 W.

## Deployment profile

| Setting | Official HF | Tuned clean vLLM | Optimized native vLLM |
| --- | ---: | ---: | ---: |
| Execution shape | In-process batches | 1 `/pooling` replica | 8 `/pooling` replicas |
| Model source | Clean upstream | Clean upstream | Patched, exact-fusion source |
| Batch / `max_num_seqs` | 64 | 64 | 40 per replica |
| Renderer workers | N/A | 4 | 4 per replica |
| Detector max batch | 32 | 16 | 16 |
| Recognizer chunk | 128 | 128 | 128 |
| Relational chunk | 128 | 128 | 128 |
| Client concurrency | N/A | 128 | 64 per replica, 512 total |
| Warmup | 128 images | 32 requests | 32 requests per replica |
| CUDA MPS share | N/A | No MPS partition | 25% per replica |
| vLLM execution | N/A | eager, async scheduling off | eager, async scheduling off |

The client uses one work-conserving queue. The next available endpoint pulls
the next JPEG request, avoiding fixed-shard tail imbalance, while native vLLM
`AsyncLLM` admission and continuous batching remain responsible for the queue
inside each `/pooling` engine.

### Why the model was not split into pipeline services

Stock vLLM pooling cannot pass arbitrary CUDA tensors between independently
queued model instances. Splitting detector, recognizer, and relational stages
would therefore require a new same-process executor or a custom CUDA IPC, NCCL,
or NIXL tensor transport plus cross-stage backpressure. It would not be a
configuration-only vLLM optimization.

The measured design keeps the complete OCR pipeline in every replica and lets
vLLM queue complete image requests. This avoids host round trips between model
stages and reached near-saturated GPU execution. A future disaggregated design
should be accepted only if its measured gain exceeds the scheduling and tensor
handoff cost.

## What changed

The final result combines deployment and source-level changes:

- a first-class vLLM pooling model and JPEG-byte I/O processor;
- compact payload rows so only declared result bytes cross the device/IPC
  boundary;
- custom CUDA kernels launched on PyTorch's current stream;
- batched relational geometry and removal of avoidable GPU synchronization;
- exact detector batch-normalization, residual, ASPP, normalization, and
  concat/upsample fusions with guarded eager fallbacks;
- eight warmed native vLLM queues under CUDA MPS; and
- work-conserving request dispatch across replicas.

The engine still uses eager execution because the model pipeline contains
dynamic detector/NMS/postprocessing shapes that are not yet safe for a single
CUDA Graph capture. The current-stream fix is a prerequisite for correct
multi-stream execution; replicas and MPS, rather than one oversized sequence
batch, delivered the large utilization gain.

## Output-agreement quality gate

The optimized detector was first checked directly at actual checkpoint batch
sizes 1, 8, and 16. Confidence tensors, rotated boxes, and feature tensors were
bitwise identical to the unfused control.

Full-pipeline candidate and control runs then processed the same 1,000 JPEG
pages twice. The optimized candidate had 18 region-sequence edits against one
control run, while the two control repeats had 19. It produced 984 exact page
text sequences versus 982 between control repeats. Numeric drift for exact-text
aligned regions stayed within the declared coordinate and confidence
tolerances. One page had a nondeterministic region split/merge, but its
space-joined text was preserved; candidate-repeat divergence was 10 edits,
better than the control-repeat divergence.

The repeat-aware verdict is `no_output_regression_detected`. This is an output
agreement envelope against repeated controls, not a labeled-ground-truth OCR
accuracy measurement. The primary one-to-one strict comparator alone is
intentionally not described as passing because it flags that single
segmentation change.

## Reproducing the runs

The dataset builder and benchmark drivers are included in
[`benchmarks/nemotron_ocr_v2`](../../benchmarks/nemotron_ocr_v2/README.md). The
external source changes are archived as
[`nemotron_ocr_model_optimizations.patch`](../../benchmarks/nemotron_ocr_v2/nemotron_ocr_model_optimizations.patch).

```bash
ROOT=/raid/vjawa/tmp/ocr_optimization
VLLM_REPO=$(pwd)
GPU_UUID=GPU-242d3c90-db9c-a49e-e2b9-ddb0b36f1ba3
DATASET="$ROOT/data/bo767_1k_pooling_jpeg_q100_444.jsonl"
CLEAN_MODEL="$ROOT/nemotron-ocr-v2-baseline"
OPT_MODEL="$ROOT/nemotron-ocr-v2"
```

Create the JPEG-byte dataset with the same encoding recipe:

```bash
"$ROOT/venv/bin/python" \
  "$VLLM_REPO/benchmarks/nemotron_ocr_v2/build_pooling_dataset.py" \
  --image-dir /path/to/document_images \
  --output "$DATASET" --limit 1000 \
  --payload-mode jpeg-bytes --jpeg-quality 100 --jpeg-subsampling 0 \
  --workers 8
sha256sum "$DATASET"
```

Run the official NVIDIA/HF reference directly in-process:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
NEMOTRON_OCR_SOURCE="$CLEAN_MODEL/nemotron-ocr/src" \
PYTHONPATH="$CLEAN_MODEL/nemotron-ocr/src" \
"$ROOT/venv/bin/python" \
  "$VLLM_REPO/benchmarks/nemotron_ocr_v2/benchmark_hf_inprocess.py" \
  --dataset-jsonl "$DATASET" \
  --model-repo "$CLEAN_MODEL" \
  --model-dir "$CLEAN_MODEL/v2_multilingual" \
  --limit 1000 --batch-size 64 --warmup 128 --replay-count 30 \
  --merge-level paragraph --infer-length 1024 \
  --detector-max-batch-size 32 \
  --recognizer-chunk-size 128 --relational-chunk-size 128 \
  --gpu-trace-device "$GPU_UUID" --gpu-trace-interval-ms 250 \
  --gpu-trace-csv /path/to/hf/gpu_trace.csv \
  --output-json /path/to/hf/result.json
```

The clean vLLM baseline uses detector 16, four renderer workers,
`max_num_seqs=64`, and client concurrency 128. The optimized servers use the
same OCR chunk sizes but eight replicas, `max_num_seqs=40`, concurrency 64 per
endpoint, and the exact-fusion environment flags. The complete sweep harness
records resolved commands, environment, repository state, extension hashes,
dataset hashes, GPU identity, failures, and telemetry in each run directory.

For first-pass single-server sweeps, the repo also includes
[`serve_sweep.json`](../../benchmarks/nemotron_ocr_v2/serve_sweep.json) and
[`bench_sweep.json`](../../benchmarks/nemotron_ocr_v2/bench_sweep.json) for
[`vllm bench sweep serve`](sweeps.md). The dedicated multi-replica harness used
for the final run extends that sweep to replica count, MPS share, detector and
recognizer chunks, and per-endpoint concurrency while keeping request admission
inside native vLLM engines.

## Source and artifact boundary

The official and clean vLLM baselines use model commit
`0e83e83f17943524b90afa6c0fd82ac2bc1a40ca`. The final run captured vLLM commit
`d5058677b204b0026c2ed87fd4a2d81317ab0f8c` plus the source patch now folded
into this PR branch, and model commit `bb392d4` plus the exact-fusion changes now
committed as `a92d75050f05c2638394e970bf8cec53c113d99b` and pushed to
[`nvidia/nemotron-ocr-v2` discussion #8](https://huggingface.co/nvidia/nemotron-ocr-v2/discussions/8).

!!! warning "The optimized result requires the external model patch"

    The optimized servers used a rebuilt CUDA extension and external Nemotron
    OCR source changes. The archived patch has SHA-256
    `19084526836ca882625383028e7273b0c4e3872315e458cf18eba788faed0cae`
    and applies to the pinned clean model checkout. Rebuild it for A100 compute
    capability 8.0 by following the
    [benchmark README](../../benchmarks/nemotron_ocr_v2/README.md). A clean
    checkout of only the vLLM branch cannot reproduce 85.3941 images/s.

The exact summaries, raw GPU traces, baseline sweep ranking, compact accuracy
evidence, and matching charts are published in
[`VibhuJawa/nemotron-vllm-ocr`](https://github.com/VibhuJawa/nemotron-vllm-ocr/tree/results/final-85-imgs/results/a100-2026-07-09-final-85-imgs).
