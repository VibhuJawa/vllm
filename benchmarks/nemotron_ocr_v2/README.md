# Nemotron OCR v2 A100 benchmark kit

This directory contains the dataset builder and benchmark clients used for the
matched `nvidia/nemotron-ocr-v2` Hugging Face and native vLLM comparison.

On one A100-SXM4-80GB, the final matched 30,000-image results were:

| System | Images/s | Speedup versus clean vLLM |
| --- | ---: | ---: |
| Official NVIDIA/HF in-process | 31.2456869 | 0.692x |
| Tuned clean-model vLLM | 45.1633991 | 1.000x |
| Optimized native vLLM | **85.3941303** | **1.891x** |

All systems received the same 1,000 ordered real document pages as JPEG Q100,
4:4:4 byte payloads and processed 30 replays. The native vLLM runs completed
with zero failed requests. See the
[full A100 report](../../docs/benchmarking/nemotron_ocr_v2_a100.md) for the
matched contract, corrected baseline sweep, GPU traces, and output-agreement
gate.

## External model patch

The optimized result requires model-side changes archived in
`nemotron_ocr_model_optimizations.patch`; they are not part of the vLLM source
tree. The patch applies to exact upstream model commit
`0e83e83f17943524b90afa6c0fd82ac2bc1a40ca` and has SHA-256
`19084526836ca882625383028e7273b0c4e3872315e458cf18eba788faed0cae`.
The corresponding model PR commit is
[`a92d75050`](https://huggingface.co/nvidia/nemotron-ocr-v2/commit/a92d75050f05c2638394e970bf8cec53c113d99b).

The patch includes current-stream CUDA launches, synchronization reduction,
batched relational geometry, and guarded exact detector fusions. It preserves
eager fallbacks when a layout or module does not satisfy a fusion contract.
It is also available for review in
[`nvidia/nemotron-ocr-v2` discussion #8](https://huggingface.co/nvidia/nemotron-ocr-v2/discussions/8).

Apply it to a clean model checkout:

```bash
MODEL_REPO=/path/to/nemotron-ocr-v2
git -C "$MODEL_REPO" checkout --detach \
  0e83e83f17943524b90afa6c0fd82ac2bc1a40ca
git -C "$MODEL_REPO" apply --check \
  "$(pwd)/benchmarks/nemotron_ocr_v2/nemotron_ocr_model_optimizations.patch"
git -C "$MODEL_REPO" apply \
  "$(pwd)/benchmarks/nemotron_ocr_v2/nemotron_ocr_model_optimizations.patch"
```

Rebuild the extension for A100 (`sm_80`) with the CUDA compiler matching the
PyTorch environment:

```bash
ROOT=/raid/vjawa/tmp/ocr_optimization
MODEL_REPO="$ROOT/nemotron-ocr-v2"
VENV="$ROOT/venv"
CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"

cd "$MODEL_REPO/nemotron-ocr"
rm -rf build src/nemotron_ocr_cpp/_nemotron_ocr_cpp*.so
CUDA_HOME="$CUDA_HOME" \
PATH="$CUDA_HOME/bin:$PATH" \
LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST=8.0 \
"$VENV/bin/python" scripts/build-extension.py

export NEMOTRON_OCR_SOURCE="$MODEL_REPO/nemotron-ocr/src"
```

## JPEG-byte dataset

`build_pooling_dataset.py` converts a sorted directory of document images into
vLLM custom-dataset JSONL. Each row carries one JPEG data URI in
`prompt.data`. The publication workload used Pillow quality 100 with
subsampling 0:

```bash
python benchmarks/nemotron_ocr_v2/build_pooling_dataset.py \
  --image-dir /path/to/document_images \
  --output /path/to/ocr_pooling_q100_444.jsonl \
  --limit 1000 \
  --payload-mode jpeg-bytes \
  --jpeg-quality 100 \
  --jpeg-subsampling 0 \
  --workers 8
```

The publication JSONL has SHA-256
`139c96ef75a85da440350722a95d9eb3bd21dd4155d43f7281253f63c07eaa16`.
Base64 and JPEG decoding remain inside the timed request path.

## Benchmark clients

`benchmark_hf_inprocess.py` runs the official `NemotronOCRV2` pipeline directly,
without vLLM. It accepts either the custom-dataset JSONL or a JPEG directory and
supports replaying the selected corpus to keep the GPU under sustained load:

```bash
python benchmarks/nemotron_ocr_v2/benchmark_hf_inprocess.py \
  --model-repo "$MODEL_REPO" \
  --dataset-jsonl /path/to/ocr_pooling.jsonl \
  --limit 1000 \
  --replay-count 30 \
  --batch-size 64 \
  --warmup 128 \
  --infer-length 1024 \
  --detector-max-batch-size 32 \
  --recognizer-chunk-size 128 \
  --relational-chunk-size 128 \
  --gpu-trace-device 0 \
  --gpu-trace-csv /path/to/results/gpu_trace.csv \
  --output-json /path/to/results/result.json
```

`benchmark_multi_endpoint_pooling.py` drives one work-conserving client queue
across already-running native vLLM `/pooling` endpoints. It validates the replay
count, traces the selected physical GPU, records endpoint-level latency and
completion, and exits nonzero if any timed request fails:

```bash
python benchmarks/nemotron_ocr_v2/benchmark_multi_endpoint_pooling.py \
  --endpoint http://127.0.0.1:8000 \
  --endpoint http://127.0.0.1:8001 \
  --dataset /path/to/ocr_pooling.jsonl \
  --model /path/to/nemotron-ocr-v2 \
  --num-prompts 30000 \
  --replay-count 30 \
  --concurrency-per-endpoint 64 \
  --warmups-per-endpoint 32 \
  --gpu 0 \
  --output-dir /path/to/results/multi_endpoint
```

## Native vLLM setting sweeps

`serve_sweep.json` and `bench_sweep.json` plug into
`vllm bench sweep serve`. Native `AsyncLLM` owns admission and continuous
batching while the sweep varies server and client settings. Use `--dry-run`
first to inspect the Cartesian product:

```bash
MODEL=/path/to/nemotron-ocr-v2
DATASET=/path/to/ocr_pooling_q100_444.jsonl
HF_OVERRIDES='{"model_type":"nemotron_ocr_v2","architectures":["NemotronOCRV2ForImageToText"],"nemotron_ocr_model_subdir":"v2_multilingual","nemotron_ocr_merge_level":"paragraph","nemotron_ocr_detector_max_batch_size":16,"nemotron_ocr_recognizer_chunk_size":128,"nemotron_ocr_relational_chunk_size":128,"nemotron_ocr_verbose_post":false,"nemotron_ocr_infer_length":1024}'

vllm bench sweep serve \
  --serve-cmd "vllm serve $MODEL --runner pooling --skip-tokenizer-init --io-processor-plugin nemotron_ocr_v2 --mm-processor-cache-gb 0 --enforce-eager --hf-overrides '$HF_OVERRIDES'" \
  --bench-cmd "vllm bench serve --backend vllm-pooling --base-url http://127.0.0.1:8000 --endpoint /pooling --model $MODEL --dataset-name custom --dataset-path $DATASET --skip-tokenizer-init --request-rate inf --disable-tqdm" \
  --serve-params benchmarks/nemotron_ocr_v2/serve_sweep.json \
  --bench-params benchmarks/nemotron_ocr_v2/bench_sweep.json \
  --output-dir /path/to/results/sweep \
  --experiment-name nemotron-ocr-v2 \
  --num-runs 3 \
  --after-bench-cmd true \
  --dry-run
```

The final multi-replica sweep adds replica count, per-replica MPS share, OCR
chunk sizes, and endpoint concurrency. Its exact harness, configs, resolved
commands, source snapshots, raw summaries, and traces are archived in the
[`final 85 images/s result bundle`](https://github.com/VibhuJawa/nemotron-vllm-ocr/tree/results/final-85-imgs/results/a100-2026-07-09-final-85-imgs).

The selected clean vLLM control is one replica with detector batch 16, four
renderer workers, `max_num_seqs=64`, and concurrency 128. The selected optimized
shape is eight MPS replicas with detector batch 16, four renderer workers,
recognizer and relational chunks 128, `max_num_seqs=40`, and concurrency 64 per
endpoint.
