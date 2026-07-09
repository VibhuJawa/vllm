# Nemotron OCR v2 model optimization patch

The vLLM benchmark's optimized result requires the external model changes in
`nemotron_ocr_model_optimizations.patch`. These changes are not part of the
vLLM source tree.

The exact model repository base is
`0e83e83f17943524b90afa6c0fd82ac2bc1a40ca` from
`nvidia/nemotron-ocr-v2`.

The matched A100 results, raw JSON/CSV telemetry, exact 1,000-page input
manifest, and final charts are published in
[`VibhuJawa/nemotron-vllm-ocr`](https://github.com/VibhuJawa/nemotron-vllm-ocr/tree/main/results/a100-2026-07-08).
The vLLM-native model/plugin and queueing changes live on
[`agent/nemotron-ocr-v2-port`](https://github.com/VibhuJawa/vllm/tree/agent/nemotron-ocr-v2-port);
this directory carries the external model patch and reproducibility drivers
needed to recreate the optimized deployment.

From the vLLM repository root, apply the patch to a clean model checkout:

```bash
MODEL_REPO=/path/to/nemotron-ocr-v2
git -C "$MODEL_REPO" checkout --detach 0e83e83f17943524b90afa6c0fd82ac2bc1a40ca
git -C "$MODEL_REPO" apply --check \
  "$(pwd)/benchmarks/nemotron_ocr_v2/nemotron_ocr_model_optimizations.patch"
git -C "$MODEL_REPO" apply \
  "$(pwd)/benchmarks/nemotron_ocr_v2/nemotron_ocr_model_optimizations.patch"
```

Rebuild the model extension for an A100 (`sm_80`) using the CUDA compiler that
matches the PyTorch environment. The following is the exact layout used by the
A100 benchmark environment; adjust `ROOT` only if the checkout was relocated:

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

## Dataset creation

`build_pooling_dataset.py` converts a sorted directory of document images into
vLLM custom-dataset JSONL. Each row contains one JPEG data URI under
`prompt.data`. The exact 1,000-image Q100, 4:4:4 recipe used for the benchmark
is Pillow quality `100` with subsampling `0`:

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

## Reproducibility drivers

`benchmark_hf_inprocess.py` runs `NemotronOCRV2` directly, without vLLM. It
accepts either a vLLM custom-dataset JSONL file or a directory of JPEGs. JSONL
JPEG payloads remain compressed bytes until the timed model pipeline, and
`--replay-count` repeats the selected image set. The result records model Git
provenance, input representation, failures through process exit, timing, peak
memory, and optional predictions and `nvidia-smi` traces. An optimized checkout
created by applying the patch above is intentionally dirty, so pass
`--allow-dirty-model-repo`; its exact dirty status is still recorded.

```bash
python benchmarks/nemotron_ocr_v2/benchmark_hf_inprocess.py \
  --model-repo "$MODEL_REPO" \
  --dataset-jsonl /path/to/ocr_pooling.jsonl \
  --limit 1000 \
  --replay-count 1 \
  --batch-size 32 \
  --allow-dirty-model-repo \
  --gpu-trace-device 0 \
  --gpu-trace-csv /path/to/results/hf_gpu_trace.csv \
  --predictions-json /path/to/results/hf_predictions.json \
  --output-json /path/to/results/hf_summary.json
```

`benchmark_multi_endpoint_pooling.py` drives a work-conserving client queue
across already-running native vLLM `/pooling` endpoints. It reads the same
custom-dataset JSONL request bodies, requires `--num-prompts` to equal the row
count times `--replay-count`, traces the explicitly selected physical GPU, and
writes endpoint-level completion/latency data plus aggregate failures and GPU
metrics. It exits nonzero if any timed request fails.

```bash
python benchmarks/nemotron_ocr_v2/benchmark_multi_endpoint_pooling.py \
  --endpoint http://127.0.0.1:8000 \
  --endpoint http://127.0.0.1:8001 \
  --dataset /path/to/ocr_pooling.jsonl \
  --model nvidia/nemotron-ocr-v2 \
  --num-prompts 2000 \
  --replay-count 2 \
  --gpu 0 \
  --output-dir /path/to/results/multi_endpoint
```

## vLLM-native parameter sweeps

The included `serve_sweep.json` and `bench_sweep.json` plug directly into
`vllm bench sweep serve`. They sweep `max_num_seqs`, renderer workers, and
client concurrency while native `AsyncLLM` owns admission and continuous
batching inside the `/pooling` server. Use `--dry-run` first to inspect the
Cartesian product and substitute your model and dataset paths:

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
