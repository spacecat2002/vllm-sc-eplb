# Benchmarking MoE Dual Batch Overlap

This guide compares online serving throughput with MoE Dual Batch Overlap
(DBO) disabled and enabled. It uses a synthetic, decode-oriented workload so
that both runs receive the same token lengths and concurrency.

DBO targets deployments that combine data parallelism and expert parallelism.
For details about how DBO works, its requirements, and its thresholds, see
[Dual Batch Overlap](../design/dbo.md).

## Prerequisites

- At least two visible NVIDIA GPUs.
- DeepEP installed in the same environment as vLLM.
- A MoE model that supports expert parallelism.
- The same vLLM commit, model weights, GPU clocks, and idle GPUs for both runs.

The example below uses two GPUs and `deepseek-ai/DeepSeek-V2-Lite`. Adjust the
model and parallelism settings for your hardware, but keep them identical
between the baseline and DBO runs.

## Common configuration

Set these variables in each terminal used for the test:

```bash
export MODEL=deepseek-ai/DeepSeek-V2-Lite
export CUDA_VISIBLE_DEVICES=0,1
export DP_SIZE=2
export A2A_BACKEND=deepep_low_latency
export RESULT_DIR=./moe-dbo-results
mkdir -p "$RESULT_DIR"
```

The following optional settings use synthetic, approximately balanced expert
routing. They help isolate DBO and communication performance from expert load
imbalance. Omit them when the goal is to measure the model's natural routing
distribution instead.

```bash
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
```

## 1. Run the baseline without DBO

Start the server in the first terminal. DBO is disabled by default, so this
command deliberately omits `--enable-dbo`.

```bash
vllm serve "$MODEL" \
  --trust-remote-code \
  --data-parallel-size "$DP_SIZE" \
  --enable-expert-parallel \
  --all2all-backend "$A2A_BACKEND" \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --dbo-decode-token-threshold 16 \
  --dbo-prefill-token-threshold 256
```

After the server is ready, run the benchmark in a second terminal:

```bash
vllm bench serve \
  --backend vllm \
  --model "$MODEL" \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 512 \
  --random-range-ratio 0 \
  --seed 0 \
  --temperature 0 \
  --num-warmups 64 \
  --num-prompts 1024 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename baseline.json \
  --metadata mode=baseline a2a_backend="$A2A_BACKEND" dp_size="$DP_SIZE"
```

Stop the baseline server after the benchmark completes. Wait for its worker
processes to exit before starting the DBO server.

## 2. Run with DBO enabled

Restart the server with the same arguments and add only `--enable-dbo`:

```bash
vllm serve "$MODEL" \
  --trust-remote-code \
  --data-parallel-size "$DP_SIZE" \
  --enable-expert-parallel \
  --all2all-backend "$A2A_BACKEND" \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --dbo-decode-token-threshold 16 \
  --dbo-prefill-token-threshold 256 \
  --enable-dbo
```

Run the identical workload and save it under a different name:

```bash
vllm bench serve \
  --backend vllm \
  --model "$MODEL" \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 512 \
  --random-range-ratio 0 \
  --seed 0 \
  --temperature 0 \
  --num-warmups 64 \
  --num-prompts 1024 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename dbo.json \
  --metadata mode=dbo a2a_backend="$A2A_BACKEND" dp_size="$DP_SIZE"
```

`--seed 0` and `--temperature 0` keep the inputs and decoding policy fixed,
while `--ignore-eos` keeps the generated length fixed. `--request-rate inf`
offers all requests immediately, while `--max-concurrency` bounds the number
in flight. With 64 concurrent decode requests, scheduled decode batches
should exceed the 16-token DBO threshold for most of the measured run.

## 3. Compare results

The console output reports request, output-token, and total-token throughput.
The same values can be read from the saved JSON files:

```bash
jq -s -r '
  ["mode", "request_req_s", "output_tok_s", "total_tok_s"],
  (.[] | [
    .mode,
    .request_throughput,
    .output_throughput,
    .total_token_throughput
  ]) | @tsv
' "$RESULT_DIR/baseline.json" "$RESULT_DIR/dbo.json"
```

Calculate the DBO-to-baseline throughput ratios with:

```bash
jq -s '
  {
    request_throughput_speedup:
      (.[1].request_throughput / .[0].request_throughput),
    output_token_throughput_speedup:
      (.[1].output_throughput / .[0].output_throughput),
    total_token_throughput_speedup:
      (.[1].total_token_throughput / .[0].total_token_throughput)
  }
' "$RESULT_DIR/baseline.json" "$RESULT_DIR/dbo.json"
```

A ratio greater than `1.0` means that DBO improved that metric. For a stable
comparison, repeat each mode at least three times, alternate their order, and
compare the median throughput. Use a unique result filename for every repeat.

## Prefill-oriented variant

For a workload dominated by prompt processing, repeat both the baseline and
DBO runs with the high-throughput DeepEP backend:

```bash
export A2A_BACKEND=deepep_high_throughput
```

In both benchmark commands, replace the synthetic lengths with a longer prompt
and shorter output:

```bash
--random-input-len 2048 \
--random-output-len 128
```

Do not compare a `deepep_low_latency` baseline against a
`deepep_high_throughput` DBO run. The all-to-all backend, workload, thresholds,
and every other server and benchmark argument must remain fixed within each
DBO comparison.
