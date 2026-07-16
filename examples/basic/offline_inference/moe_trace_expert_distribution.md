# MoE token-to-expert distribution

`moe_trace_expert_distribution.py` compares logical expert load across
datasets and per-rank inference batch sizes. Each output figure contains one
dataset, one batch size, and one MoE layer. The upper panel shows
token-to-expert distribution, the middle panel shows load imbalance as a line
plot, and the lower panel shows the scheduled-token count as a line plot. The
script uses the opt-in MoE trace in
`vllm/model_executor/layers/fused_moe/moe_trace.py` and always runs vLLM in
eager mode.

## Built-in datasets

The four built-in names are loaded through Hugging Face `datasets` in
streaming mode, so only the requested examples are read:

| Name | Source | Prompt field |
| --- | --- | --- |
| `math` | `openai/gsm8k` | `question` |
| `code` | `openai/openai_humaneval` | `prompt` |
| `chat` | `HuggingFaceH4/mt_bench_prompts` | first `prompt` turn |
| `summary` | `abisee/cnn_dailymail` | `article` |

Install the optional benchmark dependencies before using remote datasets:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e '.[bench]' --torch-backend=auto
```

## Collect and plot

This example compares all four datasets at batch sizes 1, 2, 4, and 8, using
32 prompts from each dataset and four expert-parallel ranks:

```bash
.venv/bin/python \
  examples/basic/offline_inference/moe_trace_expert_distribution.py collect \
  --model Qwen/Qwen3-30B-A3B \
  --datasets math code chat summary \
  --batch-sizes 1 2 4 8 \
  --num-prompts 32 \
  --ep-size 4 \
  --max-new-tokens 16 \
  --layers 23 \
  --output-dir /tmp/qwen3_expert_distribution
```

`--batch-sizes` is the number of requests passed to each `llm.generate()` call
on each EP/DP rank. The effective cluster-wide upper bound is therefore
`batch_size * ep_size`. It is not the number of tokens in a model forward.
Scheduler token limits and short final shards can make the actual
model-forward batch smaller. `--max-new-tokens` is kept explicit so runs use
the same generation length when comparing batch sizes.

Each dataset/batch-size combination starts new vLLM workers and is stored at:

```text
/tmp/qwen3_expert_distribution/
  dataset_math/batch_0001/activations/rank_00000/*.pt
  dataset_math/batch_0002/activations/rank_00000/*.pt
  dataset_code/batch_0001/activations/rank_00000/*.pt
  ...
```

If `--layers` is omitted, the script plots only the first common MoE layer, so
each dataset/batch-size combination produces exactly one PNG. The example
above produces separate files such as
`expert_distribution_math_batch_0001_layer_0023.png` and
`expert_distribution_math_batch_0008_layer_0023.png`; datasets and batch sizes
never share a figure. Passing multiple layer IDs explicitly produces one file
per layer. To redraw selected combinations without running inference again:

```bash
.venv/bin/python \
  examples/basic/offline_inference/moe_trace_expert_distribution.py plot \
  --work-dir /tmp/qwen3_expert_distribution \
  --datasets math code \
  --batch-sizes 1 8 \
  --layers 23 \
  --max-steps 100
```

## Local or custom datasets

`--dataset-path NAME=PATH` accepts UTF-8 TXT, JSON, or JSONL. TXT uses one
prompt per non-empty line. JSON/JSONL recognizes plain strings and common
fields such as `prompt`, `text`, `question`, `article`, `document`,
`messages`, and ShareGPT-style `conversations`.

Paths can override built-in datasets, which is useful on offline machines:

```bash
.venv/bin/python \
  examples/basic/offline_inference/moe_trace_expert_distribution.py collect \
  --model Qwen/Qwen3-30B-A3B \
  --datasets math code chat summary \
  --dataset-path math=/data/gsm8k.jsonl \
  --dataset-path code=/data/humaneval.jsonl \
  --dataset-path chat=/data/sharegpt.json \
  --dataset-path summary=/data/cnn_dailymail.jsonl \
  --batch-sizes 1 4 16 \
  --num-prompts 64 \
  --output-dir /tmp/local_expert_distribution
```

Arbitrary names are also supported when a path is supplied:

```bash
.venv/bin/python \
  examples/basic/offline_inference/moe_trace_expert_distribution.py collect \
  --model Qwen/Qwen3-30B-A3B \
  --datasets finance medical \
  --dataset-path finance=/data/finance.txt \
  --dataset-path medical=/data/medical.jsonl \
  --batch-sizes 1 8 \
  --output-dir /tmp/domain_expert_distribution
```

## How the figure is produced

For every `(dataset, batch size, layer, raw model-forward step)` group, the
script selects `VLLM_MOE_TRACE_MODE=expert_distribution`. Inside the router
trace callback, vLLM flattens logical `[num_tokens, top_k]` IDs and applies
`torch.bincount(..., minlength=num_experts)` before copying the small count
vector to CPU. A token routed to top-k experts therefore contributes one
token-expert assignment to each selected expert. Each count-only record stores
`expert_counts`, `num_scheduled_tokens`, request IDs, rank, step, and layer; it
does not store activations, router logits, top-k IDs, or top-k weights.

For compatibility, the plotting code can still read an older full trace. If a
record has no `expert_counts`, it falls back to applying `numpy.bincount` to
the saved `topk_ids`.

The count for expert `e` at a step is normalized as:

```text
share[e] = count[e] / sum(count[all experts]) * 100
```

The middle panel is a line plot using the unnormalized counts from the same
forward. A dotted horizontal reference line marks perfect balance at `1`:

```text
imbalance = max(count[all experts]) / mean(count[all experts])
```

The mean includes every logical expert, including zero-load experts. A value
of `1` is perfectly balanced; larger values mean that the hottest expert has
more work relative to the per-expert average.

The third panel is another line plot. For each forward, it sums the trace
record's `num_scheduled_tokens` across EP ranks. New traces store this value
before the 4096-token tensor-selection cap, so the line reflects scheduler
load rather than trace-file truncation. When plotting an older trace without
this field, the script falls back to `topk_ids.shape[0]` and therefore shows
only the number of recorded tokens.

All EP ranks are summed before normalization. The figure then uses a stacked
area plot: y is 0-100% of routed assignments and each colored band is one
logical expert ID. Expert colors are deterministic and remain identical
across every dataset and batch-size figure. IDs are captured before EPLB
logical-to-physical remapping.

All three panels use the same model-forward index on the x axis. This is a
contiguous `0..N-1` index over the recorded forwards, not the outer batch index
and not a token position. The dashed vertical lines mark changes in the outer
`llm.generate()` batch index. This preserves the individual chunked-prefill
forwards while still showing where one request batch ends and the next begins.

New count-only traces aggregate every scheduled routing row internally, so
none of the three panels is subject to the 4096-token full-trace tensor cap.
When reading an older full trace, the first two panels can still reflect its
truncated `topk_ids`. Keep the scheduler configuration fixed when comparing
runs.

Alongside PNG files, `expert_distribution.json` stores forward indices, the
corresponding raw model-forward steps and batch indices, the per-forward
`max_over_mean` and `scheduled_token_counts` series, plus total assignment
counts and percentages for every expert and layer. This aggregate is useful
for quantitative checks; the plots emphasize changes over scheduler forwards.
