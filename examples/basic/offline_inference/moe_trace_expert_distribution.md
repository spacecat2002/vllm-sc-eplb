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
  --top-n-experts 8 16 32 \
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
each dataset/batch-size combination produces one forward-series PNG and one
rank-total PNG per EP rank. The example above produces separate files such as
`expert_distribution_math_batch_0001_layer_0023.png` and
`expert_counts_math_batch_0001_layer_0023_rank_00000.png`. Datasets, batch
sizes, layers, and ranks never share a figure. Passing multiple layer IDs
explicitly produces both kinds of plot for every requested layer. To redraw
selected combinations without running inference again:

```bash
.venv/bin/python \
  examples/basic/offline_inference/moe_trace_expert_distribution.py plot \
  --work-dir /tmp/qwen3_expert_distribution \
  --datasets math code \
  --batch-sizes 1 8 \
  --layers 23 \
  --top-n-experts 8 16 32 \
  --max-steps 100
```

## Collect and solve without retaining traces

The `collect-solve` command captures real token-level `topk_ids`, runs the fast
placement and routing solver as soon as each dataset/batch-size experiment
finishes, and then removes the temporary trace. Only the plan JSON files under
`--solver-output-dir` are retained. If that option is omitted, plans are written
to `./moe_solver_plans`:

```bash
.venv/bin/python \
  examples/basic/offline_inference/moe_trace_expert_distribution.py \
  collect-solve \
  --model Qwen/Qwen3-30B-A3B \
  --datasets math \
  --batch-sizes 8 \
  --num-prompts 32 \
  --ep-size 4 \
  --max-new-tokens 16 \
  --solver-layers 3 7 11 15 23 \
  --solver-step 0 \
  --solver-redundant-slots 2 \
  --solver-min-quota 8 \
  --solver-route-slices 16 \
  --solver-redistribution-iters 4 \
  --solver-redistribution-min-quota 8 \
  --solver-output-dir /tmp/qwen3_moe_plans
```

For this example, one plan is retained for every requested layer, such as
`plan_math_batch_0008_layer_0003_step_000000.json` and
`plan_math_batch_0008_layer_0023_step_000000.json`. The singular alias
`--solver-layer 23` remains available for one-layer runs. Omitting
`--solver-step` selects the first captured model-forward step. Collection stops
after the selected step, although every MoE layer up to that point is
temporarily captured because all layers share the same router trace hook.

Each output compares three plans with one evaluator: the original layout,
UltraEP, and the independent `joint_balanced` redistribution/reroute heuristic.
`joint_balanced` starts
without replicas, tests single-expert and co-occurring top-k expert bundles,
screens them with a low-cost route, and fully reroutes only the best few
candidates in congestion-aware chunks. It does not use the UltraEP placement
or quotas as its starting point. Increase
`--solver-redundant-slots` when co-locating larger top-k bundles is acceptable;
`--solver-route-slices` trades solver time for finer routing splits.
Every plan JSON is accompanied by a `_rank_loads.png` figure comparing UltraEP
and `joint_balanced` compute assignments, remote sends, and remote receives for
every rank.

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
`torch.bincount(..., minlength=num_experts)` before copying the count vector to
CPU. Collection also writes `topk_ids` when `trace_config.json` contains
`capture_topk_ids=true`; this is enabled by the collection command in this
document so local-bypass optimization can use real token tuples. Older
count-only records store `expert_counts`, `num_scheduled_tokens`, request IDs,
rank, step, and layer, but do not store token-level top-k IDs.

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

Each rank-total figure separately sums all forward steps for one rank and one
layer, then sorts bars by assignment count from largest to smallest. The x-axis
labels retain the original logical expert IDs. Ties are ordered by expert ID.
Layers are never combined because the same logical ID in two layers identifies
different experts. The counts are token-expert assignments: with top-k routing,
one token contributes once to every selected expert.

Passing `--top-n-experts 8 16 32` also reports cumulative coverage for the 8,
16, and 32 most-loaded experts. For each rank and layer, experts are selected
from the same trace-wide descending order used by the bars. The plot annotation
and console output show both the covered assignment count and its percentage of
all assignments on that rank. Because count-only traces do not retain per-token
top-k combinations, this is not a count of unique tokens that hit at least one
selected expert.

For every Top-N value, the same record also reports how many selected experts
are local to that rank: `local_expert_count`, `local_expert_share_percent`, and
`local_expert_ids`. Expert ownership follows vLLM's configured placement
strategy. New collections store `expert_placement_strategy`; existing traces
without that field are interpreted as the default `linear` placement. With
EPLB or redundant physical experts, logical trace IDs are insufficient to
reconstruct the dynamic physical placement, so these local-expert fields should
not be used as physical EPLB placement metrics.

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
counts and percentages for every expert and layer. Its per-rank entries include
expert IDs and counts in the same descending order as the rank-total plots.
When `--top-n-experts` is provided, `top_n_expert_coverage` stores the selected
expert IDs, covered assignment count, and coverage percentage for each N.
Per-rank entries additionally store the selected local expert IDs and their
count and percentage within Top-N.
This aggregate is useful for quantitative checks; the plots emphasize changes
over scheduler forwards.

## Simulate replicated experts from an existing trace

`moe_trace_replica_simulation.py` replays the recorded per-forward,
per-source-rank token top-k tuples. It does not start vLLM or collect another
trace. For example:

```bash
.venv/bin/python \
  examples/basic/offline_inference/moe_trace_replica_simulation.py \
  --work-dir /tmp/qwen3_expert_distribution \
  --datasets math \
  --batch-sizes 4 \
  --layers 23 \
  --extra-replicas 0 4 8 16 \
  --communication-weights 0.1 0.3 1 3 10
```

The work directory must contain the `manifest.json`, `metadata.json`, and
`activations/rank_*` files produced by the collection command. Omit datasets,
batch sizes, or layers to use the values available in the manifest and trace.
Results are written under `WORK_DIR/replica_simulation` by default:

- `replica_simulation.json` contains all placements and detailed metrics.
- `replica_simulation.csv` contains one row per policy and replica budget.
- `replica_pareto_*.png` plots communication latency against compute latency.

The simulator compares three greedy policies. `communication_first` places
copies to maximize fully local tokens and minimize remote token transfers,
`balance_first` places copies to minimize the maximum token-expert assignment
load on any rank, and `joint` sweeps the supplied communication weights. The
optimizer routes every token's top-k tuple to a feasible replica combination. Use
`--candidate-limit 0` to evaluate every possible expert/rank copy; the default
uses locality and load shortlists.

Replica candidates are shortlisted from trace-wide aggregate demand. Candidate
placements are scored by replaying every recorded forward separately, preserving
source-rank demand and temporal load variation. `--extra-replicas` is a budget,
not a requirement: `used_extra_replicas` can be smaller when a shorter placement
prefix has the best objective.

For large traces, layout search can be bounded without changing the reported
metrics. `--search-max-tokens N` uses an evenly spaced subset of at most `N`
tokens while choosing placements; the selected layout is then replayed on the
complete trace for the output metrics. For example:

```bash
.venv/bin/python \
  examples/basic/offline_inference/moe_trace_replica_simulation.py \
  --work-dir /tmp/qwen3_expert_distribution \
  --datasets math \
  --batch-sizes 4 \
  --layers 23 \
  --extra-replicas 0 4 8 \
  --communication-weights 0.3 1 3 \
  --candidate-limit 8 \
  --search-max-tokens 4096
```

The default `--search-max-tokens 0` preserves exhaustive trace replay during
search. A smaller `--candidate-limit` also reduces the number of candidate
expert/rank placements evaluated at each greedy step.

`estimated_compute_latency_ms` sums the maximum expert-input token count across
ranks for each recorded forward. This is the token-expert assignment count used
by the real expert kernel. `estimated_communication_latency_ms` charges the sum
of remote target-rank transfers for every token, including dispatch and combine.
A token sent to two remote ranks contributes two communication units, regardless
of how many experts it has on each rank. `remote_tokens` is reported separately
as the number of tokens that cannot use the fully-local path.
`estimated_serial_latency_ms` assumes no compute/communication overlap, while
`estimated_overlap_lower_bound_ms` takes the larger component in each forward
and represents ideal overlap.
`balanced_compute_lower_bound_ms` assumes every forward's token-expert
assignments can be divided evenly among all ranks. `communication_lower_bound_ms`
independently chooses the least-communication replica combination for every
token under the selected layout.
The Pareto flag and dashed frontier use the two estimated latency dimensions;
replica count is reported as a resource cost but is not a Pareto objective.

The default coefficients are normalized values of one microsecond per expert
input token or communication unit. They are useful for comparing policies but
are not measured GPU latencies. Supply calibrated coefficients to estimate
hardware time:

```bash
.venv/bin/python \
  examples/basic/offline_inference/moe_trace_replica_simulation.py \
  --work-dir /tmp/qwen3_expert_distribution \
  --compute-us-per-token 0.08 \
  --communication-us-per-token 0.15
```

Count-only traces preserve the number of token-expert assignments but not the
top-k expert tuple for each token. The simulator then synthesizes valid token
rows from the counts. This preserves expert counts but cannot recover the
original token co-occurrence; local-token and remote-token-transfer metrics are
therefore approximate for old traces. New collections with `capture_topk_ids`
use the original token tuples.

## Replay the trace with real GPU operators

After generating `replica_simulation.json`, the distributed replay benchmark
measures the selected placements with the local-bypass path, DeepEP-HT for
non-local tokens, and the vLLM Triton expert kernel. Launch one process per
trace EP rank:

```bash
.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  --module benchmarks.kernels.benchmark_moe_trace_replay \
  --work-dir /tmp/qwen3_expert_distribution \
  --dataset math \
  --batch-size 4 \
  --layer 23 \
  --policies baseline communication_first balance_first joint \
  --extra-replicas 0 8 \
  --communication-weights 0.3 1 3 \
  --execution-mode local_bypass \
  --warmup 2 \
  --iters 5
```

`WORLD_SIZE` must equal the trace EP size. The benchmark loads the chosen
replica placements, gives every copy a physical expert slot on its target rank,
and repeats every recorded forward. For each forward it measures dispatch,
expert compute, and combine separately, then uses the maximum rank time for
each stage. The reported replay time is the sum of those stage critical paths
over all selected forwards. JSON, CSV, and a measured Pareto plot are written
next to `replica_simulation.json` as `operator_timing_*` files.

The replay also selects the configuration with the lowest measured
`serial_ms` and writes three `operator_timing_*_best_expert_load` files. The
PNG stacks each logical expert's assignment load by target rank, with experts
sorted from highest to lowest total load. The rank panel shows expert-input token
counts, which drive compute balance, alongside unique target-rank token counts,
which describe communication fanout. A star marks experts with an extra replica.
The CSV has one row per expert with its load on every rank, replica ranks, and
extra replica ranks. The JSON preserves both rank-load metrics, the assignment
matrix, and the exact best-configuration metadata. Expert loads count
token-expert assignments, so one top-k token contributes once to each selected
expert.

To generate these files from an existing `operator_timing_*.json` without
rerunning DeepEP or the expert kernel, run the module once without `torchrun`:

```bash
.venv/bin/python -m benchmarks.kernels.benchmark_moe_trace_replay \
  --work-dir /tmp/qwen3_expert_distribution \
  --dataset math \
  --batch-size 4 \
  --layer 23 \
  --plot-only
```

The default input is the `operator_timing_*` file selected by
`--execution-mode`. Pass `--output-json /path/to/operator_timing.json` when the
timing result has a custom path. Plot-only mode reconstructs the selected
configuration's assignment matrix from the original trace and
`replica_simulation.json`; it does not require CUDA or distributed launch.

All points selected in one invocation use the same padded physical expert
capacity per rank: the largest capacity required by any selected point. Unused
slots receive no tokens. This keeps kernel metadata shapes identical across
policies. To measure an unpadded native baseline, run a separate invocation
with `--policies baseline`.

Trace replay always uses `local_bypass`: a token whose entire top-k target set
is its source rank uses the local MoE kernel; all other tokens use the DeepEP
path. Placement optimization uses the same token-level criterion. The
communication objective counts remote target-rank transfers only. The balance
objective uses expert-input token count per target rank,
matching the `received_tokens` metric and the work executed by the expert kernel;
unique target-rank token counts are also reported separately.

The benchmark uses random BF16 activations and weights with the model's real
MoE dimensions, because weight values do not change the kernel shape. Built-in
dimensions are available for the Qwen3 presets listed in the benchmark. For a
local model directory, the benchmark reads `config.json` and recognizes the
usual `hidden_size`, `moe_intermediate_size`, and `num_experts_per_tok` fields,
including fields nested under `text_config`. If a custom configuration uses
different names, pass `--hidden-size`, `--intermediate-size`, and `--top-k`.

These are real operator timings, but still not end-to-end inference latency.
For count-only traces, replay synthesizes token rows from exact expert counts;
expert assignment counts remain exact, but local-token and remote-transfer
coalescing are approximate. A trace containing token-level top-k IDs is
required for exact local-bypass communication behavior.
