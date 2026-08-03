# MoE Gate LoRA

This package trains a side-channel next-gate LoRA without materializing the
full activation/logit dataset.

```bash
.venv/bin/python -m moe_gate_lora pipeline \
  --model Qwen/Qwen3-30B-A3B \
  --prompts /path/to/train_prompts.txt \
  --eval-prompts /path/to/eval_prompts.txt \
  --work-dir /tmp/moe_gate_lora \
  --ep-size 4 \
  --epochs 3 \
  --batch-size 2
```

For each inference batch, all EP ranks pause after writing their trace. The
parent process performs one optimizer step per adjacent MoE layer pair,
updates exact running overlap statistics and the plot, saves the small LoRA
checkpoints, deletes the consumed trace tensors, and releases the ranks for
the next batch.

Training keeps one vLLM worker set, one set of adapters, and one optimizer
across all epochs. Each epoch regenerates the trace in the same prompt order;
only the current batch is retained. The evaluation phase starts fresh vLLM
workers, runs once, keeps vLLM's internal SC-EPLB/LoRA path disabled, and
applies the final trained LoRA in this package while consuming each trace
batch. Its final mean and standard deviation are mathematically identical to
loading all evaluation records and computing the same metrics afterward.

This experiment targets Qwen3 MoE and assumes every layer uses the standard
vLLM `FusedTopKRouter` with softmax and renormalization. Training creates one
router from the first batch's top-k width and expert count, shares it across
all layers, and calls `_compute_routing()` instead of raw `torch.topk(logits)`.
The trace therefore does not serialize router types or parameters.

The trained parameters match an offline run only when it uses the same epoch
count, batch boundaries, order, and regenerated samples. Shuffled training is
not currently implemented.

The vLLM-internal `next_gate_lora.py` remains available for future serving-side
experiments, but this package never enables or imports it.

## Aggregate load-prediction example

`load_predictor_example.py` tests a small DeepSets-style predictor that maps
the current layer's complete Top-K route set to the next layer's normalized
expert load. It predicts a residual over an expert transition matrix and
reports metrics that match replica prefetching: hot-expert recall and routed
token coverage at a fixed replica-slot budget.

The default experiment generates scheduler forwards with token counts sampled
from 8 to 256, so absolute load changes while the prediction target remains a
normalized distribution:

```bash
.venv/bin/python -m moe_gate_lora.load_predictor_example \
  --samples 200 \
  --epochs 20 \
  --replica-slots 4
```

To train on a collected full trace, select an adjacent source layer. The trace
must use `VLLM_MOE_TRACE_MODE=lora_training`; count-only
`expert_distribution` traces do not contain the per-token Top-K combinations
used by the network.

```bash
.venv/bin/python -m moe_gate_lora.load_predictor_example \
  --trace-dir /tmp/moe_gate_lora/trace \
  --source-layer 10 \
  --epochs 20 \
  --replica-slots 4
```

The vLLM integration now has five environment variables. The pipeline manages
the first three and forcibly disables the latter two in its workers:

- `VLLM_MOE_TRACE_DIR`: per-worker batch trace directory.
- `VLLM_MOE_TRACE_MAX_STEPS`: bounded number of forwards to record.
- `VLLM_MOE_TRACE_MODE`: `lora_training` for the full training trace or
  `expert_distribution` for count-only expert-load records. This pipeline
  always selects `lora_training`.
- `VLLM_SC_EPLB`: enable the future serving-side next-gate predictor.
- `VLLM_SC_EPLB_LORA_DIR`: optional serving-side LoRA checkpoint directory.
