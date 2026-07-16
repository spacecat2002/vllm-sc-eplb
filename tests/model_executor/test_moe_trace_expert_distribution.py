import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from examples.basic.offline_inference.moe_trace_expert_distribution import (
    _aggregate_trace,
    _load_datasets,
)
from vllm.model_executor.layers.fused_moe.moe_trace import (
    MoETraceCollector,
    MoETraceConfig,
)


def test_trace_mode_is_loaded_and_validated_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_MOE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MOE_TRACE_MAX_STEPS", "2")
    monkeypatch.setenv("VLLM_MOE_TRACE_MODE", "expert_distribution")

    config = MoETraceConfig.from_env()

    assert config is not None
    assert config.mode == "expert_distribution"
    monkeypatch.setenv("VLLM_MOE_TRACE_MODE", "invalid")
    with pytest.raises(ValueError, match="VLLM_MOE_TRACE_MODE"):
        MoETraceConfig.from_env()


def test_local_dataset_overrides_preserve_domain_prompt_format(tmp_path):
    math_path = tmp_path / "math.jsonl"
    math_path.write_text(
        "\n".join(
            [json.dumps({"question": "What is 1 + 1?"}), json.dumps("Compute 2 + 2")]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        datasets=["math"],
        dataset_path=[f"math={math_path}"],
        num_prompts=2,
    )

    prompts = _load_datasets(args)["math"]

    assert prompts[0].startswith("What is 1 + 1?")
    assert "\\boxed{}" in prompts[0]
    assert prompts[1] == "Compute 2 + 2"


def test_trace_aggregation_sums_ranks_before_normalizing(tmp_path):
    experiment_dir = tmp_path / "dataset_math" / "batch_0002"
    (experiment_dir / "activations" / "rank_00000").mkdir(parents=True)
    (experiment_dir / "activations" / "rank_00001").mkdir(parents=True)
    (experiment_dir / "metadata.json").write_text(
        json.dumps(
            {
                "num_experts": 3,
                "request_batches": {
                    "0:request-0": 0,
                    "0:request-1": 0,
                    "1:request-2": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        (
            0,
            torch.tensor([2, 1, 1], dtype=torch.int64),
            2,
            ["request-0", "request-1"],
        ),
        (
            1,
            torch.tensor([1, 0, 1], dtype=torch.int64),
            1,
            ["request-2"],
        ),
    ]
    for rank, expert_counts, num_scheduled_tokens, request_ids in records:
        record = {
            "mode": "expert_distribution",
            "rank": rank,
            "step": 0,
            "layer_id": 4,
            "expert_counts": expert_counts,
            "num_scheduled_tokens": num_scheduled_tokens,
            "request_ids": request_ids,
        }
        path = (
            experiment_dir
            / "activations"
            / f"rank_{rank:05d}"
            / "step_000000_layer_0004.pt"
        )
        torch.save(record, path)

    distribution = _aggregate_trace(experiment_dir)

    layer = distribution.layers[4]
    np.testing.assert_array_equal(layer.totals, [3, 1, 2])
    np.testing.assert_allclose(layer.shares[0], [50.0, 100 / 6, 100 / 3])
    np.testing.assert_array_equal(layer.batch_indices, [0])
    np.testing.assert_allclose(layer.imbalance, [1.5])
    np.testing.assert_array_equal(layer.scheduled_tokens, [3])


def test_trace_preserves_scheduled_count_when_tensors_are_truncated(tmp_path):
    collector = MoETraceCollector(
        MoETraceConfig(
            output_dir=tmp_path,
            max_steps=1,
            capture_next_gate_base_logits=False,
        ),
        {4: "model.layers.4.mlp"},
        {},
    )
    num_tokens = 4100
    collector.begin_forward(
        num_scheduled_tokens=[num_tokens],
        num_computed_tokens=[0],
        prefill_lengths=[num_tokens],
        request_ids=["request-0"],
    )
    hidden_states = torch.zeros((num_tokens, 1))
    router_logits = torch.zeros((num_tokens, 2))
    topk_ids = torch.zeros((num_tokens, 1), dtype=torch.int64)
    topk_weights = torch.ones((num_tokens, 1))

    collector.capture(4, hidden_states, router_logits, topk_weights, topk_ids)

    record = torch.load(
        tmp_path / "rank_00000" / "step_000000_layer_0004.pt",
        weights_only=True,
    )
    assert record["mode"] == "lora_training"
    assert record["topk_ids"].shape[0] == 4096
    assert record["num_scheduled_tokens"] == num_tokens


def test_distribution_mode_saves_exact_counts_without_training_tensors(tmp_path):
    collector = MoETraceCollector(
        MoETraceConfig(
            output_dir=tmp_path,
            max_steps=1,
            capture_next_gate_base_logits=False,
            mode="expert_distribution",
        ),
        {4: "model.layers.4.mlp"},
        {},
    )
    num_tokens = 4100
    collector.begin_forward(
        num_scheduled_tokens=[num_tokens],
        num_computed_tokens=[0],
        prefill_lengths=[num_tokens],
        request_ids=["request-0"],
    )
    hidden_states = torch.zeros((num_tokens, 1))
    router_logits = torch.zeros((num_tokens, 2))
    topk_ids = torch.zeros((num_tokens, 1), dtype=torch.int64)
    topk_weights = torch.ones((num_tokens, 1))

    collector.capture(4, hidden_states, router_logits, topk_weights, topk_ids)

    record = torch.load(
        tmp_path / "rank_00000" / "step_000000_layer_0004.pt",
        weights_only=True,
    )
    assert record["mode"] == "expert_distribution"
    assert record["num_scheduled_tokens"] == num_tokens
    assert torch.equal(record["expert_counts"], torch.tensor([num_tokens, 0]))
    assert "topk_ids" not in record
    assert "router_logits" not in record
    assert "activations" not in record
