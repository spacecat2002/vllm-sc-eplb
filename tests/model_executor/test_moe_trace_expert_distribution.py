import json
from argparse import Namespace

import numpy as np
import pytest
import torch

from examples.basic.offline_inference.moe_trace_expert_distribution import (
    _aggregate_trace,
    _load_datasets,
    _sort_expert_counts,
    _top_n_expert_coverage,
    _validate_top_n_experts,
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


def test_trace_aggregation_sums_ranks_per_step_and_steps_per_rank(tmp_path):
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
                    "0:request-3": 1,
                    "1:request-2": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        (
            0,
            0,
            torch.tensor([2, 1, 1], dtype=torch.int64),
            2,
            ["request-0", "request-1"],
        ),
        (
            1,
            0,
            torch.tensor([1, 0, 1], dtype=torch.int64),
            1,
            ["request-2"],
        ),
        (
            0,
            1,
            torch.tensor([1, 2, 0], dtype=torch.int64),
            1,
            ["request-3"],
        ),
    ]
    for rank, step, expert_counts, num_scheduled_tokens, request_ids in records:
        record = {
            "mode": "expert_distribution",
            "rank": rank,
            "step": step,
            "layer_id": 4,
            "expert_counts": expert_counts,
            "num_scheduled_tokens": num_scheduled_tokens,
            "request_ids": request_ids,
        }
        path = (
            experiment_dir
            / "activations"
            / f"rank_{rank:05d}"
            / f"step_{step:06d}_layer_0004.pt"
        )
        torch.save(record, path)

    distribution = _aggregate_trace(experiment_dir)

    layer = distribution.layers[4]
    np.testing.assert_array_equal(layer.totals, [4, 3, 2])
    np.testing.assert_array_equal(layer.rank_totals[0], [3, 3, 1])
    np.testing.assert_array_equal(layer.rank_totals[1], [1, 0, 1])
    np.testing.assert_allclose(layer.shares[0], [50.0, 100 / 6, 100 / 3])
    np.testing.assert_allclose(layer.shares[1], [100 / 3, 200 / 3, 0])
    np.testing.assert_array_equal(layer.batch_indices, [0, 1])
    np.testing.assert_allclose(layer.imbalance, [1.5, 2.0])
    np.testing.assert_array_equal(layer.scheduled_tokens, [3, 1])


def test_expert_counts_are_sorted_descending_with_stable_expert_ids():
    expert_ids, counts = _sort_expert_counts(np.asarray([4, 7, 7, 0, 2]))

    np.testing.assert_array_equal(expert_ids, [1, 2, 0, 4, 3])
    np.testing.assert_array_equal(counts, [7, 7, 4, 2, 0])


def test_top_n_expert_coverage_counts_token_expert_assignments():
    coverage = _top_n_expert_coverage(np.asarray([4, 7, 7, 0, 2]), [1, 3, 5])

    assert coverage[1] == {
        "selected_expert_ids": [1],
        "token_expert_assignments": 7,
        "assignment_share_percent": 35.0,
    }
    assert coverage[3]["selected_expert_ids"] == [1, 2, 0]
    assert coverage[3]["token_expert_assignments"] == 18
    assert coverage[3]["assignment_share_percent"] == 90.0
    assert coverage[5]["token_expert_assignments"] == 20
    assert coverage[5]["assignment_share_percent"] == 100.0


def test_top_n_experts_are_deduplicated_and_range_checked():
    assert _validate_top_n_experts([3, 1, 3], 4) == [3, 1]
    with pytest.raises(ValueError, match="must be in"):
        _validate_top_n_experts([0, 5], 4)


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
