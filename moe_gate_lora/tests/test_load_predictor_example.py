# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from moe_gate_lora.load_predictor_example import (
    RouteSetLoadPredictor,
    fit_transition_matrix,
    load_trace_samples,
    synthetic_chunked_samples,
    train_predictor,
    transition_prediction,
)


def test_predictor_starts_from_transition_and_learns_residual():
    samples = synthetic_chunked_samples(
        num_samples=24,
        num_experts=8,
        seed=1,
    )
    assert len({sample.current_ids.shape[0] for sample in samples}) > 1
    transition, _ = fit_transition_matrix(samples, num_experts=8)
    predictor = RouteSetLoadPredictor(
        num_experts=8,
        embedding_dim=8,
        hidden_dim=16,
    )
    sample = samples[0]
    base = transition_prediction(sample, transition)

    with torch.no_grad():
        initial = predictor(sample.current_ids, sample.current_weights, base)
    torch.testing.assert_close(initial, base)

    losses = train_predictor(
        predictor,
        samples,
        transition,
        epochs=4,
        learning_rate=1e-2,
        replica_slots=2,
        seed=0,
        device=torch.device("cpu"),
    )
    assert losses[-1] < losses[0]


def test_loads_adjacent_layer_samples_from_full_trace(tmp_path):
    for step in range(2):
        source_ids = torch.tensor([[0, 1], [1, 2]])
        target_ids = torch.tensor([[2, 3], [3, 0]])
        source = {
            "rank": 0,
            "step": step,
            "layer_id": 1,
            "next_gate_layer_id": 2,
            "topk_ids": source_ids,
            "topk_weights": torch.full((2, 2), 0.5),
        }
        target = {
            "rank": 0,
            "step": step,
            "layer_id": 2,
            "topk_ids": target_ids,
            "router_logits": torch.zeros(2, 5),
        }
        torch.save(source, tmp_path / f"step_{step}_layer_1.pt")
        torch.save(target, tmp_path / f"step_{step}_layer_2.pt")

    samples, num_experts, pair = load_trace_samples(tmp_path, source_layer=1)

    assert pair == (1, 2)
    assert num_experts == 5
    assert len(samples) == 2
    torch.testing.assert_close(samples[0].current_weights, torch.full((2, 2), 0.5))
