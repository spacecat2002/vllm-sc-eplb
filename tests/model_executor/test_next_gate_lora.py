# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.dual_gate_lora import (
    get_dual_gate_output_workspace,
)
from vllm.model_executor.layers.fused_moe.next_gate_lora import (
    combine_gate_weights,
    materialize_lora_delta,
)


def test_combined_gate_projection_matches_separate_lora_projection():
    hidden_states = torch.randn(5, 7)
    current_weight = torch.randn(3, 7)
    next_weight = torch.randn(4, 7)
    lora_a = torch.randn(2, 7)
    lora_b = torch.randn(4, 2)
    scale = 0.75

    combined_weight, current_size = combine_gate_weights(
        current_weight,
        next_weight,
        (lora_a, lora_b, scale),
    )
    combined_logits = F.linear(hidden_states, combined_weight)
    current_logits, predicted_logits = torch.split(
        combined_logits,
        [current_size, combined_logits.shape[-1] - current_size],
        dim=-1,
    )

    expected_current = F.linear(hidden_states, current_weight)
    expected_next = F.linear(hidden_states, next_weight)
    expected_next += ((hidden_states @ lora_a.T) @ lora_b.T) * scale
    torch.testing.assert_close(current_logits, expected_current)
    torch.testing.assert_close(predicted_logits, expected_next)


def test_materialized_lora_delta_stays_separate_from_base_weight():
    base_weight = torch.randn(4, 7)
    original = base_weight.clone()
    lora_a = torch.randn(2, 7)
    lora_b = torch.randn(4, 2)

    delta = materialize_lora_delta(base_weight, (lora_a, lora_b, 0.5))

    torch.testing.assert_close(delta, (lora_b @ lora_a) * 0.5)
    torch.testing.assert_close(base_weight, original)
    assert delta.data_ptr() != base_weight.data_ptr()


def test_dual_gate_output_workspace_is_shared_and_contiguous():
    first_current, first_next = get_dual_gate_output_workspace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        slot=101,
        num_tokens=4,
        current_experts=3,
        next_experts=5,
    )
    second_current, second_next = get_dual_gate_output_workspace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        slot=101,
        num_tokens=2,
        current_experts=4,
        next_experts=6,
    )

    assert first_current.untyped_storage().data_ptr() == (
        second_current.untyped_storage().data_ptr()
    )
    assert first_next.untyped_storage().data_ptr() == (
        second_next.untyped_storage().data_ptr()
    )
    assert second_current.is_contiguous()
    assert second_next.is_contiguous()


def test_dual_gate_output_workspace_requires_fixed_dtype_per_slot():
    get_dual_gate_output_workspace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        slot=102,
        num_tokens=1,
        current_experts=1,
        next_experts=1,
    )

    with pytest.raises(ValueError, match="device and dtype must remain fixed"):
        get_dual_gate_output_workspace(
            device=torch.device("cpu"),
            dtype=torch.float16,
            slot=102,
            num_tokens=1,
            current_experts=1,
            next_experts=1,
        )
