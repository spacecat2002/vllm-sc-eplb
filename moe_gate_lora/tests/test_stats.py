import torch

from moe_gate_lora.stats import RunningMoments, topk_overlap


def test_running_moments_matches_materialized_statistics():
    chunks = [torch.tensor([0.0, 0.5]), torch.tensor([1.0, 0.5, 1.0])]
    moments = RunningMoments()
    for chunk in chunks:
        moments.update(chunk)

    expected = torch.cat(chunks).double()
    result = moments.as_dict()
    assert result["count"] == expected.numel()
    assert result["mean"] == expected.mean().item()
    assert result["std"] == expected.std(correction=0).item()


def test_topk_overlap_counts_set_intersection():
    predicted = torch.tensor([[1, 2], [3, 4]])
    actual = torch.tensor([[2, 5], [3, 4]])

    assert torch.equal(topk_overlap(predicted, actual), torch.tensor([0.5, 1.0]))
