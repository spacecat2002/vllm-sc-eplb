import json

import pytest
import torch

from moe_gate_lora.trainer import StreamingProcessor


class FakeRouter:
    def __init__(self, top_k):
        self.top_k = top_k
        self.calls = 0

    def _compute_routing(
        self,
        hidden_states,
        router_logits,
        indices_type,
        *,
        input_ids=None,
    ):
        del hidden_states, input_ids
        self.calls += 1
        weights, ids = torch.topk(router_logits, k=self.top_k, dim=-1)
        return weights.float(), ids.to(indices_type)


@pytest.fixture
def fake_routers(monkeypatch):
    routers = []

    def build(**kwargs):
        router = FakeRouter(kwargs["top_k"])
        routers.append(router)
        return router

    monkeypatch.setattr("moe_gate_lora.trainer.FusedTopKRouter", build)
    return routers


def _write_pair(root, batch, offset):
    hidden = torch.tensor(
        [[1.0 + offset, 0.5], [0.25, 1.0 + offset]], dtype=torch.float16
    )
    base = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    target = base + torch.tensor([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0]])
    source = {
        "format_version": 2,
        "rank": 0,
        "step": batch,
        "layer_id": 1,
        "next_gate_layer_id": 2,
        "activations": hidden,
        "next_gate_base_logits": base,
    }
    following = {
        "format_version": 2,
        "rank": 0,
        "step": batch,
        "layer_id": 2,
        "router_logits": target,
        "topk_ids": torch.topk(target, k=2, dim=-1).indices,
    }
    paths = [root / f"batch_{batch}_layer_1.pt", root / f"batch_{batch}_layer_2.pt"]
    torch.save(source, paths[0])
    torch.save(following, paths[1])
    return paths


def test_streaming_train_then_eval_without_retaining_records(
    tmp_path, monkeypatch, fake_routers
):
    monkeypatch.setattr(
        "moe_gate_lora.trainer.write_overlap_plot", lambda rows, output: None
    )
    lora_dir = tmp_path / "lora"
    train = StreamingProcessor(
        mode="train",
        output_dir=tmp_path / "train_metrics",
        lora_dir=lora_dir,
        rank_dim=1,
        alpha=1.0,
        lr=1e-2,
        weight_decay=0.0,
        seed=0,
        device="cpu",
    )
    for epoch in range(2):
        for batch in range(2):
            step = epoch * 2 + batch
            paths = _write_pair(tmp_path, step, float(batch))
            train.process(paths, epoch=epoch)
            assert not any(path.exists() for path in paths)
        train.finish_epoch(epoch)

    checkpoint = lora_dir / "layer_0001_to_0002.pt"
    assert checkpoint.exists()
    train_metrics = json.loads(
        (tmp_path / "train_metrics" / "metrics.json").read_text()
    )
    assert train_metrics["batches"] == 4
    assert train_metrics["epochs_completed"] == 2
    assert train_metrics["training"][0]["steps"] == 4
    adapter = train.adapters[2]
    assert adapter.optimizer is not None
    assert adapter.optimizer.state[adapter.lora_a]["step"].item() == 4
    assert fake_routers[0].calls == 8

    evaluate = StreamingProcessor(
        mode="eval",
        output_dir=tmp_path / "eval_metrics",
        lora_dir=lora_dir,
        rank_dim=1,
        alpha=1.0,
        lr=1e-2,
        weight_decay=0.0,
        seed=0,
        device="cpu",
    )
    paths = _write_pair(tmp_path, 3, 3.0)
    evaluate.process(paths, epoch=0)
    evaluate.finish_epoch(0)
    assert not any(path.exists() for path in paths)
    eval_metrics = json.loads((tmp_path / "eval_metrics" / "metrics.json").read_text())
    assert eval_metrics["rows"][0]["num_tokens"] == 2
    assert fake_routers[1].calls == 2


def test_epochs_must_be_finished_in_order(tmp_path, monkeypatch, fake_routers):
    del fake_routers
    monkeypatch.setattr(
        "moe_gate_lora.trainer.write_overlap_plot", lambda rows, output: None
    )
    processor = StreamingProcessor(
        mode="train",
        output_dir=tmp_path / "metrics",
        lora_dir=tmp_path / "lora",
        rank_dim=1,
        alpha=1.0,
        lr=1e-2,
        weight_decay=0.0,
        seed=0,
        device="cpu",
    )
    paths = _write_pair(tmp_path, 0, 0.0)
    processor.process(paths, epoch=0)
    processor.finish_epoch(0)

    with pytest.raises(ValueError, match="exactly once"):
        processor.finish_epoch(0)

    paths = _write_pair(tmp_path, 1, 1.0)
    with pytest.raises(ValueError, match="Expected epoch 1"):
        processor.process(paths, epoch=2)
    for path in paths:
        path.unlink()
