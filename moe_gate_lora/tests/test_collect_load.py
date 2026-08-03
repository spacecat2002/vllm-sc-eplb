# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import pytest

from moe_gate_lora import cli
from moe_gate_lora.collect import _archive_trace_batch


def test_archive_trace_batch_preserves_rank_boundaries(tmp_path):
    active = tmp_path / "activations"
    paths = []
    for rank in range(2):
        rank_dir = active / f"rank_{rank:05d}"
        rank_dir.mkdir(parents=True)
        path = rank_dir / "step_000003_layer_0004.pt"
        path.write_text(f"rank {rank}", encoding="utf-8")
        paths.append(path)

    archived = _archive_trace_batch(
        paths,
        tmp_path / "traces",
        epoch=1,
        batch=2,
    )

    assert [path.read_text(encoding="utf-8") for path in archived] == [
        "rank 0",
        "rank 1",
    ]
    assert all(not path.exists() for path in paths)
    assert archived[0].relative_to(tmp_path) == (
        tmp_path / "traces/epoch_0001/batch_000002/rank_00000/step_000003_layer_0004.pt"
    ).relative_to(tmp_path)


def test_archive_trace_batch_refuses_to_overwrite(tmp_path):
    active = tmp_path / "activations" / "rank_00000"
    active.mkdir(parents=True)
    first_source = active / "step_000000_layer_0000.pt"
    conflicting_source = active / "step_000000_layer_0001.pt"
    first_source.write_text("first", encoding="utf-8")
    conflicting_source.write_text("new", encoding="utf-8")
    destination = (
        tmp_path / "traces/epoch_0000/batch_000000/rank_00000/step_000000_layer_0001.pt"
    )
    destination.parent.mkdir(parents=True)
    destination.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _archive_trace_batch(
            [first_source, conflicting_source],
            tmp_path / "traces",
            epoch=0,
            batch=0,
        )

    assert first_source.read_text(encoding="utf-8") == "first"
    assert conflicting_source.read_text(encoding="utf-8") == "new"
    assert destination.read_text(encoding="utf-8") == "old"


def test_collect_load_cli_selects_trace_mode(tmp_path, monkeypatch):
    captured = {}

    def fake_collect(config):
        captured["config"] = config
        return {"trace_dir": str(tmp_path / "traces")}

    monkeypatch.setattr(cli, "collect", fake_collect)
    args = cli.parse_args(
        [
            "collect-load",
            "--model",
            "test-model",
            "--output-dir",
            str(tmp_path),
            "--ep-size",
            "2",
            "--max-num-batched-tokens",
            "512",
        ]
    )

    assert isinstance(args, argparse.Namespace)
    args.func(args)
    config = captured["config"]
    assert config.mode == "trace"
    assert config.lora_dir is None
    assert config.ep_size == 2
    assert config.max_num_batched_tokens == 512
