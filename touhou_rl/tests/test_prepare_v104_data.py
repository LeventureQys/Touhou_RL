"""Tests for ``tools/rl/prepare_v104_data.py`` (S3-1 CLI contract).

The tool is the only supported way to produce the S3-1 evidence, so its exit
codes and refusal rules are part of the contract:

* known contract/manifest/IO failures print ``v104:<CODE>: ...`` to stderr and
  exit 2; a usage error is argparse's exit 2; anything else keeps its traceback;
* ``audit`` writes a JSON report, ``cache`` writes a build + verification report,
  ``eval-manifest`` writes a ``DatasetManifest`` and *reuses* an identical one
  while refusing a different one;
* ``eval-manifest`` is refused for a dataset that already has ``manifest.json``
  and never writes a dataset's default manifest path;
* a report mode refuses to overwrite a file that is already a ``DatasetManifest``.

The tests drive ``main([...])`` in-process (the returned int *is* the exit code);
the real end-to-end runs on A/B were executed from a shell and are recorded in the
S3-1 completion report.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl import obs, vision
from touhou_rl.pipeline.manifest import index_digest, split_indices
from touhou_rl.vision_data import VisionShardWriter

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "rl" / "prepare_v104_data.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("prepare_v104_data", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_tool()

META_FIELDS = ("frame", "stage", "score", "graze", "lives", "bombs", "power", "deaths")


def _regions(value: int) -> dict[str, bytes]:
    sizes = {"playfield": (12, 16), "hud_score": (4, 8), "hud_status": (6, 8)}
    return {
        name: vision.encode_png(np.full((height, width, 3), (value * 5) % 250, dtype=np.uint8))
        for name, (height, width) in sizes.items()
    }


def _observation(seed: int = 0) -> obs.Observation:
    rng = np.random.default_rng(seed)
    return obs.Observation(
        self_feat=rng.random(obs.N_SELF, dtype=np.float32),
        bullets=rng.random((obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32).astype(np.float16),
        danger_map=rng.integers(0, 3, size=(obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
    )


def _make_dataset(root: Path, shards: int = 3, frames: int = 4) -> Path:
    """A tiny but structurally real vision dataset (``shards`` npz files)."""
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    for shard in range(shards):
        writer = VisionShardWriter(root, shard_size=frames, tag=f"collect-{shard:05d}")
        for row in range(frames):
            writer.add(
                _regions(index),
                _observation(index),
                (0, 1, 0, 0),
                {
                    "frame": float(index * 2),
                    "stage": 0,
                    "score": 0,
                    "graze": 0,
                    "lives": 2,
                    "bombs": 2,
                    "power": 0.0,
                    "deaths": 0,
                },
            )
            index += 1
        writer.close()
    return root


def _run(argv: list[str]) -> int:
    return prepare.main(argv)


# --- argument handling ------------------------------------------------------
def test_unknown_mode_is_a_usage_error(tmp_path, capsys):
    dataset = _make_dataset(tmp_path / "data")
    with pytest.raises(SystemExit) as excinfo:
        _run(["--dataset", str(dataset), "--mode", "bogus", "--out", str(tmp_path / "x.json")])
    assert excinfo.value.code == 2


def test_missing_dataset_directory_is_exit_two(tmp_path, capsys):
    code = _run([
        "--dataset", str(tmp_path / "nowhere"), "--mode", "audit", "--out", str(tmp_path / "x.json")
    ])
    assert code == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("v104:"), captured.err
    assert not (tmp_path / "x.json").exists()


# --- audit mode -------------------------------------------------------------
def test_audit_mode_writes_a_recomputable_report(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    out = tmp_path / "evidence" / "history.json"
    code = _run([
        "--dataset", str(dataset), "--mode", "audit", "--depth", "3", "--out", str(out)
    ])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["mode"] == "audit"
    assert report["history_contract"] == "causal-segment-v1"
    assert report["depth"] == 3
    assert report["frames"] == 12
    assert report["shards"] == 3
    assert report["manifest"] == {}  # this dataset has no manifest.json
    assert report["decode_cache"]["hits"] == 0  # the audit reads meta only
    assert report["repeated_history_ratio"] == pytest.approx(
        report["repeated_history_samples"] / report["frames"]
    )


def test_audit_mode_depth_one(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    out = tmp_path / "history-d1.json"
    assert _run(["--dataset", str(dataset), "--mode", "audit", "--depth", "1", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["depth"] == 1
    assert report["repeated_history_samples"] == 0


def test_audit_mode_rejects_other_depths(tmp_path, capsys):
    dataset = _make_dataset(tmp_path / "data")
    code = _run(["--dataset", str(dataset), "--mode", "audit", "--depth", "2", "--out", str(tmp_path / "h.json")])
    assert code == 2
    assert capsys.readouterr().err.startswith("v104:CONFIG: ")


def test_audit_mode_rejects_unknown_history_contract(tmp_path, capsys):
    dataset = _make_dataset(tmp_path / "data")
    code = _run([
        "--dataset", str(dataset), "--mode", "audit",
        "--history-contract", "causal-segment-v2", "--out", str(tmp_path / "h.json"),
    ])
    assert code == 2
    assert capsys.readouterr().err.startswith("v104:CONFIG: ")


# --- eval-manifest mode -----------------------------------------------------
def _eval_manifest(tmp_path, dataset: Path, out: Path) -> int:
    return _run(["--dataset", str(dataset), "--mode", "eval-manifest", "--out", str(out)])


def test_eval_manifest_is_read_only_and_complete(tmp_path):
    dataset = _make_dataset(tmp_path / "data", shards=3, frames=4)
    out = tmp_path / "evidence" / "eval-manifest.json"
    assert _eval_manifest(tmp_path, dataset, out) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["modality"] == "vision"
    assert payload["total_frames"] == 12
    assert payload["has_teacher_values"] is False
    assert payload["teacher"] == {}
    assert payload["stats"]["evaluation_only"] is True
    assert payload["provenance"]["purpose"] == "external-evaluation-only"
    assert payload["splits"]["seed"] == 7
    assert payload["splits"]["val_fraction"] == 0.1
    names = [record["name"] for record in payload["shards"]]
    assert names == sorted(names)  # stable filename order
    assert [record["frames"] for record in payload["shards"]] == [4, 4, 4]
    for record in payload["shards"]:
        path = dataset / record["name"]
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == prepare.file_digest(path)
    _train, val = split_indices(12, 7, 0.1)
    assert payload["splits"]["val_frames"] == len(val)
    assert payload["splits"]["val_index_sha256"] == index_digest(val)
    # nothing else was written into the dataset directory
    assert not (dataset / "manifest.json").exists()


def test_eval_manifest_reuses_identical_content(tmp_path, capsys):
    dataset = _make_dataset(tmp_path / "data")
    out = tmp_path / "eval-manifest.json"
    assert _eval_manifest(tmp_path, dataset, out) == 0
    first = out.read_bytes()
    first_summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert first_summary["reused"] is False
    assert _eval_manifest(tmp_path, dataset, out) == 0
    second_summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert second_summary["reused"] is True
    assert out.read_bytes() == first


def test_eval_manifest_refuses_inconsistent_existing_output(tmp_path, capsys):
    dataset = _make_dataset(tmp_path / "data")
    out = tmp_path / "eval-manifest.json"
    out.write_text(json.dumps({"version": 1, "shards": [], "splits": {}}), encoding="utf-8")
    code = _eval_manifest(tmp_path, dataset, out)
    assert code == 2
    assert capsys.readouterr().err.startswith("v104:EVIDENCE: ")
    assert json.loads(out.read_text(encoding="utf-8")) == {"version": 1, "shards": [], "splits": {}}


def test_eval_manifest_is_refused_for_a_dataset_with_a_manifest(tmp_path, capsys):
    dataset = _make_dataset(tmp_path / "data")
    (dataset / "manifest.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    code = _eval_manifest(tmp_path, dataset, tmp_path / "eval-manifest.json")
    assert code == 2
    assert capsys.readouterr().err.startswith("v104:CONFIG: ")


def test_eval_manifest_refuses_the_dataset_default_manifest_path(tmp_path, capsys):
    dataset = _make_dataset(tmp_path / "data")
    code = _eval_manifest(tmp_path, dataset, dataset / "manifest.json")
    assert code == 2
    assert capsys.readouterr().err.startswith("v104:EVIDENCE: ")
    assert not (dataset / "manifest.json").exists()


# --- cache mode -------------------------------------------------------------
def test_cache_mode_builds_and_verifies(tmp_path):
    dataset = _make_dataset(tmp_path / "data", shards=2, frames=3)
    out = tmp_path / "evidence" / "cache.json"
    assert _run(["--dataset", str(dataset), "--mode", "cache", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["mode"] == "cache"
    assert report["cache_contract"] == "v104-decode-cache/1"
    assert report["sizes"] == {
        "playfield": [224, 192],
        "hud_score": [34, 216],
        "hud_status": [126, 216],
    }
    assert report["precheck"]["free_bytes_before"] > 0
    assert report["precheck"]["required_bytes"] > 0
    build = report["build"]
    assert build["files_written"] == 6  # 2 shards x 3 regions
    assert build["files_invalid"] == 0 and build["files_repaired"] == 0
    assert build["files_present"] == 6
    assert build["bytes_total"] > 0
    verification = report["verification"]
    assert verification["files_ok"] == verification["files"] == 6
    assert verification["files_invalid"] == 0
    assert verification["payload_verified"] is True
    assert report["manifest"] == {}


def test_cache_mode_is_idempotent(tmp_path):
    dataset = _make_dataset(tmp_path / "data", shards=2, frames=3)
    out = tmp_path / "cache.json"
    assert _run(["--dataset", str(dataset), "--mode", "cache", "--out", str(out)]) == 0
    first = json.loads(out.read_text(encoding="utf-8"))["build"]
    assert _run(["--dataset", str(dataset), "--mode", "cache", "--out", str(out)]) == 0
    second = json.loads(out.read_text(encoding="utf-8"))["build"]
    assert first["files_written"] == 6
    assert second["files_written"] == 0
    assert second["files_skipped"] == first["files_written"]
    assert second["bytes_total"] == first["bytes_total"]


def test_cache_mode_limit_shards_is_a_smoke_run(tmp_path):
    dataset = _make_dataset(tmp_path / "data", shards=3, frames=3)
    out = tmp_path / "cache.json"
    assert _run([
        "--dataset", str(dataset), "--mode", "cache", "--limit-shards", "1", "--out", str(out)
    ]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["build"]["files_written"] == 3
    assert report["build"]["shards_visited"] == 1
    assert report["verification"]["files"] == 3


def test_cache_mode_refuses_to_overwrite_a_manifest(tmp_path, capsys):
    dataset = _make_dataset(tmp_path / "data")
    out = tmp_path / "manifest.json"
    out.write_text(
        json.dumps({"version": 1, "shards": [], "splits": {"seed": 7}}), encoding="utf-8"
    )
    code = _run(["--dataset", str(dataset), "--mode", "cache", "--out", str(out)])
    assert code == 2
    assert capsys.readouterr().err.startswith("v104:EVIDENCE: ")
    assert json.loads(out.read_text(encoding="utf-8"))["version"] == 1


def test_cache_mode_refuses_when_free_space_is_insufficient(tmp_path, capsys, monkeypatch):
    """The free-space precheck must fail *before* writing, and never delete data."""
    dataset = _make_dataset(tmp_path / "data")
    real_usage = prepare.shutil.disk_usage

    def tiny_free(path):
        usage = real_usage(path)
        return type(usage)(total=usage.total, used=usage.used, free=1024)

    monkeypatch.setattr(prepare.shutil, "disk_usage", tiny_free)
    out = tmp_path / "cache.json"
    code = _run(["--dataset", str(dataset), "--mode", "cache", "--out", str(out)])
    assert code == 2
    assert "v104:BUDGET:" in capsys.readouterr().err
    assert not out.exists()
    assert not list(dataset.glob("*.decoded-*.npy"))  # nothing was written


def test_unexpected_failure_keeps_the_traceback(tmp_path, capsys, monkeypatch):
    dataset = _make_dataset(tmp_path / "data")

    def explode(*_args, **_kwargs):
        raise TypeError("boom")

    monkeypatch.setattr(prepare, "build_decode_cache", explode)
    out = tmp_path / "cache.json"
    code = _run(["--dataset", str(dataset), "--mode", "cache", "--out", str(out)])
    assert code == 1
    assert "Traceback" in capsys.readouterr().err
    assert not out.exists()
