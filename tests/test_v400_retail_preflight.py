"""Strict read-only retail preflight tests; no original game is executed."""
import hashlib
import json
from pathlib import Path

from tools.rl.v400_retail_preflight import RETAIL_SHA, inspect


def fixture_inputs(tmp_path: Path):
    target = tmp_path / "th08.exe"
    target.write_bytes(b"research")
    replays = tmp_path / "replays"
    replays.mkdir()
    rows = []
    for name in ("one.rpy", "two.rpy"):
        data = name.encode()
        (replays / name).write_bytes(data)
        rows.append({"id": name, "file": name, "sha256": hashlib.sha256(data).hexdigest()})
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps({"targetSha256": RETAIL_SHA, "fixtures": rows}), encoding="utf-8")
    data = tmp_path / "th08.dat"
    data.write_bytes(b"data")
    bgm = tmp_path / "thbgm.dat"
    bgm.write_bytes(b"bgm")
    loader = tmp_path / "wine"
    loader.write_bytes(b"loader")
    return target, fixtures, replays, loader, data, bgm


def test_research_build_never_passes_as_retail(tmp_path):
    code, result = inspect(*fixture_inputs(tmp_path))
    assert code == 2
    assert result["capture_count"] == 0
    assert not result["target"]["retail_identity_verified"]
    assert result["gaps"] == ["target_not_original_japanese_th08_1.00d"]


def test_missing_original_is_blocked(tmp_path):
    inputs = fixture_inputs(tmp_path)
    inputs[0].unlink()
    code, result = inspect(*inputs)
    assert code == 3
    assert result["gaps"] == ["retail_target_missing"]


def test_valid_identity_still_requires_replay_and_tools(tmp_path, monkeypatch):
    inputs = fixture_inputs(tmp_path)
    import tools.rl.v400_retail_preflight as preflight
    monkeypatch.setattr(preflight, "RETAIL_SIZE", len(b"original"))
    monkeypatch.setattr(preflight, "RETAIL_SHA", hashlib.sha256(b"original").hexdigest())
    inputs[0].write_bytes(b"original")
    inputs[1].write_text(json.dumps({"targetSha256": preflight.RETAIL_SHA, "fixtures": [
        {"id": name, "file": name, "sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in ("one.rpy", "two.rpy")]}), encoding="utf-8")
    (inputs[2] / "two.rpy").unlink()
    inputs[3].unlink()
    monkeypatch.setattr(preflight.shutil, "which", lambda _: None)
    code, report = inspect(*inputs)
    assert code == 3
    assert report["target"]["retail_identity_verified"]
    assert "fixture_missing_or_hash_mismatch:two.rpy" in report["gaps"]
    assert "tool_missing:wine_loader_present" in report["gaps"]
    assert report["capture_count"] == 0


def test_manifest_path_traversal_rejected(tmp_path, monkeypatch):
    inputs = fixture_inputs(tmp_path)
    import tools.rl.v400_retail_preflight as preflight
    monkeypatch.setattr(preflight, "RETAIL_SIZE", len(b"original"))
    monkeypatch.setattr(preflight, "RETAIL_SHA", hashlib.sha256(b"original").hexdigest())
    inputs[0].write_bytes(b"original")
    inputs[1].write_text(json.dumps({"targetSha256": preflight.RETAIL_SHA, "fixtures": [
        {"id": "bad", "file": "../escape.rpy", "sha256": "x"},
        {"id": "ok", "file": "two.rpy", "sha256": "x"}]}), encoding="utf-8")
    code, report = inspect(*inputs)
    assert code == 3
    assert "unsafe_fixture_name" in report["gaps"]
