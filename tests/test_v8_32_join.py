#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline tests for ``tools/rl/v8_32_join.py`` — 0 engine starts, 0 git calls.

The fixtures are **synthetic but contract-shaped**: a baseline shard is written
exactly like ``.tmp_v420_data/T04/dataset/<shard>`` (``v420-dataset-2f/1/slot``
rows + ``v420-dataset-2f/1/meta`` with the obs file table) and a relabel output
exactly like ``tools/rl/v8_32_relabel.py run`` (``v832-dagger-relabel-label/1``
rows + ``v832-dagger-relabel/1`` meta whose ``source.root`` is the ``--save-obs``
directory).  Every asserted rule is therefore the real one; only the bytes are
small.

The downstream test imports ``.tmp_v832_readout01/readout01.py`` and
``.tmp_v832_s2e/s2e_train.py`` itself (read-only) and pushes the built corpus
through ``readout01.Corpus`` + ``s2e_train.merge_shards`` + ``view_corpus`` +
``dataset_2f.check_targets_record`` — the functions the training run calls.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from touhou_rl.v420 import contracts as C  # noqa: E402

READOUT01 = REPO / ".tmp_v832_readout01" / "readout01.py"
S2E_TRAIN = REPO / ".tmp_v832_s2e" / "s2e_train.py"

PLAY_SHAPE = (4, 112, 96)
HUD_SHAPE = (56, 160)
HIST_SHAPE = (8, 18)
N_CAND = 18


def _load_module(path: str | Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


JOIN = _load_module(REPO / "tools" / "rl" / "v8_32_join.py", "v8_32_join_under_test")
J = JOIN


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _history(row_index: int) -> list[int]:
    return [(row_index * 3 + step) % N_CAND for step in range(8)]


def _act_hist(rows: int) -> np.ndarray:
    array = np.zeros((rows, *HIST_SHAPE), dtype=np.uint8)
    for index in range(rows):
        for step, cand in enumerate(_history(index)):
            array[index, step, cand] = 1
    return array


def _write_planes(obs_dir: Path, rows: int, seed: int) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    planes = {
        "play96x112.npy": rng.integers(0, 256, size=(rows, *PLAY_SHAPE), dtype=np.uint8),
        "hud.npy": rng.integers(0, 256, size=(rows, *HUD_SHAPE), dtype=np.uint8),
        "act_hist.npy": _act_hist(rows),
    }
    obs_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Any]] = {}
    for name, array in planes.items():
        path = obs_dir / name
        np.save(path, array, allow_pickle=False)
        written[name] = {"path": path, "sha256": sha256_file(path),
                         "size": int(path.stat().st_size), "array": array}
    written["play96x112.npy"]["shape"] = list(planes["play96x112.npy"].shape)
    return written


def _write_jsonl(path: Path, payloads: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def _scores() -> list[float]:
    return [100.0 + index for index in range(N_CAND)]


def _verified(chosen: int | None, count: int = 1) -> list[int]:
    mask = [0] * N_CAND
    if chosen is not None and count:
        mask[chosen] = 1
    return mask


# ---------------------------------------------------------------------------
# a T04 baseline shard
# ---------------------------------------------------------------------------
def make_baseline(root: Path, *, seed: int = 10, rows: int = 5, first_frame: int = 16,
                  shard_key: str | None = None, bomb: int = 0, cand: int = 7,
                  unverified_rows: Sequence[int] = (), pre_base: int = 1000,
                  plane_rows: int | None = None) -> dict[str, Any]:
    root = Path(root)
    shard_key = shard_key or f"d1-s0d1-n3300-seed{seed}-shard"
    obs = _write_planes(root / "obs", int(plane_rows if plane_rows is not None else rows),
                        seed * 7 + 1)
    records = []
    for index in range(rows):
        frame = first_frame + 2 * index
        unverified = index in set(unverified_rows)
        chosen = None if unverified else cand
        mask = _verified(chosen)
        scores = _scores()
        records.append({
            "schema": "v420-dataset-2f/1/slot",
            "dataset_version": "v420-2f-dataset/1",
            "contract": "v420-contract/1",
            "obs_contract": "obs-v420-2f",
            "action_contract": "action-v420-2f",
            "shard_key": shard_key,
            "episode_seed": seed,
            "slot_id": index,
            "decision_frame": frame,
            "commit_frames": [frame, frame + 2],
            "duration_frames": 2,
            "obs_render_frames": [frame - 4 + step for step in range(4)],
            "history_frames": [frame - 16 + 2 * step for step in range(8)],
            "history": _history(index),
            "executed_cand": cand,
            "bomb": bomb,
            "mask": C.cand_to_mask(cand, bomb),
            "executed": {"cand": cand, "bomb": bomb,
                         "mask": C.cand_to_mask(cand, bomb), "source": "step_frames"},
            "scores": scores,
            "verified_mask": mask,
            "verified_count": sum(mask),
            "chosen": chosen,
            "chosen_verified": chosen is not None,
            "chosen_score": None if chosen is None else scores[chosen],
            "teacher_bomb": bomb,
            "margin": 16.0,
            "label_source": "teacher_2f",
            "label_verified": chosen is not None,
            "hash_pre": f"{pre_base + index:016x}",
            "hash_post": f"{pre_base + index + 1:016x}",
            "obs": {"play": "obs/play96x112.npy", "hud": "obs/hud.npy",
                    "act_hist": "obs/act_hist.npy", "row": index,
                    "hud_frame": frame - 1},
            "provenance": {
                "join_key": {"shard_key": shard_key, "slot_id": index},
                "corpus": {"slot_id": index, "decision_frame": frame,
                           "obs_row": index, "block_index": 4,
                           "label_source": "teacher_slot0" if index % 2 == 0
                           else "executed_slot1"},
                "label": {"slot_id": index, "decision_frame": frame,
                          "teacher_chosen": chosen},
            },
        })
    _write_jsonl(root / "slots.jsonl", records)
    _write_jsonl(root / "targets.jsonl", [{
        "schema": "v420-dataset-2f/1/target",
        "dataset_version": "v420-2f-dataset/1",
        "shard_key": shard_key,
        "episode_seed": seed,
        "slot_id": int(row["slot_id"]),
        "decision_frame": int(row["decision_frame"]),
        "executed_cand": int(row["executed_cand"]),
        "chosen_verified_cand": row["chosen"],
        "chosen_verified": bool(row["chosen_verified"]),
        "bomb": int(row["bomb"]),
    } for row in records])
    _write_json(root / "meta.json", {
        "schema": "v420-dataset-2f/1/meta",
        "dataset_version": "v420-2f-dataset/1",
        "counts": {"seed": seed, "shard_key": shard_key, "slots": rows,
                   "label_source_counts": {"teacher_2f": rows}},
        "obs": {"mode": "copy", "files": [{
            "name": f"obs/{name}",
            "source": obs[name]["path"].as_posix(),
            "target": None,
            "kind": "file",
            "sha256": obs[name]["sha256"],
            "size": obs[name]["size"],
        } for name in ("play96x112.npy", "hud.npy", "act_hist.npy")]},
    })
    return {"root": root, "rows": rows, "obs": obs, "records": records,
            "shard_key": shard_key}


# ---------------------------------------------------------------------------
# a v8_32_relabel.py output
# ---------------------------------------------------------------------------
def make_dagger(out: Path, *, seed: int = 12, rows: int = 3, first_frame: int = 64,
                teacher_bomb: int = 0, executed_bomb: int = 0, cand: int = 7,
                chosen_cand: int | None = None,
                unverified_rows: Sequence[int] = (), pre_base: int = 2000,
                obs_rows: int | None = None, root: Path | None = None) -> dict[str, Any]:
    out = Path(out)
    save_obs = out.parent / (out.name + "_save_obs")
    obs = _write_planes(save_obs / "obs",
                        int(obs_rows if obs_rows is not None else rows),
                        seed * 13 + 3)
    student_rows = [{"schema": "v832-takeover-obs-slot/1", "episode_seed": seed,
                     "decision_frame": first_frame + 2 * index,
                     "history": _history(index), "cand": cand, "bomb": executed_bomb,
                     "obs": {"row": index}} for index in range(rows)]
    _write_jsonl(save_obs / "slots.jsonl", student_rows)
    _write_json(save_obs / "meta.json", {
        "schema": "v832-takeover-obs/1",
        "seed": seed,
        "label_source": "dagger_student_rollout",
        "obs_source": "takeover-runner",
        "decisions": rows,
        "slots": {"file": "slots.jsonl", "rows": rows},
        "planes": {key: {"file": f"obs/{name}", "sha256": obs[name]["sha256"],
                         "rows": int(obs[name]["array"].shape[0])}
                   for key, name in (("play", "play96x112.npy"),
                                     ("hud", "hud.npy"),
                                     ("act_hist", "act_hist.npy"))},
    })
    labels = []
    for index in range(rows):
        frame = first_frame + 2 * index
        unverified = index in set(unverified_rows)
        #: ``chosen_cand`` defaults to ``cand`` so the pre-existing fixtures keep the
        #: "the student happened to agree with the teacher" shape; a fixture that
        #: wants the disagreement (the interesting case) passes ``chosen_cand``.
        chosen = None if unverified else (cand if chosen_cand is None else chosen_cand)
        mask = _verified(chosen)
        scores = _scores()
        labels.append({
            "schema": "v832-dagger-relabel-label/1",
            "episode_seed": seed,
            "decision_frame": frame,
            "commit_frames": [frame, frame + 2],
            "duration_frames": 2,
            "obs_render_frames": [frame - 4 + step for step in range(4)],
            "history_frames": [frame - 16 + 2 * step for step in range(8)],
            "history": _history(index),
            "executed_cand": cand,
            "executed": {"cand": cand, "bomb": executed_bomb,
                         "mask": C.cand_to_mask(cand, executed_bomb),
                         "source": "policy_act"},
            "mask": C.cand_to_mask(cand, executed_bomb),
            # the known pit: the top-level bomb here is the *teacher* channel
            "bomb": teacher_bomb,
            "teacher_bomb": teacher_bomb,
            "scores": scores,
            "verified_mask": mask,
            "verified_count": sum(mask),
            "chosen": chosen,
            "chosen_verified": chosen is not None,
            "chosen_score": None if chosen is None else scores[chosen],
            "margin": 1.0,
            "label_source": "teacher_2f",
            "label_verified": chosen is not None,
            "hash_pre": f"{pre_base + index:016x}",
            "hash_post": f"{pre_base + index + 1:016x}",
            "obs": {"play": "obs/play96x112.npy", "hud": "obs/hud.npy",
                    "act_hist": "obs/act_hist.npy", "row": index,
                    "hud_frame": frame - 1},
            "slot_id": None,
            "shard_key": None,
            "provenance": None,
            "relabel": {
                "schema": "v832-dagger-relabel-block/1",
                "at": "2026-01-01T00:00:00+08:00",
                "entry_frame": first_frame,
                "decision_frame": frame,
                "decision_index": index,
                "query_index": index,
                "record_source": "child",
                "nonfinite_scores": [],
                "tolerance": 0.001,
                "hash_pre": f"{pre_base + index:016x}",
                "hash_post": f"{pre_base + index + 1:016x}",
                "tape_path": ".tmp_v832_relabel/tape/slots.jsonl",
                "tape_sha256": "0" * 64,
                "engine_sha256": "1" * 64,
                "teacher": {"bomb": teacher_bomb, "chosen": chosen,
                            "chosen_score": None if chosen is None else scores[chosen],
                            "margin": 1.0, "agrees_with_chosen": False},
                "tool": {"path": "tools/rl/v8_32_relabel.py", "sha256": "2" * 64},
            },
        })
    _write_jsonl(out / "labels.jsonl", labels)
    _write_json(out / "meta.json", {
        "schema": "v832-dagger-relabel/1",
        "seed": seed,
        "complete": True,
        "exit_code": 0,
        "source": {"root": save_obs.as_posix(), "rows": rows,
                   "schema": "v832-takeover-obs/1"},
        "counts": {"rows_labelled": rows, "rows_requested": rows, "limit": None,
                   "truncated": False, "source_rows": rows},
    })
    return {"out": out, "labels": out / "labels.jsonl", "save_obs": save_obs,
            "rows": rows, "obs": obs, "records": labels}


# ---------------------------------------------------------------------------
# build helper
# ---------------------------------------------------------------------------
def run_build(out: Path, *, baselines: Sequence[Path] = (), daggers: Sequence[Path] = (),
              dedup: bool = False, prefix: str = "dagger",
              require_verified: bool = False,
              dagger_label: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if dagger_label is not None:
        kwargs["dagger_label"] = dagger_label
    return J.build(J.BuildConfig(
        baselines=tuple(str(path) for path in baselines),
        daggers=tuple(str(path) for path in daggers),
        out=str(out), prefix=prefix, dedup=dedup,
        require_verified=require_verified,
        readout01=str(READOUT01), s2e_train=str(S2E_TRAIN), **kwargs))


def read_rows(out: Path) -> list[dict[str, Any]]:
    with (out / "slots.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_meta(out: Path) -> dict[str, Any]:
    return json.loads((out / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# (1) one baseline shard
# ---------------------------------------------------------------------------
def test_01_single_baseline_merge(tmp_path: Path) -> None:
    source = make_baseline(tmp_path / "base10", seed=10, rows=5)
    out = tmp_path / "out"
    result = run_build(out, baselines=[source["root"]])

    assert result["ok"] is True
    assert result["rows"] == 5
    assert result["rows_by_provenance"] == {"t04": 5}
    assert result["label_source_counts"] == {"teacher_2f": 5}
    assert result["downstream_passed"] is True
    assert result["downstream_rows"] == 5

    rows = read_rows(out)
    assert len(rows) == 5
    assert [row["slot_id"] for row in rows] == list(range(5))
    for name in ("play96x112.npy", "hud.npy", "act_hist.npy"):
        plane = np.load(out / "obs" / name, mmap_mode="r")
        assert int(plane.shape[0]) == 5
        assert plane.dtype == np.uint8
    assert (out / "targets.jsonl").is_file()


# ---------------------------------------------------------------------------
# (2) several baselines + several dagger outputs
# ---------------------------------------------------------------------------
def test_02_multi_baseline_and_multi_dagger(tmp_path: Path) -> None:
    base10 = make_baseline(tmp_path / "base10", seed=10, rows=4)
    base11 = make_baseline(tmp_path / "base11", seed=11, rows=3, pre_base=3000)
    dagger_a = make_dagger(tmp_path / "dag_a", seed=12, rows=2, first_frame=64)
    dagger_b = make_dagger(tmp_path / "dag_b", seed=12, rows=3, first_frame=200,
                           teacher_bomb=1, executed_bomb=0, pre_base=5000)
    out = tmp_path / "out"
    result = run_build(out, baselines=[base10["root"], base11["root"]],
                       daggers=[dagger_a["labels"], dagger_b["labels"]])

    assert result["rows"] == 4 + 3 + 2 + 3
    assert result["rows_by_provenance"] == {"t04": 7, "dagger": 5}
    meta = read_meta(out)
    assert [shard["shard_key"] for shard in meta["shards"]] == [
        "v832join-s00", "v832join-s01", "v832join-s02", "v832join-s03"]
    assert [shard["rows"] for shard in meta["shards"]] == [4, 3, 2, 3]
    assert [row["provenance"] for row in read_rows(out)] == (
        ["t04"] * 7 + ["dagger"] * 5)


# ---------------------------------------------------------------------------
# (3) the bomb channel is normalised and says where it came from
# ---------------------------------------------------------------------------
def test_03_bomb_channel_normalisation(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=2, bomb=0, cand=7)
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=2, first_frame=64,
                         teacher_bomb=1, executed_bomb=0, cand=7)
    out = tmp_path / "out"
    #: ``executed`` is the pre-flag channel; this test is the v832 bomb contract on
    #: that channel (the ``chosen`` channel is pinned by test_26..test_29)
    run_build(out, baselines=[base["root"]], daggers=[dagger["labels"]],
              dagger_label="executed")
    rows = read_rows(out)

    baseline_rows = [row for row in rows if row["provenance"] == "t04"]
    dagger_rows = [row for row in rows if row["provenance"] == "dagger"]
    assert len(baseline_rows) == 2 and len(dagger_rows) == 2

    for row in baseline_rows:
        assert row["bomb_source"] == "t04_bomb"
        assert row["bomb"] == 0
        assert row["teacher_bomb"] == 0
        assert row["executed_bomb"] == 0
        assert C.mask_to_cand(row["mask"]) == (row["executed_cand"], row["executed_bomb"])

    for row in dagger_rows:
        # the label channel is the teacher's bomb, not the student's
        assert row["bomb_source"] == "teacher_bomb"
        assert row["bomb"] == 1
        assert row["teacher_bomb"] == 1
        assert row["executed_bomb"] == 0
        # the mask is the *executed* mask and therefore decodes to the student's bomb
        assert C.mask_to_cand(row["mask"]) == (7, 0)
        assert C.cand_to_mask(7, 1) != row["mask"]


# ---------------------------------------------------------------------------
# (4) slot_id / shard_key renumbering
# ---------------------------------------------------------------------------
def test_04_slot_id_and_shard_key_are_renumbered(tmp_path: Path) -> None:
    base10 = make_baseline(tmp_path / "base10", seed=10, rows=3)
    base12 = make_baseline(tmp_path / "base12", seed=12, rows=2, pre_base=4000)
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=2, first_frame=300,
                         pre_base=7000)
    out = tmp_path / "out"
    run_build(out, baselines=[base10["root"], base12["root"]],
              daggers=[dagger["labels"]])
    rows = read_rows(out)

    assert [row["slot_id"] for row in rows] == list(range(7))
    assert {row["shard_key"] for row in rows} == {
        "v832join-s00", "v832join-s01", "v832join-s02"}
    assert all(row["shard_key"] and row["slot_id"] is not None for row in rows)
    # the source ids are preserved, never reused as the primary key
    t04_keys = {row["shard_key"] for row in rows if row["provenance"] == "t04"}
    assert "d1-s0d1-n3300-seed10-shard" not in t04_keys
    assert "d1-s0d1-n3300-seed12-shard" not in t04_keys
    for row in rows:
        source = row["source"]
        if row["provenance"] == "t04":
            assert source["slot_id"] == source["row"]      # T04 slot == row here
            assert source["shard_key_source"].startswith("d1-s0d1-")
        else:
            # the relabel row's own slot_id/shard_key are null — not reused, not lost
            assert source["slot_id"] is None
            assert source["shard_key_source"] is None
        assert source["ordinal"] == int(row["shard_key"][-2:])
    # T04 row 0 still carries its own nested provenance verbatim
    assert rows[0]["provenance_source"]["join_key"] == {
        "shard_key": "d1-s0d1-n3300-seed10-shard", "slot_id": 0}
    assert dagger["records"][0]["provenance"] is None
    assert rows[-1]["provenance_source"] is None


# ---------------------------------------------------------------------------
# (5) dedup + conflict counting
# ---------------------------------------------------------------------------
def test_05_dedup_counts_conflicts_and_keeps_the_preferred_side(tmp_path: Path) -> None:
    # same (episode_seed, decision_frame, hash_pre) on both sides: seed 12, frame 64
    base = make_baseline(tmp_path / "base12", seed=12, rows=2, first_frame=64,
                         pre_base=9000)
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=2, first_frame=64,
                         pre_base=9000)
    key = (12, 64, f"{9000:016x}")

    plain = tmp_path / "plain"
    result_plain = run_build(plain, baselines=[base["root"]], daggers=[dagger["labels"]])
    assert result_plain["rows"] == 4              # nothing is dropped without --dedup
    meta_plain = read_meta(plain)
    assert meta_plain["dedup"]["enabled"] is False
    assert meta_plain["join"]["duplicate_keys"] == len(
        {key, (12, 66, f"{9001:016x}")})
    assert meta_plain["join"]["duplicate_rows"] == 2

    dagger_wins = tmp_path / "dagger_wins"
    result = run_build(dagger_wins, baselines=[base["root"]],
                       daggers=[dagger["labels"]], dedup=True)
    assert result["rows"] == 2
    meta = read_meta(dagger_wins)
    assert meta["dedup"]["enabled"] is True
    assert meta["dedup"]["precedence"] == "dagger"
    assert meta["dedup"]["dedup_conflicts"] == 2
    assert meta["dedup"]["conflict_keys"] == 2
    assert meta["dedup"]["dedup_dropped"] == 2
    assert meta["join"]["duplicate_keys"] == 2       # counted before dedup
    assert meta["join"]["duplicate_keys_after_filters"] == 0
    assert [row["provenance"] for row in read_rows(dagger_wins)] == ["dagger", "dagger"]
    assert meta["counts"]["rows_by_provenance"] == {"dagger": 2}

    baseline_wins = tmp_path / "baseline_wins"
    run_build(baseline_wins, baselines=[base["root"]], daggers=[dagger["labels"]],
              dedup=True, prefix="baseline")
    meta = read_meta(baseline_wins)
    assert meta["dedup"]["precedence"] == "baseline"
    assert meta["counts"]["rows_by_provenance"] == {"t04": 2}


def test_06_dedup_keeps_a_single_copy_of_same_side_duplicates(tmp_path: Path) -> None:
    base_a = make_baseline(tmp_path / "base_a", seed=10, rows=2, pre_base=1000)
    base_b = make_baseline(tmp_path / "base_b", seed=10, rows=2, pre_base=1000,
                           shard_key="d1-s0d1-n3300-seed10-shard-copy")
    out = tmp_path / "out"
    run_build(out, baselines=[base_a["root"], base_b["root"]], dedup=True)
    meta = read_meta(out)
    assert meta["counts"]["rows"] == 2
    assert meta["dedup"]["conflict_keys"] == 2
    assert meta["dedup"]["dedup_dropped"] == 2
    assert [row["shard_key"] for row in read_rows(out)] == ["v832join-s00"] * 2


# ---------------------------------------------------------------------------
# (6) --require-verified
# ---------------------------------------------------------------------------
def test_07_require_verified_drops_unverified_rows(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=4, unverified_rows=[1, 3])
    strict = tmp_path / "strict"
    result = run_build(strict, baselines=[base["root"]], require_verified=True)

    assert result["rows"] == 2
    meta = read_meta(strict)
    assert meta["filters"]["require_verified"] is True
    assert meta["filters"]["dropped_unverified"] == 2
    assert meta["filters"]["dropped_by_side"] == {"baseline": 2}
    rows = read_rows(strict)
    assert all(row["label_verified"] is True for row in rows)
    assert all(row["chosen"] is not None for row in rows)


def test_08_unverified_rows_make_the_corpus_unloadable(tmp_path: Path) -> None:
    """Without --require-verified a chosen=null row is not loadable: fail closed."""

    base = make_baseline(tmp_path / "base10", seed=10, rows=4, unverified_rows=[1])
    out = tmp_path / "out"
    with pytest.raises(J.JoinError) as caught:
        run_build(out, baselines=[base["root"]])
    assert caught.value.stop == "downstream_unreadable"
    assert "chosen=null" in str(caught.value)
    assert "--require-verified" in str(caught.value)


# ---------------------------------------------------------------------------
# (7) the obs planes are bit copies
# ---------------------------------------------------------------------------
def test_09_obs_planes_are_bit_copies(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=3)
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=2, first_frame=64)
    out = tmp_path / "out"
    run_build(out, baselines=[base["root"]], daggers=[dagger["labels"]])

    expected = {
        "play96x112.npy": np.concatenate([base["obs"]["play96x112.npy"]["array"],
                                          dagger["obs"]["play96x112.npy"]["array"]]),
        "hud.npy": np.concatenate([base["obs"]["hud.npy"]["array"],
                                   dagger["obs"]["hud.npy"]["array"]]),
        "act_hist.npy": np.concatenate([base["obs"]["act_hist.npy"]["array"],
                                        dagger["obs"]["act_hist.npy"]["array"]]),
    }
    for name, array in expected.items():
        produced = np.load(out / "obs" / name, mmap_mode="r")
        assert np.array_equal(np.asarray(produced), array), name
        assert produced.dtype == array.dtype
        # a bit copy means the .npy files are byte identical
        reference = tmp_path / f"reference_{name}"
        np.save(reference, array, allow_pickle=False)
        assert sha256_file(out / "obs" / name) == sha256_file(reference), name

    meta = read_meta(out)
    for entry in meta["obs"]["files"]:
        name = Path(entry["name"]).name
        assert entry["sha256"] == sha256_file(out / "obs" / name)
        assert entry["size"] == (out / "obs" / name).stat().st_size
        assert entry["kind"] == "bit-copy"


# ---------------------------------------------------------------------------
# (8) fail-closed self-checks
# ---------------------------------------------------------------------------
def _mutate_rows(path: Path, mutate: Any) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    mutate(rows)
    _write_jsonl(path, rows)


def test_10_broken_hash_chain_is_refused(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=4)
    _mutate_rows(base["root"] / "slots.jsonl",
                 lambda rows: rows[2].__setitem__("hash_pre", "deadbeefdeadbeef"))
    out = tmp_path / "out"
    with pytest.raises(J.JoinError) as caught:
        run_build(out, baselines=[base["root"]])
    assert caught.value.stop == "selfcheck_failed"
    assert "hash" in str(caught.value)
    assert not (out / "slots.jsonl").exists()


def test_11_scores_must_be_eighteen(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=2)
    _mutate_rows(base["root"] / "slots.jsonl",
                 lambda rows: rows[0].__setitem__("scores", rows[0]["scores"][:-1]))
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", baselines=[base["root"]])
    assert caught.value.stop == "selfcheck_failed"
    assert "scores_length" in caught.value.detail["violations"]


def test_12_chosen_must_be_inside_the_verified_mask(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=2)
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[1]["verified_mask"] = _verified(5)     # verified candidate is 5
        rows[1]["chosen"] = 7                       # chosen is 7
        rows[1]["verified_count"] = 1
    _mutate_rows(base["root"] / "slots.jsonl", mutate)
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", baselines=[base["root"]])
    assert caught.value.stop == "selfcheck_failed"
    assert "verified_argmax" in caught.value.detail["violations"]


def test_13_a_student_row_is_not_a_label_and_is_refused(tmp_path: Path) -> None:
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=2)
    _mutate_rows(dagger["labels"],
                 lambda rows: rows[0].__setitem__("schema",
                                                  "v832-takeover-obs-slot/1"))
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", daggers=[dagger["labels"]])
    assert caught.value.stop == "source_invalid"
    assert "label" in str(caught.value)


def test_14_a_label_bomb_that_is_not_the_teacher_bomb_is_refused(tmp_path: Path) -> None:
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=2, teacher_bomb=0)
    _mutate_rows(dagger["labels"],
                 lambda rows: rows[0].__setitem__("bomb", 1))
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", daggers=[dagger["labels"]])
    assert caught.value.stop == "selfcheck_failed"
    assert "bomb" in caught.value.detail["violations"]


# ---------------------------------------------------------------------------
# (9) meta.json
# ---------------------------------------------------------------------------
def test_15_meta_json_carries_every_recorded_fact(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=3)
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=2, first_frame=64,
                         teacher_bomb=1, executed_bomb=0)
    out = tmp_path / "out"
    run_build(out, baselines=[base["root"]], daggers=[dagger["labels"]])
    meta = read_meta(out)

    assert meta["schema"] == "v832-join/1/meta"
    assert meta["complete"] is True
    assert meta["counts"]["rows"] == 5
    assert meta["counts"]["rows_by_provenance"] == {"t04": 3, "dagger": 2}
    assert meta["counts"]["label_source_counts"] == {"teacher_2f": 5}
    assert meta["counts"]["bomb_source_counts"] == {"t04_bomb": 3,
                                                    "teacher_bomb": 2}
    assert meta["hash"]["hash_pre_unique"] == 5
    assert meta["hash"]["hash_pre_duplicate_rows"] == 0
    assert meta["join"]["join_key"] == ["episode_seed", "decision_frame", "hash_pre"]
    assert meta["join"]["duplicate_keys"] == 0
    assert meta["join"]["unique_keys"] == 5
    assert [entry["sha256"] for entry in meta["obs"]["files"]] == [
        sha256_file(out / "obs" / name)
        for name in ("play96x112.npy", "hud.npy", "act_hist.npy")]
    assert meta["obs"]["mode"] == "copy"
    assert meta["slots"]["rows"] == 5 and meta["slots"]["sha256"] == sha256_file(
        out / "slots.jsonl")
    assert meta["targets"]["rows"] == 5
    for key in ("bomb", "bomb_source", "mask", "provenance", "provenance_source",
                "slot_id", "shard_key", "decision_frame", "hash_pre/hash_post",
                "scores", "chosen", "label_verified", "obs", "dedup"):
        assert key in meta["field_notes"], key
        assert meta["field_notes"][key]
    assert meta["downstream"]["passed"] is True
    assert meta["downstream"]["readout01_gate_corpus"]["called"] is True
    # the single-episode T04 gate cannot hold on a joined corpus; the probe records
    # exactly which of its rules break instead of pretending it passed
    diagnosis = meta["downstream"]["readout01_gate_corpus"]["diagnosis"]
    assert diagnosis["decision_frame_rule_breaks"] == 2      # the two dagger rows
    assert diagnosis["slot_id_order_breaks"] == 0
    assert diagnosis["hash_chain_breaks"] == 1
    assert diagnosis["check_dataset_record_violations"]
    assert meta["downstream"]["targets_contract_errors"] == {}
    assert meta["sources"][0]["selfcheck"]["passed"] is True
    assert meta["sources"][1]["selfcheck"]["save_obs_root"].endswith("dag_save_obs")
    assert meta["sources"][0]["meta_schema"] == "v420-dataset-2f/1/meta"
    assert meta["tool"]["path"].endswith("tools/rl/v8_32_join.py")
    assert len(meta["tool"]["sha256"]) == 64


# ---------------------------------------------------------------------------
# (10) the downstream loader really reads it
# ---------------------------------------------------------------------------
def test_16_downstream_loader_reads_the_corpus(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=4)
    base11 = make_baseline(tmp_path / "base11", seed=11, rows=3, pre_base=6000)
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=3, first_frame=64)
    out = tmp_path / "out"
    run_build(out, baselines=[base["root"], base11["root"]],
              daggers=[dagger["labels"]])
    total = 4 + 3 + 3

    ro = _load_module(READOUT01, "test_downstream_readout01")
    s2e = _load_module(S2E_TRAIN, "test_downstream_s2e_train")
    rows = read_rows(out)
    planes = {name: np.load(out / "obs" / name, mmap_mode="r")
              for name in ("play96x112.npy", "hud.npy", "act_hist.npy")}

    corpus = ro.Corpus(
        key="joined", seed=int(rows[0]["episode_seed"]), dir=str(out),
        planes=planes,
        chosen=np.asarray([int(row["chosen"]) for row in rows], dtype=np.int64),
        executed=np.asarray([int(row["executed_cand"]) for row in rows],
                            dtype=np.int64),
        verified=np.asarray([[bool(value) for value in row["verified_mask"]]
                             for row in rows]),
        history=np.asarray([list(row["history"]) for row in rows], dtype=np.int64),
        scores=np.asarray([list(row["scores"]) for row in rows], dtype=np.float32),
        label_source=np.asarray([row["label_source"] for row in rows]),
        slot_id=np.asarray([int(row["slot_id"]) for row in rows], dtype=np.int64),
        gate={"key": "joined", "rows": total, "passed": True,
              "violations_total": 0, "violations": {},
              "obs_planes_all_match": True, "hash_pairs": total,
              "executed_history_match_rows": total,
              "corpus_label_source_counts": {"teacher_2f": total}})

    merged, info = s2e.merge_shards(ro, [corpus])
    assert merged.n == total
    assert info["rows_total"] == total
    assert info["rows_per_shard"] == [total]
    assert info["plane_rows"] == {"play96x112.npy": total, "hud.npy": total,
                                  "act_hist.npy": total}
    assert info["plane_dtype"] == {"play96x112.npy": "uint8", "hud.npy": "uint8",
                                   "act_hist.npy": "uint8"}
    for name in s2e.PLANE_NAMES:
        assert merged.planes[name].shape[0] == total
    assert merged.planes["play96x112.npy"].shape[1:] == PLAY_SHAPE
    assert merged.planes["hud.npy"].shape[1:] == HUD_SHAPE
    assert merged.planes["act_hist.npy"].shape[1:] == HIST_SHAPE
    for attr in s2e.LABEL_FIELDS:
        assert len(getattr(merged, attr)) == total, attr
    values, counts = np.unique(merged.label_source, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist())) == {"teacher_2f": total}
    assert merged.scores.shape == (total, N_CAND)
    assert merged.verified.shape == (total, N_CAND)
    assert merged.history.shape == (total, 8)
    assert np.isfinite(merged.scores).all()

    view = s2e.view_corpus(merged, ro, 0, total, key="joined-view",
                           seed=int(rows[0]["episode_seed"]),
                           gate=corpus.gate, directory=str(out))
    assert view.n == total
    assert np.array_equal(np.asarray(view.executed), np.asarray(merged.executed))

    targets = [json.loads(line) for line in
               (out / "targets.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert len(targets) == total
    for record, target in zip(rows, targets):
        assert ro.dataset_2f.check_targets_record(target, record) == {}


# ---------------------------------------------------------------------------
# (11) empty input is refused
# ---------------------------------------------------------------------------
def test_17_no_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out")
    assert caught.value.stop == "empty_input"
    assert not (tmp_path / "out").exists()


def test_18_a_source_with_no_rows_is_refused(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=2)
    _write_jsonl(base["root"] / "slots.jsonl", [])
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", baselines=[base["root"]])
    assert caught.value.stop == "empty_input"


# ---------------------------------------------------------------------------
# (12) a labels/obs row-count mismatch is refused
# ---------------------------------------------------------------------------
def test_19_dagger_labels_and_obs_rows_must_agree(tmp_path: Path) -> None:
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=3, obs_rows=4)
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", daggers=[dagger["labels"]])
    assert caught.value.stop == "selfcheck_failed"
    assert "rows_vs_planes" in caught.value.detail["violations"]


def test_20_baseline_slots_and_obs_rows_must_agree(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=4, plane_rows=5)
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", baselines=[base["root"]])
    assert caught.value.stop == "selfcheck_failed"
    assert "rows_vs_planes" in caught.value.detail["violations"]


# ---------------------------------------------------------------------------
# extra fail-closed surface
# ---------------------------------------------------------------------------
def test_25_load_corpora_is_the_s2e_integration(tmp_path: Path) -> None:
    """The documented one-liner the training run needs, executed for real."""

    base = make_baseline(tmp_path / "base10", seed=10, rows=3)
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=2, first_frame=64,
                         teacher_bomb=1, executed_bomb=0)
    out = tmp_path / "out"
    run_build(out, baselines=[base["root"]], daggers=[dagger["labels"]])

    ro = _load_module(READOUT01, "test_load_corpora_readout01")
    s2e = _load_module(S2E_TRAIN, "test_load_corpora_s2e_train")
    parts = J.load_corpora(out, readout01=str(READOUT01))
    assert len(parts) == 1
    assert parts[0].n == 5
    assert parts[0].planes["play96x112.npy"].shape == (5, *PLAY_SHAPE)
    merged, info = s2e.merge_shards(ro, parts)
    assert merged.n == 5
    assert info["rows_total"] == 5
    assert set(np.unique(merged.label_source).tolist()) == {"teacher_2f"}
    assert merged.scores.shape == (5, N_CAND)


def test_21_a_plane_with_the_wrong_shape_is_refused(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=2)
    hud = np.zeros((2, 57, 160), dtype=np.uint8)
    np.save(base["root"] / "obs" / "hud.npy", hud, allow_pickle=False)
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", baselines=[base["root"]])
    assert caught.value.stop == "plane_invalid"
    assert "obs-v3" in str(caught.value)


def test_22_a_plane_sha_mismatch_is_refused(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=2)
    path = base["root"] / "obs" / "act_hist.npy"
    array = np.load(path)
    array[0, 0, :] = 0
    np.save(path, array, allow_pickle=False)
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", baselines=[base["root"]])
    assert caught.value.stop == "plane_invalid"
    assert "sha256" in str(caught.value)


def test_23_a_non_empty_out_dir_is_not_overwritten(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=2)
    out = tmp_path / "out"
    run_build(out, baselines=[base["root"]])
    with pytest.raises(J.JoinError) as caught:
        run_build(out, baselines=[base["root"]])
    assert caught.value.stop == "out_dir_in_use"


def test_24_verify_reaudits_the_written_corpus(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=3)
    out = tmp_path / "out"
    run_build(out, baselines=[base["root"]])
    report = J.verify(out, readout01=str(READOUT01), s2e_train=str(S2E_TRAIN))
    assert report["ok"] is True
    assert report["rows"] == 3
    assert report["rows_by_provenance"] == {"t04": 3}
    assert all(entry["match"] for entry in report["planes"].values())

    # a corrupted plane must make the re-audit fail
    path = out / "obs" / "hud.npy"
    array = np.load(path)
    array[0, 0, 0] = (int(array[0, 0, 0]) + 1) % 256
    np.save(path, array, allow_pickle=False)
    with pytest.raises(J.JoinError) as caught:
        J.verify(out, readout01=str(READOUT01), s2e_train=str(S2E_TRAIN))
    assert caught.value.stop == "verify_failed"


# ---------------------------------------------------------------------------
# the label channel (--dagger-label)
# ---------------------------------------------------------------------------
#: a dagger row's key order in ``executed`` mode — the pre-flag order.  A row
#: written by ``--dagger-label executed`` must match this exactly (it is what makes
#: "the old behaviour is unchanged" checkable at the byte level).
EXECUTED_MODE_KEY_ORDER = [
    "schema", "dataset_version", "obs_contract", "provenance", "shard_key",
    "slot_id", "episode_seed", "decision_frame", "duration_frames",
    "commit_frames", "obs_render_frames", "history_frames", "history",
    "executed_cand", "executed_bomb", "mask", "bomb", "bomb_source",
    "teacher_bomb", "scores", "verified_mask", "verified_count", "chosen",
    "chosen_verified", "chosen_score", "margin", "label_source",
    "label_verified", "hash_pre", "hash_post", "obs", "source",
    "provenance_source", "relabel"]


def _dagger_and_baseline(tmp_path: Path, *, chosen_cand: int | None = 3,
                         cand: int = 7) -> tuple[dict[str, Any], dict[str, Any]]:
    """A fixture whose student action (7) is *not* the teacher's (3)."""

    base = make_baseline(tmp_path / "base10", seed=10, rows=2, cand=7)
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=3, first_frame=64,
                         teacher_bomb=1, executed_bomb=0, cand=cand,
                         chosen_cand=chosen_cand)
    return base, dagger


def test_26_chosen_label_channel_rewrites_the_executed_trio(tmp_path: Path) -> None:
    """``chosen`` (the default) puts the *teacher's* action in the training trio."""

    base, dagger = _dagger_and_baseline(tmp_path)
    out = tmp_path / "out"
    result = run_build(out, baselines=[base["root"]], daggers=[dagger["labels"]])
    assert result["dagger_label"] == "chosen"
    rows = read_rows(out)
    dagger_rows = [row for row in rows if row["provenance"] == "dagger"]
    assert len(dagger_rows) == 3

    for row in dagger_rows:
        # the label channel is the teacher's chosen candidate and bomb
        assert row["executed_cand"] == row["chosen"] == 3
        assert row["executed_bomb"] == row["teacher_bomb"] == row["bomb"] == 1
        assert row["mask"] == C.cand_to_mask(3, 1)
        assert C.mask_to_cand(row["mask"]) == (3, 1)
        # the transmitted mask is no longer the label mask
        assert row["mask"] != C.cand_to_mask(7, 0)

    meta = read_meta(out)
    channel = meta["label_channel"]
    assert channel["dagger_label"] == "chosen"
    assert channel["dagger_rows"] == 3
    assert channel["dagger_executed_cand_equals_chosen"] == 3
    assert channel["dagger_mask_equals_cand_to_mask_label"] == 3
    assert meta["counts"]["dagger_label"] == "chosen"
    assert meta["counts"]["dagger_executed_cand_equals_chosen"] == 3
    assert meta["counts"]["dagger_executed_bomb_equals_teacher_bomb"] == 3
    assert "student_cand" in meta["field_notes"]
    assert meta["field_notes"]["dagger_label"]
    # the target file follows the same channel
    targets = [json.loads(line) for line in
               (out / "targets.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    for record, target in zip(rows, targets):
        assert target["executed_cand"] == record["executed_cand"]
        assert target["chosen_verified_cand"] == record["chosen"]


def test_27_chosen_mode_preserves_the_student_channel(tmp_path: Path) -> None:
    """The student's own action survives verbatim beside the label channel."""

    base, dagger = _dagger_and_baseline(tmp_path)
    out = tmp_path / "out"
    run_build(out, baselines=[base["root"]], daggers=[dagger["labels"]])
    rows = read_rows(out)

    dagger_rows = [row for row in rows if row["provenance"] == "dagger"]
    for index, row in enumerate(dagger_rows):
        assert row["student_cand"] == 7          # what the student executed
        assert row["student_bomb"] == 0
        assert row["student_mask"] == C.cand_to_mask(7, 0)
        assert row["student_mask"] != row["mask"]
        assert row["chosen"] == 3                # the teacher's action, untouched
        assert index == row["slot_id"] - 2

    # a T04 row has no student channel: its executed action *is* the label
    baseline_rows = [row for row in rows if row["provenance"] == "t04"]
    for row in baseline_rows:
        for key in ("student_cand", "student_bomb", "student_mask"):
            assert key not in row
        assert C.mask_to_cand(row["mask"]) == (row["executed_cand"],
                                              row["executed_bomb"])

    # the downstream loader's ``executed`` array is the label channel
    corpus = J.load_corpora(out, readout01=str(READOUT01))[0]
    assert corpus.executed.tolist()[:2] == [7, 7]
    assert corpus.executed.tolist()[2:] == [3, 3, 3]
    assert corpus.chosen.tolist() == [7, 7, 3, 3, 3]


def test_28_t04_rows_are_byte_identical_in_both_modes(tmp_path: Path) -> None:
    """Only the dagger label channel may change: the T04 block is frozen."""

    base, dagger = _dagger_and_baseline(tmp_path)
    chosen_out = tmp_path / "chosen_out"
    executed_out = tmp_path / "executed_out"
    run_build(chosen_out, baselines=[base["root"]], daggers=[dagger["labels"]],
              dagger_label="chosen")
    run_build(executed_out, baselines=[base["root"]], daggers=[dagger["labels"]],
              dagger_label="executed")

    chosen_lines = (chosen_out / "slots.jsonl").read_text(
        encoding="utf-8").splitlines()
    executed_lines = (executed_out / "slots.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(chosen_lines) == len(executed_lines) == 5
    # the T04 block: same bytes, same keys, same values
    assert chosen_lines[:2] == executed_lines[:2]
    # the dagger block: same rows, different label channel
    for mine, theirs in zip(chosen_lines[2:], executed_lines[2:]):
        mine_row, theirs_row = json.loads(mine), json.loads(theirs)
        assert set(mine_row) - set(theirs_row) == {
            "student_cand", "student_bomb", "student_mask"}
        assert set(theirs_row) - set(mine_row) == set()
        for key in theirs_row:
            if key in ("executed_cand", "executed_bomb", "mask"):
                continue
            assert mine_row[key] == theirs_row[key], key
        assert theirs_row["executed_cand"] == mine_row["student_cand"]
        assert theirs_row["executed_bomb"] == mine_row["student_bomb"]
        assert theirs_row["mask"] == mine_row["student_mask"]

    # the obs planes are a bit copy of the same source rows in both modes
    for name in ("play96x112.npy", "hud.npy", "act_hist.npy"):
        left = np.load(chosen_out / "obs" / name)
        right = np.load(executed_out / "obs" / name)
        assert np.array_equal(left, right), name
    # the two modes disagree on exactly the dagger label counts
    assert read_meta(chosen_out)["label_channel"][
        "dagger_executed_cand_equals_chosen"] == 3
    assert read_meta(executed_out)["label_channel"][
        "dagger_executed_cand_equals_chosen"] == 0
    assert read_meta(executed_out)["label_channel"]["student_channel_fields"] == [
        "student_cand", "student_bomb", "student_mask"]
    assert read_meta(executed_out)["counts"][
        "rows_with_student_channel"] == 0


def test_29_executed_mode_is_the_pre_flag_behaviour(tmp_path: Path) -> None:
    """``--dagger-label executed`` writes exactly the v832 row, key order included."""

    base, dagger = _dagger_and_baseline(tmp_path)
    out = tmp_path / "out"
    result = run_build(out, baselines=[base["root"]], daggers=[dagger["labels"]],
                       dagger_label="executed")
    assert result["dagger_label"] == "executed"
    rows = read_rows(out)
    dagger_rows = [row for row in rows if row["provenance"] == "dagger"]
    assert len(dagger_rows) == 3

    raw = dagger["records"]
    for row, payload in zip(dagger_rows, raw):
        assert list(row) == EXECUTED_MODE_KEY_ORDER
        assert "student_cand" not in row
        assert row["executed_cand"] == payload["executed_cand"] == 7
        assert row["executed_bomb"] == payload["executed"]["bomb"] == 0
        assert row["mask"] == payload["mask"] == C.cand_to_mask(7, 0)
        # the bomb channel is the teacher's in *both* modes: it never moved
        assert row["bomb"] == row["teacher_bomb"] == 1
        assert row["chosen"] == 3
        assert row["bomb_source"] == "teacher_bomb"

    meta = read_meta(out)
    assert meta["label_channel"]["dagger_label"] == "executed"
    assert meta["label_channel"]["dagger_executed_cand_equals_chosen"] == 0
    assert meta["counts"]["dagger_student_cand_differs_from_chosen"] == 0
    assert meta["field_notes"]["dagger_label"]
    assert "executed" in meta["field_notes"]["dagger_label"]
    # a T04 row's key order is the same list minus the relabel block
    baseline_row = rows[0]
    assert list(baseline_row) == EXECUTED_MODE_KEY_ORDER[:-1]


def test_30_chosen_mode_refuses_a_row_without_a_teacher_action(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=2)
    dagger = make_dagger(tmp_path / "dag", seed=12, rows=3, first_frame=64,
                         unverified_rows=(1,))
    out = tmp_path / "out"
    with pytest.raises(J.JoinError) as caught:
        run_build(out, baselines=[base["root"]], daggers=[dagger["labels"]])
    assert caught.value.stop == "source_invalid"
    assert "chosen=null" in str(caught.value)
    assert not out.exists()
    # the same corpus is buildable on the channel that does not need a teacher
    # action (an unverified row is not loadable downstream in *either* mode, so it
    # is dropped here -- that gap predates the flag)
    run_build(out, baselines=[base["root"]], daggers=[dagger["labels"]],
              dagger_label="executed", require_verified=True)
    assert len(read_rows(out)) == 4
    assert read_meta(out)["filters"]["dropped_unverified"] == 1


def test_31_an_unknown_dagger_label_is_refused(tmp_path: Path) -> None:
    base = make_baseline(tmp_path / "base10", seed=10, rows=2)
    with pytest.raises(J.JoinError) as caught:
        run_build(tmp_path / "out", baselines=[base["root"]],
                  dagger_label="teacher")
    assert caught.value.stop == "config_invalid"
    with pytest.raises(SystemExit):
        J.build_parser().parse_args(["build", "--baseline", "x", "--out", "y",
                                     "--dagger-label", "teacher"])
