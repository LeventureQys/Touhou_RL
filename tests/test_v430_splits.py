"""v4.3.0 W09 — behaviour tests for the clean hold-out lineage validator.

Every test asserts a *decision* the validator must make (accept / reject) and the
reason it must give, not merely that a function returns a dict.  Nothing here
launches the game, reads a pixel payload or trains anything.

Coverage map (``EXECUTION_PLAN.md`` §2 W09, ``DEVELOPMENT_GUIDE.md`` §11):

* train/dev ancestor crossing is rejected and names the conflicting ancestor;
* a clean hold-out family appearing in **each** of the nine pretraining source
  categories is rejected (one case per category, including perception, reward
  detector and model selection);
* a seed inside 60000-60199 is rejected; a protected-range seed is rejected for
  train only;
* a sealed path appearing in the config is rejected (and so is
  ``read_sealed_results: true``);
* a missing ancestor field and an unknown source kind are rejected;
* the same ``(task_family, episode)`` recorded under a second seed is collapsed
  into one group (never two independent samples);
* a legal config passes, its ``split_registry_sha`` is stable, and changing a
  single field changes the digest;
* ``text/lf-normalized`` digests are equal for CRLF and LF copies of the same
  text while ``binary/raw`` digests differ;
* the strict / degraded tier: missing granularity metadata fails in strict mode
  and is reported as a degradation in non-strict mode, but a real lineage
  violation still fails in non-strict mode;
* the shipped frozen config (``configs/v430/splits.json``) passes in strict mode.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from touhou_rl.v430.data import lineage
from touhou_rl.v430.data.lineage import (
    BANNED_SEED_MAX,
    BANNED_SEED_MIN,
    DIGEST_SCOPE_BINARY,
    DIGEST_SCOPE_TEXT,
    PRETRAINING_SOURCE_KINDS,
    ancestor_id,
    binary_raw_digest,
    file_identity,
    group_key,
    text_lf_digest,
    validate_splits,
)

REPO = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO / "configs" / "v430" / "splits.json"
REGISTRY_DIR = REPO / "Document/Update/Alpha/v4.3.0/registries"
EXECUTION_PLAN = REPO / "Document/Update/Alpha/v4.3.0/EXECUTION_PLAN.md"
FROZEN_REGISTRIES = ("task_families.json", "pretraining_sources.json",
                     "holdout_families.json", "runs.json")
#: Every ``*_at``-style field the shipped config/registries must date honestly.
_TIME_FIELD_RE = re.compile(r"(?:^|_)(?:at|checked_at)$")
_TZ_SUFFIX_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
#: How far in the future a clock-skewed timestamp may sit before it is "not yet true".
_FUTURE_TOLERANCE = timedelta(minutes=5)

FAM_TRAIN = "th08-stage1-normal-team0"
HOLDOUT_A = "th08-stage4a-hard-team0"
HOLDOUT_B = "th08-stage6a-lunatic-team0"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _family(fid: str, status: str, **extra) -> dict:
    entry = {
        "family_id": fid,
        "game": "th08",
        "stage_index": 0,
        "stage_name": "Stage 1",
        "route_family": "single",
        "difficulty": "normal",
        "difficulty_index": 1,
        "team": 0,
        "status": status,
        "retail_replay_fixture_ids": [],
    }
    if status == "touched_by_training":
        entry["touched_by"] = ["teacher_query"]
        entry["training_evidence"] = [
            {"source_kind": "teacher_query", "source_id": "tq-1",
             "evidence_path": "configs/v430/splits.json", "record": "unit-test"}
        ]
    else:
        entry["clean_evidence"] = [_claim()]
    entry.update(extra)
    return entry


def _claim() -> dict:
    return {
        "claim": "no fitting source records this family",
        "records_absent_from": ["unit-test-record"],
        "evidence_paths": ["ASSET_SYNC.md"],
    }


def _ancestor(kind: str, sid: str, episode: str, seed: int, family: str, **extra) -> dict:
    entry = {"source_kind": kind, "source_id": sid, "episode": episode,
             "seed": seed, "task_family": family}
    entry.update(extra)
    return entry


def _source(kind: str, sid: str, ids: list[str], **extra) -> dict:
    entry = {
        "kind": kind,
        "source_id": sid,
        "status": "used_for_training",
        "ancestor_enumeration": "complete",
        "ancestor_ids": list(ids),
        "evidence_paths": ["ASSET_SYNC.md"],
        "uses_retail_replay_fixtures": False,
    }
    entry.update(extra)
    return entry


def _aid(kind: str, sid: str, episode: str, family: str = FAM_TRAIN) -> str:
    return ancestor_id(kind, sid, episode, family)


def base_config() -> dict:
    """A minimal config that satisfies every fail-closed rule."""

    ancestors = [
        _ancestor("teacher_query", "tq-1", "tq-ep-1", 1, FAM_TRAIN),
        _ancestor("dagger_round", "dg-1", "dg-ep-1", 2000, FAM_TRAIN),
        _ancestor("policy_rl", "pr-1", "pr-ep-1", 3000, FAM_TRAIN),
        _ancestor("perception_pretrain", "pp-1", "pp-ep-1", 1, FAM_TRAIN),
        _ancestor("reward_detector", "rd-1", "rd-ep-1", 1, FAM_TRAIN),
        _ancestor("human_replay", "hr-1", "hr-ep-1", 7, FAM_TRAIN),
        _ancestor("model_selection", "ms-1", "dev-seed-1000", 1000, FAM_TRAIN),
    ]
    sources = [
        _source("teacher_query", "tq-1", [_aid("teacher_query", "tq-1", "tq-ep-1")]),
        _source("bc_dataset", "bc-1", [_aid("teacher_query", "tq-1", "tq-ep-1")]),
        _source("dagger_round", "dg-1", [_aid("dagger_round", "dg-1", "dg-ep-1")]),
        _source("policy_rl", "pr-1", [_aid("policy_rl", "pr-1", "pr-ep-1")]),
        _source("perception_pretrain", "pp-1", [_aid("perception_pretrain", "pp-1", "pp-ep-1")]),
        _source("reward_detector", "rd-1", [_aid("reward_detector", "rd-1", "rd-ep-1")]),
        _source("human_replay", "hr-1", [_aid("human_replay", "hr-1", "hr-ep-1")]),
        _source("model_selection", "ms-1", [_aid("model_selection", "ms-1", "dev-seed-1000")]),
        _source("hyperparameter_selection", "hs-1",
                [_aid("model_selection", "ms-1", "dev-seed-1000")]),
    ]
    return {
        "schema": lineage.SCHEMA,
        "experiment_id": "unit-test-splits-001",
        "spec_revision": lineage.SPEC_REVISION,
        "frozen_at": "2026-11-15T19:40:00+08:00",
        "timezone": "Asia/Shanghai",
        "roots": {"repo": ".", "d1_corpus": ".tmp_v400_d1/d1"},
        "families": [
            _family(FAM_TRAIN, "touched_by_training"),
            _family(HOLDOUT_A, "instrument_only"),
            _family(HOLDOUT_B, "instrument_only"),
        ],
        "holdout_families": {
            "frozen": [
                {"family_id": HOLDOUT_A, "clean_evidence": [_claim()]},
                {"family_id": HOLDOUT_B, "clean_evidence": [_claim()]},
            ],
            "reserved": [],
        },
        "ancestors": ancestors,
        "splits": {
            "train": {"ancestor_ids": [
                _aid("teacher_query", "tq-1", "tq-ep-1"),
                _aid("dagger_round", "dg-1", "dg-ep-1"),
                _aid("policy_rl", "pr-1", "pr-ep-1"),
                _aid("perception_pretrain", "pp-1", "pp-ep-1"),
                _aid("reward_detector", "rd-1", "rd-ep-1"),
                _aid("human_replay", "hr-1", "hr-ep-1"),
            ]},
            "dev": {"ancestor_ids": [_aid("model_selection", "ms-1", "dev-seed-1000")]},
            "confirm": {"ancestor_ids": [],
                        "empty_reason": "unit test: no confirmation batch frozen"},
            "adapt": {"ancestor_ids": [],
                      "target_families": [HOLDOUT_A, HOLDOUT_B],
                      "empty_reason": "unit test: no adaptation episode run yet"},
        },
        "pretraining_sources": sources,
        "confirm_batches": [
            {"name": "first_stage", "seed_min": 61000, "seed_max": 61099,
             "episodes": 0, "status": "unused", "evidence": "ASSET_SYNC.md"},
            {"name": "teacher", "seed_min": 61300, "seed_max": 61399,
             "episodes": 100, "status": "spent", "evidence": "ASSET_SYNC.md"},
        ],
        "reserved_training_exclusion": {
            "seed_min": BANNED_SEED_MIN,
            "seed_max": BANNED_SEED_MAX,
            "read_sealed_results": False,
            "protected_ranges": [
                {"name": "first_stage_confirm", "seed_min": 61000, "seed_max": 61099,
                 "usage": "unused", "evidence": "ASSET_SYNC.md"},
                {"name": "teacher_confirm", "seed_min": 61300, "seed_max": 61399,
                 "usage": "spent", "evidence": "ASSET_SYNC.md"},
            ],
        },
        "sealed": {
            "read_sealed_results": False,
            "present_on_host": False,
            "path_patterns": ["sealed/"],
            "verified_absent": {"evidence_paths": ["ASSET_SYNC.md"],
                                "method": "unit test"},
        },
    }


def cfg(**overrides) -> dict:
    config = base_config()
    config.update(overrides)
    return config


def codes(report: dict) -> set[str]:
    return {p["code"] for p in report["problems"]}


def find(report: dict, code: str) -> list[dict]:
    return [p for p in report["problems"] if p["code"] == code]


def source_of(config: dict, kind: str, sid: str) -> dict:
    for entry in config["pretraining_sources"]:
        if entry["kind"] == kind and entry["source_id"] == sid:
            return entry
    raise AssertionError(f"source {kind}/{sid} not in fixture")


# ---------------------------------------------------------------------------
# 0. the shipped registry
# ---------------------------------------------------------------------------


def test_shipped_config_passes_strict() -> None:
    report = validate_splits(SHIPPED_CONFIG)
    assert report["passed"] is True, report["problems"][:5]
    assert report["problems"] == []
    assert report["mode"] == "strict"
    assert report["exit_code"] == 0
    stats = report["statistics"]
    assert stats["seed_range_check"]["banned_hits"] == []
    assert stats["seed_range_check"]["protected_hits"] == []
    assert stats["pretraining_kinds_missing"] == []
    assert len(stats["holdout_families_frozen"]) >= 2
    assert stats["splits"]["train"]["independent_ancestors"] > 0
    assert report["split_registry_sha_scope"] == DIGEST_SCOPE_TEXT


def test_base_fixture_passes() -> None:
    report = validate_splits(base_config())
    assert report["passed"] is True, report["problems"]
    assert report["statistics"]["splits"]["train"]["independent_ancestors"] == 6
    assert report["statistics"]["splits"]["dev"]["independent_ancestors"] == 1


# ---------------------------------------------------------------------------
# 1. train / dev crossing
# ---------------------------------------------------------------------------


def test_train_dev_crossing_is_rejected_and_names_the_ancestor() -> None:
    config = base_config()
    shared = _aid("model_selection", "ms-1", "dev-seed-1000")
    config["splits"]["train"]["ancestor_ids"].append(shared)
    report = validate_splits(config)
    assert report["passed"] is False
    assert report["exit_code"] == 2
    overlaps = find(report, "X_TRAIN_OVERLAP")
    assert overlaps, codes(report)
    assert shared in json.dumps(overlaps[0], ensure_ascii=False)
    assert "train" in overlaps[0]["split"] and "dev" in overlaps[0]["split"]


def test_train_adapt_crossing_is_rejected() -> None:
    config = base_config()
    config["ancestors"].append(_ancestor("evaluation", "ad-1", "adapt-ep-1", 61200, HOLDOUT_A))
    config["splits"]["adapt"]["ancestor_ids"].append(
        _aid("evaluation", "ad-1", "adapt-ep-1", HOLDOUT_A))
    config["splits"]["train"]["ancestor_ids"].append(
        _aid("evaluation", "ad-1", "adapt-ep-1", HOLDOUT_A))
    report = validate_splits(config)
    assert report["passed"] is False
    assert "X_TRAIN_OVERLAP" in codes(report)


# ---------------------------------------------------------------------------
# 2. a clean hold-out family touched by each of the nine source categories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", PRETRAINING_SOURCE_KINDS)
def test_holdout_touched_by_each_pretraining_kind(kind: str) -> None:
    config = base_config()
    target = next(s for s in config["pretraining_sources"] if s["kind"] == kind)
    target["touched_families"] = [HOLDOUT_A]
    report = validate_splits(config)
    assert report["passed"] is False, f"{kind} should not be able to touch a hold-out family"
    hits = find(report, "H_TOUCHED_BY_SOURCE")
    assert hits, sorted(codes(report))
    assert any(kind == h["kind"] for h in hits), hits
    assert any(HOLDOUT_A in h["families"] for h in hits)


def test_holdout_listed_as_an_ancestor_of_a_source_is_rejected() -> None:
    config = base_config()
    config["ancestors"].append(_ancestor("teacher_query", "tq-1", "hard-4a-ep", 42, HOLDOUT_A))
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(
        _aid("teacher_query", "tq-1", "hard-4a-ep", HOLDOUT_A))
    config["splits"]["confirm"]["ancestor_ids"].append(
        _aid("teacher_query", "tq-1", "hard-4a-ep", HOLDOUT_A))
    report = validate_splits(config)
    assert report["passed"] is False
    assert {"H_TOUCHED_BY_SOURCE", "H_HOLDOUT_IN_SPLIT"} <= codes(report)


def test_holdout_in_train_split_is_rejected() -> None:
    config = base_config()
    config["ancestors"].append(_ancestor("teacher_query", "tq-1", "lun-6a-ep", 43, HOLDOUT_B))
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(
        _aid("teacher_query", "tq-1", "lun-6a-ep", HOLDOUT_B))
    config["splits"]["train"]["ancestor_ids"].append(
        _aid("teacher_query", "tq-1", "lun-6a-ep", HOLDOUT_B))
    report = validate_splits(config)
    assert report["passed"] is False
    assert "H_HOLDOUT_IN_SPLIT" in codes(report)


def test_retail_replay_fitting_source_invalidates_holdouts() -> None:
    """The G0.1 retail corpus covers every (difficulty, stage) pair."""

    config = base_config()
    config["families"][1]["retail_replay_fixture_ids"] = ["hard-4-a", "hard-4-b"]
    config["families"][2]["retail_replay_fixture_ids"] = ["lunatic-6-a"]
    source_of(config, "human_replay", "hr-1")["uses_retail_replay_fixtures"] = True
    report = validate_splits(config)
    assert report["passed"] is False
    leak = find(report, "H_INSTRUMENT_FIXTURE_LEAK")
    assert len(leak) == 2, leak
    assert {HOLDOUT_A, HOLDOUT_B} == {p["family"] for p in leak}


def test_holdout_needs_evidence_and_clean_status() -> None:
    config = base_config()
    config["families"][1]["status"] = "touched_by_training"
    config["families"][1].pop("clean_evidence")
    report = validate_splits(config)
    assert report["passed"] is False
    assert "H_DIRTY_FAMILY" in codes(report)

    config = base_config()
    config["holdout_families"]["frozen"][0]["clean_evidence"] = []
    report = validate_splits(config)
    assert report["passed"] is False
    assert "H_NO_EVIDENCE" in codes(report)


def test_fewer_than_two_holdouts_is_rejected() -> None:
    config = base_config()
    config["holdout_families"]["frozen"] = config["holdout_families"]["frozen"][:1]
    config["splits"]["adapt"]["target_families"] = [HOLDOUT_A]
    report = validate_splits(config)
    assert report["passed"] is False
    assert "H_TOO_FEW" in codes(report)


def test_renaming_a_holdout_label_is_rejected() -> None:
    config = base_config()
    config["holdout_families"]["frozen"][0]["family_id"] = "th08-stage4a-hard-team0-renamed"
    report = validate_splits(config)
    assert report["passed"] is False
    assert "H_UNKNOWN_FAMILY" in codes(report)


# ---------------------------------------------------------------------------
# 3. banned and reserved seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [BANNED_SEED_MIN, 60100, BANNED_SEED_MAX])
def test_banned_seed_is_rejected(seed: int) -> None:
    config = base_config()
    aid = _aid("teacher_query", "tq-1", "banned-ep")
    config["ancestors"].append(_ancestor("teacher_query", "tq-1", "banned-ep", seed, FAM_TRAIN))
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(aid)
    config["splits"]["train"]["ancestor_ids"].append(aid)
    report = validate_splits(config)
    assert report["passed"] is False
    hits = find(report, "S_BANNED_SEED")
    assert hits and hits[0]["seed"] == seed


def test_protected_range_seed_is_rejected_for_train() -> None:
    config = base_config()
    aid = _aid("teacher_query", "tq-1", "confirm-ep")
    config["ancestors"].append(_ancestor("teacher_query", "tq-1", "confirm-ep", 61300, FAM_TRAIN))
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(aid)
    config["splits"]["train"]["ancestor_ids"].append(aid)
    report = validate_splits(config)
    assert report["passed"] is False
    hits = find(report, "S_RESERVED_IN_TRAIN")
    assert hits and hits[0]["range"] == "teacher_confirm"


def test_reserved_range_must_not_shrink() -> None:
    config = base_config()
    config["reserved_training_exclusion"]["seed_max"] = 60100
    report = validate_splits(config)
    assert report["passed"] is False
    assert "P_RESERVED_RANGE" in codes(report)


# ---------------------------------------------------------------------------
# 4. sealed
# ---------------------------------------------------------------------------


def test_sealed_reference_is_rejected() -> None:
    config = base_config()
    config["splits"]["dev"]["note"] = "copied from sealed/confirm-61300"
    report = validate_splits(config)
    assert report["passed"] is False
    hits = find(report, "SEALED_REFERENCED")
    assert hits and hits[0]["pattern"] == "sealed/"


def test_sealed_ancestor_artifact_is_rejected() -> None:
    config = base_config()
    aid = _aid("teacher_query", "tq-1", "sealed-ep")
    config["ancestors"].append(_ancestor("teacher_query", "tq-1", "sealed-ep", 5, FAM_TRAIN,
                                         artifact_path="sealed/results/61300.jsonl"))
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(aid)
    config["splits"]["train"]["ancestor_ids"].append(aid)
    report = validate_splits(config)
    assert report["passed"] is False
    assert "SEALED_REFERENCED" in codes(report)


def test_read_sealed_results_true_is_rejected() -> None:
    config = base_config()
    config["reserved_training_exclusion"]["read_sealed_results"] = True
    report = validate_splits(config)
    assert report["passed"] is False
    assert "P_SEALED_READ_ALLOWED" in codes(report)


def test_sealed_present_on_host_is_rejected() -> None:
    config = base_config()
    config["sealed"]["present_on_host"] = True
    report = validate_splits(config)
    assert report["passed"] is False
    assert "SEALED_PRESENT" in codes(report)


def test_sealed_block_missing_is_rejected() -> None:
    config = base_config()
    config.pop("sealed")
    report = validate_splits(config)
    assert report["passed"] is False
    assert "P_SEALED_MISSING" in codes(report)


# ---------------------------------------------------------------------------
# 5. missing fields / unknown kinds / digest discipline
# ---------------------------------------------------------------------------


def test_missing_ancestor_field_is_rejected() -> None:
    config = base_config()
    config["ancestors"][0].pop("task_family")
    report = validate_splits(config)
    assert report["passed"] is False
    assert "A_MISSING_FIELD" in codes(report)


def test_unknown_source_kind_is_rejected() -> None:
    config = base_config()
    config["pretraining_sources"][0]["kind"] = "vibes"
    report = validate_splits(config)
    assert report["passed"] is False
    assert {"P_SOURCE_KIND_UNKNOWN", "P_SOURCE_KIND_UNREGISTERED"} <= codes(report)


def test_each_unregistered_kind_is_rejected() -> None:
    """Dropping any single category must fail: 'only the final RL seed' is forbidden."""

    for kind in PRETRAINING_SOURCE_KINDS:
        config = base_config()
        config["pretraining_sources"] = [s for s in config["pretraining_sources"]
                                         if s["kind"] != kind]
        report = validate_splits(config)
        assert report["passed"] is False, f"missing {kind} was accepted"
        assert "P_SOURCE_KIND_UNREGISTERED" in codes(report), kind


def test_bare_sha256_without_scope_is_rejected() -> None:
    config = base_config()
    config["splits"]["dev"]["registry_pointer"] = {"path": "x.json", "sha256": "0" * 64}
    report = validate_splits(config)
    assert report["passed"] is False
    assert "P_DIGEST_SCOPE_MISSING" in codes(report)


def test_unknown_family_in_ancestor_is_rejected() -> None:
    config = base_config()
    config["ancestors"][0]["task_family"] = "th08-stage1-normal-team0-renamed"
    report = validate_splits(config)
    assert report["passed"] is False
    assert "A_UNKNOWN_FAMILY" in codes(report)


def test_ancestor_outside_every_split_is_rejected() -> None:
    config = base_config()
    config["ancestors"].append(_ancestor("teacher_query", "tq-1", "orphan-ep", 9, FAM_TRAIN))
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(
        _aid("teacher_query", "tq-1", "orphan-ep"))
    report = validate_splits(config)
    assert report["passed"] is False
    assert "A_UNASSIGNED" in codes(report)


def test_source_claiming_an_unknown_ancestor_is_rejected() -> None:
    config = base_config()
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(
        _aid("teacher_query", "tq-1", "never-registered"))
    report = validate_splits(config)
    assert report["passed"] is False
    assert "P_SOURCE_UNKNOWN_ANCESTOR" in codes(report)


def test_empty_split_needs_a_reason() -> None:
    config = base_config()
    config["splits"]["confirm"].pop("empty_reason")
    report = validate_splits(config)
    assert report["passed"] is False
    assert "P_SPLIT_EMPTY" in codes(report)


def test_adapt_must_target_the_frozen_families() -> None:
    config = base_config()
    config["splits"]["adapt"]["target_families"] = [HOLDOUT_A]
    report = validate_splits(config)
    assert report["passed"] is False
    assert "H_NOT_TARGETED" in codes(report)


def test_foreign_host_root_is_rejected() -> None:
    config = base_config()
    config["roots"]["engine"] = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
    report = validate_splits(config)
    assert report["passed"] is False
    assert "P_ROOT_FOREIGN_PATH" in codes(report)


# ---------------------------------------------------------------------------
# 6. "a fresh seed is not a new task"
# ---------------------------------------------------------------------------


def test_same_family_episode_with_new_seed_collapses_to_one_group() -> None:
    config = base_config()
    # the same episode re-collected in a second run (new source id, new seed)
    again = _ancestor("dagger_round", "dg-2", "dg-ep-1", 2001, FAM_TRAIN)
    config["ancestors"].append(again)
    config["pretraining_sources"].append(
        _source("dagger_round", "dg-2", [_aid("dagger_round", "dg-2", "dg-ep-1")]))
    config["splits"]["train"]["ancestor_ids"].append(_aid("dagger_round", "dg-2", "dg-ep-1"))
    report = validate_splits(config)
    assert report["passed"] is False
    hits = find(report, "A_DUPLICATE_GROUP")
    assert hits and sorted(hits[0]["seeds"]) == [2000, 2001]
    stats = report["statistics"]["splits"]["train"]
    assert stats["independent_ancestors"] == stats["declared_ancestors"] - 1
    assert len(stats["collapsed_groups"]) == 1


def test_same_group_across_splits_is_a_crossing_not_a_new_task() -> None:
    config = base_config()
    config["ancestors"].append(_ancestor("dagger_round", "dg-2", "dg-ep-1", 2001, FAM_TRAIN))
    config["pretraining_sources"].append(
        _source("dagger_round", "dg-2", [_aid("dagger_round", "dg-2", "dg-ep-1")]))
    config["splits"]["dev"]["ancestor_ids"].append(_aid("dagger_round", "dg-2", "dg-ep-1"))
    report = validate_splits(config)
    assert report["passed"] is False
    assert "X_TRAIN_OVERLAP" in codes(report)
    assert group_key(FAM_TRAIN, "dg-ep-1") == group_key(FAM_TRAIN, "dg-ep-1")


def test_shared_trajectory_digest_merges_groups() -> None:
    traj = "a" * 64
    config = base_config()
    config["ancestors"][0]["trajectory_sha256"] = traj
    config["ancestors"].append(_ancestor("teacher_query", "tq-1", "another-ep", 2, FAM_TRAIN,
                                         trajectory_sha256=traj))
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(
        _aid("teacher_query", "tq-1", "another-ep"))
    config["splits"]["train"]["ancestor_ids"].append(
        _aid("teacher_query", "tq-1", "another-ep"))
    report = validate_splits(config)
    assert report["passed"] is False
    assert "A_DUPLICATE_GROUP" in codes(report)


# ---------------------------------------------------------------------------
# 6b. "frozen" must mean "already frozen": timestamps are real and timezone-aware
# ---------------------------------------------------------------------------


def _parse_stamp(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None, f"{text!r} has no UTC offset"
    return parsed


def _iter_time_fields(obj, path="") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else key
            if isinstance(value, str) and _TIME_FIELD_RE.search(key) and _TZ_SUFFIX_RE.search(value):
                found.append((here, value))
            found.extend(_iter_time_fields(value, here))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(_iter_time_fields(value, f"{path}[{index}]"))
    return found


@pytest.mark.parametrize("name", FROZEN_REGISTRIES)
def test_frozen_registry_has_timezone_aware_frozen_at(name: str) -> None:
    """A registry without an honest ``frozen_at`` is not frozen at all."""

    payload = json.loads((REGISTRY_DIR / name).read_text(encoding="utf-8"))
    stamp = payload.get("frozen_at")
    assert isinstance(stamp, str) and stamp, f"{name} has no frozen_at"
    assert _TZ_SUFFIX_RE.search(stamp), f"{name} frozen_at has no UTC offset: {stamp!r}"
    parsed = _parse_stamp(stamp)
    now = datetime.now(timezone.utc)
    assert parsed <= now + _FUTURE_TOLERANCE, (
        f"{name} frozen_at {stamp!r} is in the future: a future freeze means the "
        "registry was not frozen yet")
    assert parsed >= now - timedelta(days=3650), f"{name} frozen_at {stamp!r} is stale"


def test_every_timestamp_in_the_shipped_config_is_real() -> None:
    """Covers ``frozen_at`` *and* ``sealed.verified_absent.checked_at`` and friends."""

    config = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    fields = _iter_time_fields(config)
    names = {path for path, _ in fields}
    assert "frozen_at" in names
    assert "sealed.verified_absent.checked_at" in names
    now = datetime.now(timezone.utc)
    for path, value in fields:
        parsed = _parse_stamp(value)
        assert parsed <= now + _FUTURE_TOLERANCE, f"{path}={value!r} is in the future"


def test_all_frozen_stamps_agree() -> None:
    config = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    stamps = {"configs/v430/splits.json": _parse_stamp(config["frozen_at"])}
    for name in FROZEN_REGISTRIES:
        payload = json.loads((REGISTRY_DIR / name).read_text(encoding="utf-8"))
        stamps[name] = _parse_stamp(payload["frozen_at"])
    unique = {stamp for stamp in stamps.values()}
    assert len(unique) == 1, stamps


# ---------------------------------------------------------------------------
# 6c. every protection range the execution plan names must be registered
# ---------------------------------------------------------------------------


def _execution_plan_ranges() -> list[tuple[int, int]]:
    """Ranges named verbatim in ``EXECUTION_PLAN.md`` §6 (fail-closed if absent)."""

    text = EXECUTION_PLAN.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "冻结注册表" in line and "60000" in line:
            found = [(int(lo), int(hi)) for lo, hi in
                     re.findall(r"(\d{5})\s*[\u2013\-]\s*(\d{5})", line)]
            assert found, f"no seed range parsed from the §6 sentence: {line!r}"
            return found
    raise AssertionError("EXECUTION_PLAN.md §6 '冻结注册表' sentence not found")


PLAN_RANGES = _execution_plan_ranges()


def test_plan_ranges_are_parsed_and_include_the_documented_set() -> None:
    assert (BANNED_SEED_MIN, BANNED_SEED_MAX) in PLAN_RANGES
    assert (61000, 61099) in PLAN_RANGES
    assert (61400, 61499) in PLAN_RANGES


@pytest.mark.parametrize("rng", PLAN_RANGES)
def test_plan_declared_range_is_registered(rng: tuple[int, int]) -> None:
    config = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    reserved = config["reserved_training_exclusion"]
    lo, hi = rng
    if (lo, hi) == (BANNED_SEED_MIN, BANNED_SEED_MAX):
        assert (reserved["seed_min"], reserved["seed_max"]) == (lo, hi)
    else:
        covered = [(r["seed_min"], r["seed_max"]) for r in reserved["protected_ranges"]]
        assert (lo, hi) in covered, f"planned protected range {lo}-{hi} missing from {covered}"


def test_no_banned_or_protected_seed_reaches_any_split() -> None:
    config = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    reserved = config["reserved_training_exclusion"]
    protected = [(r["seed_min"], r["seed_max"]) for r in reserved["protected_ranges"]]
    for ancestor in config["ancestors"]:
        seed = ancestor["seed"]
        assert not (BANNED_SEED_MIN <= seed <= BANNED_SEED_MAX), ancestor
        assert not any(lo <= seed <= hi for lo, hi in protected), ancestor

    report = validate_splits(SHIPPED_CONFIG)
    check = report["statistics"]["seed_range_check"]
    assert check["banned_hits"] == []
    assert check["protected_hits"] == []
    assert check["train_seed_max"] < BANNED_SEED_MIN


def test_every_protected_range_declares_usage_and_evidence() -> None:
    config = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    for entry in config["reserved_training_exclusion"]["protected_ranges"]:
        assert entry["usage"] in ("unused", "spent", "reserved_for_v430"), entry
        assert isinstance(entry["evidence"], str) and entry["evidence"], entry
        assert (REPO / entry["evidence"]).exists(), f"missing evidence file {entry['evidence']}"


# ---------------------------------------------------------------------------
# 7. stability of the registry digest
# ---------------------------------------------------------------------------


def test_split_registry_sha_is_stable_and_sensitive() -> None:
    first = validate_splits(base_config())
    second = validate_splits(base_config())
    assert first["passed"] is True
    assert first["split_registry_sha"] == second["split_registry_sha"]
    assert first["split_registry_sha_scope"] == DIGEST_SCOPE_TEXT
    assert first["split_registry_sha_canonicalization"] == lineage.TEXT_CANONICALIZATION

    changed = base_config()
    changed["splits"]["dev"]["note"] = "one extra field"
    third = validate_splits(changed)
    assert third["split_registry_sha"] != first["split_registry_sha"]

    reseeded = base_config()
    reseeded["ancestors"][0]["seed"] = 2
    fourth = validate_splits(reseeded)
    assert fourth["split_registry_sha"] != first["split_registry_sha"]


def test_shipped_config_digest_is_stable() -> None:
    a = validate_splits(SHIPPED_CONFIG)
    b = validate_splits(SHIPPED_CONFIG)
    assert a["split_registry_sha"] == b["split_registry_sha"]
    assert a["split_registry_sha"] is not None and len(a["split_registry_sha"]) == 64


# ---------------------------------------------------------------------------
# 8. line-ending discipline
# ---------------------------------------------------------------------------


def test_text_scope_normalises_crlf_but_binary_scope_does_not(tmp_path: Path) -> None:
    payload = b'{"schema":"x","note":"line one\\nline two\\n"}\n'
    crlf = payload.replace(b"\n", b"\r\n")
    lf_path = tmp_path / "lf.json"
    crlf_path = tmp_path / "crlf.json"
    lf_path.write_bytes(payload)
    crlf_path.write_bytes(crlf)
    assert crlf != payload

    identity_lf = file_identity(lf_path, DIGEST_SCOPE_TEXT)
    identity_crlf = file_identity(crlf_path, DIGEST_SCOPE_TEXT)
    assert identity_lf["sha256"] == identity_crlf["sha256"]
    assert identity_lf["digest_scope"] == DIGEST_SCOPE_TEXT == identity_crlf["digest_scope"]

    raw_lf = file_identity(lf_path, DIGEST_SCOPE_BINARY)
    raw_crlf = file_identity(crlf_path, DIGEST_SCOPE_BINARY)
    assert raw_lf["sha256"] != raw_crlf["sha256"]
    assert raw_lf["digest_scope"] == DIGEST_SCOPE_BINARY


def test_lone_cr_is_folded_and_scopes_are_explicit() -> None:
    assert text_lf_digest(b"a\r\nb") == text_lf_digest(b"a\nb") == text_lf_digest(b"a\rb")
    assert binary_raw_digest(b"a\r\nb") != binary_raw_digest(b"a\nb")
    with pytest.raises(lineage.LineageError):
        lineage.digest_bytes(b"x", "sha256")
    assert text_lf_digest("中文\r\n") == text_lf_digest("中文\n")


# ---------------------------------------------------------------------------
# 9. strict / degraded tier
# ---------------------------------------------------------------------------


def _degraded_config() -> dict:
    config = base_config()
    config["pretraining_sources"].append({
        "kind": "perception_pretrain",
        "source_id": "bootstrap-family-level",
        "status": "used_for_training",
        "ancestor_enumeration": "family_level",
        "ancestor_ids": [],
        "touched_families": [FAM_TRAIN],
        "evidence_paths": ["ASSET_SYNC.md"],
        "uses_retail_replay_fixtures": False,
    })
    return config


def test_family_level_without_limit_fails_strict_and_degrades_otherwise() -> None:
    config = _degraded_config()
    strict = validate_splits(config, strict=True)
    assert strict["passed"] is False
    assert strict["mode"] == "strict"
    assert "P_SOURCE_ENUMERATION_LIMIT" in codes(strict)

    relaxed = validate_splits(config, strict=False)
    assert relaxed["passed"] is True, relaxed["problems"]
    assert relaxed["mode"] == "degraded"
    assert [p["code"] for p in relaxed["degradations"]] == ["P_SOURCE_ENUMERATION_LIMIT"]
    assert relaxed["problems"] == []
    assert relaxed["statistics"]["degradation_count"] == 1


def test_declared_enumeration_limit_passes_strict() -> None:
    config = _degraded_config()
    config["pretraining_sources"][-1]["enumeration_limit"] = (
        "the unit is a counterfactual state, not an episode")
    report = validate_splits(config, strict=True)
    assert report["passed"] is True, report["problems"]
    assert report["statistics"]["pretraining_sources_family_level"] == [
        "perception_pretrain/bootstrap-family-level"]


def test_degraded_mode_never_hides_a_lineage_violation() -> None:
    config = _degraded_config()
    config["reserved_training_exclusion"]["seed_min"] = 1000     # wrong banned range
    aid = _aid("teacher_query", "tq-1", "banned-ep")
    config["ancestors"].append(_ancestor("teacher_query", "tq-1", "banned-ep", 60050, FAM_TRAIN))
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(aid)
    config["splits"]["train"]["ancestor_ids"].append(aid)
    relaxed = validate_splits(config, strict=False)
    assert relaxed["passed"] is False
    assert "S_BANNED_SEED" in codes(relaxed)


# --- the four hard classes must fail in *both* tiers -----------------------


def _viol_banned_seed() -> dict:
    config = base_config()
    aid = _aid("teacher_query", "tq-1", "banned-ep")
    config["ancestors"].append(_ancestor("teacher_query", "tq-1", "banned-ep", 60123, FAM_TRAIN))
    source_of(config, "teacher_query", "tq-1")["ancestor_ids"].append(aid)
    config["splits"]["train"]["ancestor_ids"].append(aid)
    return config


def _viol_sealed() -> dict:
    config = base_config()
    config["splits"]["dev"]["note"] = "trace copied out of sealed/results"
    return config


def _viol_holdout_touched() -> dict:
    config = base_config()
    source_of(config, "reward_detector", "rd-1")["touched_families"] = [HOLDOUT_B]
    return config


def _viol_train_dev_crossing() -> dict:
    config = base_config()
    config["splits"]["train"]["ancestor_ids"].append(
        _aid("model_selection", "ms-1", "dev-seed-1000"))
    return config


@pytest.mark.parametrize("build,code", [
    (_viol_banned_seed, "S_BANNED_SEED"),
    (_viol_sealed, "SEALED_REFERENCED"),
    (_viol_holdout_touched, "H_TOUCHED_BY_SOURCE"),
    (_viol_train_dev_crossing, "X_TRAIN_OVERLAP"),
], ids=["banned_seed", "sealed", "holdout_touched", "train_dev_crossing"])
@pytest.mark.parametrize("strict", [True, False], ids=["strict", "degraded"])
def test_critical_violations_fail_in_both_tiers(build, code: str, strict: bool) -> None:
    report = validate_splits(build(), strict=strict)
    assert report["passed"] is False
    assert report["exit_code"] == 2
    assert code in codes(report), (code, sorted(codes(report)))
    assert report["degradations"] == []


# ---------------------------------------------------------------------------
# 10. registry bindings (file identity, not inlined content)
# ---------------------------------------------------------------------------


def _write_registry(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8", newline="\n")
    return text_lf_digest(path.read_bytes())


def test_bound_registry_with_wrong_sha_is_rejected(tmp_path: Path) -> None:
    config = base_config()
    config["families"] = None
    del config["families"]
    registry = tmp_path / "task_families.json"
    real = _write_registry(registry, {"schema": lineage.FAMILY_REGISTRY_SCHEMA,
                                      "families": base_config()["families"]})
    config["registries"] = [{
        "role": "task_families",
        "path": str(registry).replace("\\", "/"),
        "sha256": "0" * 64,
        "digest_scope": DIGEST_SCOPE_TEXT,
        "frozen": True,
    }]
    report = validate_splits(config)
    assert report["passed"] is False
    assert "P_BINDING_SHA_MISMATCH" in codes(report)

    config["registries"][0]["sha256"] = real
    report = validate_splits(config)
    assert report["passed"] is True, report["problems"]


def test_missing_bound_registry_fails_strict() -> None:
    config = base_config()
    del config["families"]
    config["registries"] = [{
        "role": "task_families",
        "path": "does/not/exist.json",
        "sha256": "0" * 64,
        "digest_scope": DIGEST_SCOPE_TEXT,
        "frozen": True,
    }]
    report = validate_splits(config)
    assert report["passed"] is False
    assert {"P_BINDING_MISSING", "P_FAMILIES_MISSING"} <= codes(report)


def test_unreadable_config_path_fails_closed(tmp_path: Path) -> None:
    report = validate_splits(tmp_path / "nope.json")
    assert report["passed"] is False
    assert report["exit_code"] == 2
    assert codes(report) == {"P_CONFIG_UNREADABLE"}


def test_exit_code_helper() -> None:
    assert lineage.exit_code_for({"passed": True}) == 0
    assert lineage.exit_code_for({"passed": False}) == 2


# ---------------------------------------------------------------------------
# 11. CLI wiring — ``tools/rl/v430.py validate-splits`` (real subprocess)
#
# These four tests execute the actual CLI, so they cover the glue the pure
# function cannot: the exit codes (0 / 3 / 2), the exclusive ``--out`` contract
# and the shape of the evidence document.  ``validate_splits`` itself keeps its
# own exit-code helper (2 for a rejected registry); the CLI maps "a valid
# measurement that misses the requested gate" to 3 as
# ``EXECUTION_PLAN.md`` §4 requires — test ``..._exit_3_...`` pins that mapping.
# ---------------------------------------------------------------------------

CLI = REPO / "tools" / "rl" / "v430.py"
CLI_TIMEOUT_S = 600


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run ``tools/rl/v430.py`` as the user would, from the repository root."""

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=str(REPO), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=CLI_TIMEOUT_S, check=False)


def _repo_workdir(tmp_path: Path) -> Path:
    """Temp dir **inside** the repository tree.

    ``validate_splits`` resolves the frozen registry bindings against the
    config's directory and then its ancestors, so a copy of the config placed
    under the repository root still resolves the real registries; a copy in the
    system temp directory would only report ``P_BINDING_MISSING``.  Keeping it
    under ``.tmp_*`` keeps it out of Git and the caller removes it again.
    """

    work = REPO / ".tmp_v430_w09b_cli" / tmp_path.name
    work.mkdir(parents=True, exist_ok=True)
    return work


def test_cli_validate_splits_writes_passing_evidence(tmp_path: Path) -> None:
    out = tmp_path / "splits-001.json"
    proc = _run_cli("validate-splits",
                    "--config", "configs/v430/splits.json",
                    "--out", str(out))
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert out.is_file(), "the CLI must create the evidence document"
    payload = json.loads(out.read_text(encoding="utf-8"))

    # The CLI must record the *pure function's* digest, not one of its own.
    report = validate_splits(SHIPPED_CONFIG)
    assert report["passed"] is True, report["problems"]
    assert payload["split_registry_sha"] == report["split_registry_sha"]
    assert re.fullmatch(r"[0-9a-f]{64}", payload["split_registry_sha"])
    assert payload["digest_scope"] == report["split_registry_sha_scope"]
    assert payload["split_registry_sha_scope"] == report["split_registry_sha_scope"]

    # Exit 0 is a lineage verdict, never a capability claim.
    assert payload["capability_gate"] in ("not_run", "not_applicable")
    assert payload["command_class"] == "data"
    assert payload["exit_code"] == 0
    assert payload["execution_status"] == "ok"
    assert payload["protocol_validity"] == "valid"
    result = payload["research_result"]
    assert result["status"] == "pass"
    assert result["passed"] is True
    assert result["problems"] == []
    assert result["problem_count"] == 0
    assert result["statistics"]["splits"]["train"]["declared_ancestors"] == 328


def test_cli_validate_splits_exit_3_when_registry_is_rejected(tmp_path: Path) -> None:
    """A *valid* measurement that fails the gate must exit 3, not 2.

    Construction: copy the shipped config and append one **dev** ancestor id to
    ``splits.train.ancestor_ids``.  That is a real train/dev crossing
    (``X_TRAIN_OVERLAP`` + ``A_MULTI_SPLIT``) and nothing else: the registries
    still resolve and still match their pinned sha256, so the validator measures
    a well-formed registry and rejects it on a lineage rule.  ``lineage``'s own
    helper answers 2 for that verdict; the CLI must answer 3.
    """

    work = _repo_workdir(tmp_path)
    try:
        config = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8-sig"))
        crossing = config["splits"]["dev"]["ancestor_ids"][0]
        assert crossing not in config["splits"]["train"]["ancestor_ids"]
        config["splits"]["train"]["ancestor_ids"].append(crossing)
        mutated = work / "splits-train-dev-crossing.json"
        mutated.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8", newline="\n")

        out = work / "splits-bad.json"
        proc = _run_cli("validate-splits", "--config", str(mutated), "--out", str(out))
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["exit_code"] == 3
        assert payload["execution_status"] == "gate_fail"
        assert payload["protocol_validity"] == "valid"
        result = payload["research_result"]
        assert result["status"] == "fail"
        assert result["passed"] is False
        assert "X_TRAIN_OVERLAP" in result["problem_codes"]
        assert payload["capability_gate"] in ("not_run", "not_applicable")
        # The pure function's own verdict is preserved for the record.
        assert result["lineage_exit_code"] == 2
        assert lineage.exit_code_for(validate_splits(mutated)) == 2
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_cli_validate_splits_refuses_existing_out(tmp_path: Path) -> None:
    out = tmp_path / "already-there.json"
    sentinel = '{"sentinel": "a previous result that must survive"}\n'
    out.write_text(sentinel, encoding="utf-8", newline="\n")
    before = out.read_bytes()

    proc = _run_cli("validate-splits",
                    "--config", "configs/v430/splits.json",
                    "--out", str(out))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert out.read_bytes() == before, "an existing --out must never be overwritten"
    assert "refusing to overwrite" in (proc.stderr or "")


def test_cli_validate_splits_missing_config_is_exit_2(tmp_path: Path) -> None:
    out = tmp_path / "never-written.json"
    proc = _run_cli("validate-splits",
                    "--config", "configs/v430/does-not-exist.json",
                    "--out", str(out))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert not out.exists(), "invalid inputs must not produce an evidence file"
