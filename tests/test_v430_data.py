"""v4.3.0 W03C — behaviour tests for the dataset manifest loader.

**Synthetic fixture only.**  Every byte below is generated inside ``tmp_path``;
this file never reads a real ``.tmp_*`` corpus, never launches the game and never
trains or evaluates.  A green run therefore proves the *loader's decisions*, and
is **not** evidence of any dataset, capability, gate or training result.

**合成测试，不是游戏/能力证据。**  （Synthetic test fixture — not game or
capability evidence.  No engine was started, no real corpus was read.)

Each test asserts a decision the validator must make (accept / reject), which
``P_*`` code it must give, and — where the guide requires it — that the message
names the offending shard, field or ancestor.  Not merely "a dict came back".

Coverage map (``DEVELOPMENT_GUIDE.md`` §6.1, ten fail-closed rules):

1. a legal manifest is ``valid`` **and** ``complete`` with ``problems == []``
   (:func:`test_complete_manifest_is_valid_and_complete`);
2. a sha256 that disagrees with the bytes is rejected **and** the problem names
   the shard, the expected value and the actual value
   (:func:`test_sha256_mismatch_names_shard_expected_and_actual`);
3. a ``size_bytes`` that disagrees with the disk is rejected
   (:func:`test_size_mismatch_is_rejected`);
4. a file under an explicit root that the manifest does not list is rejected,
   while the same corpus without roots says the scan *did not run*
   (:func:`test_unregistered_shard_is_rejected`);
5. a missing required field is rejected and named, at manifest **and** shard
   level (:func:`test_missing_field_is_rejected_and_named`);
6. one ancestor group in train and dev is rejected and names the ancestor /
   group, while the same group twice inside one split is not a crossing
   (:func:`test_train_dev_ancestor_crossing_is_rejected`,
   :func:`test_same_ancestor_group_inside_one_split_is_not_a_crossing`);
7. a ``sha256`` with no ``digest_scope`` is rejected, at manifest, nested and
   shard level (:func:`test_bare_sha256_without_digest_scope_is_rejected`);
8. ``status="partial"`` is never a complete dataset, and ``partial_view`` keeps
   only completed + immutable shards
   (:func:`test_partial_manifest_is_never_a_complete_dataset`);
9. line endings: the same manifest content as CRLF and LF has the **same**
   ``text/lf-normalized`` digest and a **different** ``binary/raw`` digest
   (:func:`test_crlf_and_lf_manifest_copies_share_the_text_digest`);
10. every code that escapes is in :data:`manifest.PROBLEM_CODES`
    (:func:`test_every_emitted_code_is_registered`).

Plus the cross-checks that keep the above from being vacuous: the expected-digest
comparisons, the split-registry binding, duplicate paths, ``counts`` arithmetic
and the module's stdlib-only/exit-code discipline.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from touhou_rl.v430.data import lineage
from touhou_rl.v430.data import manifest as manifest_module
from touhou_rl.v430.data.lineage import (
    DIGEST_SCOPE_BINARY,
    DIGEST_SCOPE_TEXT,
    ancestor_id,
    binary_raw_digest,
    digest_bytes,
    group_key,
    text_lf_digest,
)
from touhou_rl.v430.data.manifest import (
    KNOWN_OBS_CONTRACTS,
    LEGACY_OBS_CONTRACTS,
    MANIFEST_SCHEMA,
    O2_OBSERVATION_CHANNELS,
    OBS_CONTRACT_O2,
    PARTIAL_DISCLAIMER,
    PROBLEM_CODES,
    ManifestError,
    assert_o2_compatible,
    exit_code_for,
    load_manifest,
    o2_compatible,
    partial_view,
    validate_manifest,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST_SOURCE = REPO / "touhou_rl" / "v430" / "data" / "manifest.py"

FAMILY_TRAIN = "th08-stage1-normal-team0"
FAMILY_DEV = "th08-stage2-normal-team0"
TRAIN_ANCESTOR = ancestor_id("human_replay", "th08-r2", "ep-0001", FAMILY_TRAIN)
DEV_ANCESTOR = ancestor_id("human_replay", "th08-r2", "ep-0007", FAMILY_DEV)

#: A *synthetic* split config.  It is deliberately not a real frozen registry:
#: the loader only binds the sha256 that ``lineage.validate_splits`` computes for
#: it, so the test never depends on the real ``configs/v430/splits.json``.
SPLIT_CONFIG: dict[str, Any] = {
    "schema": "v430-splits/1",
    "spec_revision": "v430-plan-r2",
    "experiment_id": "w03c-synthetic-manifest-fixture",
    "frozen_at": "2026-01-01T00:00:00+08:00",
    "timezone": "+08:00",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _hex(label: str) -> str:
    """A deterministic 64-hex stand-in digest (never a real artefact's)."""

    return binary_raw_digest(label.encode("utf-8"))


def _split_registry_sha() -> str:
    report = lineage.validate_splits(SPLIT_CONFIG)
    sha = report["split_registry_sha"]
    assert isinstance(sha, str) and len(sha) == 64
    return sha


def _identity(label: str, scope: str = DIGEST_SCOPE_TEXT) -> dict[str, str]:
    return {"sha256": _hex(label), "digest_scope": scope}


def _shard(
    path: str,
    payload: bytes,
    *,
    split: str,
    ancestors: list[Any],
    episodes: list[str],
    frame_range: tuple[int, int] = (0, 120),
    decisions: int = 10,
    labels_valid: int = 9,
    status: str = "completed",
    immutable: bool = True,
) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": binary_raw_digest(payload),
        "digest_scope": DIGEST_SCOPE_BINARY,
        "episodes": list(episodes),
        "ancestors": list(ancestors),
        "frame_range": list(frame_range),
        "decisions": decisions,
        "labels_valid": labels_valid,
        "duplicates": {"episode_repeats": 0, "frame_overlap": 0},
        "split": split,
        "status": status,
        "immutable": immutable,
    }


def _counts(shards: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "shards": len(shards),
        "decisions": sum(item["decisions"] for item in shards),
        "labels_valid": sum(item["labels_valid"] for item in shards),
        "episodes": sum(len(item["episodes"]) for item in shards),
    }


def _document(shards: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": "th08-w03c-synthetic",
        "parent_ids": [],
        "code_snapshot": {"git_commit": "0" * 40, "dirty": False},
        "engine_sha": _identity("th08-engine-binary", DIGEST_SCOPE_BINARY),
        "game_data_manifest_sha": _identity("game-data-manifest"),
        "obs_digest": _identity("o2-observation-parameters"),
        "action_digest": _identity("action-mask-tables"),
        "label_digest": _identity("label-validity-policy"),
        "split_registry_sha": _identity("split-registry", DIGEST_SCOPE_TEXT),
        "shards": shards,
        "counts": _counts(shards),
        "status": "complete",
    }
    doc["split_registry_sha"] = {
        "sha256": _split_registry_sha(),
        "digest_scope": DIGEST_SCOPE_TEXT,
    }
    # The fixture is O2 data by declaration: the guard refuses a manifest that
    # cannot say which observation contract its pixels were recorded under.
    doc["obs_contract"] = OBS_CONTRACT_O2
    doc.update(overrides)
    return doc


def _encode(doc: Mapping[str, Any], newline: str = "\n") -> bytes:
    text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True)
    if newline == "\r\n":
        text = text.replace("\n", "\r\n")
    return text.encode("utf-8")


@dataclasses.dataclass
class Fixture:
    tmp: Path
    root: Path
    doc: dict[str, Any]
    payloads: dict[str, bytes]

    def path(self, name: str = "manifest.json") -> Path:
        return self.tmp / name

    def write(self, doc: Mapping[str, Any], name: str = "manifest.json",
              newline: str = "\n") -> Path:
        target = self.path(name)
        target.write_bytes(_encode(doc, newline))
        return target


@pytest.fixture()
def fixture(tmp_path: Path) -> Fixture:
    """Two shards (train + dev), each with its own task ancestor."""

    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    (root / "dev").mkdir(parents=True)
    train_payload = b"TH08-TRAIN\x00\x01\x02" * 7
    dev_payload = b"TH08-DEV\x03\x04" * 5
    (root / "train" / "s1.bin").write_bytes(train_payload)
    (root / "dev" / "s1.bin").write_bytes(dev_payload)
    shards = [
        _shard("train/s1.bin", train_payload, split="train",
               ancestors=[TRAIN_ANCESTOR], episodes=["ep-0001"],
               frame_range=(0, 120), decisions=10, labels_valid=9),
        _shard("dev/s1.bin", dev_payload, split="dev",
               ancestors=[DEV_ANCESTOR], episodes=["ep-0007"],
               frame_range=(0, 60), decisions=6, labels_valid=6),
    ]
    return Fixture(
        tmp=tmp_path,
        root=root,
        doc=_document(shards),
        payloads={"train/s1.bin": train_payload, "dev/s1.bin": dev_payload},
    )


def _run(fx: Fixture, doc: Mapping[str, Any], **kwargs: Any):
    """Load the manifest from disk and validate it; returns ``(manifest, report)``."""

    path = fx.write(doc)
    manifest = load_manifest(path, roots={"data": fx.root})
    kwargs.setdefault("base_dir", fx.root)
    kwargs.setdefault("roots", {"data": fx.root})
    return manifest, validate_manifest(manifest, **kwargs)


def _only(report: Mapping[str, Any], code: str) -> dict[str, Any]:
    items = [item for item in report["problems"] if item["code"] == code]
    assert len(items) == 1, (code, [item["code"] for item in report["problems"]])
    return items[0]


def _codes(report: Mapping[str, Any]) -> set[str]:
    return {item["code"] for item in report["problems"]}


# ---------------------------------------------------------------------------
# 1. the legal case
# ---------------------------------------------------------------------------


def test_complete_manifest_is_valid_and_complete(fixture: Fixture) -> None:
    """Rule 1: a consistent, complete manifest passes with no problems at all."""

    manifest, report = _run(
        fixture,
        fixture.doc,
        expected_obs_digest=fixture.doc["obs_digest"],
        expected_action_digest=fixture.doc["action_digest"]["sha256"],
        expected_label_digest=fixture.doc["label_digest"],
        split_config=SPLIT_CONFIG,
    )

    assert report["valid"] is True
    assert report["complete"] is True
    assert report["problems"] == []
    assert report["degradations"] == []
    assert report["exit_code"] == 0
    assert report["status"] == "complete"
    assert report["dataset_id"] == "th08-w03c-synthetic"

    # Every check that could run, ran — including the two the task calls out.
    assert report["checks_performed"] == [
        "identity", "shards", "counts", "ancestors", "digest_scope",
        "expected_digests", "split_registry", "roots", "status",
    ]
    # ... and nothing was silently skipped.
    assert report["not_checked"], "a validator that checks nothing must say so"
    assert not [item for item in report["not_checked"]
                if "was not compared" in item or "did not run" in item]

    assert report["statistics"]["shards_listed"] == 2
    assert report["statistics"]["shards_readable"] == 2
    assert report["statistics"]["shards_by_split"] == {"dev": 1, "train": 1}
    assert report["statistics"]["decisions"] == 16
    assert report["statistics"]["labels_valid"] == 15
    assert report["statistics"]["ancestor_groups"] == 2
    assert report["statistics"]["ancestor_ids"] == 2

    # The manifest file itself is a *text* artefact: identity is LF-normalised.
    assert report["manifest"]["digest_scope"] == DIGEST_SCOPE_TEXT
    assert manifest.source is not None
    assert manifest.source["digest_scope"] == DIGEST_SCOPE_TEXT
    assert manifest.source["sha256"] == text_lf_digest(fixture.path().read_bytes())


# ---------------------------------------------------------------------------
# 2. sha256 vs the bytes on disk (the core check)
# ---------------------------------------------------------------------------


def test_sha256_mismatch_names_shard_expected_and_actual(fixture: Fixture) -> None:
    """Rule 2: a wrong sha256 is rejected and names shard + expected + actual."""

    doc = copy.deepcopy(fixture.doc)
    wrong = _hex("not-the-train-bytes")
    doc["shards"][0]["sha256"] = wrong

    _, report = _run(fixture, doc)

    assert report["valid"] is False
    assert report["complete"] is False
    problem = _only(report, "P_SHARD_SHA_MISMATCH")
    actual = binary_raw_digest(fixture.payloads["train/s1.bin"])
    assert problem["path"] == "train/s1.bin"
    assert problem["expected"] == wrong
    assert problem["actual"] == actual
    assert problem["digest_scope"] == DIGEST_SCOPE_BINARY
    # The message must be enough to act on without re-reading the report.
    assert "train/s1.bin" in problem["message"]
    assert wrong in problem["message"]
    assert actual in problem["message"]
    assert problem["resolved"].endswith("data/train/s1.bin")
    assert report["exit_code"] == 2

    # The *other* shard is untouched: the problem is attributed, not sprayed.
    assert [item["path"] for item in report["problems"]
            if item["code"] == "P_SHARD_SHA_MISMATCH"] == ["train/s1.bin"]


def test_size_mismatch_is_rejected(fixture: Fixture) -> None:
    """Rule 3: ``size_bytes`` must equal the number of bytes on disk."""

    doc = copy.deepcopy(fixture.doc)
    doc["shards"][1]["size_bytes"] = len(fixture.payloads["dev/s1.bin"]) + 1

    _, report = _run(fixture, doc)

    assert report["valid"] is False
    problem = _only(report, "P_SHARD_SIZE_MISMATCH")
    assert problem["path"] == "dev/s1.bin"
    assert problem["expected"] == len(fixture.payloads["dev/s1.bin"]) + 1
    assert problem["actual"] == len(fixture.payloads["dev/s1.bin"])


def test_missing_shard_file_is_rejected(fixture: Fixture) -> None:
    """Rule 2's sibling: a registered shard that is not on disk is fatal."""

    (fixture.root / "dev" / "s1.bin").unlink()

    _, report = _run(fixture, fixture.doc)

    problem = _only(report, "P_SHARD_MISSING")
    assert problem["path"] == "dev/s1.bin"
    assert report["valid"] is False


# ---------------------------------------------------------------------------
# 4. unregistered shards (only an explicit roots scan can see them)
# ---------------------------------------------------------------------------


def test_unregistered_shard_is_rejected(fixture: Fixture) -> None:
    """Rule 4: data under a root that the manifest does not name is rejected."""

    extra = fixture.root / "train" / "extra.bin"
    extra.write_bytes(b"unregistered bytes")

    _, report = _run(fixture, fixture.doc)

    assert report["valid"] is False
    problem = _only(report, "P_SHARD_UNREGISTERED")
    assert problem["path"] == "train/extra.bin"
    assert problem["root"] == "data"
    assert problem["resolved"].endswith("data/train/extra.bin")

    # Without explicit roots the scan cannot run; that is recorded, never passed.
    blind_manifest = load_manifest(fixture.path())
    blind = validate_manifest(blind_manifest, base_dir=fixture.root)
    assert blind["valid"] is True  # nothing else is wrong ...
    assert "roots" not in blind["checks_performed"]  # ... but the scan did not run
    assert [item for item in blind["not_checked"]
            if "unregistered-shard scan did not run" in item]


# ---------------------------------------------------------------------------
# 5. missing fields
# ---------------------------------------------------------------------------


def test_missing_field_is_rejected_and_named(fixture: Fixture) -> None:
    """Rule 5: a missing required field is rejected and identified by name."""

    doc = copy.deepcopy(fixture.doc)
    del doc["label_digest"]

    _, report = _run(fixture, doc)

    assert report["valid"] is False
    problem = _only(report, "P_MISSING_FIELD")
    assert problem["field"] == "label_digest"
    assert "label_digest" in problem["message"]

    # The same rule applies inside a shard, with an indexed field path.
    doc = copy.deepcopy(fixture.doc)
    del doc["shards"][1]["labels_valid"]
    _, report = _run(fixture, doc)
    missing = [item for item in report["problems"]
               if item["code"] == "P_MISSING_FIELD"]
    assert [item["field"] for item in missing] == ["shards[1].labels_valid"]


def test_missing_counts_key_is_rejected(fixture: Fixture) -> None:
    """``counts`` is mandatory per §6.1 and each count is named."""

    doc = copy.deepcopy(fixture.doc)
    del doc["counts"]["episodes"]

    _, report = _run(fixture, doc)

    problem = _only(report, "P_MISSING_FIELD")
    assert problem["field"] == "counts.episodes"


# ---------------------------------------------------------------------------
# 6. ancestor crossing
# ---------------------------------------------------------------------------


def test_train_dev_ancestor_crossing_is_rejected(fixture: Fixture) -> None:
    """Rule 6: one ancestor group in train and dev is a lineage violation."""

    doc = copy.deepcopy(fixture.doc)
    doc["shards"][1]["ancestors"] = [TRAIN_ANCESTOR]

    _, report = _run(fixture, doc)

    assert report["valid"] is False
    problem = _only(report, "P_ANCESTOR_CROSSING")
    assert problem["ancestor_group"] == group_key(FAMILY_TRAIN, "ep-0001")
    assert problem["ancestor_ids"] == [TRAIN_ANCESTOR]
    assert problem["splits"] == ["dev", "train"]
    assert TRAIN_ANCESTOR in problem["message"]
    assert "train" in problem["message"] and "dev" in problem["message"]


def test_same_ancestor_group_inside_one_split_is_not_a_crossing(fixture: Fixture) -> None:
    """A re-seeded copy inside *one* split is one sample, not a crossing.

    The grouping is ``lineage.group_key(task_family, episode)``, so a different
    source kind and a different seed for the same episode stay in the group and
    are allowed as long as the split is the same.
    """

    payload = b"TH08-TRAIN-2\x05" * 3
    (fixture.root / "train" / "s2.bin").write_bytes(payload)
    doc = copy.deepcopy(fixture.doc)
    doc["shards"].append(_shard(
        "train/s2.bin", payload, split="train",
        ancestors=[ancestor_id("dagger_round", "wp05-r3", "ep-0001", FAMILY_TRAIN)],
        episodes=["ep-0001"], frame_range=(0, 40), decisions=4, labels_valid=4))
    doc["counts"] = _counts(doc["shards"])

    _, report = _run(fixture, doc)

    assert "P_ANCESTOR_CROSSING" not in _codes(report)
    assert report["valid"] is True
    # Both ancestors collapse into one independent group.
    assert report["statistics"]["ancestor_groups"] == 2
    assert report["statistics"]["ancestor_ids"] == 3


def test_quarantine_is_exempt_from_the_crossing_rule(fixture: Fixture) -> None:
    """``quarantine`` names barred records, which may repeat a train episode."""

    payload = b"TH08-QUARANTINE\x06"
    (fixture.root / "train" / "q.bin").write_bytes(payload)
    doc = copy.deepcopy(fixture.doc)
    doc["shards"].append(_shard(
        "train/q.bin", payload, split="quarantine",
        ancestors=[TRAIN_ANCESTOR], episodes=["ep-0001"],
        frame_range=(0, 10), decisions=2, labels_valid=0,
        status="quarantined", immutable=False))
    doc["counts"] = _counts(doc["shards"])

    _, report = _run(fixture, doc, strict=False)

    assert "P_ANCESTOR_CROSSING" not in _codes(report)


# ---------------------------------------------------------------------------
# 7. digests must always carry their scope
# ---------------------------------------------------------------------------


def test_bare_sha256_without_digest_scope_is_rejected(fixture: Fixture) -> None:
    """Rule 10: an identity with sha256 but no ``digest_scope`` is refused."""

    doc = copy.deepcopy(fixture.doc)
    doc["obs_digest"] = doc["obs_digest"]["sha256"]  # bare hex, no *_scope sibling

    _, report = _run(fixture, doc)

    assert report["valid"] is False
    problem = _only(report, "P_DIGEST_SCOPE_MISSING")
    assert problem["field"] == "obs_digest"
    assert "digest_scope" in problem["message"]

    # A digest nested where no dedicated check looks is caught by the same rule.
    doc = copy.deepcopy(fixture.doc)
    doc["code_snapshot"]["source_tree"] = {"sha256": _hex("tree")}
    _, report = _run(fixture, doc)
    problem = _only(report, "P_DIGEST_SCOPE_MISSING")
    assert problem["field"] == "code_snapshot.source_tree"

    # And at shard level, where the digest is the shard's own identity.
    doc = copy.deepcopy(fixture.doc)
    del doc["shards"][0]["digest_scope"]
    _, report = _run(fixture, doc)
    problem = _only(report, "P_DIGEST_SCOPE_MISSING")
    assert problem["field"] == "shards[0]"


def test_unknown_digest_scope_is_rejected(fixture: Fixture) -> None:
    """A scope outside ``lineage.DIGEST_SCOPES`` is not "close enough"."""

    doc = copy.deepcopy(fixture.doc)
    doc["action_digest"] = {"sha256": _hex("actions"),
                            "digest_scope": "binary/raw-normalized-ish"}

    _, report = _run(fixture, doc)

    problem = _only(report, "P_DIGEST_SCOPE_UNKNOWN")
    assert problem["field"] == "action_digest"


# ---------------------------------------------------------------------------
# 8. partial is never complete
# ---------------------------------------------------------------------------


def test_partial_manifest_is_never_a_complete_dataset(fixture: Fixture) -> None:
    """Rule 9: partial is refused as a whole, and readable only as a view."""

    doc = copy.deepcopy(fixture.doc)
    doc["status"] = "partial"
    doc["shards"][1]["status"] = "writing"
    doc["shards"][1]["immutable"] = False

    manifest, strict_report = _run(fixture, doc)
    assert strict_report["valid"] is False
    assert strict_report["complete"] is False
    assert strict_report["exit_code"] == 2
    problem = _only(strict_report, "P_PARTIAL_INCOMPLETE")
    assert problem["status"] == "partial"
    assert "partial_view" in problem["message"]

    # The completed part may still be validated — completeness stays False.
    _, degraded = _run(fixture, doc, strict=False)
    assert degraded["valid"] is True
    assert degraded["complete"] is False
    assert degraded["exit_code"] == 4
    assert [item["code"] for item in degraded["degradations"]] == ["P_PARTIAL_INCOMPLETE"]
    assert degraded["problems"] == []

    view = partial_view(manifest, base_dir=fixture.root)
    assert view.complete is False
    assert view.source_status == "partial"
    assert view.disclaimer == PARTIAL_DISCLAIMER
    assert [shard.path for shard in view.shards] == ["train/s1.bin"]
    assert view.excluded == ({"path": "dev/s1.bin", "status": "writing",
                              "immutable": False, "reason": "status_writing"},)
    assert view.report is not None
    assert view.report["valid"] is True
    assert view.report["complete"] is False
    assert view.report["status"] == "partial"
    # The view's own counts describe the included shards only.
    assert view.manifest.counts["shards"] == 1
    assert view.manifest.counts["decisions"] == 10

    payload = view.to_dict()
    assert payload["complete"] is False
    assert payload["disclaimer"] == PARTIAL_DISCLAIMER
    assert [item["path"] for item in payload["excluded"]] == ["dev/s1.bin"]

    # completed_only=False keeps the mutable shard but still claims nothing.
    loose = partial_view(manifest, completed_only=False, base_dir=fixture.root)
    assert [shard.path for shard in loose.shards] == ["train/s1.bin", "dev/s1.bin"]
    assert loose.complete is False
    assert loose.excluded == ()


def test_unrecorded_shard_status_cannot_be_read(fixture: Fixture) -> None:
    """A shard that does not *say* it is completed+immutable is not readable."""

    doc = copy.deepcopy(fixture.doc)
    for shard in doc["shards"]:
        shard.pop("status")
        shard.pop("immutable")

    manifest, report = _run(fixture, doc, strict=False)
    assert report["complete"] is False

    view = partial_view(manifest, base_dir=fixture.root)
    assert view.shards == ()
    assert [item["reason"] for item in view.excluded] == ["status_not_recorded"] * 2
    # A partial view with nothing readable is not valid data either.
    assert view.report is not None
    assert view.report["valid"] is False


def test_complete_manifest_may_not_list_a_mutable_shard(fixture: Fixture) -> None:
    """``status="complete"`` with a still-writable shard is a contradiction."""

    doc = copy.deepcopy(fixture.doc)
    doc["shards"][0]["immutable"] = False

    _, report = _run(fixture, doc)

    problem = _only(report, "P_SHARD_NOT_COMPLETED")
    assert problem["path"] == "train/s1.bin"
    assert problem["immutable"] is False
    assert report["complete"] is False


# ---------------------------------------------------------------------------
# 9. line endings
# ---------------------------------------------------------------------------


def test_crlf_and_lf_manifest_copies_share_the_text_digest(fixture: Fixture) -> None:
    """Rule 4 of the task: text scope normalises CRLF, binary scope must not."""

    crlf_path = fixture.write(fixture.doc, "manifest_crlf.json", newline="\r\n")
    lf_path = fixture.write(fixture.doc, "manifest_lf.json", newline="\n")
    crlf_bytes, lf_bytes = crlf_path.read_bytes(), lf_path.read_bytes()
    assert crlf_bytes != lf_bytes  # the two files really do differ

    assert text_lf_digest(crlf_bytes) == text_lf_digest(lf_bytes)
    assert binary_raw_digest(crlf_bytes) != binary_raw_digest(lf_bytes)

    crlf_manifest = load_manifest(crlf_path, roots={"data": fixture.root})
    lf_manifest = load_manifest(lf_path, roots={"data": fixture.root})
    assert crlf_manifest.source["digest_scope"] == DIGEST_SCOPE_TEXT
    assert crlf_manifest.source["sha256"] == lf_manifest.source["sha256"]
    assert crlf_manifest.source["sha256"] == text_lf_digest(lf_bytes)
    # The CRLF copy is not a different dataset: both validate identically.
    crlf_report = validate_manifest(crlf_manifest, base_dir=fixture.root,
                                    roots={"data": fixture.root})
    lf_report = validate_manifest(lf_manifest, base_dir=fixture.root,
                                  roots={"data": fixture.root})
    assert crlf_report["valid"] is True
    assert crlf_report["problems"] == lf_report["problems"] == []

    # A shard is binary/raw: the same *logical* bytes with different line endings
    # are different data, and the dispatch goes through lineage.digest_bytes.
    payload_crlf, payload_lf = b"frame\r\nnext\r\n", b"frame\nnext\n"
    assert (digest_bytes(payload_crlf, DIGEST_SCOPE_BINARY)
            == binary_raw_digest(payload_crlf))
    assert (digest_bytes(payload_crlf, DIGEST_SCOPE_BINARY)
            != digest_bytes(payload_lf, DIGEST_SCOPE_BINARY))
    assert (digest_bytes(payload_crlf, DIGEST_SCOPE_TEXT)
            == digest_bytes(payload_lf, DIGEST_SCOPE_TEXT))
    with pytest.raises(lineage.LineageError):
        digest_bytes(payload_crlf, "raw")


# ---------------------------------------------------------------------------
# expected digests / split registry binding
# ---------------------------------------------------------------------------


def test_expected_digest_mismatch_is_rejected(fixture: Fixture) -> None:
    """Rule 7: a supplied expected digest that disagrees is a rejection."""

    _, report = _run(fixture, fixture.doc,
                     expected_obs_digest=_hex("another-observation-encoder"))
    problem = _only(report, "P_OBS_DIGEST_MISMATCH")
    assert problem["expected"] == _hex("another-observation-encoder")
    assert problem["actual"] == fixture.doc["obs_digest"]["sha256"]

    # Each identity keeps its own code: no aliasing into one generic mismatch.
    _, report = _run(fixture, fixture.doc,
                     expected_label_digest=_hex("another-label-policy"))
    _only(report, "P_LABEL_DIGEST_MISMATCH")
    assert "P_OBS_DIGEST_MISMATCH" not in _codes(report)

    _, report = _run(fixture, fixture.doc,
                     expected_action_digest=_hex("another-action-table"))
    _only(report, "P_ACTION_DIGEST_MISMATCH")

    # A *matching* expectation adds no problem and is recorded as checked.
    _, report = _run(fixture, fixture.doc,
                     expected_obs_digest=fixture.doc["obs_digest"])
    assert report["valid"] is True
    assert "expected_digests" in report["checks_performed"]


def test_split_registry_sha_mismatch_is_rejected(fixture: Fixture) -> None:
    """Rule 8: ``split_registry_sha`` must equal the split config's own sha256."""

    assert fixture.doc["split_registry_sha"]["sha256"] == _split_registry_sha()

    _, report = _run(fixture, fixture.doc, split_config=SPLIT_CONFIG)
    assert report["valid"] is True
    assert report["split_registry"]["sha256"] == _split_registry_sha()
    # The synthetic config is *not* a valid split registry (no registries bound);
    # that verdict belongs to W09 and is recorded, not folded into this report.
    assert report["split_registry"]["split_config_valid"] is False

    doc = copy.deepcopy(fixture.doc)
    doc["split_registry_sha"]["sha256"] = _hex("a-different-split-registry")
    _, report = _run(fixture, doc, split_config=SPLIT_CONFIG)
    problem = _only(report, "P_SPLIT_REGISTRY_SHA_MISMATCH")
    assert problem["expected"] == _split_registry_sha()
    assert problem["actual"] == _hex("a-different-split-registry")


# ---------------------------------------------------------------------------
# schema / duplicate paths / counts arithmetic
# ---------------------------------------------------------------------------


def test_unknown_schema_is_rejected(fixture: Fixture) -> None:
    doc = copy.deepcopy(fixture.doc)
    doc["schema"] = "v430-manifest/2"

    _, report = _run(fixture, doc)

    problem = _only(report, "P_SCHEMA")
    assert problem["expected"] == MANIFEST_SCHEMA
    assert problem["actual"] == "v430-manifest/2"


def test_duplicate_shard_path_is_rejected(fixture: Fixture) -> None:
    doc = copy.deepcopy(fixture.doc)
    doc["shards"][1]["path"] = "train/s1.bin"

    _, report = _run(fixture, doc)

    problem = _only(report, "P_SHARD_DUPLICATE")
    assert problem["path"] == "train/s1.bin"
    assert (problem["first"], problem["second"]) == (0, 1)


def test_counts_must_match_the_shards(fixture: Fixture) -> None:
    doc = copy.deepcopy(fixture.doc)
    doc["counts"]["decisions"] = 99

    _, report = _run(fixture, doc)

    problem = _only(report, "P_COUNTS_MISMATCH")
    assert problem["field"] == "counts.decisions"
    assert problem["expected"] == 99
    assert problem["actual"] == 16


def test_labels_valid_may_not_exceed_decisions(fixture: Fixture) -> None:
    doc = copy.deepcopy(fixture.doc)
    doc["shards"][0]["labels_valid"] = doc["shards"][0]["decisions"] + 1
    doc["counts"] = _counts(doc["shards"])

    _, report = _run(fixture, doc)

    problem = _only(report, "P_LABELS_EXCEED_DECISIONS")
    assert problem["field"] == "shards[0].labels_valid"


# ---------------------------------------------------------------------------
# structural fail-closed behaviour of the loader itself
# ---------------------------------------------------------------------------


def test_load_rejects_a_non_object_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b"[1, 2, 3]")
    with pytest.raises(ManifestError):
        load_manifest(path)

    missing = tmp_path / "absent.json"
    with pytest.raises(ManifestError):
        load_manifest(missing)

    broken = tmp_path / "broken.json"
    broken.write_bytes(b"{not json")
    with pytest.raises(ManifestError):
        load_manifest(broken)


def test_an_invalid_caller_expectation_is_a_caller_error(fixture: Fixture) -> None:
    """A bad *expectation* raises; it is program code, not dataset content."""

    with pytest.raises(ManifestError):
        validate_manifest(fixture.doc, expected_obs_digest="not-a-digest")


def test_every_emitted_code_is_registered(fixture: Fixture) -> None:
    """Rule 10: a rejection may not appear without a declared code."""

    seen: set[str] = set()
    mutations = []
    for mutate in (
        lambda doc: doc.__setitem__("schema", "other/1"),
        lambda doc: doc.pop("label_digest"),
        lambda doc: doc["shards"][0].__setitem__("sha256", _hex("x")),
        lambda doc: doc["shards"][0].__setitem__("size_bytes", 1),
        lambda doc: doc["shards"][1].__setitem__("ancestors",
                                                 [TRAIN_ANCESTOR]),
        lambda doc: doc["shards"][0].pop("digest_scope"),
        lambda doc: doc["shards"][0].__setitem__("split", "validation"),
        lambda doc: doc["shards"][1].__setitem__("path", "train/s1.bin"),
        lambda doc: doc["counts"].__setitem__("shards", 9),
        lambda doc: doc.__setitem__("status", "partial"),
    ):
        doc = copy.deepcopy(fixture.doc)
        mutate(doc)
        _, report = _run(fixture, doc, strict=False)
        seen |= _codes(report)
        seen |= {item["code"] for item in report["degradations"]}

    (fixture.root / "train" / "extra.bin").write_bytes(b"x")
    _, report = _run(fixture, fixture.doc)
    seen |= _codes(report)

    assert seen, "no problem was produced, so the assertion below is vacuous"
    assert seen <= set(PROBLEM_CODES), sorted(seen - set(PROBLEM_CODES))


def test_problem_codes_are_documented() -> None:
    for code, meaning in PROBLEM_CODES.items():
        assert code.startswith("P_")
        assert meaning.strip()


# ---------------------------------------------------------------------------
# module discipline
# ---------------------------------------------------------------------------


def test_module_is_pure_stdlib() -> None:
    """No numpy, no torch, no engine, no v400 import — a manifest is data only."""

    source = MANIFEST_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import torch", "touhou_rl.v400",
                      "th08_rl", "subprocess", "socket"):
        assert forbidden not in source, forbidden


def test_exit_code_helper() -> None:
    """``0`` complete, ``4`` valid-but-partial, ``2`` invalid."""

    assert exit_code_for({"valid": True, "complete": True}) == 0
    assert exit_code_for({"valid": True, "complete": False}) == 4
    assert exit_code_for({"valid": False, "complete": False}) == 2
    # "complete" alone never rescues an invalid report.
    assert exit_code_for({"valid": False, "complete": True}) == 2
    assert exit_code_for({}) == 2


def test_problem_code_table_is_used_by_the_module() -> None:
    """The code table is not decorative: a stray code raises at construction."""

    with pytest.raises(ManifestError):
        manifest_module._problem("P_NOT_A_REAL_CODE", "x")  # noqa: SLF001


# ---------------------------------------------------------------------------
# 11. the observation-contract guard (W03F)
#
# The corpora on disk were recorded under a *different* observation contract
# than ``obs-v430-o2``; ``observation.o2_digest()`` compares the contract's
# parameter block, so it cannot notice "right shape, downsampled pixels".  These
# tests pin the declaration guard: it names the contract, refuses the rest, and
# offers no conversion.
# ---------------------------------------------------------------------------

#: The **real** legacy contract name, read off disk from
#: ``.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed10-shard/meta.json`` ->
#: ``contract.obs_contract``.  Not a guess, and not re-read here: the test must
#: not depend on a 29 GB asset being present.
LEGACY_CONTRACT = "obs-v420-2f"

#: The O2 geometry, from ``OBS_CONTRACT_GAP.md`` §2 / ``observation.O2_SHAPES``.
O2_DECLARED_SHAPES: dict[str, list[int]] = {
    "global_rgb": [3, 240, 320],
    "play_gray": [4, 224, 192],
    "hud_gray": [1, 448, 160],
    "executed_key_history": [8, 7],
}

#: What the corpora actually hold: no ``global_rgb`` at all, ``play_gray`` at
#: exactly 2x downsampled, ``hud_gray`` at 8x vertically downsampled, and a
#: different ``executed_key_history`` feature width.
LEGACY_DECLARED_SHAPES: dict[str, list[int]] = {
    "play_gray": [4, 112, 96],
    "hud_gray": [56, 160],
    "executed_key_history": [8, 18],
}


def test_missing_obs_contract_is_rejected(fixture: Fixture) -> None:
    """11a: a manifest that does not name its observation contract is invalid."""

    doc = copy.deepcopy(fixture.doc)
    doc.pop("obs_contract")
    _, report = _run(fixture, doc)

    assert report["valid"] is False
    assert report["o2_compatible"] is False
    problem = _only(report, "P_MISSING_FIELD")
    assert problem["field"] == "obs_contract"
    assert "obs_contract" in problem["message"]


def test_unknown_obs_contract_is_rejected(fixture: Fixture) -> None:
    """11b: a contract name that is not registered cannot be asserted to be O2."""

    doc = copy.deepcopy(fixture.doc)
    doc["obs_contract"] = "obs-v999-imaginary"
    _, report = _run(fixture, doc)

    assert report["valid"] is False
    problem = _only(report, "P_OBS_CONTRACT_UNKNOWN")
    assert "obs-v999-imaginary" in problem["message"]
    assert OBS_CONTRACT_O2 in problem["message"]
    assert problem["actual"] == "obs-v999-imaginary"


def test_legacy_obs_contract_validates_but_is_refused_by_the_o2_gate(
        fixture: Fixture) -> None:
    """11c: the real on-disk contract is a *valid* manifest and *not* O2 data."""

    doc = copy.deepcopy(fixture.doc)
    doc["obs_contract"] = LEGACY_CONTRACT
    manifest, report = _run(fixture, doc)

    # A legacy corpus is internally true.  ``valid`` is not an O2 claim.
    assert report["valid"] is True
    assert report["obs_contract"] == LEGACY_CONTRACT
    assert report["o2_compatible"] is False

    with pytest.raises(ManifestError) as excinfo:
        assert_o2_compatible(manifest)
    message = str(excinfo.value)
    assert LEGACY_CONTRACT in message          # the actual contract
    assert OBS_CONTRACT_O2 in message          # the required contract
    assert "global_rgb" in message             # a channel that does not exist
    assert "2x" in message and "8x" in message  # play 2x, hud 8x downsampled
    assert "112" in message and "56" in message
    assert "re-captured" in message            # the only offered fix


def test_o2_contract_with_non_o2_shapes_is_rejected(fixture: Fixture) -> None:
    """11d: the O2 *name* with legacy geometry is exactly the case to refuse."""

    doc = copy.deepcopy(fixture.doc)
    doc["obs_contract"] = OBS_CONTRACT_O2
    doc["obs_shapes"] = copy.deepcopy(LEGACY_DECLARED_SHAPES)
    _, report = _run(fixture, doc)

    assert report["valid"] is False
    problem = _only(report, "P_OBS_SHAPE_CONTRACT_MISMATCH")
    for channel in O2_OBSERVATION_CHANNELS:
        assert channel in problem["message"], channel
    assert "(4, 112, 96)" in problem["message"]
    assert "(4, 224, 192)" in problem["message"]
    assert problem["differences"]

    with pytest.raises(ManifestError) as excinfo:
        assert_o2_compatible(doc)
    assert "P_OBS_SHAPE_CONTRACT_MISMATCH" in str(excinfo.value)


def test_o2_compatible_routes_every_other_case_to_false() -> None:
    """11e: the non-raising router; only a genuine O2 declaration is True."""

    # Local import: ``observation`` pulls numpy, this test module does not.
    from touhou_rl.v430.observation import O2_SHAPES  # noqa: PLC0415

    declared = {name: list(O2_SHAPES[name]) for name in O2_OBSERVATION_CHANNELS}
    assert declared == O2_DECLARED_SHAPES
    assert set(O2_OBSERVATION_CHANNELS) <= set(O2_SHAPES)

    o2_doc = {"dataset_id": "d", "obs_contract": OBS_CONTRACT_O2,
              "obs_shapes": copy.deepcopy(O2_DECLARED_SHAPES)}
    assert o2_compatible(o2_doc) is True
    # A name-only claim makes no geometry claim, so there is nothing to refute.
    assert o2_compatible({"dataset_id": "d", "obs_contract": OBS_CONTRACT_O2}) is True
    # Every other shape of input is not O2-compatible, and never raises.
    assert o2_compatible({"dataset_id": "d", "obs_contract": LEGACY_CONTRACT}) is False
    assert o2_compatible({"dataset_id": "d",
                          "obs_contract": "obs-v999-imaginary"}) is False
    assert o2_compatible({"dataset_id": "d"}) is False
    assert o2_compatible("not a manifest") is False
    # Declaring O2 geometry cannot launder a legacy corpus into O2: the name wins.
    forged = {"dataset_id": "d", "obs_contract": LEGACY_CONTRACT,
              "obs_shapes": copy.deepcopy(O2_DECLARED_SHAPES)}
    assert o2_compatible(forged) is False
    with pytest.raises(ManifestError):
        assert_o2_compatible(forged)
    # The checker returns the manifest it accepted (unchanged, no conversion).
    accepted = assert_o2_compatible(o2_doc)
    assert accepted.obs_contract == OBS_CONTRACT_O2
    assert accepted.obs_shapes == O2_DECLARED_SHAPES


def test_obs_contract_codes_are_registered() -> None:
    """11f: no code escapes the table, and no contract escapes the whitelist."""

    for code in ("P_OBS_CONTRACT_UNKNOWN", "P_OBS_SHAPE_CONTRACT_MISMATCH"):
        assert code in PROBLEM_CODES
        assert PROBLEM_CODES[code].strip()

    assert OBS_CONTRACT_O2 == "obs-v430-o2"
    assert OBS_CONTRACT_O2 in KNOWN_OBS_CONTRACTS
    assert LEGACY_CONTRACT in KNOWN_OBS_CONTRACTS
    assert LEGACY_CONTRACT in LEGACY_OBS_CONTRACTS
    assert LEGACY_CONTRACT != OBS_CONTRACT_O2
    assert LEGACY_OBS_CONTRACTS[LEGACY_CONTRACT].strip()

    with pytest.raises(ManifestError):
        manifest_module._problem("P_OBS_CONTRACT_UPGRADED", "x")  # noqa: SLF001


#: Call names that would mean "manufacture the missing channel".  ``manifest``
#: may not contain any of them, under any receiver spelling.
_RESAMPLING_CALLS = frozenset({
    "repeat", "repeat_interleave", "tile", "interpolate", "resize", "zoom",
    "pad", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like",
    "empty", "new_empty", "kron", "broadcast_to", "upsample", "resample",
    "concatenate", "concat", "stack", "vstack", "hstack", "dstack",
})
#: Array/pixel libraries: importing one at module scope is how a "conversion"
#: starts, so the top level of ``manifest`` must name none of them.
_ARRAY_ROOTS = ("numpy", "np", "torch", "cv2", "PIL", "Image", "scipy", "skimage")


def _call_name(node: ast.expr) -> str | None:
    """``numpy.zeros`` / ``np.repeat`` -> dotted text; ``None`` if not nameable."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def test_manifest_module_has_no_resampling_or_channel_synthesis() -> None:
    """11g: the guard declares and refuses — it never upgrades legacy data."""

    source = MANIFEST_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # (1) no array/pixel library imported at module scope, and the numpy-pulling
    #     ``observation`` import must stay lazy (inside ``_o2_shapes``).
    top_level: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                top_level.append(f"relative:{node.module}")
            elif node.module:
                top_level.append(node.module.split(".")[0])
    assert [name for name in top_level if name in _ARRAY_ROOTS] == [], top_level
    assert "relative:observation" not in top_level, (
        "observation must stay lazily imported: manifest is pure stdlib")

    # (2) no call in the whole module resamples/pads/synthesises an array.
    offending: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name is None:
            continue
        if name.split(".")[0] in _ARRAY_ROOTS or name.rsplit(".", 1)[-1] in _RESAMPLING_CALLS:
            offending.append(f"{name} @line {node.lineno}")
    assert offending == [], offending

    # (3) no function is *named* like a converter/upsampler/promoter.
    suspicious = [
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(word in node.name.lower() for word in
                ("upsampl", "resampl", "interpolat", "concat", "convert",
                 "promote", "synthes", "pad_", "_pad"))]
    assert suspicious == [], suspicious

    # (4) the literal shortcuts a reviewer would grep for are absent as well.
    for literal in ("np.repeat", "np.zeros", "np.pad", "np.resize", "cv2.resize",
                    "F.interpolate", "torch.nn.functional.interpolate"):
        assert literal not in source, literal
