"""Judge A-01..A-06 of the v1.0.4 freeze, from the frozen artifacts, in code.

Why this is a new tool
----------------------
``check_a01.py`` decides the v1.0.3 gates (A-01/A-02/A-03) from
``vision-summary.json`` files whose A-01 rule is "some epoch passed" and whose
A-03 rule is "does more training data still help".  v1.0.4 changed both contracts
(A-01 is the *joint* pass of the selected ``state.best`` row on the full 7,241
validation frames; A-03 is a cross-replay ratio on two manifests), so re-labelling
the old tool would have silently kept the old semantics.  ``check_a01.py`` keeps
its historical meaning and is not touched by this task.

What this tool reads
--------------------
``--artifacts PATH`` is the frozen list (``schema=v104-final/1``): the 20 required
flat keys (``state_checkpoint``/``state_checkpoint_sha256``/``state_summary``/
``policy_checkpoint``/``policy_checkpoint_sha256``/``policy_summary``/``teacher_cache``/
``teacher_fingerprint``/``manifest_a``/``manifest_a_sha256``/``manifest_b``/
``manifest_b_sha256``/``val_index_sha256``/``arch``/``depth``/``seed``/
``experiment_table``/``selection_reason``/``frozen_at``/``external_guard_uses``) plus
``evidence``.  ``state_run``/``policy_run`` are optional and only cross-checked;
``architecture`` is **not** an alias for ``arch``.  Every path is
workspace-relative and every ``*_sha256`` is 64 lowercase hex.

``evidence`` has exactly nine keys - ``state_recheck``, ``policy_full_val``,
``cross_replay``, ``eval_report``, ``pytest``, ``d9``, ``d10``, ``d11``,
``startup`` - and each value is ``{path, sha256}`` pointing at a machine-readable
index (``schema=v104-evidence/1``) with ``kind``/``status``/``created_at``/
``producer``/``command``/``cwd``/``exit_code``/``artifacts``/``observations``/
``reason``.  Both the index file and every artifact inside it are re-hashed, and
the numbers are taken from the raw artifacts: a boolean ``status`` is never
accepted as the evidence itself.

Judgements (thresholds are the frozen v1.0.4 ones, ``需求文档`` section 7)

* A-01: the frozen ``state.best`` checkpoint re-checked on the *full* validation
  set - ``frames == 7241``, ``coverage_sha256 == val_index_sha256``,
  ``player_pos_px < 10.0`` and ``danger_corr_perframe_median > 0.8``.
* A-02: the frozen ``policy.pt`` on the same full validation set, ``0-15`` danger
  bucket with ``frames > 0``, *unclipped* teacher regret ``< 1.5 px`` and a
  reported ``top3``.  An empty bucket cannot pass.
* A-03: the same ``policy.pt`` bytes on both manifests, ``stage-sample`` with
  ``--per-stage 128``, ``B_mae_raw <= 1.25 * A_mae_raw``; the selected index hashes
  are re-derived from the two manifests instead of being read from the index.
* A-04: at least 3 real-game episodes, Lunatic/team 0/frameskip 2/max_steps 12000,
  ``frames_mean > 4000``, ``hit_step_cap == 0`` and complete per-episode
  ``terminated``/``truncated``/``termination_reason``; the ``startup`` evidence
  must say a real game started (a stub/offline eval-report must not pass).
* A-05: ``pytest`` evidence with ``passed >= 178``, ``failed == 0``,
  ``command_exit == 0``; the raw log is parsed again when it is a text artifact.
* A-06: D9/D10/D11 evidence with their contracted observations, and the D9/D10
  invariants (7,241 covered, resident <= budget, LRU/value equality) re-checked.

Exit codes: ``0`` = all six pass, ``1`` = at least one fail and no blocked,
``2`` = at least one blocked (fail and blocked may coexist; every item keeps its
own status in the JSON) or the frozen list itself is unusable.

Synthetic fixtures in the unit tests are *not* an acceptance run: the tool proves
the verdict logic, not that the real experiments reached these numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from touhou_rl import spec
from touhou_rl.pipeline.manifest import (
    DatasetManifest,
    ManifestError,
    file_digest,
    index_digest,
    split_indices,
)
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vision_history import v104_error
from touhou_rl.vmodels import KNOWN_VISION_ARCHES, is_sha256

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_FINAL = "v104-final/1"
SCHEMA_EVIDENCE = "v104-evidence/1"
SCHEMA_CHECK = "v104-check/1"
SUMMARY_SCHEMA = "vision-summary/2"
CONTRACT_VERSION = "v104-A/1"

#: The 20 required flat keys.  ``evidence`` is the 21st key of the frozen list and
#: is validated separately (its nine entries are the evidence map).
REQUIRED_FLAT: tuple[str, ...] = (
    "state_checkpoint",
    "state_checkpoint_sha256",
    "state_summary",
    "policy_checkpoint",
    "policy_checkpoint_sha256",
    "policy_summary",
    "teacher_cache",
    "teacher_fingerprint",
    "manifest_a",
    "manifest_a_sha256",
    "manifest_b",
    "manifest_b_sha256",
    "val_index_sha256",
    "arch",
    "depth",
    "seed",
    "experiment_table",
    "selection_reason",
    "frozen_at",
    "external_guard_uses",
)
OPTIONAL_FLAT: tuple[str, ...] = ("state_run", "policy_run", "evidence")
EVIDENCE_KEYS: tuple[str, ...] = (
    "state_recheck",
    "policy_full_val",
    "cross_replay",
    "eval_report",
    "pytest",
    "d9",
    "d10",
    "d11",
    "startup",
)
EVIDENCE_STATUSES: tuple[str, ...] = ("pass", "fail", "blocked")
EVIDENCE_INDEX_KEYS: tuple[str, ...] = (
    "schema",
    "kind",
    "status",
    "created_at",
    "producer",
    "command",
    "cwd",
    "exit_code",
    "artifacts",
    "observations",
    "reason",
)
#: ``<name>_sha256`` keys whose ``<name>`` path must hash to the declared value.
HASHED_FILES: tuple[str, ...] = (
    "state_checkpoint",
    "policy_checkpoint",
    "manifest_a",
    "manifest_b",
)

#: Contract thresholds: 需求文档 section 7 (口径不可改).
THRESHOLDS: dict = {
    "a01_pos_px_max": 10.0,
    "a01_perframe_median_min": 0.8,
    "a02_regret_px_max": 1.5,
    "a03_ratio_max": 1.25,
    "a03_per_stage": 128,
    "a04_min_episodes": 3,
    "a04_frames_mean_min": 4000.0,
    "a04_difficulty": spec.LUNATIC,
    "a04_team": 0,
    "a04_frameskip": 2,
    "a04_max_steps": 12000,
    "a05_min_passed": 178,
    "val_frames": 7241,
    "d10_budget_bytes": 536_870_912,
    "a16_bucket": "0-15",
}
V104_CODES = frozenset(
    {"CONFIG", "DATA", "CACHE", "TEACHER", "CHECKPOINT", "COVERAGE", "BUDGET", "EVIDENCE"}
)
_V104_PREFIX = re.compile(r"^v104:([A-Z]+): ")
_SHA_ONLY = re.compile(r"^[0-9a-f]{64}$")
_PYTEST_COUNTS = re.compile(r"(\d+) (passed|failed|skipped|error|errors)")


def v104_error_code(message: object) -> str | None:
    """``"EVIDENCE"`` for ``"v104:EVIDENCE: ..."``, else ``None``."""
    match = _V104_PREFIX.match(str(message))
    if match and match.group(1) in V104_CODES:
        return match.group(1)
    return None


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _as_float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _as_int(value):
    number = _as_float(value)
    if number is None or number != int(number):
        return None
    return int(number)


def _is_relative_path(value) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    return not Path(value).is_absolute() and not re.match(r"^[A-Za-z]:[\\/]", value)


def _iso8601_with_timezone(value) -> bool:
    if not isinstance(value, str) or not value:
        return False
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).tzinfo is not None
    except ValueError:
        return False


def _resolve(root: Path, relative: str) -> Path:
    return (Path(root) / Path(str(relative).replace("\\", "/"))).resolve()


# --------------------------------------------------------------------------- #
# the frozen list
# --------------------------------------------------------------------------- #
def require_flat_keys(document: dict) -> None:
    """Missing required keys are ``KeyError`` (the caller maps them to EVIDENCE)."""
    if "arch" not in document and "architecture" in document:
        raise KeyError("'architecture' is not an alias for 'arch'; record the key 'arch'")
    missing = [key for key in REQUIRED_FLAT if key not in document]
    if missing:
        raise KeyError(f"frozen artifacts are missing the required key(s) {missing}")


def load_artifacts(path: Path | str, root: Path | str = ROOT) -> tuple[dict, dict]:
    """Read and structurally validate ``final-artifacts.json``."""
    path = Path(path)
    root = Path(root)
    if not path.is_file():
        raise v104_error("EVIDENCE", f"frozen artifacts not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise v104_error("EVIDENCE", f"frozen artifacts are unreadable: {path} ({exc})") from exc
    if not isinstance(document, dict):
        raise v104_error("EVIDENCE", f"frozen artifacts are not a JSON object: {path}")
    if document.get("schema") != SCHEMA_FINAL:
        raise v104_error(
            "EVIDENCE", f"frozen artifacts schema {document.get('schema')!r} != {SCHEMA_FINAL!r}"
        )
    require_flat_keys(document)
    arch = document["arch"]
    if arch not in KNOWN_VISION_ARCHES:
        raise v104_error(
            "EVIDENCE", f"frozen arch {arch!r} not in {list(KNOWN_VISION_ARCHES)}"
        )
    if _as_int(document["depth"]) is None or int(document["depth"]) < 1:
        raise v104_error("EVIDENCE", f"frozen depth {document['depth']!r} is not a positive integer")
    if _as_int(document["seed"]) is None:
        raise v104_error("EVIDENCE", f"frozen seed {document['seed']!r} is not an integer")
    if _as_int(document["external_guard_uses"]) is None:
        raise v104_error(
            "EVIDENCE",
            f"external_guard_uses {document['external_guard_uses']!r} is not an integer",
        )
    if not _iso8601_with_timezone(document["frozen_at"]):
        raise v104_error(
            "EVIDENCE", f"frozen_at {document['frozen_at']!r} is not an ISO8601 timestamp with timezone"
        )
    for key in REQUIRED_FLAT:
        if key.endswith("_sha256"):
            if not is_sha256(document[key]):
                raise v104_error(
                    "EVIDENCE", f"{key} must be 64 lowercase hex chars, got {document[key]!r}"
                )
        elif key in ("experiment_table", "selection_reason", "teacher_cache") or key.endswith(
            ("_checkpoint", "_summary")
        ):
            if not _is_relative_path(document[key]):
                raise v104_error(
                    "EVIDENCE", f"{key} must be a workspace-relative path, got {document[key]!r}"
                )
    if not isinstance(document["teacher_fingerprint"], (str, dict)) or not document["teacher_fingerprint"]:
        raise v104_error("EVIDENCE", "teacher_fingerprint must be a non-empty string or mapping")
    if isinstance(document["teacher_fingerprint"], str) and not (
        is_sha256(document["teacher_fingerprint"]) or len(document["teacher_fingerprint"]) >= 8
    ):
        raise v104_error("EVIDENCE", "teacher_fingerprint is too short to identify a teacher")
    evidence = document.get("evidence")
    if not isinstance(evidence, dict):
        raise v104_error("EVIDENCE", "frozen artifacts have no 'evidence' mapping")
    unknown = sorted(set(evidence) - set(EVIDENCE_KEYS))
    if unknown:
        raise v104_error(
            "EVIDENCE", f"evidence has unknown key(s) {unknown}; the contract fixes {list(EVIDENCE_KEYS)}"
        )
    for key, entry in evidence.items():
        if not isinstance(entry, dict):
            raise v104_error("EVIDENCE", f"evidence[{key!r}] is not a {{path, sha256}} mapping")
        for field in ("path", "sha256"):
            if field not in entry:
                raise KeyError(f"evidence[{key!r}] is missing {field!r}")
        if not _is_relative_path(entry["path"]):
            raise v104_error(
                "EVIDENCE", f"evidence[{key!r}].path is not workspace-relative: {entry['path']!r}"
            )
        if not is_sha256(entry["sha256"]):
            raise v104_error(
                "EVIDENCE", f"evidence[{key!r}].sha256 is not 64 lowercase hex: {entry['sha256']!r}"
            )
    for key in OPTIONAL_FLAT:
        if key in ("state_run", "policy_run") and key in document:
            if not _is_relative_path(document[key]):
                raise v104_error(
                    "EVIDENCE", f"{key} must be a workspace-relative path, got {document[key]!r}"
                )
    meta = {
        "path": str(path),
        "sha256": file_digest(path),
        "root": str(root),
    }
    return document, meta


# --------------------------------------------------------------------------- #
# frozen-file identity
# --------------------------------------------------------------------------- #
def _summary_identity(document: dict, stage: str, expected_sha: str) -> dict:
    """Does ``stages.<stage>.best`` point at the frozen checkpoint bytes?

    C4 requires ``stages.<stage>.best.checkpoint``/``checkpoint_sha256`` and the
    ``contract_version=v104-A/1`` marker; a summary that does not name the frozen
    bytes means A-01/A-02 are being judged against a different model.
    """
    path = _resolve(Path(document["_root"]), document[f"{stage}_summary"])
    record = {"path": str(path), "ok": False, "errors": [], "checkpoint": None, "sha256": None}
    if not path.is_file():
        record["errors"].append(f"{stage}_summary not found: {path}")
        return record
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record["errors"].append(f"{stage}_summary is unreadable: {exc}")
        return record
    if not isinstance(summary, dict):
        record["errors"].append(f"{stage}_summary is not a JSON object")
        return record
    if summary.get("schema") != SUMMARY_SCHEMA:
        record["errors"].append(
            f"{stage}_summary schema {summary.get('schema')!r} != {SUMMARY_SCHEMA!r}"
        )
    if summary.get("contract_version") != CONTRACT_VERSION:
        record["errors"].append(
            f"{stage}_summary contract_version {summary.get('contract_version')!r} "
            f"!= {CONTRACT_VERSION!r}"
        )
    best = _dict(_dict(summary.get("stages")).get(stage)).get("best")
    best = _dict(best)
    record["checkpoint"] = best.get("checkpoint")
    record["sha256"] = best.get("checkpoint_sha256")
    if not best:
        record["errors"].append(f"{stage}_summary has no stages.{stage}.best")
        return record
    if not isinstance(record["checkpoint"], str) or not record["checkpoint"]:
        record["errors"].append(f"stages.{stage}.best.checkpoint is missing")
    if not is_sha256(record["sha256"]):
        record["errors"].append(
            f"stages.{stage}.best.checkpoint_sha256 is not a 64-hex sha: {record['sha256']!r}"
        )
    elif record["sha256"] != expected_sha:
        record["errors"].append(
            f"stages.{stage}.best.checkpoint_sha256 {record['sha256']} != frozen "
            f"{stage}_checkpoint_sha256 {expected_sha}"
        )
    record["ok"] = not record["errors"]
    return record


def _manifest_identity(path: Path, val_index_sha256: str | None, *, require_full_val: bool) -> dict:
    """Re-derive a manifest's split; bind A's to the frozen full-validation hash.

    This is the cheapest way to prove the frozen ``val_index_sha256`` really is a
    7,241-frame contiguous-in-index validation split of the frozen manifest, and
    that the A/B manifests were not swapped after the freeze.
    """
    record: dict = {"path": str(path), "ok": False, "errors": [], "val_frames": None, "val_index_sha256": None}
    if not path.is_file():
        record["errors"].append(f"manifest not found: {path}")
        return record
    try:
        manifest = DatasetManifest.load(path)
    except (ManifestError, KeyError, ValueError) as exc:
        record["errors"].append(f"manifest is unusable: {exc}")
        return record
    splits = _dict(manifest.splits)
    try:
        _train, val = split_indices(
            int(manifest.total_frames), int(splits["seed"]), float(splits["val_fraction"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        record["errors"].append(f"manifest split is unusable: {exc}")
        return record
    digest = index_digest(val)
    record["val_frames"] = len(val)
    record["val_index_sha256"] = digest
    if digest != splits.get("val_index_sha256"):
        record["errors"].append(
            f"split reproducibility failed: recomputed {digest} != manifest {splits.get('val_index_sha256')!r}"
        )
    if val_index_sha256 is not None and digest != val_index_sha256:
        record["errors"].append(
            f"split digest {digest} != frozen val_index_sha256 {val_index_sha256}"
        )
    if require_full_val and len(val) != THRESHOLDS["val_frames"]:
        record["errors"].append(
            f"manifest_a validation split has {len(val)} frames, the frozen contract is {THRESHOLDS['val_frames']}"
        )
    record["ok"] = not record["errors"]
    return record


def _checkpoint_identity(path: Path) -> dict:
    """``{arch, depth}`` read from the checkpoint payload (C6 arch dispatch).

    The frozen list names one architecture and one depth; the payload is what a
    deployment would actually load, so a mismatch means the freeze points at a
    different model than the evidence was produced for.  Only the header keys are
    read - this does not instantiate a model.
    """
    import torch

    record: dict = {"arch": None, "depth": None, "errors": []}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # an unreadable file is an identity failure, not a crash
        record["errors"].append(f"cannot read checkpoint payload: {exc}")
        return record
    if not isinstance(payload, dict) or payload.get("kind") != "vision_scorer":
        record["errors"].append("checkpoint payload is not a kind=vision_scorer mapping")
        return record
    record["arch"] = payload.get("arch") or "v1"
    config = payload.get("config")
    if isinstance(config, dict):
        record["depth"] = _as_int(config.get("depth"))
    return record


def verify_frozen(document: dict, root: Path) -> dict:
    """Existence + hash of every declared file, and the summary/checkpoint identity."""
    root = Path(root)
    record: dict = {"root": str(root), "files": {}, "summaries": {}, "manifests": {}, "ok": True, "errors": []}
    for name in HASHED_FILES:
        path = _resolve(root, document[name])
        declared = document[f"{name}_sha256"]
        item = {"path": str(path), "declared": declared, "actual": None, "ok": False}
        if not path.is_file():
            item["error"] = "missing"
            record["errors"].append(f"{name} not found: {path}")
        else:
            item["actual"] = file_digest(path)
            item["ok"] = item["actual"] == declared
            if not item["ok"]:
                record["errors"].append(
                    f"{name} sha256 {item['actual']} != frozen {declared}"
                )
        record["files"][name] = item
    for name in ("state_checkpoint", "policy_checkpoint"):
        item = record["files"][name]
        if not item.get("ok"):
            continue
        identity = _checkpoint_identity(Path(item["path"]))
        item["header"] = identity
        for error in identity["errors"]:
            record["errors"].append(f"{name}: {error}")
        if identity["arch"] is not None and identity["arch"] != document["arch"]:
            record["errors"].append(
                f"{name} arch {identity['arch']!r} != frozen arch {document['arch']!r}"
            )
        if identity["depth"] is not None and identity["depth"] != int(document["depth"]):
            record["errors"].append(
                f"{name} config depth {identity['depth']!r} != frozen depth {document['depth']!r}"
            )
    for name in ("teacher_cache", "experiment_table"):
        path = _resolve(root, document[name])
        exists = path.exists()
        record["files"][name] = {"path": str(path), "ok": exists, "kind": "dir" if path.is_dir() else "file"}
        if not exists:
            record["errors"].append(f"{name} not found: {path}")
    record["val_index_sha256"] = document["val_index_sha256"]
    record["manifests"]["a"] = _manifest_identity(
        _resolve(root, document["manifest_a"]),
        document["val_index_sha256"],
        require_full_val=True,
    )
    record["manifests"]["b"] = _manifest_identity(
        _resolve(root, document["manifest_b"]), None, require_full_val=False
    )
    for side, manifest_record in record["manifests"].items():
        if not manifest_record["ok"]:
            record["errors"].extend(f"manifest_{side}: {error}" for error in manifest_record["errors"])
    document = dict(document)
    document["_root"] = str(root)
    record["summaries"]["state"] = _summary_identity(document, "state", document["state_checkpoint_sha256"])
    record["summaries"]["policy"] = _summary_identity(
        document, "policy", document["policy_checkpoint_sha256"]
    )
    for stage, summary_record in record["summaries"].items():
        if not summary_record["ok"]:
            record["errors"].extend(f"summary[{stage}]: {error}" for error in summary_record["errors"])
    record["ok"] = not record["errors"]
    return record


# --------------------------------------------------------------------------- #
# evidence indices
# --------------------------------------------------------------------------- #
def _artifact_observations(document: dict) -> dict:
    """Observation values carried by a raw artifact.

    Producers either nest them under ``observations`` or write them flat; both are
    accepted so the checker never has to trust the index's copy.  Nested
    structures (``by_danger`` etc.) are kept as-is.
    """
    nested = document.get("observations")
    if isinstance(nested, dict):
        merged = dict(nested)
        for key, value in document.items():
            merged.setdefault(key, value)
        return merged
    return {
        key: value
        for key, value in document.items()
        if not isinstance(value, (dict, list)) or key == "by_danger"
    }


def _compare_observation(key: str, index_value, artifact_value) -> str | None:
    """``None`` when they agree, else a human-readable mismatch."""
    if isinstance(index_value, bool) or isinstance(artifact_value, bool):
        return None if bool(index_value) == bool(artifact_value) else f"{key}: {index_value!r} != {artifact_value!r}"
    number = _as_float(index_value)
    other = _as_float(artifact_value)
    if number is None and other is None:
        # NaN/inf on both sides: the item's own finite check reports it, the
        # equality check cannot (``nan != nan``).
        if isinstance(index_value, (int, float)) and isinstance(artifact_value, (int, float)):
            return None
    if number is not None and other is not None:
        if abs(number - other) <= 1e-6 * max(1.0, abs(number)):
            return None
        return f"{key}: index {number} != artifact {other}"
    if index_value == artifact_value:
        return None
    return f"{key}: index {index_value!r} != artifact {artifact_value!r}"


def load_evidence_index(entry, key: str, root: Path, required_observations) -> dict:
    """Load one evidence index and re-verify its sha and every artifact's sha."""
    record: dict = {
        "kind": key,
        "ok": False,
        "errors": [],
        "path": None,
        "sha256": None,
        "index": {},
        "observations": {},
        "artifacts": [],
        "artifact_docs": [],
        "artifact_texts": [],
        "status": None,
        "reason": "",
    }
    if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
        record["errors"].append(f"evidence[{key!r}] is not a {{path, sha256}} mapping")
        return record
    path = _resolve(root, entry["path"])
    record["path"] = str(path)
    record["sha256"] = entry["sha256"]
    if not path.is_file():
        record["errors"].append(f"evidence[{key!r}] index not found: {path}")
        return record
    actual = file_digest(path)
    if actual != entry["sha256"]:
        record["errors"].append(
            f"evidence[{key!r}] index sha256 {actual} != frozen {entry['sha256']}"
        )
        return record
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record["errors"].append(f"evidence[{key!r}] index is unreadable: {exc}")
        return record
    if not isinstance(index, dict):
        record["errors"].append(f"evidence[{key!r}] index is not a JSON object")
        return record
    record["index"] = index
    for field in EVIDENCE_INDEX_KEYS:
        if field not in index:
            record["errors"].append(f"evidence[{key!r}] index is missing {field!r}")
    if index.get("schema") != SCHEMA_EVIDENCE:
        record["errors"].append(
            f"evidence[{key!r}] schema {index.get('schema')!r} != {SCHEMA_EVIDENCE!r}"
        )
    if index.get("kind") != key:
        record["errors"].append(f"evidence[{key!r}] kind {index.get('kind')!r} != {key!r}")
    if index.get("status") not in EVIDENCE_STATUSES:
        record["errors"].append(
            f"evidence[{key!r}] status {index.get('status')!r} not in {list(EVIDENCE_STATUSES)}"
        )
    record["status"] = index.get("status")
    record["reason"] = str(index.get("reason", ""))
    if not _as_int(index.get("exit_code")) and index.get("exit_code") != 0:
        record["errors"].append(f"evidence[{key!r}] exit_code {index.get('exit_code')!r} is not an integer")
    observations = index.get("observations")
    if not isinstance(observations, dict):
        record["errors"].append(f"evidence[{key!r}] observations is not a mapping")
        observations = {}
    record["observations"] = observations
    missing = [name for name in required_observations if name not in observations]
    if missing:
        record["errors"].append(f"evidence[{key!r}] observations are missing {missing}")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        record["errors"].append(f"evidence[{key!r}] has no artifacts list")
        artifacts = []
    for position, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or "path" not in artifact or "sha256" not in artifact:
            record["errors"].append(f"evidence[{key!r}] artifacts[{position}] is not {{path, sha256}}")
            continue
        artifact_path = _resolve(root, artifact["path"])
        node = {
            "path": str(artifact_path),
            "declared": artifact["sha256"],
            "actual": None,
            "ok": False,
        }
        if not artifact_path.is_file():
            node["error"] = "missing"
            record["errors"].append(f"evidence[{key!r}] artifact not found: {artifact_path}")
        else:
            node["actual"] = file_digest(artifact_path)
            node["ok"] = node["actual"] == artifact["sha256"]
            if not node["ok"]:
                record["errors"].append(
                    f"evidence[{key!r}] artifact sha256 {node['actual']} != declared {artifact['sha256']}"
                )
            text = artifact_path.read_text(encoding="utf-8", errors="replace")
            record["artifact_texts"].append(text)
            try:
                document = json.loads(text)
            except json.JSONDecodeError:
                document = None
            if isinstance(document, dict):
                record["artifact_docs"].append(document)
        record["artifacts"].append(node)
    # Cross-check the index's observations against what the raw artifacts say.
    if record["artifact_docs"]:
        merged: dict = {}
        for document in record["artifact_docs"]:
            merged.update(_artifact_observations(document))
        for name in required_observations:
            if name in observations and name in merged:
                mismatch = _compare_observation(name, observations[name], merged[name])
                if mismatch is not None:
                    record["errors"].append(f"evidence[{key!r}] index disagrees with artifact: {mismatch}")
    record["ok"] = not record["errors"]
    return record


REQUIRED_OBSERVATIONS: dict[str, tuple[str, ...]] = {
    "state_recheck": (
        "checkpoint_sha256",
        "frames",
        "coverage_sha256",
        "player_pos_px",
        "danger_corr_perframe_median",
        "danger_corr_perframe_valid",
    ),
    "policy_full_val": (
        "checkpoint_sha256",
        "frames",
        "coverage_sha256",
        "player_pos_px",
        "danger_corr_perframe_median",
        "danger_corr_perframe_valid",
        "by_danger",
    ),
    "cross_replay": (
        "checkpoint_sha256",
        "a_mae_raw",
        "b_mae_raw",
        "a_indices_sha256",
        "b_indices_sha256",
        "a_frames",
        "b_frames",
        "sampling_mode",
        "per_stage",
    ),
    "eval_report": (
        "checkpoint_sha256",
        "episodes",
        "frames_mean",
        "hit_step_cap",
        "episode_fields_complete",
    ),
    "pytest": ("passed", "failed", "skipped", "command_exit"),
    "d9": ("expected_frames", "actual_frames", "visited_index_sha256", "last_batch_size"),
    "d10": ("budget_bytes", "max_resident_bytes", "lru_verified", "value_equal"),
    "d11": ("expected_files", "validated_files", "bytes_total", "value_equal", "manifest_sha"),
    "startup": (
        "real_game_started",
        "pixels_only",
        "no_crash",
        "no_exception_log",
        "ui_checked",
        "screenshots",
    ),
}


# --------------------------------------------------------------------------- #
# A-03: re-derive the stage sample from the manifests
# --------------------------------------------------------------------------- #
def recompute_stage_sample(manifest_path: Path, per_stage: int, root: Path) -> dict:
    """Selected indices of one manifest, recomputed from the written rule.

    ``stage-sample`` = per stage, the first ``per_stage`` validation indices in
    ascending order.  The checker re-implements the rule (instead of importing the
    comparison tool) so that a bug in the tool cannot validate itself.
    """
    record: dict = {"path": str(manifest_path), "ok": False, "errors": [], "sha256": None, "frames": None}
    if not manifest_path.is_file():
        record["errors"].append(f"manifest not found: {manifest_path}")
        return record
    try:
        manifest = DatasetManifest.load(manifest_path)
    except (ManifestError, KeyError, ValueError) as exc:
        record["errors"].append(f"manifest is unusable: {exc}")
        return record
    splits = _dict(manifest.splits)
    try:
        _train, val = split_indices(
            int(manifest.total_frames), int(splits["seed"]), float(splits["val_fraction"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        record["errors"].append(f"manifest split is unusable: {exc}")
        return record
    digest = index_digest(val)
    if digest != splits.get("val_index_sha256"):
        record["errors"].append(
            f"split reproducibility check failed: recomputed {digest} != manifest "
            f"{splits.get('val_index_sha256')}"
        )
        return record
    dataset_dir = Path(str(manifest.root).replace("\\", "/"))
    if not dataset_dir.is_absolute():
        dataset_dir = _resolve(root, str(manifest.root))
    if not dataset_dir.is_dir():
        record["errors"].append(f"dataset directory not found: {dataset_dir}")
        return record
    with VisionDataset(dataset_dir) as dataset:
        if len(dataset) != int(manifest.total_frames):
            record["errors"].append(
                f"dataset has {len(dataset)} frames, manifest says {manifest.total_frames}"
            )
            return record
        stages = dataset.stage_of(val).tolist()
    ordered = sorted(zip(val, stages))
    selected: list[int] = []
    per_stage_counts: dict[str, dict] = {}
    for stage in sorted({int(value) for _index, value in ordered}):
        rows = [int(index) for index, value in ordered if int(value) == stage]
        take = rows[: int(per_stage)]
        per_stage_counts[str(stage)] = {"available": len(rows), "selected": len(take)}
        selected.extend(take)
    record.update(
        {
            "ok": True,
            "sha256": index_digest(selected),
            "frames": len(selected),
            "val_index_sha256": digest,
            "per_stage": per_stage_counts,
            "dataset": str(dataset_dir),
        }
    )
    return record


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #
def _item(status: str, *, source: str, record=None, values=None, thresholds=None, reason="") -> dict:
    node = {
        "status": status,
        "source": source,
        "path": None,
        "hash": None,
        "artifacts": [],
        "values": values or {},
        "thresholds": thresholds or {},
        "reason": reason,
    }
    if record is not None:
        node["path"] = record.get("path")
        node["hash"] = record.get("sha256")
        node["artifacts"] = [artifact.get("path") for artifact in record.get("artifacts", [])]
    return node


def _blocked_on_record(record, source: str) -> dict | None:
    """A ``blocked`` item when the index/artifact identity itself is unusable."""
    if record.get("ok"):
        return None
    return _item(
        "blocked",
        source=source,
        record=record,
        reason="; ".join(record.get("errors", [])) or "evidence is unusable",
    )


def _status_veto(record, source: str, values: dict, thresholds: dict) -> dict | None:
    """Honor ``status=blocked``; catch ``status=fail`` that contradicts the values."""
    if record.get("status") == "blocked":
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"the evidence index itself is blocked: {record.get('reason') or 'no reason given'}",
        )
    return None


def judge_a01(frozen: dict, evidence: dict) -> dict:
    source = "state_recheck"
    record = evidence[source]
    veto = _blocked_on_record(record, source) or _status_veto(record, source, {}, {})
    if veto is not None:
        return veto
    if not frozen["ok"]:
        return _item(
            "blocked", source=source, record=record, reason="frozen identity is unusable: " + "; ".join(frozen["errors"][:3])
        )
    observations = record["observations"]
    values = {
        key: observations.get(key)
        for key in (
            "checkpoint_sha256",
            "frames",
            "coverage_sha256",
            "player_pos_px",
            "danger_corr_perframe_median",
            "danger_corr_perframe_valid",
        )
    }
    thresholds = {
        "player_pos_px_max": THRESHOLDS["a01_pos_px_max"],
        "danger_corr_perframe_median_min": THRESHOLDS["a01_perframe_median_min"],
        "frames": THRESHOLDS["val_frames"],
        "coverage_sha256": frozen["val_index_sha256"],
    }
    if observations.get("checkpoint_sha256") != frozen["files"]["state_checkpoint"]["declared"]:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                f"state_recheck checkpoint_sha256 {observations.get('checkpoint_sha256')!r} != frozen "
                f"state_checkpoint_sha256 {frozen['files']['state_checkpoint']['declared']}"
            ),
        )
    frames = _as_int(observations.get("frames"))
    if frames != THRESHOLDS["val_frames"]:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"state_recheck covers {observations.get('frames')!r} frames, the full validation set is {THRESHOLDS['val_frames']}",
        )
    if observations.get("coverage_sha256") != frozen["val_index_sha256"]:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                f"coverage_sha256 {observations.get('coverage_sha256')!r} != frozen val_index_sha256 "
                f"{frozen['val_index_sha256']}"
            ),
        )
    position = _as_float(observations.get("player_pos_px"))
    perframe = _as_float(observations.get("danger_corr_perframe_median"))
    valid = _as_int(observations.get("danger_corr_perframe_valid"))
    if position is None or perframe is None or valid is None:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason="player_pos_px / danger_corr_perframe_median / danger_corr_perframe_valid is not finite",
        )
    if valid <= 0:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"danger_corr_perframe_valid is {valid}; a per-frame correlation median needs valid frames",
        )
    passed = position < THRESHOLDS["a01_pos_px_max"] and perframe > THRESHOLDS["a01_perframe_median_min"]
    return _item(
        "pass" if passed else "fail",
        source=source,
        record=record,
        values=values,
        thresholds=thresholds,
        reason=(
            ""
            if passed
            else f"player_pos_px {position} not < {THRESHOLDS['a01_pos_px_max']} or "
            f"danger_corr_perframe_median {perframe} not > {THRESHOLDS['a01_perframe_median_min']}"
        ),
    )


def _bucket(observations: dict, name: str) -> dict | None:
    for row in observations.get("by_danger") or []:
        if not isinstance(row, dict):
            continue
        label = str(row.get("bucket", "")).strip().replace("\u2013", "-").replace("\u2014", "-")
        if label == name:
            return row
    return None


def judge_a02(frozen: dict, evidence: dict) -> dict:
    source = "policy_full_val"
    record = evidence[source]
    veto = _blocked_on_record(record, source) or _status_veto(record, source, {}, {})
    if veto is not None:
        return veto
    if not frozen["ok"]:
        return _item(
            "blocked", source=source, record=record, reason="frozen identity is unusable: " + "; ".join(frozen["errors"][:3])
        )
    observations = record["observations"]
    bucket = _bucket(observations, THRESHOLDS["a16_bucket"])
    values = {
        "checkpoint_sha256": observations.get("checkpoint_sha256"),
        "bucket": bucket,
        "frames": None if bucket is None else bucket.get("frames"),
        "top3": None if bucket is None else bucket.get("top3"),
        "regret": None if bucket is None else bucket.get("regret", bucket.get("regret_px")),
    }
    thresholds = {
        "bucket": THRESHOLDS["a16_bucket"],
        "regret_px_max": THRESHOLDS["a02_regret_px_max"],
        "frames_min": 1,
        "top3_reported": True,
    }
    if observations.get("checkpoint_sha256") != frozen["files"]["policy_checkpoint"]["declared"]:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                f"policy_full_val checkpoint_sha256 {observations.get('checkpoint_sha256')!r} != frozen "
                f"policy_checkpoint_sha256 {frozen['files']['policy_checkpoint']['declared']}"
            ),
        )
    if bucket is None:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"policy_full_val.by_danger has no {THRESHOLDS['a16_bucket']!r} bucket",
        )
    frames = _as_int(bucket.get("frames"))
    regret = _as_float(bucket.get("regret", bucket.get("regret_px")))
    top3 = _as_float(bucket.get("top3"))
    if frames is None or regret is None:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason="the 0-15 bucket lacks a finite frames/regret value",
        )
    if top3 is None:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason="the 0-15 bucket does not report top3",
        )
    if frames <= 0:
        return _item(
            "fail",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason="the 0-15 bucket has frames=0; an empty bucket cannot pass and its zeros are not a measurement",
        )
    passed = regret < THRESHOLDS["a02_regret_px_max"]
    return _item(
        "pass" if passed else "fail",
        source=source,
        record=record,
        values=values,
        thresholds=thresholds,
        reason="" if passed else f"0-15 bucket regret {regret} not < {THRESHOLDS['a02_regret_px_max']}",
    )


def judge_a03(frozen: dict, evidence: dict, root: Path) -> dict:
    source = "cross_replay"
    record = evidence[source]
    veto = _blocked_on_record(record, source) or _status_veto(record, source, {}, {})
    if veto is not None:
        return veto
    if not frozen["ok"]:
        return _item(
            "blocked", source=source, record=record, reason="frozen identity is unusable: " + "; ".join(frozen["errors"][:3])
        )
    observations = record["observations"]
    thresholds = {
        "ratio_max": THRESHOLDS["a03_ratio_max"],
        "sampling_mode": "stage-sample",
        "per_stage": THRESHOLDS["a03_per_stage"],
    }
    values = {key: observations.get(key) for key in REQUIRED_OBSERVATIONS[source]}
    if observations.get("checkpoint_sha256") != frozen["files"]["policy_checkpoint"]["declared"]:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                f"cross_replay checkpoint_sha256 {observations.get('checkpoint_sha256')!r} != frozen "
                f"policy_checkpoint_sha256 {frozen['files']['policy_checkpoint']['declared']}"
            ),
        )
    if observations.get("sampling_mode") != "stage-sample":
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"A-03 requires sampling_mode=stage-sample, got {observations.get('sampling_mode')!r}",
        )
    per_stage = _as_int(observations.get("per_stage"))
    if per_stage != THRESHOLDS["a03_per_stage"]:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"A-03 fixes per_stage={THRESHOLDS['a03_per_stage']}, got {observations.get('per_stage')!r}",
        )
    a_mae = _as_float(observations.get("a_mae_raw"))
    b_mae = _as_float(observations.get("b_mae_raw"))
    if a_mae is None or b_mae is None or a_mae <= 0:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason="a_mae_raw/b_mae_raw is missing, non-finite, or a_mae_raw is not positive",
        )
    recomputed = {}
    for side, manifest_key, sha_key, frames_key in (
        ("a", "manifest_a", "a_indices_sha256", "a_frames"),
        ("b", "manifest_b", "b_indices_sha256", "b_frames"),
    ):
        node = recompute_stage_sample(
            _resolve(root, frozen["document"][manifest_key]), THRESHOLDS["a03_per_stage"], root
        )
        recomputed[side] = node
        if not node["ok"]:
            return _item(
                "blocked",
                source=source,
                record=record,
                values=values,
                thresholds=thresholds,
                reason=f"{manifest_key} cannot be re-derived: {'; '.join(node['errors'])}",
            )
        if node["sha256"] != observations.get(sha_key):
            return _item(
                "blocked",
                source=source,
                record=record,
                values=values,
                thresholds=thresholds,
                reason=(
                    f"{sha_key} {observations.get(sha_key)!r} != recomputed {node['sha256']} "
                    f"from {manifest_key}"
                ),
            )
        if node["frames"] != _as_int(observations.get(frames_key)):
            return _item(
                "blocked",
                source=source,
                record=record,
                values=values,
                thresholds=thresholds,
                reason=(
                    f"{frames_key} {observations.get(frames_key)!r} != recomputed {node['frames']} "
                    f"from {manifest_key}"
                ),
            )
    values["a_indices_sha256_recomputed"] = recomputed["a"]["sha256"]
    values["b_indices_sha256_recomputed"] = recomputed["b"]["sha256"]
    ratio = float(b_mae) / float(a_mae)
    values["ratio"] = ratio
    passed = ratio <= THRESHOLDS["a03_ratio_max"]
    return _item(
        "pass" if passed else "fail",
        source=source,
        record=record,
        values=values,
        thresholds=thresholds,
        reason="" if passed else f"cross-replay ratio {ratio:.6f} > {THRESHOLDS['a03_ratio_max']}",
    )


def _eval_report_document(record: dict) -> dict | None:
    for document in record["artifact_docs"]:
        if isinstance(document.get("episodes"), list):
            return document
    return None


def _startup_gate(startup: dict, thresholds: dict) -> dict | None:
    veto = _blocked_on_record(startup, "startup")
    if veto is not None:
        return _item(
            "blocked",
            source="eval_report",
            record=startup,
            thresholds=thresholds,
            reason="startup evidence is unusable: " + "; ".join(startup.get("errors", [])),
        )
    observations = startup["observations"]
    required_true = {
        "real_game_started": True,
        "pixels_only": True,
        "no_crash": True,
        "no_exception_log": True,
    }
    for key, expected in required_true.items():
        if observations.get(key) is not expected:
            return _item(
                "fail",
                source="eval_report",
                record=startup,
                values={key: observations.get(key)},
                thresholds={key: expected},
                reason=f"startup.{key} is {observations.get(key)!r}, expected {expected!r} (a stub run is not A-04)",
            )
    if observations.get("ui_checked") is not True:
        return _item(
            "fail",
            source="eval_report",
            record=startup,
            values={"ui_checked": observations.get("ui_checked")},
            thresholds={"ui_checked": True},
            reason="startup.ui_checked is not true; the UI check result is part of the startup evidence",
        )
    screenshots = observations.get("screenshots")
    if not isinstance(screenshots, list) or not screenshots:
        return _item(
            "fail",
            source="eval_report",
            record=startup,
            values={"screenshots": screenshots},
            thresholds={"screenshots": "non-empty list"},
            reason="startup.screenshots is empty; the run saved no screenshot evidence",
        )
    return None


def judge_a04(frozen: dict, evidence: dict, root: Path) -> dict:
    source = "eval_report"
    record = evidence[source]
    thresholds = {
        "min_episodes": THRESHOLDS["a04_min_episodes"],
        "frames_mean_min": THRESHOLDS["a04_frames_mean_min"],
        "hit_step_cap": 0,
        "difficulty": THRESHOLDS["a04_difficulty"],
        "team": THRESHOLDS["a04_team"],
        "frameskip": THRESHOLDS["a04_frameskip"],
        "max_steps": THRESHOLDS["a04_max_steps"],
    }
    veto = _blocked_on_record(record, source) or _status_veto(record, source, {}, thresholds)
    if veto is not None:
        return veto
    if not frozen["ok"]:
        return _item(
            "blocked", source=source, record=record, reason="frozen identity is unusable: " + "; ".join(frozen["errors"][:3])
        )
    startup = evidence["startup"]
    gate = _startup_gate(startup, thresholds)
    if gate is not None:
        return gate
    observations = record["observations"]
    values = {key: observations.get(key) for key in REQUIRED_OBSERVATIONS[source]}
    observations = record["observations"]
    if observations.get("checkpoint_sha256") != frozen["files"]["policy_checkpoint"]["declared"]:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                f"eval_report checkpoint_sha256 {observations.get('checkpoint_sha256')!r} != frozen "
                f"policy_checkpoint_sha256 {frozen['files']['policy_checkpoint']['declared']}"
            ),
        )
    report = _eval_report_document(record)
    if report is None:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason="no artifact of the eval_report evidence is an eval-report.json (needs an 'episodes' list)",
        )
    episodes = report["episodes"]
    count = len([row for row in episodes if isinstance(row, dict)])
    frames = [
        _as_int(row.get("frames_reported")) for row in episodes if isinstance(row, dict)
    ]
    if any(value is None for value in frames) or not frames:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason="an episode has no integer frames_reported; frames_mean cannot be recomputed",
        )
    frames_mean = round(sum(frames) / len(frames), 1)
    hit_step_cap = sum(
        1
        for row in episodes
        if isinstance(row, dict) and row.get("termination_reason") == "max_steps"
    )
    fields_complete = all(
        isinstance(row, dict)
        and "terminated" in row
        and "truncated" in row
        and "termination_reason" in row
        for row in episodes
    )
    values.update(
        {
            "episodes": count,
            "frames": frames,
            "frames_mean": frames_mean,
            "frames_mean_reported": observations.get("frames_mean"),
            "hit_step_cap": hit_step_cap,
            "hit_step_cap_reported": observations.get("hit_step_cap"),
            "episode_fields_complete": fields_complete,
            "difficulty": report.get("difficulty"),
            "team": report.get("team"),
            "frameskip": report.get("frameskip"),
            "practice_stage": report.get("practice_stage"),
        }
    )
    if observations.get("frames_mean") is not None:
        mismatch = _compare_observation("frames_mean", observations["frames_mean"], frames_mean)
        if mismatch is not None:
            return _item(
                "blocked",
                source=source,
                record=record,
                values=values,
                thresholds=thresholds,
                reason=f"index disagrees with the eval-report: {mismatch}",
            )
    if observations.get("hit_step_cap") is not None and _as_int(observations["hit_step_cap"]) != hit_step_cap:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                f"hit_step_cap {observations.get('hit_step_cap')!r} != recomputed {hit_step_cap}"
            ),
        )
    if _as_int(observations.get("episodes")) != count:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                f"observed episodes {observations.get('episodes')!r} != {count} episodes in the "
                "eval-report"
            ),
        )
    vision_node = _dict(report.get("vision"))
    max_steps = _as_int(vision_node.get("max_steps"))
    if max_steps is None:
        max_steps = _as_int(observations.get("max_steps"))
    if max_steps is None:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                "neither the eval-report (vision.max_steps) nor the evidence records max_steps; "
                f"A-04 is defined at {THRESHOLDS['a04_max_steps']} and cannot be asserted without it"
            ),
        )
    if max_steps != THRESHOLDS["a04_max_steps"]:
        return _item(
            "fail",
            source=source,
            record=record,
            values=values | {"max_steps": max_steps},
            thresholds=thresholds,
            reason=f"evaluation used max_steps={max_steps}, A-04 is defined at {THRESHOLDS['a04_max_steps']}",
        )
    config_checks = {
        "difficulty": (report.get("difficulty"), THRESHOLDS["a04_difficulty"]),
        "team": (report.get("team"), THRESHOLDS["a04_team"]),
        "frameskip": (report.get("frameskip"), THRESHOLDS["a04_frameskip"]),
    }
    for key, (actual, expected) in config_checks.items():
        if _as_int(actual) != expected:
            return _item(
                "fail",
                source=source,
                record=record,
                values=values | {key: actual},
                thresholds=thresholds,
                reason=f"eval-report {key} is {actual!r}, A-04 is defined at {expected!r}",
            )
    if report.get("practice_stage") not in (None, -1):
        return _item(
            "fail",
            source=source,
            record=record,
            values=values | {"practice_stage": report.get("practice_stage")},
            thresholds=thresholds,
            reason=f"eval-report practice_stage is {report.get('practice_stage')!r}; A-04 is a full run",
        )
    checkpoint_text = vision_node.get("checkpoint") or report.get("checkpoint")
    checkpoint_ok = False
    if isinstance(checkpoint_text, str) and checkpoint_text and checkpoint_text != ".":
        candidate = Path(checkpoint_text)
        if not candidate.is_absolute():
            candidate = _resolve(root, checkpoint_text)
        if candidate.is_file() and file_digest(candidate) == frozen["files"]["policy_checkpoint"]["declared"]:
            checkpoint_ok = True
        values["deployed_checkpoint"] = str(candidate)
    if not checkpoint_ok:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                "the eval-report does not resolve to the frozen policy.pt bytes "
                f"(vision.checkpoint={checkpoint_text!r}); a deployment of state-pass.pt cannot stand in for A-04"
            ),
        )
    if count < THRESHOLDS["a04_min_episodes"]:
        return _item(
            "fail",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"{count} episode(s) < {THRESHOLDS['a04_min_episodes']}",
        )
    failed = []
    if not frames_mean > THRESHOLDS["a04_frames_mean_min"]:
        failed.append(f"frames_mean {frames_mean} not > {THRESHOLDS['a04_frames_mean_min']}")
    if hit_step_cap != 0:
        failed.append(f"hit_step_cap {hit_step_cap} != 0")
    if not fields_complete:
        failed.append("per-episode terminated/truncated/termination_reason is incomplete")
    return _item(
        "pass" if not failed else "fail",
        source=source,
        record=record,
        values=values,
        thresholds=thresholds,
        reason="; ".join(failed),
    )


def _pytest_counts(text: str) -> dict:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for line in text.splitlines():
        if "passed" not in line and "failed" not in line and "error" not in line:
            continue
        if " in " not in line:
            continue
        for number, kind in _PYTEST_COUNTS.findall(line):
            key = "error" if kind.startswith("error") else kind
            counts[key] = max(counts.get(key, 0), int(number))
    return counts


def judge_a05(frozen: dict, evidence: dict) -> dict:
    source = "pytest"
    record = evidence[source]
    thresholds = {
        "min_passed": THRESHOLDS["a05_min_passed"],
        "failed": 0,
        "command_exit": 0,
    }
    veto = _blocked_on_record(record, source) or _status_veto(record, source, {}, thresholds)
    if veto is not None:
        return veto
    observations = record["observations"]
    values = {key: observations.get(key) for key in REQUIRED_OBSERVATIONS[source]}
    passed = _as_int(observations.get("passed"))
    failed = _as_int(observations.get("failed"))
    exit_code = _as_int(observations.get("command_exit"))
    if passed is None or failed is None or exit_code is None:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason="passed/failed/command_exit is not an integer",
        )
    parsed = None
    for text in record["artifact_texts"]:
        counts = _pytest_counts(text)
        if counts["passed"] or counts["failed"] or counts["error"]:
            parsed = counts
            break
    if parsed is None:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason="no pytest summary line could be parsed from the raw log artifacts",
        )
    values["parsed"] = parsed
    if parsed["passed"] != passed or parsed["failed"] != failed:
        return _item(
            "blocked",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=(
                f"index says passed={passed} failed={failed}, the raw log says "
                f"passed={parsed['passed']} failed={parsed['failed']}"
            ),
        )
    if parsed["error"] or failed:
        return _item(
            "fail",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"{failed} test(s) failed / {parsed['error']} error(s); A-05 requires 0 failed",
        )
    if passed < THRESHOLDS["a05_min_passed"]:
        return _item(
            "fail",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"{passed} passed < {THRESHOLDS['a05_min_passed']}",
        )
    if exit_code != 0:
        return _item(
            "fail",
            source=source,
            record=record,
            values=values,
            thresholds=thresholds,
            reason=f"pytest exited {exit_code}",
        )
    return _item("pass", source=source, record=record, values=values, thresholds=thresholds)


def judge_a06(frozen: dict, evidence: dict) -> dict:
    thresholds = {
        "d9_frames": THRESHOLDS["val_frames"],
        "d10_budget_bytes": THRESHOLDS["d10_budget_bytes"],
        "d11_value_equal": True,
    }
    nodes = {}
    reasons = []
    statuses = []
    for source in ("d9", "d10", "d11"):
        record = evidence[source]
        veto = _blocked_on_record(record, source)
        if veto is not None:
            return _item(
                "blocked",
                source=source,
                record=record,
                thresholds=thresholds,
                reason=f"{source} evidence is unusable: {'; '.join(record.get('errors', []))}",
            )
        if record.get("status") == "blocked":
            return _item(
                "blocked",
                source=source,
                record=record,
                thresholds=thresholds,
                reason=f"{source} evidence is blocked: {record.get('reason') or 'no reason given'}",
            )
        observations = record["observations"]
        nodes[source] = {key: observations.get(key) for key in REQUIRED_OBSERVATIONS[source]}
        statuses.append(record.get("status"))
    d9, d10, d11 = (nodes["d9"], nodes["d10"], nodes["d11"])
    values = dict(nodes)
    expected = _as_int(d9.get("expected_frames"))
    actual = _as_int(d9.get("actual_frames"))
    if expected != THRESHOLDS["val_frames"]:
        reasons.append(f"D9 expected_frames {expected!r} != {THRESHOLDS['val_frames']}")
    if actual != expected:
        reasons.append(f"D9 actual_frames {actual!r} != expected {expected!r}")
    if d9.get("visited_index_sha256") != frozen["val_index_sha256"]:
        reasons.append(
            f"D9 visited_index_sha256 {d9.get('visited_index_sha256')!r} != frozen val_index_sha256"
        )
    if _as_int(d9.get("last_batch_size")) is None:
        reasons.append("D9 last_batch_size is missing")
    budget = _as_int(d10.get("budget_bytes"))
    resident = _as_int(d10.get("max_resident_bytes"))
    if budget != THRESHOLDS["d10_budget_bytes"]:
        reasons.append(f"D10 budget_bytes {budget!r} != {THRESHOLDS['d10_budget_bytes']}")
    if resident is None or budget is None or resident > budget:
        reasons.append(f"D10 max_resident_bytes {resident!r} > budget {budget!r}")
    if d10.get("lru_verified") is not True:
        reasons.append("D10 lru_verified is not true")
    if d10.get("value_equal") is not True:
        reasons.append("D10 value_equal is not true")
    expected_files = _as_int(d11.get("expected_files"))
    validated = _as_int(d11.get("validated_files"))
    if expected_files is None or validated is None or expected_files <= 0:
        reasons.append(f"D11 expected_files/validated_files is unusable ({expected_files!r}/{validated!r})")
    elif expected_files != validated:
        reasons.append(f"D11 validated_files {validated} != expected_files {expected_files}")
    bytes_total = _as_int(d11.get("bytes_total"))
    if bytes_total is None or bytes_total <= 0:
        reasons.append(f"D11 bytes_total is unusable ({bytes_total!r})")
    if d11.get("value_equal") is not True:
        reasons.append("D11 value_equal is not true")
    if not is_sha256(d11.get("manifest_sha")):
        reasons.append(f"D11 manifest_sha is not a 64-hex sha ({d11.get('manifest_sha')!r})")
    if "fail" in statuses:
        reasons.append("one of the D9/D10/D11 indices reports status=fail")
    return _item(
        "pass" if not reasons else "fail",
        source="d9+d10+d11",
        record=evidence["d9"],
        values=values,
        thresholds=thresholds,
        reason="; ".join(reasons),
    )


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def judge(document: dict, meta: dict, root: Path) -> dict:
    frozen = verify_frozen(document, root)
    frozen["document"] = document
    evidence = {
        key: load_evidence_index(
            _dict(document["evidence"]).get(key), key, root, REQUIRED_OBSERVATIONS[key]
        )
        for key in EVIDENCE_KEYS
    }
    items = {
        "A-01": judge_a01(frozen, evidence),
        "A-02": judge_a02(frozen, evidence),
        "A-03": judge_a03(frozen, evidence, root),
        "A-04": judge_a04(frozen, evidence, root),
        "A-05": judge_a05(frozen, evidence),
        "A-06": judge_a06(frozen, evidence),
    }
    statuses = [item["status"] for item in items.values()]
    if "blocked" in statuses:
        code = 2
    elif "fail" in statuses:
        code = 1
    else:
        code = 0
    return {
        "schema": SCHEMA_CHECK,
        "artifacts": meta,
        "frozen": {
            "arch": document["arch"],
            "depth": int(document["depth"]),
            "seed": int(document["seed"]),
            "val_index_sha256": document["val_index_sha256"],
            "frozen_at": document["frozen_at"],
            "external_guard_uses": int(document["external_guard_uses"]),
            "identity_ok": frozen["ok"],
            "identity_errors": frozen["errors"],
            "files": frozen["files"],
            "manifests": frozen["manifests"],
            "summaries": frozen["summaries"],
        },
        "evidence": {
            key: {
                "ok": record["ok"],
                "status": record["status"],
                "path": record["path"],
                "hash": record["sha256"],
                "errors": record["errors"],
            }
            for key, record in evidence.items()
        },
        "items": items,
        "counts": {
            "pass": statuses.count("pass"),
            "fail": statuses.count("fail"),
            "blocked": statuses.count("blocked"),
        },
        "exit_code": code,
    }


def _print_verdicts(result: dict) -> None:
    for name, item in result["items"].items():
        detail = ""
        values = item.get("values") or {}
        if name == "A-01" and item["status"] in ("pass", "fail"):
            detail = (
                f"pos_px={values.get('player_pos_px')} "
                f"perframe={values.get('danger_corr_perframe_median')} "
                f"frames={values.get('frames')}"
            )
        elif name == "A-02" and item["status"] in ("pass", "fail"):
            detail = (
                f"bucket={THRESHOLDS['a16_bucket']} frames={values.get('frames')} "
                f"regret={values.get('regret')} top3={values.get('top3')}"
            )
        elif name == "A-03" and item["status"] in ("pass", "fail"):
            detail = (
                f"a_mae_raw={values.get('a_mae_raw')} b_mae_raw={values.get('b_mae_raw')} "
                f"ratio={values.get('ratio')} thresholds<={THRESHOLDS['a03_ratio_max']}"
            )
        elif name == "A-04" and item["status"] in ("pass", "fail"):
            detail = (
                f"episodes={values.get('episodes')} frames_mean={values.get('frames_mean')} "
                f"hit_step_cap={values.get('hit_step_cap')}"
            )
        elif name == "A-05" and item["status"] in ("pass", "fail"):
            detail = (
                f"passed={values.get('passed')} failed={values.get('failed')} "
                f"threshold>={THRESHOLDS['a05_min_passed']}"
            )
        line = f"{name} {item['status']}" + (f" {detail}" if detail else "")
        if item["status"] != "pass" and item.get("reason"):
            line += f" | {item['reason']}"
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Judge A-01..A-06 of the v1.0.4 freeze from the frozen artifacts"
    )
    parser.add_argument("--artifacts", required=True, help="Stage3/final-artifacts.json")
    parser.add_argument("--out", default="", help="write the per-item verdict JSON here")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="workspace root that the frozen relative paths resolve against (default: repo root)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only write the JSON, print the exit code line"
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    try:
        document, meta = load_artifacts(args.artifacts, root)
        result = judge(document, meta, root)
    except KeyError as exc:
        print(f"v104:EVIDENCE: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        code = v104_error_code(exc)
        if code is None:
            raise
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ManifestError) as exc:
        print(f"v104:EVIDENCE: {exc}", file=sys.stderr)
        return 2
    if args.out:
        out_path = Path(args.out)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            print(f"v104:EVIDENCE: cannot write {args.out}: {exc}", file=sys.stderr)
            return 2
    if not args.quiet:
        _print_verdicts(result)
        print(
            f"pass={result['counts']['pass']} fail={result['counts']['fail']} "
            f"blocked={result['counts']['blocked']} exit={result['exit_code']}"
        )
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
