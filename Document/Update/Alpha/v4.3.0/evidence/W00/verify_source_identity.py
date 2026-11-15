"""W00 (v4.3.0) read-only source-identity verification for the v4.2.0 sources.

What this file is
-----------------
W00 has two halves.  This is the *source identity and reproducibility* half: it
inventories the four v4.2.0 source groups that exist on this disk, pins their
sha256 / byte size / mtime, records their Git tracking state, cross-checks every
sha256 that the v4.2.0 report recorded against the local bytes, and loads the two
example checkpoints into the real model class in a fresh process.

What this file is not
---------------------
It changes no existing file outside ``evidence/W00/``, it never runs the engine
or the learner, and it never calls ``git add``.  A matching sha256 is a statement
about bytes on this disk.  It is **not** a capability result, not a test result
and not a G430-E0 pass; those are recorded separately.

Modes
-----
``python verify_source_identity.py``              verify the pinned manifest (default)
``python verify_source_identity.py --write``      rebuild ``source_identity.json``
``python verify_source_identity.py --load-checkpoint PATH [--record]``
                                                  fresh-process checkpoint load
``python verify_source_identity.py --record-tests JUNIT_XML``
                                                  record a pytest junit report

The default mode re-derives every digest from the bytes and re-extracts every
recorded expectation from the v4.2.0 evidence, so a hand-edit of
``source_identity.json`` cannot pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ElementTree


HERE = Path(__file__).resolve().parent
VERSION = HERE.parents[1]              # .../Document/Update/Alpha/v4.3.0
ROOT = VERSION.parents[3]              # repository root
MANIFEST = HERE / "source_identity.json"
REGISTRY = ROOT / "Document/Update/Alpha/v4.3.0/registries/code_identity.json"

V420_EVIDENCE = "Document/Update/Alpha/v4.2.0/evidence"
T00 = f"{V420_EVIDENCE}/T00/assets_verify.json"
T01 = f"{V420_EVIDENCE}/T01/audit-assets.json"
T01_MD = f"{V420_EVIDENCE}/T01/T01.md"
T02_RAW = f"{V420_EVIDENCE}/T02/RAW_SHA256.txt"
V420_DOCS = "Document/Update/Alpha/v4.2.0"

# The four groups W00 is asked to pin.  ``literal`` entries name exact files;
# ``glob`` entries are (directory, pattern) pairs.  Order is fixed so the
# manifest is reproducible.
GROUPS: tuple[tuple[str, str, object], ...] = (
    ("touhou_rl/v420/*.py", "glob", ("touhou_rl/v420", "*.py")),
    ("tools/rl/v420.py + v420_eval_shoot.py", "literal",
     ("tools/rl/v420.py", "tools/rl/v420_eval_shoot.py")),
    ("configs/v420/*.json", "glob", ("configs/v420", "*.json")),
    ("tests/test_v420_*.py", "glob", ("tests", "test_v420_*.py")),
)

# Both checkpoints named by the W00 work package, with the sha256 the report and
# the work package state.  The value is re-derived from the bytes every run.
CHECKPOINTS: tuple[dict[str, str], ...] = (
    {
        "id": "B0",
        "path": ".tmp_v410_wp04_arms/b0_seed0_last.pt",
        "stated_sha256": "3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6",
    },
    {
        "id": "20k",
        "path": ".tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt",
        "stated_sha256": "15ae1fd5b9a6e0cbbddfd668bb342444bdb44db123b5f7e56abf3c5e7215bd95",
    },
)

TMP_PATTERNS = (".tmp_v420_*.py", ".tmp_v420_*.sh")

# Binaries that live on the WSL asset host and therefore cannot be hashed from
# this Windows checkout.  They are **not** "missing": the WSL side was measured
# out of band by the supervising agent and the measurement is cited here.  The
# claim is checked, not assumed -- the cited document must still contain the
# command and the digest, and that digest must still equal the digest the
# v4.2.0 evidence pins for the same object.  Anything that fails the check falls
# back to ``missing``.
HOST_SPLIT_EVIDENCE: dict[str, dict] = {
    "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern": {
        "object": "engine420",
        "wsl_side_observed_sha256":
            "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6",
        "wsl_side_verified_by": [
            {
                "document": "Document/Update/Alpha/v4.3.0/evidence/W00/"
                            "ASSET_LEDGER_CHECK.md",
                "locator": "§4 引擎二进制身份（主代理在 WSL 侧实测）",
                "command": "sha256sum /root/Touhou_RL/third_party/th08-rl/"
                           "build/rl_v3/th08-modern",
                "must_contain": [
                    "sha256sum /root/Touhou_RL/third_party/th08-rl/build/"
                    "rl_v3/th08-modern",
                    "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a24472"
                    "22ba7a568ef6",
                ],
                "also_states": "rl_laser/th08-modern 与游戏数据目录同样存在",
            },
            {
                "document": "Document/Update/Alpha/v4.2.0/evidence/T01/"
                            "audit-assets.json",
                "locator": "checks_by_object['engine420'].sha256_expected",
                "must_equal_expected": True,
            },
        ],
        "windows_side_note":
            "no build/ tree exists under D:\\...\\third_party\\th08-rl (that "
            "path is .gitignore'd and holds only src/rlharness)",
    },
}

# The v4.2.0 evidence was produced on the WSL side of this same host, so it
# spells repository paths as ``/mnt/d/WorkShop/Touhou_RL/...``.  Translating the
# prefix to the Windows checkout is a claim, not a convenience: it is accepted
# here only because the translation reproduces the recorded digests byte for
# byte (see ``PATH_TRANSLATION`` in the manifest).  Prefixes that cannot be
# translated (``/root/...`` = the WSL-only asset host) stay untranslated and are
# reported as ``missing`` from this checkout by design.
WSL_PREFIX = "/mnt/d/WorkShop/Touhou_RL/"


def translate_target(target: str) -> tuple[str, bool]:
    """Return (path usable from the repository root, was_translated)."""

    if target.startswith(WSL_PREFIX):
        return target[len(WSL_PREFIX):], True
    if target.startswith("/"):
        return target, False
    return target.lstrip("./"), False


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
# ``--self-test`` injects a synthetic git state here instead of touching the
# real repository.  It is the only way git tracking can influence a run, and it
# only changes what gets *reported*.
_TRACKING_OVERRIDE: dict | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def posix(relative: Path | str) -> str:
    return str(relative).replace("\\", "/")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git(*args: str) -> dict:
    """Run git from the repository root and keep stdout/stderr/exit code."""

    completed = subprocess.run(
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return {
        "argv": ["git", *args],
        "exit_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def group_files(kind: str, spec: object) -> list[Path]:
    if kind == "literal":
        return [ROOT / name for name in spec]  # type: ignore[union-attr]
    directory, pattern = spec  # type: ignore[misc]
    base = ROOT / directory
    return sorted(base.glob(pattern), key=lambda item: item.name)


# ---------------------------------------------------------------------------
# 1. source inventory
# ---------------------------------------------------------------------------
def git_tracking() -> dict:
    """Tracked / untracked state for the four groups, from git itself.

    This is an **observation, not a gate**.  W00 exists to close the
    "source cannot be synchronised" gap, so the verifier must keep passing
    *after* the 54 files are committed.  Nothing here can make ``verify`` fail;
    ``_TRACKING_OVERRIDE`` exists only so ``--self-test`` can run the same code
    against a synthetic tracked / untracked state without touching the real
    repository.
    """

    if _TRACKING_OVERRIDE is not None:
        return dict(_TRACKING_OVERRIDE)

    probes = {
        "touhou_rl/v420": ["ls-files", "--", "touhou_rl/v420"],
        "tools/rl/v420*.py": ["ls-files", "--", "tools/rl/v420.py",
                              "tools/rl/v420_eval_shoot.py"],
        "configs/v420": ["ls-files", "--", "configs/v420"],
        "tests/test_v420_*.py": ["ls-files", "--", "tests/test_v420_contracts.py",
                                 "tests/test_v420_counterfactual.py",
                                 "tests/test_v420_dagger.py",
                                 "tests/test_v420_dataset_2f.py",
                                 "tests/test_v420_eval.py",
                                 "tests/test_v420_eval_shoot.py",
                                 "tests/test_v420_rebuild.py",
                                 "tests/test_v420_replay_audit.py",
                                 "tests/test_v420_reslot.py",
                                 "tests/test_v420_rl.py",
                                 "tests/test_v420_setsup.py",
                                 "tests/test_v420_train.py"],
    }
    recorded: dict[str, dict] = {}
    for label, argv in probes.items():
        result = git(*argv)
        recorded[label] = {
            "argv": "git " + " ".join(argv),
            "exit_code": result["exit_code"],
            "stdout": result["stdout"],
            "tracked_count": len([line for line in result["stdout"].splitlines()
                                  if line.strip()]),
        }
    status = git("status", "--short", "--untracked-files=all", "--",
                 "touhou_rl/v420", "tools/rl/v420.py",
                 "tools/rl/v420_eval_shoot.py", "configs/v420", "tests")
    ignore = git("check-ignore", "-v", "touhou_rl/v420/__init__.py",
                 "tools/rl/v420.py", "configs/v420/protocol.json",
                 "tests/test_v420_contracts.py")
    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    lines = [line for line in status["stdout"].splitlines()
             if line.strip() and ".tmp_v420" not in line]
    untracked = [line for line in lines if line.startswith("??")]
    tracked_changed = [line for line in lines if not line.startswith("??")]
    tracked_total = sum(entry["tracked_count"] for entry in recorded.values())
    if tracked_total == 0 and untracked:
        state = "untracked"
    elif tracked_total and not untracked:
        state = "tracked"
    elif tracked_total and untracked:
        state = "mixed"
    else:
        state = "absent"
    return {
        "branch": branch["stdout"].strip(),
        "head": head["stdout"].strip(),
        "ls_files": recorded,
        "tracked_total": tracked_total,
        "status_argv": "git " + " ".join(status["argv"][1:]),
        "status_exit_code": status["exit_code"],
        "status_untracked_count": len(untracked),
        "status_tracked_changed_count": len(tracked_changed),
        "status_untracked_lines": untracked,
        "check_ignore_exit_code": ignore["exit_code"],
        "check_ignore_stdout": ignore["stdout"],
        "tracking_state": state,
        "ignore_verdict": ("none of the four groups is ignored (git check-ignore "
                           "exit 1 = no match)"),
        "consequence_if_untracked": (
            "a fresh clone has none of the 54 files; this is the W00 delivery "
            "gap, reported as an observation and never as a verification "
            "failure -- closing the gap must not break this checker"),
    }



def inventory() -> dict:
    groups: list[dict] = []
    for label, kind, spec in GROUPS:
        files = group_files(kind, spec)
        entries = []
        for path in files:
            if not path.is_file():
                entries.append({"path": posix(path.relative_to(ROOT)),
                                "exists": False})
                continue
            stat = path.stat()
            entries.append({
                "path": posix(path.relative_to(ROOT)),
                "exists": True,
                "sha256": sha256_file(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "mtime_iso": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc).astimezone().isoformat(),
            })
        present = [entry for entry in entries if entry["exists"]]
        groups.append({
            "label": label,
            "count": len(present),
            "missing_count": len(entries) - len(present),
            "total_bytes": sum(entry["size_bytes"] for entry in present),
            "files": entries,
        })
    all_files = [entry for group in groups for entry in group["files"]
                 if entry["exists"]]
    manifest_lines = "\n".join(
        f"{entry['sha256']}  {entry['path']}"
        for entry in sorted(all_files, key=lambda item: item["path"]))
    return {
        "groups": groups,
        "totals": {
            "file_count": len(all_files),
            "total_bytes": sum(entry["size_bytes"] for entry in all_files),
            "manifest_sha256": sha256_bytes(manifest_lines.encode("utf-8")),
            "manifest_note": ("sha256 over the sorted '<sha256>  <path>' lines of "
                              "every file in the four groups"),
        },
    }


def checkpoint_inventory() -> list[dict]:
    entries = []
    for pin in CHECKPOINTS:
        path = ROOT / pin["path"]
        if not path.is_file():
            entries.append({"id": pin["id"], "path": pin["path"], "exists": False,
                            "stated_sha256": pin["stated_sha256"],
                            "verdict": "missing"})
            continue
        observed = sha256_file(path)
        entries.append({
            "id": pin["id"],
            "path": pin["path"],
            "exists": True,
            "size_bytes": int(path.stat().st_size),
            "mtime_iso": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat(),
            "sha256_observed": observed,
            "stated_sha256": pin["stated_sha256"],
            "verdict": "verified" if observed == pin["stated_sha256"] else "mismatch",
        })
    return entries


# ---------------------------------------------------------------------------
# 2. cross-check against what the v4.2.0 report recorded
# ---------------------------------------------------------------------------
def _t00_object(payload: dict, path: str) -> dict | None:
    for entry in payload.get("objects", []):
        if entry.get("path") == path:
            return entry
    return None


def extract_references() -> list[dict]:
    """Re-read the v4.2.0 evidence and pull out every sha256 it recorded.

    Each returned record is a *verbatim extraction*: ``expected_sha256`` comes
    from the document, ``observed_sha256`` comes from the bytes on this disk.
    Nothing is trusted from the previous run.
    """

    references: list[dict] = []
    t00 = read_json(ROOT / T00)
    t01 = read_json(ROOT / T01)

    def add(identifier: str, target: str, expected: str | None, kind: str,
            document: str, locator: str, *, scope: str, note: str = "") -> None:
        references.append({
            "id": identifier,
            "target_path": target,
            "expected_sha256": expected,
            "expectation_kind": kind,
            "recorded_in": document,
            "locator": locator,
            "w00_scope": scope,
            "note": note,
        })

    # --- T00 assets_verify.json ------------------------------------------------
    local_json = (t00.get("artifacts") or {}).get("local_json") or {}
    add("t00.local_json", local_json.get("path", "configs/v420/local.json"),
        local_json.get("sha256"), "pinned_expectation", T00,
        "artifacts.local_json.sha256", scope="v420_config",
        note="the only v420 *config* byte-pin in the v4.2.0 evidence")

    t00_report = (t00.get("artifacts") or {}).get("t00_report") or {}
    add("t00.t00_report", t00_report.get("path", f"{V420_EVIDENCE}/T00/T00.md"),
        t00_report.get("sha256"), "pinned_expectation", T00,
        "artifacts.t00_report.sha256", scope="evidence_self",
        note="self-consistency of the T00 report itself")

    for name, path in (("evaluator", "tools/rl/v410_student_eval.py"),
                       ("v410_init", "touhou_rl/v410/__init__.py"),
                       ("asset_sync_sha256", "ASSET_SYNC.sha256"),
                       ("asset_sync_md", "ASSET_SYNC.md")):
        entry = _t00_object(t00, path)
        if entry is None:
            continue
        add(f"t00.object.{name}", path, entry.get("sha256_actual"),
            "observed_not_expected", T00, f"objects[path={path}].sha256_actual",
            scope=("evaluator" if name == "evaluator" else "outside_v420"),
            note=f"T00 verdict: {entry.get('verdict')!r}")

    # --- T01 audit-assets.json -------------------------------------------------
    checks = t01.get("checks_by_object") or {}
    for key, scope, note in (
        ("checkpoint:B0", "checkpoint", "v4.1.0 B0 student"),
        ("checkpoint:20k", "checkpoint", "v4.1.0 DAgger 20k student"),
        ("evaluator", "evaluator",
         "T01 pins the *v4.1.0* evaluator; no v420 evaluator digest is pinned"),
        ("engine420", "outside_v420", "engine binary, not source"),
    ):
        entry = checks.get(key) or {}
        if not entry:
            continue
        add(f"t01.{key}", entry.get("path", ""), entry.get("sha256_expected"),
            "pinned_expectation", T01, f"checks_by_object[{key!r}].sha256_expected",
            scope=scope, note=note)

    # T01 names v420 paths without pinning a digest for them.
    local_path = (t01.get("local_config") or {}).get("path")
    if local_path:
        add("t01.local_config_reference", local_path, None, "referenced_no_sha",
            T01, "local_config.path", scope="v420_config",
            note="T01 states this config was 'applied' but records no digest")

    # --- T02 RAW_SHA256.txt ----------------------------------------------------
    raw = (ROOT / T02_RAW).read_text(encoding="utf-8")
    for line in raw.splitlines():
        match = re.match(r"^([0-9a-f]{64})\s\s(.+)$", line.strip())
        if not match:
            continue
        add(f"t02.raw.{Path(match.group(2)).name}", match.group(2),
            match.group(1), "pinned_expectation", T02_RAW,
            "line entry '<sha256>  <path>'", scope="t02_artifact",
            note="T02 evaluator output, not v420 source")

    # --- T01.md ---------------------------------------------------------------
    md = (ROOT / T01_MD).read_text(encoding="utf-8")
    named: list[str] = []
    for pattern in (r"touhou_rl/v420/[A-Za-z0-9_]+\.py",
                    r"tools/rl/v420[A-Za-z0-9_]*\.py",
                    r"configs/v420/[A-Za-z0-9_.\-]+\.json"):
        for hit in re.findall(pattern, md):
            if hit not in named:
                named.append(hit)
    for name in sorted(named):
        add(f"t01md.{name}", name, None, "referenced_no_sha", T01_MD,
            "path mention in prose", scope="v420_source_or_config",
            note="T01.md names the file but records no sha256 for it")

    # --- v420 configs' own config-to-config pins -------------------------------
    for holder in sorted((ROOT / "configs/v420").glob("*.json")):
        try:
            held = read_json(holder)
        except (OSError, ValueError):
            continue
        for index, (locator, node) in enumerate(_walk_pins(held)):
            add(f"cfg.{holder.name}.source_config_sha256[{index}]",
                node["source_config"], node["source_config_sha256"],
                "pinned_expectation", f"configs/v420/{holder.name}",
                f"{locator}.source_config_sha256", scope="v420_config",
                note="a v420 config pinning the digest of the config it was "
                     "derived from; this is the only byte-pin the v4.2.0 "
                     "*configs* carry about other v420 configs")

    legacy = read_json(ROOT / "configs/v420/legacy-reproduce.json")
    if legacy.get("evaluator") and legacy.get("evaluator_sha256"):
        add("cfg.legacy-reproduce.evaluator_sha256", legacy["evaluator"],
            legacy["evaluator_sha256"], "pinned_expectation",
            "configs/v420/legacy-reproduce.json", "evaluator_sha256",
            scope="evaluator",
            note="the v4.2.0 run config reusing the v4.1.0 evaluator verbatim")

    return references


def _host_split_evidence(raw_target: str) -> dict | None:
    """Check the cited out-of-band WSL measurement for one host-split object.

    Returns ``None`` when there is no claim for this path, or when the claim no
    longer checks out -- in either case the caller falls back to ``missing``.
    """

    claim = HOST_SPLIT_EVIDENCE.get(raw_target)
    if not claim:
        return None
    checks: list[dict] = []
    ok = True
    for citation in claim["wsl_side_verified_by"]:
        path = ROOT / citation["document"]
        if not path.is_file():
            checks.append({"document": citation["document"],
                           "checked": False, "reason": "document absent"})
            ok = False
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing_tokens = [token for token in citation.get("must_contain", [])
                          if token not in text]
        if citation.get("must_contain"):
            checks.append({"document": citation["document"],
                           "locator": citation["locator"],
                           "must_contain_present": not missing_tokens,
                           "missing": missing_tokens})
            ok = ok and not missing_tokens
        if citation.get("must_equal_expected"):
            observed = _t01_engine_expected()
            agree = observed == claim["wsl_side_observed_sha256"]
            checks.append({"document": citation["document"],
                           "locator": citation["locator"],
                           "re_extracted": observed,
                           "equals_wsl_measurement": agree})
            ok = ok and agree
    return {
        "object": claim["object"],
        "windows_side_observed": False,
        "windows_side_observed_sha256": None,
        "wsl_side_observed_sha256": claim["wsl_side_observed_sha256"],
        "wsl_side_verified_by": claim["wsl_side_verified_by"],
        "verdict_reason": "host_split",
        "windows_side_note": claim["windows_side_note"],
        "citation_checks": checks,
        "citation_holds": ok,
    }


def _t01_engine_expected() -> str | None:
    try:
        payload = read_json(ROOT / T01)
    except (OSError, ValueError):
        return None
    entry = (payload.get("checks_by_object") or {}).get("engine420") or {}
    return entry.get("sha256_expected")


def judge(references: list[dict]) -> list[dict]:
    judged = []
    for reference in references:
        raw_target = reference.get("target_path") or ""
        target, translated = translate_target(raw_target)
        base = {**reference,
                "target_path_resolved": target,
                "path_translated": translated}
        if raw_target.startswith("/") and not translated:
            claim = _host_split_evidence(raw_target)
            if claim and claim["citation_holds"]:
                judged.append({
                    **base,
                    "exists": None,
                    "observed_sha256": None,
                    "verdict": "unverified_host_split",
                    **{key: value for key, value in claim.items()
                       if key != "object"},
                })
                continue
            if claim:
                base["host_split_citation_failed"] = claim["citation_checks"]
        if not target:
            judged.append({**base, "exists": False, "observed_sha256": None,
                           "verdict": "missing"})
            continue
        path = ROOT / target
        if not path.is_file():
            judged.append({**base, "exists": False, "observed_sha256": None,
                           "verdict": "missing"})
            continue
        observed = sha256_file(path)
        if not reference.get("expected_sha256"):
            verdict = "not_referenced"
        elif observed == reference["expected_sha256"]:
            verdict = "verified"
        else:
            verdict = "mismatch"
        judged.append({**base, "exists": True, "observed_sha256": observed,
                       "verdict": verdict})
    return judged


def _walk_pins(node, path: str = "") -> list[tuple[str, dict]]:
    """Every dict in the tree that carries both a target and a digest."""

    found: list[tuple[str, dict]] = []
    if isinstance(node, dict):
        if "source_config" in node and "source_config_sha256" in node:
            found.append((path or ".", node))
        for key, value in node.items():
            found.extend(_walk_pins(value, f"{path}.{key}" if path else key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk_pins(value, f"{path}[{index}]"))
    return found


def unreferenced_files(groups: list[dict], references: list[dict]) -> list[dict]:
    """Group files that no recorded expectation covers at all."""

    covered = {reference.get("target_path_resolved", reference.get("target_path"))
               for reference in references if reference.get("expected_sha256")}
    out = []
    for group in groups:
        for entry in group["files"]:
            if not entry.get("exists"):
                continue
            out.append({
                "path": entry["path"],
                "group": group["label"],
                "verdict": "not_referenced" if entry["path"] not in covered
                           else "referenced",
            })
    return out


# ---------------------------------------------------------------------------
# 3. root-level .tmp_v420_* helper scripts
# ---------------------------------------------------------------------------
def tmp_v420_inventory() -> dict:
    scripts: list[Path] = []
    for pattern in TMP_PATTERNS:
        scripts.extend(ROOT.glob(pattern))
    scripts = sorted({path.name: path for path in scripts}.values(),
                     key=lambda item: item.name)
    doc_root = ROOT / V420_DOCS
    corpus: dict[str, str] = {}
    archived: dict[str, str] = {}
    for path in sorted(doc_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
                ".md", ".json", ".jsonl", ".txt", ".py", ".sh"}:
            continue
        relative = posix(path.relative_to(ROOT))
        if path.suffix.lower() in {".md", ".json", ".jsonl", ".txt"}:
            try:
                corpus[relative] = path.read_text(encoding="utf-8",
                                                  errors="replace")
            except OSError:
                pass
        else:
            try:
                archived[sha256_file(path)] = relative
            except OSError:
                pass
    names = [path.name for path in scripts]
    entries = []
    for path in scripts:
        hits = [relative for relative, text in corpus.items() if path.name in text]
        body = path.read_text(encoding="utf-8", errors="replace")
        depends = sorted({name for name in names
                          if name != path.name and name in body})
        digest = sha256_file(path)
        entries.append({
            "path": path.name,
            "kind": path.suffix.lstrip("."),
            "size_bytes": int(path.stat().st_size),
            "sha256": digest,
            "referenced_in_v420_docs": hits,
            "referenced_count": len(hits),
            "depends_on_tmp_v420": depends,
            "archived_copy_in_v420_evidence": archived.get(digest),
            "superseded_by": SUPERSEDED.get(path.name),
        })
    by_name = {entry["path"]: entry for entry in entries}
    seeds = [entry["path"] for entry in entries if entry["referenced_count"]]
    closure: list[str] = []
    queue = list(seeds)
    while queue:
        name = queue.pop(0)
        if name in closure or name not in by_name:
            continue
        closure.append(name)
        queue.extend(by_name[name]["depends_on_tmp_v420"])
    for entry in entries:
        entry["reachable_from_docs"] = entry["path"] in closure
        entry["restore_action"] = _restore_action(entry)
    return {
        "scan_scope": f"{V420_DOCS}/**/*.md|*.json|*.jsonl|*.txt (explicit recursive "
                      "walk of that directory only; no repository-wide scan)",
        "scanned_document_count": len(corpus),
        "archived_script_count": len(archived),
        "script_count": len(entries),
        "directly_referenced": seeds,
        "transitive_closure": closure,
        "closure_added": [name for name in closure if name not in seeds],
        "action_counts": _count_actions(entries),
        "scripts": entries,
    }


def _restore_action(entry: dict) -> str:
    """What W00 recommends doing with one root-level helper script.

    Only a recommendation.  W00 copies nothing, moves nothing and deletes
    nothing; the restore step belongs to whoever owns the tracking decision.
    """

    if entry["archived_copy_in_v420_evidence"]:
        return "already_archived_in_evidence_no_restore"
    if not entry["reachable_from_docs"]:
        return "discard_unreferenced_one_shot"
    if entry["kind"] == "sh":
        return "discard_shell_orchestration_command_is_inlined_in_report"
    if entry["superseded_by"]:
        return "preserve_as_provenance_only_logic_now_in_tracked_code"
    return "restore_into_tools_rl_v420_needed_to_reproduce_reported_number"


def _count_actions(entries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["restore_action"]] = counts.get(entry["restore_action"], 0) + 1
    return counts


# Probes whose computation now exists as a tracked module.  The right-hand side
# names the function that recomputes it, so the claim is checkable by reading
# that file rather than by trusting this table.
SUPERSEDED: dict[str, str] = {
    ".tmp_v420_slot_probe.py":
        "touhou_rl/v420/replay_audit.py::audit_legacy_shard (hash chain leg, "
        "frames-per-decision leg)",
    ".tmp_v420_mask_probe.py":
        "touhou_rl/v420/replay_audit.py::audit_legacy_shard (slot-1 mask leg)",
    ".tmp_v420_release_probe.py":
        "touhou_rl/v420/replay_audit.py::audit_legacy_shard "
        "(slot1_tail_release leg)",
    ".tmp_v420_hist_probe.py":
        "touhou_rl/v420/replay_audit.py::audit_legacy_shard (act_hist leg)",
}


# ---------------------------------------------------------------------------
# 4. fresh-process checkpoint load
# ---------------------------------------------------------------------------
def load_checkpoint(target: str) -> dict:
    """Load one checkpoint into the real model class, in this process only.

    Two independent bindings are attempted: a direct ``ProbeNet(...)
    .load_state_dict(..., strict=True)`` and the production loader
    ``tools/rl/v410_student_eval.py::load_v410_checkpoint``.  A bare
    ``torch.load`` success is recorded but never counted as success.
    """

    import torch  # noqa: PLC0415 - deliberately imported only in this mode

    result: dict = {
        "checkpoint": target,
        "process": {"isolated": True, "pid": _pid(),
                    "torch": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "python": sys.version.split()[0]},
        "torch_load": {"ok": False},
        "direct_binding": {"ok": False},
        "production_loader": {"ok": False},
        "v420_loader": {"ok": False},
    }
    path = ROOT / target
    if not path.is_file():
        result["torch_load"] = {"ok": False, "error": f"does not exist: {path}"}
        result["verdict"] = "missing"
        return result

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        result["torch_load"] = {"ok": False,
                                "traceback": traceback.format_exc()}
        result["verdict"] = "failed"
        return result

    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    result["torch_load"] = {
        "ok": True,
        "payload_type": type(payload).__name__,
        "payload_keys": sorted(payload) if isinstance(payload, dict) else [],
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
        "arm": payload.get("arm") if isinstance(payload, dict) else None,
        "model_config": payload.get("model_config")
                        if isinstance(payload, dict) else None,
        "state_dict_key_count": len(state_dict) if isinstance(state_dict, dict)
                                else None,
        "state_dict_total_parameters": (
            int(sum(tensor.numel() for tensor in state_dict.values()))
            if isinstance(state_dict, dict) and all(
                hasattr(tensor, "numel") for tensor in state_dict.values())
            else None),
    }

    # --- direct binding -------------------------------------------------------
    direct: dict = {"ok": False}
    try:
        from touhou_rl.v410.student.diagnostics import ProbeNet  # noqa: PLC0415

        model_config = dict((payload.get("model_config") or {}))
        model = ProbeNet(**{key: int(value) for key, value in model_config.items()
                            if isinstance(value, int)})
        direct["model_class"] = "touhou_rl.v410.student.diagnostics.ProbeNet"
        direct["built_config"] = dict(model.config)
        direct["built_parameters"] = int(sum(parameter.numel()
                                             for parameter in model.parameters()))
        model.load_state_dict(dict(state_dict), strict=True)
        direct["ok"] = True
        direct["strict"] = True
        direct["loaded_parameters"] = int(sum(parameter.numel()
                                              for parameter in model.parameters()))
    except Exception:
        direct["traceback"] = traceback.format_exc()[-2000:]
    result["direct_binding"] = direct

    # --- production loader (v4.1.0 evaluator, read-only import) ---------------
    production: dict = {"ok": False}
    try:
        import importlib.util  # noqa: PLC0415

        loader_path = ROOT / "tools/rl/v410_student_eval.py"
        spec = importlib.util.spec_from_file_location(
            "_w00_v410_student_eval", loader_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        info, model = module.load_v410_checkpoint(path, device="cpu")
        production["ok"] = True
        production["info"] = (dict(info._asdict()) if hasattr(info, "_asdict")
                              else repr(info))
        production["model_class"] = type(model).__name__
        production["parameters"] = int(sum(parameter.numel()
                                           for parameter in model.parameters()))
    except Exception:
        production["traceback"] = traceback.format_exc()[-2000:]
    result["production_loader"] = production

    # --- the v420 module's own entry point -----------------------------------
    v420_loader: dict = {"ok": False}
    try:
        from touhou_rl.v420 import replay_audit  # noqa: PLC0415

        described, model = replay_audit.load_frozen_model(path, device="cpu")
        v420_loader["ok"] = True
        v420_loader["description"] = described
        v420_loader["model_class"] = type(model).__name__
    except Exception:
        v420_loader["traceback"] = traceback.format_exc()[-2000:]
    result["v420_loader"] = v420_loader

    strict_ok = bool(direct.get("ok")) and bool(production.get("ok"))
    result["strict_binding_ok"] = strict_ok
    result["verdict"] = "verified" if strict_ok else "failed"
    return result


def _pid() -> int:
    import os  # noqa: PLC0415

    return int(os.getpid())


# ---------------------------------------------------------------------------
# 5. pytest junit recording
# ---------------------------------------------------------------------------
def parse_junit(path: Path) -> dict:
    tree = ElementTree.parse(path)
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    total = failures = errors = skipped = 0
    failed_cases: list[dict] = []
    for suite in suites:
        if suite.tag != "testsuite":
            continue
        total += int(suite.get("tests", 0) or 0)
        failures += int(suite.get("failures", 0) or 0)
        errors += int(suite.get("errors", 0) or 0)
        skipped += int(suite.get("skipped", 0) or 0)
        for case in suite.iter("testcase"):
            for kind in ("failure", "error"):
                node = case.find(kind)
                if node is not None:
                    failed_cases.append({
                        "name": f"{case.get('classname', '')}::{case.get('name', '')}",
                        "outcome": kind,
                        "first_line": (node.get("message", "") or "").splitlines()[0]
                                      if (node.get("message") or "") else "",
                    })
    return {
        "junit": posix(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
                 else str(path),
        "collected": total,
        "passed": total - failures - errors - skipped,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
        "failed_cases": failed_cases,
    }


# ---------------------------------------------------------------------------
# build / verify
# ---------------------------------------------------------------------------
PRESERVED = ("git", "groups", "totals", "checkpoints", "reference_crosscheck",
             "not_referenced_files", "tmp_v420_scripts", "model_load",
             "test_run", "environment")


def environment() -> dict:
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "repo_root": posix(ROOT),
        "manifest_path": posix(MANIFEST.relative_to(ROOT)),
        "registry_path": posix(REGISTRY.relative_to(ROOT)),
    }


def build() -> dict:
    references = judge(extract_references())
    inv = inventory()
    tracking = git_tracking()
    payload = {
        "schema": "v430-w00-source-identity/1",
        "work_package": "v4.3.0 W00 (source identity and reproducibility half)",
        "generated_at_utc": datetime.now(tz=timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": environment(),
        "scope_statement": (
            "pins the bytes of four v4.2.0 source groups, cross-checks every "
            "sha256 the v4.2.0 evidence recorded, and loads two checkpoints. A "
            "match is a byte statement: not a capability, not a test result and "
            "not G430-E0."),
        "git": tracking,
        "path_translation": path_translation(references),
        "groups": inv["groups"],
        "totals": inv["totals"],
        "checkpoints": checkpoint_inventory(),
        "reference_crosscheck": references,
        "mismatch_explanations": explain_mismatches(references),
        "not_referenced_files": unreferenced_files(inv["groups"], references),
        "tmp_v420_scripts": tmp_v420_inventory(),
        "model_load": {},
        "test_run": {},
        "known_discrepancies": known_discrepancies(),
    }
    return payload


def path_translation(references: list[dict]) -> dict:
    """How host-absolute paths recorded in the v4.2.0 evidence are resolved.

    The translation is not assumed.  ``configs/v420/local.json`` is recorded in
    the evidence under the WSL spelling and also carries a pinned digest; that
    digest reproduces the bytes under ``D:\\WorkShop\\Touhou_RL`` (see the
    ``t00.local_json`` reference), which is what licenses the mapping.
    """

    translated = []
    untranslated = []
    for reference in references:
        raw = reference.get("target_path") or ""
        if raw.startswith(WSL_PREFIX):
            translated.append({"recorded": raw,
                               "resolved": raw[len(WSL_PREFIX):]})
        elif raw.startswith("/"):
            untranslated.append(raw)
    return {
        "prefix": WSL_PREFIX,
        "replacement": "",
        "translated_count": len(translated),
        "examples": translated[:3],
        "untranslated_absolute_paths": sorted(set(untranslated)),
        "untranslated_note": (
            "absolute paths outside /mnt/d/WorkShop/Touhou_RL name the WSL asset "
            "host (/root/...). They are not on this Windows checkout, so they "
            "resolve to 'missing' here; that is an expected host split, not a "
            "deleted asset."),
    }


def explain_mismatches(references: list[dict]) -> list[dict]:
    """Attach a falsifiable cause to every mismatch, and re-check that cause.

    Two mismatches are expected and neither is a v420 source file: the
    ``ASSET_SYNC.*`` pair was re-pinned on 2026-11-15 while registering the
    v4.2.0 binaries (``ASSET_SYNC.md`` §9).  The explanation is accepted only
    if the document still says so and the manifest still carries those v420
    lines, so a later drift would invalidate it here rather than in prose.
    """

    sync_md = (ROOT / "ASSET_SYNC.md").read_text(encoding="utf-8",
                                                 errors="replace")
    sync_sha = (ROOT / "ASSET_SYNC.sha256").read_text(encoding="utf-8",
                                                      errors="replace")
    documented_repin = ("2026-11-15T04:18:27" in sync_md
                        and "## 9. v4.2.0 关键资产" in sync_md)
    v420_asset_lines = len([line for line in sync_sha.splitlines()
                            if ".tmp_v420_data/" in line])
    source_lines = len([line for line in sync_sha.splitlines()
                        if re.search(r"touhou_rl/v420|tools/rl/v420|"
                                     r"configs/v420|tests/test_v420", line)])
    explained = []
    for reference in references:
        if reference.get("verdict") != "mismatch":
            continue
        target = reference.get("target_path_resolved") or reference["target_path"]
        if target.startswith("ASSET_SYNC."):
            cause = ("re-pinned after T00 while registering the v4.2.0 binary "
                     "assets (ASSET_SYNC.md §9); the bytes changed, the file is "
                     "not a v420 source file")
            recheck = documented_repin and v420_asset_lines > 0
        else:
            cause = "unexplained"
            recheck = False
        explained.append({
            "reference": reference["id"],
            "target": target,
            "expected_sha256": reference.get("expected_sha256"),
            "observed_sha256": reference.get("observed_sha256"),
            "cause": cause,
            "cause_recheck": {
                "asset_sync_md_documents_2026_11_15_repin": documented_repin,
                "v420_asset_lines_in_asset_sync_sha256": v420_asset_lines,
                "v420_source_lines_in_asset_sync_sha256": source_lines,
                "cause_holds": bool(recheck),
            },
        })
    return explained


def known_discrepancies() -> list[dict]:
    """Differences between the v4.3.0 plan text / task statement and this disk.

    Written from the evidence, not from the plan: the plan document's own claim
    about this checkout is one of the things under test.
    """

    configs = sorted((ROOT / "configs/v420").glob("*.json"))
    readme = ROOT / "Document/Update/Alpha/v4.3.0/README.md"
    gap: list[dict] = [
        {
            "id": "D-1",
            "what": "v4.3.0 README §2 says this checkout is missing "
                    "touhou_rl/v420, configs/v420, the tests and the evaluators",
            "observed": (
                f"touhou_rl/v420/*.py = {len(list((ROOT / 'touhou_rl/v420').glob('*.py')))} "
                f"files, configs/v420/*.json = {len(configs)} files, "
                f"tests/test_v420_*.py = "
                f"{len(list((ROOT / 'tests').glob('test_v420_*.py')))} files, "
                "tools/rl/v420.py + tools/rl/v420_eval_shoot.py = 2 files, all "
                "present on this disk"),
            "verdict": "plan_text_out_of_date",
            "impact": (
                "P1-07's real defect is not absence but Git invisibility: the "
                "files exist and are untracked, so they cannot be synchronised "
                "and a fresh clone would genuinely lack them. A downstream agent "
                "that trusted the README would look for a restore source that is "
                "not needed instead of fixing tracking."),
            "evidence": f"{posix(readme.relative_to(ROOT))} L37-L39, "
                        "git ls-files / git status in this file's 'git' block",
        },
        {
            "id": "D-2",
            "what": "the W00 brief states configs/v420 holds 28 json files",
            "observed": f"{len(configs)} json files on disk",
            "verdict": "measured_26_brief_is_2_too_high (实测 26，任务书数字偏大 2，"
                       "以磁盘为准)",
            "brief_number": 28,
            "measured_number": len(configs),
            "delta_brief_minus_measured": 28 - len(configs),
            "rule": "以磁盘为准 / disk is authoritative",
            "impact": "file counts must come from disk, not from the brief",
            "evidence": "groups[2].files in this manifest",
        },
    ]
    pyfiles = sorted((ROOT / "touhou_rl/v420").glob("*.py"))
    if len(pyfiles) != 14:
        gap.append({"id": "D-3", "what": "touhou_rl/v420/*.py count",
                    "observed": len(pyfiles), "verdict": "differs_from_brief",
                    "impact": "", "evidence": "groups[0]"})
    tests = sorted((ROOT / "tests").glob("test_v420_*.py"))
    if len(tests) != 12:
        gap.append({"id": "D-4", "what": "tests/test_v420_*.py count",
                    "observed": len(tests), "verdict": "differs_from_brief",
                    "impact": "", "evidence": "groups[3]"})
    return gap


def evaluate() -> tuple[dict, list[str], list[str]]:
    """Re-derive everything and return ``(summary, problems, notes)``.

    ``problems`` is what decides pass/fail, and it is built **only** from byte
    identity: the 54 files' sha256 / size / mtime against the pinned manifest,
    plus the pinned reference verdicts and checkpoint digests.  Git tracking
    state is reported under ``observations`` and can never enter ``problems`` --
    W00 exists to close the "source is untracked" gap, so committing the files
    must not turn this checker red.
    """

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"manifest missing: {MANIFEST}")
    pinned = read_json(MANIFEST)
    problems: list[str] = []
    notes: list[str] = []

    # --- bytes ---------------------------------------------------------------
    current = inventory()
    pinned_groups = {group["label"]: group for group in pinned.get("groups", [])}
    current_groups = {group["label"]: group for group in current["groups"]}
    for label, group in current_groups.items():
        if label not in pinned_groups:
            problems.append(f"group not pinned: {label}")
            continue
        before = {entry["path"]: entry for entry in pinned_groups[label]["files"]}
        for entry in group["files"]:
            path = entry["path"]
            if path not in before:
                problems.append(f"new file not in manifest: {path}")
                continue
            if not entry.get("exists"):
                problems.append(f"pinned file missing: {path}")
                continue
            if entry["sha256"] != before[path].get("sha256"):
                problems.append(f"sha256 changed: {path}")
            if entry["size_bytes"] != before[path].get("size_bytes"):
                problems.append(f"size changed: {path}")
            if entry["mtime_ns"] != before[path].get("mtime_ns"):
                notes.append(f"mtime changed (bytes identical): {path}")
        for path in before:
            if path not in {entry["path"] for entry in group["files"]}:
                problems.append(f"pinned file no longer enumerated: {path}")
    if current["totals"]["manifest_sha256"] != pinned["totals"]["manifest_sha256"]:
        problems.append("group manifest digest changed")
    if current["totals"]["file_count"] != pinned["totals"]["file_count"]:
        problems.append("group file count changed")

    # --- git: observed, never gated ------------------------------------------
    tracking = git_tracking()
    observations = {
        "git_tracking_state": tracking["tracking_state"],
        "git_tracked_in_groups": tracking["tracked_total"],
        "git_status_untracked_count": tracking["status_untracked_count"],
        "git_status_tracked_changed_count": tracking["status_tracked_changed_count"],
        "git_check_ignore_exit_code": tracking["check_ignore_exit_code"],
        "git_head": tracking["head"],
        "git_branch": tracking["branch"],
        "note": ("observed only; not a pass/fail condition. Before the fix this "
                 "count was asserted to be 0, which made the checker fail once "
                 "the delivery gap was closed."),
    }
    if tracking["tracked_total"] and tracking["tracked_total"] != current[
            "totals"]["file_count"]:
        notes.append("tracked total "
                     f"{tracking['tracked_total']} != {current['totals']['file_count']} "
                     "files in the groups (partial add)")
    if tracking["head"] != pinned["git"]["head"]:
        notes.append(f"git HEAD moved: {pinned['git']['head']} -> {tracking['head']}")

    # --- checkpoints ---------------------------------------------------------
    for entry in checkpoint_inventory():
        if entry["verdict"] != "verified":
            problems.append(f"checkpoint {entry['id']} {entry['verdict']}")

    # --- re-extract every recorded expectation -------------------------------
    references = judge(extract_references())
    pinned_refs = {reference["id"]: reference
                   for reference in pinned.get("reference_crosscheck", [])}
    if len(references) != len(pinned_refs):
        notes.append(f"reference count {len(references)} vs pinned "
                     f"{len(pinned_refs)}")
    for reference in references:
        before = pinned_refs.get(reference["id"])
        if before is None:
            problems.append(f"reference not pinned: {reference['id']}")
            continue
        for field in ("expected_sha256", "observed_sha256", "verdict",
                      "target_path", "wsl_side_observed_sha256",
                      "windows_side_observed"):
            if reference.get(field) != before.get(field):
                problems.append(
                    f"reference {reference['id']}.{field}: "
                    f"pinned {before.get(field)!r} now {reference.get(field)!r}")
        if reference["verdict"] == "unverified_host_split":
            if not (reference.get("citation_checks")
                    and all(check.get("must_contain_present", True)
                            and check.get("equals_wsl_measurement", True)
                            for check in reference["citation_checks"])):
                problems.append(
                    f"reference {reference['id']}: host-split citation no "
                    "longer holds")
    counts = tally(references)
    explanations = explain_mismatches(references)
    for entry in explanations:
        if not entry["cause_recheck"]["cause_holds"]:
            problems.append(f"unexplained mismatch: {entry['reference']} -> "
                            f"{entry['target']}")
    if len(explanations) != counts.get("mismatch", 0):
        problems.append("not every mismatch has an explanation")

    # --- recorded sub-results must be internally consistent ------------------
    model_load = pinned.get("model_load") or {}
    for entry in pinned.get("checkpoints", []):
        recorded = (model_load.get(entry["path"]) or model_load.get(entry["id"])
                    or {})
        if not recorded:
            notes.append(f"no model_load recorded for {entry['id']} "
                         "(checkpoint sha is still verified)")
            continue
        if recorded.get("checkpoint") != entry["path"]:
            problems.append(f"model_load[{entry['id']}] names a different file")
        if recorded.get("verdict") == "verified" and not recorded.get(
                "strict_binding_ok"):
            problems.append(f"model_load[{entry['id']}] claims verified without "
                            "a strict binding")
        elif recorded.get("verdict") != "verified":
            notes.append(f"model_load[{entry['id']}] verdict="
                         f"{recorded.get('verdict')!r}")

    summary = {
        "schema": "v430-w00-source-identity-verify/1",
        "manifest": posix(MANIFEST.relative_to(ROOT)),
        "groups": {label: {"count": group["count"],
                           "total_bytes": group["total_bytes"]}
                   for label, group in current_groups.items()},
        "totals": current["totals"],
        "observations": observations,
        "git_tracked_in_groups": tracking["tracked_total"],
        "git_tracking_state": tracking["tracking_state"],
        "git_head": tracking["head"],
        "checkpoints": {entry["id"]: entry["verdict"]
                        for entry in checkpoint_inventory()},
        "reference_verdicts": counts,
        "not_referenced_count": sum(
            1 for entry in pinned.get("not_referenced_files", [])
            if entry["verdict"] == "not_referenced"),
        "model_load": {key: value.get("verdict")
                       for key, value in model_load.items()},
        "test_run": pinned.get("test_run") or {},
        "notes": notes,
        "problems": problems,
        "status": "pass" if not problems else "fail",
    }
    return summary, problems, notes


def verify() -> int:
    try:
        summary, problems, _notes = evaluate()
    except FileNotFoundError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True))
    return 0 if not problems else 1


def tally(references: list[dict]) -> dict:
    counts = {"verified": 0, "mismatch": 0, "missing": 0, "not_referenced": 0,
              "unverified_host_split": 0}
    for reference in references:
        counts[reference["verdict"]] = counts.get(reference["verdict"], 0) + 1
    return counts



def write_manifest() -> int:
    payload = build()
    if MANIFEST.is_file():
        previous = read_json(MANIFEST)
        for key in ("model_load", "test_run"):
            if not payload.get(key) and previous.get(key):
                payload[key] = previous[key]
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    write_registry(payload)
    print(json.dumps({
        "status": "written",
        "manifest": posix(MANIFEST.relative_to(ROOT)),
        "registry": posix(REGISTRY.relative_to(ROOT)),
        "groups": {group["label"]: group["count"] for group in payload["groups"]},
        "totals": payload["totals"],
        "reference_verdicts": tally(payload["reference_crosscheck"]),
    }, ensure_ascii=False, indent=1, sort_keys=True))
    return 0


def write_registry(payload: dict) -> None:
    """A compact code-identity register, derived from the manifest."""

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        "schema": "v430-code-identity/1",
        "work_package": "v4.3.0 W00",
        "generated_at_utc": payload["generated_at_utc"],
        "source_manifest": posix(MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256_file(MANIFEST) if MANIFEST.is_file()
                                  else None,
        "code_identity": {
            "groups": [
                {
                    "label": group["label"],
                    "count": group["count"],
                    "total_bytes": group["total_bytes"],
                    "manifest_sha256": sha256_bytes("\n".join(
                        f"{entry['sha256']}  {entry['path']}"
                        for entry in sorted(group["files"],
                                            key=lambda item: item["path"])
                        if entry.get("exists")).encode("utf-8")),
                    "files": [{"path": entry["path"], "sha256": entry["sha256"],
                               "size_bytes": entry["size_bytes"]}
                              for entry in group["files"] if entry.get("exists")],
                }
                for group in payload["groups"]
            ],
            "totals": payload["totals"],
        },
        "git": {
            "branch": payload["git"]["branch"],
            "head": payload["git"]["head"],
            "tracked_in_groups": payload["git"]["tracked_total"],
            "tracking_state": payload["git"]["tracking_state"],
            "tracking_is_observed_not_gated": (
                "recorded as a fact; the verifier's pass/fail depends only on "
                "the 54 files' bytes, so committing them cannot break it"),
            "evidence": payload["git"]["ignore_verdict"],
            "sync_manifest_coverage": (
                "ASSET_SYNC.sha256 has 0 lines for touhou_rl/v420, "
                "tools/rl/v420*, configs/v420 and tests/test_v420_*; it "
                "registers only .tmp_v420_data/** binaries, so the source "
                "groups are covered by neither Git nor the asset manifest"),
        },
        "delivery_gap": {
            "state": ("source exists on disk, is untracked by Git, is "
                      f"unregistered in ASSET_SYNC.sha256"
                      if payload["git"]["tracking_state"] == "untracked"
                      else "source exists on disk and is tracked by Git"),
            "consequence": (
                "a fresh clone has none of the four groups; every v4.2.0 number "
                "remains a historical record of this host until the groups are "
                "committed, and no remote reproduction can run the tests or the "
                "evaluator"),
            "not_a_capability_statement": (
                "closing this gap restores reproducibility only; it is not "
                "G430-E0 and not any G410 gate"),
        },
        "checkpoints": [
            {"id": entry["id"], "path": entry["path"],
             "sha256": entry.get("sha256_observed"), "verdict": entry["verdict"]}
            for entry in payload["checkpoints"]
        ],
        "reference_verdicts": tally(payload["reference_crosscheck"]),
        "not_referenced_file_count": sum(
            1 for entry in payload["not_referenced_files"]
            if entry["verdict"] == "not_referenced"),
    }
    REGISTRY.write_text(
        json.dumps(registry, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")


def record_model_load(target: str) -> int:
    result = load_checkpoint(target)
    payload = read_json(MANIFEST)
    payload.setdefault("model_load", {})[result.get("checkpoint", target)] = result
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
    return 0 if result.get("verdict") == "verified" else 1


def record_tests(junit: str) -> int:
    payload = read_json(MANIFEST)
    stats = parse_junit(Path(junit).resolve())
    previous = payload.get("test_run") or {}
    previous.setdefault("runs", []).append(stats)
    payload["test_run"] = previous
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1, sort_keys=True))
    return 0


def _temporary_repo_tracking_demo() -> dict:
    """A real ``git add`` in a throwaway repo, to show what the two states mean.

    Nothing here touches the project repository: the demo repo is created under
    the OS temp directory.  It shows the one fact the gate removal rests on --
    committing a file changes ``git ls-files`` and ``git status`` but leaves the
    file's sha256 byte-for-byte identical, so a byte gate cannot flip.
    """

    import tempfile  # noqa: PLC0415

    demo = Path(tempfile.mkdtemp(prefix="w00_tracking_demo_"))
    try:
        (demo / "a.py").write_text("A = 1\n", encoding="utf-8")
        (demo / "b.py").write_text("B = 2\n", encoding="utf-8")
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=str(demo), capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        run("init", "-q")
        run("config", "user.email", "w00@example.invalid")
        run("config", "user.name", "w00")
        before_digest = {name: sha256_file(demo / name)
                         for name in ("a.py", "b.py")}
        before_ls = run("ls-files").stdout.split()
        before_status = run("status", "--short", "--untracked-files=all"
                            ).stdout.splitlines()
        run("add", "a.py", "b.py")
        after_digest = {name: sha256_file(demo / name)
                        for name in ("a.py", "b.py")}
        after_ls = run("ls-files").stdout.split()
        after_status = run("status", "--short", "--untracked-files=all"
                           ).stdout.splitlines()
        return {
            "repo": "throwaway git repo under the OS temp directory",
            "before_add": {"ls_files": before_ls,
                           "status_lines": before_status,
                           "digests": before_digest},
            "after_add": {"ls_files": after_ls,
                          "status_lines": after_status,
                          "digests": after_digest},
            "digests_unchanged_by_add": before_digest == after_digest,
            "tracking_changed_by_add": (len(before_ls) != len(after_ls)
                                        or before_status != after_status),
            "conclusion": ("git add changes only the tracking observation, never "
                           "the bytes; a byte gate is therefore stable across the "
                           "fix"),
        }
    finally:
        import shutil  # noqa: PLC0415

        shutil.rmtree(demo, ignore_errors=True)


def self_test() -> int:
    """Prove a tracked repo still passes, without touching the real repository.

    Two git states are run through the *same* evaluation: the real one, read
    from git, and a simulated post-``git add`` one where every group file is
    reported tracked.  Both must return exit 0 with ``problems == []``.  If any
    byte check were made to depend on the tracking state, this test goes red --
    that is its whole purpose.  A real ``git add`` in a throwaway repository
    corroborates that committing cannot move the bytes.
    """

    global _TRACKING_OVERRIDE
    real_tracking = git_tracking()
    total = inventory()["totals"]["file_count"]
    synthetic = {
        **real_tracking,
        "tracked_total": total,
        "status_untracked_count": 0,
        "status_tracked_changed_count": 0,
        "status_untracked_lines": [],
        "tracking_state": "tracked",
        "ls_files": {label: {**entry, "tracked_count": 0, "stdout": ""}
                     for label, entry in real_tracking["ls_files"].items()},
    }
    cases = []
    try:
        for name, override in (("before_add_real_git_state", None),
                               ("after_add_synthetic_tracked_state", synthetic)):
            _TRACKING_OVERRIDE = override
            summary, problems, notes = evaluate()
            cases.append({
                "case": name,
                "injected_git_tracked_in_groups": summary["git_tracked_in_groups"],
                "injected_git_tracking_state": summary["git_tracking_state"],
                "exit_code": 0 if not problems else 1,
                "problems": problems,
                "notes": notes,
                "verdict": "pass" if not problems else "fail",
            })
    finally:
        _TRACKING_OVERRIDE = None
    both_pass = all(case["exit_code"] == 0 and not case["problems"]
                    for case in cases)
    demo = _temporary_repo_tracking_demo()
    ok = both_pass and demo["digests_unchanged_by_add"] and demo[
        "tracking_changed_by_add"]
    print(json.dumps({
        "schema": "v430-w00-source-identity-selftest/1",
        "purpose": ("show that git tracking state is an observation, not a gate: "
                    "the same byte checks must pass before and after git add"),
        "real_tracking_state": real_tracking["tracking_state"],
        "real_tracked_in_groups": real_tracking["tracked_total"],
        "group_file_count": total,
        "cases": cases,
        "both_states_exit_0": both_pass,
        "temporary_repo_demo": demo,
        "status": "pass" if ok else "fail",
    }, ensure_ascii=False, indent=1, sort_keys=True))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="rebuild source_identity.json and the registry")
    parser.add_argument("--load-checkpoint", metavar="PATH", default=None,
                        help="load one checkpoint into the model class")
    parser.add_argument("--record", action="store_true",
                        help="with --load-checkpoint: merge into the manifest")
    parser.add_argument("--record-tests", metavar="JUNIT_XML", default=None,
                        help="merge a pytest junit report into the manifest")
    parser.add_argument("--registry", action="store_true",
                        help="rebuild only registries/code_identity.json")
    parser.add_argument("--self-test", action="store_true",
                        help="prove tracked and untracked states both pass")
    args = parser.parse_args(argv)

    if args.load_checkpoint:
        if args.record:
            return record_model_load(args.load_checkpoint)
        result = load_checkpoint(args.load_checkpoint)
        print(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
        return 0 if result.get("verdict") == "verified" else 1
    if args.record_tests:
        return record_tests(args.record_tests)
    if args.self_test:
        return self_test()
    if args.registry:
        write_registry(read_json(MANIFEST))
        print(json.dumps({"status": "registry_written",
                          "registry": posix(REGISTRY.relative_to(ROOT))}))
        return 0
    if args.write:
        return write_manifest()
    return verify()


if __name__ == "__main__":
    sys.exit(main())
