#!/usr/bin/env python3
"""v4.3.0 CLI.  W00 owns this file's read-only ``audit`` entry point only.

``EXECUTION_PLAN.md`` section 2 (W00) and ``DEVELOPMENT_GUIDE.md`` section 10 define the
interface.  Nothing but ``audit`` is implemented here:

* ``audit --config configs/v430/assets.json --out <new result.json>``
  re-measures every registered asset and every pin the registry declares, writes
  one result JSON and returns the exit code below.  It starts no engine, no
  game, no training and never writes anything except ``--out``.
* Every other v4.3.0 subcommand listed by ``PLAN_SPEC.json`` (``future_cli``) is
  registered with its documented argument shape and fails closed: exit 2 with a
  ``not_implemented`` notice on stderr and no filesystem effect.  A reserved
  subcommand never returns 0 and never fabricates a PASS result.

Exit codes (``PLAN_SPEC.json`` -> ``future_cli.exit_codes``)

====  =====================================================================
0     every check the command actually requested is really satisfied
2     inputs/config/environment invalid (missing config, bad schema, existing
      ``--out``, reserved subcommand)
3     the measurement was valid but the requested gate is not met (any
      missing / mismatch / unreachable, or a required hash that was skipped)
4     reserved for "incomplete because of resources or interruption"; audit
      itself never produces it
other non-zero: program error
====  =====================================================================

Asset verdict vocabulary (one row per registered check)

==========================  ===============================================
``present_verified``        present and every requested verification passed
``present_unhashed``        present, but identity is not content-hashed (a
                            directory, a live-mutated file, or an object
                            above ``max_full_hash_bytes``); never counted as
                            verified, always listed in ``remaining_gaps``
``present_mismatch``        present but a pinned hash/expectation differs
``missing``                 the path does not exist
``unreachable``             the measurement could not be taken at all (WSL
                            down, command not found, timeout, unreadable)
==========================  ===============================================

An unreachable Linux-side asset is never downgraded to ``missing`` or to a
pass.  ``--out`` is created exclusively; an existing file is refused with exit
2 and left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

RESULT_SCHEMA = "v430-audit/1"
ASSETS_SCHEMA = "v430-assets/1"

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_GATE = 3
EXIT_INCOMPLETE = 4

ST_VERIFIED = "present_verified"
ST_UNHASHED = "present_unhashed"
ST_MISMATCH = "present_mismatch"
ST_MISSING = "missing"
ST_UNREACHABLE = "unreachable"

STATUSES = (ST_VERIFIED, ST_UNHASHED, ST_MISMATCH, ST_MISSING, ST_UNREACHABLE)
GATE_FAIL_STATUSES = (ST_MISSING, ST_MISMATCH, ST_UNREACHABLE)

REQUIRE_HASH = "hash"
REQUIRE_PRESENCE = "presence"
REQUIREMENTS = (REQUIRE_HASH, REQUIRE_PRESENCE)

WINDOWS_FILE = "file"
WINDOWS_DIR = "dir"
WSL_FILE = "wsl_file"
WSL_DIR = "wsl_dir"
WSL_PYTHON = "wsl_python"
RUNTIME_PYTHON = "python"
RUNTIME_TORCH = "torch"
KINDS = (WINDOWS_FILE, WINDOWS_DIR, WSL_FILE, WSL_DIR, WSL_PYTHON, RUNTIME_PYTHON, RUNTIME_TORCH)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_LINE_RE = re.compile(r"^(?P<sha>[0-9a-f]{64})\s{2}(?P<path>.+)$")
PYTHON_VERSION_RE = re.compile(r"^Python\s+(?P<version>\d+\.\d+\.\d+)")
UNSAFE_WSL_PATH_CHARS = ('"', "'", "$", "`", "\\", "\n", "\r")

DEFAULT_MAX_FULL_HASH_BYTES = 2 * 1024 ** 3
DEFAULT_WSL_TIMEOUT = 300.0

#: Reserved subcommands: name -> (owner, purpose).  Nothing here is implemented.
RESERVED: dict[str, dict[str, str]] = {
    "tactical-pilot": {"owner": "W02", "purpose": "tactical counterfactual pilot"},
    "power-pilot": {"owner": "W02", "purpose": "POWER-01 pilot"},
    "prepare-data": {"owner": "W03", "purpose": "dataset index and lineage preparation"},
    "pretrain": {"owner": "W03", "purpose": "representation pretraining"},
    "behavior-clone": {"owner": "W03", "purpose": "behaviour cloning"},
    "micro-learn": {"owner": "W03", "purpose": "micro-segment learning"},
    "train": {"owner": "W04", "purpose": "curriculum / PPO training (also W07, W08)"},
    "evaluate": {"owner": "W04", "purpose": "unified evaluation entry"},
    "confirm": {"owner": "W04", "purpose": "frozen-candidate confirmation"},
    "windows-eval": {"owner": "W06", "purpose": "Windows latency/input evaluation"},
    "feedback-eval": {"owner": "W05", "purpose": "pixel feedback evaluation"},
    "adapt": {"owner": "W10", "purpose": "held-out adaptation (also W11 transfer)"},
    "settle": {"owner": "W12", "purpose": "independent evidence settlement"},
    "plan": {"owner": "W01", "purpose": "print dependencies and budget; starts nothing"},
}


class ConfigError(Exception):
    """Invalid inputs or configuration (exit 2)."""


class Unreachable(Exception):
    """An asset host or command could not be reached (status ``unreachable``)."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now().astimezone()


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _err(message: str) -> None:
    print(f"v430: {message}", file=sys.stderr)


def _repo_root() -> Path:
    """Repository root = the ancestor that contains AGENTS.md."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "AGENTS.md").is_file():
            return parent
    return here.parents[2]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _run_capture(argv: list[str], timeout: float, stdin: str | None = None) -> tuple[int, str, str]:
    """Run ``argv`` and capture text output; spawn/timeout failures are Unreachable."""
    try:
        completed = subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise Unreachable(f"command not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise Unreachable(f"timeout after {timeout:g}s: {' '.join(argv[:4])}") from exc
    except OSError as exc:
        raise Unreachable(f"cannot execute {argv[0]}: {exc}") from exc
    return completed.returncode, completed.stdout, completed.stderr


def _git(repo: Path, *args: str, timeout: float = 120.0) -> tuple[int, str, str]:
    try:
        return _run_capture(["git", "-C", str(repo), *args], timeout)
    except Unreachable as exc:
        return -1, "", str(exc)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _need_str(container: dict, key: str, where: str) -> str:
    value = container.get(key)
    _need(isinstance(value, str) and value.strip() != "", f"{where}: '{key}' must be a non-empty string")
    return value


def _validate_wsl_path(value: Any, where: str) -> str:
    _need(isinstance(value, str) and value.startswith("/"), f"{where}: wsl path must be an absolute POSIX path")
    for char in UNSAFE_WSL_PATH_CHARS:
        _need(char not in value, f"{where}: wsl path must not contain {char!r}")
    return value


def load_config(path: Path) -> dict:
    """Parse and structurally validate ``configs/v430/assets.json`` (exit 2 on failure)."""
    _need(path.is_file(), f"config not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config {path} is not valid JSON: {exc}") from exc
    _need(isinstance(raw, dict), "config root must be a JSON object")
    _need(raw.get("schema") == ASSETS_SCHEMA, f"config schema must be {ASSETS_SCHEMA!r}, got {raw.get('schema')!r}")

    for key in ("experiment_id", "spec_revision", "frozen_at", "assets"):
        _need(key in raw, f"config is missing required field '{key}'")
    _need_str(raw, "experiment_id", "config")
    _need_str(raw, "spec_revision", "config")
    frozen_at = _need_str(raw, "frozen_at", "config")
    try:
        frozen = datetime.fromisoformat(frozen_at)
    except ValueError as exc:
        raise ConfigError(f"frozen_at is not ISO-8601: {frozen_at!r}") from exc
    _need(frozen.tzinfo is not None, "frozen_at must carry a timezone offset")

    policy = raw.get("audit_policy", {})
    _need(isinstance(policy, dict), "audit_policy must be an object")
    max_bytes = policy.get("max_full_hash_bytes", DEFAULT_MAX_FULL_HASH_BYTES)
    _need(isinstance(max_bytes, int) and not isinstance(max_bytes, bool) and max_bytes > 0,
          "audit_policy.max_full_hash_bytes must be a positive integer")
    wsl_timeout = policy.get("wsl_timeout_seconds", DEFAULT_WSL_TIMEOUT)
    _need(isinstance(wsl_timeout, (int, float)) and not isinstance(wsl_timeout, bool) and wsl_timeout > 0,
          "audit_policy.wsl_timeout_seconds must be positive")

    wsl_block = raw.get("wsl", {})
    _need(isinstance(wsl_block, dict), "wsl must be an object")
    default_distro = wsl_block.get("distro")
    if default_distro is not None:
        _need(isinstance(default_distro, str) and default_distro.strip() != "", "wsl.distro must be a string")

    assets = raw["assets"]
    _need(isinstance(assets, list) and assets, "config.assets must be a non-empty list")
    seen: set[str] = set()
    for index, asset in enumerate(assets):
        where = f"assets[{index}]"
        _need(isinstance(asset, dict), f"{where} must be an object")
        ident = _need_str(asset, "id", where)
        _need(ident not in seen, f"duplicate asset id: {ident!r}")
        seen.add(ident)
        kind = _need_str(asset, "kind", f"{where} ({ident})")
        _need(kind in KINDS, f"{where} ({ident}): unknown kind {kind!r}; expected one of {KINDS}")
        require = asset.get("require")
        _need(require in REQUIREMENTS, f"{where} ({ident}): require must be one of {REQUIREMENTS}")
        if "expect" in asset:
            _need(isinstance(asset["expect"], dict), f"{where} ({ident}): expect must be an object")
        expected_sha = asset.get("sha256_expected")
        if expected_sha is not None:
            _need(isinstance(expected_sha, str) and SHA256_RE.match(expected_sha) is not None,
                  f"{where} ({ident}): sha256_expected must be 64 lowercase hex characters")
        if require == REQUIRE_HASH:
            _need(kind in (WINDOWS_FILE, WSL_FILE),
                  f"{where} ({ident}): only file kinds can require a content hash, got kind {kind!r}")
            _need(expected_sha is not None, f"{where} ({ident}): require=hash needs sha256_expected")
        if "hash_policy" in asset:
            _need(asset["hash_policy"] == "stat_only",
                  f"{where} ({ident}): hash_policy must be 'stat_only' when present")
        if "live_mutated" in asset:
            _need(isinstance(asset["live_mutated"], bool),
                  f"{where} ({ident}): live_mutated must be a boolean")
        if kind in (WINDOWS_FILE, WINDOWS_DIR):
            _need_str(asset, "path", f"{where} ({ident})")
            base = asset.get("path_base", "repo")
            _need(base in ("repo", "absolute"), f"{where} ({ident}): path_base must be 'repo' or 'absolute'")
            if base == "absolute":
                _need(Path(asset["path"]).is_absolute(), f"{where} ({ident}): path_base=absolute needs an absolute path")
        elif kind == RUNTIME_PYTHON:
            _need_str(asset, "path", f"{where} ({ident})")
            _need(Path(asset["path"]).is_absolute(), f"{where} ({ident}): the registered interpreter path must be absolute")
        elif kind in (WSL_FILE, WSL_DIR, WSL_PYTHON):
            wsl = asset.get("wsl")
            _need(isinstance(wsl, dict), f"{where} ({ident}): wsl assets need a wsl {{distro, path}} block")
            distro = wsl.get("distro", default_distro)
            _need(isinstance(distro, str) and distro.strip() != "", f"{where} ({ident}): wsl.distro missing")
            _need(default_distro in (None, distro),
                  f"{where} ({ident}): wsl.distro {distro!r} differs from registry distro {default_distro!r}")
            _validate_wsl_path(wsl.get("path"), f"{where} ({ident})")

    plan_block = raw.get("plan_spec")
    if plan_block is not None:
        _need(isinstance(plan_block, dict), "plan_spec must be an object")
        _need_str(plan_block, "path", "plan_spec")
        _need_str(plan_block, "schema_expected", "plan_spec")
        revision = _need_str(plan_block, "revision_expected", "plan_spec")
        _need(revision == raw["spec_revision"],
              f"plan_spec.revision_expected {revision!r} must equal spec_revision {raw['spec_revision']!r}")

    manifest_block = raw.get("sync_manifest")
    if manifest_block is not None:
        _need(isinstance(manifest_block, dict), "sync_manifest must be an object")
        _need_str(manifest_block, "path", "sync_manifest")
        pinned = _need_str(manifest_block, "sha256_expected", "sync_manifest")
        _need(SHA256_RE.match(pinned) is not None, "sync_manifest.sha256_expected must be 64 lowercase hex characters")
        count = manifest_block.get("entry_count_expected")
        _need(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
              "sync_manifest.entry_count_expected must be a non-negative integer")
        external = manifest_block.get("external_entries", {})
        _need(isinstance(external, dict), "sync_manifest.external_entries must be an object")
        for entry_path, asset_id in external.items():
            _need(isinstance(entry_path, str) and entry_path.startswith("/"),
                  "sync_manifest.external_entries keys must be absolute POSIX paths")
            _need(asset_id in seen,
                  f"sync_manifest.external_entries maps {entry_path!r} to unknown asset {asset_id!r}")

    raw["_audit_policy"] = {"max_full_hash_bytes": max_bytes, "wsl_timeout_seconds": float(wsl_timeout)}
    raw["_wsl_distro"] = default_distro or "Ubuntu"
    return raw


# --------------------------------------------------------------------------- #
# WSL probe
# --------------------------------------------------------------------------- #
class WslProbe:
    """One distro binding; every Linux-side asset goes through it.

    ``wsl.exe -- <command>`` runs the command through the distribution's default
    shell, which expands ``$name`` before the command sees it.  Two transports
    avoid that: ``-e/--exec bash -lc <script>`` (no shell in between) and
    ``-e bash -s`` with the script on stdin.  The first that answers the probe
    marker is used for the rest of the run; if neither answers, the distro is
    ``unreachable`` and no Linux-side asset is downgraded to ``missing``.
    """

    TRANSPORT_EXEC = "exec"
    TRANSPORT_STDIN = "stdin"
    TRANSPORTS = (TRANSPORT_EXEC, TRANSPORT_STDIN)

    def __init__(self, distro: str, timeout: float):
        self.distro = distro
        self.timeout = timeout
        self.available = False
        self.reason = "not probed"
        self.probe_exit_code: int | None = None
        self.commands = 0
        self.transport: str = self.TRANSPORT_EXEC
        self.transport_used: str | None = None
        self.attempts: list[dict] = []

    def _argv(self, transport: str, script: str) -> list[str]:
        if transport == self.TRANSPORT_STDIN:
            return ["wsl.exe", "-d", self.distro, "-e", "bash", "-s"]
        return ["wsl.exe", "-d", self.distro, "-e", "bash", "-lc", script]

    def _invoke(self, transport: str, script: str) -> tuple[int, str, str]:
        stdin = script if transport == self.TRANSPORT_STDIN else None
        return _run_capture(self._argv(transport, script), self.timeout, stdin=stdin)

    def probe(self) -> None:
        probe_timeout = min(self.timeout, 60.0)
        saved_timeout = self.timeout
        self.timeout = probe_timeout
        try:
            for transport in self.TRANSPORTS:
                try:
                    code, out, err = self._invoke(transport, "echo v430-wsl-probe-ok")
                except Unreachable as exc:
                    self.attempts.append({"transport": transport, "error": str(exc)})
                    self.reason = str(exc)
                    if "command not found" in str(exc):
                        break
                    continue
                self.probe_exit_code = code
                if code == 0 and "v430-wsl-probe-ok" in out:
                    self.available = True
                    self.transport = transport
                    self.transport_used = transport
                    self.reason = "ok"
                    self.attempts.append({"transport": transport, "exit_code": 0, "ok": True})
                    return
                detail = (err or out or "").strip().replace("\r", " ").replace("\n", " ")[:300]
                self.attempts.append({"transport": transport, "exit_code": code,
                                      "error": detail, "ok": False})
                self.reason = f"{transport} probe exit {code}: {detail}"
        finally:
            self.timeout = saved_timeout
        self.available = False
        self.transport = self.TRANSPORT_EXEC

    def run_script(self, script: str) -> str:
        if not self.available:
            raise Unreachable(f"WSL distro {self.distro!r} unavailable: {self.reason}")
        self.commands += 1
        code, out, err = self._invoke(self.transport, script)
        if code != 0:
            detail = (err or out or "").strip().replace("\r", " ").replace("\n", " ")
            raise Unreachable(f"wsl command exit {code}: {detail[:300]}")
        return out

    def identity(self) -> dict:
        return {
            "distro": self.distro,
            "available": self.available,
            "transport": self.transport_used,
            "probe_exit_code": self.probe_exit_code,
            "probe_reason": self.reason,
            "probe_attempts": self.attempts,
            "commands": self.commands,
        }


# --------------------------------------------------------------------------- #
# the audit
# --------------------------------------------------------------------------- #
class AuditRun:
    def __init__(self, cfg: dict, cfg_path: Path, repo: Path, out: Path, started: datetime, argv: list[str]):
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.repo = repo
        self.out = out
        self.started = started
        self.argv = argv
        self.checks: list[dict] = []
        self.gaps: list[str] = []
        self.notes: list[str] = []
        self.bytes_hashed = 0
        self.files_hashed = 0
        self.identities: dict[str, Any] = {}
        policy = cfg["_audit_policy"]
        self.max_full_hash_bytes: int = policy["max_full_hash_bytes"]
        self.wsl = WslProbe(cfg["_wsl_distro"], policy["wsl_timeout_seconds"])
        self.wsl_batch_error: str | None = None
        self.manifest_present: int | None = None
        self.manifest_covered: int | None = None
        self.config_sha256 = sha256_file(cfg_path)

    # -- check scaffolding ------------------------------------------------- #
    def _new_check(self, *, ident: str, category: str, kind: str, require: str,
                   description: str, location: dict) -> dict:
        return {
            "id": ident,
            "category": category,
            "kind": kind,
            "require": require,
            "description": description,
            "location": location,
            "status": None,
            "requested_check_met": False,
            "measurement": None,
            "sha256_expected": None,
            "sha256_actual": None,
            "size_bytes": None,
            "expectations": {},
            "observed": {},
            "reason": None,
        }

    def _finish(self, check: dict, status: str, *, met: bool, reason: str | None = None,
                measurement: str | None = None) -> dict:
        if status not in STATUSES:
            raise RuntimeError(f"internal: unknown status {status!r}")
        check["status"] = status
        check["requested_check_met"] = bool(met)
        check["reason"] = reason
        check["measurement"] = measurement
        self.checks.append(check)
        return check

    def _expect(self, check: dict, key: str, expected: Any, actual: Any, *, mode: str = "exact") -> bool:
        if mode == "bool":
            ok = bool(actual) == bool(expected)
        else:
            ok = actual == expected
        check["expectations"][key] = {"expected": expected, "actual": actual, "ok": ok}
        return ok

    @staticmethod
    def _failed_expectations(check: dict) -> list[str]:
        return [key for key, row in check["expectations"].items() if not row["ok"]]

    def _resolve_windows_path(self, asset: dict) -> Path:
        if asset.get("path_base", "repo") == "absolute":
            return Path(asset["path"])
        return self.repo / asset["path"]

    # -- config pins ------------------------------------------------------- #
    def check_plan_spec(self) -> None:
        block = self.cfg.get("plan_spec")
        if not block:
            self.gaps.append("plan_spec is not bound by this registry: spec_revision is not pinned to a PLAN_SPEC.json hash.")
            return
        path = self.repo / block["path"]
        check = self._new_check(
            ident="plan_spec", category="config", kind="json-file", require="pin",
            description="PLAN_SPEC.json must carry the pinned plan schema and revision.",
            location={"path": block["path"], "resolved_absolute": str(path)},
        )
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            self._finish(check, ST_MISSING, met=False, reason="PLAN_SPEC.json does not exist")
            return
        except OSError as exc:
            self._finish(check, ST_UNREACHABLE, met=False, reason=f"{type(exc).__name__}: {exc}")
            return
        self.bytes_hashed += len(payload)
        digest = sha256_bytes(payload)
        check["sha256_actual"] = digest
        check["size_bytes"] = len(payload)
        try:
            spec = json.loads(payload.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._finish(check, ST_MISMATCH, met=False, reason=f"PLAN_SPEC.json is not valid JSON: {exc}",
                         measurement="bytes read from the repository file")
            return
        ok = self._expect(check, "schema", block["schema_expected"], spec.get("schema"))
        ok = self._expect(check, "revision", block["revision_expected"], spec.get("revision")) and ok
        check["observed"] = {"schema": spec.get("schema"), "revision": spec.get("revision")}
        self.identities["plan_spec"] = {
            "path": block["path"],
            "resolved_absolute": str(path),
            "sha256": digest,
            "schema": spec.get("schema"),
            "revision": spec.get("revision"),
        }
        if ok:
            self._finish(check, ST_VERIFIED, met=True, measurement="sha256 of the repository file + JSON fields")
        else:
            failed = ", ".join(self._failed_expectations(check))
            self._finish(check, ST_MISMATCH, met=False, reason=f"PLAN_SPEC.json disagrees on: {failed}",
                         measurement="sha256 of the repository file + JSON fields")

    def check_sync_manifest(self) -> None:
        block = self.cfg.get("sync_manifest")
        if not block:
            self.gaps.append("sync_manifest is not bound by this registry: ASSET_SYNC.sha256 is not pinned.")
            return
        path = self.repo / block["path"]
        check = self._new_check(
            ident="sync_manifest", category="config", kind="hash-manifest", require="pin",
            description="ASSET_SYNC.sha256 must be byte-identical to the pinned authority list.",
            location={"path": block["path"], "resolved_absolute": str(path)},
        )
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            self.manifest_present = None
            self._finish(check, ST_MISSING, met=False, reason="ASSET_SYNC.sha256 does not exist")
            return
        except OSError as exc:
            self._finish(check, ST_UNREACHABLE, met=False, reason=f"{type(exc).__name__}: {exc}")
            return
        self.bytes_hashed += len(payload)
        digest = sha256_bytes(payload)
        check["sha256_expected"] = block["sha256_expected"]
        check["sha256_actual"] = digest
        check["size_bytes"] = len(payload)
        self.identities["sync_manifest"] = {
            "path": block["path"],
            "resolved_absolute": str(path),
            "sha256": digest,
            "entry_count_expected": block["entry_count_expected"],
        }
        if digest != block["sha256_expected"]:
            self._finish(check, ST_MISMATCH, met=False,
                         reason="ASSET_SYNC.sha256 differs from the pinned hash; the registry is stale",
                         measurement="sha256 of the repository file")
            self.gaps.append("ASSET_SYNC.sha256 no longer matches the pinned hash, so the per-asset hashes below may be stale.")
            return
        entries, error = self._parse_manifest(payload, block)
        if entries is not None:
            self.manifest_present = len(entries)
        if error is not None:
            self._finish(check, ST_MISMATCH, met=False, reason=error,
                         measurement="sha256 of the repository file")
            return
        self._finish(check, ST_VERIFIED, met=True, measurement="sha256 of the repository file")
        self._check_manifest_coverage(entries, block)

    def _parse_manifest(self, payload: bytes, block: dict) -> tuple[list[tuple[str, str]] | None, str | None]:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            return None, f"ASSET_SYNC.sha256 is not UTF-8: {exc}"
        entries: list[tuple[str, str]] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            match = MANIFEST_LINE_RE.match(line)
            if not match:
                return None, f"ASSET_SYNC.sha256 line {number} is not '<sha256>  <path>': {line[:120]!r}"
            entries.append((match.group("sha"), match.group("path")))
        expected = block["entry_count_expected"]
        if len(entries) != expected:
            return entries, f"ASSET_SYNC.sha256 has {len(entries)} entries, registry expects {expected}"
        return entries, None

    def _check_manifest_coverage(self, entries: list[tuple[str, str]], block: dict) -> None:
        external = block.get("external_entries", {})
        by_manifest_path = {
            asset["manifest"]["entry_path"]: asset
            for asset in self.cfg["assets"]
            if isinstance(asset.get("manifest"), dict) and "entry_path" in asset["manifest"]
        }
        missing_rows: list[str] = []
        covered = 0
        for sha, entry_path in entries:
            if entry_path in external:
                asset_id = external[entry_path]
                asset = next((row for row in self.cfg["assets"] if row["id"] == asset_id), None)
                if asset is None:
                    missing_rows.append(f"{entry_path} -> unknown asset {asset_id}")
                elif asset.get("sha256_expected") != sha:
                    missing_rows.append(f"{entry_path} -> {asset_id} pins a different sha256")
                else:
                    covered += 1
                continue
            asset = by_manifest_path.get(entry_path)
            if asset is None:
                missing_rows.append(f"{entry_path} is not registered by any asset")
            elif asset.get("sha256_expected") != sha:
                missing_rows.append(f"{entry_path} -> {asset['id']} pins a different sha256")
            else:
                covered += 1
        manifest_paths = {entry_path for _sha, entry_path in entries}
        stray = sorted(set(by_manifest_path) - manifest_paths - set(external))
        if stray:
            missing_rows.append(f"registry rows without a manifest entry: {', '.join(stray[:5])}")
        self.manifest_covered = covered
        check = self._new_check(
            ident="sync_manifest:coverage", category="config", kind="hash-manifest", require="pin",
            description="Every ASSET_SYNC.sha256 entry must be verified by exactly one registered asset row.",
            location={"path": block["path"], "entry_count": len(entries)},
        )
        check["observed"] = {"entries": len(entries), "covered": covered, "uncovered": missing_rows[:10]}
        if not missing_rows:
            self._finish(check, ST_VERIFIED, met=True,
                         measurement="manifest lines joined to registered rows by path and sha256")
        else:
            self._finish(check, ST_MISMATCH, met=False,
                         reason=f"{len(missing_rows)} manifest entries are not covered by the registry",
                         measurement="manifest lines joined to registered rows by path and sha256")

    # -- asset checks ------------------------------------------------------ #
    def check_assets(self) -> None:
        engine_rows = [asset for asset in self.cfg["assets"] if asset["kind"] == WSL_FILE]
        wsl_assets = [asset for asset in self.cfg["assets"]
                      if asset["kind"] in (WSL_FILE, WSL_DIR, WSL_PYTHON)]
        if wsl_assets:
            self.wsl.probe()
            if not self.wsl.available:
                self.gaps.append(
                    f"WSL distro {self.wsl.distro!r} is unreachable: {self.wsl.reason}. "
                    "No Linux-side asset could be measured; nothing was downgraded to 'missing' or to a pass."
                )
        wsl_files = self._probe_wsl_files(engine_rows)
        for asset in self.cfg["assets"]:
            kind = asset["kind"]
            if kind in (WINDOWS_FILE, WINDOWS_DIR):
                self._check_windows(asset)
            elif kind == RUNTIME_PYTHON:
                self._check_python(asset)
            elif kind == RUNTIME_TORCH:
                self._check_torch(asset)
            elif kind == WSL_FILE:
                self._check_wsl_file(asset, wsl_files)
            elif kind == WSL_DIR:
                self._check_wsl_dir(asset)
            elif kind == WSL_PYTHON:
                self._check_wsl_python(asset)
            else:  # pragma: no cover - KINDS is validated at load time
                raise RuntimeError(f"internal: unhandled kind {kind!r}")

    def _wsl_row(self, asset: dict) -> dict:
        wsl = asset["wsl"]
        return {"wsl_distro": wsl.get("distro", self.cfg["_wsl_distro"]), "wsl_path": wsl["path"]}

    def _probe_wsl_files(self, assets: list[dict]) -> dict[str, dict]:
        if not assets or not self.wsl.available:
            return {}
        paths = [asset["wsl"]["path"] for asset in assets]
        quoted = " ".join(f'"{path}"' for path in paths)
        script = (
            "for f in " + quoted + "; do "
            'if [ -f "$f" ]; then '
            'echo V430-FILE $(stat -c %s -- "$f") $(sha256sum -- "$f" | cut -c1-64) "$f"; '
            'else echo V430-MISSING "$f"; fi; done'
        )
        try:
            output = self.wsl.run_script(script)
        except Unreachable as exc:
            self.wsl_batch_error = str(exc)
            self.gaps.append(f"WSL file hashing failed: {exc}")
            return {}
        rows: dict[str, dict] = {}
        for line in output.splitlines():
            line = line.rstrip("\r")
            if line.startswith("V430-FILE "):
                parts = line.split(" ", 3)
                if len(parts) != 4:
                    continue
                size_text, digest, path = parts[1], parts[2], parts[3]
                if SHA256_RE.match(digest) is None or not size_text.isdigit():
                    rows[path] = {"error": f"unexpected sha256sum output: {line[:160]}"}
                else:
                    rows[path] = {"size": int(size_text), "sha256": digest}
            elif line.startswith("V430-MISSING "):
                rows[line[len("V430-MISSING "):]] = {"missing": True}
        for path in paths:
            rows.setdefault(path, {"error": "no output line for this path"})
        return rows

    def _check_windows(self, asset: dict) -> None:
        is_dir_kind = asset["kind"] == WINDOWS_DIR
        resolved = self._resolve_windows_path(asset)
        check = self._new_check(
            ident=asset["id"], category="asset", kind=asset["kind"], require=asset["require"],
            description=asset.get("description", ""),
            location={"windows_path": asset["path"], "path_base": asset.get("path_base", "repo"),
                      "resolved_absolute": str(resolved)},
        )
        try:
            info = resolved.stat()
        except FileNotFoundError:
            self._finish(check, ST_MISSING, met=False, reason="path does not exist")
            return
        except OSError as exc:
            self._finish(check, ST_UNREACHABLE, met=False, reason=f"{type(exc).__name__}: {exc}")
            return
        actual_dir = stat.S_ISDIR(info.st_mode)
        if is_dir_kind:
            if not actual_dir:
                self._finish(check, ST_MISMATCH, met=False, reason="expected a directory, found a non-directory")
                return
            self._check_directory_contents(asset, check, resolved)
            return
        if not stat.S_ISREG(info.st_mode):
            self._finish(check, ST_MISMATCH, met=False, reason="expected a regular file, found something else")
            return
        self._check_file_hash(asset, check, resolved, info.st_size, reader="hashlib.sha256 over the Windows file")

    def _check_directory_contents(self, asset: dict, check: dict, resolved: Path) -> None:
        try:
            entries = sorted(os.listdir(resolved))
        except OSError as exc:
            self._finish(check, ST_UNREACHABLE, met=False, reason=f"cannot list directory: {exc}")
            return
        check["observed"] = {"entry_count": len(entries), "entries": entries[:50]}
        expected = (asset.get("expect") or {}).get("contains", [])
        present = set(entries)
        ok = True
        for name in expected:
            ok = self._expect(check, f"contains:{name}", True, name in present) and ok
        if ok:
            self._finish(check, ST_UNHASHED, met=True,
                         reason="directory; contents are registered by layout, not by content hash",
                         measurement="directory listing")
        else:
            self._finish(check, ST_MISMATCH, met=False,
                         reason=f"missing expected entries: {', '.join(self._failed_expectations(check))}",
                         measurement="directory listing")

    def _check_file_hash(self, asset: dict, check: dict, resolved: Path, size: int,
                         *, reader: str, observed_sha: str | None = None) -> None:
        expected = asset.get("sha256_expected")
        check["sha256_expected"] = expected
        check["size_bytes"] = size
        if expected is None:
            self._finish(check, ST_UNHASHED, met=asset["require"] == REQUIRE_PRESENCE,
                         reason="no sha256_expected registered (presence-only row)",
                         measurement="stat only")
            return
        if observed_sha is None and size > self.max_full_hash_bytes and asset.get("hash_policy") != "stat_only":
            self._finish(check, ST_UNHASHED, met=False,
                         reason=(f"{size} bytes exceeds max_full_hash_bytes {self.max_full_hash_bytes}; "
                                 "full sha256 was not computed, so the pin cannot be checked"),
                         measurement="stat only")
            return
        if observed_sha is None and asset.get("hash_policy") == "stat_only" and size > self.max_full_hash_bytes:
            self._finish(check, ST_UNHASHED, met=True,
                         reason=(f"{size} bytes exceeds max_full_hash_bytes {self.max_full_hash_bytes}; "
                                 "hash_policy=stat_only accepts presence without a content hash"),
                         measurement="stat only")
            return
        if observed_sha is None:
            try:
                observed_sha = sha256_file(resolved)
            except OSError as exc:
                self._finish(check, ST_UNREACHABLE, met=False, reason=f"cannot read file: {exc}")
                return
            self.bytes_hashed += size
            self.files_hashed += 1
        check["sha256_actual"] = observed_sha
        ok = self._expect(check, "sha256", expected, observed_sha)
        if ok:
            self._finish(check, ST_VERIFIED, met=True, measurement=reader)
        else:
            self._finish(check, ST_MISMATCH, met=False,
                         reason="sha256 differs from the pinned value", measurement=reader)

    def _check_python(self, asset: dict) -> None:
        check = self._new_check(
            ident=asset["id"], category="asset", kind=asset["kind"], require=asset["require"],
            description=asset.get("description", ""),
            location={"registered_executable": asset["path"], "running_executable": sys.executable},
        )
        version = platform.python_version()
        implementation = platform.python_implementation()
        check["observed"] = {
            "executable": sys.executable,
            "resolved_path": str(Path(sys.executable).resolve()),
            "version": version,
            "implementation": implementation,
            "platform": platform.platform(),
        }
        expected_exe = Path(asset["path"])
        same = os.path.normcase(str(Path(sys.executable).resolve())) == os.path.normcase(str(expected_exe.resolve()))
        ok = self._expect(check, "executable", str(expected_exe.resolve()), str(Path(sys.executable).resolve()))
        expect = asset.get("expect") or {}
        for key in ("version", "implementation"):
            if key in expect:
                actual = version if key == "version" else implementation
                ok = self._expect(check, key, expect[key], actual) and ok
        if ok:
            self._finish(check, ST_VERIFIED, met=True, measurement="sys.executable + platform probes")
        else:
            reason = "the audit is not running under the registered interpreter" if not same else \
                     f"disagrees on: {', '.join(self._failed_expectations(check))}"
            self._finish(check, ST_MISMATCH, met=False, reason=reason,
                         measurement="sys.executable + platform probes")

    def _check_torch(self, asset: dict) -> None:
        check = self._new_check(
            ident=asset["id"], category="asset", kind=asset["kind"], require=asset["require"],
            description=asset.get("description", ""), location={"python": sys.executable},
        )
        try:
            import torch  # noqa: PLC0415 - deliberately deferred and optional
        except Exception as exc:  # noqa: BLE001 - any import failure means "cannot measure"
            self._finish(check, ST_UNREACHABLE, met=False, reason=f"import torch failed: {type(exc).__name__}: {exc}")
            return
        cuda_available = bool(torch.cuda.is_available())
        observed = {
            "version": str(torch.__version__),
            "file": str(getattr(torch, "__file__", "")),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": cuda_available,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        }
        if cuda_available:
            try:
                observed["device_name"] = torch.cuda.get_device_name(0)
            except Exception as exc:  # noqa: BLE001
                observed["device_name_error"] = f"{type(exc).__name__}: {exc}"
        check["observed"] = observed
        expect = asset.get("expect") or {}
        ok = True
        for key, mode in (("version", "exact"), ("cuda_version", "exact"), ("cuda_available", "bool")):
            if key in expect:
                ok = self._expect(check, key, expect[key], observed[key], mode=mode) and ok
        if ok:
            self._finish(check, ST_VERIFIED, met=True, measurement="import torch + torch.cuda probes")
        else:
            self._finish(check, ST_MISMATCH, met=False,
                         reason=f"disagrees on: {', '.join(self._failed_expectations(check))}",
                         measurement="import torch + torch.cuda probes")

    def _check_wsl_file(self, asset: dict, rows: dict[str, dict]) -> None:
        wsl = asset["wsl"]
        check = self._new_check(
            ident=asset["id"], category="asset", kind=asset["kind"], require=asset["require"],
            description=asset.get("description", ""), location=self._wsl_row(asset),
        )
        if not self.wsl.available:
            self._finish(check, ST_UNREACHABLE, met=False,
                         reason=f"WSL distro {wsl.get('distro', self.cfg['_wsl_distro'])!r} unavailable: {self.wsl.reason}")
            return
        row = rows.get(wsl["path"])
        if row is None or "error" in row:
            reason = (row or {}).get("error") or self.wsl_batch_error or "no measurement returned"
            self._finish(check, ST_UNREACHABLE, met=False, reason=reason)
            return
        if row.get("missing"):
            self._finish(check, ST_MISSING, met=False, reason="path does not exist inside the distro")
            return
        self._check_file_hash(asset, check, Path(wsl["path"]), int(row["size"]),
                              reader="wsl sha256sum over the Linux-side file",
                              observed_sha=row["sha256"])

    def _check_wsl_dir(self, asset: dict) -> None:
        wsl = asset["wsl"]
        check = self._new_check(
            ident=asset["id"], category="asset", kind=asset["kind"], require=asset["require"],
            description=asset.get("description", ""), location=self._wsl_row(asset),
        )
        if not self.wsl.available:
            self._finish(check, ST_UNREACHABLE, met=False,
                         reason=f"WSL distro {wsl.get('distro', self.cfg['_wsl_distro'])!r} unavailable: {self.wsl.reason}")
            return
        path = wsl["path"]
        script = (
            f'd="{path}"; if [ -d "$d" ]; then echo V430-DIR "$d"; '
            "ls -1A -- \"$d\" | sed 's/^/V430-ENTRY /'; "
            'else echo V430-MISSING "$d"; fi'
        )
        try:
            output = self.wsl.run_script(script)
        except Unreachable as exc:
            self._finish(check, ST_UNREACHABLE, met=False, reason=str(exc))
            return
        lines = [line.rstrip("\r") for line in output.splitlines()]
        if any(line.startswith("V430-MISSING ") for line in lines):
            self._finish(check, ST_MISSING, met=False, reason="directory does not exist inside the distro")
            return
        if not any(line.startswith("V430-DIR ") for line in lines):
            self._finish(check, ST_UNREACHABLE, met=False,
                         reason=f"unexpected probe output: {output.strip()[:200]!r}")
            return
        entries = sorted(line[len("V430-ENTRY "):] for line in lines if line.startswith("V430-ENTRY "))
        check["observed"] = {"entry_count": len(entries), "entries": entries[:50]}
        expected = (asset.get("expect") or {}).get("contains", [])
        present = set(entries)
        ok = True
        for name in expected:
            ok = self._expect(check, f"contains:{name}", True, name in present) and ok
        if ok:
            self._finish(check, ST_UNHASHED, met=True,
                         reason="directory; layout registered, contents not content-hashed (live game files are mutated)",
                         measurement="wsl directory listing")
        else:
            self._finish(check, ST_MISMATCH, met=False,
                         reason=f"missing expected entries: {', '.join(self._failed_expectations(check))}",
                         measurement="wsl directory listing")

    def _check_wsl_python(self, asset: dict) -> None:
        wsl = asset["wsl"]
        path = wsl["path"]
        check = self._new_check(
            ident=asset["id"], category="asset", kind=asset["kind"], require=asset["require"],
            description=asset.get("description", ""), location=self._wsl_row(asset),
        )
        if not self.wsl.available:
            self._finish(check, ST_UNREACHABLE, met=False,
                         reason=f"WSL distro {wsl.get('distro', self.cfg['_wsl_distro'])!r} unavailable: {self.wsl.reason}")
            return
        script = f'p="{path}"; if [ -x "$p" ]; then "$p" --version 2>&1; else echo V430-MISSING "$p"; fi'
        try:
            output = self.wsl.run_script(script)
        except Unreachable as exc:
            self._finish(check, ST_UNREACHABLE, met=False, reason=str(exc))
            return
        if f"V430-MISSING {path}" in output:
            self._finish(check, ST_MISSING, met=False, reason="interpreter does not exist inside the distro")
            return
        match = None
        for line in output.splitlines():
            match = PYTHON_VERSION_RE.match(line.strip())
            if match:
                break
        if match is None:
            self._finish(check, ST_MISMATCH, met=False,
                         reason=f"cannot parse a Python version from: {output.strip()[:200]!r}")
            return
        version = match.group("version")
        check["observed"] = {"version": version, "raw": output.strip()[:200]}
        expect = asset.get("expect") or {}
        ok = True
        if "version" in expect:
            ok = self._expect(check, "version", expect["version"], version)
        if ok:
            self._finish(check, ST_VERIFIED, met=True, measurement="wsl <interpreter> --version")
        else:
            self._finish(check, ST_MISMATCH, met=False,
                         reason=f"disagrees on: {', '.join(self._failed_expectations(check))}",
                         measurement="wsl <interpreter> --version")

    # -- identities -------------------------------------------------------- #
    def collect_source_identity(self) -> None:
        tool_path = Path(__file__).resolve()
        tool_sha = sha256_file(tool_path)
        source: dict[str, Any] = {
            "repository_root": str(self.repo),
            "tool": {"path": "tools/rl/v430.py", "sha256": tool_sha,
                     "size_bytes": tool_path.stat().st_size},
            "config": {"path": str(self.cfg_path), "sha256": self.config_sha256},
        }
        code, out, err = _git(self.repo, "rev-parse", "HEAD")
        if code != 0:
            source["git"] = {"available": False, "reason": (err or out).strip()[:300] or "git rev-parse failed"}
        else:
            head = out.strip()
            branch_code, branch_out, _ = _git(self.repo, "rev-parse", "--abbrev-ref", "HEAD")
            status_code, status_out, status_err = _git(self.repo, "status", "--porcelain", "--untracked-files=no")
            tracked: Counter[str] = Counter()
            if status_code == 0:
                for line in status_out.splitlines():
                    code_text = line[:2].strip() or line[:2]
                    tracked[code_text] += 1
            source["git"] = {
                "available": True,
                "head": head,
                "branch": branch_out.strip() if branch_code == 0 else None,
                "tracked_changes": dict(sorted(tracked.items())),
                "tracked_change_total": sum(tracked.values()),
                "clean": status_code == 0 and sum(tracked.values()) == 0,
                "status_error": None if status_code == 0 else (status_err or "").strip()[:300],
            }
            if status_code == 0 and sum(tracked.values()):
                self.gaps.append(
                    "Working tree is dirty: "
                    f"{sum(tracked.values())} tracked file(s) changed "
                    f"({dict(sorted(tracked.items()))}); git HEAD alone does not reproduce this tree."
                )
        self.identities["source_repository"] = source

    def collect_runtime_identity(self) -> None:
        runtime: dict[str, Any] = {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "pid": os.getpid(),
        }
        torch_row = next((check for check in self.checks if check["id"] == "torch"), None)
        if torch_row is not None:
            runtime["torch"] = {
                "status": torch_row["status"],
                "observed": torch_row["observed"],
            }
        self.identities["runtime"] = runtime

    def collect_asset_identity(self) -> None:
        self.identities["wsl"] = self.wsl.identity()
        engines: dict[str, Any] = {}
        for check in self.checks:
            if check["kind"] == WSL_FILE:
                engines[check["id"]] = {
                    "wsl": {key: check["location"][key] for key in ("wsl_distro", "wsl_path")},
                    "sha256_expected": check["sha256_expected"],
                    "sha256_actual": check["sha256_actual"],
                    "size_bytes": check["size_bytes"],
                    "status": check["status"],
                }
        game = next((check for check in self.checks if check["id"] == "game_data"), None)
        self.identities["engine_and_data"] = {
            "engines": engines,
            "game_data": {
                "wsl": {"wsl_distro": game["location"]["wsl_distro"], "wsl_path": game["location"]["wsl_path"]},
                "status": game["status"],
                "observed": game["observed"],
            } if game is not None else None,
            "wsl_python": next(
                ({"status": check["status"], "observed": check["observed"]}
                 for check in self.checks if check["kind"] == WSL_PYTHON),
                None,
            ),
        }

    def collect_protocol_identity(self) -> None:
        plan = self.identities.get("plan_spec") or {}
        tool = self.identities["source_repository"]["tool"]
        parts = {
            "audit_schema": RESULT_SCHEMA,
            "assets_schema": ASSETS_SCHEMA,
            "assets_config_sha256": self.config_sha256,
            "plan_spec_sha256": plan.get("sha256"),
            "spec_revision": self.cfg["spec_revision"],
            "tool_sha256": tool["sha256"],
            "sync_manifest_sha256": (self.identities.get("sync_manifest") or {}).get("sha256"),
        }
        self.identities["protocol"] = {
            "digest": sha256_bytes(_json_bytes(parts)),
            "digest_kind": "v430-audit-identity-composite/1",
            "digest_components": parts,
            "observation_action_protocol": "not_defined_in_W00",
            "note": (
                "This digest covers the audit identity (schemas, config, plan spec, manifest, tool). "
                "The frozen observation/action protocol is W01's contracts.py and is not asserted here."
            ),
        }

    # -- result ------------------------------------------------------------ #
    def statistics(self) -> dict:
        counts = Counter(check["status"] for check in self.checks)
        by_kind = Counter(check["kind"] for check in self.checks)
        by_category = Counter(check["category"] for check in self.checks)
        unmet = [check["id"] for check in self.checks if not check["requested_check_met"]]
        for status in STATUSES:
            counts.setdefault(status, 0)
        return {
            "denominator": len(self.checks),
            "denominator_note": (
                "one row per registered check: config pins (plan spec, sync manifest, manifest coverage) "
                "plus every registered asset"
            ),
            "registered_assets": sum(1 for check in self.checks if check["category"] == "asset"),
            "verified": counts[ST_VERIFIED],
            "present_unhashed": counts[ST_UNHASHED],
            "mismatch": counts[ST_MISMATCH],
            "missing": counts[ST_MISSING],
            "unreachable": counts[ST_UNREACHABLE],
            "unmet": len(unmet),
            "unmet_ids": unmet,
            "by_category": dict(sorted(by_category.items())),
            "by_kind": dict(sorted(by_kind.items())),
            "bytes_hashed": self.bytes_hashed,
            "files_hashed": self.files_hashed,
            "manifest_entries": self.manifest_present,
            "manifest_entries_covered": self.manifest_covered,
            "wsl_commands": self.wsl.commands,
            "wsl_available": self.wsl.available,
        }

    def execute(self) -> tuple[dict, int]:
        self.collect_source_identity()
        self.check_plan_spec()
        self.check_sync_manifest()
        self.check_assets()
        live_rows = {asset["id"]: bool(asset.get("live_mutated", False)) for asset in self.cfg["assets"]}
        for check in self.checks:
            check["live_mutated"] = live_rows.get(check["id"], False)
        self.collect_runtime_identity()
        self.collect_asset_identity()
        self.collect_protocol_identity()

        stats = self.statistics()
        unmet = stats["unmet"]
        code = EXIT_OK if not unmet else EXIT_GATE
        for check in self.checks:
            if check["status"] == ST_UNHASHED:
                self.gaps.append(
                    f"{check['id']}: present but not content-hashed - {check['reason']}; not counted as verified."
                )
            elif check["status"] in GATE_FAIL_STATUSES:
                text = f"{check['id']}: {check['status']} - {check['reason']}"
                if check.get("live_mutated"):
                    text += (
                        "; this row is a live execution file (state/journal/budget) that was appended after "
                        "the pinned manifest was generated, so it is live drift rather than static corruption: "
                        "do not re-pin it blindly (DEVELOPMENT_GUIDE section 12)."
                    )
                self.gaps.append(text)
        self.gaps.append(
            "audit starts no engine, no game and no training: it measures registered assets and pins only."
        )
        self.gaps.append(
            "Observation/action protocol validity is W01's contract; this audit asserts no capability and no protocol conformance."
        )
        self.gaps.append("G430-E0 is not passed by this audit; the W00 gate also needs W01's real replay evidence.")

        finished = _now()
        result = {
            "schema": RESULT_SCHEMA,
            "experiment_id": self.cfg["experiment_id"],
            "run_id": self.out.stem,
            "phase": "W00",
            "mode": "read_only_audit",
            "command": "audit",
            "spec_revision": self.cfg["spec_revision"],
            "config_frozen_at": self.cfg["frozen_at"],
            "started_at": _iso(self.started),
            "finished_at": _iso(finished),
            "duration_seconds": round((finished - self.started).total_seconds(), 3),
            "timezone": finished.strftime("%z"),
            "argv": self.argv,
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "output_path": str(self.out),
            "exit_code": code,
            "execution_status": "completed" if code == EXIT_OK else "completed_with_unmet_checks",
            "protocol_validity": "valid",
            "protocol_validity_detail": {
                "config_schema": "ok",
                "config_frozen_at": "ok" if self.cfg.get("frozen_at") else "missing",
                "plan_spec_pin": "ok" if self.identities.get("plan_spec") else "not_bound",
                "sync_manifest_pin": "ok" if self.identities.get("sync_manifest") else "not_bound",
                "observation_action_protocol": "not_asserted_by_W00_audit",
            },
            "research_result": {
                "kind": "asset_audit",
                "requested_gate": "every registered check is satisfied",
                "passed": not unmet,
                "unmet_checks": unmet,
                "note": "an asset audit is not a capability measurement",
            },
            "capability_gate": {
                "status": "not_run",
                "gates": {"G430-E0": "not_run", "G410-0": "not_reassessed"},
                "reason": "W00 audits assets and identities only; capability needs W01 replays and later gates.",
            },
            "identities": self.identities,
            "checks": self.checks,
            "statistics": stats,
            "remaining_gaps": self.gaps,
            "notes": self.notes,
        }
        return result, code


# --------------------------------------------------------------------------- #
# result writing
# --------------------------------------------------------------------------- #
def emit_result(out: Path, payload: dict) -> str | None:
    """Create ``out`` exclusively; never overwrite an existing file."""
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError:
        return f"refusing to overwrite existing output: {out}"
    except OSError as exc:
        return f"cannot write {out}: {exc}"
    return None


def invalid_payload(*, cfg_path: Path, out: Path, argv: list[str], started: datetime,
                    message: str) -> dict:
    finished = _now()
    return {
        "schema": RESULT_SCHEMA,
        "experiment_id": None,
        "run_id": out.stem,
        "phase": "W00",
        "mode": "read_only_audit",
        "command": "audit",
        "spec_revision": None,
        "started_at": _iso(started),
        "finished_at": _iso(finished),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "timezone": finished.strftime("%z"),
        "argv": argv,
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "output_path": str(out),
        "exit_code": EXIT_INVALID,
        "execution_status": "invalid_inputs",
        "protocol_validity": "invalid",
        "protocol_validity_detail": {"config": "invalid"},
        "research_result": {
            "kind": "asset_audit",
            "requested_gate": "every registered check is satisfied",
            "passed": False,
            "unmet_checks": [],
            "note": "no measurement was taken because the inputs are invalid",
        },
        "capability_gate": {"status": "not_run", "gates": {"G430-E0": "not_run"}, "reason": "invalid inputs"},
        "identities": {
            "config": {"path": str(cfg_path), "sha256": None},
            "runtime": {"python_executable": sys.executable, "python_version": platform.python_version(),
                        "platform": platform.platform()},
        },
        "checks": [],
        "statistics": {
            "denominator": 0, "registered_assets": 0, "verified": 0, "present_unhashed": 0,
            "mismatch": 0, "missing": 0, "unreachable": 0, "unmet": 0, "unmet_ids": [],
            "by_category": {}, "by_kind": {}, "bytes_hashed": 0, "files_hashed": 0,
            "manifest_entries": None, "manifest_entries_covered": None, "wsl_commands": 0,
            "wsl_available": None,
        },
        "remaining_gaps": [f"invalid inputs: {message}"],
        "notes": ["exit 2 means nothing was measured; fix the inputs and rerun into a new --out path."],
    }


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_audit(args: argparse.Namespace) -> int:
    out = Path(args.out)
    started = _now()
    argv = list(sys.argv)
    if out.exists():
        _err(f"refusing to overwrite existing output: {out}")
        return EXIT_INVALID

    cfg_path = Path(args.config)
    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        _err(str(exc))
        payload = invalid_payload(cfg_path=cfg_path, out=out, argv=argv, started=started, message=str(exc))
        write_error = emit_result(out, payload)
        if write_error:
            _err(write_error)
        return EXIT_INVALID

    repo = _repo_root()
    run = AuditRun(cfg=cfg, cfg_path=cfg_path, repo=repo, out=out, started=started, argv=argv)
    result, code = run.execute()
    write_error = emit_result(out, result)
    if write_error:
        _err(write_error)
        return 1
    stats = result["statistics"]
    print(
        "audit: "
        f"checks {stats['denominator']} | verified {stats['verified']} | "
        f"unhashed {stats['present_unhashed']} | mismatch {stats['mismatch']} | "
        f"missing {stats['missing']} | unreachable {stats['unreachable']} | "
        f"wsl {'up' if stats['wsl_available'] else 'down'} | "
        f"gate {'passed' if result['research_result']['passed'] else 'NOT met'} | exit {code}"
    )
    if code != EXIT_OK:
        print(f"unmet checks: {', '.join(stats['unmet_ids'])}", file=sys.stderr)
    print(f"result: {out}")
    return code


# --------------------------------------------------------------------------- #
# validate-contract (W01)
# --------------------------------------------------------------------------- #
#: Schema of the evidence document this subcommand writes.
CONTRACT_RESULT_SCHEMA = "v430-contract-validation/1"
#: The only config schema ``validate-contract`` accepts.
CONTRACT_CONFIG_SCHEMA = "v430-contracts/1"
#: Files whose identity the evidence records.  All are **text** products, so all
#: are hashed under ``text/lf-normalized`` (``SHA_LINE_ENDING_FINDING.md`` §6).
CONTRACT_IDENTITY_FILES: tuple[tuple[str, str], ...] = (
    ("contracts_module", "touhou_rl/v430/contracts.py"),
    ("actions_module", "touhou_rl/v430/actions.py"),
    ("session_module", "touhou_rl/v430/session.py"),
    ("evaluate_module", "touhou_rl/v430/evaluate.py"),
    ("contracts_config", "configs/v430/contracts.json"),
    ("plan_spec", "Document/Update/Alpha/v4.3.0/PLAN_SPEC.json"),
    ("cli", "tools/rl/v430.py"),
)
#: The four protocol digests the evidence must carry.
PROTOCOL_DIGEST_PARTS: tuple[str, ...] = ("observation", "action", "reward", "terminal")
#: The eight session states the work package requires.
REQUIRED_SESSION_STATES: tuple[str, ...] = (
    "playing", "hit_pending", "respawn", "transition", "game_over", "menu",
    "ending", "unknown")


class Shape:
    """Minimal duck-typed array (``.shape`` / ``.dtype``) for contract probes.

    ``contracts.check_actor_observation`` reads exactly these two attributes for
    the image fields, so the observation contract can be exercised without
    materialising 474 kB of pixels per check.
    """

    __slots__ = ("shape", "dtype")

    def __init__(self, shape: Sequence[int], dtype: str) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _diff_dicts(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Every path where two JSON-shaped values differ (both directions)."""

    out: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else key
            if key not in expected:
                out.append(f"{child}: only in the module")
            elif key not in actual:
                out.append(f"{child}: only in the config")
            else:
                out.extend(_diff_dicts(expected[key], actual[key], child))
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(expected)} != {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            out.extend(_diff_dicts(left, right, f"{path}[{index}]"))
        return out
    if expected != actual or type(expected) is not type(actual):
        out.append(f"{path}: module {expected!r} != config {actual!r}")
    return out


class CheckLog:
    """One row per asserted property; a single failed row fails the command."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, group: str, name: str, ok: bool, detail: Any = None) -> bool:
        self.rows.append({"group": group, "name": name, "ok": bool(ok),
                          "detail": detail})
        return bool(ok)

    def eq(self, group: str, name: str, expected: Any, actual: Any) -> bool:
        ok = expected == actual and type(expected) is type(actual)
        return self.add(group, name, ok,
                        {"expected": expected, "actual": actual})

    def refuses(self, group: str, name: str, call, *, exceptions: tuple) -> bool:
        try:
            call()
        except exceptions as exc:
            return self.add(group, name, True, {"raised": type(exc).__name__,
                                                "message": str(exc)[:300]})
        except BaseException as exc:  # noqa: BLE001 - wrong exception is a failure
            return self.add(group, name, False,
                            {"raised": type(exc).__name__,
                             "message": str(exc)[:300],
                             "note": f"expected one of {exceptions}"})
        return self.add(group, name, False,
                        {"raised": None, "note": "nothing was refused"})

    def accepts(self, group: str, name: str, call) -> bool:
        try:
            detail = call()
        except BaseException as exc:  # noqa: BLE001
            return self.add(group, name, False,
                            {"raised": type(exc).__name__,
                             "message": str(exc)[:300]})
        return self.add(group, name, True, {"result": detail})

    def groups(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for row in self.rows:
            bucket = out.setdefault(row["group"], {"checks": 0, "failed": 0})
            bucket["checks"] += 1
            if not row["ok"]:
                bucket["failed"] += 1
        return out

    def failures(self) -> list[dict]:
        return [row for row in self.rows if not row["ok"]]

    def summary(self) -> dict:
        return {
            "checks": len(self.rows),
            "failed": len(self.failures()),
            "groups": self.groups(),
        }


class ContractValidation:
    """Compare ``configs/v430/contracts.json`` with the code, in both directions.

    Every check is a *behavioural* assertion where one is possible: the 72 masks
    are built, the log-probabilities are summed, the bootstrap rules are
    exercised with a legal and an illegal record, the Wilson interval is computed.
    A check that only re-read a constant would not catch a drifted implementation.
    """

    def __init__(self, *, cfg: dict, cfg_path: Path, repo: Path, out: Path,
                 started: datetime, argv: list[str]) -> None:
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.repo = repo
        self.out = out
        self.started = started
        self.argv = argv
        self.log = CheckLog()
        self.identities: dict[str, Any] = {}
        self.plan: dict = {}
        self.notes: list[str] = []
        self.gaps: list[str] = []

    # -- imports (deferred: ``audit`` must keep its original import graph) -- #
    def load_modules(self) -> tuple[Any, Any, Any, Any]:
        """Import the v430 modules, or raise :class:`ConfigError` (exit 2)."""

        try:
            from touhou_rl.v430 import contracts as contracts_module
            from touhou_rl.v430 import actions as actions_module
            from touhou_rl.v430 import evaluate as evaluate_module
            from touhou_rl.v430 import session as session_module
        except Exception as exc:  # noqa: BLE001 - an unimportable module is invalid
            raise ConfigError(
                f"cannot import the v430 modules under test: "
                f"{type(exc).__name__}: {exc}") from exc
        return contracts_module, actions_module, evaluate_module, session_module

    # -- identity ---------------------------------------------------------- #
    def collect_identities(self, evaluate_module: Any) -> None:
        for label, relative in CONTRACT_IDENTITY_FILES:
            path = self.repo / relative
            try:
                identity = evaluate_module.identify_file(
                    path, scope=evaluate_module.DIGEST_SCOPE_TEXT)
            except Exception as exc:  # noqa: BLE001
                self.log.add("identity", label, False,
                             {"path": relative, "error": str(exc)[:300]})
                continue
            identity["label"] = label
            identity["path_relative"] = relative
            self.identities[label] = identity
            self.log.add("identity", label, True,
                         {"sha256": identity["sha256"],
                          "digest_scope": identity["digest_scope"],
                          "bytes": identity["bytes"],
                          "line_endings": identity["line_endings"]})
        try:
            checked = evaluate_module.assert_identity_scopes(self.identities)
        except Exception as exc:  # noqa: BLE001
            self.log.add("identity", "every_identity_has_digest_scope", False,
                         {"error": str(exc)[:300]})
        else:
            self.log.add("identity", "every_identity_has_digest_scope", True,
                         {"identities": len(checked)})

    # -- plan spec --------------------------------------------------------- #
    def check_plan_spec(self) -> None:
        block = self.cfg.get("plan_spec") or {}
        path = self.repo / block.get("path", "")
        self.log.eq("plan_spec", "config_schema", CONTRACT_CONFIG_SCHEMA,
                    self.cfg.get("schema"))
        if not path.is_file():
            self.log.add("plan_spec", "present", False, {"path": str(path)})
            return
        try:
            self.plan = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.log.add("plan_spec", "readable_json", False, {"error": str(exc)[:300]})
            return
        self.log.add("plan_spec", "present", True,
                     {"path": block.get("path"), "schema": self.plan.get("schema"),
                      "revision": self.plan.get("revision")})
        self.log.eq("plan_spec", "schema", block.get("schema_expected"),
                    self.plan.get("schema"))
        self.log.eq("plan_spec", "revision", block.get("revision_expected"),
                    self.plan.get("revision"))
        self.log.eq("plan_spec", "config_spec_revision", self.cfg.get("spec_revision"),
                    self.plan.get("revision"))

    # -- config <-> module constants (bidirectional) ----------------------- #
    def check_constants(self, contracts_module: Any) -> None:
        module_constants = contracts_module.contract_constants()
        config_constants = self.cfg.get("constants")
        if not isinstance(config_constants, dict):
            self.log.add("config_constants", "constants_block_present", False,
                         {"got": type(config_constants).__name__})
            return
        differences = _diff_dicts(module_constants, config_constants)
        self.log.add("config_constants", "bidirectional_equality",
                     not differences,
                     {"differences": differences[:25],
                      "difference_count": len(differences),
                      "note": ("compared contracts.contract_constants() with "
                               "configs/v430/contracts.json['constants']; both "
                               "directions (module-only and config-only keys) are "
                               "reported")})
        self.log.eq("config_constants", "contract_version",
                    contracts_module.CONTRACT_VERSION,
                    config_constants.get("contract_version"))
        self.log.eq("config_constants", "spec_revision",
                    contracts_module.SPEC_REVISION,
                    config_constants.get("spec_revision"))

    # -- action space ------------------------------------------------------ #
    def check_action_space(self, C: Any, A: Any) -> None:
        plan_action = self.plan.get("action") or {}
        self.log.eq("action_space", "joint_count", 72, C.JOINT_COUNT)
        self.log.eq("action_space", "n_move", 9, C.N_MOVE)
        self.log.eq("action_space", "head_sizes", {"move": 9, "focus": 2, "shoot": 2,
                                                  "bomb": 2}, dict(C.HEAD_SIZES))
        self.log.eq("action_space", "head_order", ["move", "focus", "shoot", "bomb"],
                    list(C.HEAD_ORDER))
        self.log.eq("action_space", "encode_decode_inverse", True, C.joint_inverse())
        pairs = A.all_joint_masks()
        self.log.eq("action_space", "mask_count", 72, len(pairs))
        self.log.eq("action_space", "distinct_masks", 72,
                    len({mask for mask, _action in pairs}))
        bad = []
        for mask, action in pairs:
            try:
                A.check_agent_mask(mask)
            except C.ContractError as exc:
                bad.append({"mask": mask, "action": list(action), "error": str(exc)[:120]})
        self.log.add("action_space", "all_masks_agent_emittable", not bad,
                     {"rejected": bad[:5], "count": len(bad)})
        frames = A.compile_masks(pairs[0][1]) if pairs else ()
        self.log.eq("action_space", "compile_masks_frames", C.HOLD_GAME_FRAMES,
                    len(frames))
        self.log.eq("action_space", "config_joint_count", 72,
                    (self.cfg.get("constants") or {}).get("action", {})
                    .get("joint_count"))
        self.log.eq("action_space", "plan_joint_count", 72, plan_action.get("joint_count"))
        self.log.eq("action_space", "plan_heads", {"move": 9, "focus": 2, "shoot": 2,
                                                  "bomb": 2},
                    dict(plan_action.get("heads") or {}))
        self.log.eq("action_space", "truth_action_mask", False,
                    bool(plan_action.get("truth_action_mask")))
        self.log.eq("action_space", "forced_shoot", False,
                    bool(plan_action.get("forced_shoot")))

    # -- four-head probability semantics ----------------------------------- #
    def check_probability(self, C: Any, A: Any) -> None:
        head_lp = {"move": -0.5, "focus": -0.25, "shoot": -0.125, "bomb": -0.0625}
        head_entropy = {"move": 1.5, "focus": 0.25, "shoot": 0.125, "bomb": 0.0625}
        self.log.eq("probability", "joint_logprob_is_sum", sum(head_lp.values()),
                    C.joint_logprob_of(head_lp))
        self.log.eq("probability", "joint_entropy_is_sum", sum(head_entropy.values()),
                    C.joint_entropy_of(head_entropy))
        self.log.eq("probability", "joint_logprob_components",
                    sum(head_lp.values()),
                    C.check_joint_logprob(head_lp["move"], head_lp["focus"],
                                          head_lp["shoot"], head_lp["bomb"]))
        self.log.refuses("probability", "nan_logprob_refused",
                         lambda: C.check_head_logprob(float("nan"), "move"),
                         exceptions=(C.ContractError,))
        self.log.refuses("probability", "negative_entropy_refused",
                         lambda: C.check_head_entropy(-1.0, "move"),
                         exceptions=(C.ContractError,))
        self.log.eq("probability", "forced_shoot_constant", False, C.forced_shoot())
        self.log.eq("probability", "truth_action_mask_constant", False,
                    C.truth_action_mask())
        # No implicit SHOOT: the emitted mask's shoot bit is the request's bit.
        shoot_bit = 1 << C.shoot_bit_index()
        strip = 0
        mismatched = []
        for mask, action in A.all_joint_masks():
            move, focus, shoot, bomb = action
            if shoot == 0 and (mask & shoot_bit):
                mismatched.append({"action": [move, focus, shoot, bomb],
                                   "mask": f"0x{mask:04x}", "problem": "shoot bit set"})
            if shoot == 1 and not (mask & shoot_bit):
                mismatched.append({"action": [move, focus, shoot, bomb],
                                   "mask": f"0x{mask:04x}", "problem": "shoot bit clear"})
            strip += 1
        self.log.add("probability", "shoot_bit_equals_request", not mismatched,
                     {"checked": strip, "mismatched": mismatched[:5],
                      "shoot_bit": shoot_bit})
        self.log.refuses(
            "probability", "forced_shoot_rule_refused",
            lambda: C.check_no_forced_shoot_rule([shoot_bit, shoot_bit],
                                                 shoot_requests=[0, 0]),
            exceptions=(C.ContractError,))
        factorization = C.action_parameters().get("factorization")
        self.log.eq("probability", "factorization_string",
                    factorization,
                    (self.cfg.get("constants") or {}).get("action", {})
                    .get("factorization"))
        self.log.add("probability", "factorization_is_summed",
                     isinstance(factorization, str) and "summed" in factorization,
                     {"factorization": factorization})

    # -- observation contract ---------------------------------------------- #
    def check_observation(self, C: Any) -> None:
        config_obs = (self.cfg.get("constants") or {}).get("observation") or {}
        plan_o2 = (self.plan.get("observation") or {}).get("O2") or {}
        specs = {spec.name: spec for spec in C.ACTOR_OBSERVATION_FIELDS}
        self.log.eq("observation", "field_names", ["global_rgb", "play_gray",
                                                  "hud_gray", "executed_key_history",
                                                  "history_valid", "elapsed_capture_s",
                                                  "memory"],
                    list(C.ACTOR_OBSERVATION_KEYS))
        self.log.eq("observation", "field_count", 7, len(specs))
        shape_keys = {
            "global_rgb": "global_rgb_chw",
            "play_gray": "play_gray_chw",
            "hud_gray": "hud_gray_chw",
            "executed_key_history": "executed_key_history_shape",
            "history_valid": "history_valid_shape",
            "memory": "memory_shape",
        }
        for name, key in shape_keys.items():
            spec = specs[name]
            self.log.eq("observation", f"config_shape:{name}", list(spec.shape),
                        config_obs.get(key))
        self.log.eq("observation", "config_shape:elapsed_capture_s",
                    len(specs["elapsed_capture_s"].shape),
                    config_obs.get("elapsed_capture_s_rank"))
        plan_shapes = {
            "global_rgb": plan_o2.get("global_rgb_chw"),
            "play_gray": plan_o2.get("play_gray_chw"),
            "hud_gray": plan_o2.get("hud_gray_chw"),
            "executed_key_history": plan_o2.get("executed_key_history_shape"),
            "history_valid": plan_o2.get("history_valid_shape"),
        }
        for name, expected in plan_shapes.items():
            self.log.eq("observation", f"plan_shape:{name}", list(specs[name].shape),
                        expected)
        self.log.eq("observation", "plan_key_order", list(C.KEY_ORDER),
                    plan_o2.get("key_order"))
        self.log.eq("observation", "config_key_order", list(C.KEY_ORDER),
                    config_obs.get("key_order"))
        self.log.eq("observation", "source_rgb_wh", list(C.SOURCE_RGB_WH),
                    list((self.plan.get("observation") or {}).get("source_rgb_wh") or []))
        self.log.eq("observation", "play_crop", list(C.O2_PLAY_CROP_XYWH),
                    plan_o2.get("play_crop_xywh"))
        self.log.eq("observation", "hud_crop", list(C.O2_HUD_CROP_XYWH),
                    plan_o2.get("hud_crop_xywh"))
        self.log.eq("observation", "image_bytes_per_decision", 474112,
                    C.O2_IMAGE_BYTES_PER_DECISION)
        self.log.eq("observation", "config_image_bytes", 474112,
                    config_obs.get("image_bytes_per_decision"))
        self.log.eq("observation", "plan_image_bytes", 474112,
                    plan_o2.get("image_bytes_per_decision"))
        self.log.eq("observation", "o2_digest_matches_config", C.o2_digest(),
                    config_obs.get("digest"))
        self.log.eq("observation", "storage_dtype", "uint8", C.O2_STORAGE_DTYPE)
        self.log.eq("observation", "gray_rule", "(77*R+151*G+28*B)>>8", C.O2_GRAY_RULE)
        config_fields = {row.get("name"): row for row in config_obs.get("fields") or []}
        self.log.eq("observation", "config_field_rows", 7, len(config_fields))
        for name, spec in specs.items():
            row = config_fields.get(name) or {}
            self.log.eq("observation", f"config_dtype:{name}", spec.dtype,
                        row.get("dtype"))
            self.log.eq("observation", f"config_shape_row:{name}", list(spec.shape),
                        list(row.get("shape") or []))

        # Positive control: a full, valid observation must be *accepted*.
        observation = {
            "global_rgb": Shape(specs["global_rgb"].shape, "uint8"),
            "play_gray": Shape(specs["play_gray"].shape, "uint8"),
            "hud_gray": Shape(specs["hud_gray"].shape, "uint8"),
            "executed_key_history": [[0] * 7 for _ in range(8)],
            "history_valid": [True] * 8,
            "elapsed_capture_s": 1.0 / C.REFERENCE_HZ,
            "memory": [0.0] * 256,
        }
        self.log.accepts("observation", "valid_observation_accepted",
                         lambda: sorted(C.check_actor_observation(observation)))

        def _mutate(key: str, value: Any):
            copy = dict(observation)
            copy[key] = value
            return lambda: C.check_actor_observation(copy)

        self.log.refuses("observation", "missing_field_refused",
                         lambda: C.check_actor_observation(
                             {k: v for k, v in observation.items() if k != "memory"}),
                         exceptions=(C.ContractError,))
        self.log.refuses("observation", "extra_field_refused",
                         _mutate("stage_truth", [0]),
                         exceptions=(C.ContractError,))
        self.log.refuses("observation", "resized_image_refused",
                         _mutate("global_rgb", Shape((3, 224, 320), "uint8")),
                         exceptions=(C.ContractError,))
        self.log.refuses("observation", "wrong_dtype_refused",
                         _mutate("memory", Shape((256,), "uint8")),
                         exceptions=(C.ContractError,))
        self.log.refuses("observation", "history_hole_refused",
                         _mutate("history_valid", [True, False] + [True] * 6),
                         exceptions=(C.ContractError,))
        self.log.refuses("observation", "key_history_2_refused",
                         _mutate("executed_key_history", [[2] * 7] + [[0] * 7] * 7),
                         exceptions=(C.ContractError,))
        self.log.refuses("observation", "negative_capture_interval_refused",
                         _mutate("elapsed_capture_s", -0.5),
                         exceptions=(C.ContractError,))

    # -- privileged fields ------------------------------------------------- #
    def check_privileged(self, C: Any, S: Any) -> None:
        config_constants = self.cfg.get("constants") or {}
        plan_forbidden = list(((self.plan.get("observation") or {}).get("O2") or {})
                              .get("forbidden_actor_fields") or [])
        self.log.eq("privileged", "config_forbidden_actor_fields",
                    list(C.FORBIDDEN_ACTOR_FIELDS),
                    list(config_constants.get("forbidden_actor_fields") or []))
        self.log.eq("privileged", "plan_forbidden_actor_fields", plan_forbidden,
                    list(C.FORBIDDEN_ACTOR_FIELDS))
        self.log.eq("privileged", "config_observation_forbidden",
                    list(C.FORBIDDEN_ACTOR_FIELDS),
                    list(((config_constants.get("observation") or {})
                          .get("forbidden_actor_fields")) or []))
        self.log.eq("privileged", "config_privileged_field_names",
                    list(C.PRIVILEGED_FIELD_NAMES),
                    list(config_constants.get("privileged_field_names") or []))
        accepted = []
        for name in plan_forbidden:
            try:
                C.assert_not_privileged({name: 1}, what="probe")
            except C.ContractError:
                accepted.append(name)
        self.log.eq("privileged", "every_forbidden_name_refused", plan_forbidden,
                    accepted)
        session_truth = ("stage_id", "stage_truth", "stage_clear", "game_over",
                         "lives", "bomb_count", "resource_truth", "engine_frame")
        missing = [name for name in session_truth
                   if name not in S.SESSION_TRUTH_FIELDS]
        self.log.add("privileged", "session_refuses_engine_truth", not missing,
                     {"missing": missing,
                      "session_truth_fields": list(S.SESSION_TRUTH_FIELDS)})
        refused = []
        for name in session_truth:
            try:
                S.assert_pixel_only({name: 1}, what="probe")
            except C.ContractError:
                refused.append(name)
        self.log.eq("privileged", "session_refuses_each_truth_name",
                    list(session_truth), refused)

    # -- time / step contract ---------------------------------------------- #
    def check_time(self, C: Any) -> None:
        config_time = (self.cfg.get("constants") or {}).get("time") or {}
        config_action = (self.cfg.get("constants") or {}).get("action") or {}
        plan_action = self.plan.get("action") or {}
        plan_ppo = self.plan.get("ppo") or {}
        self.log.eq("time", "hold_game_frames", 2, C.HOLD_GAME_FRAMES)
        self.log.eq("time", "config_action_hold", C.HOLD_GAME_FRAMES,
                    config_action.get("hold_game_frames"))
        self.log.eq("time", "config_time_hold", C.HOLD_GAME_FRAMES,
                    config_time.get("hold_game_frames"))
        self.log.eq("time", "plan_hold", C.HOLD_GAME_FRAMES,
                    plan_action.get("hold_game_frames"))
        self.log.eq("time", "reference_hz", 60, C.REFERENCE_HZ)
        self.log.eq("time", "config_reference_hz", C.REFERENCE_HZ,
                    config_time.get("reference_hz"))
        self.log.eq("time", "duration_sources", list(C.DURATION_SOURCES),
                    list(config_time.get("duration_sources") or []))
        self.log.eq("time", "duration_tolerance_frames",
                    C.DURATION_TOLERANCE_FRAMES,
                    config_time.get("duration_tolerance_frames"))
        external = C.check_duration(
            duration_source=C.DURATION_SOURCE_EXTERNAL,
            elapsed_capture_s=2.0 / C.REFERENCE_HZ)
        self.log.add("time", "external_two_frame_interval",
                     abs(external.effective_frames - 2.0) < 1e-9,
                     {"effective_frames": external.effective_frames,
                      "duration_source": external.duration_source})
        self.log.accepts("time", "two_frame_step_is_normal",
                         lambda: C.assert_normal_two_frame(external).as_dict())
        three = C.check_duration(
            duration_source=C.DURATION_SOURCE_EXTERNAL,
            elapsed_capture_s=3.0 / C.REFERENCE_HZ)
        self.log.refuses("time", "three_frame_step_refused",
                         lambda: C.assert_normal_two_frame(three),
                         exceptions=(C.ContractError,))
        engine = C.check_duration(duration_source=C.DURATION_SOURCE_ENGINE,
                                  elapsed_game_frames=2)
        self.log.add("time", "engine_source_reports_frames",
                     engine.effective_frames == 2.0,
                     {"effective_frames": engine.effective_frames})
        self.log.refuses("time", "external_source_rejects_engine_frames",
                         lambda: C.check_duration(
                             duration_source=C.DURATION_SOURCE_EXTERNAL,
                             elapsed_game_frames=2, elapsed_capture_s=2.0 / 60.0),
                         exceptions=(C.ContractError,))
        self.log.refuses("time", "engine_source_requires_frames",
                         lambda: C.check_duration(
                             duration_source=C.DURATION_SOURCE_ENGINE,
                             elapsed_capture_s=2.0 / 60.0),
                         exceptions=(C.ContractError,))
        invalid = C.check_duration(duration_source=C.DURATION_SOURCE_INVALID)
        self.log.add("time", "invalid_source_has_no_frame_count",
                     invalid.effective_frames is None,
                     {"effective_frames": invalid.effective_frames,
                      "duration_source": invalid.duration_source})
        self.log.eq("time", "effective_frames_from_capture", 2.0,
                    C.effective_frames_from_capture(2.0 / C.REFERENCE_HZ))
        self.log.add("time", "plan_r1_r2_clock_is_external",
                     "external" in str(plan_ppo.get("duration_source_R1_R2", "")),
                     {"plan": plan_ppo.get("duration_source_R1_R2")})
        self.log.add("time", "reference_hz_matches_plan",
                     str(C.REFERENCE_HZ) in str(plan_ppo.get("reference_hz")),
                     {"plan": plan_ppo.get("reference_hz")})

    # -- terminal / transition contract ------------------------------------ #
    def check_terminal(self, C: Any) -> None:
        config_transition = (self.cfg.get("constants") or {}).get("transition") or {}
        module_transition = C.contract_constants().get("transition") or {}
        differences = _diff_dicts(module_transition, config_transition)
        self.log.add("terminal", "config_transition_block_matches", not differences,
                     {"differences": differences[:15],
                      "difference_count": len(differences)})
        self.log.eq("terminal", "end_flags", ["terminated", "truncated", "segment_end"],
                    list(C.TRANSITION_END_FLAGS))
        self.log.eq("terminal", "end_flag_count", 3, len(set(C.TRANSITION_END_FLAGS)))
        record = {"terminated": False, "truncated": False, "segment_end": True}
        self.log.eq("terminal", "check_end_flags_accepts", (False, False, True),
                    C.check_end_flags(record))
        self.log.refuses("terminal", "done_alias_refused",
                         lambda: C.check_end_flags({**record, "done": True}),
                         exceptions=(C.ContractError,))
        self.log.refuses("terminal", "terminated_and_truncated_refused",
                         lambda: C.check_end_flags({"terminated": True,
                                                    "truncated": True,
                                                    "segment_end": False,
                                                    "truncation_reason": "other"}),
                         exceptions=(C.ContractError,))
        self.log.refuses("terminal", "truncated_without_reason_refused",
                         lambda: C.check_end_flags({"terminated": False,
                                                    "truncated": True,
                                                    "segment_end": False}),
                         exceptions=(C.ContractError,))
        self.log.eq("terminal", "terminated_bootstrap_is_zero", 0,
                    C.check_bootstrap(terminated=True, truncated=False,
                                      segment_end=False, bootstrap_mask=0))
        self.log.refuses("terminal", "terminated_bootstrap_one_refused",
                         lambda: C.check_bootstrap(terminated=True, truncated=False,
                                                   segment_end=False,
                                                   bootstrap_mask=1),
                         exceptions=(C.ContractError,))
        self.log.refuses("terminal", "truncated_bootstrap_without_terminal_obs",
                         lambda: C.check_bootstrap(terminated=False, truncated=True,
                                                   segment_end=False,
                                                   bootstrap_mask=1),
                         exceptions=(C.ContractError,))
        self.log.refuses("terminal", "continuing_bootstrap_without_next_obs",
                         lambda: C.check_bootstrap(terminated=False, truncated=False,
                                                   segment_end=False,
                                                   bootstrap_mask=1),
                         exceptions=(C.ContractError,))
        self.log.refuses("terminal", "reset_image_never_bootstraps",
                         lambda: C.check_bootstrap(
                             terminated=False, truncated=False, segment_end=False,
                             bootstrap_mask=1, next_obs_ref="obs:same",
                             reset_observation_ref="obs:same"),
                         exceptions=(C.ContractError,))
        self.log.eq("terminal", "bootstrapping_continuing_step", 1,
                    C.check_bootstrap(terminated=False, truncated=False,
                                      segment_end=False, bootstrap_mask=1,
                                      next_obs_ref="obs:next"))

    # -- result JSON / exit codes ------------------------------------------ #
    def check_results(self, C: Any) -> None:
        plan_cli = self.plan.get("future_cli") or {}
        config_results = (self.cfg.get("constants") or {}).get("results") or {}
        self.log.eq("results", "result_fields", list(C.RESULT_FIELDS),
                    list(plan_cli.get("result_fields") or []))
        self.log.eq("results", "config_result_fields", list(C.RESULT_FIELDS),
                    list(config_results.get("result_fields") or []))
        self.log.eq("results", "exit_codes", ["0", "2", "3", "4"],
                    sorted(str(code) for code in C.EXIT_CODES))
        self.log.eq("results", "plan_exit_codes", ["0", "2", "3", "4"],
                    sorted(str(code) for code in (plan_cli.get("exit_codes") or {})))
        self.log.eq("results", "exit_code_by_status",
                    {"ok": 0, "invalid": 2, "gate_fail": 3, "incomplete": 4},
                    dict(C.EXIT_CODE_BY_EXECUTION_STATUS))
        self.log.eq("results", "capability_gate_default", "not_run",
                    C.CAPABILITY_GATE_DEFAULT)
        self.log.eq("results", "config_capability_gate_default", "not_run",
                    config_results.get("capability_gate_default"))
        legal = C.new_result(command_class="contract",
                             research_result={"status": "pass"})
        self.log.eq("results", "new_result_capability_gate", "not_run",
                    legal["capability_gate"])
        self.log.eq("results", "exit_code_for_ok", 0, C.exit_code_for(legal))
        self.log.accepts("results", "assert_result_schema_accepts_legal_result",
                         lambda: C.assert_result_schema(legal)["execution_status"])
        self.log.refuses("results", "constructor_cannot_claim_capability",
                         lambda: C.new_result(capability_gate="pass"),
                         exceptions=(C.ContractError,))
        promoted = C.set_capability_gate(C.new_result(command_class="training"),
                                        "pass",
                                        judged_from={"judge": "x", "sha256": "0" * 64})
        self.log.refuses("results", "training_command_capability_refused",
                         lambda: C.assert_result_schema(promoted),
                         exceptions=(C.ContractError,))
        self.log.refuses("results", "invalid_protocol_with_ok_refused",
                         lambda: C.assert_result_schema(
                             {**legal, "protocol_validity": "invalid"}),
                         exceptions=(C.ContractError,))
        self.log.refuses("results", "mismatched_exit_code_refused",
                         lambda: C.assert_result_schema({**legal, "exit_code": 3}),
                         exceptions=(C.ContractError,))

    # -- session module (W01 slice B) -------------------------------------- #
    def check_session(self, C: Any, S: Any) -> None:
        plan_action = self.plan.get("action") or {}
        missing = [state for state in REQUIRED_SESSION_STATES
                   if state not in S.SESSION_STATES]
        self.log.add("session", "required_states_present", not missing,
                     {"missing": missing, "states": list(S.SESSION_STATES)})
        driving = []
        idle = []
        for state in S.SESSION_STATES:
            action = S.session_action_for_state(state)
            if action is None:
                idle.append(state)
                continue
            payload = action.as_dict()
            if payload["actor_loss_mask"] != 0 or payload["controller_source"] != "session":
                driving.append({"state": state, "action": payload})
            else:
                driving.append(state)
        problems = [row for row in driving if isinstance(row, dict)]
        self.log.add("session", "every_state_maps_to_an_action", not problems,
                     {"session_driven": [row for row in driving
                                         if not isinstance(row, dict)],
                      "actor_driven": idle, "problems": problems})
        self.log.eq("session", "unknown_presses_nothing", True,
                    S.session_action_for_state(S.UNKNOWN) is None)
        self.log.eq("session", "session_actor_loss_mask", 0,
                    S.SESSION_ACTOR_LOSS_MASK)
        self.log.eq("session", "contract_session_actor_loss_mask",
                    C.SESSION_ACTOR_LOSS_MASK, S.SESSION_ACTOR_LOSS_MASK)
        self.log.eq("session", "plan_session_actor_loss_mask", 0,
                    plan_action.get("session_actor_loss_mask"))
        self.log.eq("session", "owner_exclusive", True,
                    bool(plan_action.get("owner_exclusive")))
        try:
            audit = S.session_source_audit()
        except Exception as exc:  # noqa: BLE001
            self.log.add("session", "source_imports_pixel_only", False,
                         {"error": str(exc)[:300]})
        else:
            self.log.add("session", "source_imports_pixel_only", True,
                         {"imports": audit["imports"],
                          "sha256": audit["sha256"],
                          "digest_scope": audit["digest_scope"]})

    # -- evaluation module (W01 slice B) ----------------------------------- #
    def check_evaluation(self, C: Any, E: Any, S: Any) -> None:
        plan_formal = ((self.plan.get("evaluation") or {}).get("formal") or {})
        self.log.eq("evaluation", "digest_scopes", ["text/lf-normalized", "binary/raw"],
                    list(E.DIGEST_SCOPES))
        self.log.eq("evaluation", "session_pixel_scope", E.DIGEST_SCOPE_BINARY,
                    S.DIGEST_SCOPE_PIXELS)
        crlf = b'{\r\n  "a": 1\r\n}\r\n'
        lf_only = b'{\n  "a": 1\n}\n'
        self.log.eq("evaluation", "text_scope_ignores_line_endings",
                    E.digest_bytes(crlf, scope=E.DIGEST_SCOPE_TEXT),
                    E.digest_bytes(lf_only, scope=E.DIGEST_SCOPE_TEXT))
        self.log.add("evaluation", "binary_scope_sees_line_endings",
                     E.digest_bytes(crlf, scope=E.DIGEST_SCOPE_BINARY)
                     != E.digest_bytes(lf_only, scope=E.DIGEST_SCOPE_BINARY),
                     {"note": "binary/raw must not normalise"})
        self.log.refuses("evaluation", "identity_without_scope_refused",
                         lambda: E.check_identity({"sha256": "0" * 64}),
                         exceptions=(E.MissingDigestScopeError,))
        self.log.refuses(
            "evaluation", "scope_mismatch_is_distinguished",
            lambda: E.verify_identity(
                {"sha256": E.digest_bytes(crlf, scope=E.DIGEST_SCOPE_BINARY),
                 "digest_scope": E.DIGEST_SCOPE_TEXT, "path": "probe"},
                crlf),
            exceptions=(E.DigestScopeMismatch,))
        self.log.refuses(
            "evaluation", "content_mismatch_is_distinguished",
            lambda: E.verify_identity(
                {"sha256": "0" * 64, "digest_scope": E.DIGEST_SCOPE_TEXT,
                 "path": "probe"}, crlf),
            exceptions=(E.DigestMismatch,))
        self.log.refuses("evaluation", "zero_denominator_refused",
                         lambda: E.wilson_interval(0, 0),
                         exceptions=(E.ZeroDenominatorError,))
        low, high = E.wilson_interval(int(plan_formal.get("min_successes", 50)),
                                      int(plan_formal.get("episodes", 100)))
        self.log.add("evaluation", "plan_wilson_bound_achievable",
                     low >= float(plan_formal.get("wilson_min_lower_bound", 0.0)),
                     {"successes": plan_formal.get("min_successes"),
                      "episodes": plan_formal.get("episodes"),
                      "wilson_low": low, "wilson_high": high,
                      "required_low": plan_formal.get("wilson_min_lower_bound")})
        self.log.refuses("evaluation", "normal_approximation_edge_refused",
                         lambda: E.wilson_interval(
                             int(plan_formal.get("episodes", 100)) + 1,
                             int(plan_formal.get("episodes", 100))),
                         exceptions=(E.EvaluationError,))
        gate = E.GateRequest.from_plan_spec(self.plan)
        self.log.eq("evaluation", "gate_from_plan_spec",
                    (int(plan_formal.get("episodes", 0)),
                     int(plan_formal.get("min_successes", 0))),
                    (gate.episodes, gate.min_successes))
        self.log.eq("evaluation", "capability_gate_never_promoted", "not_run",
                    C.CAPABILITY_GATE_DEFAULT)
        self.log.eq("evaluation", "invalid_policy_is_full_batch", True,
                    "invalidate full batch"
                    in str(plan_formal.get("invalid_policy", "")))
        try:
            isolation = E.assert_judge_isolation()
        except Exception as exc:  # noqa: BLE001
            self.log.add("evaluation", "judge_isolation", False,
                         {"error": str(exc)[:300]})
        else:
            self.log.add("evaluation", "judge_isolation", True,
                         {"imports": isolation["imports"],
                          "sha256": isolation["sha256"],
                          "digest_scope": isolation["digest_scope"]})
        self.log.refuses("evaluation", "manual_aggregate_refused",
                         lambda: E.validate_episode(
                             {"episode_id": "e", "ancestor_id": "a", "seed": 1,
                              "split": "dev", "outcome": "clear",
                              "protocol_validity": "valid",
                              "policy_sha256": "a" * 64, "frames": 1,
                              "mean": 0.5},
                             index=0),
                         exceptions=(E.ManualAggregateError,))
        self.log.refuses("evaluation", "missing_episode_field_refused",
                         lambda: E.validate_episode({}, index=0),
                         exceptions=(E.MissingFieldError,))
        self.log.refuses("evaluation", "hand_edited_success_refused",
                         lambda: E.validate_episode(
                             {"episode_id": "e", "ancestor_id": "a", "seed": 1,
                              "split": "dev", "outcome": "game_over",
                              "protocol_validity": "valid",
                              "policy_sha256": "a" * 64, "frames": 1,
                              "success": True},
                             index=0),
                         exceptions=(E.InvalidRecordError,))
        self.log.refuses("evaluation", "missing_batch_refused",
                         lambda: E.load_batch(
                             self.repo / "nonexistent-batch.json"),
                         exceptions=(E.InvalidInputError,))

    # -- protocol digests -------------------------------------------------- #
    def protocol_digests(self, C: Any, E: Any) -> dict[str, Any]:
        blocks = {
            "observation": C.observation_parameters(),
            "action": C.action_parameters(),
            "reward": (self.plan.get("ppo") or {}).get("main_reward") or {},
            "terminal": C.contract_constants().get("transition") or {},
            "time": C.time_parameters(),
            "results": C.result_parameters(),
        }
        sources = {
            "observation": "touhou_rl/v430/contracts.py::observation_parameters()",
            "action": "touhou_rl/v430/contracts.py::action_parameters()",
            "reward": "PLAN_SPEC.json::ppo.main_reward",
            "terminal": "touhou_rl/v430/contracts.py::contract_constants().transition",
            "time": "touhou_rl/v430/contracts.py::time_parameters()",
            "results": "touhou_rl/v430/contracts.py::result_parameters()",
        }
        digests: dict[str, Any] = {}
        for name, block in blocks.items():
            payload = _canonical_bytes(block)
            digests[name] = {
                "sha256": E.digest_bytes(payload, scope=E.DIGEST_SCOPE_TEXT),
                "digest_scope": E.DIGEST_SCOPE_TEXT,
                "bytes": len(payload),
                "source": sources[name],
                "normalization": "canonical-json, sorted keys, separators=(',',':'); "
                                 "LF only, so CRLF->LF is a no-op",
            }
        digests["observation"]["o2_digest"] = C.o2_digest()
        composite_parts = {name: digests[name]["sha256"]
                           for name in PROTOCOL_DIGEST_PARTS}
        digests["composite"] = {
            "sha256": E.digest_bytes(_canonical_bytes(composite_parts),
                                     scope=E.DIGEST_SCOPE_TEXT),
            "digest_scope": E.DIGEST_SCOPE_TEXT,
            "parts": composite_parts,
            "note": "sha256 over the four protocol digests (observation/action/"
                    "reward/terminal)",
        }
        self.log.eq("digest", "four_protocol_parts", list(PROTOCOL_DIGEST_PARTS),
                    [name for name in digests if name in PROTOCOL_DIGEST_PARTS])
        self.log.eq("digest", "reward_digest_nonempty",
                    bool((self.plan.get("ppo") or {}).get("main_reward")),
                    bool(digests["reward"]["bytes"]))
        for name in PROTOCOL_DIGEST_PARTS:
            self.log.eq("digest", f"scope:{name}", E.DIGEST_SCOPE_TEXT,
                        digests[name]["digest_scope"])
        return digests

    # -- driver ------------------------------------------------------------ #
    def execute(self) -> tuple[dict, int]:
        C, A, E, S = self.load_modules()
        self.collect_identities(E)
        self.check_plan_spec()
        self.check_constants(C)
        self.check_action_space(C, A)
        self.check_probability(C, A)
        self.check_observation(C)
        self.check_privileged(C, S)
        self.check_time(C)
        self.check_terminal(C)
        self.check_results(C)
        self.check_session(C, S)
        self.check_evaluation(C, E, S)
        digests = self.protocol_digests(C, E)

        summary = self.log.summary()
        failed = self.log.failures()
        code = EXIT_OK if not failed else EXIT_GATE
        self.gaps.extend([
            "validate-contract checks code/config/plan consistency only; it starts "
            "no engine and produces no game evidence.",
            "G430-E0 is not passed here: W01 still needs real success/failure "
            "replays through this contract on the asset host.",
            "capability_gate stays not_run: a protocol check is not a capability.",
        ])
        for row in failed[:10]:
            self.gaps.append(f"failed check {row['group']}/{row['name']}: "
                             f"{json.dumps(row['detail'], ensure_ascii=False)[:220]}")
        if not self.plan:
            self.gaps.append("PLAN_SPEC.json was not readable, so the plan-side "
                             "cross-checks are missing rather than passing.")

        finished = _now()
        result = {
            "schema": CONTRACT_RESULT_SCHEMA,
            "experiment_id": "v430-w01-contract-001",
            "run_id": self.out.stem,
            "phase": "W01",
            "mode": "contract_validation",
            "command": "validate-contract",
            "command_class": "contract",
            "spec_revision": self.plan.get("revision") or self.cfg.get("spec_revision"),
            "config_frozen_at": self.cfg.get("frozen_at"),
            "started_at": _iso(self.started),
            "finished_at": _iso(finished),
            "duration_seconds": round((finished - self.started).total_seconds(), 3),
            "timezone": finished.strftime("%z"),
            "argv": self.argv,
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "output_path": str(self.out),
            "exit_code": code,
            "execution_status": "ok" if code == EXIT_OK else "gate_fail",
            "protocol_validity": "valid",
            "research_result": {
                "kind": "contract_validation",
                "status": "pass" if code == EXIT_OK else "fail",
                "requested_gate": ("every config<->module constant, action mask, "
                                   "observation shape, privileged-field, time and "
                                   "terminal check agrees"),
                "passed": code == EXIT_OK,
                "checks": self.log.rows,
                "summary": summary,
            },
            "capability_gate": "not_run",
            "capability_gate_detail": {
                "status": "not_run",
                "reason": ("validate-contract is a protocol/consistency check "
                           "(command_class='contract'). It cannot report "
                           "capability_gate='pass': contracts.assert_result_schema "
                           "refuses pass/fail without an independent capability "
                           "judgement, and this command has none."),
                "gates": {"G430-E0": "not_run", "G410-0": "not_reassessed"},
            },
            "contract_version": C.CONTRACT_VERSION,
            "protocol_digests": digests,
            "identities": self.identities,
            "identity_count": len(E.assert_identity_scopes(self.identities)),
            "remaining_gaps": self.gaps,
            "notes": self.notes,
        }
        # The guard. contracts.assert_result_schema() is what refuses a
        # capability claim here; the write path re-checks it before creating the
        # file (see write_checked_result).
        C.assert_result_schema(result)
        return result, code


def write_checked_result(out: Path, payload: dict) -> tuple[bool, str | None]:
    """Validate a result with ``assert_result_schema``, then create ``out``.

    The order is the point: the ``capability_gate`` guard lives in
    ``contracts.assert_result_schema``, so a code path that constructs a result
    and writes it directly would bypass the guard.  Nothing touches the
    filesystem until the schema check has passed, and ``emit_result`` still
    refuses an existing path.
    """

    try:
        from touhou_rl.v430 import contracts as contracts_module
    except Exception as exc:  # noqa: BLE001 - an unimportable contract is invalid
        return False, f"cannot import touhou_rl.v430.contracts: {exc}"
    try:
        contracts_module.assert_result_schema(payload)
    except contracts_module.ContractError as exc:
        return False, (f"refusing to write a result that fails "
                       f"assert_result_schema: {exc}; nothing was written")
    error = emit_result(out, payload)
    return (error is None), error


def cmd_validate_contract(args: argparse.Namespace) -> int:
    out = Path(args.out)
    started = _now()
    argv = list(sys.argv)
    if out.exists():
        _err(f"refusing to overwrite existing output: {out}")
        return EXIT_INVALID

    cfg_path = Path(args.config)
    try:
        cfg = load_contract_config(cfg_path)
    except ConfigError as exc:
        _err(str(exc))
        return EXIT_INVALID

    repo = _repo_root()
    run = ContractValidation(cfg=cfg, cfg_path=cfg_path, repo=repo, out=out,
                             started=started, argv=argv)
    try:
        result, code = run.execute()
    except ConfigError as exc:
        _err(str(exc))
        return EXIT_INVALID
    ok, message = write_checked_result(out, result)
    if not ok:
        _err(message or "cannot write result")
        return EXIT_INVALID
    summary = result["research_result"]["summary"]
    print(
        "validate-contract: "
        f"checks {summary['checks']} | failed {summary['failed']} | "
        f"groups {len(summary['groups'])} | "
        f"gate {'passed' if result['research_result']['passed'] else 'NOT met'} | "
        f"exit {code}"
    )
    if code != EXIT_OK:
        for row in run.log.failures()[:10]:
            print(f"failed: {row['group']}/{row['name']}", file=sys.stderr)
    print(f"result: {out}")
    return code


def load_contract_config(path: Path) -> dict:
    """Parse and structurally validate ``configs/v430/contracts.json`` (exit 2)."""

    _need(path.is_file(), f"config not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config {path} is not valid JSON: {exc}") from exc
    _need(isinstance(raw, dict), "config root must be a JSON object")
    _need(raw.get("schema") == CONTRACT_CONFIG_SCHEMA,
          f"config schema must be {CONTRACT_CONFIG_SCHEMA!r}, got {raw.get('schema')!r}")
    for key in ("spec_revision", "frozen_at", "constants", "plan_spec"):
        _need(key in raw, f"config is missing required field '{key}'")
    _need_str(raw, "spec_revision", "config")
    _need(isinstance(raw["constants"], dict), "config.constants must be an object")
    _need(isinstance(raw["plan_spec"], dict), "config.plan_spec must be an object")
    _need_str(raw["plan_spec"], "path", "plan_spec")
    _need_str(raw["plan_spec"], "schema_expected", "plan_spec")
    _need_str(raw["plan_spec"], "revision_expected", "plan_spec")
    raw.setdefault("experiment_id", "v430-w01-contract-001")
    return raw


# --------------------------------------------------------------------------- #
# validate-splits (W09)
# --------------------------------------------------------------------------- #
#: Schema of the evidence document this subcommand writes.  It is deliberately
#: *not* ``lineage.REPORT_SCHEMA`` (``v430-splits-validation/1``): the pure
#: function's report is embedded verbatim under ``research_result``, so the
#: document and the report must be distinguishable by schema.
SPLITS_RESULT_SCHEMA = "v430-validate-splits/1"
#: The only config schema this subcommand accepts (``configs/v430/splits.json``).
SPLITS_CONFIG_SCHEMA = "v430-splits/1"
#: Registered ``contracts.COMMAND_CLASSES`` value closest to this command: W09
#: validates a *data* registry.  It is not ``"contract"`` (that is W01's
#: config<->module protocol check) and not ``"evaluation"`` (no episode is run).
SPLITS_COMMAND_CLASS = "data"
#: ``lineage`` problem codes that mean "the input cannot be measured at all"
#: (exit 2) rather than "a valid registry was measured and rejected" (exit 3).
SPLITS_INVALID_INPUT_CODES = frozenset({"P_CONFIG_UNREADABLE", "P_SCHEMA"})


def load_splits_config(path: Path) -> dict:
    """Parse ``configs/v430/splits.json`` (exit 2 on any input failure).

    Only *input* validity is decided here (readable, JSON object, declared
    schema).  Every lineage rule stays in ``lineage.validate_splits``: this
    module must not second-guess, relax or duplicate them.
    """

    _need(path.is_file(), f"config not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config {path} is not valid JSON: {exc}") from exc
    _need(isinstance(raw, dict), "config root must be a JSON object")
    _need(raw.get("schema") == SPLITS_CONFIG_SCHEMA,
          f"config schema must be {SPLITS_CONFIG_SCHEMA!r}, got {raw.get('schema')!r}")
    return raw


def _text_identity(path: Path, lineage_module: Any) -> dict:
    """``sha256`` **plus** its scope for a text product (``text/lf-normalized``)."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        return {"path": str(path), "sha256": None, "digest_scope": None,
                "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": lineage_module.text_lf_digest(payload),
        "digest_scope": lineage_module.DIGEST_SCOPE_TEXT,
    }


class SplitsValidation:
    """Run ``lineage.validate_splits`` and shape its report into a result JSON.

    Nothing is measured here: the validator reads the frozen registries and the
    config, and this class only records the decision it returned.  The
    ``capability_gate`` therefore stays ``not_run`` - a lineage validation is not
    a capability measurement (``contracts.assert_result_schema`` is what enforces
    that a non-judged command cannot claim otherwise).
    """

    def __init__(self, *, cfg: dict, cfg_path: Path, repo: Path, out: Path,
                 started: datetime, argv: list[str]) -> None:
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.repo = repo
        self.out = out
        self.started = started
        self.argv = argv
        self.notes: list[str] = []
        self.gaps: list[str] = []

    def execute(self) -> tuple[dict, int]:
        try:
            from touhou_rl.v430.data import lineage as lineage_module
        except Exception as exc:  # noqa: BLE001 - an unimportable validator is invalid
            raise ConfigError(
                f"cannot import touhou_rl.v430.data.lineage: "
                f"{type(exc).__name__}: {exc}") from exc

        identities: dict[str, Any] = {
            "splits_config": _text_identity(self.cfg_path, lineage_module),
            "lineage_module": _text_identity(
                self.repo / "touhou_rl/v430/data/lineage.py", lineage_module),
            "cli": _text_identity(self.repo / "tools/rl/v430.py", lineage_module),
        }

        try:
            report = lineage_module.validate_splits(str(self.cfg_path), strict=True)
        except Exception as exc:  # noqa: BLE001 - the validator itself must not crash
            raise ConfigError(
                f"lineage.validate_splits raised {type(exc).__name__}: {exc}") from exc
        if not isinstance(report, dict):
            raise ConfigError(
                "lineage.validate_splits did not return a mapping "
                f"(got {type(report).__name__}); no legal result can be built")

        problems = [row for row in (report.get("problems") or []) if isinstance(row, dict)]
        codes = [str(row.get("code")) for row in problems]
        passed = report.get("passed") is True
        # A registry that cannot be read at all was never measured: that is an
        # invalid input (exit 2), not a gate failure (exit 3).
        unusable = sorted(set(codes) & SPLITS_INVALID_INPUT_CODES)
        if unusable:
            raise ConfigError(
                "the split config could not be measured at all "
                f"(lineage problem codes {unusable}); nothing was written")

        code = EXIT_OK if (passed and not problems) else EXIT_GATE
        scope = report.get("split_registry_sha_scope")
        digest = report.get("split_registry_sha")
        stats = report.get("statistics") or {}

        self.notes.extend([
            "lineage.validate_splits is called with strict=True; the CLI has no "
            "flag that relaxes it and none of its rules live in this file.",
            "lineage.exit_code_for() returns 2 for a rejected registry because the "
            "pure function classifies its own verdict as an input/protocol "
            "decision.  This CLI maps 'the measurement was valid but the requested "
            "gate is not met' to exit 3 (EXECUTION_PLAN.md section 4), so the "
            "verdict is carried in research_result.status while the process exit "
            "code follows the plan.",
            "split_registry_sha is the digest of the whole frozen config "
            "(sorted-key compact JSON, text/lf-normalized).  It is recorded here "
            "with split_registry_sha_scope so no bare digest travels alone.",
            "The result file is created exclusively; an existing --out path is "
            "refused with exit 2 and left untouched.",
        ])
        self.gaps.extend([
            "validate-splits reads the config and the frozen registries only: it "
            "opens no asset, no pixel payload and no sealed record, and it starts "
            "no engine and no training.",
            "capability_gate stays 'not_run': a split/lineage validation is not a "
            "capability measurement (command_class='data').",
            "G410-6 is not passed here: this command only proves the frozen split "
            "registry is admissible; the clean hold-out adaptation still needs "
            "W10's real episodes on the two frozen families.",
        ])
        for row in problems[:10]:
            self.gaps.append(
                "rejected lineage problem "
                f"{row.get('code')}: {str(row.get('message'))[:220]}")

        finished = _now()
        result = {
            "schema": SPLITS_RESULT_SCHEMA,
            "experiment_id": (report.get("experiment_id")
                              or self.cfg.get("experiment_id")
                              or "v430-w09-clean-holdout-splits-001"),
            "run_id": self.out.stem,
            "phase": "W09",
            "mode": "lineage_validation",
            "command": "validate-splits",
            "command_class": SPLITS_COMMAND_CLASS,
            "spec_revision": report.get("spec_revision") or self.cfg.get("spec_revision"),
            "config_path": str(self.cfg_path),
            "config_schema": report.get("config_schema") or self.cfg.get("schema"),
            "config_frozen_at": report.get("frozen_at") or self.cfg.get("frozen_at"),
            "started_at": _iso(self.started),
            "finished_at": _iso(finished),
            "duration_seconds": round((finished - self.started).total_seconds(), 3),
            "timezone": finished.strftime("%z"),
            "argv": self.argv,
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "output_path": str(self.out),
            "exit_code": code,
            "execution_status": "ok" if code == EXIT_OK else "gate_fail",
            "protocol_validity": "valid",
            "split_mode": report.get("mode"),
            "split_registry_sha": digest,
            "digest_scope": scope,
            "split_registry_sha_scope": scope,
            "split_registry_sha_canonicalization":
                report.get("split_registry_sha_canonicalization"),
            "research_result": {
                "kind": "split_registry_validation",
                "status": "pass" if code == EXIT_OK else "fail",
                "requested_gate": (
                    "the frozen train/dev/confirm/adapt lineage is admissible under "
                    "every strict lineage rule (no crossing, no touched hold-out, "
                    "no banned seed, no sealed reference, no digest without scope)"),
                "passed": code == EXIT_OK,
                "mode": report.get("mode"),
                "lineage_report_schema": report.get("schema"),
                "lineage_exit_code": report.get("exit_code"),
                "problems": problems,
                "problem_count": len(problems),
                "problem_codes": codes,
                "degradations": list(report.get("degradations") or []),
                "statistics": stats,
                "checks_performed": list(report.get("checks_performed") or []),
                "not_checked": list(report.get("not_checked") or []),
                "split_registry_sha": digest,
                "digest_scope": scope,
            },
            "capability_gate": "not_run",
            "capability_gate_detail": {
                "status": "not_run",
                "reason": (
                    "a split/lineage validation measures no capability: it decides "
                    "whether a frozen data registry is admissible. Only an "
                    "independent capability judgement may promote this field, and "
                    "this command has none."),
                "gates": {"G430-E0": "not_run", "G410-6": "not_run"},
            },
            "identities": identities,
            "statistics": {
                "problem_count": len(problems),
                "degradation_count": len(report.get("degradations") or []),
                "checks_performed": len(report.get("checks_performed") or []),
                "not_checked": len(report.get("not_checked") or []),
                "splits": stats.get("splits"),
                "ancestors": stats.get("ancestors"),
                "holdout_families_frozen": stats.get("holdout_families_frozen"),
                "pretraining_sources": stats.get("pretraining_sources"),
                "seed_range_check": stats.get("seed_range_check"),
            },
            "remaining_gaps": self.gaps,
            "notes": self.notes,
        }
        # Guard before the write path runs (write_checked_result re-checks it):
        # a result that cannot pass contracts.assert_result_schema is never
        # handed to emit_result, so nothing is written first and validated after.
        from touhou_rl.v430 import contracts as contracts_module
        contracts_module.assert_result_schema(result)
        return result, code


def cmd_validate_splits(args: argparse.Namespace) -> int:
    out = Path(args.out)
    started = _now()
    argv = list(sys.argv)
    if out.exists():
        _err(f"refusing to overwrite existing output: {out}")
        return EXIT_INVALID

    cfg_path = Path(args.config)
    try:
        cfg = load_splits_config(cfg_path)
    except ConfigError as exc:
        _err(str(exc))
        return EXIT_INVALID

    repo = _repo_root()
    run = SplitsValidation(cfg=cfg, cfg_path=cfg_path, repo=repo, out=out,
                           started=started, argv=argv)
    try:
        result, code = run.execute()
    except ConfigError as exc:
        _err(str(exc))
        return EXIT_INVALID
    ok, message = write_checked_result(out, result)
    if not ok:
        _err(message or "cannot write result")
        return EXIT_INVALID
    research = result["research_result"]
    print(
        "validate-splits: "
        f"mode {result['split_mode']} | problems {research['problem_count']} | "
        f"splits {len((research['statistics'].get('splits') or {}))} | "
        f"sha {str(result['split_registry_sha'])[:12]} | "
        f"gate {'passed' if research['passed'] else 'NOT met'} | exit {code}"
    )
    if code != EXIT_OK:
        for row in research["problems"][:10]:
            print(f"problem {row.get('code')}: {row.get('message')}", file=sys.stderr)
    print(f"result: {out}")
    return code


# --------------------------------------------------------------------------- #
# train-bc (W03G)
# --------------------------------------------------------------------------- #
#: Schema of the evidence document ``train-bc`` writes.
TRAIN_BC_RESULT_SCHEMA = "v430-train-bc/1"
#: Schema of the ``--batches`` specification file.  Synthetic batches only: this
#: slice never reads a corpus (the v4.2.0 corpus is a legacy contract and cannot
#: be fed to an O2 actor).
TRAIN_BC_BATCHES_SCHEMA = "v430-synthetic-batches/1"
#: The only dataset kind this command can train on, recorded verbatim so a
#: synthetic run can never be read as a real-corpus run.
TRAIN_BC_DATASET_KIND = "synthetic"
#: ``contracts.COMMAND_CLASSES`` value.  ``train-bc`` trains; it never judges a
#: capability, so ``capability_gate`` stays ``not_run``.
TRAIN_BC_COMMAND_CLASS = "training"
#: Typo guards for the specification (a bad number is refused, not allocated).
TRAIN_BC_MAX_BATCHES = 256
TRAIN_BC_MAX_BATCH_SIZE = 64
TRAIN_BC_MAX_STEPS = 16
TRAIN_BC_DEFAULT_CAPTURE_S = 0.016
TRAIN_BC_IMAGE_FIELDS: tuple[str, ...] = ("global_rgb", "play_gray", "hud_gray")


def load_batches_spec(path: Path) -> dict:
    """Parse and validate the ``--batches`` specification (exit 2 on failure).

    The file describes **synthetic** batches.  Nothing is read from a corpus and
    no tensor is loaded from disk: every observation and label is generated
    inside the CLI from the file's own ``seed`` with an explicit
    ``torch.Generator``, so the same file yields bit-identical batches.

    Format (schema ``v430-synthetic-batches/1``)::

        {
          "schema": "v430-synthetic-batches/1",   # optional; must match if present
          "n_batches": 4,                         # required; 1..256
          "batch_size": 2,                        # required; 1..64
          "seed": 1234,                           # required; any int (tensor seed)
          "steps": 1,                             # optional; 1..16 (GRU sequences)
          "elapsed_capture_s": 0.016,             # optional; finite and > 0
          "observation_shapes": {                 # optional; must equal O2_SHAPES
            "global_rgb": [3, 240, 320],
            "play_gray": [4, 224, 192],
            "hud_gray": [1, 448, 160],
            "executed_key_history": [8, 7],
            "history_valid": [8]
          }
        }

    ``observation_shapes`` is a *declaration*, not an override: every shape in it
    is compared with :data:`touhou_rl.v430.observation.O2_SHAPES` and a single
    disagreement is refused (the model never interpolates an observation into
    another shape).  Image fields are filled with ``uint8`` in ``0..255``,
    ``executed_key_history`` with ``uint8`` ``0/1``, ``history_valid`` with a
    cold-start all-``False`` window and ``elapsed_capture_s`` with one float32
    interval per sample.  ``steps`` is the sequence length for
    ``--memory-mode gru`` and must be ``1`` for ``--memory-mode none``.
    """

    _need(path.is_file(), f"batches spec not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read batches spec {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"batches spec {path} is not valid JSON: {exc}") from exc
    _need(isinstance(raw, dict), "batches spec root must be a JSON object")
    if "schema" in raw:
        _need(raw["schema"] == TRAIN_BC_BATCHES_SCHEMA,
              f"batches spec schema must be {TRAIN_BC_BATCHES_SCHEMA!r}, "
              f"got {raw['schema']!r}")
    for key in ("n_batches", "batch_size", "seed"):
        _need(key in raw, f"batches spec is missing required field '{key}'")

    spec: dict[str, Any] = {
        "n_batches": _batch_int(raw["n_batches"], "n_batches", 1, TRAIN_BC_MAX_BATCHES),
        "batch_size": _batch_int(raw["batch_size"], "batch_size", 1,
                                 TRAIN_BC_MAX_BATCH_SIZE),
        "seed": _plain_int(raw["seed"], "seed"),
        "steps": 1,
        "elapsed_capture_s": TRAIN_BC_DEFAULT_CAPTURE_S,
    }
    if "steps" in raw:
        spec["steps"] = _batch_int(raw["steps"], "steps", 1, TRAIN_BC_MAX_STEPS)
    if "elapsed_capture_s" in raw:
        value = raw["elapsed_capture_s"]
        _need(not isinstance(value, bool) and isinstance(value, (int, float)),
              f"batches spec: 'elapsed_capture_s' must be a number, got {value!r}")
        _need(math.isfinite(float(value)) and float(value) > 0.0,
              f"batches spec: 'elapsed_capture_s' must be finite and > 0, got {value!r}")
        spec["elapsed_capture_s"] = float(value)

    declared = raw.get("observation_shapes")
    if declared is not None:
        _need(isinstance(declared, dict), "batches spec: 'observation_shapes' must be an object")
        try:
            from touhou_rl.v430 import observation as observation_module
        except Exception as exc:  # noqa: BLE001 - an unimportable encoder is invalid
            raise ConfigError(
                f"cannot import touhou_rl.v430.observation: {type(exc).__name__}: {exc}") from exc
        known = {name: list(shape) for name, shape in observation_module.O2_SHAPES.items()}
        for name, shape in declared.items():
            _need(name in known,
                  f"batches spec: observation_shapes has unknown field {name!r}; "
                  f"known {sorted(known)}")
            _need(isinstance(shape, list) and shape,
                  f"batches spec: observation_shapes.{name} must be a non-empty list")
            _need(all(not isinstance(dim, bool) and isinstance(dim, int) for dim in shape),
                  f"batches spec: observation_shapes.{name} must hold integers, "
                  f"got {shape!r}")
            _need(shape == known[name],
                  f"batches spec: observation_shapes.{name} {shape} != frozen O2 shape "
                  f"{known[name]}; the model never reshapes or interpolates an "
                  f"observation into another shape")
        spec["observation_shapes"] = {name: list(shape) for name, shape in declared.items()}
    return spec


def _batch_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    number = _plain_int(value, name)
    _need(minimum <= number <= maximum,
          f"batches spec: '{name}' must be in [{minimum}, {maximum}], got {number}")
    return number


def _plain_int(value: Any, name: str) -> int:
    _need(not isinstance(value, bool) and isinstance(value, int),
          f"batches spec: '{name}' must be an integer, got {value!r}")
    return int(value)


def _training_stack() -> dict[str, Any]:
    """Import the v4.3.0 training stack; an unimportable stack is invalid input."""

    try:
        import torch  # noqa: PLC0415 - deliberately deferred: audit must not need torch
        from touhou_rl.v430 import contracts as contracts_module
        from touhou_rl.v430 import learn as learn_module
        from touhou_rl.v430 import observation as observation_module
        from touhou_rl.v430.models import actor as actor_module
    except Exception as exc:  # noqa: BLE001 - any import failure means "cannot train"
        raise ConfigError(
            f"cannot import the v4.3.0 training stack: {type(exc).__name__}: {exc}") from exc
    if TRAIN_BC_COMMAND_CLASS not in contracts_module.TRAINING_COMMAND_CLASSES:
        raise ConfigError(
            f"contracts.TRAINING_COMMAND_CLASSES no longer contains "
            f"{TRAIN_BC_COMMAND_CLASS!r}; the anti-promotion guard is not wired")
    return {
        "torch": torch,
        "contracts": contracts_module,
        "learn": learn_module,
        "observation": observation_module,
        "actor": actor_module,
    }


def synthetic_batches(stack: dict[str, Any], spec: dict[str, Any], *,
                      memory_mode: str) -> list[dict[str, Any]]:
    """Generate the batches the specification describes (deterministic, uint8).

    Every tensor comes from the one explicit generator seeded with
    ``spec["seed"]``; the global RNG is never consulted, so the returned batches
    are a pure function of ``(spec, memory_mode)``.  Observations are ``uint8``
    exactly as ``observation.O2_SHAPES`` requires — never floats, because the
    ``/255`` normalisation belongs inside the model.
    """

    torch = stack["torch"]
    shapes = stack["observation"].O2_SHAPES
    size = int(spec["batch_size"])
    steps = int(spec["steps"])
    if memory_mode == "none" and steps != 1:
        raise ConfigError(
            f"batches spec: 'steps'={steps} with --memory-mode none; a sequence of "
            f"observations needs memory_mode='gru' (only the last step would be "
            f"trained otherwise)")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec["seed"]))
    interval = float(spec["elapsed_capture_s"])
    batches: list[dict[str, Any]] = []
    for _index in range(int(spec["n_batches"])):
        observations = []
        targets = []
        for _step in range(steps):
            sample: dict[str, Any] = {}
            for name in TRAIN_BC_IMAGE_FIELDS:
                shape = (size,) + tuple(int(dim) for dim in shapes[name])
                sample[name] = torch.randint(0, 256, shape,
                                             generator=generator).to(torch.uint8)
            sample["executed_key_history"] = torch.randint(
                0, 2, (size,) + tuple(int(dim) for dim in shapes["executed_key_history"]),
                generator=generator).to(torch.uint8)
            sample["history_valid"] = torch.zeros(
                (size,) + tuple(int(dim) for dim in shapes["history_valid"]),
                dtype=torch.bool)
            sample["elapsed_capture_s"] = torch.full((size,), interval, dtype=torch.float32)
            observations.append(sample)
            targets.append({
                head: torch.randint(0, int(stack["contracts"].HEAD_SIZES[head]), (size,),
                                    generator=generator, dtype=torch.int64)
                for head in stack["contracts"].HEAD_ORDER
            })
        batches.append({"observations": observations if memory_mode == "gru"
                        else observations[0],
                        "targets": targets if memory_mode == "gru" else targets[0]})
    return batches


def batches_signature(batches: Sequence[dict[str, Any]]) -> str:
    """sha256 over every generated tensor's bytes, in a fixed order.

    This is the audit hook for "the same seed really produced the same data": two
    runs of the same specification must report the same digest, and it is what the
    result records next to the per-step losses.
    """

    digest = hashlib.sha256()
    for index, batch in enumerate(batches):
        digest.update(f"batch:{index}:".encode("utf-8"))
        observations = batch["observations"]
        targets = batch["targets"]
        obs_steps = list(observations) if isinstance(observations, list) else [observations]
        target_steps = list(targets) if isinstance(targets, list) else [targets]
        for step_index, (observation_row, target_row) in enumerate(
                zip(obs_steps, target_steps)):
            digest.update(f"step:{step_index}:".encode("utf-8"))
            for name in sorted(observation_row):
                digest.update(name.encode("utf-8"))
                tensor = observation_row[name]
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            for head in sorted(target_row):
                digest.update(head.encode("utf-8"))
                digest.update(target_row[head].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TrainBCRun:
    """One ``train-bc`` execution: synthetic batches -> actor -> TrainingReport."""

    def __init__(self, *, spec: dict, batches_path: Path, out: Path, started: datetime,
                 argv: list[str], epochs: int, lr: float, seed: int,
                 max_steps: int | None, memory_mode: str, shuffle: bool,
                 checkpoint_out: Path | None):
        self.spec = spec
        self.batches_path = batches_path
        self.out = out
        self.started = started
        self.argv = argv
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.max_steps = max_steps
        self.memory_mode = memory_mode
        self.shuffle = shuffle
        self.checkpoint_out = checkpoint_out

    # -- training ---------------------------------------------------------- #
    def _write_checkpoint(self, learn_module: Any, model: Any, report: Any) -> dict | None:
        if self.checkpoint_out is None:
            return None
        try:
            written = learn_module.save_checkpoint(model, self.checkpoint_out, extra={
                "command": "train-bc",
                "command_class": TRAIN_BC_COMMAND_CLASS,
                "dataset_kind": TRAIN_BC_DATASET_KIND,
                "memory_mode": report.memory_mode,
                "seed": int(report.seed),
                "epochs": int(report.epochs),
                "lr": float(report.lr),
                "n_steps": int(report.n_steps),
                "shuffle_batches": bool(report.shuffle_batches),
            })
        except learn_module.LearnError as exc:
            raise ConfigError(
                f"cannot write --checkpoint-out {self.checkpoint_out}: {exc}") from exc
        target = Path(written)
        return {
            "path": str(target),
            "sha256": sha256_file(target),
            "size_bytes": target.stat().st_size,
            "o2_digest": None,  # filled by the caller from contracts.o2_digest()
        }

    def execute(self, stack: dict[str, Any]) -> tuple[dict, int]:
        torch = stack["torch"]
        contracts_module = stack["contracts"]
        learn_module = stack["learn"]
        actor_module = stack["actor"]

        batches = synthetic_batches(stack, self.spec, memory_mode=self.memory_mode)
        signature = batches_signature(batches)
        # Determinism is a promise this result makes, so the two knobs that break
        # it are pinned here and recorded: one CPU thread (a multi-threaded
        # reduction order changes the last bits of a gradient) and an explicit
        # manual_seed before the actor is built (the starting weights must be a
        # function of --seed, otherwise "same seed, same losses" is an accident).
        torch.set_num_threads(1)
        torch.manual_seed(int(self.seed))
        model = actor_module.build_actor()
        parameters_before = actor_module.parameter_report(model)["total"]
        try:
            report = learn_module.train_bc(
                model, batches, epochs=self.epochs, lr=self.lr, seed=self.seed,
                memory_mode=self.memory_mode, max_steps=self.max_steps,
                shuffle_batches=self.shuffle)
        except learn_module.LearnError as exc:
            raise ConfigError(f"train_bc refused the run: {exc}") from exc
        parameters = actor_module.parameter_report(model)
        if parameters["total"] != parameters_before:  # pragma: no cover - structural
            raise ConfigError(
                f"the parameter count changed during training "
                f"({parameters_before} -> {parameters['total']}); the actor grew")
        checkpoint_row = self._write_checkpoint(learn_module, model, report)
        o2 = contracts_module.o2_digest()
        if checkpoint_row is not None:
            checkpoint_row["o2_digest"] = o2

        losses = list(report.losses)
        finished = _now()
        tool_path = Path(__file__).resolve()
        result = contracts_module.new_result(
            execution_status="ok",
            protocol_validity="valid",
            capability_gate=contracts_module.CAPABILITY_GATE_DEFAULT,
            research_result={
                "status": "not_run",
                "kind": "behaviour_cloning",
                "dataset_kind": TRAIN_BC_DATASET_KIND,
                "passed": False,
                "requested_gate": None,
                "note": ("train-bc runs behaviour cloning on synthetic batches and "
                         "reports the run; it claims no research result and no "
                         "capability (capability_gate='not_run')"),
            },
            **{
                "schema": TRAIN_BC_RESULT_SCHEMA,
                "command": "train-bc",
                "command_class": TRAIN_BC_COMMAND_CLASS,
                "owner": "W03G",
                "started_at": _iso(self.started),
                "finished_at": _iso(finished),
                "duration_seconds": round((finished - self.started).total_seconds(), 3),
                "argv": self.argv,
                "cwd": str(Path.cwd()),
                "python_executable": sys.executable,
                "torch_version": str(torch.__version__),
                "output_path": str(self.out),
                "exit_code": EXIT_OK,
                "dataset_kind": TRAIN_BC_DATASET_KIND,
                "batches": {
                    "path": str(self.batches_path),
                    "sha256": sha256_file(self.batches_path),
                    "spec": dict(self.spec),
                    "signature_sha256": signature,
                    "generator": ("torch.Generator(device='cpu') seeded with "
                                  "spec.seed; the global RNG is not consulted"),
                    "observation_dtype": "uint8",
                    "label_dtype": "int64",
                },
                "memory_mode": report.memory_mode,
                "epochs": report.epochs,
                "lr": report.lr,
                "seed": report.seed,
                "determinism": {
                    "torch_num_threads": int(torch.get_num_threads()),
                    "global_seed_for_initial_weights": int(self.seed),
                    "batch_generator": "explicit torch.Generator(device='cpu')",
                    "global_rng_used_for_training": False,
                },
                "optimizer": report.optimizer,
                "shuffle_batches": report.shuffle_batches,
                "max_steps": report.max_steps,
                "n_steps": report.n_steps,
                "steps": [dict(step) for step in report.steps],
                "losses": losses,
                "first_loss": losses[0] if losses else None,
                "last_loss": losses[-1] if losses else None,
                "n_batches": int(self.spec["n_batches"]),
                "batch_size": int(self.spec["batch_size"]),
                "o2_digest": o2,
                "parameter_report": parameters,
                "trainable_parameter_count": parameters["total"],
                "checkpoint": checkpoint_row,
                "identities": {
                    "cli": {"path": "tools/rl/v430.py", "sha256": sha256_file(tool_path)},
                    "batches_spec": {"path": str(self.batches_path),
                                     "sha256": sha256_file(self.batches_path)},
                    "o2_digest": o2,
                    "contract_version": contracts_module.CONTRACT_VERSION,
                    "spec_revision": contracts_module.SPEC_REVISION,
                    "runtime": {"python_executable": sys.executable,
                                "python_version": platform.python_version(),
                                "platform": platform.platform(),
                                "torch_version": str(torch.__version__)},
                },
                "remaining_gaps": [
                    ("dataset_kind='synthetic': no real corpus was read, so these "
                     "losses say nothing about the game and are not an evaluation"),
                    ("capability_gate='not_run': a training exit 0 is not a "
                     "capability; no G430 gate was judged"),
                    ("no checkpoint was written (--checkpoint-out was not given)"
                     if checkpoint_row is None else
                     f"checkpoint written to {checkpoint_row['path']}; it was not "
                     f"reloaded and re-evaluated in this slice"),
                    "no episode was run: no engine, no game, no evaluation entry point",
                ],
                "notes": [
                    ("the batches are generated inside the CLI from the spec's seed; "
                     "the same spec reproduces the same signature and the same losses"),
                    ("observations are uint8 in the frozen O2 shapes; the /255 "
                     "normalisation happens inside the model"),
                ],
            },
        )
        # Guard before the write path (write_checked_result re-checks it): a result
        # that cannot pass assert_result_schema is never handed to emit_result.
        try:
            contracts_module.assert_result_schema(result)
        except contracts_module.ContractError as exc:
            raise ConfigError(
                f"internal: the train-bc result fails assert_result_schema: {exc}") from exc
        return result, EXIT_OK


def cmd_train_bc(args: argparse.Namespace) -> int:
    out = Path(args.out)
    started = _now()
    argv = list(sys.argv)
    if out.exists():
        _err(f"refusing to overwrite existing output: {out}")
        return EXIT_INVALID
    checkpoint_out = Path(args.checkpoint_out) if args.checkpoint_out else None
    if checkpoint_out is not None and checkpoint_out.exists():
        _err(f"refusing to overwrite existing checkpoint: {checkpoint_out}")
        return EXIT_INVALID

    try:
        stack = _training_stack()
        memory_mode = str(args.memory_mode)
        _need(memory_mode in stack["learn"].MEMORY_MODES,
              f"--memory-mode {memory_mode!r} is not one of "
              f"{list(stack['learn'].MEMORY_MODES)}")
        spec = load_batches_spec(Path(args.batches))
        _need(not isinstance(args.epochs, bool) and int(args.epochs) >= 1,
              f"--epochs must be >= 1, got {args.epochs}")
        _need(math.isfinite(float(args.lr)) and float(args.lr) > 0.0,
              f"--lr must be finite and > 0, got {args.lr}")
        max_steps = None if args.max_steps is None else int(args.max_steps)
        _need(max_steps is None or max_steps >= 1,
              f"--max-steps must be >= 1 when given, got {args.max_steps}")
        run = TrainBCRun(spec=spec, batches_path=Path(args.batches), out=out,
                         started=started, argv=argv, epochs=int(args.epochs),
                         lr=float(args.lr), seed=int(args.seed), max_steps=max_steps,
                         memory_mode=memory_mode, shuffle=bool(args.shuffle),
                         checkpoint_out=checkpoint_out)
        result, code = run.execute(stack)
    except ConfigError as exc:
        _err(str(exc))
        return EXIT_INVALID

    ok, message = write_checked_result(out, result)
    if not ok:
        _err(message or "cannot write result")
        return EXIT_INVALID
    print(
        "train-bc: "
        f"dataset {result['dataset_kind']} | memory {result['memory_mode']} | "
        f"steps {result['n_steps']} | last loss {result['last_loss']:.6f} | "
        f"params {result['trainable_parameter_count']} | "
        f"gate {result['capability_gate']} | exit {code}"
    )
    print(f"result: {out}")
    return code


def cmd_reserved(args: argparse.Namespace) -> int:
    meta = RESERVED[args.command]
    notice = {
        "schema": "v430-cli-notice/1",
        "command": args.command,
        "status": "not_implemented",
        "exit_code": EXIT_INVALID,
        "owner": meta["owner"],
        "purpose": meta["purpose"],
        "implemented_here": ["audit"],
        "message": (
            f"'{args.command}' is reserved for {meta['owner']} in v4.3.0 and is not implemented by W00. "
            "Nothing was executed, no result was written and no gate was evaluated."
        ),
    }
    print(json.dumps(notice, ensure_ascii=False))
    return EXIT_INVALID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v430.py",
        description=(
            "v4.3.0 CLI. W00 implements the read-only 'audit' subcommand only; every other "
            "subcommand fails closed with exit 2 and the 'not_implemented' notice."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    audit = subparsers.add_parser("audit", help="read-only asset/identity audit (W00)")
    audit.add_argument("--config", required=True, help="path to configs/v430/assets.json")
    audit.add_argument("--out", required=True,
                       help="new result JSON path; an existing file is refused (exit 2)")
    audit.set_defaults(handler=cmd_audit)

    contract = subparsers.add_parser(
        "validate-contract",
        help="config<->module contract validation (W01)")
    contract.add_argument("--config", required=True,
                          help="path to configs/v430/contracts.json")
    contract.add_argument("--out", required=True,
                          help="new result JSON path; an existing file is refused (exit 2)")
    contract.set_defaults(handler=cmd_validate_contract)

    splits = subparsers.add_parser(
        "validate-splits",
        help="frozen split registry / task lineage validation (W09)")
    splits.add_argument("--config", default="configs/v430/splits.json",
                        help=("path to configs/v430/splits.json "
                              "(default: configs/v430/splits.json)"))
    splits.add_argument("--out", required=True,
                        help="new result JSON path; an existing file is refused (exit 2)")
    splits.set_defaults(handler=cmd_validate_splits)

    train_bc = subparsers.add_parser(
        "train-bc",
        help="behaviour cloning of the O2 actor on synthetic batches (W03G)")
    train_bc.add_argument(
        "--batches", required=True,
        help=("new path to a v430-synthetic-batches/1 JSON specification; the "
              "batches are generated inside the CLI and no corpus is read"))
    train_bc.add_argument("--epochs", type=int, default=1,
                          help="training epochs (default 1)")
    train_bc.add_argument("--lr", type=float, default=1e-4,
                          help="AdamW learning rate (default 1e-4)")
    train_bc.add_argument("--seed", type=int, default=0,
                          help="seed for the initial weights and the batches (default 0)")
    train_bc.add_argument("--max-steps", type=int, default=None,
                          help="stop after this many optimizer steps (optional)")
    train_bc.add_argument("--memory-mode", default="none",
                          help="'none' (D-pixel) or 'gru' (D-pixel-memory); default none")
    train_bc.add_argument("--shuffle", action="store_true",
                          help="permute the batch order each epoch with the seeded generator")
    train_bc.add_argument("--checkpoint-out", default=None,
                          help="new checkpoint path; an existing file is refused (exit 2)")
    train_bc.add_argument("--out", required=True,
                          help="new result JSON path; an existing file is refused (exit 2)")
    train_bc.set_defaults(handler=cmd_train_bc)

    for name, meta in RESERVED.items():
        reserved = subparsers.add_parser(name, help=f"reserved for {meta['owner']}: not implemented")
        reserved.add_argument("--config")
        reserved.add_argument("--out")
        reserved.add_argument("--registry")
        reserved.add_argument("--dry-run", action="store_true")
        reserved.add_argument("--execute", action="store_true")
        reserved.set_defaults(handler=cmd_reserved)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - program error, never a gate verdict
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
