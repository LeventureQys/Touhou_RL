"""Is this WSL2 distro actually ready to train TH08? (v4.0.0 P0-WP0 gate)

Run **inside** Ubuntu 24.04 on WSL2, from the repository root:

    source ~/v400-venv/bin/activate
    python tools/rl/v400_env_check.py            # human-readable verdict
    python tools/rl/v400_env_check.py --json     # machine-readable snapshot
    python tools/rl/v400_env_check.py --dry-run  # list the checks, execute nothing

This tool never launches the game.  It only asks whether the pieces the training
ground needs are installed and whether the game assets are the bytes the rest of
the project was measured against.

Exit codes: ``0`` ready, ``3`` a required dependency is missing, ``2`` the
repository layout is wrong.  Warnings (GPU, fonts, disk) never fail the run unless
``--strict`` is passed.

Why the asset hash is pinned: a different ``th08.dat`` means different sprites,
different stage data, and therefore incomparable pixels.  ``ENVIRONMENT.md``
records the sizes; the hashes below were computed from this repository's copy on
2026-11-07 and belong in EXP-P0-001.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Source of truth for the bytes: ENVIRONMENT.md section 1 (sizes) plus the
# SHA-256 recorded here on 2026-11-07 from third_party/th08/build/play.
EXPECTED_ASSETS = {
    "th08.dat": {
        "bytes": 46_838_025,
        "sha256": "9d7edf43b8ddd347cbb641836f6b5050745dd936f688daebbf9382ca557043bb",
    },
    "thbgm.dat": {
        "bytes": 449_961_024,
        "sha256": "2c2ef05f9ff6f43f752dae5da519a2477372c3f8875b7b44996a8f69fdad4d89",
    },
}

# Package names checked by pkg-config.  These mirror 10-P0 WP0.
PKG_CONFIG_LIBS = ("sdl2", "SDL2_image", "SDL2_ttf", "fontconfig", "freetype2", "gl")
# Binaries that must be on PATH.
REQUIRED_BINARIES = ("g++", "gcc", "cmake", "ninja", "pkg-config", "git", "xvfb-run", "Xvfb", "xauth")
# Budget headroom from D-05 (<= 300 GB new on disk); warn below this.
DISK_WARN_BYTES = 350 * 1024**3
EXPECTED_PYTHON = (3, 12)


@dataclass
class Check:
    name: str
    status: str = "ok"  # ok | warn | fail | skip
    required: bool = True
    detail: str = ""
    data: dict = field(default_factory=dict)


def _run(argv: list[str], timeout: float = 60.0, env: dict | None = None) -> tuple[int, str, str]:
    """Run a command, never raise; return (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", f"not found: {argv[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:.0f}s: {' '.join(argv)}"
    except Exception as exc:  # pragma: no cover - diagnostics only
        return 1, "", f"{type(exc).__name__}: {exc}"


def _sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------
def check_platform() -> Check:
    if sys.platform != "linux":
        return Check(
            "platform",
            "fail",
            True,
            f"sys.platform={sys.platform!r}: this check must run inside the WSL2 distro",
            {"platform": sys.platform},
        )
    proc_version = ""
    try:
        proc_version = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    is_wsl = bool(re.search(r"microsoft|wsl", proc_version, re.IGNORECASE))
    data = {"platform": sys.platform, "kernel": os.uname().release, "wsl": is_wsl}
    if not is_wsl:
        return Check("platform", "warn", True, "Linux, but not WSL - v4.0.0 assumes WSL2", data)
    return Check("platform", "ok", True, f"WSL2 kernel {os.uname().release}", data)


def check_python() -> Check:
    version = sys.version_info
    data = {
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    }
    if (version.major, version.minor) != EXPECTED_PYTHON:
        expected = ".".join(str(part) for part in EXPECTED_PYTHON)
        return Check("python", "warn", True, f"{data['version']} (10-P0 WP0 asks for {expected})", data)
    return Check("python", "ok", True, f"{data['version']} at {sys.executable}", data)


def check_venv() -> Check:
    in_venv = sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))
    if in_venv:
        return Check("venv", "ok", False, f"inside {sys.prefix}", {"prefix": sys.prefix})
    return Check("venv", "warn", False,
                 "not in a virtualenv - the system interpreter works but drifts", {"prefix": sys.prefix})


def check_binaries() -> Check:
    found: dict[str, str] = {}
    missing = []
    for name in REQUIRED_BINARIES:
        path = shutil.which(name)
        if path:
            found[name] = path
        else:
            missing.append(name)
    if missing:
        return Check("binaries", "fail", True, f"missing: {', '.join(missing)}", {"found": found, "missing": missing})
    versions = {}
    for name, argv in (("g++", ["g++", "--version"]), ("cmake", ["cmake", "--version"]),
                       ("ninja", ["ninja", "--version"]), ("git", ["git", "--version"])):
        rc, out, _ = _run(argv, timeout=30)
        versions[name] = _first_line(out) if rc == 0 else f"rc={rc}"
    return Check("binaries", "ok", True, "; ".join(versions.values()), {"found": found, "versions": versions})


def check_pkg_config() -> Check:
    versions: dict[str, str] = {}
    missing = []
    for lib in PKG_CONFIG_LIBS:
        rc, out, err = _run(["pkg-config", "--modversion", lib], timeout=30)
        if rc == 0 and out.strip():
            versions[lib] = out.strip()
        else:
            missing.append(lib)
    if missing:
        return Check("pkg_config", "fail", True, f"missing from pkg-config: {', '.join(missing)}",
                     {"versions": versions, "missing": missing})
    return Check("pkg_config", "ok", True,
                 ", ".join(f"{name} {ver}" for name, ver in versions.items()), {"versions": versions})


def check_gl() -> Check:
    command = ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24", "glxinfo", "-B"]
    rc, out, err = _run(command, timeout=90)
    if rc != 0:
        return Check("gl", "fail", True, f"glxinfo under xvfb failed (rc={rc}): {_first_line(err) or _first_line(out)}",
                     {"command": command})
    renderer = ""
    vendor = ""
    for line in out.splitlines():
        low = line.lower()
        if "opengl renderer string" in low:
            renderer = line.split(":", 1)[-1].strip()
        elif "opengl vendor string" in low:
            vendor = line.split(":", 1)[-1].strip()
    data = {"command": command, "renderer": renderer, "vendor": vendor}
    if "llvmpipe" in renderer.lower():
        return Check("gl", "ok", True, f"{vendor} / {renderer} (software, expected under xvfb)", data)
    return Check("gl", "ok", True, f"{vendor} / {renderer}", data)


def check_torch() -> Check:
    rc, out, err = _run([sys.executable, "-c",
                         "import torch,json;print(json.dumps({'version':torch.__version__,"
                         "'cuda':bool(torch.cuda.is_available()),"
                         "'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"],
                        timeout=180)
    if rc != 0:
        return Check("torch", "warn", False, f"torch not importable (rc={rc}): {_first_line(err)}", {})
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except Exception:
        return Check("torch", "warn", False, f"unparseable torch probe: {_first_line(out)}", {})
    status = "ok" if data.get("cuda") else "warn"
    detail = f"torch {data.get('version')} cuda={data.get('cuda')}"
    if data.get("devices"):
        detail += " devices=" + ",".join(data["devices"])
    return Check("torch", status, False, detail, data)


def check_cuda() -> Check:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return Check("cuda", "warn", False, "nvidia-smi not on PATH (P0 is CPU-bound; P3/P4 need a GPU)", {})
    rc, out, err = _run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                        timeout=60)
    if rc != 0:
        return Check("cuda", "warn", False, f"nvidia-smi failed (rc={rc}): {_first_line(err)}", {})
    return Check("cuda", "ok", False, _first_line(out), {"gpus": [line for line in out.splitlines() if line.strip()]})


def _asset_dirs(root: Path) -> list[Path]:
    """Where game assets may live, most authoritative first."""
    dirs = []
    env_data = os.environ.get("V400_DATA")
    if env_data:
        dirs.append(Path(env_data) / "game")
    else:
        dirs.append(Path.home() / "v400-data" / "game")
    dirs.append(root / "third_party/th08/build/play")
    return dirs


def check_assets(root: Path, skip_hash: bool = False) -> Check:
    candidates = _asset_dirs(root)
    chosen = next((directory for directory in candidates
                   if all((directory / name).is_file() for name in EXPECTED_ASSETS)), None)
    if chosen is None:
        listed = ", ".join(str(directory) for directory in candidates)
        return Check("assets", "fail", True, f"th08.dat/thbgm.dat not found in: {listed}",
                     {"searched": [str(directory) for directory in candidates]})
    results: dict[str, dict] = {}
    problems = []
    for name, expected in EXPECTED_ASSETS.items():
        path = chosen / name
        size = path.stat().st_size
        entry: dict = {"path": str(path), "bytes": size, "expected_bytes": expected["bytes"]}
        if size != expected["bytes"]:
            entry["sha256"] = None
            problems.append(f"{name}: {size} B != {expected['bytes']} B")
        elif skip_hash:
            entry["sha256"] = "skipped"
        else:
            digest = _sha256(path)
            entry["sha256"] = digest
            if digest != expected["sha256"]:
                problems.append(f"{name}: sha256 {digest[:16]}... != {expected['sha256'][:16]}...")
        results[name] = entry
    data = {"directory": str(chosen), "assets": results}
    if problems:
        return Check("assets", "fail", True, "; ".join(problems), data)
    suffix = " (hash skipped)" if skip_hash else ""
    return Check("assets", "ok", True, f"verified in {chosen}{suffix}", data)


def check_repo(root: Path) -> Check:
    required = {
        "third_party/th08/src": root / "third_party/th08/src",
        "third_party/th08/scripts": root / "third_party/th08/scripts",
        "tools/rl": root / "tools/rl",
        "touhou_rl": root / "touhou_rl",
    }
    missing = [name for name, path in required.items() if not path.is_dir()]
    if missing:
        return Check("repo", "fail", True, f"missing directories: {', '.join(missing)}",
                     {"root": str(root), "missing": missing})
    sources = sorted((root / "third_party/th08/src").glob("*.cpp"))
    if not sources:
        return Check("repo", "fail", True, "third_party/th08 has no .cpp sources - submodule not initialised",
                     {"root": str(root)})
    rc, out, _ = _run(["git", "-C", str(root / "third_party/th08"), "rev-parse", "--abbrev-ref", "HEAD"], timeout=30)
    branch = out.strip() if rc == 0 else "?"
    rc, out, _ = _run(["git", "-C", str(root / "third_party/th08"), "rev-parse", "--short", "HEAD"], timeout=30)
    head = out.strip() if rc == 0 else "?"
    return Check("repo", "ok", True, f"{len(sources)} .cpp, th08 at {branch}@{head}",
                 {"root": str(root), "th08_branch": branch, "th08_head": head, "cpp_count": len(sources)})


def check_disk(root: Path) -> Check:
    env_data = os.environ.get("V400_DATA")
    target = Path(env_data) if env_data else Path.home()
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return Check("disk", "warn", False, f"cannot stat {target}: {exc}", {})
    data = {"path": str(target), "free_gb": round(usage.free / 1024**3, 1),
            "warn_below_gb": round(DISK_WARN_BYTES / 1024**3, 1)}
    status = "ok" if usage.free >= DISK_WARN_BYTES else "warn"
    return Check("disk", status, False,
                 f"{data['free_gb']} GB free on {target} (D-05 budget needs headroom)", data)


def check_cores() -> Check:
    cores = os.cpu_count() or 0
    data = {"cores": cores}
    if cores <= 0:
        return Check("cores", "fail", True, "os.cpu_count() returned nothing", data)
    return Check("cores", "ok", True, f"{cores} logical cores", data)


def check_fonts() -> Check:
    fc_list = shutil.which("fc-list")
    if not fc_list:
        return Check("fonts", "warn", False, "fc-list missing (fonts-vlgothic not installed?)", {})
    rc, out, _ = _run([fc_list], timeout=60)
    if rc != 0:
        return Check("fonts", "warn", False, f"fc-list failed (rc={rc})", {})
    hits = [line for line in out.splitlines() if re.search(r"VL\s*Gothic|vlgothic", line, re.IGNORECASE)]
    if hits:
        return Check("fonts", "ok", False, f"VL Gothic present ({len(hits)} faces)", {"faces": len(hits)})
    return Check("fonts", "warn", False, "VL Gothic not found; the game may fall back to a default font", {})


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
CHECKS = (
    "platform",
    "python",
    "venv",
    "binaries",
    "pkg_config",
    "gl",
    "repo",
    "assets",
    "torch",
    "cuda",
    "cores",
    "disk",
    "fonts",
)


def collect(root: Path, skip_hash: bool = False) -> dict:
    results = [
        check_platform(),
        check_python(),
        check_venv(),
        check_binaries(),
        check_pkg_config(),
        check_gl(),
        check_repo(root),
        check_assets(root, skip_hash=skip_hash),
        check_torch(),
        check_cuda(),
        check_cores(),
        check_disk(root),
        check_fonts(),
    ]
    failed_required = [check.name for check in results if check.status == "fail" and check.required]
    warned = [check.name for check in results if check.status == "warn"]
    repo_failed = "repo" in failed_required
    if failed_required:
        exit_code = 2 if repo_failed else 3
    else:
        exit_code = 0
    return {
        "schema": "v400-env/1",
        "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(root),
        "checks": [asdict(check) for check in results],
        "verdict": {
            "ready": not failed_required,
            "failed_required": failed_required,
            "warnings": warned,
            "exit_code": exit_code,
        },
    }


def render(facts: dict) -> str:
    """ASCII-safe console rendering."""
    marks = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "skip": "skip"}
    lines = [
        "v4.0.0 environment check (nothing launched)",
        f"  root : {facts['root']}",
        f"  time : {facts['checkedAt']}",
        "",
    ]
    for check in facts["checks"]:
        mark = marks.get(check["status"], "?   ")
        required = "*" if check["required"] else " "
        lines.append(f"  [{mark}]{required} {check['name']:<12} {check['detail']}")
    verdict = facts["verdict"]
    lines += [
        "",
        "  (* = required; warnings do not block P0)",
        f"  verdict: ready={verdict['ready']}  exit={verdict['exit_code']}",
    ]
    if verdict["failed_required"]:
        lines.append(f"  blocking: {', '.join(verdict['failed_required'])}")
    if verdict["warnings"]:
        lines.append(f"  warnings: {', '.join(verdict['warnings'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("v400-env-check", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the machine-readable snapshot only")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the checks and exit without executing anything")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--skip-asset-hash", action="store_true",
                        help="check asset sizes only (saves ~1 GB of hashing)")
    parser.add_argument("--root", default="", help="repository root (testing/CI override)")
    parser.add_argument("--baseline", default="", help="write the JSON snapshot to this path")
    args = parser.parse_args(argv)

    if args.dry_run:
        print("v4.0.0 environment check - dry run, nothing executed")
        for name in CHECKS:
            print(f"  - {name}")
        print(f"\n  root: {Path(args.root).resolve() if args.root else ROOT}")
        return 0

    root = Path(args.root).resolve() if args.root else ROOT
    facts = collect(root, skip_hash=args.skip_asset_hash)

    if args.strict and facts["verdict"]["warnings"] and facts["verdict"]["exit_code"] == 0:
        facts["verdict"]["exit_code"] = 3
        facts["verdict"]["ready"] = False
        facts["verdict"]["strict"] = True

    if args.baseline:
        target = Path(args.baseline)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(facts, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[v400-env] snapshot written: {target}", file=sys.stderr)

    if args.json:
        print(json.dumps(facts, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render(facts))
    return int(facts["verdict"]["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
