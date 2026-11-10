"""Read-only identity preflight for two original TH08 1.00d retail captures.

Exit 0: two fixture replays and Wine loader ready; 3: resource/tool blocked;
2: executable identity mismatch (never run a reconstruction as retail).
The report contains identities and gaps, not a claim of legal ownership or capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import hashlib as _hashlib
import shutil
from pathlib import Path

RETAIL_SHA = "330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924"
RETAIL_SIZE = 840704
ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(target: Path, fixtures: Path, replay_dir: Path, wine: Path,
            game_data: Path, bgm_data: Path) -> tuple[int, dict]:
    report = {"schema": "v400-retail-preflight/1", "status": "blocked",
              "capture_count": 0, "authorization": "not established by file hash",
              "target": {"path": str(target)}, "replays": [], "tools": {}, "gaps": []}
    if not target.is_file():
        report["gaps"].append("retail_target_missing")
        return 3, report
    size, sha = target.stat().st_size, digest(target)
    report["target"].update(size=size, sha256=sha,
                            retail_identity_verified=size == RETAIL_SIZE and sha == RETAIL_SHA)
    if not report["target"]["retail_identity_verified"]:
        report["gaps"].append("target_not_original_japanese_th08_1.00d")
        return 2, report
    try:
        manifest = json.loads(fixtures.read_text(encoding="utf-8"))
        if manifest.get("targetSha256") != RETAIL_SHA or len(manifest["fixtures"]) < 2:
            raise ValueError("fixture target or count mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["gaps"].append(f"fixture_manifest_invalid:{exc}")
        return 3, report
    for fixture in manifest["fixtures"][:2]:
        name = fixture["file"]
        # Names from an external manifest must never escape the replay folder.
        if Path(name).name != name or name in (".", ".."):
            report["gaps"].append("unsafe_fixture_name")
            return 3, report
        path = replay_dir / name
        row = {"id": fixture["id"], "path": str(path), "expected_sha256": fixture["sha256"]}
        row["present"] = path.is_file()
        if row["present"]:
            row["sha256"] = digest(path)
            row["matches_fixture"] = row["sha256"] == fixture["sha256"]
        report["replays"].append(row)
        if not row.get("matches_fixture", False):
            report["gaps"].append(f"fixture_missing_or_hash_mismatch:{name}")
    for label, path in (("game_data", game_data), ("bgm_data", bgm_data)):
        row = {"path": str(path), "present": path.is_file()}
        if row["present"]:
            row.update(size=path.stat().st_size, sha256=digest(path))
        report[label] = row
        if not row["present"]:
            report["gaps"].append(f"{label}_missing")
    report["tools"] = {"wine_loader": str(wine), "wine_loader_present": wine.is_file(),
                       "wineboot": shutil.which("wineboot"), "xdotool": shutil.which("xdotool"),
                       "Xvfb": shutil.which("Xvfb"),
                       "python_hashlib_file_digest": hasattr(_hashlib, "file_digest")}
    for key in ("wine_loader_present", "wineboot", "xdotool", "Xvfb", "python_hashlib_file_digest"):
        if not report["tools"][key]:
            report["gaps"].append(f"tool_missing:{key}")
    if not report["gaps"]:
        report["status"] = "ready_to_attempt_capture_not_captured"
        return 0, report
    return 3, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, default=ROOT / "third_party/th08/resources/th08.exe")
    p.add_argument("--fixtures", type=Path, default=ROOT / "third_party/th08/config/replay-fixtures.json")
    p.add_argument("--replay-dir", type=Path, default=ROOT / "th08_replay")
    p.add_argument("--wine", type=Path, default=Path("/usr/lib/wine/wine"))
    p.add_argument("--game-data", type=Path, default=Path("/root/v400-data/game/th08.dat"))
    p.add_argument("--bgm-data", type=Path, default=Path("/root/v400-data/game/thbgm.dat"))
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    code, result = inspect(a.target, a.fixtures, a.replay_dir, a.wine, a.game_data, a.bgm_data)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if a.output:
        a.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
