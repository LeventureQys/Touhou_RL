"""Read-only checks for the 2026-11-15 roadmap review; never runs the game.

--write exclusively creates this review's JSON report; it never overwrites one.
This is evidence arithmetic and document QA, not a capability evaluator.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from urllib.parse import unquote


HERE = Path(__file__).resolve().parent
VERSION = HERE.parent
ROOT = VERSION.parents[3]
OLD = ROOT / "Document/Update/Alpha/v4.2.0"
REPORT = HERE / "review_checks.json"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> dict:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=False)
    check(proc.returncode == 0, f"git {args!r} failed: {proc.stderr}")
    return {"exit_code": proc.returncode, "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    check((ROOT / "AGENTS.md").is_file(), "Cannot resolve repository root")

    source_paths = sorted((OLD / "evidence").rglob("*.json"))
    check(len(source_paths) == 89, f"Historical evidence inventory changed: {len(source_paths)}")
    for path in source_paths:
        read_json(path)

    data = read_json(OLD / "evidence/T08/clear_remeasure.json")
    rows = []
    total = 0
    for arm, suites in data["arms"].items():
        for suite, block in suites.items():
            episodes = block["episodes"]
            frames = [ep["frames"] for ep in episodes]
            n = len(episodes)
            successes = sum(bool(ep["stage_clear"]) for ep in episodes)
            mean = statistics.mean(frames)
            check(n == block["n"] == block["n_measured"] == 20, f"Bad n: {arm}/{suite}")
            check(successes == block["clears"] == 0, f"Clear count: {arm}/{suite}")
            check(math.isclose(mean, block["frames"]["mean"], abs_tol=1e-9),
                  f"Frame mean: {arm}/{suite}")
            check(len({ep["seed"] for ep in episodes}) == 20, f"Duplicate seed: {arm}/{suite}")
            stops = dict(collections.Counter(ep["stop_reason"] for ep in episodes))
            expected_stop = "game_over" if suite == "s1n" else "death_stop"
            check(stops == {expected_stop: 20}, f"Unexpected stops: {arm}/{suite}")
            total += n
            rows.append({"arm": arm, "suite": suite, "episodes": n, "successes": successes,
                         "frames_mean": mean, "frames_max": max(frames), "stop_reasons": stops})
    check(total == data["episode_count"] == 160, "Total episode count")
    check(max(row["frames_max"] for row in rows) == 3914, "Deepest episode")

    a1 = read_json(OLD / "evidence/T08/a1_safe_set_full.json")
    sizes = []
    arms_count = 0
    for state in a1["states"]:
        arms = state["arms"]
        check(len(arms) == 18 and {arm["cand"] for arm in arms} == set(range(18)), "A1 candidates")
        check(all(arm["outcome"] in {"engine_alive", "engine_death"} for arm in arms), "A1 unknown")
        alive = sorted(arm["cand"] for arm in arms if arm["outcome"] == "engine_alive")
        check(alive == sorted(state["safe_set"]), "A1 state safe-set mismatch")
        check(len(alive) == state["safe_size"], "A1 state safe-size mismatch")
        check(state["prefix_hash_mismatches"] == 0, "A1 prefix mismatch")
        sizes.append(len(alive))
        arms_count += len(arms)
    proper = sum(0 < n < 18 for n in sizes)
    multi = sum(1 < n < 18 for n in sizes)
    check(len(sizes) == 200 and arms_count == 3600, "A1 table shape")
    check(proper == a1["proper"]["count"] == 184 and multi == 178, "A1 proper definition")

    setloss = read_json(OLD / "evidence/T08/setloss_form.json")
    setloss_rows = []
    for suite, block in setloss["measurement"]["rulers"].items():
        check(len(block["frames"]) == block["n"] == 20, "Setloss n")
        mean = statistics.mean(block["frames"])
        check(math.isclose(mean, block["frames_mean"], abs_tol=1e-9), "Setloss mean")
        check(block["clears"] == 0, "Setloss clear count")
        setloss_rows.append({"suite": suite, "episodes": block["n"], "frames_mean": mean,
                             "successes_reported": block["clears"]})

    ledger_counts = {}
    for name in ["journal.jsonl", "budget.jsonl"]:
        lines = (OLD / name).read_text(encoding="utf-8-sig").splitlines()
        ledger_counts[name] = sum(bool(line.strip()) for line in lines)
        for line in lines:
            if line.strip():
                json.loads(line)

    local_paths = ["touhou_rl/v420", "configs/v420", "tools/rl/v420.py",
                   "tools/rl/v420_eval_shoot.py", ".tmp_v420_data", "touhou_rl/v410"]
    availability = {path: (ROOT / path).exists() for path in local_paths}
    availability["tests/test_v420*.py count"] = len(list((ROOT / "tests").glob("test_v420*.py")))

    docs = sorted(VERSION.glob("*.md"))
    missing_links = []
    link_count = 0
    for doc in docs:
        content = doc.read_text(encoding="utf-8-sig")
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", content):
            target = match.group(1).strip().strip("<>")
            if "://" in target or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0])
            resolved = (doc.parent / target).resolve()
            link_count += 1
            if not resolved.exists() and not (args.write and resolved == REPORT):
                missing_links.append({"document": doc.name, "target": target})
    check(not missing_links, f"Broken local document links: {missing_links}")

    extra_sources = [OLD / name for name in ["README.md", "PROGRESS.md", "ROUTE_CONCLUSION.md",
                                             "state.json", "journal.jsonl", "budget.jsonl"]]
    extra_sources += [ROOT / path for path in ["AGENTS.md", "ASSET_SYNC.md", "ASSET_SYNC.sha256",
        "touhou_rl/v400/gym/actions.py", "touhou_rl/v400/student/obs_spec.py",
        "touhou_rl/v400/student/obs_spec_v3.py"]]
    source_hashes = {path.relative_to(ROOT).as_posix(): sha(path)
                     for path in source_paths + extra_sources}
    if REPORT.exists():
        previous = read_json(REPORT)
        check(previous["source_sha256"] == source_hashes,
              "Source evidence changed since the recorded review; do not overwrite the report")

    result = {
        "schema": "v430-roadmap-review/1", "review_date": "2026-11-15",
        "timezone": "Asia/Shanghai", "scope": "local evidence arithmetic and roadmap QA only",
        "command": "python Document/Update/Alpha/v4.3.0/evidence/verify_review.py --write",
        "verification_exit_code": 0,
        "git_head": git("rev-parse", "HEAD")["stdout"],
        "tracked_diff": git("diff", "--name-only"),
        "parsed_evidence_json": len(source_paths), "parsed_ledgers": ledger_counts,
        "remeasure_rows": rows, "standard_s1n_episode_records": 80,
        "first_death_episode_records": 80, "pooled_independent_confirmation_n": None,
        "independence_note": "Same dev seeds across arms; do not pool two protocols or models as one binomial sample",
        "a1_recount": {"states": 200, "arms": arms_count, "proper_0_lt_size_lt_18": proper,
                       "multimember_1_lt_size_lt_18": multi, "horizon_frames": a1["horizon_frames"]},
        "setloss_recount": setloss_rows, "local_availability": availability,
        "old_gate_snapshot": read_json(OLD / "state.json")["gates"],
        "document_files": [path.name for path in docs], "local_links_checked": link_count,
        "missing_local_links": missing_links, "source_sha256": source_hashes,
        "not_performed": ["game or engine execution", "training", "Windows deployment",
                          "historical pytest rerun", "asset-host verification", "formal confirmation"],
        "limitations": ["Historical JSON may describe experiments that were invalid or incomplete",
                        "Checking recorded outcomes does not re-execute the original experiment",
                        "Remote assets were not enumerated; local absence is not remote absence"],
    }
    if args.write:
        with REPORT.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"status": "review_checks_passed", "exit_code": 0,
                      "evidence_json": len(source_paths), "remeasure_groups": len(rows),
                      "standard_s1n": "0/80 records across four models",
                      "a1_states": 200, "a1_arms": arms_count, "proper": proper,
                      "document_files": len(docs), "local_links_checked": link_count,
                      "report_created": args.write, "capability_tests_run": 0},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
