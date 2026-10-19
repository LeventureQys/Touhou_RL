"""Run the v1.0.3 recipe matrix in order, one directory and one table row per attempt.

Why this exists
---------------
Stage 3 compares several architecture recipes at several data sizes.  Two rules
make that comparison credible:

* every attempt leaves a row behind (requirement R2) - a recipe that was tried and
  lost is still evidence, and an interrupted batch must not hide what already ran;
* a completed run is never overwritten: a ``runs/v103-exp-<NN>/`` directory that
  already holds a ``vision-summary.json`` is frozen, so re-running the same recipe
  file resumes instead of repeating GPU hours.

Directory numbers are allocated as "lowest free slot from 01", so a later recipe
file (confirmation run, R3 data points) continues the numbering instead of
colliding with the first batch.  A recipe that already has a record in
``sweep-report.json`` resumes the directory recorded there, which makes re-entry
exact: same recipe file, same directories, no duplicate table row (rows are keyed
by the ``序号`` column).

Usage::

    python tools/rl/v103_sweep.py --recipes Document/Update/Alpha/v1.0.3/Stage3/recipes.json
    python tools/rl/v103_sweep.py --recipes ... --only R1-a,R1-b --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Recipe args map 1:1 onto ``python -m touhou_rl.vision_train`` flags, underscores
# turned into hyphens.  The order is the order they are emitted in, which keeps
# the printed command line stable between runs and therefore reviewable.
ALLOWED_ARGS = (
    "stage",
    "epochs_state",
    "epochs_policy",
    "batch_size",
    "lr",
    "depth",
    "train_fraction",
    "max_train_batches",
    "max_val_batches",
    "seed",
    "note",
    "heatmap_sigma",
    "w_heatmap",
    "w_self",
    "w_danger",
    "pf_width",
    "hud_width",
)
ARCHES = ("v1", "v2")
DIR_PREFIX = "v103-exp-"
SUMMARY_NAME = "vision-summary.json"
BUCKET = "0-15"
FROZEN_DIR_MARKER = "pilot-v1"
MAX_DIR_NUMBER = 999
TABLE_HEADER = (
    "| 序号 | 配方 | arch | 训练帧数 | max_train_batches | 位置误差 px | "
    "perframe corr 中位数 | 0-15 桶后悔值 | 用时 s | summary 路径 |"
)
TABLE_RULE = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
DEFAULT_OUT_JSON = "Document/Update/Alpha/v1.0.3/Stage3/sweep-report.json"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _as_number(value):
    """Return ``value`` as a number (ints stay ints) if it is one, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if isinstance(value, int) else float(value)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _fmt(value, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_int(value) -> str:
    return "n/a" if value is None else str(int(value))


def _fmt_arg(value) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def format_command(cmd: list[str]) -> str:
    """Render a command line that can be pasted back into a shell."""
    return " ".join(f'"{token}"' if (" " in token or "\t" in token) else token for token in cmd)


# --------------------------------------------------------------------------- #
# recipe file validation
# --------------------------------------------------------------------------- #
def _require(payload: dict, key: str, source: str):
    if key not in payload:
        raise SystemExit(f"{source}: missing required key {key!r}")
    return payload[key]


def validate_recipes(payload, source: str = "recipes") -> dict:
    """Validate the whole recipe file before a single GPU second is spent."""
    if not isinstance(payload, dict):
        raise SystemExit(f"{source}: top level must be a JSON object")
    dataset = _require(payload, "dataset", source)
    runs_root = _require(payload, "runs_root", source)
    table = _require(payload, "table", source)
    recipes = _require(payload, "recipes", source)
    for label, value in (("dataset", dataset), ("runs_root", runs_root), ("table", table)):
        if not isinstance(value, str) or not value:
            raise SystemExit(f"{source}: {label} must be a non-empty string, got {value!r}")
    if FROZEN_DIR_MARKER in Path(runs_root).as_posix():
        raise SystemExit(
            f"{source}: refusing runs_root {runs_root!r}: {FROZEN_DIR_MARKER} holds v1.0.2 evidence"
        )
    if not isinstance(recipes, list) or not recipes:
        raise SystemExit(f"{source}: recipes must be a non-empty list")

    seen: dict[str, int] = {}
    for position, recipe in enumerate(recipes, start=1):
        if not isinstance(recipe, dict):
            raise SystemExit(f"{source}: recipe #{position} must be a JSON object")
        recipe_id = recipe.get("id")
        if not isinstance(recipe_id, str) or not recipe_id:
            raise SystemExit(f"{source}: recipe #{position} needs a non-empty string id")
        if recipe_id in seen:
            raise SystemExit(f"{source}: duplicate recipe id {recipe_id!r}")
        seen[recipe_id] = position
        arch = recipe.get("arch")
        if arch not in ARCHES:
            raise SystemExit(
                f"{source}: recipe {recipe_id!r} has arch {arch!r}, expected one of {list(ARCHES)}"
            )
        args = recipe.get("args", {})
        if not isinstance(args, dict):
            raise SystemExit(f"{source}: recipe {recipe_id!r} args must be a JSON object")
        for key, value in args.items():
            if key not in ALLOWED_ARGS:
                raise SystemExit(f"unknown recipe arg {key!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise SystemExit(
                    f"{source}: recipe {recipe_id!r} arg {key!r} must be int, float or str"
                )
        max_val_batches = args.get("max_val_batches", 0)
        if _as_number(max_val_batches) != 0:
            raise SystemExit("max_val_batches must be 0: validation must always be full")
    return {
        "dataset": dataset,
        "runs_root": runs_root,
        "table": table,
        "recipes": recipes,
    }


def build_command(recipe: dict, dataset: str, out_dir: Path, python: str | None = None) -> list[str]:
    """Assemble one ``vision_train`` command line from a recipe.

    ``--max-val-batches 0`` is always emitted, even when the recipe leaves it out:
    design doc section 7.4 requires the experiment script to say so explicitly, and a
    shrunken validation set makes two recipes' ``val_player_pos_px`` incomparable.
    """
    cmd = [
        python or sys.executable,
        "-m",
        "touhou_rl.vision_train",
        "--dataset",
        dataset,
        "--out",
        out_dir.as_posix(),
        "--arch",
        recipe["arch"],
    ]
    args = recipe.get("args") or {}
    for key in ALLOWED_ARGS:
        present = key in args or key == "max_val_batches"
        if present:
            value = args[key] if key in args else 0
            cmd.extend([f"--{key.replace('_', '-')}", _fmt_arg(value)])
    return cmd


# --------------------------------------------------------------------------- #
# summary -> table row
# --------------------------------------------------------------------------- #
def summary_metrics(summary) -> dict:
    """Numbers the experiment table and the report need; ``None`` when absent."""
    summary = _dict(summary)
    stages = _dict(summary.get("stages"))
    state_best = _dict(_dict(stages.get("state")).get("best"))
    policy_best = _dict(_dict(stages.get("policy")).get("best"))
    config = _dict(summary.get("config"))
    regret = None
    for row in policy_best.get("by_danger") or []:
        if isinstance(row, dict) and row.get("bucket") == BUCKET:
            regret = _as_number(row.get("regret_px"))
            break
    return {
        "train_frames": _as_number(summary.get("train_frames")),
        "max_train_batches": _as_number(config.get("max_train_batches")),
        "pos_px": _as_number(state_best.get("val_player_pos_px")),
        "perframe_median": _as_number(state_best.get("val_danger_corr_perframe_median")),
        "regret_0_15": regret,
    }


def table_row(index: str, recipe_id: str, arch: str, metrics: dict, seconds: float, summary_path: Path) -> str:
    return (
        f"| {index} | {recipe_id} | {arch} | {_fmt_int(metrics.get('train_frames'))} | "
        f"{_fmt_int(metrics.get('max_train_batches'))} | {_fmt(metrics.get('pos_px'), 2)} | "
        f"{_fmt(metrics.get('perframe_median'), 3)} | {_fmt(metrics.get('regret_0_15'), 2)} | "
        f"{_fmt(seconds, 1)} | {summary_path.as_posix()} |"
    )


def _table_index(line: str):
    """First cell of a Markdown table row, or ``None`` when it is not a row."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return cells[0] if cells else None


def write_table_row(table_path: Path, index: str, row: str) -> str:
    """Append one row unless ``序号`` is already present.  Returns what happened.

    Rows are immutable once written: a re-run of the same recipe must not shift
    history, only fill in a row that was never written (an interrupted batch).
    """
    table_path.parent.mkdir(parents=True, exist_ok=True)
    existing = table_path.read_text(encoding="utf-8") if table_path.is_file() else ""
    lines = existing.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if any(_table_index(line) == index for line in lines):
        return "present"
    if not any(_table_index(line) == "序号" for line in lines):
        lines = [TABLE_HEADER, TABLE_RULE] + lines
    lines.append(row)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "appended"


# --------------------------------------------------------------------------- #
# directory allocation and the sweep report
# --------------------------------------------------------------------------- #
def dir_name(number: int) -> str:
    return f"{DIR_PREFIX}{number:02d}"


def _has_summary(out_dir: Path) -> bool:
    return (out_dir / SUMMARY_NAME).is_file()


def allocate_dir(runs_root: Path, taken: set[int]) -> tuple[int, Path]:
    """Lowest number whose directory holds no ``vision-summary.json`` yet."""
    number = 1
    while number <= MAX_DIR_NUMBER:
        out_dir = runs_root / dir_name(number)
        if number not in taken and not _has_summary(out_dir):
            taken.add(number)
            return number, out_dir
        number += 1
    raise SystemExit(f"no free {DIR_PREFIX}<NN> left under {runs_root.as_posix()} (>= {MAX_DIR_NUMBER})")


def dir_number(out_dir: Path):
    name = out_dir.name
    if not name.startswith(DIR_PREFIX):
        return None
    suffix = name[len(DIR_PREFIX):]
    return int(suffix) if suffix.isdigit() else None


def load_report(path: Path) -> list[dict]:
    """Previous sweep records; an unreadable report is a warning, not a failure."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: ignoring unreadable report {path.as_posix()}: {exc}", file=sys.stderr)
        return []
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("entries")
    else:
        entries = None
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict) and entry.get("id")]


def prior_dirs(entries: list[dict]) -> dict[str, Path]:
    """Recipe id -> directory of an earlier real attempt (dry runs own no directory)."""
    mapping: dict[str, Path] = {}
    for entry in entries:
        if entry.get("status") == "dry-run":
            continue
        out_dir = entry.get("out_dir")
        if isinstance(out_dir, str) and out_dir:
            mapping[entry["id"]] = Path(out_dir)
    return mapping


def merge_entries(prior: list[dict], fresh: list[dict]) -> list[dict]:
    """Fold this invocation into the previous report, keyed by recipe id."""
    by_id = {entry["id"]: entry for entry in prior}
    order = [entry["id"] for entry in prior]
    for entry in fresh:
        recipe_id = entry["id"]
        old = by_id.get(recipe_id)
        if old is not None and old.get("status") != "dry-run" and entry.get("status") == "dry-run":
            # a dry run must never erase a real result
            continue
        if recipe_id not in order:
            order.append(recipe_id)
        by_id[recipe_id] = entry
    return [by_id[recipe_id] for recipe_id in order]


def write_report(path: Path, spec: dict, recipes_path: Path, entries: list[dict], dry_run: bool) -> None:
    merged = merge_entries(load_report(path), entries)
    document = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "recipes": recipes_path.as_posix(),
        "dataset": spec["dataset"],
        "runs_root": spec["runs_root"],
        "table": spec["table"],
        "entries": merged,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# one attempt
# --------------------------------------------------------------------------- #
def _execute(cmd: list[str]) -> int:
    """Run one training command.  A non-zero code is reported, never raised."""
    return int(subprocess.run(cmd, check=False).returncode)


def _read_summary(summary_path: Path) -> dict:
    if not summary_path.is_file():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: unreadable summary {summary_path.as_posix()}: {exc}", file=sys.stderr)
        return {}
    return payload if isinstance(payload, dict) else {}


def _select(recipes: list[dict], only: str) -> list[dict]:
    if not only.strip():
        return list(recipes)
    wanted = [item.strip() for item in only.split(",") if item.strip()]
    known = {recipe["id"] for recipe in recipes}
    for recipe_id in wanted:
        if recipe_id not in known:
            raise SystemExit(f"unknown recipe id {recipe_id!r}")
    return [recipe for recipe in recipes if recipe["id"] in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the v1.0.3 recipe matrix, one directory and one table row per attempt"
    )
    parser.add_argument("--recipes", required=True, metavar="PATH", help="recipe JSON file")
    parser.add_argument(
        "--only", default="", metavar="ID,ID", help="run only these recipe ids (comma separated)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands, create nothing, write no table row",
    )
    parser.add_argument(
        "--out-json",
        default=DEFAULT_OUT_JSON,
        metavar="PATH",
        help=f"sweep report (default: {DEFAULT_OUT_JSON})",
    )
    args = parser.parse_args(argv)

    recipes_path = Path(args.recipes)
    if not recipes_path.is_file():
        raise SystemExit(f"recipes file not found: {args.recipes}")
    try:
        payload = json.loads(recipes_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"recipes file is not valid JSON: {args.recipes} ({exc})") from exc
    spec = validate_recipes(payload, args.recipes)

    selected = _select(spec["recipes"], args.only)
    runs_root = Path(spec["runs_root"])
    table_path = Path(spec["table"])
    report_path = Path(args.out_json)
    prior = load_report(report_path)
    resumed = prior_dirs(prior)

    taken: set[int] = set()
    entries: list[dict] = []
    failures = 0
    for position, recipe in enumerate(selected, start=1):
        recipe_id = recipe["id"]
        recorded = resumed.get(recipe_id)
        if recorded is not None:
            number = dir_number(recorded)
            if number is not None:
                taken.add(number)
            out_dir = recorded
        else:
            number, out_dir = allocate_dir(runs_root, taken)
        index = f"{number:02d}" if number is not None else "n/a"
        cmd = build_command(recipe, spec["dataset"], out_dir)
        summary_path = out_dir / SUMMARY_NAME

        print(f"[{position}/{len(selected)}] {recipe_id} -> {out_dir.as_posix()}")
        print(f"  {format_command(cmd)}")

        if args.dry_run:
            entries.append(
                {
                    "id": recipe_id,
                    "arch": recipe["arch"],
                    "note": recipe.get("note", ""),
                    "command": format_command(cmd),
                    "out_dir": out_dir.as_posix(),
                    "summary": summary_path.as_posix(),
                    "index": index,
                    "status": "dry-run",
                    "returncode": None,
                    "seconds": 0.0,
                    "metrics": {},
                }
            )
            continue

        if summary_path.is_file():
            status, returncode, seconds = "skipped", None, 0.0
            print(f"  status: skipped (frozen result: {summary_path.as_posix()})")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            returncode = _execute(cmd)
            seconds = round(time.monotonic() - started, 1)
            if returncode != 0:
                status = "failed"
            elif summary_path.is_file():
                status = "ok"
            else:
                status = "no-summary"
            print(f"  status: {status} (returncode {returncode}, {seconds:.1f} s)")
            if status != "ok":
                failures += 1

        metrics = summary_metrics(_read_summary(summary_path))
        entries.append(
            {
                "id": recipe_id,
                "arch": recipe["arch"],
                "note": recipe.get("note", ""),
                "command": format_command(cmd),
                "out_dir": out_dir.as_posix(),
                "summary": summary_path.as_posix(),
                "index": index,
                "status": status,
                "returncode": returncode,
                "seconds": seconds,
                "metrics": metrics,
            }
        )
        row = table_row(index, recipe_id, recipe["arch"], metrics, seconds, summary_path)
        print(f"  table: {write_table_row(table_path, index, row)} -> {table_path.as_posix()}")
        print(
            f"  {recipe_id} {status} pos_px={_fmt(metrics.get('pos_px'), 2)} "
            f"perframe_median={_fmt(metrics.get('perframe_median'), 3)} "
            f"regret_0_15={_fmt(metrics.get('regret_0_15'), 2)}"
        )

    write_report(report_path, spec, recipes_path, entries, args.dry_run)
    print(f"report: {report_path.as_posix()}")
    if args.dry_run:
        print(f"dry run: {len(entries)} command(s) printed, nothing executed")
        return 0
    print(f"table: {table_path.as_posix()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
