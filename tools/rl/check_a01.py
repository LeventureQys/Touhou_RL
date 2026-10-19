"""Decide A-01 / A-02 / A-03 from ``vision-summary.json`` files, in code.

Why this exists
---------------
The v1.0.3 gate is: fix the vision architecture first (A-01: player position error
< 10 px **and** per-frame danger correlation median > 0.8), then ask whether more
data still helps (A-03).  Both numbers are easy to read charitably out of a training
report after the fact, which is exactly what the failure-mode list forbids
(problem list D4/Q2).  So the verdicts live here, not in prose.

Three properties are deliberate and must not be "simplified":

* **A-01 is "some epoch passed", not "the best epoch passed".**  ``state.pt`` is
  chosen by ``val_player_pos_px`` alone, so an epoch can be the position winner and
  still miss the joint bar, while an earlier epoch met both.  Reading only
  ``stages.state.best`` would report a false failure.  Every row of
  ``stages.state.history`` is scanned.
* **The per-frame key is the only danger metric accepted.**  If
  ``val_danger_corr_perframe_median`` is absent the verdict is ``blocked``; there is
  no fallback to the pooled ``val_danger_corr``, whose definition is what made the
  old number look better than it was.
* **A-03 never authorises R4.**  ``has_gain`` only justifies collecting 3 more
  replays for a cross-replay curve; ``r4_authorized`` is always ``false``.

Exit codes: 0 when A-01 and A-02 both pass, 1 otherwise, 2 when an input file is
missing or unusable (never silently treated as zero).

Usage::

    python tools/rl/check_a01.py \
        --summary runs/v103-exp-01/vision-summary.json \
        --primary runs/v103-exp-05/vision-summary.json \
        --r3 0.50=runs/v103-exp-06/vision-summary.json \
        --r3 1.00=runs/v103-exp-05/vision-summary.json \
        --out Document/Update/Alpha/v1.0.3/Stage3/acceptance-v103.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCHEMA = "vision-summary/2"
POS_KEY = "val_player_pos_px"
PERFRAME_KEY = "val_danger_corr_perframe_median"
BUCKET = "0-15"
A01_POS_MAX = 10.0
A01_PERFRAME_MIN = 0.8
A02_REGRET_MAX = 1.5
A03_POS_RATIO = 0.85
A03_REGRET_RATIO = 0.90
A03_REQUIRED = (0.5, 1.0)
A03_RULE = "pos(1.00) <= 0.85*pos(0.50) or regret(1.00) <= 0.90*regret(0.50)"
R4_NOTE = "A-03 has_gain only authorizes collecting 3 more replays for a cross-replay curve"
NO_PRIMARY_REASON = "no full v2 run (train_fraction=1.0, max_train_batches=0)"


class InputError(Exception):
    """An input that cannot be used at all: missing file, unreadable JSON, no state stage."""


# --------------------------------------------------------------------------- #
# generic readers
# --------------------------------------------------------------------------- #
def _dict(value):
    return value if isinstance(value, dict) else {}


def _as_float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value):
    number = _as_float(value)
    return None if number is None else int(number)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def load_summary(path_text: str) -> dict:
    """Read one summary file.  Anything unreadable is an ``InputError``."""
    path = Path(path_text)
    if not path.is_file():
        raise InputError(f"summary not found: {path_text}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"summary is not valid JSON: {path_text} ({exc})") from exc
    except OSError as exc:
        raise InputError(f"summary cannot be read: {path_text} ({exc})") from exc
    if not isinstance(payload, dict):
        raise InputError(f"summary is not a JSON object: {path_text}")
    schema = payload.get("schema")
    if schema != SCHEMA:
        warn(f"{path_text}: schema is {schema!r}, expected {SCHEMA!r}")
    return payload


def state_stage(payload: dict) -> dict:
    return _dict(_dict(payload.get("stages")).get("state"))


def policy_stage(payload: dict) -> dict:
    return _dict(_dict(payload.get("stages")).get("policy"))


def state_history(payload: dict) -> list:
    history = state_stage(payload).get("history")
    return history if isinstance(history, list) else []


def state_best(payload: dict) -> dict:
    return _dict(state_stage(payload).get("best"))


def policy_best(payload: dict) -> dict:
    return _dict(policy_stage(payload).get("best"))


def bucket_regret(node: dict):
    """``regret_px`` of the ``0-15`` danger bucket, or ``None``."""
    for row in _dict(node).get("by_danger") or []:
        if isinstance(row, dict) and row.get("bucket") == BUCKET:
            return _as_float(row.get("regret_px"))
    return None


# --------------------------------------------------------------------------- #
# A-01: does *some* epoch meet both thresholds?
# --------------------------------------------------------------------------- #
def _a01_skeleton(thresholds: dict, reason: str) -> dict:
    return {
        "status": "blocked",
        "epoch": None,
        "pos_px": None,
        "perframe_median": None,
        "best_epoch": None,
        "best_pos_px": None,
        "best_perframe_median": None,
        "epochs_checked": 0,
        "thresholds": thresholds,
        "reason": reason,
    }


def evaluate_a01(history: list) -> dict:
    """Scan every history row; a single joint-qualifying epoch is a pass."""
    thresholds = {"pos_px": A01_POS_MAX, "perframe_median": A01_PERFRAME_MIN}
    if not history:
        return _a01_skeleton(thresholds, "stages.state.history is missing or empty")

    samples = []
    for position, row in enumerate(history, start=1):
        if not isinstance(row, dict):
            return _a01_skeleton(thresholds, f"stages.state.history[{position - 1}] is not a JSON object")
        epoch = _as_int(row.get("epoch"))
        samples.append(
            {
                "epoch": position if epoch is None else epoch,
                "pos": _as_float(row.get(POS_KEY)),
                "perf": _as_float(row.get(PERFRAME_KEY)),
            }
        )

    without_perf = sum(1 for sample in samples if sample["perf"] is None)
    if without_perf:
        return _a01_skeleton(
            thresholds,
            f"{without_perf}/{len(samples)} stages.state.history rows have no {PERFRAME_KEY}; "
            "refusing to fall back to the pooled danger_corr",
        )
    without_pos = sum(1 for sample in samples if sample["pos"] is None)
    if without_pos:
        return _a01_skeleton(
            thresholds,
            f"{without_pos}/{len(samples)} stages.state.history rows have no {POS_KEY}",
        )

    qualifying = [
        sample
        for sample in samples
        if sample["pos"] < A01_POS_MAX and sample["perf"] > A01_PERFRAME_MIN
    ]
    # Closest miss, for the report: most thresholds met, then the higher per-frame
    # correlation, then the earlier epoch (deterministic).
    closest = max(
        samples,
        key=lambda sample: (
            int(sample["pos"] < A01_POS_MAX) + int(sample["perf"] > A01_PERFRAME_MIN),
            sample["perf"],
            -sample["epoch"],
        ),
    )
    if qualifying:
        chosen = min(qualifying, key=lambda sample: (sample["pos"], sample["epoch"]))
        return {
            "status": "pass",
            "epoch": chosen["epoch"],
            "pos_px": chosen["pos"],
            "perframe_median": chosen["perf"],
            "best_epoch": closest["epoch"],
            "best_pos_px": closest["pos"],
            "best_perframe_median": closest["perf"],
            "epochs_checked": len(samples),
            "thresholds": thresholds,
            "qualifying_epochs": [sample["epoch"] for sample in qualifying],
        }
    return {
        "status": "fail",
        "epoch": None,
        "pos_px": None,
        "perframe_median": None,
        "best_epoch": closest["epoch"],
        "best_pos_px": closest["pos"],
        "best_perframe_median": closest["perf"],
        "epochs_checked": len(samples),
        "thresholds": thresholds,
        "reason": (
            f"no epoch with {POS_KEY} < {A01_POS_MAX} and "
            f"{PERFRAME_KEY} > {A01_PERFRAME_MIN}"
        ),
    }


# --------------------------------------------------------------------------- #
# primary run selection
# --------------------------------------------------------------------------- #
def _min_pos(payload: dict):
    values = [
        value
        for value in (_as_float(row.get(POS_KEY)) for row in state_history(payload) if isinstance(row, dict))
        if value is not None
    ]
    return min(values) if values else None


def infer_primary(payloads: list[tuple[str, dict]]) -> dict | None:
    """Lowest ``val_player_pos_px`` among full v2 runs; screening runs excluded."""
    candidates = []
    for path_text, payload in payloads:
        if not isinstance(state_stage(payload).get("history"), list):
            warn(f"skipping {path_text}: no stages.state.history")
            continue
        if payload.get("arch") != "v2":
            warn(f"skipping {path_text}: arch is {payload.get('arch')!r}, not 'v2'")
            continue
        fraction = _as_float(payload.get("train_fraction"))
        if fraction != 1.0:
            warn(f"skipping {path_text}: train_fraction is {payload.get('train_fraction')!r}, not 1.0")
            continue
        max_train_batches = _as_float(_dict(payload.get("config")).get("max_train_batches"))
        if max_train_batches != 0:
            warn(
                f"skipping {path_text}: config.max_train_batches is "
                f"{_dict(payload.get('config')).get('max_train_batches')!r}, not 0 (screening run)"
            )
            continue
        min_pos = _min_pos(payload)
        if min_pos is None:
            warn(f"skipping {path_text}: no {POS_KEY} in stages.state.history")
            continue
        candidates.append((min_pos, path_text, payload))
    if not candidates:
        return None
    # Ties keep the order the files were given in (sorted() is stable).
    candidates.sort(key=lambda item: item[0])
    min_pos, path_text, payload = candidates[0]
    print(f"primary: {path_text} (best {POS_KEY} = {min_pos:.3f})")
    return {"path": path_text, "payload": payload}


def build_primary(a01: dict, primary: dict | None) -> dict:
    if primary is None:
        return {
            "path": None,
            "arch": None,
            "epoch": None,
            "checkpoint": None,
            "pos_px": None,
            "perframe_median": None,
            "regret_px_bucket_0_15": None,
        }
    payload = primary["payload"]
    node = {
        "path": primary["path"],
        "arch": payload.get("arch"),
        "epoch": None,
        "checkpoint": "state.pt",
        "pos_px": None,
        "perframe_median": None,
        "regret_px_bucket_0_15": bucket_regret(policy_best(payload)),
    }
    if a01["status"] == "pass":
        node.update(
            {
                "epoch": a01["epoch"],
                "checkpoint": "state-pass.pt",
                "pos_px": a01["pos_px"],
                "perframe_median": a01["perframe_median"],
            }
        )
        return node
    # Fail or blocked: report the checkpoint that ``state.pt`` actually holds.
    best = state_best(payload)
    node["epoch"] = _as_int(best.get("epoch"))
    node["pos_px"] = _as_float(best.get(POS_KEY))
    node["perframe_median"] = _as_float(best.get(PERFRAME_KEY))
    if node["epoch"] is None or node["pos_px"] is None or node["perframe_median"] is None:
        warn(
            f"{primary['path']}: stages.state.best is missing "
            f"epoch/{POS_KEY}/{PERFRAME_KEY}; primary epoch left empty"
        )
    return node


# --------------------------------------------------------------------------- #
# A-02: policy regret inside 15 px, read from ``best`` (never ``final``)
# --------------------------------------------------------------------------- #
def evaluate_a02(primary: dict | None) -> dict:
    node = {"status": "blocked", "regret_px_bucket_0_15": None, "threshold": A02_REGRET_MAX}
    if primary is None:
        node["reason"] = NO_PRIMARY_REASON
        return node
    payload = primary["payload"]
    if not isinstance(_dict(payload.get("stages")).get("policy"), dict):
        node["reason"] = "primary run has no policy stage"
        return node
    best = policy_best(payload)
    if not best:
        node["reason"] = "primary run has no stages.policy.best"
        return node
    regret = bucket_regret(best)
    if regret is None:
        node["reason"] = f"stages.policy.best.by_danger has no {BUCKET!r} bucket with regret_px"
        return node
    node["regret_px_bucket_0_15"] = regret
    node["status"] = "pass" if regret < A02_REGRET_MAX else "fail"
    return node


# --------------------------------------------------------------------------- #
# A-03: does the 100% point still beat the 50% point?
# --------------------------------------------------------------------------- #
def r3_point(fraction: float, path_text: str, payload: dict) -> dict:
    best = state_best(payload)
    return {
        "fraction": fraction,
        "path": path_text,
        "train_frames": _as_int(payload.get("train_frames")),
        "val_frames": _as_int(payload.get("val_frames")),
        "pos_px": _as_float(best.get(POS_KEY)),
        "regret_px_bucket_0_15": bucket_regret(policy_best(payload)),
    }


def evaluate_a03(points: dict[float, dict], a01_status: str) -> dict:
    node = {
        "status": "blocked",
        "rule": A03_RULE,
        "points": [points[fraction] for fraction in sorted(points)],
        "gain_pos": None,
        "gain_regret": None,
    }
    if a01_status == "fail":
        node["reason"] = "A-01 not met; R3 premise (architecture first) unmet"
        return node
    if a01_status == "blocked":
        node["reason"] = "A-01 blocked; R3 premise (architecture first) unverified"
        return node
    missing = [fraction for fraction in A03_REQUIRED if fraction not in points]
    if missing:
        node["status"] = "incomplete"
        node["reason"] = "missing R3 point(s): " + ", ".join(f"{fraction:.2f}" for fraction in missing)
        return node

    point_50, point_100 = points[0.5], points[1.0]
    gain_pos, gain_regret = None, None
    if point_50["pos_px"] is not None and point_100["pos_px"] is not None:
        gain_pos = point_100["pos_px"] <= A03_POS_RATIO * point_50["pos_px"]
    if (
        point_50["regret_px_bucket_0_15"] is not None
        and point_100["regret_px_bucket_0_15"] is not None
    ):
        gain_regret = point_100["regret_px_bucket_0_15"] <= (
            A03_REGRET_RATIO * point_50["regret_px_bucket_0_15"]
        )
    node["gain_pos"] = gain_pos
    node["gain_regret"] = gain_regret

    if gain_pos or gain_regret:
        node["status"] = "has_gain"
        return node
    if gain_pos is False and gain_regret is False:
        node["status"] = "plateau"
        return node
    node["status"] = "incomplete"
    node["reason"] = "R3 point(s) lack " + "/".join(
        name
        for predicate, name in (
            (gain_pos is None, POS_KEY),
            (gain_regret is None, f"regret_px_bucket_{BUCKET}"),
        )
        if predicate
    )
    return node


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def _parse_r3(specs: list[str], parser: argparse.ArgumentParser) -> dict[float, str]:
    points: dict[float, str] = {}
    for spec in specs:
        if "=" not in spec:
            parser.error(f"--r3 expects FRACTION=PATH, got {spec!r}")
        fraction_text, path_text = spec.split("=", 1)
        try:
            fraction = round(float(fraction_text), 2)
        except ValueError:
            parser.error(f"--r3 fraction is not a number: {fraction_text!r}")
        if not path_text:
            parser.error(f"--r3 expects FRACTION=PATH, got {spec!r}")
        points[fraction] = path_text
    return points


def _print_verdicts(a01: dict, a02: dict, a03: dict) -> None:
    if a01["status"] == "pass":
        print(
            f"A-01 pass epoch={a01['epoch']} pos_px={a01['pos_px']:.2f} "
            f"perframe_median={a01['perframe_median']:.3f} "
            f"(all {a01['epochs_checked']} epoch(s) scanned)"
        )
    else:
        print(f"A-01 {a01['status']}: {a01.get('reason', '')}")
        if a01["best_epoch"] is not None:
            print(
                f"      closest epoch={a01['best_epoch']} pos_px={a01['best_pos_px']:.2f} "
                f"perframe_median={a01['best_perframe_median']:.3f}"
            )
    if a02["status"] in ("pass", "fail"):
        print(
            f"A-02 {a02['status']} regret_0_15={a02['regret_px_bucket_0_15']:.2f} "
            f"(threshold < {A02_REGRET_MAX})"
        )
    else:
        print(f"A-02 {a02['status']}: {a02.get('reason', '')}")
    print(f"A-03 {a03['status']}" + (f": {a03['reason']}" if a03.get("reason") else ""))


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    r3_paths = _parse_r3(args.r3, parser)
    summary_paths = list(args.summary)

    payloads: list[tuple[str, dict]] = []
    for path_text in summary_paths:
        payloads.append((path_text, load_summary(path_text)))
    r3_payloads = {
        fraction: load_summary(path_text) for fraction, path_text in sorted(r3_paths.items())
    }

    if args.primary:
        primary_payload = load_summary(args.primary)
        if not isinstance(_dict(primary_payload.get("stages")).get("state"), dict):
            raise InputError(f"--primary has no stages.state: {args.primary}")
        primary = {"path": args.primary, "payload": primary_payload}
        print(f"primary: {args.primary} (given explicitly)")
    else:
        primary = infer_primary(payloads)

    a01 = evaluate_a01(state_history(primary["payload"])) if primary is not None else _a01_skeleton(
        {"pos_px": A01_POS_MAX, "perframe_median": A01_PERFRAME_MIN}, NO_PRIMARY_REASON
    )
    if a01["status"] == "fail" and primary is not None:
        best = state_best(primary["payload"])
        if _as_float(best.get("epoch")) is None or _as_float(best.get(POS_KEY)) is None:
            a01 = _a01_skeleton(
                a01["thresholds"],
                "stages.state.best is missing; cannot point the primary run at a checkpoint",
            )
    a02 = evaluate_a02(primary)
    points = {
        fraction: r3_point(fraction, path_text, r3_payloads[fraction])
        for fraction, path_text in sorted(r3_paths.items())
    }
    a03 = evaluate_a03(points, a01["status"])

    document = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "summaries": summary_paths,
            "r3": {f"{fraction:.2f}": path_text for fraction, path_text in sorted(r3_paths.items())},
        },
        "primary": build_primary(a01, primary),
        "a01": a01,
        "a02": a02,
        "a03": a03,
        "r4_authorized": False,
        "note": R4_NOTE,
    }
    _print_verdicts(a01, a02, a03)
    if args.out:
        out_path = Path(args.out)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            raise InputError(f"cannot write {args.out}: {exc}") from exc
        print(f"written: {args.out}")
    else:
        print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0 if (a01["status"] == "pass" and a02["status"] == "pass") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decide A-01/A-02/A-03 from vision-summary.json files"
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="PATH",
        help="one run summary; repeat for each candidate (used to infer the primary run)",
    )
    parser.add_argument(
        "--primary",
        default="",
        metavar="PATH",
        help="summary of the full run that A-01/A-02 judge (disables inference)",
    )
    parser.add_argument(
        "--r3",
        action="append",
        default=[],
        metavar="FRACTION=PATH",
        help="R3 data-scale point, e.g. 0.50=runs/v103-exp-06/vision-summary.json",
    )
    parser.add_argument("--out", default="", metavar="PATH", help="where to write the JSON verdict")
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
