"""EXP-P3-009: evaluation power for S1N closed-loop rows.

Why this file exists
--------------------
EXP-P3-004 → EXP-P3-008 compared closed-loop S1N rows as **five-episode means**
and read differences of 150–550 frames as effects (``+552`` for 8→27 shards,
``+447.8`` for ``geo_restore``, ``−147.5`` for ``verify_k`` 1→3, ``−165/−354`` for
the v2j action set).  EXP-P3-008 §7.4 then measured two checkpoints that share
init, data order, split and τ — M1 vs M1b, identical config, M1b merely trained
73 % longer — and they differ by **1,330 frames** while their validation top-1
differs by 0.0004.  That is not evaluation noise (re-running one checkpoint
reproduces every episode frame for frame); it is *training* noise: which step you
stop at is worth more than every lever the project has chased.

So a closed-loop row has two variance components, and five seeds only measures
the second one:

    σ²_total = σ²_ckpt + σ²_episode / n

* ``σ²_episode`` — per-episode spread inside one checkpoint.  Twenty seeds nail
  this.  It is *not* the quantity that decides whether an arm is better.
* ``σ²_ckpt`` — spread of checkpoint means over retrainings of the same recipe.
  Five seeds cannot see it at all, because they all come from one training run.

This module does four things, all of them arithmetic on recorded episodes:

1. ``noise`` — one-way random-effects decomposition of a set of evaluated
   checkpoints into the two components above, plus ICC.
2. ``power`` — how many episodes × how many checkpoints are needed to resolve a
   given frame difference at ``power`` / ``alpha``.
3. ``claim`` — the criterion template: **a difference is an effect only when both
   arms report a CI and those CIs are disjoint**.  A comparison with no CI is
   refused (exit 2), never reported as "small".
4. ``baseline`` — the redefined baseline row: mean, sd, se, bootstrap CI, and the
   checkpoint-level interval implied by the measured ``σ²_ckpt``.

Nothing here touches the engine, the protocol or a checkpoint: it reads
``summary.json`` / ``episodes.jsonl`` directories that
``tools/rl/v400_student_eval.py`` wrote.

Run::

    python -X utf8 tools/rl/v400_eval_power.py noise \
      --runs m1=.tmp_v400_power/m1,m1b=.tmp_v400_power/m1b,armF=.tmp_v400_power/armF \
      --json .tmp_v400_power/noise.json
    python -X utf8 tools/rl/v400_eval_power.py power \
      --noise .tmp_v400_power/noise.json --deltas 300,550
    python -X utf8 tools/rl/v400_eval_power.py claim \
      --a m1=.tmp_v400_power/m1 --b m1b=.tmp_v400_power/m1b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

SCHEMA = "v400-eval-power-1"
DEFAULT_LEVEL = 0.95
DEFAULT_ITERS = 20000
#: EXP-P3-009 protocol floor: fewer episodes than this cannot bound the mean.
MIN_EPISODES = 20
#: Two-sided by default: nobody pre-registers a direction they would accept losing.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
#: The four effects the project has published as "levers", in frames.
DEFAULT_DELTAS = (300.0, 550.0)


class PowerError(RuntimeError):
    """The inputs cannot support the requested statement."""


# --------------------------------------------------------------------------- #
# normal distribution without scipy
# --------------------------------------------------------------------------- #

def norm_cdf(value: float) -> float:
    """Φ(x) via ``erf`` (max abs error ~1e-7, ample for power arithmetic)."""

    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def norm_ppf(probability: float) -> float:
    """Φ⁻¹(p), Acklam's rational approximation (|ε| < 1.15e-9)."""

    p = float(probability)
    if not 0.0 < p < 1.0:
        raise PowerError(f"norm_ppf needs 0 < p < 1, got {p!r}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    low, high = 0.02425, 1.0 - 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def z_critical(alpha: float = DEFAULT_ALPHA) -> float:
    """Two-sided critical value: ``z`` with ``P(|Z| > z) = alpha``."""

    if not 0.0 < float(alpha) < 1.0:
        raise PowerError(f"alpha {alpha!r} must be in (0, 1)")
    return norm_ppf(1.0 - float(alpha) / 2.0)


# --------------------------------------------------------------------------- #
# reading recorded episodes
# --------------------------------------------------------------------------- #

def load_arm(path: str | Path, *, level: float = DEFAULT_LEVEL,
             iters: int = DEFAULT_ITERS, seed: int = 0) -> dict:
    """Read one evaluated arm from a directory written by the eval script.

    Accepts ``episodes.jsonl`` (preferred: per-episode frames), or a
    ``summary.json`` that already carries a CI.  A summary with a bare
    ``mean_frames`` and no episodes yields ``ci_low = None`` — deliberately, so
    every downstream verdict can refuse it instead of pretending it has an
    interval.
    """

    root = Path(path)
    if root.is_file():
        root = root.parent
    episodes_path = root / "episodes.jsonl"
    summary_path = root / "summary.json"
    frames: list[float] = []
    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt artefact
            raise PowerError(f"{summary_path} is not JSON: {exc}") from exc
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise PowerError(f"{episodes_path} has a bad line: {exc}") from exc
            if record.get("end") == "error":  # a failed episode is not a measurement
                continue
            frames.append(float(record["frames"]))
    elif isinstance(summary.get("frames"), list):
        frames = [float(value) for value in summary["frames"]]
    stats = frames_statistics(frames, level=level, iters=iters, seed=seed)
    stats["frames"] = [float(value) for value in frames]
    if stats["ci_low"] is None and summary.get("ci_low") is not None:
        # A summary written by hand (or by an older tool) may carry the interval
        # without the episodes; trust it, but say where it came from.
        stats.update(mean=summary.get("mean_frames"), n=summary.get("episodes"),
                     ci_low=summary["ci_low"], ci_high=summary.get("ci_high"),
                     ci_level=summary.get("ci_level", level),
                     sd=summary.get("sd_frames"), se=summary.get("se_frames"),
                     ci_source=str(summary_path))
    stats["label"] = summary.get("label") or root.name
    stats["path"] = str(root)
    stats["checkpoint_sha256"] = summary.get("checkpoint_sha256")
    stats["seeds"] = summary.get("seed_list") or summary.get("seeds")
    stats["obs_window"] = summary.get("obs_window")
    stats["max_frames"] = summary.get("max_frames")
    stats["stage"] = summary.get("stage")
    stats["stop_on_death"] = summary.get("stop_on_death")
    return stats


def frames_statistics(frames, *, level: float = DEFAULT_LEVEL,
                      iters: int = DEFAULT_ITERS, seed: int = 0) -> dict:
    """Mean / sd / se / percentile-bootstrap CI of per-episode frame counts."""

    array = np.asarray(list(frames), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None, "median": None,
                "min": None, "max": None, "ci_low": None, "ci_high": None,
                "ci_level": float(level), "bootstrap_iters": int(iters),
                "ci_half_width": None}
    n = int(array.size)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else None
    se = float(sd / math.sqrt(n)) if sd is not None else None
    lo = hi = None
    if n > 1:
        rng = np.random.default_rng(int(seed))
        draws = rng.integers(0, n, size=(int(iters), n))
        means = array[draws].mean(axis=1)
        alpha = (1.0 - float(level)) / 2.0
        lo, hi = (float(value) for value in
                  np.quantile(means, [alpha, 1.0 - alpha]))
    else:
        lo = hi = mean
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "median": float(np.median(array)),
        "min": float(array.min()), "max": float(array.max()),
        "ci_low": lo, "ci_high": hi, "ci_level": float(level),
        "bootstrap_iters": int(iters),
        "ci_half_width": float((hi - lo) / 2.0) if lo is not None else None,
    }


# --------------------------------------------------------------------------- #
# variance components
# --------------------------------------------------------------------------- #

def variance_components(groups: dict[str, list[float]] | dict[str, np.ndarray]) -> dict:
    """One-way random-effects decomposition over checkpoints.

    ``y_ij`` = episode *j* of checkpoint *i*.  With *k* checkpoints of sizes *n_i*:

    * ``σ²_episode = MSW`` — pooled within-checkpoint variance,
    * ``σ²_ckpt    = (MSB − MSW) / n0`` — the between-checkpoint variance
      (moment estimator; truncated at 0, the untruncated value is reported),
    * ``n0 = (N − Σ n_i²/N) / (k − 1)``.

    Interpretation that drives the whole card: the variance of an arm's reported
    mean over *n* episodes is ``σ²_ckpt + σ²_episode/n``.  The second term can be
    driven to zero with more seeds; the first cannot.
    """

    clean = {}
    for label, values in groups.items():
        array = np.asarray(list(values), dtype=np.float64)
        if array.size:
            clean[str(label)] = array
    k = len(clean)
    if k < 2:
        raise PowerError(f"variance decomposition needs >= 2 checkpoints, got {k}")
    sizes = np.array([array.size for array in clean.values()], dtype=np.float64)
    total = int(sizes.sum())
    if total - k < 1:
        raise PowerError("every checkpoint needs >= 2 episodes to separate the "
                         "two components")
    means = {label: float(array.mean()) for label, array in clean.items()}
    grand = float(np.sum(sizes * np.array([means[label] for label in clean])) / total)
    ss_between = float(np.sum(sizes * np.array(
        [(means[label] - grand) ** 2 for label in clean])))
    ss_within = float(np.sum([float(((array - means[label]) ** 2).sum())
                              for label, array in clean.items()]))
    df_between = k - 1
    df_within = total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    n0 = (total - float(np.sum(sizes ** 2)) / total) / df_between
    sigma2_episode = ms_within
    raw_ckpt = (ms_between - ms_within) / n0
    sigma2_ckpt = max(0.0, raw_ckpt)
    sd_episode = math.sqrt(sigma2_episode)
    sd_ckpt = math.sqrt(sigma2_ckpt)
    denom = sigma2_ckpt + sigma2_episode
    return {
        "checkpoints": {label: {"n": int(array.size), "mean": means[label],
                                "sd": float(array.std(ddof=1)) if array.size > 1 else None}
                        for label, array in clean.items()},
        "k": k,
        "n_total": total,
        "n0": n0,
        "grand_mean": grand,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "df_between": df_between,
        "df_within": df_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "sigma2_episode": sigma2_episode,
        "sigma2_checkpoint": sigma2_ckpt,
        "sigma2_checkpoint_raw": raw_ckpt,
        "sd_episode": sd_episode,
        "sd_checkpoint": sd_ckpt,
        "icc": (sigma2_ckpt / denom) if denom > 0 else None,
        "sd_of_arm_mean_one_checkpoint": math.sqrt(sigma2_ckpt + sigma2_episode),
        "se_of_arm_mean_with_n": {
            str(n): math.sqrt(sigma2_ckpt + sigma2_episode / n)
            for n in (5, 10, 20, 40, 80)
        },
        "note": "sigma2_checkpoint is a moment estimate on a handful of "
                "checkpoints: it is a *floor* on how much a retrain moves the "
                "number, not a calibrated prior.",
    }


# --------------------------------------------------------------------------- #
# power
# --------------------------------------------------------------------------- #

def se_of_difference(sigma2_ckpt: float, sigma2_episode: float, *,
                     k: int = 1, n: int = 20) -> float:
    """SE of (arm A mean − arm B mean) for ``k`` checkpoints × ``n`` episodes each.

    Both arms are independent retrainings, so the checkpoint components add; they
    do **not** cancel, which is why a single-checkpoint pair of arms has a floor
    ``sqrt(2 σ²_ckpt)`` that no number of episodes can lower.
    """

    if k < 1 or n < 1:
        raise PowerError("k and n must be >= 1")
    return math.sqrt(2.0 * (float(sigma2_ckpt) / k + float(sigma2_episode) / (k * n)))


def achievable_power(delta: float, se: float, *, alpha: float = DEFAULT_ALPHA) -> float:
    """Two-sided power of a normal test at ``|Δ| / se``."""

    if se <= 0:
        return 1.0
    z = z_critical(alpha)
    shift = abs(float(delta)) / float(se)
    return norm_cdf(shift - z) + norm_cdf(-shift - z)


def required_design(sigma2_ckpt: float, sigma2_episode: float, delta: float, *,
                    power: float = DEFAULT_POWER, alpha: float = DEFAULT_ALPHA,
                    max_checkpoints: int = 12, max_episodes: int = 400) -> dict:
    """Minimum (k, n) — and the cheapest such design — to resolve ``delta``.

    Returns the required SE, the checkpoint-count floor (``k_min``), and for each
    ``k`` the smallest ``n`` meeting the target power, so the caller can see the
    exchange rate between retrainings and episodes.
    """

    if not 0.0 < float(power) < 1.0:
        raise PowerError(f"power {power!r} must be in (0, 1)")
    if float(delta) == 0:
        raise PowerError("delta must be non-zero")
    z_alpha = z_critical(alpha)
    z_power = norm_ppf(float(power))
    se_required = abs(float(delta)) / (z_alpha + z_power)
    floor_terms = 2.0 * float(sigma2_ckpt)  # SE² with k -> inf
    k_floor = (floor_terms / se_required ** 2) if se_required > 0 else float("inf")
    rows = []
    best = None
    for k in range(1, int(max_checkpoints) + 1):
        need = se_required ** 2
        # se² = 2σ²_ckpt/k + 2σ²_ep/(k n)  ->  n = 2σ²_ep / (k·need − 2σ²_ckpt)
        remaining = k * need - 2.0 * float(sigma2_ckpt)
        if remaining <= 0:
            rows.append({"k": k, "n": None, "episodes": None,
                         "se": se_of_difference(sigma2_ckpt, sigma2_episode, k=k, n=1),
                         "power": achievable_power(delta, se_of_difference(
                             sigma2_ckpt, sigma2_episode, k=k, n=max_episodes),
                             alpha=alpha),
                         "note": "checkpoint floor: no n resolves this Δ at this k"})
            continue
        n = int(math.ceil(2.0 * float(sigma2_episode) / remaining))
        n = max(n, 2)
        if n > int(max_episodes):
            rows.append({"k": k, "n": None, "episodes": None,
                         "se": se_of_difference(sigma2_ckpt, sigma2_episode, k=k, n=n),
                         "power": achievable_power(delta, se_of_difference(
                             sigma2_ckpt, sigma2_episode, k=k, n=n), alpha=alpha),
                         "note": f"needs n={n} > max_episodes={max_episodes}"})
            continue
        se = se_of_difference(sigma2_ckpt, sigma2_episode, k=k, n=n)
        row = {"k": k, "n": n, "episodes": k * n, "se": se,
               "power": achievable_power(delta, se, alpha=alpha),
               "ci_half_width_per_arm": z_alpha * math.sqrt(
                   sigma2_ckpt + sigma2_episode / n)}
        rows.append(row)
        if best is None or row["episodes"] < best["episodes"]:
            best = row
    feasible = [row for row in rows if row["n"]]
    return {
        "delta": float(delta),
        "power_target": float(power),
        "alpha": float(alpha),
        "se_required": se_required,
        "z_alpha": z_alpha,
        "z_power": z_power,
        "checkpoint_floor_k": (k_floor if math.isfinite(k_floor)
                               else float(k_floor)),
        "se_floor_with_many_checkpoints": math.sqrt(floor_terms) if floor_terms else 0.0,
        "feasible": bool(feasible),
        "cheapest": best,
        "cheapest_by_cost": min(feasible, key=lambda row: (row["episodes"], row["k"]))
        if feasible else None,
        "rows": rows,
        "note": "normal approximation, two-sided, independent retrainings; "
                "paired-by-seed designs are strictly more powerful, so these are "
                "upper bounds on the cost.",
    }


# --------------------------------------------------------------------------- #
# the criterion template
# --------------------------------------------------------------------------- #

VERDICT_EFFECT = "有效应"
VERDICT_OVERLAP = "无法判定（CI 重叠）"
VERDICT_NO_CI = "无法判定（缺 CI）"
VERDICT_BELOW_FLOOR = "无法判定（分辨率不足）"

CRITERION_TEMPLATE = (
    "差异只有在【两个臂都给出同一置信水平的 CI】且【两个 CI 不重叠】时才可称为"
    "'有效应'；否则一律记为'无法判定'，禁止写成'小幅效应'/'方向一致'/'趋势'。")


def compare(a: dict, b: dict, *, alpha: float = DEFAULT_ALPHA) -> dict:
    """Apply :data:`CRITERION_TEMPLATE` to two arm descriptions.

    ``a``/``b`` are dicts with at least ``label``, ``mean``, ``ci_low``,
    ``ci_high``, ``n`` (as produced by :func:`load_arm`).  Fail closed: a missing
    interval is a refusal, not a warning.
    """

    problems = []
    for role, arm in (("a", a), ("b", b)):
        if arm.get("ci_low") is None or arm.get("ci_high") is None:
            problems.append(f"{role} ({arm.get('label')}) has no CI")
        if arm.get("mean") is None:
            problems.append(f"{role} ({arm.get('label')}) has no mean")
    result = {
        "criterion": CRITERION_TEMPLATE,
        "a": {"label": a.get("label"), "mean": a.get("mean"),
              "ci": [a.get("ci_low"), a.get("ci_high")], "n": a.get("n"),
              "sd": a.get("sd")},
        "b": {"label": b.get("label"), "mean": b.get("mean"),
              "ci": [b.get("ci_low"), b.get("ci_high")], "n": b.get("n"),
              "sd": b.get("sd")},
    }
    if problems:
        result.update(verdict=VERDICT_NO_CI, delta=None, problems=problems)
        return result
    delta = float(a["mean"]) - float(b["mean"])
    lo_a, hi_a = float(a["ci_low"]), float(a["ci_high"])
    lo_b, hi_b = float(b["ci_low"]), float(b["ci_high"])
    disjoint = hi_a < lo_b or hi_b < lo_a
    overlap = [max(lo_a, lo_b), min(hi_a, hi_b)]
    warnings = []
    for arm in (a, b):
        if arm.get("n") is not None and int(arm["n"]) < MIN_EPISODES:
            warnings.append(
                f"{arm.get('label')}: n={arm['n']} < protocol floor {MIN_EPISODES}")
    result.update(
        delta=delta,
        ci_overlap=bool(not disjoint),
        overlap_interval=overlap if not disjoint else None,
        warnings=warnings,
        verdict=(VERDICT_EFFECT if disjoint else VERDICT_OVERLAP),
        problems=[],
    )
    return result


def apply_checkpoint_floor(delta: float, sigma2_ckpt: float, *, k: int = 1,
                           alpha: float = DEFAULT_ALPHA) -> dict:
    """Can a single-checkpoint-pair delta survive the checkpoint-variance floor?"""

    se_floor = math.sqrt(2.0 * float(sigma2_ckpt) / max(int(k), 1))
    needed = z_critical(alpha) * se_floor
    return {
        "delta": float(delta),
        "se_floor": se_floor,
        "resolution_limit": needed,
        "resolvable": abs(float(delta)) > needed,
        "interpretation": ("|Δ| 小于 σ²_ckpt 给出的分辨极限，即使 n=∞ 也无法判定"
                           if abs(float(delta)) <= needed else
                           "|Δ| 超过 checkpooint 分辨极限，但仍需 CI 不重叠才算效应"),
    }


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #

def parse_named_run(text: str) -> tuple[str, str]:
    """``"label=dir"`` (a bare path uses the directory name as label)."""

    if "=" in text:
        label, path = text.split("=", 1)
        label, path = label.strip(), path.strip()
    else:
        label, path = Path(text).name, text
    if not label or not path:
        raise PowerError(f"--runs entry {text!r} is not 'label=dir'")
    return label, path


def format_stats(stats: dict) -> str:
    if stats.get("mean") is None:
        return f"{stats.get('label')}: n=0"
    if stats.get("ci_low") is None:
        return (f"{stats.get('label')}: n={stats['n']} mean={stats['mean']:.1f} "
                f"(no CI)")
    return (f"{stats.get('label')}: n={stats['n']} mean={stats['mean']:.1f} "
            f"sd={stats['sd'] if stats['sd'] is None else round(stats['sd'], 1)} "
            f"se={stats['se'] if stats['se'] is None else round(stats['se'], 1)} "
            f"CI{int(round(stats.get('ci_level', DEFAULT_LEVEL) * 100))}="
            f"[{stats['ci_low']:.1f}, {stats['ci_high']:.1f}]")


def _json_dump(payload: dict, path: str | None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    noise = sub.add_parser("noise", help="variance components over checkpoints")
    noise.add_argument("--runs", action="append", default=[],
                       help="label=dir (repeatable); each dir is one evaluated "
                            "checkpoint")
    noise.add_argument("--same-config", action="append", default=[],
                       help="labels that share one config (one such group is the "
                            "pure σ²_ckpt estimate)")
    noise.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    noise.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    noise.add_argument("--json", default=None)

    power = sub.add_parser("power", help="episodes x checkpoints for a frame delta")
    power.add_argument("--noise", required=True, help="noise.json from 'noise'")
    power.add_argument("--deltas", default=",".join(str(int(d)) for d in DEFAULT_DELTAS))
    power.add_argument("--target-power", type=float, default=DEFAULT_POWER)
    power.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    power.add_argument("--max-checkpoints", type=int, default=12)
    power.add_argument("--max-episodes", type=int, default=400)
    power.add_argument("--json", default=None)

    claim = sub.add_parser("claim", help="apply the criterion template")
    claim.add_argument("--a", required=True, help="label=dir")
    claim.add_argument("--b", required=True, help="label=dir")
    claim.add_argument("--delta-label", default=None,
                       help="what the comparison is, for the record")
    claim.add_argument("--sigma2-checkpoint", type=float, default=None,
                       help="measured σ²_ckpt, to also report the resolution floor")
    claim.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    claim.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    claim.add_argument("--json", default=None)
    claim.add_argument("--allow-missing-ci", action="store_true",
                       help="NOT supported: refusal on a missing CI is the point")

    baseline = sub.add_parser("baseline", help="one arm, with its interval")
    baseline.add_argument("--run", required=True, help="dir")
    baseline.add_argument("--sigma2-checkpoint", type=float, default=None)
    baseline.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    baseline.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    baseline.add_argument("--json", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "noise":
        arms = {}
        for text in args.runs:
            label, path = parse_named_run(text)
            stats = load_arm(path, level=args.level, iters=args.iters)
            arms[label] = stats
        if len(arms) < 2:
            raise SystemExit("noise needs >= 2 --runs")
        groups = {label: arms[label].get("frames")
                  for label in arms if arms[label].get("frames")}
        if any(not value for value in groups.values()):
            missing = [label for label in arms if not arms[label].get("frames")]
            raise SystemExit(f"no per-episode frames for {missing}; noise needs "
                             f"episodes.jsonl or summary.frames")
        payload = {"schema": SCHEMA, "command": "noise",
                   "arms": {label: {key: stats.get(key) for key in
                                    ("label", "path", "n", "mean", "sd", "se",
                                     "ci_low", "ci_high", "ci_level", "min",
                                     "max", "seeds", "obs_window", "max_frames",
                                     "stage", "stop_on_death",
                                     "checkpoint_sha256")}
                            for label, stats in arms.items()}}
        payload["variance_components_all"] = variance_components(groups)
        if args.same_config:
            wanted = [part.strip()
                      for value in args.same_config
                      for part in str(value).split(",") if part.strip()]
            subset = {label: groups[label] for label in wanted if label in groups}
            missing = [label for label in wanted if label not in groups]
            if missing:
                payload["same_config_missing"] = missing
            if len(subset) >= 2:
                payload["variance_components_same_config"] = variance_components(subset)
            else:
                payload["variance_components_same_config"] = None
        else:
            payload["variance_components_same_config"] = None
        _json_dump(payload, args.json)
        return 0
    if args.command == "power":
        noise = json.loads(Path(args.noise).read_text(encoding="utf-8"))
        component = (noise.get("variance_components_same_config")
                     or noise.get("variance_components_all"))
        if not component:
            raise SystemExit(f"{args.noise} has no variance components")
        sigma2_ckpt = float(component["sigma2_checkpoint"])
        sigma2_ep = float(component["sigma2_episode"])
        deltas = [float(part) for part in args.deltas.split(",") if part.strip()]
        payload = {"schema": SCHEMA, "command": "power",
                   "sigma2_checkpoint": sigma2_ckpt,
                   "sigma2_episode": sigma2_ep,
                   "sd_checkpoint": math.sqrt(sigma2_ckpt),
                   "sd_episode": math.sqrt(sigma2_ep),
                   "component_source": ("same_config"
                                        if noise.get("variance_components_same_config")
                                        else "all"),
                   "designs": [required_design(
                       sigma2_ckpt, sigma2_ep, delta,
                       power=args.target_power, alpha=args.alpha,
                       max_checkpoints=args.max_checkpoints,
                       max_episodes=args.max_episodes) for delta in deltas]}
        _json_dump(payload, args.json)
        return 0
    if args.command == "claim":
        if args.allow_missing_ci:
            raise SystemExit("--allow-missing-ci is not supported: a comparison "
                             "without a CI has no verdict")
        label_a, path_a = parse_named_run(args.a)
        label_b, path_b = parse_named_run(args.b)
        arm_a = load_arm(path_a, level=args.level, iters=args.iters)
        arm_b = load_arm(path_b, level=args.level, iters=args.iters)
        arm_a["label"], arm_b["label"] = label_a, label_b
        verdict = compare(arm_a, arm_b)
        verdict["schema"] = SCHEMA
        verdict["command"] = "claim"
        verdict["delta_label"] = args.delta_label
        if args.sigma2_checkpoint is not None and verdict.get("delta") is not None:
            floor = apply_checkpoint_floor(verdict["delta"], args.sigma2_checkpoint)
            verdict["checkpoint_floor"] = floor
            if not floor["resolvable"] and verdict["verdict"] == VERDICT_EFFECT:
                # Condition 3 of the template: a CI-disjoint pair still cannot
                # license an effect when |Δ| is inside the retraining band.
                verdict["ci_verdict"] = verdict["verdict"]
                verdict["verdict"] = VERDICT_BELOW_FLOOR
                verdict.setdefault("problems", []).append(
                    f"|Δ|={abs(verdict['delta']):.1f} < resolution limit "
                    f"{floor['resolution_limit']:.1f}: a single-training pair "
                    f"cannot separate this from retraining drift")
        _json_dump(verdict, args.json)
        return 0 if verdict["verdict"] == VERDICT_EFFECT else 2
    if args.command == "baseline":
        stats = load_arm(args.run, level=args.level, iters=args.iters)
        payload = {"schema": SCHEMA, "command": "baseline",
                   "arm": stats, "criterion": CRITERION_TEMPLATE,
                   "protocol_floor_episodes": MIN_EPISODES}
        if args.sigma2_checkpoint is not None:
            sigma2 = float(args.sigma2_checkpoint)
            if stats.get("mean") is not None:
                n = int(stats.get("n") or 0)
                sd_arm = math.sqrt(sigma2 + (float(stats["sd"]) ** 2 if stats.get("sd")
                                             else 0.0) / max(n, 1))
                half = z_critical() * sd_arm
                payload["checkpoint_level_interval"] = {
                    "sigma2_checkpoint": sigma2,
                    "se_of_arm_mean": sd_arm,
                    "ci_low": float(stats["mean"]) - half,
                    "ci_high": float(stats["mean"]) + half,
                    "note": "the same row if the recipe were retrained: this is the "
                            "interval a rival arm actually has to beat",
                }
        _json_dump(payload, args.json)
        return 0
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
