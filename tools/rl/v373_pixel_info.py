"""Round27: is the F1 required direction present in the pixels, and learnable?

R20/R21 showed that F1's coverage gap is not rendered before the lethal frame and
that adding a telegraph made the policy *worse*.  That left two readings:

* the information is still not in the pixels (task design is broken), or
* the information is there and the frozen policy simply does not use it.

This tool answers that question without training any policy:

1. a *direct reader* (no learning) reconstructs the gap and the player from the
   rendered playfield and checks the sign, on the 12 valid R18 worlds and on 120
   generated ones;
2. a small CNN is trained on generated **base** worlds only and tested on their
   **mirrored** twins (the transfer that the policy fails), plus the 12 valid
   worlds as an untouched holdout;
3. both are repeated on the pre-telegraph renderer as a control arm, where the
   same information should be absent.

Pre-registration: ``Document/Update/Alpha/v3.7.3/preregistration/p1-pixel-info-round27.md``.

    python tools/rl/v373_pixel_info.py --out Document/Update/Alpha/v3.7.3/evidence/pixel-info-round27
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools" / "rl") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "rl"))

PLAYFIELD_LEFT, PLAYFIELD_TOP, PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT = 32, 16, 384, 448
GAP_EDGE_WIDTH = 4          # mirrors v373_telegraph.EDGE_WIDTH
TELEGRAPH_LEAD = 45
DOWNSCALE = (112, 96)       # (height, width) fed to the probe
DECISION_LEAD = 44          # even frame inside [wall-45, wall)

FILL_COLOUR = np.array([44, 58, 74], dtype=np.uint8)
EDGE_COLOUR = np.array([96, 148, 128], dtype=np.uint8)
PLAYER_COLOUR = np.array([90, 225, 255], dtype=np.uint8)
BULLET_COLOUR = np.array([255, 205, 65], dtype=np.uint8)
BACKGROUND_COLOUR = np.array([10, 14, 24], dtype=np.uint8)

VALID_DIR = ROOT / "Document/Update/Alpha/v3.7.3/evidence/worldfilter-round18/run/filters/F1"
COVERAGES = ((8.0, 376.0), (8.0, 372.0), (12.0, 376.0), (16.0, 368.0))


# ---------------------------------------------------------------------------
# Direct pixel reader (no world state, no learning)
# ---------------------------------------------------------------------------
def _clusters(columns) -> list[list[int]]:
    groups: list[list[int]] = []
    for column in columns:
        if groups and column - groups[-1][-1] <= 2:
            groups[-1].append(int(column))
        else:
            groups.append([int(column)])
    return groups


def read_direction(image: np.ndarray) -> dict:
    """Read ``sign(gap_center - player_x)`` from the pixels alone.

    Returns ``{"ok": False, "reason": ...}`` when the image carries no readable
    preview band (which is the expected result on the pre-telegraph arm).
    """
    playfield = image[
        PLAYFIELD_TOP : PLAYFIELD_TOP + PLAYFIELD_HEIGHT,
        PLAYFIELD_LEFT : PLAYFIELD_LEFT + PLAYFIELD_WIDTH,
        :,
    ]
    edge_mask = np.all(playfield == EDGE_COLOUR, axis=-1)
    fill_mask = np.all(playfield == FILL_COLOUR, axis=-1)
    player_mask = np.all(playfield == PLAYER_COLOUR, axis=-1)

    if not fill_mask.any():
        return {"ok": False, "reason": "no preview band", "bandPixels": 0}
    edge_columns = np.nonzero(edge_mask.any(axis=0))[0]
    groups = _clusters(edge_columns)
    if len(groups) != 2:
        return {
            "ok": False,
            "reason": f"{len(groups)} edge clusters (expected 2)",
            "bandPixels": int(fill_mask.sum()),
        }
    player_columns = np.nonzero(player_mask.any(axis=0))[0]
    if not player_columns.size:
        return {"ok": False, "reason": "no player sprite", "bandPixels": int(fill_mask.sum())}
    gap_left = groups[0][-1] + 1
    gap_right = groups[1][0] - 1
    if gap_right < gap_left:
        return {"ok": False, "reason": "empty gap between edge markers"}
    gap_center = 0.5 * (gap_left + gap_right)
    player_x = float(np.median(player_columns))
    direction = 1 if gap_center > player_x else -1
    return {
        "ok": True,
        "gapLeft": gap_left,
        "gapRight": gap_right,
        "gapCenter": gap_center,
        "gapWidth": gap_right - gap_left + 1,
        "playerX": player_x,
        "direction": direction,
        "bandPixels": int(fill_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------
def load_template_scenario() -> dict:
    sample = json.loads(sorted(VALID_DIR.glob("wf18-F1-*-m0.json"))[0].read_text(encoding="utf-8"))
    return dict(sample["scenario"])


def load_valid_worlds() -> list[dict]:
    from touhou_rl import tactical_teacher_validation as tv

    worlds = []
    for path in sorted(VALID_DIR.glob("wf18-F1-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario = dict(payload["scenario"])
        worlds.append(
            {
                "eventId": payload["event_id"],
                "group": payload["group"],
                "mirror": bool(scenario["mirror"]),
                "source": "valid-r18",
                "scenario": scenario,
                "world": tv.ValidationWorld(**scenario),
            }
        )
    return worlds


def generate_worlds(groups: int, seed: int) -> list[dict]:
    """Base/mirror pairs where the player's x carries no information about the label.

    Two confounding generators were caught by the probe's own control before this
    one: (a) ``start_x`` drawn *relative* to the gap, and (b) independent draws
    followed by ``|gap-start| >= 30``, which makes one label impossible near the
    screen edges.  Here x is confined to the band where both signs are feasible
    for every x, and the sign is drawn 50/50, so ``P(label | x) = 0.5`` exactly.
    """
    from touhou_rl import tactical_teacher_validation as tv

    template = load_template_scenario()
    rng = random.Random(seed)
    gap_min, gap_max = 80.0, 304.0
    x_min, x_max = gap_min + 30.0, gap_max - 30.0   # 110 .. 274: both signs feasible
    worlds = []
    positives = 0
    negatives = 0
    index = 0
    attempts = 0
    target_each = max(1, groups // 2)
    while (positives < target_each or negatives < target_each) and attempts < groups * 400:
        attempts += 1
        coverage = rng.choice(COVERAGES)
        gap_width = rng.choice((12.0, 14.0, 16.0, 18.0, 20.0))
        canonical_start = rng.uniform(x_min, x_max)
        sign = 1 if (positives < target_each and negatives >= target_each) else (
            -1 if (negatives < target_each and positives >= target_each) else rng.choice((1, -1))
        )
        bound = (gap_max - canonical_start) if sign > 0 else (canonical_start - gap_min)
        if bound < 30.0:
            continue
        distance = rng.uniform(30.0, min(120.0, bound))
        gap_center = canonical_start + sign * distance
        gap = [gap_center - gap_width / 2.0, gap_center + gap_width / 2.0]
        if gap[0] < coverage[0] + 4.0 or gap[1] > coverage[1] - 4.0:
            continue
        wall_frame = float(rng.choice(range(48, 92, 2)))
        if sign > 0:
            positives += 1
        else:
            negatives += 1
        for mirror in (False, True):
            scenario = dict(template)
            scenario.update(
                {
                    "event_id": f"gen27-F1-{index:03d}-m{int(mirror)}",
                    "template_group": f"gen27-F1-{index:03d}",
                    "family": "F1",
                    "seed": seed + index,
                    "split": "development",
                    "start_x": 384.0 - canonical_start if mirror else canonical_start,
                    "mirror": bool(mirror),
                    "coverage": [float(coverage[0]), float(coverage[1])],
                    "gap": [float(gap[0]), float(gap[1])],
                    "internal_gap": True,
                    "wall_frame": int(wall_frame),
                    "depletion_pressure": False,
                    "model_known": True,
                }
            )
            worlds.append(
                {
                    "eventId": scenario["event_id"],
                    "group": scenario["template_group"],
                    "mirror": bool(mirror),
                    "source": "generated",
                    "scenario": scenario,
                    "world": tv.ValidationWorld(**scenario),
                }
            )
        index += 1
    return worlds


def screen_gap_center(world) -> float:
    import v373_telegraph as tg

    gap = tg.telegraph_gap(world)
    return 0.5 * (gap[0] + gap[1])


def decision_frame(world) -> int:
    return int(max(2, world.wall_frame - DECISION_LEAD))


def state_at(world, frame: int):
    from touhou_rl import tactical_teacher_validation as tv
    from touhou_rl.tactical_scenarios import WorldState

    state = WorldState(frame=0, x=float(world.start_x), bullets=(), hit=False, emissions=())
    while state.frame < frame:
        state = tv.advance(world, state, 0)
        if state.hit:
            break
    return state


def render(world, state, arm: str) -> np.ndarray:
    import v373_telegraph as tg
    from touhou_rl.tactical_synthetic_pixels import render_state

    if arm == "telegraph":
        return tg.render_state_telegraph(world, state)
    return render_state(world, state)


def downscale(image: np.ndarray) -> np.ndarray:
    from PIL import Image

    playfield = image[
        PLAYFIELD_TOP : PLAYFIELD_TOP + PLAYFIELD_HEIGHT,
        PLAYFIELD_LEFT : PLAYFIELD_LEFT + PLAYFIELD_WIDTH,
        :,
    ]
    pil = Image.fromarray(playfield).resize((DOWNSCALE[1], DOWNSCALE[0]), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Probe model
# ---------------------------------------------------------------------------
def build_model(kind: str, device):
    import torch
    from torch import nn

    if kind == "cnn":
        net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            # Keep a coarse spatial layout: global average pooling would make the
            # model blind to *where* the gap is relative to the player, which is
            # the whole question (round27's head did exactly that).
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1),
        )
    elif kind == "linear":
        net = nn.Sequential(nn.Flatten(), nn.Linear(3 * DOWNSCALE[0] * DOWNSCALE[1], 1))
    else:
        raise ValueError(kind)
    return net.to(device)


def train_probe(images, labels, kind: str, seed: int, steps: int = 300, device=None):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(kind, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device)
    # BCEWithLogits expects {0, 1}; the world labels are {-1, +1}.  Passing the
    # signed labels makes the loss negative and the probe untrainable (this bug
    # made round27/27b/27c report "not learned" while training nothing).
    y = torch.tensor([1.0 if label > 0 else 0.0 for label in labels], dtype=torch.float32)
    y = y.to(device).view(-1, 1)
    losses = []
    model.train()
    for _ in range(steps):
        pick = rng.integers(0, len(x), size=min(32, len(x)))
        logits = model(x[pick])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[pick])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if losses and (min(losses) < 0.0 or not np.isfinite(min(losses))):
        raise RuntimeError(f"probe loss left the BCE domain: min={min(losses)}")
    return model, losses


def evaluate_probe(model, images, labels, device) -> dict:
    import torch

    if not images:
        return {"n": 0, "accuracy": None, "predictions": []}
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        logits = model(x).view(-1).cpu().numpy()
    predictions = [1 if value > 0 else -1 for value in logits]
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return {
        "n": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels),
        "predictions": [int(p) for p in predictions],
    }


def player_x_only_baseline(worlds) -> dict:
    """Logistic regression on the player's own x: balanced data must keep this at chance."""
    xs = np.asarray([item["playerX"] for item in worlds], dtype=np.float64)
    ys = np.asarray([1.0 if item["label"] > 0 else 0.0 for item in worlds], dtype=np.float64)
    mean, std = xs.mean(), xs.std() or 1.0
    z = (xs - mean) / std
    w = 0.0
    b = 0.0
    for _ in range(2000):
        logits = z * w + b
        prob = 1.0 / (1.0 + np.exp(-logits))
        error = prob - ys
        w -= 0.1 * float((error * z).mean())
        b -= 0.1 * float(error.mean())
    predictions = np.where(z * w + b > 0, 1, -1)
    targets = np.where(ys > 0, 1, -1)
    return {
        "accuracy": float((predictions == targets).mean()),
        "w": w,
        "b": b,
        "n": int(len(ys)),
        "balanced": abs(float(ys.mean()) - 0.5) <= 0.1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("v373-pixel-info")
    parser.add_argument(
        "--out",
        default="Document/Update/Alpha/v3.7.3/evidence/pixel-info-round27",
    )
    parser.add_argument("--groups", type=int, default=60)
    parser.add_argument("--seed", type=int, default=27001)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args(argv)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(part) for part in args.seeds.split(",")]
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_worlds(args.groups, args.seed)
    valid = load_valid_worlds()
    result: dict = {
        "schema": "v373-pixel-info/1",
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "seeds": seeds,
        "steps": args.steps,
        "counts": {"generated": len(generated), "valid": len(valid)},
        "sources": {
            "validDir": str(VALID_DIR.relative_to(ROOT)),
            "validFilesSha256": {
                path.name: sha256_file(path) for path in sorted(VALID_DIR.glob("wf18-F1-*.json"))
            },
        },
        "arms": {},
    }

    # -- labels, states, renders -------------------------------------------
    dataset = {"telegraph": {"generated": [], "valid": []}, "original": {"generated": [], "valid": []}}
    reader_rows = []
    occluded_readable = 0
    occluded_total = 0
    for bucket, worlds in (("generated", generated), ("valid", valid)):
        for item in worlds:
            world = item["world"]
            frame = decision_frame(world)
            state = state_at(world, frame)
            gap_center = screen_gap_center(world)
            label = 1 if gap_center > state.x else -1
            item["label"] = int(label)
            item["decisionFrame"] = frame
            item["gapCenter"] = gap_center
            item["playerX"] = float(state.x)
            item["offset"] = gap_center - state.x
            for arm in ("telegraph", "original"):
                image = render(world, state, arm)
                dataset[arm][bucket].append(
                    {"image": downscale(image), "label": int(label), "eventId": item["eventId"]}
                )
                reading = read_direction(image)
                reading.update(
                    {
                        "arm": arm,
                        "bucket": bucket,
                        "eventId": item["eventId"],
                        "mirror": item["mirror"],
                        "label": int(label),
                        "frame": frame,
                        "gapCenterTruth": gap_center,
                        "playerXTruth": float(state.x),
                        "correct": bool(reading.get("ok")) and reading.get("direction") == label,
                    }
                )
                reader_rows.append(reading)
                if arm == "telegraph":
                    # P2: with the preview colours replaced by background the
                    # reader must find nothing, proving it reads the band.
                    occluded = image.copy()
                    painted = np.all(occluded == FILL_COLOUR, axis=-1) | np.all(
                        occluded == EDGE_COLOUR, axis=-1
                    )
                    occluded[painted] = BACKGROUND_COLOUR
                    occluded_total += 1
                    if read_direction(occluded).get("ok"):
                        occluded_readable += 1
            # Q6: outside the telegraph window both renderers must be identical.
            quiet_frame = 2
            if world.wall_frame - TELEGRAPH_LEAD > quiet_frame:
                quiet_state = state_at(world, quiet_frame)
                same = np.array_equal(
                    render(world, quiet_state, "telegraph"), render(world, quiet_state, "original")
                )
                item.setdefault("quietFramesIdentical", same)

    reader_telegraph = [row for row in reader_rows if row["arm"] == "telegraph"]
    reader_original = [row for row in reader_rows if row["arm"] == "original"]
    readable_t = sum(1 for row in reader_telegraph if row["ok"])
    readable_o = sum(1 for row in reader_original if row["ok"])
    correct_t = sum(1 for row in reader_telegraph if row["correct"])
    correct_o = sum(1 for row in reader_original if row["correct"])
    result["arms"]["directReader"] = {
        "telegraph": {
            "total": len(reader_telegraph),
            "readable": readable_t,
            "correct": correct_t,
            "accuracy": correct_t / len(reader_telegraph),
            "validCorrect": sum(
                1 for row in reader_telegraph if row["bucket"] == "valid" and row["correct"]
            ),
            "validTotal": sum(1 for row in reader_telegraph if row["bucket"] == "valid"),
            "generatedCorrect": sum(
                1 for row in reader_telegraph if row["bucket"] == "generated" and row["correct"]
            ),
            "generatedTotal": sum(1 for row in reader_telegraph if row["bucket"] == "generated"),
        },
        "original": {
            "total": len(reader_original),
            "readable": readable_o,
            "correct": correct_o,
            "accuracy": correct_o / len(reader_original) if reader_original else None,
        },
        "rows": reader_rows,
        "occlusionControl": {
            "readable": occluded_readable,
            "total": occluded_total,
            "passed": occluded_total > 0 and occluded_readable == 0,
        },
    }

    # -- CNN probe ----------------------------------------------------------
    probe = {"seeds": {}}
    for arm in ("telegraph", "original"):
        base = [item for item in dataset[arm]["generated"] if _base(item["eventId"])]
        mirror = [item for item in dataset[arm]["generated"] if not _base(item["eventId"])]
        valid_items = dataset[arm]["valid"]
        per_seed = []
        for seed in seeds:
            model, losses = train_probe(
                [item["image"] for item in base],
                [item["label"] for item in base],
                "cnn",
                seed,
                args.steps,
                device,
            )
            mirror_eval = evaluate_probe(
                model, [item["image"] for item in mirror], [item["label"] for item in mirror], device
            )
            valid_eval = evaluate_probe(
                model,
                [item["image"] for item in valid_items],
                [item["label"] for item in valid_items],
                device,
            )
            per_seed.append(
                {
                    "seed": seed,
                    "trainN": len(base),
                    "finalLoss": round(float(np.mean(losses[-20:])), 5),
                    "trainAccuracy": evaluate_probe(
                        model,
                        [item["image"] for item in base],
                        [item["label"] for item in base],
                        device,
                    )["accuracy"],
                    "mirrorAccuracy": mirror_eval["accuracy"],
                    "mirrorCorrect": mirror_eval["correct"],
                    "mirrorN": mirror_eval["n"],
                    "validAccuracy": valid_eval["accuracy"],
                    "validCorrect": valid_eval["correct"],
                    "validN": valid_eval["n"],
                }
            )
        probe["seeds"][arm] = per_seed

    # Exploratory (not a gate): can the same probe fit when both orientations are
    # in the training set at all?  A ~chance result here means the probe cannot
    # represent the relation, so its failure says nothing about the information.
    mixed_items = dataset["telegraph"]["generated"]
    group_ids = sorted({item["eventId"][:-3] for item in mixed_items})
    split_rng = np.random.default_rng(seeds[0])
    order = split_rng.permutation(len(group_ids))
    holdout_groups = {group_ids[i] for i in order[: max(1, len(group_ids) // 5)]}
    mixed_train = [item for item in mixed_items if item["eventId"][:-3] not in holdout_groups]
    mixed_test = [item for item in mixed_items if item["eventId"][:-3] in holdout_groups]
    mixed_runs = []
    for seed in seeds:
        model, _ = train_probe(
            [item["image"] for item in mixed_train],
            [item["label"] for item in mixed_train],
            "cnn",
            seed,
            args.steps,
            device,
        )
        mixed_runs.append(
            {
                "seed": seed,
                "trainN": len(mixed_train),
                "testN": len(mixed_test),
                "trainAccuracy": evaluate_probe(
                    model,
                    [item["image"] for item in mixed_train],
                    [item["label"] for item in mixed_train],
                    device,
                )["accuracy"],
                "heldOutGroupAccuracy": evaluate_probe(
                    model,
                    [item["image"] for item in mixed_test],
                    [item["label"] for item in mixed_test],
                    device,
                )["accuracy"],
            }
        )
    probe["exploratoryMixedOrientation"] = mixed_runs
    # Baselines on the same generated set: linear-on-pixels and player-x-only.
    base_items = [item for item in dataset["telegraph"]["generated"] if _base(item["eventId"])]
    mirror_items = [item for item in dataset["telegraph"]["generated"] if not _base(item["eventId"])]
    linear_model, _ = train_probe(
        [item["image"] for item in base_items],
        [item["label"] for item in base_items],
        "linear",
        seeds[0],
        args.steps,
        device,
    )
    probe["linearOnPixels"] = evaluate_probe(
        linear_model,
        [item["image"] for item in mirror_items],
        [item["label"] for item in mirror_items],
        device,
    )
    probe["playerXOnly"] = player_x_only_baseline(
        [item for item in generated if item["label"] in (-1, 1)]
    )
    result["arms"]["probe"] = probe

    # -- verdict ------------------------------------------------------------
    def mean(values):
        return float(sum(values) / len(values)) if values else None

    telegraph_reader_valid = result["arms"]["directReader"]["telegraph"]["validCorrect"]
    telegraph_reader_total = result["arms"]["directReader"]["telegraph"]["validTotal"]
    gen_reader = (
        result["arms"]["directReader"]["telegraph"]["generatedCorrect"],
        result["arms"]["directReader"]["telegraph"]["generatedTotal"],
    )
    mirror_acc = [entry["mirrorAccuracy"] for entry in probe["seeds"]["telegraph"]]
    valid_acc = [entry["validAccuracy"] for entry in probe["seeds"]["telegraph"]]
    original_acc = [entry["mirrorAccuracy"] for entry in probe["seeds"]["original"]]
    quiet_checked = [
        item for item in generated + valid if "quietFramesIdentical" in item
    ]
    q6 = bool(quiet_checked) and all(item["quietFramesIdentical"] for item in quiet_checked)
    gates = {
        "Q1_direct_reader": {
            "passed": telegraph_reader_valid == telegraph_reader_total
            and gen_reader[0] == gen_reader[1],
            "detail": {
                "validCorrect": telegraph_reader_valid,
                "validTotal": telegraph_reader_total,
                "generatedCorrect": gen_reader[0],
                "generatedTotal": gen_reader[1],
            },
        },
        "Q2_reader_absent_without_telegraph": {
            "passed": readable_o <= 0.7 * len(reader_original),
            "detail": {"readable": readable_o, "total": len(reader_original)},
        },
        "Q3_cnn_mirror_transfer": {
            "passed": mean(mirror_acc) is not None and mean(mirror_acc) >= 0.9,
            "detail": {"perSeed": mirror_acc, "mean": mean(mirror_acc)},
        },
        "Q4_cnn_valid_holdout": {
            "passed": mean(valid_acc) is not None and mean(valid_acc) >= 10.0 / 12.0,
            "detail": {"perSeed": valid_acc, "mean": mean(valid_acc)},
        },
        "Q5_control_and_shortcut": {
            "passed": mean(original_acc) is not None
            and mean(original_acc) <= 0.7
            and (probe["playerXOnly"]["accuracy"] or 1.0) <= 0.6,
            "detail": {
                "originalArmPerSeed": original_acc,
                "originalMean": mean(original_acc),
                "playerXOnly": probe["playerXOnly"]["accuracy"],
            },
        },
        "Q6_renderer_quiet_outside_window": {
            "passed": q6,
            "detail": {
                "checked": len(quiet_checked),
                "identical": sum(
                    1 for item in quiet_checked if item["quietFramesIdentical"]
                ),
            },
        },
    }
    if gates["Q1_direct_reader"]["passed"] and gates["Q3_cnn_mirror_transfer"]["passed"]:
        conclusion = "information present in pixels and learnable"
    elif gates["Q1_direct_reader"]["passed"]:
        conclusion = "information present but not learned at this scale"
    else:
        conclusion = "rendering does not encode the direction"
    result["verdict"] = {
        "gates": gates,
        "conclusion": conclusion,
        "allPassed": all(gate["passed"] for gate in gates.values()),
    }
    result["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    (out_dir / "worlds.json").write_text(
        json.dumps(
            [
                {
                    "eventId": item["eventId"],
                    "group": item["group"],
                    "mirror": item["mirror"],
                    "source": item["source"],
                    "label": int(item["label"]),
                    "gapCenter": item["gapCenter"],
                    "playerX": item["playerX"],
                    "offset": item["offset"],
                    "decisionFrame": item["decisionFrame"],
                    "scenario": item["scenario"],
                }
                for item in generated + valid
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / "reader.jsonl").open("w", encoding="utf-8") as handle:
        for row in reader_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["verdict"], indent=2))
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


def _base(event_id: str) -> bool:
    return event_id.endswith("m0")


if __name__ == "__main__":
    raise SystemExit(main())
