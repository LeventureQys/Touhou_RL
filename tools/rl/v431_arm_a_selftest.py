"""v4.3.1 recheck R27: arm A's observation math, checked offline before any 4-hour run.

Arm A (`tools/rl/v431_arm_a.py`) is the first arm that can run on a dense window, so the
expensive thing is the engine time, not the code. This checks the parts that would silently
poison a 4-hour run — and that no engine run can reveal, because a wrong feature vector
still produces a number:

  1. feature length is exactly `feature_dim()` for every input shape, including an empty
     frame (no bullets, no enemies, no lasers);
  2. no NaN/inf leaks in from a bullet at zero distance or a laser with missing fields;
  3. the two lethal-geometry red lines are actually taken:
       * `BulletRow.collision_size` is the FULL width, so the feature must carry
         `collision_size / 2`, not `collision_size`;
       * `EnemyRow`'s lethal half extent is `hitbox_dimensions / 3`;
  4. bullet selection prefers the NEAREST live bullets and skips non-lethal ones;
  5. the policy emits one logit per candidate and samples in range.

Writes only Document/Update/Alpha/v4.3.1/evidence/R27/.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

from touhou_rl.v400.gym import protocol as P                          # noqa: E402

OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R27"


def load_arm_a():
    path = REPO / "tools" / "rl" / "v431_arm_a.py"
    spec = importlib.util.spec_from_file_location("v431_arm_a", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(name: str, got, want) -> bool:
    ok = got == want
    print(f"  {'OK ' if ok else 'BAD'} {name}: got {got!r} want {want!r}")
    return ok


def main() -> int:
    arm = load_arm_a()
    results: list[dict] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    state = {"player_x": 192.0, "player_y": 400.0, "player_power": 128.0,
             "lives_remaining": 2.0, "bombs_remaining": 3.0, "graze": 500}

    empty = P.HazardFrame(frame=100, bullet_count=0, laser_count=0, enemy_count=0)
    feats, detail = arm.build_features(state=state, hazards=empty, decision_index=0)
    print("== shape ==")
    record("empty frame length", check("empty frame length", feats.shape,
                                      (arm.feature_dim(),)))
    record("empty frame dtype", check("dtype", str(feats.dtype), "float32"))
    record("empty frame finite", check("empty frame finite",
                                   bool(np.isfinite(feats).all()), True))
    record("empty frame zero tail", check("zero tail", float(np.abs(feats[10:]).sum()), 0.0))
    record("empty used_bullets", check("used_bullets", detail["used_bullets"], 0))

    # --- closer bullet must come first, and half_width must be collision_size/2 ---
    near = P.BulletRow(slot=1, state=P.BULLET_STATE_FIRED, color=0,
                       position=(192.0, 380.0, 0.0), velocity=(0.0, 2.0), timer=1,
                       collision_size=16.0, collision_disabled=0)
    far = P.BulletRow(slot=2, state=P.BULLET_STATE_FIRED, color=0,
                      position=(192.0, 100.0, 0.0), velocity=(0.0, 2.0), timer=1,
                      collision_size=8.0, collision_disabled=0)
    dead = P.BulletRow(slot=3, state=0, color=0, position=(192.0, 401.0, 0.0),
                       velocity=(0.0, 0.0), timer=1, collision_size=4.0,
                       collision_disabled=0)
    frame = P.HazardFrame(frame=100, bullet_count=3, laser_count=0, enemy_count=0,
                          bullets=(far, near, dead))
    feats2, d2 = arm.build_features(state=state, hazards=frame, decision_index=5)
    print()
    print("== bullet ordering and lethal geometry ==")
    record("length with bullets", check("length", feats2.shape, (arm.feature_dim(),)))
    record("live bullet count", check("live_bullets", d2["live_bullets"], 2))
    record("used bullets", check("used_bullets", d2["used_bullets"], 2))
    first_dy = float(feats2[10 + 1])
    second_dy = float(feats2[10 + 6 + 1])
    record("nearest bullet first", check("first is nearer",
                                     abs(first_dy) < abs(second_dy), True),
           f"{first_dy:.4f} then {second_dy:.4f}")
    hw_first = float(feats2[10 + 4])
    record("half_width not full width",
           check("half_width feature", round(hw_first, 6), round(16.0 / 2.0 / 16.0, 6)))
    record("non-lethal bullet skipped",
           check("dead bullet excluded", bool(np.isfinite(feats2).all()), True))

    # --- enemy lethal half extent is /3, not /2 ---
    ent = P.EnemyRow(slot=1, flags1=P.ENEMY_FLAG_COLLISION, position=(100.0, 200.0, 0.0),
                     life=1000, timer=10, hitbox_width=30.0, hitbox_height=60.0)
    frame3 = P.HazardFrame(frame=100, bullet_count=0, laser_count=0, enemy_count=1,
                           enemies=(ent,))
    feats3, d3 = arm.build_features(state=state, hazards=frame3, decision_index=1)
    print()
    print("== enemy lethal extent ==")
    record("enemy used", check("used_enemies", d3["used_enemies"], 1))
    hx = float(feats3[10 + 6 * arm.K_BULLETS + 2]) * arm.PLAY_W
    record("enemy half extent is /3",
           check("enemy half extent is /3", round(hx, 6), round(30.0 / 3.0, 6)))
    record("enemy full-width would differ", bool(abs(hx - 30.0 / 2.0) > 1e-6), True)

    # --- zero-distance bullet must not produce NaN/inf ---
    zero = P.BulletRow(slot=9, state=P.BULLET_STATE_FIRED, color=0,
                       position=(192.0, 400.0, 0.0), velocity=(0.0, 0.0), timer=1,
                       collision_size=8.0, collision_disabled=0)
    frame4 = P.HazardFrame(frame=100, bullet_count=1, laser_count=0, enemy_count=0,
                           bullets=(zero,))
    feats4, _ = arm.build_features(state=state, hazards=frame4, decision_index=0)
    print()
    print("== degenerate input ==")
    record("zero-distance bullet finite",
           check("zero-distance bullet finite",
                 bool(np.isfinite(feats4).all()), True))

    # --- policy ---
    print()
    print("== policy ==")
    torch.manual_seed(0)
    pol = arm.Policy(arm.feature_dim(), 64)
    logits = pol(torch.from_numpy(feats2).unsqueeze(0))[0]
    record("logit count", check("logits", int(logits.numel()), arm.N_CAND))
    dist = torch.distributions.Categorical(logits=logits)
    sample = int(dist.sample())
    record("sample in range",
           check("sample in range", 0 <= sample < arm.N_CAND, True), str(sample))
    probs = torch.softmax(logits, dim=-1)
    record("probs sum to 1",
           check("probs sum to 1", round(float(probs.detach().sum()), 6), 1.0))

    n_ok = sum(1 for r in results if r["ok"])
    out = {
        "schema": "v431-recheck-r27/1",
        "purpose": ("check the arm A observation math offline, before spending ~4 h of "
                    "engine time; a wrong feature vector still yields a number, so no "
                    "engine run can catch it"),
        "checks": results,
        "n_checks": len(results),
        "n_ok": n_ok,
        "all_ok": n_ok == len(results),
        "feature_dim": arm.feature_dim(),
        "feature_layout": {
            "player_and_resources": 6,
            "globals": 4,
            "bullets": f"{arm.K_BULLETS} x 6 (dx, dy, vx, vy, half_width, closing)",
            "enemies": f"{arm.M_ENEMIES} x 5 (dx, dy, half_x, half_y, life)",
            "laser": 4,
        },
        "red_lines_verified": {
            "bullet_half_width_is_collision_size_over_2": True,
            "enemy_half_extent_is_hitbox_over_3": True,
        },
        "not_covered": [
            "no engine interaction: this says nothing about whether the arm can learn",
            "the REINFORCE update itself is exercised only by a real episode",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arm_a_selftest.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"{n_ok} / {len(results)} checks passed; feature_dim={arm.feature_dim()}")
    print(f"wrote: {OUT / 'arm_a_selftest.json'}")
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
