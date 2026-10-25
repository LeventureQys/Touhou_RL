"""D3 data-quality gates (执行计划-D3补遗 §3), computed before any strategy verdict.

D3-1 slots coverage       100%
D3-2 damageable both states >= 10% of boss decisions
D3-3 phase_starting_life jumps >= 2 per episode
D3-4 attribution rate 30%-80%
D3-5 slot-based spawn rate 0.2-6 per decision
"""

import collections
import json
import pathlib

import numpy as np

from touhou_rl.bullet_track import spawns_by_slot

PROBE = pathlib.Path("Document/Update/Alpha/v3.6.0/data/probe/d3-fields")

episodes = [json.loads(line) for line in (PROBE / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

print(f"{'episode':14s} {'dec':>5} {'bullets':>8} {'slot<0':>7} {'boss_dec':>8} {'dmg_yes':>8} {'dmg_no':>7} {'phase_jumps':>11} {'spawns':>7} {'spawn/dec':>9} {'attr_rate':>9}")
totals = collections.Counter()
rows = []
for ep in episodes:
    name = ep["episode"]
    bullets = np.load(PROBE / "bullets" / f"{name}.npz")
    offsets = bullets["offsets"]
    slots_all = bullets["slots"] if "slots" in bullets else None
    n_bullets = int(offsets[-1])
    slot_missing = int((slots_all < 0).sum()) if slots_all is not None else n_bullets

    boss_decisions = 0
    damageable_yes = 0
    damageable_no = 0
    phase_values = []
    spawns = 0
    attributed = 0
    decisions = 0
    prev_slots = None
    records = [json.loads(line) for line in (PROBE / "trace" / f"{name}.jsonl").open(encoding="utf-8")]
    for i, record in enumerate(records):
        start, end = int(offsets[i]), int(offsets[i + 1])
        cur = slots_all[start:end] if slots_all is not None else None
        if prev_slots is not None and cur is not None:
            new_mask = spawns_by_slot(prev_slots, cur)
            spawns += int(new_mask.sum())
            decisions += 1
            pos = bullets["positions"][start:end]
            spawn_pos = pos[new_mask]
            enemies = np.array(
                [[float(e.get("x", 0.0)), float(e.get("y", 0.0))] for e in (record.get("enemies") or [])],
                dtype=np.float32,
            ).reshape(-1, 2)
            if spawn_pos.shape[0] and enemies.shape[0]:
                d = np.sqrt(
                    (spawn_pos[:, None, 0] - enemies[None, :, 0]) ** 2
                    + (spawn_pos[:, None, 1] - enemies[None, :, 1]) ** 2
                )
                attributed += int((d.min(axis=1) <= 32.0).sum())
        bosses = [e for e in (record.get("enemies") or []) if e.get("boss")]
        if bosses:
            boss_decisions += 1
            if any(e.get("damageable") for e in bosses):
                damageable_yes += 1
            else:
                damageable_no += 1
            for e in bosses:
                phase_values.append(int(e.get("phase_starting_life") or 0))
        prev_slots = cur

    jumps = sum(1 for a, b in zip(phase_values, phase_values[1:]) if a != b)
    rate = attributed / spawns if spawns else None
    rows.append(
        {
            "episode": name,
            "decisions": len(records),
            "bullets": n_bullets,
            "slots_missing": slot_missing,
            "boss_decisions": boss_decisions,
            "damageable_yes": damageable_yes,
            "damageable_no": damageable_no,
            "damageable_yes_frac": damageable_yes / boss_decisions if boss_decisions else None,
            "phase_jumps": jumps,
            "spawns": spawns,
            "spawns_per_decision": spawns / decisions if decisions else None,
            "attributed": attributed,
            "attribution_rate": rate,
        }
    )
    totals["bullets"] += n_bullets
    totals["slots_missing"] += slot_missing
    totals["boss_decisions"] += boss_decisions
    totals["damageable_yes"] += damageable_yes
    totals["damageable_no"] += damageable_no
    totals["spawns"] += spawns
    totals["attributed"] += attributed
    totals["decisions"] += decisions
    print(
        f"{name:14s} {len(records):5d} {n_bullets:8d} {slot_missing:7d} {boss_decisions:8d} "
        f"{damageable_yes:8d} {damageable_no:7d} {jumps:11d} {spawns:7d} "
        f"{(spawns / decisions if decisions else 0):9.2f} {(rate if rate is not None else 0):9.3f}"
    )

print()
gates = {
    "D3-1": {
        "name": "slots 覆盖率",
        "value": 1.0 - totals["slots_missing"] / max(1, totals["bullets"]),
        "threshold": 1.0,
        "pass": totals["slots_missing"] == 0,
    },
    "D3-2": {
        "name": "damageable 两态比例（Boss 在场）",
        "value_yes": totals["damageable_yes"] / max(1, totals["boss_decisions"]),
        "value_no": totals["damageable_no"] / max(1, totals["boss_decisions"]),
        "threshold": 0.10,
        "pass": min(
            totals["damageable_yes"] / max(1, totals["boss_decisions"]),
            totals["damageable_no"] / max(1, totals["boss_decisions"]),
        )
        >= 0.10,
    },
    "D3-3": {
        "name": "phase_starting_life 每局跳变次数",
        "value": min(row["phase_jumps"] for row in rows),
        "threshold": 2,
        "pass": all(row["phase_jumps"] >= 2 for row in rows),
    },
    "D3-4": {
        "name": "归属率（32 px）",
        "value": totals["attributed"] / max(1, totals["spawns"]),
        "threshold": [0.30, 0.80],
        "pass": 0.30 <= totals["attributed"] / max(1, totals["spawns"]) <= 0.80,
    },
    "D3-5": {
        "name": "槽位法新弹率/决策",
        "value": totals["spawns"] / max(1, totals["decisions"]),
        "threshold": [0.2, 6.0],
        "pass": 0.2 <= totals["spawns"] / max(1, totals["decisions"]) <= 6.0,
    },
}
for key, gate in gates.items():
    print(f"{key} {gate['name']}: value={gate['value'] if 'value' in gate else gate.get('value_yes')} pass={gate['pass']}")
print("overall:", all(g["pass"] for g in gates.values()))

out = pathlib.Path("Document/Update/Alpha/v3.6.0/evidence/d3-data-quality.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "schema": "v360-d3-data-quality/1",
            "probe": str(PROBE),
            "gates": gates,
            "overall": {"pass": all(g["pass"] for g in gates.values())},
            "per_episode": rows,
            "totals": dict(totals),
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print("wrote", out)
