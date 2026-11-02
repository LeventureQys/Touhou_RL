"""Controlled goal/deadline diagnostic; no claim of learned pixel behaviour."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from touhou_rl.escape.forecast import DangerField, ForecastConfig
from touhou_rl.escape.viability import ViabilityConfig, search_viability


def main():
    field = DangerField(frame_base=0, positions=np.zeros((0,2)), velocities=np.zeros((0,2)),
                        halves=np.zeros((0,2)), spawn=np.zeros(0), despawn=np.zeros(0),
                        config=ForecastConfig())
    logits = np.zeros(18)
    logits[0] = 10  # old policy wants to idle
    result = search_viability(field, (192.,400.), (0,0), logits,
               config=ViabilityConfig(horizon_frames=18, target_radius_px=0,
                                      target_deadline_frames=6), target=(212.,400.))
    assert not result.witness_mask[0]
    assert result.route.arrival_frame <= 6
    report = {"schema":"v372-goal-diagnostic/1", "synthetic":True,
              "scenario":"currently safe, declared exit20px-right must be reached by frame6",
              "baseline_candidate":0, "teacher_candidate":result.chosen_candidate,
              "witness_mask":list(result.witness_mask), "arrival_frame":result.route.arrival_frame,
              "positions":result.route.positions, "status":result.status,
              "warning":"Goal/deadline are declared for a controlled test, not inferred from game pixels. This is a teacher test, not network behaviour."}
    path = Path("Document/Update/Alpha/v3.7.2/evidence/goal-diagnostic.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
