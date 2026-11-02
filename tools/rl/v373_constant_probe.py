"""Constant-policy probe: does a single always-on action already solve a world set?

If it does, that set cannot distinguish "playing" from "hugging a wall", and any
aggregate success count computed on it is confounded. Read-only, no training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from touhou_rl.tactical_synthetic_closed_loop import FrozenNetworks, run_event, BASELINE, HEADS
from touhou_rl.tactical_teacher_validation import make_world
from tools.rl.v373_seed_confirm import confirmation_worlds

R11 = Path('Document/Update/Alpha/v3.7.3/evidence/synthetic-closed-loop-round11/preregistration.json')
CONSTANTS = {'hold': 0, 'right': 3, 'left': 7}


def world_sets():
    return {
        'r11_validation': [dict(event_id=e['event_id'], scenario=e['scenario'])
                           for e in json.loads(R11.read_text(encoding='utf-8'))['events']],
        'r16_confirmation': confirmation_worlds(),
    }


def run(out, device='cuda'):
    out = Path(out)
    if out.exists():
        raise FileExistsError('new output directory required')
    torch.set_num_threads(4)
    network = FrozenNetworks(BASELINE, HEADS, device)
    result = {'schema': 'v373-constant-probe/1', 'training_steps': 0, 'formal_gate': False,
              'constants': CONSTANTS, 'sets': {}}
    for set_name, events in world_sets().items():
        rows = []
        for raw in events:
            spec = make_world(raw['scenario'])
            for name, candidate in CONSTANTS.items():
                r = run_event(spec, 'train_mode', 'horizontal_masked', network, candidate)
                rows.append({'set': set_name, 'event_id': r['event_id'], 'family': r['family'],
                             'constant': name, 'feasible': r['feasible'],
                             'survived': r['survived'], 'first_hit': r['first_collision_frame']})
        summary = {}
        for fam in ('F1', 'F2', 'F3', 'all'):
            for name in CONSTANTS:
                group = [r for r in rows if r['constant'] == name and (fam == 'all' or r['family'] == fam)]
                summary.setdefault(fam, {})[name] = {
                    'events': len(group), 'feasible': sum(r['feasible'] for r in group)}
        result['sets'][set_name] = {'summary': summary, 'events': rows}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v['summary'] for k, v in result['sets'].items()}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path,
                   default=Path('Document/Update/Alpha/v3.7.3/evidence/constant-probe-round17'))
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    run(a.out, a.device)
