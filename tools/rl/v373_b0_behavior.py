"""Read-only characterization of the frozen base policy on R16 confirmation worlds.

Answers: what behaviour does B0 actually exhibit, and therefore what the 200-step
fine-tuning is destroying. No training, no labels, no gate changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import torch

from tools.rl.v373_seed_confirm import confirmation_worlds
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.tactical_synthetic_closed_loop import FrozenNetworks, executed_history
from touhou_rl.tactical_policy import TacticalPolicy, TacticalConfig
from touhou_rl.vmodels import load_vision_checkpoint


class ResearchNet:
    """Read-only adapter for R16 schema research heads; no training, no gate changes."""

    def __init__(self, baseline, checkpoint, device):
        self.device = torch.device(device)
        backbone, _ = load_vision_checkpoint(baseline, map_location='cpu')
        backbone = backbone.to(device).eval()
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload.get('schema') not in ('v373-seed-confirm-heads/1', 'v373-synthetic-render-only-heads/1'):
            raise ValueError('unknown research schema')
        if payload['baseline_sha256'] != file_sha(baseline):
            raise ValueError('baseline binding mismatch')
        self.policy = TacticalPolicy(backbone, TacticalConfig(**payload['config'])).to(device).eval()
        self.policy.heads.load_state_dict(payload['heads'], strict=True)

    @torch.inference_mode()
    def __call__(self, arm, regions, previous, hidden, reset):
        regions = {k: v.to(self.device) for k, v in regions.items()}
        core = self.policy.vision_model.forward_core(regions)
        f = core['features'][:, None]
        b = core['policy_logits'][:, None]
        out = self.policy.forward_features(f, b, executed_history(f, previous), hidden,
                                           reset_mask=torch.tensor([[reset]], device=self.device))
        return out['policy_logits'][0, 0].cpu(), out['hidden']
from touhou_rl.tactical_synthetic_closed_loop import FrozenNetworks, run_event, BASELINE, HEADS
from touhou_rl.tactical_data import file_sha

OUT = Path('Document/Update/Alpha/v3.7.3/evidence/b0-behavior-round17')
NAMES = {0: 'hold', 3: 'right', 7: 'left', 9: 'hold_focus', 12: 'rightfocus', 16: 'leftfocus'}


def run(out=OUT, device='cuda', checkpoint=None, arm='B0'):
    out = Path(out)
    if out.exists():
        raise FileExistsError('new output directory required')
    torch.set_num_threads(4)
    if checkpoint:
        network = ResearchNet(BASELINE, Path(checkpoint), device)
    else:
        network = FrozenNetworks(BASELINE, HEADS, device)
    rows = []
    for raw in confirmation_worlds():
        spec = make_world(raw['scenario'])
        result = run_event(spec, arm, 'horizontal_masked', network, 0)
        acts = [r['actual_candidate'] for r in result['steps']]
        first = result['steps'][0]['actual_candidate'] if result['steps'] else None
        rows.append({
            'event_id': result['event_id'], 'family': result['family'],
            'mirror': bool(spec.mirror), 'feasible': result['feasible'],
            'survived': result['survived'], 'goal_satisfied': result['goal_satisfied'],
            'first_hit': result['first_collision_frame'], 'first_action': first,
            'action_counts': {NAMES.get(k, str(k)): v for k, v in sorted(Counter(acts).items())},
            'switches': sum(a != b for a, b in zip(acts, acts[1:])),
            'x_trace': [r['x_after'] for r in result['steps']][:8],
        })
    summary = {}
    for fam in ('F1', 'F2', 'F3', 'all'):
        group = [r for r in rows if fam == 'all' or r['family'] == fam]
        counts = Counter()
        for r in group:
            for name, n in r['action_counts'].items():
                counts[name] += n
        summary[fam] = {
            'events': len(group),
            'feasible': sum(r['feasible'] for r in group),
            'survived': sum(r['survived'] for r in group),
            'action_share': {k: round(v / max(1, sum(counts.values())), 4) for k, v in counts.most_common()},
            'first_actions': dict(Counter(NAMES.get(r['first_action'], str(r['first_action'])) for r in group)),
            'median_switches': sorted(r['switches'] for r in group)[len(group) // 2] if group else None,
        }
    report = {
        'schema': 'v373-b0-behaviour/1',
        'baseline_sha256': file_sha(BASELINE),
        'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': file_sha(checkpoint) if checkpoint else file_sha(HEADS),
        'arm': arm,
        'condition': 'horizontal_masked, causal depth3, actual-history, sticky hit to horizon',
        'note': 'B0 path uses only frozen baseline policy logits. With --checkpoint/--arm N1 the research heads are loaded read-only.',
        'summary': summary,
        'events': rows,
        'training_steps': 0,
        'formal_gate': False,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'sha256.json').write_text(json.dumps({p.name: file_sha(p) for p in out.iterdir() if p.is_file()}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--device', default='cuda')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--arm', default='B0')
    a = p.parse_args()
    run(a.out, a.device, a.checkpoint, a.arm)
