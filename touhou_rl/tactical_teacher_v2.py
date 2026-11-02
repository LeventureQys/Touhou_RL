"""Controlled geometry teacher R9; synthetic audit only, no official labels.

Policy uses causal ValidationWorld.advance only. Reference solve/verify is imported
only inside the audit runner, never used to select a policy action.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
from .tactical_scenarios import initial_state
from .tactical_teacher import Decision, TacticalTeacher
from .tactical_teacher_validation import advance


class TacticalTeacherV2:
    """Receding six-step (12-frame) bounded causal safety search, beam width 20."""
    lookahead_steps = 6
    beam_width = 20

    def target(self, s, state):
        x = s.canonical_x(state.x)
        if s.family == 'F1':
            lo, hi = s.coverage
            regions = [(8., min(376., lo-4.)), (max(8., hi+4.), 376.)]
            if s.internal_gap and s.gap[1]-s.gap[0] >= 4:
                regions.append((max(8., s.gap[0]+2.), min(376., s.gap[1]-2.)))
            regions = [r for r in regions if r[0] <= r[1]]
            if not regions: return None
            region = min(regions, key=lambda r: (max(r[0]-x, x-r[1], 0), r[0]))
            return region
        if s.family == 'F3':
            if state.frame+2 < s.window[0]:
                return (s.threshold+8., s.threshold+16.)
            goal = min(s.goal-4., s.threshold-8.)
            return (max(8., goal-8.), max(8., goal))
        return None

    def preferred(self, s, state, region):
        sign = -1 if s.mirror else 1
        x = s.canonical_x(state.x)
        if region is not None:
            if x < region[0]: return sign*(2 if region[0]-x >= 8 else 1)
            if x > region[1]: return -sign*(2 if x-region[1] >= 8 else 1)
            return 0
        last = state.emissions[-1] if state.emissions else None
        age = state.frame-last['frame'] if last else -1
        return sign if 0 <= age < 4 and x < 368 else 0

    def decide(self, s, state):
        if not s.model_known:
            return Decision(0, 'LOCAL_RECOVER', 'UNKNOWN', confidence='unknown')
        region = self.target(s, state)
        desired = self.preferred(s, state, region)
        # Evaluate each first action independently. Every candidate includes both
        # actual logical-frame collision checks and action-dependent bullet births.
        # Keep a short safe beam; no complete-solution/infeasibility interpretation.
        scores = []
        for first in (-2, -1, 0, 1, 2):
            nxt = advance(s, state, first)
            beam = [(nxt, 0.)] if not nxt.hit else []
            safe_depth = 1 if beam else 0
            best_cost = float('inf')
            for depth in range(1, self.lookahead_steps):
                children = []
                for st, cost in beam:
                    reg = self.target(s, st)
                    pref = self.preferred(s, st, reg)
                    for a in (-2, -1, 0, 1, 2):
                        ns = advance(s, st, a)
                        if ns.hit: continue
                        cx = s.canonical_x(ns.x)
                        dist = max(reg[0]-cx, cx-reg[1], 0) if reg else 0
                        c = cost + dist + .2*abs(a-pref)
                        children.append((ns, c))
                if not children: break
                children.sort(key=lambda p: p[1])
                # Quantized x dedup avoids spending the beam on duplicate paths.
                seen = set(); beam = []
                for item in children:
                    key = (item[0].x, tuple((e['frame'], e['aim_x']) for e in item[0].emissions[-2:]))
                    if key in seen: continue
                    seen.add(key); beam.append(item)
                    if len(beam) == self.beam_width: break
                safe_depth = depth+1
                best_cost = beam[0][1]
            cx = s.canonical_x(nxt.x)
            dist = max(region[0]-cx, cx-region[1], 0) if region else 0
            scores.append((nxt.hit, -safe_depth, dist, abs(first-desired), best_cost, abs(first), first))
        selected = min(scores)
        action = selected[-1]
        collision = selected[0]
        reason = 'BOUNDED_NO_SAFE_CANDIDATE_UNKNOWN' if collision else (
            'LOCAL_COLLISION_OVERRIDE' if action != desired else 'GEOMETRY_PROGRESS' if region else 'CAUSAL_EMISSION_RULE')
        return Decision(action, 'LOCAL_RECOVER' if action != desired else 'TRANSIT' if action else 'HOLD_STREAM',
                        reason, region, 'unknown' if collision else 'bounded_synthetic_rule')


def rollout(s, policy=None):
    policy = policy or TacticalTeacherV2()
    state = initial_state(s); actions = []; decisions = []
    while state.frame < s.horizon:
        d = policy.decide(s, state)
        decisions.append(dict(frame=state.frame, **asdict(d)))
        actions.append(d.action); state = advance(s, state, d.action)
    return actions, decisions


def hashes():
    from . import tactical_teacher, tactical_teacher_validation, tactical_scenarios
    files = [Path(__file__), Path(tactical_teacher.__file__), Path(tactical_teacher_validation.__file__), Path(tactical_scenarios.__file__)]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def confirmation_definitions():
    groups = {
        'F1': [dict(start_x=210., wall_frame=46, coverage=(76.,296.)),
               dict(start_x=170., wall_frame=56, coverage=(108.,278.)),
               dict(start_x=222., wall_frame=50, coverage=(94.,302.), internal_gap=True, gap=(214.,230.)),
               dict(start_x=194., wall_frame=42, coverage=(86.,292.), internal_gap=True, gap=(226.,242.))],
        'F2': [dict(start_x=172., fire_interval=15, bullet_speed=7.4, origin=(150.,85.)),
               dict(start_x=212., fire_interval=21, bullet_speed=9.4, origin=(234.,115.)),
               dict(start_x=182., fire_interval=23, bullet_speed=8.2, burst=True),
               dict(start_x=202., fire_interval=25, bullet_speed=7.8, origin=(166.,95.), burst=True)],
        'F3': [dict(start_x=350., fire_interval=15, window=(40,96), threshold=178., goal=154.),
               dict(start_x=322., fire_interval=21, window=(28,92), threshold=194., goal=166.),
               dict(start_x=346., fire_interval=23, window=(44,110), threshold=174., goal=150.),
               dict(start_x=326., fire_interval=25, window=(24,100), threshold=198., goal=170., origin=(218.,105.))]}
    rows = []
    for family, params in groups.items():
        for i, p in enumerate(params):
            for mirror in (False, True):
                kw = dict(p); x = kw.pop('start_x'); group = f'confirmation-{family}-{i:02d}'
                rows.append(dict(event_id=f'{group}-m{int(mirror)}', template_group=group,
                    seed=9000+i, family=family, split='confirmation', mirror=mirror,
                    start_x=384.-x if mirror else x, depletion_pressure=family=='F3', **kw))
    return rows


def audit(out, mode):
    from .tactical_teacher_validation import definitions, make_world, metrics, solve, verify
    import html
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if mode == 'freeze':
        path = out/'confirmation-manifest.json'
        if path.exists(): raise FileExistsError('frozen manifest exists')
        value = dict(schema='R9-confirmation-v1', source_sha256=hashes(), groups=confirmation_definitions(),
                     test_execution='R8 test forbidden', policy_frozen=True)
        path.write_text(json.dumps(value, indent=2), encoding='utf-8'); return value
    if mode == 'confirmation':
        manifest = json.loads((out/'confirmation-manifest.json').read_text(encoding='utf-8'))
        if manifest['source_sha256'] != hashes(): raise ValueError('frozen source changed')
        specs = manifest['groups']; destination = out/'confirmation'
    else:
        specs = [s for s in definitions() if s['split'] in ('development', 'validation')]
        destination = out/'development'
    if destination.exists(): raise FileExistsError('preserve prior audit output')
    destination.mkdir(); before = hashes(); rows = []
    for spec in specs:
        s = make_world(spec)
        actions, decisions = rollout(s)
        old_actions, old_decisions = rollout(s, TacticalTeacher())
        ref = solve(s)
        if ref['status'] == 'feasible': assert verify(s, ref['actions']) == 'feasible'
        row = dict(scenario=asdict(s), v2=metrics(s, actions), old_teacher=metrics(s, old_actions),
                   reference_search=ref, reference=metrics(s, ref['actions']) if ref['actions'] else None,
                   decisions=decisions, old_decisions=old_decisions, official_label=False)
        (destination/f'{s.event_id}.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620">',
                 '<rect width="1000" height="620" fill="white"/>',
                 f'<text x="25" y="25">{html.escape(s.event_id)} — synthetic R9; x=time, y=canonical x; NOT pixels/B0</text>',
                 f'<text x="25" y="48">coverage={s.coverage}; gap={s.gap}; wall={s.wall_frame}; window={s.window}; threshold={s.threshold}; goal={s.goal}</text>']
        for i, (arm, color) in enumerate([('v2', '#16803c'), ('old_teacher', '#c33'), ('reference', '#36c')]):
            m = row[arm]
            if m is None: continue
            pts = ' '.join(f'{40+t["frame"]*5},{450-s.canonical_x(t["x"])}' for t in m['trace'])
            lines.extend([f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>',
                          f'<text x="25" y="{490+i*22}" fill="{color}">{arm}: {m["verdict"]}; collision={m["first_collision_frame"]}</text>'])
        lines.append('</svg>'); (destination/f'{s.event_id}.svg').write_text('\n'.join(lines), encoding='utf-8')
        rows.append(row)
    summary = {}
    for family in ('F1', 'F2', 'F3'):
        rr = [r for r in rows if r['scenario']['family'] == family]
        summary[family] = dict(instances=len(rr), reference_feasible=sum(r['reference_search']['status']=='feasible' for r in rr),
            **{arm: sum(r[arm] is not None and r[arm]['verdict']=='feasible' for r in rr) for arm in ('v2', 'old_teacher', 'reference')})
    failures = [dict(event_id=r['scenario']['event_id'], verdict=r['v2']['verdict'], collision=r['v2']['first_collision_frame'], reference=r['reference_search']['status']) for r in rows if r['v2']['verdict']!='feasible']
    r8manifest = out.parent/'teacher-validation-round8'/'manifest.json'
    audit_record = dict(source_sha256_before=before, source_sha256_after=hashes(), executed_events=len(rows), executed_R8_test=0,
        r8_manifest_sha256=hashlib.sha256(r8manifest.read_bytes()).hexdigest(), official_labels_created=0, training=False, game_started=False, formal_gate='NOT_OPENED')
    assert before == hashes()
    for name, value in [('summary', summary), ('failures', failures), ('execution-audit', audit_record)]:
        (destination/f'{name}.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('development', 'freeze', 'confirmation'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.command), indent=2))


if __name__ == '__main__': main()
