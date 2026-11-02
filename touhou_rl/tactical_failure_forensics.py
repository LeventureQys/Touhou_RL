"""Round12 frozen-trace forensic research, never training/labels or network scoring.

A teacher disagreement is descriptive only. Independent finite open-loop schedules
and a separate logical-frame physics replay certify positive witnesses only.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from math import hypot
from pathlib import Path
from .tactical_scenarios import Bullet, WorldState, initial_state
from .tactical_teacher_validation import advance, make_world, programs
from .tactical_teacher_v2 import TacticalTeacherV2

EVIDENCE = Path('Document/Update/Alpha/v3.7.3/evidence')
OLD = EVIDENCE / 'synthetic-closed-loop-round11'
OUT = EVIDENCE / 'failure-forensics-round12'
ACTIONS = (-2, -1, 0, 1, 2)
HORIZONTAL = {0: 0, 3: 2, 7: -2, 9: 0, 12: 1, 16: -1}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def logical_step(s, state, action):
    """Independent one-logical-frame implementation, not advance/teacher/verify.

    Keep the ORIGINAL horizon (including its emitter cutoff); truncating horizon
    to obtain a single frame would silently change future emissions.
    """
    if action not in ACTIONS or state.frame >= s.horizon:
        raise ValueError('invalid logical action/state')
    f = state.frame
    x = state.x
    bullets, emissions = list(state.bullets), list(state.emissions)
    firing = (s.family in ('F2', 'F3') and
              not (s.family == 'F3' and s.window[0]-1 <= f < s.window[1]) and
              f < s.horizon-30 and
              (f % s.fire_interval == 0 or s.burst and f % s.fire_interval == 4))
    if firing:
        ox, oy = s.origin
        if s.mirror:
            ox = 384.-ox
        length = hypot(x-ox, 400.-oy)
        b = Bullet(f, ox, oy, s.bullet_speed*(x-ox)/length,
                   s.bullet_speed*(400.-oy)/length, x)
        bullets.append(b)
        emissions.append(dict(frame=f, aim_x=x, vx=b.vx, vy=b.vy))
    x = max(8., min(376., x+2.*action))
    cx = 384.-x if s.mirror else x
    hit_now = any(abs(b.x+b.vx*(f+1-b.born)-x) <= 3. and
                  abs(b.y+b.vy*(f+1-b.born)-400.) <= 3. for b in bullets)
    if s.family == 'F1' and s.wall_frame <= f+1 < s.wall_frame+12:
        hit_now |= (s.coverage[0] <= cx <= s.coverage[1] and
                    not (s.internal_gap and s.gap[0] <= cx <= s.gap[1]))
    if s.family == 'F3' and s.depletion_pressure:
        hit_now |= ((f+1 < s.window[0] and cx <= s.threshold) or
                    (f+1 >= s.window[1] and cx >= s.threshold))
    return WorldState(f+1, x, tuple(bullets), bool(state.hit or hit_now), tuple(emissions))


def independent_replay(s, actions, state=None):
    state = initial_state(s) if state is None else state
    rows = [dict(frame=state.frame, x=state.x, hit=state.hit)]
    first = state.frame if state.hit else None
    for action in actions:
        for _ in range(min(2, s.horizon-state.frame)):
            state = logical_step(s, state, action)
            rows.append(dict(frame=state.frame, x=state.x, hit=state.hit, action=action))
            if first is None and state.hit:
                first = state.frame
    goal = s.family != 'F3' or any(
        (384.-r['x'] if s.mirror else r['x']) <= s.goal
        for r in rows if r['frame'] % 2 == 0)
    verdict = ('unknown' if not s.model_known else 'rejected_witness' if state.hit else
               'feasible' if state.frame == s.horizon and goal else 'unknown')
    return dict(status=verdict, first_collision_logical=first,
                full_horizon=state.frame == s.horizon, goal_satisfied=goal,
                trace=rows), state


def replay_actual(s, trace):
    """Replay every actual action, including sticky-hit postcollision suffix."""
    if trace['arm'] != 'N1' or trace['condition'] != 'horizontal_masked':
        raise ValueError('only frozen N1 horizontal trace')
    state = initial_state(s)
    states = [state]
    actions = []
    first = None
    expected_positions = [dict(frame=0, x=state.x, hit=False)]
    for row in trace['steps']:
        if (row['frame'] != state.frame or row['x_before'] != state.x or
                row['actual_candidate'] not in HORIZONTAL or
                HORIZONTAL[row['actual_candidate']] != row['actual_action'] or
                row['actual_hold'] != 2 or row['invalid_action'] or row['fallback']):
            raise ValueError('actual trace contract mismatch')
        before = state
        state = advance(s, state, row['actual_action'])
        if state.hit and not before.hit:
            first = state.frame
        if (row['x_after'] != state.x or row['hit'] != state.hit or
                row['first_collision_frame'] != first):
            raise ValueError('actual x/hit/firsthit mismatch')
        actions.append(row['actual_action'])
        states.append(state)
        expected_positions.append(dict(frame=state.frame, x=state.x, hit=state.hit))
    independent, independent_state = independent_replay(s, actions)
    if (trace['trace'] != expected_positions or trace['first_collision_frame'] != first or
            state.frame != s.horizon or independent_state != state or
            [r for r in independent['trace'] if r['frame'] % 2 == 0] !=
            [dict(r, **({'action': actions[i-1]} if i else {}))
             for i, r in enumerate(expected_positions)]):
        raise ValueError('full replay or independent logical replay mismatch')
    return actions, states, dict(verified=True, decisions=len(actions),
        first_collision_frame=first, first_collision_logical=independent['first_collision_logical'],
        final_state_sha256=hashlib.sha256(repr(state).encode()).hexdigest(),
        x_hit_verified_all_decision_frames=True, independent_logical_replay_verified=True)


def reference_candidates(s, prefix, state, budget=223):
    """Fixed prefix + one forced candidate + finite reference-program suffix.

    No teacher import/call in this routine. Every positive replays initial through
    complete prefix/candidate/suffix using independent logical physics.
    """
    if not s.model_known or state.hit:
        raise ValueError('trusted, unhit state required')
    rows = []
    for action in ACTIONS:
        first, after = independent_replay(s, [action], state)
        remaining = (s.horizon-after.frame)//2
        witness = None
        checked = 0
        if not after.hit:
            for program in programs(s):
                if checked >= budget:
                    break
                checked += 1
                route = list(prefix)+[action]+list(program[:remaining])
                result, _ = independent_replay(s, route)
                if result['status'] == 'feasible':
                    witness = dict(actions=route, prefix_actions=list(prefix), candidate=action,
                                   suffix_actions=route[len(prefix)+1:],
                                   independent_verification=result)
                    break
        rows.append(dict(action=action, first_two_logical_frames=first['trace'][1:],
                         first_two_hit=after.hit, programs_checked=checked,
                         status='feasible' if witness else 'unknown', witness=witness,
                         unknown_reason=None if witness else
                         'candidate_collision_no_witness' if after.hit else 'bounded_program_search_no_witness'))
    return rows


def action_stats(actions):
    return dict(decisions=len(actions), stays=sum(a == 0 for a in actions),
                action_switches=sum(a != b for a, b in zip(actions, actions[1:])),
                direction_reversals=sum(a*b < 0 for a, b in zip(actions, actions[1:])))


def analyze_event(s, trace, representatives=2, budget=223, rescue=True):
    actions, states, replay = replay_actual(s, trace)
    out = dict(event_id=s.event_id, family=s.family, replay=replay,
               full_trace_action_stats=action_stats(actions), windows=[], representatives=[])
    collision = trace['first_collision_frame']
    if collision is None:
        return out
    if not s.model_known:
        out['excluded_reason'] = 'unsupported_model'
        return out
    teacher = TacticalTeacherV2()
    first_frame = max(0, collision-16)
    indices = [i for i, st in enumerate(states[:-1])
               if first_frame <= st.frame < collision and not st.hit]
    for i in indices:
        st = states[i]
        d = teacher.decide(s, st)
        out['windows'].append(dict(frame=st.frame, state=asdict(st),
            actual_candidate=trace['steps'][i]['actual_candidate'], actual_action=actions[i],
            teacher=asdict(d), differs=actions[i] != d.action))
    differing = [r['frame'] for r in out['windows'] if r['differs']]
    first_div = differing[0] if differing else None
    changes = [states[i].frame for i in indices if i and actions[i] != actions[i-1]]
    later_change = next((f for f in changes if first_div is not None and f > first_div), None)
    out.update(window_start=first_frame, window_action_stats=action_stats([actions[i] for i in indices]),
        pre_collision_action_stats=action_stats(actions[:collision//2]),
        first_divergence_in_window=first_div, disagreement_decisions=len(differing),
        late_reaction_descriptive=dict(first_subsequent_action_change=later_change,
            lag_frames=None if later_change is None else later_change-first_div,
            meaning='descriptive lag only; not a safety deadline or causal avoidability claim'))
    selected = sorted(set([indices[0], indices[-1]]))[:representatives] if indices else []
    for i in selected:
        st = states[i]
        candidates = reference_candidates(s, actions[:i], st, budget)
        row = dict(frame=st.frame, prefix_actions=actions[:i], actual_action=actions[i],
                   teacher_action=next(r['teacher']['action'] for r in out['windows'] if r['frame'] == st.frame),
                   candidates=candidates)
        actual = next(c for c in candidates if c['action'] == actions[i])
        alternatives = [c['action'] for c in candidates if c['status'] == 'feasible' and c['action'] != actions[i]]
        row['interpretation'] = dict(actual_candidate_status=actual['status'],
            witnessed_alternative_actions=alternatives,
            death_avoidable_by_witnessed_suffix=bool(alternatives or actual['status'] == 'feasible'),
            limitation='existential finite synthetic suffix, not N1 capability; unknown never means infeasible')
        if rescue:
            rs = st
            suffix, decisions = [], []
            while rs.frame < s.horizon:
                d = teacher.decide(s, rs)
                decisions.append(dict(frame=rs.frame, **asdict(d)))
                suffix.append(d.action)
                rs = advance(s, rs, d.action)
            result, _ = independent_replay(s, actions[:i]+suffix)
            row['privileged_teacher_rescue'] = dict(actions=actions[:i]+suffix,
                prefix_actions=actions[:i], suffix_actions=suffix, decisions=decisions,
                independent_verification=result, network_score=False,
                original_trace_modified=False, privileged_model_access=True)
        out['representatives'].append(row)
    return out


def preregister(out=OUT):
    out = Path(out)
    if (out/'preregistration.json').exists():
        raise FileExistsError('preserve preregistration')
    old = json.loads((OLD/'preregistration.json').read_text(encoding='utf-8'))
    files = [OLD/'preregistration.json', OLD/'run/summary.json',
             Path('Document/Update/Alpha/v3.7.3/reports/P5-synthetic-round11.md'),
             Path(__file__), Path('touhou_rl/tests/test_tactical_failure_forensics.py'),
             Path('touhou_rl/tactical_synthetic_closed_loop.py'),
             Path('touhou_rl/tactical_teacher_v2.py'), Path('touhou_rl/tactical_teacher_validation.py'),
             Path('touhou_rl/tactical_scenarios.py')]
    # Preserve every old round11 input identity plus the complete old run artifact
    # inventory (including pixels, controls and figures), not merely selected traces.
    files += [Path(p) for p in old['identities']]
    files += sorted((OLD/'run').rglob('*'))
    files = sorted(set(p for p in files if p.is_file()))
    p = dict(schema='v373-failure-forensics-prereg/1',
        old_input_sha256={str(f): sha(f) for f in files}, events=old['events'],
        arm='N1', condition='horizontal_masked', max_failures=6,
        window_logical_frames=16, representatives_per_failure=2,
        representative_rule='first and last trusted unhit decision in [reported firstcollision-16, firstcollision)',
        candidate_actions=list(ACTIONS), reference_program_budget=223,
        reference='world.programs schedules; full actual prefix + candidate + fixed finite program suffix; independent logical replay',
        unknown='no witness is unknown, never infeasible; candidate first two logical collision frames always retained',
        teacher_comparison='only trusted model and unhit pre-collision states; at most16 logical frames',
        first_divergence='first actual-vs-teacher action difference within window, not global event',
        late_reaction='lag from first window disagreement to next actual action change; descriptive, not causal',
        rescue='teacherV2 from both representative states to horizon, separately privileged, independent full prefix+suffix replay',
        no_training=True, formal_labels_created=0, validation_mixed_into_training=False,
        no_game=True, original_traces_immutable=True, network_scores_unchanged=True)
    save(out/'preregistration.json', p)
    return p


def run(out=OUT):
    out = Path(out)
    p = json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
    for file, digest in p['old_input_sha256'].items():
        if sha(file) != digest:
            raise ValueError('frozen input changed: '+file)
    dest = out/'run'
    dest.mkdir(exist_ok=False)
    rows = []
    failures = 0
    for e in p['events']:
        trace = json.loads((OLD/'run/horizontal_masked'/e['event_id']/'N1/trace.json').read_text(encoding='utf-8'))
        if trace['first_collision_frame'] is not None:
            failures += 1
            if failures > p['max_failures']:
                raise ValueError('failure budget exceeded')
        row = analyze_event(make_world(e['scenario']), trace, p['representatives_per_failure'], p['reference_program_budget'])
        save(dest/(e['event_id']+'.json'), row)
        print(e['event_id'], 'replay verified', len(row['representatives']), flush=True)
        rows.append(row)
    unchanged = all(sha(f) == h for f, h in p['old_input_sha256'].items())
    if not unchanged:
        raise ValueError('old input changed during forensic execution')
    summary = dict(schema='v373-failure-forensics/1', preregistration_sha256=sha(out/'preregistration.json'),
        events=len(rows), replays_verified=sum(r['replay']['verified'] for r in rows),
        failures=failures, representative_states=sum(len(r['representatives']) for r in rows),
        comparisons=sum(len(r['windows']) for r in rows), old_inputs_unchanged=unchanged,
        training_steps=0, official_labels_created=0, game_started=False,
        formal_P5=False, T21_passed=False, development_only=True, network_scores_unchanged=True,
        event_summaries=[dict(event_id=r['event_id'], replay=r['replay'],
            first_divergence_in_window=r.get('first_divergence_in_window'),
            disagreement_decisions=r.get('disagreement_decisions', 0),
            pre_collision_action_stats=r.get('pre_collision_action_stats'),
            window_action_stats=r.get('window_action_stats'),
            late_reaction_descriptive=r.get('late_reaction_descriptive'),
            representatives=[dict(frame=t['frame'], **t['interpretation'],
                rescue_status=t['privileged_teacher_rescue']['independent_verification']['status'])
                for t in r['representatives']]) for r in rows])
    save(dest/'summary.json', summary)
    save(out/'old-input-sha256-after.json', {f: sha(f) for f in p['old_input_sha256']})
    save(dest/'sha256.json', {str(f.relative_to(dest)): sha(f) for f in dest.rglob('*') if f.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preregister', 'run'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(preregister(args.out) if args.command == 'preregister' else run(args.out), indent=2))


if __name__ == '__main__':
    main()
