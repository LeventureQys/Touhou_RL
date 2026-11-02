"""Round8 synthetic holdout audit. Never creates official labels or executes test."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from math import hypot
import argparse
import hashlib
import json
from .tactical_scenarios import Scenario, WorldState, Bullet, initial_state
from .tactical_teacher import TacticalTeacher


@dataclass(frozen=True)
class ValidationWorld(Scenario):
    coverage: tuple = (100., 284.)
    gap: tuple = (180., 204.)
    window: tuple = (34, 104)
    threshold: float = 180.
    goal: float = 160.
    origin: tuple = (192., 100.)
    burst: bool = False

    def firing(self, frame):
        if self.family not in ('F2', 'F3'):
            return False
        if self.family == 'F3' and self.window[0]-1 <= frame < self.window[1]:
            return False
        return frame < self.horizon-30 and (frame % self.fire_interval == 0 or
               self.burst and frame % self.fire_interval == 4)


def definitions():
    """Declarative group definitions; test is serialized, never constructed."""
    groups = {
      'development': {
       'F1': [dict(start_x=184., wall_frame=64), dict(start_x=200., wall_frame=68),
              dict(start_x=192., wall_frame=72, internal_gap=True)],
       'F2': [dict(start_x=184., fire_interval=16), dict(start_x=200., fire_interval=18),
              dict(start_x=192., fire_interval=20, bullet_speed=9.)],
       'F3': [dict(start_x=336., fire_interval=16), dict(start_x=328., fire_interval=18),
              dict(start_x=344., fire_interval=20, bullet_speed=9.)]},
      'validation': {
       'F1': [dict(start_x=216., wall_frame=40, coverage=(72.,300.)),
              dict(start_x=176., wall_frame=52, coverage=(112.,272.)),
              dict(start_x=216., wall_frame=48, internal_gap=True, gap=(208.,224.)),
              dict(start_x=192., wall_frame=44, internal_gap=True, gap=(220.,236.))],
       'F2': [dict(start_x=168., fire_interval=14, bullet_speed=7., origin=(144.,80.)),
              dict(start_x=216., fire_interval=22, bullet_speed=10., origin=(240.,120.)),
              dict(start_x=176., fire_interval=24, bullet_speed=8.5, burst=True),
              dict(start_x=208., fire_interval=26, bullet_speed=7.5, origin=(160.,90.), burst=True)],
       'F3': [dict(start_x=352., fire_interval=14, window=(42,94), threshold=176., goal=152.),
              dict(start_x=320., fire_interval=22, window=(26,90), threshold=196., goal=168.),
              dict(start_x=348., fire_interval=24, window=(46,112), threshold=172., goal=148.),
              dict(start_x=324., fire_interval=26, window=(22,98), threshold=200., goal=172., origin=(224.,110.))]},
      'test': {
       'F1': [dict(start_x=x, wall_frame=t, coverage=(lo,hi), internal_gap=g, gap=(202.,214.))
              for x,t,lo,hi,g in [(228.,38,80.,310.,False),(164.,58,106.,260.,False),
                                  (208.,46,90.,294.,True),(188.,62,116.,278.,True)]],
       'F2': [dict(start_x=x, fire_interval=f, bullet_speed=v, origin=(o,105.), burst=b)
              for x,f,v,o,b in [(160.,28,7.2,136.,False),(224.,30,9.2,248.,False),
                                (180.,32,8.2,176.,True),(204.,34,8.8,208.,True)]],
       'F3': [dict(start_x=x, fire_interval=f, window=w, threshold=t, goal=g)
              for x,f,w,t,g in [(356.,28,(48,100),174.,150.),(316.,30,(24,88),198.,170.),
                                (340.,32,(38,108),182.,158.),(332.,34,(30,96),190.,164.)]]}}
    rows=[]
    for split, families in groups.items():
        for family, params in families.items():
            for k, p in enumerate(params):
                group=f'{split}-{family}-structure-{k:02d}'
                for mirror in (False, True):
                    kw=dict(p)
                    x=kw.pop('start_x')
                    rows.append(dict(event_id=f'{group}-m{int(mirror)}', family=family,
                         template_group=group, seed=8000+k, split=split, mirror=mirror,
                         start_x=384-x if mirror else x,
                         depletion_pressure=family=='F3', **kw))
    return rows


def make_world(spec):
    if spec['split']=='test':
        raise ValueError('sealed test: manifest only; execution forbidden')
    return ValidationWorld(**spec)


def advance(s, state, action):
    if action not in (-2,-1,0,1,2):
        raise ValueError('two-frame action outside frozen action set')
    x, bullets, events, hit=state.x,list(state.bullets),list(state.emissions),state.hit
    for frame in range(state.frame, min(state.frame+2,s.horizon)):
        if s.firing(frame):
            ox,oy=s.origin
            # Mirrored emitter geometry, actual current player target at EVERY birth.
            if s.mirror: ox=384.-ox
            dx,dy=x-ox,400.-oy
            length=hypot(dx,dy)
            b=Bullet(frame,ox,oy,s.bullet_speed*dx/length,s.bullet_speed*dy/length,x)
            bullets.append(b)
            events.append(dict(frame=frame,aim_x=x,vx=b.vx,vy=b.vy))
        x=min(376.,max(8.,x+action*2.))
        cx=s.canonical_x(x)
        collision=any(abs(b.x+b.vx*(frame+1-b.born)-x)<=3. and
                      abs(b.y+b.vy*(frame+1-b.born)-400.)<=3. for b in bullets)
        if s.family=='F1' and s.wall_frame<=frame+1<s.wall_frame+12:
            collision |= s.coverage[0]<=cx<=s.coverage[1] and not (
                s.internal_gap and s.gap[0]<=cx<=s.gap[1])
        if s.family=='F3' and s.depletion_pressure:
            collision |= (frame+1<s.window[0] and cx<=s.threshold or
                          frame+1>=s.window[1] and cx>=s.threshold)
        hit |= collision
    return WorldState(min(state.frame+2,s.horizon),x,tuple(bullets),bool(hit),tuple(events))


def replay(s, actions, early=False):
    state=initial_state(s)
    trace=[dict(frame=0,x=state.x,hit=False)]
    for a in actions:
        if state.frame>=s.horizon: break
        state=advance(s,state,a)
        trace.append(dict(frame=state.frame,x=state.x,hit=state.hit,action=a,actual_hold=2))
        if early and state.hit: break
    return state,trace


def verify(s, actions, early=False):
    state,trace=replay(s,actions,early)
    if state.hit: return 'rejected_witness'
    if state.frame!=s.horizon: return 'unknown'
    if s.family=='F3' and not any(s.canonical_x(t['x'])<=s.goal for t in trace): return 'unknown'
    return 'feasible'


def programs(s):
    """Independent bounded schedule class; no teacher geometry/options consulted."""
    n=s.horizon//2
    yield (0,)*n
    sign=-1 if s.mirror else 1
    for start in (0,8,16,24,32,40,48,64,80,96,112):
        for direction in (-2,-1,1,2):
            for duration in (8,24,48,80,160):
                yield tuple(sign*direction if start<=2*k<start+duration else 0 for k in range(n))
    for direction in (-1,1):
        yield tuple(sign*direction if 2*k % s.fire_interval<4 else 0 for k in range(n))


def solve(s, delay=0, prefix_action=0):
    if delay<0 or delay%2 or delay>s.horizon: raise ValueError('delay must be even in horizon')
    n=s.horizon//2
    prefix=(prefix_action,)*(delay//2)
    # Once prefix collides, all suffixes in this prefix class fail; still UNKNOWN.
    if replay(s,prefix,early=True)[0].hit:
        return dict(status='unknown',reason='prefix_collision_no_witness',checked=0,actions=[])
    for checked,p in enumerate(programs(s),1):
        actions=prefix+p[:n-len(prefix)]
        if verify(s,actions,early=True)=='feasible':
            # All positives explicitly rechecked without early termination.
            assert verify(s,actions)=='feasible'
            return dict(status='feasible',reason='full_two_frame_finite_horizon_witness',
                        checked=checked,actions=list(actions))
    return dict(status='unknown',reason='bounded_program_search_no_witness',checked=checked,actions=[])


def windows(entries):
    result=[]
    for e in entries:
        if e['status']!='feasible': continue
        d=e['delay']
        if result and result[-1][1]+2==d: result[-1][1]=d
        else: result.append([d,d])
    return result


def scan(s, current):
    result={}
    for name,a in [('hold',0),('continue_current',current)]:
        entries=[dict(delay=d,**solve(s,d,a)) for d in range(0,s.horizon+1,2)]
        result[name]=dict(prefix_action=a,entries=entries,witnessed_windows=windows(entries),
             exact_latest_upper_bound=None,unwitnessed_semantics='unknown_not_infeasible')
    return result


def rollout(s):
    state=initial_state(s)
    actions=[]; decisions=[]
    teacher=TacticalTeacher()
    while state.frame<s.horizon:
        d=teacher.decide(s,state)
        decisions.append(dict(frame=state.frame,**asdict(d)))
        actions.append(d.action)
        state=advance(s,state,d.action)
    return actions,decisions


def metrics(s,actions):
    state,trace=replay(s,actions)
    occupancy=0
    for frame in range(1,s.horizon+1):
        if s.family=='F3' and frame>=s.window[1]: continue
        for b in state.bullets:
            age=frame-b.born
            if age<1: continue
            bx=s.canonical_x(b.x+b.vx*age); by=b.y+b.vy*age
            occupancy+=77<=bx<=163 and 357<=by<=423
    aims=[s.canonical_x(e['aim_x']) for e in state.emissions]
    corridor=(40,160) if s.family=='F3' else (80,304)
    return dict(verdict=verify(s,actions),survived=not state.hit and state.frame==s.horizon,
        first_collision_frame=next((t['frame'] for t in trace if t['hit']),None),
        collision_time_resolution_frames=2, # actual physics checks every logical frame
        corridor_occupancy_bullet_frames=occupancy,
        corridor_ray_count=sum(corridor[0]<=x<=corridor[1] for x in aims),
        final_horizontal_space_px=min(state.x-8,376-state.x),
        reached_goal=s.family=='F3' and any(s.canonical_x(t['x'])<=s.goal for t in trace),
        trace=trace,emission_events=list(state.emissions),actions=list(actions))


def onset(s,m):
    tr=m['trace']; sign=1 if s.mirror else -1
    if s.family=='F1' and s.internal_gap and s.gap[0]<=s.canonical_x(s.start_x)<=s.gap[1]:
        return None # no forced exit for a reliable internal gap
    for k in range(len(tr)-3):
        diffs=[sign*(tr[k+j+1]['x']-tr[k+j]['x']) for j in range(3)]
        if min(diffs)>0 and sum(diffs)>=8: return tr[k]['frame']
    return None


def svg(s,row,path):
    import html
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650">',
        '<rect width="1000" height="650" fill="white"/>',
        f'<text x="25" y="24">{s.event_id}: synthetic, NOT pixels/B0; x=time, y=canonical player x</text>']
    for name,color in [('teacher','#16803c'),('broad_oscillation','#c33'),('reference','#36c')]:
        m=row[name]
        if not m: continue
        points=' '.join(f"{40+t['frame']*5},{450-s.canonical_x(t['x'])}" for t in m['trace'])
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="40" y="{490+20*len([z for z in lines if "verdict=" in z])}" fill="{color}">{name} verdict={m["verdict"]}</text>')
    for d in row['decisions'][::8]:
        txt=html.escape(f"{d['option']} target={d['target_region']}")
        lines.append(f'<text x="{40+5*d["frame"]}" y="{56+12*((d["frame"]//16)%3)}" font-size="7">{txt}</text>')
    for i,(name,sc) in enumerate(row.get('delay_scan',{}).items()):
        text=html.escape(f'{name}: witnessed-only windows {sc["witnessed_windows"]}; exact upper UNKNOWN')
        lines.append(f'<text x="25" y="{580+i*22}" font-size="11">{text}</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines),encoding='utf-8')


def source_hash():
    from . import tactical_teacher
    return hashlib.sha256(Path(tactical_teacher.__file__).read_bytes()).hexdigest()


def preregister(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema='teacher-validation-round8-v1',teacher_sha256=source_hash(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_execution='forbidden_manifest_only',groups=definitions(),
        frozen_baseline='R1 canonical +/-2 each 32 logical frames; initial -2',
        exact_deadline_gate='not_claimed')
    path=out/'manifest.json'
    if path.exists(): raise FileExistsError('refuse to replace frozen manifest')
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def evaluate(out):
    out=Path(out)
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if manifest['teacher_sha256']!=source_hash(): raise ValueError('frozen teacher changed')
    if manifest['evaluator_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('evaluator differs from preregistered source')
    rows=[]
    for spec in manifest['groups']:
        if spec['split']=='test': continue
        s=make_world(spec); actions,decisions=rollout(s); sign=-1 if s.mirror else 1
        n=s.horizon//2
        ref=solve(s)
        row=dict(scenario=asdict(s),teacher=metrics(s,actions),decisions=decisions,
            reference_search=ref,reference=metrics(s,ref['actions']) if ref['actions'] else None,
            broad_oscillation=metrics(s,[sign*(-2 if (k//16)%2==0 else 2) for k in range(n)]),
            hold=metrics(s,[0]*n),premature_cross=metrics(s,[-2*sign]*n),
            late_cross=metrics(s,[0 if k*2<112 else -2*sign for k in range(n)]),
            label_source='synthetic_audit_only_not_official_supervision')
        if s.family in ('F1','F3'):
            row['delay_scan']=scan(s,actions[0]); start=onset(s,row['teacher'])
            row['late_start']={}
            for name,sc in row['delay_scan'].items():
                seen=[e['delay'] for e in sc['entries'] if e['status']=='feasible']
                row['late_start'][name]=dict(teacher_onset_frame=start,
                    greatest_witnessed_delay=max(seen) if seen else None,
                    later_than_seen_window=start>max(seen) if start is not None and seen else None,
                    exact_latest_upper_bound=None,precise_lateness_gate='not_passed_not_established')
        (out/f'{s.event_id}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        svg(s,row,out/f'{s.event_id}.svg')
        rows.append(row)
    summary={}
    for split in ('development','validation'):
        summary[split]={}
        for family in ('F1','F2','F3'):
            rr=[r for r in rows if r['scenario']['split']==split and r['scenario']['family']==family]
            d=dict(instances=len(rr),reference_feasible=sum(r['reference_search']['status']=='feasible' for r in rr),
                   reference_unknown=sum(r['reference_search']['status']=='unknown' for r in rr))
            for arm in ('teacher','broad_oscillation','hold','premature_cross','late_cross'):
                d[arm]={key:sum(r[arm][key] for r in rr) for key in
                    ('survived','corridor_occupancy_bullet_frames','corridor_ray_count','final_horizontal_space_px','reached_goal')}
                d[arm]['verified']=sum(r[arm]['verdict']=='feasible' for r in rr)
            d['reference_conditioned_teacher_success']=sum(r['teacher']['verdict']=='feasible' for r in rr if r['reference_search']['status']=='feasible')
            summary[split][family]=d
    failures=[dict(event_id=r['scenario']['event_id'],split=r['scenario']['split'],
        teacher_verdict=r['teacher']['verdict'],first_collision_frame=r['teacher']['first_collision_frame'],
        reference_status=r['reference_search']['status']) for r in rows if r['teacher']['verdict']!='feasible']
    unknown=[dict(event_id=r['scenario']['event_id'],prefix=name,delay=e['delay'],reason=e['reason'])
        for r in rows for name,sc in r.get('delay_scan',{}).items() for e in sc['entries'] if e['status']=='unknown']
    for name,value in [('summary',summary),('failures',failures),('unknown',unknown),
                       ('execution-audit',dict(executed_events=len(rows),executed_test_events=0,
                         sealed_test_entries=sum(s['split']=='test' for s in manifest['groups']),
                         teacher_sha256_before=manifest['teacher_sha256'],teacher_sha256_after=source_hash(),
                         full_p2_gate='NOT_PASSED',official_labels_created=0,training=False,game_started=False))]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('preregister','evaluate'))
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result=preregister(args.output) if args.command=='preregister' else evaluate(args.output)
    print(json.dumps(result if args.command=='evaluate' else {'manifest_entries':len(result['groups'])},indent=2))

if __name__=='__main__': main()
