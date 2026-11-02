"""Bounded live motion probe with consumed-input evidence and fail-fast reset.

No training labels or deployment. User must authorize foreground calibration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from touhou_rl import winapi, spec
from touhou_rl.env import Th08Env,EnvConfig
from tools.rl.v373_calibrate import state_record


def bracket(rows,mask):
    prev=None
    for r in rows:
        if r['gui_input_current']&255==mask:
            return [None if prev is None else prev['gameplay_frame'],r['gameplay_frame']]
        prev=r
    return None


def run(out, protocol='actions'):
    if protocol not in ('actions','boundaries'):
        raise ValueError('unknown motion protocol')
    out=Path(out)
    if out.exists(): raise FileExistsError('new probe directory required')
    out.mkdir(parents=True)
    report={'schema':'v373-motion-probe/1','trials':[],'boundary':[], 'P0_passed':False,
        'status':'running','resource':{'shoot':True,'bomb':False},'keys_released':False}
    env=Th08Env(EnvConfig(run_dir=out/'session',practice_stage=0,difficulty=3,
        frameskip=2,use_postmessage_input=False))
    try:
        env.reset(); s=env.session
        if not winapi.force_foreground(s.hwnd): raise RuntimeError('foreground unavailable')
        reader=s.reader
        s.keyboard.set_keys({'z'})
        s.wait_for(lambda st: st.player_state==0,timeout=8,interval=.001,description='alive')
        initial=reader.read_state(with_bullets=False,with_title=False)
        deaths=initial.deaths
        def read():
            if winapi.user32.GetForegroundWindow()!=s.hwnd: raise RuntimeError('foreground lost')
            st=reader.read_state(with_bullets=False,with_title=False)
            if st.player_state!=0 or st.deaths!=deaths or not st.is_playing():
                raise RuntimeError(f'non-alive/reset/death frame={st.gameplay_frame} state={st.player_state}')
            return state_record(st)
        def collect(frames):
            first=read(); rows=[first]; deadline=time.perf_counter()+3
            while rows[-1]['gameplay_frame']<first['gameplay_frame']+frames:
                if time.perf_counter()>deadline: raise TimeoutError('clock stall')
                cur=read()
                if cur['gameplay_frame']<rows[-1]['gameplay_frame']: raise RuntimeError('rollback')
                if cur!=rows[-1]: rows.append(cur)
                time.sleep(.0005)
            return rows
        # Alternating two-frame commands, then observe complete release transition.
        for i in range(20 if protocol=='actions' else 0):
            keys={'right','z'} if i%2==0 else {'left','z'}
            mask=spec.BTN_SHOOT|(spec.BTN_RIGHT if i%2==0 else spec.BTN_LEFT)
            s.keyboard.set_keys({'z'}); neutral=collect(3)
            before=read(); s.keyboard.set_keys(keys); send_after=read()
            held=collect(2)
            release_before=read(); s.keyboard.set_keys({'z'}); release_after=read()
            released=collect(3)
            row={'id':i,'keys':sorted(keys),'send_interval':[before['gameplay_frame'],send_after['gameplay_frame']],
                 'neutral':neutral,'held':[send_after]+held,'release_send_interval':[release_before['gameplay_frame'],release_after['gameplay_frame']],
                 'released':[release_after]+released,
                 'effective_start':bracket([send_after]+held,mask),
                 'effective_release':bracket([release_after]+released,spec.BTN_SHOOT)}
            report['trials'].append(row)
        # Boundary push and reversal: short enough to remain an exploratory window.
        boundary_schedule=[('right_boundary',{'right','z'},55),('left_reverse',{'left','z'},8)]
        if protocol=='boundaries':
            boundary_schedule=[('left_boundary',{'left','z'},55),('right_reverse',{'right','z'},8),
                ('upper_boundary',{'up','z'},105),('down_reverse',{'down','z'},8),
                ('lower_boundary',{'down','z'},112),('up_reverse',{'up','z'},8)]
        report['protocol']=protocol
        for name,keys,frames in boundary_schedule:
            s.keyboard.set_keys(keys)
            rows=collect(frames)
            report['boundary'].append({'name':name,'rows':rows})
        report['status']='completed_exploratory'
    except Exception as e:
        report['status']='aborted_evidence_preserved'; report['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        try:
            if env.session and env.session.keyboard:
                env.session.keyboard.release_all(); report['keys_released']=True
            env.close()
        finally:
            (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in report.items() if k not in ('trials','boundary')},indent=2),flush=True)
    print('trials',len(report['trials']),'boundary',len(report['boundary']))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--confirm-foreground',action='store_true',required=True)
    p.add_argument('--protocol',choices=('actions','boundaries'),default='actions')
    a=p.parse_args(); run(a.out,a.protocol)


if __name__=='__main__': main()
