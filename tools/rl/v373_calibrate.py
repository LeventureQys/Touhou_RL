"""Bounded user-confirmed v3.7.3 foreground calibration; no model deployment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from touhou_rl.env import Th08Env, EnvConfig
from touhou_rl import winapi, spec


def state_record(s):
    return {key: getattr(s,key,None) for key in ('gameplay_frame','player_x','player_y',
        'player_state','deaths','input_mask','gui_input_current','gui_input_previous',
        'focus_mode','player_vx','player_vy','stage','supervisor_state')}


def run(out):
    out=Path(out)
    if out.exists():
        raise FileExistsError('new calibration directory required')
    out.mkdir(parents=True)
    report={'schema':'v373-live-calibration/1','foreground_confirmed':True,
        'windows':10,'shoot':True,'bomb':False,'backend':'SendInput',
        'team':0,'difficulty':3,'practice_stage':0,'trials':[],
        'passed':False,'limitations':['boundary saturation not tested',
           '2-frame deterministic actuation not established',
           'capture bracketing cannot independently prove compositor freshness']}
    env=Th08Env(EnvConfig(run_dir=out/'session',practice_stage=0,difficulty=3,
        frameskip=2,windowed=True,use_postmessage_input=False))
    begun=time.perf_counter()
    try:
        env.reset()
        session=env.session
        if not winapi.force_foreground(session.hwnd):
            raise RuntimeError('cannot acquire game foreground')
        reader=session.reader
        report['exe_sha256']=hashlib.sha256((out/'session/th08.exe').read_bytes()).hexdigest()
        report['initial_state']=state_record(reader.read_state(with_bullets=False,with_title=False))
        categories=[('normal',{'right','z'}),('normal',{'left','z'}),
                    ('focus',{'right','shift','z'}),('focus',{'left','shift','z'}),
                    ('diagonal',{'up','right','z'}),('diagonal',{'down','left','z'}),
                    ('focus_diagonal',{'up','right','shift','z'}),
                    ('focus_diagonal',{'down','left','shift','z'})]
        def read():
            return reader.read_state(with_bullets=False,with_title=False)
        def wait_frame(start,delta):
            deadline=time.perf_counter()+2
            last=read()
            polls=[]
            while last.gameplay_frame < start+delta:
                if time.perf_counter()>deadline:
                    raise TimeoutError('game frame stalled')
                if last.gameplay_frame < start:
                    raise RuntimeError('game clock rollback')
                polls.append(state_record(last))
                time.sleep(.0005)
                last=read()
            polls.append(state_record(last))
            return last,polls
        session.keyboard.set_keys({'z'})
        session.wait_for(lambda s: s.player_state == 0, timeout=8., interval=.002,
                         description='stable alive player after spawn')
        for window in range(10):
            for index,(category,keys) in enumerate(categories):
                if winapi.user32.GetForegroundWindow()!=session.hwnd:
                    raise RuntimeError('foreground lost: abort before further inputs')
                session.keyboard.set_keys({'z'})
                neutral_start=read()
                neutral_end,neutral_polls=wait_frame(neutral_start.gameplay_frame,4)
                before=read()
                send_begin=time.perf_counter()
                session.keyboard.set_keys(keys)
                sent=read()
                after,polls=wait_frame(sent.gameplay_frame,4)
                session.keyboard.set_keys({'z'})
                released=read()
                # Capture interval and full raw pool observation are separately bracketed.
                cap_before=read()
                image=session.grab()
                cap_after=read()
                pool_before=read()
                pool=reader.read_bullets()
                pool_after=read()
                sid=f'w{window:02d}-t{index:02d}'
                Image.fromarray(image).save(out/f'{sid}.png')
                active=pool.active_mask()
                np.savez_compressed(out/f'{sid}-pool.npz',
                    active_mask=active,position=pool.position,velocity=pool.velocity,
                    half_size=pool.half_size,state=pool.state,disabled=pool.disabled)
                aligned=(cap_before.gameplay_frame==cap_after.gameplay_frame==pool_before.gameplay_frame==pool_after.gameplay_frame)
                trial={'window':window,'category':category,'keys':sorted(keys),
                    'neutral_before':state_record(neutral_start),'neutral_after':state_record(neutral_end),
                    'neutral_polls':neutral_polls,
                    'before':state_record(before),'send_after':state_record(sent),
                    'hold_end':state_record(after),'release_after':state_record(released),
                    'capture_interval':[cap_before.gameplay_frame,cap_after.gameplay_frame],
                    'pool_interval':[pool_before.gameplay_frame,pool_after.gameplay_frame],
                    'alignment_status':'same_bracket_frame_freshness_unverified' if aligned else 'alignment_unknown',
                    'active_bullets':int(active.sum()),'pool_slots':len(active),
                    'capture':f'{sid}.png','pool':f'{sid}-pool.npz',
                    'dx':after.player_x-before.player_x,'dy':after.player_y-before.player_y,
                    'frame_delta':after.gameplay_frame-before.gameplay_frame,
                    'send_frame_interval':[before.gameplay_frame,sent.gameplay_frame],
                    'release_frame_interval':[after.gameplay_frame,released.gameplay_frame],
                    'polls':polls,'wall_ms':(time.perf_counter()-send_begin)*1000,
                    'death_during_trial':after.deaths!=before.deaths,
                    'invalid_player_state':before.player_state!=0 or after.player_state!=0}
                report['trials'].append(trial)
                print(json.dumps({k:trial[k] for k in ('window','category','dx','dy','frame_delta','alignment_status','death_during_trial')}),flush=True)
        report['completed']=True
        report['final_state']=state_record(read())
        report['summary']={}
        for category in ('normal','focus','diagonal','focus_diagonal'):
            rows=[t for t in report['trials'] if t['category']==category]
            usable=[t for t in rows if not t['death_during_trial'] and not t['invalid_player_state'] and t['frame_delta']>0]
            speeds=[float(np.hypot(t['dx'],t['dy'])/t['frame_delta']) for t in usable]
            report['summary'][category]={'trials':len(rows),'no_death_trials':len(usable),
                'speed_px_per_logical_frame_median':float(np.median(speeds)) if speeds else None,
                'speed_min':min(speeds) if speeds else None,'speed_max':max(speeds) if speeds else None}
        report['same_bracket_frame_count']=sum(t['alignment_status'].startswith('same_') for t in report['trials'])
        report['status']='exploratory_calibration_completed_not_full_P0_gate'
    except Exception as exc:
        report['completed']=False
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['keys_released_on_exit']=False
        try:
            if env.session is not None and env.session.keyboard is not None:
                env.session.keyboard.release_all()
                report['keys_released_on_exit']=True
            env.close()
        finally:
            report['seconds']=time.perf_counter()-begun
            (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirm-foreground',action='store_true',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    run(args.out)


if __name__=='__main__':
    main()
