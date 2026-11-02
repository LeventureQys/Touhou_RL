"""RAM buffered bounded pixel/state synchronization diagnostic, no training data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from touhou_rl.env import Th08Env,EnvConfig
from touhou_rl import winapi
from tools.rl.v373_calibrate import state_record


def run(out, protocol='development'):
    if protocol not in ('development','confirmation'):
        raise ValueError('unknown capture protocol')
    out=Path(out)
    if out.exists(): raise FileExistsError('new diagnostic directory required')
    out.mkdir(parents=True)
    env=Th08Env(EnvConfig(run_dir=out/'session',practice_stage=0,difficulty=3,
        frameskip=2,use_postmessage_input=False))
    frames=[]; trace=[]
    report={'schema':'v373-continuous-capture/1','training_data':False,
        'alignment':'unknown','keys_released':False,'gdi_flush_used':False,
        'note':'user-authorized diagnostic actions; no image freshness claim'}
    try:
        env.reset(); session=env.session
        if not winapi.force_foreground(session.hwnd): raise RuntimeError('foreground unavailable')
        session.keyboard.set_keys({'z'})
        session.wait_for(lambda s:s.player_state==0,timeout=8,interval=.001,description='alive')
        initial=session.reader.read_state(with_bullets=False,with_title=False)
        baseline_deaths=initial.deaths
        # Non-periodic turns and pauses, small excursion, no death-state continuation.
        schedule=[(0,{'z'}),(7,{'right','z'}),(20,{'z'}),(25,{'left','up','z'}),
                  (36,{'down','z'}),(45,{'right','z'}),(54,{'z'}),(61,{'left','z'}),
                  (72,{'right','down','z'}),(81,{'up','z'}),(92,{'left','z'}),
                  (101,{'z'}),(107,{'right','z'}),(115,{'z'})]
        if protocol=='confirmation':
            schedule=[(0,{'z'}),(5,{'left','z'}),(16,{'z'}),(22,{'right','up','z'}),
                      (34,{'down','z'}),(42,{'left','z'}),(53,{'z'}),(58,{'right','z'}),
                      (69,{'left','down','z'}),(80,{'up','z'}),(91,{'right','z'}),
                      (103,{'z'}),(109,{'left','z'}),(117,{'z'})]
        report['protocol']=protocol
        report['schedule']=[{'start':f,'keys':sorted(k)} for f,k in schedule]
        current_keys=None
        begin=initial.gameplay_frame
        deadline=time.perf_counter()+6
        while True:
            if time.perf_counter()>deadline: raise TimeoutError('bounded capture wall deadline')
            if winapi.user32.GetForegroundWindow()!=session.hwnd or not winapi.window_visible(session.hwnd):
                raise RuntimeError('window not foreground visible')
            pre=session.reader.read_state(with_bullets=False,with_title=False)
            delta=pre.gameplay_frame-begin
            if delta<0: raise RuntimeError('clock rollback')
            if delta>=122: break
            if pre.player_state!=0 or pre.deaths!=baseline_deaths: raise RuntimeError('nonalive/death')
            keys=next(keys for start,keys in reversed(schedule) if delta>=start)
            if keys!=current_keys:
                session.keyboard.set_keys(keys);current_keys=keys
            t0=time.perf_counter_ns()
            image=np.ascontiguousarray(session.grab())
            t1=time.perf_counter_ns()
            post=session.reader.read_state(with_bullets=False,with_title=False)
            if post.player_state!=0 or post.deaths!=baseline_deaths: raise RuntimeError('nonalive/death after capture')
            index=len(frames)
            frames.append(image)
            trace.append({'index':index,'pre':state_record(pre),'post':state_record(post),
                'capture_interval':[pre.gameplay_frame,post.gameplay_frame],
                'capture_start_ns':t0,'capture_end_ns':t1,'keys':sorted(keys),
                'phase_verified':False})
            # 120Hz candidate observer; preserve duplicate game frames explicitly.
            time.sleep(.004)
        report['status']='completed'
    except Exception as e:
        report['status']='aborted'; report['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        try:
            if env.session and env.session.keyboard:
                env.session.keyboard.release_all();report['keys_released']=True
            env.close()
        finally:
            if frames:
                np.savez_compressed(out/'frames.npz',rgb=np.stack(frames))
            (out/'trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
            report['captures']=len(frames)
            report['distinct_frames']=len({t['pre']['gameplay_frame'] for t in trace})
            report['same_frame_brackets']=sum(t['pre']['gameplay_frame']==t['post']['gameplay_frame'] for t in trace)
            report['shape']=list(frames[0].shape) if frames else None
            (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirm-foreground',action='store_true',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--protocol',choices=('development','confirmation'),default='development')
    a=p.parse_args();run(a.out,a.protocol)


if __name__=='__main__':main()
