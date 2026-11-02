import numpy as np
import pytest
from touhou_rl.tactical_capture_audit import estimate_lag


@pytest.mark.parametrize('lag',[-1,0,1,3])
def test_nonperiodic_lag_sign_and_budget(lag):
    rng=np.random.default_rng(7373)
    xy=np.cumsum(rng.normal(0,4,(180,2)),axis=0)
    memory={i:tuple(x) for i,x in enumerate(xy)}
    pixels=[{'frame':i,'xy':(xy[i-lag]+[5,8]).tolist(),'confident':True,'phase_verified':True}
            for i in range(20,150)]
    result=estimate_lag(memory,pixels)
    assert result['lag_candidates']==[lag]
    assert result['status']==('outside_one_frame' if lag==3 else 'diagnostic_within_one_frame')
    assert not result['real_pixel_gate_passed']


@pytest.mark.parametrize('moving',[False,True])
def test_static_or_constant_speed_not_identifiable(moving):
    memory={i:(i*4 if moving else 0,0) for i in range(180)}
    pixels=[{'frame':i,'xy':[memory[i][0]+10,2],'confident':True,'phase_verified':True}
            for i in range(20,150)]
    assert estimate_lag(memory,pixels)['status']=='unknown'


def test_uncertain_phase_and_missing_frames_do_not_interpolate():
    memory={i:(i,i*i) for i in range(50)}
    pixels=[{'frame':i,'xy':memory[i],'confident':True,'phase_verified':False} for i in range(50)]
    assert estimate_lag(memory,pixels)['reason']=='insufficient_common_lag_coverage'


def test_periodic_trace_keeps_multiple_lag_candidates():
    memory={i:((i%4)*3,0) for i in range(180)}
    pixels=[{'frame':i,'xy':memory[i],'confident':True,'phase_verified':True}
            for i in range(20,150)]
    result=estimate_lag(memory,pixels)
    assert result['status']=='unknown'
    assert len(result['lag_candidates'])>1


def test_duplicate_frame_pixel_phase_cannot_be_collapsed_silently():
    memory={i:(i*i,i) for i in range(180)}
    row={'frame':40,'xy':memory[40],'confident':True,'phase_verified':True}
    assert estimate_lag(memory,[row,row])['reason']=='duplicate_frame_phase_ambiguity'
