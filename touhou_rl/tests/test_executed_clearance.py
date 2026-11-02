import pytest
from touhou_rl.executed_clearance import audit
from touhou_rl import spec

def rows():
    obs=[]
    for frame in range(3):
        obs.append(dict(stage=0,frame=frame,gui_input_current=spec.BTN_SHOOT,read_stable=True,
          player_position=[100.,100.],player_state=0,deaths=0,
          bullets=dict(slot_id=[1],position=[[100.,120.-frame]],velocity=[[0.,-1.]],half=[1.])))
    return [dict(episode=1,decision=0,executed_action=[0,1,0,0],input_timing=dict(
      requested_buttons=spec.BTN_SHOOT,effective_first_observed=dict(stage=0,frame=0),end=dict(stage=0,frame=2),observations=obs))]

def test_executed_linear_clearance_matches_measured_endpoint():
    r=audit(rows());assert r['count']==1 and r['clearance_error_p95']==pytest.approx(0,abs=1e-5)
    assert r['records'][0]['predicted_clearance']==pytest.approx(18-2.325)
    assert r['false_safe_count']==0

def test_newborn_threat_can_invalidate_initial_prediction():
    x=rows();x[0]['input_timing']['observations'][-1]['bullets']=dict(slot_id=[2],position=[[100,100]],velocity=[[0,0]],half=[1])
    r=audit(x);assert r['false_safe_count']==1

def test_missing_full_pool_unknown_not_safe():
    x=rows();x[0]['input_timing']['observations'][1]['bullets']=None
    r=audit(x);assert r['count']==0 and r['exclusions']['pool-missing']==1
