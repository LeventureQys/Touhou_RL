"""Safety counterexamples for the forced action teacher and geometry calibration."""
import numpy as np
from touhou_rl.chunk_safety import predict

def test_safe_prefix_can_become_unsafe_at_long_duration():
    x=dict(schema='v200-raw-safety-snapshot/1',complete=True,frame=0,stage=0,player_state=0,
      player_position=[100,100],bullets=dict(position=[[100,120]],velocity=[[0,-2]],half=[1]))
    r=predict(x)
    assert r['clearance'][0,0]>0 and r['clearance'][0,2]<0
    assert np.all(np.diff(r['clearance'],axis=1)<=1e-5)

def test_uncontrollable_phase_has_no_training_labels():
    x=dict(schema='v200-raw-safety-snapshot/1',complete=True,frame=0,stage=0,player_state=1,
      player_position=[100,100],bullets=dict(position=[],velocity=[],half=[]))
    assert not predict(x)['model_valid'].any()
