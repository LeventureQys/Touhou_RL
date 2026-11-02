import numpy as np
from touhou_rl.bullet_continuity import ObservedBulletTracker

def raw(x=1):return dict(slot_id=[1],position=[[x,2]],velocity=[[1,0]])
def test_contiguous_prediction_and_gap_have_separate_masks():
    t=ObservedBulletTracker();assert t.update(raw(),0,1)['observed_continuity_valid']==[False]
    a=t.update(raw(2),0,2);assert a['observed_continuity_valid']==[True] and not a['engine_generation_known']
    b=t.update(raw(3),0,4);assert b['observed_continuity_valid']==[False]
    assert b['observed_generation'][0]>a['observed_generation'][0]
