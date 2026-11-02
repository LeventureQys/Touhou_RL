import pytest
from touhou_rl.tactical_motion_envelope import MotionEnvelope,old_action_carryover


def test_boundaries_and_diagonal_speed():
    e=MotionEnvelope()
    assert e.advance((192,384),(-1,0),55)==(8,384)
    assert e.advance((40,384),(0,-1),105)==(40,16)
    assert e.advance((40,44),(0,1),112)==(40,432)
    x,y=e.advance((100,100),(1,1),1)
    assert ((x-100)**2+(y-100)**2)**.5==pytest.approx(4)


def test_axis_switch_keeps_old_action_for_lag():
    e=MotionEnvelope()
    assert old_action_carryover(e,(36,384),(1,0),(0,-1),105,(1,1))==[(40,16)]
    assert old_action_carryover(e,(376,384),(1,0),(-1,0),8,(1,1))==[(348,384)]


def test_lag_unknown_set_not_slower_averaged_motion():
    e=MotionEnvelope()
    assert old_action_carryover(e,(100,100),(1,0),(0,-1),2,(0,2))==[(100,92),(104,96),(108,100)]
    with pytest.raises(ValueError):
        old_action_carryover(e,(100,100),(1,0),(0,-1),2,(2,1))
