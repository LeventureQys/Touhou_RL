from touhou_rl.tactical_calibration import summarize_trial


def row(frame,mask=129,x=0,state=0):
    return {'gameplay_frame':frame,'gui_input_current':mask,'input_mask':0,'player_x':x,'player_y':0,
            'player_state':state,'deaths':0}


def trial(records):
    return {'keys':['right','z'],'window':0,'category':'normal',
            'send_after':records[0],'polls':records[1:],'send_frame_interval':[0,0]}


def test_already_active_has_no_exact_latency():
    out=summarize_trial(trial([row(2),row(3,x=3)]))
    assert out['input_status']=='already_active_no_lower_bound'
    assert out['effective_frame_interval']==[None,2]
    assert out['stable_readback_deltas']==[{'dx':3,'dy':0,'frame':3}]


def test_transition_excludes_wrong_input_and_spawning():
    out=summarize_trial(trial([row(0,mask=0),row(1,state=3),row(2,x=3),row(3,x=6)]))
    assert out['effective_frame_interval']==[0,1]
    assert len(out['stable_readback_deltas'])==1
    assert out['rejected_transitions']['transition_or_wrong_input']==1
    assert out['rejected_transitions']['non_alive_player_state']==1


def test_frame_jump_not_measured_as_single_frame_speed():
    out=summarize_trial(trial([row(0),row(2,x=6)]))
    assert not out['stable_readback_deltas']
    assert out['rejected_transitions']=={'non_unit_frame':1}
