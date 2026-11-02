import json
from copy import deepcopy
import pytest
from touhou_rl.audit_chunk_calibration import audit_rows, aggregate_reports, SCHEMA
from touhou_rl import spec

def row(dec=0, frame=10):
    return dict(episode=1,decision=dec,stage=0,frame=frame,aligned=True,fresh_pixels=True,
                deaths_before=0,deaths_after=0,executed_action=[3,1,0,0],
                action_duration_game_frames=99, actual_hold_frames=2,
                raw_safety_snapshot=dict(schema='v200-raw-safety-snapshot/1',complete=True,stage=0,frame=frame,
                  player_position=[100.,100.],bullets=dict(position=[],velocity=[],half=[])),
                input_timing=dict(requested_buttons=spec.BTN_RIGHT|spec.BTN_SHOOT,
                  effective_first_observed=dict(stage=0,frame=frame),end=dict(stage=0,frame=frame+2),
                  observations=[dict(stage=0,frame=frame+i,gui_input_current=spec.BTN_RIGHT|spec.BTN_SHOOT,
                      read_stable=True,player_position=[100.+i*4,100.],deaths=0,player_state=0) for i in range(3)]))

def test_actual_observed_motion_not_legacy_delta():
    r=audit_rows([row()]);a=r['actions'][0]
    assert r['schema']==SCHEMA and a['displacement_residual']==pytest.approx(0)
    assert a['actual_observed_hold_frames']==2 and a['model_clearance']==60
    assert r['counterfactual_calibration'] is False
    assert len(r['identity_hash']['value'])==64

def test_readback_gap_mask_and_unstable_fail_closed():
    for mutation in ('gap','mask','unstable','missing'):
        x=row(); t=x['input_timing']
        if mutation=='gap': t['observations'].pop(1)
        if mutation=='mask': t['observations'][1]['gui_input_current']=0
        if mutation=='unstable': t['observations'][1]['read_stable']=False
        if mutation=='missing': x.pop('input_timing')
        a=audit_rows([x])['actions'][0]
        assert not a['usable'] and a['displacement_residual'] is None

def test_death_window_not_immediate_death_and_actual_death_invalid():
    x=row();x['death_window']=True
    assert audit_rows([x])['actions'][0]['usable']
    x['deaths_after']=1
    assert 'death' in audit_rows([x])['actions'][0]['unavailable_reasons']

def test_settling_pre_observations_ignored_not_fabricated():
    x=row();x['input_timing']['observations'].insert(0,dict(stage=0,frame=9,gui_input_current=0,read_stable=True,player_position=[99.,100.],deaths=0))
    assert audit_rows([x])['actions'][0]['displacement_residual']==0

def test_duration_mapping_and_snapshot_timing():
    x=row();x['raw_safety_snapshot']['frame']=9
    a=audit_rows([x])['actions'][0]
    assert a['model_clearance'] is None and a['clearance_unavailable_reason']=='snapshot-to-effective-not-aligned'
    x=row();x['input_timing']['observations'].append(dict(x['input_timing']['observations'][-1],frame=13,player_position=[112.,100.]))
    x['input_timing']['end']['frame']=13
    assert audit_rows([x])['actions'][0]['clearance_unavailable_reason']=='duration-not-modeled'

def test_respawn_phase_is_unknown_not_selected_by_residual():
    x=row();x['input_timing']['observations'][1]['player_state']=1
    a=audit_rows([x])['actions'][0]
    assert not a['usable'] and 'uncontrollable-or-unknown-player-phase' in a['unavailable_reasons']


def test_duplicate_and_reorder_rejected():
    with pytest.raises(ValueError):audit_rows([row(),row()])
    with pytest.raises(ValueError):audit_rows([row(1),row(0)])

def test_cross_episode_is_not_joined_and_aggregate_safe_json():
    a=row();b=row();b['episode']=2
    r=audit_rows([a,b]);assert r['residual_sample_count']==2
    merged=aggregate_reports([r]);assert merged['usable_count']==2
    json.dumps(merged,allow_nan=False)
