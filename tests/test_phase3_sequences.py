import copy
import numpy as np
import pytest

from touhou_rl.phase3_sequences import (
    build_sequence_index, previous_history, build_windows, chunk_targets, evaluate_joint,
)


def rows(n=5):
    return [dict(rollout_id='one', episode=1, decision=i, sample_index=i,
                 frame=100+i*2, stage=1, aligned=True, fresh_pixels=True,
                 raw_safety_snapshot=dict(complete=True, player_state=0, frame=100+i*2, stage=1),
                 deaths_before=0, deaths_after=0, terminated=False, truncated=False,
                 executed_action=[i % 9, 1, i % 2, 0], teacher_action=[8, 1, 1, 0],
                 action_duration_decisions=(1, 3, 5)[i % 3]) for i in range(n)]


def test_shard_allowed_episode_not_allowed():
    r = rows()
    for i, row in enumerate(r):
        row['shard'] = i // 2
    assert build_sequence_index(r, {1:'train'}) == [[0,1,2,3,4]]
    for i in (3,4):
        r[i]['episode'] = 2
        r[i]['decision'] -= 3
    assert build_sequence_index(r, {1:'train', 2:'test'}) == [[0,1,2],[3,4]]
    with pytest.raises(ValueError, match='split'):
        build_sequence_index(r, {1:'train'})


@pytest.mark.parametrize('kind', ['duplicate_sample','duplicate_decision','reorder','rollout','episode'])
def test_reject_identity_and_order(kind):
    r = rows()
    if kind == 'duplicate_sample': r[2]['sample_index'] = 1
    if kind == 'duplicate_decision': r[2]['decision'] = 1
    if kind == 'reorder': r[1], r[2] = r[2], r[1]
    if kind == 'rollout': r[2]['rollout_id'] = 'two'
    if kind == 'episode': r[0]['episode'] = 2
    with pytest.raises(ValueError):
        build_sequence_index(r, {1:'train',2:'test'})


@pytest.mark.parametrize('kind', ['missing','unaligned','stale','raw','player','stamp'])
def test_invalid_centres_omitted_and_break(kind):
    r = rows()
    if kind == 'missing': r[2]['sample_index'] = None
    if kind == 'unaligned': r[2]['aligned'] = False
    if kind == 'stale': r[2]['fresh_pixels'] = False
    if kind == 'raw': r[2]['raw_safety_snapshot']['complete'] = False
    if kind == 'player': r[2]['raw_safety_snapshot']['player_state'] = 1
    if kind == 'stamp': r[2]['raw_safety_snapshot']['frame'] += 1
    assert build_sequence_index(r, {1:'train'}) == [[0,1],[3,4]]


@pytest.mark.parametrize('kind', ['death','late_death','terminated','truncated','stage','repeat','gap','decision_gap'])
def test_boundaries(kind):
    r = rows()
    if kind == 'death':
        r[1]['deaths_after'] = 1
        for row in r[2:]: row.update(deaths_before=1, deaths_after=1)
    if kind == 'late_death':
        for row in r[2:]: row.update(deaths_before=1, deaths_after=1)
    if kind in ('terminated','truncated'): r[1][kind] = True
    if kind == 'stage':
        for row in r[2:]:
            row['stage'] = row['raw_safety_snapshot']['stage'] = 2
    if kind in ('gap','repeat'):
        shift = 50 if kind == 'gap' else -2
        for row in r[2:]:
            row['frame'] += shift
            row['raw_safety_snapshot']['frame'] += shift
    if kind == 'decision_gap':
        for row in r[2:]: row['decision'] += 1
    assert build_sequence_index(r, {1:'train'}) == [[0,1],[2,3,4]]


def test_history_causal_bos_and_teacher_invariance():
    r = rows()
    segs = [[0,1,2],[3,4]]
    expected = {0:(18,3), 1:(0,0), 2:(10,1), 3:(18,3), 4:(12,0)}
    assert previous_history(r, segs) == expected
    for row in r:
        row.update(teacher_action=[0,0,0,0], death_window=True, teacher_clearance=-10000)
    assert previous_history(r, segs) == expected
    r[-1]['executed_action'] = [8,1,1,0]
    assert previous_history(r,segs) == expected  # last action cannot influence past


@pytest.mark.parametrize('duration', [None,2,4,3.0,True])
def test_legacy_delta_never_used(duration):
    r = rows(2)
    r[0]['action_duration_decisions'] = duration
    r[0]['action_duration_game_frames'] = 3
    r[0]['frame_delta'] = 3
    assert previous_history(r, [[0,1]])[1] == (18,3)


def test_history_rejects_episode_leak_and_duplicate():
    r = rows()
    r[2]['episode'] = 2
    r[2]['decision'] = 0
    with pytest.raises(ValueError): previous_history(r[:3], [[1,2]])
    with pytest.raises(ValueError): previous_history(rows(), [[0,1],[1,2]])


def test_windows_fixed_nonoverlap_carry_padding():
    windows = build_windows([list(range(35)), [40,41]], length=16, burn_in=4)
    assert len(windows) == 4
    assert [w['carry'] for w in windows] == [False,True,True,False]
    assert [w['reset_mask'].sum() for w in windows] == [1,0,0,1]
    assert [w['loss_mask'].sum() for w in windows] == [12,16,3,0]
    actual = [w['sample_indices'][w['valid_mask']].tolist() for w in windows]
    assert actual == [list(range(16)),list(range(16,32)),[32,33,34],[40,41]]
    assert np.all(windows[2]['sample_indices'][3:] == -1)
    assert np.array_equal(windows[0]['bos_mask'], windows[0]['reset_mask'])
    # A fake recurrent sum agrees with segment traversal, not random one-frame resets.
    hidden, outputs = 0, []
    for w in windows:
        for sample, valid, reset in zip(w['sample_indices'],w['valid_mask'],w['reset_mask']):
            if not valid: continue
            if reset: hidden = 0
            hidden += sample
            outputs.append(hidden)
    assert outputs[34] == sum(range(35))
    assert outputs[-2:] == [40,81]


def clearance():
    return np.tile([8.,4.,1.], (2,18,1))


def test_target_longest_safe_strict_threshold_none_safe_and_counts():
    c = clearance()
    c[0,0] = [9,7,3]  # class2
    c[0,1] = [2,1,-1]  # none strictly >2; valid risky class0
    c[0,2] = [4,2,0]  # exact threshold is not safe
    target = chunk_targets(c, np.ones_like(c, dtype=bool))
    assert target['best_duration'][0,:4].tolist() == [2,0,0,1]
    assert target['duration_valid'].all()
    assert target['risk_dangerous'][0,1]
    assert target['duration_class_counts'].tolist() == [2,33,1]
    assert target['joint_target'].shape == (2,18,3)
    assert target['joint_class'][0] == 0


def test_masks_and_monotone():
    c = clearance()
    valid = np.ones_like(c, dtype=bool)
    valid[0,0,2] = False
    c[1] = np.nan
    target = chunk_targets(c,valid)
    assert target['best_duration'][0,0] == -1
    assert not target['duration_valid'][0,0]
    assert target['joint_class'][1] == -1
    assert not target['joint_valid'][1]
    c = clearance()
    c[0,0] = [2,3,1]
    with pytest.raises(ValueError, match='monotone'): chunk_targets(c,valid)
    with pytest.raises(ValueError): chunk_targets(clearance(),np.ones((2,18)))
    with pytest.raises(ValueError): chunk_targets(clearance(),np.full((2,18,3),2))


def test_collapse_counts_reported_not_fabricated():
    c = np.tile([4.,1.,-1.],(3,18,1))
    result = chunk_targets(c, np.ones_like(c))
    assert result['duration_class_counts'].tolist() == [54,0,0]


def test_joint_evaluation_horizon_matching_no_shorter_safety_claim():
    c = clearance()
    c[:,0] = [7,3,0]
    result = evaluate_joint(c, np.array([0,0]), np.array([0,1]))
    assert result['chosen_chunk_clearance'].tolist() == [7,3]
    assert result['chosen']['mean_horizon'] == 2
    assert result['fixed_duration3']['mean_horizon'] == 3
    assert result['chosen']['short_duration_rate'] == .5
    assert result['duration_class_counts'].tolist() == [1,1,0]
    assert result['matched_horizon_regret'] == 1
    assert result['matched_horizon_regret_diff'] == 0
    assert result['duration_matching'][1]['count'] == 1
    assert result['duration_matching'][5]['mean_clearance'] is None
    c[1,1,0] = np.nan
    result = evaluate_joint(c, np.array([0,0]), np.array([0,1]))
    assert result['count'] == 1
    assert result['valid_mask'].tolist() == [True,False]
    with pytest.raises(ValueError): evaluate_joint(c,np.array([0.,0.]),np.array([0,1]))
    with pytest.raises(ValueError): evaluate_joint(c,np.array([0,0]),np.array([3,1]))
