import copy
import json
import numpy as np
import pytest
from touhou_rl.tactical_alignment_intervals import RULES, analyze, freeze, main, write_new


def fixture(static=False):
    trace = []
    for f in range(25):
        s = {'gameplay_frame': f, 'player_x': 100. if static else 100.+3*f,
             'player_y': 200., 'player_state': 0, 'focus_mode': 0}
        trace.append({'index': f, 'pre': s, 'post': dict(s),
                      'keys': ['right'] if f < 12 else ['up']})
    locs = [{'index': f, 'confident': f == 12, 'xy': [trace[f]['pre']['player_x']+32, 216.]} for f in range(25)]
    return trace, {'role': 'confirmation', 'localizations': locs}, {'rules': copy.deepcopy(RULES), 'anchor_offset': [32.,16.]}


def test_dynamic_exact_geometry_and_closed_budget():
    trace, bank, pre = fixture()
    q = bank['localizations'][12]
    q['xy'][0] += 1
    result = analyze(trace, bank, pre)
    r = result['rows'][0]
    assert r['candidate_frames'] == [12,13]  # residuals 1 and exactly 2
    assert r['candidate_relative_frames'] == [0,1]
    assert r['relative_memory_error_interval_xy'] == [[0.,3.],[0.,0.]]
    assert r['relative_memory_error_interval_px'] == [0.,3.]
    assert r['candidate_residual_interval_px'] == [1.,2.]
    assert r['all_candidates_within_one'] is True
    assert r['dynamic'] and r['near_turn']
    assert len(r['comparisons']) == 17
    assert result['summaries']['dynamic_near_turn']['denominator'] == 1
    assert result['summaries']['all_confident']['inflation_diagnostic']['2.0']['count'] == 0
    assert result['summaries']['all_confident']['inflation_diagnostic']['4.0']['count'] == 1
    assert not result['phase_verified'] and not result['real_pixel_gate_passed']
    assert not result['bulletpool_same_frame_proven']


def test_stationary_preserves_all_frames_not_nearest():
    trace, bank, pre = fixture(True)
    r = analyze(trace, bank, pre)['rows'][0]
    assert r['candidate_frames'] == list(range(4,21))
    assert r['relative_frame_interval_hull'] == [-8.,8.]
    assert r['all_candidates_within_one'] is False
    assert r['max_relative_displacement_px'] == 0


@pytest.mark.parametrize('mode', ['missing','conflict','nonfinite'])
def test_any_incomplete_window_unknown_no_interpolation(mode):
    trace, bank, pre = fixture()
    if mode == 'missing':
        trace.pop(4)
        bank['localizations'].pop(4)
    elif mode == 'conflict':
        trace[4]['post']['player_x'] += .001
    else:
        trace[4]['post']['player_x'] = float('nan')
    result = analyze(trace, bank, pre)
    r = result['rows'][0]
    assert r['status'] == 'unknown'
    assert r['candidate_frames'] == [12]  # partial evidence retained, not exhaustive
    assert not r['candidate_set_exhaustive_in_window']
    assert r['all_candidates_within_one'] is None
    assert r['max_relative_displacement_px'] is None
    assert result['summaries']['all_confident']['counts']['unknown'] == 1


def test_unstable_and_edge_unknown():
    trace, bank, pre = fixture()
    trace[12]['post']['gameplay_frame'] = 13
    r = analyze(trace, bank, pre)['rows'][0]
    assert 'unstable_capture_bracket' in r['unknown_reasons']
    trace, bank, pre = fixture()
    bank['localizations'][0]['confident'] = True
    assert analyze(trace, bank, pre)['rows'][0]['status'] == 'unknown'


def test_duplicate_pixel_pairwise_conflict_invalidates_every_q():
    trace, bank, pre = fixture(True)
    for i, delta in [(25,-.6),(26,.6)]:
        r = copy.deepcopy(trace[12]); r['index'] = i; trace.append(r)
        bank['localizations'].append({'index': i,'confident': True,'xy':[132+delta,216.]})
    result = analyze(trace, bank, pre)
    assert result['pixel_conflicting_counters'] == [12]
    assert all(r['status'] == 'unknown' for r in result['rows'])


def test_identical_duplicates_are_separate_capture_diagnostics():
    trace, bank, pre = fixture()
    r = copy.deepcopy(trace[12]); r['index'] = 25; trace.append(r)
    q = copy.deepcopy(bank['localizations'][12]); q['index'] = 25; bank['localizations'].append(q)
    result = analyze(trace, bank, pre)
    assert len(result['rows']) == 2
    assert result['pixel_conflicting_counters'] == []
    assert all(r['candidate_frames'] == [12] for r in result['rows'])


def test_empty_candidates_are_not_vacuously_within_one():
    trace, bank, pre = fixture()
    bank['localizations'][12]['xy'][1] += 10
    r = analyze(trace, bank, pre)['rows'][0]
    assert r['status'] == 'complete_empty'
    assert r['all_candidates_within_one'] is None
    assert r['relative_frame_interval_hull'] is None


def test_repeat_nonadjacent_position_keeps_discrete_set():
    trace, bank, pre = fixture()
    trace[16]['pre']['player_x'] = trace[16]['post']['player_x'] = trace[12]['pre']['player_x']
    r = analyze(trace, bank, pre)['rows'][0]
    assert r['candidate_frames'] == [12,16]
    assert r['relative_frame_interval_hull'] == [0.,4.]
    assert not r['all_candidates_within_one']


def test_freeze_only_development_and_no_refit(tmp_path):
    dev = tmp_path/'dev.json'; out = tmp_path/'prereg.json'
    dev.write_text(json.dumps({'role':'development','counter_lag_diagnostic': {'scores':[
        {'lag':0,'anchor_offset':[32,16]}, {'lag':1,'anchor_offset':[900,900]}]}}))
    frozen = freeze(dev,out)
    assert frozen['anchor_offset'] == [32,16]
    with pytest.raises(FileExistsError): freeze(dev,out)
    dev.write_text(json.dumps({'role':'confirmation'}))
    with pytest.raises(ValueError, match='development'): freeze(dev,tmp_path/'other.json')
    trace, bank, pre = fixture()
    pre['rules']['residual_budget_px'] = 10
    with pytest.raises(ValueError, match='frozen'): analyze(trace,bank,pre)


def test_indices_must_exactly_match_and_output_no_overwrite(tmp_path):
    trace,bank,pre = fixture()
    bank['localizations'].pop()
    with pytest.raises(ValueError,match='indices'): analyze(trace,bank,pre)
    path = tmp_path/'out.json'; write_new(path, {'original':True})
    with pytest.raises(FileExistsError): write_new(path, {})
    with pytest.raises(FileExistsError):
        main(['analyze','--prereg','absent','--bank','absent','--trace','absent','--output',str(path)])
