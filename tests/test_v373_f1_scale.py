"""Contract tests for round25's F1 data-scale test; CPU only, no training run, no CUDA, no formal run.

What these tests exist for:
  1. the search is round18's generator and round18's per-world criterion at a *declared*
     budget and seed -- one family, one target, no relaxed threshold -- and the declared
     budget actually reaches the target before it is frozen;
  2. the frozen signature union really is R8/R9/R10/R16 + R18 and an injected R18 world
     is rejected, so "the new groups never repeat an old world" is a checked assertion;
  3. the split rule is a pure function: alternate by group, mirrors atomic, the three frozen
     R19 validation groups held out, >= 12 train and >= 5 held-out, the one documented flip,
     a recorded shortfall branch, and no leakage between the train split and either
     evaluation set;
  4. the small arm is exactly the six old groups and the large arm exactly the train split;
  5. round21's renderer, its preview window and its teacher records are the ones used:
     every frozen F1 train event's stored preview equals `telegraph_window`, and the stored
     teacher actions equal the frozen R19 teacher_v2 records;
  6. the direction/mechanism read-out is round22's code reused verbatim: the round24
     recorded base/mirror counts recompute from round24's own cells;
  7. the judgement rule is a pure function and cannot return "scale helps" when the mirror
     count is zero on every seed;
  8. the checkpoint contract is the round20 one, so the closed loop really runs the frozen
     inference path;
  9. JSON is opened in x mode, weights in xb mode, directories with exist_ok=False.
"""
from __future__ import annotations

import json

import pytest

from tools.rl import v373_f1_scale as fs
from tools.rl import v373_spatial_fuse as r22
from tools.rl import v373_telegraph as r21
from tools.rl import v373_valid_curriculum as r20
from tools.rl import v373_world_filter as r18
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_teacher_validation import make_world


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def frozen():
    manifest = fs.load_manifest()
    split = fs.load_split(manifest)
    return dict(manifest=manifest, split=split, old=fs.old_groups(manifest, split),
                r19=fs.r19_frozen_groups(split), assets=fs.r21_assets())


@pytest.fixture(scope='module')
def search(frozen):
    return fs.search_new_groups()


@pytest.fixture(scope='module')
def backbone():
    from touhou_rl.vmodels import load_vision_checkpoint
    model, _ = load_vision_checkpoint(fs.BASELINE, map_location='cpu')
    model.eval()
    return model


def synth_group(index, prefix='wf25'):
    group = f'{prefix}-F1-{index:03d}'
    rows = {}
    for mirror in (False, True):
        event_id = group + f'-m{int(mirror)}'
        rows[event_id] = dict(event_id=event_id, family='F1', template_group=group, seed=25000 + index,
                              start_x=100. + 20 * index if not mirror else 384. - (100. + 20 * index),
                              mirror=mirror, horizon=160, split='development', depletion_pressure=False)
    return dict(group=group, family='F1', source='wf25', signature=['F1', 100. + 20 * index],
                events=list(rows), scenarios=rows, specs=list(rows.values()))


def synthetic_split(count, frozen):
    groups = frozen['old'] + [synth_group(index) for index in range(count)]
    partitioned = fs.partition(groups)
    return groups, partitioned, fs.split_payload(partitioned, groups)


# ---------------------------------------------------------------- 1. frozen search
def test_search_spec_is_frozen_and_f1_only():
    assert fs.SEARCH['family'] == 'F1'
    assert fs.SEARCH['seed'] == 25001 and fs.SEARCH['budget'] == 400 and fs.SEARCH['target_groups'] == 16
    assert fs.SEARCH['group_prefix'] == 'wf25'
    assert fs.FAMILY == 'F1' and fs.SEEDS == (7, 17, 27) and fs.UPDATES == 200
    assert fs.LR == .001 and fs.CLIP == 1. and fs.CONDITION == 'horizontal_masked'
    assert fs.ARMS == ('S', 'L') and fs.CONTROL == 'A'
    assert fs.SPLIT_FLOORS == dict(train_groups=12, heldout_groups=5)
    criterion = fs.criterion()
    assert criterion['search']['domains'] == r18.DOMAINS['F1']
    assert criterion['search']['budget'] == fs.SEARCH['budget']
    assert criterion['search']['target_groups'] == fs.SEARCH['target_groups']
    assert criterion['search']['seed'] == fs.SEARCH['seed']
    assert criterion['disjointness']['frozen'][-2:] == ['r18_worlds',
                                                        'r19_f1_split (the same fourteen R18 groups)']


def test_declared_budget_reaches_the_target(search):
    counts = search['counts']
    assert counts['target_met'], 'the frozen budget must reach 16 groups before the pre-registration'
    assert len(search['accepted']) == fs.SEARCH['target_groups']
    assert counts['drawn'] <= fs.SEARCH['budget']
    assert counts['budget_exhausted'] is False
    for row in search['accepted']:
        assert row['base']['verdict'] == 'valid' and row['mirror']['verdict'] == 'valid'
        assert row['base']['passing_baselines'] == [] and row['mirror']['passing_baselines'] == []
        assert row['base']['reference']['status'] == 'feasible'
        assert row['base']['reference']['witness_replay'] and row['mirror']['reference']['witness_replay']
        assert len(row['base']['reference']['actions']) == fs.DECISIONS
        # round18's six-constant diagnostic is reported, not part of the criterion
        assert isinstance(row['base']['any_constant_feasible'], list)
    assert len({row['group'] for row in search['accepted']}) == len(search['accepted'])
    assert all(row['group'].startswith('wf25-F1-') for row in search['accepted'])


def test_search_groups_are_mirror_atomic():
    geometry = dict(family='F1', coverage=(8., 376.), internal_gap=True, gap=(96., 108.), wall_frame=52,
                    start_x=140.)
    specs = fs.new_group_specs(3, geometry)
    assert [spec['event_id'] for spec in specs] == ['wf25-F1-003-m0', 'wf25-F1-003-m1']
    assert {spec['mirror'] for spec in specs} == {False, True}
    assert {spec['template_group'] for spec in specs} == {'wf25-F1-003'}
    assert r18.signature(specs[0]) == r18.signature(specs[1])
    assert specs[0]['start_x'] + specs[1]['start_x'] == 384.
    assert all(spec['split'] == 'development' for spec in specs)


# ---------------------------------------------------------------- 2. frozen signatures
def test_frozen_union_is_r8_r9_r10_r16_plus_r18(frozen):
    union, census = fs.frozen_signature_union()
    assert census['r8'] == 21 and census['r9'] == 12 and census['r10'] == 8 and census['r16_confirmation'] == 12
    assert census['r10_subset_of_r8'] is True
    assert census['r18_worlds'] == 14 and census['r19_f1_groups'] == 6
    assert census['frozen_union'] == 59 == len(union)


def test_injected_r18_world_is_rejected(frozen):
    world = frozen['manifest']['spec']['worlds'][0]
    injected = dict(group=world['group'], signature=list(world['signature']))
    r18_sig = r18.signature(world['scenario'])
    injected['signature'] = json.loads(json.dumps(r18_sig))
    with pytest.raises(ValueError, match='duplicates a frozen signature'):
        fs.assert_new_disjoint([injected])
    duplicate = dict(injected)
    duplicate['group'] = 'wf25-F1-999'
    with pytest.raises(ValueError, match='duplicates a frozen signature'):
        fs.assert_new_disjoint([duplicate])


# ---------------------------------------------------------------- 3./4. split
def test_partition_meets_floors_and_keeps_mirrors_atomic(frozen):
    groups, partitioned, payload = synthetic_split(16, frozen)
    assert partitioned['counts'] == dict(total_groups=22, train_groups=12, heldout_groups=10, new_groups=16,
                                         new_heldout_groups=7, train_events=24, heldout_events=20,
                                         new_heldout_events=14)
    assert partitioned['floors'] == dict(train_groups=True, heldout_groups=True, forced_heldout=True,
                                         new_heldout_groups=7, new_heldout_floor=True, new_groups_floor=True,
                                         disjoint=True, covers_all=True)
    assert len(partitioned['flips']) == 1 and partitioned['flips'][0]['change'] == 'heldout_to_train'
    assert set(fs.R19_FROZEN_VALIDATION) <= set(partitioned['heldout'])
    assert set(fs.R19_FROZEN_TRAIN) <= set(partitioned['train'])
    assert payload['counts']['train_groups'] == 12 and payload['counts']['heldout_groups'] == 10
    assert len(payload['r19_frozen_events']) == 6 and len(payload['new_heldout_events']) == 14
    for row in payload['groups']:
        assert len(row['events']) == 2
        assert {event['mirror'] for event in row['events']} == {False, True}
    # every group keeps its two events on one side, and the new held-out groups are new
    assert all(row['source'] == 'wf25' for row in payload['groups'] if row['split'] == 'heldout'
               and row['group'] not in fs.R19_FROZEN_VALIDATION)
    assert len({row['group'] for row in payload['groups'] if row['split'] == 'train'}) == 12


@pytest.mark.parametrize('count,train,heldout,new_heldout', [(17, 12, 11, 8), (18, 12, 12, 9),
                                                             (20, 13, 13, 10), (11, 12, 5, 2)])
def test_partition_other_group_counts(frozen, count, train, heldout, new_heldout):
    groups, partitioned, payload = synthetic_split(count, frozen)
    assert partitioned['counts']['train_groups'] == train
    assert partitioned['counts']['heldout_groups'] == heldout
    assert partitioned['counts']['new_heldout_groups'] == new_heldout
    assert set(fs.R19_FROZEN_VALIDATION) <= set(partitioned['heldout'])


def test_partition_shortfall_branches(frozen):
    groups, partitioned, payload = synthetic_split(10, frozen)
    assert partitioned['counts']['train_groups'] == 11 and partitioned['counts']['heldout_groups'] == 5
    assert partitioned['floors']['train_groups'] is False
    single = fs.partition(frozen['old'] + [synth_group(0)])
    assert single['shortfall'] == 'fewer_than_eight_new_groups:train_uses_every_group_except_the_frozen_heldout'
    assert set(single['heldout']) == set(fs.R19_FROZEN_VALIDATION)
    assert single['counts']['new_heldout_groups'] == 0


def test_no_leakage_between_train_and_evaluation(frozen):
    groups, partitioned, payload = synthetic_split(16, frozen)
    table = {row['group']: row for row in payload['groups']}
    small = {event['event_id'] for group in fs.R19_FROZEN_TRAIN + fs.R19_FROZEN_VALIDATION
             for event in table[group]['events']}
    large = set(payload['train_events'])
    evaluation = set(payload['r19_frozen_events']) | set(payload['new_heldout_events'])
    assessments = fs.split_assertions(payload, {'S': small, 'L': large})
    assert assessments['ok'], assessments['errors']
    assert not (large & evaluation)
    assert not (small & set(payload['new_heldout_events']))
    assert small & set(payload['r19_frozen_events'])
    assert assessments['small_arm_trained_on_r19_frozen'] is True
    assert assessments['large_arm_trains_on_r19_frozen'] is False
    assert assessments['train_heldout_disjoint'] is True
    assert assessments['train_groups'] == 12 and assessments['heldout_groups'] == 10
    broken = dict(payload, train_events=list(payload['train_events']) + [payload['r19_frozen_events'][0]])
    assert fs.split_assertions(broken, {'S': small, 'L': set(broken['train_events'])})['ok'] is False


def test_frozen_r19_split_is_what_round19_froze(frozen):
    r19 = frozen['r19']
    assert set(r19) == set(fs.R19_FROZEN_TRAIN + fs.R19_FROZEN_VALIDATION)
    for group_id, row in r19.items():
        assert row['split'] == ('train' if group_id in fs.R19_FROZEN_TRAIN else 'validation')
        assert len(row['events']) == 2 and len(row['signature']) == 13
    assert [row['group'] for row in frozen['old']] == ['wf18-F1-002', 'wf18-F1-005', 'wf18-F1-006',
                                                       'wf18-F1-007', 'wf18-F1-009', 'wf18-F1-010']


# ---------------------------------------------------------------- 5. renderer / teacher
def test_round21_preview_window_matches_stored_freeze(frozen):
    for event_id, entry in frozen['assets'].items():
        if entry['family'] != 'F1':
            continue
        world = make_world(frozen['manifest']['index'][event_id]['scenario'])
        assert world.family == 'F1'
        window = r21.telegraph_window(world)
        stored = entry['render_freeze']['preview']
        assert stored == [max(0, window[0]), window[1]]
        expected = [frame for frame in r21.DECISION_FRAMES if window[0] <= frame < window[1]]
        assert entry['render_freeze']['changed_decision_frames'] == len(expected)
        assert entry['render_freeze']['first_changed_frame'] == expected[0]
        assert entry['render_freeze']['last_changed_frame'] == expected[-1]
        assert r21.TELEGRAPH_LEAD == 45


def test_frozen_teacher_actions_reproduce_round19(frozen):
    f1_assets = {event_id: entry for event_id, entry in frozen['assets'].items() if entry['family'] == 'F1'}
    assert len(f1_assets) == 6
    for event_id, entry in f1_assets.items():
        actions = fs.r19_teacher_actions(event_id)
        assert len(actions) == fs.DECISIONS
        rollout = json.loads((fs.R21_EVIDENCE / 'run' / entry['rollout']['path']).read_text(encoding='utf-8'))
        assert actions == list(rollout['actions'])
        assert file_sha(fs.R21_EVIDENCE / 'run' / entry['rollout']['path']) == entry['rollout']['sha256']
    for group in fs.R19_FROZEN_TRAIN + fs.R19_FROZEN_VALIDATION:
        for mirror in (0, 1):
            event_id = f'{group}-m{mirror}'
            assert len(fs.r19_teacher_actions(event_id)) == fs.DECISIONS


# ---------------------------------------------------------------- 6. direction read-out
def test_round24_direction_recomputes_from_round24_cells():
    summary = json.loads(fs.R24_SUMMARY.read_text(encoding='utf-8'))
    rows = summary['validation']['f1_cells']
    arms = {'A': [f'A-s{seed}' for seed in fs.SEEDS], 'X': [f'X-s{seed}' for seed in fs.SEEDS]}
    extracted = {}
    for arm, networks in arms.items():
        subset = {key: row for key, row in rows.items() if row['network'] in networks}
        extracted[arm] = r22.group_summary(subset, networks)
    for group in ('base', 'mirror'):
        for key, value in fs.R24_REFERENCE['direction'][group].items():
            assert extracted['A'][group][key] == value, (group, key)
    screen = dict(right=extracted['A']['base']['screen_right'] + extracted['A']['mirror']['screen_right'],
                  left=extracted['A']['base']['screen_left'] + extracted['A']['mirror']['screen_left'],
                  hold=extracted['A']['base']['screen_hold'] + extracted['A']['mirror']['screen_hold'])
    assert screen == fs.R24_REFERENCE['direction']['screen']
    assert fs.r24_reference()['direction'] == fs.R24_REFERENCE['direction']


def test_screen_share_counts_every_cell():
    rows = {f'L-s7/e{index}': dict(network='L-s7', screen_direction=direction)
            for index, direction in enumerate(['right', 'right', 'left', 'hold'])}
    share = fs.screen_share(rows, ['L-s7'])
    assert share == dict(cells=4, right=2, left=1, hold=1, right_share=.5)
    assert fs.screen_share({}, ['L-s7']) == dict(cells=0, right=0, left=0, hold=0, right_share=None)


# ---------------------------------------------------------------- 7. judgement
def make_cell(network, mirror, need_screen, moved, index):
    """One synthetic F1 cell with a chosen screen need and a chosen executed screen direction."""
    left_gap, right_gap = (40., 60.), (140., 160.)
    canonical_need = need_screen if not mirror else ('left' if need_screen == 'right' else 'right')
    gap = right_gap if canonical_need == 'right' else left_gap
    toward = (moved == need_screen)
    return dict(network=network, mirror=mirror, gap=list(gap), start_canonical=100.,
                screen_direction=moved, mirror_group='mirror' if mirror else 'base',
                verdict='toward' if toward else ('hold' if moved == 'hold' else 'away'),
                in_gap_at_wall=False, feasible=False)


def scenario(l_flip, s_flip=0, b0_flip=0, l_keep=7, need_left=7, need_right=7, left_base=5):
    """A held-out set with `need_left` cells the screen-right prior gets wrong and `need_right` it gets right."""
    def value(spec, seed):
        return spec[fs.SEEDS.index(seed)] if isinstance(spec, (list, tuple)) else spec
    rows = {}
    for seed in fs.SEEDS:
        for arm, flip_count, keep_count in (('L', value(l_flip, seed), l_keep),
                                            ('S', value(s_flip, seed), need_right),
                                            ('B0', value(b0_flip, seed), need_right)):
            network = 'B0' if arm == 'B0' else f'{arm}-s{seed}'
            cells = []
            for index in range(need_left):
                cells.append(make_cell(network, index >= left_base, 'left',
                                       'left' if index < flip_count else 'right', index))
            for index in range(need_right):
                cells.append(make_cell(network, index < need_right - 2, 'right',
                                       'right' if index < keep_count else 'left', 100 + index))
            for index, cell in enumerate(cells):
                rows[f'{network}/{index}'] = cell
    return dict(new_heldout=rows, r19_frozen={})


def test_screen_need_flips_with_the_mirror():
    base_right = make_cell('L-s7', False, 'right', 'right', 0)
    mirror_left = make_cell('L-s7', True, 'left', 'left', 1)
    assert fs.screen_need(base_right) == 'right'
    assert fs.screen_need(mirror_left) == 'left'
    flipped = dict(base_right, mirror=True)
    assert fs.screen_need(flipped) == 'left'


def test_conditioned_flip_and_keep():
    rows = scenario(0, need_left=7, need_right=7)
    counts = fs.conditioned(rows['new_heldout'], ['L-s7'])
    assert counts['need_left'] == 7 and counts['need_right'] == 7
    assert counts['flip'] == 0 and counts['keep'] == 7
    assert counts['flip_rate'] == 0. and counts['keep_rate'] == 1.
    rows = scenario(3)
    counts = fs.conditioned(rows['new_heldout'], ['L-s7'])
    assert counts['flip'] == 3 and counts['flip_rate'] == round(3 / 7, 6)


def test_raw_mirror_toward_can_be_satisfied_by_the_prior():
    """The confound the metric exists for: all screen-right, five mirror cells still 'toward'."""
    rows = scenario(0, need_left=7, need_right=7)
    mirror = [row for row in rows['new_heldout'].values()
              if row['network'] == 'L-s7' and row['mirror_group'] == 'mirror']
    raw_toward = sum(1 for row in mirror if row['verdict'] == 'toward')
    counts = fs.conditioned(rows['new_heldout'], ['L-s7'])
    assert raw_toward == 5, 'the raw mirror rate is geometry, not behaviour'
    assert counts['flip_mirror'] == 0 and counts['flip'] == 0


def test_judgement_zero_flip_is_not_scale():
    result = fs.judgement(scenario(0))
    assert result['P1']['value'] == 'scale_not_the_cause'
    assert result['P1']['large_flip_by_seed'] == [0, 0, 0]
    assert result['P2']['value'] == 'base_kept'
    assert result['verdict'] == 'scale_not_the_cause'


def test_judgement_positive_flip_scales():
    result = fs.judgement(scenario([1, 1, 0]))
    assert result['P1']['value'] == 'scale_helps'
    assert result['P1']['large_flip_by_seed'] == [1, 1, 0]
    assert result['P2']['value'] == 'base_kept'
    assert result['verdict'] == 'scale_helps'
    assert result['P1_mirror']['large_flip_mirror_by_seed'] == [0, 0, 0]


def test_judgement_partial_and_base_lost():
    partial = fs.judgement(scenario([1, 0, 0]))
    assert partial['P1']['value'] == 'partial_scale' and partial['verdict'] == 'partial_scale'
    mixed = fs.judgement(scenario([1, 1, 1], l_keep=1))
    assert mixed['P2']['value'] == 'base_lost' and mixed['verdict'] == 'mixed_scale'
    assert mixed['P2']['large_keep_rate_by_seed'] == [round(1 / 7, 6)] * 3


def test_judgement_rule_is_the_frozen_one():
    rule = fs.criterion()['judgement']
    assert 'two of three seeds are > 0' in rule['P1']
    assert 'all three are 0' in rule['P1']
    assert rule['P2'].startswith('base survives iff')
    assert 'need-left' in rule['P1'] and 'need_right' in rule['P2']
    assert 'B0 scores 5/7' in rule['geometry']
    assert 'no_relaxation' in rule


# ---------------------------------------------------------------- 8. checkpoint contract
def test_checkpoint_contract_is_round20(backbone, tmp_path):
    from dataclasses import asdict
    from tools.rl import v373_seed_confirm as r16
    model = r16.fresh_policy(backbone, 7)
    sha = r16.state_sha(model.heads.state_dict())
    payload = dict(schema=r20.SCHEMA, arm=fs.CONTROL, training_set='S', seed=7,
                   config=asdict(model.tactical_config), baseline_sha256=file_sha(fs.BASELINE),
                   final_head_state_sha256=sha, anchor_weight=0.,
                   dataset=dict(groups=['wf18-F1-002'], events=['wf18-F1-002-m0'], events_available=12,
                                updates=fs.UPDATES, epochs=16.666667),
                   heads={key: value.detach().cpu() for key, value in model.heads.state_dict().items()})
    path = tmp_path / 'research-heads.pt'
    fs.write_torch(path, payload)
    network = fs.load_network(backbone, path, 'cpu', 'S', 7, sha)
    assert network.heads_sha256 == sha
    with pytest.raises(ValueError, match='arm/seed'):
        fs.load_network(backbone, path, 'cpu', 'L', 7, sha)
    with pytest.raises(ValueError, match='arm/seed'):
        fs.load_network(backbone, path, 'cpu', 'S', 17, sha)


# ---------------------------------------------------------------- 9. file contracts
def test_json_is_opened_exclusively(tmp_path):
    path = tmp_path / 'value.json'
    fs.write_json(path, dict(a=1))
    with pytest.raises(FileExistsError):
        fs.write_json(path, dict(a=2))
    assert json.loads(path.read_text(encoding='utf-8')) == dict(a=1)


def test_weights_are_opened_exclusively(tmp_path):
    path = tmp_path / 'value.pt'
    fs.write_torch(path, dict(a=1))
    with pytest.raises(FileExistsError):
        fs.write_torch(path, dict(a=2))


def test_directories_are_created_exclusively(tmp_path):
    path = tmp_path / 'run'
    assert fs.new_dir(path) == path
    with pytest.raises(FileExistsError):
        fs.new_dir(path)


# ---------------------------------------------------------------- misc contracts
def test_identity_and_criterion_are_frozen_tables(frozen):
    paths = fs.identity_paths()
    assert len(paths) >= 30
    assert all(path.is_file() for path in paths)
    identities = fs.identities()
    assert len(identities) == len(paths)
    from pathlib import Path
    assert str(Path('tests/test_v373_f1_scale.py')) in identities
    assert fs.criterion_sha() == fs.criterion_sha()
    assert len(fs.criterion_sha()) == 64
    assert fs.r24_reference()['summary_sha256'] == fs.identities()[str(fs.R24_SUMMARY)]


def test_no_test_split_is_ever_constructed(frozen):
    assert all(world['scenario']['split'] == 'development' for world in frozen['manifest']['spec']['worlds'])
    assert frozen['split']['spec']['assertions']['test_split'] is False
    assert fs.criterion()['search']['split_label'] == 'development'
    assert fs.criterion()['training']['real_cuda'] is True
    assert fs.criterion()['training']['precision'].startswith('FP32')
    assert fs.criterion()['arms']['S']['groups'] == 6 and fs.criterion()['arms']['S']['events'] == 12


def test_round18_module_is_not_modified_by_this_round():
    assert r18.BUDGET == {'F1': 200, 'F2': 400, 'F3': 200}
    assert r18.TARGET_GROUPS == 6 and r18.ALTERNATE_BLOCK == 16
    assert r18.SEEDS == {'F1': 18001, 'F2': 18002, 'F3': 18003}
    assert sorted(r18.SUPPORTED) == [0, 3, 7, 9, 12, 16]
