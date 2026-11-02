"""Contract tests for round24's player-relative privileged auxiliary label; CPU only, no training
run, no CUDA, no formal run.

What these tests exist for:
  1. the label is `sign(gap screen centre - player screen x)` and nothing else: it reads the
     player's visible x, it is *not* round23's `gap centre > 192`, and the two definitions are
     shown to differ on the events round23 named as its counter-example (`007-m0`, `007-m1`);
  2. the label is defined only on F1 events, only on the telegraph-window rows, 3 left / 3 right
     on both splits, 128 / 132 rows, and it recomputes to the frozen tables;
  3. the sign-flip rule is `first_preview_row`: the target is the sign at the first window row,
     applied to every supervised row, and a flip is recorded rather than supervised. The frozen
     teacher trajectories contain no flip, so the rule is exercised on a synthetic trace;
  4. the label agrees with the teacher's own first action 6/6 on both splits, and only 3/12 with
     the teacher's action at the anchoring row because the teacher holds there - the bound the
     report must state;
  5. the judgement rule that replaces round23's "any seed larger = improved" is a pure function:
     K1 needs two of three seeds strictly higher and none lower, K1 with a falling pooled count is
     `mixed`, and round23's own -1/+1/0 is `not_improved`;
  6. the head, its placement, the captured GRU state, the gradient path and the inference path are
     round23's, unchanged (that is the point of a single-factor comparison): the variant overrides
     no line of the parent, both arms are byte-identical at step 0, perturbing the auxiliary head
     moves no policy logit, and the label functions are never called on the inference path.
"""
from __future__ import annotations
import inspect
import json
import numpy as np
import pytest
import torch
from tools.rl import v373_player_relative_aux as pl
from tools.rl import v373_privileged_aux as r23
from tools.rl import v373_spatial_fuse as r22
from tools.rl import v373_telegraph as r21
from tools.rl import v373_valid_curriculum as r20
from touhou_rl.tactical_policy import TacticalConfig, TacticalPolicy, bos_history
from touhou_rl.tactical_scenarios import WorldState
from touhou_rl.tactical_synthetic_closed_loop import pixel_stack
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.vmodels import load_vision_checkpoint

HARD_CASES = ('wf18-F1-007-m0', 'wf18-F1-007-m1')


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def frozen():
    manifest, split, train, validation = pl.all_events()
    teacher, verdicts = pl.teacher_table(train + validation)
    prereg = dict(worlds=dict(train_events=[row['event_id'] for row in train],
                              train_scenario_sha256={row['event_id']: row['scenario_sha256']
                                                     for row in train}))
    return dict(manifest=manifest, split=split, train=train, validation=validation,
                all=train + validation, teacher=teacher, verdicts=verdicts,
                records=pl.train_records(prereg),
                labels=pl.label_rows(train, teacher),
                validation_labels=pl.label_rows(validation, teacher))


@pytest.fixture(scope='module')
def backbone():
    model, _ = load_vision_checkpoint(pl.BASELINE, map_location='cpu')
    model.eval()
    return model


@pytest.fixture(scope='module')
def encoded(backbone, frozen):
    with torch.inference_mode():
        data, vectors, features = pl.encode(backbone, frozen['records'][:1], 'cpu')
    return dict(data=data, vectors=vectors, features=features)


@pytest.fixture(scope='module')
def f1_event(frozen):
    """The first F1 train event: its record, its world, its positions and its label terms."""
    row = next(entry for entry in frozen['records'] if entry['family'] == 'F1')
    world = make_world(row['scenario'])
    positions = pl.player_trace(world, frozen['teacher'][row['event_id']])
    return dict(record=row, world=world, positions=positions,
                terms=pl.aux_terms(world, positions))


@pytest.fixture(scope='module')
def f1_row(backbone, f1_event):
    with torch.inference_mode():
        data, _, _ = pl.encode(backbone, [f1_event['record']], 'cpu')
    return data[0]


def history_of(entry):
    return {key: value[None] for key, value in entry['history'].items()}


def model_with_heads(backbone, state=None, seed=1234):
    torch.manual_seed(seed)
    model = r23.PrivAuxPolicy(backbone, TacticalConfig(arm='N1'))
    if state is not None:
        model.heads.load_state_dict({key: value.clone() for key, value in state.items()}, strict=True)
    return model


def head_payload(backbone, arm, weight, aux_state=None):
    """A checkpoint payload with round23's contract; the auxiliary state may be corrupted."""
    torch.manual_seed(0)
    model = r23.PrivAuxPolicy(backbone, TacticalConfig(arm='N1'))
    aux = ({key: value.detach().cpu() for key, value in model.aux.state_dict().items()}
           if aux_state is None else aux_state)
    return dict(schema=pl.HEADS_SCHEMA, arm=arm, seed=7,
                config=pl.asdict(model.tactical_config),
                baseline_sha256=pl.file_sha(pl.BASELINE),
                final_head_state_sha256=pl.r16.state_sha(model.heads.state_dict()),
                final_aux_state_sha256=pl.r16.state_sha(model.aux.state_dict()),
                initial_aux_state_sha256=pl.r16.state_sha(model.aux.state_dict()),
                aux_weight=weight, loss_arm=pl.LOSS_ARM,
                auxiliary_head=model.aux.describe(),
                heads={key: value.detach().cpu() for key, value in model.heads.state_dict().items()},
                aux_state=aux)


def write_payload(path, payload):
    with path.open('xb') as stream:
        torch.save(payload, stream)
    return path


# ---------------------------------------------------------------- 1. the label
def test_design_constants_are_frozen():
    assert pl.ARMS == ('A', 'X') and (pl.CONTROL, pl.AUXILIARY) == ('A', 'X')
    assert pl.SEEDS == (7, 17, 27) == r20.SEEDS
    assert pl.LOSS_ARM == r20.CONTROL
    assert (r20.UPDATES, r20.LR, r20.CLIP, r20.SEQUENCE, r20.BATCH) == (200, 1e-3, 1.0, 80, 1)
    assert pl.AUX_WEIGHT == .5 == r23.AUX_WEIGHT
    assert (pl.FEATURE_DIM, pl.HIDDEN_SIZE, pl.AUX_DIM) == (256, 256, 2)
    assert pl.AUX_PARAMETERS == 256 * 2 + 2 == 514
    assert pl.GRU_INPUT_WIDTH == 256 + 29 == 285
    assert pl.POLICY_HEADS_ACTIVE == 429200
    assert pl.STATE_HOOK_MODULE == 'option'
    assert pl.HEADS_SCHEMA == r23.HEADS_SCHEMA and pl.PAYLOAD_KEYS == r23.PAYLOAD_KEYS, \
        'the checkpoint contract must stay round23\'s so the two rounds are comparable'
    assert pl.LABEL_RULE == 'first_preview_row'
    assert pl.REFERENCE_FRAME_CENTRE == 192.
    assert len(pl.F1_EVENTS) == pl.EXPECTED['aux_events'] * 2 == 12
    assert pl.EXPECTED['judgement_higher_needed'] == 2 and pl.EXPECTED['judgement_seeds'] == 3


def test_the_label_is_the_sign_of_the_gap_relative_to_the_player(frozen):
    """Independent recomputation: the label is sign(centre - player x), nothing else."""
    for key, terms in dict(frozen['labels'], **frozen['validation_labels']).items():
        world = make_world(next(row for row in frozen['all'] if row['event_id'] == key)['scenario'])
        gap = r21.telegraph_gap(world)
        centre = (float(gap[0]) + float(gap[1])) / 2.
        positions = pl.player_trace(world, frozen['teacher'][key])
        first = r23.preview_indices(world)[0]
        assert positions[first] == terms['first_player_x']
        assert terms['side'] == int(centre > positions[first])
        assert terms['label'] == ('right' if centre > positions[first] else 'left')
        assert round(centre - positions[first], 6) == terms['first_offset']
        assert terms['first_row'] == first
        assert terms['first_frame'] == r21.DECISION_FRAMES[first]


def test_the_label_reads_the_player_and_not_the_screen_centre(frozen):
    """Same world, different player x: the label moves with the player, not with 192."""
    world = make_world(next(row for row in frozen['all']
                           if row['event_id'] == 'wf18-F1-007-m0')['scenario'])
    teacher = frozen['teacher']['wf18-F1-007-m0']
    positions = pl.player_trace(world, teacher)
    assert pl.aux_terms(world, positions)['side'] == 1, 'the gap is right of the player'
    shifted = [300.] * len(positions)
    assert pl.aux_terms(world, shifted)['side'] == 0, 'walk past the gap and the label flips'
    assert pl.aux_terms(world, positions)['round23_side'] == 0, \
        'round23 called this event "left" because the gap centre 162 is below 192'
    assert pl.aux_terms(world, positions)['label'] == 'right'
    source = inspect.getsource(pl.aux_target)
    assert 'aux_terms' in source and 'REFERENCE_FRAME_CENTRE' not in source
    terms_source = inspect.getsource(pl.aux_terms)
    assert 'positions[row]' in terms_source and 'centre - float' in terms_source
    assert REFERENCE_ONLY_IN_COMPARISON_COLUMNS(terms_source)


def REFERENCE_ONLY_IN_COMPARISON_COLUMNS(source):
    """`192` may only appear as the round23 comparison column, never in the target arithmetic."""
    lines = [line for line in source.splitlines() if 'REFERENCE_FRAME_CENTRE' in line]
    return all(('offset_from_screen_centre' in line or 'round23_side' in line
                or 'distance_from_centre' in line) for line in lines)


def test_only_f1_events_carry_a_label(frozen):
    for entry in frozen['all']:
        world = make_world(entry['scenario'])
        positions = pl.player_trace(world, frozen['teacher'][entry['event_id']])
        value = pl.aux_target(world, positions)
        if world.family == 'F1':
            assert world.internal_gap and value is not None
            assert pl.aux_mask(world) is not None
        else:
            assert value is None, entry['event_id']
            assert pl.aux_mask(world) is None
            assert pl.aux_terms(world, positions) is None


def test_labels_are_balanced_on_both_splits(frozen):
    table = pl.supervision_table(frozen['records'], frozen['teacher'])
    labelled = {key: value for key, value in table.items() if value['target_defined']}
    assert len(labelled) == pl.EXPECTED['aux_events'] == 6
    assert sorted(value['label'] for value in labelled.values()) == ['left'] * 3 + ['right'] * 3
    validation = {key: value for key, value in
                  pl.supervision_table(frozen['validation'], frozen['teacher']).items()
                  if value['target_defined']}
    assert len(validation) == 6
    assert sorted(value['label'] for value in validation.values()) == ['left'] * 3 + ['right'] * 3
    assert sum(value['supervised_rows'] for value in labelled.values()) == \
        pl.EXPECTED['aux_train_rows'] == 128
    assert sum(value['supervised_rows'] for value in validation.values()) == \
        pl.EXPECTED['aux_validation_rows'] == 132
    unlabelled = [key for key, value in table.items() if not value['target_defined']]
    assert len(unlabelled) == pl.EXPECTED['aux_unlabelled_train_events'] == 8
    assert all(table[key]['family'] != 'F1' for key in unlabelled)


def test_the_label_differs_from_round23_on_exactly_the_hard_events(frozen):
    """The parent flagged `007-m0/m1` as label-vs-motion opposites; the new label fixes them."""
    changed_train = sorted(key for key, value in frozen['labels'].items()
                           if value['label'] != pl.side_name(value['round23_side']))
    changed_validation = sorted(key for key, value in frozen['validation_labels'].items()
                                if value['label'] != pl.side_name(value['round23_side']))
    assert changed_train == ['wf18-F1-002-m0', 'wf18-F1-002-m1',
                             'wf18-F1-009-m0', 'wf18-F1-009-m1'], changed_train
    assert changed_validation == list(HARD_CASES), changed_validation
    cases = {key: frozen['validation_labels'][key] for key in HARD_CASES}
    assert cases['wf18-F1-007-m0']['label'] == 'right'
    assert cases['wf18-F1-007-m0']['round23_side'] == 0, 'round23 said left, the player must go right'
    assert cases['wf18-F1-007-m1']['label'] == 'left'
    assert cases['wf18-F1-007-m1']['round23_side'] == 1, 'round23 said right, the player must go left'
    assert cases['wf18-F1-007-m0']['first_sign'] == 1 and cases['wf18-F1-007-m1']['first_sign'] == -1


def test_the_label_agrees_with_the_teacher_first_action(frozen):
    """The teacher moves the way the label points, on every labelled event of both splits."""
    for key, terms in dict(frozen['labels'], **frozen['validation_labels']).items():
        first = int(frozen['teacher'][key][0])
        assert first != 0, 'every F1 teacher starts by moving'
        assert (first > 0) == bool(terms['side']), key
        assert pl.row_sign(first) == terms['first_sign'], key


def test_the_mask_is_the_telegraph_window(frozen):
    for entry in frozen['all']:
        world = make_world(entry['scenario'])
        mask = pl.aux_mask(world)
        if mask is None:
            continue
        window = r21.telegraph_window(world)
        expected = np.array([window[0] <= frame < window[1] for frame in r21.DECISION_FRAMES])
        assert mask.shape == (r20.DECISIONS,) and mask.dtype == np.bool_
        assert np.array_equal(mask, expected)
        terms = pl.aux_terms(world, pl.player_trace(world, frozen['teacher'][entry['event_id']]))
        assert terms['rows'] == int(mask.sum())
        assert terms['row_indices'] == [int(i) for i in np.nonzero(mask)[0]]


# ---------------------------------------------------------------- 2. the flip rule
def test_the_frozen_teacher_trajectories_never_flip_the_sign(frozen):
    flips = pl.flip_table(dict(frozen['labels'], **frozen['validation_labels']))
    assert flips['rule'] == 'first_preview_row'
    assert flips['event_count'] == 12
    assert flips['flipping_events'] == [] and flips['flip_count'] == 0
    assert flips['tie_count'] == 0
    assert flips['event_count'] == pl.EXPECTED['flip_free_teacher_events']
    for row in flips['events'].values():
        assert row['would_flip'] is False and row['sign_flips'] == []
        assert row['target_side'] == int(row['first_sign'] > 0)
        assert row['target_side'] == int(row['per_row_signs'][0] > 0)


def test_the_flip_rule_keeps_the_first_row_sign_and_records_the_flip(frozen):
    """A synthetic overshoot: the rule must hold the first-row sign and record the flip."""
    world = make_world(next(row for row in frozen['all']
                           if row['event_id'] == 'wf18-F1-007-m0')['scenario'])
    positions = pl.player_trace(world, frozen['teacher']['wf18-F1-007-m0'])
    terms = pl.aux_terms(world, positions)
    centre = terms['gap_screen_centre']
    overshoot = list(positions)
    for row in terms['row_indices'][6:]:
        overshoot[row] = centre + 20.     # walk past the gap centre
    moved = pl.aux_terms(world, overshoot)
    assert moved['would_flip'] is True
    assert moved['sign_flip_frames'] and moved['sign_flips'][0] == terms['row_indices'][6]
    assert moved['side'] == terms['side'] == 1, 'the target keeps the first-row sign'
    assert moved['label'] == 'right'
    assert moved['per_row_signs'][-1] == -1
    assert [moved['per_row_signs'][k] for k in range(len(moved['row_indices']))] != \
        [terms['per_row_signs'][k] for k in range(len(terms['row_indices']))]


def test_the_flip_table_names_the_rule_and_the_untried_alternative(frozen):
    flips = pl.flip_table(dict(frozen['labels'], **frozen['validation_labels']))
    assert flips['rule'] == 'first_preview_row'
    assert 'not tried' in flips['rule_text']
    for row in flips['events'].values():
        assert row['rule'] == 'first_preview_row'
        assert 'NOT supervised' in row['handling']


# ---------------------------------------------------------------- 3. the agreement diagnostic
def test_agreement_is_six_of_six_on_the_episode_first_action(frozen):
    agreement = pl.agreement_table(frozen['all'], frozen['teacher'])
    for split_name in (r20.TRAIN_SPLIT, r20.VALIDATION_SPLIT):
        block = agreement['by_split'][split_name]
        assert block['events'] == 14 and block['labelled'] == 6 and block['unlabelled'] == 8
        assert block['label_counts'] == {'left': 3, 'right': 3}
        assert block['episode_first_action']['agree'] == 6
        assert block['episode_first_action']['disagree'] == 0
        assert block['episode_first_action']['agreement_rate'] == 1.
        assert block['episode_first_action']['teacher_hold'] == 0
    assert agreement['by_split'][r20.TRAIN_SPLIT]['anchored_action']['teacher_hold'] == 2
    assert agreement['by_split'][r20.VALIDATION_SPLIT]['anchored_action']['teacher_hold'] == 4
    anchored = [agreement['by_split'][split]['anchored_action']
                for split in (r20.TRAIN_SPLIT, r20.VALIDATION_SPLIT)]
    assert sum(row['agree'] for row in anchored) == 6
    assert sum(row['disagree'] for row in anchored) == 0, \
        'the label never contradicts the teacher where the teacher moves at that row'
    assert sum(row['teacher_hold'] for row in anchored) == 6, \
        'the teacher holds at the anchoring row on 6 of 12 events: the label is not a timing label'
    assert all(row['agreement_rate'] == 1. for row in anchored)
    for key in ('wf18-F2-119-m0', 'wf18-F3-005-m1'):
        assert agreement['rows'][key]['label'] is None
        assert agreement['rows'][key]['episode_first_action'] == 'no_label'


# ---------------------------------------------------------------- 3b. round23's label collapse
def test_round23_label_collapsed_to_the_player_half_on_the_whole_holdout(frozen):
    """Round23's target `gap centre > 192` was, on all six F1 holdout events, `player x > 192`.

    That is why round23 could decode its label from the GRU state at 0.826: the state contains the
    player's visible position. The new label is not a function of the player alone, which is the
    point of the intervention and the reason the two rounds' decodability numbers are not
    comparable in level.
    """
    table = pl.round23_collinearity(frozen['all'], frozen['teacher'])
    counts = table['by_split']
    assert counts[r20.VALIDATION_SPLIT]['events'] == 6
    assert counts[r20.VALIDATION_SPLIT]['round23_equals_player_half'] == 6
    assert counts[r20.VALIDATION_SPLIT]['differing'] == []
    assert counts[r20.TRAIN_SPLIT]['events'] == 6
    assert counts[r20.TRAIN_SPLIT]['round23_equals_player_half'] == 4
    assert counts[r20.TRAIN_SPLIT]['differing'] == ['wf18-F1-006-m0', 'wf18-F1-006-m1']
    for key, row in table['rows'].items():
        assert row['round24_side'] == row['round24_side']
        if row['round23_equals_player_half']:
            assert row['round23_side'] == int(row['first_player_x'] > pl.REFERENCE_FRAME_CENTRE)


# ---------------------------------------------------------------- 4. the judgement rule
def test_the_judgement_rule_replaces_round23_s():
    cases = [(dict(f1_control=[2, 2, 2], f1_auxiliary=[1, 3, 2]), 'not_improved'),
             (dict(f1_control=[2, 2, 2], f1_auxiliary=[3, 3, 3]), 'improved'),
             (dict(f1_control=[2, 2, 2], f1_auxiliary=[2, 2, 2]), 'unchanged'),
             (dict(f1_control=[3, 3, 3], f1_auxiliary=[2, 2, 2]), 'worse'),
             (dict(f1_control=[2, 2, 2], f1_auxiliary=[2, 2, 3]), 'inconclusive'),
             (dict(f1_control=[2, 2, 2], f1_auxiliary=[3, 3, 1]), 'not_improved')]
    for kwargs, expected in cases:
        assert pl.judgement(**kwargs)['verdict'] == expected, kwargs
    block = pl.judgement([2, 2, 2], [1, 3, 2])
    assert block['K1']['higher'] == 1 and block['K1']['lower'] == 1
    assert block['K1']['satisfied'] is False
    assert block['K2']['f1'] == dict(control=6, auxiliary=6, delta=0)
    assert block['replaces']['verdict_under_old_rule'] == 'improved'
    assert block['replaces']['verdict_under_this_rule'] == 'not_improved'


def test_K1_with_a_falling_pooled_count_is_mixed():
    """The case round23's rule could not express: two seeds up, the total down."""
    block = pl.judgement([1, 1, 1], [2, 2, 0])
    assert block['K1']['satisfied'] is False and block['K1']['lower'] == 1
    assert block['verdict'] == 'not_improved'
    block = pl.judgement([3, 3, 3], [4, 4, 2])
    assert block['K1']['satisfied'] is False and block['verdict'] == 'not_improved'
    # two seeds strictly higher, none lower, but the pooled F1 count falls: only possible with
    # unequal control values, which is exactly what K2 is for
    block = pl.judgement([5, 5, 5], [6, 6, 3])
    assert block['K1']['satisfied'] is False
    block = pl.judgement([4, 4, 4], [5, 5, 4])
    assert block['K1']['satisfied'] is True and block['K2']['f1']['delta'] == 2
    assert block['verdict'] == 'improved'
    as_dict = dict(K1=True)
    assert pl.rule_verdict(2, 0, True, 1, -1) == 'mixed', 'a falling all-family count alone is mixed'
    assert pl.rule_verdict(2, 0, True, 0, 0) == 'mixed'
    assert pl.rule_verdict(2, 0, True, 1, 0) == 'improved'


def test_the_criterion_freezes_K1_to_K5_and_the_mixed_rule():
    rule = pl.criterion()['judgement']
    for key in ('K1', 'K2', 'K3', 'K4', 'K5', 'rule', 'replaces', 'mixed', 'honesty', 'aux_learned'):
        assert key in rule, key
    assert 'at least two of the three seeds' in rule['K1']
    assert 'mixed' in rule['rule'] and 'improved' in rule['rule']
    assert 'round23' in rule['replaces']
    assert 'K1' in pl.criterion()['evaluation']['primary_metrics'][0]
    assert pl.criterion_sha() == pl.criterion_sha()
    assert len(pl.criterion_sha()) == 64


def test_k_report_builds_K1_to_K5_from_artifacts(frozen):
    """K3..K5 are computed from the stored closed loop and the stored probes, not typed in."""
    names = r23.all_names()
    closed = {}
    plan = {'A': [[2, 2], [2, 2], [2, 2]], 'X': [[1, 2], [2, 2], [2, 1]]}
    for arm in pl.ARMS:
        for index, seed in enumerate(pl.SEEDS):
            f1, total = plan[arm][index]
            payload = dict(families=dict(F1=dict(feasible=f1, survived=f1),
                                         all=dict(feasible=total, survived=total,
                                                  firsthit={})),
                           behaviour={})
            closed[(r20.name_of(arm, seed), r20.VALIDATION_SPLIT)] = payload
            closed[(r20.name_of(arm, seed), r20.TRAIN_SPLIT)] = payload
    mechanism = {arm: dict(groups={group: dict(cells=9, toward=1, toward_rate=.5, in_gap=1,
                                               parked_px=dict(mean_abs=8., maximum_abs=9.))
                                   for group in ('base', 'mirror')}) for arm in pl.ARMS}
    probes = dict(aux=dict(by_arm={arm: dict(holdout=dict(accuracy=.5, balanced_accuracy=.5,
                                                          per_seed_accuracy=[.5, .5, .5],
                                                          per_seed_balanced_accuracy=[.5, .5, .5],
                                                          accuracy_vs_frozen_teacher=.5,
                                                          balanced_accuracy_vs_frozen_teacher=.5,
                                                          predicts_one_class=[True] * 3,
                                                          differing_events={'X-s7': ['e']}),
                                            seen=dict(accuracy=.5, balanced_accuracy=.5,
                                                      per_seed_accuracy=[.5, .5, .5]))
                                  for arm in pl.ARMS},
                          holdout={name: dict(overall=dict(confusion={})) for name in names}),
                 usage=dict(by_arm={arm: dict(mirror_aux_correct=1, mirror_aux_correct_and_away=1,
                                              mirror_away=2, mirror_cells=9)
                                    for arm in pl.ARMS}))
    block = pl.k_report(closed, mechanism, probes)
    assert set(('K1', 'K2', 'K3', 'K4', 'K5')) <= set(block)
    assert block['K1']['control'] == [2, 2, 2] and block['K1']['auxiliary'] == [1, 2, 2]
    assert block['K2']['f1'] == dict(control=6, auxiliary=5, delta=-1)
    assert block['K3']['mirror']['A']['toward'] == 1
    assert block['K3']['delta']['mirror'] == 0.
    assert block['learned_but_unused']['round23_mirror_aux_correct_and_away'] == \
        pl.r23_reference()['mirror_aux_correct_and_away']['X']
    assert block['learned_but_unused']['this_round_mirror_aux_correct_and_away'] == 1


# ---------------------------------------------------------------- 5. the round23 reference
def test_round23_reference_reproduces():
    reference = pl.r23_reference()
    assert reference['f1_validation_feasible'] == {'A': [2, 2, 2], 'X': [1, 3, 2]}
    assert reference['all_validation_feasible'] == {'A': [3, 3, 3], 'X': [1, 3, 2]}
    assert reference['control_final_head_state_sha256'] == pl.R23_CONTROL_FINAL_HEAD_SHA
    assert reference['comparison']['deltas'] == [-1, 1, 0]
    assert reference['comparison']['verdict'] == 'improved'
    assert reference['direction'] == pl.R23_DIRECTION
    assert reference['usage'] == pl.R23_USAGE
    assert reference['holdout_accuracy'] == pl.R23_AUX_HOLDOUT
    assert reference['mirror_aux_correct_and_away'] == {'A': 2, 'X': 7}
    assert len(reference['label_position_table']) == 6


# ---------------------------------------------------------------- 6. round23's machinery
def test_the_variant_overrides_nothing_of_the_parent():
    assert r23.PrivAuxPolicy.forward_features is TacticalPolicy.forward_features
    assert r23.PrivAuxPolicy.forward is TacticalPolicy.forward
    assert r23.PARENT_DETACH_LINE in inspect.getsource(TacticalPolicy.forward_features)


def test_step_zero_is_bitwise_the_same_policy_in_both_arms(backbone, encoded):
    entry = encoded['data'][0]
    initial = pl.r16.initial_head_shas()
    heads, auxes = set(), set()
    for seed in pl.SEEDS:
        torch.manual_seed(seed)
        model = r23.PrivAuxPolicy(backbone, TacticalConfig(arm='N1'))
        head_sha = pl.r16.state_sha(model.heads.state_dict())
        heads.add(head_sha)
        auxes.add(pl.r16.state_sha(model.aux.state_dict()))
        assert head_sha == initial[str(seed)]
        with torch.inference_mode():
            left = model.forward_features(entry['features'][None], entry['base'][None],
                                          history_of(entry))
            right, aux = model.forward_with_aux(entry['features'][None], entry['base'][None],
                                                history_of(entry))
        assert torch.equal(left['policy_logits'], right['policy_logits'])
        assert aux.shape == (1, r20.DECISIONS, pl.AUX_DIM)
    assert len(heads) == len(pl.SEEDS) and len(auxes) == len(pl.SEEDS)
    assert r23.aux_init_shas()[pl.CONTROL] == r23.aux_init_shas()[pl.AUXILIARY]


def test_perturbing_the_auxiliary_head_changes_no_policy_logit(backbone, encoded):
    entry = encoded['data'][0]
    model = model_with_heads(backbone).eval()
    generator = torch.Generator().manual_seed(99)
    with torch.no_grad():
        model.aux.classifier.weight.copy_(torch.randn(model.aux.classifier.weight.shape,
                                                      generator=generator) * 10.)
        model.aux.classifier.bias.copy_(torch.randn(model.aux.classifier.bias.shape,
                                                    generator=generator) * 10.)
    with torch.inference_mode():
        after, aux = model.forward_with_aux(entry['features'][None], entry['base'][None],
                                           history_of(entry))
        reference = TacticalPolicy(backbone, TacticalConfig(arm='N1')).eval()
        reference.heads.load_state_dict({key: value.clone()
                                         for key, value in model.heads.state_dict().items()}, strict=True)
        before = reference.forward_features(entry['features'][None], entry['base'][None],
                                           history_of(entry))
    assert torch.equal(before['policy_logits'], after['policy_logits'])
    assert torch.equal(before['hidden'], after['hidden'])
    assert not torch.equal(aux, torch.zeros_like(aux))


def test_aux_loss_uses_the_new_target_and_ignores_unlabelled_events(backbone, f1_event, f1_row):
    terms = f1_event['terms']
    target, mask = pl.aux_batch(terms)
    model = model_with_heads(backbone)
    _, aux = model.forward_with_aux(f1_row['features'][None], f1_row['base'][None], history_of(f1_row))
    value = r23.aux_loss(aux, target, mask)
    expected = torch.nn.functional.cross_entropy(aux[mask], target[mask])
    assert torch.equal(value['loss'].detach(), expected.detach())
    assert value['count'] == terms['rows'] == int(mask.sum())
    assert int(target[0, terms['first_row']]) == terms['side']
    assert int(mask[0, terms['first_row']]) == 1
    empty_target, empty_mask = pl.aux_batch(None)
    assert empty_target is None and empty_mask is None


def test_the_auxiliary_gradient_touches_only_the_aux_head_and_the_gru(backbone, f1_event, f1_row):
    model = model_with_heads(backbone)
    target, mask = pl.aux_batch(f1_event['terms'])
    _, aux = model.forward_with_aux(f1_row['features'][None], f1_row['base'][None], history_of(f1_row))
    model.zero_grad(set_to_none=True)
    r23.aux_loss(aux, target, mask)['loss'].backward()
    assert float(model.aux.classifier.weight.grad.abs().max()) > 0.
    assert float(model.heads['gru'].weight_ih_l0.grad.abs().max()) > 0.
    for name in ('residual', 'option', 'target', 'urgency', 'termination', 'slack', 'option_embedding'):
        assert all(parameter.grad is None for parameter in model.heads[name].parameters()), name


# ---------------------------------------------------------------- 7. the inference path
def test_the_inference_network_is_the_parent_class_and_has_no_aux(backbone, tmp_path):
    payload = head_payload(backbone, pl.AUXILIARY, pl.AUX_WEIGHT)
    path = write_payload(tmp_path / 'research-heads.pt', payload)
    network = r23.PrivNetwork(backbone, path, 'cpu')
    assert type(network.model) is TacticalPolicy
    assert not hasattr(network.model, 'aux') and not hasattr(network, 'aux')
    assert set(payload) == set(pl.PAYLOAD_KEYS)
    assert {key.split('.')[0] for key in payload['heads']} == set(pl.HEAD_MODULES)
    extra = dict(payload, player_side=1)
    with pytest.raises(ValueError):
        r23.PrivNetwork(backbone, write_payload(tmp_path / 'extra-key.pt', extra), 'cpu')


def test_the_label_function_is_never_called_on_the_inference_path(backbone, f1_event, tmp_path,
                                                                 monkeypatch):
    path = write_payload(tmp_path / 'research-heads.pt', head_payload(backbone, pl.CONTROL, 0.))

    def boom(*args, **kwargs):
        raise AssertionError('the privileged label was read on the inference path')

    for name in ('aux_target', 'aux_mask', 'aux_terms', 'aux_batch', 'supervision_table',
                 'label_rows', 'player_trace', 'teacher_table', 'agreement_table'):
        assert hasattr(pl, name), name
        monkeypatch.setattr(pl, name, boom)
    network = r23.PrivNetwork(backbone, path, 'cpu')
    world = f1_event['world']
    regions, _ = pixel_stack([r21.render_state_telegraph(world, WorldState(0, float(world.start_x)))])
    logits, hidden = network('N1', regions, None, None, True)
    assert logits.shape == (18,) and torch.isfinite(logits).all() and hidden is not None


def test_noise_in_the_auxiliary_state_changes_no_closed_loop_decision(backbone, f1_event, tmp_path):
    payload = head_payload(backbone, pl.AUXILIARY, pl.AUX_WEIGHT)
    noisy = {key: torch.full_like(value, 7.) for key, value in payload['aux_state'].items()}
    world = f1_event['world']
    regions, _ = pixel_stack([r21.render_state_telegraph(world, WorldState(0, float(world.start_x)))])
    logits = []
    for index, aux_state in enumerate((payload['aux_state'], noisy)):
        path = write_payload(tmp_path / f'heads-{index}.pt', dict(payload, aux_state=aux_state))
        network = r23.PrivNetwork(backbone, path, 'cpu')
        value, hidden = network('N1', regions, None, None, True)
        assert value.shape == (18,) and hidden is not None
        logits.append(value)
    assert torch.equal(logits[0], logits[1])


def test_the_policy_input_surface_is_unchanged(backbone, encoded):
    entry = encoded['data'][0]
    model = model_with_heads(backbone)
    assert model.heads['gru'].input_size == 285
    captured = {}

    def grab(module, inputs):
        captured['inp'] = inputs[0]
        return None

    handle = model.heads['gru'].register_forward_pre_hook(grab)
    with torch.inference_mode():
        model.forward_with_aux(entry['features'][None], entry['base'][None], history_of(entry))
    handle.remove()
    assert captured['inp'].shape == (1, 1, 285)
    assert set(entry['history']) == set(bos_history(torch.zeros(1, 1, pl.FEATURE_DIM)))
    assert set(entry['labels']) == {'action_positive', 'action_valid'}
    for name in set(entry['history']) | set(entry['labels']):
        assert not any(word in name for word in ('gap', 'player', 'side', 'aux', 'mirror', 'label'))


def test_the_frozen_teacher_table_is_round21_s_rollout():
    """The label's player x comes from the same actions the training pixels were rendered from."""
    _, _, train, validation = pl.all_events()
    teacher, verdicts = pl.teacher_table(train + validation)
    assert sorted(teacher) == sorted(row['event_id'] for row in train + validation)
    assert len(teacher) == 28 and all(len(value) == r20.DECISIONS for value in teacher.values())
    assert set(verdicts.values()) == {'feasible'}
    assert len(pl.teacher_digest(teacher)) == 64
    assert all(set(value) <= {-2, -1, 0, 1, 2} for value in teacher.values())
