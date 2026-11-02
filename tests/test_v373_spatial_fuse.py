"""Contract tests for the round22 spatial fusion; CPU only, no training run, no CUDA.

The five things these tests exist for:
  1. the 94-d spatial vector is exactly the frozen definition, and the F4/F8 it is built
     from are the tensors `forward_core` itself used (recomputed tail, bitwise equal);
  2. the variant head's forward is the parent head's forward bitwise, its body is the parent
     body with exactly one line changed (`features.detach()` removed), and that one line is
     *why* the fusion can be trained at all;
  3. step 0 of arm S is bitwise arm A: identical policy logits, identical masked action,
     delta exactly zero - the fusion cannot perturb the initial policy;
  4. the fusion layer really receives gradient (zero at step 0 because the residual head is
     also zero-initialised, non-zero from step 1) while arm A's fusion stays frozen at zero;
  5. the round22 mechanism metrics (direction toward the gap, signed parking error)
     reproduce round21's frozen f1-gap table on round21's own rows.
"""
from __future__ import annotations
import inspect
import json
import numpy as np
import pytest
import torch
from tools.rl import v373_spatial_fuse as sf
from tools.rl import v373_valid_curriculum as r20
from touhou_rl import vmodels3
from touhou_rl.tactical_policy import TacticalConfig, TacticalPolicy, bos_history
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.vmodels import load_vision_checkpoint


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def frozen():
    manifest = r20.load_worlds()
    split = r20.load_split(manifest)
    train = r20.world_events(manifest, split, r20.TRAIN_SPLIT)
    validation = r20.world_events(manifest, split, r20.VALIDATION_SPLIT)
    prereg = dict(worlds=dict(train_events=[row['event_id'] for row in train],
                              train_scenario_sha256={row['event_id']: row['scenario_sha256']
                                                     for row in train}))
    return dict(manifest=manifest, split=split, train=train, validation=validation,
                all=train + validation, records=sf.train_records(prereg))


@pytest.fixture(scope='module')
def backbone():
    model, _ = load_vision_checkpoint(sf.BASELINE, map_location='cpu')
    model.eval()
    return model


@pytest.fixture(scope='module')
def regions(frozen):
    """Four real decision rows of one stored round21 train event, as the encoder feeds them."""
    entry = frozen['records'][0]
    pixels = r20.pixel_tensors(sf.R21_EVIDENCE / 'run' / entry['pixels']['path'])
    return {key: value[:4].clone() for key, value in pixels.items()}


@pytest.fixture(scope='module')
def encoded(backbone, frozen):
    """One real event through the encoder, exactly as the run builds it (CPU)."""
    with torch.inference_mode():
        data, vectors, feature_table = sf.spatial_vectors(
            backbone, frozen['records'][:1], sf.R21_EVIDENCE / 'run', 'cpu')
    return dict(data=data, vectors=vectors, features=feature_table)


def code_lines(function):
    """Source lines of a function body, without the docstring, blank lines and comments."""
    lines = inspect.getsource(function).splitlines()[1:]
    out, in_doc = [], False
    for line in lines:
        stripped = line.strip()
        if not in_doc and (stripped.startswith('"""') or stripped.startswith("'''")):
            quote = '"""' if stripped.startswith('"""') else "'''"
            if stripped.count(quote) >= 2:
                continue
            in_doc = True
            continue
        if in_doc:
            if '"""' in line or "'''" in line:
                in_doc = False
            continue
        if not stripped or stripped.startswith('#'):
            continue
        out.append(line.rstrip())
    return out


def model_with_heads(backbone, state=None, seed=1234):
    torch.manual_seed(seed)
    model = sf.SpatialHead(backbone, TacticalConfig(arm='N1'))
    if state is not None:
        model.heads.load_state_dict({key: value.clone() for key, value in state.items()}, strict=True)
    return model


# ---------------------------------------------------------------- 1. design and vector
def test_frozen_design_constants():
    assert sf.ARMS == ('A', 'S') and (sf.CONTROL, sf.SPATIAL) == ('A', 'S')
    assert sf.SEEDS == (7, 17, 27) == r20.SEEDS
    assert sf.LOSS_ARM == r20.CONTROL, 'both arms are the round20/21 control loss'
    assert (r20.UPDATES, r20.LR, r20.CLIP, r20.SEQUENCE, r20.BATCH) == (200, 1e-3, 1.0, 80, 1)
    assert sf.SPATIAL_POOL == (7, 6)
    assert (sf.SPATIAL_COLUMNS, sf.SPATIAL_ROWS, sf.SPATIAL_DIM) == (24, 28, 94)
    assert 42 + 24 + 28 == 94
    assert sf.FEATURE_DIM == 256
    assert (sf.F4_SHAPE, sf.F8_SHAPE, sf.PLAYFIELD_HW) == ((56, 48), (28, 24), (224, 192))
    assert sf.FUSION_PARAMETERS == 94 * 256 + 256 == 24320
    assert sf.PARENT_DETACH_LINE in inspect.getsource(TacticalPolicy.forward_features)
    assert sf.PARENT_DETACH_LINE not in inspect.getsource(sf.SpatialHead.forward_features)
    assert sf.VARIANT_DETACH_LINE in inspect.getsource(sf.SpatialHead.forward_features)
    assert sf.EXPECTED['checkpoints'] == 6 and sf.EXPECTED['closed_loop_files'] == 14
    assert sf.EXPECTED['f1_gap_rows'] == 84 and sf.EXPECTED['baseline_rows'] == 168
    assert sf.EXPECTED['train_events'] == 14 and sf.EXPECTED['validation_events'] == 14


def test_fusion_layer_starts_at_exactly_zero():
    for enabled in (True, False):
        fusion = sf.SpatialFusion(enabled=enabled)
        assert float(fusion.projection.weight.detach().abs().max()) == 0.
        assert float(fusion.projection.bias.detach().abs().max()) == 0.
        assert [p.requires_grad for p in fusion.projection.parameters()] == [enabled, enabled]
        described = fusion.describe()
        assert described['parameters'] == sf.FUSION_PARAMETERS
        assert described['active'] == (sf.FUSION_PARAMETERS if enabled else 0)
    assert sf.fusion_init_shas()[sf.CONTROL] == sf.fusion_init_shas()[sf.SPATIAL]


def test_spatial_vector_is_the_frozen_definition(backbone, regions):
    f4, f8 = sf.SpatialFusion.spatial_maps(backbone, regions)
    assert tuple(f4.shape) == (4, 64) + sf.F4_SHAPE
    assert tuple(f8.shape) == (4, 96) + sf.F8_SHAPE
    vector = sf.SpatialFusion.vector(f8)
    assert tuple(vector.shape) == (4, 94)
    pooled = torch.nn.functional.adaptive_avg_pool2d(f8, (7, 6)).mean(dim=1).flatten(1)
    assert torch.equal(vector[:, :42], pooled)
    assert torch.equal(vector[:, 42:66], f8.mean(dim=(1, 2)))
    assert torch.equal(vector[:, 66:], f8.mean(dim=(1, 3)))
    synthetic = torch.zeros(1, 2, 28, 24)
    synthetic[0, :, :, 5] = 3.
    column_block = sf.SpatialFusion.vector(synthetic)[0, 42:66]
    assert int(column_block.argmax()) == 5
    assert float(column_block[5]) == 3. and float(column_block.sum()) == 3.


def test_the_spatial_maps_are_the_tensors_forward_core_used(backbone, regions):
    """Recompute forward_core's tail from the extracted F4/F8: bitwise equality, not closeness."""
    core = backbone.forward_core(regions)
    f4, f8 = sf.SpatialFusion.spatial_maps(backbone, regions)
    heatmap = backbone.locate(f4, f8)
    position, stats = vmodels3.decode_location(heatmap)
    location = backbone.location_mlp(torch.cat([position, stats], dim=-1))
    checked = backbone._checked_regions(regions)
    fused = torch.cat([backbone.local_features(f4, position), backbone.global_features(f8),
                       backbone.hud_score(backbone._hud_input(checked['hud_score'])),
                       backbone.hud_status(backbone._hud_input(checked['hud_status'])),
                       location], dim=-1)
    features = backbone.trunk(fused)
    assert torch.equal(core['heatmap_logits'], heatmap)
    assert torch.equal(core['features'], features)
    assert torch.equal(core['policy_logits'], backbone.policy_head(features))
    assert torch.equal(core['position'], position)


def test_spatial_vectors_are_not_inference_tensors(encoded):
    """Ordinary tensors are required: an autograd layer cannot take an inference tensor."""
    for row in encoded['data']:
        assert not row['spatial'].is_inference()
        assert not row['features'].is_inference()
        assert row['spatial'].shape == (r20.DECISIONS, 94)
        assert row['features'].shape == (r20.DECISIONS, 256)
        assert row['base'].shape == (r20.DECISIONS, 18)


# ---------------------------------------------------------------- 2. the variant head
def test_variant_body_is_the_parent_body_with_one_line_changed():
    parent = code_lines(TacticalPolicy.forward_features)
    child = code_lines(sf.SpatialHead.forward_features)
    assert len(parent) == len(child)
    differences = [(index, left, right) for index, (left, right) in enumerate(zip(parent, child))
                   if left != right]
    assert len(differences) == 1, differences
    index, left, right = differences[0]
    assert left.strip() == sf.PARENT_DETACH_LINE
    assert right.strip() == sf.VARIANT_DETACH_LINE


def test_variant_forward_is_the_parent_forward_bitwise(backbone, regions):
    torch.manual_seed(0)
    parent = TacticalPolicy(backbone, TacticalConfig(arm='N1')).eval()
    child = model_with_heads(backbone, parent.heads.state_dict()).eval()
    with torch.inference_mode():
        core = backbone.forward_core(regions)
        features = core['features'][None]
        base = core['policy_logits'][None]
        history = bos_history(features)
        history['previous_candidate'].fill_(3)
        history['previous_hold'].fill_(2)
        history['frame_delta'].fill_(2)
        history['history_valid'].fill_(True)
        left = parent.forward_features(features, base, history)
        right = child.forward_features(features, base, history)
    for key in ('policy_logits', 'residual_logits', 'option_logits', 'target', 'slack',
                'urgency', 'termination_logits', 'hidden'):
        assert torch.equal(left[key], right[key]), key
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(2, 5, 256, generator=generator)
    base = torch.randn(2, 5, 18, generator=generator)
    with torch.inference_mode():
        left = parent.forward_features(features, base)
        right = child.forward_features(features, base)
    assert torch.equal(left['policy_logits'], right['policy_logits'])
    assert torch.equal(left['hidden'], right['hidden'])


def test_the_parent_detach_would_freeze_the_fusion_forever(backbone, regions):
    """Measure the reason for the one-line deviation instead of asserting it."""
    torch.manual_seed(0)
    parent = TacticalPolicy(backbone, TacticalConfig(arm='N1'))
    child = model_with_heads(backbone, parent.heads.state_dict())
    # A fresh residual is zero, which blocks every feature-side gradient for an unrelated
    # reason; give both heads the same small non-zero residual so the path exists.
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(parent.heads.residual.weight.shape, generator=generator) * .01
    with torch.no_grad():
        parent.heads.residual.weight.copy_(weight)
        child.heads.residual.weight.copy_(weight)
    core = backbone.forward_core(regions)
    features = core['features'][None]
    base = core['policy_logits'][None]

    def gradient_through(head):
        fusion = sf.SpatialFusion(enabled=True)
        fused = features + fusion.projection(torch.zeros(1, 1, 94) + .5)
        output = head.forward_features(fused, base)
        output['policy_logits'].sum().backward()
        grad = fusion.projection.weight.grad
        return None if grad is None else float(grad.abs().max())

    assert gradient_through(parent) is None, 'the parent detaches `features`: the fusion is unreachable'
    assert gradient_through(child) > 0., 'the variant keeps the path the fusion needs'


# ---------------------------------------------------------------- 3./4. step 0 and gradients
def test_step0_of_S_is_bitwise_arm_A(backbone, encoded):
    model = model_with_heads(backbone).eval()
    arm_a = sf.SpatialFusion(enabled=False).eval()
    arm_s = sf.SpatialFusion(enabled=True).eval()
    with torch.inference_mode():
        for entry in encoded['data']:
            left = sf.fused_infer(model, arm_a, entry)
            right = sf.fused_infer(model, arm_s, entry)
            assert torch.equal(left['policy_logits'], right['policy_logits'])
            assert torch.equal(left['hidden'], right['hidden'])
            for step in (0, -1):
                assert sf.masked_candidate(left['policy_logits'][0, step]) == \
                       sf.masked_candidate(right['policy_logits'][0, step])
            delta = arm_s.fuse(entry['features'][None], entry['spatial'][None]) - entry['features'][None]
            assert float(delta.abs().max()) == 0.
    assert arm_s.projection.weight.grad is None


def test_fusion_gradient_is_zero_at_step0_and_live_from_step1(backbone, encoded):
    """Zero at step 0 is structural: the residual head is zero-initialised too, so at step 0
    there is no path from `features` to the logits at all. From step 1 the path exists."""
    entry = encoded['data'][0]
    labels = {key: value[None] for key, value in entry['labels'].items()}
    model = model_with_heads(backbone)
    fusion = sf.SpatialFusion(enabled=True)
    parameters = [p for p in model.parameters() if p.requires_grad] + \
                 [p for p in fusion.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=r20.LR)
    norms = []
    for _ in range(2):
        output = sf.fused_infer(model, fusion, entry)
        head = r20.arm_loss(sf.LOSS_ARM, output, labels)
        assert head['parts']['has_supervision']
        optimizer.zero_grad(set_to_none=True)
        head['loss'].backward()
        norms.append(float(fusion.projection.weight.grad.norm()))
        optimizer.step()
    assert norms[0] == 0., 'the zero-initialised residual head blocks the path at step 0'
    assert norms[1] > 0., 'from step 1 the fusion must receive gradient'
    assert float(fusion.projection.weight.detach().abs().max()) > 0.


def test_arm_A_keeps_its_fusion_frozen_at_zero(backbone, encoded):
    entry = encoded['data'][0]
    labels = {key: value[None] for key, value in entry['labels'].items()}
    model = model_with_heads(backbone)
    fusion = sf.SpatialFusion(enabled=False)
    parameters = [p for p in model.parameters() if p.requires_grad] + \
                 [p for p in fusion.parameters() if p.requires_grad]
    assert len(parameters) == len([p for p in model.parameters() if p.requires_grad])
    optimizer = torch.optim.Adam(parameters, lr=r20.LR)
    for _ in range(2):
        output = sf.fused_infer(model, fusion, entry)
        head = r20.arm_loss(sf.LOSS_ARM, output, labels)
        optimizer.zero_grad(set_to_none=True)
        head['loss'].backward()
        optimizer.step()
    assert float(fusion.projection.weight.detach().abs().max()) == 0.
    assert fusion.projection.weight.grad is None


def test_initial_head_sha_is_the_frozen_table(backbone):
    computed = {str(seed): sf.r16.state_sha(sf.fresh_head(backbone, seed).heads.state_dict())
                for seed in sf.SEEDS}
    frozen_prereg = json.loads((sf.R21_EVIDENCE / 'preregistration.json').read_text(encoding='utf-8'))
    assert computed == frozen_prereg['initialization']['per_seed']
    assert computed == sf.r16.initial_head_shas()
    assert len(set(computed.values())) == len(sf.SEEDS)


def test_parameter_budget(backbone):
    model = model_with_heads(backbone)
    fusion = sf.SpatialFusion(enabled=True)
    head_active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert head_active == 429200, 'the round21 control head budget'
    assert sum(p.numel() for p in fusion.parameters() if p.requires_grad) == 24320
    assert head_active + 24320 == 453520


# ---------------------------------------------------------------- 5. round21 mechanics
def test_direction_counts_reproduce_round21():
    reference = sf.r21_reference()
    gaps = json.loads((sf.R21_EVIDENCE / 'run/f1-gap.json').read_text(encoding='utf-8'))
    round21_names = [r20.name_of(arm, seed) for arm in r20.ARMS for seed in r20.SEEDS]
    control = sf.direction_counts(gaps['validation'],
                                  networks=[n for n in round21_names if n.startswith('A-')])
    assert control == reference['direction']['control'] == sf.R21_DIRECTION['control']
    heads = sf.direction_counts(gaps['validation'], networks=round21_names)
    assert heads == reference['direction']['heads'] == sf.R21_DIRECTION['heads']
    assert sf.direction_counts(gaps['validation'], networks=['B0']) == sf.R21_DIRECTION['b0']
    assert heads['base'] == {'toward': 15, 'hold': 3, 'away': 0}
    assert heads['mirror'] == {'toward': 3, 'hold': 0, 'away': 15}


def test_signed_error_matches_round21_parking_distance():
    gaps = json.loads((sf.R21_EVIDENCE / 'run/f1-gap.json').read_text(encoding='utf-8'))
    checked = 0
    for row in gaps['validation'].values():
        value = sf.signed_error(row['gap'], row['canonical_at_wall'])
        if value is None:
            continue
        assert abs(value) == pytest.approx(row['distance_to_gap_at_wall'], abs=1e-9)
        at_wall, (low, high) = row['canonical_at_wall'], row['gap']
        if value < 0:
            assert at_wall < low and row['mechanism'] != 'in_the_gap_at_the_wall'
        elif value > 0:
            assert at_wall > high and row['mechanism'] != 'in_the_gap_at_the_wall'
        else:
            assert low <= at_wall <= high and row['in_gap_at_wall'] is True
        checked += 1
    assert checked == len(gaps['validation'])


def test_direction_classification_of_hand_built_rows():
    assert sf.classify_direction(dict(gap=[248., 264.], start_canonical=216., direction='right')) == \
        dict(need='right', verdict='toward')
    assert sf.classify_direction(dict(gap=[248., 264.], start_canonical=216., direction='left'))['verdict'] == 'away'
    assert sf.classify_direction(dict(gap=[248., 264.], start_canonical=216., direction='hold'))['verdict'] == 'hold'
    assert sf.classify_direction(dict(gap=[100., 120.], start_canonical=200., direction='left')) == \
        dict(need='left', verdict='toward')
    assert sf.signed_error([248., 264.], 216.) == -32.
    assert sf.signed_error([248., 264.], 300.) == 36.
    assert sf.signed_error([248., 264.], 256.) == 0.


def test_geometry_table_matches_round21s_frozen_table(frozen):
    table = sf.world_table(frozen['all'])
    assert len(table) == 28
    for entry in frozen['records']:
        stored = json.loads((sf.R21_EVIDENCE / 'run' / entry['rollout']['path'])
                            .read_text(encoding='utf-8'))
        row = table[entry['event_id']]
        if row['family'] != 'F1':
            assert row['gap_screen'] is None and row['preview_window'] is None
            assert row['decisions_in_preview'] is None and row['last_preview_frame'] is None
            continue
        assert row['internal_gap'] is True
        stored_window = stored['telegraph']['window']
        assert row['decisions_in_preview'] == len([frame for frame in range(0, 160, 2)
                                                   if stored_window[0] <= frame < stored_window[1]])
        assert row['last_preview_frame'] == row['wall_frame'] - 1
        assert row['gap_screen'] == stored['telegraph']['gap']
        assert row['preview_window'] == stored_window
        world = make_world(entry['scenario'])
        assert [list(span) for span in sf.r21.telegraph_spans(world)] == stored['telegraph']['spans']


def test_round21_reference_is_self_consistent():
    reference = sf.r21_reference()
    assert reference['f1_validation_feasible'] == {'A': [2, 2, 2], 'B': [2, 2, 2]}
    assert reference['all_validation_feasible'] == {'A': [3, 3, 3], 'B': [2, 2, 2]}
    assert reference['control_final_head_state_sha256'] == sf.R21_CONTROL_FINAL_HEAD_SHA
    assert len(set(reference['control_final_head_state_sha256'].values())) == 3


# ---------------------------------------------------------------- 6. probe machinery
def _synthetic_worlds(base_scenario):
    """Three F1 geometries x two mirrors, with gaps on both screen sides."""
    centres = {'g0': (88., 104.), 'g1': (200., 220.), 'g2': (140., 160.)}
    worlds = {}
    for group, gap in centres.items():
        for mirror in (False, True):
            spec = dict(base_scenario)
            spec.update(event_id=f'{group}-m{int(mirror)}', template_group=group, mirror=mirror,
                        family='F1', internal_gap=True, gap=gap,
                        start_x=168. if mirror else 216.)
            worlds[spec['event_id']] = make_world(spec)
    return worlds


def test_ridge_recovers_a_planted_linear_target(frozen):
    """Plumbing only: if the target is linear in the vector, the probe must recover it.

    The nuisance dimensions are held constant across rows on purpose: with 94 dimensions and
    a few hundred rows a ridge is effectively a kernel interpolator, and the point of this
    test is the probe machinery, not the p >> n regime (which the round reports separately,
    with a lambda sweep, on the real vectors).
    """
    generator = np.random.default_rng(7)
    worlds = _synthetic_worlds(frozen['train'][0]['scenario'])
    vectors, positions, rows_by_event = {}, {}, {}
    for event_id, world in worlds.items():
        gap = sf.r21.telegraph_gap(world)
        centre = (gap[0] + gap[1]) / 2.
        rows = 6
        value = np.zeros((rows, sf.SPATIAL_DIM), dtype=np.float64)
        value[:, 0] = centre / 100.
        value[:, 1] = 1.
        vectors[event_id] = value
        positions[event_id] = [centre - 10. * row for row in range(rows)]
        rows_by_event[event_id] = list(range(rows))
    result = sf.spatial_probe(vectors, vectors, worlds, positions, rows_by_event)
    assert result['spatial_94']['side_accuracy'] == 1.
    assert result['spatial_94']['centre_mae_px'] < 20.
    assert set(result['spatial_94']['held_out_groups']) == {'g0', 'g1', 'g2'}
    assert result['design']['ridge']['lambda_'] == sf.RIDGE['lambda_'] == 1.
    assert set(result['spatial_94']['lambda_sweep']) == {str(value) for value in sf.RIDGE['lambda_sweep']}
    assert result['features_256']['centre_mae_px'] < 20.


def test_mirror_symmetry_plumbing():
    worlds = {f'g0-m{int(mirror)}': type('W', (), dict(template_group='g0', mirror=mirror))()
              for mirror in (False, True)}
    left = np.zeros((3, sf.SPATIAL_DIM), dtype=np.float32)
    left[:, 42:66] = np.arange(24, dtype=np.float32)
    right = np.zeros((3, sf.SPATIAL_DIM), dtype=np.float32)
    right[:, 42:66] = np.arange(24, dtype=np.float32)[::-1]
    result = sf.mirror_symmetry({'g0-m0': left, 'g0-m1': right}, worlds)
    assert result['pairs'][0]['correlation'] == pytest.approx(1.)
    assert result['pairs'][0]['mean_abs_difference'] == pytest.approx(0.)


def test_flip_summary_counts_reversals():
    probes = [dict(network='X', event_id='e', steps=4, stored_candidate_matches=4,
                   rows=[dict(action=2, action_flipped=-2), dict(action=1, action_flipped=-2),
                         dict(action=0, action_flipped=0), dict(action=2, action_flipped=2)])]
    summary = sf.flip_summary(probes)
    assert summary['X']['rows'] == 4 and summary['X']['flips'] == 2
    assert summary['X']['same'] == 2 and summary['X']['holds_true'] == 1
    assert summary['X']['flip_rate'] == 0.5
    assert summary['X']['moving_rows'] == 3 and summary['X']['moving_flips'] == 2
    assert summary['X']['flip_rate_when_moving'] == pytest.approx(2 / 3)
    assert summary['X']['replay_fidelity'] == 1.


def test_the_probe_rebuilds_the_runner_inputs_exactly(frozen, backbone):
    """Run the frozen B0 runner on one F1 holdout event, then replay its own trace.

    This is the test that pins the input-mirror probe to reality: `replay_images` must rebuild
    the runner's stacks, so the replay's masked argmax must equal the stored candidate on
    every one of the 80 decisions. An off-by-one image list would fail here immediately.
    """
    from tools.rl import v373_telegraph as r21
    entry = next(row for row in frozen['validation'] if row['event_id'] == 'wf18-F1-005-m0')
    world = make_world(entry['scenario'])
    with torch.inference_mode():
        row = sf.r20.compact(r21.run_event_rendered(world, 'B0', r20.CONDITION,
                                                   r20.BaselineNetwork(backbone, 'cpu'),
                                                   render=r21.render_state_telegraph))
        assert len(row['steps']) == r20.DECISIONS
        images = sf.replay_images(world, row['steps'])
        assert len(images) == len(row['steps'])
        assert np.array_equal(images[0], r21.render_state_telegraph(
            world, sf.WorldState(0, float(world.start_x))))
        assert np.array_equal(images[7], r21.render_state_telegraph(
            world, sf.WorldState(int(row['steps'][7]['frame']), float(row['steps'][7]['x_before']))))
        probe = sf.flip_probe(backbone, 'cpu', world, row['steps'], 'B0', head_only=True)
    assert probe['steps'] == len(row['steps'])
    assert probe['stored_candidate_matches'] == len(row['steps']), \
        'the replay must reproduce every stored decision, not most of them'
    assert len(probe['rows']) == len([step for step in row['steps']
                                      if 3 <= int(step['frame']) < 48])
