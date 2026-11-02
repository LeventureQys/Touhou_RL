"""Contract tests for round23's training-only privileged auxiliary task; CPU only, no training
run, no CUDA, no formal run.

The seven things these tests exist for:
  1. the label is exactly `s.gap` under `s.mirror` - a pure world property, present only on F1
     events, balanced 3/3 on both splits, supervised only on the telegraph-window rows, and it
     recomputes to the row counts the pre-registration freezes;
  2. the auxiliary head lives outside `heads`, so the policy head state dict and its SHA stay
     round22's, and the two arms are byte-identical objects at step 0;
  3. `PrivAuxPolicy` overrides **no** line of `TacticalPolicy.forward_features` - the parent
     body is used verbatim (function-object identity, not text equality) and its forward is
     bitwise the parent's on real data rows;
  4. the captured tensor really is the GRU state: all five heads that consume it are
     recomputed from it and compared bitwise with the forward's own outputs;
  5. the auxiliary gradient reaches only the auxiliary head and the shared GRU - the residual,
     option, target, urgency, termination and slack heads see none of it - and a zero weight
     leaves the auxiliary head bitwise untouched;
  6. the inference path cannot read the label: `PrivNetwork` is a plain `TacticalPolicy`, the
     checkpoint's policy keys carry no auxiliary tensor, the label function is never called on
     that path (measured with a raising stub), the pixel-region set is unchanged and the GRU
     input stays 256 + 29 = 285 wide, so nothing was concatenated into features or history;
  7. the round22 reference table and the round23 mechanism helpers reproduce the frozen
     numbers of the two previous rounds.
"""
from __future__ import annotations
import inspect
import json
import numpy as np
import pytest
import torch
from tools.rl import v373_privileged_aux as pa
from tools.rl import v373_spatial_fuse as r22
from tools.rl import v373_valid_curriculum as r20
from touhou_rl.tactical_policy import TacticalConfig, TacticalPolicy, bos_history
from touhou_rl.tactical_scenarios import WorldState
from touhou_rl.tactical_synthetic_closed_loop import pixel_stack
from touhou_rl.tactical_synthetic_pixels import SIZES
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
                all=train + validation, records=pa.train_records(prereg))


@pytest.fixture(scope='module')
def backbone():
    model, _ = load_vision_checkpoint(pa.BASELINE, map_location='cpu')
    model.eval()
    return model


@pytest.fixture(scope='module')
def encoded(backbone, frozen):
    """One real F1 train event through the encoder, exactly as the run builds it (CPU)."""
    with torch.inference_mode():
        data, vectors, features = pa.encode(backbone, frozen['records'][:1], 'cpu')
    return dict(data=data, vectors=vectors, features=features)


@pytest.fixture(scope='module')
def f1_event(frozen):
    """The first F1 train event: its record, its world and its auxiliary terms."""
    row = next(entry for entry in frozen['records'] if entry['family'] == 'F1')
    world = make_world(row['scenario'])
    return dict(record=row, world=world, terms=pa.aux_terms(world))


@pytest.fixture(scope='module')
def f1_row(backbone, f1_event):
    """The encoded rows of that event (the same tensors the training loop consumes)."""
    with torch.inference_mode():
        data, _, _ = pa.encode(backbone, [f1_event['record']], 'cpu')
    return data[0]


def model_with_heads(backbone, state=None, seed=1234):
    torch.manual_seed(seed)
    model = pa.PrivAuxPolicy(backbone, TacticalConfig(arm='N1'))
    if state is not None:
        model.heads.load_state_dict({key: value.clone() for key, value in state.items()}, strict=True)
    return model


def history_of(entry):
    return {key: value[None] for key, value in entry['history'].items()}


def head_payload(backbone, arm, weight, aux_state=None):
    """A checkpoint payload with the round23 schema; the auxiliary state may be corrupted."""
    torch.manual_seed(0)
    model = pa.PrivAuxPolicy(backbone, TacticalConfig(arm='N1'))
    aux = ({key: value.detach().cpu() for key, value in model.aux.state_dict().items()}
           if aux_state is None else aux_state)
    return dict(schema=pa.HEADS_SCHEMA, arm=arm, seed=7,
                config=pa.asdict(model.tactical_config),
                baseline_sha256=pa.file_sha(pa.BASELINE),
                final_head_state_sha256=pa.r16.state_sha(model.heads.state_dict()),
                final_aux_state_sha256=pa.r16.state_sha(model.aux.state_dict()),
                initial_aux_state_sha256=pa.r16.state_sha(model.aux.state_dict()),
                aux_weight=weight, loss_arm=pa.LOSS_ARM,
                auxiliary_head=model.aux.describe(),
                heads={key: value.detach().cpu() for key, value in model.heads.state_dict().items()},
                aux_state=aux)


def write_payload(path, payload):
    with path.open('xb') as stream:
        torch.save(payload, stream)
    return path


# ---------------------------------------------------------------- 1. the label
def test_label_is_exactly_the_gap_under_the_mirror(frozen):
    """The label is a pure world property: recomputed from s.gap and s.mirror, nothing else."""
    source = inspect.getsource(pa.aux_target)
    assert 'telegraph_gap' in source and 'AUX_THRESHOLD' in source
    assert 'canonical_x' not in source and 'start_x' not in source, \
        'the label must not depend on the player position'
    for entry in frozen['all']:
        world = make_world(entry['scenario'])
        expected = None
        if world.family == 'F1' and world.internal_gap:
            screen = r22.r21.telegraph_gap(world)
            expected = int((screen[0] + screen[1]) / 2. > 192.)
        assert pa.aux_target(world) == expected, entry['event_id']


def test_only_f1_events_carry_a_label(frozen):
    for entry in frozen['all']:
        world = make_world(entry['scenario'])
        value = pa.aux_target(world)
        if world.family == 'F1':
            assert world.internal_gap and value is not None
        else:
            assert value is None, entry['event_id']
            assert pa.aux_mask(world) is None
            assert pa.aux_terms(world) is None


def test_labels_are_balanced_on_both_splits(frozen):
    table = pa.supervision_table(frozen['train'])
    labelled = {key: value for key, value in table.items() if value['target_defined']}
    assert len(labelled) == pa.EXPECTED['aux_events'] == 6
    assert sorted(value['label'] for value in labelled.values()) == ['left'] * 3 + ['right'] * 3
    validation = {key: value for key, value in pa.supervision_table(frozen['validation']).items()
                  if value['target_defined']}
    assert len(validation) == 6
    assert sorted(value['label'] for value in validation.values()) == ['left'] * 3 + ['right'] * 3
    assert sum(value['supervised_rows'] for value in labelled.values()) == \
        pa.EXPECTED['aux_train_rows'] == 128
    assert sum(value['supervised_rows'] for value in validation.values()) == \
        pa.EXPECTED['aux_validation_rows'] == 132
    unlabelled = [key for key, value in table.items() if not value['target_defined']]
    assert len(unlabelled) == pa.EXPECTED['aux_unlabelled_train_events'] == 8
    assert all(table[key]['family'] != 'F1' for key in unlabelled)


def test_the_mask_is_the_telegraph_window(frozen):
    """Supervision is exactly the rows whose frame the renderer draws the band on."""
    for entry in frozen['all']:
        world = make_world(entry['scenario'])
        mask = pa.aux_mask(world)
        if mask is None:
            continue
        window = r22.r21.telegraph_window(world)
        assert mask.shape == (r20.DECISIONS,) and mask.dtype == np.bool_
        expected = np.array([window[0] <= frame < window[1] for frame in r22.r21.DECISION_FRAMES])
        assert np.array_equal(mask, expected)
        assert int(mask.sum()) == sum(1 for frame in r22.r21.DECISION_FRAMES
                                      if window[0] <= frame < window[1])
        assert pa.aux_terms(world)['rows'] == int(mask.sum())


def test_labels_are_not_an_input_surface(encoded):
    """No label, mirror flag or gap number is a history field, a pixel region or a feature."""
    history_keys = set(bos_history(torch.zeros(1, 1, pa.FEATURE_DIM)))
    entry = encoded['data'][0]
    assert set(entry['history']) == history_keys
    assert set(entry['labels']) == {'action_positive', 'action_valid'}
    for name in set(entry['history']) | set(entry['labels']):
        assert not any(word in name for word in ('gap', 'side', 'aux', 'mirror', 'label'))
    assert set(SIZES) == {'playfield', 'hud_score', 'hud_status'}
    assert not any('gap' in name or 'aux' in name for name in SIZES)


# ---------------------------------------------------------------- 2./3. architecture
def test_design_constants_are_frozen():
    assert pa.ARMS == ('A', 'X') and (pa.CONTROL, pa.AUXILIARY) == ('A', 'X')
    assert pa.SEEDS == (7, 17, 27) == r20.SEEDS
    assert pa.LOSS_ARM == r20.CONTROL, 'both arms are the round20/21/22 control loss'
    assert (r20.UPDATES, r20.LR, r20.CLIP, r20.SEQUENCE, r20.BATCH) == (200, 1e-3, 1.0, 80, 1)
    assert pa.AUX_WEIGHT == .5
    assert (pa.FEATURE_DIM, pa.HIDDEN_SIZE, pa.AUX_DIM) == (256, 256, 2)
    assert pa.AUX_THRESHOLD == 192.
    assert pa.AUX_PARAMETERS == 256 * 2 + 2 == 514
    assert pa.GRU_INPUT_WIDTH == 256 + 29 == 285
    assert pa.POLICY_HEADS_ACTIVE == 429200
    assert pa.STATE_HOOK_MODULE == 'option'
    assert pa.EXPECTED['checkpoints'] == 6 and pa.EXPECTED['closed_loop_files'] == 14
    assert pa.EXPECTED['f1_gap_rows'] == 84 and pa.EXPECTED['baseline_rows'] == 168
    assert pa.EXPECTED['aux_events'] == 6 and pa.EXPECTED['aux_unlabelled_train_events'] == 8
    assert len(pa.HEAD_MODULES) == 9 and 'aux' not in pa.HEAD_MODULES


def _own_members(cls):
    """Class-dict names, minus the interpreter-supplied dunders."""
    skip = {'__module__', '__doc__', '__qualname__', '__dict__', '__weakref__',
            '__firstlineno__', '__static_attributes__', '__annotations__'}
    return {name for name in cls.__dict__ if name not in skip}


def test_the_variant_overrides_nothing_of_the_parent():
    """`forward_features` is literally the parent's function object, not an equal copy."""
    assert pa.PrivAuxPolicy.forward_features is TacticalPolicy.forward_features
    assert pa.PrivAuxPolicy.forward_regions is TacticalPolicy.forward_regions
    assert pa.PrivAuxPolicy.forward is TacticalPolicy.forward
    assert pa.PrivAuxPolicy.train is TacticalPolicy.train
    own, parent = _own_members(pa.PrivAuxPolicy), _own_members(TacticalPolicy)
    assert own & parent == {'__init__'}, own & parent
    assert own - parent == {'_capture_state', 'state', 'aux_logits', 'forward_with_aux', 'describe'}
    assert pa.PARENT_DETACH_LINE in inspect.getsource(TacticalPolicy.forward_features)
    init_source = inspect.getsource(pa.PrivAuxPolicy.__init__)
    assert 'super().__init__' in init_source and 'register_forward_pre_hook' in init_source
    assert 'forward_features' not in init_source


def test_auxiliary_head_is_outside_heads(backbone):
    model = model_with_heads(backbone)
    assert 'aux' not in model.heads
    assert {key.split('.')[0] for key in model.heads.state_dict()} == set(pa.HEAD_MODULES)
    assert not any('aux' in key for key in model.heads.state_dict())
    assert set(model.aux.state_dict()) == {'classifier.weight', 'classifier.bias'}
    assert sum(p.numel() for p in model.aux.parameters()) == pa.AUX_PARAMETERS == 514
    described = model.aux.describe()
    assert described['parameters'] == pa.AUX_PARAMETERS
    assert described['outside_heads'] is True and described['never_an_input'] is True


def test_step_zero_is_bitwise_the_same_policy_in_both_arms(backbone, encoded):
    """The two arms differ in one number (the weight) and in nothing else."""
    entry = encoded['data'][0]
    initial = pa.r16.initial_head_shas()
    heads, auxes = set(), set()
    for seed in pa.SEEDS:
        torch.manual_seed(seed)
        model = pa.PrivAuxPolicy(backbone, TacticalConfig(arm='N1'))
        head_sha = pa.r16.state_sha(model.heads.state_dict())
        heads.add(head_sha)
        auxes.add(pa.r16.state_sha(model.aux.state_dict()))
        assert head_sha == initial[str(seed)]
        with torch.inference_mode():
            left = model.forward_features(entry['features'][None], entry['base'][None], history_of(entry))
            right, aux = model.forward_with_aux(entry['features'][None], entry['base'][None],
                                               history_of(entry))
        assert torch.equal(left['policy_logits'], right['policy_logits'])
        assert aux.shape == (1, r20.DECISIONS, pa.AUX_DIM)
        assert pa.masked_candidate(left['policy_logits'][0, 0]) == \
            pa.masked_candidate(right['policy_logits'][0, 0])
    assert len(heads) == len(pa.SEEDS) and len(auxes) == len(pa.SEEDS)


def test_initial_auxiliary_state_is_identical_in_both_arms():
    shas = pa.aux_init_shas()
    assert shas[pa.CONTROL] == shas[pa.AUXILIARY]
    assert len(shas[pa.CONTROL]) == 64


def test_perturbing_the_auxiliary_head_changes_no_policy_logit(backbone, encoded):
    """The strongest form of "it only adds gradient": an arbitrary auxiliary head is inert."""
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
    assert torch.equal(before['residual_logits'], after['residual_logits'])
    assert torch.equal(before['hidden'], after['hidden'])
    assert not torch.equal(aux, torch.zeros_like(aux)), 'the perturbation must be visible in the aux head'


# ---------------------------------------------------------------- 4. the captured state
def test_the_captured_tensor_is_the_gru_state(backbone, encoded):
    """All five heads that consume the state are recomputed from the captured tensor."""
    entry = encoded['data'][0]
    model = model_with_heads(backbone).eval()
    with torch.inference_mode():
        output, _ = model.forward_with_aux(entry['features'][None], entry['base'][None],
                                           history_of(entry))
        captured = model.state()
        assert captured.shape == (1, r20.DECISIONS, pa.HIDDEN_SIZE)
        assert torch.equal(model.heads['option'](captured), output['option_logits'])
        assert torch.equal(model.heads['target'](captured), output['target'])
        assert torch.equal(model.heads['urgency'](captured).sigmoid().squeeze(-1), output['urgency'])
        assert torch.equal(model.heads['termination'](captured), output['termination_logits'])
        assert torch.equal(model.heads['slack'](captured).squeeze(-1), output['slack'])
    with pytest.raises(RuntimeError):
        pa.PrivAuxPolicy(backbone, TacticalConfig(arm='N1')).state()


def test_the_variant_forward_is_the_parent_forward_bitwise(backbone, encoded):
    entry = encoded['data'][0]
    torch.manual_seed(0)
    parent = TacticalPolicy(backbone, TacticalConfig(arm='N1')).eval()
    child = model_with_heads(backbone, parent.heads.state_dict()).eval()
    with torch.inference_mode():
        left = parent.forward_features(entry['features'][None], entry['base'][None], history_of(entry))
        right, _ = child.forward_with_aux(entry['features'][None], entry['base'][None], history_of(entry))
    for key in ('policy_logits', 'residual_logits', 'option_logits', 'target', 'slack', 'urgency',
                'termination_logits', 'hidden', 'base_logits', 'valid_mask'):
        assert torch.equal(left[key], right[key]), key
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(2, 5, 256, generator=generator)
    base = torch.randn(2, 5, 18, generator=generator)
    with torch.inference_mode():
        left = parent.forward_features(features, base)
        right, _ = child.forward_with_aux(features, base)
    assert torch.equal(left['policy_logits'], right['policy_logits'])
    assert torch.equal(left['hidden'], right['hidden'])


# ---------------------------------------------------------------- 5. the loss and the gradient
def test_aux_loss_is_exactly_zero_without_a_label(backbone, encoded):
    entry = encoded['data'][0]
    model = model_with_heads(backbone)
    _, aux = model.forward_with_aux(entry['features'][None], entry['base'][None], history_of(entry))
    empty = torch.zeros((1, r20.DECISIONS), dtype=torch.bool)
    target = torch.zeros((1, r20.DECISIONS), dtype=torch.long)
    value = pa.aux_loss(aux, target, empty)
    assert value['count'] == 0 and value['supervised'] is False and value['accuracy'] is None
    assert float(value['loss'].detach()) == 0.
    value['loss'].backward()
    assert float(model.aux.classifier.weight.grad.abs().max()) == 0.


def test_aux_loss_is_the_cross_entropy_of_the_masked_rows(backbone, f1_event, f1_row):
    """Real row set, real label, real mask: the loss is CE over exactly the supervised rows."""
    terms = f1_event['terms']
    model = model_with_heads(backbone)
    _, aux = model.forward_with_aux(f1_row['features'][None], f1_row['base'][None],
                                   history_of(f1_row))
    target, mask = pa.aux_batch(f1_event['world'])
    assert int(mask.sum()) == terms['rows']
    value = pa.aux_loss(aux, target, mask)
    expected = torch.nn.functional.cross_entropy(aux[mask], target[mask])
    assert torch.equal(value['loss'].detach(), expected.detach())
    assert value['count'] == terms['rows']
    assert value['correct'] == int((aux[mask].argmax(-1) == target[mask]).sum())
    assert value['accuracy'] == value['correct'] / value['count']
    with pytest.raises(ValueError):
        pa.aux_loss(aux, target.to(torch.float32), mask)
    with pytest.raises(ValueError):
        pa.aux_loss(aux, target + 2, mask)
    with pytest.raises(ValueError):
        pa.aux_loss(aux, target, mask.to(torch.long))


def test_the_auxiliary_gradient_touches_only_the_aux_head_and_the_gru(backbone, f1_event, f1_row):
    """The decision heads are downstream of the state: the auxiliary term cannot reach them."""
    model = model_with_heads(backbone)
    target, mask = pa.aux_batch(f1_event['world'])
    _, aux = model.forward_with_aux(f1_row['features'][None], f1_row['base'][None],
                                   history_of(f1_row))
    model.zero_grad(set_to_none=True)
    pa.aux_loss(aux, target, mask)['loss'].backward()
    assert model.aux.classifier.weight.grad is not None
    assert float(model.aux.classifier.weight.grad.abs().max()) > 0.
    assert model.heads['gru'].weight_ih_l0.grad is not None
    assert float(model.heads['gru'].weight_ih_l0.grad.abs().max()) > 0., \
        'the shared GRU must receive the auxiliary gradient - that is the intervention'
    for name in ('residual', 'option', 'target', 'urgency', 'termination', 'slack', 'option_embedding'):
        assert all(parameter.grad is None for parameter in model.heads[name].parameters()), name


def test_zero_weight_leaves_the_auxiliary_head_bitwise_untouched(backbone, f1_event, f1_row):
    model = model_with_heads(backbone)
    start = pa.r16.state_sha(model.aux.state_dict())
    labels = {key: value[None] for key, value in f1_row['labels'].items()}
    target, mask = pa.aux_batch(f1_event['world'])
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=r20.LR)
    for _ in range(2):
        output, aux = model.forward_with_aux(f1_row['features'][None], f1_row['base'][None],
                                            history_of(f1_row))
        loss = r20.arm_loss(pa.LOSS_ARM, output, labels)['loss'] \
            + 0. * pa.aux_loss(aux, target, mask)['loss']
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert pa.r16.state_sha(model.aux.state_dict()) == start, \
        'a zero-weight auxiliary term must not move its own head'


def test_a_positive_weight_moves_the_auxiliary_head_and_the_gru(backbone, f1_event, f1_row):
    model = model_with_heads(backbone)
    aux_start = pa.r16.state_sha(model.aux.state_dict())
    gru_start = {key: value.clone() for key, value in model.heads['gru'].state_dict().items()}
    labels = {key: value[None] for key, value in f1_row['labels'].items()}
    target, mask = pa.aux_batch(f1_event['world'])
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=r20.LR)
    for _ in range(2):
        output, aux = model.forward_with_aux(f1_row['features'][None], f1_row['base'][None],
                                            history_of(f1_row))
        loss = r20.arm_loss(pa.LOSS_ARM, output, labels)['loss'] \
            + pa.AUX_WEIGHT * pa.aux_loss(aux, target, mask)['loss']
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert pa.r16.state_sha(model.aux.state_dict()) != aux_start
    after = model.heads['gru'].state_dict()
    assert any(not torch.equal(after[key], gru_start[key]) for key in gru_start)


# ---------------------------------------------------------------- 6. the inference path
def test_the_inference_network_is_the_parent_class_and_has_no_aux(backbone, tmp_path):
    payload = head_payload(backbone, pa.AUXILIARY, pa.AUX_WEIGHT)
    path = write_payload(tmp_path / 'research-heads.pt', payload)
    network = pa.PrivNetwork(backbone, path, 'cpu')
    assert type(network.model) is TacticalPolicy
    assert not hasattr(network.model, 'aux')
    assert not hasattr(network, 'aux')
    assert set(network.model.heads.state_dict()) == set(payload['heads'])
    assert set(payload) == set(pa.PAYLOAD_KEYS)
    assert {key.split('.')[0] for key in payload['heads']} == set(pa.HEAD_MODULES)
    with pytest.raises(ValueError):
        pa.PrivNetwork(backbone, path, 'cpu', expected_aux_sha='0' * 64)
    wrong = dict(payload, aux_weight=0.)
    with pytest.raises(ValueError):
        pa.PrivNetwork(backbone, write_payload(tmp_path / 'wrong-arm.pt', wrong), 'cpu')
    extra = dict(payload, label_side=1)
    with pytest.raises(ValueError):
        pa.PrivNetwork(backbone, write_payload(tmp_path / 'extra-key.pt', extra), 'cpu')


def test_noise_in_the_auxiliary_state_changes_no_closed_loop_decision(backbone, f1_event, tmp_path):
    """The closed loop never loads the auxiliary tensors, so corrupting them is unobservable."""
    payload = head_payload(backbone, pa.AUXILIARY, pa.AUX_WEIGHT)
    noisy = {key: torch.full_like(value, 7.) for key, value in payload['aux_state'].items()}
    world = f1_event['world']
    regions, _ = pixel_stack([r22.r21.render_state_telegraph(world, WorldState(0, float(world.start_x)))])
    logits = []
    for index, aux_state in enumerate((payload['aux_state'], noisy)):
        path = write_payload(tmp_path / f'heads-{index}.pt', dict(payload, aux_state=aux_state))
        network = pa.PrivNetwork(backbone, path, 'cpu')
        value, hidden = network('N1', regions, None, None, True)
        assert value.shape == (18,) and hidden is not None
        logits.append(value)
    assert torch.equal(logits[0], logits[1])


def test_the_label_function_is_never_called_on_the_inference_path(backbone, f1_event, tmp_path,
                                                                 monkeypatch):
    """Measured, not asserted: make every label entry point raise and run the closed-loop path."""
    path = write_payload(tmp_path / 'research-heads.pt',
                         head_payload(backbone, pa.CONTROL, 0.))

    def boom(*args, **kwargs):
        raise AssertionError('the privileged label was read on the inference path')

    for name in ('aux_target', 'aux_mask', 'aux_terms', 'aux_batch', 'supervision_table'):
        monkeypatch.setattr(pa, name, boom)
    network = pa.PrivNetwork(backbone, path, 'cpu')
    world = f1_event['world']
    regions, _ = pixel_stack([r22.r21.render_state_telegraph(world, WorldState(0, float(world.start_x)))])
    logits, hidden = network('N1', regions, None, None, True)
    assert logits.shape == (18,) and torch.isfinite(logits).all() and hidden is not None


def test_the_policy_input_surface_is_unchanged(backbone, encoded):
    """No privileged channel: the GRU input is still 256 + 29, the regions are still the four ROIs."""
    entry = encoded['data'][0]
    model = model_with_heads(backbone)
    assert model.heads['gru'].input_size == 285
    captured = {}

    def grab(module, inputs):
        captured['inp'] = inputs[0]
        captured['calls'] = captured.get('calls', 0) + 1
        return None

    handle = model.heads['gru'].register_forward_pre_hook(grab)
    with torch.inference_mode():
        model.forward_with_aux(entry['features'][None], entry['base'][None], history_of(entry))
    handle.remove()
    assert captured['inp'].shape == (1, 1, 285)
    assert captured['calls'] == r20.DECISIONS, 'the GRU is stepped once per decision row'
    assert entry['features'].shape == (r20.DECISIONS, 256)
    assert pa.FEATURE_DIM + 19 + 4 + 1 + 1 + 2 + 1 + 1 == 285


# ---------------------------------------------------------------- 7. the reference and the mechanics
def test_round22_reference_reproduces():
    reference = pa.r22_reference()
    assert reference['f1_validation_feasible'] == {'A': [2, 2, 2], 'S': [2, 2, 2]}
    assert reference['all_validation_feasible'] == {'A': [3, 3, 3], 'S': [3, 3, 3]}
    assert reference['control_final_head_state_sha256'] == pa.R22_CONTROL_FINAL_HEAD_SHA
    assert reference['comparison']['deltas'] == [0, 0, 0]
    assert reference['comparison']['verdict'] == 'unchanged'
    assert reference['comparison']['identical_first_hit_cells'] == [14, 14, 14]
    assert reference['direction']['A'] == pa.R22_DIRECTION['A']
    assert reference['direction']['S'] == pa.R22_DIRECTION['S']
    assert reference['six_head_direction'] == pa.R22_DIRECTION['six_heads']
    assert reference['spatial_sha256'] == pa.R22_SPATIAL_SHA
    assert len(reference['feature_norm']) == 14


def test_mechanism_helpers_reproduce_round22_direction_counts():
    reference = json.loads((pa.R22_EVIDENCE / 'run/f1-gap.json').read_text(encoding='utf-8'))
    round22_names = [r20.name_of(arm, seed) for arm in r22.ARMS for seed in r20.SEEDS]
    control = r22.direction_counts(reference['validation'],
                                   networks=[name for name in round22_names if name.startswith('A-')])
    assert control == pa.R22_DIRECTION['A'] == {'base': {'toward': 6, 'hold': 3, 'away': 0},
                                                'mirror': {'toward': 3, 'hold': 0, 'away': 6}}
    heads = r22.direction_counts(reference['validation'], networks=round22_names)
    assert heads == pa.R22_DIRECTION['six_heads']
    assert heads['mirror'] == {'toward': 6, 'hold': 1, 'away': 11}
    assert heads['base'] == {'toward': 12, 'hold': 3, 'away': 3}
    assert r22.direction_counts(reference['validation'], networks=['B0']) == \
        {'base': {'toward': 3, 'hold': 0, 'away': 0}, 'mirror': {'toward': 0, 'hold': 0, 'away': 3}}


def test_mechanism_block_accepts_this_round_s_arm_names():
    """The round22 mechanism block is parameterised by network name, so A/X names work."""
    rows = {}
    for name in ['B0'] + pa.all_names():
        for mirror in (False, True):
            rows[f'{name}/{mirror}'] = dict(network=name, mirror=mirror, verdict='hold',
                                            signed_error_px=-8.0, in_gap_at_wall=False,
                                            entered_gap=False, mechanism='stopped_left_of_the_gap',
                                            screen_direction='right', toward_rate=None)
    block = pa.mechanism_block(rows)
    assert set(block) == {'A', 'X', 'B0'}
    assert block['A']['groups']['base']['hold'] == 3
    assert block['X']['flip_rate']['mirror_toward_rate'] == 0.
    assert block['A']['per_seed']['7']['base']['cells'] == 1
    assert block['B0']['groups']['mirror']['cells'] == 1


def test_aux_summary_is_the_confusion_of_the_given_pairs():
    summary = pa.aux_summary([0, 0, 1, 1], [0, 1, 1, 1])
    assert summary['rows'] == 4 and summary['correct'] == 3
    assert summary['accuracy'] == .75
    assert summary['balanced_accuracy'] == (1. + .5) / 2.
    assert summary['chance'] == .5 and summary['majority_baseline'] == .5
    assert summary['confusion'] == {'left': {'left': 1, 'right': 1}, 'right': {'left': 0, 'right': 2}}
    assert summary['per_class']['left'] == dict(support=2, correct=1, recall=.5)


def test_the_encode_pass_is_round22_s_on_the_same_device(encoded):
    """The frozen per-event bytes are what make the feature identity a measurement.

    The stored vectors were produced on CUDA (round22's run device); this test runs on CPU, so
    the agreement is float32 rounding rather than bit equality - the *exact* digest equality is
    asserted by `preregister`, `verify_prereg` and `run`, which all execute on CUDA. The gap
    measured here is reported so the weaker claim cannot hide behind the stronger one.
    """
    reference = pa.r22_reference()
    stored = json.loads((pa.R22_EVIDENCE / 'run/records.json').read_text(encoding='utf-8'))['spatial']
    worst = 0.
    for event_id, vector in encoded['vectors'].items():
        path = pa.R22_EVIDENCE / 'run' / stored[event_id]['path']
        assert pa.file_sha(path) == stored[event_id]['sha256'], event_id
        with np.load(path) as archive:
            reference_vector = np.array(archive['spatial'])
        assert reference_vector.shape == vector.shape == (r20.DECISIONS, 94)
        worst = max(worst, float(np.abs(reference_vector - vector).max()))
        assert abs(pa.feature_norms(encoded['data'])[event_id]
                   - reference['feature_norm'][event_id]) <= 1e-5, event_id
    assert worst < 1e-5, 'the CPU recomputation must be the CUDA value to float32 rounding'
