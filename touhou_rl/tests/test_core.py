"""Unit tests for the parts of the framework that do not need a live game.

Run from the repository root::

    python -m pytest touhou_rl/tests -q
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from touhou_rl import config_file, dataset, menu, models, obs, reward, spec  # noqa: E402
from touhou_rl.memory import ProcessMemory  # noqa: E402
from touhou_rl.state import BulletField, GameState, GameStateReader  # noqa: E402


# ---------------------------------------------------------------------------
# spec / action contract
# ---------------------------------------------------------------------------
def test_action_bijection_covers_all_72():
    seen = set()
    for move in range(9):
        for shoot in range(2):
            for focus in range(2):
                for bomb in range(2):
                    flat = spec.encode_flat(move, shoot, focus, bomb)
                    assert 0 <= flat < spec.N_ACTIONS
                    assert spec.decode_flat(flat) == (move, shoot, focus, bomb)
                    seen.add(flat)
    assert len(seen) == 72


def test_decode_flat_rejects_out_of_range():
    with pytest.raises(ValueError):
        spec.decode_flat(72)
    with pytest.raises(ValueError):
        spec.decode_flat(-1)


def test_action_button_mapping_round_trips_for_movement():
    for move in range(9):
        for shoot in (0, 1):
            for focus in (0, 1):
                mask = spec.action_to_buttons((move, shoot, focus, 0))
                decoded = spec.buttons_to_action(mask)
                assert decoded[:3] == (move, shoot, focus)
    # Diagonal + shoot + focus is the common expert input.
    mask = spec.action_to_buttons((2, 1, 1, 0))
    assert mask == spec.BTN_UP | spec.BTN_RIGHT | spec.BTN_SHOOT | spec.BTN_FOCUS
    assert spec.buttons_to_action(mask) == (2, 1, 1, 0)


def test_action_rejects_invalid_move():
    with pytest.raises(ValueError):
        spec.action_to_buttons((9, 0, 0, 0))


# ---------------------------------------------------------------------------
# th08.cfg
# ---------------------------------------------------------------------------
def test_config_default_is_valid_size_and_version(tmp_path: Path):
    cfg = config_file.GameConfig.default()
    assert len(cfg.raw) == config_file.CONFIG_SIZE
    assert cfg.version == config_file.GAME_VERSION
    path = tmp_path / "th08.cfg"
    cfg.save(path)
    loaded = config_file.GameConfig.load(path)
    assert loaded.version == 0x80001
    assert loaded.windowed == 1
    assert loaded.default_difficulty == spec.LUNATIC


def test_config_opt_bits_match_source_layout():
    cfg = config_file.GameConfig.default()
    cfg.set_opt(config_file.OPT_DONT_USE_DIRECTINPUT, True)
    cfg.set_opt(config_file.OPT_DISABLE_VSYNC, True)
    assert cfg.opts == 0x4800  # bit 11 and bit 14
    cfg.set_opt(config_file.OPT_DONT_USE_DIRECTINPUT, False)
    assert cfg.opts == 0x4000
    assert cfg.get_opt(config_file.OPT_DISABLE_VSYNC)
    assert not cfg.get_opt(config_file.OPT_DONT_USE_DIRECTINPUT)


def test_config_field_offsets_match_source_asserts():
    cfg = config_file.GameConfig.default()
    cfg.life_count = 4
    cfg.bomb_count = 2
    cfg.windowed = 0
    cfg.default_difficulty = 1
    assert cfg.raw[0x1C] == 4
    assert cfg.raw[0x1D] == 2
    assert cfg.raw[0x22] == 0
    assert cfg.raw[0x21] == 1
    assert cfg.music_volume == 0


def test_prepare_config_sets_runtime_profile(tmp_path: Path):
    path = tmp_path / "th08.cfg"
    cfg = config_file.prepare_config(
        path, windowed=True, default_difficulty=spec.LUNATIC, use_postmessage_input=False
    )
    assert path.exists()
    assert cfg.version == config_file.GAME_VERSION
    assert cfg.windowed == 1
    assert cfg.default_difficulty == spec.LUNATIC
    assert cfg.music_mode == config_file.MUSIC_OFF
    assert cfg.get_opt(config_file.OPT_DISABLE_VSYNC)


# ---------------------------------------------------------------------------
# state decoding primitives (synthetic buffers, no game process)
# ---------------------------------------------------------------------------
def _synthetic_bullet_field(n: int = 4) -> BulletField:
    raw = np.zeros((spec.BULLET_COUNT, spec.BULLET_SIZE), dtype=np.uint8)
    positions = np.ndarray(
        shape=(spec.BULLET_COUNT, 2),
        dtype=np.float32,
        buffer=raw,
        offset=spec.OFF_BULLET_POSITION,
        strides=(spec.BULLET_SIZE, 4),
    )
    velocities = np.ndarray(
        shape=(spec.BULLET_COUNT, 2),
        dtype=np.float32,
        buffer=raw,
        offset=spec.OFF_BULLET_VELOCITY,
        strides=(spec.BULLET_SIZE, 4),
    )
    states = np.ndarray(
        shape=(spec.BULLET_COUNT,),
        dtype=np.uint16,
        buffer=raw,
        offset=spec.OFF_BULLET_STATE,
        strides=(spec.BULLET_SIZE,),
    )
    disabled = np.ndarray(
        shape=(spec.BULLET_COUNT,),
        dtype=np.int8,
        buffer=raw,
        offset=spec.OFF_BULLET_COLLISION_DISABLED,
        strides=(spec.BULLET_SIZE,),
    )
    half = np.ndarray(
        shape=(spec.BULLET_COUNT, 2),
        dtype=np.float32,
        buffer=raw,
        offset=spec.OFF_BULLET_HURTBOX_X,
        strides=(spec.BULLET_SIZE, 4),
    )
    for i in range(n):
        positions[i] = (100.0 + 10 * i, 200.0 + 5 * i)
        velocities[i] = (0.5, -1.0)
        states[i] = 1
        half[i] = (3.0, 3.0)
    states[n] = spec.BULLET_DESPAWNING  # must be filtered out
    states[n + 1] = 1
    disabled[n + 1] = 1  # collision-disabled must be filtered out
    return BulletField(
        raw=raw,
        position=positions,
        velocity=velocities,
        state=states,
        disabled=disabled,
        half_size=half,
    )


def test_bullet_field_active_mask_filters_unused_and_disabled():
    field = _synthetic_bullet_field(4)
    mask = field.active_mask()
    assert int(mask.sum()) == 4
    assert not mask[4]  # despawning
    assert not mask[5]  # collision disabled


def test_game_state_playing_and_over_flags():
    state = GameState(supervisor_state=spec.STATE_GAME_MANAGER, gameplay_frame=10, lives=2.0)
    assert state.is_playing()
    assert not state.is_run_over()
    demo = GameState(supervisor_state=spec.STATE_GAME_MANAGER, gameplay_frame=10, is_demo=True)
    assert not demo.is_playing()
    dead = GameState(supervisor_state=spec.STATE_GAME_MANAGER, gameplay_frame=10, lives=-1.0)
    assert dead.is_run_over()


# ---------------------------------------------------------------------------
# observation contract
# ---------------------------------------------------------------------------
def _sample_state(**overrides) -> GameState:
    base = GameState(
        supervisor_state=spec.STATE_GAME_MANAGER,
        gameplay_frame=100,
        stage=0,
        player_x=200.0,
        player_y=380.0,
        lives=2.0,
        bombs=3.0,
        power=64.0,
        field_x=32.0,
        field_y=16.0,
        field_w=384.0,
        field_h=448.0,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_observation_shapes_and_ranges():
    state = _sample_state()
    observation = obs.encode(state, _synthetic_bullet_field(4))
    assert observation.self_feat.shape == (obs.N_SELF,)
    assert observation.self_feat.dtype == np.float32
    assert observation.bullets.shape == (obs.N_BULLETS, obs.N_BULLET_FEATURES)
    assert observation.bullets.dtype == np.float16
    assert observation.danger_map.shape == (obs.DANGER_ROWS, obs.DANGER_COLS)
    assert np.isfinite(observation.self_feat).all()
    valid = observation.bullets[:, 5]
    assert int(valid.sum()) == 4
    # Player-relative normalisation: the nearest bullet sits up and to the left.
    first = observation.bullets[0]
    assert float(first[0]) < 0.0
    assert float(first[1]) < 0.0


def test_observation_without_bullets_has_zero_validity():
    observation = obs.encode(_sample_state(), None)
    assert int(observation.bullets[:, 5].sum()) == 0
    assert int(observation.danger_map.sum()) == 0
    # No bullets means all danger rays are "clear" (distance 1.0).
    assert observation.self_feat[-obs.RAY_COUNT :].tolist() == [1.0] * obs.RAY_COUNT


def test_observation_danger_rays_shrink_with_proximity():
    field = _synthetic_bullet_field(1)
    field.position[0] = (200.0, 300.0)  # 80 px directly above the player
    field.velocity[0] = (0.0, 2.0)
    near = obs.encode(_sample_state(), field)
    field.position[0] = (200.0, 260.0)  # 120 px above
    far = obs.encode(_sample_state(), field)
    assert near.self_feat[-obs.RAY_COUNT] < far.self_feat[-obs.RAY_COUNT]


def test_danger_map_counts_bullets():
    field = _synthetic_bullet_field(4)
    for i in range(4):
        field.position[i] = (200.0, 100.0)  # all in the same coarse cell
    state = _sample_state()
    observation = obs.encode(state, field)
    assert int(observation.danger_map.max()) == 4
    assert int(observation.danger_map.sum()) == 4


def test_observation_bullets_sorted_by_distance():
    field = _synthetic_bullet_field(3)
    field.position[0] = (200.0, 370.0)  # 10 px
    field.position[1] = (200.0, 330.0)  # 50 px
    field.position[2] = (200.0, 300.0)  # 80 px
    observation = obs.encode(_sample_state(), field)
    dists = np.abs(observation.bullets[:3, 1].astype(np.float32)) * 224.0
    assert dists[0] < dists[1] < dists[2]


# ---------------------------------------------------------------------------
# reward contract
# ---------------------------------------------------------------------------
def test_reward_first_step_is_zero_and_weights_validated():
    engine = reward.RewardEngine()
    state = _sample_state()
    # Without a baseline sample there is nothing to difference against.
    vector, info = engine.compute(state)
    assert np.allclose(vector, 0.0)
    assert info == {}
    # A repeated identical sample yields only the survival term.
    engine.reset(state)
    vector, _ = engine.compute(state)
    assert vector[0] == pytest.approx(1.0)
    assert vector[2:].tolist() == [0.0, 0.0, 0.0, 0.0]
    bad = reward.RewardWeights(survive=-1.0)
    with pytest.raises(ValueError):
        reward.RewardEngine(weights=bad)


def test_reward_survive_progress_and_death():
    engine = reward.RewardEngine()
    first = _sample_state(score=1000, graze=10, deaths=0)
    engine.reset(first)
    second = _sample_state(score=21000, graze=110, deaths=1, lives=1.0)
    vector, info = engine.compute(second)
    names = reward.COMPONENTS
    comp = dict(zip(names, vector))
    assert comp["survive"] == pytest.approx(1.0)
    assert comp["score"] == pytest.approx(20000 / 1e6)
    assert comp["graze"] == pytest.approx(100 / 100.0)
    assert comp["event"] == pytest.approx(-engine.death_penalty)
    assert info["deaths"] == 1
    assert engine.scalar(vector) == pytest.approx(
        float(np.dot(vector, engine.weights.as_array()))
    )


def test_reward_bomb_usage_and_stage_clear():
    engine = reward.RewardEngine()
    engine.reset(_sample_state(stage=0, bombs_used=0, stage_clear_flag=0))
    vector, info = engine.compute(
        _sample_state(stage=1, bombs_used=1, stage_clear_flag=1, gameplay_frame=200)
    )
    comp = dict(zip(reward.COMPONENTS, vector))
    assert comp["resource"] == pytest.approx(-engine.bomb_penalty)
    assert comp["event"] == pytest.approx(engine.stage_clear_bonus)
    assert info["stage_advanced"]


def test_reward_component_clipping():
    engine = reward.RewardEngine()
    engine.reset(_sample_state(score=0, graze=0))
    vector, _ = engine.compute(_sample_state(score=10**9, graze=10**6))
    comp = dict(zip(reward.COMPONENTS, vector))
    assert comp["score"] == pytest.approx(engine.clip)
    assert comp["graze"] == pytest.approx(engine.clip)


# ---------------------------------------------------------------------------
# dataset shards
# ---------------------------------------------------------------------------
def _random_observation(rng: np.random.Generator) -> obs.Observation:
    return obs.Observation(
        self_feat=rng.standard_normal(obs.N_SELF).astype(np.float32),
        bullets=rng.standard_normal((obs.N_BULLETS, obs.N_BULLET_FEATURES)).astype(np.float16),
        danger_map=rng.integers(0, 12, (obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
    )


def test_shard_writer_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(0)
    writer = dataset.ShardWriter(tmp_path, shard_size=3, tag="unit", provenance={"replay": "x"})
    for i in range(5):
        writer.add(_random_observation(rng), (i % 9, 1, 0, 0), {
            "frame": i, "stage": 0, "score": i * 10, "graze": i, "lives": 2, "bombs": 3, "power": 1, "deaths": 0
        })
    summary = writer.close()
    assert summary["frames"] == 5
    assert len(summary["shards"]) == 2
    assert dataset.count_frames(tmp_path) == 5
    shards = dataset.discover_shards(tmp_path)
    first = dataset.load_shard(shards[0])
    assert first["self_feat"].shape == (3, obs.N_SELF)
    assert first["action"].tolist() == [[0, 1, 0, 0], [1, 1, 0, 0], [2, 1, 0, 0]]
    assert first["meta"][2][0] == 2.0  # frame index round-trips
    sidecar = (tmp_path / "unit-00000.npz.json").read_text(encoding="utf-8")
    assert "replay" in sidecar


def test_replay_dataset_stacked_batch(tmp_path: Path):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    writer = dataset.ShardWriter(tmp_path, shard_size=4, tag="unit")
    for i in range(9):
        writer.add(_random_observation(rng), (0, 0, 0, 0), {
            "frame": i, "stage": 0, "score": 0, "graze": 0, "lives": 2, "bombs": 3, "power": 0, "deaths": 0
        })
    writer.close()
    ds = dataset.ReplayDataset(tmp_path)
    assert len(ds) == 9
    batch = dataset.stacked_batch(ds, [0, 4, 8])
    assert batch["self_feat"].shape == (3, obs.N_SELF)
    assert batch["bullets"].shape == (3, obs.N_BULLETS, obs.N_BULLET_FEATURES)
    assert batch["action"].shape == (3, 4)
    assert batch["action"].dtype == torch.int64
    sample = ds[5]
    assert sample["danger"].shape == (obs.DANGER_ROWS, obs.DANGER_COLS)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def test_model_forward_and_action_shapes():
    torch = pytest.importorskip("torch")
    model = models.DanmakuPolicy()
    batch = {
        "self_feat": torch.zeros(2, obs.N_SELF),
        "bullets": torch.zeros(2, obs.N_BULLETS, obs.N_BULLET_FEATURES),
        "danger": torch.zeros(2, obs.DANGER_ROWS, obs.DANGER_COLS),
    }
    out = model(batch)
    assert out["move_logits"].shape == (2, 9)
    assert out["shoot_logits"].shape == (2, 2)
    assert out["value"].shape == (2,)
    assert out["reward_pred"].shape == (2, 6)
    actions, logp, value, entropy = model.act(batch)
    assert actions.shape == (2, 4)
    assert logp.shape == (2,)
    logp2, entropy2, value2 = model.evaluate_actions(batch, actions)
    assert torch.allclose(logp, logp2, atol=1e-6)
    assert model.num_parameters() > 10_000


def test_model_masked_pooling_ignores_invalid_bullets():
    torch = pytest.importorskip("torch")
    model = models.DanmakuPolicy()
    base = {
        "self_feat": torch.zeros(1, obs.N_SELF),
        "bullets": torch.zeros(1, obs.N_BULLETS, obs.N_BULLET_FEATURES),
        "danger": torch.zeros(1, obs.DANGER_ROWS, obs.DANGER_COLS),
    }
    with torch.no_grad():
        empty = model.encode(base)
        padded = dict(base)
        padded["bullets"] = torch.zeros(1, obs.N_BULLETS, obs.N_BULLET_FEATURES)
        # Insert garbage into slots whose validity flag is 0.
        padded["bullets"][0, :, :5] = 7.5
        padded["bullets"][0, 0, 0] = 1.0
        padded["bullets"][0, 0, 5] = 1.0
        padded["bullets"][0, 1:, 5] = 0.0
        partial = model.encode(padded)
        assert partial.shape == empty.shape


def test_checkpoint_roundtrip(tmp_path: Path):
    torch = pytest.importorskip("torch")
    model = models.DanmakuPolicy(models.NetConfig(trunk_hidden=128))
    path = tmp_path / "ckpt.pt"
    models.save_checkpoint(path, model, {"epoch": 3})
    loaded, extra = models.load_checkpoint(path, map_location="cpu")
    assert extra["epoch"] == 3
    batch = {
        "self_feat": torch.zeros(1, obs.N_SELF),
        "bullets": torch.zeros(1, obs.N_BULLETS, obs.N_BULLET_FEATURES),
        "danger": torch.zeros(1, obs.DANGER_ROWS, obs.DANGER_COLS),
    }
    with torch.no_grad():
        assert torch.allclose(model(batch)["value"], loaded(batch)["value"], atol=1e-5)


# ---------------------------------------------------------------------------
# behaviour cloning helpers
# ---------------------------------------------------------------------------
def test_split_indices_are_disjoint_and_cover_everything():
    from touhou_rl.bc import split_indices

    train, val = split_indices(1000, 0.1, seed=3)
    assert len(train) + len(val) == 1000
    assert len(set(train.tolist()) & set(val.tolist())) == 0


def test_env_config_validation():
    from touhou_rl.env import EnvConfig

    with pytest.raises(ValueError):
        EnvConfig(frameskip=0).validate()
    with pytest.raises(ValueError):
        EnvConfig(reset_mode="teleport").validate()
    with pytest.raises(ValueError):
        EnvConfig(practice_stage=9).validate()
    EnvConfig(practice_stage=3, frameskip=1).validate()


# ---------------------------------------------------------------------------
# replay installation (menu)
# ---------------------------------------------------------------------------
def test_install_replay_replaces_previous_files(tmp_path: Path):
    replay_src = ROOT / "third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/th8_ud2ce5.rpy"
    if not replay_src.exists():
        pytest.skip("replay fixture not available")
    target = menu.install_replay(tmp_path, replay_src)
    assert target.exists()
    assert (tmp_path / "replay" / "th8_01.rpy").exists()
    assert len(list((tmp_path / "replay").glob("th8_*.rpy"))) == 1
    # Installing another replay removes the first one.
    menu.install_replay(tmp_path, replay_src, index=2)
    assert not (tmp_path / "replay" / "th8_01.rpy").exists()
    assert (tmp_path / "replay" / "th8_02.rpy").exists()


# ---------------------------------------------------------------------------
# Win32 / memory primitives
# ---------------------------------------------------------------------------
def test_read_own_process_memory():
    pytest.importorskip("touhou_rl.memory")
    data = ctypes.create_string_buffer(b"touhou-rl-probe")
    address = ctypes.addressof(data)
    mem = ProcessMemory(os.getpid())
    try:
        assert mem.read(address, 16) == b"touhou-rl-probe\x00"
        assert mem.read(address, 6) == b"touhou"
        with pytest.raises(OSError):
            mem.read(0x1, 16)  # unmapped address must raise, never silently zero-fill
    finally:
        mem.close()


def test_find_window_returns_zero_for_unknown_pid():
    from touhou_rl import winapi

    assert winapi.find_window(0x7FFFFFF0) == 0
    assert winapi.find_dialog(0x7FFFFFF0) == 0


def test_list_process_ids_contains_self_python():
    from touhou_rl import winapi

    name = os.path.basename(sys.executable)
    assert os.getpid() in winapi.list_process_ids(name)
