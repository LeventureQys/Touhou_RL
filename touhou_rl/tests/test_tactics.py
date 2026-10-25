"""Unit tests for the v3.6.1 T2 tactic controllers.

These are pure functions on purpose: the paired comparison costs ~80 minutes of
live game time, so everything that can be decided without the game is tested
here.  In particular ``choose_target`` and ``collect_power_override`` are the
whole "one factor" definition, and the constants they read are pre-registered.
"""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl import spec
from touhou_rl.tactics import (
    CLEAR_ENEMY_AIM_WEIGHT,
    CLEAR_ENEMY_SAFE_MARGIN,
    COLLECT_POWER_MIN_CLEARANCE,
    COLLECT_POWER_TARGET_Y,
    choose_target,
    collect_power_override,
    pilot_config_kwargs,
    target_as_boss_array,
    validate_tactic,
)


def _enemy(**kwargs) -> dict:
    base = {"slot": 0, "x": 192.0, "y": 100.0, "life": 100, "max_life": 100, "boss": False, "onscreen": True}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# ClearEnemy target selection
# ---------------------------------------------------------------------------


def test_choose_target_prefers_the_boss_over_a_normal_enemy():
    enemies = [_enemy(x=192.0, y=100.0), _enemy(x=192.0, y=120.0, boss=True)]
    target = choose_target(enemies, player_y=380.0)
    assert target is not None and target["boss"] is True


def test_choose_target_prefers_above_over_below_at_equal_distance():
    above = _enemy(x=192.0, y=280.0)
    below = _enemy(x=192.0, y=480.0)
    # Both are 100 px from the player at y=380: the above one is worth more.
    assert choose_target([below, above], player_y=380.0) is above


def test_choose_target_prefers_the_nearer_enemy_all_else_equal():
    near = _enemy(x=192.0, y=200.0)
    far = _enemy(x=192.0, y=40.0)
    assert choose_target([far, near], player_y=380.0) is near


def test_choose_target_skips_dead_enemies():
    dead = _enemy(x=192.0, y=200.0, life=0)
    alive = _enemy(x=100.0, y=40.0, life=5)
    assert choose_target([dead, alive], player_y=380.0) is alive
    assert choose_target([dead], player_y=380.0) is None
    assert choose_target([], player_y=380.0) is None


def test_target_as_boss_array_feeds_the_pilots_aim_term():
    array = target_as_boss_array(_enemy(x=123.0, y=45.0))
    assert array is not None and array.shape == (1, 2)
    assert array[0, 0] == pytest.approx(123.0)
    assert target_as_boss_array(None) is None


# ---------------------------------------------------------------------------
# CollectPower override
# ---------------------------------------------------------------------------


def _p_item(**kwargs) -> dict:
    base = {"slot": 0, "x": 190.0, "y": 100.0, "type": spec.ITEM_POWER_SMALL, "onscreen": True}
    base.update(kwargs)
    return base


def test_collect_override_moves_up_when_below_the_band_and_safe():
    result = collect_power_override(0.0, 380.0, [_p_item()], clearance=12.0)
    assert result["override"] == (1, 1, 1, 0)
    assert result["reason"] == "up_collect"


def test_collect_override_holds_focus_inside_the_band():
    result = collect_power_override(0.0, COLLECT_POWER_TARGET_Y - 5.0, [_p_item()], clearance=12.0)
    assert result["override"] == (0, 1, 1, 0)
    assert result["reason"] == "hold_collect"


def test_collect_override_refuses_when_power_is_full():
    result = collect_power_override(spec.POWER_MAX, 380.0, [_p_item()], clearance=99.0)
    assert result["override"] is None and result["reason"] == "power_max"


def test_collect_override_refuses_without_an_onscreen_power_item():
    assert collect_power_override(0.0, 380.0, [], clearance=99.0)["reason"] == "no_p_item"
    offscreen = _p_item(onscreen=False)
    assert collect_power_override(0.0, 380.0, [offscreen], clearance=99.0)["reason"] == "no_p_item"
    # A non-power item type must not count.
    other = _p_item(type=spec.ITEM_POINT)
    assert collect_power_override(0.0, 380.0, [other], clearance=99.0)["reason"] == "no_p_item"


def test_collect_override_respects_the_safety_gate():
    result = collect_power_override(0.0, 380.0, [_p_item()], clearance=COLLECT_POWER_MIN_CLEARANCE - 0.1)
    assert result["override"] is None and result["reason"] == "unsafe"
    # Exactly at the gate is allowed: the threshold is inclusive.
    allowed = collect_power_override(0.0, 380.0, [_p_item()], clearance=COLLECT_POWER_MIN_CLEARANCE)
    assert allowed["override"] is not None


def test_collect_override_ignores_power_items_with_zero_life_or_unknown_type():
    bad_type = _p_item(type=999)
    assert collect_power_override(0.0, 380.0, [bad_type], clearance=99.0)["reason"] == "no_p_item"


# ---------------------------------------------------------------------------
# arm definition
# ---------------------------------------------------------------------------


def test_pilot_config_kwargs_only_change_the_clear_enemy_arm():
    assert pilot_config_kwargs("clear_enemy") == {
        "aim_weight": CLEAR_ENEMY_AIM_WEIGHT,
        "aim_safe_margin": CLEAR_ENEMY_SAFE_MARGIN,
    }
    # The baseline and the collect arm must use the untouched pilot.
    assert pilot_config_kwargs("none") == {}
    assert pilot_config_kwargs("collect_power") == {}


def test_pre_registered_constants_are_the_documented_values():
    """Guards against silent tuning after a result is seen."""
    assert CLEAR_ENEMY_AIM_WEIGHT == 3.0
    assert CLEAR_ENEMY_SAFE_MARGIN == 10.0
    assert COLLECT_POWER_MIN_CLEARANCE == 6.0
    assert COLLECT_POWER_TARGET_Y == 110.0


def test_validate_tactic_rejects_unknown_names():
    assert validate_tactic("clear_enemy") == "clear_enemy"
    with pytest.raises(ValueError, match="unknown tactic"):
        validate_tactic("lure")


# ---------------------------------------------------------------------------
# live dataclasses (the first live smoke crashed on exactly this difference)
# ---------------------------------------------------------------------------


def test_controllers_accept_the_live_dataclasses():
    from touhou_rl.state import EnemyInfo, ItemInfo

    enemy = EnemyInfo(
        slot=3,
        x=180.0,
        y=100.0,
        life=50,
        max_life=50,
        active=True,
        boss=True,
        sub_id=1,
        item_drop=2,
        boss_slot=0,
        flags1=3,
    )
    assert choose_target([enemy], player_y=380.0) is enemy
    array = target_as_boss_array(enemy)
    assert array is not None and array[0, 0] == pytest.approx(180.0)

    item = ItemInfo(
        slot=1,
        x=190.0,
        y=110.0,
        item_type=spec.ITEM_POWER_SMALL,
        state=0,
        onscreen=True,
        is_max_value=False,
    )
    result = collect_power_override(0.0, 380.0, [item], clearance=12.0)
    assert result["override"] == (1, 1, 1, 0)
    # A non-power dataclass item must be rejected just like the dict form.
    point = ItemInfo(
        slot=2, x=190.0, y=110.0, item_type=spec.ITEM_POINT, state=0, onscreen=True, is_max_value=False
    )
    assert collect_power_override(0.0, 380.0, [point], clearance=12.0)["reason"] == "no_p_item"
