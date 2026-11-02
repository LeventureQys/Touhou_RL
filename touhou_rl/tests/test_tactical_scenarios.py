import pytest
from touhou_rl.tactical_scenarios import (development_scenarios, initial_state, advance,
    replay, fingerprint, Scenario)


def test_six_families_replay_and_group_isolation():
    scenarios = development_scenarios()
    assert len(scenarios) == 72
    assert {s.family for s in scenarios} == {f"F{i}" for i in range(1,7)}
    assert len({fingerprint(s) for s in scenarios}) == 72
    for s in scenarios:
        a = (0,)*(s.horizon//2)
        assert replay(s, a) == replay(s, a)
        assert s.split == "development"
        assert s.template_group.startswith("dev-")


@pytest.mark.parametrize("family", ["F2", "F3"])
def test_aimed_birth_responds_to_executed_history(family):
    s = next(s for s in development_scenarios() if s.family == family)
    still, moved = initial_state(s), initial_state(s)
    for _ in range(9):
        still = advance(s, still, 0)
        moved = advance(s, moved, -1)
    # First emission same origin/history; SECOND emission genuinely action dependent.
    assert still.emissions[0] == moved.emissions[0]
    assert still.emissions[1]["aim_x"] != moved.emissions[1]["aim_x"]
    assert still.emissions[1]["vx"] != moved.emissions[1]["vx"]
    # Existing bullet direction remains fixed rather than homing after birth.
    assert still.bullets[0] == moved.bullets[0]


def test_rest_under_f1_wall_and_internal_gap_counterexample():
    worlds = development_scenarios()
    wall = next(s for s in worlds if s.family == "F1" and not s.internal_gap)
    gap = next(s for s in worlds if s.family == "F1" and s.internal_gap and s.start_x == 192)
    assert replay(wall, (0,)*80)[0].hit
    assert not replay(gap, (0,)*80)[0].hit


def test_invalid_actions_rejected():
    s = development_scenarios()[0]
    with pytest.raises(ValueError):
        advance(s, initial_state(s), 99)
