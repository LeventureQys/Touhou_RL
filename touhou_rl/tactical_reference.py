"""Independent bounded open-loop program search; NEVER certifies infeasibility.

No import of tactical_teacher. Enumerates wait/travel/stop and streaming programs
from a different policy class, then replays the full causal environment. Positive
witnesses certify only this finite synthetic horizon, not a viability kernel.
"""
from dataclasses import dataclass
from .tactical_scenarios import replay, initial_state, advance

@dataclass(frozen=True)
class ReferenceResult:
    status: str
    actions: tuple[int, ...] = ()
    checked: int = 0
    reason: str = ""


def verify_route(scenario, actions):
    if not scenario.model_known:
        return ReferenceResult("unknown", reason="unsupported_future_emitter_model")
    state, trace = replay(scenario, actions)
    if state.frame != scenario.horizon:
        return ReferenceResult("unknown", reason="incomplete_horizon")
    if state.hit:
        return ReferenceResult("rejected_witness", reason="replayed_collision_not_infeasibility_proof")
    if scenario.family == "F3" and not any(scenario.canonical_x(t["x"]) <= 160 for t in trace):
        return ReferenceResult("unknown", reason="survival_without_required_restream_goal")
    return ReferenceResult("feasible", tuple(actions), reason="full_causal_replay_positive_witness")


def _programs(s):
    n = (s.horizon+1)//2
    yield (0,)*n
    # Enumerated schedules do not read teacher output or its option logic.
    for direction in (-2, -1, 1, 2):
        for start in (0, 8, 16, 24, 32, 40, 48, 64, 80, 96):
            for duration in (8, 16, 32, 64, 96, 160):
                yield tuple(direction if start <= 2*k < start+duration else 0 for k in range(n))
    for direction in (-1, 1):
        for switch in (32, 40, 48, 64, 80, 96):
            yield tuple(direction if (2*k//16)%2 == 0 else 0 for k in range(n))
            yield tuple((2*direction if switch <= 2*k < switch+64 else
                         -direction if 2*k % s.fire_interval < 4 else 0) for k in range(n))


def solve(scenario, max_programs=300, wait_frames=0):
    if max_programs < 0 or wait_frames < 0 or wait_frames % 2:
        raise ValueError("budget nonnegative and delay must be nonnegative/even")
    if not scenario.model_known:
        return ReferenceResult("unknown", reason="unsupported_future_emitter_model")
    checked = 0
    for program in _programs(scenario):
        if checked >= max_programs:
            break
        prefix = (0,) * (wait_frames//2)
        actions = (prefix+program)[:len(program)]
        checked += 1
        result = verify_route(scenario, actions)
        if result.status == "feasible":
            return ReferenceResult("feasible", actions, checked, result.reason)
    return ReferenceResult("unknown", checked=checked, reason="bounded_program_search_no_witness")


def scan_start_windows(scenario, delays=range(0, 66, 2), max_programs=300):
    """Nonmonotone full scan, unknown entries do not define a latest deadline."""
    return [{"delay": delay, "status": (r := solve(scenario, max_programs, delay)).status,
             "reason": r.reason, "checked": r.checked} for delay in delays]


def trajectory_metrics(s, actions):
    """Independent event measurements; geometrical pollution is diagnostic only.

    At birth, project the fixed ray onto player y. Count rays aiming into the
    pre-reserved opposite-side corridor. Not an all-time hazard-occupancy proof.
    """
    state, trace = replay(s, actions)
    aims = [s.canonical_x(e["aim_x"]) for e in state.emissions]
    corridor = (40, 160) if s.family == "F3" else (80, 304)
    pollution = sum(corridor[0] <= x <= corridor[1] for x in aims)
    canonical = [s.canonical_x(t["x"]) for t in trace]
    occupancy = 0
    for frame in range(1, s.horizon+1):
        if s.family == "F3" and frame >= 104:
            continue
        for b in state.bullets:
            age = frame-b.born
            if age < 1:
                continue
            bx, by = s.canonical_x(b.x+b.vx*age), b.y+b.vy*age
            if 77 <= bx <= 163 and 357 <= by <= 423:
                occupancy += 1
    return {"survived": not state.hit and state.frame == s.horizon,
            "corridor_occupancy_bullet_frames": occupancy,
            "corridor_ray_count": pollution, "emissions": len(aims),
            "aim_span_px": max(aims)-min(aims) if aims else 0.,
            "final_horizontal_space_px": min(state.x-8, 376-state.x),
            "restreamed": s.family == "F3" and min(canonical) <= 160,
            "trace": trace, "emission_events": list(state.emissions)}
