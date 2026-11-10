"""Closed-loop S1N evaluation of an obs-v1 student checkpoint.

Runs real engine episodes under ``xvfb-run`` inside WSL with the ``env`` role and
pixel readback, so the policy sees only ``play64`` + ``hud`` planes and never
privileged state.  Bookkeeping (frame cap, game-over terminal rule, death/lives
sampling) mirrors ``tools/rl/v400_bench_run.py`` so the numbers sit on the same
ruler as the ``random``/``still`` rows in ``bench/scoreboard.md``.

This is **not** a gate evaluation: five dev seeds are a smoke test, and the
checkpoint under test is a behaviour-cloned student (V0: the collection
policies; BC v1: the teacher's soft labels).

Stage numbering: the harness ``RESET`` stage argument and ``STATE.f1`` are
**0-based** — ``stage 0`` is the first stage (``Stage1`` in
``bench/human_reference.json``, ``stage_index: 0``).  ``S1N`` is therefore
``RESET stage 0``, normal, cap 19887 (= ceil(1.5 x 13258 measured human
``Stage1`` frames)), which is exactly how EXP-P1-012 (G1) and EXP-P1-018 (D0)
ran it.  ``--stage 1`` starts the real Stage 2.

Run inside WSL::

    xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
      tools/rl/v400_student_eval.py \
      --engine /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \
      --data-dir /root/Touhou_RL/third_party/th08-rl/data \
      --checkpoint .tmp_v400_p3v1/checkpoint-v1.pt \
      --seeds 1000-1019 --max-frames 19887 \
      --stage 0 --difficulty normal --stop-on-death \
      --out-dir .tmp_v400_p3v1/eval

EXP-P3-009 (evaluation power) changed two things about this ruler and nothing
about the engine:

1. ``--seeds`` accepts ranges (``1000-1019``) and lists (``1000,1001-1005``), so a
   row can be measured on more than the historical five dev seeds.  ``--repeats``
   runs every seed ``N`` times and labels each episode with ``repeat``; the S1N
   runner is deterministic per (checkpoint, seed) — EXP-P3-008 §7.4 verified a
   repeat reproduced the first five episodes frame for frame — so repeats exist to
   *prove* that on a new checkpoint, not to buy precision.
2. The summary now carries ``frames_stats`` (n / sd / se / bootstrap CI) and
   ``per_seed`` rows.  A five-episode mean with no interval is no longer a
   reportable result: ``tools/rl/v400_eval_power.py`` refuses to call a difference
   an effect unless both arms come with a CI and those CIs are disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from touhou_rl.v400.gym import actions as gym_actions  # noqa: E402
from touhou_rl.v400.gym.protocol import HarnessClient  # noqa: E402
from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID, SharedBlock  # noqa: E402
from touhou_rl.v400.labels.state import state_from_reply  # noqa: E402
from touhou_rl.v400.student import obs_spec  # noqa: E402
from touhou_rl.v400.student import obs_spec_v2  # noqa: E402
from touhou_rl.v400.student.model import checkpoint_digest, load_checkpoint  # noqa: E402
from touhou_rl.v400.student.model_v2 import (  # noqa: E402
    ARCH_GRU,
    GRUCache,
    load_checkpoint_v2,
)

EPISODE_SCHEMA = "v400-student-eval-episode-1"
SUMMARY_SCHEMA = "v400-student-eval-summary-1"
#: ``RESET`` stage argument; 0-based, so ``0`` is the first stage.  ``S1N`` is
#: ``stage 0`` normal (``bench/human_reference.json``: ``stage_index 0`` is
#: ``Stage1``, 13258 gameplay frames; EXP-P1-012 + EXP-P1-018 both ran S1N as
#: ``--stage 0``).  ``1`` would start the real Stage 2 under the Stage-1 cap.
DEFAULT_STAGE = 0
DEFAULT_DIFFICULTY = 1     # "normal"
DIFFICULTIES = ("easy", "normal", "hard", "lunatic")
DECISION_FRAMES = obs_spec.DECISION_FRAMES
#: EXP-P3-004 §4.  ``consecutive`` = one render per game frame (the 4-frame stack
#: is frames[t-3..t], i.e. the training window); ``strided`` = the legacy one
#: render per decision (the stack spans 4 * DECISION_FRAMES = 8 frames).
OBS_WINDOWS = ("consecutive", "strided")

#: Same three dev seeds, same cap class, from ``bench/scoreboard.md``
#: (cap_source = human_reference, max_frames 19887).  Only these three seeds are
#: reproduced, so the comparison is per-seed, not against the 20-seed means.
REFERENCE_ROWS = {
    "random": {
        1000: {"frames": 2586, "deaths": 3.0, "score": 41193},
        1001: {"frames": 1790, "deaths": 3.0, "score": 26031},
        1002: {"frames": 2004, "deaths": 3.0, "score": 34615},
    },
    "still": {
        1000: {"frames": 1748, "deaths": 3.0, "score": 59953},
        1001: {"frames": 1692, "deaths": 3.0, "score": 63693},
        1002: {"frames": 1748, "deaths": 3.0, "score": 63762},
    },
}
REFERENCE_DEV20 = {
    "random": {"episodes": 20, "deaths": 3.0, "score": 39331.2},
    "still": {"episodes": 20, "deaths": 3.0, "score": 64089.2},
}
#: Historical *student* rows, kept verbatim so a comparison cannot drift.
REFERENCE_STUDENT_ROWS = {
    "bc_v0_stage1": {
        "what": "BC v0 (behaviour cloning of random/still/dive), 3 dev seeds",
        "seeds": [1000, 1001, 1002],
        "mean_frames": 1634.0,
        "mean_deaths": 3.0,
        "clears": 0,
        "stage_argument_used": 1,
        "stage_warning": (
            "that run passed --stage 1, which starts the real Stage 2 under the "
            "Stage-1 cap, so it is NOT an S1N row and not per-seed comparable with "
            "a --stage 0 row"),
        "source": ".tmp_v400_p3/eval/summary.json (EXP-P3-001)",
    },
    "teacher_s1n": {
        "what": "P3 teacher plan argmax, S1N = --stage 0 normal, cap 19887",
        "seeds": "1000-1019",
        "seed_count": 20,
        "mean_frames": 13565.2,
        "min_frames": 12812,
        "max_frames": 15204,
        "mean_deaths": 0.0,
        "clears": "20/20",
        "source": "Document/Update/Alpha/v4.0.0/experiments/EXP-P1-012-g1-evaluation.md",
    },
}


class EvalError(RuntimeError):
    """An episode or its record violated the evaluation contract."""


class EnvHarnessClient(HarnessClient):
    """HarnessClient that starts the engine in ``env`` role (pixels + state)."""

    def start(self) -> None:  # noqa: D401 - same contract as the parent
        if self._proc is not None:
            raise RuntimeError("client already started")
        import subprocess

        argv = [str(self.binary), "--data-dir", str(self.data_dir),
                "--rl-control", "0,1", "--rl-role", "env"]
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            bufsize=0,
        )
        if os.name == "nt":
            self._start_pipe_reader()


def _readings(state: dict) -> dict[str, float]:
    return {
        "deaths": float(state["deaths"]),
        "lives": float(state["lives_remaining"]),
        "bombs": float(state["bombs_remaining"]),
        "score": int(state["score"]),
    }


def decision_steps(obs_window: str, budget: int) -> tuple[int, int]:
    """``(STEP units, frames per STEP)`` for one decision under this window.

    EXP-P3-004 §4.  Both windows spend the **same** number of game frames on a
    decision (``units * frames == budget``), so the frame ruler is unchanged;
    they differ only in how many frames get rendered, which is what the 4-frame
    observation stack is built from:

    * ``consecutive`` -> ``(budget, 1)``: one STEP per frame, one render per
      frame -> the stack is the last FRAME_STACK *consecutive* frames, i.e. the
      training window ``frames[t-3..t]``;
    * ``strided`` -> ``(1, budget)``: one STEP of ``budget`` frames and a single
      render -> the FRAME_STACK pushes are ``budget`` frames apart, so the stack
      spans ``FRAME_STACK * budget`` frames.
    """

    if obs_window not in OBS_WINDOWS:
        raise EvalError(f"obs_window {obs_window!r} not in {sorted(OBS_WINDOWS)}")
    if int(budget) <= 0:
        raise EvalError(f"budget must be >= 1 game frame, got {budget!r}")
    if obs_window == "consecutive":
        return int(budget), 1
    return 1, int(budget)


REQUIRED_EPISODE_FIELDS = {
    "schema": str,
    "checkpoint": str,
    "checkpoint_sha256": str,
    "obs_spec_version": str,
    "obs_spec_digest": str,
    "seed": int,
    "suite": str,
    "stage": int,
    "difficulty": str,
    "max_frames": int,
    "frames": int,
    "deaths": float,
    "score": int,
    "end": str,
    "privilege": str,
    "wall_s": float,
    "engine_sha256": str,
    "policy_input": str,
    "not_a_gate": bool,
}
VALID_ENDS = {"game_over", "death_stop", "stage_clear", "frame_cap", "engine_exit",
              "reset_failed", "error"}
#: Optional fields this controller always writes and the validator type-checks.
#: Keeping them optional means an older record (EXP-P3-001, written before the
#: stage/clear bookkeeping existed) is still readable, while a *present* field of
#: the wrong type or value is rejected.
OPTIONAL_EPISODE_FIELDS = {
    "stage_start": int,
    "stage_end": int,
    "cleared": bool,
    "exit": str,
    "stop_on_death": bool,
    "obs_window": str,
    #: EXP-P3-009: which repeat of this (checkpoint, seed) pair the episode is.
    "repeat": int,
    "renders": int,
    "window_alignment": dict,
    #: EXP-P3-008: what the policy actually emitted, so a collapsed arm is visible in
    #: the closed-loop record instead of only in the offline metrics.
    "action_histogram": list,
    "action0_share": float,
    "focus_classes_covered": int,
}
VALID_OBS_WINDOWS = set(OBS_WINDOWS)
VALID_EXITS = {"engine_exit", "game_over", "controller", "none"}


def validate_episode_record(record: dict, *, expected_spec: tuple[str, str] | None = None) -> None:
    """Fail closed on a malformed or privilege-dirty evaluation record.

    ``expected_spec`` is the ``(version, digest)`` of the checkpoint's observation
    contract (obs-v1 for the legacy arms, obs-v2 for EXP-P3-008).  Both are frozen,
    so an episode that observed either one is checkable; what is refused is observing
    a spec other than the checkpoint's.
    """

    missing = [key for key in REQUIRED_EPISODE_FIELDS if key not in record]
    if missing:
        raise EvalError(f"episode record missing fields: {missing}")
    for key, expected in REQUIRED_EPISODE_FIELDS.items():
        value = record[key]
        if expected is float:
            if not isinstance(value, (int, float)):
                raise EvalError(f"episode field {key} is not numeric: {value!r}")
        elif not isinstance(value, expected) or isinstance(value, bool) and expected is int:
            raise EvalError(f"episode field {key} is not {expected.__name__}: {value!r}")
    for key, expected in OPTIONAL_EPISODE_FIELDS.items():
        if key not in record:
            continue
        value = record[key]
        if expected is list and not isinstance(value, list):
            raise EvalError(f"episode field {key} is not a list: {value!r}")
        if expected is bool and not isinstance(value, bool):
            raise EvalError(f"episode field {key} is not bool: {value!r}")
        if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
            raise EvalError(f"episode field {key} is not int: {value!r}")
        if expected is str and not isinstance(value, str):
            raise EvalError(f"episode field {key} is not str: {value!r}")
        if expected is float and not isinstance(value, (int, float)):
            raise EvalError(f"episode field {key} is not numeric: {value!r}")
    if "exit" in record and record["exit"] not in VALID_EXITS:
        raise EvalError(f"episode exit {record['exit']!r} not in {sorted(VALID_EXITS)}")
    if "obs_window" in record:
        if record["obs_window"] not in VALID_OBS_WINDOWS:
            raise EvalError(
                f"episode obs_window {record['obs_window']!r} not in "
                f"{sorted(VALID_OBS_WINDOWS)}")
        alignment = record.get("window_alignment") or {}
        expected_renders = (DECISION_FRAMES if record["obs_window"] == "consecutive" else 1)
        if int(alignment.get("renders_per_decision", -1)) != expected_renders:
            raise EvalError(
                f"episode obs_window {record['obs_window']} must render "
                f"{expected_renders}x per decision, window_alignment says "
                f"{alignment.get('renders_per_decision')!r}")
    if "cleared" in record and "stage_start" in record and "stage_end" in record:
        if record["cleared"] != (record["stage_end"] > record["stage_start"]):
            raise EvalError(
                "episode cleared flag disagrees with the stage counter: "
                f"start={record['stage_start']} end={record['stage_end']} "
                f"cleared={record['cleared']}")
    if record["schema"] != EPISODE_SCHEMA:
        raise EvalError(f"episode schema {record['schema']!r} != {EPISODE_SCHEMA!r}")
    if record["end"] not in VALID_ENDS:
        raise EvalError(f"episode end {record['end']!r} not in {sorted(VALID_ENDS)}")
    if record["privilege"] != "deploy_pixel":
        raise EvalError(f"episode privilege {record['privilege']!r} is not deploy_pixel")
    version, digest = expected_spec or (obs_spec.OBS_SPEC_VERSION, obs_spec.obs_spec_digest())
    if record["obs_spec_version"] != version:
        raise EvalError(
            f"episode observed obs spec {record['obs_spec_version']!r}, the checkpoint "
            f"is {version!r}")
    if record["obs_spec_digest"] != digest:
        raise EvalError("episode observed a different obs spec digest")
    if record["frames"] < 0 or record["max_frames"] <= 0:
        raise EvalError("episode frame counters are invalid")
    if record["not_a_gate"] is not True:
        raise EvalError("episode must be labelled not_a_gate=True")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_episode(
    *,
    engine: str,
    data_dir: str,
    checkpoint_path: str,
    model,
    seed: int,
    stage: int,
    difficulty: str,
    max_frames: int,
    shm_name: str,
    stop_on_death: bool = False,
    timeout: float = 240.0,
    obs_window: str = "consecutive",
    obs_v2: bool = False,
    arch: str = "mlp",
    repeat: int = 0,
) -> dict:
    """One real S1N episode; returns a validated record.

    ``stop_on_death`` ends the episode on the **first** death (G1's rule) instead
    of playing on until the spare lives are gone, so a row measures "time to
    first death" rather than the saturated ``lives_start + 1`` deaths every
    game-over row reports.  A stage advance also ends the episode
    (``end="stage_clear"``), which is how EXP-P1-012 measured S1N: continuing
    into Stage 2 would report Stage-2 time under a Stage-1 cap.

    ``obs_window`` (EXP-P3-004 §4) selects *how many frames each decision
    renders*: ``consecutive`` steps the DECISION_FRAMES frames one at a time and
    pushes each one, so the stack is 4 consecutive frames (what the D0/DAgger
    shards stored); ``strided`` is the legacy single render per decision, whose
    stack spans 8 frames.  The frame budget per decision is identical either
    way, so the two modes share the frame ruler.
    """

    import torch

    if obs_window not in OBS_WINDOWS:
        raise EvalError(f"obs_window {obs_window!r} not in {sorted(OBS_WINDOWS)}")

    started = time.monotonic()
    spec_version = (obs_spec_v2.OBS_SPEC_VERSION if obs_v2
                    else obs_spec.OBS_SPEC_VERSION)
    spec_digest = (obs_spec_v2.obs_spec_v2_digest() if obs_v2
                   else obs_spec.obs_spec_digest())
    identity = {
        "schema": EPISODE_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        "obs_spec_version": spec_version,
        "obs_spec_digest": spec_digest,
        "seed": int(seed),
        "repeat": int(repeat),
        "suite": "S1N",
        "stage": int(stage),
        "difficulty": difficulty,
        "max_frames": int(max_frames),
        "decision_frames": DECISION_FRAMES,
        "obs_window": obs_window,
        "window_alignment": {
            "mode": obs_window,
            "frames_per_decision": DECISION_FRAMES,
            "renders_per_decision": DECISION_FRAMES if obs_window == "consecutive" else 1,
            "stack_span_frames": (obs_spec.FRAME_STACK if obs_window == "consecutive"
                                  else obs_spec.FRAME_STACK * DECISION_FRAMES),
            "matches_training_obs": obs_window == "consecutive",
            "training_window": (
                "DecisionObservationBuffer.stack_for(4k) = frames[4k-4..4k-1]: four "
                "consecutive game frames, one decision = four frames"),
        },
        "stage_argument_semantics": "RESET stage is 0-based; S1N = stage 0 (Stage1)",
        "stop_on_death": bool(stop_on_death),
        "privilege": "deploy_pixel",
        "policy_input": ("play64 4-frame stack + hud strip + act_hist (8x18 one-hot of "
                         "the policies own last 8 decisions); no privileged state"
                         if obs_v2 else
                         "play64 4-frame stack + hud strip; no privileged state"),
        "referee_state_source": "engine STATE reply after each decision",
        "engine": str(engine),
        "engine_sha256": _sha256(engine),
        "not_a_gate": True,
        "not_the_confirm_split": True,
        "controller": "tools/rl/v400_student_eval.py",
    }
    client = None
    shm = None
    frames = 0
    steps = 0
    renders = 0
    action_counts = np.zeros(obs_spec.N_MOVE_CLASSES, dtype=np.int64)
    zero_advances = 0
    end = "error"
    error = None
    start = None
    previous = None
    end_state = None
    pixels_warmup_frames = 0
    hello_role = None
    stage_start = None
    exit_reason = "none"
    try:
        client = EnvHarnessClient(engine, data_dir, shm_name=shm_name, timeout=timeout)
        client.start()
        hello = client.hello()
        hello_role = hello.get("role")
        client.attach_shm()
        client.readback(True)
        shm = SharedBlock(client.shm_name or shm_name)
        reply = client.reset(stage=stage, difficulty=DIFFICULTIES.index(difficulty),
                             team=0, seed=seed)
        if reply.get("ready") != "1":
            raise EvalError(f"RESET did not reach a playable state: {reply}")
        state = state_from_reply(client.request("STATE"))
        start = _readings(state)
        previous = start
        stage_start = int(state["stage"])
        if not (shm.header()["flags"] & FLAG_PIXELS_VALID):
            # The collector needed one step before the first valid readback; do the
            # same and record how many frames it cost, never hide them.
            warm = client.step(gym_actions.cand_to_mask(0, 0), 1)
            pixels_warmup_frames += int(warm.get("advanced", 0))
            frames += pixels_warmup_frames
            state = state_from_reply(client.request("STATE"))
            previous = _readings(state)
            stage_start = int(state["stage"])
        buffer = obs_spec.ObservationBuffer()
        history = obs_spec_v2.ActionHistoryBuffer() if obs_v2 else None
        gru_cache = (GRUCache(model.seq_len) if (obs_v2 and arch == ARCH_GRU) else None)
        play, hud = obs_spec.encode_planes(shm.pixels())
        buffer.push(play, hud)
        renders = 1

        while frames < max_frames:
            arrays = buffer.arrays()
            play_t = torch.from_numpy(arrays["play"][None]).float()
            hud_t = torch.from_numpy(arrays["hud"][None]).float()
            if obs_v2:
                # (B,S,18): the model takes the 8x18 window, not the flattened 144
                hist_t = torch.from_numpy(history.rows()[None]).float()
                if gru_cache is not None:
                    cand, bomb = model.act_step(play_t, hud_t, hist_t, gru_cache)
                else:
                    cand, bomb = model.act(play_t, hud_t, hist_t)
            else:
                cand, bomb = model.act(play_t, hud_t)
            mask = gym_actions.cand_to_mask(cand, bomb)
            action_counts[int(cand)] += 1
            budget = min(DECISION_FRAMES, max_frames - frames)
            units, want = decision_steps(obs_window, budget)
            steps += 1
            ended = None
            for _unit in range(units):
                step_reply = client.step(mask, want)
                advanced = int(step_reply.get("advanced", 0))
                frames += advanced
                if advanced <= 0:
                    zero_advances += 1
                    if zero_advances >= 3 and step_reply.get("exit") != "1":
                        raise EvalError(f"STEP advanced no frames: {step_reply}")
                else:
                    zero_advances = 0
                if step_reply.get("exit") == "1":
                    if step_reply.get("game_over") == "1":
                        ended = ("game_over", "game_over")
                    else:
                        ended = ("engine_exit", "engine_exit")
                    break
                state = state_from_reply(client.request("STATE"))
                current = _readings(state)
                if int(state["stage"]) > stage_start:
                    # The stage counter advanced: Stage 1 was cleared.  Stop here,
                    # as EXP-P1-012 did, so `frames` stays Stage-1 time.
                    ended = ("stage_clear", "controller")
                    break
                if current["deaths"] > previous["deaths"]:
                    if current["lives"] <= 0:
                        ended = ("game_over", "game_over")
                        break
                    if stop_on_death:
                        ended = ("death_stop", "controller")
                        break
                previous = current
                play, hud = obs_spec.encode_planes(shm.pixels())
                buffer.push(play, hud)
                renders += 1
            if obs_v2:
                # The history at decision t must hold a[0..t-1]; push only after the
                # action has been applied, so the current decision can never leak in.
                history.push(cand)
            if ended is not None:
                end, exit_reason = ended
                break
        else:
            end = "frame_cap"
            exit_reason = "controller"
        try:
            end_state = state_from_reply(client.request("STATE"))
        except Exception:  # noqa: BLE001 - keep the last real reading, say so
            end_state = state
        identity["final_state_source"] = (
            "engine STATE reply after the final STEP" if end_state is not state
            else "last successful per-step STATE reply")
    except BaseException as exc:  # noqa: BLE001 - a failed episode is a recorded episode
        error = f"{type(exc).__name__}: {exc}"
        end = "reset_failed" if start is None else "error"
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:  # noqa: BLE001
                pass
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        for path in Path("/dev/shm").glob(f"{shm_name}*"):
            try:
                path.unlink()
            except OSError:
                pass

    finish = _readings(end_state) if end_state is not None else (start or {"deaths": 0.0, "lives": 0.0, "bombs": 0.0, "score": 0})
    stage_end = int(end_state["stage"]) if end_state else 0
    if stage_start is None:
        stage_start = stage_end
    if exit_reason == "none" and end not in ("error", "reset_failed"):
        exit_reason = "controller"
    record = dict(identity)
    record.update(
        max_frames=int(max_frames),
        frames=int(frames),
        steps=int(steps),
        renders=int(renders),
        pixels_warmup_frames=int(pixels_warmup_frames),
        deaths=float(finish["deaths"]),
        lives_end=float(finish["lives"]),
        bombs_start=float(start["bombs"]) if start else None,
        bombs_end=float(finish["bombs"]),
        bombs_used=(round(start["bombs"] - finish["bombs"], 6) if start else None),
        score=int(finish["score"]),
        score_start=int(start["score"]) if start else None,
        state_frame=int(end_state["frame"]) if end_state else 0,
        gameplay_frame=int(end_state["gameplay_frame"]) if end_state else 0,
        stage_start=int(stage_start),
        stage_end=int(stage_end),
        cleared=bool(stage_end > stage_start),
        action_histogram=[int(value) for value in action_counts],
        action0_share=float(action_counts[0] / action_counts.sum())
        if action_counts.sum() else 0.0,
        focus_classes_covered=int(sum(1 for value in range(gym_actions.MOVES_PER_CAND + 1,
                                                           obs_spec.N_MOVE_CLASSES)
                                      if action_counts[value] > 0)),
        exit=exit_reason,
        end=end,
        wall_s=round(time.monotonic() - started, 3),
        harness_role=hello_role,
        error=error,
    )
    validate_episode_record(record, expected_spec=(spec_version, spec_digest))
    return record


#: EXP-P3-009 defaults.  ``iters`` is large enough that the percentile endpoints
#: move by <0.5 % of an episode sd between seeds, which is far below the
#: checkpoint-level noise this protocol exists to expose.
BOOTSTRAP_ITERS = 20000
CI_LEVEL = 0.95
#: A seed range is refused beyond this width: a mistyped ``1000-9999`` would
#: silently schedule 9,000 real engine episodes.
MAX_SEEDS = 256


def parse_seeds(spec: str) -> list[int]:
    """``"1000-1019"`` / ``"1000,1001"`` / ``"1000,1005-1007"`` -> ``[int, ...]``.

    Order is preserved and duplicates are dropped (the first occurrence wins), so
    ``--seeds 1000-1004,1002`` runs five episodes, not six.  Fail closed: an empty
    spec, a reversed range, a negative seed or a range wider than :data:`MAX_SEEDS`
    raises :class:`EvalError` instead of silently evaluating something else.
    """

    seeds: list[int] = []
    seen: set[int] = set()
    text = str(spec or "").strip()
    if not text:
        raise EvalError("no seeds given")
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            bounds = piece.split("-")
            if len(bounds) != 2:
                raise EvalError(f"seed range {piece!r} is not 'lo-hi'")
            lo_text, hi_text = bounds[0].strip(), bounds[1].strip()
            if not lo_text.isdigit() or not hi_text.isdigit():
                raise EvalError(f"seed range {piece!r} is not numeric")
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise EvalError(f"seed range {piece!r} is reversed")
            if hi - lo + 1 > MAX_SEEDS:
                raise EvalError(
                    f"seed range {piece!r} covers {hi - lo + 1} seeds, "
                    f"the cap is {MAX_SEEDS}")
            values = range(lo, hi + 1)
        else:
            if not piece.isdigit():
                raise EvalError(f"seed {piece!r} is not a non-negative integer")
            values = (int(piece),)
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            seeds.append(int(value))
    if not seeds:
        raise EvalError("no seeds given")
    if len(seeds) > MAX_SEEDS:
        raise EvalError(f"{len(seeds)} seeds requested, the cap is {MAX_SEEDS}")
    return seeds


def bootstrap_ci(values, *, iters: int = BOOTSTRAP_ITERS, level: float = CI_LEVEL,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the **mean** of ``values``.

    Non-parametric on purpose: S1N per-episode frames are right-skewed near the
    death point (a policy either survives the opening or does not), so a normal
    interval is the wrong shape.  The generator is seeded, so a CI is reproducible
    from the recorded episode list alone.
    """

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise EvalError("cannot bootstrap an empty sample")
    if not 0.0 < float(level) < 1.0:
        raise EvalError(f"CI level {level!r} must be in (0, 1)")
    if int(iters) < 1:
        raise EvalError(f"bootstrap iters {iters!r} must be >= 1")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, array.size, size=(int(iters), array.size))
    means = array[draws].mean(axis=1)
    alpha = (1.0 - float(level)) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def frames_statistics(values, *, iters: int = BOOTSTRAP_ITERS,
                      level: float = CI_LEVEL, seed: int = 0) -> dict:
    """Mean / sd / se / bootstrap CI for one arm's per-episode frame counts."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None,
                "median": None, "min": None, "max": None,
                "ci_low": None, "ci_high": None, "ci_level": float(level),
                "bootstrap_iters": int(iters), "ci_method": "percentile-bootstrap"}
    n = int(array.size)
    mean = float(array.mean())
    # Sample sd (ddof=1) is the unbiased episode-level variance; with n=1 it is
    # undefined, not zero -- an arm measured on one episode has no interval.
    sd = float(array.std(ddof=1)) if n > 1 else None
    se = float(sd / np.sqrt(n)) if (sd is not None and n > 0) else None
    lo, hi = bootstrap_ci(array, iters=iters, level=level, seed=seed)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": float(level),
        "bootstrap_iters": int(iters),
        "ci_method": "percentile-bootstrap",
        "ci_half_width": float((hi - lo) / 2.0),
    }


def summarise(records: list[dict], *, disclosure: str | None = None,
              policy_note: str | None = None,
              bootstrap_iters: int = BOOTSTRAP_ITERS,
              ci_level: float = CI_LEVEL) -> dict:
    measured = [record for record in records if record["end"] != "error"]
    clears = sum(1 for record in records if record.get("cleared"))
    frames = [record["frames"] for record in measured]
    stats = frames_statistics(frames, iters=bootstrap_iters, level=ci_level)
    seeds_in_order: list[int] = []
    for record in records:
        if record["seed"] not in seeds_in_order:
            seeds_in_order.append(int(record["seed"]))
    per_seed = [
        {
            "seed": seed,
            "repeats": len([r for r in measured if r["seed"] == seed]),
            "mean_frames": float(np.mean([r["frames"] for r in measured
                                          if r["seed"] == seed])),
            "frames": [int(r["frames"]) for r in measured if r["seed"] == seed],
        }
        for seed in seeds_in_order
    ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "episodes": len(records),
        "measured": len(measured),
        "seeds": [record["seed"] for record in records],
        "seed_list": seeds_in_order,
        "seed_count": len(seeds_in_order),
        "repeats": (max(int(record.get("repeat", 0)) for record in records) + 1
                    if records else 0),
        "stage": records[0]["stage"] if records else None,
        "difficulty": records[0]["difficulty"] if records else None,
        "stop_on_death": records[0].get("stop_on_death") if records else None,
        "obs_window": records[0].get("obs_window") if records else None,
        "window_alignment": records[0].get("window_alignment") if records else None,
        "max_frames": records[0]["max_frames"] if records else None,
        # EXP-P3-009: the point estimate travels with its interval.  ``frames_stats``
        # is the only field a comparison may read; a bare ``mean_frames`` is a
        # descriptor, not a result.
        "frames_stats": stats,
        "frames": [int(value) for value in frames],
        "per_seed": per_seed,
        "mean_frames": stats["mean"],
        "median_frames": stats["median"],
        "sd_frames": stats["sd"],
        "se_frames": stats["se"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "mean_score": float(np.mean([r["score"] for r in measured])) if measured else None,
        "mean_deaths": float(np.mean([r["deaths"] for r in measured])) if measured else None,
        "clears": int(clears),
        "clear_rate": (clears / len(records)) if records else None,
        "end_counts": {end: sum(1 for r in records if r["end"] == end)
                       for end in sorted({r["end"] for r in records})},

        "obs_spec_version": records[0].get("obs_spec_version", obs_spec.OBS_SPEC_VERSION)
        if records else obs_spec.OBS_SPEC_VERSION,
        "obs_spec_digest": records[0].get("obs_spec_digest", obs_spec.obs_spec_digest())
        if records else obs_spec.obs_spec_digest(),
        "comparison": {
            "reference_source": "Document/Update/Alpha/v4.0.0/bench/scoreboard.md "
                                "(S1N, cap_source=human_reference, max_frames=19887)",
            "same_three_seeds": REFERENCE_ROWS,
            "dev20_means": REFERENCE_DEV20,
            "student_rows": REFERENCE_STUDENT_ROWS,
            "note": "random/still are control policies; bc_v0_stage1 was recorded with "
                    "--stage 1 (the real Stage 2) and is not an S1N row; teacher_s1n is "
                    "the P3 teacher on the same cap and stage semantics. n=5 cannot "
                    "support a gate claim.",
        },
        "not_a_gate": True,
        # EXP-P3-009: the criterion template, shipped with every summary so a
        # reader cannot quote the point estimate without seeing what it licenses.
        "criterion": {
            "template": "差异只有在【两个臂都给出同一置信水平的 CI】且【两个 CI 不重叠】"
                        "时才可称为'有效应'（a difference is an effect only when both "
                        "arms report a CI at the same level AND those intervals do not "
                        "overlap）；否则一律记为'无法判定'，禁止写成'小幅效应'/'方向"
                        "一致'/'趋势'。",
            "requires": ["ci_low", "ci_high", "n", "sd"],
            "min_episodes_per_arm": 20,
            "min_checkpoints_per_arm": "1 for a paired claim on the same corpus; "
                                       "3+ or an explicit checkpoint-variance "
                                       "term for an arm-level claim",
            "checkpoint_variance_reference": "EXP-P3-009: same-config M1 vs M1b "
                                             "differ by 1,330 frames while val top-1 "
                                             "differs by 0.0004",
            "this_summary_has_ci": stats["ci_low"] is not None,
            "tool": "tools/rl/v400_eval_power.py claim --a <dir> --b <dir>",
        },
        "window_note": (
            "EXP-P3-004 §4: obs_window='consecutive' renders every game frame so the "
            "4-frame stack is frames[t-3..t], the window the D0/DAgger shards gave the "
            "student during training.  The EXP-P3-002/003 rows (1,084.0 / 2,399.6) were "
            "measured with the legacy 'strided' ruler (one render per decision, stack "
            "spanning 8 frames), so they are only comparable against a 'strided' re-run."),
        "disclosure": disclosure or (
            "V0 behaviour cloning of random/still/dive; not teacher labels, not a P3 "
            "or G3 result."),
    }
    if policy_note:
        summary["policy"] = policy_note
    histograms = [record.get("action_histogram") for record in records
                  if record.get("action_histogram")]
    if histograms:
        total = np.sum(np.asarray(histograms, dtype=np.int64), axis=0)
        focus = range(gym_actions.MOVES_PER_CAND + 1, obs_spec.N_MOVE_CLASSES)
        summary["action_distribution"] = {
            "histogram": [int(value) for value in total],
            "action0_share": float(total[0] / total.sum()) if total.sum() else 0.0,
            "focus_classes_covered": int(sum(1 for value in focus if total[value] > 0)),
            "per_episode_action0_share": [round(float(record.get("action0_share", 0.0)), 4)
                                          for record in records],
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", default="1000,1001,1002,1003,1004",
                        help="seed list and/or ranges: '1000-1019', '1000,1001', "
                             "'1000,1005-1009'.  EXP-P3-009: five seeds cannot "
                             "resolve a 300-frame effect, twenty is the floor")
    parser.add_argument("--repeats", type=int, default=1,
                        help="run every seed N times (repeat index is recorded; "
                             "this proves the ruler is deterministic per "
                             "checkpoint, it does not buy statistical precision)")
    parser.add_argument("--bootstrap-iters", type=int, default=BOOTSTRAP_ITERS,
                        help="percentile bootstrap resamples for the frames CI")
    parser.add_argument("--ci-level", type=float, default=CI_LEVEL,
                        help="confidence level for the frames CI (default 0.95)")
    parser.add_argument("--max-frames", type=int, default=19887,
                        help="19887 = human_reference cap for S1N; 3000 = provisional")
    parser.add_argument("--stage", type=int, default=DEFAULT_STAGE,
                        help="RESET stage argument (0-based); S1N = 0")
    parser.add_argument("--difficulty", default="normal", choices=DIFFICULTIES)
    parser.add_argument("--stop-on-death", action="store_true",
                        help="end the episode on the first death (G1's rule)")
    parser.add_argument("--obs-window", choices=OBS_WINDOWS, default="consecutive",
                        help="EXP-P3-004 §4: 'consecutive' renders every game frame so "
                             "the 4-frame stack matches the training window; 'strided' "
                             "is the legacy one-render-per-decision ruler that produced "
                             "the 1,084.0 / 2,399.6 rows (stack spans 8 frames)")
    parser.add_argument("--obs-v2", action="store_true",
                        help="EXP-P3-008: the checkpoint is an obs-v2 student, so the "
                             "policy is fed act_hist (the 8x18 one-hot of its own last 8 "
                             "decisions) in addition to the obs-v1 planes; the GRU arm "
                             "carries its 8-decision embedding cache across decisions")
    parser.add_argument("--out-dir", default=".tmp_v400_p3/eval")
    parser.add_argument("--device", default="cpu")
    return parser


def _policy_identity(payload: dict) -> tuple[str, str]:
    """(disclosure, policy note) selected from the checkpoint's own train block."""

    train = payload.get("train") or {}
    version = str(train.get("bc_version") or "")
    if payload.get("obs_spec_version") == obs_spec_v2.OBS_SPEC_VERSION:
        model = payload.get("model") or {}
        return (
            "BC v1 on obs-v2 (EXP-P3-008): teacher soft labels + the 8-decision action "
            "history; not a G3 pass, not PSN, not the confirm split.",
            f"obs-v2 arch={model.get('arch')} tau={train.get('tau')} "
            f"parameters={payload.get('parameters')} "
            f"obs_digest={payload.get('obs_spec_digest')}",
        )
    if version == "bc-v1-teacher-soft":
        return (
            str(train.get("disclosure") or
                "BC v1 = behaviour cloning of the P3 teacher's soft labels; not a G3 "
                "pass, not PSN, not the confirm split."),
            f"bc-v1-teacher-soft tau={train.get('tau')} parameters={payload.get('parameters')}",
        )
    return (
        "V0 behaviour cloning of random/still/dive; not teacher labels, not a P3 or "
        "G3 result.",
        f"bc-v0 parameters={payload.get('parameters')}",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.obs_v2:
        model, payload = load_checkpoint_v2(args.checkpoint, device=args.device)
        arch = str((payload.get("model") or {}).get("arch", ""))
        if arch not in ("mlp", "gru"):
            raise SystemExit(f"obs-v2 checkpoint arch {arch!r} is not mlp/gru")
    else:
        model, payload = load_checkpoint(args.checkpoint, device=args.device)
        arch = "mlp"
    model.eval()
    disclosure, policy_note = _policy_identity(payload)
    seeds = parse_seeds(args.seeds)
    if int(args.repeats) < 1:
        raise SystemExit(f"--repeats must be >= 1, got {args.repeats!r}")
    if not seeds:
        raise SystemExit("no seeds given")
    records = []
    for repeat in range(int(args.repeats)):
        for seed in seeds:
            shm_name = (f"p3eval-{os.getpid()}-{seed}-"
                        f"{int(time.time() * 1000) % 100000}")
            record = run_episode(
                engine=args.engine,
                data_dir=args.data_dir,
                checkpoint_path=args.checkpoint,
                model=model,
                seed=seed,
                stage=args.stage,
                difficulty=args.difficulty,
                max_frames=args.max_frames,
                shm_name=shm_name,
                stop_on_death=bool(args.stop_on_death),
                obs_window=str(args.obs_window),
                obs_v2=bool(args.obs_v2),
                arch=arch,
                repeat=repeat,
            )
            records.append(record)
            print(json.dumps({key: record.get(key) for key in
                              ("seed", "repeat", "frames", "deaths", "cleared",
                               "score", "stage_start", "stage_end", "end", "exit",
                               "wall_s", "error")},
                             ensure_ascii=False), flush=True)
    summary = summarise(records, disclosure=disclosure, policy_note=policy_note,
                        bootstrap_iters=int(args.bootstrap_iters),
                        ci_level=float(args.ci_level))

    (out_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    failed = [record for record in records if record["end"] == "error"]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
