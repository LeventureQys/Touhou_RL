"""Unit tests for the round28 privilege audit — no game, no teacher rollouts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools" / "rl") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "rl"))

import v373_privilege_audit as audit  # noqa: E402


def _world(family: str = "F1", **overrides):
    from touhou_rl import tactical_teacher_validation as tv

    fields = {
        "event_id": f"test-{family}-m0",
        "family": family,
        "template_group": f"test-{family}",
        "seed": 1,
        "start_x": 120.0,
    }
    fields.update(overrides)
    return tv.ValidationWorld(**fields)


def _load_module_with_read(tmp_path: Path, field: str):
    """A stand-in for a teacher module that reads one privileged field."""
    path = tmp_path / f"fake_teacher_{field}.py"
    path.write_text(
        "def read(masked):\n"
        f"    return masked.{field}\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(f"fake_teacher_{field}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path.name


def test_masked_world_blocks_reads_made_by_the_teacher_module(tmp_path):
    module, filename = _load_module_with_read(tmp_path, "gap")
    masked = audit.MaskedWorld(_world(), (filename,))
    with pytest.raises(audit.PrivilegeBlocked) as info:
        module.read(masked)
    assert info.value.field == "gap"


def test_masked_world_forwards_reads_from_other_modules():
    world = _world()
    masked = audit.MaskedWorld(world, ("tactical_teacher.py",))
    # This test file is not the teacher module, so the same read is forwarded.
    assert masked.gap == world.gap
    assert masked.coverage == world.coverage


def test_masked_world_forwards_ordinary_attributes():
    world = _world()
    masked = audit.MaskedWorld(world, ("tactical_teacher.py",))
    assert masked.family == world.family
    assert masked.mirror == world.mirror
    assert masked.canonical_x(100.0) == world.canonical_x(100.0)
    with pytest.raises(AttributeError):
        masked.not_a_field


def test_perturb_changes_privileged_fields_without_mutating_the_world():
    world = _world(internal_gap=True)
    changed = audit.perturb(world, "F1")
    assert changed.gap != world.gap
    assert changed.coverage != world.coverage
    assert changed.wall_frame != world.wall_frame
    assert world.gap == (180.0, 204.0)  # the frozen original is untouched
    # F2/F3 parameters move too, but structural flags are preserved.
    changed_f3 = audit.perturb(_world("F3"), "F3")
    assert changed_f3.window != world.window
    assert changed_f3.internal_gap == _world("F3").internal_gap


def test_run_teacher_reports_actions_and_verdict():
    class HoldTeacher:
        def decide(self, s, state):
            return SimpleNamespace(action=0)

    world = _world()
    outcome = audit.run_teacher(HoldTeacher(), world, world)
    assert outcome["blockedField"] is None
    assert outcome["error"] is None
    assert outcome["verdict"] in {"feasible", "rejected_witness", "unknown"}
    assert len(outcome["actions"]) > 0
