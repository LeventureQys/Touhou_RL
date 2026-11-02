"""No sealed-test worlds are instantiated or executed in these tests."""
from copy import deepcopy
import pytest

from touhou_rl import tactical_task_templates as templates
from touhou_rl import tactical_tasks as world
from touhou_rl import tactical_task_teacher as teacher


def test_manifest_split_hash_integrity_only():
    manifest = templates.build_manifest()
    assert templates.audit_manifest(manifest) == {
        "counts": {"development": 24, "validation": 24, "test": 12},
        "groups": 8, "hash_integrity": True, "group_disjoint": True}
    assert manifest == templates.build_manifest()
    # Sealed entries are metadata only. Do not instantiate, solve or roll them out.
    for entry in manifest["entries"]:
        assert entry["sha256"] == templates.digest(entry["task"])
        assert entry["sealed"] == (entry["task"]["split"] == "test")
    changed = deepcopy(manifest)
    changed["entries"][-1]["task"]["seed"] += 1
    with pytest.raises(ValueError, match="hash"):
        templates.audit_manifest(changed)


def test_group_leak_is_rejected():
    manifest = templates.build_manifest()
    entry = manifest["entries"][24]
    entry["task"]["template_group"] = manifest["entries"][0]["task"]["template_group"]
    entry["sha256"] = templates.digest(entry["task"])
    manifest["sha256"] = templates.digest({k: v for k, v in manifest.items() if k != "sha256"})
    with pytest.raises(ValueError, match="crosses split"):
        templates.audit_manifest(manifest)


def test_development_occupancy_matches_existing_world():
    mapping = {"single_wait": ("F4", "wait_pass"), "single_exit": ("F4", "immediate_exit"),
               "single_narrow": ("F5", "narrow_gap"), "scattered_pair": ("F5", "scattered")}
    for entry in templates.build_manifest()["entries"]:
        record = entry["task"]
        if record["split"] != "development":
            continue
        task = templates.instantiate(record)
        family, subtype = mapping[task.subtype]
        legacy = world.Task("legacy", family, subtype, "legacy", 1, task.offset, task.mirror)
        for frame in range(1, 41):
            for x in range(41):
                assert world.dangerous(task, frame, x) == world.dangerous(legacy, frame, x)
        for action in world.ACTIONS:
            assert world.advance(task, world.initial_state(task), action) == world.advance(legacy, world.initial_state(legacy), action)
        assert teacher.rollout(task)["actions"] == teacher.rollout(legacy)["actions"]


def test_validation_duck_typing_without_teacher_tuning():
    for entry in templates.build_manifest()["entries"]:
        if entry["task"]["split"] != "validation":
            continue
        task = templates.instantiate(entry["task"])
        state = world.initial_state(task)
        assert len(task.regions) == 2
        decision = teacher.decide(task, state)
        assert decision.action in world.ACTIONS
        assert world.advance(task, state, decision.action).frame == 2
        assert world.verify_route(task, ()).status == "unknown"


def test_time_geometry_and_mirror_transform():
    region = templates.DangerRegion(4, 8, 10, 12)
    assert region.contains(4, 10) and region.contains(8, 12)
    assert not region.contains(3, 10) and not region.contains(9, 12)
    assert not region.contains(4, 13)
    with pytest.raises(ValueError):
        templates.DangerRegion(9, 4, 10, 12)
    for entry in templates.build_manifest()["entries"]:
        if entry["task"]["split"] == "test":
            continue
        task = templates.instantiate(entry["task"])
        for x in range(1, 40):
            assert task.canonical_x(task.physical_x(x)) == x


def test_requires_preregistration_and_preserves_existing(tmp_path):
    with pytest.raises(ValueError, match="preregistration"):
        templates.write_manifest(tmp_path)
    (tmp_path / "preregistration.md").write_text("frozen protocol", encoding="utf-8")
    templates.write_manifest(tmp_path)
    with pytest.raises(FileExistsError):
        templates.write_manifest(tmp_path)
