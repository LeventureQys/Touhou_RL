"""Preregistered synthetic structural holdout; sealed test is metadata only.

Reuse world.advance/goal/solve and the unchanged rule teacher via duck typing.
No game IO, training, test execution, dynamic function cloning or monkeypatching.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from . import tactical_tasks as world
from . import tactical_task_teacher as teacher


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DangerRegion:
    """Inclusive canonical space/time rectangle; outside=True makes a safe gate."""
    first: int
    last: int
    lo: int
    hi: int
    outside: bool = False

    def __post_init__(self):
        if not 1 <= self.first <= self.last <= 40 or not 0 <= self.lo <= self.hi <= 40:
            raise ValueError("invalid danger geometry/time")

    def contains(self, frame, x):
        return self.first <= frame <= self.last and ((not self.lo <= x <= self.hi) if self.outside else self.lo <= x <= self.hi)


@dataclass(frozen=True)
class TemplateTask:
    event_id: str
    family: str
    subtype: str
    template_group: str
    seed: int
    split: str
    regions: tuple[DangerRegion, ...]
    target: tuple[int, int]
    canonical_start: int
    offset: int = 0
    mirror: bool = False
    model_known: bool = True
    horizon: int = 40

    def __post_init__(self):
        if self.split not in ("development", "validation"):
            raise ValueError("sealed test cannot be instantiated")
        if self.family not in ("F4", "F5") or self.horizon != 40 or self.offset not in (-1, 0, 1):
            raise ValueError("outside registered world")
        if not 0 <= self.target[0] <= self.target[1] <= 40 or not 0 <= self.canonical_start <= 40:
            raise ValueError("invalid goal/start")

    def canonical_x(self, x):
        return (40 - x if self.mirror else x) - self.offset

    def physical_x(self, x):
        x += self.offset
        return 40 - x if self.mirror else x

    @property
    def start_x(self):
        return self.physical_x(self.canonical_start)

    def dangerous(self, frame, x):
        return any(r.contains(frame, self.canonical_x(x)) for r in self.regions)


def _record(split, family, subtype, group, regions, target, start, offset, mirror, variant):
    return dict(event_id=f"r3-{split}-{subtype}-{variant}-{offset+1}-{int(mirror)}", family=family,
                subtype=subtype, template_group=group, seed=737300 + variant,
                split=split, regions=[asdict(DangerRegion(*r)) for r in regions], target=list(target),
                canonical_start=start, offset=offset, mirror=mirror, model_known=True, horizon=40)


def build_manifest():
    """Pure metadata, including sealed test specs; never instantiate or evaluate."""
    specs = [
        ("development", "F4", "single_wait", "dev-single-wait", [(1,16,18,22)], (30,34),10),
        ("development", "F4", "single_exit", "dev-single-exit", [(11,40,18,22)], (30,34),10),
        ("development", "F5", "single_narrow", "dev-single-narrow", [(12,20,18,22,True)], (20,22),16),
        ("development", "F5", "scattered_pair", "dev-scattered-pair", [(6,10,15,17),(14,18,19,21)], (24,26),16),
        ("validation", "F4", "double_gate_a", "val-double-blocked-gates", [(1,16,18,22),(23,40,26,28)], (30,34),10),
        ("validation", "F4", "double_gate_b", "val-double-blocked-gates", [(1,12,16,18),(21,40,24,26)], (28,30),10),
        ("validation", "F5", "double_safe_a", "val-double-safe-gates", [(10,14,18,20,True),(20,26,22,24,True)], (24,26),16),
        ("validation", "F5", "double_safe_b", "val-double-safe-gates", [(8,12,14,16,True),(18,24,18,20,True)], (20,22),16),
        ("test", "F4", "staggered_narrow", "test-staggered-narrow-gates", [(1,10,16,18),(9,22,22,24),(21,32,28,30)], (32,34),10),
        ("test", "F5", "multi_local", "test-multisegment-local-regions", [(4,8,15,17),(12,16,19,21),(20,24,23,25),(28,32,18,20)], (22,24),16),
    ]
    entries = []
    for variant, (split, family, subtype, group, regions, target, start) in enumerate(specs):
        for offset in (-1, 0, 1):
            for mirror in (False, True):
                record = _record(split, family, subtype, group, regions, target, start, offset, mirror, variant)
                entries.append({"task": record, "sha256": digest(record), "sealed": split == "test"})
    payload = {"protocol": "template-round3-v1", "formal_training_labels": False, "entries": entries}
    return {**payload, "sha256": digest(payload)}


def audit_manifest(manifest):
    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    if digest(payload) != manifest["sha256"]:
        raise ValueError("manifest hash mismatch")
    groups, ids, counts = {}, set(), {}
    for entry in manifest["entries"]:
        t = entry["task"]
        if digest(t) != entry["sha256"] or t["event_id"] in ids:
            raise ValueError("task hash/identity mismatch")
        ids.add(t["event_id"])
        if entry["sealed"] != (t["split"] == "test") or t["split"] not in ("development", "validation", "test"):
            raise ValueError("invalid seal/split")
        splits = groups.setdefault(t["template_group"], set())
        splits.add(t["split"])
        if len(splits) != 1:
            raise ValueError("structural group crosses split")
        counts[t["split"]] = counts.get(t["split"], 0) + 1
    return {"counts": counts, "groups": len(groups), "hash_integrity": True, "group_disjoint": True}


def instantiate(record):
    if record["split"] == "test":
        raise ValueError("sealed test cannot be instantiated")
    data = dict(record)
    data["regions"] = tuple(DangerRegion(**r) for r in data["regions"])
    data["target"] = tuple(data["target"])
    return TemplateTask(**data)


def write_manifest(output):
    out = Path(output)
    if not (out / "preregistration.md").is_file():
        raise ValueError("preregistration required before manifest")
    paths = {"template_source": Path(__file__), "world_source": Path(world.__file__),
             "teacher_source": Path(teacher.__file__), "preregistration": out / "preregistration.md"}
    manifest = build_manifest()
    hashes = {name: file_hash(path) for name, path in paths.items()}
    hashes["manifest"] = manifest["sha256"]
    for name, value in (("manifest.json", manifest), ("hashes.json", hashes)):
        with (out / name).open("x", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
    return manifest, hashes


def evaluate(output):
    out = Path(output)
    if (out / "summary.json").exists() or (out / "development").exists() or (out / "validation").exists():
        raise ValueError("preserve prior evaluation, including failures")
    manifest, hashes = write_manifest(out)
    audit = audit_manifest(manifest)
    summaries = {}
    for split in ("development", "validation"):
        destination = out / split
        destination.mkdir()
        summary = {}
        for entry in manifest["entries"]:
            if entry["task"]["split"] != split:
                continue
            task = instantiate(entry["task"])
            result = teacher.rollout(task)
            verification = world.verify_route(task, result["actions"])
            reference = world.solve(task, max_expansions=100000)
            row = {"task": entry["task"], "task_sha256": entry["sha256"], "teacher": result,
                   "teacher_verification": asdict(verification), "reference": asdict(reference),
                   "reference_metrics": world.metrics(task, reference.actions), "formal_training_labels": False}
            with (destination / f"{task.event_id}.json").open("x", encoding="utf-8") as f:
                json.dump(row, f, indent=2)
            group = summary.setdefault(task.family, {"instances": 0, "goal_success": 0, "survived": 0, "reference_feasible": 0, "failures": []})
            group["instances"] += 1
            group["goal_success"] += int(result["metrics"]["goal_success"])
            group["survived"] += int(result["metrics"]["survived"])
            group["reference_feasible"] += int(reference.status == "feasible")
            if not result["metrics"]["goal_success"]:
                group["failures"].append({"event_id": task.event_id, "reason": result["failure"], "reference": reference.status})
        summaries[split] = summary
    current = {"template_source": file_hash(__file__), "world_source": file_hash(world.__file__),
               "teacher_source": file_hash(teacher.__file__), "preregistration": file_hash(out / "preregistration.md")}
    if any(current[k] != hashes[k] for k in current):
        raise ValueError("source changed during evaluation; preserve rows")
    result = {"protocol": "template-round3-v1", "summary": summaries, "audit": audit,
              "source_hashes_unchanged": True, "hashes": hashes, "test_executed": False,
              "test_status": "sealed_manifest_only", "formal_training_labels": False,
              "diagnostic_gate": {s: all(g["goal_success"] == g["instances"] == g["reference_feasible"] for g in groups.values()) for s, groups in summaries.items()},
              "limitations": "Fixed-y synthetic only; no game, training, B0, full six families, statistical noninferiority or full P1/P2 gate claim."}
    with (out / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output), indent=2))


if __name__ == "__main__":
    main()
