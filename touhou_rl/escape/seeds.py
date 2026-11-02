"""Deterministic scenario seed lists (pre-registered, not chosen after results).

Why hash-derived seeds
----------------------
Picking "30 seeds" by hand invites the question of whether they were picked after
seeing which ones pass.  The list here is a pure function of a committed master
seed and the family id, so the same rule reproduces the same list on any machine
and the formal list can be frozen before any run.

Rule (v1)::

    seed(family, k) = int(sha256(f"v370/{family}/{k}/{master}").hexdigest()[:8], 16) % 2**31

Families follow the acceptance matrix ``SC-01`` .. ``SC-17``; the analytic edge
instances (``E-01`` .. ``E-20``) are hand-designed and listed separately, because
their purpose is to be individually inspectable rather than sampled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "v370-seeds/1"

MASTER_SYNTHETIC = 37000
MASTER_LIVE = 37001
SEEDS_PER_FAMILY = 30
SPLIT_DEV_FRACTION = 0.4  # first 12 seeds per family are dev, the last 18 formal

FAMILIES: tuple[tuple[str, str], ...] = (
    ("SC-01", "wide fan with a reachable outer exit"),
    ("SC-02", "single gap in a bullet wall"),
    ("SC-03", "expanding ring, inside gap only"),
    ("SC-04", "asymmetric exits, one closes earlier"),
    ("SC-05", "delayed second wave after the first"),
    ("SC-06", "corner/wall closure"),
    ("SC-07", "same position occupied then vacated"),
    ("SC-08", "crossing between two observation frames"),
    ("SC-09", "far wall / bullet count above the old cap"),
    ("SC-10", "exit changes during evacuation"),
    ("SC-11", "P-point or damage temptation off the safe route"),
    ("SC-12", "no solution / unknown / budget exhausted"),
    ("SC-13", "accelerating or turning bullets with estimation noise"),
    ("SC-14", "mirror/translation and initial focus changes"),
    ("SC-15", "wide-local-gap dead end vs narrow exit"),
    ("SC-16", "field locally empty, exit closes within 90-180 frames"),
    ("SC-17", "isolated sparse bullets (local-skill regression)"),
)

EDGE_INSTANCES: tuple[dict[str, str], ...] = (
    {"id": "E-01", "target": "SC-08", "what": "bullet crosses the hurtbox at t=3 between t=2/4 samples"},
    {"id": "E-02", "target": "SC-08", "what": "diagonal crossing exactly through a corner"},
    {"id": "E-03", "target": "SC-07", "what": "two bullets occupy the same cell 30 frames apart"},
    {"id": "E-04", "target": "SC-07", "what": "occupancy never returns: permanent block is correct"},
    {"id": "E-05", "target": "SC-02", "what": "gap 3 px wider than the hurtbox, no margin"},
    {"id": "E-06", "target": "SC-02", "what": "gap 1 px narrower than the hurtbox: must refuse"},
    {"id": "E-07", "target": "SC-06", "what": "player already in the corner, one action to leave"},
    {"id": "E-08", "target": "SC-06", "what": "clamp boundary itself: centre exactly at 376/432"},
    {"id": "E-09", "target": "SC-04", "what": "nearer exit closes first, farther exit is still open"},
    {"id": "E-10", "target": "SC-04", "what": "both exits close at the same frame (tie must be explicit)"},
    {"id": "E-11", "target": "SC-12", "what": "provably no route (enclosure), not budget exhaustion"},
    {"id": "E-12", "target": "SC-12", "what": "budget exhausted on an open field (unknown, not blocked)"},
    {"id": "E-13", "target": "SC-16", "what": "field empty now, wall arrives exactly at 180 frames"},
    {"id": "E-14", "target": "SC-16", "what": "wall arrives at 181 frames: outside the horizon, must be unknown"},
    {"id": "E-15", "target": "SC-15", "what": "local clearance is maximal toward the dead end"},
    {"id": "E-16", "target": "SC-15", "what": "narrow route needs one focus frame to fit"},
    {"id": "E-17", "target": "SC-17", "what": "single slow bullet, no macro closure: keep local behaviour"},
    {"id": "E-18", "target": "SC-17", "what": "two bullets crossing far from the player"},
    {"id": "E-19", "target": "SC-13", "what": "linear model is exact (control for the noise family)"},
    {"id": "E-20", "target": "SC-14", "what": "mirror pair where the baseline must not prefer a side"},
)


def family_seed(master: int, family: str, index: int) -> int:
    payload = f"v370/{family}/{index}/{master}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31)


def build(master: int = MASTER_SYNTHETIC) -> dict:
    split = int(round(SEEDS_PER_FAMILY * SPLIT_DEV_FRACTION))
    families = []
    for family, description in FAMILIES:
        seeds = [family_seed(master, family, index) for index in range(SEEDS_PER_FAMILY)]
        families.append(
            {
                "family": family,
                "description": description,
                "seeds": seeds,
                "dev": seeds[:split],
                "formal": seeds[split:],
            }
        )
    document = {
        "schema": SCHEMA,
        "master_synthetic": master,
        "master_live_pair_order": MASTER_LIVE,
        "seeds_per_family": SEEDS_PER_FAMILY,
        "split": {
            "dev_fraction": SPLIT_DEV_FRACTION,
            "dev_per_family": split,
            "formal_per_family": SEEDS_PER_FAMILY - split,
        },
        "rule": "seed(family, k) = int(sha256('v370/{family}/{k}/{master}').hexdigest()[:8], 16) % 2**31",
        "families": families,
        "edge_instances": [dict(item) for item in EDGE_INSTANCES],
    }
    document["digest"] = hashlib.sha256(
        json.dumps({k: v for k, v in document.items() if k != "digest"}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return document


def write(path: Path, master: int = MASTER_SYNTHETIC) -> dict:
    document = build(master)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return document
