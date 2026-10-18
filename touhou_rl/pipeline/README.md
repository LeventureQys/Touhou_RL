"""Three-stage data pipeline: **collect → process → train**.

Why a pipeline and not three scripts
------------------------------------
The stages already existed as scripts, but they were coupled by convention: each
one re-opened the dataset and re-derived its own idea of what was there.  The
first symptom was concrete - a capture that was interrupted mid-shard, then a
training run whose report claimed a split it had not actually used.

This package makes the coupling explicit through a single artefact,
``manifest.json`` (:mod:`touhou_rl.pipeline.manifest`):

```
     game                      raw shards                    manifest + labels        checkpoint + metrics
  ┌─────────┐   collect    ┌──────────────┐   process   ┌──────────────────┐  train  ┌──────────────────┐
  │ th08.exe│ ───────────▶ │ *.npz +      │ ──────────▶ │ manifest.json    │ ──────▶ │ *.pt +           │
  │ replay  │              │ collect-     │             │ teacher-values   │         │ train-report.json│
  └─────────┘              │ report.json  │             │ (offline labels) │         └──────────────────┘
                           └──────────────┘             └──────────────────┘
```

Stage guarantees
----------------
``collect``  never computes labels and never splits; it only turns a game into
             shards plus a provenance report.  Re-running it is the only stage
             that costs wall-clock game time.
``process``  validates every shard, records its hash, computes teacher values
             offline, and pins the train/validation split to
             ``(seed, val_fraction)`` **plus a hash of the exact validation
             indices**.  It is the only stage allowed to write ``manifest.json``.
``train``    refuses to run unless the manifest verifies against the directory,
             then stores hyper-parameters and dataset identity next to the
             checkpoint.  It never touches the game.

Command line
------------
```powershell
# 1) collect (needs the game; this is the only slow stage)
python -m touhou_rl.pipeline collect --modality vision --replay th08_replay/Th8_ud3936.rpy ^
    --out third_party/th08/build/rl/datasets/vision-lunatic --max-seconds 160

# 2) process (offline: validate + label + manifest)
python -m touhou_rl.pipeline process --raw third_party/th08/build/rl/datasets/vision-lunatic

# 3) train (offline)
python -m touhou_rl.pipeline train --model vision ^
    --manifest third_party/th08/build/rl/datasets/vision-lunatic/manifest.json ^
    --out third_party/th08/build/rl/runs/vision-v2

# 4) audit at any time
python -m touhou_rl.pipeline inspect --manifest .../manifest.json
```

Per-stage documentation lives next to the code:

* :doc:`collect/README.md <collect>` - sources, modalities, cost per frame
* :doc:`process/README.md <process>` - checks, teacher labelling, split pinning
* :doc:`train/README.md <train>` - model kinds, metric selection, report fields
"""

from __future__ import annotations

__all__ = ["manifest", "collect", "process", "train"]
