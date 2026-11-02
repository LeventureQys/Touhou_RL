# WITHDRAWN: first round18 pre-registration used an 8-decision oscillation block

This directory is kept only as the audit trail of a **withdrawn specification**. It is
not evidence for any round18 claim and must not be cited as such. The live evidence is
in `../worldfilter-round18/`.

## What was wrong

The frozen criterion defined the alternation baseline as `left/right` switching every
**8 decisions (16 logical frames)**. The repository convention is 16 decisions:

```
Document/Update/Alpha/v3.7.3/evidence/teacher-validation-round8/preregistration.md
  "fixed R1 broad oscillation: canonical -2/+2 switching every 16 decisions
   (32 logical frames)"
```

and `v373_arm_diagnostic` / `tactical_teacher.evaluate_development` both alternate on
`(k // 16) % 2` over decisions. The task text ("每 16 步") matches that convention, so the
8-decision block was a specification error, not a modelling choice.

## How it was found

A pre-publication sensitivity check replayed both readings over the 36 worlds this
directory had accepted:

| reading | withdrawn 36 worlds feasible |
|---|---:|
| 8 decisions / 16 logical frames (what this directory froze) | 0/36 |
| 16 decisions / 32 logical frames (repository convention) | **10/36** |

Under the correct convention 10 of these 36 worlds would have been invalid ("a single
alternation already passes"), so the directory cannot be used.

## What was done

1. The criterion was corrected to 16 decisions / 32 logical frames and re-frozen in a
   new pre-registration (`../worldfilter-round18/preregistration.json`).
2. Seeds (18001/18002/18003), candidate domains and budgets (200/400/200) were kept
   exactly as first declared: nothing was relaxed, re-tuned or widened.
3. One new generation run was executed against the corrected criterion.
4. No world, verdict or action from this directory was reused.

This directory also predates a derived-field correction: its `run/summary.json` and
`run/comparison.json` old-set `constant_passable` column is 0 because the old-set probe
row omitted `any_constant_feasible`; its `run-correction/` holds the read-only rebuild.
Both defects are superseded by the new directory; nothing here is repaired in place.
