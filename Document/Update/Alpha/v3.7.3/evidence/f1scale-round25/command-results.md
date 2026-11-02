# round25 — commands, failures and exit codes

All commands were run from the repository root `D:\WorkShop\Touhou_RL`.
`-p no:cacheprovider --basetemp .tmp/...` keeps pytest away from the read-protected
`.pytest_cache` in the repository root; no permission was changed.

## 1. Test iterations (before the pre-registration was frozen)

| # | command | result |
|---|---|---|
| tests-1 | `python -m pytest tests/test_v373_f1_scale.py -q -p no:cacheprovider --basetemp .tmp/v373-f1scale-tests-1` | **3 failed / 25 passed** — (a) `R21` records do not carry a `scenario` key (it lives in `rollout.json`); (b) `r19_teacher_actions` was called for the F2 asset; (c) the identity key was an absolute path while the test looked for a relative one |
| tests-2 | `... --basetemp .tmp/v373-f1scale-tests-2` | **2 failed / 26 passed** — the `r21` import was removed with the unused `torch` import; the identity key uses Windows backslashes |
| tests-3 | `... --basetemp .tmp/v373-f1scale-tests-3` | **28 passed** (this was the pre-registered metric's test set) |
| tests-4 | `... --basetemp .tmp/v373-f1scale-tests-4` | **1 failed / 30 passed** — a test asserted the literal substring `need_left`, the rule text says `need-left` |
| tests-5 | `... --basetemp .tmp/v373-f1scale-tests-5` | **31 passed** |
| final pre-prereg | `... --basetemp .tmp/v373-f1scale-tests-final-pre` | **31 passed in 3.51 s**, exit 0 |

The final test file is 31 tests: the search contract (the declared budget must reach the
target), the frozen signature union, mirror atomicity, the split floors and the recorded
flip, the shortfall branch, the no-leakage assertions, the frozen R19 split, the round21
preview window and teacher records, the round22 direction read-out recomputed from
round24's own cells, the direction-conditioned flip/keep counts (including the proof that
the raw mirror rate can be satisfied by the frozen prior), the judgement ladder, the
round20 checkpoint contract, and the `x`/`xb`/`exist_ok=False` file contracts.

## 2. Scratch smoke runs (disclosed; never evidence)

All of them used `--out .tmp/f1scale-smoke`, i.e. outside the evidence tree. No evidence
directory was created, read or modified by them. The first six attempts found six real
defects in this round's own code; attempt 7 succeeded and its numbers are byte-identical
to the official run (see §4).

| attempt | failure | cause / fix |
|---|---|---|
| 1 | `KeyError: 'domains'` | `verify_prereg` read `p['search']['domains']`; the domains are frozen inside `p['criterion']['search']['domains']` |
| 2 | `ValueError: the frozen R19 F1 signatures changed` | the pre-registration stored only the three validation signatures while the verifier compared all six frozen groups |
| 3 | `ValueError: stored signature drift: wf25-F1-001` | `loaded_search` compared a tuple signature (in memory) with a JSON list (after `run`'s round trip) |
| 4 | `FileNotFoundError: .../validteacher-round19/run/worlds/F1/wf25-F1-001-m0.json` | the R19 teacher-record equality check was applied to the new worlds too; only the twelve old-group events have an R19 record |
| 5 | `KeyError: 'split'` | `round20.six_baselines` needs `entry['split']`; the evaluation event rows were built without it |
| 6 | run reached the summary | end-to-end success, but the judgement was `mixed_scale` and the raw "mirror toward" rate was **5/7 for the frozen baseline B0**: the new held-out set mixes gap sides, so the raw mirror rate is geometry, not behaviour.  The metric was replaced by the direction-conditioned flip/keep read-out **before** the official pre-registration; criterion SHA `d2ae92b6…` → `5ad3ea35…`, and the tests were updated |
| 6b | `TypeError: string indices must be integers` in `verify` | `verify_artifacts` took arm events from group event-id strings but treated them as event dicts |
| 7 | — | `preregister`, `run`, `verify` all exit 0; verdict `scale_not_the_cause`; the six checkpoint SHAs equal the official run's |

Attempts 1–5 and 6b are engineering iterations of this round's tool and test file only.
Attempt 6 is the design check that forced the metric revision.  The pre-registration was
**not** frozen when any of them ran, no evidence directory existed, and no number from
them is used in the report.

## 3. Official execution (one pre-registration, one run)

```powershell
# 1) final contract tests, before freezing anything
python -m pytest tests/test_v373_f1_scale.py -q -p no:cacheprovider --basetemp .tmp/v373-f1scale-tests-final-pre
#    31 passed in 3.51s，exit 0

# 2) the single pre-registration
python -m tools.rl.v373_f1_scale preregister
#    exit 0；criterion SHA 5ad3ea35a02ab4aa86d673bb908fc9f0826b4a8719dc8ab49e5479cfa5a065d8
#    preregistration.json SHA d450dd4d6694f6077a4f699ef73b2a53b1169ec599e72345ad71c8315fd25b89
#    36 frozen identities；search seed 25001 / budget 400 / target 16

# 3) the single evidence run (background job, collected with job_output)
python -m tools.rl.v373_f1_scale run
#    exit 0；wall clock ≈ 235 s（search 2.403 s / data 36.563 s / encode 2.558 s /
#    training 65.799 s / closed loop 122.986 s）；180 SHA entries, 181 files under run/

# 4) read-only verification
python -m tools.rl.v373_f1_scale verify
#    exit 0；180 run SHA entries (181 files) / 182 root entries / 120 asset bindings / 6 checkpoints /
#    14 closed-loop files / 140 F1 cells / filter_valid on both evaluation sets

# 5) contract tests after the run (the frozen test file must still pass)
python -m pytest tests/test_v373_f1_scale.py -q -p no:cacheprovider --basetemp .tmp/v373-f1scale-tests-after-run
#    31 passed，exit 0

# 6) round18 regression (the generator/criterion this round reuses)
python -m pytest tests/test_v373_world_filter.py -q -p no:cacheprovider --basetemp .tmp/v373-f1scale-regression-r18
#    15 passed，exit 0

# 7) verification after the logs were written (logs are outside the root SHA table)
python -m tools.rl.v373_f1_scale verify
#    exit 0
```

Exit codes, in order: **0 / 0 / 0 / 0 / 0 / 0 / 0**.
`logs/` and this file were written after the run's root `sha256.json`; they are therefore
not in that table, exactly as in rounds 21–24, and step 7 re-verified the frozen artifacts.

## 4. Determinism

The six checkpoint SHA-256 values of the final scratch smoke and of the official run are
identical (`4c4dd3c7…`, `8273e257…`, `399684c9…`, `7f6a4080…`, `4153be147…`, `58b76828…`),
and `search.json`, `split.json`, `f1-gap.json`, `baselines.json` are byte-identical.  The
smoke and the official run are the same computation; the smoke is disclosed only because
it forced the metric revision in §2 attempt 6.

## 5. Logs in this directory

| file | content |
|---|---|
| `logs/preregister.log` | the pre-registration command's stdout |
| `logs/execution.log` | the single run's full stdout (renders, training, closed loop) |
| `logs/verify.log` | the read-only verification output |
| `logs/tests-before-prereg.log` | 31 passed; the pre-freeze invocation printed the same, and the identical command was replayed after the run so that the log exists (same frozen test file, same result) |
| `logs/tests-after-run.log` | 31 passed after the run |
| `logs/regression-round18.log` | 15 passed, round18 suite |
| `logs/verify-after-logs.log` | verification re-run after the logs were written |
