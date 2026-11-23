# v8.4.0 deviations scratch (fold into RUN.md)

1. **One cell failed at launch and was retried.**
   `eval-R4-d600-s0` (the first `R4`/seed-0 cell of the wave) died during parent
   start with `HarnessError: harness closed the control pipe` —
   `rc=4`, wall `7.792 s`, started `2026-11-22T10:12:25+08:00`, finished
   `10:12:32`. Evidence: `logs/wave-d600.out` (`[eval-R4-d600-s0] rc=4
   wall=7.792s`) and the per-cell TSV. The cell was re-launched detached at
   `10:13:15` (`logs/retry-R4-s0.out`) and its published rows come from the
   retry (`rc=0`). No other cell failed.
2. **The pre-edit runner was archived after the A/B before-run, not before it.**
   `tools/rl/v800_bomb.py` was not copied before the before-run; it is archived
   byte-for-byte from `.tmp_v810_bomb/v800_bomb.at_launch.py`
   (sha256 `f884a4be…b918`, mtime `08:57`, i.e. before the `10:07` before-run,
   and identical to the hash v8.2.0 recorded for its shipped runner). The gap is
   closed behaviourally: `refs/repro_before_check.json` re-runs that archived file
   for the same first 2 episodes and gets the same rows (2/2 identical).
3. **`bombs_used == honoured presses` is not an identity for every arm.**
   Measured: one honoured press can spend **two** bombs (a double decrement in
   the same decision). Second smoke caught it; the audit reports the counts and
   lists the doubled presses instead of asserting the v8.0.0 identity.
4. **The runner's legacy smoke gate returns 7 for R2 and R3.**
   `--mode smoke`'s built-in `checks.all_pass` requires
   `counter_bookkeeping_closes` (`bombs_used == in-decision decrements`), which
   item 3 breaks; the *rule* semantics are checked separately and pass for all
   five rules. R2's smoke output also contains episodes that die before decision
   200 and therefore never bomb — recorded, not hidden.
5. **`--mode smoke`'s durable copy is a single shared path**
   (`<work>/smoke/bomb_smoke.json`); five smoke runs in parallel therefore race
   on it. The canonical per-rule files are `smoke/bomb_smoke_R{1..5}.json`, which
   are the ones published and audited.
