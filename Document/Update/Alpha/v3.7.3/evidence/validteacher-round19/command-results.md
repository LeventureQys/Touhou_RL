# Round19 command results

All commands were run from the repository root on Windows, CPU only, with the tool
and the test file frozen by preregistration.json (SHA 48a8ccb24846c9194e29c81ae10cb77070aa2eac02ebf0e6a203c9629d8aa605).

| # | command | exit code | result |
|---|---|---:|---|
| 1 | python -m pytest tests/test_v373_valid_teacher_eval.py -q -p no:cacheprovider --basetemp .tmp/v373-validteacher-round19-tests-01 | 0 | 15 passed, 1 skipped (the post-run test skips itself) |
| 2 | python -m tools.rl.v373_valid_teacher_eval preregister | 0 | 13 identities frozen, 28 events / 14 groups, manifest SHA 9bd3a1b8d31784d6e0fd546fc701f93be611311d1743c78f159e77a4f5fe7fa7 |
| 3 | python -m pytest tests/test_v373_valid_teacher_eval.py -q -p no:cacheprovider --basetemp .tmp/v373-validteacher-round19-tests-02 | 0 | 15 passed, 1 skipped (same frozen source still passes) |
| 4 | python -m tools.rl.v373_valid_teacher_eval run (background job, collected once) | 0 | 28 worlds written, 168 baseline rows reproduced, 28/28 witnesses replaid, split frozen |
| 5 | python -m pytest tests/test_v373_valid_teacher_eval.py -q -p no:cacheprovider --basetemp .tmp/v373-validteacher-round19-tests-03 | 0 | 16 passed (post-run SHA and split tests now execute) |

Logs: 	ests-first.log, 	ests-after-preregister.log, 	ests-completion.log,
preregister.log, execution.log (the 
un output, written by PowerShell 5.1
Out-File -Encoding utf8).

Run wall clock: 
un/split.json written at 20:52:09.226 and 
un/sha256.json at
20:52:17.747, i.e. about 8.5 s for the twenty-eight worlds, the hundred and sixty
eight baseline rows and the twenty-eight witness replays (interpreter start and
imports excluded). One single run, no rerun, no repair pass.

The test pass before pre-registration already evaluates exactly one world
(wf18-F1-002-m0) end to end as a contract test. That is intentional and it was
not used to choose the criterion, the split or anything else: the criterion is a
module constant frozen in preregistration.json afterwards, and the full
twenty-eight-world result exists only in 
un/.
