# HYBRID_H2S — the H2s arm per dev window (20 episodes × 4 dev windows × 3 seeds)

**Round**: v8.12.0 / `HYBRID_H2S`
**Producer**: `tools/rl/v8d0_h2s.py` (new this round; the episode loop is the frozen
`tools/rl/v830_bombpol.py::run_episode36`, imported, not re-implemented)
**Host**: training machine, WSL2 (`/root/v400-venv/bin/python`, engine
`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`)
**Work dir**: `.tmp_v8d0_h2s/` (new; `.tmp_v8c0_hybrid/` was **not** touched)
**Evidence dir**: `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/` (this directory)

## 0. Verdict

**COMPLETE — 240/240 episodes measured** (12 cells × 20 episodes, 144,000 decisions,
288,000 frames), **0 invalid rows**.

| window | n | full | rate | CP95 | R5, same window (prereg §0) | H2s − R5 |
| --- | --- | --- | --- | --- | --- | --- |
| **11** | 60 | 60 | **100.00 %** | [94.04 %, 100.00 %] | 81.1 % | **+18.9 pp** |
| **12** | 60 | 59 | **98.33 %** | [91.06 %, 99.96 %] | 79.2 % | **+19.1 pp** |
| **17** | 60 | 57 | **95.00 %** | [86.08 %, 98.96 %] | 70.4 % | **+24.6 pp** |
| **18** | 60 | 60 | **100.00 %** | [94.04 %, 100.00 %] | 77.7 % | **+22.3 pp** |
| **pooled** | **240** | **236** | **98.33 %** | **[95.79 %, 99.54 %]** | — | — |

`rate` = `survived` fraction at **600 decisions = 1,200 frames** (2 frames/decision),
PARK frame 8663; CP95 = Clopper-Pearson exact two-sided 95 % interval (pure Python,
no scipy). 4 deaths in 240 episodes (§7.5). Per seed: s0 78/80 (97.5 %), s1 80/80
(100 %), s2 78/80 (97.5 %).

This file answers exactly one question — **does H2s hold up on windows other than
11?** — and does no other work: no training, no collection, no labels, no other arm.

## 1. Why this file exists

v8.12.0's H2s implementation lives in `tools/rl/v8c0_hybrid.py`, whose `--mode eval`
uses the v4.5.0/v8.6.0 protocol of **300 episodes per window** with the episode loop
nested inside the window loop. The H2s cell was interrupted mid-window-11, so what
survived was one window only and no cross-window statement could be made: at 12:33
the three files `.tmp_v8c0_hybrid/eval/eval_H2s_d600_seed{0,1,2}.jsonl` held **55
rows each, all `window=11`, all `survived=true`** (and three `v8c0_hybrid --mode
eval` processes were still alive; they were **not** touched).

Re-running 3,600 episodes is neither affordable nor necessary to answer it. This
round re-runs **one** arm with a deliberately small protocol:

```
for seed in (0, 1, 2):            # torch.Generator().manual_seed(seed*1000+7)
  for window in (11, 12, 17, 18): # dev windows, PARK frame 8663
    for episode in range(20):     # 20 per window, not 300
      run_one_episode(budget_decisions=600, policy="H2s")
```

## 2. H2s, verbatim

| element | value |
| --- | --- |
| policy | `.tmp_v860_bombdag/ckpt/seed{seed}/bombdag.pt`, `Policy(106, 64, 36)` via `v860_bombdag.load_policy36` |
| candidate | `score[c] = max(logit[2c], logit[2c+1])`, `cand ~ softmax(score / 1.0)`, **one `torch.multinomial` draw per decision**, generator `torch.Generator().manual_seed(seed*1000 + 7)` |
| bomb | **external R5 rule only** (`tools/rl/v800_bomb.py`): `nearest_all_px < {3: 80, 2: 60, 1: 40}[bombs_remaining]` **and** `bombs_remaining > 0`; the policy's bomb head is never used |
| masks | `cand_to_mask(cand, bomb)` on the decision's first frame, `cand_to_mask(cand, 0)` afterwards (inside the imported loop) |
| start / horizon | PARK frame 8663, 2 frames per decision, 600 decisions = 1,200 frames |

The sampler is audited per episode: `samples_consumed == decisions_measured`
(600/600 or 591/590 at a death), `r5_bombs == bomb_emits`, and
`bombs_used == bombs_start - bombs_end`. `cand_counts` in every row shows **15–18 of
the 18 candidates used per episode** (mean sampled probability ≈ 0.11), i.e. the arm
really samples; it is not an argmax in disguise.

## 3. Identities (fail closed; `ckpt_assertions_start.json` / `_final.json`)

| item | sha256 | matches |
| --- | --- | --- |
| engine `th08-modern` | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | ✅ |
| `.tmp_v860_bombdag/ckpt/seed0/bombdag.pt` | `9c6d5d4ac7c9a6120ecd33aa9de7a0762c916413cc8f52e0c82f8ac41f8a483b` | ✅ |
| `.tmp_v860_bombdag/ckpt/seed1/bombdag.pt` | `36a51090d56149dc185d86e5b386f89304e5522c3612d9e919a4092e99f8e84c` | ✅ |
| `.tmp_v860_bombdag/ckpt/seed2/bombdag.pt` | `885dde70c57f18bb7f1e3d0351737d239a52ce064aeb47fda025866bd6e45943` | ✅ |
| `.tmp_v450_cf/ckpt/seed0/round2.pt` (frozen v4.5.0 `S`) | `4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc` | ✅ |
| `.tmp_v450_cf/ckpt/seed1/round2.pt` | `a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4` | ✅ |
| `.tmp_v450_cf/ckpt/seed2/round2.pt` | `c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01` | ✅ |

`--mode assert-ckpt` also hashes the 8 protected sources (`v432_fork_arm.py`,
`v450_cf_label.py`, `v431_arm_a.py`, `v800_bomb.py`, `v830_bombpol.py`,
`v860_bombdag.py`, `touhou_rl/v420/contracts.py`, `touhou_rl/v400/gym/actions.py`)
and itself, before (`phase=start`, exit 0) and after (`phase=final`, exit 0) the
grid. Nothing under `touhou_rl/**`, `third_party/**`, `.tmp_v8c0_hybrid/**` or an
existing version directory was written.

## 4. D3 fidelity gate — 5/5 PASS

Run through the frozen `tools/rl/v432_fork_arm.py --mode fidelity --seeds 12
--learner frozen` (exit 0, wall **11.393 s**); the round is void on anything short of
5/5, so this gate ran **before** the smoke and the grid (`d3_gate.json`, raw record
`.tmp_v8d0_h2s/fidelity_d3.json`, sha256 `08b5a5c0e16395c928cfea81bb5995870a3cb2c0…`).

| # | assertion | result |
| --- | --- | --- |
| 1 | `HELLO` replies `role=shadow` | PASS |
| 2 | `DEBUG` replies `threads=1`, before and after PARK | PASS |
| 3 | 2,165/2,165 checkpoints to frame 8663, 0 mismatches, final frame 8663 | PASS |
| 4 | the R24 18 constants reproduce `first_death_offset` 18/18 | PASS |
| 5 | `PARK` replies `park=1 … frame=8663` | PASS |

## 5. Audio (AGENTS.md §9.3)

* `SDL_AUDIODRIVER=dummy`, `DISPLAY`/`WAYLAND_DISPLAY` unset in every process
  (`v450.prepare_env`, recorded per eval process in `seed{seed}_windows.json`);
* a **freshly generated and `--inspect`ed** `th08.cfg` before **every** engine start
  (`v432.build_private_data_dir`: `musicMode=0 playSounds=0 musicVolume=0
  sfxVolume=0`, 60 bytes, `muted_four_zero=true`; per-window cfg sha256
  `cc062cfbf9fd2c14…` in `seed{seed}_windows.json`);
* `audio_audit.json`: **12 cfg files, 12 muted, 0 audible, 0 errors** (exit 0).

## 6. Commands, exit codes, wall clock

All under WSL: `cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python …`.
Machine-readable record: `commands.tsv` (label/exit/wall/start/stop/argv) and
`eval_exits.tsv` (the three eval processes, written by their own wrapper).

| # | command (`tools/rl/v8d0_h2s.py` unless stated) | exit | wall |
| --- | --- | --- | --- |
| 1 | `-m compileall -q tools/rl/v8d0_h2s.py` | 0 | 0.3 s |
| 2 | `--mode assert-ckpt --phase start` | 0 | 1.0 s |
| 3 | `v432_fork_arm.py --mode fidelity --seeds 12 --learner frozen` (via `--mode d3`) | 0 | 12.5 s |
| 4 | `--mode smoke --policy-seed 0 --windows 11 --decisions 600 --episodes 1` (first attempt) | 0 | 25.2 s |
| 5 | `-m compileall` (after the audit-check fix) | 0 | 0.4 s |
| 6 | `--mode smoke …` (final, the run whose output is published) | 0 | 25.9 s |
| 7 | `--mode eval --policy-seed 0 --windows 11,12,17,18 --episodes 20 --decisions 600` — attempt 1 | **4** | 660 s |
| 8 | `--mode eval --policy-seed 2 …` — attempt 1 | **4** | 936 s |
| 9 | `--mode eval --policy-seed 1 …` — attempt 1 | **130** (my SIGINT) | 950 s |
| 10 | `--mode eval --policy-seed 0 …` — final | 0 | 1,385 s |
| 11 | `--mode eval --policy-seed 1 …` — final | 0 | 1,389 s |
| 12 | `--mode eval --policy-seed 2 …` — final | 0 | 1,378 s |
| 13 | `--mode summary` | 0 | 0.5 s |
| 14 | `--mode audio-audit` | 0 | 0.5 s |
| 15 | `--mode process-audit` | 0 | 0.3 s |
| 16 | `--mode assert-ckpt --phase final` | 0 | 0.8 s |
| 17 | `determinism_check.py` (first attempt, strict key set) | **1** | 0.2 s |
| 18 | `determinism_check.py` (final) | 0 | 0.2 s |
| 19 | `--mode publish` | 0 | 1.0 s |

The three eval processes are **21.5–23.2 min each** of measured work
(`wall_s` inside the run: 1,293.4 / 1,297.8 / 1,285.9 s; the rest is `torch` import
and process start under load). Attempts 1 are documented in §8.3 — **all 12 final
cells come from the three final processes**; the attempts-1 partial files are kept
only in `.tmp_v8d0_h2s/eval/ref1_*` and are not published.

`commands.tsv` in this directory is the exact 20-row file from
`.tmp_v8d0_h2s/commands.tsv` (its last row is the publish run that snapshotted it);
the artifact hashes below are the ones in `manifest.json`.

## 7. Results

### 7.1 Per (seed, window) — `summary.json` → `cells`

| seed | window | n | full | rate | CP95 | alive_frames mean | bombs mean | distinct trajectories |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 11 | 20 | 20 | 100.00 % | [83.16 %, 100.00 %] | 1200.00 | 3.00 | 20 |
| 1 | 11 | 20 | 20 | 100.00 % | [83.16 %, 100.00 %] | 1200.00 | 3.00 | 20 |
| 2 | 11 | 20 | 20 | 100.00 % | [83.16 %, 100.00 %] | 1200.00 | 3.00 | 20 |
| 0 | 12 | 20 | 19 | 95.00 % | [75.13 %, 99.87 %] | 1198.90 | 3.00 | 20 |
| 1 | 12 | 20 | 20 | 100.00 % | [83.16 %, 100.00 %] | 1200.00 | 3.00 | 20 |
| 2 | 12 | 20 | 20 | 100.00 % | [83.16 %, 100.00 %] | 1200.00 | 3.00 | 20 |
| 0 | 17 | 20 | 19 | 95.00 % | [75.13 %, 99.87 %] | 1199.75 | 3.00 | 19 |
| 1 | 17 | 20 | 20 | 100.00 % | [83.16 %, 100.00 %] | 1200.00 | 3.00 | 19 |
| 2 | 17 | 20 | 18 | 90.00 % | [68.30 %, 98.77 %] | 1188.90 | 3.00 | 20 |
| 0 | 18 | 20 | 20 | 100.00 % | [83.16 %, 100.00 %] | 1200.00 | 2.95 | 20 |
| 1 | 18 | 20 | 20 | 100.00 % | [83.16 %, 100.00 %] | 1200.00 | 3.00 | 20 |
| 2 | 18 | 20 | 20 | 100.00 % | [83.16 %, 100.00 %] | 1200.00 | 3.00 | 20 |

`distinct trajectories` = distinct `(alive_frames, bombs_used, bomb_decisions)`
signatures per cell: 20 of 20 in 10 cells, 19 in the two w17 cells that hold a death,
i.e. the 20 episodes are 20 **different** rollouts (as expected of a sampled arm), not
replays. Per-window pooled: w11 60/60, w12 59/60, w17 57/60, w18 60/60.

### 7.2 Pooled

`n = 240`, `full = 236`, `rate = 0.983333`, **CP95 = [0.957879, 0.995441]**,
`invalid_rows = 0`, `complete = true`.

### 7.3 Bomb-usage distribution (the R5 rule's behaviour, `summary.json` → `bomb_usage`)

| quantity | value |
| --- | --- |
| bombs_used per episode | **0: 0, 1: 0, 2: 1, 3: 239** — mean **2.9958**, median 3 |
| episodes ending with 0 bombs | 239 / 240 |
| bomb emits | 2,384 total (mean **9.93** per episode) → 719 honoured, 1,665 ignored while a bomb was already active |
| first_bomb_decision | all 240 fired at **decision 0** (mean 0, min 0, max 0) |
| bomb decisions per episode | 1–9: 172 episodes, 10–24: 68 episodes |
| per window (bombs/episode) | w11 3.000, w12 3.000, w17 3.000, w18 2.983 |
| per seed (bombs/episode) | s0 2.988, s1 3.000, s2 3.000 |

Two facts worth recording. (a) At this PARK state the frozen R5 gate sees
`nearest_all_px < 80 px` **at decision 0 in all 240 episodes**, so the whole stock is
committed in the opening seconds: 239/240 episodes end with no bomb left, unlike
R5's own 1.36–1.75 bombs per episode at the same horizon. (b) The frozen gate has no
cooldown, so ~70 % of its emits land inside an active bomb and are ignored
(`bomb_ignored`); the 719 honoured emits are what `bombs_used` counts. Both are
properties of the frozen rule, reported here, not changes made by this round.

### 7.4 Deaths (4 / 240)

| cell | episode | alive_frames | death frame | decisions | bombs | bomb decisions |
| --- | --- | --- | --- | --- | --- | --- |
| s0/w12 | 10 | 1,178 | 9,842 | 589 | 3 | 0,1,2,3,4,5,6,303,457 |
| s0/w17 | 1 | 1,195 | 9,859 | 597 | 3 | 0,1,2,3,4,5,6,126,131,132,332 |
| s2/w17 | 6 | 982 | 9,646 | 491 | 3 | 0,1,2,3,4,5,6,130,345 |
| s2/w17 | 19 | 1,196 | 9,860 | 598 | 3 | 0,1,2,3,4,5,6,125,367 |

All four are `terminal=first_death` (the frozen `counterfactual._terminal_of` rule),
and all four used their 3 bombs. w17 is the only window where the rate leaves the
ceiling (57/60); w11 and w18 are 60/60.

### 7.5 H2s beside R5, window by window

| window | H2s (n=60) | R5 (prereg §0, same window) | difference |
| --- | --- | --- | --- |
| 11 | 100.00 % | 81.1 % | +18.9 pp |
| 12 | 98.33 % | 79.2 % | +19.1 pp |
| 17 | 95.00 % | 70.4 % | +24.6 pp |
| 18 | 100.00 % | 77.7 % | +22.3 pp |

Both arms use the same R5 bomb rule and the same four dev windows at 600 decisions;
they differ in the **candidate direction** (H2s: the v8.6.0 sampled candidate head;
R5: the v4.5.0 `S` student's own stochastic candidate). No test is claimed here — the
parent asked for the rates side by side, and n=60 per window is a descriptive sample,
not a hypothesis test. Note also that R5's own record spends 1.36–1.75 bombs per
episode while H2s spends 3.00 (§7.3), so the two arms differ in *both* factors at
once; this round measured H2s, it did not re-measure R5.

## 8. Deviations (all documented, none silent)

1. **Execution shape.** The specified loop is `seed → window → episode`. I ran one
   process per seed **concurrently** (3 at once), and each process loops
   `window → episode` with 20 episodes per window. The cells are the same; the seeds
   share no state (own `torch.Generator`, own checkpoint, own private data dir), so
   only wall time is affected.
2. **Generator scope.** One `torch.Generator` per **seed process**, created once
   before the window loop and consumed once per decision across all 80 episodes —
   the discipline of `v860_bombdag.mode_eval` and of v8c0's H2s (re-seeding per window
   or per episode would turn the 20 episodes into replays). Recorded in
   `seed{seed}_windows.json`.
3. **Attempt 1 aborted by my own audit check, then a full clean re-run.** The first
   grid aborted at seed0/w12 ep10 (and seed2/w17 ep6) because my per-episode
   bookkeeping check compared samples to `decisions_executed`; on a death the fatal
   decision is **measured but not counted** by the frozen loop
   (`run_episode36` breaks before its increment), so the check was wrong, not the
   episode. Fix: `samples_consumed == decisions_measured` (with the `+1` slack) plus a
   new `frames_consistent` check; validated offline on the very row that failed
   (`589 executed / 590 measured / 590 samples / 1 frame of the fatal decision`).
   Then **all three seeds were re-run from scratch** (not resumed) so that every cell
   of a seed comes from one continuous generator stream. Attempts 1 are preserved in
   `.tmp_v8d0_h2s/eval/ref1_*` and are **not** published; `commands.tsv` keeps their
   exit codes (4, 4, 130).
4. **`checks` key set grew between the two w11 runs** (`frames_consistent` was
   added). The determinism comparison therefore compares only the checks **both**
   runs carry; everything else is compared field for field.
5. **Determinism evidence (from the re-run).** The 60 w11 rows were produced twice by
   two processes: **60/60 identical field for field** (`wall_s`/`spawn_ms` excluded),
   `determinism_check.json`, exit 0. The aborted attempt's dying row reappears
   identically in the clean run (s0/w12 ep10, death frame **9,842**, alive 1,178,
   `bomb_decisions = 0,1,2,3,4,5,6,303,457`).
6. **Cross-implementation check against the previous round.** My seed0/w11 episode 0
   (the published smoke) is identical to row 0 of the previous round's
   `.tmp_v8c0_hybrid/eval/eval_H2s_d600_seed0.jsonl`: `survived=true`,
   `alive_frames=1200`, `bombs_used=3`, 9 emits, `bomb_decisions=0,1,2,3,4,5,6,339,528`.
   So this file's H2s is the previous round's H2s, not a look-alike.
7. **Contention on a shared machine.** At start (`12:33` host time) `pgrep -af
   th08-modern` showed **56 engine processes from other rounds** (11
   `v8b0_eventsample --mode collect` processes, 3 still-running `v8c0_hybrid --mode
   eval H2s` processes, and orphans), load average ≈ 45 on 20 cores. Per-episode wall
   grew from ~10 s (previous round, quieter) to ~19 s (w11/w12) and ~14 s (w17/w18).
   None of those processes were touched; this round's own engine processes are **0**
   (`process_residue.json`: 0 matching, 17 other-round processes untouched).
8. **The previous round's H2s count.** The task said 153 rows; what I observed at
   12:33 was **55 rows per seed (165 total), all window 11**, in files that were still
   being appended to by live processes. I report the observation, not the figure.
9. **No test, no verdict.** With 20 episodes per window, per-window CP95 is wide
   (n=60). This round produces rates and intervals only; it runs no Fisher test, no
   P-gate and no prereg verdict. The v8.12.0 prereg's own P1–P5 were not re-run.

## 9. Artifacts and sha256

12 eval files (`eval_H2s_w{window}_seed{seed}.jsonl`, 20 rows each) + the summary and
audits below; `manifest.json` lists every file with sha256/bytes.

| file | sha256 | bytes |
| --- | --- | --- |
| `eval_H2s_w11_seed0.jsonl` | `6992f9f6ee2dcf74bd66e1f21f7b0b3a3505726103ee054176190fc6a6d1cbef` | 87,143 |
| `eval_H2s_w11_seed1.jsonl` | `f3b1efcaa9ad6ab38f65309d4da9e5b4b6bafdca08990f98ce682183837c9f3d` | 87,978 |
| `eval_H2s_w11_seed2.jsonl` | `2b8b57191642ee806c11192111831dfd5f1c1832e48193f10cca55cfd19fdd97` | 87,798 |
| `eval_H2s_w12_seed0.jsonl` | `8adf5c4bcf6292aa121a153768116a08b5d9018246fa7b1cdd8626d3a279bbd0` | 86,743 |
| `eval_H2s_w12_seed1.jsonl` | `88033e8fb0e449a741c9fe21ea528031709a1aa10911eab77e52aa8506b084ca` | 87,839 |
| `eval_H2s_w12_seed2.jsonl` | `cc82532eb5ae30888f06baee9e4e7c65ef670aa2636af774125ce53485950f52` | 87,924 |
| `eval_H2s_w17_seed0.jsonl` | `89b6a7e947200571aeef2f86f80fae3141a93dd485987d430a7de582fac16d31` | 87,080 |
| `eval_H2s_w17_seed1.jsonl` | `f0e7a9bc0c36c1ecd5c0032a15978f9e22e5741addead20a914d2c5744eb239e` | 88,227 |
| `eval_H2s_w17_seed2.jsonl` | `b52187a01be69c92b3d0fb7cfb8a0a639191744549342fb2f03fa2372c8dbbfa` | 86,809 |
| `eval_H2s_w18_seed0.jsonl` | `04a10692e52ac984896ac5ccb5409f7fbc2c47a0a653782e2cfa4efb99a23de4` | 86,229 |
| `eval_H2s_w18_seed1.jsonl` | `7cc8e76f8cb01dcd4f595606ccde3ae6b49a45b1edc127152701845f5269b6b9` | 87,438 |
| `eval_H2s_w18_seed2.jsonl` | `20291b75033ba28b7f6082cad57ea2bd474aeffb4055c3cab3fd5b8719d74cfe` | 87,255 |
| `summary.json` | `8dd6cf2aa1a9d2690cac958c213e3d49d6c56c206bd550a9798ceb3e82e527e6` | 16,171 |
| `d3_gate.json` | `b1c59559b8773d849ef9a7d1f9bcb421589165e0fa62f29ff46d31faeaec5bd2` | 5,033 |
| `ckpt_assertions_start.json` | `942de94ee0c33b1875a284a1732725950e7c84de6a05543a6c10463a1b85d910` | 4,492 |
| `ckpt_assertions_final.json` | `72bdabf5f4ad70c4174c918eda53d10326c81e120e29fd2f5539ef43405dc5ee` | 4,492 |
| `audio_audit.json` | `db0aa62a5f54aeebf2f80cb0fd09c8c8f2147dba159a2fa89e9edf7954fc20a6` | 3,635 |
| `process_residue.json` | `c60206bc7d9630666c54af4a0b04183a162ce4b09b4024ee650492bb3bdbc01d` | 508 |
| `determinism_check.json` | `a4540e0545e19154c8c19fd4a53ee0ad9124bbdf64985f79cf3825b96238f77e` | 1,259 |
| `smoke.json` | `900a9a7b6dc2b0ce1133005c7d7bb1cf328aa27916b621ffc68e8a6e6aec7e5a` | 8,526 |
| `commands.tsv` | `a4d7891698cb2be39686ef0e0face290ebd592319f22ed768ef46c6dafb6458e` | 4,156 |
| `RUN.md` | this file; its sha256 is the one `manifest.json` lists | — |

`smoke.json` is the published smoke run: **seed0/w11 episode 0** of the 20 —
`survived=true`, `bombs_used=3`, `alive_frames=1200`, `frames=1200`,
`decisions_executed=600`, `bomb_emits=9`, `distinct_cands=18`,
`cfg_muted_four_zero=true`, `gate_all_pass=true`, all four audit checks true
(wall 19.5 s, whole command 25.9 s).

## 10. Not done (by instruction)

No other arm (H1/H2/H3), no training, no collection, no labels, no checkpoint writes,
no hyper-parameter search, no pixel arm, no git add/commit, no writes under
`touhou_rl/**`, `third_party/**`, `.tmp_v8c0_hybrid/**`, existing version directories
or the preregistration. This round's engine processes: **0** at exit.
