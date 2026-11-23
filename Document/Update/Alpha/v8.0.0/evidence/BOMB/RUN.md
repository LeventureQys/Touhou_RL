# v8.0.0 BOMB-001 — what is a bomb worth, measured by a script rule (A0/A1/A2/A3)

Prereg: [`PREREGISTRATION_BOMB_001.md`](../../PREREGISTRATION_BOMB_001.md) (FROZEN 2026-11-22,
before any v8.0.0 result). Executor: [`tools/rl/v800_bomb.py`](../../../../../../tools/rl/v800_bomb.py).
Status: **the 4-arm grid ran to completion, all gates passed, all raw data published.**
**This file contains no statistical inference** (prereg §5; the P1/P2 tests belong to the
analysis pass).

---

## 0. One-paragraph answer

Adding **one script rule** — *"if the nearest bullet is closer than T px and `bombs_remaining > 0`,
press `BTN_BOMB` on this decision's first frame"* — moves the frozen v4.5.0 `S` policy from
**484 / 3,600** episodes that walk the full 600 frames (A0, 13.4 %) to
**3,600 / 3,600** (A1, A2 and A3, 100 %), with mean alive frames **216.97 → 600.00** and
median **107 → 600**. The bomb really fires: `bombs_remaining` drops **4,880 / 5,734 / 6,288**
times in A1/A2/A3, exactly matching the number of presses the game honoured (decrement latency
measured at **2 frames**), and **0** times in A0. The effect is not "a well-timed bomb": the rule
fires on the **very first decision of every one of the 3,600 episodes** in every bomb arm, and
**~1.4–1.7 of the 3** starting bombs is enough for the whole segment. The threshold (40/60/80 px)
makes **no difference at all** at this budget — the endpoint is saturated.

---

## 1. What was run

| arm | `--bomb-rule` | threshold | checkpoint | windows | episodes | decisions | frames/decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A0** | `none` | — | `.tmp_v450_cf/ckpt/seed{k}/round2.pt` | 11, 12, 17, 18 | 300 / window | 300 | 2 |
| **A1** | `r40` | 40 px | same | same | same | same | same |
| **A2** | `r60` | 60 px | same | same | same | same | same |
| **A3** | `r80` | 80 px | same | same | same | same | same |

- policy: the frozen v4.5.0 `S` checkpoint, `stochastic` = `torch.multinomial` on
  `softmax(z / 1.0)` from **one `torch.Generator` per (arm, seed) process seeded
  `seed * 1000 + 7`** — v4.5.0's own rule (`v450_cf_label.policy_action` imported);
- state arm only (no pixels): `PARK` at frame **8663**, `SPAWN`/`CHILD`/`QUIT`, 4 dev windows;
- 4 arms × 3 seeds × 4 windows × 300 = **14,400 episodes**, all `status: complete`, 0 invalid;
- engine `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
  sha256 `691bfd77…8ef6` (asserted, matches);
- **bomb implementation** (prereg §1): every decision reads the child's `HAZARDS`, computes the
  minimum Euclidean distance (2-D playfield px) from `PLAYER` to the bullet list; if
  `distance < T` **and** `bombs_remaining > 0`, that decision's **first** frame is sent as
  `cand_to_mask(cand, bomb=1)` and its remaining frames as `cand_to_mask(cand, bomb=0)`.
  The mask table is the frozen one (`touhou_rl/v400/gym/actions.py::cand_to_mask`,
  via `touhou_rl/v420/contracts.py::cand_to_mask`); A0 never sets the bomb bit.

### 1.1 Why `run_episode_bomb` is a copy, and how that is proven

`tools/rl/v432_fork_arm.py` may not be modified (round constraint), and its `run_episode`
computes **one** mask per decision (`cand_to_mask(cand)`) with no hook for a per-frame mask. So
`v800_bomb.run_episode_bomb` is that loop **copied line for line**, with exactly one behavioural
difference: the mask sent on each frame of a decision. `--mode equiv` measures the copy against
the original **on the engine**, from two freshly seeded generators:

```
window 12, seed 0, 4 episodes, --bomb-rule none
frozen v432_fork_arm.run_episode   vs   v800_bomb.run_episode_bomb
ep0 frames=252 survived=False actions=126   ep1 600/True/300
ep2 260/False/130                           ep3 269/False/135
=> 4/4 episodes: every compared field identical (frames, alive_frames,
   first_death_offset, first_death_frame, return_decisions, decisions_executed,
   survived, terminal) AND the full 18-candidate action sequence identical
```

(`equiv_check.json`; rc 0.) That is what makes A0 an honest control rather than a second,
untested implementation.

---

## 2. Gates (all passed)

| step | command | exit | wall |
| --- | --- | --- | --- |
| ckpt/engine assertions (`start`) | `v800_bomb.py --mode assert-ckpt --phase start` | **0** | 0.53 s |
| **D3 fidelity** | `v432_fork_arm.py --mode fidelity --seeds 12 --learner frozen` | **0** | 4.60 s |
| smoke (A3 × 4) | `v800_bomb.py --mode smoke --bomb-rule r80 --windows 12 --episodes 4` | 1 → 7 → **0** | see §3 |
| A0 equivalence control | `v800_bomb.py --mode equiv --bomb-rule none` | **0** | 5.59 s |
| 12 eval cells | `v800_bomb.py --mode eval …` ×12 | **0** ×12 | 408–879 s each |
| bomb audit / audio audit / process audit | `--mode audit` / `audio-audit` / `process-audit` | **0** | 0.4 / 0.5 / 0.1 s |
| diagnostics | `--mode diagnostics` | **0** | 0.62 s |
| ckpt/engine assertions (`final`) | `v800_bomb.py --mode assert-ckpt --phase final` | **0** | 0.32 s |

**D3 fidelity (5/5 assertions, `fidelity_d3.json`)** — `1_hello_role_shadow` ✅,
`2_debug_threads_1` ✅, `3_prefix_2165_checkpoints` ✅ (2,165/2,165, 0 mismatches,
final frame 8663), `4_r24_18_constants` ✅ (18/18 `first_death_offset` = R24's table),
`5_park_frame_8663` ✅.

**Checkpoint assertions (`ckpt_assertions.json`, phase `final`)** — the three `S` checkpoints
this round evaluated are the frozen v4.5.0 `round2.pt`, and the mask table / runner / protected
sources did not move:

| ckpt | sha256 | expected prefix | match |
| --- | --- | --- | --- |
| `seed0/round2.pt` | `4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc` | `4abee069` | ✅ |
| `seed1/round2.pt` | `a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4` | `a93746bb` | ✅ |
| `seed2/round2.pt` | `c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01` | `c69960fc` | ✅ |

| protected source | sha256 (16) | same as v7.1.0's recorded start phase |
| --- | --- | --- |
| `tools/rl/v432_fork_arm.py` | `3323550607730854` | ✅ |
| `tools/rl/v450_cf_label.py` | `b7ab0ebeee38df4f` | ✅ |
| `tools/rl/v431_arm_a.py` | `96923930fed5d1df` | ✅ |
| `touhou_rl/v420/contracts.py` | `1357c69c32a74fda` | ✅ |
| `touhou_rl/v400/gym/actions.py` (mask table) | `a67145b2bec7c96b` | mtime 2026-11-08, predates this round |

The working tree was **already dirty before this round** (`git status` lists hundreds of
pre-existing modifications from earlier rounds, none of them created here); the only paths this
round added are `tools/rl/v800_bomb.py` and `Document/Update/Alpha/v8.0.0/**`.

---

## 3. Smoke first (prereg §1: prove the bomb path fires, before 14,400 episodes)

A3 (`r80`), seed 0, window 12, 4 episodes, per-decision trace (`bomb_audit.json` → `smoke`):

| episode | frames | `bombs_remaining` | emits | first emit decision |
| --- | --- | --- | --- | --- |
| 0 | 600 | 3 → 2 | 7 | 0 |
| 1 | 600 | 3 → 1 | 14 | 0 |
| 2 | 600 | 3 → 1 | 14 | 0 |
| 3 | 600 | 3 → 1 | 14 | 0 |
| **total** | 4 × 600 | 12 → 5 | **49** | **0** |

`smoke.checks`: `bomb_path_fired=true`, `bombs_remaining_decreased=true`,
`counter_bookkeeping_closes=true` (7 honoured presses = 7 bombs spent),
`first_emit_decremented=true`, `all_pass=true`.

**Two measured facts that the naive implementation would have got wrong**:

1. **The counter drops 2 frames after the press, not on the next frame.** Frame-by-frame:
   press on frame 1 of the decision → `bombs_remaining` **unchanged** after frame 1, **−1**
   after frame 2. Measured latency over **all 16,902** honoured presses in the grid:
   `latency_frames = {"2": 4880/5734/6288}` — a single value, 2, in every arm. (The first
   version of this round's check tested "after frame 1" and reported 0 decrements while the
   bombs were in fact firing; the check was corrected to compare the count after the decision's
   **last** frame, and the raw per-frame readings are kept in every smoke row.)
2. **A press while a bomb is already active is ignored by the game.** `emits 49 / honoured 7`
   in the smoke, `34,891 / 4,880` in A1, `44,248 / 5,734` in A2, `55,442 / 6,288` in A3. The
   game spends exactly one bomb per accepted press, so `bombs_used == honoured_presses` is an
   exact identity in every arm — that identity is the audit's invariant, not a claim.

---

## 4. Full grid — raw counts, pooled over 4 windows × 3 seeds (n = 3,600 per arm)

**No test, no p-value, no interval here.** `alive_frames` = frames executed before the first
death, **death frame excluded** (R24's rule): `alive_frames == first_death_offset` on a death,
`600` when the episode survived. `frames` = executed frames including the death frame
(`v432_fork_arm.run_episode`'s field). "full 600" = the episode walked all 300 decisions.

| arm | rule | thr | n | alive mean | alive median | q1 | q3 | min | max | **full 600** | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A0** | none | — | 3,600 | **216.9714** | **107.0** | 73.75 | 311.0 | 58 | 600 | **484** (13.44 %) | 3,116 |
| **A1** | r40 | 40 px | 3,600 | **600.0** | **600.0** | 600.0 | 600.0 | 600 | 600 | **3,600** (100 %) | 0 |
| **A2** | r60 | 60 px | 3,600 | **600.0** | **600.0** | 600.0 | 600.0 | 600 | 600 | **3,600** (100 %) | 0 |
| **A3** | r80 | 80 px | 3,600 | **600.0** | **600.0** | 600.0 | 600.0 | 600 | 600 | **3,600** (100 %) | 0 |

A0's per-window means (mean of the three seeds' means) / full-600 counts:
w11 **123.9978** / 23 of 900, w12 **320.9389** / 159, w17 **315.0544** / 271,
w18 **107.8944** / 31. A1/A2/A3 are **600.0 / 900 of 900** in every one of the four windows.

Every one of the 12 bomb cells (3 arms × 3 seeds × 4 windows) is exactly **300/300** — there is
no cell in which the rule failed to reach the full 600 frames.

<details>
<summary>Per (arm, seed, window): alive mean / full-600 (all 48 cells)</summary>

```
A0 s0w11:127.4567/8    s0w12:394.15/101  s0w17:338.8833/106 s0w18:91.5533/5
   s1w11:124.39/4      s1w12:233.1433/33 s1w17:321.0/96     s1w18:98.5233/10
   s2w11:120.1467/11   s2w12:335.5233/25 s2w17:285.28/69    s2w18:133.6067/16
A1 s0w11..s2w18: 600.0/300 in all 12 cells
A2 s0w11..s2w18: 600.0/300 in all 12 cells
A3 s0w11..s2w18: 600.0/300 in all 12 cells
```
</details>

### 4.1 The control is exact: A0 **is** v4.5.0's recorded stochastic dev evaluation

`diagnostics.json → a0_vs_v450`: for each seed, A0's 1,200 episode rows are compared with
`.tmp_v450_cf/eval/seed{0,1,2}/eval_stochastic_dev.jsonl` on
`(window, episode, frames, first_death_offset, return_decisions, survived, decisions_executed)`:

| seed | episodes compared | episodes with differences | identical |
| --- | --- | --- | --- |
| 0 | 1,200 | **0** | ✅ |
| 1 | 1,200 | **0** | ✅ |
| 2 | 1,200 | **0** | ✅ |

So A0 is not merely "a run with the bomb disabled": it reproduces the frozen v4.5.0 evaluation
bit for bit (3 × 1,200 = 3,600 episode records).

---

## 5. Bomb audit (prereg §2 P3 — descriptive only)

| arm | episodes | presses (emits) | honoured presses = bombs spent | ignored presses | episodes that bombed | `bombs_used`/ep mean · median · max | episodes ending at 0 bombs | exhaustion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 3,600 | 0 | 0 | 0 | 0 | 0 · 0 · 0 | 0 | 0 % |
| A1 (40 px) | 3,600 | 34,891 | **4,880** | 30,011 | 3,600 (100 %) | **1.3556** · 1 · 3 | 64 | **1.78 %** |
| A2 (60 px) | 3,600 | 44,248 | **5,734** | 38,514 | 3,600 (100 %) | **1.5928** · 2 · 3 | 196 | **5.44 %** |
| A3 (80 px) | 3,600 | 55,442 | **6,288** | 49,154 | 3,600 (100 %) | **1.7467** · 2 · 3 | 335 | **9.31 %** |

- **budget effect** (presses/episode, mean): A1 9.69, A2 12.29, A3 15.40 — the wider threshold
  presses more often, but only ~1 in 8.8 presses is honoured (the game ignores the rest while a
  bomb is active), so the wider threshold mostly **wastes input**, not bombs;
- **`bombs_used` distribution** (episodes): A1 `{1: 2384, 2: 1152, 3: 64}`,
  A2 `{1: 1662, 2: 1742, 3: 196}`, A3 `{1: 1247, 2: 2018, 3: 335}` — every episode starts with
  **3** bombs (the parked frame-8663 template), so the maximum spendable is 3;
- **decisions below threshold** (all 1,080,000 measured decisions per arm):
  A1 35,515, A2 46,315, A3 58,968. Restricted to *lethal* bullets the same counts are
  4,755 / 6,220 / 8,017 — see §7.2;
- **audit invariants** (`bomb_audit.json → arms.{A1,A2,A3}.invariants`, all true):
  `bomb_path_fired`, `bombs_remaining_decreased`, `counter_bookkeeping_closes`
  (`bombs_used == honoured presses`), `emits_at_least_used`, `used_at_most_eight_per_episode`,
  `rule_fired_somewhere`. A0's invariants: `no_bomb_emitted`, `bombs_used_zero`,
  `bombs_untouched` — all true.

### 5.1 The mechanism is visible in the data: the fire happens at decision 0

`first emit decision` over all 3,600 episodes of every bomb arm: **min = 0, median = 0,
max = 0**. In other words the parked frame-8663 state already has a bullet **21.28 px** from the
player (smoke trace; lethal and non-lethal identical there), every bomb rule fires on the first
decision, and after that bomb the nearest *lethal* bullet is **103–110 px** away (smoke trace).
That single bomb carries the episode to frame 600 in ≥ 66 % of episodes (A1: 2,384/3,600 episodes
spend exactly one bomb).

Consequences, stated plainly:

- this round measures **"a bomb at the start of the segment"**, not "good bomb timing";
- because every episode starts with only 3 bombs and the endpoint is 600 frames, the design
  **cannot see a threshold effect** — A1/A2/A3 all sit on the ceiling, so the 40/60/80 px factor
  is **not resolved** by this round (a floor-effect/ceiling-effect situation, not evidence that
  the threshold does not matter in general).

---

## 6. Deviations and corrections (complete list)

1. **`run_episode_bomb` is a copy of `v432_fork_arm.run_episode`** (that file may not be
   modified this round). Proven equivalent by `--mode equiv` (§1.1, 4/4 episodes, fields **and**
   action sequences). A0 runs through the same copy, so no arm gets a different code path.
2. **The smoke block in `bomb_audit.json` comes from a smoke re-run.** The pre-run smoke did run
   first (08:20:32 → 08:21:34, three attempts, rc 1 → 7 → 0; `logs/smoke-a3.out`,
   `commands.tsv`), and reported the same thing (49 emits / 7 honoured / 7 bombs / first emit at
   decision 0). While merging it into `bomb_audit.json` a bug in the merge dropped the block
   (**fixed**: `--mode smoke` now also writes `smoke/bomb_smoke.json`, which the audit reads).
   The block that ships is the **re-run** (08:37:38, `logs/smoke-a3-rerun.out`, identical
   numbers) — the shipped JSON is a live measurement, but it is not the pre-run one.
3. **The first smoke check was wrong and was corrected before the grid.** It tested
   `bombs_remaining` after the decision's *first* frame and therefore reported
   `decremented=0` while bombs were in fact firing (latency is 2 frames). Raw per-frame readings
   (`emit_remaining_after_first_frame`, `emit_remaining_after_last_frame`,
   `emit_latency_frames`) are kept in every episode row so the correction is auditable.
4. **`--mode assert-ckpt --phase start` and `… final` write the same file**, so
   `ckpt_assertions.json` ships the **final** phase; the start check (rc 0, 08:20:23) is recorded
   in `commands.tsv` and `logs/assert-start.out`. (Same shape as v7.1.0.)
5. **Distance scope.** The rule tests the minimum over **all** `HAZARDS` bullets — the literal
   reading of prereg §1 ("从 `HAZARDS` 的子弹列表里取与玩家位置的最小欧氏距离"). The lethal-only
   minimum is recorded on every decision as well, so the alternative reading can be analysed
   without re-running anything (§7.2).
6. **Parents are not closed between windows** (they are closed at the end of each cell) — the
   frozen `v450_cf_label.mode_eval` shape, unchanged, so 4 parent processes exist per cell at
   most. It does not affect any recorded field.
7. **Nothing else was touched**: no change to `touhou_rl/**`, `third_party/**`,
   `tools/rl/v432_fork_arm.py`, `tools/rl/v450_cf_label.py`, `tools/rl/v431_arm_a.py`, any
   earlier version directory or prereg, no `git add`/`commit`, no deleted or moved asset.

---

## 7. What this round does **not** claim

### 7.1 Prereg §5, verbatim in effect

- **Not** "the policy learned to bomb" — the bomb is an **external script rule**; the policy
  never chooses it;
- the bomb arms are **not** "the same policy improved": they are the same frozen checkpoint
  under an **added external intervention**, so they must not be pooled with the 18-action
  historical numbers;
- **no** pixel-arm conclusion (no GL path here);
- the 12 (seed, window) cells are **not** independent samples and are **not** aggregated as if
  they were;
- **no statistics**: no Fisher, no permutation, no p-value, no Holm correction, no confidence
  interval, no "significant" wording. P1/P2 (`full-600` count, mean alive frames) are **not
  decided here** — the raw counts are above and the tests belong to the analysis pass.
  Read against prereg §2 this round's raw counts are *consistent with* **S-ok** (a bomb arm's
  raw `full-600` count is 3,600 vs A0's 484) and *nothing in it looks like* **S-harm** — but the
  prereg's endpoint tests, including the three Holm-corrected comparisons, were **not run**, so
  no stop condition is declared by this file.

### 7.2 Open items this round exposed (for the next prereg, not conclusions)

- **Ceiling.** 100 % survival at 300 decisions / 600 frames cannot distinguish A1/A2/A3.
  A next round needs either a longer segment (more frames), an arm that disallows the
  decision-0 bomb, or a per-episode bomb budget below what the rule wants.
- **Non-lethal bullets dominate the distance.** With the all-bullets reading, 35,515 (A1) to
  58,968 (A3) decisions are "below threshold" but only 4,755–8,017 are below it when restricted
  to lethal bullets; after the decision-0 bomb the nearest lethal bullet is > 100 px while
  non-lethal slots sit at 15–26 px, so the rule keeps pressing into an ignored input.
  A "lethal-only + not-already-bombing" rule is the obvious next variant (and is the direction
  prereg §2.1 S-harm anticipates: "只在无路可走时放").
- **`bombs_remaining` is 3 at frame 8663** for every window and seed — the template state, not a
  choice of this round.

---

## 8. Files in this directory

| file | what it is |
| --- | --- |
| `eval_A{0,1,2,3}_seed{0,1,2}.jsonl` | **raw per-episode rows** (12 files, 14,400 rows): frames, `alive_frames`, `first_death_offset`, `survived`, `return_decisions`, the bomb audit of that episode (`bomb_emits`, `bomb_decisions`, `bomb_frames`, `emit_remaining_before`, `emit_remaining_after_first_frame`, `emit_remaining_after_last_frame`, `emit_decremented`, `emit_latency_frames`, `bombs_start/end/used`, `min_nearest_px`, `min_nearest_lethal_px`, `decisions_below_threshold*`), ckpt sha256, generator seed |
| `eval_A*_seed*.summary.json` | per-cell summaries (per window: alive-frames quantiles, full-600, bomb totals, gate prefix, cfg sha256) |
| `bomb_audit.json` | §3 smoke block (with the per-decision trace) + §5 per-arm audit, per (seed, window), histograms, invariants |
| `diagnostics.json` | offline per-arm / per-(seed,window) / pooled counts, means, medians, quantiles; the A0-vs-v4.5.0 reproduction check; no statistics |
| `equiv_check.json` | the A0 loop-equivalence control (§1.1) |
| `fidelity_d3.json` | the D3 5-assertion gate (18/18 R24 constants) |
| `ckpt_assertions.json` | engine sha, the three checkpoint sha256, protected-source sha256 (phase `final`) |
| `audio_audit.json` | AGENTS.md §9.3: every `th08.cfg` this round created — **51 files, 51 muted, 0 audible** |
| `process_residue.json` | own engine processes alive at the end: **0** |
| `commands.tsv` | every command with exit code, wall time, start/finish |
| `logs/*.out` | stdout/stderr of every step |
| `scripts/exec.sh`, `status.sh`, `finalize.sh`, `tables.py`, `numbers.py`, `inspect_smoke.py`, `show.py` | the harness, the publish/finalise step and the read-only helpers used above |
| `manifest.json` | sha256 + bytes of every file in this directory |

### 8.1 Environment (AGENTS.md §9.3, audio off in three places)

- `SDL_AUDIODRIVER=dummy`, `DISPLAY`/`WAYLAND_DISPLAY` unset in the harness **and** in
  `v450_cf_label.prepare_env()` inside every engine process (recorded as `launch_env` in each
  summary: `{"SDL_AUDIODRIVER": "dummy", "DISPLAY": null, "WAYLAND_DISPLAY": null}`);
- each private data dir's `th08.cfg` is regenerated by `tools/rl/v400_make_cfg.py` and
  `--inspect`ed before the engine starts, refusing to run unless `musicMode` / `playSounds` /
  `musicVolume` / `sfxVolume` are all 0 (`cfg.muted_four_zero = true` in all summaries);
- post-hoc audit of this round's own cfgs: **51 / 51 muted, 0 audible**
  (`audio_audit.json`, written to this round's own path — the shared audit JSON of earlier
  versions was not read or written).
