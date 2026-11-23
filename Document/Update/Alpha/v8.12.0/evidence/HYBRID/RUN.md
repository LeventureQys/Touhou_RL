# v8.12.0 HYBRID-001 — **hybrid deployment**: learned dodge + rule bomb

Executor: `tools/rl/v8c0_hybrid.py` (new this round; imports every frozen module, re-implements none).
Prereg (FROZEN 2026-11-22, before any v8.12.0 evaluation):
`Document/Update/Alpha/v8.12.0/PREREGISTRATION_HYBRID_001.md`,
sha256 `c937c1685af536b2d6aa9ac2cf378a9bfdcfb9603c5889516063be5d391636b4`.
Host: training machine (WSL, `/root/v400-venv`). Engine
`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`, sha256
`691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`.
**Zero training, zero collection.**

---

## 1. What was asked, and what was actually run

| arm | candidate direction | bomb decision | policy checkpoint |
| --- | --- | --- | --- |
| **H1** | v8.6.0 36-action policy, stochastic 36-way sample | the policy's **own bomb head** | `.tmp_v860_bombdag/ckpt/seed{k}/bombdag.pt` |
| **H2** | v8.6.0 policy, `argmax_c max(logit[2c], logit[2c+1])` | **external R5 rule** | same |
| **H3** | v8.3.0 policy, same argmax | **external R5 rule** | `.tmp_v830_bombpol/ckpt/seed{k}/round2.pt` |
| **H2s** | v8.6.0 policy, `softmax_c max(logit[2c], logit[2c+1])` **sampled** | **external R5 rule** | same as H2 |

`H2s` was **added at the supervising agent's direction after the round started**.
Reason: the frozen prereg's H2 uses `argmax`, which makes it deterministic, while
R5's 77.11 % was measured with a **sampling** policy — so H2 and R5 are not
comparable. H2s samples the candidate (temperature 1.0, generator
`seed*1000+7`, one `torch.multinomial` draw per decision) and keeps the bomb bit
external, so its 720-episode pool is 720 independent rollouts and it **is**
comparable to R5. **P1 is judged on H2s.**

Every arm runs on the **same** episode loop: `tools/rl/v830_bombpol.py::run_episode36`
(imported). The only difference between the arms is the `decide` callback, i.e.
where the candidate and the bomb bit come from. That is what makes "the only
change is the bomb source" a measured fact rather than a promise.

Protocol: dev windows `(11, 12, 17, 18)`, PARK at frame `8663`, decision budgets
`{300, 600, 900}`, 2 frames per decision, policy seeds `{0, 1, 2}`, `stochastic`
= `softmax(z / 1.0)` sampled by `torch.multinomial` from one
`torch.Generator().manual_seed(seed * 1000 + 7)`.

The R5 rule is not re-typed: `H2/H2s/H3` call
`tools/rl/v800_bomb.py::rule_gate("staged", …)` (the 3→80 / 2→60 / 1→40 px
thresholds) and `v800_bomb.py::nearest_bullet_distances` (the live `HAZARDS`
reading), with v8.4.0's gate `nearest_all_px < threshold` **and**
`bombs_remaining > 0`.

---

## 2. Gates — all passed

| gate | result | file |
| --- | --- | --- |
| pinned imports (8 files) + prereg | **8/8 + prereg match** | `preflight.json` |
| engine sha256 | `691bfd77…` **match** | `preflight.json`, `ckpt_assertions.json` |
| **D3 fidelity, 5 assertions** | **5/5 PASS**, 18/18 R24 offsets | `fidelity_d3.json` |
| v4.5.0 `round2.pt` (behaviour policy of the v8.3/v8.6 states) | `4abee069…` / `a93746bb…` / `c69960fc…` **3/3 match** | `ckpt_assertions.json` |
| feature dim | 106 | `preflight.json` |
| assert-ckpt phase `start` and phase `final` | **all_pass True** | `ckpt_assertions{,_final}.json` |
| audio (AGENTS.md §9.3) | 32 `th08.cfg` decoded, **32 muted, 0 audible** | `audio_audit.json` |
| own engine processes at the end | **0** | `process_residue.json` |

The D3 gate was run through `tools/rl/v432_fork_arm.py --mode fidelity --seeds 12`
(item 4 alone re-checks 18/18 R24 first-death offsets).

### 2.1 Evaluated checkpoints (sha256, computed this round)

| arm | seed 0 | seed 1 | seed 2 | bytes |
| --- | --- | --- | --- | ---: |
| H1 / H2 / H2s — `.tmp_v860_bombdag/ckpt/seed{k}/bombdag.pt` | `9c6d5d4ac7c9a6120ecd33aa9de7a0762c916413cc8f52e0c82f8ac41f8a483b` | `36a51090d56149dc185d86e5b386f89304e5522c3612d9e919a4092e99f8e84c` | `885dde70c57f18bb7f1e3d0351737d239a52ce064aeb47fda025866bd6e45943` | 39,069 |
| H3 — `.tmp_v830_bombpol/ckpt/seed{k}/round2.pt` | `6e508206b9e951e00dac6215e96a0a08cc16260167421df3e648cdd389a974d3` | `a04366c305938982a71a42af191643943315a319136a86d73744cbec7d1998dc` | `5e9bcb60059d596155b5ae3e8e1c39c1284606cc7b8f7bfecf29fe15ac8d361d` | 39,059 |

---

## 3. Smoke (before any parallel run) — passed

H2, seed 0, **window 13**, **1 episode**, 300 decisions:

```
bombs 3->1   used=2   emits=12   survived=True
bomb_decisions = [0,1,2,3,4,5,6,136,139,140,141,142]
bombs_remaining curve (300 decisions): 3x1, 2x136, 1x163
checks: bomb_path_fired=True  bombs_remaining_decreased=True
        counter_bookkeeping_closes=True  emits_honoured=1
```

`smoke.json` holds the full per-decision trace.

---

## 4. H1 is the v8.6.0 eval grid (measured, not assumed)

The v8.6.0 d300/d600/d900 grid finished on this machine **minutes before this
round started** (all 9 summaries `status=complete`), on the same protocol, the
same checkpoints and the same generator rule. H1 *is* that arm, so re-running
10,800 episodes would have spent the whole round's budget reproducing numbers
that already existed.

Before adopting it, this file's own H1 code path was required to reproduce it:

* `--mode crosscheck`: 20 episodes of H1 (seed 0, d600, window 11) re-run
  through `v8c0_hybrid.py`, compared field by field with
  `.tmp_v860_bombdag/eval/eval_d600_seed0.jsonl` → **20/20 identical** on
  `frames, alive_frames, first_death_offset, first_death_frame, survived,
  decisions_executed, terminal, bombs_start, bombs_end, bombs_used, bomb_emits,
  bomb_decisions, bomb_ignored` (`crosscheck.json`).
* Only then did `--mode adopt-h1` copy the nine v8.6.0 files into
  `eval_H1_d{300,600,900}_seed{k}.jsonl`, recording each source sha256
  (`h1_adoption.json`).

The first crosscheck attempt **failed 1/20** because H1's generator was being
re-seeded inside the episode loop, which made every episode a replay of the
first. That is a bug in this round's code; it was fixed (`process_generator` is
now called once per (arm, budget, seed) process, exactly as
`v860_bombdag.mode_eval` does) and the re-run is the 20/20 above. The failed
attempt is kept in this narrative on purpose.

---

## 5. Results

### 5.1 The P1 verdict arm — **H2s, 600 decisions, n = 720**

3 seeds × 4 dev windows × **60 episodes** = **720 independent episodes**
(there are 711 distinct trajectories in the pool, so the episodes really are
independent draws, not replays).

| window | n | survived | rate |
| --- | ---: | ---: | ---: |
| w11 | 180 | 180 | **100.00 %** |
| w12 | 180 | 177 | **98.33 %** |
| w17 | 180 | 170 | **94.44 %** |
| w18 | 180 | 178 | **98.89 %** |
| **pooled** | **720** | **705** | **97.92 %** (CP95 **96.59 % – 98.83 %**) |

Per seed: s0 235/240 = 97.92 %, s1 237/240 = 98.75 %, s2 233/240 = 97.08 %.

**The cross-window spread is small (94.4 % – 100 %), so the result is not a
w11 artefact.** The weakest window is w17 (170/180).

Against the prereg's references:

| comparison | value | test |
| --- | --- | --- |
| **H2s 97.92 % vs R5 77.11 %** | **+20.81 pp** | one-sided Fisher **p < 1e-8** (2,776/824 for R5 at n = 3,600) |
| **H2s 97.92 % vs H1 29.58 %** | **+68.33 pp** | one-sided Fisher **p < 1e-8** (table `[[705,15],[1065,2535]]`) |

**P1 (≥ 70 %) holds: 97.92 %, lower CP95 bound 96.59 %.** Stop condition
**S-deploy**.

Caveat, stated plainly: H2s's n is 720, not 3,600, so its CP95 is wider than
R5's quoted 3,600-episode figure. The gap (+20.8 pp) is far larger than either
interval, so the limitation does not affect the verdict.

### 5.2 Bomb usage (P4)

| arm | budget | n | distribution (bombs used) | mean |
| --- | ---: | ---: | --- | ---: |
| H1 | 300 / 600 / 900 | 3,600 each | `{3: 3600}` in all three | **3.0000** |
| **H2s** | 600 | 720 | `{2: 8, 3: 712}` | **2.9889** |
| H2 | 600 | 9 (w11 only) | `{3: 9}` | 3.0000 |
| H3 | 600 | 9 (w11 only) | `{3: 9}` | 3.0000 |

H2s's first bomb is **always** pressed on decision **0** (720/720), and its last
bomb lands at median decision **475** (q1 433, q3 526, range 265–597). 7,141
presses were emitted in total but only 2,000-ish were honoured — 4,989 presses
(69.9 %) fell inside an already-active bomb and were correctly counted as
ignored, exactly the v8.4.0 bookkeeping.

**P4 is not a failure of H2s**: at the **600-decision** budget R5 itself uses
`{3: 3540, 2: 60}` (v8.1.0 record) — i.e. three bombs in 98.3 % of episodes.
The prereg §0 figure of 1.36–1.75 bombs belongs to the **300-decision** budget.
H2s spending 2.99 bombs at 600 decisions is therefore *the same economy as R5*,
not worse. (The one place where the hybrid is visibly more economical is the
300-decision budget: the deterministic H2 cells there spent **1.00** bomb and
survived 16/16 — but those cells are partial, see §6.)

### 5.3 Deterministic arms — H2 and H3

H2 and H3 use `argmax` on the candidate and a deterministic distance rule on the
bomb, so **an episode has no random input**: the 300 episodes of one
(seed, window) cell are replays of one trajectory. Measured, not assumed: an
11-episode probe of six cells (66 episodes) produced **exactly one distinct
`(alive_frames, bombs_used, bomb_decisions)` signature per cell**
(`determinism_probe.json`).

**Therefore a deterministic arm's effective n is the number of (seed, window)
cells, not the number of episodes.** Every table below carries
`distinct_trajectories` for this reason.

| arm | budget | episodes | distinct | survived | rate | bombs | windows measured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H2 | 300 | 16 | 3 | 16 | 100.00 % | 1.00 | w11 only |
| H2 | 600 | 9 | 3 | 9 | 100.00 % | 3.00 | w11 only |
| H2 | 900 | 6 | 3 | 2 | 33.33 % | 3.00 | w11 only |
| H3 | 300 | 17 | 3 | 17 | 100.00 % | 1.00 | w11 only |
| H3 | 600 | 9 | 3 | 9 | 100.00 % | 3.00 | w11 only |
| H3 | 900 | 6 | 3 | 0 | **0.00 %** | 3.00 | w11 only |

**These H2/H3 cells are partial and cover window 11 only** (they were
deprioritised by the supervising agent once H2s was added; the deeper budgets
were stopped mid-window). They are reported as measured, but they must **not** be
read as the prereg's pooled 3,600-episode numbers: at 3 distinct cells each they
are 3 independent observations. The only cross-window statement they support is
"on w11, deterministic H2/H3 survive all 600 decisions and die out by 900".

### 5.4 H1 (the v8.6.0 control) — full protocol, 3,600 episodes per budget

| budget | survived | rate | bombs/ep | alive mean | alive median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 300 | 3,598 / 3,600 | **99.94 %** | 3.0000 | 600.00 | 600.0 |
| 600 | 1,065 / 3,600 | **29.58 %** | 3.0000 | 1,026.99 | 1,065.0 |
| 900 | 106 / 3,600 | **2.94 %** | 3.0000 | 1,096.67 | 1,052.5 |

This reproduces the prereg §0 line for v8.6.0 (99.94 % at 10 s, ~29 % at 20 s,
3.00 bombs) — a useful check that the adopted grid is the arm it claims to be.

### 5.5 Side-by-side with the earlier rounds

| arm | d300 | d600 | d900 | bombs/ep @600 |
| --- | ---: | ---: | ---: | ---: |
| v4.5.0 `S`, no bomb (v8.0.0 A0) | 13.44 % | — | — | 0 |
| R5 rule, v4.5.0 `S` (v8.4.0) | 100 % | **77.11 %** | 17.61 % | 3.00 (98.3 % of eps) |
| v8.3.0 own bomb head | 99.94 % | — | 0.61 % | 3.00 |
| **H1 = v8.6.0 own bomb head** | **99.94 %** | **29.58 %** | **2.94 %** | **3.00** |
| **H2s = learned dodge + R5 bomb** | not run | **97.92 %** | not run | **2.99** |
| H2 = argmax dodge + R5 bomb (deterministic) | (w11) 100 % | (w11) 100 % | (w11) 33.3 % | 3.00 |

---

## 6. Deviations from the frozen prereg (all of them)

1. **H2s added after the start** (supervising agent's instruction). It is not in
   the frozen prereg. Justification: the frozen H2 is deterministic, so it
   cannot be compared with R5's sampling measurement; H2s is. **P1 is judged on
   H2s; H2 is reported as the deterministic upper bound.**
2. **Episode counts are below the prereg's 300 per window for every arm except
   H1.** The prereg's §3 cost estimate (32,400 episodes ≈ 2–2.5 h) assumed ~2.7 s
   per episode. Measured throughput on this machine:
   * a **surviving** 600-decision episode is **1,200 frames** and costs
     **≈ 10.5–22 s** wall clock (≈ 2× the v8.6.0 average, because the hybrid
     reads `HAZARDS` once per decision to compute `nearest_all_px` and because
     episodes no longer die early). At that rate 300 episodes × 4 windows per
     cell is **3.7 h per cell**, i.e. ~11 h for H2s alone.
   * **H2s was therefore run at 60 episodes per window (720 total)** — enough
     for a CP95 of ±1.6 pp at the observed 97.9 %.
   * **H2/H3** were stopped after 6–17 episodes per cell (window 11 only) once
     H2s was added; the parent's instruction was to stop them.
   * **H1 carries the prereg's full 3,600 episodes per budget** (adopted grid).
3. **The 300-episode pooled protocol is meaningless for deterministic arms**
   (§5.3). Reporting the pooled 3,600 for H2 would be pseudo-replication; the
   distinct-cell count is reported instead.
4. **H1 adopted rather than re-run** (§4), with a 20/20 crosscheck as the
   precondition.

Nothing else was changed: no training, no collection, no edits to
`touhou_rl/**`, `third_party/**`, any existing version directory or prereg, and
no existing runner was modified (all are imported by path and sha256-pinned).

---

## 7. What this round shows and does not show

**Shows.** With the v8.6.0 candidate head choosing *where to dodge* and the R5
rule choosing *when to bomb*, the 600-decision survival rate is **97.92 %**
(705/720, CP95 96.59–98.83 %) versus **77.11 %** for R5 alone and **29.58 %** for
v8.6.0's own bomb head. The spread across the four dev windows is 94.4–100 %.
Stop condition **S-deploy**.

**Does not show.** That the policy learned resource allocation — the bomb
decision is a hand-written rule (prereg §5). That the hybrid is "a learned
result" — it is a learned component plus a rule. That the game is cleared. That
H2s's 97.92 % can be compared with R5's 77.11 % at equal precision: H2s has
n = 720, not 3,600. And the 300-decision and 900-decision budgets are **not**
measured for H2s, so P3 is **not evaluated** this round.

---

## 8. Independent replication (a parallel run, not mine to certify)

A **separate agent on the same machine ran H2s independently** and published its
own evidence at `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/` (its
`schema`, files and `RUN.md` are its own). It used 20 episodes per
(seed, window) cell and a different loop order (`for seed: for window: for
episode`), so its RNG stream — and therefore its episodes — are **not** the same
draws as this round's 720.

The two runs agree closely, window by window:

| window | this round (n = 180) | HYBRID_H2S (n = 60) | R5 (HYBRID_H2S's reference) |
| --- | ---: | ---: | ---: |
| w11 | 100.00 % | 100.00 % | 81.1 % |
| w12 | 98.33 % | 98.33 % | 79.2 % |
| w17 | 94.44 % | 95.00 % | 70.4 % |
| w18 | 98.89 % | 100.00 % | 77.7 % |
| **pooled** | **97.92 %** (705/720) | **98.33 %** (236/240) | 77.11 % |

Bomb usage also agrees: 2.9889 (this round) vs 2.9958 (HYBRID_H2S) bombs per
episode, with the first bomb on decision 0 in every episode of both runs.

Read this as an **independent replication of the same effect**, not as
extra pooled evidence: the two datasets are not merged, and this round's
`verdict.json` uses only its own 720 episodes. The R5 per-window column is taken
from `HYBRID_H2S/summary.json`, which in turn quotes the supervising agent;
it is not re-measured here.

---

## 9. Files

| file | content |
| --- | --- |
| `preflight.json` | pins, engine gate, the nine checkpoints, R5 thresholds |
| `ckpt_assertions.json`, `ckpt_assertions_final.json` | checkpoint/engine/prereg assertions, phase start and final |
| `fidelity_d3.json` | the D3 gate (5/5, 18/18 R24 offsets) |
| `smoke.json` | the single H2 episode with a per-decision trace |
| `crosscheck.json` | H1's code path vs the recorded v8.6.0 rows (20/20) |
| `h1_adoption.json` | provenance of the adopted H1 grid (source sha256 per cell) |
| `determinism_probe.json` | the 66-episode probe proving the deterministic arms replay |
| `eval_H1_d{300,600,900}_seed{0,1,2}.jsonl` + `.summary.json` | H1, 3,600 episodes per budget |
| `eval_H2s_d600_seed{0,1,2}.jsonl` + `.summary.json` | H2s, 720 episodes |
| `eval_H2_d{300,600,900}_seed{0,1,2}.jsonl` | H2, partial (window 11 only) |
| `eval_H3_d{300,600,900}_seed{0,1,2}.jsonl` | H3, partial (window 11 only) |
| `diagnostics.json` | per-cell / per-window / per-seed counts, means, medians, quantiles, first/last bomb decision, distinct-trajectory counts |
| `bomb_usage.json` | the bomb-usage distribution per cell |
| `verdict.json` | P1–P5, Clopper-Pearson and one-sided Fisher, both on the episode pool and on distinct trajectories |
| `audio_audit.json` | 32 `th08.cfg`, 32 muted, 0 audible |
| `process_residue.json` | 0 own engine processes at the end |
| `commands.tsv`, `contention.tsv`, `exec.sh` | every command with rc/wall, machine contention log, the harness |
| `manifest.json` | sha256 + bytes of every published file |

## 10. Reproduction

```bash
cd /mnt/d/WorkShop/Touhou_RL
./.tmp_v8c0_hybrid/exec.sh preflight
./.tmp_v8c0_hybrid/exec.sh start
./.tmp_v8c0_hybrid/exec.sh fidelity
./.tmp_v8c0_hybrid/exec.sh smoke
./.tmp_v8c0_hybrid/exec.sh crosscheck
./.tmp_v8c0_hybrid/exec.sh adopt-h1
EPISODES=60 ./.tmp_v8c0_hybrid/exec.sh h2s      # 720 independent episodes
./.tmp_v8c0_hybrid/exec.sh diag
./.tmp_v8c0_hybrid/exec.sh verdict
./.tmp_v8c0_hybrid/exec.sh audit
./.tmp_v8c0_hybrid/exec.sh final
./.tmp_v8c0_hybrid/exec.sh publish
```
