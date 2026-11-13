# Getting from 30 decisions/s to 10⁴–10⁶: a literature + engineering review for TH08 RL

**Scope.** This review answers five questions for a project that must train RL on a live, unmodified
60 fps Touhou TH08 instance (≈30 decisions/s, ~900 decisions/episode), and that happens to own a
99.92 %-by-bytes, bit-exact from-source reconstruction of that game plus 148 recorded full-run replays.

**Evidence labels used throughout** — every number below carries one of these:

| Label | Meaning |
|---|---|
| **[M]** | Measured by me in this workspace, on this machine (RTX 4070 SUPER, torch 2.14.1+cu132, Python 3.12.10, numpy 2.5.4). Reproduce with the scripts named in the text. |
| **[P]** | Published/primary claim, quoted or paraphrased from the cited source. |
| **[D]** | **Derived** arithmetic from published or measured quantities. The inputs are cited; the division is mine. |
| **[E]** | **Estimate** — my engineering judgement, no direct measurement. Explicitly not a fact. |

**Hardware note.** The GPU in this machine is an **RTX 4070 SUPER** (56 SMs, 12 GB, 504 GB/s), not a
plain 4070 (46 SMs, 504 GB/s). **[E]** JAX/torch throughput on a plain 4070 should be ≈15–20 % lower
than the numbers below; memory-bound kernels scale with SMs, so the batched-physics numbers are the
most sensitive and the small-MLP numbers the least.

---

## 0. Executive summary

1. **The published successes are 10³–10⁵× faster than this project's live loop, and almost all of
   that comes from *parallelism*, not from a faster per-environment step.** EnvPool reaches
   1.07 M agent-steps/s on 256 CPU cores; Sample Factory reaches 135,893 Atari frames/s on a
   36-core + RTX 2080 Ti box; OpenAI Five consumed "approximately 2 million frames every 2 seconds"
   ≈ 10⁶ frames/s. A single Atari environment, by contrast, steps at a few hundred frames/s.
   **[P]** The gap is *aggregate* parallel throughput.
2. **Faster-than-real-time execution of a real commercial game is standard practice when the game
   exposes a headless/scriptable interface.** StarCraft II via PySC2: "10x real-time is not uncommon
   and can be much higher for simple maps" **[P]**. Super Smash Bros Melee via libmelee: headless
   Dolphin plus gecko codes for fast-forward, "allows the game to run much faster than normal" **[P]**.
   Atari via EnvPool is a C++ emulator inside the training process **[P]**. Nobody in the published
   literature speed-hacks a game *without* a sanctioned headless path — and TH08 has none, which is
   exactly why the reconstruction matters.
3. **This project has already proved the single most important prerequisite:** gameplay state is
   invariant to a **128× time-scale change** — 108 recordings / 12,311,460 calculations reproduce
   bit-exactly on 25 fields, with a scaled `CLOCK_MONOTONIC_RAW` and pixel rasterization disabled
   **[P]** (first-party). TH08 logic is frame-count-driven, not wall-clock-driven. A headless build
   is therefore *mechanically* sound; only its throughput is unknown.
4. **Surrogate simulation "from logs" is not an established, named technique for commercial games.**
   What exists is (a) Dyna/model-based RL, whose central documented failure mode is exactly the one
   that will bite here — model bias exploited by the optimiser (MBPO's own abstract frames the
   trade-off as "the ease of data generation must be weighed against the bias of model-generated
   data") **[P]**; (b) offline RL from fixed logged data, which cannot query counterfactuals; and
   (c) imitation/ghost-style training on recorded behaviour, which is well established as a
   *bootstrap* (AlphaStar's first agent was supervised on human replays) but never as a substitute
   for interaction.
5. **I measured a vectorised player+bullet surrogate on the actual GPU: 1.77 × 10⁷ env steps/s at
   ≈100 bullets on screen** (128-slot pool, batch 4096, CUDA graph), **3.2 × 10⁶ steps/s** with a
   1024-slot pool, and **2.35 × 10⁶ decisions/s end-to-end including an MLP policy forward pass**
   **[M]**. A single-env pure-numpy version already does **34,059 steps/s** — 1,100× the live loop
   **[M]**. At this scale the *policy network*, not the physics, becomes the bottleneck: a small CNN
   costs 0.4–0.5 M
   forwards/s and 0.17 M training samples/s **[M]**.
6. **Ranked recommendation** (§7): (1) build the headless reconstruction first — 2–4 engineer-weeks,
   low risk, expected 10³–10⁴ decisions/s per process and 10⁴–10⁵ aggregate, at *perfect* fidelity,
   and it is the instrument that generates every other option's data; (2) build the vectorised
   surrogate *on top of it* (spawn schedules recorded from the real game) for 10⁵–10⁶ decisions/s;
   (3) train the hybrid curriculum: surrogate → headless reconstruction → live game. A full learned
   surrogate of the whole game is the only path to 10⁶ with visual observations, and it is the
   highest-risk item on the list — do not start there.

---

## Q1. How fast do the successful game-RL setups actually run?

### 1.1 The table

Units matter more than the numbers here: in Atari convention a "frame" is a raw emulator frame and
agents act every 4th frame (frameskip 4), so **decisions/s ≈ frames/s ÷ 4**. I give both columns and
say which convention each source uses.

| System | Year | Reported throughput | ≈ decisions/s | Total experience | Units / caveat | Source |
|---|---|---|---|---|---|---|
| **DQN** (Nature) | 2015 | not stated as steps/s; protocol 50 M frames; "38 days of game experience" | — | 50 M frames (12.5 M decisions) | training wall-clock never published in the paper | [Mnih et al. 2015, doi:10.1038/nature14236](https://doi.org/10.1038/nature14236) |
| DQN, re-stated by EnvPool | — | "DQN takes eight days and 200 million frames to train an agent to play a single Atari game" | **≈290 frames/s → 72 decisions/s** **[D]** | 200 M frames | ambiguity: whether "frames" here are raw or agent steps gives 72–290 decisions/s | [EnvPool §1](https://ar5iv.labs.arxiv.org/html/2206.10558) |
| **Rainbow DQN** | 2018 | 200 M frames in **35 h** (as re-measured by BTR) | **≈1,587 frames/s → 397 decisions/s** **[D]** | 200 M frames | explicit "200 million Atari frames", frameskip 4 | [BTR §1 and §5.1](https://ar5iv.labs.arxiv.org/html/2411.03820) |
| Rainbow DQN, aggregate cost | 2018 | **34,200 GPU-hours** for Atari-57 (Ceron & Castro 2021, cited by BTR) | — | 200 M frames × 57 games | ≈600 GPU-hours per game | [BTR §1](https://ar5iv.labs.arxiv.org/html/2411.03820) |
| **BTR** ("Beyond the Rainbow") | 2024 | **200 M frames in <12 h on one desktop PC**; 0.9 A100-GPU-days for Atari-60 | **≈4,630 frames/s → 1,157 decisions/s** **[D]** | 200 M frames | 64 vectorised envs; 2.9 M-param net | [BTR abstract, §4.1, Table 3](https://ar5iv.labs.arxiv.org/html/2411.03820) |
| **Dopamine** | 2018 | — | — | 200 M frames (standard protocol) | I could **not** verify a primary steps/s or wall-clock figure in this session → **not verified** | [Dopamine paper](https://arxiv.org/abs/1812.06110) |
| **EnvPool** (Atari, C++ backend) | 2022 | laptop 12 cores: **49,439**; workstation 32 cores: **200,428**; DGX-A100 256 cores: **891,286** async / **1,069,922** NUMA+async | same numbers (their "FPS" = `env.step()` with frameskip 4) | benchmark, no fixed total | Python subprocess baseline on DGX: 71,943 | [EnvPool Table 1](https://ar5iv.labs.arxiv.org/html/2206.10558) |
| EnvPool (MuJoCo) | 2022 | DGX-A100: **3,134,287** physics steps/s | — | — | 5 substeps per step | same |
| **Sample Factory** (APPO) | 2020 | **135,893** Atari FPS, **146,551** VizDoom FPS, **42,149** DMLab FPS on 36-core + RTX 2080 Ti | **Atari ≈33,973**; VizDoom ≈146,551 (skip 1 for Duel/Deathmatch = 2) | 2.5 × 10⁹ frames per agent in self-play; 8 agents ≈ "18 years of simulated experience" | "pure simulation" ceiling on the same box: 181,740 Atari / 322,907 VizDoom | [Sample Factory Table 1, §4.3](https://ar5iv.labs.arxiv.org/html/2006.11751) |
| **IMPALA** | 2018 | "24 K FPS for a single machine with 48 CPU cores, and up to 250 K FPS on a cluster with 500 CPUs" | ≈6,000 / ≈62,500 | — | quoted by Sample Factory, not by IMPALA directly in my fetch | [Sample Factory §2](https://ar5iv.labs.arxiv.org/html/2006.11751) ; [IMPALA](https://arxiv.org/abs/1802.01561) |
| **OpenAI Five** (Dota 2) | 2019 | "batches of approximately **2 million frames every 2 seconds**" → **≈10⁶ frames/s**; trained for **10 months** | ≈10⁶ (30 fps game ticks) | — | 1 s wall = ≈33,333 s of game time **[D]** | [Berner et al., arXiv:1912.06680 abstract](https://arxiv.org/abs/1912.06680) |
| **AlphaStar** | 2019 | no steps/s published; 16 TPU v3 per agent, league training over ~10 days/agent | — | — | **partially verified**: the Nature paper is paywalled and I could not fetch the Methods; the 16-TPU figure comes from a secondary wiki | [Vinyals et al., Nature 2019](https://www.nature.com/articles/s41586-019-1724-z) ; secondary: [aiwiki](https://aiwiki.ai/wiki/alphastar) |
| **GT Sophy** | 2022 | real-time 60 Hz racing sim; no env-steps/s in the abstract | — | — | **not verified** — Nature Methods not fetched | [Wurman et al., Nature 2022](https://doi.org/10.1038/s41586-021-04357-7) |
| **Isaac Gym** | 2021 | "2–3 orders of magnitude improvements compared to conventional RL training that uses a CPU based simulator" | — | — | headline claim only; per-task FPS tables are in the (unfetched) PDF | [Makoviychuk et al., arXiv:2108.10470 abstract](https://arxiv.org/abs/2108.10470) |
| **Isaac Lab** | 2024 | — | — | — | **not verified** in this session | — |
| **Brax** | 2021 | JAX; "training of performant policies on common MuJoCo-like tasks **in minutes**" | — | — | no absolute steps/s in abstract | [Freeman et al., arXiv:2106.13281 abstract](https://arxiv.org/abs/2106.13281) |
| **MJX / MuJoCo Playground** | 2025 | "train policies **in minutes on a single GPU**"; MJX, batched on-device | — | — | the technical report's RTX-4090 steps/s table is a PDF I could not fetch | [playground.mujoco.org](https://playground.mujoco.org/) |

### 1.2 What the table actually says

- **Aggregate throughput is bought with parallel environments, not with a faster single environment.**
  Sample Factory's own accounting makes this explicit: on the same machine, "pure simulation"
  (random actions, no network) reaches 181,740 Atari FPS while the full APPO learner reaches
  135,893 — i.e. the *system* is within 75 % of the bare environment engine **[P]**. The engine's
  per-core cost is what it is; the multiplier is the number of cores/instances.
- **The floor is embarrassingly low.** Vanilla DQN's implied rate is **72–290 decisions/s** **[D]**;
  Rainbow's is ~400 **[D]**. Those systems solved 49–57 games. This is the most important single
  fact for the present project: *the live 30.8 decisions/s loop is within one order of magnitude of
  what the original DQN did per environment.* What the field added since 2015 is not a faster game;
  it is 64–65,536 copies of the game.
- **Two orders of magnitude are available on one desktop, if the environment is in-process and
  vectorised.** BTR gets 1,157 decisions/s on one PC **[D]**; EnvPool gets 200 k steps/s on 32 cores
  **[P]**; the vectorised surrogate in §Q5 gets 3.2 M steps/s on one GPU **[M]**.

### 1.3 The project's position, quantified

| Quantity | Value | Label |
|---|---|---|
| Live loop, skip = 1 | **30.8 decisions/s** | **[M]** first-party (`Document/Dialogs/2026-10-18_v1.0.2.md:54`) |
| Live loop, frames | 60.0 fps foreground; 14.3 fps when unfocused | **[M]** same file, line 52 |
| Decisions per 9 h | 10⁶ → 9.26 h | **[D]** |
| Decisions per 4 days | 10⁶ × 10.4 | **[D]** |
| Atari DQN equivalent | 72–290 decisions/s per env | **[D]** from EnvPool §1 |
| Gap to 10⁴ decisions/s | ×325 | **[D]** |
| Gap to 10⁶ decisions/s | ×32,500 | **[D]** |

---

## Q2. Headless / faster-than-real-time simulation of a *real* game

### 2.1 The evidence table

| System | Mechanism | Documented speed | Source |
|---|---|---|---|
| **StarCraft II (PySC2)** | shipped headless Linux client; the API "allows you to control the clock" | "**10x real-time is not uncommon and can be much higher for simple maps**"; 16–22 sim steps/s at normal speed | [PySC2 docs/environment.md](https://github.com/google-deepmind/pysc2/blob/master/docs/environment.md) |
| **Super Smash Bros Melee (libmelee / Slippi Dolphin)** | headless Dolphin (`gfx_backend="Null"`, `disable_audio=True`) + gecko codes for EXI inputs and fast-forward, which "internally disable melee's rendering" | "allows the game to run **much faster than normal**" (no factor published) | [libmelee README](https://github.com/vladfi1/libmelee) |
| **Atari (EnvPool)** | the emulator is re-implemented in C++ inside the training process; no Python, no subprocess | 1.07 M steps/s on 256 cores; **≈2× even for a single environment** vs ALE | [EnvPool §1, §4.1](https://ar5iv.labs.arxiv.org/html/2206.10558) |
| **Gym Retro / libretro** | libretro cores + savestates + deterministic replay; the standard RL wrapper for emulated consoles | — (no measured factor found in this session) | [Gym Retro paper](https://arxiv.org/abs/1809.02839) |
| **MAME (MAMEToolkit)** | MAME + memory reads + frame capture | — (not verified) | [MAMEToolkit](https://github.com/M-J-Murray/MAMEToolkit) |
| **Doom (ViZDoom)** | engine-level headless scenario runner (Doom is open source — the "reimplementation" case) | — (numbers not verified) | [ViZDoom paper](https://arxiv.org/abs/1605.02097) |
| **BizHawk / TAS tooling** | deterministic savestates + frame advance; the basis of tool-assisted speedruns | determinism is the product; speed depends on core | [BizHawk](https://tasvideos.org/BizHawk) |
| **TH08 reconstruction (this project)** | Wine + scaled `CLOCK_MONOTONIC_RAW` (`TH08_REPLAY_CLOCK_RATE=128`) + Mesa `LP_NO_RAST` rasterization bypass; "Game calculation **and draw callbacks** still run" | clock ×128; **traces bit-identical over 12,311,460 calculations** | [REPLAY_TESTING.md:67-74](third_party/th08/docs/REPLAY_TESTING.md), [REPLAY_RESULTS.md](third_party/th08/docs/REPLAY_RESULTS.md) |

### 2.2 The three architectural patterns behind every fast real-game setup

1. **The game ships a headless/server mode** (SC2, Dota 2, Rocket League, CS). This is the cheapest
   path and it is unavailable to TH08.
2. **The game is open source or has a reimplementation** (Doom, Quake, OpenTTD, NetHack, MineRL's
   Java edition, microRTS). You then delete rendering and the clock from the loop.
   **TH08's reconstruction is this case, bought at the price of a full reverse-engineering project
   that has already been paid.**
3. **The platform is emulated and the emulator is in-process** (Atari/EnvPool, libretro/Gym-Retro,
   Dolphin/libmelee, BizHawk). Savestates + determinism give you free episode resets and branching.

**[E]** Pattern 2 is the only one that applies here, and it is *strictly better* than pattern 3 for
this game: an emulator would have to emulate a 1.7 GHz Pentium-4-era x86 plus D3D8 at maybe 10–50×
real time, whereas the reconstruction is native 64-bit C++ with the renderer already swappable.

### 2.3 Measured / documented TH08 facts that de-risk the headless build

| Fact | Evidence |
|---|---|
| 108 replays × 12,311,460 calculations reproduce bit-exactly on 25 fields (incl. RNG generation counts, scores, positions, frame counters) | **[P]** `docs/REPLAY_RESULTS.md:3-8` |
| Same result under a **128× scaled monotonic clock** and no rasterization | **[P]** `docs/REPLAY_RESULTS.md:41-47`, `docs/REPLAY_TESTING.md:67-74` |
| The clock shim only touches `CLOCK_MONOTONIC_RAW` (`clock_id == 4`); "Game code, input records, other clocks … are unchanged" | **[P]** `scripts/replay-clock.c:1-5,23` |
| The main loop is a **fixed-timestep accumulator**: `while (lastFrameTime < curTimestamp) lastFrameTime += 1.0f/60;` and `Sleep(0)` when ahead | **[P]** `src/main.cpp:307-376` |
| Draw chain and calc chain are separate: `RunCalcChain()` then, only if `frameskipConfig <= framesSinceRedraw`, `RunDrawChain()` | **[P]** `src/main.cpp:338-365` |
| All Win32 timing is already funnelled through a small shim on Linux (`Sleep`, `timeGetTime`, `QueryPerformance*`) | **[P]** `src/modern/linux/linux_compat.cpp:447-456` |
| The RNG is **seeded from `timeGetTime()`** at startup | **[P]** `src/Supervisor.cpp:725-727` |
| The retail process holds a named mutex `Touhou 08 App` → one instance per machine | **[M]** first-party (`Document/Dialogs/2026-10-18_v1.0.2.md:50`) |
| Replay input is read from `g_GuiMessageInputCurrent` (`0x0161C674`), not `g_CurFrameInput` | **[M]** first-party (`…v1.0.2.md:81-86`) |
| Player speeds are **data-driven** per shot type: `normalAxisSpeed`, `normalDiagonalSpeed`, `focusedAxisSpeed`, `focusedDiagonalSpeed` | **[P]** `src/Player.cpp:766-786` |

### 2.4 What is *not* known, and how to find out in one day

Nobody has published a clean logic-only frame rate for the reconstruction. The best available floor
is **[D]** from the documented suite wall-time:

```
x86_64 full suite: 4 h 14 min = 15,240 s for 12,311,460 gameplay calculations
→ 808 calculations/s  (≥13.5× real time)
```

…while running Wine + Xvfb + llvmpipe, with a **GDB/ptrace hardware-breakpoint observer attached at
every calculation**, per-recording process launches and menu navigation. That is a floor, not a
ceiling. **[E]** A clean build with rasterization skipped and no observer should land at
**3,000–10,000 logic frames/s** (50–170× real time); the uncertainty is entirely in how much of the
16.7 ms frame budget is draw submission vs. game logic. **A one-day experiment settles it**: add a
`--fps-probe N` flag that runs N frames with `Present()` stubbed and prints frames/s. Everything
downstream depends on this number; measure it before committing to any option.

---

## Q3. Surrogate environments, recorded opponents, and the sim-to-true gap

### 3.1 Is there established literature on distilling a fast simulator from logged transitions?

**No — not under that name, and not for commercial games.** I searched for "environment
distillation", "learned simulator for RL speedup", "two-fidelity simulation RL", "surrogate
environment RL", "simulation from logs". What exists is adjacent, and each adjacent thing has a
documented failure mode:

| Technique | What it buys | Documented limitation | Source |
|---|---|---|---|
| **Dyna** (Sutton 1990) — learn a model from real transitions, plan with it | the canonical "replay your own experience faster" | model error compounds; requires the model to be queried in the same state distribution | [Sutton, "Integrated architectures for learning, planning, and reacting"](https://doi.org/10.1016/B978-1-55860-141-3.50030-4) |
| **MBPO** — short model rollouts branched from real states | SOTA sample efficiency; "matches the asymptotic performance of the best model-free algorithms" | the paper's framing *is* the warning: "the ease of data generation must be weighed against **the bias of model-generated data**"; hence short branches (horizon ≪ episode) | [Janner et al., arXiv:1906.08253 abstract](https://arxiv.org/abs/1906.08253) |
| **World models / Dreamer / MuZero** | learn the dynamics, train or plan inside | succeed where the observation is low-dimensional and the model class matches the true dynamics; no evidence of transfer to a 1,500-bullet danmaku game | [Ha & Schmidhuber](https://arxiv.org/abs/1803.10122), [Hafner et al.](https://arxiv.org/abs/2301.04104) |
| **Offline RL** (BCQ/CQL/IQL/TD3+BC/DT) | learn a policy from a *fixed* logged dataset | cannot query counterfactuals; performance is bounded by dataset coverage; distribution shift is the core problem | [CQL, arXiv:2006.04779](https://arxiv.org/abs/2006.04779) |
| **Recorded-opponent / ghost / replay-based training** | a stationary, cheap opponent or a supervised bootstrap | a fixed opponent is exploitable and does not co-adapt; usable as a *pretraining* stage, not as the final environment | §3.3 |

**Assessment [E].** "Build a fast approximate simulator from logged transitions of a slow
ground-truth environment, train a policy inside it, transfer" is *not* a recipe with a track record
in game RL. The nearest published analogues are (a) model-based RL with short branched rollouts, and
(b) sim-to-real in robotics with domain randomisation — and in both the transfer step is the hard,
measured part. Presenting it as an established technique would be wrong; presenting it as an
unexplored, plausible, high-risk idea is right.

### 3.2 What *is* established: the surrogate-to-true gap is the whole problem

- Isaac Gym's pitch is explicitly a *fidelity-preserving* speedup ("physics simulation and the
  neural network policy training reside on GPU … 2–3 orders of magnitude … compared to …
  a CPU based simulator") **[P]** — the simulator is not approximated, it is the same physics moved
  onto the accelerator. That is the safe version of "faster simulation", and it is what the TH08
  reconstruction is: the *same* game, not a model of it.
- Where the simulator *is* approximate, the literature's answer is domain randomisation plus
  system-identification, and even then transfers are reported as bounded successes, not as
  drop-in equivalence **[E]** (I did not verify a quantitative survey in this session — treat the
  general claim as background, not as a citation).
- For this project the gap is *measurable*, which is the decisive advantage: the reconstruction is
  the ground truth, and any surrogate can be scored against it frame-by-frame on the same replay
  inputs (the project already has the 25-field comparator). **A surrogate whose per-frame state
  error is measured on 10⁶ real frames is a much weaker risk than a learned simulator whose error is
  unknown.**

### 3.3 Recorded opponents, ghosts, and replay-based training

- **AlphaStar** is the strongest precedent for *using* recordings: its first agent was trained by
  supervised learning on human replays before any RL **[P]** ([Vinyals et al., Nature 2019](https://www.nature.com/articles/s41586-019-1724-z)); the commonly quoted figure of
  ~971,000 replays is **secondary-source only** in this session ([aiwiki](https://aiwiki.ai/wiki/alphastar)) and I flag it as unverified.
  The structural lesson is solid: replays buy you a *policy prior and a state distribution*, not a
  dynamics model.
- **Drivatar (Forza)** is the shipped commercial version of "train on recorded human behaviour, then
  race the resulting ghosts": Microsoft's Drivatar system learns driver behaviour offline from
  recorded human laps and runs it as an opponent **[P]** (journalistic primary:
  [Ars Technica, "War Stories: How Forza learned to love neural nets to train AI drivers"](https://arstechnica.com/gaming/2020/09/war-stories-how-forza-learned-to-love-neural-nets-to-train-ai-drivers/)).
  It is *opponent* modelling, not environment modelling.
- **This project already does the replay-as-teacher version at scale**: 108 single-stage fixture
  recordings (12,311,460 calculations, used as the determinism oracle) plus 40 new full-run replays
  (**4,581,131 input frames ≈ 21.2 h**, 39 Lunatic runs), a per-frame expert-action channel that is
  non-zero on a median 99.6 % of frames, BC baselines, and a privileged teacher distilled into a
  vision student **[M]** first-party (`readme.md:118-233`,
  `third_party/th08/docs/REPLAY_RESULTS.md`). The recorded data is being used exactly as the
  literature says to use it.

**Bottom line for Q3 [E]:** the honest framing is *"a hand-built, validated, partial surrogate"*,
not *"a learned simulator"*. The player+bullet dynamics are simple enough to reimplement exactly
(§Q5), which removes the model-bias problem entirely for that subsystem; the enemy/boss behaviour is
what must be replayed, and replaying it is a fidelity choice, not a modelling choice.

---

## Q4. Determinism and replay as an accelerator

### 4.1 The technique that the literature actually endorses

**Go-Explore** is the canonical exploitation of a deterministic engine. Its abstract states the
principle directly: *"solve simulated environments through any available means (including by
introducing determinism), then robustify via imitation learning"*, together with *"(1) remember
previously visited states, (2) first return to a promising state (without exploration), then explore
from it"* **[P]** ([Ecoffet et al., arXiv:1901.10995](https://arxiv.org/abs/1901.10995)). The results
are the ones everyone quotes: >43 k mean on Montezuma's Revenge, >650 k with domain knowledge,
max ≈18 M — "meeting even the strictest definition of superhuman" **[P]**.

The mechanism is exactly "rollout by replay": store a state, restore it, replay the action suffix,
branch. Its cost model is the one that matters here:

```
cost(branch) = restore + replay(prefix) + explore(suffix)
```

If restore is O(1) (savestate / memcpy of the game struct) and replay is at 10³–10⁴ frames/s, a
branch of 100 frames costs ~10–100 ms. **TH08 gives you O(1) restore for free**: the game state is a
handful of global structs (the project already snapshots "player, spellcard, bullet, enemy, item,
effect objects" for its failure diagnostics — `docs/REPLAY_TESTING.md:100-105`).

### 4.2 Determinism in this specific game

| Property | Status | Evidence |
|---|---|---|
| Logic is frame-count-driven, not wall-clock-driven | **proved** under a 128× clock over 12.3 M frames | **[P]** `REPLAY_RESULTS.md:41-52` |
| Replay = input stream only, replayed forward | standard Touhou replay format; validated by the fixture corpus | **[P]** `REPLAY_TESTING.md:12-20` |
| Bit-exact float behaviour on x86_64 | validated: "Floating-point values retain exact bit comparison" | **[P]** `REPLAY_TESTING.md:90` |
| RNG | seeded once from `timeGetTime()` (`Supervisor.cpp:727`); the suite compares RNG *generation counts*, with raw seeds kept as metadata | **[P]** `REPLAY_TESTING.md:87-90` |
| Determinism blockers for a training loop | the time-derived RNG seed, and any `Sleep()`-paced path (`GameManager.cpp:1060-1070`, `GameManagerSetup.cpp:417-427`, `AnmManager.cpp:2432`, `TitleScreen.cpp:3743`) | **[P]** source |

**The practical consequence [E]:** to make the reconstruction a *deterministic* training environment
you must pin the RNG seed (an env var read where `Supervisor.cpp:727` currently calls
`timeGetTime()`), and you must not rely on any path that waits on the wall clock. Both are small,
local changes; the hard part (numeric determinism) is already demonstrated.

### 4.3 The two cheap replay tricks this enables

1. **Rollout-by-replay (spawn-schedule replay).** Run the real game once per (replay, stage); record
   every bullet spawn (frame, x, y, vx, vy, radius) plus enemy/boss spawn events. Thereafter the
   player's *dodging* problem can be trained against that schedule in a vectorised surrogate at
   10⁵–10⁶ decisions/s **[M]** (§Q5), with the schedule acting as the "opponent". This is the single
   highest-leverage idea in this review, and it is only possible because the game is deterministic
   and the replays exist.
2. **State-restore branching.** For credit assignment over long horizons (a death 300 frames after
   the fatal mistake), restore a savestate and re-simulate the suffix with different actions. At
   10³–10⁴ frames/s this is 100–1,000× cheaper than doing it live. Go-Explore's success is the
   evidence that this style of search works when the engine is deterministic **[P]**.

**What replay cannot do [E]:** it cannot answer "what would the boss have done if I had moved left?"
Enemy behaviour that is a function of player state (targeted shots, phase changes on HP, RNG draws
that depend on player actions) will diverge the moment the policy deviates from the recording. This
is the *specific* fidelity risk of the surrogate, and it must be measured, not assumed.

---

## Q5. Vectorised bullet-hell physics: measured on the target GPU

### 5.1 The system I built and measured

`tools/bench_surrogate.py` — a minimal, fully vectorised surrogate of *player + bullets only*:

- field 384 × 448 px; player hitbox radius 0.825 px; circle-vs-circle collision;
- player velocities 4.0 / 2.8284 (fast cardinal/diagonal) and 2.0 / 1.4142 (focused), matching the
  game's data-driven `*AxisSpeed` / `*DiagonalSpeed` fields (**[P]** `src/Player.cpp:766-786`);
- bullets: fixed velocity, linear motion, culled off-field;
- **spawn schedule replayed from a precomputed table** (the "recorded from the real game" part);
  no enemy, no AI, no boss, no RNG;
- state: `[B, C, …]` tensors; one action per step; reward/collision mask computed but no auto-reset.

Two configurations were measured: `C = 1024` slots (the real game's pool is 1537 — **[M]** project
readme: "整池子弹 1537 发") and `C = 256`.

### 5.2 Results — all **[M]**, one RTX 4070 SUPER, torch 2.14.1+cu132

| Implementation | Batch | env steps/s | Notes |
|---|---:|---:|---|
| numpy, CPU, **single env** | 1 | **34,059** | pure-Python per-step overhead dominates (~29 µs/step) — yet already 1,100× the live loop |
| torch, CPU | 1 | 6,126 | |
| torch, CPU | 64 | 79,696 | |
| torch, GPU, eager | 256 | 395,279 | one kernel launch per op |
| torch, GPU, eager | 1,024 | 1,665,899 | |
| torch, GPU, eager | 4,096 | 3,458,637 | |
| torch, GPU, CUDA-graph | 256 | 1,033,581 | graph replay buys 2.6× at small batch |
| torch, GPU, CUDA-graph | 1,024 | 3,119,997 | |
| **torch, GPU, CUDA-graph** | **4,096** | **3,207,837** | 1.28 ms per batched step (488 steps in 623 ms) |
| torch, GPU, CUDA-graph | 16,384 | 2,414,211 | bandwidth-limited |
| torch, GPU, CUDA-graph | 65,536 | 2,377,732 | |
| torch, GPU, CUDA-graph, **C = 256 slots** | 4,096 | **13,636,660** | same code, 4× smaller pool |
| **end-to-end: physics + MLP policy forward** | 8,192 | **2,353,363 decisions/s** | one CUDA graph containing both |
| MLP `344→256→256→16` forward | 4,096 | 34,721,869 fwd/s | privileged-state policy |
| MLP training (fwd+bwd+Adam) | 8,192 | 10,896,124 samples/s | learner can keep up |
| small CNN `4×84×84→16` forward | 512 | 523,303 fwd/s | vision policy |
| **small CNN training** | 256 | **170,575 samples/s** | **the real bottleneck for a vision agent** |

### 5.3 The direct answer: 100 bullets on screen

Because cost tracks the *pool size*, not the live bullet count, I scanned capacity at fixed batch
(4096) with best-of-3 CUDA-graph timing — all **[M]**:

| Bullet slots simulated (C) | env steps/s | ns per env-step | ≈ × the live 30.8 decisions/s |
|---:|---:|---:|---:|
| 64 | 19,288,724 | 212 | 626,000× |
| **128 (≈100 bullets on screen)** | **17,697,550** | **231** | **575,000×** |
| 256 | 13,963,253 | 293 | 453,000× |
| 512 | 7,482,318 | 547 | 243,000× |
| 1,024 | 3,210,230 | 1,276 | 104,000× |
| 1,537 (the game's real pool) | ≈5.1 × 10⁵ **[D, extrapolated]** | ≈1,960 | 16,600× |

The fit is linear in slots with a fixed per-step cost:

```
per-env-step ≈ (100–135 ns fixed) + (1.1–1.3 ns per simulated bullet slot)
```

Practical readings:

- **A 128-slot (≈100 live bullets) surrogate sustains ≈1.8 × 10⁷ player-steps/s on one RTX 4070
  SUPER** — 575,000× the live loop, ~18,000× a hypothetical 1,000 fps headless build. At 900
  decisions/episode that is **≈20,000 episodes per second [D]**.
- **Simulate the bullets that are on screen, not the engine's whole pool.** All 1,537 slots cost ≈8×
  more than 128 for no behavioural gain, and even that conservative choice (extrapolated 5 × 10⁵
  steps/s) clears 10⁴–10⁶ by 1–2 orders of magnitude.
- **Do not oversell the GPU.** A pure-numpy single-environment surrogate at **34 k steps/s** **[M]**
  is already enough for a single-agent loop, and it needs no GPU at all. The GPU matters only when
  you want 10⁵–10⁶ *and* a network in the loop.
- **At 10⁶ decisions/s, the policy network is the binding constraint.** MLP inference (34.7 M/s) and
  MLP learning (10.9 M/s) both clear 3.2 M steps/s, so a state-based agent can run at the physics
  ceiling. A small CNN clears neither: 0.52 M forwards/s and 0.17 M training samples/s **[M]** →
  a vision agent is capped near **1.7 × 10⁵ decisions/s** with a 1:1 update ratio. Still 5,500× the
  live game, and exactly the regime where the reconstruction's own visual pipeline (28 fps capture)
  is hopeless.
- **[E]** A JAX/MJX reimplementation with `jit` + `vmap` should land in the same order of magnitude as
  the CUDA-graph numbers (fused kernels, same bandwidth ceiling); I did **not** measure JAX (not
  installed here). torch CUDA graphs are the closest available proxy, and they eliminate per-op
  launch overhead, which is the main thing `jit` would also buy.

### 5.4 What this benchmark does *not* prove

The surrogate has no enemies, no boss, no lasers, no RNG, no items, no deathbomb, no auto-reset, and
its spawn schedule is synthetic. Its **speed** is a hard measurement; its **fidelity is zero by
construction**. The engineering question is not "can this run fast" (yes, by 10⁴–10⁵×) but "can a
dodging policy trained here survive contact with a real stage" — which is empirically testable
against the reconstruction, and which nobody has published for danmaku games.

---

## 6. Where the 10⁴–10⁶ actually comes from: a cost model for each option

| Option | Mechanism | Decisions/s (per instance) | Aggregate on one 16-core + 4070 box | Label |
|---|---|---:|---:|---|
| **0. Live game (status quo)** | memory-read + key injection, 60 fps | 30.8 | 30.8 (mutex forbids a 2nd instance) | **[M]** |
| **1. Live game, tuned** | larger action repeat, cheaper reads | 30–60 | 30–60 | **[E]** |
| **2. Headless reconstruction, virtual clock, no rasterization, direct input injection** | delete the clock from `GameWindow::Render`, stub `Present`, keep draw callbacks | **1,500–10,000** (frames/s ÷ action-repeat; acting every frame gives the top of the range) | **2.4 × 10⁴ – 1.6 × 10⁵** with 16 processes | **[E]** for the rate; ≥808 frames/s **[D]** is the documented floor |
| **3. Replay-driven vectorised player+bullet surrogate** | this report's benchmark, with spawn schedules recorded from (2) | **1.8 × 10⁷** (C=128); **3.2 × 10⁶** (C=1024); **1.7 × 10⁵** (CNN obs, learner-bound) | same (it *is* the parallel one) | **[M]** |
| **4. Learned surrogate of the whole game** | fit dynamics + policy from logs | 10⁵–10⁶ | 10⁵–10⁶ | **[E]**, high risk, no precedent |
| **5. Hybrid** | train dodge in (3), fine-tune in (2), evaluate in (0) | 10⁶ → 10⁴ → 30 | — | **[E]** |

**The key structural observation [E]:** options 2 and 3 are not competitors. Option 2 is the only way
to *generate* the spawn schedules that option 3 consumes, and option 2 is the only *validator* that
option 3's policies are real. The cheapest correct plan is 2 → 3 → 5, not 3 alone.

---

## 7. Ranked recommendation

### Rank 1 — Headless reconstruction with a virtual clock and rasterization bypass (do this first)

**Why first:** perfect fidelity (it *is* the game), reuses 12.3 M frames of validated replay
evidence, costs the least, and is the precondition for everything else. Expected **10³–10⁴
decisions/s per process**, **10⁴–10⁵ aggregate** across processes **[E]**.

**What it takes (concretely):**

| Step | File / line | Work |
|---|---|---|
| 1. Virtual clock | `src/modern/linux/linux_compat.cpp:447-456` (all of `Sleep`/`timeGetTime`/`QueryPerformance*` is already funnelled here) | return a counter that advances exactly 1/60 per frame; `Sleep*` → no-op |
| 2. Unconditional fixed step | `src/main.cpp:307-376` (`while (lastFrameTime < curTimestamp) …`, `Sleep(0)`) | replace the accumulator with one unconditional tick; delete `Sleep(0)` |
| 3. Null present | `src/main.cpp:379-419` (`Present()`, `TakeScreencaptures()`), `src/modern/linux/d3d8_compat.cpp:463` | stub `Present`, gate screencaptures |
| 4. Keep the draw *callbacks*, skip rasterization | — | already proven: `LP_NO_RAST` profile matched 16,080 demo frames (`REPLAY_TESTING.md:67-74`). Do **not** skip `RunDrawChain()` — ANM/VM callbacks carry gameplay state |
| 5. Skip FPS calibration | `src/Supervisor.cpp:616-706` (`CheckFps`, up to 1,800 `Present` calls + `timeGetTime` windows) | bypass; it also flips `disableVsync` on slow hosts |
| 6. Pin the RNG | `src/Supervisor.cpp:725-727` | seed from an env var, default 0 |
| 7. Direct action injection | `g_GuiMessageInputCurrent` (`0x0161C674`), `g_CurFrameInput` | write the per-frame input word instead of synthesising keystrokes (the replay path already does exactly this) |
| 8. Remove remaining sleeps | `GameManager.cpp:1060-1070`, `GameManagerSetup.cpp:417-427`, `AnmManager.cpp:2432`, `TitleScreen.cpp:3743` | no-op under the virtual clock |
| 9. Multi-instance | retail mutex `Touhou 08 App` | namespace/remove it in the headless build; one process per game instance (all state is global) |
| 10. Observations | existing memory reader | already implemented and cheap (0.019 ms scalar / 0.537 ms for the 6.58 MB bullet pool — **[M]** `…v1.0.2.md:53`) |

**What would break, and how you would find out:**

- **`CheckFps` and the vsync calibration** will happily run for seconds before your first frame; and
  if the virtual clock is "too fast" the calibration logic will tear down vsync state mid-run. Stub
  it (step 5) rather than letting it run.
- **Anything that reads the framebuffer** (`TakeScreencaptures`, snapshot key, 3D background effects
  that need a read-back) must be gated, or you will spend your speed budget on `GetRenderTargetData`.
- **Audio** (`g_SoundPlayer.ProcessQueues()` every frame, MIDI timer threads) must be off; the replay
  suite already runs muted — copy that.
- **X11 dependency.** The current port needs Xvfb; a truly headless build needs the window layer
  stubbed (`src/modern/linux/`). **[E]** 2–5 days of work, and Xvfb is an acceptable fallback at this
  scale.
- **RNG divergence** is silent: two runs will differ unless the seed is pinned (step 6). The replay
  suite survives this only because it compares RNG *generation counts*; a training loop will not be
  so forgiving.
- **Determinism regression risk is low but must be checked**: re-run the 108-replay suite against the
  headless build; every one of the 25 fields must still match. The project already has the tooling.
- **The real unknown is the wall-clock rate.** If the logic is draw-submission-bound, the headless
  build might only reach ~1,000 frames/s and the case for it weakens to "10³ decisions/s, still 30×
  better than today". Measure before building (the `--fps-probe` experiment in §2.4).

**Cost [E]:** 2–4 engineer-weeks to a gym-style headless environment with a determinism gate;
+1 week for process-parallel orchestration. **Risk:** low on mechanism (the clock invariance is
already proved), medium on the resulting speed.

### Rank 2 — Vectorised player+bullet surrogate driven by recorded spawn schedules

**Why second but highest-ceiling:** **3.2 × 10⁶ steps/s measured** (13.6 × 10⁶ with a compacted
pool), 2.35 × 10⁶ end-to-end with a policy net, and it composes with GPU-parallel training. It is the
only route to 10⁶ in this project.

**What it takes:** extract spawn schedules (frame, x, y, vx, vy, r) from replays by running the
*headless reconstruction* over the 148 recordings (this is Rank 1's job); reimplement the player
kinematics and collision exactly (they are simple and data-driven); add reward, auto-reset, and a
vectorised gym API; and — non-negotiable — a **fidelity gate** that replays the recorded human actions
through the surrogate and scores per-frame state error against the reconstruction's trace.

**What would break:** aimed bullets and boss phases that depend on player state (schedule becomes
stale the moment the policy deviates); lasers (not modelled); death/bomb/respawn; item collection;
and the sim-to-true gap on the *policy* side, which is measurable and must be measured. **Cost [E]:**
3–6 engineer-weeks. **Risk:** medium. **Payoff:** 10⁵ (vision) to 10⁶ (state) decisions/s.

### Rank 3 — Hybrid curriculum: surrogate → headless reconstruction → live game

Train the dodge skill in Rank 2 at 10⁶, fine-tune in Rank 1 at 10⁴ with the full enemy/RNG machinery,
validate on the live game at 30. This is the only configuration that is simultaneously fast *and*
faithful, and it maps onto the project's existing BC→PPO→vision-distillation plan. **Cost [E]**:
2–4 engineer-weeks on top of Ranks 1–2. **Risk:** low (it is integration, not invention).

### Rank 4 (rejected) — Learned surrogate of the entire game from logs

**Do not start here.** No published precedent for a commercial game; the documented failure mode
(model bias exploited by the optimiser, MBPO's own framing **[P]**) is exactly the failure mode you
would hit; and you would be unable to distinguish "the policy is bad" from "the simulator is wrong"
without the reconstruction — which is Rank 1 anyway.

### The one-day experiment that should precede all of it

Add `--fps-probe N` to the reconstruction: run N logic frames with `Present()` stubbed, print
frames/s. Three data points decide the whole plan:

| Probe result (logic frames/s) | Decision |
|---|---|
| ≥ 10,000 | Rank 1 is a 30–300× win on its own; do it first, unconditionally |
| 2,000–10,000 | Rank 1 still first (10³–10⁴ decisions/s aggregate), Rank 2 becomes the main event |
| ≤ 1,000 | Rank 1 is a data-generation tool only; Rank 2 is the main event |

---

## Appendix A — Reproducing my measurements

```powershell
python tools/bench_surrogate.py    # physics surrogate: CPU/GPU/vectorisation/policy nets
python tools/bench_learner.py      # fwd+bwd+Adam throughput (MLP and CNN)
```

Environment: `Python 3.12.10`, `numpy 2.5.4`, `torch 2.14.1+cu132`,
`NVIDIA GeForce RTX 4070 SUPER` (56 SMs, 12 GB). All GPU numbers use 20–50 warm-up iterations and
`torch.cuda.synchronize()` around the timed region; CUDA-graph numbers capture the whole N-step loop
into one graph and time `graph.replay()`.

## Appendix B — What I could not verify (stated, not papered over)

1. **Dopamine's** wall-clock per 200 M-frame run — no primary figure found.
2. **AlphaStar's** env-steps/s and TPU-days — the Nature paper is paywalled and unfetchable here;
   the "16 TPU v3 per agent" figure is secondary. Cite it as such.
3. **GT Sophy's** training scale — the Nature Methods section was not accessible.
4. **Isaac Lab / MJX absolute steps/s** — the relevant numbers live in PDFs the fetch tool refuses
   (`application/pdf`); I substituted the abstract-level claims plus my own measurements.
5. **A clean logic-only frame rate for the TH08 reconstruction** — genuinely unknown; §2.4 gives the
   one-day experiment that resolves it.
6. **JAX** — not installed; torch CUDA graphs are my proxy for `jit`-style kernel fusion.

## Appendix C — Source index

Primary, fetched and quoted: [EnvPool](https://ar5iv.labs.arxiv.org/html/2206.10558) ·
[Sample Factory](https://ar5iv.labs.arxiv.org/html/2006.11751) ·
[Beyond the Rainbow](https://ar5iv.labs.arxiv.org/html/2411.03820) ·
[OpenAI Five / Dota 2](https://arxiv.org/abs/1912.06680) ·
[PySC2 environment docs](https://github.com/google-deepmind/pysc2/blob/master/docs/environment.md) ·
[libmelee](https://github.com/vladfi1/libmelee) ·
[Go-Explore](https://arxiv.org/abs/1901.10995) · [MBPO](https://arxiv.org/abs/1906.08253) ·
[Brax](https://arxiv.org/abs/2106.13281) · [Isaac Gym](https://arxiv.org/abs/2108.10470) ·
[MuJoCo Playground](https://playground.mujoco.org/) ·
[Forza Drivatar (Ars Technica)](https://arstechnica.com/gaming/2020/09/war-stories-how-forza-learned-to-love-neural-nets-to-train-ai-drivers/).

First-party (this workspace): `third_party/th08/docs/REPLAY_RESULTS.md`, `REPLAY_TESTING.md`,
`scripts/replay-clock.c`, `src/main.cpp`, `src/Supervisor.cpp`, `src/Player.cpp`,
`src/modern/linux/linux_compat.cpp`; `Document/Dialogs/2026-10-18_v1.0.2.md`; `readme.md`.
