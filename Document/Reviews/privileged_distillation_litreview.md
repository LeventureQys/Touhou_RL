# Privileged distillation, demonstration bootstrapping, offline→online RL, shaping and shielding: what the literature actually supports for a pixel-only bullet-hell policy

**Setting under review.** TH08 (Touhou 8), 60 fps, decision every 2 frames → 30 decisions/s, 18 discrete actions (9 directions × focus on/off). Deployment: 3 cropped image regions + own action history. Training: full memory state (exact bullet positions/velocities, player position). Scripted privileged expert: beam search over linear bullet extrapolation, ~6500 frames survived; uniform-random ~2420; current pixel student ~1766; target >4000. Sparse reward (3 deaths ends the episode). Single real instance, ~30 decisions/s, no parallel envs. Current method: behavioural cloning / distillation of per-frame candidate values, plateaued at 0.4% closed-loop gain over 18 rounds.

**Verification convention used throughout.** Every number below is either (a) quoted from a primary source I fetched, with the URL, or (b) explicitly marked `[NOT VERIFIED]`. Where a paper's evidence is *only* in a learning-curve figure (no table), I say so, because that materially weakens the claim. Section 7 is a ledger of what I could not confirm.

---

## 1. Privileged information / asymmetric learning

### 1.1 Learning by Cheating — Chen, Zhou, Koltun, Krähenbühl (CoRL 2019), arXiv:1912.12294
URL: https://arxiv.org/abs/1912.12294 · full text: https://arxiv.org/html/1912.12294v1

**How the teacher is trained.** The privileged agent receives a ground-truth bird's-eye-view map `M ∈ {0,1}^{W×H×7}` (one channel each for road, lane boundaries, vehicles, pedestrians, traffic lights) anchored at the agent's position, plus speed `v` and a high-level command `c`. It predicts `K` future waypoints as heatmaps + differentiable soft-argmax, and is trained by **plain behaviour cloning** (L1 on ground-truth future waypoints) on 100 expert trajectories — "about 157K frames ... and 4 hours of driving" at 10 fps — with rotation/shift augmentation applied to the map and propagated to the waypoints. **The teacher is not RL-trained and never sees an image.**

**Exactly how the student is distilled.** Two stages, and the distinction matters:
1. *Off-policy:* L1 between the student's predicted waypoints (projected through a fixed closed-form perspective transform) and the teacher's map-frame waypoints, "on the same set of states as in the first stage" — i.e. **the expert's state distribution**, sampled with a loss-prioritised resampling scheme. The student is supervised on **all four command branches simultaneously** ("white-box" supervision), because the teacher is conditional and can be queried for every branch.
2. *On-policy:* "we train the sensorimotor agent online via DAgger [21], using the privileged agent as an oracle" — i.e. **the student's own state distribution**, with the teacher queried at every state the student reaches.

Two design details that are easy to miss and probably load-bearing: the student imitates **waypoints (an intermediate spatial representation), not the low-level steering/throttle/brake actions**; and the teacher is queried as a *function* at arbitrary states, not replayed from a fixed dataset.

**Key quantitative result — the ablation is the whole paper for your purposes.** Table 1, CoRL2017 "navigation" condition, test town, test weather (success %):

| Supervision | white-box | on-policy | Success |
|---|---|---|---|
| Direct (image → action from expert data) | – | – | **20** |
| Two stage | – | – | **16** |
| Two stage | – | ✓ | **64** |
| Two stage | ✓ | – | **96** |
| Two stage | ✓ | ✓ | **100** |

Headline results: "for the first time, 100% success rate on all tasks in the original CARLA benchmark", "advancing the state of the art by 18 percentage points (absolute) in the hardest, dense-traffic condition", and "reduces the frequency of infractions by at least an order of magnitude in most conditions" (abstract).

Also recorded, and important: "The second stage alone – without pre-training on the original trajectories – works equally well in the final accuracy. However, the online training with DAgger is slower than training on pre-existing trajectories, so we use the first stage to accelerate the overall training process."

**Relevance: HIGH — and diagnostic.** Read the table again: *two-stage off-policy distillation of a privileged teacher scores 16, i.e. worse than plain direct imitation at 20.* The gains come from (i) querying the teacher as a white-box function over all action/command branches at each visited state (16 → 96) and (ii) collecting states from the **student's** distribution (16 → 64). Your pipeline (off-policy distillation of per-frame candidate values on a fixed/expert-derived state distribution) is the `16` cell of this table. LBC's own ablation predicts exactly the plateau you observe, and says the fix is not more rounds of the same loss but a change in *whose states you supervise* and *how much of the teacher's output you distil per state*. Note also that the teacher here is a BC clone of a scripted expert — structurally identical to your setup (scripted privileged expert → learned teacher → pixel student).

### 1.2 Asymmetric Actor Critic — Pinto, Andrychowicz, Welinder, Zaremba, Abbeel (arXiv:1710.06542, 2017)
URL: https://arxiv.org/abs/1710.06542 · full text: https://arxiv.org/html/1710.06542v1

**Method.** DDPG + Hindsight Experience Replay, multi-goal. The replay buffer stores **both** `(s_t, o_t, a_t, r_t, s_{t+1}, o_{t+1}, g, g^o)`. The critic is trained on full states `{s,a,r,s',g}`; the actor is trained on rendered observations `{o,a,r,o',g^o}` — "since the critic works on full state, it learns the state-action value function much faster, which also allows for better updates for the actor." At test time only the actor is used. They also propose a **bottleneck**: an auxiliary loss making an intermediate actor layer predict the full state ("constrain one of the actor network's intermediary layers to predict the full state"). Environments: Particle, Reacher (2D, 100×100×3), Fetch Pick (3D, 100×100×4 RGBD); real-robot picking / forward pushing / block move with domain randomisation.

**Key quantitative result.** `[NOT VERIFIED — the results are learning curves]`. The results section states "asymmetric inputs for training outperforms symmetric inputs by significant margins" and introduces Fig. 5 with "The shaded region corresponds to ±1 ..." but I did not retrieve any success-rate table; the fetch truncated at the figure. **Do not quote a number for this paper unless you obtain the PDF.**

**Relevance: HIGH (mechanism), MEDIUM (evidence).** The mechanism is the cheapest one available to you and costs nothing at deployment: keep a critic (or Q-head) on **memory state** during training, and let the pixel actor be trained by the critic's gradient. That is precisely the "privileged critic" half of your architecture, and it is the standard alternative to distilling a teacher's actions. The bottleneck idea is a free auxiliary loss you can add to the student now: predict the privileged state (bullet occupancy grid / player position) from the student's features.

### 1.3 Learning Quadrupedal Locomotion over Challenging Terrain — Lee, Hwangbo, Wellhausen, Koltun, Hutter (Science Robotics 5(47):eabc5986, 2020), arXiv:2010.11251
URL: https://arxiv.org/abs/2010.11251 · full text: https://arxiv.org/html/2010.11251v1

**Method (the most directly transferable parts).** Teacher: MLPs over robot state plus privileged `x_t` — "noiseless information that we receive directly from a physics engine ... foot-ground interactions such as terrain profile, foot contact states and forces, friction coefficients, and external disturbance forces", with terrain represented by "the elevation of 9 scan points around each foot ... along a circle with a 10 cm radius". **The teacher is trained by RL, not by imitating an expert**: "we do not rely on expert demonstrations to train the privileged policy; rather, the teacher policy is trained via reinforcement learning." Student: a temporal convolutional network over a history of proprioceptive measurements, trained by **supervised imitation of both the teacher's latent embedding `l̄_t` and its action `ā_t`**. Deployed zero-shot, "with no fine-tuning".

**Key quantitative results (Table 1, natural terrain, average speed m/s):** ours 0.452 vs baseline 0.199 on moss; 0.338 vs 0.197 on mud; 0.248 vs (baseline failed) on vegetation. DARPA Subterranean Challenge: "drove two ANYmal-B robots in four missions of 60 minutes. The controller exhibited a zero failure rate throughout the competition." With an unsimulated 10 kg payload (22.7% of body weight) it still traversed steps "up to 13.4 cm", and "heading error ... stays within 10°"; the baseline "is incapable of traversing any steps under any command speed with the payload". Baseline's catastrophic failures were excluded from the average-speed table because "when the baseline fails, it is reset by a human operator".

**The single most useful sentence in the paper for you:** "We have found that training a rough-terrain locomotion policy directly via RL was not successful: the supervisory signal was sparse and the presented network failed to learn locomotion within reasonable time budgets."

**Relevance: HIGH.** Three transferable facts: (i) sparse-reward direct RL on a hard control problem failed for a well-resourced lab, which is evidence *against* "just run PPO on the 30 Hz environment and wait"; (ii) the student was distilled on the teacher's **latent** as well as its action — a dense, well-shaped regression target that is much easier than 18-way action matching; (iii) the privileged information that mattered was *contact/geometry near the agent*, i.e. the analogue of "bullets near the player", which is exactly what your 3 crops should be engineered to contain.

### 1.4 Learning to See before Learning to Act — Yen-Chen, Zeng, Song, Isola, Lin (ICRA 2020), arXiv:2107.00646
URL: https://arxiv.org/abs/2107.00646

**Attribution correction:** the paper with this exact title is **Yen-Chen et al., ICRA 2020**, arXiv:2107.00646 (Andy Zeng is second author). I found no 2018 paper with this title. The arXiv comment reads "Accepted to ICRA 2020".

**Method.** Pre-train a vision network on a passive vision task, then transfer parameters into an affordance-prediction network for manipulation; keys are *which* parts to transfer and that "outputs of standard vision models highly correlate with affordance maps". Fine-tune with a small amount of robot experience.

**Key quantitative result (abstract, verbatim):** "With just 10 minutes of suction experience or 1 hour of grasping experience, our method achieves ~80% success rate at picking up novel objects", and "this can result in successful zero-shot adaptation, where a robot can pick up certain objects with zero robotic experience."

**Relevance: MEDIUM (concept), LOW (numbers).** It supports a specific decomposition you may be under-using: *first* train the perception stack on a dense supervised task with privileged labels (predict bullet positions/danger from the 3 crops), *then* learn control. Your privileged state gives you unlimited free labels for the perception stage; this is a stronger use of it than per-frame action distillation. But the reported gains are for manipulation and do not transfer quantitatively.

### 1.5 Two more privileged-teacher references worth knowing (evidence not audited here)
- **RMA: Rapid Motor Adaptation** (Kumar, Fu, Pathak, Malik, RSS 2021) — two-stage: privileged base policy + an adaptation module regressing the privileged latent from proprioceptive history. `[NOT VERIFIED in this review — I did not fetch it]`.
- **Learning agile and dynamic motor skills for legged robots** (Hwangbo et al., Science Robotics 2019) — the earlier ETH teacher/student scheme that Lee et al. build on. `[NOT VERIFIED in this review]`.

---

## 2. Demonstration bootstrap + RL fine-tuning ("start from a mediocre imitation, then exceed it")

### 2.1 DQfD — Hester, Vecerik, Pietquin, Lanctot, Schaul, et al. (AAAI 2018), arXiv:1704.03732
URL: https://arxiv.org/abs/1704.03732 · full text: https://ar5iv.labs.arxiv.org/html/1704.03732

**Method.** Four losses, pre-train on demos first, then interact with demos **kept permanently in the replay buffer** and given a priority bonus so the demo/self-generated ratio self-regulates:
1. 1-step double-Q loss; 2. n-step double-Q loss with `n = 10` ("helps propagate the values of the expert's trajectory to all the earlier states");
3. a **large-margin supervised classification loss** `J_E(Q) = max_{a∈A}[Q(s,a) + l(a_E,a)] − Q(s,a_E)`, where `l = 0` for the expert action and positive otherwise — "This loss forces the values of the other actions to be at least a margin lower than the value of the demonstrator's action. Adding this loss grounds the values of the unseen actions to reasonable values";
4. L2 on weights.

The rationale for (3) is exactly your failure mode: "many state-actions have never been taken and have no data to ground them to realistic values. If we were to pre-train the network with only Q-learning updates towards the max value of the next state, the network would update towards the highest of these ungrounded variables and the network would propagate these values throughout the Q function."

**Key quantitative results (all from the fetched full text/abstract).**
- 42 Atari games; human demos "range from 5,574 to 75,472 transitions per game".
- "DQfD performed better than PDD DQN on the first million steps on 41 of 42 games."
- "On average, PDD DQN does not surpass the performance of DQfD until 83 million steps into the task and never surpasses it in mean scores."
- "it learns to play better than the best demonstration episode in 14 of the games"; "DQfD out-performs pure imitation learning in mean score on 39 of 42 games"; SOTA on 11 of 42.
- "In comparison, pure imitation learning is worse than the demonstrator's performance in every game."
- For calibration of effort: "DQN learns from over 200 million frames" (their own comparison point).

**Relevance: HIGH.** This is the single best-documented recipe for "I have a decent scripted expert and a sparse reward". Three things transfer directly: the margin loss (a *relative* ranking objective over the 18 actions, much closer to what closed-loop survival needs than a per-frame MSE on values); demo retention with a priority bonus (fixes catastrophic forgetting of rare survivable states); and n-step returns (propagates sparse death signals backwards — directly relevant when 3 deaths end the episode and the useful signal is a long, delayed "you died"). The 83M-frame number is the honest cost of catching up *without* demos.

### 2.2 Kickstarting — Schmitt, Hudson, Zidek, Osindero, Doersch, Czarnecki, Leibo, Kuttler, Zisserman, Simonyan, Eslami (arXiv:1803.03835, 2018)
URL: https://arxiv.org/abs/1803.03835

**Method.** Add a distillation term to the RL loss so a *student* is trained against one or more *teachers* while still maximising its own return; the mixing weight is regulated so that the student is not pinned to the teacher. "Our method places no constraints on the architecture of the teacher or student agents, and it regulates itself to allow the students to surpass their teachers in performance." Benchmark: DMLab-30 (multi-task, pixel input).

**Key quantitative result (abstract, verbatim).** "with the kickstarted agent matching the performance of an agent trained from scratch in almost 10x fewer steps, and surpassing its final performance by 42 percent."

**Relevance: HIGH.** This is the cleanest published statement of the property you actually need: a distillation term that is **annealed/regulated so the student can exceed the teacher** rather than converge to it. If your distillation weight is fixed, then by construction the best you can do is reproduce the teacher's per-frame values on the visited distribution — you have no mechanism for exceeding it, and you should expect exactly the plateau you see.

### 2.3 VPT — Baker, Akkaya, Zhokhov, Huizinga, Tang, Ecoffet, Houghton, Sampedro, Clune (arXiv:2206.11795, 2022; commonly cited as NeurIPS 2022)
URL: https://arxiv.org/abs/2206.11795

**Method.** Semi-supervised imitation: a small labelled contractor dataset trains an inverse-dynamics model that labels a huge corpus of unlabelled human videos; a behavioural prior is trained on the pseudo-labelled data; then **fine-tune with RL while keeping a KL penalty to the prior policy**. (The KL-to-prior term is the "don't unlearn the prior / don't reward-hack the sparse reward" regulariser.)

**Key quantitative results I could verify (abstract).** "the first to report computer agents that can craft diamond tools, which can take proficient humans upwards of 20 minutes (24,000 environment actions) of gameplay to accomplish"; "it can be fine-tuned, with both imitation learning and reinforcement learning, to hard-exploration tasks that are impossible to learn from scratch via reinforcement learning".
`[NOT VERIFIED: the exact KL coefficient, the contractor dataset size in hours, and the ablation numbers on the KL term. The paper's own statement that the KL term was necessary is widely quoted, but I did not retrieve it, so do not attribute a specific coefficient to it on my authority.]`

**Relevance: HIGH (as the "anchor" recipe).** VPT's structure is your structure at a different scale: a prior trained by imitation, then RL that is *prevented* from destroying the prior. The KL term is the standard fix for the specific pathology you will hit if you move to RL — the policy collapses onto a degenerate "safe corner" behaviour, or games the sparse reward.

---

## 3. Offline RL → online fine-tuning with very little online interaction

Framing for your constraint: at 30 decisions/s, single instance, the wall-clock cost of online interaction is brutal — 100k decisions = 55 min, **1M decisions = 9.3 h, 10M decisions = 92.6 h** (arithmetic from your stated 30 decisions/s; not a literature number). This is the number that should decide your algorithm choice: anything needing 10M+ environment steps is off the table; anything needing ≤1M is an overnight run.

### 3.1 AWAC — Nair, Gupta, Dalal, Levine (arXiv:2006.09359, 2020)
URL: https://arxiv.org/abs/2006.09359

**Method.** Off-policy TD critic + a *maximum-likelihood* actor update weighted by advantage `exp(A/λ)`, which yields an **implicit** policy constraint that "does not require any explicit modeling of the behavior policy". Offline pre-training on the prior dataset, then online fine-tuning on a mixture of offline and online data.

**Key quantitative/qualitative results (verified quotes from the fetched text).**
- Prior dataset for the analysis task (HalfCheetah): "15 demonstrations from an expert policy and 100 suboptimal trajectories sampled from a behavioral clone of these demonstrations".
- On-policy fine-tuning baselines: "on-policy methods like DAPG and Monte-Carlo return methods like AWR and MARWIL are an order of magnitude slower than off-policy actor-critic methods".
- The crucial negative results, which are the reason to cite this paper: "SAC-scratch, which does not receive the prior data, performs similarly to SACfD-prior, which does have access to the prior data, indicating that the off-policy RL algorithm is not actually able to make use of the off-policy data for pre-training. Moreover, even if the SAC policy is pre-trained by behavior cloning, labeled SACfD-pretrain, we still observe an initial decrease in performance, and performance similar to learning from scratch."
- On pure offline methods: BEAR fine-tunes "very slowly"; the diagnosis is that "the accuracy of the behavior models ... reduces during online fine-tuning" so "constrained optimization becomes too conservative".
`[NOT VERIFIED: exact online step budgets and final success numbers — the evidence is in learning curves (Fig. 3, Fig. 4), and my fetch truncated before the results tables.]`

**Relevance: HIGH.** AWAC's diagnostic is the most useful thing in this literature for you: **naively putting offline data in the replay buffer does nothing, and BC pre-training followed by naive off-policy RL can lose the pre-trained performance before it improves.** If you move from distillation to RL, expect an initial dip; that dip is not a bug in your environment.

### 3.2 IQL — Kostrikov, Nair, Levine (arXiv:2110.06169, 2021)
URL: https://arxiv.org/abs/2110.06169

**Method.** Never query the Q-function on out-of-distribution actions. Fit `V` by expectile regression of `Q` (an implicit "best action in the data" estimate), back it up into `Q`, extract the policy by **advantage-weighted behavioural cloning**. Abstract: "we can approximate the policy improvement step implicitly by treating the state value function as a random variable ... and then taking a state conditional upper expectile".

**Key claims (abstract, verbatim).** "IQL demonstrates the state-of-the-art performance on D4RL"; "We also demonstrate that IQL achieves strong performance fine-tuning using online interaction after offline initialization."
`[NOT VERIFIED: number of gradient steps, and the D4RL scores.]`

**Relevance: HIGH, and possibly the best fit to your compute.** IQL is the one algorithm here that needs **zero environment interaction** to improve on the behaviour data, works with discrete actions, and requires only a Q/V head on states you already have. Because it never evaluates actions absent from the data, it is robust to the exact pathology your plateau suggests (ungrounded per-frame values on states the expert never visits).

### 3.3 TD3+BC — Fujimoto, Gu (NeurIPS 2021 Spotlight), arXiv:2106.06860
URL: https://arxiv.org/abs/2106.06860

**Method.** Take an online off-policy algorithm (TD3), add a **behaviour-cloning term to the policy loss** and normalise the reward/state scales; no generative model, no extra hyperparameters beyond the BC coefficient.

**Key claims (abstract, verbatim + verified comment line).** "we can match the performance of state-of-the-art offline RL algorithms by simply adding a behavior cloning term to the policy update of an online RL algorithm and normalizing the data ... while more than halving the overall run time by removing the additional computational overhead of previous methods." arXiv comment: "NeurIPS 2021 Spotlight".
`[NOT VERIFIED: step counts / D4RL tables.]`

**Relevance: MEDIUM.** Not a fine-tuning method (no online phase); included because it establishes that a *plain BC regulariser on top of a Q-learner is a competitive offline baseline*. That is the minimal change from your current distillation setup.

### 3.4 Cal-QL — Nakamoto, Zhai, Singh, Mark, Ma, Finn, Kumar, Levine (NeurIPS 2023), arXiv:2303.05479
URL: https://arxiv.org/abs/2303.05479 · project page: https://nakamotoo.github.io/Cal-QL/

**Method.** Diagnose *why* offline initialisations fine-tune badly: conservative Q-values are much smaller than any valid policy's true return, so the first online samples make bad actions look good, and the policy **unlearns** the initialisation. Fix = "calibration": require `E_{a~π}[Q(s,a)] ≥ V^μ(s)` for a reference policy `μ` (the behaviour policy), implemented as a one-line change to CQL's regulariser: `E_{s~D,a~π}[max(Q(s,a), V^μ(s))] − E_{s,a~D}[Q(s,a)]`. Also gives a cumulative-regret analysis.

**Key quantitative results (verified).**
- "Cal-QL outperforms state-of-the-art methods on 9/11 fine-tuning benchmark tasks" (abstract); "the best fine-tuned performance in 9 out of 11 tasks" and "the best performance among 8 / 11 tasks" on cumulative regret (project page).
- "We show that Cal-QL matches or outperforms the best methods on all tasks, in some cases by 30-40%" (paper, §1).
- Their diagnostic experiment: "Fine-tuning begins at 50K steps" (Fig. 3 caption) — i.e. 50k offline steps then online.
- Environments: AntMaze, Franka Kitchen, Adroit, and **Visual Robotic Manipulation** (pixel input).
- Prior methods compared under the same protocol: CQL (unlearns), IQL, TD3+BC, AWAC ("attain slow asymptotic performance"), PEX, RLPD, SAC, SAC+offline, Hybrid RL.
`[NOT VERIFIED: the exact online step budget (I believe 1M online steps, but the axis values were not in any text I fetched).]`

**Relevance: HIGH.** Two reasons. (i) It is the state of the art specifically for "offline init → little online interaction", which is precisely your budget. (ii) The *calibration* concept maps onto your setting in a non-obvious way: your per-frame candidate values are uncalibrated with respect to the actual survival return, so a small number of online samples can convince the optimiser that a bad action is good. Also note the downstream evidence: Kumar et al. 2022 "Pre-Training for Robots: Offline RL Enables Learning New Tasks from a Handful of Trials" (arXiv:2210.05178) uses Cal-QL for real-world autonomous fine-tuning — the closest published analogue to "a few hours of real interaction".

### 3.5 RLPD — Ball, Smith, Kostrikov, Levine (ICML 2023), arXiv:2302.02948
URL: https://arxiv.org/abs/2302.02948

**Method.** No new algorithm. Put the offline data in the replay buffer, sample 50/50 offline/online, add layer normalisation, ensembles, and a high update-to-data ratio. "can we simply apply existing off-policy methods to leverage offline data when learning online? ... the answer is yes; however, a set of minimal but important changes to existing off-policy RL algorithms are required".

**Key quantitative claim (abstract, verbatim).** "correct application of these simple recommendations can provide a **2.5×** improvement over existing approaches across a diverse set of competitive benchmarks, with no additional computational overhead."
`[NOT VERIFIED: demo counts (commonly 20 demos) and the 1M-step budget.]`

**Relevance: HIGH (engineering), MEDIUM (novelty).** If you do move to off-policy RL, this is the checklist to follow so that the offline data actually helps instead of being ignored (AWAC's negative result above).

---

## 4. Reward shaping with privileged state

### 4.1 Ng, Harada, Russell — "Policy invariance under reward transformations: Theory and application to reward shaping" (ICML 1999)
URL (canonical PDF; I could not text-extract it with the tools available, see §7): https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf

**What potential-based shaping guarantees.** For any potential function `Φ: S → ℝ`, the shaping reward
`F(s, a, s') = γΦ(s') − Φ(s)`
added to the base reward leaves the **set of optimal policies unchanged** (policy invariance), for both discounted and undiscounted (absorbing) formulations, and for any `Φ`. The mechanism is a telescoping sum: over a trajectory the shaping terms collapse to `γ^T Φ(s_T) − Φ(s_0)`, so the return of every trajectory changes by an amount that depends only on its endpoints, not on the actions taken. Correspondingly, the Q-function shifts by exactly `−Φ(s)`, `Q'(s,a) = Q(s,a) − Φ(s)`, so greedy action selection is preserved. The paper's contribution is precisely that this is the *only* form of state-only shaping with this guarantee.

**Caveats that matter operationally (and that I am flagging as analysis, not as quoted theorem text).**
1. **`Φ` must be a function of state only** — not of time, not of the action, not of an action-history window. If you shape with "the critic's value at this frame" and the critic is updated during training, `Φ` becomes time-varying and the 1999 guarantee no longer applies; the extension to time-varying potentials is Devlin & Kudenko (AAMAS 2012), and it requires the potential be evaluated consistently at `s` and `s'`.
2. **Terminal transitions are the classic implementation bug.** Because the guarantee relies on discounting, the potential at an absorbing/death state must be handled consistently (typically `Φ(terminal) = 0`), otherwise the agent can be paid to die, or paid to avoid the terminal transition. In a 3-deaths-ends-episode environment this is not a corner case — it is the main case.
3. **It guarantees the objective is unchanged; it does not guarantee faster learning.** There is no sample-complexity theorem attached. Empirically it usually helps a lot and occasionally hurts; the guarantee is about *what* you optimise, not *how fast*.
4. **Policy invariance ≠ policy-order invariance.** Arbitrary `Φ` preserves the optimal policy, but the induced ranking of *suboptimal* policies can change; only a `V*`-based potential provably preserves the ordering of all policies (Wiewiora, Cottrell, Elkan, ICML 2003 — `[NOT VERIFIED in this review]`). That matters if you diagnose progress by comparing mediocre policies.

**Relevance: HIGH as a constraint, not a recipe.** If you are going to shape the sparse death signal with privileged information — and you should, because a per-frame "danger" signal is dense and free — then use the PBRS form with `Φ` computed from privileged state (e.g. `Φ(s) = −(expected time-to-death from s)` or the scripted expert's beam-search score at `s`). This gets you a dense signal *without* changing the objective you are evaluated on, which is the only way to add shaping to a survival task without silently redefining "good".

### 4.2 Using a learned/privileged critic as a dense signal for a pixel actor
I could not find a paper that turns a privileged critic into an *additive dense reward* for a pixel actor with verified quantitative results. What exists, and what it actually is:
- **Advantage-weighted regression / AWAC-style actor updates** use the critic as a **per-sample weight** on the actor's log-likelihood: `log π(a|s) · exp(A(s,a)/λ)`. This *is* "privileged critic as a dense per-sample training signal for a pixel actor", and it is the mechanism behind AWAC (above) and AWR/Peng et al. 2019. If the critic reads memory state and the actor reads pixels, this is an asymmetric-actor-critic with an implicit (rather than explicit) distillation objective. **Evidence: AWAC's curves; no clean ablation isolating the critic's privileged input.**
- **VPT's KL-to-prior** is a dense per-step signal that is not a reward, and it is the one that has been shown to matter at scale for sparse-reward fine-tuning.
- **MARWIL / Filtered BC / one-step RL** (Brandfonbrener et al., NeurIPS 2021) — advantage-filtered BC is the degenerate case (weight 1 or 0). `[NOT VERIFIED in this review]`

**Relevance: MEDIUM; the honest summary is that "privileged critic as reward" is usually implemented as "privileged critic as actor-loss weight", and PBRS is the only version of "as reward" that has a theorem.**

---

## 5. Safety shields / safety filters over a learned policy

### 5.1 Alshiekh, Bloem, Ehlers, Könighofer, Niekum, Topcu — "Safe Reinforcement Learning via Shielding" (AAAI 2018), arXiv:1708.08611
URL: https://arxiv.org/abs/1708.08611

**What it needs.** A safety specification in the safety fragment of LTL, **plus "an abstraction of the agent's environment dynamics"**: "The shield is computed upfront from the safety part of the given system specification and an abstraction of the agent's environment dynamics." Shield synthesis is a two-player game solve over that abstraction, and the abstraction must be conservative (an over-approximation) for correctness to carry over to the real system.

**Two placements.** *Preemptive* shielding: "the shield acts each time the learning agent is about to make a decision and provides a list of safe actions" (the learner is restricted to that set). *Post-posed* shielding: "it monitors the actions selected by the learning agent and corrects them if and only if the chosen action causes a violation of the specification."

**What it guarantees.** Correctness of the execution w.r.t. the specification "during the learning and controller execution phases, regardless of how fast the learning process converges"; *minimum interference* ("forbids actions only if they could endanger safe system behavior"); and preservation of the learner's convergence guarantees under stated conditions on the shield.

**Quantitative results.** Experiments are on "a grid world example", "a self-driving car example", "Atari 2600 Seaquest", and "the water tank example" (section headings, verified). `[NOT VERIFIED: no experiment numbers were retrieved — the fetch truncated before §8. I will not guess them.]`

**Relevance: HIGH (structurally), with a hard blocker.** The mapping is almost embarrassingly direct: your scripted expert (beam search over linear bullet extrapolation, ~6500 frames) is *already* an abstraction-plus-backup-controller, i.e. a shield. You could deploy it as a **post-posed shield over the student**: the student acts; if its action leaves the set of states from which the expert's recovery policy survives the next N frames, override. The blocker is stated in the paper itself: the shield needs the state, and **your deployment policy sees only 3 crops and action history**. A shield at deployment therefore requires a perception module (bullet-state estimator) whose error becomes part of the safety argument. That is the whole difficulty, and it is why §5.3 matters.

### 5.2 Model predictive shielding (MPS) / Robust MPS — Li & Bastani (arXiv:1910.10885); and arXiv:1905.10691
URLs: https://arxiv.org/abs/1910.10885 (robust MPS) · https://arxiv.org/abs/1905.10691 (MPS with nonlinear dynamics)

**Mechanism.** Maintain the invariant that a *backup controller* can always return the system to a safe equilibrium. If `x` is recoverable but `f(π̂(x))` is not, use the backup controller; otherwise let the learned policy act. Formally: "a state `x ∈ X` that satisfies this invariant is *recoverable* ... as long as the system remains in `X_rec`, then MPS can guarantee safety." The backup controller in the robust version is a tube-based robust NMPC, with the tube (forward reachable set) estimated by sampling and given a high-probability guarantee via statistical learning theory.

**What it needs — quote, because this is the crux.** "We focus on the setting where the nominal dynamics are known, and are subject to additive stochastic disturbances with known distribution." Also: the disturbances are explicitly allowed to model estimation error — "it can be used to model estimation error in the robot's state (e.g., uncertainty in its position)".

**Quantitative results.** Demonstrated on "cart-pole and a non-holonomic particle with random obstacles", with DDPG as the learned policy. `[NOT VERIFIED: specific numbers; fetch truncated in §IV.]`

**Relevance: HIGH — the highest of anything in this section, for a reason that inverts your current architecture.** MPS's requirements are *exactly* your assets: a known nominal model (linear bullet extrapolation is a nominal model with small, roughly quantifiable process noise), a reachable-set/backup computation (your beam search *is* a bounded-horizon backup), and a stochastic disturbance model (bullet acceleration, player velocity, timing jitter). It also tells you why a shield changes the learning problem: **if the shield absorbs tail risk, the student no longer needs to learn it**, so a shielded policy can have a much lower survival rate than the shield+policy system, and evaluating the student alone will look worse than deployment. Conversely: if you evaluate with the shield on, your 6500-frame expert baseline and your student numbers are not comparable.

### 5.3 Control-barrier-function filters, and what happens when the state estimate is noisy
CBF safety filters take a nominal (learned) control input and project it onto the set of inputs satisfying `L_f h(x) + L_g h(x) u + α h(x) ≥ 0` for a barrier `h` whose superlevel set is the safe set. Requirements are stronger than MPS's: a **known control-affine model**, a valid CBF (often found by hand or by learning with a certificate), and a QP solved at control rate. The relevant reference for *noisy perception* is Dean, Taylor, Cosner, Recht, Ames, "Guaranteeing Safety of Learned Perception Modules via Measurement-Robust Control Barrier Functions" (CoRL 2020) — record: https://authors.library.caltech.edu/records/ecz9b-j5q17 — which is explicitly about certifying safety when the barrier is evaluated on the output of a *learned perception module*. Background: Ames et al., "Control Barrier Functions: Theory and Applications" (ECC 2019), record: https://authors.caltech.edu/94624/.
`[NOT VERIFIED: I did not retrieve quantitative results or the exact error-bound condition for either paper. Treat the mechanism as established and the numbers as unquoted.]`

**The general answer to "what happens when the state estimate is noisy", as supported by the sources above:** the guarantee does not degrade gracefully on its own — it is *conditional on a bound*. Concretely, (i) MPS handles it by folding the estimation error into the disturbance set `W` and inflating the reachable-set tube (and gets only a **high-probability** guarantee, not a hard one); (ii) measurement-robust CBFs require an explicit bound on perception error to define an inflated safe set; (iii) Alshiekh-style shields are only correct relative to the abstraction, so a mis-estimated state silently voids the guarantee while the shield still *appears* to work — the worst failure mode, because violations look like ordinary bad luck. There is no shield/filter result I am aware of that gives you safety from an unbounded-error perception pipeline.

**Relevance: HIGH (as a warning), MEDIUM (as a technique).** For your system the practical consequence is: a shield built on estimated bullet positions needs an explicit, measured bound on estimator error (e.g. the 99th percentile of position error at 2-frame horizons), and the shield's recovery horizon must be inflated by that bound. If the inflation makes the safe set empty, the shield is useless and you find out before deploying, not after.

---

## 6. Sample-complexity reality check: how many frames does pixel-input RL actually need?

**Anchors I could verify in this review.**
- DQfD's comparison baseline: "DQN learns from over 200 million frames" (arXiv:1704.03732). DQfD's own results are reported "over 200 million steps" for the SOTA comparison, and its best-window scores are taken over a 3M-step window.
- Human-level Atari 200x faster (Kapturowski et al., arXiv:2209.07550): "Agent57 was the first agent to surpass the human benchmark on all 57 games, but this came at the cost of poor data-efficiency, requiring nearly 80 billion frames of experience to achieve. ... we employ a diverse set of strategies to achieve a 200-fold reduction of experience needed to out perform the human baseline." (Read this as the scale of the problem: even after a 200× reduction, the remaining budget is still enormous by your standards.)
- Demonstration initialisation, measured as *catch-up frames saved*: DQfD beats PDD DQN for the first 1M steps on 41/42 games and PDD DQN needs **83M steps on average** to catch up; but DQfD's demo data was only 5,574–75,472 transitions. So: ~10^4 demo transitions bought ~8×10^7 frames of head start, and did not change the asymptote much.
- The sample-efficient Atari line (Kaiser et al. 2020, "Model-Based Reinforcement Learning for Atari", arXiv:1903.00374; Data-Efficient Rainbow; CURL; DrQ; SPR) exists precisely because the standard budgets are 10^8–10^9 frames. `[NOT VERIFIED: I did not fetch any of these papers, so I am not quoting their scores or the exact 100k-frame budget figure. Do not cite a number for them on my authority.]`
- Negative result on pure exploration: Lee et al. 2020 — direct sparse-reward RL of a hard locomotion policy "failed to learn locomotion within reasonable time budgets", which is why they went two-stage.

**Converting to your units (arithmetic, not literature).** At 30 decisions/s and 2 frames/decision: 1 decision = 1/30 s; one 3-life episode at ~4000 frames ≈ 2000 decisions ≈ **67 s**. A 1M-decision budget = **9.3 h**; 10M = **92.6 h**; 100M (one "standard" Atari budget) = **38.6 days**. Compare: the whole 200M-frame Atari DQN budget would be ~2 years of wall-clock on a single TH08 instance.

**Verdict.** The literature supports a sharp conclusion for you: (a) pixel-input RL from scratch on a *single* game is a 10^8-frame proposition in the standard literature and 10^5–10^6 only in the heavily-engineered sample-efficient regime, and (b) demonstration/privileged initialisation reliably removes the *early* part of the curve (DQfD: better on 41/42 games in the first 1M steps) but is not documented to remove the asymptote. You cannot substitute frames for algorithm design; you can substitute a *privileged model* for frames, which is what your expert already is.

---

## 7. The three recipes with the strongest empirical track record for this exact setting, ranked

Before the ranking, the diagnosis that determines it. Your teacher survives ~6500 frames and your target is >4000. **The teacher is already good enough; the gap is entirely student-side.** So the ranking is ordered by strength of evidence for "close a teacher→student gap in a pixel policy", not by "get a better teacher".

### Rank 1 — LBC-style on-policy DAgger with white-box, full-18-action supervision, and an intermediate (image-space) distillation target
**Evidence base:** LBC ablation (16 → 96 → 100 in the same table, same benchmark) — the only paper I found that ablates *exactly* your two failure axes (whose state distribution; how much of the teacher's output per state) in a privileged-teacher → pixel-student pipeline whose teacher is itself a scripted-expert clone. Supporting: Lee et al. 2020 distilling the teacher's latent as well as its action; AAC's bottleneck auxiliary loss.

**Concretely, three changes to what you do now:**
1. **Roll out the student and label the student's own states** with the expert's beam search (this is DAgger; the expert is a function, so you can query it anywhere). Your current pipeline cannot fix covariate shift because it never observes the states that break the student.
2. **Supervise the full candidate-value vector over all 18 actions at every visited state** (LBC's "white-box" advantage, worth 16→96), using a **ranking / margin loss** rather than MSE on values — DQfD's `max_a[Q(s,a)+l(a_E,a)] − Q(s,a_E)` is the canonical form and is exactly the right objective when only the argmax matters for survival.
3. **Distil an image-space intermediate target** — e.g. per-region future danger / predicted bullet occupancy in the crop frame — not just 18 logits. The student's bottleneck is perception; give it a perception-shaped target.

**Expected failure modes.** (i) *Irreducible perception loss*: if the 3 crops + action history do not identify the bullet configuration (occlusion, bullets entering from outside the crops, temporal aliasing at 2-frame decisions), no distillation scheme reaches 4000 and the loss curve will show a floor — measure this directly by training a supervised bullet-position estimator from the crops and looking at its error at 2-frame horizons; that number bounds everything else. (ii) *Teacher noise*: the teacher's per-frame values change discontinuously when the beam search flips; a margin loss on near-tied actions will add noise, so use a margin on the *gap* and skip near-ties. (iii) *DAgger cost*: 30 decisions/s means each round of on-policy rollouts is slow — budget by 1M decisions ≈ 9.3 h, so plan ~5–10 rounds, not 18. (iv) *Compounding*: DAgger's guarantee is no-regret under an interactive expert; your expert is expensive per query (beam search), so query sparsely and cache.

### Rank 2 — Privileged-critic offline RL (IQL/Cal-QL on the logged memory-state dataset) used as the *training signal* for the pixel actor, with PBRS shaping from the same critic
**Evidence base:** IQL (offline-only improvement over the behaviour policy, no OOD action queries), Cal-QL (9/11 tasks; +30–40% in cases; explicit diagnosis of the "unlearn the initialisation" failure), AWAC (advantage-weighted actor = critic as dense per-sample signal for a pixel actor; plus its negative results on naive buffer seeding and BC-pretrain-then-RL), DQfD (margin loss + demo retention), RLPD (2.5× from minimal changes), Ng et al. 1999 (PBRS keeps the survival objective intact).

**Concretely:** fit `Q(s,a)` and `V(s)` on **memory states** with IQL (all your logged expert + student rollouts, with returns-to-go; no environment interaction). Then (a) extract an improved *privileged* policy by `argmax_a Q` — if it beats 6500 frames, you have a better teacher for Rank 1 for free, and if it does not, you have learned that the expert is near-optimal in its own representation; (b) train the pixel student by advantage-weighted regression against that critic, with `Φ = V` used as a PBRS potential so the shaped reward is `γV(s') − V(s)` (objective-preserving by Ng et al.). Optionally fine-tune online with Cal-QL calibration against `V^μ` of the scripted expert, for ≤1M decisions.

**Expected failure modes.** (i) *Offline data coverage*: IQL/AWAC are only as good as the state coverage; your student's own rollouts (surviving 1766 frames) give you the on-policy region, the expert's give you the good region, and the *interesting* states (near-death, recoverable) are rare in both — this is exactly where Cal-QL's unlearning pathology lives. (ii) *Critic mis-specification*: `Q` on 18 discrete actions with a sparse 3-death reward and 2000-decision episodes has a very long effective horizon; discount choice will dominate results and a wrong `γ` makes `Φ = V` a bad potential. (iii) *PBRS implementation*: terminal-state handling (see §4.1 caveat 2) — get this wrong and you will train a policy that dies faster, and it will look like the shaping "helped early learning". (iv) *Distribution mismatch between privileged-state critic and pixel policy*: the critic is evaluated on memory states during training but the actor only has pixels; the actor inherits the critic's errors in a way the critic never sees.

### Rank 3 — Deploy a shield: the existing scripted expert as a post-posed backup controller over the student, driven by a supervised bullet-state estimator, with the recovery horizon inflated by the estimator's measured error
**Evidence base:** Alshiekh et al. (correctness during learning and execution, minimum interference, convergence preserved), MPS/RMPS (known nominal dynamics + backup controller + reachable-set check; estimation error folded into the disturbance set; only a high-probability guarantee), measurement-robust CBFs (safety certified only under a perception-error bound).

**Why it is ranked third despite being the most likely to *immediately* raise the survival number:** it changes what you are measuring. The shield's guarantee is conditional on the state estimate, and your deployment state estimate is a learned perception module. It also removes the learning signal for exactly the cases you most want the student to learn (tail risk), so it can *mask* a plateau rather than fix it. Use it as (a) a diagnostic — the gap between student-only and student+shield survival tells you how much of the deficit is decision quality vs. perception, and (b) a safety net for data collection, not as the primary answer to "reach 4000 frames".

**Expected failure modes.** (i) *Estimator error voids the guarantee silently* — the shield looks fine and violations look like bad luck; you must measure the error distribution, not assume it. (ii) *Overriding too often*: if the estimator is noisy, the shield will fire constantly and you will have re-implemented the expert badly at 2× the cost; instrument the override rate (MPS's "minimizing how often π̂ is overridden" is the metric to watch). (iii) *Model error*: linear bullet extrapolation is a nominal model with real process noise (acceleration, spawning, aimed bullets) — the reachable-set tube must be inflated for it, and if the inflation makes the recoverable set empty, the shield contributes nothing. (iv) *Evaluation apples-to-oranges*: never compare shielded-student survival against unshielded-expert survival.

### What I would *not* do, and why
- **PPO/RL from scratch on the live 30 Hz instance.** Lee et al. report direct sparse-reward RL failing on a comparable control problem; the verified Atari budgets are 10^8–10^9 frames (§6); 10M decisions is 92.6 h of wall-clock. There is no published result suggesting this works in your budget.
- **More rounds of the same per-frame value distillation.** LBC's ablation says the two-stage-off-policy cell is *below* direct imitation; 18 rounds of a 0.4% gain is what that cell looks like. The variable to change is the state distribution and the per-state supervision, not the number of rounds.
- **A learned reward model as the objective.** PBRS (§4.1) is the only shaping form with a policy-invariance theorem; a free-form learned dense reward silently redefines "good" and your metric is survival.

---

## 8. Verification ledger (what I could not confirm)

| Claim | Status |
|---|---|
| LBC Table 1 ablation numbers (20/16/64/96/100), 157K frames, 4 h, DAgger oracle wording, "second stage alone works equally well" | **Verified** (arXiv HTML) |
| LBC +18 points dense traffic, 10× infractions, 100% CARLA | **Verified** (abstract) |
| LBC NoCrash per-condition success rates | **NOT VERIFIED** (fetch truncated after Table 2) |
| AAC: "significant margins", bottleneck description, DDPG+HER, tasks | **Verified** |
| AAC: any numeric success rate / comparison table | **NOT VERIFIED** — evidence is learning curves only |
| Lee et al. 2020: privileged info contents, TCN student, latent+action distillation, "direct RL not successful", Table 1 speeds, DARPA zero failures, 13.4 cm with payload | **Verified** |
| Lee et al. 2020: teacher/student sample counts and distillation loss values | **NOT VERIFIED** |
| Yen-Chen et al.: ~80% success with 10 min suction / 1 h grasping; ICRA 2020 | **Verified** (abstract + comment) |
| DQfD: 5,574–75,472 demo transitions; 41/42 first-1M-step wins; 83M steps to catch up; 14/42 beat best demo; 11/42 SOTA; pure imitation worse than demo in every game; loss definitions incl. n=10 | **Verified** (full text) |
| DQfD: the numeric margin value `l` (commonly 0.8) | **NOT VERIFIED** (paper gives the form, not the constant in the text I fetched) |
| Kickstarting: 10× fewer steps to match, +42% final performance, DMLab-30 | **Verified** (abstract) |
| VPT: diamond tools, 20 min / 24,000 actions, "impossible from scratch" | **Verified** (abstract) |
| VPT: KL coefficient, contractor data hours, KL ablation | **NOT VERIFIED** |
| AWAC: quotes on on-policy being "an order of magnitude slower", SACfD-prior ≈ SAC-scratch, BC-pretrain dip, BEAR too conservative, 15 demos + 100 suboptimal trajectories | **Verified** (full text) |
| AWAC: online step budgets, final success numbers | **NOT VERIFIED** (curves only) |
| IQL: expectile + AWR extraction, SOTA D4RL claim, fine-tuning claim | **Verified** (abstract) · step counts **NOT VERIFIED** |
| TD3+BC: BC term + normalisation, "match SOTA", "more than halving run time", NeurIPS 2021 Spotlight | **Verified** (abstract + comment) |
| Cal-QL: calibration definition, `max(Q, V^μ)` mask, 9/11 tasks, 8/11 regret, "30-40%", 50K-step fine-tuning start, unlearning diagnosis | **Verified** (paper + project page) |
| Cal-QL: exact online step budget (1M?) | **NOT VERIFIED** |
| RLPD: 2.5× claim, minimal-changes message | **Verified** (abstract) · demo counts/budget **NOT VERIFIED** |
| Ng et al. 1999 theorem statement and `Q' = Q − Φ` shift | **Reconstructed from the standard statement + the telescoping identity; the PDF could not be text-extracted by my tools.** Treat the mathematical content as reliable, the exact wording as not quoted. |
| Alshiekh: mechanism, two placements, abstraction requirement, minimum interference, convergence preservation, experiment list | **Verified** (full text) |
| Alshiekh: any experiment numbers | **NOT VERIFIED** (truncated before §8) |
| RMPS: known nominal dynamics + known disturbance distribution, recoverability invariant, tube-based NMPC backup, estimation error as disturbance, cart-pole/particle demos | **Verified** (full text) · numbers **NOT VERIFIED** |
| Measurement-robust CBF / Ames CBF tutorial: existence, venue, topic | **Verified via publisher records** · contents **NOT VERIFIED** |
| Atari "200M frames" for DQN/DQfD; Agent57 "nearly 80 billion frames"; 200× reduction | **Verified** (DQfD full text; 2209.07550 abstract) |
| Atari 100k benchmark budget and any sample-efficient Atari scores | **NOT VERIFIED** (not fetched) |
| 30 decisions/s → 1M = 9.3 h, 10M = 92.6 h; 4000-frame episode ≈ 67 s | **Arithmetic on your stated numbers, not literature** |
