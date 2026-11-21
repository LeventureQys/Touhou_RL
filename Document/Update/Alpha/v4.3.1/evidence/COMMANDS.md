# v4.3.1 复核：真实命令、退出码与产物

全部命令在 `D:\WorkShop\Touhou_RL` 下以 `pwsh -Command` 执行。
`PYTHONPATH` 必须设为仓库根，否则 `import touhou_rl` 失败（入口问题，不是代码失败）。

```powershell
cd D:\WorkShop\Touhou_RL
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
```

环境实测：

```text
python 3.12   torch 2.14.1+cu132 (CPU)   numpy 2.5.4   torch.get_num_threads()=14
```

---

## R1 划分恢复与正式基线

```powershell
python tools\rl\v431_recheck_baseline.py
```

- 退出码：**0**
- 产物：`Document/Update/Alpha/v4.3.1/evidence/R1/baseline_recheck.json`
- 关键输出：

```text
corpora loaded: ['seed12', 'seed10', 'seed11']
seed12 n = 1099

== split recovery ==
  int_0p8      n_train=  879 n_val= 220 matches 880/219: False
  fixed_200    n_train=  899 n_val= 200 matches 880/219: False
  random_200   n_train=  899 n_val= 200 matches 880/219: False
  claimed in ROUTE_REVIEW: 880 train / 219 validation

== move-head baselines on the recovered hold-out (train-only fitting) ==
evaluation set             n  train_mle   laplace  uniform  H(train)  maj_acc  old -log
seed12_val__move         220     1.6586    1.6589   2.1972    1.6937   0.4500    0.8680
seed10_full__move       1099     1.7126    1.7125   2.1972    1.6937   0.4249    0.8680
seed10_suffix__move      880     1.7122    1.7118   2.1972    1.6937   0.4364    0.8680

== cross-tape common prefix (cand level) ==
  {'common_cand_prefix_len': 219, 'seed10_n': 1099, 'seed10_suffix_n': 880,
   'seed12_n_minus_prefix': 880, 'fraction_of_tape': 0.19927206551410373}

== documented numbers re-derived ==
  seed12_full_histogram = [468, 28, 203, 172, 32, 21, 78, 76, 21]
  seed12_full_n = 1099
  neg_log_full_majority = 0.8536876584859746
  full_seed12_entropy = 1.6859891972159187
  seed12_full_prior_on_seed10_full = 1.7074972455584387
```

---

## R2 跨磁带重叠审计

```powershell
python tools\rl\v431_recheck_overlap.py
```

- 退出码：**0**
- 产物：`evidence/R2/overlap_audit.json`
- 关键输出（原文）：

```text
seed12__seed10: cand common prefix = 219 (19.927% of shorter tape)
    prefix play_gray              identical    0/219  (  0.0%)  meanAbsDiff   0.622  max 251
    prefix hud_gray               identical  120/219  ( 54.8%)  meanAbsDiff   0.168  max 251
    prefix global_rgb             identical    0/219  (  0.0%)  meanAbsDiff   0.444  max 255
    prefix executed_key_history   identical  219/219  (100.0%)  meanAbsDiff   0.000  max 0
    prefix history_valid          identical  219/219  (100.0%)  meanAbsDiff   0.000  max 0
    suffix play_gray              same-index identical    0/880   shared-values-anywhere 0
    suffix executed_key_history   same-index identical  167/880   shared-values-anywhere 55

seed12__seed11: cand common prefix = 225 (20.473% of shorter tape)
    prefix play_gray              identical    0/225  (  0.0%)  meanAbsDiff   0.723  max 251

seed10__seed11: cand common prefix = 219 (19.927% of shorter tape)
    prefix play_gray              identical    0/219  (  0.0%)  meanAbsDiff   0.619  max 248

== within-tape ==
  seed12: distinct play_gray 1099/1099  exact dup rows 0  adjacent meanAbsDiff 21.485
  seed10: distinct play_gray 1099/1099  exact dup rows 0  adjacent meanAbsDiff 23.349
  seed11: distinct play_gray 1099/1099  exact dup rows 0  adjacent meanAbsDiff 23.054
  control (random pairs, seed12): 39.122

== ancestry ==
  seed12: tape=.tmp_v420_data\T04\dataset\d1-s0d1-n3300-seed12-shard\slots.jsonl
      sha=f759de7d...  slots=6588  frames [12..13188]  gap [6]
  seed10: sha=52402de4...  slots=6592  frames [12..13188]  gap [6]
  seed11: sha=f75e632b...  slots=6592  frames [12..13188]  gap [6]
```

---

## R3 近重复结构

```powershell
python tools\rl\v431_recheck_duplication.py
```

- 退出码：**0**
- 产物：`evidence/R3/duplication_structure.json`
- 关键输出：

```text
== A. label identity over the cand prefix ==
  seed12__seed10: prefix 219  move identical 219/219  focus identical 219/219
      suffix: move identical 342/880
  seed12__seed11: prefix 225  move identical 225/225
      suffix: move identical 333/874
  seed10__seed11: prefix 219  move identical 219/219
      suffix: move identical 300/880

== B. pixel structure over the prefix (play_gray) ==
  seed12__seed10: n=219  meanAbsDiff=0.622  pixels differing at all=5.46%
      >=8: 1.90%  >=32: 0.44%    samples with zero differing pixels: 0.0%

== C. random-split optimism inside seed12 ==
  n_val=220 sampled=44
  val -> temporally nearest train : 20.648
  val -> random train             : 39.809
  ratio (near twin / random)      : 0.519

== D. cross-tape distance ==
  prefix same-index 0.622  (min 0.002 max 4.374)  control 34.286
```

---

## R4 两个小模型 BC 臂重执行（新身份 v431-R4）

```powershell
python tools\rl\v431_recheck_models.py
```

- 第一次退出码 **1**：`TypeError: linear(): argument 'input' (position 1) must be Tensor, not numpy.ndarray`
  —— 求值集张量在 seed10/seed11 分支仍是 numpy，`score()` 未转换。已修正为
  `if not isinstance(x, torch.Tensor): x = torch.from_numpy(np.ascontiguousarray(x))`。
- 第二次退出码：**0**
- 产物：`evidence/R4/model_recheck.json`
- 关键输出：

```text
== split ==
  n=1099 cut=879 train=879 val=220

arm     evaluation set       n  model CE  base MLE  base lap  model acc  maj acc    verdict
-------------------------------------------------------------------------------------------
down16  seed12_train       879    1.2827    1.6937    1.6938     0.5358   0.4198      BEATS
down16  seed12_val         220    1.5306    1.6586    1.6589     0.4045   0.4500      BEATS
down16  seed10_full       1099    1.6020    1.7126    1.7125     0.4149   0.4249      BEATS
down16  seed10_suffix      880    1.6450    1.7122    1.7118     0.4011   0.4364      BEATS
down16  seed11_full       1099    1.5568    1.6755    1.6760     0.4104   0.4404      BEATS
down16  seed11_suffix      874    1.5903    1.6639    1.6641     0.3924   0.4588      BEATS
down4   seed12_train       879    0.1558    1.6937    1.6938     0.9829   0.4198      BEATS
down4   seed12_val         220    2.1299    1.6586    1.6589     0.4591   0.4500         no
down4   seed10_full       1099    2.1161    1.7126    1.7125     0.4832   0.4249         no
down4   seed10_suffix      880    2.5070    1.7122    1.7118     0.3875   0.4364         no
down4   seed11_full       1099    2.0074    1.6755    1.6760     0.4923   0.4404         no
down4   seed11_suffix      874    2.3844    1.6639    1.6641     0.3936   0.4588         no

params: down16=30,793, down4=487,753
```

注意最后一行：`down4 seed12_train ... 0.9829 ... BEATS` 是**训练集自评**，
`BEATS` 在这里只说明它拟合了训练集（0.1558 / 0.9829），**不是**泛化证据。

---

## R5 配对 bootstrap：16× 的 CE 优势是否可与零区分

```powershell
python tools\rl\v431_recheck_significance.py
```

- 退出码：**0**
- 产物：`evidence/R5/significance.json`
- 做法：按 R4 的完全相同代码与 RNG 序列重训 30,793 参数模型（重训后 `seed12_val` CE
  仍为 1.5306，与 R4 一致），保存逐样本 CE，对 `基线CE_i − 模型CE_i` 做 20,000 次配对 bootstrap。
- 关键输出：

```text
n_train=879 n_val=220 params=30,793
  seed10 common prefix 219
  seed11 common prefix 225
  seed12_val     model 1.5306  base 1.6586  delta +0.1280  CI [+0.0350, +0.2215]  excl0=True
  seed10_full    model 1.6020  base 1.7126  delta +0.1106  CI [+0.0657, +0.1549]  excl0=True
  seed10_suffix  model 1.6450  base 1.7122  delta +0.0673  CI [+0.0149, +0.1190]  excl0=True
  seed11_full    model 1.5568  base 1.6755  delta +0.1187  CI [+0.0765, +0.1602]  excl0=True
  seed11_suffix  model 1.5903  base 1.6639  delta +0.0736  CI [+0.0265, +0.1206]  excl0=True

set                n    model  baseline    delta                 95% CI  excl 0     acc     maj
seed12_val       220   1.5306    1.6586  +0.1280 [  +0.0350,  +0.2215]    True  0.4045  0.4500
seed10_full     1099   1.6020    1.7126  +0.1106 [  +0.0657,  +0.1549]    True  0.4149  0.4249
seed10_suffix    880   1.6450    1.7122  +0.0673 [  +0.0149,  +0.1190]    True  0.4011  0.4364
seed11_full     1099   1.5568    1.6755  +0.1187 [  +0.0765,  +0.1602]    True  0.4104  0.4404
seed11_suffix    874   1.5903    1.6639  +0.0736 [  +0.0265,  +0.1206]    True  0.3924  0.4588
```

解读边界：CI 下界在两条去前缀后缀上只到 +0.0149 / +0.0265，所以结论只能是
"幅度很小但可与零区分的 CE 改善"，**不是**"BC 学到可迁移策略"；同一模型 argmax 准确率仍低于多数类。

---

## R10 / R11 / R12 复用 v4.2.0 T08 A1 的引擎核验资产

```powershell
python tools\rl\v431_safeset_evidence.py    # exit 0 -> evidence/R10/safe_set_evidence.json
python tools\rl\v431_state_cost.py          # exit 0 -> evidence/R11/state_cost_profile.json
python tools\rl\v431_seg001_manifest.py     # exit 0 -> evidence/R12/seg001_state_manifest.json
```

只读输入：`Document/Update/Alpha/v4.2.0/evidence/T08/a1_safe_set_full.json`
（`schema=v420-a1-safe-set-full/1`，`status=pass`，`exit_code=0`）。

`R10` 输出：

```text
states=200 arms_expected=3600 arms_present=3600 horizon=120 frames
== safe-set size histogram ==
{"0": 11, "1": 6, "2": 6, "3": 15, "4": 10, "5": 18, "6": 17, "7": 10, "8": 8,
 "9": 11, "10": 13, "11": 9, "12": 12, "13": 14, "14": 11, "15": 13, "16": 8,
 "17": 3, "18": 5}
  buckets: {"S_eq_18": 5, "S_between_2_and_17": 178, "S_eq_1": 6, "S_eq_0": 11}
== discriminating (0 < |S| < 18) ==
  count=184 of 200 complete; rate_over_all=0.92 rate_over_complete=0.92
  proper count=184/200 rate=0.92
    wilson95={'lower': 0.87401, 'upper': 0.95016}
== measured engine cost of this pass ==
  prefix_share_of_frames = 0.9399
  branch_share_of_frames = 0.0601
  prefix_frames_per_arm = 1471.2
  branch_frames_per_arm = 94.1
  wall_s_per_arm = 12.864
  engine_frames_per_s = 121.7
== outcome counts ==
  {"states": 200, "complete_states": 200, "invalid_states": 0, "arms": 3600,
   "outcomes": {"engine_death": 1899, "engine_alive": 1701, "unknown": 0},
   "distinct_seed_ancestors": 50, "distinct_trajectories": 50}
```

`R11` 输出：

```text
states=200 proper=184 seeds=50 (3042..3091)
offsets_to_death=[30, 31, 60, 61, 90, 91, 120, 121]
prefix_frames: {'min': 704, 'p25': 1307, 'median': 1390, 'p75': 1516, 'max': 3020, 'mean': 1469.4}
prefix hash mismatches: 0 over 0 states
state_wall_s: {'min': 142.7, 'median': 226.8, 'max': 461.1, 'mean': 237.6}
death share of the 18 actions at proper states: min 0.056 median 0.5 max 0.944 mean 0.514
  states where all 18 die: 0   where none die: 0

== online-attempt cost at the median proper-state prefix ==
  segment_2s   prefix   11.4s + segment   1.0s =  12.4s/attempt   prefix share 92.0%   ...
  segment_3s   prefix   11.4s + segment   1.5s =  12.8s/attempt   prefix share 88.5%   200 attempts 0.71 h   3 arms 2.14 h
  segment_5s   prefix   11.4s + segment   2.5s =  13.8s/attempt   prefix share 82.2%   200 attempts 0.77 h   3 arms 2.30 h
  segment_10s  prefix   11.4s + segment   4.9s =  16.3s/attempt   prefix share 69.7%   200 attempts 0.91 h   3 arms 2.72 h
```

`R12` 输出（冻结清单）：

```text
states 200  proper 184
train pool 150 -> selected 68 across 35 seeds
dev   pool 34 -> selected 16 across 8 seeds
dev seeds: [3083, 3084, 3085, 3086, 3088, 3089, 3090, 3091]

== budget ==
  attempt_s_train = 13.4
  attempt_s_dev = 13.7
  h_200_episodes_one_arm = 0.74
  h_three_arms_200_each = 2.23
  n_dev_attempts = 192
  h_dev_evaluation = 0.73
```

---

## R14 / R15 / R16 游戏静音现状、候选动作空间、A1 判定可靠性

```powershell
python tools\rl\v431_audio_cfg_audit.py       # exit 0 -> evidence/R14/audio_cfg_audit.json
python tools\rl\v431_candidate_space.py       # exit 0 -> evidence/R15/candidate_space.json
python tools\rl\v431_verdict_reliability.py   # exit 0 -> evidence/R16/verdict_reliability.json
```

`R14`（对应 `AGENTS.md` §9）：

```text
th08.cfg found: 105   decoded: 105   errors: 0
  muted   (playSounds=0, volumes<=0): 0
  audible (any of those not off)    : 105

root                      muted  audible
.tmp_v420_data                0       71
.tmp_v430_o2                  0        2
.tmp_v410_wp05_round1         0       32

  audible: playSounds=1 music=39 sfx=36  \.tmp_v420_data\replay\env-seed10\th08.cfg
```

模板（`v400_make_cfg.py --dry-run`）是 `musicMode 0 / playSounds 0 / musicVolume 0 / sfxVolume 0`，
但 105 个已部署 cfg 一个都没静音 → 静音目前只靠"WSL 没有声卡"。

`R15`：

```text
candidates=18  distinct masks=18  collisions=0
roundtrip_ok=True  bomb_always_0=True  problems=0
varying bits across the candidate set: 0xf4
   0      1    0x1
   9      5    0x5      <- bit 0x04 = focus
focus pairs: 9; all differ in mask: True
```

`R16`：

```text
== A. repeat agreement of the A1 arm verdicts ==
  pairs=4  agreed=4  rate=1.0

== B. does the recorded (student) action die at the selected states? ==
  selected states           : 84
  recorded action dies      : 81 (96.4%)
  states with unknown arms  : 0
  arms judged / expected    : 1512 / 1512
  safe size range           : 2..8
  mean lethal arms per state: 12.96 / 18
```

---

## R17 判据算术自测（无引擎）

```powershell
python tools\rl\v431_seg001_gate.py --selftest
```

- 退出码：**0**
- 输出：`selftest: 25 checks passed`
- 覆盖：Wilson 区间（含 `p=0`/`p=1` 端点与 8/10 的手算值）、区间重叠判定、
  `strictly_better` 要求点估计更高**且**区间不重叠、aborted 从分子分母同时排除、
  S1/S2/S4/S6 触发、S7 只发 note 不拦截、C-priv=100% 时**不得**阻断门槛、空输入不崩。

运行时的用法（未来）：

```powershell
python tools\rl\v431_seg001_gate.py --records <seg001-*.jsonl> --out evidence/SEG001/gate.json
```

---

## R18 恢复探针：真实引擎上重跑 A1 的 state 0（本轮唯一一次启动引擎）

前置（只读既有资产 + 一处静音模板）：

```powershell
python .tmp_v431_seg001\make_probe_config.py        # exit 0 -> configs/v431/seg001-probe.json
python tools\rl\v400_make_cfg.py --out ".tmp_v431_seg001\full\data\template-env\th08.cfg"
```

```bash
# WSL：建隔离工作目录（704 KB；两个大 blob 是软链，不复制大文件）
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/setup.sh     # exit 0
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/smoke.sh     # exit 0, "SMOKE OK"（不启引擎）
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_probe.sh # exit 0, wall 303 s（真实引擎）
```

对比与取证：

```powershell
python tools\rl\v431_probe_compare.py
```

- 退出码：**0**
- 产物：`evidence/R18/probe_reproduction.json`；原始分片 `.tmp_v431_seg001/full/raw/state_000.json`
- 关键输出：

```text
state 0: seed 3042 frame 2010 expected_hash 6cade82be08f8a95
shard status=pass exit=0 arms=18 complete=True
prefix checks=36180 mismatches=0

audio muted: True  (cfg sha cc062cfbf9fd2c14a1225376e182beffc52f964c342c187b409045f495d5f0fa)

outcome agreement: 18/18 = 1.0
safe set  A1=[2, 7, 11]
safe set new=[2, 7, 11]   equal=True

cand             A1          probe  agree  a1_death  pr_death
   0   engine_death   engine_death   True      2041      2041
   2   engine_alive   engine_alive   True      None      None
  17   engine_death   engine_death   True      2040      2040

cost: {"engine_starts": 1, "resets": 19, "reset_s": 14.524, "prefix_frames": 38190,
       "prefix_s": 269.781, "branch_frames": 1124, "student_forwards": 567,
       "student_s": 0.96, "encode_s": 1.05, "engine_frames": 39314, "arms": 19,
       "shard_wall_s": 295.747, "ms_per_arm": 16430.39}
```

结论：恢复协议、逐帧哈希门、引擎/checkpoint/数据**今天仍然成立**，
且静音（`playSounds=0` + `SDL_AUDIODRIVER=dummy`）**不改变**判定与死亡帧。

---

## R19 策略驱动 episode：harness 验证 + `safe_set` 语义更正（rev D）

驱动（新）：`tools/rl/v431_seg001_driver.py`（逐帧策略驱动，**不走**冻结学生的臂）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_harness.sh   # exit 0，3 个 episode
```

```powershell
python tools\rl\v431_harness_evidence.py          # exit 0 -> evidence/R19/harness_validation.json
```

- 关键输出：

```text
== harness checks ==
  restore_gate_all_ok                      True
  recorded_reproduced_a1_death_frame       True
  a1_recorded_death_frame                  2041
  probe_recorded_death_frame               2041

== episodes (state 0, seed 3042, frame 2010) ==
policy      frames      terminal         death@  offset  alive120  alive180  wall_s
recorded        31   first_death           2041      31     False     False   16.05
const:2         35   first_death           2045      35     False     False   16.16
const:7        180          None           None    None      True      True   16.80

== safe-set semantics ==
  A1 safe set              : [2, 7, 11]
  A1 says cand 2           : engine_alive
  holding cand 2           : first_death at offset 35
  holding cand 7           : None after 180 frames
  VERDICT                  : S is continuation-conditional: membership in S does not mean
                             the action can be held

== admissibility as a 180-frame segment ==
  constant cand 7 survived 180 frames : True
  state 0 admissible                 : False
```

三条结论：

1. `recorded` 在**另一条代码路径**上复现 A1 的死亡帧 `2041` → harness 交叉验证通过。
2. `safe_set` 是"续跑条件化"的标签集，**不是**可持有的动作集（cand 2 死、cand 7 活）。
3. **state 0 在 180 帧口径下不合格**（恒定动作可打完）→ `C4` 校准门必须恢复（rev D）。

> 同一个 `run_harness.sh` 第一次跑时 `alive120/alive180` 报了 `True`——那是我的 bug：
> 把**绝对帧号** `2041` 直接和**偏移** `120/180` 比。已改为 `died_at - decision_frame` 后再比，
> 上面的输出是修正后的。

---

## R20 `C4` 校准门 + 状态位置证据（真实引擎）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_c4.sh 0 1 3 165    # exit 0, 1009 s
# dev 状态首次全 18 臂失败：student_traces/seed3083.json 没拷进隔离目录
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/fix_traces.sh          # 补 50 条 trace（15 MB）
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_c4.sh 165          # exit 0, 273 s
```

```powershell
python tools\rl\v431_c4_screen.py        # exit 0 -> evidence/R20/c4_screen.json
python tools\rl\v431_state_position.py   # exit 0 -> evidence/R20/state_position.json
```

- 关键输出：

```text
states screened          : 4 (complete 18/18: 4)
admissible               : 0 (train 0, dev 0) rate 0.000
errors                   : 0
S1b (< 4 admissible/side) : True

 state  split  18/18  survivors  admissible  best const  best surv
     0  train   True          7       False           6       180f
     1  train   True          8       False           2       180f
     3  train   True          9       False           0       180f
   165    dev   True          6       False           1       180f
```

```text
A1 proper states            : 184 (seeds 49)
decision_frame              : min 704 p25 1306 median 1383 p75 1512 max 3020
stage length                : 13190 frames
median share of stage       : 10.5%
states beyond half the stage: 0
states beyond frame 8000    : 0

ink percentile at A1 states : mean 16.14 median 16.8
ink percentile in dense wins: mean 82.19 median 84.5
```

**结论**：`C4` 合格率 **0/4**，`S1b` 触发。原因不是引擎问题，而是 A1 状态全在关卡早期稀疏弹幕区
（中位数 10.5%、无一越过 frame 8,000），所以恒按一个方向就能活 3 秒。
每个状态的最强恒定动作都能打满整段 → `C-const-best` 在开发状态上就是 **100%**，
任何策略最多打平，**没有学习余量**。rev B/rev D 的 4–5 h 预算是靠换成简单任务买来的；
诚实预算回到 22–23 h（rev A 量级），除非先做引擎存档。

> 两个我自己引入的缺陷，都记在这里：
> ① `run_c4.sh` 第一次因第 41 行多一个引号直接语法失败（`bash: unexpected EOF`），`bash -n` 定位后修正；
> ② driver 在"输入缺失"这类早期异常下**不写任何产物**，dev 状态 18 臂全失败却只留下日志。
> 已加 `main()` 包装：任何异常都写一条 stub 记录（`error` 字段），失败也要留下证据。

---

## R21 / R22 轨迹上限、教师磁带哈希链、密集区间阻断

```powershell
python tools\rl\v431_pool_ceiling.py       # exit 0 -> evidence/R21/pool_ceiling.json
```

```text
== A. A1 student trace lengths (the ceiling on where a state can sit) ==
  traces            : 50
  last_frame        : min 825 median 1482 max 3051
  beyond frame 4000 : 0
  beyond frame 8000 : 0
  every trace ends at first_death : True

== B. teacher tape (seed12) ==
  slots             : 6588  frames 16..13190
  2-frame grid      : True
  hash chain breaks : 0 (first None)
  target frame      : 8666 is_slot_boundary=True
  target hash_pre   : 235b20476408e0c8
  prefix cost       : 8666 frames = 71.2 s
  synthetic prefix viable : True
```

离线构造校验（不启引擎）：

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/check_prefix.sh   # exit 0, "OFFLINE PREFIX OK"
```

```text
frame 8666: masks=8666 hashes=4327 expected=235b20476408e0c8 hash_at=235b20476408e0c8 match=True
hash_pre agreement over all slots up to 8666: 4326 OK, 0 bad
```

真实引擎（**失败，正是要报告的东西**）：

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_dense_smoke.sh   # exit 1
```

```text
DENSE_SMOKE exit=1 wall=5s
[seg001] error=CounterfactualError: frame 96: the recorded two-frame decision carries two
         different masks (0x0085 then 0x0084); the action history cannot be rebuilt
```

```powershell
python tools\rl\v431_dense_blocker.py      # exit 0 -> evidence/R22/dense_blocker.json
```

```text
 seed  slots  split-mask slots                  first split  reachable floor
   10   6592               137             slot 39 frame 94               94
   11   6592               137             slot 39 frame 94               94
   12   6588               137             slot 39 frame 94               94

no replayable prefix reaches frame 8000: True
```

**结论**：两条起始轨迹来源互斥——学生轨迹可重放但止于 frame 3,051；教师磁带覆盖密集区间
但越过 **frame 94** 就被 `_pair_cand`（`counterfactual.py:3924-3929`，fail-closed）拒绝。
所以**密集区间的 3–10 s 片段今天构造不出来，与预算无关**。

> 我自己的两个误解也记在这里：
> ① 第一版自造 prefix 用 `mask` 重复两帧，引擎在 frame 96 分叉——真正的逐帧 mask 在
> `T02c/rebuild/*/2f/slots.jsonl` 的 `corpus.committed_masks` 里（137 个槽两帧不同）；
> ② 换成正确的逐帧 mask 后，**离线 4,326 个哈希全对**，但引擎仍然拒绝——因为
> `_pair_cand` 不接受"一个决策两个 mask"。离线正确不等于可重放。

---

## R23 密集区间可达：绕开 `replay()` 的手动逐帧重放（真实引擎）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_dense_manual.sh   # exit 0, 75 s
```

```powershell
python tools\rl\v431_dense_reachable.py    # exit 0 -> evidence/R23/dense_reachable.json
```

```text
[dense] seed 12 frame 8664 policy recorded: terminal=None death_off=None
        alive120=True alive180=True checks=2165 mismatch=0 wall=72.444s error=None

== dense reachability ==
  target frame        : 8664 (dense >= 8000: True)
  frames walked       : 8663
  hash checks         : 2165
  hash mismatches     : 0
  engine frame        : 8663
  verified            : True
  cost                : wall 72.444s over 8843 engine frames (8.192 ms/frame)

== S1c corpus census ==
  shards              : 12
  all have split blocks: True
  all first split @92 : True
  identical step_frames: [(149, 149, 133, 132)]
```

**结论**：`_pair_cand` 只在 `replay()` 里被调用，所以用 `step_frame` 手动逐帧走完 8,663 帧
就能到达密集区间，并且每 4 帧一个记录哈希、**2,165 个全对**。
这**窄化**了 R22 的结论：密集区间对**臂 A 与两个基线**可达，对**像素臂 B/C 仍不可达**。

顺带解释 frame 96 的拒绝：v4.1.0 S1c 的 12 条磁带**每条都有 137 个双 mask 决策块，
第一个都在 frame 92，`step_frames` 完全相同 `[149,149,133,132]`**——是教师行为与
"一决策一 mask"契约的冲突。

> 本轮我自己的三个操作缺陷，一并记下：
> ① `Set-Content -Encoding utf8` 给 .sh 加了 BOM，导致 `#!/usr/bin/env` 失效；
> ② 内联 `python -c` 反复被 PowerShell 的引号解析弄坏（三次），最后改用脚本文件；
> ③ C4-dense 的批量 runner 跑了 13 分钟**一个产物都没有**且不报错——逐条进度回显是必需的，
> 这本身就是一条应当写进规范的教训。C4-dense 的正式测量留到下一轮用纯脚本文件重跑。

---

## R24 `C4` 校准门通过：密集窗口 @ 10 s（真实引擎）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_c4_dense.sh   # exit 0, 1290 s
```

```powershell
python tools\rl\v431_c4_dense_summary.py   # exit 0 -> evidence/R24/c4_dense.json
```

```text
window seed 12 frame 8664   segment 600 frames (10.0 s)
candidates      : 18  complete 18  errors 0
survivors       : 0 []
ADMISSIBLE      : True
death offsets   : min 58  median 277.5  max 520 (8.67 s)
min admissible segment : 521 frames (8.68 s); 10 s clears by 80 frames (13.3%)
hash gate       : all 2,165 checks per episode = True, mismatches 0
cost            : 18 episodes, mean 72.6 s, total 1306.6 s

cand      terminal  death_off  alive120  alive180
   2   first_death         58     False     False
  15   first_death        520      True      True
  17   first_death        304      True      True
```

**三条结论**：

1. **合格**：18/18 恒定动作在 600 帧（10 s）内全部死亡，0 存活。
2. **最短合格段长 = 521 帧 = 8.68 s**（最强恒定动作活到 520）。10 s 余量 13.3%。
3. **3 s 在本窗口根本不合格**：18 个恒定动作里 **13 个能活过 180 帧**。

边界：一个窗口一个种子；像素臂仍不可跑；门只排除"一个恒定键"，
不排除"519 帧时切一次"。

> 上一轮我说这个 runner"13 分钟一个产物都没有"，**是误判**：它其实正常，
> 是我的 WSL 内联 `ls | wc -l` 失败了，让我以为没落盘。教训：
> 跨 shell 边界时用脚本文件、并直接读产物目录，别用内联管道判断。

---

## R25 C4 跨磁带复跑 + 观测通道核查（真实引擎）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_c4_dense_multi.sh   # exit 0, ~65 min
```

```powershell
python tools\rl\v431_c4_multi_summary.py   # exit 0 -> evidence/R25/c4_multi.json
```

```text
== C4 across tapes (segment 600 frames = 10.0 s) ==
 seed  frame  cands  survivors   adm  minOff  maxOff  minAdmF  >=180f    wall
   10   8664     18          1 False      64     374      375      10    71.9
   11   8664     18          0  True      63     544      545       7    71.9
   12   8664     18          0  True      58     520      521      13    72.6

admissible windows: 2 / 3
verdict           : 2 of 3 windows admissible; admissibility is window-specific,
                    not a regime property

== observation channel ==
  STATE fields          : 25
  bullets/boss in STATE : False ([])
  child snapshot verbs  : ['HAZARDS', 'PLAYER', 'EVENTS', 'HUD']
  status                : open - not resolved in this round
```

**两条结论**：

1. **合格是 (窗口, 段长) 的属性**：seed10 的 cand 12 跑满 600 帧，该窗口不合格。
   只跑 seed12 就宣布"密集区间可用"是错的。
2. **臂 A 的观测定义有错**：`STATE` 的 25 个字段里没有任何弹幕/Boss 字段，
   所以预注册里"玩家坐标、power、Boss HP、最近弹幕几何"的臂 A **用 STATE 造不出来**。
   `HAZARDS` 是子进程快照动词，在 `env` 会话上是否可用本轮未验证。

---

## R26 `HAZARDS` 探针：臂 A 能不能看见弹幕（真实引擎，75 s）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_hazprobe.sh   # exit 0, 75 s
```

```powershell
python tools\rl\v431_hazards_evidence.py   # exit 0 -> evidence/R26/hazards_available.json
```

```text
walk: {'frames': 8663, 'hash_checks': 2165, 'hash_mismatches': 0, 'engine_frame': 8663}
  HAZARDS  OK   len= 12778  is_err=False
           head='OK frames=8663 nb=183 nl=0 ne=4 bullets=2:1:8:42d4eea6:...'
  PLAYER   OK   len=   331  head='OK frames=8663 hp=40000000 lives=2 bombs=40400000 power=41600000'
  EVENTS   OK   len=   127
  HUD      OK   len=   172
  STATE    OK   len=   249
arm A choice: A-HAZARDS: bullets are visible on the env session
```

**五个动词在 env 会话上全部可用**，所以臂 A 定为 **A-HAZARDS**（能看见 183 颗弹幕、4 个敌人）。
观测契约：`session.client.hazards()` → `protocol.HazardFrame`，行结构为
`BulletRow(slot,state,color,position,velocity,timer,collision_size,collision_disabled)`、
`EnemyRow(slot,flags1,position,life,timer,world_position,hitbox_width,hitbox_height)`、
`LaserRow(16 字段，含 damageable)`。

**实现坑**：`collision_size` 是**全宽**，致死半宽是 `half_width`；`EnemyRow` 致死半延是
`hitbox_dimensions/3`。把全宽当半宽正是 11-P1 §2.1 点名的旧教师错误。

**成本**：单帧 `HAZARDS` 回复 12.8 KB，所以臂 A 应**每决策查一次**（2 帧）而非每帧。

---

## R28 扩筛到 9 个窗口 + 冻结 train/dev 切分（真实引擎，3.2 h）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_c4_dense_w2.sh   # seeds 13-16
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_c4_dense_w3.sh   # seeds 17-18
```

```powershell
python tools\rl\v431_screen_split.py   # exit 0 -> evidence/R28/screen_split.json
```

```text
== C4 screen at frame 8664, segment 600 frames (10.0 s) ==
 seed  cands  judged  surv    adm  minOff  maxOff  minBlock  >=180f
   10     18    True     1  False      64     374       375      10
   11     18    True     0   True      63     544       545       7
   12     18    True     0   True      58     520       521      13
   13     18    True     0   True      59     518       519       4
   14     18    True     0   True      60     494       495      11
   15     18    True     0   True      59     551       552       5
   16     18    True     0   True      85     448       449      11
   17     18    True     0   True      72     491       492       8
   18     18    True     0   True      57     515       516       7

admissible      : 8 / 9 -> seeds [11..18]
rejected        : [10]
split           : train (13, 14, 15, 16) (4 ok)  dev (11, 12, 17, 18) (4 ok)  disjoint=True
S1b (>=4/side)  : True  (train 4/4, dev 4/4)
segment needed  : 552 frames = 9.2 s (margin 48 frames at 600)
gate integrity  : all episodes 2,165 hash checks = True, mismatches 0
```

**三条结论**：

1. **`S1b` 达成**：9 个窗口 8 个合格，train/dev 各 4，且磁带互不相同。
2. **段长是算出来的**：最强恒定动作（seed15 cand12）活到 **551** 帧，所以段长必须 >552 帧
   （9.2 s）；选 600 帧（10 s），余量 48 帧（8.7%）。
3. **门完整性**：162 个 episode 全部 2,165 次逐帧哈希检查、0 不符。

> 两个操作缺陷：
> ① 用 Python `write_text` 复制 .sh 时把 LF 转成 CRLF，`bash -n` 报 `$'do\r'` —— 必须
> `newline=""`。
> ② 汇总脚本一开始把**未跑完**的窗口记成"不合格"，口径错；已改为 `judged` 与
> `admissible` 分离，且给脚本加 `--seeds/--out`，避免新窗口污染 R25 的历史快照。

---

## R29 臂 A 首次真跑：12 训练 + 10 dev + 10 C-fair（真实引擎）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_smoke.sh     # 3 eps, 224 s
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_fixcheck.sh  # 6 eps 修 bug 后
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_train12.sh   # 12 eps + 存盘
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_eval.sh      # dev seed11 x10
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_cfair.sh           # C-fair seed11 x10
```

```powershell
python tools\rl\v431_arm_a_run.py   # exit 0 -> evidence/R29/arm_a_run.json
```

```text
== what works ==
  train eps 12, full survivals 1, mean 133.9, best 300.0
  dev   eps 10, survivals 0, mean 75.1

== arm A vs C-fair on dev seed 11 ==
  arm A : mean 75.1  median 77.0  max 117.0  survived 0/10
  C-fair: mean 71.8  median 60.8  max 182.5  survived 0/10
  bootstrap mean difference: 3.279 CI95 [-31.1, 33.75] excludes_zero=False
  verdict: no signal

== why ==
  12 episodes = 6% of the prereg budget
    200 episodes ->    3.9 h (one window)     46.9 h (3 seeds x 4 windows)
   1000 episodes ->   19.6 h (one window)    234.7 h (3 seeds x 4 windows)
```

**三条结论**：

1. **回路通了**：12 个训练 episode 里有 1 个跑满 300 决策（完整 10 s），而该窗口最强恒定动作只能活 259。
2. **一个真 bug**：`loss = sum(-logp) × adv` 的梯度量级 ≈ 9×10⁴，把 return 从 84 打到 38；
   改成 `mean(logp) × adv/decisions` + lr 3e-3 + 梯度裁剪后恢复到 83，第 12 个到 300。
3. **dev 无信号**：75.1 vs 71.8，bootstrap CI 跨 0。但**只跑了预注册预算的 6%**，
   所以这**不构成对方法的否定**——只能说按 70.4 s/episode 的代价，跑到能下结论的样本量超出本次预算。

> 操作缺陷：`write_text` 再次把 .sh 的 LF 转成 CRLF（这次改用二进制 `replace(b"\r\n", b"\n")` 修）；
> 以及 job 取消后 WSL 子进程（python + 引擎）仍在跑，必须显式 `pkill` 才算真的停。
> 这两条都已写进本文件的原因：跨 shell/进程边界时，工具返回值不等于事实。

---

## R30 / R30b 前缀 run 结构与 STEP 批量微基准（真实引擎，28 s）

```powershell
python tools\rl\v431_prefix_runs.py                                   # 离线, exit 0
```

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_step_bench.sh      # exit 0, 28 s
```

```text
prefix of 8663 frames, maximal runs of identical masks:
 seed   runs  mean run  max run  reduction
   11   2242      3.86       95      3.86x
   12   2231      3.88       95      3.88x

 group  cmds/block   median ms/frame
     1         250             7.655
     5          50             7.718
    25          10             7.837

speedup (group5 vs group1): 0.992x  batching_helps=False
```

**结论：批量 `STEP` 完全省不了时间**——7.7 ms/帧是引擎仿真成本，不是每命令开销。
70.4 s/episode 已贴着仿真下限（66.7 s）。**客户端层面没有可优化空间了。**

---

## R31 基线画像：`C-fair` 四个 dev 窗口 × 10（真实引擎，37.7 min）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_cfair_dev.sh
```

```powershell
python tools\rl\v431_baseline_profile.py   # exit 0 -> evidence/R31/baseline_profile.json
```

```text
 seed   side  const-best dec  fair n  fair surv  fair mean  fair max  armA mean
   11    dev           272.0      10          0       71.8     182.5       75.1
   12    dev           260.0      10          0       65.3     101.0       None
   17    dev           245.5      10          0      118.0     246.0       None
   18    dev           257.5      10          0       49.6     115.5       None

C-fair pooled : n=40 survived=0 mean=76.2 median=59.5 max=246.0
difficulty    : strongest constant survives 272.0 decisions vs 76.2 for random, i.e. 3.57x
```

**结论：任务的天花板不是随机，是一个恒定键。**
"打败 `C-fair`" 是弱门（随机只有 76 决策）；真门是 `C-const-best` ≈ **250–272**。
臂 A 现在 75.1——差 `C-fair` 一个决策，差 `C-const-best` 一百八十多个。

---

## R32 fork-server 探针：`PARK` / `SPAWN` / `CHILD`（真实引擎，12 s）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_park_spawn.sh
```

```powershell
python tools\rl\v431_park_analysis.py   # exit 0 -> evidence/R32/park_spawn.json
```

```text
role=shadow DISPLAY_unset=True
  reset                  {'ready': '1', 'hash': '83d953750afaea5a', ...}
  warmup_frame           400
ERROR: HarnessError: 'PARK' -> ERR park-unsafe role=shadow-required threads=5 setup=0 null_device=0
```

**发现**：`third_party/th08-rl/src/rlharness/fork.cpp` **已经实现完整 fork-server**：

```text
PARK                 -> OK park=1 template=<id> pid=<pid> frame=<f>
SPAWN <template_id>  -> OK spawned=1 pid=<pid> handle=<h> template=<id>
CHILD <handle> <cmd> -> 转发子进程回复（子进程可答 STATE/PLAYER/HAZARDS 并可 STEP 到 100000 帧）
```

**但 `PARK` 在本构建里被三个前置挡住**：

| 条件 | 要求 | 实测 |
| --- | --- | ---: |
| `threads` | `ThreadCount() == 1` | **5** |
| `setup` | `gameplaySetupState == COMPLETE` | **0** |
| `null_device` | `g_Supervisor.d3dDevice != NULL` | **0** |

第三条是 **harness 内部矛盾**：`null_device.cpp` 只要求 `DISPLAY` 为空，从不填 `d3dDevice`，
于是"唯一允许 fork 的模式"里它恰好是 NULL。

**旁证（有价值）**：shadow 模式能无头启动，`RESET` 哈希 `83d953750afaea5a`
**与 S1c seed12 磁带的 reset 哈希逐位相同**——shadow 与 env 起的是同一初始状态。

**结论**：fork-server 不是"要写"，是"要修门"；修门仍需改 `third_party`，按协作规则 §3 需用户批准。

---

## R33 多帧重试 PARK：是"太早"还是"结构上关着"（真实引擎，148 s）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_park_frames.sh
```

```powershell
python tools\rl\v431_park_frames_analysis.py   # exit 0 -> evidence/R33/park_frames.json
```

```text
 frame  ok  threads  setup  null  hash_ok  reply
   400 False        5      0     0      100  ERR park-unsafe ...
  1200 False        5      0     0      300  ERR park-unsafe ...
  2400 False        5      0     0      600  ERR park-unsafe ...
  4800 False        5      0     0     1200  ERR park-unsafe ...
  8664 False        5      0     0     2166  ERR park-unsafe ...

distinct unsafe triples : [[5, 0, 0]]
invariant across frames : True
was it merely early     : False
shadow walk hash checks : 2166 (mismatches 0)
```

**结论**：三个数字在 frame 400→8,664 之间**一个都没变**，所以 `PARK` **不是时机问题，
是结构上关着**——修复是真的代码改动（3 处判定），不是调参。

**旁证（重要）**：shadow 路径走完 8,664 帧，**2,166 个记录哈希 0 不符**，
与 env 路径逐位一致。一旦 `PARK` 修好，臂可直接搬到 shadow 上跑，观测无需重验。

> 操作缺陷：`request()` 对 `ERR` 回复是**抛异常**而不是返回字符串，
> 第一版探针因此直接结束（只探到一个帧）。必须逐次 `try/except` 才能把多帧都探完。

---

## R34 训练/留出迁移：train 涨、dev 显著变差（真实引擎，1.8 引擎小时）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_cont60.sh   # +60 episodes, 4451 s
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_eval.sh      # dev seed11 x10, 736 s
```

```powershell
python tools\rl\v431_transfer.py   # exit 0 -> evidence/R34/transfer.json
```

```text
== train (seed13) ==
  cumulative episodes        : 72 (12 pretrain + 60)
  first 20 mean              : 87.7 decisions
  last 20 mean               : 108.3 decisions (+20.6)
  full 300-decision survivals: 4 / 60  (best constant on seed13 dies at 259)

== dev (seed11, never trained on) ==
  at 12 episodes : mean 75.1  survivals 0/10
  at 72 episodes : mean 48.5  survivals 0/10
  C-fair         : mean 71.8  survivals 0/10
  dev72 - C-fair: -23.36 CI95 [-53.35, -1.2] excludes_zero=True
  dev72 - dev12 : -26.42 CI95 [-46.7, -6.7] excludes_zero=True

verdict: learned something that pays on the training tape and costs on the held-out tape
cost   : 102 episodes, 1.8 engine hours
```

**结论**：臂 A 在训练磁带上确实学到东西（+23%，4 次跑满一个恒定动作做不到的 300 决策），
但在留出磁带上**显著比均匀随机还差**。这是**记住了 seed13 的弹幕图案**，不是学会躲弹幕。

**方法学后果**：单窗口训练 + 单窗口评测会给出**相反**结论；
跨磁带判据从"最好有"变成"必需"。

> 过程缺陷：评测 runner 复用同一输出文件名，72-episode 评测**覆盖**了 12-episode 原始文件；
> 旧值从 `R29` 冻结证据取回。文件名必须带 checkpoint 标识。

---

## R35 多窗口训练：检验 R34 的过拟合解释（真实引擎，1.76 引擎小时）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_multi4.sh   # 80 eps, 6011 s
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_eval.sh     # dev seed11 x10, 744 s
```

```powershell
python tools\rl\v431_multi_window.py   # exit 0 -> evidence/R35/multi_window.json
```

```text
== multi-window training (13/14/15/16, shared policy) ==
  episodes            : 80
  train mean return   : 101.0 decisions
  first third -> last : 91.6 -> 119.1 (+27.5)
  train survivals     : 3
    seed13: 20 eps, mean 96.7, survivals 1
    seed14: 20 eps, mean 94.1, survivals 1
    seed15: 20 eps, mean 89.3, survivals 0
    seed16: 20 eps, mean 124.0, survivals 1

== dev seed11 (never trained on) ==
  single-window (72 ep) : 48.5
  multi-window  (80 ep) : 57.3
  C-fair                : 71.8
  multi - C-fair        : -14.6 CI95 [-46.4, 10.15] excludes_zero=False
  multi - single        : +8.8 (mean-to-mean)
```

**结论**：多窗口训练确实有效（dev 48.5 → 57.3，4 条磁带分均值全为正），
但**仍低于均匀随机 71.8**，且差距**不再显著**（CI 跨 0）。
这更正了 `R34` 里那个显著差于随机的强调——它只对单磁带策略成立。

> 过程缺陷：`eval-seed11*.jsonl` 的 glob + 取最后一个会挑错文件（字母序），
> 已改为按 checkpoint 命名输出文件并读显式文件名。

---

## R36 臂 A 跑满 200 episode：dev 首次越过随机，但是尾部事件（真实引擎，1.14 h）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_cont48.sh   # +48 eps, 3661 s
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_eval.sh     # dev seed11 x10, 748 s
```

```powershell
python tools\rl\v431_budget_200.py   # exit 0 -> evidence/R36/budget_200.json
```

```text
== arm A budget ==
   12 episodes  seed13 only            ckpt-12   (R29)
   60 episodes  seed13 continued       ckpt-72   (R34)
   80 episodes  4 windows shared       ckpt-80   (R35)
   48 episodes  4 windows continued    ckpt-128  (R36)
  total 200 / prereg 200  -> reaches budget: True

== continuation (+48, four windows) ==
  mean return     : 188.3
  first -> last   : 152.7 -> 219.3 (+66.6)
  survivals       : 2
    seed13: mean 125.6 | seed14: mean 180.9 | seed15: mean 212.2 | seed16: mean 234.4

== multi-window cumulative (128) ==
  train mean 133.8, survivals 5

== dev seed11 ==
  128 ep : mean 78.7  survivals 1/10 (values [43,45,46,47,48,49,50,73,86,300])
  80 ep  : mean 57.3
  C-fair : mean 71.8
  128 - C-fair: +6.78 CI95 [-39.8, 65.6] excludes_zero=False
```

**结论**：预算跑满（200/200）。train 后段 +66.6、累计 5 次跑满 300；
dev **首次出现一次 300 决策的全长存活**（该窗口最强恒定动作只能到 272，随机 0/40），
均值 **78.7 越过随机 71.8**——**但中位数 48.5 低于随机的 59.5**，均值改善由单个尾部事件贡献，
CI 跨 0。**双峰行为：要么早早死，要么跑满。**

> 这一轮**推翻了我跑之前记下的预测**（更多 episode 只会缩小差距、不会改变符号）。
> 符号确实变了，靠一次尾部事件变的。把预测记在证据里、再记它错了，比事后解释有用。

---

## R37 全长存活跨磁带复现：4/40 on 3 个未见磁带（真实引擎，2,264 s）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_arm_a_eval.sh   # seeds 12/17/18 x10
```

```powershell
python tools\rl\v431_dev_all_windows.py   # exit 0 -> evidence/R37/dev_all_windows.json
```

```text
 seed  eps    mean  median    max  surv const-best cfair mean
   11   10    78.7    48.5  300.0     1      272.0       71.8
   12   10   134.7    90.0  300.0     2      260.0       65.3
   17   10   166.1   205.0  300.0     1      245.5      118.0
   18   10    39.9    37.0   60.0     0      257.5       49.6

pooled arm A : n=40 mean=104.8 median=49.5 survivals=4 (windows: [11, 12, 17])
pooled C-fair: n=40 mean=76.2 median=59.5 survivals=0
arm A - C-fair: +28.82 CI95 [-3.24, 63.48] excludes_zero=False
survival rates: arm A 0.100 vs C-fair 0.000  table=[[4, 36], [0, 40]]  Fisher p=0.1156
```

**结论**：`R36` 的单次存活**不是 seed11 的偶然**——全长存活出现在 **3 个不同的未见磁带**上，
而这三个窗口的最强恒定动作（272/260/245.5）**都到不了 300**，随机 40 次 0 次做到。
**但统计仍未达显著**：均值 CI 下界 **−3.24**（差一点），Fisher **p=0.116**。

**下一步算得出来了**：不需训练、不需 `PARK`，只要把 dev 每臂从 n=40 扩到 n=80
（+80 episode ≈ **1.6 h**）；若 10% 存活率保持，**8/80 vs 0/80 → Fisher p ≈ 0.006**。

---

## R38 n=80 评测扩展：效应**变小**，随机也做成了一次（真实引擎，3.13 h）

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v431_seg001/run_eval_ext.sh   # 80 arm A + 80 C-fair
```

```powershell
python tools\rl\v431_n80.py   # exit 0 -> evidence/R38/n80.json
```

```text
 seed  arm n  arm mean  median  surv  ext mean  ext surv  cfair n cfair mean
   11     20      63.3    48.0     1      47.9         0       20       69.2
   12     20     109.2    58.5     2      83.8         0       20       78.0
   17     20     152.8   154.5     2     139.6         1       20      109.7
   18     20      51.0    37.0     1      62.0         1       20       61.8

pooled arm A : n=80 mean=94.1 median=48.0 survivals=6 windows=[11, 12, 17, 18]
pooled C-fair: n=80 mean=79.7 median=63.0 survivals=1
arm A - C-fair: +14.48 CI95 [-6.97, 37.58] excludes_zero=False
survival rates: 0.075 vs 0.013  table=[[6, 74], [1, 79]]  Fisher p=0.1167

n=40 was: arm 104.8 (4/40), cfair 76.2, Fisher p=0.1156
```

**三条对我不利的更正**：
1. **随机策略也跑满了一次 300**（1/80）。`R37` 里"没有任何基线做到"在小样本下成立、n=80 下**不成立**。
2. 均值差 **+28.8 → +14.5**，CI 仍跨 0。
3. 相对优势仍在（存活率 7.5% vs 1.25%，约 6 倍），但两项检验都不显著（Fisher p=0.117）。

**功效估算**：检出 7.5% vs 1.25% 需每组约 **163 episode**（现各 80），补足需再约 **3.3 h**。

> **流程缺陷（差一步就伪造出显著结果）**：第一次扩展没有传 `--policy-seed`，
> 扩展批与首批**逐值相同**（seed11 两次都是 78.7、1/10）——那是 40 个副本。
> 我靠"两批均值一模一样"发现，删掉重跑，并在 runner 里加了"与首批逐值相同就中止"的守卫。
> 若没发现，n=80 会得出一个用副本堆出来的显著性。

---

## R39 引文审计：把每条更正回源到 v4.3.0 文档（离线）

```powershell
python tools\rl\v431_quote_audit.py       # 逐字 8 字符片段
python tools\rl\v431_summary_anchors.py   # 摘要行的关键数字定位
```

```text
rows: 27  found: 8  not found: 6  internal: 13

rows: 6  anchored: 5
    C1 anchored=True  C2 anchored=True  C3 anchored=True
    C7 anchored=True  C9 anchored=True  C14 anchored=False  docs=NONE
```

**审计发现 1 处真缺陷**：`C14` 的第一列把我的实测数字（0.4995/0.5369/0.4249/0.4404）
当成了原报告的说法，而这 4 个数字在 v4.3.0 的 9 个文档里**一个都没有**，
原文说的是相反的"全部劣于恒预测多数类"。已改为显式标注来源，并记为 `C28`。

其余 13 条可核（8 逐字 + 5 数字可定位），所以"更正清单基于伪造引文"这个最坏情形没有发生。

---

## R40 未复现/未执行清单复核（离线）

```powershell
python tools\rl\v431_verify_report.py   # 942 项，含 N 行统计与 §7.2 条目断言
```

```text
§6.3: 14 条 -> 4 条闭合（N7/N8/N9/N12），10 条开放
§7.2: 2 行过时 -> 删除「臂 A 跑到 200 episode」；「跨磁带」拆成评测已做 / 缺 3 个训练种子
§5.1: 「开发祖先 8 个，不够 ≥30 独立留出」 -> 改为「靠教师 4 窗口 × 20 = 80 episode 实现」
```

**这类审计的价值**：清单过时不会让任何数字变错，但会让读者高估未完成量（把已做的当没做）
或低估风险（把需授权的当成自己能做）。本轮修了 3 处，其中 1 处（§5.1 的祖先行）从 `R20` 起就过时了。

---

## R41 早期证据重跑：R1–R6 在当前工作区是否仍可重现（离线，约 6.5 min）

```powershell
python tools\rl\v431_rerun_early.py     # 重跑并比对
python tools\rl\v431_rerun_verdict.py   # 汇总 -> evidence/R41/rerun_verdict.json
```

```text
  R1 v431_recheck_baseline.py       exit=0   0.2s  diff(timestamp)
  R2 v431_recheck_overlap.py        exit=0   3.2s  same
  R3 v431_recheck_duplication.py    exit=0   1.0s  same
  R4 v431_recheck_models.py         exit=0   7.1s  same
  R5 v431_recheck_significance.py   exit=0   4.6s  same
  R6 v431_recheck_actor.py          exit=0 374.3s  diff(float_jitter)

R6 scalar: 3 distinct values in 5 runs, spread 2.07e-10 abs / 1.45e-07 rel;
           last three runs agreed bit-for-bit
```

**结论**：6/6 跑通、4 逐字节相同；`R1` 只差时间戳，`R6` 只有一个标量在 1e-10 量级移动。
报告引用的 `R6` 数字是四位小数，**不受影响**。

> **我自己的假读数（必须记）**：审计第一版每次运行都覆盖快照，于是第二次比较变成
> "当前文件 vs 自己"，打印出"六个全同"。**该读数已撤回**；快照改为只写一次，
> `--compare-only` 拒绝创建快照。
> 教训：**审计工具本身也会给出假阳性**，而它比被审计对象更危险——因为它看起来像证据。

---

## SELF_AUDIT 自检清单（离线，7 条命令）

新增 `Document/Update/Alpha/v4.3.1/SELF_AUDIT.md`，把 R39–R41 三轮自我审计沉淀成可复用清单。
全部 7 条命令实测 exit 0：

```text
v431_verify_report.py     exit=0   （969 项断言）
v431_seg001_gate.py --selftest  exit=0   （24 checks）
v431_rerun_early.py       exit=0   （R1 只差时间戳；R2-R6 本次全同）
v431_rerun_verdict.py     exit=0   （substantive 0）
v431_quote_audit.py       exit=0   （28 行：14 自标 + 9 逐字 + 5 摘要）
v431_summary_anchors.py   exit=0   （6 行中 5 行数字可定位；C14 的 numbers 仍是我的）
v431_arm_a_selftest.py    exit=0   （18/18）
```

**清单内容**：§0 一条命令跑完；§1 数值骨架复跑；§2 引文回源；§3 承诺清单复核；
§4 报告结构断言；§5 判据算术自检；§6 资产与范围检查；
**§7 八条复现性陷阱**（前两条会直接制造假证据：快照被覆盖、评测批次共用 RNG 种子）；
§8 明确列出清单**不**覆盖什么（引擎侧、资产全量校验、方法学正确性）。

**清单自己承认的局限**：快照只写一次，而修复发生在第 3 次运行之后，
所以现在盘上的快照是**修复后的输出**——今天再跑 `R6` 大概率全同，那是基线换了，
不是"当年没差过"。原始 vs 首跑的真实差异只存在于 `R41/rerun_verdict.json` 的常量表里。

**顺带发现 `R39` 的分类会自我改变**：`C14` 改写后自己也变成"逐字可定位"了，
因为新单元格引用了原文那句话——但它的**数字仍是我的**（`summary_anchors` 里 `anchored=False`）。
两者一起看才准确。

---

## R42 逐窗口方差分析：为什么同一份权重在 seed17 得 152.8、seed18 只有 39.9（离线）

```powershell
python tools\rl\v431_window_variance.py   # exit 0 -> evidence/R42/window_variance.json
```

```text
 seed  c4mean   c4max   c4sd  >=200     arm     f10     e10     sd    fair  arm-fair
   11   105.9   272.0   80.2      3    63.3    78.7    47.9   55.5    69.2      -5.9
   12   126.0   260.0   57.6      1   109.2   134.7    83.8   99.7    78.0      31.3
   17   127.8   245.5   81.2      7   152.8   166.1   139.6  100.1   109.7      43.1
   18    87.1   257.5   65.6      2    51.0    39.9    62.0   57.7    61.8     -10.8

between-window sd 40.2  mean within-window sd 80.3  ratio 0.5
batch deltas (ext - first): [-30.8, -50.9, -26.5, 22.1]
rank by arm        : [18, 11, 12, 17]
rank by c4 mean    : [18, 11, 12, 17]
advantage by window: -5.9, 31.3, 43.1, -10.8  monotone=True  exact p(one-sided)=0.0417
```

**三个结论必须一起说**：
1. seed18 是**最难**窗口（18 恒定动作均值 87.1），而臂 A 的排名与该难度排名**完全相同**；
2. **但窗口间 sd 40.2 只有窗口内 sd 80.3 的一半**，n=20 下排名不成立；
3. 真正的信号是**优势随窗口宽阔度单调上升**（4 秩精确 p=0.0417，只是 hint）。

**意义**：臂 A 的"6 倍存活率优势"不是均匀的，而是**依赖窗口有没有余量**。
一个只在有余量的地方有效的方法，**不能**被当作会躲弹幕的证据。

---

## SUMMARY_FOR_DECISION.md：2 页决策摘要（离线）

正文已有约 1,600 行，而决策者需要决定的事只有两件。新增 2 页摘要，其数字与证据**逐条一致**
（由 `v431_verify_report.py` 断言：摘要里的每个关键数字都必须与证据文件里的对应值同时出现）。

```text
SUMMARY rows    < 160 行（断言）
携带的关键数字  1.6586 / 1.7122 / 1.6639 / 879 / 220 / 0.622/255
                94.1 / 79.7 / +14.48 / 0.1167 / 6/80 / 1/80
                −10.8 / +31.3 / +43.1 / 0.0417 / 234.7
两项决定        决定 1（批准改 third_party）与 决定 2（不建议的 3.3 h）
资产声明        我改动的既有文件 = 0；音频全程静音
```

**摘要的结构**：① 结论一句话 → ② 原报告错在哪（4 条）→ ③ 三臂做到/没做到 → 
④ 臂 A 的最终数字 + `R42` 的窗口依赖 → ⑤ 需要你决定的两件事 → ⑥ 我遵守了什么 → ⑦ 证据索引。

**为什么单独写一份**：报告的可核性靠长正文与 42 组证据，但**可读性与可核性是两件事**。
把两者塞进一个文件，会让需要决定的人淹没在证据里，也会让需要核证据的人找不到细节。

---

## R43 基线对平滑方案的敏感性（离线，回应复核文档第 73 行）

```powershell
python tools\rl\v431_smoothing_sensitivity.py   # exit 0 -> evidence/R43/smoothing.json
```

```text
              baseline classes missing        MLE    Laplace    |delta|
      seed12_val__move       9       0   1.658629   1.658860   2.31e-04
     seed12_val__focus       2       0   0.401915   0.402085   1.69e-04
     seed12_val__shoot       2       1  -0.000000   0.001136   1.14e-03
      seed12_val__bomb       2       1  -0.000000   0.001136   1.14e-03
     seed10_full__move       9       0   1.712588   1.712482   1.06e-04
   seed10_suffix__move       9       0   1.712241   1.711807   4.34e-04

move heads have no missing class: True
max |MLE - Laplace| on move     : 4.34e-04 nats
smallest claimed effect (R5)    : 0.0673 nats -> 155.1x larger
```

**结论**：复核文档要求"缺失类别的平滑方案须预先固定"。报告引用的 `move` 头**没有任何类别缺失**，
MLE 有定义；换 Laplace 最多动 **4.34e-4 nats**，比最小效应（0.0673）小 **155 倍**。
唯一的敏感项是 `shoot`/`bomb`——**退化头**（训练标签只有一个取值），已标为零信号仪器属性。

**所以"基线稳健"这句话现在有数字支撑，不是断言。**

---

## R44 把 §2.6 的每个"文档值"回源到 v4.3.0（离线）

```powershell
python tools\rl\v431_documented_trace.py   # exit 0 -> evidence/R44/documented_trace.json
```

```text
                           row      doc                           found in       mine  match
 ActorPolicy seed12-train move   0.0293 ROUTE_AND_CONCLUSIONS.md   0.029261   True
   ActorPolicy seed12-val move   1.8807 ROUTE_AND_CONCLUSIONS.md   1.880707   True
       ActorPolicy seed10 move   1.8837 ROUTE_AND_CONCLUSIONS.md   1.883694   True
      ActorPolicy seed10 focus   0.6461 evidence/W03_DATA_LEG.md   0.646054   True
  ActorPolicy seed10-val focus   0.5589            ** NOT FOUND **   0.558894   True
         16x seed12-train move   1.2827 ROUTE_AND_CONCLUSIONS.md   1.282733   True
           16x seed12-val move   1.5306 ROUTE_AND_CONCLUSIONS.md   1.530596   True
               16x seed10 move   1.6020 ROUTE_AND_CONCLUSIONS.md   1.602032   True
          4x seed12-train move   0.1558 ROUTE_AND_CONCLUSIONS.md   0.155834   True
            4x seed12-val move   2.1299 ROUTE_AND_CONCLUSIONS.md   2.129872   True
                4x seed10 move   2.1161 ROUTE_AND_CONCLUSIONS.md   2.116118   True
         4x seed10-suffix move   2.5070 ROUTE_AND_CONCLUSIONS.md   2.506951   True

documented 11/12; exact 11/11; not documented at all 1
fabricated rows: ['ActorPolicy seed10-val focus']
```

**两条方向相反的更正**：
1. `0.6461` **一直是精确的**（`R6` 的 `batch_mean_ce.focus` = 0.646054）；§2.6 原来只看
   `sample_mean_ce`（0.6425），所以误标 ≈。**"10 精确 + 2 近似"其实是"11 精确 + 0 近似"。**
2. 但 `0.5589` 在 v4.3.0 里**全文 0 命中**——它是我的 `batch_mean_ce` 值被写进了"文档"列。
   该行已删除，记为 `C29`。

**为什么值得单独一轮**：0.5589 与我的实测值**完全一致**，看上去像"完美复现"，
任何数值断言都抓不到。**只有回到源头搜索才能发现它不该在那里。**
这就是"复现"（这个数算得对吗）与"对账"（这个数该出现在这里吗）的区别。

---

## R45 全报告"归因数字"扫描：C14/C29 那类问题还有没有第三处（离线）

```powershell
python tools\rl\v431_attribution_sweep.py   # exit 0 -> evidence/R45/attribution_sweep.json
```

**方法**：逐行扫描报告里**同时含**"文档/原报告/原说法/原文/旧报告/复核文档"**和数字**的行——
那是"把数字归因给原文档"的行；每个数字到 `v4.3.0/**/*.{md,json,jsonl}` 里查字面命中。

```text
numbers checked on attribution lines : 44
absent from v4.3.0                  : 29
          full-precision : 3     （文档值的更高精度形式，4 位小数前缀在文档里）
    mine-self-explaining : 20     （就是解释 C14/C28/C29/C30 的那几行）
         own-measurement : 6      （我的数落在只是提到文档的行上）
mislabelled now: 0
```

**发现第三处**：§3 表格"三盘是否同一条轨迹的变体"的**依据**原写"文档 + `R2`"，
但 `0.463`/`0.456` 在 v4.3.0 里 **0 命中**，文档从未给过任何动作一致率，
只在 `evidence/W09/W09.md:70` 提出过**去重风险**。三个数字全是我算的。
依据已改为仅 `R2`，记为 **`C30`**。

**这类缺陷三次相隔约十二轮**（`C14`→`R39`、`C29`→`R44`、`C30`→本轮），
**每次都是回源才发现的**，而且每次的数值都**完全正确**——错的只是"它该出现在哪里"。

> 实现缺陷：扫描脚本最初按**行号**分类，而修补会让行号位移，于是分类全部失效
> （出现 3 个 UNCLASSIFIED）。已改为**按内容规则**分类（审计标记 / 文档+ 模式 / 高精度前缀），
> 与行号无关。

---

## R46 证据引用完整性：45 引用 / 42 目录 / 0 缺失 / 0 孤儿（离线）

```powershell
python tools\rl\v431_evidence_refs.py   # exit 0 -> evidence/R46/evidence_refs.json
```

```text
distinct R-groups referenced : 45
  with non-empty evidence dir: 42
  missing                    : none
  declared process steps     : [13, 17, 40]

evidence dirs on disk        : 42
  never cited                : none
```

**三个"流程步骤"不是缺失**：`R13`（报告自检）、`R17`（判据算术自测）、`R40`（清单复核）
的输出就是断言本身，没有独立文件。把它们写在 `PROCESS_GROUPS` 里、并要求每条给出理由——
**这样"无文件"是一个被记录的决定，而不是被静默豁免**。

**顺带抓到一处写了三十轮的错数字**：§1 表说判据自测"**25 项**通过"，实际是
`selftest: 24 checks passed`。这个数**不是任何证据文件里的数字，所以没有任何断言去核它**。
已改为 24，并加了两条断言（§1 用 24；旧值只允许出现在记录该更正的三处）。

> 与此前 `C14`/`C29`/`C30` 的区别：那三处是"数字对、位置错"；这一处是**数字本身错**。
> 引用完整性检查的价值正是把"我记得是这样"换成"文件里是这样"。

---

## R47 停止条件能不能执行（离线，`S6` 已重述）

```powershell
python tools\rl\v431_stop_conditions.py   # exit 0 -> evidence/R47/stop_conditions.json
```

```text
  id         status  condition
  S1  NOT_TRIGGERED  恢复门哈希不匹配率 > 5%
        {'episodes_with_a_gate': 162, 'hash_checks': 350730, 'hash_mismatches': 0, 'rate': 0.0}
 S1b  NOT_TRIGGERED  C4 合格窗口 < 4        {'train_admissible': 4, 'dev_admissible': 4}
  S2      TRIGGERED  臂 A ≤ C-fair 或 ≤ C-const-best
        {'arm_mean': 94.1, 'cfair_mean': 79.7, 'const_best': 272.0, 'le_cfair': False, 'le_const_best': True}
  S3  NOT_EVALUABLE  臂 A 学到、臂 B 学不到        （暂不可用；解阻需 PARK 或契约变更）
  S4  NOT_EVALUABLE  臂 B 与 C 都 ≤ 两个基线       （同上）
  S5  NOT_TRIGGERED  引擎墙钟超预算        {'spent_h': 13.5, 'budget_h': 23.0}
  S6      TRIGGERED  同一窗口在 3 s 与 10 s 下的合格判定是否一致（原 S6 的可算等价式）
        {'constants_alive_at_180_frames': {10:10,11:7,12:13,13:4,14:11,15:5,16:11,17:8,18:7},
         'constants_alive_at_600_frames': {10:1,11:0,...,18:0},
         'windows_admissible_at_180': [], 'windows_admissible_at_600': [11..18],
         'windows_whose_verdict_flips': [11,12,13,14,15,16,17,18]}
  S7  NOT_TRIGGERED  训练 100% 但开发 ≤ 基线   {'train_survival_rate': 0.0391, ...}

triggered 2  not_triggered 4  not_evaluable 2
```

**`S6` 的重述是本轮最实质的产出**：原条件用 `alive@120/180`，那是 A1 阶段（3 s 段）的指标，
密集阶段不再产出。但它的**意图**可以不启引擎就算——每个 C4 episode 都存了 `alive_180` 与
`first_death_offset`，于是同一窗口能在两个阈值下各判一次：

| 判定阈值 | 存活 180 帧 | 存活 600 帧 | 合格窗口 |
| --- | ---: | ---: | ---: |
| **3 s（180 帧）** | **76 / 162** | — | **0 / 9** |
| **10 s（600 帧）** | — | **1 / 162** | **8 / 9** |

**8 个窗口的合格判定随段长翻转**——这就是"必须先统一段长再判"的实测依据，
也让 600 帧这个选择不再任意（最短合格段长 521–552 帧，`R28`）。

**`S3`/`S4` 处置不同**：依赖从未运行的臂 B，解阻需修 `PARK` 或放宽契约。
在此之前它们是**当前架构无法检验的假设**，不是待办。

> 两个我自己的口径错误已修：① 各轮证据 JSON 的 `restore_gate` 字段名不同，导致读到 0 次检查，
> 改为直接对 162 个原始 episode 求和（350,730 次）；② `S2` 最初用全窗口最强恒定动作 275.5，
> 而 `S2` 判 dev 侧，应为 272.0。

---

## R48 段长上下文：每个"合格"结论都必须带段长（离线）

```powershell
python tools\rl\v431_segment_context.py   # exit 0 -> evidence/R48/segment_context.json
```

```text
规则：含"多少个合格"的行必须同时含段长标记（3 s / 10 s / 180 帧 / 600 帧 / 段长 / 521 / 552）
豁免：S1b 停止条件本身（按定义就是每侧、每段长分别计数）

lines mentioning admissibility : 42
results without a segment     : 0
```

**初测 1 行违规**（`4.2m` 的 `**8 / 9 合格**`），补成"**10 s 段下** 8 / 9 合格"后归零；
连带补了 §0 两行、`R20` 结论行、`R25` 标题与通过率、`R28` 标题与导语。

**理由**：`R47` 已证明 8 个窗口的合格判定**随段长翻转**（3 s 下 0/9，10 s 下 8/9）。
**一个在两种设定下会给出相反答案的指标，它的每一个数字都必须带着设定走。**

---

## R49 存活率上下文：每个比率都要带 n 与门槛（离线）

```powershell
python tools\rl\v431_survival_context.py   # exit 0 -> evidence/R49/survival_context.json
```

```text
规则：含存活"比率"（百分比或 N 倍）的行必须同时含 n 标记与门槛标记

lines mentioning survival : 46
rates missing context     : 0
```

**初测 10 处违规**（5 处只有"6 倍"这类比率；5 处有 n 但没写门槛），修完后归零。

**为什么必须两个设定都在**：同一句对比在 n=40 时是 4/40 vs 0/40，在 n=80 时是 6/80 vs 1/80——
`R37` 的"随机 0 次存活"到 `R38` 就变成了 1 次，**这正是 n 不写清就能被误读的地方**。

> `R48`/`R49` 合起来的方法论：报告最容易错的不是算错，而是**把依赖设定的量写得像绝对量**。
> "合格"依赖段长、"存活率"依赖 n 与门槛，两者在不同设定下会给出相反结论。

---

## R50 均值是否被 300 决策上限截断，以及一个漏掉的基线（离线）

```powershell
python tools\rl\v431_censoring_audit.py   # exit 0 -> evidence/R50/censoring.json
```

```text
                                     group    n  at cap    mean  median         kind
                         arm A (dev, n=80)   80       6    94.1    48.0  LOWER_BOUND
                        C-fair (dev, n=80)   80       1    79.7    63.0  LOWER_BOUND
             all constants pooled (18 x 4)   72       0   111.7    88.2        EXACT

 seed     arm  @cap    pool    best  arm-pool  arm-best
   11    63.3     1   105.9   272.0     -42.6    -208.7
   12   109.2     2   126.0   260.0     -16.8    -150.8
   17   152.8     2   127.8   245.5     +25.1     -92.7
   18    51.0     1    87.1   257.5     -36.1    -206.6

arm beats the constant pool on 1/4 windows, the best constant on 0/4
```

**两个结论**：
1. **臂 A 均值 94.1 是下界**（6/80 撞到 300 上限），`C-fair` 79.7 也是下界（1/80），
   而恒定动作池 111.7 是**精确**值——所以报告里拿臂 A 与 `C-const-best` 比时，
   **是用对臂 A 最有利的口径在比，它仍输**。
2. **臂 A 只在 1/4 个窗口上胜过"随便挑一个动作固定不变"**（池均值 111.7 > 94.1），
   **0/4 胜过最强恒定动作**。它跑满的次数更多（6/80 vs 1/80），但**平均表现更差**
   （中位数 48.0 vs 88.2）。

**`C33`**：`R27` 定基线时选了"最随机"与"最强"两端，**漏掉最自然的中间基线**。
选两端会让被测方法看起来"比最随机的强、比最强的最弱"，从而**掩盖它不如固定动作**这个事实。
这不是算错，是**比较对象的挑选问题**。

---

## R51 基线阶梯：`C-const-best` 是神谕，可部署版更严格（离线）

```powershell
python tools\rl\v431_baseline_ladder.py   # exit 0 -> evidence/R51/baseline_ladder.json
```

```text
candidates           : 18
oracle per-window    :   258.8  (ORACLE)
const chosen on train:   172.4  (cand #12, DEPLOYABLE)
const pool mean (dev):   111.7  (DEPLOYABLE)
arm A                :    94.1

oracle inflation     : +86.4 over the deployable version (1.5x)
arm A vs oracle      : -164.7
arm A vs train-chosen: -78.3
arm A vs pool        : -17.6
```

**两条结论**：
1. `C-const-best`（每窗口在 18 个里挑最大、又在同窗口评分）是**神谕**，比可部署版本高 86.4。
   所以 `S2` 的"臂 A ≤ `C-const-best`"**对任何非神谕方法都会触发**——不可证伪。
   报告用它论证"不启动像素臂"是**误用**（`C34`）：真正挡住像素臂的是 `_pair_cand` 与 `PARK`。
2. 但换成**可部署**的 172.4（只用 train 窗口选一个动作）之后，臂 A 落后 **78.3**——
   **结论仍成立，而且更硬**：一个诚实选出的固定动作是臂 A 的 **1.83 倍**。

---

## R52 "臂 A 胜过随机"换个统计量就反向（离线）

```powershell
python tools\rl\v431_statistic_flip.py   # exit 0 -> evidence/R52/statistic_flip.json
```

```text
  policy    n    mean    min     q1  median      q3     max  at cap
   arm_a   80    94.1   29.0   36.0    48.0   110.5   300.0       6
  c_fair   80    79.7   29.5   44.6    63.0    98.8   300.0       1

mean diff   +14.4   (arm A better)
median diff -15.0   (random better)
Q1 diff      -8.6   (random better)
Q3 diff     +11.7   (arm A better)
P(arm > cfair median)  35.0%
P(cfair > arm median)  70.0%
```

**符号随统计量翻转**：均值与 Q3 是臂 A 好，**中位数与 Q1 是随机好**。
**七成的随机 episode 比臂 A 的中位数更长**，而只有 35% 的臂 A episode 比随机的中位数更长。

**所以"臂 A 胜过随机"只在"比均值"时成立**，而这个均值靠 **6/80** 跑满的 episode 撑起来。
这是继 `R48`（段长）、`R49`（n 与门槛）之后**第三个"设定一变结论就反"的量**——这次变的是统计量本身。

---

## R53 判据敏感性：「未判定」只对均值口径成立（离线）

```powershell
python tools\rl\v431_criterion_sensitivity.py   # exit 0 -> evidence/R53/criterion_sensitivity.json
```

```text
                         criterion      stat              ci95         points
                       均值 > C-fair     14.43  [-7.59, 37.14]        for_arm
                      中位数 > C-fair     -15.0  [-22.5, -6.5]    against_arm   <- 排除 0
                       Q1 > C-fair      -7.50  [-16.5, -1.0]    against_arm   <- 排除 0
                       Q3 > C-fair      11.70                 -        for_arm
                   均值 > 恒定动作池     -17.61                 -    against_arm
          均值 > train 上选出恒定动作    -78.31                 -    against_arm
                     均值 > 神谕最强   -164.71                 -    against_arm
          跑满 300 的次数（6/80 vs 1/80） Fisher p=0.1167       -   inconclusive

for arm 2  against arm 5  inconclusive 1
median diff CI [-22.5, -6.5]  significant_negative=True
```

**关键**：中位数差 **−15.0，CI [−22.5, −6.5] 排除 0**；Q1 差 **−7.5，CI [−16.5, −1.0]** 也排除 0。
**随机策略在中位数与下四分位上显著优于臂 A。** 均值之所以「不显著但为正」，
是因为 **6/80** 个跑满的 episode 把均值拉了上去。

**两条自加限制**：
1. **多重比较**：8 条判据 2 条显著，未校正时全局零假设下「至少一条显著」≈ **34%**，
   但两条方向相同、机制相同，且与 `R52` 独立算出的符号翻转一致。
2. **预注册 vs 事后**：均值与存活率是**预注册**判据；中位数/Q1/Q3 与可部署基线是**事后补的**，
   只能补充说明预注册判据掩盖了什么，**不能替代它**。

---

## R54 我自己的 36 处自更正朝哪个方向（离线）

```powershell
python tools\rl\v431_correction_direction.py   # exit 0 -> evidence/R54/correction_direction.json
```

```text
     helps : 2      (C23, C24 -- 都是我把负面说过头了)
  helps_bc : 1      (C4 -- 帮 BC，刻意排除在臂 A 这条轴之外)
     hurts : 7      (C22, C27, C32, C33, C34, C35, C36)
   neutral : 26
```

**不是单向的。** 那 2 处"让臂 A 更好"恰恰最说明问题：
`C23`（我曾说"臂 A 没有学习信号"——错，它有）与 `C24`（我曾说"显著差于随机"——过强）
**都是我把负面说过头了**，后来被自己的重算纠正回来。有意唱衰的复核不会产生这种更正。

**但有一个结构性不对称**：**每新增一个基线或判据，被测者只会更难过关。**
加基线是单调的，它不会让臂 A 变好。所以"更差"条数偏多是方法本身的性质，不是判断倾向。

**对读者的意义**：本报告对臂 A 的结论越来越负面，**不代表证据在累积变强**——
7 条里至少 4 条（`C33`/`C34`/`C35`/`C36`）来自"换了更严格的看法"，而不是"跑出了新数据"。
**这两种"更负面"必须分开读。**

---

## R55 反向检验：对臂 A 最有利的判据集 6/8 支持它（离线）

```powershell
python tools\rl\v431_favourable_criteria.py   # exit 0 -> evidence/R55/favourable_criteria.json
```

```text
                         criterion       arm  baseline  favours
                        跑满 300 的次数      6/80      1/80     True
            存活率单侧 Fisher（最宽松的合法检验）  0.075     0.013    False  <- p=0.0584
                           上四分位 Q3     110.5      98.8     True
                前 10 个 episode 的均值     294.8     187.3     True
               ≥200 决策的 episode 比例     15/80      3/80     True
                 单个最好窗口（seed17）的均值     152.8     109.7     True
                       seed17 的中位数     154.5      70.2     True
                  四个窗口里胜出的窗口数（按均值）       2/4         -    False

favours arm: 6/8
one-sided Fisher p = 0.0584  clears 0.05 = False
```

**所以"臂 A 什么都没学到"是错的**：它最好的 10 个 episode 平均 **294.8**（几乎每次跑满），
随机最好的 10 个只有 **187.3**；seed17 上它的中位数是随机的 **2.2 倍**。

**但两件事必须同时说**：
1. **所有有利判据都是"上尾"或"单窗口"判据**，没有一条说"典型 episode 更好"；
2. **最宽松的合法检验也差一点**：单侧 Fisher **p = 0.0584**（双侧 0.1167），仍过不了 0.05。

**两个判据集并排**：`R53`（系统判据）**2 支持 / 5 反对**；`R55`（上尾判据）**6 支持 / 2 反对**。
**同一批数据，换判据集结论就反**——这正是 `C36` 的问题，现在从两个方向都量化了。

---

## R56 三份交付物对臂 A 的结论是否一致（离线）

```powershell
python tools\rl\v431_verdict_consistency.py   # exit 0 -> evidence/R56/verdict_consistency.json
```

```text
     doc arm A mentions    not_significant upper_tail_favours   typical_does_not
  report            133                yes                yes                yes
 summary             14                yes                yes                yes
  prereg             85                yes                yes                yes

SUMMARY §3 (one section, 1384 chars): not_significant=yes upper_tail_favours=yes typical_does_not=yes
complete: 3/3  incomplete: none
```

**三要素**：① 预注册检验不显著（不能说"学会了"）；② 上尾支持它（不能说"没有差别"）；
③ 典型 episode 不支持它（均值不是全部故事）。

**对摘要的额外要求**：2 页文档只有 14 处提到臂 A，所以**三要素必须落在同一节里**
（已断言在 §3 的 1,384 字符内）——否则只读摘要的人会拿到单向结论。

---

## R57 全套离线审计：22 个脚本一次跑完，连跑两遍仍全绿

```powershell
& .tmp_v431_seg001\run_all_audits.ps1   # -> evidence/R57/audit_suite.json
```

```text
ok  tools\rl\v431_seg001_gate.py  exit=0
ok  tools\rl\v431_rerun_early.py  exit=0
  ...（共 22 行）
ok  tools\rl\v431_evidence_refs.py  exit=0
ok  tools\rl\v431_verify_report.py  exit=0
----
non-zero: 0 of 22
```

**第一次跑就抓到两个真问题**：

1. `R39` 的引文审计会随 C 行增加而重算行数（28 → 37），而 `verify_report.py` 写死了 28，
   于是**跑完审计套件之后 verify 反而挂了**。已改为断言**不变量**（`found + not_found + internal == rows`）。
2. **顺序也有依赖**：`evidence_refs.py` 会重写 `R46`，而 verify 断言它——verify 必须排在**最后**，
   否则它比的是**上一轮**的证据。已把 verify 移到末位。

**这条为什么重要**：一份报告的可复核性不是"每个脚本单独能跑"，
而是**"任何人按顺序全跑一遍都通过，且跑完还能再跑一遍"**。前者只是能演示，后者才是能交接。

---

## R58 决策摘要刷新：6 处过时，已修并加断言

```powershell
python tools\rl\v431_verify_report.py   # 1212 项，含 10 条摘要新鲜度断言
```

| 摘要里的说法 | 问题 | 改成 |
| --- | --- | --- |
| "R1–R42 全部证据"、"≈1,600 行" | 证据到 `R57`，正文 ≈1,700 行 | `R1–R57`、≈1,700 行 |
| "`SELF_AUDIT.md`（7 条命令）" | 现在一条命令跑 22 个脚本 | 已改 |
| "rev Y" | 预注册到 **rev AP** | 已改 |
| "约 6 倍存活率优势" | 单向表述，与 `R53`/`R55` 双向结论不符 | 双向 + 172.4 |
| "最强恒定动作 272" | 272 是单窗口最大，口径是神谕均值 **258.8** | 已改 |
| "音频静音已实测为哈希中性" | 只有**单状态**证据（`R18`） | 已加限定 |

**一条自我指涉的断言**：`check("report really is ~1,700 lines", _rep_lines > 1650)`——
正文再长 200 行而摘要仍写"≈1,700"时这条会失败。**摘要不可能再悄悄落后于正文。**

**为什么值得单独一轮**：报告与摘要不一致时，**没人会去读正文**——摘要给的是"最终印象"。

---

## SELF_AUDIT §9：12 条"审计工具自己骗我"的错误

```powershell
python tools\rl\v431_verify_report.py   # 1218 项，含 §9 的 12 行表与元教训断言
```

```text
1  审计打印"六个全同"            快照是证据，不是缓存
2  套件断言永不恢复               断言不能包含被测对象自己的输出
3  分类全变 UNCLASSIFIED          不要用行号做键
4  断言抓不到自己                 断言与被断言量可能同源出错
5  每加行就要改断言               断言不变量，不断言快照
6  跑完套件 verify 反挂           有写依赖的检查要排序
7  "0 非零"是假的                脚本自己的报错会被读成"通过"
8  FormatError                    {1,-40s} 不是合法对齐
9  SyntaxError                    内联 python -c 的引号，一律写脚本文件
10 自己新写的一段被抓出           规则对自己也生效，这是好事
11 "新"批次与旧批逐值相同        重复运行不等于新样本
12 读到错误的历史文件             不要靠排序猜，用显式文件名
```

**元教训**：12 条里有 **5 条**（2、4、5、7、10）的共同点是
**"检查器与被检查对象共享了同一个错误来源"**——同一个变量、同一份计数、同一个退出码。
**好的检查要在结构上与被检查对象分离**：不同的文件、不同的进程、或至少不同的计数来源。

---

## R59 跨节逻辑冲突通读：断言查不出的那一类（离线）

```powershell
python tools\rl\v431_cross_section.py   # exit 0 -> evidence/R59/cross_section.json
```

```text
自读     7 处   D=3 A=2 F=1 B=1
委派通读 24 处  A=12 F=4 C=3 B=2 D=2 E=1
合计    31 处   全部已就地修正或标注
```

**最要紧的五处**：
1. 摘要"臂 A 胜出 **3/4** 个窗口"，逐窗口数据是 **2/4**（3/4 是 n=10 旧数）→ 已改；
2. `C22`（真门是恒定动作）与 `C34`（那是神谕、不可证伪）互相矛盾 → 加交叉引用，收窄为**可部署 172.4**；
3. `S2` 基线有**三种口径**（272.0 / 258.8 / 172.4）+ 脚注"应为 272.0" → 统一为**可部署 172.4**；
4. **预注册正文是 rev E 冻结快照**（180 帧、`alive@180`）→ 顶部横幅列四处差异 + 两处就地标注，**保留原文**；
5. **自证循环**："每个数字都被机检"的依据就是机检，而机检没覆盖的数字（行数、断言数、K/C/N）恰好错了 → 行数改实测 **2,264**，此类计数改为不变量。

**另两处是我自己分析的样本量问题**：
- **172.4 是 4 个确定性值的均值**（258.0 / 180.5 / 221.0 / **30.0**），不是 n=80；**seed18 上恒定动作 30.0 < 臂 A 51.0**；
- §4.2ii 的 **+14.43 是两样本（非配对）**，`R38` 的配对值是 **+14.48**，不可混称。

**为什么这一轮比前面几十轮重要**：前 30 轮改的是**数字**，这一轮改的是**论证**。
数字错了断言会叫；**论证错了没人叫**——它只会让读者在一个自洽但不成立的故事里点头。

---

## R60 只查「结论强于证据」：11 处，10 处确认（离线通读 + 逐条复核）

```text
委派全文通读 -> 11 处 -> 我逐条回原文复核 -> 10 确认 / 1 不确定 -> 11 处全部改写
```

| # | 原表述 | 为什么过强 | 改成 |
| --- | --- | --- | --- |
| **1** | "只要 1.6 h 评测就能推到**可判定**"＋"本轮最重要的可执行结论" | 上一行是"**若** 10% 存活率保持"的条件式外推；`R38` 实测 **p=0.1167** 当场推翻 | **撤回该评价** |
| 2 | "首次全长存活说明**确实**学到了可迁移的东西" | **n=1**，与运气不可区分 | 两种解释都一致 |
| 3 | "多窗口训练**确实有效**" | +8.8 未检验；唯一检验过的是 **−14.6，CI 跨 0** | 去掉"确实有效" |
| 4 | "唯一的胜利**完全**是共同前缀造成的" | 前后对比同时改变 n 与类别比例 | 改为"去掉后消失"+ 声明混杂 |
| **5** | "**唯一**办法/杠杆"、"客户端可省 **0**" | 只测了一种客户端优化（`STEP` 0.992×） | 限定为"**已测的**" |
| 6 | "seed18 **确实最难**" | 4 秩、n=20，排名不成立 | 降级为排序一致 |
| 7 | "96.4% **证明**选的是真实失败状态" | `safe_set` 按定义就是会死 | 声明为定义性一致 |
| 8 | "mtime **证明** third_party 未被触碰" | mtime 可保留；且有既有 dirty 标记 | 改为"未见改动迹象" |
| 9 | "样本量是**唯一**缺口" | `R38` 显示缺口不止样本量 | 改为"当时已知的缺口" |
| **10** | "均值 78.7 **完全由**那一个 300 拉动" | **算术不成立**：去掉后 **54.1**，该次贡献 **24.6（31%）** | 改为定量贡献 |
| 11 | "journal.jsonl 不是我写的" | mtime 只能排除那一次探针 | 改为"非本次探针所改" |

**第 1 条最该记**：把"若……保持"写成"只要……就能"，还给了"本轮最重要"的最高评价，
而**下一节自己的实测就推翻了它**。这类错误不让任何数字变错，**它会直接推动一个不该做的决策**。

**第 10 条是可验算的算术错误**：`[43,45,46,47,48,49,50,73,86,300]` 的均值 78.7（含 300）
与 **54.1**（不含）之差是 **24.6**，所以"完全由它拉动"是假的。

**两轮合起来的方法论**：断言查数字，通读查**说服力**。前者能自动化，
后者只能靠"换一个只会挑一种毛病的人来读"。

---

## R61 只查「两个判据混成一个『优于』」：3 处（离线）

```text
扫 13 处「优于」/ 10 处「更好」/ 4 处「更强」/ 15 处「种子」 -> 3 处混用，全部改写
```

| # | 位置 | 混在哪 | 改成 |
| --- | --- | --- | --- |
| **1** | §0 的"一句话" | "从『不优于』变成『优于』"**没写判据**（§0 其余行都写了 CE） | "在 **CE** 上…；argmax 准确率仍输——**换个判据就反向**" |
| 2 | §0 第 5 行 / §2.5 | "唯一优于基线的臂" | "唯一**在 CE 上**优于基线的臂" |
| **3** | 全篇"种子" | "3 种子"指**训练种子**，别处 `seed` 指**磁带种子** | 6 处改为"3 个**训练**种子"；加术语表 |

**第 3 处最值得说**：它直接支撑"234.7 h vs 9 h"这个决策项。若读者把"3 种指 × 4 窗口"
读成 3 条磁带 × 4 个窗口（或 12 条磁带），他对"要不要批准修 `PARK`"的判断就建立在错误成本上。
已加术语表并写明"**不是** 3 条磁带 × 4 窗口，也**不是** 12 条磁带"。

**检查过但写对了的**（以示不是漏查）：§2.5 分两段并列写 CE 与 argmax 并声明不矛盾；
§9 的一句话写了 CE；§4.2hh/§4.2ii 每判据单独成行；§5.4 的 S6 明写"同一窗口在 3 s 与 10 s 下"。

**顺带根除两个漂移类**：摘要不再写"R1–Rxx"的范围，也不再引行数——
它们每轮都会变，而"变了没改"正是本轮抓到的问题之一。

---

## R62 接手入口：索引从磁盘生成（离线）

```powershell
python tools\rl\v431_make_index.py   # -> Document/Update/Alpha/v4.3.1/README_INDEX.md
```

```text
documents 5  groups 57  evidence 2083 KB
```

**为什么生成而不是手写**：索引是最容易漂移的文档——它引用的每个数字都会变。
现在它是**从 `evidence/R*/` 与四份文档现场算出来的**，新增证据后重跑即可。

**并且加了完整性断言**：每个证据组必须有一条**人工描述**（`DESCRIPTIONS`），
没有就 `assert` 失败。否则索引会静默退化成 `v431-recheck-r3/1` 这种无信息字符串——
那比没有索引更糟，因为它看起来像有内容。

索引内容：① 按"你要做什么"分四条路径；② 三十秒版本；③ 57 组证据各一行说明；
④ 一条命令复跑全部检查；⑤ 资产与范围声明。

---

## R63 产物缺席：把「没找到」变成「搜过」+ 分类（离线，约 4 min）

```powershell
python tools\rl\v431_artifact_absence.py   # exit 0 -> evidence/R63/artifact_absence.json
```

```text
files seen 148,987  dirs 97,602
checkpoint-like        : 5146
prediction-like        : 780
files naming a BC config: 53
```

**逐个分类后没有一个是训练好的 BC 权重**：
1,198 个在 `.tmp_v420_data`（早期引擎实例）；~1,100 个在 `.tmp_r*` / `.tmp_v400_replay_*`（早期 replay）；
177 个在 `.tmp_v430_pytest_w0*`（v4.3.0 的**测试**目录，`.bin`/`.pt` 是夹具）；
**唯一落在 v4.3.0 期的 `.tmp_w03g_ckpt.bin` 只有 8 字节，内容就是字面量 `SENTINEL`**。
780 个预测类文件同样全在夹具目录。同批 `.tmp_w03g_*.json` 是 `v430-train-bc/1` 的**合成**信封
（`dataset_kind: synthetic`、`status: not_run`）。

**顺带发现一个 v4.3.0 自身的瑕疵**：`.tmp_w03g_bad1.json` 带 **UTF-8 BOM**，
普通 `json.loads` 会抛 `Unexpected UTF-8 BOM`。

> **为什么这个脚本不在 22 个的套件里**：它要遍历 14.9 万个文件（约 4 min），
> 而结果在资产不变时是稳定的。套件保持快（~90 s），这条单独跑并在 `SELF_AUDIT` 里注明。

---

## R64 穷尽 N4/N5：N5 写错了，且定位了 0.8537 的出处（离线，约 4 min）

```powershell
python tools\rl\v431_n4_n5.py   # exit 0 -> evidence/R64/n4_n5.json
```

```text
N4: 2789 log-like files, 0 mention a BC config -> holds=True
N5: bc_generalisation.json  3 epochs (800/200)
    bc_split_comparison.json 2 runs x 3 epochs (800/200)
    cross_tape_generalisation.json  no history key
majority: seed12-train/seed12-val share 468/1099 = 0.42584 ; -ln = 0.8536876584859746
```

**三个结论**：
1. **`N4` 成立**（0 / 2,789）；
2. **`N5` 写错了**——per-epoch 曲线**存在**，只是只有 3 epoch、且跑在 **n=1,000（800/200）** 上；
   正确表述是"没有那三个 25-epoch 配置的曲线"；
3. **`0.8537` 的出处定位到产物**：`move_majority_share = 468/1099`，`−ln` 就是 0.8536876584859746；
   且 seed12 的两个子集**共用按整条磁带算的 majority**——`C3` 现在有**物证**。

**另记一条样本量事实**：两个聚合文件跑在 **n=1,000（800+200）**，磁带是 **1,099**——
v4.3.0 至少混用了两种样本量，而"22% 留出"在不同文件里分母不同。

> **我自己也在这里踩了一次**：第一版脚本把 `isinstance(entries, list)` 写成 `isinstance(d, list)`，
> dict 分支永不成立，于是**漏报了三条真实存在的曲线**。是"跑出来看着合理"救不了它，
> 只有手工把文件打开逐字看才发现。

---

## R65 「未复现」清单的证据类型审计（离线）

```powershell
python tools\rl\v431_evidence_types.py   # exit 0 -> evidence/R65/evidence_types.json
```

```text
 id  sources  claim
 N1        2  880/219 划分从未存在（实际 879/220）
 N2        2  三个模型的 checkpoint 从未保存
 N3        2  逐样本预测从未保存
 N4        2  三个实验没有 stdout
 N5        2  没有 25-epoch 的 per-epoch 曲线
 N6        2  cross_tape_honest.py 的输出不在盘上，但 2.5070 已复现
```

**补上两个第二来源的过程本身有价值**——它们都**扩大了范围**：

1. **`N4` 原来只说"仓库里没有"**。W00 文档写明 stdout 抓取在
   `%TEMP%\w00_probe_restore_20261115\out\`（**仓库外**）。该目录**仍在**（20 个文件），
   逐一搜索：**0 个含 BC 训练内容**。所以 `N4` 在扩大范围后**仍然成立**。
2. **`N5` 的第二来源在文档自己身上**：`W03_DATA_LEG.md` §8.2 只保留了 25-epoch 那次
   （**n=1,099**）的 **3 个点**（epoch **1/6/25**）。而 JSON 里的 3-epoch 曲线属于
   **n=1,000** 的冒烟运行。两处都指向"没有完整曲线"。

**这条本身值得记下来**：**「未复现」不是"我放弃了"，它应该和"已复现"一样有证据等级。**

---

## R65b 用同一标准复查 N7–N14：又抓到三处描述不准

```powershell
python tools\rl\v431_evidence_types.py   # 现在覆盖 14 条
```

| # | 原描述 | 问题 | 改成 |
| --- | --- | --- | --- |
| `N7` | "**10/10** 文档数字逐位重现" | `R44` 之后是 **11/11** | 改正 + 注明 `R41` 复跑 5 次稳定 |
| `N9` | "密集窗口上 3 s 也不合格（**13/18**）" | 13/18 是 **seed12 一个窗口**；9 个窗口实际 **4–13** | 改为区间 |
| `N14` | "`third_party` **未被**本会话触碰（mtime 均早于会话）" | 与 `SELF_AUDIT` §6 改掉的**同一类过强断言** | 改为"没有发现我改过它的证据" |

**扩展后的 14 条里，唯一单源的是 `N13`**（双 mask 决策 137 个只由 `R22` 统计过一次）。
而它决定的正是"**要不要重录数据**"这个**需授权**的资产决策——**不构成"我自己说了算"的缺口**。
写出来比含糊地说"都有两个来源"更诚实。

---

## R66 把「7 个脚本都这样」逐条列出来：其实是 5 + 1，两种语料

```powershell
python tools\rl\v431_split_scripts.py   # exit 0 -> evidence/R66/split_scripts.json
```

```text
          bc_generalisation.py    33  fixed_n_val  n_val = 200
        bc_split_comparison.py    99    ratio_0.8  cut = int(N * 0.8)
  cross_tape_generalisation.py    97    ratio_0.8  cut = int(n * 0.8)
          cross_tape_honest.py    69    ratio_0.8  cut = int(len(tr["y"]) * 0.8)
           small_model_test.py    64    ratio_0.8  cut = int(n * 0.8)
        small_model_test_d4.py    64    ratio_0.8  cut = int(n * 0.8)

counts: 6 scripts with a split (5 ratio + 1 fixed) of 102 total
1099 -> 879/220    1000 -> 800/200
```

**两处修正**：① 用 `0.8` 的是 **5 个**，不是 7 个；② 还有 **1 个**写死 `n_val = 200`。
**且它们跑在两种语料上**——`bc_generalisation.py` 自己的文档串就写着
"800 train / 200 validation split of the **1000-sample corpus**"。

**`N1` 的结论仍成立，但理由改写**：实际存在的切分只有 **879/220**（1,099 磁带）与
**800/200**（1,000 子集）两种，**880/219 哪个都不是**。这比原来"所有脚本都是 879/220"更强，
因为它不依赖"脚本数恰好是 7"这个我从未核对过的细节。

---

## R67 两条全称断言：一个成立，另一个的数字**无法复现**

```powershell
python tools\rl\v431_agreement_18.py   # exit 0 -> evidence/R67/agreement_and_18.json
```

```text
断言 B（每窗口 18 个恒定动作）：9 窗口 × 18 = 162，distinct cand 全为 18 -> 成立

断言 A（全盘一致率 0.448 / 0.463 / 0.456）：
                pair  prefix     full  suffix_len   suffix
      seed10__seed11     219   0.4568         880   0.3216
      seed10__seed12     219   0.4850         880   0.3568
      seed11__seed12     225   0.4823         874   0.3490
claimed: [0.448, 0.463, 0.456]  reproducible: False
```

**两种口径都不是那三个数**，而且全库搜索**没有任何脚本或证据产出过它们**——
唯一的出现处是我自己的叙述。**所以它们是我从早期笔记继承进报告、却从未核过的数字。**

这比 `C29`（我把自算值写进"文档"列）**更严重**：`C29` 至少那个值是真的；**这三个我根本算不出来**。

**但结论反而更强**：全盘一致率 0.46–0.49 只比多数类占比 0.4258 高一点、离 1.0 很远；
**去掉共同前缀后只剩 0.32–0.36**——三盘在前缀之后**分歧更大**。

**教训**：一条断言如果"看起来像常识"，它最容易带着未核的数字活下来。`R66`/`R67` 各抓到一个，
两个都是**全称/概括性**的句子。

---

## R68 两张表的「差」列同名反号

```powershell
python tools\rl\v431_sign_conventions.py   # exit 0 -> evidence/R68/sign_conventions.json
```

```text
             set     n   model     base      m-b      b-m  CI(b-m)
      seed12_val   220  1.5306   1.6586   -0.128   0.1280  [0.035, 0.2215]
     seed11_full  1099  1.5568   1.6755   -0.119   0.1187  [0.0765, 0.1602]
     seed10_full  1099  1.6020   1.7126   -0.111   0.1106  [0.0657, 0.1549]
   seed11_suffix   874  1.5903   1.6639   -0.074   0.0736  [0.0265, 0.1206]
   seed10_suffix   880  1.6450   1.7122   -0.067   0.0673  [0.0149, 0.119]
```

**五对全部复现，5/5 的 95% CI 全部排除 0。** 但报告里两张表**都有一列叫"差"**：
§2.5 基线比较表用 `模型−基线`（负 = 更好），bootstrap 表用 `基线−模型`（正 = 更好）。
**同一对 (1.5306 vs 1.6586) 在一张表里是 −0.128、另一张是 +0.1280**，而两处都没写口径。

**不是数错，是两套符号共用一个名字**（C 类）。已把列名写成
`差（模型−基线；负 = 更好）` 与 `差（基线−模型；正 = 更好）`，并统一正文四处。

顺带：`−0.067` 是 `−0.0673` 的三位舍入、`−0.128` 是 `−0.1280` 的三位舍入——**两个端点都带舍入**。

---

## R69 104 张表的表头口径审计

```powershell
python tools\rl\v431_headers.py   # exit 0 -> evidence/R69/headers.json
```

```text
before:  tables 104  distinct cells 266  numeric cols needing a unit 25
after :  tables 104  distinct cells 268  numeric cols needing a unit 0
```

**最实质的一处是 `n`**：

| 出现处 | `n` 的含义 | 行值 |
| --- | --- | --- |
| CE 对比表 | **样本数** | 220 / 1099 |
| 臂 A 各表 | **episode 数** | 10 / 20 / 80 |

**同一个字母两种东西**，而这两个数恰好都用来支撑结论（CE 的显著性 vs 存活率的显著性）。
跨表比较时会被读成同一个量。

**已修 25 处**：`n` → `n（样本）`/`n（episode）`；`均值/中位数/最大值/Q1/Q3/标准差/平均差/跑满` → 加（决策）；
`用时` → （s）；`占` → （%）；`合格数` → （窗口）；`数量` → （文件）；`规模` → （episode / 份）。

**结论**：`R68` 的"差"**不是孤例**——表头普遍假设读者知道上下文。这类缺陷不改变任何数字，
但**让数字在跨表阅读时悄悄改变含义**，而那正是最容易被忽略的错法。

---

## R70 107 个百分比里只有 9 个能自证；我的第一版检查器报了 25 个假错

```powershell
python tools\rl\v431_percentages.py   # exit 0 -> evidence/R70/percentages.json
```

```text
percentages 119  self-checking 12  hold 12  fail 0
manual median crossings: arm>cfair_med 28/80 = 0.350, cfair>arm_med 56/80 = 0.700  holds=True
```

**9 个自带分子分母的百分比全部成立、0 个失败。** 另外 98 个**不自证**——不是错，
但读者无法从文本核对。三个最载重的已补上分母并直接用原始 episode 复核：
70% = **56/80**、35% = **28/80**、7.5%/1.25% = **6/80、1/80**、3.91% = **5/128**。

**但这一轮最重要的是我的第一版检查器错了**：它把每个 `X%` 与**同行最近的 a/b** 配对，
报了 **25 处"不符"——全部是假的**：

- "约 20% 训练数据原文"旁边是"0/219 相同"——**原报告的说法 vs 我的更正**，两个量；
- "95% CI"被当成 95% 与 `5/5` 配对——**置信水平不是百分比份额**；
- "42.6%"是多数类先验，`342/880（38.9%）` 是一致率——两个量。

**按距离配对不是证据。** 改成"只认自带分子分母"后，结论从"25 处错"变成
**"0 处错、但 98 处不可核"**——后者才是真实状态。

---

## R71 成本链重算：9/9 成立，但 70.4 s 是满长上界；抓到 90→72

```powershell
python tools\rl\v431_cost_chain.py   # exit 0 -> evidence/R71/cost_chain.json
```

```text
                 3 训练种子 × 4 窗口 × 1,000 episode   12000   234.70   234.67 True
                            计账 643 episode     643    12.60    12.57 True
measured R20: 72 eps in 1282 s = 17.8 s/episode   [states_complete x 18; wall time in evidence json: False]
measured R31: 40 eps in 2264 s = 56.6 s/episode   [c_fair_pooled.n; wall time in evidence json: False]
```

**9/9 全部成立**（偏差 ≤0.06 h）。但 **70.4 s 是满长 episode（300 决策）的成本**，
而实测 `R20` 是 **17.8 s/episode**、`R31` 是 **56.6**——早死的便宜得多。
所以用 70.4 乘含死亡的组得到的是**上界**："计账 643 ≈ 12.6 h"是上界。

**顺带抓到真错**：§4.2f 原写"**90 个 episode** / 1,282 s"，而 4 状态 × 18 = **72**；
`R20` 证据是 `states_complete: 4` 且每状态 `complete_18: true`，**§7.1 本来写的就是 72**。
记为 `C40`。这 72 个也给出真实单位成本 **17.8 s/episode**。

> 附注：`R31` 的 wall time **不在它的证据 JSON 里**（只有 `c_fair_pooled.n`），
> 所以那 2,264 s 只能标为"运行记录"。这不是错，但它是**不可从证据文件复算的输入**。

---

## R72 §6 的 K 表引用核验：6 条无引用、K9 带已证伪数、K3 引用错组

```powershell
python tools\rl\v431_k_rows.py   # exit 0 -> evidence/R72/k_rows.json
```

```text
before:  K rows 18  numbers 44  found 30   no citation: [3, 7, 8, 9, 10, 11]
after :  K rows 18  numbers 49  found 42   no citation: []
```

**① 6 条完全没有引用**：`K3`/`K7`/`K8`/`K9`/`K10`/`K11`，已全部补上。

**② `K9` 还带着已证伪的数**：仍写"全盘动作一致率 **0.448/0.463/0.456**"——
我在 `R67` 里证伪了这三个数、改了 §3 与 `C38`，**却漏了 K 表里同一句话**。
现改为实测的 **0.4568/0.4850/0.4823**（去前缀后 **0.3216/0.3568/0.3490**）。记为 `C41`。

**③ `K3` 的引用错组**：三个基线统一引 `R1`，但 `R1` 的 `baselines` 只算了
`seed12_val`、`seed10_full`、`seed10_suffix` —— **没有 seed11**。
`1.6639`（seed11 后缀）与 `1.6755`（seed11 全盘）**只存在于 `R4`**。
这不是数错，是**引用指错了文件**；而它恰好指向用户问题 2 的核心数字。记为 `C42`。

剩下 8 个"找不到"的多数是**派生量**（`563 = 149+149+133+132`、`155` 是比值）
或**格式差**（4 位小数 vs 2 位），已逐条看过，不是错。

---

## R73 按**值**反查残留：0 处真残留，29 个假阳性

```powershell
python tools\rl\v431_value_leftovers.py   # exit 0 -> evidence/R73/value_leftovers.json
```

```text
tokens 11  occurrences 55
classes {'retraction': 23, 'subject': 15, 'correct-use': 15, 'substring': 2}
unreviewed: 0
```

| 分类 | 处数 | 说明 |
| --- | ---: | --- |
| **更正语境** | **23** | 引用旧值以更正它 |
| **主题对象** | **15** | 旧值是这段话的主题（"880/219 从哪来"整节在证明它不存在） |
| **本来就正确** | **15** | `1.707` 是演算值、`1.6 h` 是成本、`78.7` 是实测均值 |
| **子串误匹配** | **2** | "5 倍"命中在"**155** 倍"里 |

**没有一处把旧值当作现行结论**——`C38`→`K9` 那类残留已清零。

**但真正的产出是那 29 个假阳性**：第一版用关键词启发式把 29 处判成"仍在用"，**读完一处都不成立**。
误报必然出现在三种情况：**主题**（旧值是讨论对象）、**正确用法**（`1.6 h` 作为成本是对的，
被撤回的是"只要 1.6 h 就能判定"这个**外推**）、**子串**（`5 倍` 在 `25 倍`/`155 倍` 里）。

**和 `R70` 的按距离配对是同一类错误**：机械规则替代不了阅读。
工具的价值只在**枚举**——55 处可读，2,400 行不可读；**分类必须靠读**。

---

## SELF_AUDIT §9 补两条：机械规则替代语义判断（13、14）

这两条不是新脚本，是把 `R70`/`R73` 的教训写进自检清单。

| # | 症状 | 根因 | 修法 | 教训 |
| --- | --- | --- | --- | --- |
| 13 | `R70` 报"**25 处百分比不符**" | 把 `X%` 与**同行最近的 `a/b`** 配对 | 只认**自带分子分母**的形式，其余标"不可核" | **按邻近性配对不是证据**；实测 25 处全假 |
| 14 | `R73` 报"**29 处旧值残留**" | 用关键词判"引用以更正 vs 仍在用" | 保留枚举、**分类逐处人工**，结果里写明假阳性数量与类型 | **机械规则替代不了阅读** |

**两条元教训**（`SELF_AUDIT` §9 末尾）：

1. 14 条里 **5 条**（2、4、5、7、10）是**"检查器与被检查对象共享同一错误来源"**——
   好的检查要在结构上分离（不同文件/进程/计数来源）。
2. 另有 **2 条**（13、14）是**"用机械规则替代语义判断"**——这类检查器不是随机出错，
   而是**系统性地在三种情况误报**（主题、正确用法、子串），所以输出**看起来像一批真发现**。
   可操作做法：**工具只做枚举与归一，分类留给读**。

---

## 一条命令自查（给人用的聚合器，不是断言门）

```powershell
python tools\rl\v431_selfcheck.py          # 5 个廉价检查，实测 0.2–0.3 s
python tools\rl\v431_selfcheck.py --full   # 加 6 个，含两次约 4 min 的全树遍历
python tools\rl\v431_selfcheck.py --list   # 只列出会跑什么
```

```text
mode: fast   checks: 5
  [ok  ] R69 表头口径            0.04s
  [ok  ] R70 百分比自证           0.07s
  [ok  ] R71 成本链             0.04s
  [ok  ] R72 K 表引用           0.05s
  [ok  ] R73 旧值残留           0.06s
non-zero: 0 of 5   总耗时 0.3s
```

它按**证据层**而不是按节检查：表头口径（`R69`）、百分比自证性（`R70`）、成本链（`R71`）、
K 表引用（`R72`）、旧值残留（`R73`）；`--full` 再加产物缺席（`R63`）、N4/N5（`R64`）、
证据类型（`R65`）、切分脚本（`R66`）、全称断言（`R67`）、符号口径（`R68`）。

> **为什么单独一条命令而不是并进套件**：套件是**断言门**（`v431_verify_report.py` 最后跑，
> 断言覆盖全部已存证据）；这条是给**人**用的聚合器——它自己不做断言，只汇总各检查的退出码。
> 把两次全树遍历塞进断言门只会让套件慢一倍，**不增加任何保证**。
> 每个检查是**独立进程**，一个崩了不会掩盖其余。

---

## R74 臂 B/C 的阻断到底卡在哪（读源码 + 运行时常量）

```powershell
python tools\rl\v431_pair_cand_scope.py   # exit 0 -> evidence/R74/pair_cand_scope.json
```

```text
pair_cand def: L3924
call sites   : [(3989, 'replay')]      <- 只有一个，在 replay() 里
constants    : DECISION_FRAMES=2  TEACHER_REPLAN_DECISIONS=1  COMMIT_FRAMES=2
replay counts mismatch, not raises: True   first_mismatch recorded: True
live arm one mask per block: True
```

**结论（`C43`）**：报告原写"obs-v3 需要逐决策动作历史，`_pair_cand` 拒绝双 mask 决策"——
**把两件真事粘成了一句错的因果**。实际是：

1. `_pair_cand` **只在 `replay()` 里被调一次**，校验的是**录像前缀**；
2. 实时臂 `COMMIT_FRAMES=2` 且**只命名一个 mask**，历史**按构造**合法；
3. **阻断 = 到不了起始帧**：磁带 prefix 从 frame 92 起有双 mask 决策，`replay()` 第 92 帧抛错，
   而合格段落在 frame 8,664；
4. `replay()` 对哈希不符**只计数不抛错** ⇒ 存在"**可到达但无效**"的路（起始状态不是录制状态）。

**四条解阻选项因此是穷尽的**：存档（无此动词）／重录干净磁带（需授权）／
改用前缀 <92 帧的段（`C4` 已判 0/4）／替换 prefix（可到达但无效）。

---

## 0.0 任务要求覆盖表（本轮新增，直接面向裁定）

报告开头新增一张**要求覆盖表**：5 项子任务 + 7 项要求，逐条给出“在哪 answered / 状态”。
状态只有四种：**已完成 / 部分完成 / 被阻断（需授权）/ 未触发（前置不成立）**。

摘要里也放了 5 行的精简版（`SUMMARY_FOR_DECISION.md` §0）。

**两处“不是已完成”都写了原因**：
- 臂 B/C **被技术阻断**——`R74` 已把原因从“不能提交两个 mask”改为“**到不了起始帧**”；
- 升级阶梯第 2 步 **未触发**——第 1 步的成功率检验 p=0.0584（单侧）⇒ 无信号，按预注册**不进入**扩样本。

---

## R75 存活率检验的功效与粒度（离线精确枚举，0.6 s）

```powershell
python tools\rl\v431_power.py   # exit 0 -> evidence/R75/power.json
```

```text
observed: 6/80 vs 1/80 -> one-sided Fisher p = 0.058373

  n/arm   extra eps   p=0.040  p=0.050  p=0.075  p=0.100  p=0.150
     80           0      9.7%    18.4%    46.2%    71.4%    95.0%   (0.0 h)
    120          80     20.2%    34.0%    68.2%    88.9%    99.4%   (1.26 h)
    163         166     31.0%    49.2%    84.0%    96.8%   100.0%   (2.61 h)
    240         320     48.8%    70.4%    95.8%    99.7%   100.0%   (5.03 h)
    320         480     63.4%    83.5%    99.0%   100.0%   100.0%   (7.55 h)

null rejection rate by n: {'80': 0.00126, '163': 0.00839, '320': 0.01927}
smallest rejectable point: {'80': {'k_in_arm_needed': 5, 'p_at_that_point': 0.0293}}
```

**三件事一起看才有意义**：

1. **那 3.3 h 买的是**：真值 5% → **49%（约抛硬币）**；7.5% → **84%**；4% → **31%**；
2. **这个检验在低事件率下保守**：名义 α=0.05，实际拒绝率只有 **0.126%**（n=80）/ 0.839%（163）
   ⇒ **`p=0.0584` 不该读成"刚差一点"**，这个规则的真实一类错误率是 0.1%–2%；
3. **粒度很粗**：n=80 时最极端可达点是 **臂 5 / 对照 0 → p=0.0293**，实测 **6 / 1 → 0.0584**——
   **差的就是那一个对照成功**。

**独立复现**：精确枚举重写的 Fisher 给出 `0.058373`，与报告 `0.0584` 一致。

> **第一版把超几何分母写成 `C(total, n1)`**（应为 `C(total, 总成功数)`），算出 p≈1e-37、
> 功效全 100%。**是这张表自己把自己抓出来的**——加一步"零假设下拒绝率应≈5%"的自检就够了。
> 这一步值得成为所有检验类工具的默认动作。

---

## R76 判据校准审计：Fisher vs Boschloo vs 二项（离线，0.09 s）

```powershell
python tools\rl\v431_boschloo.py   # exit 0 -> evidence/R76/boschloo.json
```

```text
observed 6/80 vs 1/80
Fisher   p = 0.058373   level (sup over p0) = 3.0006% at p0=0.139
Boschloo p <= 0.034291 at p0=0.13   (region size 150)
one-sample vs p0=1/80: p = 5.20142e-4 ; at the control CI edge 6.8%: P = 0.4634
```

**三个数必须一起看**：

| 口径 | p | 怎么读 |
| --- | ---: | --- |
| **Fisher**（预注册） | **0.0584** | 判据未通过；但**水平 3.0%** 与**实测速率下的拒绝率 0.126%** 是两个不同的 α，后者才是本实验相关的 |
| **Boschloo** | **≤0.0343** | 合法 α=0.05 检验，显著；**但上确界落在 p0≈13%**（实测对照率的 10 倍），所以这个数值由远离数据的区域驱动 |
| 单样本二项 | 0.00052 | **不可引用**：把对照率当已知；在上界 6.8% 下同一个数变成 **0.46** |

**结论（`C44`）**："预注册判据未通过"**不能**写成"无信号"。
一个在零假设下拒绝率 0.126% 的判据**几乎不可能**给出 p<0.05。

---

## R13 报告自检

```powershell
python tools\rl\v431_verify_report.py
```

- 退出码：**0**
- 输出：`checks run: 1428` / `all report figures match the stored evidence`
- 它从 `evidence/R1`–`R38`（报告定稿） 里重新读出报告引用的每个数字并断言，任何不符即失败。

---

## R6 `ActorPolicy`（2,726,671 参数）25 epoch 重执行

```powershell
python tools\rl\v431_recheck_actor.py
```

- 退出码：**0**（第三次尝试；前两次被会话中断，均未产出 JSON）
- 产物：`evidence/R6/actor_recheck.json`，日志 `.tmp_v431_r6.log`
- 关键输出：

```text
train 879 val 220
params 2,726,671
...
== move / focus vs TRAIN-FIT baseline ==
set                 n |  move CE    base              d +CI    acc    maj |  focus CE    base              d +CI
seed12_train      879 |   0.0293  1.6937 +1.6644[+1.6071,+1.7226] 1.0000 0.4198 |    0.0105  0.4515 +0.4409[+0.4015,+0.4806]
seed12_val        220 |   1.8504  1.6586 -0.1918[-0.4007,+0.0182] 0.4864 0.4500 |    0.5265  0.4019 -0.1245[-0.2808,+0.0199]
seed10_full      1099 |   1.8680  1.7126 -0.1554[-0.2607,-0.0502] 0.4995 0.4249 |    0.6425  0.4299 -0.2126[-0.2876,-0.1395]
seed10_suffix     880 |   2.2266  1.7122 -0.5144[-0.6241,-0.4033] 0.4034 0.4364 |    0.7731  0.4420 -0.3311[-0.4218,-0.2455]
seed11_full      1099 |   1.8175  1.6755 -0.1420[-0.2525,-0.0346] 0.5369 0.4404 |    0.5643  0.4123 -0.1519[-0.2236,-0.0827]
seed11_suffix     874 |   2.1756  1.6639 -0.5117[-0.6315,-0.3952] 0.4462 0.4588 |    0.6806  0.4199 -0.2607[-0.3466,-0.1802]
```

（上表 CE 为**逐样本**平均。原脚本用 **batch 平均（bs=128）**，见下。）

```powershell
python tools\rl\v431_r6_report.py     # exit 0，打印两种平均的对照
```

```text
set                n  move batch-mean  move sample-mean  doc value
seed12_train     879           0.0293            0.0293     0.0293
seed12_val       220           1.8807            1.8504     1.8807
seed10_full     1099           1.8837            1.8680     1.8837
```

→ 文档的 0.0293 / 1.8807 / 1.8837 在 **batch 平均**口径下**逐位重现**。
两种平均差最多 0.03，报告必须声明用哪一种。

**这张表最重要的不是 CE，是 `acc` 与 `maj` 两列**：`ActorPolicy` 在 seed10/seed11 **全盘**上
准确率 0.4995 / 0.5369 高于多数类 0.4249 / 0.4404，但切掉共同前缀后掉到
0.4034 / 0.4462，**低于**多数类——全盘上的"胜利"是近重复前缀造成的假象。

---

## R7 片段候选与目视

```powershell
python tools\rl\v431_recheck_segment.py     # exit 0 -> evidence/R7/segment_candidates.json
python tools\rl\v431_render_segment.py      # exit 0 -> evidence/R7/segment_{3,5,10}s_*.png
python tools\rl\v431_segment_cost.py        # exit 0 -> evidence/R7/segment_cost_ranking.json
```

关键输出：

```text
slots=6588 frames=[16..13190] chain pre/post equal at 6587/6587 boundaries
play96x112 (6588, 4, 112, 96) uint8 283.3 MB
windows = 2142, passing criteria = 61

measured replay rate: 104.8 frames/s wall
passing windows: 61

== best combat, any width ==
  slot  4590+ 90 ( 3.00s) bullet%= 81.24 cands= 15 prefix=   87.6s ratio= 51.5x
  slot  4575+120 ( 4.00s) bullet%= 79.56 cands= 15 prefix=   87.3s ratio= 38.0x
  slot  4545+150 ( 5.00s) bullet%= 76.89 cands= 15 prefix=   86.7s ratio= 29.9x

== any passing window starting before slot 3000 ==
  count = 0
```

（上表是 R7 以 `stride=15` 粗搜的结果；R8 改用 `stride=5` 细搜后在同一判据类下把最好的
3 s 窗口挪到 slot 4325，见上节 R8。两者不矛盾，是搜索步长不同。）

---

## R7d / R7e / R7f 三祖先与判据诊断

```powershell
python tools\rl\v431_segment_all_ancestors.py   # exit 0
python tools\rl\v431_segment_diagnose.py        # exit 0
python tools\rl\v431_margin_check.py            # exit 0
```

- `R7d`：同一冻结规则下 seed10/seed11 **0 个合格片段**（`passing=0/2143`），seed12 有 61 个。
- `R7e`：失败判据是 `margin < 8`；seed10/seed11 全盘最大"紧口"槽数 = **0**。
- `R7f`：`margin` 三盘中位数与最大值都 = **16.000**；`< 8` 在 seed12 上 4/6588、其余 0；
  前二分差 `< 8` 在 6588/6588 上都成立 → 该判据**测的是仪器尾部事件**，已弃用。

---

## R8 最终片段选择与目视确认

```powershell
python tools\rl\v431_segment_final.py   # exit 0 -> evidence/R8/segment_spec_final.json
python tools\rl\v431_render_final.py    # exit 0 -> evidence/R8/*.png
```

```text
== frozen selection ==
  train_5s        : seed12 slot 6210..6360 (5.0s, frames 12436..12734) ink% 88.89 cands 13 prefix 118.5s
  train_3s        : seed12 slot 4325..4415 (3.0s, frames 8666..8844)  ink% 94.58 cands 15 prefix 82.5s
  train_10s       : seed12 slot 6060..6360 (10.0s, frames 12136..12734) ink% 83.82 cands 14 prefix 115.6s
  dev_seed10_5s   : seed10 slot 5090..5240 (5.0s, frames 10196..10494) ink% 86.59 cands 17 prefix 97.1s
  dev_seed11_5s   : seed11 slot 4255..4405 (5.0s, frames 8526..8824)  ink% 89.29 cands 16 prefix 81.2s
```

目视：`evidence/R8/train_3s_seed12_3s.png`、`dev_seed11_5s_seed11_5s.png` 均为 Boss + 密集弹幕；
`evidence/R7/control_quiet_5s.png`（ink 百分位 2.65）为带对话框的过场。

---

## 只读旁证（未启动引擎）

```powershell
# 引擎二进制、venv、xvfb、数据根均在（只读列举）
wsl -d Ubuntu -- bash -lc "ls -d /root/Touhou_RL /root/v400-data; ls /root/v400-venv/bin/python*; which xvfb-run Xvfb"
# 会话期间另有 3 个扩容引擎在跑（未干预）
wsl -d Ubuntu -- ps aux | grep th08-modern
```

```text
/root/Touhou_RL   /root/v400-data
/root/v400-venv/bin/python  ...python3  ...python3.10
/usr/bin/xvfb-run  /usr/bin/Xvfb

root 136944 ... th08-modern --data-dir .../T08/expand/run/arm-state692-136942 --rl-control 0,1 --rl-role env
root 137021 ... th08-modern --data-dir .../T08/expand/run/arm-state691-137019 --rl-control 0,1 --rl-role env
root 137144 ... th08-modern --data-dir .../T08/expand/run/arm-state693-137142 --rl-control 0,1 --rl-role env
```

引擎**没有** state snapshot 接口：`th08-rl/src/rlharness/rlharness.cpp` 只解析
`--rl-control` / `--rl-role` / `--data-dir`；恢复只能 `RESET` + 逐 mask 重放（`touhou_rl/v420/reslot.py`）。

---

## 未执行的命令（明确记录，不用替代物冒充）

```text
# 以下均未运行：属于 SEG-001，需先取得用户对 22–23 h 引擎预算的授权
# 且会话期间有 3 个扩容引擎占用 CPU，不得抢占
tools/rl/v410_student_eval.py ...       未运行
tools/rl/v420_eval_shoot.py ...         未运行
python -m tools.rl.v430 train-bc ...    未运行
任何 xvfb-run + th08-modern 的在线学习   未运行
```
