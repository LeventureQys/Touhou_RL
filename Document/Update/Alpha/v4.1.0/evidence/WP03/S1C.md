# D410-S1 干净训练种子补采与合并审计（WP03.1c）

- 时间：2026-11-12T19:52Z（WSL UTC）/ 2026-11-13 03:52 +08:00
- 目标：用 12 个 **train seeds 9–20** 补采 stage-0 obs，与已有 8 个 **train seeds 1–8**
  合并成 **干净训练种子 S1**；**不使用 dev seeds 1000–1011**。
- 结论：**S1 passed=true**（`s1c_full_audit.json`，20 个 train seeds / 66,143 obs decisions /
  seed policy 全部通过 / 无 dev-confirm 污染 / 无合成填充）。WP03.1b 的 train/dev 混用缺陷已被
  新增的 seed-policy 闸门判死（负控 `failed_seed_policy`，exit 2）。

## 0. 复核对象与拒绝理由

| 项 | WP03.1b（被拒） | WP03.1c（本步） |
|---|---|---|
| seeds | 1–8（train）+ 1000–1011（dev） | **1–20，全部 train** |
| dev 污染 | 有（12 个 dev seed 进入训练集） | **无**（dev 1000–1099 未采、未选） |
| passed 依据 | 只判 50k/20 seeds 阈值 | 阈值 **且** seed policy **且** 逐 shard 完整性 |
| 判定 | `passed=true`（主代理拒绝） | `passed=true`（三条闸门同时通过） |

`S1_REVIEW.md` 的四条拒绝理由在 `s1c_full_audit.json → review_response.items` 里逐条对应，
全部 `ok=true`；负控（见 §7）证明旧混集会被判 `failed_seed_policy`。

## 1. 命令与退出码（实际运行，WSL）

引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，
sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`
（WSL `sha256sum` 复核一致；采集器每个 shard 前再校验一次，失配即 fail-closed）。
采集器 `tools/rl/v400_d1_collect.py` 只读、未改；所有日志在 `.tmp_v410_s1c/logs/`，
`run_phase.sh` 把完整 argv + UTC 窗口 + `EXIT_CODE` 写进日志。

### 1.1 Smoke（seed 9，1 worker）

| phase | argv（除共同尾部） | 退出码 | 日志 |
|---|---|---|---|
| probe | `--phase probe --stages 0 --seeds 9 --probe-cap-frames 16000` | **0** | `logs/smoke_probe.log` |
| plan | `--phase plan --plan-name smoke --stages 0 --seeds 9 --decisions-per-shard 3300 --plan-limit-per-stage 1` | **0** | `logs/smoke_plan.log` |
| run | `--phase run --plan-name smoke --workers 1` | **0** | `logs/smoke_run.log` |

共同尾部（三条完全一致）：
`--work-dir .tmp_v410_s1c --out-dir .tmp_v410_s1c/d1 --data-root-prefix /root/v400-data/s1c --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3 --expected-sha 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`
（前面是 `env -u DISPLAY /root/v400-venv/bin/python tools/rl/v400_d1_collect.py`；
完整 argv 原样记录在 `s1c_plan.json → commands[].argv`。）

诚实说明：smoke probe 第一次调用时外层 shell 未捕获退出码（输出本身是 `status=ok`，
3,467 decisions / 13,868 frames）；随后用**完全相同**的 argv 经 `run_phase.sh` 重跑并捕获
`EXIT_CODE 0`，两次结果逐位相同。下表与 §2 用的是带退出码的那次。

### 1.2 主采（train seeds 9–20，4 workers）

| phase | argv（除共同尾部） | 退出码 | 日志 |
|---|---|---|---|
| probe | `--phase probe --stages 0 --seeds 9-20 --probe-cap-frames 16000` | **0** | `logs/main_probe.log` |
| plan | `--phase plan --plan-name s1c --stages 0 --seeds 9-20 --decisions-per-shard 3300 --plan-limit-per-stage 12` | **0** | `logs/main_plan.log` |
| run | `--phase run --plan-name s1c --workers 4` | **0** | `logs/main_run.log` |

共同尾部同 §1.1。主采 `run` 19:25:04Z → 19:49:51Z（24m47s），12 个 shard 全部
`exit_code=0`、`failed=[]`；seed 9 由 smoke 产出，主采 `pending=11` 并正确 resume 跳过它。

### 1.3 重标、合并审计、测试

| 命令 | 退出码 |
|---|---|
| `python tools/rl/v410_relabel_d1.py --d1-root .tmp_v410_s1c/d1 --work-root .tmp_v410_s1c --out .tmp_v410_s1c/relabel --manifest .../s1c_relabel.json --s1-audit .tmp_v410_s1c/relabel/s1c_partial_audit.json` | **2**（见下） |
| `python tools/rl/v410_relabel_d1.py --d1-root .tmp_v410_s1c/d1 --work-root .tmp_v410_s1c/work --out .tmp_v410_s1c/relabel --manifest Document/Update/Alpha/v4.1.0/evidence/WP03/s1c_relabel.json --s1-audit .tmp_v410_s1c/relabel/s1c_partial_audit.json --quiet` | **0** |
| `python tools/rl/v410_s1c_audit.py --existing-manifest .../relabel_manifest.json --new-manifest .../s1c_relabel.json --out .../s1c_full_audit.json --quiet` | **0** |
| `python -m pytest -p no:cacheprovider --basetemp .tmp_v410_s1c/pytest_run -q tests/test_v410_s1c_audit.py` | **0**（20 passed） |

`--work-root` 说明（实测，不是猜测）：任务模板里的 `--work-root .tmp_v410_s1c` 实测 **exit 2**，
报 `no shard has both observations and work labels under .tmp_v410_s1c/d1 / .tmp_v410_s1c`
——采集器把 work labels 写在 `<work-dir>/work/<key>/labels.jsonl`，所以正确的 work-root 是
`.tmp_v410_s1c/work`。两条命令都在 `logs/relabel_literal.log` / `logs/relabel.log` 有据。
`s1-audit` 落在运行目录（不入库），它是 **12 个新 shard 的 partial 切片**（39,518 decisions /
12 seeds），其 `shortfall` 是预期且**不是** S1 判定；S1 判定只看合并后的 `s1c_full_audit.json`。

## 2. Smoke 数字（seed 9）

- probe：stage 0，`decisions 3467 / frames 13,868`，`stop_reason=stage_changed`，
  `stage_clear=true`，`deaths_observed=0`，`capped=false`，197.6s，teacher 70.2 fps。
  第二次 probe（wrapper 重跑）给出完全相同的 3467/13,868，主采 probe 亦然 —— 复现性有据。
- plan：`status=ok`，1 个 shard，`decisions=3300`，`frames_planned=13,200`，
  coverage shortfall 原因 `plan_limit_per_stage`（故意的 1-shard 切片）。
- run：shard `d1-s0d1-n3300-seed9-shard`，exit 0，295.7s。

smoke shard 保留情况（独立校验器 `.tmp_v410_s1c/tools/verify_shard.py`，exit 0）：

| 项 | 值 |
|---|---|
| decisions / frames | 3,300 / 13,200（`stop_reason=max_frames`，**未死亡**，shard 上限先于过面） |
| obs planes | `play96x112 (3300,4,112,96)`、`hud (3300,56,160)`、`act_hist (3300,8,18)`，uint8，rows==decisions |
| obs sha256 | 三个平面均与 `shard.json` 记录一致（play `77580b50…2972`，hud `1aff2a1f…5b0c`，act_hist `c1259480…e00b`） |
| obs spec | `obs-v3`，digest `ef2fbd15…a105`（冻结值） |
| shard labels | 3,300 行，sha256 `7d58b03d…ab4d` 与记录一致，3,300 行带 `verified_mask` |
| work labels | **保留**：`.tmp_v410_s1c/work/<shard>/labels.jsonl`，3,301 行（1 report + 3,300 decision），3,300 条 decision 全带 `verified` 明细（died=false 3,300，plan entries 3,300），sha256 `6fe4dafc…6cb2` |
| hash 对齐 | 6,600/6,600 checkpoints 相等，`transition_trim=null` |

## 3. 主采数字（train seeds 9–20）

12 个 shard、12 个 train seed，全部 `status=ok` / `exit_code=0`，**0 死亡**：

| shard | seed | decisions | frames | stop_reason | stage_clear | deaths | exit | checkpoints |
|---|---|---|---|---|---|---|---|---|
| d1-s0d1-n3300-seed9-shard | 9 | 3300 | 13,200 | max_frames | false | 0 | 0 | 6600/6600 |
| d1-s0d1-n3300-seed10-shard | 10 | 3300 | 13,200 | max_frames | false | 0 | 0 | 6600/6600 |
| d1-s0d1-n3300-seed11-shard | 11 | 3300 | 13,200 | max_frames | false | 0 | 0 | 6600/6600 |
| d1-s0d1-n3300-seed12-shard | 12 | 3298 | 13,192 | stage_changed | true | 0 | 0 | 6596/6596 |
| d1-s0d1-n3300-seed13-shard | 13 | 3274 | 13,096 | stage_changed | true | 0 | 0 | 6548/6548 |
| d1-s0d1-n3300-seed14-shard | 14 | 3300 | 13,200 | max_frames | false | 0 | 0 | 6600/6600 |
| d1-s0d1-n3300-seed15-shard | 15 | 3300 | 13,200 | max_frames | false | 0 | 0 | 6600/6600 |
| d1-s0d1-n3300-seed16-shard | 16 | 3300 | 13,200 | max_frames | false | 0 | 0 | 6600/6600 |
| d1-s0d1-n3300-seed17-shard | 17 | 3274 | 13,096 | stage_changed | true | 0 | 0 | 6548/6548 |
| d1-s0d1-n3300-seed18-shard | 18 | 3298 | 13,192 | stage_changed | true | 0 | 0 | 6596/6596 |
| d1-s0d1-n3300-seed19-shard | 19 | 3300 | 13,200 | max_frames | false | 0 | 0 | 6600/6600 |
| d1-s0d1-n3300-seed20-shard | 20 | 3274 | 13,096 | stage_changed | true | 0 | 0 | 6548/6548 |

- 合计：**12 shards / 39,518 decisions / 158,072 frames / 0 deaths**；
  `stage_clear=5`（seed 12、13、17、18、20 在 3,274–3,298 decisions 就过面）、
  `max_frames=7`（其余 7 个 seed 的 shard 在 13,200 帧上限处停止，**未死**，只是切片先到上限）。
- 每 shard 墙钟 403.3–481.3s（4 workers 并发）；三个 obs 平面 sha256 逐 shard 记录在
  `s1c_run.json → obs_planes_sha256`，且每个都与 `shard.json` 记录一致
  （`obs_sha256_all_match_recorded=true`）。
- work labels 全部保留：39,523 decision 行 + 12 report 行，**39,523/39,523 行带 `verified`**；
  5 个过面 shard 各多 1 条 trailing transition decision（只能 trim、不能配对）。
- **没有拿死亡凑数**：本批 0 死亡、0 补齐、0 重复计数；7 个 `max_frames` shard 是"先到帧上限"，
  不是"过不了面"。seed 9 的 shard 由 smoke 产出，主采以同一 key resume 复用（同一个 shard，
  不是第二个 shard、也不重复计入 39,518）。

## 4. 重标数字（`tools/rl/v410_relabel_d1.py`，exit 0）

- 规则：`touhou_rl.v410.teacher_calibration.convert_teacher_row_calibrated`，
  tolerance **8.0**（data-derived，来源 `.tmp_v410_wp02_audit/teacher1000.jsonl`），
  horizon 32、decision_frames 4；label schema `v410-teacher-label-calibrated/1`。
- 12 shards / **39,518** calibrated decisions；`work_decisions=39,523`，`trimmed_decisions=5`。
- survival（per-plan 计数）：**alive 39,330 / dead 0 / unknown 188**
  （188 全部是 `model_mismatch`，`incomplete=0`）；`moving_alive=21,845`、`idle_alive=17,485`。
- `horizon_missing=39,518`（32 帧 horizon 在 4 帧决策的 tape 上不可达，全量记录，不当负样本）；
  `conservative_unknown=31,229`。
- 完整性：每条 label 带 `obs` 指针（shard/row/decision_frame/planes），obs 与 work labels 的
  sha256 全部与记录一致（任一失配即 fail-closed）。

## 5. 合并与 seed policy（`s1c_full_audit.json`，pass）

| 指标 | 值 |
|---|---|
| shards | **20**（8 既有 + 12 新） |
| seeds | **1–20**，20 个唯一值 |
| total / unique decisions | **66,143 / 66,143** |
| obs rows | 66,143（三平面 rows==decisions） |
| observation frames | 264,572（4 帧/决策） |
| duplicate decision keys | **0** |
| observations_present | **true** |
| synthetic_padding | **false** |
| alive / dead / unknown | **65,818 / 0 / 325** |
| trimmed decisions | 12（既有侧 7 + 新侧 5，逐 shard 记录） |
| per_seed decisions | 3,274–3,371（每个 seed 都 > 0） |
| 合并后 S1 | **passed=true，status=passed**，`against_s1.status=passed`，shortfall 0/0 |

seed policy 显式校验（全部 true，`s1c_full_audit.json → seed_policy.checks`）：

1. `seeds_count_is_plan`：20 个 seed；
2. `seeds_unique`：20 个互不相同；
3. `seeds_all_train`：每个 seed ∈ `[1, 49999]` 且不在 dev/confirm 块内；
4. `seeds_no_dev`：**dev 1000–1099 命中 0 个**（`dev_overlap=[]`）；
5. `seeds_no_confirm`：confirm 60000–60199 命中 0 个；
6. `seeds_match_s1c_plan`：集合恰为 `1..20`（missing=[] extra=[]）；
7. `existing_seeds_are_train_1_8`：既有侧恰为 1–8；
8. `new_seeds_are_train_9_20`：新侧恰为 9–20；
9. `selected_shards_are_train`：被选中的 shard 无一是非 train；
10. `input_non_train_entries=[]`：两个输入 manifest 的全部条目里**没有任何** dev/confirm seed
    （连未被选中的也没有）。

逐 shard 完整性（`integrity.checks` 全 true，`problems=[]`）：obs 三平面 sha256 同时匹配
`shard.json` 与重标 manifest；work labels 仍在且每条 observed decision 都有 verified 明细，
work `hash_pre` 序列与 shard labels 序列一致；`labels_v410.jsonl` 存在、sha256 与 manifest
一致、行数==decisions；obs-v3 契约（版本/冻结 digest/平面/形状/uint8）逐 shard 相同；
duplicate keys 0；无合成行。

判定逻辑（`tools/rl/v410_s1c_audit.py`）：`passed` 要求
**coverage 判定 ∧ seed policy ∧ 全部完整性检查**；任一失败写 `shortfall` /
`failed_seed_policy` / `failed_integrity` / `failed_label_coverage` 并附缺口，绝不由决策数单独决定。

独立复核（**不 import 审计工具**，重写 seed policy 并从 `s1c_full_audit.json` + 磁盘
`shard.json` 重新推导）：`.tmp_v410_s1c/tools/verify_s1c.py`，**exit 0**，19/19 检查通过
（20 shards / seeds 1–20 / 66,143 decisions / dev=[] confirm=[] / observations_present /
duplicate 0 / synthetic false / per_seed min 3,274 / 磁盘 seed-stage-digest-sha256 交叉核对全对 /
`independent_passed == audit_passed == true`）。日志 `.tmp_v410_s1c/logs/verify_s1c.log`。

## 6. 与 S1_REVIEW 的对应

| S1_REVIEW 拒绝点 | 本步处置 |
|---|---|
| train/dev 混用（1–8 + 1000–1011） | 新侧改为 train 9–20；合并集 1–20 全 train |
| `plan_seed_set.matches_plan=false` 未导致 fail | 新增 `seeds_match_s1c_plan` / 两侧 seed 身份检查，偏离即 fail |
| dev seeds 进入训练会污染 WP05 评估 | dev 1000–1099 未采集、未选中、未出现在任何输入条目 |
| 阈值通过但协议身份不成立 | `passed` 现在需要三闸门同时通过；阈值单独通过不再足够 |

## 7. 负控（证明闸门真的会拒绝）

用同一工具审 WP03.1b 的实际混集（既有 1–8 + `s1b_relabel.json` 的 1000–1011）：

- 结果：20 shards / 65,636 decisions / 20 seeds，**覆盖率判定仍 passed**，
  但 `seed_policy.passed=false`、`dev_overlap=[1000..1011]`、
  `status=failed_seed_policy`、`passed=false`、**exit 2**。
- 即：同样的数据在旧审计里是 `passed=true`，在新闸门下被判死 —— 这正是 S1_REVIEW 要求的。
  证据：`.tmp_v410_s1c/negative_control_s1b.json` + `logs/negative_control_s1b.log`（不入库）。

## 8. 限制

1. **只覆盖 stage 0**（D410-S1 第一面）；7 个既有 stage-1 shard 明确排除并在
   `inputs.existing.excluded_non_stage0` 列出，不参与 S1。
2. **shard 上限 3,300 decisions（13,200 帧）**：seed 9 的 probe 测得该 seed 在 3,467 decisions
   /13,868 帧过面，因此 7 个 `max_frames` shard 是"前 13,200 帧"的切片（缺尾部约 668 帧），
   不是死亡、也没有补齐。5 个 shard 在 3,274–3,298 decisions 自然过面。
3. **未训练**：本步只产出 obs+labels+审计，没有任何拟合/训练/评估。
4. **不是 D410-MULTI**：仅 1 个 stage，`multi_partial` 必为 shortfall。
5. **dev seeds 保留**：1000–1011 仍是 WP05 开发评估用；WP03.1b 的 12 个 dev shard 未删除、
   未改动，只是不再进入 S1。
6. `verified_alive` 是 WP02.4 校准后的模型一致性判定，不是生存保证；`unknown` 永不当负样本。
7. 重标标签与 obs 指针写在 `.tmp_v410_s1c/relabel/`（运行目录，不入库）；仓库内只提交
   manifest/audit 证据。
8. 证据里的 `.tmp_*` 路径为运行期路径；本步**未 commit/push**，是否入库由主代理决定。

## 9. 下一步建议

1. 主代理复核并决定提交 `tools/rl/v410_s1c_audit.py`、`tests/test_v410_s1c_audit.py` 与
   `evidence/WP03/{S1C.md,s1c_plan.json,s1c_run.json,s1c_relabel.json,s1c_full_audit.json}`。
2. 若要求"每个 seed 覆盖到过面"，把 `--decisions-per-shard` 提到 3,467（或按 seed 逐个 probe）
   重采 7 个 `max_frames` shard；这会改变 shard key（`n3467`），需重跑重标与审计。
3. S1 通过后可进入 D410-MULTI（8 stages / 2,000,000 帧 / 每 stage 150,000 帧）的补采规划，
   注意 dev seeds 仍须隔离。
4. 训练前建议再加一层"train/val/test 按原始轨迹分组"的切分检查，避免同一 seed 跨 split。

## 附录 A：20 个 S1 shard 的 obs 平面 SHA

`play96x112` 全值如下；`hud` / `act_hist` 的 sha256 逐 shard 记录在
`s1c_full_audit.json → shards[].obs`。三平面 `sha256_matches_shard_json` 与
`sha256_matches_manifest` 全部为 true（`all three planes verified: True`）。

| seed | shard | decisions | frames | play96x112 sha256 | hud rows | act_hist rows |
|---|---|---|---|---|---|---|
| 1 | `d1-s0d1-n3371-seed1-shard` | 3370 | 13480 | `ff9cda704d2aa1a5beacdbd3071875ae5edeb52c57a2f16371e1cab01b5070e0` | 3370 | 3370 |
| 2 | `d1-s0d1-n3371-seed2-shard` | 3322 | 13288 | `0afddc4174b476bae88d4159b1ac385edaeea31038641d441dd576a5ee752713` | 3322 | 3322 |
| 3 | `d1-s0d1-n3371-seed3-shard` | 3346 | 13384 | `551316dbed6fea2e0ed4e971bdb6969d8f9e27424da042d337bab4d9ff2ece37` | 3346 | 3346 |
| 4 | `d1-s0d1-n3371-seed4-shard` | 3274 | 13096 | `c9e071fc6d6474ceedeb4efb8b767e4c56ba9333bf3ec46b2ec17de640e9077b` | 3274 | 3274 |
| 5 | `d1-s0d1-n3371-seed5-shard` | 3371 | 13484 | `4314330c49b8eab35e1fe44331415e6de722e885b25030e496eaa68490d4293e` | 3371 | 3371 |
| 6 | `d1-s0d1-n3371-seed6-shard` | 3274 | 13096 | `b295f3a9422a43c08a3832054899aa842ebd8df5554fbc6f8b7826ebbd760edb` | 3274 | 3274 |
| 7 | `d1-s0d1-n3371-seed7-shard` | 3346 | 13384 | `a5337cd28473bc073ef2948a49b9f25a08e770a1bbb20010a159e28545a2ee7c` | 3346 | 3346 |
| 8 | `d1-s0d1-n3371-seed8-shard` | 3322 | 13288 | `ba9a8732d44ccdf1eaa3763a74e33449a789540d703704aef4a234e20442602c` | 3322 | 3322 |
| 9 | `d1-s0d1-n3300-seed9-shard` | 3300 | 13200 | `77580b50684327be4417757b516779c375500e0fbcda6a693a197ce230342972` | 3300 | 3300 |
| 10 | `d1-s0d1-n3300-seed10-shard` | 3300 | 13200 | `7625377a802867b5443aeb11d06628256b87a506ec6917e30fbf839ba093a3e9` | 3300 | 3300 |
| 11 | `d1-s0d1-n3300-seed11-shard` | 3300 | 13200 | `ef5d5e7cc8530886adc18915d9c3b8430fe875b3d44aa985d371bdf4fb0557a8` | 3300 | 3300 |
| 12 | `d1-s0d1-n3300-seed12-shard` | 3298 | 13192 | `9946291c300273297e8f496d6610e393a0aef88dd34456b3410baeff41912f2c` | 3298 | 3298 |
| 13 | `d1-s0d1-n3300-seed13-shard` | 3274 | 13096 | `aac84a385aab64f62caf78f7a6b1bcc0182ba34225759367a9aef101745cc84f` | 3274 | 3274 |
| 14 | `d1-s0d1-n3300-seed14-shard` | 3300 | 13200 | `f76c12e4c0f9000e89210caf357c1c3d2293133258a89d1d10f31349163692a6` | 3300 | 3300 |
| 15 | `d1-s0d1-n3300-seed15-shard` | 3300 | 13200 | `6ee1e713e185dc32c65dc017ac6c1d14e4dd78013763774715213661d6b1e239` | 3300 | 3300 |
| 16 | `d1-s0d1-n3300-seed16-shard` | 3300 | 13200 | `48dfcc0d405df11fd1c3335b399ff63cfbd67a2f64b53f85aee08a6a3ffcedfb` | 3300 | 3300 |
| 17 | `d1-s0d1-n3300-seed17-shard` | 3274 | 13096 | `5f3adc91ea03b81c8003f7ef365b56fa338a1c3a41d0fa2b19bb35d21383945c` | 3274 | 3274 |
| 18 | `d1-s0d1-n3300-seed18-shard` | 3298 | 13192 | `baac53fa92c64408e3a6247ef8afc8d45c7bcaf252e99b86919068a2e4ba7bd3` | 3298 | 3298 |
| 19 | `d1-s0d1-n3300-seed19-shard` | 3300 | 13200 | `6803941291c4824d7ba1ea57e754b432031b33b7912ce65b9d338350f1d8524e` | 3300 | 3300 |
| 20 | `d1-s0d1-n3300-seed20-shard` | 3274 | 13096 | `3d494ca960b4bcd3f81c95fcf23b741f240c882c4abb165ce61b3d56228d4569` | 3274 | 3274 |
