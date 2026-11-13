# WP03.1b — D410-S1 Stage-0 obs 补采（8 seeds → 20 seeds）

- 时间：2026-11-13T03:16:23+0800
- 版本：v4.1.0 / 工作包 WP03.1b
- 结论：**S1 仍未通过（`s1_accepted=false`、`status=rejected_seed_provenance`）**。采集与重标链路全部成立、`validate_s1_coverage` 的数值门槛也全部通过（20 seeds / 65,636 decisions / obs 显式 true / 0 重复 / 0 padding、数值缺口 0/0），但合并集把 **train seeds 1–8** 与 **dev seeds 1000–1011** 混在一起，主代理在 [`S1_REVIEW.md`](S1_REVIEW.md) 中拒绝把它作为 S1 通过证据（会污染 WP05 的 dev 评估，且不是计划的 20-seed 协议集合）。
- 本步骤**通过**的是：D1 obs 采集（12 seeds）+ calibrated 重标 + obs/labels 对齐链路；**没有通过**的是：D410-S1 数据集本身。二者必须分开读。
- 口径声明：合并集 = WP03.1a 已有的 8 个 stage-0 shard（seeds 1–8，train，26,625 decisions）+ 本次新采的 12 个 seeds（1000–1011，dev，39,011 decisions）。**dev seeds 不得进入 S1 训练语料**；补 train seeds 9–20 属于 WP03.1c。
- 未训练、未 commit、未修改 `v400/`/`v410/` 源码或既有证据；D1 obs 契约与 WP03.1a 相同（`obs-v3` digest `ef2fbd15…a105`，60/60 平面 SHA 复核一致）。

## 1. 目标与起点

| 项 | 值 |
| --- | --- |
| 起点（WP03.1a） | 8 shards / 26,625 decisions / 8 seeds（train seeds 1–8），`observations_present=true` |
| D410-S1 门槛（数值） | ≥ 50,000 decisions、≥ 20 seeds、obs 必须显式为 true、synthetic padding 必须为 false |
| D410-S1 门槛（协议） | 20 seeds 必须同为 train，且不与 dev/confirm 重叠（[`S1_REVIEW.md`](S1_REVIEW.md) §2/§4） |
| 起点缺口（数值） | 23,375 decisions / 12 seeds |
| 本次补采 | 12 seeds（1000–1011，dev seeds），stage 0、Normal、team 0、`fast` 教师 |
| 合并后（数值） | 20 shards / 65,636 decisions / 20 seeds |
| 合并后（协议） | **不成立**：train 1–8 + dev 1000–1011 混用 → `s1_accepted=false` |

## 2. 环境与只读输入

- 引擎：build 目录 `/root/Touhou_RL/third_party/th08-rl/build/rl_v3`，实际二进制 `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，SHA256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`
- 引擎 SHA 与 WP03.1a 的 D1 obs 完全相同；collector 在每个 shard 运行前校验该 SHA，不匹配即 fail-closed（见 §2.1）。**未使用 `rl_laser`**。
- 只读输入（未修改，SHA256 记录于 `s1b_plan.json.read_only_inputs`）：
  - `tools/rl/v400_d1_collect.py` — `c9d310f7bb7d3d1745a2efd12c7f0506fe226dff9da747b761bc41c024c908a1`
  - `tools/rl/v400_teacher.py` — `5c98e088fe6b77a875d9cc4fc8e884a75b370cbfdd2671a7a40e1f0d078d23ed`
  - `tools/rl/v410_relabel_d1.py` — `d669f7a76995f1816595fcd2a9294b9cfad094dc21801dd3291459a353f7b4a4`
  - `touhou_rl/v410/dataset_plan.py` — `04c69360414f97f68f0c9d41911f5b02620c49702c3b22b9b7bfb74ffccd2373`
- **没有新增 wrapper**：`tools/rl/v400_d1_collect.py` 直接完成任务，因此没有新增 `tools/rl/v410_s1b_collect.py`，也没有新增 `tests/test_v410_s1b_collect.py`；仓库内没有任何新增 `.py`，所以 `compileall` 不适用于本步骤。
- 回归信号（只跑既有测试，未新增测试）：`python -m pytest -p no:cacheprovider --basetemp .tmp_v410_s1b_pytest -q tests/test_v410_dataset_plan.py tests/test_v410_relabel_d1.py` → **65 passed in 8.40s**，退出码 0（日志 `.tmp_v410_s1b/logs/pytest_wp03.log`）；这两个套件覆盖本步骤只读使用的 `dataset_plan` 与 `v410_relabel_d1`。

### 2.1 Preflight 失败（记录，不是缺陷）

第一次把 `--engine` 指向了二进制本身 `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，collector 会在其后追加 `/th08-modern`，于是 fail-closed：

```text
{"status": "engine_sha_mismatch", "reason": "engine binary not found: /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern/th08-modern; engine_dir: /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern", "expected_sha256": "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6"}
```

collector 把这次失败当作 probe 证据写入（`status=failed`，wall 0.328 s），没有崩栈；`tee` 管道吞掉了进程码。修正为 build 目录后 probe 正常。该次的 `probe/stage0/*` 与 `reachability.json` 已被成功重跑覆盖，所以它只是 preflight 记录，不进入证据链。

## 3. 命令与退出码

所有命令都在 WSL2 Ubuntu 内、`cwd=/mnt/d/WorkShop/Touhou_RL`、`env -u DISPLAY` 下执行。smoke 与主采分开记录；最后一列说明退出码是真实捕获还是由 collector 代码路径推导（早期三条 smoke 命令经 `tee` 管道，进程码被管道吞掉）。

### 3.1 Smoke（1 seed）

| 阶段 | 完整 argv（WSL，cwd=`/mnt/d/WorkShop/Touhou_RL`） | 退出码 | 来源 |
| --- | --- | --- | --- |
| smoke / probe | `env -u DISPLAY /root/v400-venv/bin/python tools/rl/v400_d1_collect.py --phase probe --stages 0 --seeds 1000 --probe-cap-frames 16000 --work-dir .tmp_v410_s1b --out-dir .tmp_v410_s1b/d1 --data-root-prefix /root/v400-data/s1b --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3 --expected-sha 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | **0** | run_cmd.py 捕获的真实退出码 |
| smoke / plan | `env -u DISPLAY /root/v400-venv/bin/python tools/rl/v400_d1_collect.py --phase plan --plan-name smoke --stages 0 --seeds 1000 --decisions-per-shard 3300 --plan-limit-per-stage 1 --work-dir .tmp_v410_s1b --out-dir .tmp_v410_s1b/d1 --data-root-prefix /root/v400-data/s1b --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3 --expected-sha 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | **0** | run_cmd.py 捕获的真实退出码 |
| smoke / run | `env -u DISPLAY /root/v400-venv/bin/python tools/rl/v400_d1_collect.py --phase run --plan-name smoke --workers 1 --work-dir .tmp_v410_s1b --out-dir .tmp_v410_s1b/d1 --data-root-prefix /root/v400-data/s1b --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3 --expected-sha 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | **0** | `tee` 管道吞掉进程码；由 collector 代码路径推导（该阶段 payload `status=ok`），详见 `s1b_plan.json.commands[*].exit_code_source` |

### 3.2 主采（12 seeds）

| 阶段 | 完整 argv（WSL，cwd=`/mnt/d/WorkShop/Touhou_RL`） | 退出码 | 来源 |
| --- | --- | --- | --- |
| main / probe | `env -u DISPLAY /root/v400-venv/bin/python tools/rl/v400_d1_collect.py --phase probe --stages 0 --seeds 1000-1011 --probe-cap-frames 16000 --work-dir .tmp_v410_s1b --out-dir .tmp_v410_s1b/d1 --data-root-prefix /root/v400-data/s1b --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3 --expected-sha 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | **0** | run_cmd.py 捕获的真实退出码 |
| main / plan | `env -u DISPLAY /root/v400-venv/bin/python tools/rl/v400_d1_collect.py --phase plan --plan-name s1b --stages 0 --seeds 1000-1011 --decisions-per-shard 3300 --plan-limit-per-stage 12 --work-dir .tmp_v410_s1b --out-dir .tmp_v410_s1b/d1 --data-root-prefix /root/v400-data/s1b --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3 --expected-sha 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | **0** | run_cmd.py 捕获的真实退出码 |
| main / run | `env -u DISPLAY /root/v400-venv/bin/python tools/rl/v400_d1_collect.py --phase run --plan-name s1b --workers 4 --work-dir .tmp_v410_s1b --out-dir .tmp_v410_s1b/d1 --data-root-prefix /root/v400-data/s1b --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3 --expected-sha 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | **0** | run_cmd.py 捕获的真实退出码 |

### 3.3 重标 / 合并校验

| 步骤 | 完整 argv | 退出码 |
| --- | --- | --- |
| 重标（按验收命令原样的 `--work-root .tmp_v410_s1b`） | `env -u DISPLAY /root/v400-venv/bin/python tools/rl/v410_relabel_d1.py --d1-root .tmp_v410_s1b/d1 --work-root .tmp_v410_s1b --out .tmp_v410_s1b/relabel --manifest Document/Update/Alpha/v4.1.0/evidence/WP03/s1b_relabel.json --s1-audit .tmp_v410_s1b/relabel/s1b_partial_audit.json --quiet` | **2** |
| 重标（`--work-root .tmp_v410_s1b/work`，实际生效） | `env -u DISPLAY /root/v400-venv/bin/python tools/rl/v410_relabel_d1.py --d1-root .tmp_v410_s1b/d1 --work-root .tmp_v410_s1b/work --out .tmp_v410_s1b/relabel --manifest Document/Update/Alpha/v4.1.0/evidence/WP03/s1b_relabel.json --s1-audit .tmp_v410_s1b/relabel/s1b_partial_audit.json` | 0 |
| 合并 + `validate_s1_coverage` | `env -u DISPLAY /root/v400-venv/bin/python .tmp_v410_s1b/tools/build_s1_audit.py --existing-manifest Document/Update/Alpha/v4.1.0/evidence/WP03/relabel_manifest.json --new-manifest Document/Update/Alpha/v4.1.0/evidence/WP03/s1b_relabel.json --out Document/Update/Alpha/v4.1.0/evidence/WP03/s1_full_audit.json` | 2 |
| 合并校验复算 | `env -u DISPLAY /root/v400-venv/bin/python .tmp_v410_s1b/tools/verify_s1_audit.py Document/Update/Alpha/v4.1.0/evidence/WP03/s1_full_audit.json` | 2 |

- 第一条重标命令**故意按验收命令原样执行**并以退出码 2 fail-closed：collector 把 channel-A work labels 写在 `<work-dir>/work/<shard>/labels.jsonl`，而 `v410_relabel_d1.py` 的 discovery 在 `<work-root>/<shard>/labels.jsonl` 找，所以 `--work-root` 必须是 `.tmp_v410_s1b/work`。错误提示：`no shard has both observations and work labels under .tmp_v410_s1b/d1 / .tmp_v410_s1b`。两次尝试都记录在此，没有隐藏第一次的失败。
- 合并校验两步都以退出码 **2** 结束，这是**预期且正确**的 fail-closed：`build_s1_audit.py` 的退出码 2 表示 `s1_accepted=false`（数值门槛通过、seed 协议被拒，见 §7/§8.1）；`verify_s1_audit.py` 用双门（`validator_passed && s1_accepted`）复算，前一门 true、后一门 false → 退出码 2。`embedded_matches_recheck=true`（证据内嵌的 `against_s1` 与独立重算一致）。
- 合并 + 复算脚本共执行 8 次（其中 2 次是脚本自身 bug 的重试，见 §8.10）；最终这次退出码 2，含义是 `s1_accepted=false`（协议被拒），不是脚本错误。

## 4. Smoke（seed 1000）

- probe：`status=ok`，**3,251 decisions / 13,004 frames**，`stop_reason=stage_changed`、`stage_clear=true`、`capped=false`，teacher 76.8 fps；同一 hparams 重跑一次得到**完全相同**的 3,251/13,004（退出码 0，187.2 s），说明该测量可复现。
- plan：`--decisions-per-shard 3300` 被 `reachable // 4` 夹住 → 每 shard **3,251 decisions（13,004 frames）**；`--plan-limit-per-stage 1` → 1 个 shard。
- run：1 shard、1 worker、**exit 0**，wall 287.8 s；存储 **3,250 decisions / 13,000 frames**，`stop_reason=stage_changed`、`trim=1（过面 transition 行没有像素）`。
- obs：`play96x112` [3250, 4, 112, 96]、`hud` [3250, 56, 160]、`act_hist` [3250, 8, 18] —— 三者行数都等于 3,250；SHA256 与 `shard.json` 记录一致，帧非零。
- work labels：`.tmp_v410_s1b/work/d1-s0d1-n3251-seed1000-shard/labels.jsonl` 保留，3,252 行 = 1 report + 3,251 decision（每个 decision 都带 `verified` / `verified_mask` / `chosen_plan.verified` 明细，SHA256 `2915fc5cd6a054e277e055bd5fd4fd0a…`）。
- 对齐：hash checkpoint 6,500/6,500 相等（method=hash-checkpoint）；obs load gate 3,250/3,250，`digest_matches_frozen=true`、`unresolved=0`。

**Smoke 通过 → 才继续 12 seeds。**

## 5. 主采（12 seeds / 1000–1011）

- probe：3,251 decisions / 13,004 frames，`stage_clear=true`（probe 只跑 seeds 的首个 seed，即 1000）；plan：12 shards / 156,048 frames（`coverage_shortfall_reason=plan_limit_per_stage`，这是刻意的 12-shard 切片，不是完整的 2,000,000 帧 D1 计划）。
- run：`--workers 4`，**12/12 shards exit 0、status=ok**，wall 1491.4 s；其中 seed 1000 的 shard 与 smoke 同 key（同 stage/seed/决策数/start_mode），被 resume 跳过并复用（11 个 shard 本次实跑、1 个复用 smoke），所以主采合计是 39,011 decisions，而不是 12 × 3,251。
- 合计：**39,011 decisions / 156,044 frames**，obs 平面 2,032,960,024 字节（1.89 GiB）；work labels 保留 **12/12** 个 shard，work 决策行 39,012 行，比存储多 1 行（1 条 stage-transition，已在存储前 trim）。

| seed | shard | decisions | frames | stop_reason | 过面 | exit | 状态 | wall (s) | 哈希对齐 | obs load | work labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | `d1-s0d1-n3251-seed1000-shard` | 3,250 | 13,000 | stage_changed | 是 | 0 | ok | 287.8 | 6,500/6,500 | 3,250/3,250 | 3,251 决策行（保留） |
| 1001 | `d1-s0d1-n3251-seed1001-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 456.5 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1002 | `d1-s0d1-n3251-seed1002-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 469.5 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1003 | `d1-s0d1-n3251-seed1003-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 457.4 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1004 | `d1-s0d1-n3251-seed1004-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 473.2 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1005 | `d1-s0d1-n3251-seed1005-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 459.8 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1006 | `d1-s0d1-n3251-seed1006-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 473.4 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1007 | `d1-s0d1-n3251-seed1007-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 464.8 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1008 | `d1-s0d1-n3251-seed1008-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 463.1 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1009 | `d1-s0d1-n3251-seed1009-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 409.9 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1010 | `d1-s0d1-n3251-seed1010-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 404.3 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |
| 1011 | `d1-s0d1-n3251-seed1011-shard` | 3,251 | 13,004 | max_frames | 否 | 0 | ok | 410.2 | 6,502/6,502 | 3,251/3,251 | 3,251 决策行（保留） |

### 5.1 obs 平面 SHA256（每 shard 三个平面，全量）

```text
seed  play96x112                                                        hud                                                               act_hist
1000  1612e3d0c2bb6dfe3852eae3542abf7014f42475ce7c13e714da942bb8aec2af a97120bd40db946d7e481070f6f24219f38ab7b2bae31a400b10e1cdead3d5a6 8459a9624c9ce3ce3550d522e7b2aa5fadb9a5764b02087ef4b8d7931175a66a
1001  14d118b9cf8ac7c16efd868027189a914ca6d912f6954ee3d0e7aea91f759421 7999d19c66ff3541ad455edde3e648a18d3353514c08ffffb3b717d370bc7f0d 4635cbcb5bd644ac5f653b4269b392b7b61172e11f468588a539ac6552b60b61
1002  76a5c981c94622c7fa3e39b07ec0c2b66bea38f901a35765d8b391862d47596e 5d0887bdcbc3c3c5087979fe0be5f3f3c967f11b205f3bc9b8628c692f3b1cc6 25a7fc520e5214137120068bb2ecd75f8c7d02e2116820213c1783fc1b0060d7
1003  7b5709032b6bf0d5e7c5a52de1896c2897509b09b3746c94bb000d5a3807f339 2e8073f1ef86a1a3db811b0803b526ebeb61151a2cf21c4ebd2232fd7f20b0cf 989b1f1997347bbf5c217448a6119556e2b8924abe7a308756943446d6397d0c
1004  75eb539a7740cb3e687c648e708aa81c36daeca13c7b97ebe4fd12013865ae16 756e7ece8b5c15daaf65f7e83ed45676bc629620f154a1e456df19cdbfdbcd47 390f9f0d66aa35643eb4d6e9464b65a9b8992f1d7b129a0e499d8d341fdd9817
1005  8d6f2e31d9ec23fdf56c9c49d15432613fef6d6684b0d8c64527abec5921551c bb29e910fae09789ed98eb00303f1af0522ca60102acf0e03a8e8f080b686b50 a025d5277270aea4d7b7bbece65062950696b45531e7a5c43405f5bf7434cc9b
1006  c81d40d1a8694e286cdb30f16a61ba02c63baecf8296b3bab186153283402625 8c81f34b1a2f98d72ef49449da2f5f150460f1a3226e3824117e1b07bec3cc46 b71968cabded5dbde566dadad67060a5fd8165d792ee2d850793c44473d6c8e4
1007  5f752ff0f8bb75f011c84fa7c433035cc1aac5937eeec4b0d7c386eb44635b00 bb11a1d80f9978fda95c66cb4a59e0ca5b2ab71ec1339e6edc682895dec739b0 bbff691e16e615fc9100ff30aac5f3d1d2b40a2eab671855ea3988b300175a8b
1008  ee135ca309ba6fab7faa4fc0ea2e176ba86ff64ac565790df0832530fdd52853 c9eee9c63a01a06ee7b7c54d52feb09a3dedc3921bf167aaf9dc2b29bc146426 de3cfe8dc07e32b157c270f432303e3c9771c4df2ebe8777a454d8587e226a72
1009  e201b5b5dd0a4567b2fd790d6b79f0f8a90ae164c9dd009dd4f6d81977a601d3 e6b480adf913b9ce5515498667642f68b1a6e4376338e6453b695ca7178bdc41 476cf9e459b1ac96d5329340d496d109f8bdf80f83b6a32df2ac9bb41a4c07f2
1010  378611f367614e4e7139e64ea7fb2d255ae0fcfa3a220962289b2785392fe9ce f07493b5dc45c8fe8b4ffc1d59b5ad2c42f2b4e8ea60dcda03f4e10dc2e4adea 50aee3d6e25f3f5433de82f27d8950392305932bbb48bd6b320673e0649e7af9
1011  d43840b378da8ec1944311df2d8ae14d5d06e6dc11ff843b119c79552b762fec 9c7340ab80b1775a7afac9cd23719183f750700c063aa01f99c9e8551c32144f ef4b6e8ac105c607577ea58206b179c98b7e68e09c63c43f27db4fb942523b2d
```

- 60/60 平面 SHA 与 `shard.json` 记录一致，60/60 与重标 manifest 一致，60/60 行数等于 decisions（见 §7）。
- 12/12 shard 的 `obs_spec_version=obs-v3`、`obs_spec_digest=ef2fbd15…a105`，与 WP03.1a 的 8 个 shard 同契约；`play96x112` (rows,4,112,96)、`hud` (rows,56,160)、`act_hist` (rows,8,18)、`uint8`。
- 11/12 个新 shard 的 `stop_reason=max_frames`（教师没有在 13,004 帧内过面，被请求上限截断），只有 seed 1000 是 `stage_changed`（过面 → trim 1 行）。**没有任何 shard 是死亡 episode**：`deaths_observed=0`，所以不存在用死亡样本凑数的问题；反过来说，这批数据也不构成“过面能力”证据。

## 6. 重标（calibrated labels）

- 命令：`v410_relabel_d1.py --d1-root .tmp_v410_s1b/d1 --work-root .tmp_v410_s1b/work --out .tmp_v410_s1b/relabel`，退出码 0，wall 20.4 s；输出 12 个 `labels_v410.jsonl`（**不复制 obs**，只引用）。
- `source`：`shard_dirs_seen=12`、`shards_with_work_labels=12`、`shards_processed=12`、`shards_failed=[]`、`shard_dirs_without_work_labels=0`。

| 指标 | 新采 12 shards |
| --- | --- |
| decisions | 39,011 |
| work decisions | 39,012 |
| trimmed | 1（stage transition，无像素） |
| **verified_alive** | **38,858** |
| **verified_dead** | **0** |
| **unknown** | **153**（`model_mismatch`=153） |
| moving_alive / idle_alive | 21,555 / 17,303 |
| conservative_alive（旧保守口径） | 8,030 |
| verified_plans_total | 39,011 |
| horizon_missing | 39,011（horizon=32 的窗口在尾段缺失，只记 unknown 明细，不伪造） |
| frames_from_obs_decisions | 156,044 |

- 每 shard 的 alive/dead/unknown 明细在 `s1b_relabel.json.shards[*].counts`；**没有 verified_dead**，unknown 只记录、绝不当负样本。
- `v410_relabel_d1.py` 自己产出的 `against_s1`（**只看新 12 shards**）是 `status=shortfall`、缺口 10,989 decisions / 8 seeds —— 这是预期的：单看新切片本就不够 50k/20，S1 的判定必须用合并后的 §7。

## 7. 合并（8 + 12）与 S1 校验

- 合并口径：WP03.1a 的 8 个 stage-0 shard（来自 `relabel_manifest.json`）+ 本次 12 个（来自 `s1b_relabel.json`），逐 shard 从磁盘复核 `shard.json`、`labels.jsonl`、三个 obs 平面。

| 指标 | 合并后 |
| --- | --- |
| shards | 20（8 + 12） |
| **total decisions** | **65,636**（起点 26,625 + 新采 39,011） |
| unique_decisions | 65,636 |
| **unique seeds** | **20** |
| duplicate decision keys | 0（键 = seed+stage+decision） |
| duplicate hash_pre | 0（更强的同一状态去重口径） |
| **observations_present** | **true** |
| synthetic_padding | false |
| alive / dead / unknown | 65,346 / 0 / 290 |
| obs 平面 SHA 复核 | 60/60 与 `shard.json` 一致，60/60 与重标 manifest 一致 |
| frames_from_obs_decisions | 262,544 |
| min_seed_decisions | 3,250 |
| `validate_s1_coverage`（数值） | `passed=true`、`status=passed` |
| **数据集判定** | **`s1_accepted=false`、`status=rejected_seed_provenance`** |

### 7.1 per-seed

| seed | decisions | 来源 |
| --- | --- | --- |
| 1 | 3,370 | train seed（WP03.1a 已有） |
| 2 | 3,322 | train seed（WP03.1a 已有） |
| 3 | 3,346 | train seed（WP03.1a 已有） |
| 4 | 3,274 | train seed（WP03.1a 已有） |
| 5 | 3,371 | train seed（WP03.1a 已有） |
| 6 | 3,274 | train seed（WP03.1a 已有） |
| 7 | 3,346 | train seed（WP03.1a 已有） |
| 8 | 3,322 | train seed（WP03.1a 已有） |
| 1000 | 3,250 | dev seed（WP03.1b 新采） |
| 1001 | 3,251 | dev seed（WP03.1b 新采） |
| 1002 | 3,251 | dev seed（WP03.1b 新采） |
| 1003 | 3,251 | dev seed（WP03.1b 新采） |
| 1004 | 3,251 | dev seed（WP03.1b 新采） |
| 1005 | 3,251 | dev seed（WP03.1b 新采） |
| 1006 | 3,251 | dev seed（WP03.1b 新采） |
| 1007 | 3,251 | dev seed（WP03.1b 新采） |
| 1008 | 3,251 | dev seed（WP03.1b 新采） |
| 1009 | 3,251 | dev seed（WP03.1b 新采） |
| 1010 | 3,251 | dev seed（WP03.1b 新采） |
| 1011 | 3,251 | dev seed（WP03.1b 新采） |

### 7.2 `validate_s1_coverage` 逐项结果

| check | ok | detail |
| --- | --- | --- |
| `total_decisions_ge_min` | ✅ true | 65636 >= 50000 |
| `unique_seeds_ge_min` | ✅ true | 20 >= 20 |
| `every_seed_has_decisions` | ✅ true | all seeds have decisions > 0 |
| `per_seed_total_matches_total_decisions` | ✅ true | 65636 == 65636 |
| `synthetic_padding_is_false` | ✅ true | synthetic_padding is false |
| `duplicate_decision_keys_zero` | ✅ true | 0 duplicate key(s) |
| `unique_decisions_ge_min` | ✅ true | 65636 >= 50000 |
| `observations_present_is_true` | ✅ true | observations_present is true |

```json
{
 "validator_passed": true,
 "validator_status": "passed",
 "labels_cover_s1": true,
 "observations_ok": true,
 "reasons": [],
 "numeric_shortfall": {
  "decisions": 0,
  "seeds": 0
 },
 "s1_accepted": false,
 "status": "rejected_seed_provenance"
}
```

**结论：S1 数据集未通过。** 数值上 65,636 ≥ 50,000 decisions、20 ≥ 20 seeds、`observations_present=true`，`validate_s1_coverage` 的 8 项检查全为 true、数值缺口 0/0；但 seed 协议不成立（train 1–8 与 dev 1000–1011 混用，且不是计划的 1000–1019），主代理在 [`S1_REVIEW.md`](S1_REVIEW.md) 中拒绝 `passed=true`，本审计因此记录 `s1_accepted=false`、`status=rejected_seed_provenance`。该结论只覆盖 D410-S1 的 stage-0 obs 平面 + 校准标签，不代表 D410-MULTI，也不代表模型能力。

## 8. 限制

1. **阻塞级：train/dev seed 混用（S1 未通过的唯一原因）**：`plan_seed_set` 记录 planned seeds = 1000–1019、actual seeds = **1–8（train）+ 1000–1011（dev）**、`matches_plan=false`。`validate_s1_coverage` 只检查数量/obs/重复/padding，**不检查 seed 协议**，所以它给出 `passed=true`；主代理复核（[`S1_REVIEW.md`](S1_REVIEW.md) §2）指出把 dev seeds 放进 S1 训练语料会污染 WP05 的 dev 评估、且 20 seeds 不是同一协议集合，因此拒绝该通过。本审计据此写 `s1_accepted=false`、`status=rejected_seed_provenance`，并把 `remaining_work` 指向 WP03.1c（补 train seeds 9–20）。**这不是可接受的偏差，而是协议身份不成立**；在补齐 train seeds 之前，S1 不能写 passed。
2. **只有 stage 0**：本次只补 stage 0；D410-MULTI（8 面 / 2,000,000 帧 / 每面 ≥150,000）没有进展，`multi_partial` 仍然是 shortfall。
3. **没有死亡样本**：新采 12 shard 的 `deaths_observed=0`、`verified_dead=0`；合并集 `dead=0`。死亡分支的统计意义仍然为零（WP03.1a 的 6 个 dead 全来自 stage-1 的 seed 2，未进入本数据集）。
4. **unknown 290（0.44%）**：`model_mismatch`，只作 unknown 明细，绝不当负样本。
5. **11/12 新 shard 未过面**：`max_frames` 截断，说明教师在这 13,004 帧内没有 clear；这批数据可用，但不能当作“过面能力”或“全程覆盖”的证据。
6. **并发降速**：4 workers 下 teacher_fps 从单跑 76.8 掉到约 39，env_fps 约 105；这是吞吐折衷，不影响数据正确性（hash 对齐仍 100%）。
7. **契约一致但有批次差别**：obs-v3 digest 与 60/60 平面 SHA 复核证明新 obs 与 WP03.1a 同契约；但两批数据的采集时间/机器负载不同，帧分布不做逐帧相等假设。
8. **未跑 `--phase gate` / `--phase report`**：hash 对齐已由每个 shard 的 `alignment.checkpoints_equal`（6,500–6,502 / 6,500–6,502，0 mismatch）与 obs load gate 逐 shard 证明；gate 会把 20 个 shard 的标签/obs 再全量加载一遍，本步骤未重复执行。
9. **未训练**：没有拟合、没有 checkpoint，本步骤只产出数据与证据。
10. **过程修正（透明记录）**：运行目录内的合并脚本（不入库）有两次自身 bug —— ① 用 `entry["shard"]`（shard 名字）而不是 `inputs.shard_dir`（路径）拼 shard 目录，退出码 1 FileNotFoundError；② 把 `shard.json.obs` 的 `bytes_per_decision`/`bytes_total` 当成平面，导致 `observations_present=false`、退出码 2。两次都是脚本缺陷、不是数据缺陷；修正后重新执行得到 §7 的 `validator_passed=true`，且 `verify_s1_audit.py` 独立复算一致。数据本身从未被改写。

## 9. 下一建议

1. **WP03.1c：补 train seeds 9–20（解除 S1 阻塞的唯一路径）**。已有的 8 个 train seeds 1–8 保持不变，新采 12 个 train seeds（9–20，stage 0，约 12 × 3,251 ≈ 39,000 decisions / 156,000 帧），合并成 **全 train 的 20-seed** 集；dev seeds 1000–1011 只保留为 dev/诊断用途，不进入 S1 训练语料。复核器需要新增显式 seed-policy 检查（20 seeds 全为 train 且与 dev/confirm 不相交），`validate_s1_coverage` 本身不覆盖这一点。
2. **补 D410-MULTI**：stage 1–7 各自 ≥150,000 帧、总计 2,000,000 帧；当前 MULTI 帧数约 374,544（WP03.1a 的 218,500 + 新采 156,044）。
3. **死亡/过面样本**：需要能稳定复现的死亡切片，才能让 `verified_dead` 有统计意义；目前的探针都停在 `max_frames` 或 clear。
4. **训练前先跑一次全量 loader**：`obs_v3_dataset.load_teacher_samples_v3(require_observations=True)` 覆盖 20 个 shard / 65,636 行，确认训练入口读的是同一批（`shard.json` 已逐 shard 证明，但入口级复核更稳）。
5. 可选：对合并集跑 `--phase gate`。`--plan-name s1b` 只覆盖 12 个新 shard；20 个 shard 的 gate 需要把 WP03.1a 的 8 个 shard 也写进同一个 plan，属于新编排工作，建议与其他 S1 收尾一起做。

## 10. 证据清单

| 文件 | 字节 | SHA256（前 32 位） |
| --- | --- | --- |
| [`s1b_plan.json`](s1b_plan.json) | 15,864 | `dd9dc40344f736e105d6cefd8410a722…` |
| [`s1b_run.json`](s1b_run.json) | 57,048 | `d83609cb81d86f0126756d888c5d5804…` |
| [`s1b_relabel.json`](s1b_relabel.json) | 72,441 | `2012bc19bac931b6f79ff987aa83d42a…` |
| [`s1_full_audit.json`](s1_full_audit.json) | 59,311 | `403bb3312a7c88c9ce61beab5fb15f8e…` |
| [`S1B.md`](S1B.md) | 本文 | — |

- 运行目录（`.tmp_*/` 已在 `.gitignore` 中，**不入库**）：
  - `.tmp_v410_s1b/d1/shards/<12 shards>/`：obs（`play96x112.npy`/`hud.npy`/`act_hist.npy`）+ `labels.jsonl` + `shard.json`
  - `.tmp_v410_s1b/work/<shard>/labels.jsonl`：channel-A work labels（含 verified 明细）
  - `.tmp_v410_s1b/relabel/<shard>/labels_v410.jsonl`：calibrated labels（不复制 obs）
  - `.tmp_v410_s1b/plan.smoke.json`、`plan.s1b.json`、`run.smoke.json`、`run.s1b.json`、`reachability.json`、`reports/<shard>.json`、`logs/commands.jsonl`（argv + 退出码账本）
  - `/root/v400-data/s1b/<shard>/`：每个 shard 的引擎数据副本（WSL ext4，不在仓库内）
- 旧证据只读：`Document/Update/Alpha/v4.1.0/evidence/WP03/` 下的 `plan.json`、`PLAN.md`、`RELABEL.md`、`relabel_manifest.json`、`s1_label_audit.json`、`s1_partial_audit.json` 由本步骤只读引用，**没有修改**。`STATUS.md` 与 `S1_REVIEW.md` 由主代理在复核阶段更新（S1_REVIEW 记录了对本审计 `passed=true` 的拒绝），这两份文件不由本步骤写入；本文件与 `s1_full_audit.json` 中的 `s1_accepted=false` 与之一致。
