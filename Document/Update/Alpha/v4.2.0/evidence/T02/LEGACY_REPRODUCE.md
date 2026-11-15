# T02a — 历史学生复现腿（v4.2.0 / WP420-00）

- **结论：复现成立（`REPRODUCED`）**
- 生成时间（UTC）：`2026-11-13T09:36:22Z`
- 任务：v4.2.0 / WP420-00 — T02a legacy student reproduction leg
- 范围：re-runs the existing tools/rl/v410_student_eval.py only; no new evaluator code, no protocol/ABI change
- 依据：仅重跑既有评测器 `tools/rl/v410_student_eval.py`；**未新增/修改任何代码、测试、配置**。

> **墙钟口径说明**：以下所有 `wall_s` 均为「争用条件下观测值」——采集期间另有一个 v4.1.0 `v410_dagger_collect.py` 进程（PID 1884178）在并发运行，因此墙钟被系统性拉长，**不构成性能基线**。

---

## 1. 四次运行：退出码与墙钟

| arm | 重复 | 退出码 | 开始 (UTC) | 结束 (UTC) | 墙钟 s（争用条件下观测值） | 输出 JSON |
|---|---|---|---|---|---|---|
| `b0` | rep1 | **0** | `2026-11-13T09:24:41Z` | `2026-11-13T09:27:04Z` | 142.974 | `b0_dev3_rep1.json` |
| `b0` | rep2 | **0** | `2026-11-13T09:27:04Z` | `2026-11-13T09:29:22Z` | 137.819 | `b0_dev3_rep2.json` |
| `beta20k` | rep1 | **0** | `2026-11-13T09:29:22Z` | `2026-11-13T09:32:41Z` | 198.367 | `beta20k_dev3_rep1.json` |
| `beta20k` | rep2 | **0** | `2026-11-13T09:32:41Z` | `2026-11-13T09:36:00Z` | 198.973 | `beta20k_dev3_rep2.json` |

全部退出码 0：yes

### 1.1 完整 argv（以 rep1 为例，两次重复仅 `--out` 不同）

`b0`（`.tmp_v420_data/T02/b0_dev3_rep1.json`）：

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp04_arms/b0_seed0_last.pt \
  --seeds 1000,1001,1002 \
  --stage 0 \
  --difficulty normal \
  --max-frames 19887 \
  --stop-on-death \
  --device cpu \
  --out /mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T02/b0_dev3_rep1.json
```

同一 argv 的逐 token 原始记录（`b0_dev3_rep1.argv.txt`）：

```
--engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern --data-dir /root/v400-data/game --checkpoint .tmp_v410_wp04_arms/b0_seed0_last.pt --seeds 1000,1001,1002 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death --device cpu --out /mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T02/b0_dev3_rep1.json
```

`beta20k`（`.tmp_v420_data/T02/beta20k_dev3_rep1.json`）：

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt \
  --seeds 1000,1001,1002 \
  --stage 0 \
  --difficulty normal \
  --max-frames 19887 \
  --stop-on-death \
  --device cpu \
  --out /mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T02/beta20k_dev3_rep1.json
```

同一 argv 的逐 token 原始记录（`beta20k_dev3_rep1.argv.txt`）：

```
--engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern --data-dir /root/v400-data/game --checkpoint .tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt --seeds 1000,1001,1002 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death --device cpu --out /mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T02/beta20k_dev3_rep1.json
```

### 1.2 环境与资产溯源

| 项 | 值 |
|---|---|
| checkpoint `b0` | `.tmp_v410_wp04_arms/b0_seed0_last.pt` |
| checkpoint sha256 | `3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6` |
| sha256 与期望一致 | yes |
| checkpoint `beta20k` | `.tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt` |
| checkpoint sha256 | `15ae1fd5b9a6e0cbbddfd668bb342444bdb44db123b5f7e56abf3c5e7215bd95` |
| sha256 与期望一致 | yes |
| engine sha256 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| engine 全为 691bfd77… | yes |
| python | `/root/v400-venv/bin/python` (3.10.12) |
| 运行位置 | WSL2，`cwd=/mnt/d/WorkShop/Touhou_RL` |
| seeds / 预算 | `1000,1001,1002`，`--stage 0 --difficulty normal --max-frames 19887 --stop-on-death --device cpu` |

---

## 2. 三向逐项比对（历史基准 vs rep1 vs rep2）

三向 = `baseline=rep1`、`baseline=rep2`、`rep1=rep2`。数值列均给出实测原值。

### 2.1 `b0` — `.tmp_v410_wp04_arms/b0_seed0_last.pt`

#### seed 1000

| 字段 | 历史基准 | rep1 实测 | rep2 实测 | baseline=rep1 | baseline=rep2 | rep1=rep2 |
|---|---|---|---|---|---|---|
| `frames` ★ | `1398` | `1398` | `1398` | yes | yes | yes |
| `steps` | `699` | `699` | `699` | yes | yes | yes |
| `score` ★ | `13102` | `13102` | `13102` | yes | yes | yes |
| `stop_reason` ★ | `death_stop` | `death_stop` | `death_stop` | yes | yes | yes |
| `deaths` ★ | `1.0` | `1.0` | `1.0` | yes | yes | yes |
| `action_sequence_sha256` ★ | `85a528fd29c83f513d1464f8a9bf3ef20de52c352ed8aa64bc366d8b59cedac6` | `85a528fd29c83f513d1464f8a9bf3ef20de52c352ed8aa64bc366d8b59cedac6` | `85a528fd29c83f513d1464f8a9bf3ef20de52c352ed8aa64bc366d8b59cedac6` | yes | yes | yes |
| `policy_input_sha256` | `41d171cccc2d9486dc5ac5de85216f2f59c7be389e38410746112a4dbb442b15` | `41d171cccc2d9486dc5ac5de85216f2f59c7be389e38410746112a4dbb442b15` | `41d171cccc2d9486dc5ac5de85216f2f59c7be389e38410746112a4dbb442b15` | yes | yes | yes |

- 裁决字段（★：frames / score / action_sequence_sha256 / stop_reason / deaths）三向全等：**yes**
- 七个字段全部相等（含 steps / policy_input_sha256）：yes
- 差异明细：无
- 墙钟（争用条件下观测值）：rep1 `50.37` s，rep2 `53.623` s

#### seed 1001

| 字段 | 历史基准 | rep1 实测 | rep2 实测 | baseline=rep1 | baseline=rep2 | rep1=rep2 |
|---|---|---|---|---|---|---|
| `frames` ★ | `818` | `818` | `818` | yes | yes | yes |
| `steps` | `409` | `409` | `409` | yes | yes | yes |
| `score` ★ | `1512` | `1512` | `1512` | yes | yes | yes |
| `stop_reason` ★ | `death_stop` | `death_stop` | `death_stop` | yes | yes | yes |
| `deaths` ★ | `1.0` | `1.0` | `1.0` | yes | yes | yes |
| `action_sequence_sha256` ★ | `57f61f9958f4b4e567372c227ec38fc3f888033cdf851317f896724dc92c93fe` | `57f61f9958f4b4e567372c227ec38fc3f888033cdf851317f896724dc92c93fe` | `57f61f9958f4b4e567372c227ec38fc3f888033cdf851317f896724dc92c93fe` | yes | yes | yes |
| `policy_input_sha256` | `cad86d0e35787943731ada4c6089f6d47dcee6683f1d38259b0403516821f8cf` | `cad86d0e35787943731ada4c6089f6d47dcee6683f1d38259b0403516821f8cf` | `cad86d0e35787943731ada4c6089f6d47dcee6683f1d38259b0403516821f8cf` | yes | yes | yes |

- 裁决字段（★：frames / score / action_sequence_sha256 / stop_reason / deaths）三向全等：**yes**
- 七个字段全部相等（含 steps / policy_input_sha256）：yes
- 差异明细：无
- 墙钟（争用条件下观测值）：rep1 `36.134` s，rep2 `30.238` s

#### seed 1002

| 字段 | 历史基准 | rep1 实测 | rep2 实测 | baseline=rep1 | baseline=rep2 | rep1=rep2 |
|---|---|---|---|---|---|---|
| `frames` ★ | `1251` | `1251` | `1251` | yes | yes | yes |
| `steps` | `626` | `626` | `626` | yes | yes | yes |
| `score` ★ | `7438` | `7438` | `7438` | yes | yes | yes |
| `stop_reason` ★ | `death_stop` | `death_stop` | `death_stop` | yes | yes | yes |
| `deaths` ★ | `1.0` | `1.0` | `1.0` | yes | yes | yes |
| `action_sequence_sha256` ★ | `864101aed6fe0d13736477e6d9402f1fd0bd4e8eb3281a135bbb443f165e3827` | `864101aed6fe0d13736477e6d9402f1fd0bd4e8eb3281a135bbb443f165e3827` | `864101aed6fe0d13736477e6d9402f1fd0bd4e8eb3281a135bbb443f165e3827` | yes | yes | yes |
| `policy_input_sha256` | `bff6d25adfd3b8eb7230dc02cba7e1bc4729f9ae8b27065ddba9c3ed3771c7c1` | `bff6d25adfd3b8eb7230dc02cba7e1bc4729f9ae8b27065ddba9c3ed3771c7c1` | `bff6d25adfd3b8eb7230dc02cba7e1bc4729f9ae8b27065ddba9c3ed3771c7c1` | yes | yes | yes |

- 裁决字段（★：frames / score / action_sequence_sha256 / stop_reason / deaths）三向全等：**yes**
- 七个字段全部相等（含 steps / policy_input_sha256）：yes
- 差异明细：无
- 墙钟（争用条件下观测值）：rep1 `47.974` s，rep2 `45.237` s

### 2.2 `beta20k` — `.tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt`

#### seed 1000

| 字段 | 历史基准 | rep1 实测 | rep2 实测 | baseline=rep1 | baseline=rep2 | rep1=rep2 |
|---|---|---|---|---|---|---|
| `frames` ★ | `1612` | `1612` | `1612` | yes | yes | yes |
| `steps` | `806` | `806` | `806` | yes | yes | yes |
| `score` ★ | `12471` | `12471` | `12471` | yes | yes | yes |
| `stop_reason` ★ | `death_stop` | `death_stop` | `death_stop` | yes | yes | yes |
| `deaths` ★ | `1.0` | `1.0` | `1.0` | yes | yes | yes |
| `action_sequence_sha256` ★ | `c2ee8fc1335dbfa7d487f79fbefb119615e3326e029471cdd871581fcefb4c30` | `c2ee8fc1335dbfa7d487f79fbefb119615e3326e029471cdd871581fcefb4c30` | `c2ee8fc1335dbfa7d487f79fbefb119615e3326e029471cdd871581fcefb4c30` | yes | yes | yes |
| `policy_input_sha256` | `79adb9a5ca13ec0fe6c2fbe57f2d9a93051303ef50a388815d55ad02f4799fe9` | `79adb9a5ca13ec0fe6c2fbe57f2d9a93051303ef50a388815d55ad02f4799fe9` | `79adb9a5ca13ec0fe6c2fbe57f2d9a93051303ef50a388815d55ad02f4799fe9` | yes | yes | yes |

- 裁决字段（★：frames / score / action_sequence_sha256 / stop_reason / deaths）三向全等：**yes**
- 七个字段全部相等（含 steps / policy_input_sha256）：yes
- 差异明细：无
- 墙钟（争用条件下观测值）：rep1 `60.989` s，rep2 `57.234` s

#### seed 1001

| 字段 | 历史基准 | rep1 实测 | rep2 实测 | baseline=rep1 | baseline=rep2 | rep1=rep2 |
|---|---|---|---|---|---|---|
| `frames` ★ | `1970` | `1970` | `1970` | yes | yes | yes |
| `steps` | `985` | `985` | `985` | yes | yes | yes |
| `score` ★ | `18920` | `18920` | `18920` | yes | yes | yes |
| `stop_reason` ★ | `death_stop` | `death_stop` | `death_stop` | yes | yes | yes |
| `deaths` ★ | `1.0` | `1.0` | `1.0` | yes | yes | yes |
| `action_sequence_sha256` ★ | `d2cd9917d99b151e1f479ec94857caa2b7cbe866d6e70f2cd4b96e10aea116c5` | `d2cd9917d99b151e1f479ec94857caa2b7cbe866d6e70f2cd4b96e10aea116c5` | `d2cd9917d99b151e1f479ec94857caa2b7cbe866d6e70f2cd4b96e10aea116c5` | yes | yes | yes |
| `policy_input_sha256` | `6a51026c142093252534a3c9efe71d960c395cfb32d3369c8b816af9e2a32258` | `6a51026c142093252534a3c9efe71d960c395cfb32d3369c8b816af9e2a32258` | `6a51026c142093252534a3c9efe71d960c395cfb32d3369c8b816af9e2a32258` | yes | yes | yes |

- 裁决字段（★：frames / score / action_sequence_sha256 / stop_reason / deaths）三向全等：**yes**
- 七个字段全部相等（含 steps / policy_input_sha256）：yes
- 差异明细：无
- 墙钟（争用条件下观测值）：rep1 `67.88` s，rep2 `69.809` s

#### seed 1002

| 字段 | 历史基准 | rep1 实测 | rep2 实测 | baseline=rep1 | baseline=rep2 | rep1=rep2 |
|---|---|---|---|---|---|---|
| `frames` ★ | `1532` | `1532` | `1532` | yes | yes | yes |
| `steps` | `766` | `766` | `766` | yes | yes | yes |
| `score` ★ | `12124` | `12124` | `12124` | yes | yes | yes |
| `stop_reason` ★ | `death_stop` | `death_stop` | `death_stop` | yes | yes | yes |
| `deaths` ★ | `1.0` | `1.0` | `1.0` | yes | yes | yes |
| `action_sequence_sha256` ★ | `e86f54d5964fb63cfdf54d7df50f429e03c5b0db44391a80a4a5c2b76b12cfa3` | `e86f54d5964fb63cfdf54d7df50f429e03c5b0db44391a80a4a5c2b76b12cfa3` | `e86f54d5964fb63cfdf54d7df50f429e03c5b0db44391a80a4a5c2b76b12cfa3` | yes | yes | yes |
| `policy_input_sha256` | `e796af3763a4a7a796b31f469e0a24e8ffd39e6e297d44f5663a87cb2a07614e` | `e796af3763a4a7a796b31f469e0a24e8ffd39e6e297d44f5663a87cb2a07614e` | `e796af3763a4a7a796b31f469e0a24e8ffd39e6e297d44f5663a87cb2a07614e` | yes | yes | yes |

- 裁决字段（★：frames / score / action_sequence_sha256 / stop_reason / deaths）三向全等：**yes**
- 七个字段全部相等（含 steps / policy_input_sha256）：yes
- 差异明细：无
- 墙钟（争用条件下观测值）：rep1 `58.193` s，rep2 `60.199` s

---

## 3. 判定口径与结论

判定规则（任务规定）：只有 **frames、score、action_sequence_sha256、stop_reason、deaths** 全部与历史基准相同，**且**两次重复彼此相同，才写「复现成立」。任何一项不同即如实写 false，不使用「接近」「一致」等模糊表述。

| checkpoint | 裁决字段三向全等 | 七字段全等 |
|---|---|---|
| `b0` | yes | yes |
| `beta20k` | yes | yes |

**结论：复现成立（`REPRODUCED`）**

- 六行（2 checkpoint × 3 seed）裁决字段三向全等：**yes**
- 补充说明：all seven compared fields also match (steps and policy_input_sha256 included)

---

## 4. 边界声明

- 本腿**只做复现**，不构成新能力证据，也不是任何 gate 的通过凭据。
- 未读取/未运行 confirm 种子（61000–61099、61400–61499），未触碰 60000–60199；所用种子 1000–1002 均在 dev 白名单 1000–1099 内。
- 未修改任何代码/测试/配置，未触碰 `Document/Update/Alpha/v4.1.0/**`、`ASSET_SYNC.*` 及任何 `.tmp_v410_*` / `.tmp_v400_*` 资产；历史基准 JSON 以只读方式打开。
- 未干扰并发中的 v4.1.0 采集进程（PID 1884178）。
- 未执行任何 git add / commit / push。

## 5. 证据清单（`Document/Update/Alpha/v4.2.0/evidence/T02/`）

| 文件 | 说明 |
|---|---|
| `legacy_reproduce.json` | 机器可读汇总：运行 argv / 退出码 / 墙钟 / 实测值 / 期望值 / match 布尔 / 差异列表 |
| `LEGACY_REPRODUCE.md` | 本文 |
| `b0_dev3_rep1.json` / `b0_dev3_rep2.json` | B0 checkpoint 原始评测产物 |
| `beta20k_dev3_rep1.json` / `beta20k_dev3_rep2.json` | 20k checkpoint 原始评测产物 |
| `RAW_SHA256.txt` | 上述原始产物的 sha256 清单 |

过程日志（argv、stdout/stderr 尾部、runs.jsonl）位于 `.tmp_v420_t02a/`。

