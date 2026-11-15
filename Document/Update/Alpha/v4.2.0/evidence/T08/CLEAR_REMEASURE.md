# T08 — CLEAR_REMEASURE: 新仪器下的第一次有效过关测量

**结论（预注册判据，跑前写死）**：160 局（4 臂 × 2 口径 × 20 dev seed）中
`stage_clear` = **0**。

- 能力判据（任一局过关）→ **未满足**
- 排序判据（有臂均值超过 R0）→ **diag 口径 D2, D3、s1n 口径 D2, D3**（按均值；配对 95% CI 全部跨 0，见第 6 节）
- 因此成立的是 **分支 2**：① 有过关 / ② 无过关但有臂超 R0、帧数改善而能力未达 / ③ 无过关且无臂超 R0 / ④ 过关数与帧数不一致。本次为 **②** —— 没有任何一局过关，但 D2、D3 的两口径均值都高于 R0。

> 本文件只报告评测。没有训练、没有微调、没有使用特权信息；
> 策略输入只有像素（obs-v3）+ 自身动作历史，引擎 STATE 仅作裁判。
> 单训练 seed（seed 0）、每臂每口径 n=20 dev 局，不构成 gate，也不构成过关率。

## 1. 仪器

| 项 | 值 |
|---|---|
| 工具 | `tools/rl/v420_eval_shoot.py` |
| 工具 sha256 | `7dbc2a31…6e8519` |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` |
| 引擎 sha256 | `691bfd77…a568ef6` |
| SHOOT 释放 | 每 48 决策，该决策最后一个 STEP 帧 `mask & ~(BTN_SHOOT\|BTN_BOMB)` |
| 协议 | stage 0 / normal / `--max-frames 19887` / `--obs-window consecutive` / `--device cpu` / `--timeout 240` |
| 口径 | `diag` = `--stop-on-death`；`s1n` = 不停止（三条命，game_over） |
| seeds | dev 1000–1019（4 批 × 5 seed） |

旧仪器 `tools/rl/v410_student_eval.py` 在 `cand_to_mask` 中无条件 OR `BTN_SHOOT`，
没有任何释放路径，所以引擎永远看不到离开换面等待状态所需的 `SHOOT` 上升沿——
**此前所有 0/N 都不是对“能否过关”的测量**。新仪器的正负对照已在本 T08 目录的
`CLEAR_REACHABILITY.md` / `EVAL_SHOOT_PULSE.md` 中给出：同一条教师录像带，
period 48 时 `stage_clear=true`（13,195 帧 / 2,631,430 分），period 1000000007 时
`game_over`（15,564 帧）；两者 `mask_trace_sha256` 与独立实现逐位相同。

## 2. 被测 checkpoint（复制到私有目录后校验）

| 臂 | 私有 checkpoint | sha256 | 说明 |
|---|---|---|---|
| D2 | `.tmp_v420_data/T08/remeasure-d2/ckpt/d2_seed0_step5000.pt` | `9f284e48c230…3751b6` | D2 = DAgger round, beta=0 (the longest-surviving DAgger arm under the old ruler) |
| D3 | `.tmp_v420_data/T08/remeasure-d3/ckpt/d3_seed0_step5000.pt` | `91d11d22187b…68a603` | D3 = DAgger round, beta=0 |
| R3 | `.tmp_v420_data/T08/remeasure-r3/ckpt/r3_seed0_50000f.pt` | `83f1ac4436c7…4b4685` | R3@50k = the best RL arm under the old ruler |
| R0 | `.tmp_v420_data/T08/remeasure-r0/ckpt/a0_seed0_step5000.pt` | `96cb219eac83…6b89ad` | R0 = the untrained-but-anchored A0 BC checkpoint (the reference arm) |

源文件与副本 sha256 全部一致（见 `.tmp_v420_remeasure/sha_sources.txt` 与
`.tmp_v420_remeasure/sha_private_checkpoints.txt`）；源目录只读。

注意：这四个 checkpoint 的 schema 里 `arm` 都写成 `B0`（它们都是 B0 派生/续训的），
所以臂的身份只由**路径与 sha256** 区分，不能由 checkpoint 里的 `arm` 字段区分。

## 3. 四臂 × 两口径（各 20 局）

| 臂 | 口径 | n | 过关 | 均值帧 | 中位 | 最小 | 最大 | sd | stop_reason | errors | invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| D2 | diag | 20 | **0** | 1757.5 | 1764.0 | 1434 | 2455 | 259.4 | death_stop:20 | 0 | 0 |
| D2 | s1n | 20 | **0** | 3010.3 | 2954.0 | 2162 | 3616 | 373.7 | game_over:20 | 0 | 0 |
| D3 | diag | 20 | **0** | 1707.5 | 1573.5 | 1122 | 2253 | 354.0 | death_stop:20 | 0 | 0 |
| D3 | s1n | 20 | **0** | 2943.8 | 3026.0 | 2179 | 3914 | 568.3 | game_over:20 | 0 | 0 |
| R3 | diag | 20 | **0** | 1597.9 | 1497.5 | 816 | 2337 | 326.5 | death_stop:20 | 0 | 0 |
| R3 | s1n | 20 | **0** | 2765.1 | 2645.0 | 1920 | 3578 | 504.6 | game_over:20 | 0 | 0 |
| R0 | diag | 20 | **0** | 1650.4 | 1562.0 | 1397 | 2308 | 267.4 | death_stop:20 | 0 | 0 |
| R0 | s1n | 20 | **0** | 2824.3 | 2863.0 | 2137 | 3478 | 360.7 | game_over:20 | 0 | 0 |

合计：**160 局**，`stage_clear` = **0**，errors = 0，invalid = 0。

最深的一局 = **3914 帧**（D3，seed 1010，s1n 口径），比过关所需的 **13,195 帧** 少 **9281 帧**；160 局里没有任何一局达到 12,000 帧。

R0 参考（新仪器）：diag **1650.4**（n=20）、s1n **2824.35**（n=20）；
s1n 仅取 seed 1000–1014 时 = **2748.0**（这就是任务书里 n=15 的 2748.0）。

**R0 数据的出处（本任务未重跑全部 R0 行）**：R0 的 40 局里，35 局是 T08 setup 任务
已写好的新仪器批次（`.tmp_v420_data/T08/setsup/eval/r0-{diag,s1n}-*.json`，本任务开始时
只读快照到 `.tmp_v420_data/T08/remeasure-r0/out/setsup-snapshot-*.json`），
另 5 局（s1n 1015–1019，即任务书要求补到 20 的那一批）由本任务用私有 data 目录重跑；
重跑与快照逐 seed 帧数/分数完全一致（第 7 节），所以 R0 行与本次三臂可比。

## 4. 逐局 stage_clear（每格 = 该 seed 的帧数，全部 `stage_clear=false`）

### 4.1 diag（--stop-on-death）

| seed | D2 | D3 | R3 | R0 |
|---|---|---|---|---|
| 1000 | 1833 | 1454 | 1966 | 1409 |
| 1001 | 1485 | 1434 | 816 | 1914 |
| 1002 | 1938 | 2220 | 2337 | 1499 |
| 1003 | 1446 | 1572 | 1494 | 1557 |
| 1004 | 1470 | 2253 | 1410 | 1410 |
| 1005 | 1489 | 2253 | 1492 | 1605 |
| 1006 | 1731 | 1432 | 1945 | 1397 |
| 1007 | 1451 | 1412 | 1546 | 1408 |
| 1008 | 1759 | 2016 | 1816 | 1431 |
| 1009 | 1912 | 1980 | 1489 | 1492 |
| 1010 | 1945 | 2168 | 1363 | 1588 |
| 1011 | 1566 | 1940 | 1485 | 1413 |
| 1012 | 1935 | 1943 | 1500 | 1912 |
| 1013 | 1940 | 1122 | 1568 | 1728 |
| 1014 | 1434 | 1301 | 1971 | 2138 |
| 1015 | 1938 | 1487 | 1295 | 2308 |
| 1016 | 2455 | 1367 | 1980 | 1567 |
| 1017 | 1769 | 1575 | 1495 | 1494 |
| 1018 | 1715 | 1649 | 1544 | 1945 |
| 1019 | 1939 | 1572 | 1446 | 1793 |

### 4.2 s1n（不停止）

| seed | D2 | D3 | R3 | R0 |
|---|---|---|---|---|
| 1000 | 2973 | 2520 | 3103 | 2137 |
| 1001 | 2668 | 2362 | 1920 | 2654 |
| 1002 | 2782 | 3380 | 3507 | 2185 |
| 1003 | 3328 | 2228 | 2407 | 2741 |
| 1004 | 2893 | 3306 | 2210 | 2456 |
| 1005 | 2529 | 3381 | 2480 | 2579 |
| 1006 | 3079 | 3032 | 2981 | 2572 |
| 1007 | 3336 | 2383 | 2889 | 3280 |
| 1008 | 3483 | 3618 | 3509 | 3181 |
| 1009 | 2730 | 3323 | 2373 | 3000 |
| 1010 | 3034 | 3914 | 2180 | 3091 |
| 1011 | 2977 | 2732 | 3578 | 2540 |
| 1012 | 2859 | 3020 | 3391 | 2935 |
| 1013 | 3589 | 2492 | 2537 | 2791 |
| 1014 | 2162 | 3240 | 3270 | 3078 |
| 1015 | 3490 | 2179 | 2660 | 3110 |
| 1016 | 3616 | 2222 | 2630 | 2550 |
| 1017 | 2913 | 3307 | 2258 | 3478 |
| 1018 | 2935 | 3857 | 3007 | 3064 |
| 1019 | 2830 | 2381 | 2412 | 3065 |

逐局明细（含 score、deaths、stop_reason、error、来源文件）见 `clear_remeasure.json` 的
`arms.<臂>.<口径>.episodes`。

## 5. 与旧仪器的同名臂差值

| 臂 | 口径 | 旧仪器均值 | 新仪器均值 | 差（新−旧） | 旧仪器来源 |
|---|---|---:|---:|---:|---|
| D2 | diag | 1902.10 | 1757.5 | -144.6 | `Document/Update/Alpha/v4.2.0/evidence/T05/dagger_d2.json` |
| D2 | s1n | 3185.70 | 3010.3 | -175.4 | `Document/Update/Alpha/v4.2.0/evidence/T05/dagger_d2.json` |
| D3 | diag | 1716.45 | 1707.5 | -9.0 | `Document/Update/Alpha/v4.2.0/evidence/T05/dagger_d3.json` |
| D3 | s1n | 3025.20 | 2943.8 | -81.3 | `Document/Update/Alpha/v4.2.0/evidence/T05/dagger_d3.json` |
| R3 | diag | 1589.20 | 1597.9 | +8.7 | `Document/Update/Alpha/v4.2.0/evidence/T06/budget_200k.json (R3_50k block)` |
| R3 | s1n | 2891.60 | 2765.1 | -126.5 | `Document/Update/Alpha/v4.2.0/evidence/T06/budget_200k.json (R3_50k block)` |
| R0 | diag | 1636.80 | 1650.4 | +13.6 | `Document/Update/Alpha/v4.2.0/evidence/T06/budget_200k.json (R0 block, from T06/dev/merged-R0.json)` |
| R0 | s1n | 3011.50 | 2824.3 | -187.2 | `Document/Update/Alpha/v4.2.0/evidence/T06/budget_200k.json (R0 block, from T06/dev/merged-R0.json)` |

**差值不可当作漂移证据。** 两台仪器测的是不同的游戏：旧仪器每帧都按住 SHOOT，
新仪器每 48 决策释放一帧。同一个 checkpoint 在两条通道上会走出不同的轨迹，
所以上表的差是**通道差**，不是同一个策略的时间漂移，也不是训练效果的变化。
旧数字只用于说明每个臂此前绑定在哪台仪器上。

## 6. 四条分支判定（跑前写死）

| 分支 | 条件 | 本次是否成立 |
|---|---|---|
| ① | 160 局里 stage_clear ≥ 1 | 否 |
| ② | 无过关，但有臂 diag 或 s1n 均值超 R0 | **是** |
| ③ | 无过关且无臂超 R0 | 否 |
| ④ | 有臂的过关数与帧数严重不一致 | 前置不成立（无任何过关） |

**成立的是分支 ②。** 160 局 `stage_clear` = 0，但按均值 D2、D3 在两个口径上都超过 R0；
能力判据（过一次关）**没有**满足，所以这是“帧数改善、能力未达”，不是能力证据。

排序读数（新仪器，n=20；配对差 = 同 seed 相对 R0 的逐 seed 差均值，正 = 比 R0 长）：

| 臂 | diag 均值 | vs R0 | 配对差 | 配对差 95% CI | s1n 均值 | vs R0 | 配对差 | 配对差 95% CI |
|---|---:|---:|---:|---|---:|---:|---:|---|
| D2 | 1757.5 | +107.1 | +107.1 | (-63.4, +277.6) | 3010.3 | +186.0 | +185.9 | (-46.2, +418.1) |
| D3 | 1707.5 | +57.1 | +57.1 | (-181.5, +295.7) | 2943.8 | +119.5 | +119.5 | (-169.6, +408.6) |
| R3 | 1597.9 | -52.5 | -52.5 | (-277.8, +172.8) | 2765.1 | -59.2 | -59.2 | (-372.2, +253.7) |
| R0 | 1650.4 | +0.0 | +0.0 | (+0.0, +0.0) | 2824.3 | +0.0 | +0.0 | (+0.0, +0.0) |

**这个排序要打折看**：

1. 所有配对 95% CI 都跨 0（D2 diag 的 CI 是 −63…+278 帧，D2 s1n 是 −46…+418 帧），
   所以“D2/D3 超过 R0”只成立在**均值**这条预注册判据上，不构成“已证明更快”的统计结论；
   帧数 sd 在 259–568 帧之间，比臂间差（57–186 帧）大一个量级。
2. 把 R0 限制到 seed 1000–1014（n=15，即任务书里的 2748.0）时，R3 在两个口径上都翻到 R0 之上
   （R3 diag n=15 = 1613.2 vs R0 1593.4；R3 s1n n=15 = 2822.3 vs R0 2748.0；n=20 时 R3 = 1597.9 / 2765.1，都低于 R0 的 1650.4 / 2824.3）。
   样本量差 5 局就能翻转 R3 的符号，说明这几百帧的差不是稳定排序。
3. 因此本报告只主张：**在新仪器下没有任何一局过关**；三臂/R0 的帧数排序是描述性的，
   且“旧仪器上 D2 最长”这一点在新仪器上按均值仍然成立（D2 > D3 > R0 > R3）。

## 7. 仪器可信性与本次测量的自检

- 正/负对照：见第 1 节（本任务只读引用，未重跑）
- 脉冲确实在跑：160 局每局释放帧数都 > 0（= `steps // 48`），合计 **3731** 个释放帧，没有一局是 0 释放（0 释放只会出现在 period 大于整局的负对照里）
- R0 行可复现性：先前观察到的 40 个 R0 帧数与本次快照**完全一致**
- 我自己的 R0 s1n 1015–1019 补测与既有批次逐 seed 帧数/分数**完全一致**
- 本任务 24 条批次 + 1 条 R0 补测（另有 1 条单 seed 冒烟）退出码全部 0；errors = 0；invalid = 0
- 每个臂每个口径用各自的私有 `cp -a` data 目录；源 checkpoint 与 T04/T05/T06 只读，未写入
- 每批 5 seed，命令模板见第 9 节；单条命令墙钟 27–223 s

## 8. not_claimed

- single training seed per arm (seed 0); no seed-variance claim
- n = 20 dev episodes per arm per ruler; no gate, no pass rate, no Wilson interval is claimed as a capability rate
- this is a pixel-only evaluation (obs-v3 playfield + HUD + action history); the engine STATE reply is a referee for death/stage/score, never a policy input, and was not used as privileged information
- it sits outside engine_truth: no engine reward or termination signal feeds the policy, so this is an R1-scoped (pixels-only) evaluation, but with NO training, NO fine-tuning and NO privileged information — pure evaluation of existing checkpoints
- dev seeds only: no confirm seed (60000-60199), no corpus seed (1-20), no gate seed
- the frames are not a clear rate; a stage_clear count of zero does not bound the clear rate above by anything except n
- no causal claim about why a checkpoint survives longer or shorter; no arm was trained or tuned in response to these numbers

## 9. 命令、退出码、墙钟

命令模板（三个臂各 8 条 = 4 批 × 2 口径，共 24 条；R0 s1n 补测 1 条；
R0 其余 7 批是 T08 setup 任务已写好的新仪器批次，本任务未重跑，只做只读快照与校验）：

```
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v420_eval_shoot.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir .tmp_v420_data/T08/remeasure-<arm>/<arm>-<ruler>-data \
  --checkpoint .tmp_v420_data/T08/remeasure-<arm>/ckpt/<ckpt>.pt \
  --seeds <lo>-<hi> --stage 0 --difficulty normal --max-frames 19887 \
  --obs-window consecutive --device cpu --timeout 240 \
  --shoot-pulse-period 48 --out .tmp_v420_data/T08/remeasure-<arm>/out/<arm>-<ruler>-<lo>_<hi>.json \
  [--stop-on-death]
```

| # | 批（tag） | 退出码 | 墙钟 s | load 起→止 | 启动时匹配 `th08-modern` 的外部进程数 |
|---|---|---:|---:|---|---:|
| 1 | r0-s1n-1015_1019-topup | 0 | 223.0 | 2.69→7.12 | na |
| 2 | d2-diag-1000_1004 | 0 | 136.4 | 6.17→6.10 | 12 |
| 3 | d2-diag-1005_1009 | 0 | 135.0 | 6.10→5.75 | 9 |
| 4 | d2-diag-1010_1014 | 0 | 136.5 | 5.75→5.17 | 9 |
| 5 | d2-diag-1015_1019 | 0 | 143.3 | 5.17→4.73 | 3 |
| 6 | d2-s1n-1000_1004 | 0 | 191.6 | 4.43→2.48 | 3 |
| 7 | d2-s1n-1005_1009 | 0 | 200.8 | 2.48→4.62 | 0 |
| 8 | d2-s1n-1010_1014 | 0 | 196.9 | 4.62→2.90 | 0 |
| 9 | d2-s1n-1015_1019 | 0 | 206.4 | 2.67→3.90 | 0 |
| 10 | d3-diag-1000_1004 | 0 | 137.2 | 3.59→3.89 | 0 |
| 11 | d3-diag-1005_1009 | 0 | 132.3 | 3.89→2.71 | 0 |
| 12 | d3-diag-1010_1014 | 0 | 126.1 | 2.50→4.14 | 0 |
| 13 | d3-diag-1015_1019 | 0 | 115.2 | 3.80→3.09 | 0 |
| 14 | d3-s1n-1000_1004 | 0 | 182.1 | 3.09→3.22 | 0 |
| 15 | d3-s1n-1005_1009 | 0 | 205.4 | 2.96→3.20 | 0 |
| 16 | d3-s1n-1010_1014 | 0 | 197.2 | 3.03→3.85 | 0 |
| 17 | d3-s1n-1015_1019 | 0 | 182.3 | 3.85→5.54 | 0 |
| 18 | r3-diag-1000_1004 | 0 | 119.0 | 5.10→1.97 | 0 |
| 19 | r3-diag-1005_1009 | 0 | 124.2 | 1.81→2.46 | 0 |
| 20 | r3-diag-1010_1014 | 0 | 117.8 | 2.46→3.79 | 0 |
| 21 | r3-diag-1015_1019 | 0 | 116.2 | 3.79→1.95 | 0 |
| 22 | r3-s1n-1000_1004 | 0 | 174.1 | 1.95→4.20 | 0 |
| 23 | r3-s1n-1005_1009 | 0 | 188.7 | 4.20→4.28 | 0 |
| 24 | r3-s1n-1010_1014 | 0 | 192.2 | 4.28→4.17 | 0 |
| 25 | r3-s1n-1015_1019 | 0 | 174.8 | 3.84→4.78 | 0 |

另有一条单 seed 冒烟（`d2-smoke-1000`，seed 1000，27.5 s，rc=0，1833 帧），只用于确认私有 cp -a data 目录可用，不计入 160 局。

`load` 是 `/proc/loadavg` 的 1 分钟值。最后一列是 `pgrep -fc 'th08-modern'` 的计数：
它会把 xvfb-run / python / 引擎三层外壳都算进去（约 3 个进程 = 1 条并发评测），所以 12 表示
约 4 条别人的评测在跑，0 表示机器上只有本任务这一条。**争用标注**：本任务开始时（约 13:06 WSL 时间）
机器上已有另一个 T08 setup 任务在并发跑 3–4 条评测（seed 1005–1014 的 r0/setsup 复跑），
头 4 条批次的 load 起点为 2.7–6.2（R0 补测期间升到 7.1），其余批次在 load 1.8–5.5 下完成。
争用只影响墙钟，不影响帧数/过关：同一 seed 的帧数在两次独立运行中逐位一致（第 7 节）。

## 10. 复现

```
wsl -e bash .tmp_v420_remeasure/00_setup.sh      # 校验 sha、复制 checkpoint、建私有 cp -a data 目录
wsl -e bash .tmp_v420_remeasure/run_one.sh <ARM> <diag|s1n> <lo-hi> <CKPT> <DATADIR> <TAG> <OUT>
wsl -e bash .tmp_v420_remeasure/aggregate.py     # 汇总并写本文件 + clear_remeasure.json
```

所有批次记录在 `.tmp_v420_remeasure/timing.tsv`，完整 stdout 在 `.tmp_v420_remeasure/logs/`，
每份 summary 的 sha256 见 `clear_remeasure.json` 的 `batches`。

