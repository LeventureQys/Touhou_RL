# EXP-P0-048 — WP7 真实 random/still 对局（provisional cap）

- 日期：2026-11-08（America/Los_Angeles）/ 2026-11-09（WSL，CST）
- Provider / 模型：DeepSeek / DeepSeek v4.1 flash
- 工作包：WP7（基准执行与 scoreboard）
- 状态：**真实 6 局已跑完并产出 scoreboard；cap 为 provisional，不是门评估、不是 20 种子开发集。**
- 前序阻塞：`EXP-P0-024`、`EXP-P0-042`（缺 human_reference 即 exit 3）

---

## 1. 本次真实能力

| 项目 | 值 |
|---|---|
| 启动方式 | `tools/rl/v400_wsl_env.WslHarnessClient`（`wsl.exe -d Ubuntu -- env -u DISPLAY SDL_AUDIODRIVER=dummy <ELF> ... --rl-role shadow`） |
| 引擎 ELF | `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern` |
| 二进制 sha256（本次 6 局） | `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb` |
| 引擎 git HEAD | `f563df58c44037b44208c393a86540b434b0c83b` |
| 数据目录 | `/root/v400-data/game` |
| WSL / 内核 | Ubuntu / `6.18.54.1-microsoft-standard-WSL2` |
| 协议 | 未改动：`HELLO / RESET / STEP / STATE / QUIT`（`--rl-control 0,1 --rl-role shadow`，null device，无 SDL/GL/窗口/音频） |
| 裁判 | `touhou_rl.v400.bench.judge.judge`，输入为真实 `STATE` 25 字段（`TrackedFields()`，与 SHM 发布的是同一记录；Windows 侧无法 mmap WSL `/dev/shm`，本 runner 也不挂 SHM） |

## 2. 字段映射核实（回答 deaths=3.0 / lives=0.0 是否映射错误）

结论：**映射正确**（`F_DEATHS=6`→`globals.deaths`，`F_LIVES=7`→`globals.livesRemaining`，`F_BOMBS=8`→`globals.bombsRemaining`）。
用真实引擎、`still`、seed 1000，逐步读取 `STATE`（每决策步 2 帧）验证：

| 观测（真实 f0 帧号） | f6 deaths | f7 lives | f8 bombs | f4 score | 解释 |
|---|---:|---:|---:|---:|---|
| reset | 0.0 | 2.0 | 3.0 | 0 | `RESET` 后初始：0 死、2 残机、3 雷 |
| f0=1028 | 0→1 | 2.0 | 3.0 | 33303 | 死亡提交（`AddToDeaths(1)`），power 归零 |
| f0=1032 | 1.0 | 2→1 | 3.0 | 33405 | 30 帧后复活检查 `AddLives(-1)` |
| f0=1434 | 1→2 | 1.0 | 3.0 | 49844 | 第二次死亡提交 |
| f0=1438 | 2.0 | 1→0 | 3.0 | 49844 | 复活检查，残机耗尽 |
| f0=1748 | 2→3 | 0.0 | 3.0 | 59953 | 第三次死亡提交（`GetLives()<=0`，进入 game over 分支） |
| f0=6000（继续空推） | 3.0 | 0.0 | 3.0 | 59953 | 分数、`f3` 冻结 |

交叉证据：
- `DEBUG` 全程 `curState=2 / setupState=0 / globals=1 / stage=1 / difficulty=1 / shot=0 / threads=1`，**不随 game over 变化**，因此 DEBUG 不能作为终局信号；
- `f3`（`gameplayFrameCounter`）在 f0=1753 后冻结（f0 仍在涨），`f4` 分数同步冻结——这是 retry menu 已接管、游戏逻辑已停止的直接证据；
- 攻击无 bomb 输入，`f8` 恒为 3.0，与策略从不按 bomb 一致；
- 源码对应：`src/Global.hpp`（`deaths` 0x64 / `livesRemaining` 0x74 / `bombsRemaining` 0x80）、`src/rlharness/rlharness.cpp` `TrackedFields()`、`src/Player.cpp`（`AddToDeaths(1)` 与 `if (GetLives() > 0) AddLives(-1) else showRetryMenu = 1`）。

## 3. 终局判据修正（这才是 3000 帧异常的根因）

**根因：不是映射错误，而是终局判据缺失。** 第三次死亡时引擎设置 `showRetryMenu=1`，但
`PumpOneFrame()` 不会因此返回 false，`STEP` 也不会返回 `exit=1`；retry menu 只推进全局帧计数
（f0 1748→6000），`gameplayFrameCounter`（f3）与分数冻结。旧 runner 因此把菜单时间当成了对局，跑满 3000 帧。

修正后的判据（`tools/rl/v400_bench_run.py`）：
- 每决策步读一次真实 `STATE`；当 `deaths` 增加且 `livesRemaining <= 0` 时判定 `end="game_over"` 立即结束；
  这正对应该死亡分支随后的 `showRetryMenu`（同一 `GetLives()<=0` 条件），比菜单早约 30 帧的死亡动画起点；
- 仍保留 `STEP exit=1` 与 provisional `max_frames` 两条退出路径；
- 每局记录 `gameplay_frame`、`terminal_rule`、`cap_hit`、`deaths/lives/bombs/score`（全部来自真实 `STATE` 经 `judge`）。

## 4. 旧结果处理（append-only，不删不改）

旧 6 行为“帧上限落在 retry menu 内”的无效对局，已在 `bench/results.jsonl` 追加 1 条
`status="invalid"` 审计记录（含 6 个 `started_at`、原因、被作废行的 agent/seed/frames/score/end），
旧行本身**未被编辑或删除**；`--summarize` 会把这些行排除出表格并在 “Superseded attempts” 段落列出。
审计记录时间：`2026-11-08T15:32:26.962501+00:00`。

## 5. 命令与真实退出码

```powershell
# 1) 单元/契约测试
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp7_real_t -q tests/test_v400_bench_runner.py
# exit 0, 27 passed

# 2) 验收命令（第一局 random/1000，真实引擎）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools/rl/v400_bench_run.py --agent random --suite S1N --seed 1000 --execute --max-frames 3000
# exit 0（stdout 已落盘 .tmp_v400_wp7_real_run2_random_1000.json）

# 3) 其余 5 局（同命令换 agent/seed）
# random/1001, random/1002, still/1000, still/1001, still/1002 -> 每条 exit 0

# 4) 生成 scoreboard
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools/rl/v400_bench_run.py --summarize
# exit 0, {"episodes": 6, "errors": 0, "invalidated": 6, "status": "ok"}
```

## 6. 真实 6 局结果（valid）

| Agent | Suite | Seed | Frames（到 game over） | Gameplay frame | Deaths | Bombs used | Lives end | Score | End | Wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| random | S1N | 1000 | 2586 | 2588 | 3.0 | 0.0 | 0.0 | 41193 | game_over | 2.344 |
| random | S1N | 1001 | 1790 | 1792 | 3.0 | 0.0 | 0.0 | 26031 | game_over | 2.094 |
| random | S1N | 1002 | 2004 | 2006 | 3.0 | 0.0 | 0.0 | 34615 | game_over | 2.094 |
| still | S1N | 1000 | 1748 | 1750 | 3.0 | 0.0 | 0.0 | 59953 | game_over | 2.062 |
| still | S1N | 1001 | 1692 | 1694 | 3.0 | 0.0 | 0.0 | 63693 | game_over | 2.094 |
| still | S1N | 1002 | 1748 | 1750 | 3.0 | 0.0 | 0.0 | 63762 | game_over | 2.093 |

- 6/6 `status="ok"`，`end="game_over"`，`cap_hit=false`（全部在 3000 帧上限前死亡）。
- `bench/results.jsonl` 现共 13 行：6 条作废 ok 行 + 1 条 invalid 审计记录 + 6 条有效 ok 行（只追加）。
- scoreboard：`Document/Update/Alpha/v4.0.0/bench/scoreboard.md`（`--summarize` 生成）。

## 7. 确定性旁证

`random/1000` 三次独立运行（旧二进制 sha `a6b146ff…`、新二进制 sha `591c521d…`、以及独立探针）均得到：
`reset_hash=3eafc035dc4fe6a9`、`score=41193`、第三次死亡发生在同一逻辑位置（旧行 3000 帧截断、
新行 2586 帧 game over）。二进制在 23:27:10 被并行工作流重建（a6b146ff → 之后又回 591c521d），
故旧/新两批行的 `harness_binary_sha256` 不同；**本表 6 行全部为 `591c521d…`**，批次内一致。

## 8. 剩余限制（必须与分数同时阅读）

1. **cap 是 provisional**：无 human_reference（`bench/human_reference.json` 仍不存在），3000 帧为操作者设定；本次 6 局都未触及 cap（`cap_hit=false`），cap 只是上界。
2. **样本量**：每策略 3 个 dev 种子（1000–1002），共 6 局；**不是 20 种子开发集**，不是 confirm split（60000+），不是 PSN；n=3 的 bootstrap 区间无推断意义，仅描述这 6 次尝试。
3. **单一条件**：只有 S1N（stage 1 / normal / standard），不含多 stage 套件（runner 对多 stage 套件显式拒绝，不伪造）。
4. **deaths 饱和**：终局规则使每局都在“耗尽残机的死亡”处结束，因此 deaths 恒为 `lives_start+1`，不能区分策略；可区分量是 frames（存活时长）与 score。
5. **cleared 不可解释**：无 stage clear 检测，`cleared=false` 表示“未观察到通关”，clear-rate 列不是通过率。
6. **bombs_used 结构性为 0**：两个控制策略从不按 bomb，只能反映计数不动，不反映投弹能力。
7. **不是 3 条命耗尽即提前结束的模型错误**：已用真实引擎逐步验证映射与终局，旧 3000 帧行已作废（见 §2/§3/§4）。
8. 未做人机对比、未做门评估、未做像素策略评估（控制策略不读观测，`privilege=deploy_pixel` 是“策略未使用任何特权信息”的构造性事实）。
