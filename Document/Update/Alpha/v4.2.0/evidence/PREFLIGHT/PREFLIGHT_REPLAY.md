# PREFLIGHT — HASH 重建 6 次重放（G420-0 的「HASH 重建零未解释差异」腿）

- 时间：2026-11-13T17:3x+08:00
- 作用：证明旧 S1 语料的**逐帧掩码可以在新进程里逐比特重建引擎状态**。
  这是 G420-0 第二条腿（HASH 重建）的实测，不是整门通过。
- 脚本：[`replay/replay_probe.py`](replay/replay_probe.py)、编排 `.tmp_v420_replay_all.sh`
- 原始 JSON：`.tmp_v420_data/replay/seed{10,11,12}_rep{1,2}.json`（每个约 186 KB，含逐决策 4 个逐帧 HASH）

## 1 方法

1. 读 shard 的 `labels.jsonl` → 每行 `step_frames`（4 个**逐帧**按钮掩码）；
2. `Th08Env(engine, data_dir, shm_name=…, role="env", decision_frames=2)` +
   `StageStart(stage=0, difficulty="normal", team=0, seed=<shard seed>, resources="standard")` 开局；
3. 逐决策 `advance(mask, 1)` ×4，取第 4 帧引擎 HASH 与记录 `hash_post` 比较；
4. 引擎 data 目录用 shard 自带 `env/` 的**副本**（`score.dat`、`th08.cfg` 复制，
   `th08.dat`/`thbgm.dat` 走符号链接），**不写语料**。

每 seed 跑 500 决策 = **2,000 帧**，共 2 次重复。

## 2 结果（6/6 全部 0 差异）

| seed | rep | 决策 | 帧 | `reset_hash`==首行 `hash_pre` | `hash_post` 不一致 | 首个不一致 | 墙钟(s) | 退出码 |
|---|---|---:|---:|---|---:|---|---:|---:|
| 10 | 1 | 500 | 2000 | **true** | **0** | null | 34.47 | 0 |
| 10 | 2 | 500 | 2000 | **true** | **0** | null | 36.01 | 0 |
| 11 | 1 | 500 | 2000 | **true** | **0** | null | 32.95 | 0 |
| 11 | 2 | 500 | 2000 | **true** | **0** | null | 43.40 | 0 |
| 12 | 1 | 500 | 2000 | **true** | **0** | null | 44.39 | 0 |
| 12 | 2 | 500 | 2000 | **true** | **0** | null | 44.48 | 0 |
| **合计** | | **3,000** | **12,000** | **6/6** | **0 / 3,000** | — | 235.7 | **6×0** |

- 吞吐（争用条件下）：2,000 帧 / 33–44 s ≈ **45–61 fps**（`role="env"`，每帧渲染）。
- 无 `exception`、无 `exit`、无 `advanced != 1`。

## 3 结论与边界

**成立**：旧 S1 在**它自己记录的 4 帧节拍**上，可以用新进程 + 副本 data 目录逐比特重建；
每次重放的 `reset_hash` 与语料首行 `hash_pre` 相同，3,000 个决策边界 HASH 全等。
所以「HASH 重建」这一条对旧语料是干净的，且**前缀重放机制可用于 T03 的 park/fork**。

**不成立/不能推出**：

- 这不证明 2 帧契约成立。槽 1（帧 `d+2..d+3`）仍然**无观测、无 HUD、无标签**，
  4 帧节拍的 `hash_post` 只校验每 4 帧一个边界，**没有**校验每 2 帧的边界。
- 不能用它给「所有已消费样本的时间/动作索引一致」背书——那条腿仍然失败。

## 4 下一步

T02 的正式 `audit-time --execute` 要在合成一次更完整的检查：每 2 帧一个槽的索引、
4 个连续渲染帧、`history` 8×2 帧口径、逐帧掩码与引擎实际生效的一致性、以及
冻结模型的在线/离线前向一致性。本 PREFLIGHT 只提供已验证的重放基座与成本数字。
