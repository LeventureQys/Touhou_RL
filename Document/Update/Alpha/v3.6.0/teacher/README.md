# v3.6.0 `teacher/`：战术教师轨迹与标签的索引

- 日期：2026-10-25
- 作用：需求文档 §7 要求交付 `teacher/`（战术教师轨迹与标签）。本目录**不再复制数据**（trace 合计约 1 GB，且可由命令重采），只提供**索引与字段说明**，让每一条标签都能被定位、被复核。
- 机器可读：[index.json](<./index.json>)

---

## 1 教师是谁

本版没有训练任何学习型教师。所有"教师"都是**规则控制器**：

| 角色 | 实现 | 说明 |
|---|---|---|
| 局部避弹教师（baseline） | `touhou_rl/pilot.py::AvoidPilot` | 18 动作束搜索、20 帧线性外推、最坏情形 clearance 排序；bomb 仅在 clearance < −1.5 时触发 |
| ClearEnemy 战术教师 | `touhou_rl/tactics.py::choose_target` + pilot 的 `aim_weight`/`aim_safe_margin` | 只替换"点瞄谁"，其余不变 |
| CollectPower 战术教师 | `touhou_rl/tactics.py::collect_power_override` | 条件满足时覆盖为"上移/停留+聚焦" |
| 人类教师（回放） | `probe35 --replay` | 输入来自 `g_GuiMessageInputCurrent`，已验证方向一致 13/13 |

---

## 2 轨迹在哪

| 语料 | 路径 | 用途 | 重采命令 |
|---|---|---|---|
| D1（教师，9 局） | `../v3.5.0/probe/s0-teacher/` | T1 离线门（含真实死亡 18 次） | `python -m touhou_rl.probe35 collect --out … --episodes 9` |
| D2（人类，6 局） | `../v3.6.0/data/probe/d2-replay/` | T1 复现（含人类行为分布） | `probe35 collect --replay …` |
| D2 补充（整面 2 局） | `../v3.6.0/data/probe/d2-stageclear/` | T0 的 stage_clear 证据 | 同上，`--max-steps 9000` |
| D3（人类 + 新字段，6 局） | `../v3.6.0/data/probe/d3-fields/` | T1 判定语料（含 `slots` 与相位字段） | 同上 |
| T2-ENEMY（真机 40 局） | `../v3.6.1/data/probe/t2-enemy/` | T2 paired comparison | `probe35 collect --tactic none --tactic clear_enemy --episodes 40 --arm-seed 20261025` |
| T2-POWER（真机 40 局） | `../v3.6.1/data/probe/t2-power/` | T2 paired comparison | 同上，`--tactic collect_power --arm-seed 20261026` |

除 D1 外均在 `.gitignore` 中（体积原因），`episodes.jsonl`、`manifest.json`、`s0-report.json` 是入库的元数据。

---

## 3 逐决策标签在哪

| 标签 | 落盘位置 | 生成方式 | 可用语料 |
|---|---|---|---|
| 动作 + 教师 clearance | `trace/*.jsonl` 的 `action`、`teacher` | 采集时写入 | 全部 |
| 16 个未来标签（`death_60`、`min_clearance_60`、`attributed_spawns_60` …） | `data/features-*.npz` 的 `label_*` | `touhou_rl/ablate36.py::build_labels`，**只读未来快照** | D1/D2/D3 |
| 8 个特征块（132 列） | `data/features-*.npz` 的 `block_*` + `cols_*` | `ablate36.extract_episode` | D1/D2/D3 |
| 敌人相位/身份 | `trace/*.jsonl` 的 `enemies[].{damageable, phase_starting_life, enemy_index, flags1, flags2, shoot_interval_frames}` | `state.read_enemies`（偏移均有源码 `C_ASSERT`） | D3 起 |
| 子弹槽位（判"新弹"） | `bullets/*.npz: slots` | `trace35.TraceWriter` | D3 起 |
| 战术意图 / 中止原因 | `trace/*.jsonl` 的 `controller`（`tactic`、`reason`、`override`、`target_slot`、`abort_reason`） | `probe35._choose_action` | T2 两轮 |
| 逐局指标 | `experiments/*/ablation.json`（T1）、`experiments/t2-*/paired.json`（T2） | 分析脚本 | 全部 |

---

## 4 契约与不可读项

字段级契约见 [../contract/策略标签契约.md](../contract/策略标签契约.md)。仍然**不可读**的项（不得在报告中当作特征）：

| 项 | 状态 | 原因 |
|---|---|---|
| bullet → enemy 归属 | `unknown`（用槽位+就近归属重建，归属率 74.9%） | `struct Bullet` 无 owner 字段 |
| 符卡相位 | `unknown` | `currentSpellCardNumber` 正常游玩不写（D2 全程 −1） |
| 无敌/阶段转换 | `damageable` / `phase_starting_life` 可读，但"不可伤害"态只占 7.2%（D3-2 未达 10%） | 数值已可读，方差不足 |
| 激光 | 未落盘 | 采集未写，`state.read_lasers` 已有 |

---

## 5 复现路径

```powershell
# 特征 + 标签（离线，约 1 分钟/语料）
python -m touhou_rl.ablate36 --probe <语料> --out <实验目录> --cache <缓存> --speed-mode measure
# T2 配对
python tools/rl/analyze_t2_pairs.py --probe <语料> --tactic <臂> --out <实验目录>
```
