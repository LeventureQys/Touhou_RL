# S3-5 PPO / 评估 / play 任务书

- 所属 Stage：Stage 3（开发）
- 依赖前置：S3-2（`Th08Env`）、S3-4（`DanmakuPolicy` 与检查点）
- 并行状态：需等 S3-2/S3-4
- 所属阶段：开发

---

## 1. 背景与目标（零上下文）

三件事：① 在真实游戏上做 PPO 微调（并防止把模仿先验忘掉）；② 用真实游戏给模型打分
（存活帧数、触达关卡、死亡次数）；③ 交付「一条命令就能自己开打」的脚本（需求原文：模型不需要
会选菜单，只需会玩）。

## 2. 交付物

```
touhou_rl/ppo.py       # PPOConfig / train_ppo / _gae / CLI
touhou_rl/evaluate.py  # EvalConfig / run_evaluation / CLI
touhou_rl/play.py      # play() / CLI
```

## 3. 契约要点

| 契约节 | 要求 |
| --- | --- |
| §5 | 标量奖励来自 `RewardEngine`，向量同时保留在 `info` |
| §6 | 只用 `env.reset/step`，不直接读内存 |
| §8 | 检查点格式与 BC 一致 |
| §9/§10 | CLI 参数、退出码、超时与错误行为 |
| §3 | 评估/play 的观测必须与训练同源（同一 `obs.encode`） |

## 4. 算法要求

- PPO：`gamma=0.995`、`lam=0.95`、`clip=0.2`、`vf_coef=0.5`、`ent_coef=0.01`、`lr=1e-4`；
- GAE 用 `terminated` 掩码（终止步不 bootstrap）；
- **KL 锚定**：`kl_coef>0` 且给出 `--bc-checkpoint` 时，对四个头的分布加
  `KL(当前策略 ‖ BC 策略)` 惩罚（BC 模型冻结、`eval` 模式）；
- rollout 长度默认 1024 决策（≈35–50 s 墙钟），训练 batch 256，4 epoch；
- 每 `save_every` 步存 `ppo-<step>.pt`，始终更新 `last.pt`。

## 5. 输入 / 输出

- `ppo`：`--out`（必填）、`--bc-checkpoint`/`--init-checkpoint`/`--kl-coef`/`--total-steps`/`--frameskip`/`--practice-stage`/`--max-steps`；
  输出 `ppo-summary.json`（每 iteration 的奖励均值、value loss、步/秒）+ 检查点；
- `evaluate`：`--checkpoint --out --episodes --difficulty --team --practice-stage --frameskip --deterministic`；
  输出 `eval-report.json`（每局 steps/frames/stages/deaths/score/graze/screenshot）+ 汇总；
- `play`：`--checkpoint --minutes --difficulty --team --practice-stage`；终端逐局 JSON，可选 `--out` 落盘。

## 6. 可观察的验收标准

1. `ppo` 冒烟（`--total-steps 64 --rollout-steps 32 --max-steps 40`）能跑完一个 iteration 并保存检查点；
2. `evaluate` 跑 1 局：`eval-report.json` 含 `frames_reported > 0`、`stages_reached ≥ 0`、`wall_seconds`、截图路径；
3. `play` 在 3 分钟内能完成至少 1 局，并输出 JSON；
4. 三个命令都不要求人做任何菜单操作；
5. 若模型明显弱于随机策略，报告必须如实反映（不得用训练奖励冒充游戏表现）。

## 7. 禁止事项

- 禁止在评估中使用训练奖励作为唯一指标（必须有真实游戏帧数）；
- 禁止让 `play` 依赖 `pytest`/数据集/探针；
- 禁止在 `ppo` 里改动 `env` 的观测语义；
- 禁止把 `kl_coef` 默认设 0（那就失去本版的核心设计）。

## 8. 出错行为

| 情况 | 期望 |
| --- | --- |
| 检查点不存在/维度不符 | 加载时抛 `FileNotFoundError`/`ValueError` |
| 游戏启动失败 | `env.reset` 抛 `RuntimeError`（含对话框文本），CLI 退出码非 0 |
| 评估中途游戏崩溃 | 该局标记 `terminated` 并继续下一局；汇总里记录 |
| 达到时间预算 | `play` 正常收尾并写报告 |
