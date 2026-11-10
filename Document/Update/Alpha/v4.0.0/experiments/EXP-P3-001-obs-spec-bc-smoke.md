# EXP-P3-001 观测规格冻结 + BC 冒烟 + 首次 S1N 评测（P3 起步切片 S9）

- 日期 / 切片：2026-11-09 / **P3-S9 仅此一片**（规格冻结、schema 契约、BC 冒烟、3 局 S1N）
- 依赖：[EXP-P2-003](<EXP-P2-003-reward-detector-g2.md>)（P2-WP4 分辨率对照）、13-P3 §2–§5、`obs-spec-v1.md`
- 结论先说：**观测规格 obs-v1 已冻结；dataset schema 与 fail-closed 校验落地；`checkpoint-v0` 是
  `random/still/dive` 行为克隆，不是教师标签、不是 P3 成绩、不是 G3 成绩**。3 局 S1N 全部
  `game_over`（0 过关），存活帧数低于 `random`、略低于 `still`。本卡不声称任何能力提升。

## 1. 范围与预注册

1. 本切片只做 S9：冻结 obs 规格（P2-WP5 交付物）、学生样本 schema（P3-WP1-a）、5 分钟内 BC 冒烟、
   3 局 S1N 冒烟、测试与实验卡。
2. **不做**：DAgger、教师提速、PSN、REAL-S1N、G3 评估、模型容量调优。
3. 所有 BC 数字标注为 **V0 行为克隆**（标签 = 采集策略自己的按键掩码），任何报告不得把它写成
   P3/G3 成绩；`checkpoint-v0` 的 `train`/`provenance` 里带 `V0_DISCLOSURE`。
4. 评测 cap 选 **19887**（`S1N` human_reference cap，与 `bench/scoreboard.md` 正式 dev 表同一把尺），
   不选 3000：3000 是早期 provisional cap，会改变帧上限类别，而 V0 实际 ~1,634 帧就结束，
   用 19887 不会多花时间（实测每局 ~22 s）。
5. 未改引擎、未改协议、未改依赖、未删除文件、未重建 build。

## 2. obs 规格冻结（obs-v1）

完整契约见 [`obs-spec-v1.md`](<../obs-spec-v1.md>)，实现唯一来源
`touhou_rl/v400/student/obs_spec.py`，摘要
`obs_spec_digest = 24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381`。

| 项 | 冻结值 |
|---|---|
| 场地观测 | `play64` 64×64 灰度 uint8，BT.601 整数亮度 `(77R+151G+28B)>>8`，384×448 场地的 6×7 块整数均值 |
| 帧堆叠 | 4 帧（`t-3..t`，最旧在前），episode 开头重复首帧补齐 |
| HUD | 包含；`hud` 56×160 绿色通道原生分辨率（score 48–72 / lives 80–96 / bombs 96–112，列 480–640），只取当前 1 帧 |
| 决策周期 | **2 个逻辑帧**（与 `gym.env.DECISION_FRAMES`、`teacher.config.DECISION_FRAMES` 一致） |
| 归一化 | `float32 / 255.0`，每平面独立，范围 [0,1]，不减均值/不除方差 |
| 动作 | `cand = focus*9+move` 18 类 × bomb 2 类，shoot 恒开；掩码由 `cand_to_mask` 唯一生成 |
| 动作持续时间 | 学生 1 个动作 = 1 个决策 = 2 逻辑帧；教师 `Plan.duration` 单位是决策槽，`duration=1` 才与学生一个动作等价 |
| 教师接口 | `π_teacher ∝ softmax(scores[18]/τ)`，`τ=1.0`（训练配置，可标定）；学生输出 `move_logits[18]+bomb_logits[2]`；主损失 `KL(π_t‖π_s)`，bomb 用 BCE；`verified_mask` 只作信任/诊断 |
| 排除输入 | 特权状态、动作历史（留给 obs-v2）、P2 奖励头输出 |

**"冻结"的含义**（写进 obs-spec-v1 §1）：改变观测含义（分辨率/通道/堆叠/周期/归一化/HUD/
动作语义/教师对应）必须新版本号 + 新文档 + 旧 checkpoint 显式不兼容；`load_checkpoint()` 用
`check_obs_spec()` 校验版本与摘要。训练超参数（τ、batch、lr、β 日程）不算观测改动。

## 3. 数据集 schema 契约与校验结果

`touhou_rl/v400/student/dataset.py` 定义并实现了：

- `bc-v0` 样本：`play (4,64,64) uint8` + `hud (56,160) uint8` + `move ∈ [0,18)` + `bomb ∈ {0,1}`；
  窗口固定为 `frames[i-3..i] -> mask[i+1]`（采集器先写 mask 再 STEP、后读像素，用同帧 mask
  会把动作的即时效果泄漏给模型）。
- `teacher` 样本：`scores[18] + verified_mask[18] + bomb + chosen_plan + margin + obs 引用`；
  兼容 `scores` 与教师子进程日志里的 `geo_score` 两种写法。
- Fail-closed：schema / label_version / plane 形状与大小 / 关卡级切分 / 空 split / mask 不可映射 /
  教师标签缺字段或非有限 / 教师标签无观测引用 / < 50,000 样本 → `DatasetError` / `DatasetInsufficient`。

真实数据校验（`/root/v400-data/p2det/out-main`，engine sha
`f172399f…6c264e`，label `p2-v2-1`）：

| 项 | train | val | test |
|---|---:|---:|---:|
| shards | 80 | 16 | 32 |
| frames | 192,000 | 38,400 | 69,600 |
| windows（stack=4） | 191,680 | 38,336 | — |
| 来源 | dive 115,200 / random 38,400 / still 38,400 | **random 38,400（全部）** | — |
| 动作掩码可映射 | 192,000/192,000 = 100% | 38,400/38,400 = 100% | — |

**教师标签 fail-closed（真实读数）**：`.tmp_v400_teacher_child_stage1.jsonl` 有
**1500 个 decision、0 个带观测引用、0 个 schema 非法**；`load_teacher_samples()` 抛出：

```text
teacher labels carry no observation reference: 1500 decisions across 1 file(s),
0 with an 'obs'/'play'/'shard' key; a pixel student cannot be trained from labels
whose frames were not stored.
```

即：**当前教师标签既不够量、也没有可对齐的观测**，P3 的教师数据集（WP1）尚不存在，
BC 只能用 300k 帧行为克隆做 V0 基线。

## 4. BC 冒烟（≤5 分钟）

| 项 | 值 |
|---|---|
| 模型 | `StudentNet`：play 3×Conv(32/64/64, s2)+GAP、hud 3×Conv(16/32/32, s2)+GAP、concat→Linear128→18/2 |
| 参数量 | **88,436**（13-P3 目标 2–6 M，V0 故意更小） |
| 设备 | RTX 4070 SUPER（WSL CUDA），训练时 `nvidia-smi` 采样 **2,034 MiB / 12,282 MiB、GPU 6%**（基线 1,296 MiB/19%，无其他训练任务） |
| 数据 | train 191,680 窗口（dive/random/still），batch 256，AdamW lr 2e-3 |
| 预算 | `--max-seconds 240`，实际 **239.33 s**（≤5 分钟） |
| 步数/样本 | **109 step / 27,904 样本**（≈0.15 epoch；瓶颈是随机窗口 I/O，实测 **116 样本/s**） |
| 训练 loss | 最后 50 步窗口均值 **0.7034** |
| checkpoint | `.tmp_v400_p3/checkpoint-v0.pt`，373,207 B，sha256 `67544353bc6185ed51bb3e6092b85882acdfbf2cc71f2ff62cdd03cceb3e416b` |

**准确率（18 类 move_focus）**：

| 评测集 | n | top-1 | top-3 | bomb acc |
|---|---:|---:|---:|---:|
| **val**（stage 5，全部 `random`） | 12,000 | **7.01%** | **22.08%** | 99.63% |
| train 抽样（混合） | 12,000 | 79.86% | 84.18% | 99.92% |
| ├ `still` | 2,421 | 100.00% | 100.00% | 100% |
| ├ `dive` | 7,201 | 97.10% | 99.51% | 100% |
| └ `random` | 2,378 | 7.15% | 21.66% | 99.58% |

解读（必须一起读）：

1. **val 是 random-only，且 random 是无记忆均匀策略**：18 类理论 top-1 上限 ≈ 5.56%、
   top-3 ≈ 16.67%。V0 的 7.01%/22.08% 只略高于随机猜测，说明"从像素预测 random 策略的
   下一个按键"本身几乎不可学，不是模型没训好。
2. `still`（恒 SHOOT）与 `dive`（SHOOT+UP）是确定性行为，V0 能拟合到 97–100%；
   这说明观测/标签/训练通路是通的，但学到的是**采集策略的常量行为**，不是飞行能力。
3. bomb acc ≈ 99.6–100% 只是类别极不平衡（300k 帧里 bomb 事件 114 帧，采集策略默认不 bomb），
   **不能**当成 bomb 决策能力。

## 5. 首次 S1N 评测（3 局，seed 1000–1002）

命令：`xvfb-run` + `env` role（唯一会渲染并读回像素的 role）+ pixel readback，策略输入只有
`play64 4 帧 + hud`，不读特权状态；`privilege=deploy_pixel`。引擎用
`build/rl/th08-modern`（sha `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`，
与 scoreboard 同一引擎）；cap 19887（human_reference），终局规则与 `v400_bench_run.py` 一致
（`exit=1` 或 death 增加且 `lives<=0`）。

| seed | frames | deaths | score | bombs_used | end | wall s |
|---:|---:|---:|---:|---:|---|---:|
| 1000 | 1,646 | 3.0 | 66,221 | 0.0 | game_over | 22.0 |
| 1001 | 1,610 | 3.0 | 58,712 | 0.0 | game_over | 21.5 |
| 1002 | 1,646 | 3.0 | 64,036 | 0.0 | game_over | 22.3 |
| **均值** | **1,634.0** | **3.0** | **62,989.7** | 0.0 | game_over ×3 | 21.9 |

3/3 局都是 `game_over`（`lives_start=2`，死亡数饱和于 3），**0 局过关**；进程退出码 **0**
（每局记录都通过 schema 校验，无 `error`）。

### 5.1 与 random / still 对照（同 3 个种子、同 cap）

参照行来自 `bench/scoreboard.md`（`cap_source=human_reference`，max_frames 19887）：

| 策略 | seed 1000 frames/score | seed 1001 frames/score | seed 1002 frames/score | 3 局均值 frames / score |
|---|---|---|---:|---:|
| random | 2,586 / 41,193 | 1,790 / 26,031 | 2,004 / 34,615 | 2,126.7 / 33,946.3 |
| still | 1,748 / 59,953 | 1,692 / 63,693 | 1,748 / 63,762 | 1,729.3 / 62,469.3 |
| **V0 学生** | 1,646 / 66,221 | 1,610 / 58,712 | 1,646 / 64,036 | **1,634.0 / 62,989.7** |

- 与 `random` 比：V0 存活帧数 **少 23.2%**（1,634 vs 2,127），score 高 85.6%——但 score 高只因为
  它一直开火，`random` 也开火，差别在站位（V0 学到的 dive 行为贴顶、点杀伤多、死得更快）。
- 与 `still` 比：存活帧数 **少 5.5%**，score 高 0.8%：**与"站桩输出"同量级，没有任何证据表明更好**。
- 样本量：3 局 vs scoreboard 的 20 局 dev 表（random 39,331 / still 64,089 均值）。
  **随机/静止基线是不同局数、不同统计口径，只能作量级参考，不能直接比较或做显著性推断。**
- 死亡数在两种口径下都饱和（lives_start+1），**不能区分策略**；`frames` 与 `score` 才是区分量。

## 6. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3/pytest_final2 -q tests/test_v400_student.py
```

- 结果：**19 passed in 1.27 s**，退出码 **0**。
- 覆盖：obs 常量/摘要/版本闸门、`encode_planes` 与 P2 采集器**逐字节等价**、归一化与补帧、
  dataset 正常加载与 4 类 fail-closed（label_version/plane 大小/缺 hud/不可映射 mask/跨 split 泄漏）、
  教师标签 scan 与 3 类 fail-closed（无观测引用/样本不足/合法 geo_score 别名）、
  模型前向形状与参数量、checkpoint 往返与摘要篡改拒绝、评测记录 schema 正反例、
  动作空间与教师 18 候选一致。
- `python -m compileall -q touhou_rl\v400\student tools\rl\v400_student_bc.py tools\rl\v400_student_eval.py` → 退出码 0。

## 7. 命令与退出码（可复现）

```bash
# WSL：BC 冒烟（≤5 分钟，GPU）
cd /mnt/d/WorkShop/Touhou_RL && export PYTHONPATH=/mnt/d/WorkShop/Touhou_RL
/root/v400-venv/bin/python tools/rl/v400_student_bc.py \
  --index /root/v400-data/p2det/out-main/index.json \
  --split /root/v400-data/p2det/split.json \
  --checkpoint /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3/checkpoint-v0.pt \
  --out .tmp_v400_p3/bc-metrics.json --max-seconds 240 --batch-size 256 --eval-samples 12000
# → exit 0（109 step / 27,904 样本 / 239.33 s / val top1 7.01% / top3 22.08%）
```

```bash
# WSL：3 局 S1N（唯一引擎进程，并发 1；xvfb + env role + pixel readback）
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3/run_eval.sh
# → exit 0；episodes.jsonl + summary.json 写入 .tmp_v400_p3/eval/
```

```powershell
# Windows：测试
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3/pytest_final2 -q tests/test_v400_student.py
# → 19 passed, exit 0
```

产物：`.tmp_v400_p3/checkpoint-v0.pt`、`.tmp_v400_p3/bc-metrics.json`、
`.tmp_v400_p3/eval/episodes.jsonl`、`.tmp_v400_p3/eval/summary.json`。

## 8. 距离 G3 还差什么（按证据排序）

1. **教师数据集 D0 不存在**：13-P3 WP1 要 ~200 万帧、带教师 `scores[18]` 与观测的样本；
   当前只有 stage 1 的 **1500 个 decision，且 0 个带观测引用**。这是 P3 的**首要阻塞**。
2. **教师太慢**：526 ms/decision、单局 13 min（前置事实）。按 13-P3 §4，每轮 50 万 decision
   需要 ~5.5 h/轮；必须先做"单层搜索 + 只在学生动作与教师首选不同/危险余量小的状态做完整搜索"
   或 P1 快速教师模式，并实测标注速率，再定 M/K。
3. **DAgger 循环未建**：β 日程、16 环境并行、影子进程查询、按轮聚合（最近 R 轮 + D0 20%）、
   按死亡点前 5 秒的起点课程，全部未实现；本切片只完成了数据契约与评测通路。
4. **学生容量与损失未到位**：V0 只有 88k 参数、无 GRU、无感知辅助头、无价值头，损失是硬标签 CE；
   P3 目标是 2–6 M、`KL(软标签)+BCE(bomb)+λ_aux+λ_v·Huber`。
5. **S1N 过关率 0/3**：G3 要求确认种子 100 局点估计 ≥50%、95% 下界 ≥35%。V0 连"活到人类
   关卡长度"都差一个数量级：1,634 帧 vs 人类 Stage1 **13,258** 帧（12.3%）；对照 random 2,127、
   still 1,729，V0 没有优势。
6. **PSN 无法评测**：现有 runner 在 suite 多关卡时直接拒绝（`v400_bench_run.py` 的 stage-boundary
   限制），G3 要求的 8 面 PSN 数据一条都没有。
7. **REAL-S1N 部署通路未建**：P3-WP5 的 Windows 真机适配器、20 局真实游戏评测都不存在；
   动作延迟（真机 1 帧）与 obs-v1 的 play64/hud 在真机像素上的对齐也未验证。
8. **感知辅助标签不可靠**：P2 的 death 检测器 G2 未通过（64×64 只在 dive 域 33.3% 事件召回，
   still 0/48），把它的"危险图/死亡前兆"当 P3 辅助头目标会引入策略域偏差。
9. **obs-v1 还缺 13-P3 §2.1 要求的动作历史**（8 个决策 one-hot）：已明确排除、留给 obs-v2；
   在补齐前，学生对"自己刚做了什么"无记忆（不过 4 帧像素 + 价值/辅助头是否足够，需要实验）。
10. **bomb 能力为零**：采集数据里 bomb 事件 114/300k 帧，V0 在 3 局里 `bombs_used=0`；
    G3 虽不直接考 bomb，但"会死的动作概率 <1e-3"与 bomb 决策都需要教师软标签才能学。
11. **零特权门未跑**：本卡评测的 `policy_input` 只有像素、`privilege=deploy_pixel`，但 G3 要求
    bench 的零特权单元测试通过并留 `privilege=deploy_pixel` 日志；该测试属 P3-WP7，尚未执行。

## 9. 变更与产物

新增（均在任务白名单内）：

- `Document/Update/Alpha/v4.0.0/obs-spec-v1.md`（obs-v1 冻结文档）
- `touhou_rl/v400/student/{__init__,obs_spec,dataset,model}.py`
- `tools/rl/v400_student_bc.py`、`tools/rl/v400_student_eval.py`
- `tests/test_v400_student.py`（19 用例）
- `.tmp_v400_p3/{checkpoint-v0.pt,bc-metrics.json,eval/*,run_bc.sh,run_eval.sh,probe_env.sh}`

未改引擎、未改协议、未改依赖、未重建 build、未删除文件；Windows 零售 trace 批跑未受影响，
WSL 引擎在评测期间并发 1（无第二个引擎进程）。

## 10. 限制（不要过度解读）

- n=3 局、单 checkpoint、单 seed 训练：没有方差估计，不能推断策略强弱。
- val 只覆盖 `random`（stage 5 全 random），所以 val 准确率天然接近随机猜测；这不是"模型失败"，
  而是"random 策略不可从像素预测"。
- `frames` 受死亡数饱和影响：3 局都是 game_over，死亡数不区分策略。
- 本卡所有数字都是**真实引擎/真实数据**的读数；没有任何一项被当作 P3 或 G3 成绩。
