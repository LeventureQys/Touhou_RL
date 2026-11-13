# WP01.1b — 真实引擎一致性切片（3 seeds × 3 resets × 1,000 frames）

- 状态：**完成**（真实引擎已运行；未训练、未采集数据集、未修改 v4.0.0）
- 日期：2026-11-12（`generated_utc` 以 JSON 内时间戳为准）
- 范围：`tools/rl/v410_consistency.py`、`touhou_rl/v410/runtime.py`、`tests/test_v410_consistency.py`、本文件、[`consistency.json`](consistency.json)
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
- 引擎 SHA256：`691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`（4,836,008 bytes，WSL 内 `hashlib` 分块读取）
- 数据目录：`/root/v400-data/game`；WSL2 Ubuntu，Python 3.10.12，`xvfb-run -a`

## 1. 结论

| 项 | 结果 |
| --- | --- |
| 主切片 `status` | **pass** |
| 规模 | 3 seeds（1/2/3）× 3 resets × 1,000 frames = **9,000 帧** |
| 同 seed 重开逐帧一致 | **是**，所有 seed 的 3 次重开 `mismatch_count = 0`、`first_mismatch_frame = null` |
| `RESET` hash 一致 | 是（每 seed 三次重开的 `reset_hash` 完全相同） |
| `stale` / `errors` | **0 / 0** |
| 跨 seed | 三对均在第 0 帧分歧（**预期行为，不作为失败**） |
| 像素检查（`role=env`） | pass：16 个采样点、2 次重开的 play/hud 字节 SHA 完全一致 |
| `role=env` 与 `role=shadow` 前缀一致 | 是：seed 1 的 200 帧序列 SHA 两者相同（`69c08f05…`） |
| 跨进程复跑主切片 | 三个 seed 的 `sequence_sha256` 与首次运行逐字节相同 |
| 稳定性（`--minutes 2`） | pass：69 resets / 68,608 帧 / errors=0 / stale=0 |
| 稳定性（`--minutes 30`） | **pass**：1,039 resets / **1,038,399 帧** / errors=0 / stale=0 / wall=1800.003 s（见 §7） |

同一 seed 的逐帧序列 SHA（1,000 帧）：

| seed | resets | frames | sequence_sha256 | mismatch |
| ---: | ---: | ---: | --- | ---: |
| 1 | 3 | 3,000 | `00227489501db23342dab80d7160e4dfbc4681f4a6df6f82d48a64f6f9f9e339` | 0 |
| 2 | 3 | 3,000 | `92d643b65e541bf19566893eb08504a71215469feb27e24bfcd51032a1b71f45` | 0 |
| 3 | 3 | 3,000 | `388d0f4aa4ec472ff3c2d3e14fb3aa1fc3e6431c6373d9b32e3406a515949811` | 0 |

每 seed 三次重开的 `sequence_sha256` 与 `reset_hash` 逐项相同，这就是“同 seed 重开逐帧相同”的直接证据。`consistency.json` 内每个 reset 还记录了 `hash_count=1000`、`wall_seconds`、`fps`、`stale`、`errors`、`budget_cut=false`。

## 2. 命令与退出码（全部实际执行）

### 2.1 smoke（WSL，20 帧，shadow）

```bash
cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python tools/rl/v410_consistency.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --seeds 1 --resets 1 --frames 20 --role shadow --out /tmp/v410_cons_smoke.json
```

- 退出码 **0**；`status=pass`；`wall_seconds=1.763`；`totals={frames:20,resets:1,errors:0,stale:0}`
- 序列 SHA `51eb102889e063bddb2dfab472f0ac674c561cd49b336bb0d791be0cfd259a51` 在两次独立运行中完全相同（跨进程可复现）

### 2.2 主切片（WSL，3 seeds × 3 resets × 1,000 帧，shadow）

```bash
cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python tools/rl/v410_consistency.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --seeds 1,2,3 --resets 3 --frames 1000 --role shadow \
  --out Document/Update/Alpha/v4.1.0/evidence/WP01/consistency.json
```

- 退出码 **0**；`status=pass`；`wall_seconds=15.832`（含 9 次引擎启动）
- 逐帧吞吐 1,767–1,865 fps（shadow 空设备，不含进程启动时间）
- `consistency.json` 由该命令生成，非手写；`schema="v410-consistency/1"`

### 2.3 渲染角色 + 像素检查（WSL，env，2 resets × 200 帧）

```bash
cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python tools/rl/v410_consistency.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --seeds 1 --resets 2 --frames 200 --role env --pixel-check --out /tmp/v410_cons_pixels.json
```

- 退出码 **0**；`status=pass`；`wall_seconds=15.015`；1,895–1,982 ms / 200 帧（约 101–106 fps，含 llvmpipe 渲染）
- 每 25 帧取一次像素，经 `encode_planes_v3` 得到 play/hud 平面并分别计算字节 SHA
- 每个 reset 采样 8 次（共 16 次）；`pixel_sequence_sha256 = 2fe48f0d6dab3965302c21dc2e1a68ae2c57fe2d0765f0407db43a7112e7ecc1`，两次重开完全相同，`pixel_identical=true`
- 帧 hash 序列 SHA `69c08f050ccee3f3b7d1d8cfc0f25a5950835d712ac332fd17d5ae63df106623`，与 2.4 的 shadow 运行一致

### 2.4 角色一致性对照（WSL，shadow，200 帧）

```bash
cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python tools/rl/v410_consistency.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --seeds 1 --resets 1 --frames 200 --role shadow --out /tmp/v410_cons_shadow200.json
```

- 退出码 **0**；`status=pass`；`sequence_sha256 = 69c08f05…`，与 2.3 的 `role=env` 完全相同 → 渲染与否不改变 25 字段物理轨迹（前 200 帧逐帧 hash 序列的 SHA 相同）

### 2.5 稳定性烟测（WSL，`--minutes 2`，shadow）

```bash
cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python tools/rl/v410_consistency.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game --minutes 2 --role shadow \
  --out /tmp/v410_cons_stability_2min.json --quiet
```

- 退出码 **0**；`status=pass`；`wall_seconds=120.003`
- 69 resets / 68,608 帧 / `errors=0` / `stale=0` / `seeds_identical=true`
- 各 seed 的 `sequence_sha256` 与 §1 主切片完全一致（seed 2 末次迭代被墙钟截断为 608 帧，`budget_cut=true`，不计长度差异）
- 逐 seed：seed 1 → 24 resets / 24,000 帧；seed 2 → 24 resets / 23,608 帧；seed 3 → 21 resets / 21,000 帧

### 2.6 主切片复跑（跨进程可复现性）

```bash
cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python tools/rl/v410_consistency.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --seeds 1,2,3 --resets 3 --frames 1000 --role shadow --quiet \
  --out /tmp/v410_cons_main_rerun.json
```

- 退出码 **0**；`status=pass`；9,000 帧 / 9 resets / errors=0 / stale=0
- 三个 seed 的 `sequence_sha256` 与 §2.2 首次运行**逐字节相同**（`002274…`、`92d643…`、`388d0f…`），说明结果不是单次进程内的偶然一致，而是可跨进程复现

### 2.7 `cycle` 输入模式 smoke（WSL，20 帧，shadow）

```bash
cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python tools/rl/v410_consistency.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --seeds 1 --resets 1 --frames 20 --role shadow --pattern cycle \
  --out /tmp/v410_cons_cycle.json
```

- 退出码 **0**；`status=pass`；`sequence_sha256 = d08673b57bfb84fa6270fea7b00c26b6dbe5557b4c035b6c93a9f1c16f1e2a09`
- 仅证明 `cycle`（`cand_to_mask((frame//2)%18,0)`）在真实引擎上可用且帧计数/哈希校验通过；**未**做 3 seeds × 3 resets × 1,000 帧的 cycle 长切片

### 2.8 验收命令（Windows）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp011b_pytest -q tests/test_v410_consistency.py
python -m compileall -q tools/rl/v410_consistency.py touhou_rl/v410
```

- pytest：**20 passed in 0.24s**，退出码 **0**（`-p no:cacheprovider` + `--basetemp` 绕开 `.pytest_cache` ACL 问题）
- compileall：退出码 **0**（仅语法通过，不代表真实运行）
- 测试全部使用 fake env/client，不启动引擎：相同序列 pass、单帧差异 fail（`first_mismatch_frame=2`）、frame 计数跳变 fail（`stale=1`）、`advanced=2` fail、`hash=None` fail、reset 失败/引擎启动失败/journal 记录、reset hash 不一致 fail、像素序列不一致 fail、跨 seed 分歧不算失败、`--minutes` 墙钟模式、参数校验

## 3. 实现

### 3.1 `tools/rl/v410_consistency.py`（CLI）

- 参数：`--engine`、`--data-dir`、`--seeds 1,2,3`、`--resets 3`、`--frames 1000`、`--role shadow|env`、`--pattern idle|cycle`、`--out`、`--minutes`、`--pixel-check`、`--stage/--difficulty/--team`、`--quiet`
- `idle` = 掩码 `0`（不按键）；`cycle` = `cand_to_mask((frame//2) % 18, 0)`
- 每个 seed/reset 使用唯一 `shm_name`：`v410c-<pid>-<seed>-<reset>-<counter>`
- 每帧校验：`advanced == 1`；`hash` 非空；帧计数严格 `prev + advanced`（即 +1，抖动计 `stale`）；`game_over`/`exit` 作为终止事件记录后停止该次重开
- 退出码：`0` pass、`1` 一致性失败、`2` 参数/环境不可用（引擎或数据目录缺失、`--pixel-check` 未配 `role=env` 等）
- 可在 WSL 内直接运行，也可由 Windows `wsl.exe ... bash -lc` 调用

### 3.2 `touhou_rl/v410/runtime.py`（可 import）

- `run_consistency(...)`：注入点 `env_factory(engine, data_dir, shm_name, role)` 与 `start_factory(**StageStart kwargs)`，测试用 fake 即可跑完整流程
- 纯函数：`sequence_sha256`、`compare_sequence_pair`、`first_divergence`、`normalize_hash`、`action_mask`、`build_report`、`report_exit_code`、`write_report`
- `--minutes` 模式：按 seed/reset 计划循环，墙钟到点后停止；进行中的那次重开标记 `budget_cut=true`，比较时忽略长度差异但仍比较已有前缀
- 报告字段：`schema`、`status`、`mode`、`engine{path,sha256,size_bytes}`、`data_dir`、`host`、`config`、`seeds[]`（每 seed 的 `resets[]` 含 `frames/hash_count/wall_seconds/fps/sequence_sha256/reset_hash/mismatch/first_mismatch_frame/stale/errors/game_over_frame/exit_frame/budget_cut` 及可选的 `pixel_sequence_sha`）、`cross_seed[]`、`pixel_check`、`totals`、`stability`（墙钟模式）、`failures`

### 3.3 判定规则

`status=pass` 当且仅当：
1. 每个 seed 的所有重开与第 0 次重开逐帧相同（`mismatch_count=0`）；
2. 每 seed 的 `reset_hash` 相同；
3. `totals.errors=0` 且 `totals.stale=0`；
4. 启用 `--pixel-check` 时同 seed 的像素序列 SHA 也相同。

不同 seed 之间只记录 `first_divergence_frame`，不参与 pass/fail。

## 4. 变更文件（严格白名单）

| 文件 | 说明 |
| --- | --- |
| `tools/rl/v410_consistency.py` | 新增：CLI |
| `touhou_rl/v410/runtime.py` | 新增：可 import 的驱动/比较/报告实现 |
| `tests/test_v410_consistency.py` | 新增：20 个 fake 测试（不启动引擎） |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/CONSISTENCY.md` | 新增：本文件 |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/consistency.json` | 由 §2.2 命令真实生成 |

未修改：`touhou_rl/v400/`、`v380/`、`v390/`、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、`touhou_rl/v410/protocol.py`、`adapters.py`、`parity.py`、`evidence.py`、`tools/rl/v410.py`、`state.json`、`journal.jsonl`、`budget.jsonl`、`assets.json`。未 commit / 未 push，未删除或覆盖旧文件。

## 5. 未完成项与边界

- 本次切片只覆盖 `stage=0 / difficulty=normal / team=0 / resources=standard`、`idle` 输入、1,000 帧上限；**不代表**全部关卡、全部输入模式或长时训练确定性。
- `idle` 不按键，因此 1,000 帧内没有触发 `game_over`，也没有覆盖“重开中途死亡/续关”路径；`cycle` 只做了真实引擎 20 帧 smoke（§2.7）与单测，**未**跑 1,000 帧长切片。
- 像素检查只做 `seeds=1, resets=2, frames=200`（16 个采样点），未扩展到主切片规模。
- 主切片为 `role=shadow`；`role=env` 只做了 200 帧前缀对照（§2.4），未做 1,000 帧 × 3 resets 的渲染角色全长运行。
- 未验证引擎二进制之外的构建产物（如 `.so`/资源）逐字节一致；SHA 只锁定 `th08-modern` 本体。
- `--minutes 30` 已完成并通过（§7）；其中仅 1 次重开因墙钟到点被截断，属预期行为。

## 6. 下一建议

1. 把 `cycle` 输入模式跑 3 seeds × 3 resets × 1,000 帧，覆盖“按键 + 射击 + 死亡”路径，验证输入参与下的同 seed 确定性。
2. 在 `role=env` 下补 3 × 3 × 1,000 像素切片，确认渲染角色的长时像素一致性（当前只有 200 帧前缀）。
3. 将 `--minutes 30` 纳入 CI/夜间任务，并把 `consistency.json` 的 `sequence_sha256` 与 `reset_hash` 作为回归基线（同引擎 SHA 下应完全复现）。
4. 若后续换引擎构建，先跑本工具对比 `reset_hash` 与帧 0–999 序列，再决定是否重建数据集基线。

## 7. 30 分钟稳定性运行（已完成）

```bash
cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python tools/rl/v410_consistency.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game --minutes 30 --role shadow \
  --out /tmp/v410_cons_stability.json --quiet
```

- 退出码 **0**；`status=pass`；`mode=stability`；`wall_seconds=1800.003`（按墙钟停止，非 frame 计数）
- **1,039 resets / 1,038,399 帧 / errors=0 / stale=0 / failures=0**
- 逐 seed：
  - seed 1：348 resets / 348,000 帧 / mismatch=0 / `sequence_sha256=002274…`
  - seed 2：346 resets / 345,399 帧 / mismatch=0 / `sequence_sha256=92d643…`（末次 iteration 345 的 reset 0 被墙钟截断为 399/1000 帧，`budget_cut=true`；比较时忽略长度差，已有前缀逐帧一致）
  - seed 3：345 resets / 345,000 帧 / mismatch=0 / `sequence_sha256=388d0f…`
- 三个 seed 的 `sequence_sha256` 与 `reset_hash`（`7f04c736…`、`ebe40d6b…`、`9c14f837…`）同 §2.2 主切片完全一致 → 30 分钟内 1,039 次重开无一次漂移
- 逐次重开 fps 909.99–1928.34（低值出现在并行跑其他切片、CPU 争用期间；一致性判定不依赖 fps）
- 1,039 次重开中 1,038 次完整跑满 1,000 帧，仅 1 次（seed 2 末次）因墙钟到点截断
- 结果文件：`/tmp/v410_cons_stability.json`（由该命令生成，未复制进仓库证据目录；仓库证据仍以 `consistency.json` 为主切片基线）
