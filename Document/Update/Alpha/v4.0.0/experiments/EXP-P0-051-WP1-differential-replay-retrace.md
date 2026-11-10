# EXP-P0-051：WP1 退路 C — 差分回放重放（harness ON vs 原样 OFF）

日期：2026-11-08。Provider=DeepSeek，模型=DeepSeek v4.1 flash。

## 结论

**本轮 3/3 份真实回放通过「差分验证」：同一 replay 在 harness ON 构建与原样 portable OFF 构建下，以相同参数、同一纯记录模式播放到自然结束，25 字段追踪行文件逐字节一致（295,040 行/侧，SHA-256 两侧相同，全部 exit 0）。**

这是 `10-P0-训练场.md` WP1 第 4 条的**退路 C（差分参考）**：只证明「harness 构建没有改变原样构建的回放行为」，**不等同于**与零售版逐位一致。WP1/G0.1 的「108 份回放与零售轨迹逐位一致」仍未达成；门报告必须写明证明力下降，不得据此翻转 G0.1。

附带的一致性检查：官方 fixture `easy-0-b`（`th8_ud2cdc.rpy`）本次产生 **107,296** 行，与 `config/replay-fixtures.json` 的 `retailTrace.calculationFrames=107296` 相同。这只是行数一致性，**不是** retail trace SHA 比对（retail trace 是外部观察器 22 字段 JSONL，与本记录器 25 字段二进制行格式不同，不能互比）。

## 依据与范围

- 零售参考轨迹三次尝试 0 帧失败：`EXP-P0-014`、`EXP-P0-037`、`EXP-P0-045`。本轮不重试零售采集。
- 退路 C 依据：`10-P0-训练场.md` WP1 第 4 条——用原样构建做差分参考，并在门报告中说明证明力下降。
- 本轮只读使用游戏数据 `/root/v400-data/game`（`th08.dat` / `thbgm.dat` 以符号链接只读挂入每次运行的临时 runtime）与真实 `.rpy`。未改零售 EXE、未改游戏逻辑、未删除/提交文件。

## 纯记录模式实现（仅研究用，默认关闭）

唯一改动文件：`third_party/th08-rl/src/modern/linux/replay_trace.cpp`（+47/−15 行；无需改头文件）。上游 `Initialize()` 在缺 `TH08_REPLAY_TRACE_REFERENCE` 时直接禁用、且逐行 memcmp 无法自举，因此新增：

- 新增 `enabled` / `recordOnly` 状态；`Finish`、`ReplayTraceRecord`、`ReplayTraceFinish`、`ReplayTraceShouldExit` 的守卫由 `trace.reference` 改为 `trace.enabled`。
- 仅当 `TH08_REPLAY_TRACE_REFERENCE` 缺失**且** `TH08_REPLAY_TRACE_RECORD_ONLY` 恰好为 `"1"` 时进入纯记录模式：只打开 rows/completion 输出，不打开参考文件、不做 memcmp；进度日志改印 `recorded calculations`。
- `complete.json` 结构与字段完全沿用原实现（`rows`/`ended`/`naturalReplayExit`/`errors`/`stages`/`stageFrames`/`endScores`/`rngOrigins`）。
- 已有 reference 时行为不变；未设置该环境变量或设为 `0` 时仍是上游行为（无 reference 即禁用）。
- 未改任何游戏逻辑；记录器唯一的状态写入是上游原有的 `g_GameManager.replayMode = 2`（fast-forward），两侧构建完全一致。

默认行为不变的真实探针（OFF 构建，`Th8_ud1858`，25s/15s 有界）：

| 探针 | 环境 | 退出码 | rows.bin | complete.json | record-only 标记 |
|---|---|---|---|---|---|
| A | 无 `RECORD_ONLY`、无 reference | 124（运行至超时） | 不存在 | 不存在 | 0 |
| B | `RECORD_ONLY=0`、无 reference | 124（运行至超时） | 不存在 | 不存在 | 0 |

## 构建（只用独立目录）

共享目录 `build/rl` 与 `build/portable-linux-x86_64` **未用于本轮最终证据**；两个变体各自新建目录，源码相同、Debug/Ninja、编译器相同，缓存差异只有下列两项：

| 缓存项 | ON（harness） | OFF（原样 portable） |
|---|---|---|
| `TH08_RL_HARNESS` | `ON` | `OFF` |
| `TH08_ENABLE_REPLAY_TRACE` | `OFF`（harness 路径强制编译记录器） | `ON` |

```bash
cd /root/Touhou_RL/third_party/th08-rl
cmake -S . -B build/rl_retrace -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTH08_RL_HARNESS=ON
cmake --build build/rl_retrace --parallel 8
cmake -S . -B build/portable-retrace -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTH08_ENABLE_REPLAY_TRACE=ON
cmake --build build/portable-retrace --parallel 8
```

配置/构建退出码：`ON_CONFIGURE_EXIT=0`、`ON_BUILD_EXIT=0`、`OFF_CONFIGURE_EXIT=0`、`OFF_BUILD_EXIT=0`。

构建身份：

| 变体 | 可执行文件 | SHA-256 | 字节 | `ReplayTraceRecord` 符号 |
|---|---|---|---|---|
| ON | `build/rl_retrace/th08-modern` | `c2ce960dff8cf6d5eee4062c7c88cbad3c00c42dc4ed9c32d3af240819c6f79d` | 4,807,664 | 存在 |
| OFF | `build/portable-retrace/th08-modern` | `f920e14004fb065955d9baae39cddfa575c78bcf90482f862b700dcf153bd2f8` | 4,629,432 | 存在 |

源树身份：`third_party/th08-rl` HEAD `f563df58c44037b44208c393a86540b434b0c83b`，tracked 工作树 dirty（含既有 harness 改动）；记录器源码 SHA-256 `78cd4fe3834027cdd61014cff92f2ff998c33c490345cc6f93b4c27d92989398`。

## 运行参数

每份回放、每个构建各启动一个独立进程，runtime 目录每次新建：

```text
xvfb-run -a -s "-screen 0 1024x768x24" timeout --signal=TERM --kill-after=5s 180s <exe> --data-dir <runtime>
```

```text
SDL_AUDIODRIVER=dummy LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=0 LP_NO_RAST=true
TH08_AUTOPLAY_REPLAY=./replay/audit.rpy  TH08_AUTOPLAY_STAGE=<该 replay 首个录制面>
TH08_RENDER_AUDIT_FAST=1  TH08_RENDER_AUDIT=0
TH08_REPLAY_TRACE_RECORD_ONLY=1
TH08_REPLAY_TRACE_ROWS=<runtime>/rows.bin
TH08_REPLAY_TRACE_COMPLETION=<case>/<side>/complete.json
（不设置 TH08_REPLAY_TRACE_REFERENCE / TH08_REPLAY_TRACE_MAXIMUM）
```

`th08.cfg` 为上游 suite 的 muted 配置（60 字节，SHA-256 `6401f8aa0a8ad7c0a2b4fc08174d7b6b9c2d8f58c110ee8329b397fdb7b18a52`，与 `third_party/th08/scripts/capture-portable-replay.py` 的 `muted_config()` 逐字节相同）。单份超时上限 180s，两侧参数完全相同。

## 结果（ON vs OFF，逐字节）

| # | 回放 | 难度/机体 | 起始面 | 输入记录 | ON 行数 | OFF 行数 | ON rows SHA-256 | OFF rows SHA-256 | 一致 | ON exit | OFF exit | ON 秒 | OFF 秒 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `th8_ud2cdc.rpy`（官方 fixture `easy-0-b`） | Easy / Border Team | 0 | 106,466 | 107,296 | 107,296 | `bbd82ee6fbacea87716f676b47fc07663268583b8f3764432168cea63b1466e5` | `bbd82ee6fbacea87716f676b47fc07663268583b8f3764432168cea63b1466e5` | 是 | 0 | 0 | 55.5 | 54.8 |
| 2 | `Th8_ud243c.rpy` | Lunatic / Youmu | 0 | 137,089 | 138,022 | 138,022 | `0b8402c4f64ba2554f01d9d346d2220b6f5338323f74112895d623cf18e6ca8b` | `0b8402c4f64ba2554f01d9d346d2220b6f5338323f74112895d623cf18e6ca8b` | 是 | 0 | 0 | 94.4 | 100.7 |
| 3 | `Th8_ud1858.rpy` | Extra / Marisa | 8 | 49,229 | 49,722 | 49,722 | `a5f0c2e3e636c6ecae7e96d456efd06f03e47a42f29096e712f59f0be0524119` | `a5f0c2e3e636c6ecae7e96d456efd06f03e47a42f29096e712f59f0be0524119` | 是 | 0 | 0 | 45.2 | 40.4 |

- 合计 **295,040 行/侧**，首个差异：**无**（逐字节相同）。
- 每份的 `complete.json` 两侧也逐字节相同（SHA-256 分别为 `2c04f551a250956862d26752940f348536eb83b41facd569369fa1948b9c03af`、`53006a99c0a31ebea950b6960fafdee7702244fde767ba75f31fe60508cfc7fc`、`44cdba51608eee450a349c221958ffacfd728e8cbd02dc50ec19f15a6c08f148`），其中 `stageFrames`/`endScores`/`rngOrigins` 一致；`ended=true`、`naturalReplayExit=true`、`errors=[]`。`easy-0-b` 的 6 面行数与分数（如 `endScores` 合计 100,043,139）两侧相同。
- 3 份 replay 的 SHA-256：`79739491e1cb1d735ef9a14814a79dfe8e2c4872d5f2bf0174c79f40ada36989`（easy-0-b，与官方 manifest 一致）、`75a5cf92dd04a6689c178118a57f6f9884d6592b03d68471c823e9b2afe9f8c0`、`f522dcfc1e8b1c08cfb055d0e8ad41aab6958c38d6de966dcddd23ab69668421`。
- 单一工具退出码 **0**（`identical 3/3 replays; 295040 rows per side`）。

## 证明力与边界（必须读）

1. **这是差分验证（ON vs 原样 OFF），不是与零售版逐位一致。** 零售参考轨迹仍未取得（0/108），退路 C 的结论不能替代 G0.1 的零售位一致要求；门报告必须注明证明力下降。
2. 两侧共享同一份记录器源码与同一读点/字段定义；若记录器本身对状态读取有偏差（读点、字段语义、边界），差分无法发现。差分只能发现「构建差异导致的回放行为差异」。
3. 两侧共享同一 replay、同一 `th08.cfg`、同一环境变量、同一 Debug 构建与同一 timeout；差异只有 harness 编译开关。
4. 只覆盖 **3/108** 份回放、3 种难度/面组合（Easy 6 面、Lunatic 6 面、Extra 单面），合计 295,040 个计算帧；不代表全部 108 份，也不代表其他机体/路线。
5. 25 字段只含分数、资源、浮点位置位模式、RNG、输入、计时器等内部状态；不含渲染、音频、内存布局或其他时序观测。
6. 行数对齐 manifest 的 `calculationFrames` 仅说明计算帧计数一致；记录器输出格式与 retail trace（22 字段 JSONL + 外部 SHA）不同，两者 SHA 不可互比。
7. 本模式会给 `g_GameManager.replayMode` 置 2（上游原有行为，两侧相同）；它只影响快进/渲染节奏，不改变逐计算帧的模拟结果——这一点由两侧行一致所支持，但未用零售轨迹独立证明。
8. 未验证 paced（非快进）模式；本轮全部使用 fast clock（`TH08_RENDER_AUDIT_FAST=1` + `LP_NO_RAST=true`）。

## 构建目录处置记录（build/rl 事故）

为遵守并行验收对 `build/rl/th08-modern` 的身份要求，如实记录并已恢复：

| 时点 | 路径 | SHA-256 |
|---|---|---|
| 本轮开始时（原始） | `build/rl/th08-modern` | `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb` |
| 首次重建（纯记录补丁） | 同上 | `2514870233aa3948a80da55c4fdd45f75c22cf67299119dab8f425ad023d3b7c` |
| 进度文案补丁后重建 | 同上 | `a6b146ff73ec7ba149ed3dc3f65681f76ad55e93d8405e9bd969a2f134af1c90` |
| **还原补丁前源码并重建** | 同上 | **`591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`（与原始逐字节一致，`RL_RESTORE_EXIT=0`）** |

- `build/portable-replay/th08-modern` 也被重建过，其**本轮前 SHA 未记录**（不可核验）；现为 `724f7e9850915afd9914539bfb11fef04685c71b63e0369d47a77fc4a43d0bf6`（以补丁前源码重建）。
- 从本卡起，**不再使用 `build/rl` 与 `build/portable-replay` 作为证据构建**；最终证据只来自 `build/rl_retrace` 与 `build/portable-retrace`。
- 注意：当前源树 `replay_trace.cpp` 已重新含纯记录补丁（默认关闭）；`build/rl` 的 `591c521d…` 二进制对应补丁前源码。**若有人重建 `build/rl`，会带上该补丁，SHA 会再次变化**（功能默认不变）。若要长期冻结 `591c521d…`，需先将 `replay_trace.cpp` 还原为补丁前版本（备份在 `.tmp_v400_replay_retrace/replay_trace.cpp.before`），再重建。

## 证据与复现

- 机器报告：[`report.json`](../../../../../.tmp_v400_replay_retrace/run-20261109/report.json)（含构建身份、源树身份、每份 replay/每个构建的 exit code、耗时、`complete.json`、rows SHA 与比较结果）。
- 逐例原始证据：`.tmp_v400_replay_retrace/run-20261109/<case>/{on,off}/` 下的 `run.log`、`complete.json`、`runtime/rows.bin`、`result.json`、`command.json`（环境变量快照）。
- 工具：[`v400_replay_retrace.py`](../../../../../tools/rl/v400_replay_retrace.py)；单测：[`test_v400_replay_retrace.py`](../../../../../tests/test_v400_replay_retrace.py)。
- 默认行为探针：`.tmp_v400_replay_retrace/probe_default_a/`、`probe_default_b/` 及汇总 [`default_behavior.log`](../../../../../.tmp_v400_replay_retrace/default_behavior.log)（A/B 均：exit 124、无 rows、无 complete、无 record-only 标记）。
- 记录器补丁脚本与备份：`.tmp_v400_replay_retrace/patch_record_only.py`、`patch2_progress.py`、`replay_trace.cpp.before`（补丁前）、`replay_trace.cpp.patched`（补丁后）。
- 构建日志：`.tmp_v400_replay_retrace/{configure_on,configure_off,build_on_retrace,build_off_retrace,restore_rl,restore_pr}.log`。

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_replay_retrace_pytest -q tests/test_v400_replay_retrace.py
# 24 passed, TEST_EXIT=0
```

```bash
cd /mnt/d/WorkShop/Touhou_RL
python3 tools/rl/v400_replay_retrace.py \
  --on-build  /root/Touhou_RL/third_party/th08-rl/build/rl_retrace/th08-modern \
  --off-build /root/Touhou_RL/third_party/th08-rl/build/portable-retrace/th08-modern \
  --replay    /mnt/d/WorkShop/Touhou_RL/.tmp_v400_retail_fixtures/th8_ud2cdc.rpy \
  --replay    /mnt/d/WorkShop/Touhou_RL/th08_replay/Th8_ud243c.rpy \
  --replay    /mnt/d/WorkShop/Touhou_RL/th08_replay/Th8_ud1858.rpy \
  --output-dir /mnt/d/WorkShop/Touhou_RL/.tmp_v400_replay_retrace/run-20261109 \
  --timeout 180
# exit 0；输出 identical 3/3 replays; 295040 rows per side
# 复跑时 --output-dir 必须是尚不存在的新目录（工具 fail-closed，拒绝复用证据目录）
```

## 能否扩到 108

**能，工具与数据都已具备，只差运行时间与磁盘。**

- 官方 accepted corpus 就在本机：`third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/` 共 **108** 份；与 `config/replay-fixtures.json` 的 `sha256` 逐份核对 **108/108 全部匹配、0 缺失、0 不符**（另有 3 份 `retail-incomplete` 不属于 accepted）。
- 规模：108 份合计 **12,311,460** 个计算帧（平均 113,995、最小 44,106、最大 149,986）。
- 吞吐实测：单进程含启动/换面约 1,100–1,933 行/秒。按此估算，单侧约 **1.8–3.1 小时**，两侧合计约 **3.6–6.2 小时**；rows.bin 两侧合计约 **2.5 GB**（`/root` 当前剩余约 941 GB）。
- 180s 单份上限：最大 fixture 149,986 行约需 78–136s，**仍在上限内但余量不大**；建议 108 轮保持 180s 并对超时个案单独提高上限重跑，或把上限放宽到 300s。
- 20 核可用，建议 2–4 个并发 runner 分片，可把两侧墙钟时间压到约 1.5–3 小时；每份仍按「ON 与 OFF 相同参数、同一 runtime 模板」执行。

## 剩余阻塞

1. **WP1/G0.1 的零售位一致仍未达成**（0/108 零售参考轨迹）。retail capture 的 0 帧失败（`EXP-P0-014/037/045`）未解除；本卡只是退路 C 的差分证据，须在门报告中标注证明力下降。
2. 本卡只完成 **3/108** 的差分覆盖；扩到 108 属于可执行的后续（约 4–6 小时、约 2.5 GB），本轮未执行。
3. 纯记录模式与差分本身不能证明记录器读点对零售语义无偏差；只有取得 retail `rows.jsonl`+`complete.json` 后，才能用同一记录器的 `TH08_REPLAY_TRACE_REFERENCE` 模式做真正的零售位比对。
