# P3 接入方案：把 `frame_proof=exact` 接进 `capture_dataset.py`

- 日期：2026-11-02。预注册：[p3-exact-capture-v1.md](<../preregistration/p3-exact-capture-v1.md>)。
- 本文是**方案**，不是已完成的采集。它写清改动点、CLI、清单字段、预检与失败语义，使实现可以按图施工、按图验收。
- 前置事实（全部已实测）：round26 给出解冻规则与 60/60 样张；`FenceFrameSampler(accept_current=True)` 把吞吐从 3.8 提到 45.2 张/s；`touhou_rl/frame_proof.py` 已提供可执行的逐帧 schema 与 16 项单测。
- round28 的约束：**F1/F3 的教师标签在真实会话里不存在**，所以第一版只做"无标签的精确帧数据门"；标签另立预注册（只可能来自 F2 类）。

## 1. 现状与缺口

`touhou_rl/capture_dataset.py` 现在是"回放自走 + 自由轮询"：

| 位置 | 现状 | 缺口 |
|---|---|---|
| `capture_replay()` 主循环 | `ReadProcessMemory` 轮询 `gameplay_frame` 变化才写一帧 | 没有呈现栅栏：帧与像素的归属不可证 |
| 数据分片 | `ShardWriter.add(observation, action, meta)` | `meta` 没有帧证明字段 |
| 清单 | `process` 生成 `manifest.json` | 没有任何 proof 语义；新旧数据无法区分 |
| 报告 | `CaptureReport` | 没有 ticket/检查/失败计数 |

## 2. 改动点（按文件、按函数）

### 2.1 `touhou_rl/capture_dataset.py`

1. **新参数**（保持默认行为不变）：
   ```python
   def capture_replay(..., frame_proof: str = PROOF_UNKNOWN,
                      fence_addresses: Path | None = None,
                      fence_max_us: int = 250_000) -> CaptureReport
   ```
   `frame_proof=PROOF_UNKNOWN` 时走今天的老路径，逐字节不变。

2. **新分支 `_capture_exact(...)`**（同文件私有函数）：
   - 用 `third_party/th08/build_fence_template` 作为模板（`scripts/build.py --build-type fence` 的产物目录），并设 `TH08_ADDRESS_FILE`；
   - 预检（见 §4）不过就抛错，**不启动游戏**；
   - 启动 → `menu.enter_replay` → 开 fence → `configure(enable=True, pixels=True, hold_every=1)`；
   - `FenceFrameSampler(session, fence, with_pixels=True, with_enemies=True, accept_current=True)`；
   - 每个 accepted sample：写 PNG 到 `frames/`、写 `FrameProofRecord` 到 `exact-frames.jsonl`、`writer.add(obs.encode(...), action, meta)`，其中 `meta` 增加 `ticket`、`frameProof`、`exactIndex`；
   - 任一 sample 不 ok 或 hold 预算耗尽 → 记 `discarded` 与原因，按 E10 标记 `invalid_exact` 并停止。

3. **`CaptureReport` 新字段**：`frameProof`、`tickets`、`checksPassed`、`discarded`、`duplicateImages`、`build`、`environment`。全部写进 `capture-report.json`。

4. **`ShardWriter` provenance**：`{"frameProof": ..., "build": {...}, "exactFrames": "exact-frames.jsonl"}`。

### 2.2 `touhou_rl/pipeline/process`（清单生成）

- 读 `exact-frames.jsonl`（若存在）：用 `frame_proof.require_exact()` + `summarize()` 校验；不通过则**整个数据集失败**（不降级、不猜测）。
- `manifest.json` 增加：
  ```json
  "frameProof": "exact",
  "exactValidFrames": 3000,
  "frames": 3000,
  "checkNames": [...],
  "build": {"exeSha256": "...", "textSha256": "...", "addressTableSha256": "...", "mapSha256": "..."},
  "environment": {"platform": "...", "gpu": ["..."], "python": "...", "gitHead": "..."}
  ```
- 老数据集（无 `exact-frames.jsonl`）写 `"frameProof": "alignment_unknown"`，行为不变。

### 2.3 `pipeline inspect`

- 声称 `frameProof=exact` 但 `exactValidFrames != frames` → 失败；
- 磁盘上有 `frames/` 之外的 PNG、或 PNG 数与 `frames` 不符 → 失败；
- 分片里的 `meta.frameProof` 与清单不一致 → 失败。

## 3. CLI 契约

```powershell
# 试采集（冻结的回放、单面、3000 帧）
python -m touhou_rl.capture_dataset `
  --replay th08_replay/Th8_ud3936.rpy `
  --out third_party/th08/build/rl/datasets/exact-pilot-v1 `
  --tag exact-pilot-v1 --frame-proof exact --stage -1

# 处理（校验 + 写清单；exact 不通过就失败）
python -m touhou_rl.pipeline process --raw third_party/th08/build/rl/datasets/exact-pilot-v1

# 审计
python -m touhou_rl.pipeline inspect --manifest third_party/th08/build/rl/datasets/exact-pilot-v1/manifest.json
```

- `--frame-proof exact` 时 `--template` 必须指向 fence 构建；否则退出码非 0 并给出原因。
- 新增 `--fence-addresses`（默认 `third_party/th08/build_fence/frame-fence-addresses.json`）。

## 4. 预检（启动游戏之前全部完成）

| 编号 | 检查 | 不过的处理 |
|---|---|---|
| P1 | `build_fence/th08.exe` 存在，SHA 与 `build_fence/frame-fence-addresses.json` 记录的 map 属同一次链接 | 拒绝启动 |
| P2 | 地址表的 `mapSha256` 等于当前 `build_fence/th08.map` 的 SHA | 拒绝（表过期会静默读错全局） |
| P3 | 模板 exe 的 `.text` SHA 等于地址表所对应的变体 | 拒绝 |
| P4 | `hold_every` 能生效（试 ack 一帧后 ticket 增量 == 1） | 拒绝 |
| P5 | 前 10 张样张 13 项全过且 `pixels_match_backbuffer` 为真 | 丢弃这 10 张并把数据集标 `invalid_exact` |

## 5. 失败语义

- 采集过程中任一样张失败：**停止**，写 `capture-report.json`（`endedReason="invalid_exact"`）与已采部分，不删除、不补齐。
- `process` 遇到失败标记：拒绝生成 manifest，打印首个失败帧。
- 重跑必须换输出目录并披露（预注册 §6）。

## 6. 验收计划（实现后按此执行）

1. **单测**：`touhou_rl/tests/test_frame_proof.py`（已有 16 项）+ 新增 `test_capture_exact.py`：
   - 假 sampler/假 writer：检查 `exact-frames.jsonl` 字段、PNG 命名、失败即停；
   - 负例：canonical 模板被拒、地址表过期被拒、混合 proof 被拒。
2. **真机试采集**：3,000 帧；要求 `discarded == 0`、`distinctImages == 3000`、`checksPassed == 3000×13`、elapsed 记录。
3. **负例真机**：用 `build/play` 模板请求 exact → 必须在启动游戏前失败。
4. **清单审计**：`inspect` 通过；手工改一个 PNG 后 `inspect` 必须失败。
5. **吞吐记录**：按 round26 的 45.2 张/s 估算，3,000 帧约 66 s 采集 + 启动；若实测低于 30 张/s 再回看 hold 预算。

## 7. 成本与风险

| 项 | 估计 |
|---|---|
| 代码 | `capture_dataset.py` +~200 行；`process/inspect` +~60 行；两个测试文件 |
| 真机时间 | 启动 ~20 s + 采集 ~70 s；负例 ~20 s |
| 磁盘 | 3,000 张 PNG × ~188 KB ≈ **560 MB**（无损，不压缩） |
| 主要风险 | ① 回放期间 `pixels_match_backbuffer` 是否同样成立（round26 只在 attract demo 与 practice 上测过）；② 长时 hold 对回放推进的影响；③ 前台焦点（SendInput 菜单） |

第①条是**实现后第一件要测的事**：先在回放模式下抓 60 帧，确认 60/60，再跑 3,000 帧。

## 8. 不在本方案内

- 战术标签与事件切分（round28：合法标签源只有 F2 4 例，需要先扩课程）。
- 训练与拆分（`process` 的 split 逻辑另议）。
- canonical 镜像的 exact 化（不可能，规则见 round26）。

## 9. 与既有文档的关系

- 解冻规则：[P0-fence-round26 §解冻规则 v2](<P0-fence-round26.md#解冻规则-v2可执行版>)。
- 采样器吞吐：[P0-fence-round26 §补充测量](<P0-fence-round26.md#补充测量采样器浪费与修复同日轮后>)。
- 标签源审计：[P2-privilege-round28](<P2-privilege-round28.md>)。
- 数据门 schema：`touhou_rl/frame_proof.py`（16 项单测）。
