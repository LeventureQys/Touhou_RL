# O2 观测契约与既有语料的形状缺口（主代理实测）

日期：2026-11-15。发现者：主代理。**这不是子代理报告，是我直接读磁盘数组头得到的。**

## 1 结论先说

**现有全部语料（`.tmp_v400_d1` 29 GB、`.tmp_v410_s1c` 2.34 GB、T02c、T04）里存的是 v4.2.0 的降采样观测，
不是 v4.3.0 的 O2。它们不能直接训练 O2 模型，且其中 `global_rgb` 这一路完全不存在。**

## 2 逐项对照（磁盘实测 vs `O2_SHAPES`）

| 通道 | v4.3.0 O2 要求 | 语料实际存储 | 关系 |
| --- | --- | --- | --- |
| `global_rgb` | `(3, 240, 320)` | **不存在** | **完全缺失**（回读里有 RGBA，但既有编码器从不产出全局 RGB） |
| `play_gray` | `(4, 224, 192)` | `(4, 112, 96)`（`play96x112.npy`） | **同一裁剪、同一公式、更粗的块平均**：同为 BT.601 `(77R+151G+28B)>>8` 作用于 `(32,16)` 384×448 视口，但 legacy 用 **4×4** 块平均，O2 用 **2×2** 块平均 → 恰好 2× 分辨率差 |
| `hud_gray` | `(1, 448, 160)` | `(56, 160)`（`hud.npy`） | **不是降采样，是窄带取样**：legacy 只取帧行 `HUD_BANDS=((48,72),(80,96),(96,112))` 共 **56 行**的**绿通道**；O2 要 `y=16..464` 共 **448 行**。legacy **从未采过**其余 392 行 |
| `executed_key_history` | `(8, 7)` | `(8, 18)`（`act_hist.npy`） | 槽位数相同（8），**特征维 18 vs 7** |

### 2.1 一处我先前写错、现更正

我先前的表里写 hud 是"纵向 8× 降采样"——**机制说错了**。448/56 恰好是 8，但 legacy 的 hud 不是把
448 行压成 56 行，而是**只挑了三段带**（48–72、80–96、96–112）并取绿通道。所以 O2 的 hud 里
**有 392 行是 legacy 从未采集过的信息**。

同理我先前说"v4.2.0 只看绿通道"——**只对 HUD 成立**。legacy 的 **play 平面用的是完整 RGBA 的 R/G/B**
（`red*77 + green*151 + blue*28) >> 8`），与 O2 **公式完全相同**。这个更正很重要，因为它证明：
**O2 放宽 alpha 与既有已验证通路完全一致**（legacy 也从不看 alpha），而不是引入新风险。

## 3 实测证据（可复现）

`.tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed1-shard/obs/`：

```
play96x112.npy  shape=(3370, 4, 112, 96)  dtype=uint8  bytes=144,937,482
hud.npy         shape=(3370, 56, 160)     dtype=uint8  bytes= 30,195,722
act_hist.npy    shape=(3370, 8, 18)       dtype=uint8  bytes=    485,802
```

`.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed10-shard/obs/`：

```
play96x112.npy  shape=(6592, 4, 112, 96)  dtype=uint8
hud.npy         shape=(6592, 56, 160)     dtype=uint8
act_hist.npy    shape=(6592, 8, 18)       dtype=uint8
```

**两者的源目录 `T02c/rebuild/…/2f/` 里也只有这三个 .npy**，没有保留任何 640×480 RGBA 原始帧。

## 4 为什么这是路线级问题

1. `DEVELOPMENT_GUIDE.md` / `EXECUTION_PLAN.md` **完全没有提到** D1 的观测是另一套契约
   （我为这两个文件 grep `d1|重渲染|96x112|obs-v420|原始帧`，命中 **0**）。
   `TRAINING_PLAN.md` 只在一处提到 D1，且只讲谱系（"旧 D1/多面 checkpoint 的谱系不清时可作能力开发"），
   **没有讲像素契约**。

2. 因此"用 D1 做 O2 预训练"这句话，在当前资产下**不成立**。可行的只有三条路：

   | 路径 | 代价 | 风险 |
   | --- | --- | --- |
   | (a) 用 v430 采集通路**重新采集** | 每局约 130 s 引擎时间，且要重新跑教师 | 慢，但干净 |
   | (b) 定义**降级适配器**：把 112×96 上采样 2× 当 play、56×160 上采样 8× 当 hud，`global` 用零填或由 play 拼 | 便宜 | **训练/部署不一致**：模型看到的细节比真实部署少，且 `global_rgb` 是编造的 |
   | (c) 给 O2 增加一个**显式 legacy 模式**，并把契约写清楚 | 中 | 必须如实登记为降级，不得冒充 O2 |

3. **绝不能做的是**：默默用 (b) 训练，然后在报告里写"o2_digest 一致"。
   `observation.o2_digest()` 会捕获契约不匹配，但**捕获不到"形状对、内容来自降采样"**这种情况。

## 4.4 **重大更正：路径 (a) 不需要改引擎，现有代码就能出 O2**

我在 §4 暗示重采集可能很贵。继续查证后发现**代价远低于预期**：

| 事实 | 出处 |
| --- | --- |
| 引擎后端返回的是**完整 `(480, 640, 4)` uint8 RGBA 回读** | `touhou_rl/v400/student/obs_spec_v3.py:299` 断言 `pixels.shape[2] == 4`；`touhou_rl/v400/rl/env.py:532` `np.zeros((480, 640, 4), dtype=np.uint8)` |
| v4.2.0 的 `play96x112` / `hud` 只是**从这个 RGBA 回读裁剪**出来的 | `obs_spec_v3.encode_planes_v3` 的 docstring：*"640x480 RGBA8 readback -> (play96x112, hud) uint8 planes"* |
| 回读通过 `self.backend.pixels()` 取得，`v420/rl.py:1204` 就在用它 | `play, hud = obs_spec_v3.encode_planes_v3(self.backend.pixels())` |

**结论**：既有重放通路**已经**握着 O2 需要的原始 RGBA 帧，只是把它交给了 v4.2.0 编码器。
要得到 O2，**只需把那一行编码调用换成 `observation.encode_o2`**——不改引擎、不改协议、不重新采集教师。

所以 (a) 的真实代价只是"重放一遍磁带 + 编码 + 落盘"，约每局 **130 秒引擎时间**，
而不是"重新采集数据"。这让 (a) 成为**最优先**的选项，(b) 更没有存在理由。

## 4.5 `encode_o2` 的"占位帧"防护只查 alpha，不查内容（主代理实测）

我按 docstring 列的拒绝条件逐条实测，六条全部成立（`DuplicateFrameError`/`FrameAlphaError`/
`ShortFrameStackError`/`FrameFormatError`/`KeyHistoryError`，报文都精确）。**但发现一处覆盖缺口**：

| 输入 | 结果 |
| --- | --- |
| 四帧完全相同的帧 | 拒绝（`DuplicateFrameError`） |
| alpha 恒为 7（非 255/0） | 拒绝（`FrameAlphaError`） |
| alpha 恒为 128（半透明） | 拒绝（`FrameAlphaError`） |
| 短栈（2 帧） | 拒绝（`ShortFrameStackError`） |
| 非 480×640 | 拒绝（`FrameFormatError`） |
| **全黑但不透明**（RGB=0, alpha=255，四帧互不相同） | **接受**，`global_rgb` 全零（sum=0） |
| **全 128 灰但不透明** | **接受** |

原因：那句 `uniform (placeholder) frame` 的实现判据是 **alpha 平面**（"alpha plane is constant 7;
only fully opaque (255) or unused (0) alpha can be drawn"），**不看 RGB 内容**。

**风险**：引擎在加载、淡入淡出、换面时会返回**不透明但全黑/全灰**的帧。这类帧会被当成合法观测
编码进数据集，**静默地让模型在这些空白输入上学习**——而形状、dtype、alpha 全都合法，任何形状级检查都拦不住。

**采集侧的最小缓解（已要求 W03h 执行）**：采到后统计 `global_rgb.sum() == 0` 与近零帧的数量，
**如实报告这个数字**；若占比显著，则该部分样本必须标记或剔除，不能默认全部有效。

**另外一并记录**：把帧**上下翻转**后 `encode_o2` **仍然接受**（形状合法）。所以"多翻一次"是
**静默错误**，不会被任何检查捕获——`shm.pixels()` 已经翻成 top-down，采集侧不得再翻。

## 4.6 **第三个缺口：O2 的 alpha 严格性把唯一真实数据源挡在门外**（主代理实测）

第一次真实采集尝试（W03h，引擎真实回读）报：

```
FrameAlphaError: alpha plane holds intermediate values (202..219);
the buffer is blended/premultiplied and dropping alpha is not lossless
```

### 这不是引擎的问题，是 O2 规格过严

`observation.py` 的 `ALPHA_MODES = ('opaque_255', 'unused_0', 'binary_mixed')`，
而它自己的 `ALPHA_POLICY` 写着：

> the alpha channel is **dropped, never blended**; constant 0 (unused), constant 255 (opaque)
> and binary 0/255 are accepted, any intermediate alpha is refused

**既然 alpha 被丢弃、从不参与混合**，那么"必须均匀"对**观测值没有任何保护作用**——它只拒绝数据源。

另一半证据（我先前写错一次，此处已更正）：**v4.2.0 的既有通路根本不看 alpha**。
`obs_spec_v3.encode_planes_v3` 对 play 用 `(77R+151G+28B)>>8`（**与 O2 公式完全相同**，且用了 R/G/B 三通道），
对 HUD 只取绿通道 —— **两条路都不看 alpha**。28.8 GB 语料就是这么产出的，那批模型是能跑的。

所以"alpha 必须均匀"是**当初 O2 规格里我加的、没有依据的严格性**；放宽它与既有已验证通路
**完全一致**，不是引入新风险。我先前说"v4.2.0 只看绿通道"是不准确的——那只对 HUD 成立。

### 处置（正在进行）

给 `encode_o2` 增加**显式可选** `alpha_mode`：
- **默认仍是严格**（既有 47 条测试语义不变）；
- 新增 `"any"`：承认 alpha 被丢弃、不参与任何观测通道，接受真实引擎的 blended alpha；
- **返回对象必须记录实际使用的模式**，让"这批数据是用宽松模式采的"可被审计；
- 代价写明：`"any"` 失去"半透明帧"这一弱信号，调用方**必须自行**做占位帧检测（W03h 已有 `is_blank`/`near_blank` 统计）。

**绝不做的**：在采集脚本里私自把 alpha 归一成 255 再喂——那等于在调用层篡改契约，事后无法审计。

### 一个仍需观察的点

alpha 在 202..219 之间，若 RGB **是被预乘的**，则画面会系统性偏暗。已要求 W03h 如实报
`global_rgb.mean()` 的 min/median/max 与 alpha 的 min/max/mean，**只报数，不做转换**。
若确认预乘，需要单独讨论（那属于真正的数据损坏，不能靠放宽 alpha 掩盖）。

## 5 我建议
  并让 `validate_manifest` 能识别 `obs_contract` 字段不等于 O2 时**拒绝**交给 O2 模型。
- 中期：走 (a)，用 W01 的采集通路重新收一批 O2 数据；规模先小（例如 1 个 shard），
  验证"真实 O2 → 训练 → 评估"闭环。
- **W03 的数据腿在 (a) 完成前，只能算"合成闭环"，不能算真实预训练。** 这一点必须写进 W03 的证据。

## 6 未做

- 未尝试任何上采样/拼接实现（那属于 (b)，需先由用户或计划裁定）。
- 未删除、移动或改写任何语料。
- 未修改 `TRAINING_PLAN.md` / `DEVELOPMENT_GUIDE.md`（应由子代理按白名单改，且需用户同意方向）。
