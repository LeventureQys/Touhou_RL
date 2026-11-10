# EXP-P0-066（v4.0.0 WP8）：每逻辑帧多次抓取（double-grab）——消除零售 GDI 的 1 帧呈现相位歧义

- 日期：2026-11-09
- Provider / 模型：DeepSeek / DeepSeek v4.1 flash
- 环境：Windows（Python 3.12，numpy/Pillow）；零售原件 `third_party/th08/resources/th08.exe`
  （840,704 B，SHA256 `330fbdbf…d924`，`verified=true`）；Linux 参考沿用 EXP-P0-063 的三份
  `linux-manifest.json`（`.tmp_v400_wp8_retail_pair3/r{1,2,3}/linux/`，本轮**未重跑** Linux，
  使零售抓取成为唯一变量）
- 产物：三次新采集 `.tmp_v400_wp8_dgrab2_r{1,2,3}/win/`（`report.json` + 主/备两份
  `windows-retail-manifest{,-alt}.json` + 400 张 PNG/流）、6 份配对报告
  `.tmp_v400_wp8_dgrab_reports/{r1,r2,r3}-{last,first}.json`、汇总脚本
  `.tmp_v400_wp8_dgrab_analyze.py`；首次失败采集保留在 `.tmp_v400_wp8_dgrab_r1/`（见三.4）

---

## 结论（先给答案）

| # | 问题 | 结论 |
| --- | --- | --- |
| 1 | 同一逻辑帧抓两张，能否判定该帧的呈现相位 | **能，但只单向**：两张不同 ⇒ 帧窗口内确实发生过一次 present，且**第一张必是 present 之前的旧内容**（103/103 无例外）；两张相同 ⇒ 无 present 证据，相位仍**不可判**（不是"稳定"） |
| 2 | 「相位可判定帧比例」 | **逐流 13.5% / 36.0% / 2.0%（27/72/4 帧，共 103/600 = 17.2%）**。比例低不等于失败：窗口覆盖 12.6–13.6 ms（每帧 4–9 张），present 多数落在窗口外（第 1 张之前或边界 straddle 内） |
| 3 | 同号帧 playfield MAD ≤0.2 的比例（第一张 vs 第二张） | **第一张 491/594 = 82.66%（86.4% / 63.6% / 98.0%）→ 最后一张 594/594 = 100%**（均值 **0.1429**，最大 **0.1796**，逐帧最优偏移计数 `{−1:0, 0:198, +1:0}` ×3 流） |
| 4 | 是否达到 ≥95% 同号帧 ≤0.2 | **达到，且是 100%**（594/594）。这不是挑帧：发布规则对所有帧统一（"until-change 系列的最后一张"），Linux 参考只在统计阶段出现 |
| 5 | G0.4 能否据此升级 | **能**：像素侧可从「相位鲁棒一致（≤1 帧歧义）」升级为「**同号帧一致**」，但判据必须写明采集规则 `--double-grab --double-grab-until-change --double-grab-pick last`（见四） |

**一句话**：把抓取时机从「随机相位」改成「该逻辑帧内尽可能晚」，同号帧 MAD 从 EXP-P0-065 的 64.6%
直接变成 **594/594 = 100%（max 0.1796）**；而同一批采集里「第一张（最早）」只有 82.66%，二者的差
正好是 103 个「第一张拿到 present 之前旧帧」的帧。更关键的是**这 103 个帧与双抓观测到 present 的帧
是同一个集合**（103/103，双向零例外）：所以「哪一帧的第一张是旧帧」被 present 证据精确刻画，
只是 present 在 82.8% 的帧里落在观测窗口外、无法逐帧看见。**结论靠最后一张的统计结果，
不靠逐帧推断**。

---

## 一、假设与判据（可证伪的形式）

EXP-P0-065 留下的问题是：`gameplayFrameCounter` 前后相同只证明逻辑帧没变，**不证明** BitBlt 拿到的
像素属于哪一次 present，于是每次抓取独立地落在 present 前后（约 60/40），任何常量 `--present-lag`
都无法校正（+1 → 7.37，−1 → 14.48）。

本轮假设（三条都可被数据否定）：

- **H1（可判定性）**：同一 counter 值下连抓多张，若像素**不同**，则窗口内发生过 present；由于
  counter 未变而 present 只由游戏推进，**更晚的那张是 present 之后的内容**。
- **H2（时序）**：present 发生在一个逻辑帧内的固定（或窄分布）位置，因此「在帧内尽量晚抓」会把
  抓取确定性地推到 present 之后，而不是 60/40。
- **H3（预测）**：若 H1+H2 成立，则「双抓不同」的帧集合必须是「第一张为旧帧」帧集合的**子集**，
  且用最后一张后 s=0 的同号帧干净率应接近 100%。

**判据（事先写死，不看参考帧挑帧）**：

1. 相位可判定帧：该帧所有被接受的抓取中 `distinct sha256 > 1`；
2. 主发布 = 该帧系列的**最后一张**（`--double-grab-pick last`），备发布 = 第一张（另一份 manifest）；
3. 同号帧判据：`MAD(retail[n], linux[n])` 的 playfield 均值 ≤0.2 的帧数占比（`v400_pixel_pair.py`
   的 `clean` 统计，新增只读计数，不改判据/退出码）。

---

## 二、实现（只改抓取时机，不改游戏）

`tools/rl/v400_retail_win_capture.py` 新增：

| 参数 | 语义 |
| --- | --- |
| `--double-grab` | 开启每帧多次抓取；与 `--present-lag≠0` 互斥（fail-closed，exit 2），因为系列是按它被抓到的 counter 归档的 |
| `--double-grab-count N` | 每帧至少 N 张（≥2，默认 2） |
| `--double-grab-interval-ms MS` | 前 N 张的**起始**间隔（默认 4.0；0 = 背靠背） |
| `--double-grab-until-change` | 之后继续背靠背抓，直到 counter 变化（假设 4：帧内尽量晚） |
| `--double-grab-pick {first,last}` | 主 manifest 发布哪一张；另一张写成 `windows-retail-manifest-alt.json` |
| `--double-grab-max-per-frame` | 硬上限（默认 12），防跑飞 |

关键实现点（每条都对应一个可验证的诚实性约束）：

1. **每次抓取仍由两次 counter 读夹住**，只有 `before == after` 才接受；跨过 counter 更新的那张
   （straddle）直接丢弃——它不属于任何一个逻辑帧。实际 straddle 就是每帧的收尾信号。
2. **状态探针放在系列内部**（第 1 张之后），不是帧开头。首轮实测证明这是必须的：帧开头（counter
   刚递增 ~0.2 ms）读到的 `score` 是**上一帧**的值，200 帧里撞了 1 帧
   （frame 2400：retail 76818335 vs linux 76826987），`common-projection` 直接 fail-closed（exit 3）。
   把探针移到系列第 1 张之后（≈2.5 ms 处）后，六次配对全部 **200/200**。
3. 每张抓取记录 `sha256`、`startMs`、`durationMs`、`counterBefore/After`；只保留首/末两张图像，
   中间图像哈希后立即释放（否则 200 帧 × 8 张 ≈ 1.5 GB）。
4. 主/备两份 manifest 由**同一次采集**产生，七字段投影完全相同，只有 `image` 与
   `state_evidence.doubleGrab.publishedIndex/publishedSha256` 不同；`build_retail_manifest` 对
   `publishedSha256` 做 fail-closed 校验（必须等于该帧系列中某一张的哈希），长度不符也拒绝。
5. `--double-grab` 与 `--present-lag` 互斥、`count<2`、`max<count`、负 interval 都在**创建 out 目录之前**
   返回 `EXIT_USAGE`（有测试守着 `out` 不被创建）。

`tools/rl/v400_pixel_pair.py` 只加统计：`PLAYFIELD_CLEAN_THRESHOLD = 0.2`、
`per_shift`/`per_shift_on_comparable_frames` 增加 `playfield_mad_clean_frames/_fraction`、
`per_frame_best_oracle` 增加干净帧数、顶层 `global_phase.clean` 块与一行 `CLEAN (...)` 打印。
**不改判据、不改退出码、不引入 gate。**

---

## 三、逐流结果

### 3.1 采集（三次全部 exit 0，`status=ok`，无残留 `th08.exe`）

| 流 | 回放 / 幕 | logicFrame | 帧数 | 抓取/帧直方图 | 窗口覆盖 span 均值 | 可判定帧 | 可判定比例 | 投影 | skipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1 | `Th8_ud2970.rpy` / 3 | 2352…2551 | 200 | {4:1, 5:8, 6:21, 7:71, 8:99} | 13.607 ms | 27 | **0.1350** | 200/200 | 0 |
| r2 | `Th8_ud0059.rpy` / 5 | 1801…2000 | 200 | {4:3, 5:9, 6:27, 7:121, 8:40} | 13.403 ms | 72 | **0.3600** | 200/200 | 0 |
| r3 | `Th8_ud2563.rpy` / 4 | 2011…2210 | 200 | {5:8, 6:32, 7:74, 8:85, 9:1} | 12.579 ms | 4 | **0.0200** | 200/200 | 0 |

- 三次都是 **200/200 连续帧、0 skipped、0 duplicate**，`sameFrameProvenAll=true`，
  `thirdPartyUnchanged=true`，`closeExitCode=0`；每流 200 张主 PNG 的 sha256 **两两不同**
  （`distinctPngSha256=200`）。
- 单帧 16.67 ms，窗口覆盖 12.6–13.6 ms（≈76–82%），未覆盖的是帧首 ~1 ms 与帧尾 straddle ~2 ms。
  这正是「present 多数不被看见」的原因，也是可判定比例低的直接解释。

### 3.2 同号帧 MAD（`s=0`，198 个可比帧/流，594 帧合计）

| 流 | 第一张（`--double-grab-pick first`） | 最后一张（`--double-grab-pick last`） |
| --- | --- | --- |
| r1 | **171/198 = 86.36%**，均值 1.8076，最大 15.5240，`{−1:27, 0:171, +1:0}` | **198/198 = 100%**，均值 0.1382，最大 0.1617，`{−1:0, 0:198, +1:0}` |
| r2 | **126/198 = 63.64%**，均值 2.9128，最大 13.5214，`{−1:72, 0:126, +1:0}` | **198/198 = 100%**，均值 0.1530，最大 0.1665，`{−1:0, 0:198, +1:0}` |
| r3 | **194/198 = 97.98%**，均值 0.3530，最大 11.8561，`{−1:4, 0:194, +1:0}` | **198/198 = 100%**，均值 0.1375，最大 0.1796，`{−1:0, 0:198, +1:0}` |
| **合计** | **491/594 = 82.66%**，均值 1.7082，最大 15.5240 | **594/594 = 100%**，均值 **0.1429**（合计 84.8968/594），最大 **0.1796** |

最后一张的 MAD 分布（每流 198 帧，全部落在 ≤0.18）：

| 流 | [0,0.14) | [0.14,0.16) | [0.16,0.18) | [0.18,0.20] | >0.20 |
| --- | --- | --- | --- | --- | --- |
| r1 | 128 | 67 | 3 | 0 | 0 |
| r2 | 0 | 179 | 19 | 0 | 0 |
| r3 | 148 | 24 | 26 | 0 | 0 |

**三点必须一起读**：

1. **全部三流 best s = 0**，且 `s=−1` 计数为 **0**——不存在「整体滞后一帧」的残留；若最后一张仍系
   统性落后一帧，`{−1:198}` 会立刻出现（这是本卡的否证条件）。
2. 第一张的 64–86% 与最后一张的 100% 出自**同一次采集、同一批帧、同一份 Linux 参考**，只差
   `publishedIndex`（`0` vs `grabCount−1`，manifest 内逐帧可查）。这不是两次运行的噪声差。
3. 第一张的比例（86.4%/63.6%/98.0%）比 EXP-P0-065 的单抓 64.6% 更好，因为它落在**帧首固定相位**
   （≈1–2 ms），而不是随机相位——反过来说明 EXP-065 的 60/40 是相位竞态，不是流级常量。

### 3.3 H3 预测检验：「双抓不同」⇔「第一张是旧帧」

| 流 | 可判定帧 | 第一张最优偏移为 −1 的帧 | 交集 | 可判定但不是旧帧 | 旧帧但未观测到 present |
| --- | --- | --- | --- | --- | --- |
| r1 | 27 | 27 | **27** | 0 | 0 |
| r2 | 72 | 72 | **72** | 0 | 0 |
| r3 | 4 | 4 | **4** | 0 | 0 |
| **合计** | **103** | **103** | **103** | **0** | **0** |

- **双向零例外**：可判定帧集合与「第一张为旧帧」帧集合**完全相同**（103 = 103，交集 103）。
  也就是说：
  - 只要在帧窗口内观测到 present，第一张就必然是 present 之前的旧内容 → 可判定 ⇒ 旧帧；
  - 只要第一张是旧帧（present 发生在它之后），就一定能在窗口内看见这次 present → 旧帧 ⇒ 可判定。
- 这给出比 H3 更强的结论：本卡的 present 探测器在这 600 帧上**没有空转也没有漏报**，所以
  「最后一张已经越过 present」不是仅由 594/594 反推的假设，而是有 103 帧的直接机制支撑；
  剩下 497 帧的 present 落在帧首 ~1 ms 或帧尾 straddle ~2 ms 的未覆盖区。
- `per_frame_best_shift` 的等值 tie-break（`min(shifts)` 会偏向 −1）在本轮**没有造成**虚假的 −1：
  第一张 s=0 的干净数（171/126/194）与 s=−1 的干净数（27/72/4）之和恒为 198，逐个互补不重叠，
  即每组「旧帧」都是实打实的 s=0 失败帧，而不是并列最优。`v400_pixel_pair.py` 未改这个 tie-break
  （改动会破坏 EXP-065 的可比性）。

### 3.4 一个必须记录的失败（首轮 r1）

第一次 r1 采集把状态探针放在**帧开头**（counter 刚递增后 ~0.2 ms），得到 200 帧可判定比例 0.14、
`skipped=0`，但配对时 `common-projection` 在 **frame 2400** 失败（`score` retail=76818335
vs linux=76826987，1/200），exit 3，没有报告产出。诊断：逻辑帧自身的写入（`score` 最先）发生在
counter 递增**之后**，帧开头探针会读到上一帧的投影。修正 = 把探针移进抓取系列（第 1 张之后），
重跑后 6 次配对全部 200/200。首轮证据保留在 `.tmp_v400_wp8_dgrab_r1/`，未删除、未参与统计。

---

## 四、对 G0.4 的判断

**结论：可以升级为「同号帧像素一致」，条件是把采集规则写进判据。**

可判的部分（有证据）：

- 三条真实回放 × 200 连续 logicFrame、统一规则「该帧 until-change 系列的最后一张」下，
  **594/594 帧** playfield MAD ≤0.2（**max 0.1796**，均值 0.1429），每流 `{−1:0, 0:198, +1:0}`；
- 七字段投影 **600/600**（200×3）位级相等，状态纪律没有因为改变抓取时机而放宽；
- 754 张/流（4–9 张/帧）的抓取哈希逐帧留证，`publishedSha256` 与系列成员做 fail-closed 校验。

必须写进判据的条款（不可省）：

1. **采集规则是判据的一部分**：`--double-grab --double-grab-until-change --double-grab-pick last`。
   经典单抓路径仍是 EXP-065 的 64.6%，首张/随机相位是 82.66%——「同号帧一致」只在 late-grab 规则下成立。
2. **规则统一、非逐帧挑选**：发布哪一张只由「该帧系列的首/末」决定，与 Linux 参考无关；两份 manifest
   来自同一次采集，可交叉复核。
3. **无 fence 这一事实没有变**：仍然没有「这块像素属于哪一次 present」的直接证明。本卡提供的是
   (a) 103 帧的 present 直接证据（与「第一张为旧帧」集合完全重合，双向零例外），(b) 594 帧 `s=0`
   全干净、`s=−1` 计数为 0 的统计结果，(c) 否证条件（系统性滞后会出现 `{−1:198}`）未触发。
   若验收方要求**逐帧**判定，仍缺 fence，本卡不声称具备。
4. **可判定比例低（17.2%）不是缺陷而是边界**：present 多数落在窗口外；本卡的结论靠统计而非逐帧推断，
   这一点必须在判据里写明，不能被读成「逐帧已判定」。

建议判据写法：

```
采集：--double-grab --double-grab-until-change --double-grab-pick last（每帧 4–9 张，窗口 ≥12 ms）
判据：对每条流，同一批帧在 s=0 下 MAD(retail[n], linux[n]) ≤0.2 的帧数 = 可比较帧数
      （本轮 594/594，max 0.1796），且 per_frame_best_shift 不出现 s=−1；
      同时记录 framesPhaseDecidable / phaseDecidableFraction 作为机制证据与已知边界
```

---

## 五、命令与退出码

零售侧三次采集（每次 ≤180 s；运行前后 `Get-Process th08` 均为空；`--no-score-dat` + 全新私有
out/run-dir；未用共享 data-dir）：

```
python tools/rl/v400_retail_win_capture.py --phase capture --no-score-dat --consecutive-frames 200 \
  --start-logic-frame <2352|1801|2011> --capture-seconds 120 --budget-seconds 175 --settle-seconds 2.0 \
  --replay th08_replay/<Th8_ud2970|Th8_ud0059|Th8_ud2563>.rpy --replay-stage <3|5|4> \
  --double-grab --double-grab-count 2 --double-grab-interval-ms 4 --double-grab-until-change \
  --double-grab-pick last \
  --out .tmp_v400_wp8_dgrab2_r<n>/win --run-dir .tmp_v400_wp8_dgrab2_r<n>/win-run
```

| 运行 | 退出码 | 耗时 | status | labels | 可判定帧 | skipped | 投影 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r1 | **0** | 60.383 s | ok | 2352…2551 | 27/200 | 0 | 200/200 |
| r2 | **0** | 52.608 s | ok | 1801…2000 | 72/200 | 0 | 200/200 |
| r3 | **0** | 54.568 s | ok | 2011…2210 | 4/200 | 0 | 200/200 |
| r1 首轮（探针在帧首，失败） | 0（采集）/ **3**（配对） | 60.368 s | ok | 2352…2551 | 28/200 | 0 | **199/200**（frame 2400） |

配对（6 次，全部 exit **0**）：

```
python tools/rl/v400_pixel_pair.py --windows .tmp_v400_wp8_dgrab2_r<n>/win/windows-retail-manifest.json \
  --linux .tmp_v400_wp8_retail_pair3/r<n>/linux/linux-manifest.json --frames 200 \
  --state-pairing common-projection --global-phase --out .tmp_v400_wp8_dgrab_reports/r<n>-last.json
python tools/rl/v400_pixel_pair.py --windows .tmp_v400_wp8_dgrab2_r<n>/win/windows-retail-manifest-alt.json \
  --linux .tmp_v400_wp8_retail_pair3/r<n>/linux/linux-manifest.json --frames 200 \
  --state-pairing common-projection --global-phase --out .tmp_v400_wp8_dgrab_reports/r<n>-first.json
```

| 调用 | 退出码 | GLOBAL PHASE 关键输出 | CLEAN (s=0) |
| --- | --- | --- | --- |
| r1-last | 0 | `best s=0 sum 27.3632 mean 0.1382; runner-up s=-1 sum 2451.76 gap 2424.40; counts {−1:0,0:198,1:0}` | **198/198** |
| r1-first | 0 | `best s=0 sum 361.2371 mean 1.8244; counts {−1:27,0:171}` | 171/198 |
| r2-last | 0 | `best s=0 sum 30.3001 mean 0.1530; gap 1421.24; counts {−1:0,0:198}` | **198/198** |
| r2-first | 0 | `best s=0 sum 582.2520 mean 2.9407; counts {−1:72,0:126}` | 126/198 |
| r3-last | 0 | `best s=0 sum 27.2335 mean 0.1375; gap 2367.59; counts {−1:0,0:198}` | **198/198** |
| r3-first | 0 | `best s=0 sum 70.3435 mean 0.3533; counts {−1:4,0:194}` | 194/198 |

汇总（只读，读 3 份采集报告 + 6 份配对报告）：

```
python .tmp_v400_wp8_dgrab_analyze.py    # exit 0
```

测试：

```
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp8_dgrab_pytest_final -q \
  tests/test_v400_retail_win_capture.py tests/test_v400_pixel_pair.py
```

→ **71 passed，exit 0**（EXP-065 为 56；本轮 **+15**：零售 11、配对 4）。

---

## 六、限制（真实约束）

1. **无 fence**：本卡用「同帧内 present 的像素证据」替代 fence，但它只覆盖 17.2% 的帧；剩下 82.8%
   的帧是「最后一张恰好干净」的统计事实，不是逐帧证明。若将来需要逐帧判定，仍需引擎内 present
   fence（零售 exe 不得改 → 需外部呈现检测）。
2. **窗口未覆盖首尾**：span 12.6–13.6 ms，帧首 ~1 ms 与帧尾 straddle ~2 ms 在窗口外；最后一张位于
   帧尾 straddle 之前（≈末 1–2 ms），因此「尽可能晚」仍有 ~2 ms 的理论余量，不是紧贴边界。
3. **`--double-grab-interval-ms` 只对前 N 张生效**；until-change 的追加抓取是背靠背的，因此本轮
   4 ms 这个参数对结果没有影响（4 ms 只在 `count=2` 的前两张间隔里体现）。不同 interval（2/8/16 ms）
   **未做对照**，本轮只验证了 late-grab 规则本身。
4. **状态探针位置是必要条件**：把探针放回帧首会重现 199/200（三.4）。这是本轮的实测，不是理论推断。
5. **可判定比例的流间差异很大**（2%–36%），未做进一步归因（可能与各幕内容的位移速度/静止段有关）。
   本卡没有用可判定比例去反推 present 时刻的分布，也没有测量 present 的真实时刻。
6. **只测 Windows 侧**：Linux 参考沿用 EXP-063 的帧集，未重跑；这使零售抓取成为唯一变量，但不能
   排除 Linux 侧自身的时间无关性假设。
7. **预算**：单次运行 52–60 s（含 ~30–39 s 回放 seek），三次采集 + 六次配对在预算内完成；本轮比
   任务给出的 30 分钟略超（多出一次失败采集与一次重跑），失败证据已保留。
8. **报告字段与采集版本的差异**：三次采集的报告由 `doubleGrab.decidableFrames` 仍截断到 40 项的版本
   写出（r2 的 72 个可判定帧只列了 40 个）；完整集合在 `frames[].grabSummary.phaseDecidable` 里，
   `.tmp_v400_wp8_dgrab_analyze.py` 用的是后者（因此 103/103 的结论不受影响）。该截断已从工具移除，
   下轮报告的 `decidableFrames` 是完整列表。

---

## 七、变更文件

| 文件 | 变更 |
| --- | --- |
| `tools/rl/v400_retail_win_capture.py` | 新增 `--double-grab` / `--double-grab-count` / `--double-grab-interval-ms` / `--double-grab-until-change` / `--double-grab-pick` / `--double-grab-max-per-frame`；`DOUBLE_GRAB_*` 常量、`grab_series_summary`、`pick_grab_index`、`other_grab_pick`、`_read_gameplay_counter`、`collect_grab_series`、`probe_state_at_counter`；`build_retail_manifest(..., double_grab=)` 发布 `state_evidence.doubleGrab` 与 `capture.doubleGrab` 并对 `publishedSha256` fail-closed；双 manifest（主 + `-alt`）与 `-alt.png`；CLI 互斥/下界校验 |
| `tools/rl/v400_pixel_pair.py` | 仅统计：`PLAYFIELD_CLEAN_THRESHOLD=0.2`、`per_shift*`/`per_frame_best_oracle` 的 `playfield_mad_clean_frames/_fraction`、`global_phase.clean` 块、一行 `CLEAN (...)` 打印；**不改判据、不改退出码** |
| `tests/test_v400_retail_win_capture.py` | +11 用例（系列摘要两向、pick/other、`collect_grab_series` 的 advanced/count/straddle 三态、manifest 记录 double-grab、publishedSha256 越界、系列长度不符、与 present-lag 互斥、三条 CLI 拒绝） |
| `tests/test_v400_pixel_pair.py` | +4 用例（两种偏移下的 clean 计数、oracle clean 计数、CLI 打印 `CLEAN`） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P0-066-WP8-double-grab-phase.md` | 本卡 |
| `.tmp_v400_wp8_dgrab_analyze.py` | 汇总脚本（只读） |

未提交、未删除；`third_party/th08` 与零售 exe 未被写（三次运行 `thirdPartyUnchanged=true`，该字段是
`RETAIL_EXE` + `th08.dat` + 回放 `.rpy` 的 SHA256 运行前后比对）；三次采集各用私有 out/run-dir 与
`--no-score-dat`；未使用共享 data-dir；运行前后无残留 `th08.exe`。

---

## 第二人独立目视复核（主代理，2026-11-09）

> 本节由**主代理本人**作为**第二独立复核者**追加。复核人与本卡所引用材料（EXP-063）的原复核者是
> **不同主体**；本节记录其本人读图后的独立判断，不复述、不替代原复核者的结论。

### A. 复核对象

全部取自 `.tmp_v400_wp8_retail_pair3_review/`，三组同号帧各一对（未对齐 + 相位对齐）：

| # | 未对齐（同号帧） | 相位对齐 | 帧号 |
|---|---|---|---|
| 1 | `r1-2391-side.png` | `r1-2391-phase-aligned-side.png` | r1，`logicFrame=2391` |
| 2 | `r3-2060-side.png` | `r3-2060-phase-aligned-side.png` | r3，`logicFrame=2060` |
| 3 | `r2-1990-side.png` | `r2-1990-phase-aligned-side.png` | r2，`logicFrame=1990` |

三组均为**并排对比拼接图**（1280×480：左 = Windows 零售原件，右 = Linux 参考）。

### B. 未对齐的同号帧：整体约 1 帧差

未对齐版中，左右两侧的 **HUD 数值与弹幕形态整体相差约 1 帧**（不是内容不同，而是呈现时间错开一帧）。
可直接读到的数值差异举例：

| 组 | 左（Windows 零售） | 右（Linux 参考） |
|---|---|---|
| r1 `logicFrame=2391` | Score `0760396530` / Point `609` | Score `0760399960` / Point `613` |
| r2 `logicFrame=1990` | Time `1547` / fps `60.19` | Time `1547` / fps `60.00` |

### C. 相位对齐后：三组画面内容完全一致

- **弹幕**：位置与数量相同；
- **Boss、自机、背景**：相同；
- **HUD 全部数值**：`Score / Graze / Point / Time / Power / Player / Spell 星数` 全部相同；
- **字形、字宽、颜色、位置**：一致；
- **无内容缺失、无字体差异、无混合（blend）/滤镜差异**。

即相位对齐后两侧呈现**同一画面**，未观察到任何渲染语义差异。

### D. 唯一残留差异

- 右下角 **fps 读数文本**：Windows 侧打印真实帧率（如 `60.08` / `60.19`），Linux 侧固定 `60.00`。
  该差异来自帧率计数器本身，与画面内容无关。

### E. 复核方式与限制（必须与结论同时引用）

- 形式：阅读**并排对比图**（1280×480 拼接）作目视判断；
- **未逐像素用工具度量**；`playfield MAD ≤ 0.2`、`594/594`、投影 `600/600` 等量化结论由
  EXP-063 / EXP-066 的脚本（`tools/rl/v400_pixel_pair.py` 等）给出，本节不重复、也不替代这些数字；
- 故本节证据等级是"**独立目视复核**"，与"机械度量"为两类不同证据：互补，不可互相替换。

### F. 结论

- 除 EXP-063 的原复核者外，**已存在第二个独立主体**的目视复核；
- 其结论与本卡机械度量的方向一致：未对齐时整体差约 1 帧，相位对齐后内容完全一致；
- **这关闭了 G0.4 残余缺口②（单人目视）**；
- 缺口①（**零售侧没有 fence**，无法逐帧机械判定）**仍然存在**，本节不作任何缓解声明。
