# EXP-P1-002 — P1-S1 激光几何门控：真实对局里 `LASERS` 到底给不给数据（2026-11-09）

**结论：激光几何可得，但门控是"有条件通过"，不是无条件通过。**

- 用固定构建 `build/rl/th08-modern`（SHA-256 `591c521d…`，**未重建**）在 WSL 跑了 **141,000 帧**真实对局：
  `LASERS` 返回非空行的**只有 replay Final B（stage slot 7）**——12,000 帧里 **1,118 帧非空**，单帧峰值 **230 条**，累计 **220,320 条**。
- 任务指定的三个来源（`RESET 0 1 0 0 0` stage 1 normal / `Th8_ud0059.rpy` **Stage 5** / practice **Final A**）**激光全为 0**。
  「Stage 5 有激光」这个前提被证伪；本卡改用 11-P1 §2.1 允许的「**或其它已知有激光的面**」= **Final B**，并把它作为显式来源跑满 12,000 帧。
- 字段：本 build 的 `LASERS` 行是 **11 个 token**，顺序
  `slot:state:posX:posY:posZ:angle:startOffset:endOffset:currentWidth:width:timer`；
  **没有 `inUse`**（该字段是源码后续 revision 才追加的），`currentWidth` 在观测到的所有帧恒为 `0.0`，**厚度只能用 `width`**。
- **`STATE` 没有 `frameEventFlags`（也没有任何事件字段）**；`EVENTS` / `PLAYER` / `HAZARDS` 在本 build 都是 `ERR unknown-command`。
  → 11-P1 §4 的 PLAYER_HIT 决死 bomb **当前不可得**，需最小扩展（§6）。
- 附带发现：`BULLETS` 在 **replay 播放期间不可用**（`ERR bullets-not-ready reset-required`）；`BULLETS count` 恒含 1 个
  `BULLET_STATE_SENTINEL` 槽（slot `0x600`、state 6、坐标全 0）；practice 模式死亡后 `STEP` 仍报 `advanced=1` 但世界冻结。
- 单帧读取成本（HazardTape 预算）：普通帧 `LASERS` **0.032 ms**、`BULLETS` **0.038 ms**、两者合计 **0.070 ms**；
  激光密集区间（均值 104.6 条、平均 7.0 页翻页）`LASERS` **0.292 ms**（最多 14 页）；`STEP` **0.083 ms/帧**。

本卡**不**声称 HazardTape 已能录制激光（那要 P1-WP1 实现），只回答"激光几何在真实引擎里是否可得、字段是否够用、覆盖在哪"。

---

## 1. 门控问题与判定

11-P1 §2.1 把子弹/激光/敌人本体的致死几何列为 HazardTape 的必录项，其中激光是**前置门控**：
若真实对局里 `LASERS` 长期为空，激光建模就无从验证，必须如实记 INVALID 而不是假装覆盖。

| 判定项 | 结果 |
|---|---|
| `LASERS` 命令是否存在 | 存在（`OK frames=… count=… offset=… limit=… lasers=…`） |
| 是否有任何来源非空 | **是**（replay Final B） |
| 三个指定来源是否非空 | **否**（0 / 0 / 0） |
| 字段是否够 §2.1 用 | 位置/角度/起止偏移/宽度/计时都有；**缺 `inUse`**，但行本身就是 in-use 过滤的结果 |
| 覆盖是否足够 | **不足**：只有 replay 形式的 Final B 命中；practice 形式的 Final B 以及另外 9 个 stage slot（各 12,000 帧）在有效帧内均为 0 |
| 门控状态 | `VALID`（.tmp_v400_p1s1/laser_gate.json `verdict.status`），**附"仅 Final B replay"这一强限定** |

所以：不是 INVALID，但**不能**据此说"激光在 stage 1/Stage 5/练习模式都有覆盖"。Stage 5 replay 的 3,000 帧
（以及后来单独 12,000 帧的复扫）里 `LASERS` 一条都没有。

---

## 2. 命令与退出码

工具：`tools/rl/v400_p1_laser_gate.py`（本次新增，仅标准库；在 WSL 内直接驱动引擎）。

```sh
# 记录在 .tmp_v400_p1s1/run_gate.sh 的完整命令（在 WSL 内执行）
wsl.exe -d Ubuntu -- sh /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p1s1/run_gate.sh

# 脚本实际调用的工具命令：
cd /mnt/d/WorkShop/Touhou_RL
PYTHONPATH=/mnt/d/WorkShop/Touhou_RL nice -n 10 python3 tools/rl/v400_p1_laser_gate.py \
    --engine /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \
    --data-dir /root/v400-data/p1s1/game \
    --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p1s1/laser_gate.json \
    --frames 3000 --sample-every 500 \
    --stage 5 --practice-stage 6 \
    --replay /mnt/d/WorkShop/Touhou_RL/th08_replay/Th8_ud0059.rpy \
    --extra-replay-stages 7 --extra-replay-frames 12000 \
    --scan "replay:0,1,2,4,5;practice:3,4,5,6,7" --scan-frames 12000 \
    --timing-laser-stage 7 --timing-laser-start 9300 --timing-laser-frames 200
```

- **退出码 `0`**（`exit=0`，`run.log` 末行；verdict VALID）。整轮 **15 次引擎启动**、60 s（`11:10:10 → 11:11:10`）。
- 引擎每次都以 `--rl-role shadow`（null D3D device，无需 X）启动，**逐帧** `STEP 0 1`，串行、同一时刻只有 1 个 WSL 引擎。
- 引擎身份：`591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`（JSON `engine.sha256`，未重建）。
- 数据目录用独立副本 `/root/v400-data/p1s1/game`（`th08.dat`/`thbgm.dat` 符号链接，`score.dat`/`th08.cfg` 复制），避免与其它实验互踩 `score.dat`。
- 单元测试：`python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p1s1_pytest -q tests/test_v400_p1_laser_gate.py` → **24 passed**。

---

## 3. 四个显式来源

| 来源 id | 请求 | 观测帧 | 有效帧* | `LASERS` 非空帧 | 峰值条数 | 累计条目 | `BULLETS` 非空帧 | `BULLETS` 最大 |
|---|---|---|---|---|---|---|---|---|
| `reset-stage1-normal` | `RESET 0 1 0 0 0`（Stage 1 normal） | 3,000 | 3,000 | **0** | 0 | 0 | 2,582 | 56 |
| `replay-stage5` | `TH08_AUTOPLAY_STAGE=5`（Stage 5） | 3,000 | 3,000 | **0** | 0 | 0 | 不可用† | 不可用† |
| `practice-stage6` | `RESET 6 1 0 0 1`（Final A） | 3,000 | 3,000 | **0** | 0 | 0 | 2,909 | 1,494 |
| `replay-finalb` | `TH08_AUTOPLAY_STAGE=7`（Final B） | 12,000 | 12,000 | **1,118** | **230** | **220,320** | 不可用† | 不可用† |

\* 有效帧 = 最后一个 `f3 gameplayFrameCounter` 仍在推进的采样点（`lastLiveSampleFrame`）。
† replay 播放不发 `RESET`，而 `HandleBullets` 要求 `g_resetReady` 为真，所以整段 replay 里 `BULLETS` 恒回
`ERR bullets-not-ready reset-required`；`LASERS` 只检查 `globals`，因此在 replay 里正常。

关键数字：

- 激光在 Final B 的大致位置：第 **9,090 帧**起出现（首行见 §5），**9,450 帧**达到峰值 230 条；
  9,500 / 10,000 帧的取样为 220 / 230 条，**10,500 帧取样已归零**——非空区间落在 9,090–10,500 帧之间
  （两次运行的入场帧略有差异，故帧号有 ±2 抖动，条数直方图逐位相同）。
- Final B 的激光是"每帧新增 10 条、逐渐拉长"的墙式弹幕：
  条数直方图 `10:10, 20:10, …, 180:10, 210:364, 220:456, 230:84`（键=该帧条数，值=帧数）。
- `BULLETS` 的 56 / 1,494 是**扣掉 sentinel 槽之后**的 `maxCount`；原始 header 是 57 / 1,495（见 §8.2）。

---

## 4. 覆盖扫描（额外，12000 帧/槽）

为回答"到底哪一面有激光"，对能进入的每个 stage slot 各跑 12,000 帧（约一整个 stage）：

| slot | 名称 | 进入方式 | 帧 | 有效帧 | `LASERS` 非空帧 | `BULLETS` 最大 |
|---|---|---|---|---|---|---|
| 0 | Stage 1 | replay | 12,000 | 12,000 | 0 | 不可用 |
| 1 | Stage 2 | replay | 12,000 | 12,000 | 0 | 不可用 |
| 2 | Stage 3 | replay | 12,000 | 12,000 | 0 | 不可用 |
| 4 | Stage 4B | replay | 12,000 | 12,000 | 0 | 不可用 |
| 5 | Stage 5 | replay | 12,000 | 12,000 | 0 | 不可用 |
| 3 | Stage 4A | practice | 12,000 | **2,500** | 0 | 586 |
| 4 | Stage 4B | practice | 12,000 | **2,500** | 0 | 586 |
| 5 | Stage 5 | practice | 12,000 | **3,500** | 0 | 438 |
| 6 | Final A | practice | 12,000 | **5,500** | 0 | 1,494 |
| 7 | Final B | practice | 12,000 | **3,000** | 0 | 799 |

- stage slot 名取自 `third_party/th08/scripts/replay_file.py:10-11` 的 `STAGES`；
  `TH08_AUTOPLAY_STAGE`/`RESET` 用的是**槽位号**（Stage 5 = 5，游戏内第 6 面 Final A = 6），不是 1-based 标签。
  请求空槽位（3=4A 或 6）时引擎会自动跳到下一个非空槽（实测请求 3 进 4、请求 6 进 7）。
- practice 行"有效帧"远小于 12,000：**玩家静止不动会死**，死亡后 `STEP` 仍回 `advanced=1` 但 `f3` 冻结在 0，
  之后的"帧"是同一个世界被重复读（`BULLETS` 会一直报同一个 count）。所以 practice 的 12,000 帧里只有前
  2,500–5,500 帧是有效覆盖，表里的 `BULLETS` 计数也不能当"这一面有这么多弹"用。
- 4A / 4B 的 practice 统计**逐位相同**（有效帧 2,500、`BULLETS` 最大 586），本卡没有解释；可能 practice 选择
  4A/4B 时落到同一段脚本，若要用于数据采集必须先澄清（§9）。

---

## 5. 字段清单与证据

`LASERS` 行的实际形态（Final B 第 9,090 帧，原始字节）：

```
count=10
0 | 1 | 434148bb | 42f03d45 | 00000000 | bfb462d1 | 00000000 | 408ccccd | 00000000 | 41200000 | 1
1 | 1 | 43466e3e | 42b1128d | 00000000 | bfb462d1 | 00000000 | 408ccccd | 00000000 | 41200000 | 1
```

按 `%08x` 还原成 float 后：

| # | 名称 | 第 9,090 帧样例 | 第 9,450 帧（峰值）样例 | 说明 |
|---|---|---|---|---|
| 0 | `slot` | 0,1,2,… | 0,1,2,3 | 0x100 池的槽号，升序 |
| 1 | `state` | 1 | 1 | 发射态 |
| 2 | `posX` | 193.28 | 13.47 / **-14.76** | playfield 坐标，可为负 |
| 3 | `posY` | 120.12 | 223.42 | |
| 4 | `posZ` | 0.0 | 0.0 | |
| 5 | `angle` | -1.4093 rad | 2.6508 rad | 旋转角 |
| 6 | `startOffset` | 0.0 | 572.4 | 起始偏移（激光生长时从 0 起） |
| 7 | `endOffset` | 4.4 → 8.8 | 752.4 | 末端偏移（逐帧拉长） |
| 8 | `currentWidth` | 0.0 | **0.0** | 观测中恒 0，**不能当厚度用** |
| 9 | `width` | 10.0 | 10.0 | 全宽；§2.1 的半厚 = `width/4` |
| 10 | `timer` | 1 → 2 | 171 | 逐帧 +1 |

**没有第 12 个 token `inUse`**：本 build 的 `LASERS` 行宽是 **11**，`classify_layout` 的判定是
`verifiedPrefix: true`（`documented[0:11]` 逐 token 通过取值域校验）、`verified: false`
（完整 12 字段布局未被证明）、`missingDocumentedSuffix: ["inUse"]`。
成因可从 `BULLETS` 侧交叉印证：`BULLETS` 行只有 **9** 个 token，而文档布局是 11 个
（`collisionSize` / `collisionDisabled` 是同一批后续追加的字段）——两个命令都早于那次追加。

对 HazardTape 的直接含义：

1. 不要从行里读 `inUse`；**能出现在行里就说明是 in-use**（引擎按池过滤后发出，`headerPagedMismatchFrames = 0`：
   1,108 个翻页帧里 header `count` 与实际发出的行数完全一致）。
2. 厚度取 `width/4`（观测 `width = 10.0` → 半厚 2.5）；`currentWidth` 在本次所有观测里都是 0，**不要用**。
3. 中心点按 §2.1 的公式：`position.x + startOffset + (endOffset-startOffset)/2` 再按 `angle` 旋转；
   观测里 `endOffset - startOffset` 从 4.4 长到 180（峰值帧 752.4-572.4），说明**必须按帧读几何**，不能只记一次。
4. 一条激光从 spawn 到长成要很多帧（timer 171 时长度 180），HazardTape 若只按 `state`/`inUse` 判定会漏掉"还没长出来"的帧，
   建议按 `startOffset..endOffset` 的实际矩形 + `timer` 一起记。

---

## 6. 事件提取前提（P1 bomb 规则）

`STATE` 的实际回包只有固定 25 个字段 + `frames`：

```
OK frames=0 f0=0 f1=0 … f24=1
```

`events`/`hit` 类字段数量 **0**；`EVENTS`、`PLAYER`、`HAZARDS` 在本 build 都是 `ERR unknown-command`。

**因此：11-P1 §4 需要的 `PLAYER_HIT`（`frameEventFlags & 0x2`）当前不可得。**
决死 bomb 现在只能靠"下一帧 `f13 playerState` 变成非 ALIVE"这种**滞后一帧**的替代信号，而 §4 的窗口本身只有
`bombs×6(+7)` 帧且输入还有 1 帧延迟——滞后一帧就可能已经错过窗口。

最小扩展建议（本 slice **没有改任何代码**）：

1. 只读命令 `EVENTS`（或 `STATE` 行末追加 `ev=<hex>`，**不要**进那 25 个被 hash 的字段，否则 G0 的 hash 全变）：
   `g_ReplayManager ? g_ReplayManager->frameEventFlags : 0`。
   来源：`ReplayManager.hpp:213`（`u16 frameEventFlags`，`C_ASSERT` 偏移 `0xda`）；`REPLAY_FRAME_EVENT_PLAYER_HIT = 0x0002`
   （`ReplayManager.hpp:163`）；置位点在 `Player.cpp:329 / 356 / 466`。
2. 边界：`g_ReplayManager` 在 replay 链被切断后是野指针，`TrackedFields` 已经用
   `g_ReplayManager && flags.isReplay` 保护过同类读取——新命令必须用同样的守卫，取不到就回 `ev=undefined`，
   **不要**回 0（0 会被误读成"没有事件"）。
3. 若还要给像素学生用，再决定是否进 SHM v2；本 slice 不建议改 SHM v1 布局。

---

## 7. 单帧耗时（HazardTape 预算）

在同一轮真实对局里测（每项 200 帧，`time.perf_counter()` 包住一次请求-应答）：

| 场景 | 调用 | 平均 | 中位 | p95 | 最大 |
|---|---|---|---|---|---|
| Stage 1 normal，帧 1–200 | `LASERS 0 16`（不翻页） | **0.032 ms** | 0.029 | 0.052 | 0.099 |
| 同上 | `BULLETS 0 16` | **0.038 ms** | 0.032 | 0.057 | 0.318 |
| 同上 | 两者合计 | **0.070 ms** | 0.060 | 0.104 | 0.390 |
| 同上 | `STEP 0 1`（单帧推演） | 0.083 ms | 0.075 | 0.128 | 0.251 |
| Final B，帧 9,301–9,500（激光均值 104.6 条） | `LASERS` **完整翻页** | **0.292 ms** | 0.284 | — | 0.764 |

- 翻页探针：200 帧里平均 **7.0 页**、最多 **14 页**、单帧最多 **210 条**；`meanPages` 直接决定成本。
- 预算：峰值帧 `STEP + 完整翻页 LASERS + BULLETS ≈ 0.083 + 0.292 + 0.038 ≈ 0.41 ms/帧`，
  即读取路径本身可跑到 ~2,400 帧/s（远高于 60 fps 实时要求）；`STEP` 才是主成本。
- 注意：这是 `--rl-role shadow` + 空载机器上的数字；零售 Windows 长跑（trace 批采集）同时在跑时会有噪声。

---

## 8. 附带发现（会影响 HazardTape 实现）

### 8.1 replay 播放期间 `BULLETS` 不可用

```
BULLETS 0 16 -> ERR bullets-not-ready reset-required
```

`HandleBullets` 的门是 `g_resetReady && globals`，而 `g_resetReady` 只在 `RESET` 后为真；replay 播放不走 `RESET`。
后果：**想从 replay 采 HazardTape，子弹侧拿不到**（激光侧可以，`HandleLasers` 只查 `globals`）。
两条可选路：给 `BULLETS` 加"只读、不依赖 reset"的分支（要单独评审），或走 `BRANCH` 影子进程在正常局里录。

### 8.2 `BULLETS count` 恒含 1 个 sentinel 槽

`BULLET_STATE_SENTINEL = 6` 占 `bullets[0x600]`，而过滤条件是 `state != UNUSED`，于是每帧 `count` 至少为 1：

```
frame 1:   count=1  -> ["1536","6","0","00000000","00000000","00000000","00000000","00000000","0"]
frame 500: count=7  -> 最后一条仍是 slot 1536 / state 6 / 全 0
```

直接按 `count` 记 HazardTape 会在**每一帧**多出一个位于 (0,0) 的幽灵子弹。本工具用
`sentinelAdjustment=1` 给出修正值（JSON 同时保留 `rawMaxCount`/`rawNonEmptyFrames`）：
Stage 1 的"非空帧 2,582 / 最大 56"是修正后的数字，**原始 header 是每帧都 ≥1、最大 57**（3,000 帧全部"非空"）。

### 8.3 practice 死亡后世界冻结，但 `STEP` 仍成功

`f3 gameplayFrameCounter` 在死亡后冻结为 0，`STEP 0 1` 依然回 `advanced=1`。
采集器必须用 `f3`（或 `DEBUG curState`）判定"世界还在推进"，否则会把同一帧重复记成上千帧。
本工具在 JSON 里给出 `lastLiveSampleFrame` / `frozenTailSamples`。

### 8.4 practice Final B ≠ replay Final B（未解释）

replay Final B 在 9,090–10,500 帧有激光；practice Final B 给到 **20,000 帧**（另跑一次，见
`.tmp_v400_p1s1/probe_practice_finalb.json`）**一条都没有**，且 `BULLETS` 峰值停在 799。
这两者不是同一段脚本（或 practice 从别的检查点开始）。本卡只报事实，不做因果推断。

---

## 9. 缺什么 / 下一步

1. **Stage 5 与"后期面 = 有激光"的假设要修正**：本语料里只有 Final B replay 命中。若要更广的激光覆盖，
   应扫更多 replay（尤其 Final A / Final B 通关档），而不是继续加长 Stage 5。
2. **replay 侧的 `BULLETS` 门**（§8.1）是 P1 用 replay 产 HazardTape 的硬阻塞；要么改协议（需评审），
   要么只走 `BRANCH` 影子进程的正常局。
3. **`inUse` 缺失**（§5）意味着 HazardTape 不能依赖该字段；如果后续要升级到带 `inUse` 的构建，
   需要重跑本卡做字段对照。
4. **PLAYER_HIT 不可得**（§6）：决死 bomb 规则在扩展落地前只能 fail-closed（不放 bomb）或接受滞后一帧，
   两种都要在 P1-WP2 里写清楚。
5. **`currentWidth` 恒 0** 这一点只覆盖了 Final B 的一个弹幕；收点线/ECL 光束是否会用别的字段，未验证。
6. practice 4A/4B 统计逐位相同（§4），以及 practice Final B 无激光（§8.4），都需要单独定位后再用于数据。

---

## 10. 复现与产物

```sh
# 1) 单测（Windows 侧即可，不需要引擎）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p1s1_pytest -q tests/test_v400_p1_laser_gate.py
# -> 24 passed

# 2) 真实对局（WSL，串行单引擎）
wsl.exe -d Ubuntu -- sh /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p1s1/run_gate.sh
# -> exit=0，60 s，15 次引擎启动
```

| 产物 | 路径 |
|---|---|
| 完整 JSON（4 个显式来源 + 10 个扫描槽 + 逐 500 帧取样 + 字段证据 + 耗时） | `.tmp_v400_p1s1/laser_gate.json` |
| 运行日志（含 SHA-256 与 `exit=0`） | `.tmp_v400_p1s1/run.log` |
| 命令记录 | `.tmp_v400_p1s1/run_gate.sh` |
| 每个来源的引擎 stderr | `.tmp_v400_p1s1/stderr/*.stderr.txt` |
| practice Final B 20,000 帧补测 | `.tmp_v400_p1s1/probe_practice_finalb.json` |
| 工具 | `tools/rl/v400_p1_laser_gate.py` |
| 单测 | `tests/test_v400_p1_laser_gate.py` |

## 修订记录

| 日期 | 修改 | 原因 |
|---|---|---|
| 2026-11-09 | 初版 | P1-S1 激光几何前置门控 |
