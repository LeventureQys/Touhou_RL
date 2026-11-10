# EXP-P0-056：WP8 真实双端像素配对首例（同一 `.rpy` / 同一 stage / 同一 logicFrame）

- 日期：2026-11-08
- 环境：Windows `sys.platform=win32`（RTX 4070 SUPER / 驱动 617.42，交互式 session 1），
  WSL Ubuntu（`xvfb-run` 私有 Xvfb + Mesa 软件 GL）
- Windows 侧：`third_party/th08/build_fence/th08.exe`，905,728 B，
  SHA256 `bc1894fed48beac739f628836629b8b1b1c1b63dbe0097239c2762cb4c8d8565`，
  `.text` SHA256 `34A8070B4854672C92A66C95472F9DF7FB3B83C0AFEB6FBD15BBAED45A8593C7`，
  地址表 `79237a295567edf0bad6c557bcc1aaf8a7d1db369d2c54429fbc1b779b85634a`，
  `th08.map` `ce7b6672f9488909a5f700e6a52900c65358a4201b0de5e62ee904f8a752bb54`
- Linux 侧：`/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，4,807,320 B，
  SHA256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`，
  源码 HEAD `101ce5eb9f442e9fbfd5e7e10e99495b4873d1ff`，数据 `/root/v400-data/game`
- 回放：`th08_replay/Th8_ud2970.rpy`，41,398 B，
  SHA256 `e621086d814fba4838a2c84e4081162524c3c7e84d1bc4ca3f6621f9ca52d96f`，
  stage 0，difficulty 1（Normal），shot 6（两侧 `DEBUG` 一致）
- 前置：EXP-P0-043（FAIL-CLOSED）、EXP-P0-047/053（Windows 研究构建可采帧）

## 结论

**完成首例。** 同一份 `.rpy`、同一 stage、同一 `logicFrame` 区间 **304–351（48 帧）**，
Windows 帧栅栏渲染与 Linux harness 渲染双双落盘为 640×480 RGBA 无损 PNG，
逐像素统计写入 `.tmp_v400_wp8_pair_report/pixel-report.json`。

实测（48 对，RGB 通道）：

| 区域 | 平均绝对差 | 差异像素比例 | 最大通道差 |
| --- | --- | --- | --- |
| playfield `x=[32,416)` | **0.130595**（min 0.126935 / max 0.134769） | **0.298965**（max 0.307617） | **13** |
| HUD（其余列） | **0.654780**（min 0.373844 / max 0.725345） | **0.187075**（max 0.187614） | **251** |
| HUD 去掉 fps 读数（附加） | 0.322811 | 0.143684 | 15 |
| 整帧去掉 fps 读数（附加） | 0.207039 | 0.134598 | 15 |

HUD 的 251 全部来自 **fps 读数**这一个矩形：Windows 研究构建打印被栅栏 hold 之后的
真实帧率（本窗口 48 帧出现 7 种不同读数，例如 `15.12fps`），Linux 侧该矩形 48 帧完全
一致（目视 `60.00fps`）。这是 **HUD 数值差异，不是渲染差异**，
报告把它单列为 `fps_readout_box` + `*_excluding_fps_readout`，但规定的 HUD 主指标仍按
任务要求保持原样（未掩蔽）。

**不宣称状态一致**：两侧状态摘要算法不同（Windows `bulletDigest`/`enemyDigest`，
Linux 25 字段 FNV-1a），`--state-pairing same-logic-frame` 只强制 replay SHA 与
`logicFrame` 集合一致，报告 `state_consistency = "not-asserted"` 且 48 帧
`state_sha256_equal` 全为 `false`。严格模式在同一份真实数据上 **exit 3**
（`BLOCKED: frame 304: state mismatch; cannot attribute pixel difference`），
证明默认严格语义未被放宽。

## 一、先验问题：`TH08_AUTOPLAY_REPLAY` 能不能在 harness 里真的放回放

**能**，但有两个必须先解决的前提。

1. **继承的 `DISPLAY` 不能用。** 本 WSL 会话 `DISPLAY=:0`（WSLg）；env 角色在该显示上
   `HELLO` 前就挂住（`.tmp_v400_wp8_pair_debug.sh`：40 s 超时、`poll=None`、无 stderr）。
   换成 `xvfb-run -a` 私有 Xvfb 后立刻应答。因此
   `tools/rl/v400_linux_frame_capture.py` 默认 `--xvfb auto` **强制重新 exec 到私有 Xvfb**，
   不信继承值。
2. **autoplay 不是首帧生效的。** `StartDeveloperReplayIfRequested()` 挂在标题画面的
   `TitleCurrentScreenState_Ready`（`TitleScreen.cpp:450`），必须先泵帧到 Ready 才会被消费。
   实测：harness 帧 58/59 时 `DEBUG` 才出现 `isReplay=1`，此刻
   `replay frameCounter=0`、`gameplayFrameCounter=0`。

实测证据（真实 run D，`.tmp_v400_wp8_pair_linux_d/capture.json`）：

```
HELLO  -> OK hello proto=1 harness=1 role=env frames=0
（泵帧到 autoplay 生效）DEBUG -> curState=2 setupState=0 subthread=0 globals=1 stage=0
                               difficulty=1 shot=6 isReplay=1 isDemo=0 frames=58
STATE  -> f2=0 (replay frameCounter)  f3=0 (gameplayFrameCounter)
```

即：**回放从 `logicFrame 0` 开始，`f2 == f3`**（回放帧计数与
`g_GameManager.gameplayFrameCounter` 同步），与 Windows 侧 `logicFrame` 的取值口径一致
（见 §二）。推进到 304–351 全部命中，48/48 无跳号。

真正被证伪的只有"继承 DISPLAY 直接跑"这一条；"用 `TH08_AUTOPLAY_REPLAY` 播放指定 `.rpy`
并推进到指定 `logicFrame`" 成立，因此不需要退回到"同一 practice/seed 起点"，也没有伪造同帧。

## 二、同一 `logicFrame` 的语义证据

- Windows：`framefence.cpp:514` `g_telemetry->logicFrame = g_GameManager.gameplayFrameCounter;`
- Linux：`rlharness.cpp` `TrackedFields()` 的 `f3 = g_GameManager.gameplayFrameCounter`，
  `f2 = g_ReplayManager->frameCounter`（仅回放期非零）

两侧是**同一个变量**，且都在关卡内从 0 起算。48 帧实测交叉核对：

| 核对项 | 结果 |
| --- | --- |
| Windows `stage_frames` == Linux `f3` == `logicFrame` | 48/48 |
| 玩家 `player_x` 浮点位相等（Windows `state.player_x` vs Linux `f10`） | 48/48 |
| 玩家 `player_y` 浮点位相等（vs `f11`） | 48/48 |
| `player_state` / `score` / `lives` / `graze` 相等 | 48/48 |

这是**时间对齐证据**，不是状态等式证明：本窗口处于 Stage 1 开场、玩家未移动
（x=376, y=384，score=0，graze=0，lives=2），所以 `player_x/y` 相等本身较弱；
它能排除"两帧错位一格"，但不能替代状态一致性验证。

窗口不是静止画面：同一侧 `logicFrame 304 → 351` 的 playfield 变化像素数
Windows 137,250 / Linux 136,815（307,200 个 playfield 像素中），两侧相差 435 px（0.14%），
说明两边在同样的帧号上播放着同一段演化中的动画；跨侧同帧 playfield MAD 在 304 帧为
0.127170。

## 三、真实命令与退出码

### 3.1 Windows 采集（研究构建，48 帧）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python .tmp_v400_wp8_pair_win_capture.py --out .tmp_v400_wp8_pair_win_d `
  --run-dir .tmp_v400_wp8_pair_win_d_run --replay-file th08_replay/Th8_ud2970.rpy `
  --replay-stage 0 --frames 48
```

- 退出码 **0**；`capture.wallSeconds=16.312`，PowerShell 计时 19.3 s（单次 ≤180 s 满足）
- `framesCaptured=48`，`logicFrame` **304–351** 连续，`aliveAtFrame=302`
- 48/48 帧 `checks_all_pass=true`（13 项栅栏校验：`hold_stable`/`pixels_frozen`/
  `pixels_match_backbuffer`/`bullet_digest`/`enemy_digest`/`gameplay_frame`/`stage_frames` …）
- 运行前后 `Get-Process th08` 均无输出（无残留进程）

### 3.2 Linux 采集（env 角色 + 私有 Xvfb）

```powershell
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && python3 tools/rl/v400_linux_frame_capture.py capture \
  --out .tmp_v400_wp8_pair_linux_d --targets 304,305,...,351 \
  --replay /root/v400-data/pair/Th8_ud2970.rpy --stage 0"
```

- 退出码 **0**（`status=captured`，`framesCaptured=48`）
- 引擎子进程 `exit_code=0`，`wallSeconds=14.376`，`display=:107`（xvfb-run 自动分配）
- `replay_start.harness_frame=58`，`replay_frame_counter=0`
- 共享内存：`magic=0x30303456`、`version=1`、`640×480`、`flags` 同时带 state/pixels 有效位；
  每帧都校验 `header.frame == STATE frames` 与 `header.stateHash == HASH 回复`

### 3.3 RGBA PNG 落盘（Windows 侧转换，逐字节校验）

```powershell
python tools\rl\v400_linux_frame_capture.py convert --capture .tmp_v400_wp8_pair_linux_d\capture.json
```

- 退出码 **0**；48 张 `frames/NNNNN.png`，RGBA 640×480
- 转换时校验：raw 字节 SHA256 未变 → 行翻转（OpenGL 自下而上 → 自上而下）→ PNG
  回读与数组 `array_equal`（无损）→ 图片模式/尺寸
- 原始 SHM RGBA 保留在 `raw/NNNNN.rgba`（48×1,228,800 B = 58,982,400 B），
  `raw_rgba_sha256` 与 `image_sha256_rgba` 同时入 manifest，翻转可审计

### 3.4 配对比较（新模式）

```powershell
python tools\rl\v400_pixel_pair.py `
  --windows .tmp_v400_wp8_pair_win_d\windows-manifest.json `
  --linux   .tmp_v400_wp8_pair_linux_d\linux-manifest.json `
  --frames 48 --state-pairing same-logic-frame `
  --out .tmp_v400_wp8_pair_report\pixel-report.json
```

- 退出码 **0**，stdout：`MEASURED 48 frames (same-logic-frame); …` + 显式 WARNING
- 报告：`schema=v400-wp8-paired-pixel-audit/2`、`status=measured-not-G0-pass`、
  `state_pairing=same-logic-frame`、`state_consistency=not-asserted`、`pairs=48`

### 3.5 严格模式对照（同一份真实数据）

```powershell
python tools\rl\v400_pixel_pair.py --windows ...windows-manifest.json --linux ...linux-manifest.json `
  --frames 48 --out .tmp_v400_wp8_pair_report\pixel-report-strict.json
```

- 退出码 **3**；stderr `BLOCKED: frame 304: state mismatch; cannot attribute pixel difference`
- **未写出任何报告文件**（fail-closed；该 `--out` 路径实际不存在）
- 说明：默认 `strict` 未被削弱；新模式必须显式给出

### 3.6 单元测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp8_pair_pytest_final -q tests/test_v400_pixel_pair.py
```

- 退出码 **0**，`11 passed`

### 3.7 早期 24 帧探路（同一回放，303–326）

在正式 48 帧之前先用 3 帧、24 帧各跑通一次（均退出码 0），用于确认工具链；48 帧 run
取代其成为正式证据。24 帧那一轮统计（旧摘要口径）：playfield MAD 0.130443 / 比例 0.298642。

## 四、变更文件

| 文件 | 变更 |
| --- | --- |
| `tools/rl/v400_pixel_pair.py` | 新增 `--state-pairing {strict,same-logic-frame}`（默认 strict）；报告 schema 升到 `/2`，新增 `state_pairing`/`state_consistency`/`warnings`/`aggregate`/`fps_readout_box`/`aux_regions`，逐帧保留两侧 `state_evidence`、`state_sha256_equal`；strict 仍要求 `state_sha256` 逐帧相等 |
| `tests/test_v400_pixel_pair.py` | 11 项：strict 默认仍拒绝不一致；same-logic-frame 仍强制 replay SHA 与 `logicFrame` 集合；新模式不宣称状态一致、保留 evidence、CLI 暴露模式；未知模式被拒绝 |
| `tools/rl/v400_linux_frame_capture.py` | **新增**。`capture`（WSL 内驱动 env 角色、autoplay 回放、按 `logicFrame` 精确定位、落原始 SHM RGBA）/ `convert`（Windows 侧翻转 + 无损 RGBA PNG + manifest） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P0-056-WP8-paired-frames-first.md` | 本卡 |
| `.tmp_v400_wp8_pair_win_capture.py`、`.tmp_v400_wp8_pair_*.sh`、`.tmp_v400_wp8_pair_*` | 采集脚本、探针脚本与证据目录（临时区，未提交） |

未改动：`third_party/th08` 源码/构建、零售原件、依赖、协议/ABI；未删除或提交任何仓库文件。

## 五、统计明细

报告：`.tmp_v400_wp8_pair_report/pixel-report.json`（形状同 schema `/2`）。
逐帧数值范围：

- playfield MAD：0.126935 – 0.134769，均值 0.130595；差异像素比例 0.290489 – 0.307617
- HUD MAD：0.373844 – 0.725345，均值 0.654780；差异像素比例 0.183455 – 0.187614
- 最大通道差：playfield 13；HUD 251（仅在 fps 矩形内）；去掉 fps 矩形后全帧 15
- 最差帧（按 HUD MAD）：`logicFrame` **330–339**（HUD MAD 0.725345、差异比例 0.187598、
  最大通道差 251），其后 340–351 相近

`fps_readout_box`（实测定位，非假设）：`y=[463,480)`、`x=[512,581)`。
判据：48 帧并集里**只有**该矩形出现 >16/255 的通道差；矩形外最大通道差恒为 15。
同窗口内 fps 矩形的不同渲染数：Windows **7 / 48 帧**，Linux **1 / 48 帧**（目视恒定 `60.00fps`）。

## 六、双侧确定性（同回放、同 stage、两次独立运行）

| 侧 | 重叠帧 | 像素一致 | 状态摘要一致 |
| --- | --- | --- | --- |
| Windows run d vs run b | 47（304–350） | 去掉 fps 矩形 **47/47** 逐字节相同；整帧 0/47（仅 fps 数字变化） | `state_sha256` 47/47 |
| Linux run d vs run b | 47（304–350） | 原始 SHM RGBA **47/47** 逐字节相同 | `state_sha256` 47/47 |

说明：`state_sha256` 只覆盖**游戏状态**——Windows 排除 `digest_ticket`（采集票据），
Linux 排除 `harness_frame`、`fields.f0`（进程泵帧计数）与 `fnv1a64`（harness 自己的
FNV-1a 把 `f0` 折进去，因而逐次运行不同）。被排除项仍原样保留在 evidence 里，
两侧 evidence 都在报告中逐帧保留。若不排除，两次运行的摘要会不同（已实测并修正）。

## 七、限制（必读）

1. **Windows 侧是研究构建，不是零售原件。** `build_fence/th08.exe`（905,728 B）自带帧栅栏
   插桩并重新编译；零售原件（840,704 B）本轮未参与。任何"Windows 行为 = 零售行为"的
   推断在本卡中都不成立。
2. **状态一致性未建立。** Windows 只给出 `bulletDigest`/`enemyDigest`/玩家位模式/分数等，
   Linux 给出 25 字段与 FNV-1a；两者不是同一摘要函数，也没有做字段级交叉编码。
   因此**像素差异不能完全归因于渲染**，报告不得被读作"状态已确认一致"。
   本卡只提供了 §二 的**时间对齐**证据（同变量计数、玩家坐标位相等），不是状态等式证明。
3. **样本量有限。** 1 份回放、1 个 stage（stage 0）、1 个连续窗口（304–351，48 帧）、
   1 个难度/机体、1 次双端配对。窗口位于 Stage 1 开场（标题卡仍在屏、无弹幕/敌机压力），
   未覆盖中段弹幕、Boss、Bomb、暂停等场景。
4. **两侧渲染栈不同。** Windows 是 D3D8 + 真实 GPU（RTX 4070 SUPER/617.42）；Linux 是
   Xvfb + Mesa **软件** GL（`LIBGL_ALWAYS_SOFTWARE=1`、`LP_NUM_THREADS=1`）。所以
   0.13 的 playfield MAD 至少混合了"实现差异 + 驱动/光栅化差异"。
5. **HUD fps 读数按构造不同。** Windows 打印被栅栏 hold 后的真实帧率（本窗口 48 帧
   7 种读数，`15.12fps` 一例），Linux 侧该矩形 48 帧恒定（目视 `60.00fps`）；
   这是数值差异，不是渲染差异。HUD 主指标仍包含它，
   看 HUD 数值时必须同时看 `hud_excluding_fps_readout`。
6. **像素来源不同。** Windows PNG 是栅栏 hold 期间的窗口 GDI 抓屏（并校验与游戏自身
   backbuffer 副本逐字节相同），alpha 是合成的常量 255（栅栏像素源是 RGB24）；
   Linux PNG 是 GL FBO 的 `glReadPixels` RGBA（alpha 是真实 GL alpha，且在绘制后、
   Present 前由 `AfterDrawFrame` 发布）。比较只用 RGB。
7. **不是 G0 通过。** 报告 `status=measured-not-G0-pass`；G0.4 仍需三份真实回放与
   人工目视复核。
8. **fps 矩形的定位是经验值**（本窗口实测唯一 >16 的超差矩形，向四周各留 1–2 px）。
   换分辨率/字体/HUD 布局后必须重新测量，不能当作常量。

## 八、剩余阻塞

1. **零售原件与授权采集**：本轮只有研究构建；零售 Windows 采集路径仍未打通
   （EXP-P0-043 的 fixture/data 缺口不变）。
2. **跨实现状态摘要**：要么把两侧映射到同一组可比较的状态投影（例如
   stage/`gameplayFrameCounter`/player x,y/score/lives/graze/power/point items 的
   规范化编码），要么定义逐字段容差并写明证明范围；在拿到之前，像素归因只能停在
   "时间对齐 + 渲染差异候选"。
3. **样本扩展**：至少三份不同回放 × 多个 stage × 含弹幕/Boss 的窗口，并覆盖
   practice/seed 起点；当前 48 帧只够证明"首次真实配对可复现"。
4. **fps 掩蔽政策**：需要在 WP8 验收口径里明确"HUD 主指标是否掩蔽 fps 读数"，
   否则 HUD 的 251 会长期主导最大通道差。
5. **G0.4 视觉复核**：48 帧已有人眼可读的并排图
   （`.tmp_v400_wp8_pair_shot_304.png`、`..._shot_330.png`、`..._diff_330.png`、
   `..._fps_crop_330.png`），但尚未形成评审记录。

## 九、证据路径

- 报告：`.tmp_v400_wp8_pair_report/pixel-report.json`（179,668 B；strict 对照因 exit 3 未落盘）
- Windows：`.tmp_v400_wp8_pair_win_d/windows-manifest.json` + `frames/000{304..351}.png`
  （另有两轮独立重复：`.tmp_v400_wp8_pair_win_b`、`.tmp_v400_wp8_pair_win_c`）
- Linux：`.tmp_v400_wp8_pair_linux_d/capture.json`、`linux-manifest.json`、
  `raw/*.rgba`、`frames/*.png`（重复轮：`.tmp_v400_wp8_pair_linux_b`）
- 采集脚本：`.tmp_v400_wp8_pair_win_capture.py`；探针脚本 `.tmp_v400_wp8_pair_probe.sh`、
  `.tmp_v400_wp8_pair_orient.sh`、`.tmp_v400_wp8_pair_debug.sh`
- 并排/差分图：`.tmp_v400_wp8_pair_shot_304.png`、`.tmp_v400_wp8_pair_shot_330.png`、
  `.tmp_v400_wp8_pair_diff_330.png`、`.tmp_v400_wp8_pair_fps_crop_330.png`
