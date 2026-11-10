# EXP-P1-001 harness 逐帧致死物导出（BULLETS/ENEMIES/LASERS 追加字段 + PLAYER + HAZARDS）

- 日期 / 工作包：2026-11-09 / P1-WP1（前置：G0 训练场）
- 假设：`build/rl` 已有的 `BULLETS`/`ENEMIES`/`LASERS` 只要**追加**字段并新增 `PLAYER` 与批量 `HAZARDS`，就能提供 [11-P1 §2.1](<../11-P1-搜索教师.md>) 需要的全部逐帧致死物几何，且不改变游戏逻辑与 25 字段 `HASH`。
- 预注册（主指标，六项，全部在运行前固定）：
  1. 追加字段在真实帧上非零且与源码常量一致：子弹 `collisionSize` 为**全宽**（4.0 / 6.0）、`collisionDisabled` 为 0、敌人 `hitboxDimensions` 为 24.0、自机 `hurtboxBounds` 半宽 0.825；
  2. `HAZARDS` 的 `nb`/`nl`/`ne` 与 `BULLETS`/`LASERS`/`ENEMIES` 的 `count=` 逐帧相等；
  3. `PLAYER` 的四个速度等于当前机体 `.sht` 数据（team 0 = 4.0 / 2.8284 / 2.0 / 1.4142），读不到时必须 `ERR player-speed-unavailable`；
  4. 扩展前后 `HASH` 前 100 帧**完全一致**；
  5. `build/rl` 的 SHA 保持 `591c521d…`；
  6. 旧字段前缀**字节不变**（用真实行的前缀切片在解析层验证）。
  通过阈值 = 六项全部成立。样本量 = 每个新字段 ≥ 1 次真实运行 + 31 项单元测试。
- 设置：
  - 源码：`/root/Touhou_RL/third_party/th08-rl/src/rlharness/rlharness.cpp`（sha256 `ec333a04…` → `18fb7294…`，34853 → 46642 字节，1029 → 1283 行；非 ASCII 字节数前后都是 21，原有注释一个字节没动）
  - 新构建目录：`third_party/th08-rl/build/rl_hazard`，`cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTH08_RL_HARNESS=ON -DTH08_ENABLE_REPLAY_TRACE=OFF -DTH08_DX8_SDK_ROOT=…/scripts/prefix/mssdk`
  - `CMakeCache.txt` 关键项与 `build/rl` **逐项相同**：`CMAKE_BUILD_TYPE=Debug`、`CMAKE_CXX_COMPILER=/usr/bin/c++`、`CMAKE_GENERATOR=Ninja`、`CMAKE_MAKE_PROGRAM=/usr/bin/ninja`、`TH08_RL_HARNESS=ON`、`TH08_ENABLE_REPLAY_TRACE=OFF`、`TH08_DX8_SDK_ROOT`、`CMAKE_HOME_DIRECTORY`
  - 说明：`TH08_DX8_SDK_ROOT` 指向的 `scripts/prefix/mssdk` 在本 checkout 里**不存在**（`build/rl` 的缓存里也是同一个不存在的值），它只在 `WIN32` 分支被使用（CMakeLists.txt:15），Linux 构建不受影响。为了"照抄"才原样传入，不是新引入的依赖。
  - 二进制：`build/rl_hazard/th08-modern`，4826176 字节，sha256 `8053dfaa3d5cde0f720ba31d7f63aeba1db929a6721999c1b95e8ac00ba7b32a`
  - 对照二进制：`build/rl/th08-modern`，sha256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`（构建前后各测一次，未变）
  - 运行环境：WSL2 Ubuntu / xvfb `:99` / `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1 SDL_AUDIODRIVER=dummy`，`--rl-role env`。`--rl-role shadow` 在本 checkout 里跑不起来：`DISPLAY` 为空时预检通过，随后 `shadow null-device backend unavailable; no window or GL created` 退出（`src/rlharness/null_device.cpp` 目前只有 `NullDevicePreflight`，没有真正的空设备后端）。本次所有真实数据都来自 `env` 角色。
  - 关卡：`RESET <stage> <1|3> 0 1 0`，stage 0 normal 与 stage 4/6/7 Lunatic；seed 固定 1
  - 数据目录：每次运行前用 `score.dat.orig`/`th08.cfg.orig` 重建，`th08.dat`/`thbgm.dat` 用软链，避免 score.dat 被写回后污染下一次运行
- 结果：**六项全部成立**。

## 1. 新增/修改的命令与字段

追加规则：G0 客户端已解析的字段位置与编码**一位不动**，新字段一律追加在行尾。`STATE`/`HASH` 的 25 字段完全未触碰。

| 命令 | G0 行格式（不变的前缀） | 追加字段 |
|---|---|---|
| `BULLETS [offset [limit]]` | `slot:state:color:posXyz:velXy:timer` | `collisionSize`（`%08x` floatBits，**全宽**，单位像素）、`collisionDisabled`（整数，0 = 有判定） |
| `ENEMIES [offset [limit]]` | `slot:flags1:posXyz:life:timer3318` | `worldPositionXyz`（3×`%08x`）、`hitboxWidth`、`hitboxHeight`（`%08x` floatBits，**原始** `hitboxDimensions`） |
| `LASERS [offset [limit]]` | `slot:state:posXyz:angle:startOffset:endOffset:currentWidth:width:timer` | `inUse`（整数） |
| `PLAYER`（新） | — | `hp lives bombs power focus isYoukai playerState invulnState px py hurtboxMinX hurtboxMinY hurtboxMaxX hurtboxMaxY speedNormalAxial speedNormalDiagonal speedFocusedAxial speedFocusedDiagonal` |
| `HAZARDS [limit]`（新） | — | `frames nb nl ne bullets=<行,…> lasers=<行,…> enemies=<行,…>` |

要点：

- **`collisionSize` 是全宽，不是半宽。** 直接来自 `Bullet::sprites.collisionSize.x`，未做任何缩放。`BulletManager.cpp:1630-1706` 的每一处赋值都把 `x` 与 `y` 设成同一个值（4 / 6 / 5 / 8 / 10 / 24），所以只导出一个分量；判定半宽由调用方取 `collisionSize / 2`（`Player::CheckLethalCollision` 里 `position ± size/2`，Player.cpp:345-348）。这正是 [11-P1 §2.1](<../11-P1-搜索教师.md>) 指出的旧教师两倍误差的来源。
- **敌人 `hitboxWidth/Height` 是原始尺寸。** `EnemyManager.cpp:852` 会把它 `/1.5` 后交给 `CheckLethalCollision`，而后者把参数当**全宽**用，因此致死半宽是 `hitboxDimensions / 3`，不是 `/1.5`。
- **`worldPosition` 不是 `position` 的复制。** stage 0 第 1440 帧的 slot 2 实测 `position=(16.5638, 61.8194)`、`worldPosition=(219.7726, 186.9997)`（带父级位移的敌人），而 `flags1=0x11003349` 的 bit2 为 0，所以它**不是**致死物——几何侧必须同时看 `worldPosition` 和 `flags1` bit2。
- **`PLAYER` 的速度不是常量。** 直接读引擎自己会用的两个文件：不聚焦用 `primaryShtFile`，聚焦用 `secondaryShtFile`（Player.cpp:783-806）。任一为空就返回 `ERR player-speed-unavailable`，不猜默认值。这个分支在真实运行里被触发过：stage 7 Lunatic 跑完 3000 帧后 `PLAYER` 返回 `ERR player-speed-unavailable primary_sht=0 secondary_sht=0`。
- **`hp` 与 `lives` 的关系**：TH08 没有独立血条，`ZunGlobals::livesRemaining` 就是耗尽即死亡的量，所以 `hp` 给它的原始 floatBits、`lives` 给整数部分，不构造新量。
- **`HAZARDS` 一行就是整帧。** 三条列表用的是与 `BULLETS`/`LASERS`/`ENEMIES` **完全相同**的行编码（含本次追加字段），因此一套解析器覆盖两者，`nb` 与 `BULLETS` 的 `count=` 可以直接比对。回复在固定 256 KiB 缓冲里组装后一次 `fwrite`（控制通道是 `_IONBF`，逐行 `fprintf` 会变成每个致死物一次 `write` 系统调用）；装不下就 `ERR hazards-too-large`，绝不截断成一条看似完整的帧。

## 2. 新旧构建 HASH 前 100 帧

同一输入序列（`RESET 0 1 0 1 0` + 100×`STEP <mask> 1`，mask 按 `[0,1,2,4,8,16,32,64,3,9,12,10,6,5,1,2]` 循环），两个二进制各用一份独立数据目录：

```text
exit_old=0  exit_new=0
hash 采样数                     : 102 / 102（RESET 帧 0 + 100×STEP + 末尾 HASH）
all_hashes_identical            : true      ← 全部 102 个逐字节相同
prefix_hashes_identical (前 101) : true
first_divergence_index          : null
frame0 hash                     : 7f04c7361749ed9f
frame100 hash                   : b7f16ce3ce844995
final hash                      : b7f16ce3ce844995
OK advanced= 回复               : 100 条逐字节相同
```

`build/rl` 甚至不认识新命令，这是它没被改动的直接证据：

```text
build/rl        : ERR unknown-command HAZARDS
build/rl_hazard : OK frames=… nb=… nl=… ne=…
```

## 3. 真实帧上的字段值

`RESET 0 1 0 1 0` 后的真实读数（team 0，判定中心 192,384）：

```text
frame 0   : hp=2.0 lives=2 bombs=3.0 power=0.0 playerState=3 invulnState=1
frame 300 : playerState=0 invulnState=0
frame 900 : hp=2.0 → 被弹后 playerState=3 invulnState=1
frame 1440: hp=0.0 lives=0 playerState=2（DYING）
hurtbox   : min=(191.175, 383.175) max=(192.825, 384.825) → 半宽 0.825
speed     : 40800000=4.0  403504f3=2.828427  40000000=2.0  3fb504f3=1.414214
```

四速与 [11-P1 §2.2](<../11-P1-搜索教师.md>) 里 team 0 的实测值一致，而它们是**从 `.sht` 读出来的**（源码路径 `PlayerRawShtFile::normalAxisSpeed` 等），代码里没有任何 4.0/2.8284 的字面量。

子弹 `collisionSize` 在真实数据里出现 4.0（`40800000`）与 6.0（`40c00000`）两种，`collisionDisabled` 采样全为 0；敌人 `hitboxDimensions` 为 24.0（`41c00000`）。

计数一致性（`HAZARDS` vs 分页命令的 `count=`，逐帧全等）：

| 帧 | `nb` / `BULLETS count` | `nl` / `LASERS count` | `ne` / `ENEMIES count` |
|---|---|---|---|
| stage 0 f0 | 1 / 1 | 0 / 0 | 1 / 1 |
| stage 0 f300 | 1 / 1 | 0 / 0 | 1 / 1 |
| stage 0 f600 | 7 / 7 | 0 / 0 | 8 / 8 |
| stage 0 f900 | 3 / 3 | 0 / 0 | 18 / 18 |
| stage 0 f1440 | 57 / 57 | 0 / 0 | 15 / 15 |
| stage 6 Lunatic f480 | **1512 / 1512** | 0 / 0 | 16 / 16 |

**已知缺口（激光）**：为找带活激光的帧一共采样了 31,200 帧——stage 0..7 × practice {0,1} × 难度 1，每段 1,200 帧；再加 stage 4/6/7 难度 3 与 stage 4 难度 1 各 3,000 帧——`LASERS count` 与 `HAZARDS nl` **全部为 0**，没有取到任何一帧带活激光的真实样本。所以激光行的 `inUse`/`state`/`startOffset`/`endOffset`/`currentWidth`/`width`/`angle`/`position` 只在编码层验证过（真实空页 + 合成行），`inUse` 与池筛选用的是同一表达式，非空池不可能输出 0。激光几何本身没有拿到真实数字，这一项留给 P1-WP1 的后续帧。同一批采样附带确认：这 31,200 帧里子弹池最多同时有 1,512 发（stage 6 Lunatic 第 480 帧），已经接近 1,537 的上限。

## 4. HAZARDS 单帧耗时（100 帧平均）

同一连接上交替发 `PING` 与 `HAZARDS`，取均值；"净成本" = `HAZARDS` 均值 − `PING` 均值（两者都含同一往返与 Python 开销，相减抵掉）。在 31,200 帧的真实采样里最密的一帧是 stage 6 Lunatic 第 480 帧：1512 子弹 + 16 敌人 = 1528 行。

| 采样帧 | 行数 | 回复字节 | 占 256 KiB | 样本 | `HAZARDS` 均值 | 净成本 |
|---|---|---|---|---|---|---|
| stage 0 normal f1440 | 72 | 5,304 | 2.0 % | 100 | 54.15 µs | 22.99 µs |
| stage 7 Lunatic f720 | 961 | 65,843 | 25.1 % | 60 | 276.88 µs | 253.78 µs |
| **stage 6 Lunatic f480** | **1528** | **103,744** | **39.6 %** | **100** | **423.96 µs** | **401.68 µs** |

三点线性拟合：净成本 ≈ **5 µs + 0.246 µs/行**（实测斜率 0.225 / 0.246 / 0.260）。

**HazardTape 48 帧预计**（H=48，[11-P1 §5](<../11-P1-搜索教师.md>) 完整模式）：

| 场景 | 只算 `HAZARDS` | 净成本口径 |
|---|---|---|
| 最密真实帧（1528 行） | 48 × 423.96 µs = **20.35 ms** | 48 × 401.68 µs = **19.28 ms** |
| 常见 stage 0 帧（72 行） | 48 × 54.15 µs = **2.60 ms** | 48 × 1.10 ms |

端到端对照（`STEP 0 1` + `HAZARDS` 一次算一帧）：

```text
最密帧 ：8,978.02 µs/帧 → 48 帧 = 430.9 ms，其中 HAZARDS 净成本占 4.5 %
stage0 f1440：7,944.25 µs/帧 → 48 帧 = 381.3 ms，其中 HAZARDS 净成本占 0.29 %
纯 STEP 0 1440 推进：9.840 s / 1440 = 6.83 ms/帧
```

结论：一条 48 帧 HazardTape 的成本由**引擎推帧**（约 6.8–9.0 ms/帧）决定，`HAZARDS` 净成本只占 0.29 %～4.5 %；把 600+ 槽位从"每帧多次分页往返"改成一次批量，省下的是往返次数（三处分页 → 一次，且每页 16 个槽位 → 全部），而不是这 400 µs。按 256 KiB 上限估算全槽位最坏情况（1537 子弹 + 256 激光 + 481 敌人 ≈ 203 KiB）仍在容量内，所以 `ERR hazards-too-large` 在真实玩法里没有被触发——实测最密帧只用了 39.6 %。

## 5. 测试

```text
python -m pytest -p no:cacheprovider -q tests/test_v400_hazards.py tests/test_v400_protocol.py
38 passed in 0.12s          # 新增 31 项 + 既有协议 7 项
python -m pytest -p no:cacheprovider -q tests/test_v400_hazards.py tests/test_v400_protocol.py \
    tests/test_v400_env.py tests/test_v400_obs_spec.py tests/test_v400_shm.py \
    tests/test_v400_state_export.py tests/test_v400_wp5.py
58 passed in 11.18s
python -m pytest -p no:cacheprovider -q tests        # 全量回归
1034 passed, 3 skipped in 73.17s
```

`tests/test_v400_hazards.py`（31 项）覆盖：

- 追加字段解码：`collisionSize`/`collisionDisabled`/`worldPosition`/`hitboxWidth`/`hitboxHeight`/`inUse`；
- **向后兼容**：把真实行按前缀切片成"旧引擎行"，逐字段比对前缀值并断言前缀子串**字节相同**；
- `PLAYER`：真实回复解出 4.0/2.8284/2.0/1.4142、`hp/lives/bombs/power`、半宽 0.825，缺速度字段时为 `None` 而不是 0.0，`ERR player-speed-unavailable` 会抛出；
- `HAZARDS`：真实帧 7/8/0 计数与分页命令一致、600 行大批量解析、`limit` 截行但保留引擎总数、`ERR hazards-too-large` 失败关闭、容量常量 262144；
- 旧标量回复（`HASH`/`STATE`/`LASERS` 空页）解析不变。

## 6. 有效性自检

- V1 是：`collisionSize`/`hitboxDimensions`/速度都是引擎内存里的原始值（floatBits 直传），不是重算值 —— 4.0/6.0/24.0/4.0/2.8284 与 `BulletManager.cpp:1630-1706`、`EnemyManager.cpp:852`、`PlayerRawShtFile` 的定义对得上。
- V2 是：对照是**同一输入序列下的 `build/rl`**，不是历史数字；两边各自独立数据目录。
- V3 是：`HAZARDS nb` 与 `BULLETS count` 在 6 个真实帧上全等，包括 1512 的那个。
- V4 是：`build/rl` 前后 SHA 相同，`CMakeCache` 关键项逐项相同，`build/rl` 直接拒绝新命令。
- V5 是：平凡基线是"扩展只改注释/不改语义"——若如此，100 帧 HASH 必须一致；实测一致，`first_divergence=null`。
- V6 是：耗时是 60～100 次同连接采样，同时报了 `PING` 基线，净成本可复算。
- V7 未做：单进程单种子，未做跨进程重复；确定性由 EXP-P0-049（5000 帧）覆盖，本次只要求"扩展没改动它"。
- V8 是：`PLAYER` 的失败关闭分支在真实运行里被触发并返回 `ERR player-speed-unavailable`，说明代码路径真的会走，而不是只写在注释里。
- V9 是：六项主指标在运行之前写定，没有事后加指标。
- 未覆盖：**带活激光的真实帧**（见 §3 已知缺口），以及 `ERR hazards-too-large` 的真实触发。

## 7. 结论

**成立。** P1 需要的逐帧致死物几何现在可以一次往返取到：子弹全宽判定、敌人 `worldPosition` + 原始 `hitboxDimensions` + `flags1` bit2、自机 `hurtboxBounds` 与按机体 `.sht` 读取的速度表；`STATE`/`HASH` 与 `build/rl` 的确定性没有任何变化。激光几何仍缺真实样本，是下一步唯一没关掉的洞。

## 8. 成本

GPU 0 h；墙钟约 2.5 h（其中真实引擎采样约 55 min，含 31,200 帧的激光搜索）；磁盘约 0.02 GB（`build/rl_hazard` 构建树 18 MB，其中二进制 4.8 MB；游戏数据用软链，不复制）。

## 9. 下一步

1. 抓一段带活激光的真实帧（需要能推进到有激光符卡的时点，例如用真实 replay 或让自机活过 stage 4/6 的中段），补上激光几何的真实数字。
2. 在 `HazardTape` 录制器里用 `HAZARDS` 替换三次分页往返，并实测 48 帧端到端。
3. `PLAYER` 与 `HAZARDS` 接进 `touhou_rl/v400/teacher/`（P1-WP2）。
