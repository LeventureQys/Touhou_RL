# EXP-P0-049：WP5 / G0.2 第 1 项——同一 StageStart + 同一 5,000 帧随机输入，两个真实 shadow 进程逐帧哈希对账

- 日期：2026-11-08。Provider=DeepSeek，模型=DeepSeek v4.1 flash。
- 工作包：P0-WP5；验收门：G0.2 第 1 项（`10-P0-训练场.md` §WP5 验收第 1 条）。
- 结论：**G0.2 第 1 项 PASS-with-limit**。两个**独立真实 shadow 进程**（各自全新 data-dir、各自独立启动）在**同一 `RESET 0 1 0 1234 1`** 后执行**同一段 5,000 帧掩码序列**（每帧一次 `STEP <mask> 1`），**5,000/5,000 帧逐帧 `hash=` 完全相等**，0 个不等帧、无首个不等帧；两进程 `exit_code=0/0`；`DEBUG threads=1`、`subthread=0`，结束后 `curState=2`、`frames=5000`。限制见文末，不据此翻转 G0。

## 预注册

- 主指标：同一 `StageStart` + 同一 5,000 帧掩码序列，两个独立真实 shadow 进程的**每帧** `hash=` 逐一相等。
- 通过阈值（全部满足才算通过）：①两进程都是真实 `role=shadow`；②`RESET` 都 `ready=1` 且帧号归零；③每帧恰好推进 1 帧且都返回 16 位十六进制 `hash=`；④各 5,000 帧、共 5,000 个逐帧哈希比对 `different_frames=0`；⑤RESET 帧 0 哈希也相等；⑥两进程 `exit_code=0`。
- 样本量：2 个独立进程 × 5,000 帧 = 10,000 次 STEP（每帧一次），不是抽样、不是批量 STEP。

## 设置（真实环境）

- 引擎：WSL Linux ELF `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，SHA-256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`，4,807,320 B。
- harness 源树：`/root/Touhou_RL/third_party/th08-rl`，HEAD `f563df58c44037b44208c393a86540b434b0c83b`，**dirty=true**（构建身份只能以 binary SHA-256 + HEAD 基线共同引用）。
- 数据模板：`/root/v400-data/game`（`th08.dat` `9d7edf43b8ddd347cbb641836f6b5050745dd936f688daebbf9382ca557043bb`、`thbgm.dat` `2c2ef05f9ff6f43f752dae5da519a2477372c3f8875b7b44996a8f69fdad4d89`、`th08.cfg` `4664d8d44cb845a9faac07bd9a7973ce24fab7234bf082fa5dc3b743bc2d6574`、`score.dat` `c7e3cb8dc675ef139d448f1173ba0d94f10f5ce80689400f9a3df865c3bc5f01`）。
- 两个 **全新且原本不存在** 的 data-dir（工具对已存在路径直接拒绝，从不删除任何既有目录/文件）：
  - `/root/v400-wp5-determinism-5000-1/data-a`
  - `/root/v400-wp5-determinism-5000-1/data-b`
  - 每个目录内：`th08.dat`/`thbgm.dat` 为指向模板的**只读符号链接**，`th08.cfg`/`score.dat` 为**复制**（复制后 SHA-256 与模板一致；JSON 中两份 `copies[*].sha256` 完全相同：`th08.cfg` `4664d8d4…c2d6574`、`score.dat` `c7e3cb8d…c3bc5f01`）。
  - 说明：模板目录的 `th08.cfg`/`score.dat` 是会被其它实验复用的**可写**文件。本实验创建两份 data-dir 时它们逐字节相同；实验结束后（23:26:47）模板中的这两个可写文件被本实验之外的进程更新，故现在直接重算模板哈希会与本记录不同。只读档案 `th08.dat`/`thbgm.dat` 在实验前后哈希不变（`9d7edf43…7043bb` / `2c2ef05f…dad4d89`），本实验的两个进程全程只读它们，且各自只写自己的 data-dir。
- 进程：`--rl-control 0,1 --rl-role shadow`；`env -u DISPLAY`（null-device 预检要求 `DISPLAY` 为空，stderr 实测 `shadow null-device preflight=ok`），另设 `SDL_AUDIODRIVER=dummy`、`LIBGL_ALWAYS_SOFTWARE=1`、`LP_NUM_THREADS=1`。未设置 CPU affinity。
- `StageStart`（两进程完全相同）：`RESET 0 1 0 1234 1`（Stage 1 / Normal / team 0 / 游戏种子 1234 / practice=1）。两进程 `RESET` 回复：`ready=1 frames=0 setup_frames=305` 与 `ready=1 frames=0 setup_frames=302`（setup 帧数允许不同，帧 0 哈希必须相同）。

## 掩码序列生成方式（固定种子，可复现）

- 种子 **400**；每帧一个掩码，只允许 `0x00F7` 白名单位：`shoot|bomb|focus|up|down|left|right`；`BTN_MENU(0x08)` 与全部系统位从不置位。
- 生成算法（SplitMix64，逐帧取输出低 7 位，一位对应一个白名单按键）：

```text
state = seed
for each frame:
    state = (state + 0x9E3779B97F4A7C15) mod 2^64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9)     mod 2^64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB)     mod 2^64
    value = z ^ (z >> 31)
    mask  = OR of MASK_BITS[i] for i in 0..6 where bit i of value is 1
# MASK_BITS = (0x01, 0x02, 0x04, 0x10, 0x20, 0x40, 0x80) = shoot,bomb,focus,up,down,left,right
```

- 5,000 帧掩码的 SHA-256（文本形式：逗号分隔、小写 `%04x`）= `4173ff413d4bc18e27b771e86de16a441f765ac92b273c68ab7e32f5ee585e1f`；`unique_mask_count=128`，全部掩码 `& ~0x00F7 == 0`。
- 头 16 个（十六进制）：`00b1 00f6 0040 0073 00a6 0020 00b4 0094 00b4 00c7 0050 0046 00b3 0033 00f3 00e3`；尾 8 个：`0081 00b1 00e6 00f5 00f1 0003 0036 0001`。完整 5,000 个掩码数组已写入 JSON。

## 真实命令与退出码

1. 验收主运行（WSL，`WSL_EXIT=0`）：

```powershell
wsl.exe -d Ubuntu -- bash -lc 'env -u DISPLAY /root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_wp5_determinism.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-template /root/v400-data/game --run-root /root/v400-wp5-determinism-5000-1 --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_wp5_determinism_5000.json'
```

工具单行输出：`{"status": "pass", "frames_per_run": [5000, 5000], "equal_frames": 5000, "different_frames": 0, "first_differing_frame": null, "exit_codes": [0, 0], "mask_sha256": "4173ff…5e1f", "wall_s": 5.339, "warnings": []}`。

2. 前置冒烟（5 帧，**不计入验收**；工具按设计拒绝把少于 5,000 帧当验收证据，`WSL_EXIT=3`，JSON 落在 `/root` 而非仓库）：

```powershell
wsl.exe -d Ubuntu -- bash -lc 'env -u DISPLAY /root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_wp5_determinism.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-template /root/v400-data/game --run-root /root/v400-wp5-det-preflight-1 --out /root/v400-wp5-det-preflight-1.json --last-frame 5'
```

3. 逻辑单测（Windows，仅合成逻辑，**不是**引擎证据；`PYTEST_EXIT=0`）：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_wp5_determinism.py
# 27 passed in 0.09s
```

4. 主代理侧独立复核 JSON（重新计算两数组是否相等、掩码 SHA-256、退出码、DEBUG；`PY_EXIT=0`）：从 `.tmp_v400_wp5_determinism_5000.json` 读回 `runs[0].hashes` 与 `runs[1].hashes`，`a == b` 为 `True`、`first_diff=None`；按文档文本形式重算掩码 SHA-256 与 `mask.sha256` 一致；`exit_codes=[0,0]`、`roles=['shadow','shadow']`、`debug threads='1'/subthread='0'`。

## 结果

| 项目 | run-a | run-b |
|---|---|---|
| data-dir（全新） | `/root/v400-wp5-determinism-5000-1/data-a` | `/root/v400-wp5-determinism-5000-1/data-b` |
| `HELLO` role | shadow | shadow |
| `RESET` reply | `ready=1 setup_frames=305 frames=0 hash=d97b43478cc24bae` | `ready=1 setup_frames=302 frames=0 hash=d97b43478cc24bae` |
| DEBUG after RESET | `curState=2 threads=1 subthread=0` | `curState=2 threads=1 subthread=0` |
| 逐帧 STEP 数 | 5,000（每帧 `STEP <mask> 1`） | 5,000 |
| 记录到的帧哈希数 | 5,000 | 5,000 |
| 末帧 HASH 交叉校验 | 与末帧 `STEP` 哈希一致 | 与末帧 `STEP` 哈希一致 |
| DEBUG after steps | `curState=2 frames=5000 threads=1 subthread=0` | `curState=2 frames=5000 threads=1 subthread=0` |
| 引擎 exit code | **0** | **0** |
| 进程墙钟 / STEP 墙钟 | 2.379 s / 0.784 s | 2.336 s / 0.744 s |

- **compare：`frames_compared=5000`、`equal_frames=5000`、`different_frames=0`、`first_differing_frame=null`**；帧 0（`RESET`）哈希也相等 `d97b43478cc24bae`；两进程的 5,000 个哈希按换行连接后的 SHA-256 均为 `75132e28dc224dc8ecedccecdb498baf9e6ee93f0741d5fd8f03b458e92ba473`。
- 首/末逐帧哈希：`46e2f54401a0fdb7`（帧 1）… `4ede365c6cb537e1`（帧 5000）；两进程完全相同。
- 总墙钟 **5.339 s**（远低于 ~10 分钟，无需分批；每进程 5,000 次 `STEP <mask> 1` 单次约 0.15 ms）。
- 未出现 `exit=1`/`game_over`：practice=1 的 `lifeCount=8` 资源下这一段随机输入撑满 5,000 帧。

## 原始证据

- 逐帧哈希与全部元数据：`.tmp_v400_wp5_determinism_5000.json`（344,611 B；含两进程各 5,000 个 `hash=`、完整 5,000 个掩码、RESET/DEBUG/exit/stderr 记录与 binary/harness/data 身份）。
- 进程 stderr 日志与两个一次性 data-dir（保留供审计，未删除）：`/root/v400-wp5-determinism-5000-1/`（`run-a.stderr.log`、`run-b.stderr.log`、`data-a/`、`data-b/`）。
- 工具：`tools/rl/v400_wp5_determinism.py`；单测：`tests/test_v400_wp5_determinism.py`。

## 限制（必须与结论同时引用）

1. **practice=1 是 StageStart 的一部分**：引擎在该模式下把 `lifeCount` 设为 8，这才使单局覆盖 5,000 帧可行。本实验不证明“不死”“完整 6 面流程”或零售对照下的 5,000 帧。
2. 逐帧 `hash=` 是引擎对其 25 个跟踪字段的 FNV-1a 摘要（16 位十六进制），**不是全内存哈希**；相同哈希证明这 25 个字段逐帧一致，不覆盖子弹/敌人全池。
3. harness 源树 **dirty**：HEAD 只是提交基线，精确构建身份需与 binary SHA-256 `591c52…a25d3eb` 一起引用。
4. 只覆盖这 1 条 5,000 帧输入序列、Stage 1 Normal、team 0、seed 1234、practice；不证明其它阶段/难度/机体/种子或更长运行（如 10,000 帧）的确定性。
5. 单线程证据来自 `DEBUG threads=1`（且 shadow 无 SDL/GL 线程），未强制 CPU affinity；跨机器/跨构建不保证。
6. 本结果只对应 **G0.2 第 1 项**。G0.2 第 2/3/4/5 项与 G0 其余门的状态不因本实验改变，G0 仍不得判 PASS。
