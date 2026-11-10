# EXP-P0-044 — WP5 第4项：真实轨迹 + harness 像素字节再生成（2026-11-08）

**本次 100 帧有界实验：PASS（该轨迹的原始 RGBA 缓存与重新生成像素逐字节相同）。** 不依赖零售参考，亦不代表三条 replay 的 WP8 / G0 全局通过。二进制源码工作树不干净，`HEAD` 仅为提交基线，**不可把该 HEAD 单独当作精确构建身份**；同时冻结二进制 SHA-256 和实际轨迹、缓存。

## 执行、来源与边界

- 真实 ELF：`/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，本次 SHA-256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`。WSL 源树 `git rev-parse HEAD=f563df58c44037b44208c393a86540b434b0c83b`，tracked 工作树为 dirty；此前历史二进制哈希不同，不能混用历史卡片中的构建身份。
- 真机命令：`wsl.exe -d Ubuntu -- xvfb-run -a python3 /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_wp5_pixel_rebuild.py`，**脚本/引擎全部 exit 0**。运行窗口严格每实例 100 帧；源模板 `/root/v400-data/game` 只读输入；两个新建且原本不存在的目录 `/mnt/d/WorkShop/Touhou_RL/.tmp_v400_wp5_pixel_rebuild_20261108/data-original` 与 `.../data-replay`，各自复制 `th08.cfg` / `score.dat`，`th08.dat` / `thbgm.dat` 仅只读引用符号链接。**未删/清空/覆盖既有目录**；再次运行同一个路径会 fail-closed。
- `--rl-role env` 两个独立真实进程；`xvfb-run`、`SDL_AUDIODRIVER=dummy LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1`。双方 `HELLO role=env`、`RESET 0 1 0 1 0 ready=1`、`READBACK 1`、`SHM active=1 bytes=1228992`；逐帧 `STEP 0 1` 且校验返回 `advanced=1` 和对应 frame；逐帧 `PIXELS pixels=640x480 nonzero>0`，从 `/dev/shm` 校验头部 frame、magic、版本、尺寸、pixel-valid flag，随后取偏移 192 起 **1,228,800 个原始 RGBA 字节**。不伪造 PNG，不裁剪/遮盖 HUD FPS，不拿 PIXELS checksum、PNG 元数据或 state hash 代替字节比较。
- 原始轨迹固定为 StageStart `[stage=0,difficulty=1,team=0,seed=1,practice=0]`，逐帧 `masks=[0]×100`；第一进程真实读回逐帧连续写入原始 RGBA 缓存；第二进程按同样逐帧输入重生成，将每一帧与缓存对应字节 `baseline != rgba` 比较，并分别记录 SHA-256。单一固定全零轨迹是最小真实切片，不代表随机输入覆盖。原始缓存 SHA-256 `7ba9973f767671a029edb160866a24ecc43231d88b90639660134e851733e4d8`，即 100×1,228,800 字节。

## 原始证据与实测结果

- 原始机器报告：[`report.json`](../../../../../.tmp_v400_wp5_pixel_rebuild_20261108/report.json)；真实未加工像素缓存：[`original-rgba640x480x4.bin`](../../../../../.tmp_v400_wp5_pixel_rebuild_20261108/original-rgba640x480x4.bin)。报告含完整 100 项输入 mask、每帧两次 RGBA SHA-256、两次 step state hash、各进程逐条 stdout 协议响应、stderr、argv、exit code 和不等帧列表。复核命令 `wsl.exe -d Ubuntu -- python3 -c "import json; p=json.load(open('/mnt/d/WorkShop/Touhou_RL/.tmp_v400_wp5_pixel_rebuild_20261108/report.json')); print('capture counts:',[x['frames'] for x in p['captures']]); print('exit codes:',[x['exit_code'] for x in p['processes']]); print('different frames:',len(p['captures'][1]['different_frames'])); print('hash agreement:',p['captures'][0]['rgba_sha256']==p['captures'][1]['rgba_sha256']); print('cache digest:',p['original_cache_sha256'])"`：`capture counts: [100, 100]`、`exit codes: [0, 0]`、`different frames: 0`、`hash agreement: True`，复核命令 `AUDIT_EXIT=0`。
- 单进程先行能力探针：`wsl.exe -d Ubuntu -- xvfb-run -a python3 .../tools/rl/v400_wp5_pixel_rebuild.py`（脚本尚为单帧探针时）进程 `PROBE_EXIT=0`，`PIXELS nonzero=1228800 checksum=6db4d07889dc98f1`、`SHM active=1`、原始 RGBA SHA-256 `0f6085f8f8a7d7296aa7989156148ac612499bc4f1be338b1a7fddfe8722e094`；它不是100帧验收。
- 验收脚本：[`v400_wp5_pixel_rebuild.py`](../../../../../tools/rl/v400_wp5_pixel_rebuild.py)；单测：[`test_v400_wp5_pixel_rebuild.py`](../../../../../tests/test_v400_wp5_pixel_rebuild.py)。`$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp5_pixel_pytest2 -q tests/test_v400_wp5_pixel_rebuild.py`：**3 passed, 1 skipped，TEST_EXIT=0**。首次命令因 Windows 不能创建测试符号链接报 `3 passed, 1 failed, TEST_EXIT=1`；改为该权限场景跳过后复跑成功；单测不替代真机证据。

**限制：** 此次证明当前实测二进制哈希 + HEAD 基线（dirty）在该 100 帧真机轨迹上逐字节再现，不证明所有构建可由干净 `harness commit` 单独重建；亦不证明其他轨迹、环境或零售对照。此项局部通过不能覆盖其它 G0.2 / G0 验收缺口。
