# EXP-P0-020 WP3 ITEMS 真实状态导出（Windows 调用 WSL 构建）

- 日期：2026-11-08；范围：只补真实 harness 的 ITEMS 合同测试，不修改协议、游戏代码或第三方。
- 执行位置：Windows Python/pytest 经 `wsl --exec` 调用既存 Linux RL 构建 `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`；游戏资源 `/mnt/d/WorkShop/Touhou_RL/.tmp/calibration/th08.dat`。`shadow` 无 DISPLAY，`env` 经 `xvfb-run -a`；这是 **Windows 发起的 WSL 真实引擎验证**，不是原生 Windows 引擎构建验证。
- 测试：[test_v400_state_export.py](<../../../../../tests/test_v400_state_export.py>)，两个参数分别启动独立真实引擎进程；缺失可执行文件/资源或 xvfb-run 时跳过，不能把跳过当验收通过。命令先请求未 RESET 的 `ITEMS`，再 `RESET 0 1 0 1234`、`STEP 0 1200`、`ITEMS 0 16`、`ITEMS 1 2`、`HASH`、`STATE`、`QUIT`。严格解析 `frames/count/offset/limit/items`、索引顺序和字段格式；要求实际 `count > 2` 且完整首批覆盖全数，第二页必须逐字等于真实首批的第 2～3 条。未 RESET 必须回复 `OK count=undefined items=undefined frames=0`，绝不接受假造空列表或合成列表作为真实证据。
- 原始 shadow 标准输出：[.tmp_v400_items_live.txt](<../../../../../.tmp_v400_items_live.txt>)；运行 `python -c ...` 采集，`LIVE_EXIT=0`。主要原始回复：
  ```text
  OK hello proto=1 harness=1 role=shadow frames=0
  OK count=undefined items=undefined frames=0
  OK reset=1 ready=1 setup_frames=540 frames=0 stage=0 difficulty=1 team=0 seed=1234 setup_state=0 hash=ba6850c823c24bae
  OK advanced=1200 frame=1200 hash=b2e32b34ffc7f218
  OK frames=1200 count=6 offset=0 limit=16 items=6:2:0:43a17ca0:41892fa1:3b5a7400:92,7:0:0:4387281c:414a4044:3b5a7400:92,8:0:0:43310c5c:c19b6c0b:3b5a7400:92,9:0:0:439aca2e:42834e6f:3b5a7400:92,10:0:0:42b5d13c:430a88f9:3b5a7400:92,11:0:0:42f0c8ce:42583010:3b5a7400:92
  OK frames=1200 count=6 offset=1 limit=2 items=7:0:0:4387281c:414a4044:3b5a7400:92,8:0:0:43310c5c:c19b6c0b:3b5a7400:92
  ```
- 验收：`$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_items_pytest -q tests/test_v400_state_export.py`；以最终重跑输出为准。`git diff --check --no-index /dev/null Document/Update/Alpha/v4.0.0/experiments/EXP-P0-020-WP3-items.md` 用于未跟踪新文档的空白检查（`--no-index` 成功比较时返回 1，空白错误返回 2）。
- 边界：只验证该 seed 在第 1200 帧的真实活动道具列表、分页及未 RESET 状态；不声称原版零售引擎适配、所有种子覆盖或完整 WP3 验收。
