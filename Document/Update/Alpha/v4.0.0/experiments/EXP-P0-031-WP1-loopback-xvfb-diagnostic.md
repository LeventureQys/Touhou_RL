# EXP-P0-031：WP1 官方 capture-replay loopback TCP Xvfb 诊断

- 日期：2026-11-08（America/Los_Angeles）
- Provider/model：DeepSeek / DeepSeek v4.1 flash
- 允许修改范围：本记录及 `.tmp_v400_retail_*`；未修改原始可执行文件、第三方脚本/数据、系统配置。
- 结论：**BLOCKED / STOP**。没有运行官方 capture-replay，没有伪造轨迹，没有尝试第二例。

## 请求的有界路径

目标命令要求以 `Xvfb :99 -nolisten unix -listen tcp 127.0.0.1` 建立仅 loopback 的 TCP X server，并使用临时 Xauthority、独立 `WINEARCH=win32` prefix、`DISPLAY=127.0.0.1:99`，对 `easy-0-b` 最多运行 150 秒。

实际执行的是一次有界 shell（外层工具 160 秒等待，官方 collector 若启动则 `timeout 150s`），临时目录：

```text
.tmp_v400_retail_diag_easy0_xvfb_tcp/
```

并尝试：

```text
Xvfb :99 -screen 0 1280x720x24 -nolisten unix -listen tcp 127.0.0.1 -auth "$XAUTHORITY"
```

## 真实结果

- `Xvfb` 路径：`/usr/bin/Xvfb`
- Wine 路径：`/usr/bin/wine`
- X server 进程：**未启动**（`XSERVER_NOT_RUNNING`）
- capture-replay：**未执行**（因此无 rows、无 complete、无 SHA）
- Xvfb stderr：`Unrecognized option: 127.0.0.1`，随后 `Fatal server error`，退出码 1。

证据文件：

- `.tmp_v400_retail_diag_easy0_xvfb_tcp/attestation.txt`
- `.tmp_v400_retail_diag_easy0_xvfb_tcp/xvfb.log`

## 精确阻塞

本环境的 Xvfb 版本将 `-listen` 的参数解析为协议名（例如 `tcp`），不接受 `127.0.0.1` 作为该选项的值；因此用户指定的精确仅 loopback 语法在 X server 初始化阶段失败。按照“若 X server/Wine ptrace 仍 0 行立即写 EXP-P0-031 诊断并停止”的要求，本轮在 X server 失败后立即停止，未改用可能暴露到非 loopback 的 `-listen tcp`，也未重复此前 Unix socket 或 `DISPLAY=:0` 路径。

后续只能由人工提供/确认适用于该 Xvfb 构建的 loopback-only 绑定方式（或兼容的 X server）；在此之前不得宣称 capture 成功或继续第二例。
