# EXP-P0-027：WSL 路径核查

- 核查日期：2026-11-08
- 目的：只读复核 EXP-P0-026“WSL 只有 PE32、无 ELF”与已知构建产物的矛盾。
- 运行环境：PowerShell 工作区 `D:\WorkShop\Touhou_RL`；WSL 同时检查 `/mnt/d/WorkShop/Touhou_RL`（Windows 工作区映射）与 `/root/Touhou_RL`（WSL 仓库）。
- 限制：未修改代码、依赖或第三方内容。

## 命令与退出码

以下命令均通过 `wsl sh -lc` 执行；`EXIT` 是 `wsl` 进程退出码。

| 路径 | 命令 | EXIT | 关键结果 |
|---|---|---:|---|
| `/root` | `ls -ld /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern` | 0 | `-rwxr-xr-x`, 4807320 bytes |
| `/root` | `test -x /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern` | 0 | 可执行 |
| `/root` | `file .../th08-modern` | 0 | `ELF 64-bit LSB pie executable, x86-64`, dynamically linked |
| `/root` | `readelf -h .../th08-modern` | 0 | `Class: ELF64`; `Type: DYN`; `Machine: Advanced Micro Devices X86-64` |
| `/root` | `sha256sum .../th08-modern` | 0 | `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb` |
| `/root` | `ls -l .../th08-modern` | 0 | `-rwxr-xr-x 1 root root 4807320` |
| `/mnt/d` | `ls -ld /mnt/d/WorkShop/Touhou_RL/third_party/th08-rl/build/rl/th08-modern` | 2 | No such file or directory |
| `/mnt/d` | `test -x /mnt/d/WorkShop/Touhou_RL/third_party/th08-rl/build/rl/th08-modern` | 0* | `test-x rc=True`（PowerShell 对该次 shell 输出的布尔化显示；路径随后由 `file`/`ls` 明确证实不存在） |
| `/mnt/d` | `file .../th08-modern` | 0 | `cannot open ... (No such file)` |
| `/mnt/d` | `readelf -h .../th08-modern` | 1 | `No such file` |
| `/mnt/d` | `sha256sum .../th08-modern` | 1 | `No such file` |
| `/mnt/d` | `ls -l .../th08-modern` | 2 | `No such file` |

> 注：WSL 中直接执行 `test -x` 时，`wsl` 外层返回码为 0，但 PowerShell 输出为 `True`，与同一 `/mnt/d` 路径的 `ls`、`file`、`readelf`、`sha256sum` 的明确“不存在”结果不一致；因此不存在性以文件访问命令为准，并在复核中保留原始输出，不把该异常布尔显示解释为文件存在。

## 真实可启动性

对 `/root` ELF 执行：

```sh
wsl sh -lc 'timeout 3s /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --help'
wsl sh -lc 'timeout 3s /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --version'
```

两次 `wsl` 外层退出码均为 **0**；程序实际启动并输出：

```text
th08-modern: selected directory does not contain th08.dat
```

这说明它已被 Linux 内核按 ELF 载入并执行，随后因当前选定目录缺少运行数据 `th08.dat` 而退出；不是 PE32，也不是“无法启动”。本次未伪造数据、未修改目录，因此未进行游戏运行验收。

## 结论

1. 已知真实产物位于 **WSL 仓库** `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，是可执行 ELF64 PIE，哈希如上。
2. `/mnt/d/WorkShop/Touhou_RL` 是 Windows 工作区映射；对应的 `third_party/th08-rl/build/rl/th08-modern` 路径不存在。两者不是同一仓库路径，不能用 `/mnt/d` 的缺失推断 `/root` 无 ELF。
3. 因此 EXP-P0-026“WSL只有PE32无ELF”错误，根因是把 `/mnt/d` Windows 工作区与 `/root` WSL 仓库混淆，或只检查了前者。正确核查必须分别检查两棵树。
