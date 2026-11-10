# EXP-P0-059：WP0 共享数据模板被意外改写——安全恢复与留证

日期：2026-11-09（本地 UTC+08:00）。Provider=DeepSeek，模型=DeepSeek v4.1 flash。

范围：仅涉及 `/root/v400-data/game/` 下的 `score.dat`、`score.dat.orig`、`th08.cfg`、`th08.cfg.orig`；
未删除任何文件，未修改 `th08.dat` / `thbgm.dat`，未改引擎/协议/第三方。

## 结论

1. **恢复完成且可验证**：`score.dat` 已从 `score.dat.orig` 恢复为 17,773 B / `sha256=60e49abc…`，
   `th08.cfg` 已从 `th08.cfg.orig` 恢复为 60 B / `sha256=48bf7621…`；两次恢复都用 `cmp` 证明与 `.orig` 逐字节相同。
2. **恢复后再次被写脏，已做第二次恢复**：第一次恢复（00:20:53）后，仍有一条使用共享目录作 `--data-dir`
   的引擎运行在 00:21:33 / 00:22:15 再次改写 `score.dat`（17,776 B）。第二次恢复（00:22:26）后按 15 s 间隔
   连测三次（00:22:26 / 00:22:41 / 00:22:58），SHA 稳定为 `60e49abc…`，且复查时**已无任何进程**以
   `/root/v400-data/game` 作 data-dir。
3. **原因（已定位）**：`WP8 pair3` 帧采集套件曾以 `th08-modern --data-dir /root/v400-data/game`
   直接跑在共享模板上（本次捕获到进程 PID 23125，2026-11-09 00:20:17 启动），
   该目录因此被当作可写数据目录使用，引擎在退出时重写 `score.dat`/`th08.cfg` 并在 `backup/` 生成 `.bak`。
   `tools/rl/v400_linux_frame_capture.py` 的 `--data-dir` 默认值就是该共享目录（第 547 行），
   `tools/rl/v400_bench_run.py` 的 `DEFAULT_DATA_DIR` 同样是该共享目录（第 77 行）。
4. **未用任务书中的 SHA 前缀**：任务书给出的 `b7e7806a… → 69964c8a…` 与本次所有实测值均不符
   （见「声明值与实测值的差异」），本卡只记录实测值，不猜测、不伪造。
5. **残余风险**：只要再有工具以共享目录为可写 data-dir 启动，模板会再次变脏；需要把 bench/capture
   的 data-dir 指向每个 run 私有的 runtime 目录。本次 00:24:16 重启的 WP8 pair3 采集已改用
   `--data-dir /root/v400-data/runtime3/...`，00:24:52 复查共享模板仍为干净值（整改生效的实证）。

## 时间线（本地时间 UTC+08:00；UTC 另注）

| 时间 | 事件 | 证据 |
| --- | --- | --- |
| 2026-11-07 14:05:52–14:05:55 | 共享模板落盘：`th08.dat`、`thbgm.dat`、`score.dat.orig`(17,773 B, 755)、`th08.cfg.orig`(60 B, 755) 同一批 mtime | `stat`/`ls -l --time-style=full-iso` |
| 2026-11-08 23:11:52（15:11:52 UTC） | 既有 `v400-data-manifest.json` 已把 `score.dat` 记为 **16,047 B / `eef06ede…`**——说明模板在 23:11 之前就已被改写，已不是 17,773 B 的 `.orig` | `Document/Update/Alpha/v4.0.0/v400-data-manifest.json` |
| 2026-11-08 23:27–23:56（15:27–15:56 UTC） | `results.jsonl` 中 52 条 `status="ok"` 的 episode 落盘（其中 6 条后被作废）；bench runner 默认 data-dir 即共享模板 | `bench/results.jsonl`、`v400_bench_run.py:77,378` |
| 2026-11-09 00:03:21–00:04:52 | 引擎在共享目录留下 `backup/score_5..3.261109.bak`（16,047/16,049 B） | `backup/` 清单 |
| 2026-11-09 00:15:13 / 00:15:20 / 00:15:21 / 00:15:22 | `th08.cfg` mtime 更新（脏值 `4664d8d4…`）、`backup/score_2`、`score.dat` mtime（16,048 B）、`modern-files.txt` 记录 `th08-modern` 打开 `th08.cfg`/`score.dat` | `ls -l --time-style=full-iso`；`modern-files.txt` |
| 00:17:57 | **本次第一次现状记录**：`score.dat`=16,048 B / `3db0e03f…`，`th08.cfg`=60 B / `4664d8d4…` | 本卡「恢复前 SHA」表 |
| 00:18 | 检查 `/proc/*/fd`：无进程持有 `score.dat`/`th08.cfg` 句柄（只有 `th08.dat`/`thbgm.dat` 只读句柄） | `/proc/*/fd` 列表 |
| 00:20:16–00:20:23 | `th08.cfg` 再写（仍为脏值 `4664d8d4…`）；00:20:17 `v400_linux_frame_capture.py` 以 `th08-modern --data-dir /root/v400-data/game` 启动（PID 23125）；00:20:22 `backup/score_1`；00:20:23 `score.dat`=16,049 B / `543eca88…` | `ps`、`ls -l`、`sha256sum` |
| 00:20:53 | **恢复 #1**：先把脏值另存 `score.dat.dirty-20261109-001521`、`th08.cfg.dirty-20261109-001521`，再从 `.orig` 用 `cp -p` 复制回；`cmp` 证明一致 | 本卡「恢复动作与证据」 |
| 00:21:33 / 00:22:09 / 00:22:15 / 00:22:16 | 恢复 #1 之后又有一次共享目录运行：`backup/score_2`(17,776 B, 00:21:33)、`th08.cfg` 00:22:09 重写为**同一干净值** `48bf7621…`、`score.dat` 00:22:15 变为 **17,776 B / `240683c9…`**、`modern-files.txt` 00:22:16 | `backup/`、`stat`、`sha256sum` |
| （主代理观测） | 主代理在此期间另测得一次脏值 17,776 B / `2418e203…`；与本次 00:22:26 实测 `240683c9…` 不同，说明 00:21:33–00:22:15 之间至少还有一次写入，两个值都是脏值 | 仅记为二手观测，不作为恢复基线 |
| 00:22:26 | **恢复 #2**：另存 `score.dat.dirty-20261109-002226`（17,776 B / `240683c9…`），从 `.orig` 复制回；T+0 起连测 3 次 | 本卡「恢复动作与证据」 |
| 00:22:26 → 00:22:58 | T+0 / T+15 / T+30 三次 `sha256sum` 均为 `60e49abc…`（17,773 B）；复查无 `--data-dir /root/v400-data/game` 的写入进程 | 稳定性检查输出 |
| 00:24:16 | WP8 pair3 采集同一目标重启，但命令行已改用**私有** data-dir `/root/v400-data/runtime3/r1-normal-stage4a/game`（整改生效，不再指向共享模板） | `ps -eo args` 全量命令行 |
| 00:24:21 / 00:24:52 / 00:26:05 | 共享模板复查仍为干净值（`60e49abc…` / `48bf7621…`），`modern-files.txt` 停留在 00:22:16，未被再次打开写入 | `sha256sum`、`ls -l` |
| 00:26:05 | 最终：`score.dat`=17,773 B / `60e49abc…`（=`.orig`），`th08.cfg`=60 B / `48bf7621…`（=`.orig`），目录 13 个条目、无任何删除 | 最终清单 |

## 恢复前 SHA（实测）

| 文件 | 大小 | SHA-256 | 说明 |
| --- | ---: | --- | --- |
| `score.dat`（00:17:57 首次记录值） | 16,048 B | `3db0e03f4b39faddb56fb3209ea9aab2119ba1e7ff78c97833fe28418497373d` | 脏值 A |
| `score.dat`（00:20:53 恢复 #1 前值） | 16,049 B | `543eca881e80077f6de6518e144d7ee9c2f6f7017421d34d564b9415a84e5b0a` | 脏值 B，已备份 `score.dat.dirty-20261109-001521` |
| `score.dat`（00:22:26 恢复 #2 前值） | 17,776 B | `240683c96d536c879ee9402b8ac36069133d9c5b76d419d504fbb3eb3e37adf6` | 脏值 C，已备份 `score.dat.dirty-20261109-002226` |
| `score.dat.orig` | 17,773 B | `60e49abc8f0a955aa8fa883769a30c66913417be453632c9a8a99b0b8a5c5191` | 模板快照（未被修改） |
| `th08.cfg`（00:17:57 / 00:20:53 前值） | 60 B | `4664d8d44cb845a9faac07bd9a7973ce24fab7234bf082fa5dc3b743bc2d6574` | 脏值，已备份 `th08.cfg.dirty-20261109-001521` |
| `th08.cfg.orig` | 60 B | `48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a` | 模板快照（未被修改） |

### 恢复后 SHA（两次恢复均一致）

| 文件 | 大小 | SHA-256 | 与 `.orig` 关系 |
| --- | ---: | --- | --- |
| `score.dat` | 17,773 B | `60e49abc8f0a955aa8fa883769a30c66913417be453632c9a8a99b0b8a5c5191` | `cmp` 逐字节相同 |
| `th08.cfg` | 60 B | `48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a` | `cmp` 逐字节相同 |

## `th08.cfg` 差异分析

`cmp -l th08.cfg.orig th08.cfg.dirty-20261109-001521`（1-based 偏移）：仅 3 字节不同。

| 偏移 | `.orig` | 脏值 |
| ---: | ---: | ---: |
| 35 | `0x01` | `0x00` |
| 40 | `0x27` | `0x64` |
| 41 | `0x24` | `0x50` |

即字节序列 `01 01 01 00 02 00 00 27 24` → `01 01 00 00 02 00 00 64 50`。
本卡只记录字节差异，**不推断**这两个字段的语义（未做逆向验证）。恢复 #1 后该文件曾被引擎于 00:22:09
重写，但内容与 `.orig` 完全相同（`48bf7621…`），因此恢复 #2 无需再动 `th08.cfg`。

## `score.dat.orig` 可信性核查（恢复前置条件）

| 检查项 | 结果 |
| --- | --- |
| 存在性 | 存在 |
| 大小 | 17,773 B，与任务书「17,773 B 左右」一致 |
| 权限/属主 | `-rwxr-xr-x root root`，与同批模板资产 `th08.dat`/`thbgm.dat` 一致 |
| mtime 溯源 | `2026-11-07 14:05:55.294297879`，与 `th08.dat`(14:05:52)、`thbgm.dat`(14:05:55)、`th08.cfg.orig`(14:05:55) 同一落盘批次 |
| 是否被本次任务改动 | 否：恢复前后均为 `60e49abc…`、17,773 B、mtime 未变 |
| 格式旁证 | 恢复为 17,773 B 后，引擎在 00:21–00:22 的共享目录运行中写出的下一版为 17,776 B（同尺寸谱系，+3 B）；而 16,047–16,049 B 的脏值与模板相差约 1.7 KB，是引擎在「无既有档」时自建的另一尺寸谱系（对照：replay-retrace 私有 runtime 内引擎自建的首个 `score.dat` 为 282 B，随后增长到 16,048 B）。旁证 `.orig` 是正确模板格式 |
| 独立对照 | `tools/rl/v400_wp8_pair_suite.py` 的 `EXPECTED_SHARED_TEMPLATE`（第 68–74 行）把 pristine 值固定为 `score.dat=60e49abc…`、`th08.cfg=48bf7621…`，与本次恢复值一致（该工具注释把事故记作 EXP-P0-058；该卡在本目录内尚未落盘） |

结论：`score.dat.orig` 存在、大小合理、溯源清晰，可作为恢复基线；**不触发**「不存在或不可信→只记录阻塞」分支。

## 恢复动作与证据（可复现）

```bash
cd /root/v400-data/game
# 0) 现状
ls -l --time-style=full-iso score.dat score.dat.orig th08.cfg th08.cfg.orig
sha256sum score.dat score.dat.orig th08.cfg th08.cfg.orig
# 1) 新的带时间戳备份（只复制，不删除）
cp -p score.dat score.dat.dirty-20261109-001521     # 16,049 B / 543eca88…
cp -p th08.cfg  th08.cfg.dirty-20261109-001521      # 60 B / 4664d8d4…
# 2) 恢复 #1
cp -p score.dat.orig score.dat
cp -p th08.cfg.orig  th08.cfg
cmp score.dat score.dat.orig && echo identical
cmp th08.cfg  th08.cfg.orig  && echo identical
# 3) 恢复 #1 后又被写脏（17,776 B / 240683c9…）→ 备份并恢复 #2
cp -p score.dat score.dat.dirty-20261109-002226     # 17,776 B / 240683c9…
cp -p score.dat.orig score.dat
# 4) 间隔 15 s 连测三次
sha256sum score.dat; sleep 15; sha256sum score.dat; sleep 15; sha256sum score.dat
# 5) 最终核对
sha256sum score.dat score.dat.orig th08.cfg th08.cfg.orig
cmp score.dat score.dat.orig && echo identical
```

实际执行退出码：两次恢复脚本均为 **0**（`wsl -e bash <script>` 的 `$LASTEXITCODE=0`）。
稳定性三次读数：`00:22:26`、`00:22:41`、`00:22:58` 均为 `60e49abc…`（17,773 B）。

### 未删除任何文件（目录清单核对）

恢复前后对比：恢复前目录 10 个条目（`&1`、`backup/`、`log.txt`、`modern-files.txt`、`score.dat`、
`score.dat.orig`、`th08.cfg`、`th08.cfg.orig`、`th08.dat`、`thbgm.dat`）全部保留，
只**新增**三份脏值备份：`score.dat.dirty-20261109-001521`、`score.dat.dirty-20261109-002226`、
`th08.cfg.dirty-20261109-001521`，恢复后目录为 13 个条目。`backup/` 下 5 份引擎自写 `.bak` 一份未删。

| 新增备份 | 大小 | SHA-256 |
| --- | ---: | --- |
| `score.dat.dirty-20261109-001521` | 16,049 B | `543eca881e80077f6de6518e144d7ee9c2f6f7017421d34d564b9415a84e5b0a` |
| `score.dat.dirty-20261109-002226` | 17,776 B | `240683c96d536c879ee9402b8ac36069133d9c5b76d419d504fbb3eb3e37adf6` |
| `th08.cfg.dirty-20261109-001521` | 60 B | `4664d8d44cb845a9faac07bd9a7973ce24fab7234bf082fa5dc3b743bc2d6574` |

## 声明值与实测值的差异（如实记录）

| 项 | 任务书声明 | 实测 |
| --- | --- | --- |
| `score.dat` 改写前 SHA | `b7e7806a…` | 本次能实测到的所有值（`.orig`=`60e49abc…`、manifest 在 23:11 记录的 16,047 B 值 `eef06ede…`）都不等于该前缀；更早的历史值无法回溯，故不据此推断 |
| `score.dat` 改写后 SHA | `69964c8a…` | 三个脏值实测为 `3db0e03f…`(16,048 B)、`543eca88…`(16,049 B)、`240683c9…`(17,776 B)，另主代理观测到 `2418e203…`(17,776 B)，均不等于该前缀 |
| 改写时间 | 23:56 | 23:56 是 bench 最后一条 episode 的 `finished_at`(15:56 UTC)；共享文件的实测 mtime 为 00:15:13–00:22:16；manifest 显示 23:11 之前模板已是 16,047 B 脏值 |
| `score.dat.orig` 大小 | 17,773 B 左右 | 17,773 B，一致 |

处置：**以实测为准**，未按声明值反推或伪造任何 SHA；上表差异作为审计事实保留。

## 对既有实验的影响判断

### 可能读到/写到脏模板的 run

1. **`v400_bench_run.py` 产出的 bench episode（`bench/results.jsonl` 53 行：1 条 `invalid` 审计行 +
   52 条 `status="ok"` 的 measured，其中 6 条被审计行显式作废，最终有效 46 条）**：
   该工具 `DEFAULT_DATA_DIR = "/root/v400-data/game"`（第 77 行），并把 `data_dir` 直接传给 harness client
   （第 378 行）启动 `th08-modern`。而 manifest 在 23:11（早于首条 episode 的 23:27）已记录脏值；
   在未显式覆盖 `--data-dir` 的前提下（结果行不记录 data-dir，无法逐条硬证），这些 episode
   都跑在非原始模板上，其中 23:53–23:56 的 40 条与 23:56 时间点吻合。
2. **`WP8 pair3` 帧采集（`.tmp_v400_wp8_pair3_suite/r1-normal-stage4a` 等）**：00:20:17 的
   `th08-modern --data-dir /root/v400-data/game`（PID 23125）与 00:21:33 / 00:22:15 的后续写入
   都发生在共享目录；该采集捕获的帧/回放是在脏模板下产生的。
3. **`v400_linux_frame_capture.py` 的任何未显式覆盖 `--data-dir` 的调用**：默认值即共享目录（第 547 行），
   具备同样暴露面，需按调用记录逐个审计。
4. **108 份 replay-retrace 差分任务（4 个 shard，当前仍在运行）**：不受影响。其命令行使用
   `--data-dir /mnt/d/.../th8_xxxx/{on,off}/attempt-1/runtime`；`run_capture()` 只把共享目录的
   `th08.dat`/`thbgm.dat` 以只读符号链接放进 runtime（第 298–299 行，`/proc/*/fd` 实测为只读句柄），
   `th08.cfg` 由 `muted_config()` 就地生成（第 300 行），**不复制**共享 `score.dat`
   （引擎在 runtime 内自建：首见 282 B，随后增长到 16,048 B）。它们不读也不写共享 `score.dat`/`th08.cfg`。
5. **`th08.dat` / `thbgm.dat`**：mtime 仍为 2026-11-07 14:05，SHA 与 manifest 记录一致，未被改写。

### 影响程度与建议

- `score.dat` 是持久化分数/符卡历史档，`th08.cfg` 是配置档；bench 判分读取的是引擎共享状态块
  （deaths/bombs/lives/score/frame），判分链路上没有读取 `score.dat` 内容的代码
  （全仓 grep 仅见 `session.prepare_run_dir` 的复制、`v400_replay_retrace.game_data_snapshot` 的
  provenance 哈希与工具侧模板完整性校验），因此**已记录的 episode 数值不需要重算**。
- 但「共享只读模板」这一前置条件对上述 run 已被破坏：这些 episode 的模板 provenance 应标记为
  **template-contaminated（非原始模板）**，而不是声称跑在原始模板上。若后续报告要求「原始模板」证据，
  应在模板修复 + 工具改用私有 data-dir 后重跑对应切片。
- 已作废的 6 条 episode 的作废原因是另一件事（post-game-over 菜单尾段、terminal rule 缺失，见 EXP-P0-048），
  与本次模板污染无关，不应混为一谈。
- 建议（不在本卡改动范围）：`v400_bench_run.py`、`v400_linux_frame_capture.py` 的默认 data-dir 改为
  每次 run 的私有 runtime 目录，并只读挂载/复制 `th08.dat`、`thbgm.dat`、`score.dat.orig`、`th08.cfg.orig`；
  对共享模板目录加写保护（只读挂载或 ACL）。

## 阻塞与残余风险

- **无恢复阻塞**：`score.dat.orig` 存在且可信，两次恢复均成功并验证。
- **残余风险 1**：恢复后曾被再次写脏（有实证）；在写入方改用私有 data-dir 之前，模板仍可能被再次改写。
  本次最终复查（00:24:21 / 00:24:52 / 00:26:05）无 `--data-dir /root/v400-data/game` 的进程，且 00:24:16 重启的
  WP8 pair3 采集已改用 `/root/v400-data/runtime3/...`，共享模板保持干净值。
- **残余风险 2**：若后续 bench/capture 仍以共享目录为 data-dir，将使更多 run 的模板 provenance 受污染。
- 未杀任何进程、未改任何工具默认值（超出本卡授权范围），仅记录并上报。
