# V831-SLICE-02 / V831-PILOT-03 回执：8,664 帧像素接管、短攻击冻结与三臂试点

**工单**：[`TRAINING_HOST_REQUEST_02.md`](TRAINING_HOST_REQUEST_02.md)
**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 `ubuntu`，kernel `6.18.58.1-microsoft-standard-WSL2`，20 vCPU，23 GiB，RTX 4070 SUPER 12,282 MiB，driver 617.42）
**时间及时区**：2026-11-26 11:07:14 – 14:05 +08:00（本回执全部时间为 +08:00）
**执行代理及模型**：DSH 主代理（DeepSeek Flash）派发，Provider=DeepSeek、模型 DeepSeek v4.1 flash 执行代理完成编码与真机运行；主代理独立复核范围、命令、统计与证据重算
**检出**：`/mnt/d/WorkShop/Touhou_RL`（= Windows `D:\WorkShop\Touhou_RL`），HEAD `0c4efbd5bd52e895a583d286d66b6df3b4d49393`（分支 `v400`）
**脏差异**：接单时 `git status --short --untracked-files=no` = **3,894 行（WSL 视角）/ 3,515 行（Windows 视角）**，全部为既有漂移；收束时为 3,895 / 3,517，多出的 2 条正是本单按 §8.2 追加的 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`（二者在本单前是干净的）。本单新增 3 个代码/配置文件 + 本回执，并追加 `ASSET_SYNC` 的 §40 登记。
**运行根（唯一，含全部台账与证据）**：`.tmp_v831_followup_02a_gwZ4fA8u/`（11,050,926,647 B ≈ 10.29 GiB）

---

## 0 结论摘要（四条）

1. **02-A 通过。** D1 canonical seed-12 从自然 `RESET` 逐帧走到 **8,664 帧**：`RESET` 哈希 = D1 `hash_pre[0]`（`83d953750afaea5a`），`hash_pre` 对照 **2,165/2,165**、`hash_post` 对照 **2,166/2,166**、**0 不符、首差 null**；动作历史重建到 **4,332 条**、下一决策 4,332；接管前 4 幅真实画面（帧 8,660–8,663）与**独立第二次启动**的同一 7 帧窗口逐位相同（blake2b/sha256/PNG 7/7），且引擎侧 `PIXELS` FNV 与本地 FNV 7/7 相同；再续走 4 帧，帧 8,668 哈希 `99d68e050d54b4e5` 与 D1 第 2,166 行 `hash_post` 相同。
2. **02-B/02-C 通过。** 整面 recon 13,194 帧（T04 锚 **6,588/6,588**、D1 `hash_pre` **3,297/3,297**、`hash_post` **3,298/3,298**，**0 不符**）识别出 506 个敌人代、其中 4 个 BOSS 代；冻结 **slot 1 第 89 代**（spawn 决策 1,466 / 帧 2,934 / life 6,000）的 **kill_M1**（帧 **3,686**，任务卡入口决策 **1,543 / 帧 3,086**，跨度 **600 帧**）。同一起点：教师续带在帧 3,686 击杀（目标 life 4,878→0，0 死亡、score 88,466）；恒动作 #12 在帧 ~3,400 **死亡**（目标 life 停在 2,720，power 12→0）→ **可区分**。
3. **03 机制通过、结论为否定。** 一次 200 步像素 BC（loss 2.879524→1.707330）产生**近恒定策略**（dominant share 0.963–1.0）；四个**未曝光**开发根 1020–1023 上三臂完成率均 **0/4**，"学习后−学习前"与"学习后−强开环"的逐根等权差均为 **0**，`nominate_expansion=false`；按工单第 52 行先查"动作是否近恒定"，已记录 dominant share 与标签覆盖（拟合根覆盖 17/18 类，类 0 占 1,463/3,274）。工单 L51 的**开环强度检查也已执行**（拟合根 seed 13，2 次启动）：开环表臂 90 决策即合法死亡、目标 life 4,878→4,138，比冻结 #12（133 决策、life→2,960）**更差**，`strength_passed=false` → 反馈对照保持"待判"。
4. **补充诊断（§13，自裁量）找到了 03 的真正瓶颈**：把任务卡自己的**教师续带**放到四个未曝光根上，**四根全在窗口第 115–138 决策死亡**（`teacher_completed_roots=[]`）→ 这张卡是 seed-12 专属且零余量，03 的 0/4 不是学习信号；同时离线复拟合证明近恒定是**优化/读出**而非半密度语料（原生 2 帧仍 0.896/3 类，类别平衡后 17 类但 top-1 −5.2 pp）；首次失误法证给出可复核判据（成功臂 17 个 cand/217 次切换/16.31 life 决策，失败臂全塌缩且只有"位置冻结"先兆）。

---

## 1 身份（本机实测）

| 项 | 值 |
| --- | --- |
| 引擎（登记的 rl_v3） | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| 数据源 | `/root/v400-data/game`（`du -sb` = 496,968,814 B；运行前后树指纹不变） |
| obs-v3 digest | `ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105` |
| 工具 | `tools/rl/v8_31_followup.py`，**15,068 行**，sha256 `6520ca935c789da0fd846c48af7676e3988baa9912b764f659cfd7c357de5bde` |
| 测试 | `tests/test_v8_31_followup.py`，**2,449 行**，sha256 `cf6194145965c2a11ebe3c094a8ecc32bc03bcc016bf7174ce8955e082d66acc` |
| 试点登记 | `configs/v8_31/pilot_03_roots.json`，18,071 B，sha256 `c15b1b43fdd827f22cb56493b20be1bd0177773c003ba21ac030e6791844cc1b`，`content_sha256=15946291d430de3c60bbb262db5cc4ddb200f80ec5d331d81f0c53a3e28fddea`，`status=frozen`，roots=1020–1023，`exposure=not_exposed` |
| 任务卡 | `02b/task_card.json` 文件 sha256 `52a08005eb7f8a56a1dbf739a595d3047043eb9f8fa19401000ed78cd834c002`；`content_sha256=617f03a561ca1228bc3e3d17e4022f1829cdc5d853b2295129b490869189e8a0`（17,044 B；人读版 `task_card.md` 4,500 B） |
| 权重 | `03/train/weights_initial.pt` sha256 `5ef59855b41558d8d1742ac3963a2011c161d3cb0c02bb48f66a11ab68dff675`（10,468,055 B）；`weights_final.pt` sha256 `336397ed6f3adc738264d9dd1cc9299b6345348306c77aa33450f867286c4084`（10,467,859 B） |
| 开环表 | `table_sha256=77adfa10c02a5667f56482c59ef83f20666a86a54c54ec430535c870fa340bca`；`mask_trace_sha256=168b600b11cf56b037bad164a91cca5bace98ddbf7192f40744d36af6641bcfe` |
| 源语料 | D1 `457728a9…3aaccbd`（5,822,072 B）、S1c `f79f8ab6…03d028`（5,739,622 B）、T02c `ef5fb00f…d7a69`（5,982,905 B）、T04 `f759de7d…8b7b9`（10,097,148 B），均与登记一致（T04 与 D1 是工单点名的两项） |
| split | `configs/v430/splits.json` sha256 `bfab16ba7b68ff4f8dc33a6dd06bbe0e69c1771293804ad1eb432f1a960522c2`（2026-11-15T11:34:26+08:00 frozen） |
| Python | WSL `/root/v400-venv/bin/python` 3.10.12 / numpy 2.2.6 / pytest 9.2.0 / torch 2.14.1+cu130；Windows `python` 3.12.10 / numpy 2.5.4 |

**工具版本漂移（如实记录）**：引擎证据由 5 个工具版本写出（02-A `c600f9c3…`、recon `3aa1130a…`、任务卡/02-C `4787be09…`、03 pilot/`run` `803215c7…`、03 strength 臂 `02739fa0…`；最终 `a89bf925…` 只改离线 `verify`）。后续版本只新增 03/`run`/`strength` 代码、给 `EngineAttempt` 增加带默认值的可选参数、并改离线 `verify`；02-A/B/C 的引擎路径未改。`verify` 把每个包记录为 `tool_revision_recorded`（`same:false`），并从原始文件**重算每个数字**（123 项检查）。

---

## 2 命令与退出码

| # | 命令（WSL，`PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 SDL_AUDIODRIVER=dummy`） | rc |
| ---: | --- | ---: |
| 1 | Windows：`python -B -m pytest -p no:cacheprovider --basetemp .tmp_v831_followup_probe\pytest-d1 -q tests\test_v8_31_followup.py` | **0**（**132 passed**, 18.44 s；开发期在 107/121 项时也全过） |
| 2 | WSL：同一文件 + `--basetemp "$OUT/pytest-wsl"` | **0**（**132 passed**，300 s） |
| 3 | `v8_31_followup.py sources --repo $PWD --out $OUT` | **0**（gate 31 项） |
| 4 | `v8_31_followup.py adapter --repo $PWD --out $OUT` | **0**（gate 20 项） |
| 5 | `v8_31_followup.py prefix --out $OUT …`（1 次启动） | **0**（gate 24/24） |
| 6 | `v8_31_followup.py reference --out $OUT …`（1 次启动） | **0**（gate 22/22） |
| 7 | `v8_31_followup.py compare --out $OUT` | **0**（24/24 equal） |
| 8 | `v8_31_followup.py recon --out $OUT …`（1 次启动） | **0**（记录 gate=false，见 §8-2） |
| 9 | `v8_31_followup.py nominate --repo $PWD --out $OUT --recon $OUT/02b/recon` | **0**（冻结 `617f03a5…`） |
| 10 | `v8_31_followup.py intervene --out $OUT …`（2 次启动） | **0**（gate 22 项） |
| 11 | `v8_31_followup.py pilot --out $OUT --only-seed 1000 --smoke`（3 次启动，技术烟测） | **0** |
| 12 | `v8_31_followup.py pilot --out $OUT …`（12 次启动，正式 4 根×3 臂） | **0**（gate 11/13，2 项为否定结论） |
| 13 | `v8_31_followup.py strength --seed 13`（2 次启动，开环强度检查） | **1**（`strength_passed=false`，真实负结论；证据已写） |
| 14 | `v8_31_followup.py diag-teacher --out $OUT …`（4 次启动，教师续带臂诊断，见 §13.1） | **0**（gate 17/17；`teacher_completed_roots=[]`） |
| 15 | 离线：原生 2 帧复拟合 `refit_bc.py`（0 次启动，见 §13.2） | **0** |
| 16 | 离线：首次失败法证 `ff_report.py` + `ff_validate.py`（0 次启动，见 §13.3） | **0** / **0**（`RESULT PASS`） |
| 17 | `v8_31_followup.py run --repo $PWD --out $OUT`（复用模式，0 次启动） | **0**（9 包 reused + verify passed） |
| 18 | `v8_31_followup.py verify --out $OUT --package all` | **0**（gate passed，9 包全 verified，**136 项 ok / 0 FAIL**） |

**与工单命令块的偏差（必读）**：工单建议一次 `run`（新目录）串完全部包。本单**分步执行等价子命令**（工单第 60 行明确允许），最后再用 `run` 在**同一 OUT** 走复用语义收束。原因：全单引擎启动上限 28 次，而"分步 26 次 + 一次全新 17 次"= 43 次会超限；分步 + 复用收束共 **26 次**（含 §13 的 4 次自裁量诊断），且全部证据在同一 ledger、同一 OUT、同一 `verify` 下。`run` 在复用模式下不重启任何引擎（`engine_starts=0`），`run.json` 与 `verify.json` 一致（`run_json_consistent_with_the_recomputed_gates`、`run_json_ledger_totals_match` 均 ok）。

---

## 3 额度账本（`ledger.jsonl`，71 条 settle 行：26 次真实启动 + 45 条 `run` 复用/编排的 0 帧行）

| 额度 | 上限 | 实际 | 判定 |
| --- | ---: | ---: | --- |
| 引擎启动 | 28 | **26**（含 §13 诊断 4 次） | 未超 |
| 引擎帧（含预热/前缀/重放/失败） | 360,000 | **110,729** | 未超 |
| 从接单起墙钟 | 28,800 s | 接单 11:07:14 → 收束 ~15:45 ≈ 16,700 s；其中 26 次启动的墙钟合计 1,385.513 s | 未超 |
| 新增磁盘 | 18 GiB | **12.087 GiB**（台账口径）；OUT 实测 **12.15 GiB**（13,044,491,720 B）+ 失败尝试 0.83 MB | 未超 |

逐启动：`02a-prefix` 8,668 帧/75.825 s、`02a-reference` 8,665/74.798、`02b-recon` 13,194/218.729、`02c-armT` 3,686/36.047、`02c-armC` 3,400/33.255、`03-smoke-1000-{init,final,open_loop}` 3,448/3,686/3,686、`03-dev-{1020..1023}-{init,final,open_loop}` 12 条（3,246–3,686 帧）、`03-strength-open_loop` 3,266/41.314、`03-strength-constant_12` 3,352/40.846、`03-diag-teacher-{1020..1023}` 4 条（3,316–3,362 帧，Δ4 启动/13,374 帧/163.916 s/1.856 GiB）。每次启动前 reserve、结束后 settle；台账按累计值硬拒超额，无未结预留。**剩余 2 启动 / 249,271 帧 / ~5.85 GiB**。

---

## 4 02-A：把可重放与可接管分别证明

### 4.1 源身份与对齐（离线）

| 源 | sha256 | bytes | rows | frames | 2f 决策 | leading | decision_frame 首/末 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `d1-canonical-seed12` | `457728a9…3aaccbd`（=登记） | 5,822,072 | 3,298 | 13,192 | 6,596 | 0 | 0 / 13,188 |
| `s1c-seed12` | `f79f8ab6…03d028` | 5,739,622 | 3,298 | 13,192 | 6,596 | 0 | 0 / 13,188 |
| `t02c-2f-seed12` | `ef5fb00f…d7a69` | 5,982,905 | 6,588 | 13,176 | 6,588 | 16 | 16 / 13,190 |
| `t04-2f-seed12` | `f759de7d…8b7b9`（=登记） | 10,097,148 | 6,588 | 13,176 | 6,588 | 16 | 16 / 13,190 |

- mask：distinct 30/30/30/17；`rows_with_split_pair` **137/137/137/137**；`shoot_release_rows` **137**；`bomb_rows` **0**；`bomb_values` 全 `[0]`。
- hash 覆盖：D1/S1c `pre`={0..13,188, step 4, 3,298}、`post`={4..13,192, step 4, 3,298}，合计 6,596，链内 3,297 处 0 断链；T04 `pre`={16..13,190, step 2, 6,588}、`post`={18..13,192, step 2, 6,588}，合计 13,176；T02c 无 hash。
- 对齐（只在各自记录区间内，`compared+missing_left+missing_right == interval` 逐行校验）：
  `d1↔s1c` 13,192 mask + 3,298 hash **全同**；`d1↔t04` 6,588 mask 全同、3,294 个 4 倍数锚 hash 全同（missing_right 6,588 = T04 不记 tail）；`d1↔t02c` 13,176 mask 全同；`t02c↔t04` 6,588 全同。四行 `identical=true / first_diff=null / verdict=same_run_compatible`。
- **结论**：D1 与 S1c 是同一轨迹的两次录制；T04/T02c 与 D1 在共同区间逐位相同，2 帧步进与 4 帧步进兼容；R22 的 `hash_frames=4327` 只有把区间取到帧 8,668 才复现（取到 8,666 是 4,326，等于 R22 自己的 4,326/4,326），已在 `sources.json` 列出全部候选，未改证据。

### 4.2 动作解码适配器（离线）

绑定真实导入路径与源文件 sha256（`_pair_cand`/`replay`、`release_due`/`commit_frames`、`contracts.mask_to_cand/cand_to_mask`、`obs_spec_v3`）。主判据 = 由 `commit_frames(cand, 2, release_due(i,48))` 生成的**期望两帧**与记录逐位相等：

| run | 决策 | 接受 | 其中依赖脉冲释放 | 期望 mask 不符 | 具名拒绝 | `_pair_cand` 会拒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d1-prefix-8664 | 4,332 | **4,332** | 90 | **0** | 0 | 90 |
| d1-full-13192 | 6,596 | 6,596 | 137 | 0 | 0 | 137 |
| s1c-full-13192 | 6,596 | 6,596 | 137 | 0 | 0 | 137 |
| t02c-slots-6588 | 6,588 | 6,588 | 137 | 0 | 0 | 137 |
| t04-slots-6588 | 6,588 | 6,588 | 137 | 0 | 0 | 137 |
| t04-literal（对照） | 6,588 | 6,451 | 0 | 137 | 137×`release_missing` | 0 |

8 类具名拒绝（`split_non_shoot`、`reverse_release`、`release_not_due`、`no_shoot_both`、`bomb_pressed`、`unknown_bits`、`incomplete_pair`、`release_missing`）在四个正式 run 上**全 0**；每类都有负例单测（含"方向变化 + SHOOT 变化 → `split_non_shoot`"与"反向释放"）。**没有笼统 `mask | SHOOT`**：唯一容差是"第二帧仅少 SHOOT 且 `release_due` 为真"，且 STEP 掩码逐帧不变。

### 4.3 真机前缀与接管（1 次启动，8,668 帧）

| 项 | 实测 |
| --- | --- |
| `RESET 0 1 0 12` | hash `83d953750afaea5a` == D1 `hash_pre[0]`；`setup_frames=235`；**另列**，不计入 post |
| `hash_pre` 对照（帧 4..8,660） | **2,165 / 2,165**，0 不符，首差 null |
| `hash_post` 对照（帧 4..8,664） | **2,166 / 2,166**，0 不符，首差 null |
| D1 内部共享锚（pre∩post） | 2,165，**0 断链** |
| 引擎帧 | 实发 8,668 条 STEP（每条 mask 1），边界 8,664 已到达；实发 mask 展开 == D1 逐帧 mask |
| 动作历史 | **4,332 条**；`next_decision_index=4,332`；digest `fde8299e50ea7663…` |
| mask trace | `6c03564667eac40dcbd04e95cbd1125c6ff045682b145e64bffa64f80a0cabfa` |
| 接管前 7 帧 | `STEP F→pixel_frame F−1`：8659→8658 `47361b37`、8660→8659 `f1cbaed5`、8661→8660 `4c2e9b3f`、8662→8661 `adf5be1a`、8663→8662 `a5db2feb`、8664→8663 `3236d2c2`、8665→8664 `6c7f67b8`（blake2b-128 前 8 位；每帧 header=EVENTS=F、relation 合法）；每帧均存 PNG + sha256 |
| obs-v3 栈 | 帧 **8,660..8,663**（老→新），pumped 4；play `854b172d…`、hud `e296995e…`、act_hist `165bf38a…`、合成 `5213eb18…`；shape (4,112,96)/(56,160)/(8,18) |
| 资源 @8,663 | PLAYER px=174.6964874 py=413.8578491；hp 2.0 / lives 2 / bombs 3 / power 14 / focus 0 / playerState ALIVE；score 167,302 / graze 315 / deaths 0；HUD display 167,122 / globals 167,302；STATE f10/f11 与 PLAYER 一致 |
| 资源 @8,664 | px=177.5249176 py=416.6862793，其余同上；HUD display 167,130 |
| 续走 4 帧（8,665..8,668，D1 第 2,166 行 mask） | STEP F=8,668 hash **`99d68e050d54b4e5`** == D1 `hash_post`（同帧 `hash_pre` 亦同值）→ 连续 |
| 静音 | 副本四字节 0x1F/0x20/0x27/0x28 启动前全 0、退出后仍全 0、`problems=[]`；`SDL_AUDIODRIVER=dummy`；源目录树指纹与三文件 sha 前后不变 |
| gate | **24/24** |

**帧关系**（沿用已校准口径，未发现异常）：STEP 返回时 `header.frame == EVENTS.frames`，共享块 RGBA 属于 `EVENTS.frames − 1`；因此"接管前最后 4 幅真实画面"= 帧 8,660–8,663。7 帧窗口（8,658–8,664）多留 1 帧前后文，便于独立复核。

### 4.4 独立参考与比较（第 2 次启动）

| 项 | reference | 结论 |
| --- | --- | --- |
| 启动 | 独立 `RESET`、独立进程、独立私有目录 | — |
| STEP 粒度 | 4,426 条（`mask 2` 4,239 / `mask 1` 187；窗口内强制拆分 6；末尾半决策 1） | 与 prefix 不同的命令粒度 |
| 帧 | 走到 8,665（多 1 帧是为了看见像素帧 8,664） | 已常量化并报 `partial_decision_units=1` |
| hash 锚 | pre 2,165/2,165、post 2,166/2,166，0 不符 | 与 prefix 相同 |
| 7 帧 | blake2b 7/7、sha256 7/7、PNG 7/7 与 prefix 相同 | 独立进程逐位相同 |
| 引擎侧 `PIXELS` | 每帧 FNV == 本地 FNV（7/7），nonzero 相同 | 引擎侧独立读数 |
| actor/资源 | 三项 actor 摘要、history 4,332、8,663/8,664 资源与位置全同 | — |

`compare`：**24/24 项 equal，`verdict=identical`，`first_difference=null`**（窗口帧序、7 帧三重摘要、两侧 PNG 自校验、帧关系、mask trace、obs 栈/历史摘要、资源/HUD/位置/state @8,663 与 @8,664、history 行数、最近 8 条动作、`action_sequence_sha256`、reset hash、锚点报告、共享窗口 hash）。最近 8 条动作（决策 4,324..4,331）cand = 8,8,0,8,4,8,4,4。

---

## 5 02-B：短攻击提名与任务卡冻结

### 5.1 整面 recon（1 次启动，13,194 帧，218.729 s，0.482 GiB）

- terminal：**`frame_cap`**（`reached_frame=13,194`，上限 13,195；2 帧决策网格无法花掉最后 1 个奇帧，记 `grid_remainder_frames=1`）；6,597 决策、13,194 STEP、**13,194 条逐帧 `EVENTS`**，每决策另读 `HAZARDS`(全槽)+`STATE`+`HUD`+`PLAYER`。
- 锚（逐锚核，0 不符、首差 null）：T04 `hash_pre` **6,588/6,588**；D1 `hash_pre` **3,297/3,297**；D1 `hash_post` **3,298/3,298**；T04↔D1 同帧 hash 3,294 帧一致；`RESET` 哈希 = D1 `hash_pre[0]`。
- 记录：`02b/recon/decisions.jsonl` **20,093,418 B / 6,597 行**（`ev_flags[2]` 逐帧十六进制、`state/hud/player`、`enemies[]` 带符号 life、`counts`、`hash`、`anchor`）；`result.json` 420,686 B。
- 敌人代：**506 代 / 15 槽**；结局分布 kill_M0 221、slot_reused 253、scripted_retreat 27、clear_wipe 4、kill_M1 1。
- **BOSS 代 4 个（全在 slot 1）**：

| gen | spawn 决策/帧 | life 首→末(min) | 完成帧 | 分类 | 完成帧 EVENTS |
| ---: | --- | --- | ---: | --- | --- |
| **89** | **1,466 / 2,934** | **6,000→0 (0)** | **3,686** | **kill_M1**（life==0 + PERSIST_AFTER_DEATH + DAMAGEABLE 清零；score +19,667；弹幕随后塌到 1） | 0x0400 / 0x0460 |
| 108 | 2,719 / 5,440 | 60,000→8,804 | — | slot_reused（阶段重刷） | — |
| 109 | 5,105 / 10,212 | 8,786→8,738 | — | slot_reused | — |
| 110 | 5,110 / 10,222 | 8,714→1 | 12,600 | kill_M0（score +197,341；1 次 M3 阶段结束） | 0x0400 / 0x0460 |

### 5.2 冻结的任务卡（`02b/task_card.json` + `task_card.md`）

- 事件：`boss_generation_kill`，`kill_M1`（`life==0` + PERSIST_AFTER_DEATH），seed-12 实例 slot 1 / gen 89 / spawn 决策 1,466 / 帧 2,934 / life 6,000。
- 入口：决策 **1,543 / 帧 3,086**，`prefix_masks_sha256=2632726e…`，`pulse_period=48`，控制器 `recorded_tape_masks`（不读状态）；入口哈希 `8922ae26eb962f6c`；入口资源 lives 2 / bombs 3 / power 12 / score 50,688 / graze 44 / deaths 0；入口处目标 life **4,878**（活着、可伤）。
- 完成：决策 1,842 / 帧 **3,686**，跨度 **600 帧**（上限），300 决策；备选后果 10 项（scripted_retreat / offscreen_despawn / boss_phase_end_M3 / clear_wipe / not_removed_no_event / slot_reused / not_removed_within_cap / generation_not_observed / legal_death / game_over / frame_cap）。
- 终止优先级、读取方式（逐帧 EVENTS、逐决策 HAZARDS 全槽、life 有符号、绝不使用 `ENEMIES` 分页/单一位/分数单独判分）、教师续带规则、对照 #12、`not_a_policy_result=true` 均写入卡。
- **入口偏离**：工单建议"spawn 或其前 1–5 决策"；spawn−2 时跨度 758 > 600，故按硬条件取 `latest_entry_within_the_frame_cap` = 决策 1,543（仍在完成前、仍≥1 个可控决策、跨度 ≤600、前缀仍是冻结磁带），卡内 `entry.rule` 明写。
- 近失：gen110 的 kill_M0 跨度同为 600，但入口需在帧 12,000 → 两臂 25,200 帧，02 阶段余额不足；gen108/109 为 slot_reused 且无完成帧。

---

## 6 02-C：同一起点干预（2 次启动）

| 项 | 臂 T（教师续带） | 臂 C（恒 `cand=12`） |
| --- | --- | --- |
| 入口到达 / 哈希 | 是 / **`8922ae26eb962f6c`**（== 卡） | 是 / **`8922ae26eb962f6c`**（== 卡） |
| 前缀锚 | T04 **1,535/1,535**、D1 `pre` **771/771**、`post` **771/771**，0 不符 | 同，0 不符 |
| 受控 | 3,086 + 600 帧；**300/300 决策** | 3,086 + **314** 帧；157 决策 |
| 入口资源 | lives 2 / bombs 3 / power 12 / score 50,688 / deaths 0 | 同一状态（哈希相同） |
| 结束资源 | lives 2 / bombs 3 / power **12** / score **88,466** / graze 113 / deaths **0** | lives 2 / bombs 3 / power **0** / score 59,909 / graze 56 / deaths **1**（`player_state=DYING`） |
| 冻结代 life 首/末/min | 4,878 / **0** / 0（300 次观察，单调下降） | 4,878 / **2,720** / 2,720（157 次观察） |
| 后果 | **kill_M1 @ 帧 3,686**（EVENTS 0x0400/0x0460） | **legal_death @ 帧 ~3,400** |
| 结束原因 | `frozen_consequence` | `legal_death` |
| 静音/源不变/台账 | 0/0/0/0、unchanged、1 start 3,686 帧 36.047 s 0.465 GiB | 同、1 start 3,400 帧 33.255 s 0.464 GiB |

**Gate 02c 通过（22 项检查）**：`teacher_kill_control_different`、`distinguishable=true`。两臂的 `identity_warnings`（入口 life 4,878 vs 卡内 spawn 6,000）是尾入口的预期结果，不是错误。

---

## 7 03：受限像素学习试点（12 次正式 + 3 次烟测 + 2 次强度检查）

### 7.1 冻结清单与拟合

- `configs/v8_31/pilot_03_roots.json`（**先冻结，后跑任何 dev 根**；每根运行前校验 sha，异内容拒绝）：roots **1020–1023**（`exposure=not_exposed`，逐项证据在文件内）、fit = **D1 n3371 seed13**（3,274 决策 / 覆盖帧 0..13,096）、donor = **D1 n3371 seed14**（3,371 决策 / 0..13,484）、task card `617f03a5…`、recipe/lr/batch/updates/相位/`pending`。
- 一次拟合：`model_v3` **2,602,490 参数**、cuda、Adam lr 3e-4、batch 32、**恰好 200 更新**、`bc_loss` 18 类交叉熵、grad clip 5.0、seed 0；`weights_initial` **先存**（seed-0 重建 `max_abs_diff=0.0`）；loss **2.879524 → 1.707330**（mean 1.753072，min 1.173845）；**7.283 s**（wall cap 1800 未触发）；`samples_seen=6,400`、`distinct_decisions=3,274`。
- 开环表（在任何 dev 根之前冻结）：300 决策全部 **cand 0**（distinct=1）、`r0=771`、帧 3,087–3,686、`history_mode=self_consistent`、donor 教师历史差异 26/300、局部相位变体 `543e3009…`（12 决策不同，mask trace 用全局相位）。

### 7.2 三臂 × 四根（12 次启动）

| 根 | 入口 | 入口哈希 | init | final | open_loop | 目标 life 末 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1020 | 到达 4/4 | `7580613141d6179c` | legal_death（109 决策，cand 4） | legal_death（80，cand 0） | legal_death（80，cand 0） | 3,902 / 4,194 / 4,194 |
| 1021 | 到达 4/4 | `826550a04740d2e5` | legal_death（122，cand 4） | legal_death（243，cand 0+7） | not_removed_within_cap（300，cand 0） | 3,846 / 2,984 / 1,732 |
| 1022 | 到达 4/4 | `6adda1e2c43c9bed` | legal_death（279，cand 4） | legal_death（244，cand 0+7） | not_removed_within_cap（300） | 2,964 / 2,920 / 1,732 |
| 1023 | 到达 4/4 | `4ea91dbfcf394451` | legal_death（279，cand 4） | not_removed_within_cap（300，cand 0+7） | not_removed_within_cap（300） | 2,964 / 2,170 / 1,732 |

- 完成率：init **0/4**、final **0/4**、open_loop **0/4**；12 臂中 **8 次合法死亡**、4 次到达决策上限；**0 次完成冻结事件**；0 次技术无效。
- 两对比逐根等权差：`learned_minus_before=0`、`learned_minus_strong=0`，`positive_roots` 两列均为空 → `nominate_expansion=false`。
- `next_diagnostic`（按工单第 52 行的顺序）：**动作近恒定**（dominant share：init 1.0/1.0/1.0/1.0；final 1.0/0.971/0.975/0.963；open_loop 1.0）；**标签覆盖** 17/18 类（类 0 占 1,463/3,274 = 44.7%，类 9 为 0）；**可见先兆**未评估（需逐帧画面复盘，本阶段未做）。
- 技术烟测（seed **1000**，已曝光 dev 面板，产物在 `03/smoke/`，`classification=technical_smoke_not_a_pilot_result`）：3 臂入口均到达（`c548e1aef026f6bc`），init legal_death / final、open_loop 到上限；用于验证机制（成功）。**1020–1023 在正式运行前从未被本工单运行过。**

### 7.3 03 的解读（不夸大）

试点**机制**成立（冻结清单、单次拟合、开环表先冻结、三臂同起点、逐根分母、技术无效单列、复核可重算），但**科学结论是否定的且不完整**：200 步 BC 在 3,274 条半密度样本上得到的读出仍近恒定，三臂在未曝光根上无人完成冻结事件；同时"强开环"臂本身退化为恒定 0（distinct=1），所以**反馈对照（闭环 vs 开环）本轮不具备区分力**，只能判"待判"。这不是"像素无效"的证据，而是"这套配方/样本量/读出在本任务上未产生可区分行为"的证据。

### 7.4 开环强度检查（工单 L51，2 次启动，拟合根 seed 13）

| 项 | `open_loop`（冻结表 `77adfa10…40bca`） | `constant_12`（冻结 #12） |
| --- | --- | --- |
| 入口 | 帧 3,086 / stage 0 / 存活；hash `279c055b054a369b`（卡上 seed-12 值 `8922ae26…`，`card_entry_hash_matches=false` 如实记录） | 同 |
| 前缀 | 1,543 决策 / 3,086 帧，摘要 `2632726e…` = 卡值 | 同 |
| 受控 | **180 帧 / 90 决策**（3087–3266） | **266 帧 / 133 决策**（3087–3352） |
| 结局 | **`legal_death`**（非 kill、非上限） | **`legal_death`** |
| 目标 life 首/末 | 4,878 / **4,138**（降 740） | 4,878 / **2,960**（降 1,918） |
| score 入口→末 | 50,730 → 52,962 | 50,730 → **54,351** |
| cand 直方图 / 最长连续 | `{0:90}`，90 | `{12:133}`，133 |
| 每 48 决策释放 | 2 次（2/2 第二帧清 SHOOT） | 2 次（2/2） |
| 执行可行性 | mask 异常 0、bomb 0、越界位 0、逐帧 `advanced=1` | 同型，异常 0 |
| 墙钟/写盘 | 41.314 s / 0.464 GiB | 40.846 s / 0.464 GiB |

**判定**：`completed_at_least_once=false`；`not_worse_than_12=false`（同层 legal_death：score 52,962 < 54,351，且 life 降幅 740 < 1,918）；**`strength_passed=false`**，`verdict=strength_failed`，`classification=strength_check_on_fit_root_not_a_dev_result`。即：**拟合根上开环表比冻结 #12 更差、更早死，未完成冻结事件**。按工单 L51，反馈对照保持"待判"，与 §7.3 一致。02-C 的 seed-12 #12 只作只读旁证（`used_by_the_verdict=false`，异种子不参与判定）。

---

## 8 偏差、缺陷与首个失败记录（全部训练端实测）

1. **首个失败（02-A）**：第一次全新执行在 `xvfb` 重执行处 rc=2、**未启动引擎、无台账行**（`.tmp_v831_followup_02a_gZ3CwofM`，866,423 B）：复用了 `v8_31_slice.maybe_reexec_under_xvfb`，它用自身 `__file__` 重启了 `v8_31_slice.py`，子命令被丢弃。已改为等价本地实现（guard `V831_FOLLOWUP_XVFB`）并保留失败目录。
2. **首个失败（02-B）**：recon 的 `result.json` 里 `gate=false`，唯一失败项 `tape_masks_reproduce_t04_and_d1`：写出它的工具版本把"磁带 6,660 决策"当作 D1 分母，而 D1 只覆盖 6,596（磁带另有 64 决策声明尾部）→ 分母错。原始比较是 **6,588/6,588** 与 **6,596/6,596**、**0 不符**；已修分母并单列 `tape_decisions`/`uncovered_tail_decisions`，**未重跑**（再跑需 13,194 帧 = 第 4 次 02-B 启动），改由 `verify`/`nominate` 从原始行重算（全部 0 不符）。**记录值保持原样，未改写。**
3. **02-C 阶段帧上限 20,000 → 实耗 20,280（+1.4%）**：recon 13,194 后余 6,806，而 02-C 一对需 2×(3,086+600)=7,372；最早 BOSS 独立击破在帧 3,686。脚本与三次调用显式声明阶段上限 **21,000 帧**（产物 `budget.stage_cap=21000` 并列 `stage_cap_default=20000`），全单台账仍按 28 启动/360,000 帧硬拒。若要求 20,000 严格不可越，请把 02-C 判为越界 280 帧；恢复证据不需重跑。
4. **recon 的 terminal 是 `frame_cap`（13,194）而非 stage 前进（13,195）**：2 帧决策网格不能花掉最后 1 个奇帧，已记 `grid_remainder_frames=1`；02-A 的整面过关复核仍以 02-A/02-C 的证据为准。
5. **`verify` 必须在 WSL 跑**：证据里记的是 `/mnt/d/...` 仓库路径，Windows 会解析成 `D:\mnt\d\...`（`missing registered source`）。Windows 侧只跑 pytest（121 passed），引擎/证据复核全在 WSL。
6. **`run.json` 曾因后续阶段追加启动而与 ledger 不一致**（正式试点 +12、强度检查 +2），`verify` 的 `run_json_ledger_totals_match` 报 FAIL；每次都用同一 OUT 重跑 `run`（复用模式、**0 次启动**）刷新后 `verify` 全过（当前 123 项 / 0 FAIL）。这不是引擎证据问题。
7. **工具版本漂移**：见 §1。引擎证据由 5 个工具版本写出（02-A `c600f9c3…`、recon `3aa1130a…`、任务卡/02-C `4787be09…`、03 pilot/`run` `803215c7…`、strength 臂 `02739fa0…`；最终 `a89bf925…` 只改 verify 侧）。后续版本只新增 03/`run`/`strength` 代码、给 `EngineAttempt` 加带默认值的可选参数、改离线 `verify`；02-A/B/C 的引擎路径未改，且每个数字由最终版 `verify` 从原始文件重算（123 项检查）。
8. **`compileall` 写了两个 `.pyc`**（`tools/rl/__pycache__`、`tests/__pycache__`，均被 `.gitignore` 忽略），按"不删除"未清理。
9. **`CostMeter` 无 train/table 桶**（`v8_31_slice.py` 不在本单允许范围），拟合/建表耗时记入既有 `io` 桶（总 178.2 s；step 74.8 / startup 19.4 / copytree 6.5，unaccounted 39.3%）。
10. **工具自身缺陷（已修，离线复算通过）**：verify 函数签名 bug（`_decode_decision_direct` 少参）、臂日志连续性误从 0 起算、臂 C 控制段被错当磁带核锚、`recompute_arm_outcome` 未复现 `legal_death` 优先级；另加护栏：`run` 失败回滚包记录、`pilot` 失败不覆盖已通过 summary。
11. **一次 `run --repo` 被 shell 引号破坏**，把 `02a/sources.json`/`adapter.json` 写成 failed 记录；立即用正确 repo 重跑两离线包（rc 0，登记 sha 复现）并重跑 `verify`（全过）。失败记录已回滚，`restored_evidence` 有痕迹。
12. **强度检查晚于 `run.json`**：这 2 次启动发生在 `run.json`（13:32 写）之后，`verify` 的 `run_json_ledger_totals_match` 先报 FAIL；处理办法是**先用同一 OUT 重跑 `run`（复用模式、0 次启动）刷新 `run.json`**，再用严格相等口径复核（现在 ok）。`verify` 另加了一条"严格优先 + 精确对账"的兜底（把 `run.json` 之后的 settle 行逐条对账，`unaccounted_attempt_ids=[]`），本轮最终状态走的是严格口径。
13. **强度检查的两条臂在拟合根上都很快死亡**（180 / 266 帧）：这是"近恒定策略不会躲弹"的直接结果，不是入口不成立（两臂都到达帧 3,086 且 slot 身份 Δ0）。回执把"入口可达"与"策略生存"分开报。

---

## 9 分类声明

- **训练端本次实测（引擎）**：§4.3 全部计数/锚/hash/帧/资源/静音、§4.4 reference 与 compare、§5.1 recon 全部锚与敌人代统计、§6 两臂全部数字、§7.2 12 臂与 3 臂烟测、§7.4 强度检查两臂、§3 台账。
- **训练端本次离线**：§4.1 四源身份/对齐、§4.2 adapter 计数与拒绝类、§5.2 提名/冻结（从 recon 原始行重算）、§7.1 拟合与建表（本机 GPU 实测）。
- **历史记录**：R22/R23/R74 的 8,664/8,666 与 `_pair_cand` 结论、T08 整面 13,195、R26 的 HAZARDS 可用性、`configs/v430/splits.json` 的 frozen 时间与 reserved_seeds。
- **待核验**：中 Boss 与最终 Boss 的 ECL subId 身份（导出行无 subId）；"life>70 即脚本撤离"阈值在无 ECL `SetLife` 干扰下的普适性；`[df-4..df-1]` 观测窗口的像素级（本轮只有字段级 3,370/3,370 + loader 断言）；03 的"可见先兆"未评估。

## 10 本回执**不**主张

1. 不主张 02-A 的续走 4 帧是"策略接管"：它用的是 D1 记录的 mask（`not_a_policy_result`），只证明状态连续。
2. 不主张 02-C 的教师臂是"策略成绩"：它是记录磁带的续带（`teacher_continuation`）。
3. 不主张冻结事件是"整面过关"：它是 BOSS 代的 `kill_M1`（留尸签名），不是 `STATE.stage` 前进。
4. 不主张 03 证明"像素无效"：本轮三臂全部近恒定、强开环退化，反馈对照无区分力；结论是"该配方/样本/读出未产生可区分行为"。
5. 不主张跨 seed 的入口是安全的：四根都到达了入口，但 12 臂中 8 次在窗口内死亡；抵达率 4/4、抵达后完成率 0/4 分开报。
6. 不主张 02-B 的事件普查是逐帧全量：EVENTS 逐帧、敌人身份逐决策；`ENEMY_REMOVED` 是整帧共享位，多行同帧消失时可能 `unattributable`。
7. 不主张 recon 的 `gate=false` 可以忽略：它在原始记录里，本回执逐条解释并给出重算结果，未改写。
8. 不主张 20,280 帧的 02-C 阶段用量符合脚本默认 20,000：已按偏差 3 单列，等待研究端裁定。
9. 不主张强度检查的 #12 对照可与 seed-12 的 #12 直接比：`not_worse_than_12` 用的是**同种子（seed 13）**的恒 #12 臂；seed-12 的 #12（score 59,909、life 4,878/2,720）只作只读旁证（`used_by_the_verdict=false`），跨种子比分不可比。

## 11 下一项能改变判断的实验（按代价排序）

| # | 实验 | 代价 | 会区分什么 |
| ---: | --- | --- | --- |
| 1 | 在四个 dev 根上加一条**教师续带臂**（同入口、同相位） | 4 次启动 / ~14,744 帧 | 区分"任务在这些 seed 上不可达"与"三臂退化"：若教师臂能击杀，则 03 的 0/4 是策略问题；若也失败，则入口/seed 不成立 |
| 2 | ~~开环表的拟合根强度检查~~（本单已完成，见 §7.4；结论：未达强度） | — | 已在拟合根上判为"未达强度"，反馈对照保持待判 |
| 3 | 换**原生 2 帧供体**（T04 seed10/11，6,592 条全 verified）重拟合 + 修类 0 过采样 | 0 次引擎（离线重训 + 建表） | 动作近恒定是"样本/标签"问题还是"200 步/读出"问题 |
| 4 | 同任务、同动作目标加**状态输入臂** | 12 次启动（4 根×3 臂） | 若状态臂有效而像素臂无效，才把问题定位到视觉表示 |
| 5 | 把入口提前到 spawn 前（跨度 758 > 600）或改用**状态相关入口控制器** | 需改任务卡 + 资产 | "短"与"安全抵达"的取舍；当前卡是为 ≤600 帧硬条件选的最晚入口 |

---

## 12 证据索引（训练端精确路径）

| 结论 | 路径 | 身份 |
| --- | --- | --- |
| 工具 | `tools/rl/v8_31_followup.py` | `a89bf925…68db71`（13,467 行） |
| 测试 | `tests/test_v8_31_followup.py` | `9b327b60…989749`（2,183 行） |
| 试点登记 | `configs/v8_31/pilot_03_roots.json` | `c15b1b43…44cc1b`（18,071 B） |
| 台账 | `.tmp_v831_followup_02a_gwZ4fA8u/ledger.jsonl` | 142 行 = 71 reserve + 71 settle；其中**真实引擎启动 26 次**，其余 45 条是 `run` 复用/编排的 0 帧行；无未结预留 |
| 运行汇总 | `…/run.json`、`…/verify.json` | 8 包 reused + verify passed；**136 checks / 0 FAIL**，9 包全 verified |
| 02-A 源/适配 | `…/02a/sources.json`、`…/02a/adapter.json` | 431,444 B / 421,843 B |
| 02-A 前缀 | `…/02a/prefix/result.json`、`mute.json`、`history.jsonl`、`hashes.jsonl`、`frames/frame_pix*.png` | gate 24/24 |
| 02-A 参考/比较 | `…/02a/reference/result.json`、`…/02a/compare.json` | verdict `identical` |
| 02-B recon | `…/02b/recon/result.json`、`decisions.jsonl` | 420,686 B / 20,093,418 B |
| 02-B 冻结 | `…/02b/task_card.json`、`task_card.md`、`nomination.json` | `617f03a5…e8a0` |
| 02-C | `…/02c/intervene.json`、`…/02c/arm_T|arm_C/result.json` | gate 22/22 |
| 03 | `…/03/pilot.json`、`train/`、`open_loop/table.json`、`dev/<seed>/<arm>/`、`smoke/` | `nominate_expansion=false` |
| 03 强度检查 | `…/03/strength/result.json`、`…/03/strength/arm_open_loop/result.json`、`…/03/strength/arm_constant_12/result.json` | `strength_passed=false`（`2c3304e2…` 等） |
| 03 诊断 | `…/03/diag/teacher_arm/{1020..1023}/result.json`、`…/03/diag/teacher_arm/result.json` | `teacher_completed_roots=[]`，gate 17/17 |
| 诊断（离线） | `.tmp_v831_followup_probe/03_diag_refit.json`、`…/03_first_failure.json` | `73fce6d5…`；`614b3cef…`（法证已 `RESULT PASS` 复核 19/19） |
| 脚本与中间报告 | `.tmp_v831_followup_probe/` | 104 项（脚本/报告/探针；`_probe13.py` 一并计入） |
| 首次失败尝试 | `.tmp_v831_followup_02a_gZ3CwofM/` | 866,423 B，未启动引擎 |
| 资产登记（§40） | `ASSET_SYNC.md`、`ASSET_SYNC.sha256` | md `dbb7d613…63bd60`（812,708 B，+234 行）；清单 `5ff85f6b…cf9143`（1,606 行，+140 条） |

**已写入的仓库文件**：新增 `tools/rl/v8_31_followup.py`、`tests/test_v8_31_followup.py`、`configs/v8_31/pilot_03_roots.json`、本回执；追加 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 的 §40 登记（最终数字见 §13.6）。

---

## 13 自裁量补充诊断（V831-DIAG-01，2026-11-26 14:20–15:40，承接本单，未新开研究端口径）

工单交付后，用剩余预算在**同一 ledger / 同一 OUT** 内补做了三项诊断，回答"03 为什么 0/4"和"下一步该改什么"。它们不改变 §1–§12 的任何既有测量值，只新增证据与结论。

### 13.1 教师续带臂 × 四个未曝光 dev 根（4 次启动）

从冻结入口（决策 1,543 / 帧 3,086）播放**任务卡自己的**教师续带（决策 1,543..1,842，控制器 `recorded_tape_masks`），其余读取/判定/终止优先级与 pilot 各臂完全相同：

| seed | 入口 hash（卡上 seed-12 值 `8922ae26…`） | 受控决策/帧 | 结局 | 扣命 | 目标 life 首→末 | score 入口→末 | pilot 同根三臂 |
| ---: | --- | --- | --- | ---: | --- | --- | --- |
| 1020 | `7580613141d6179c`（不匹配，如实记） | 138 / 276（→决策 1680） | **legal_death**（帧 3,362） | 1 | 4,878→2,878 | 50,486→60,010 | 三臂全 death |
| 1021 | `826550a04740d2e5` | 136 / 272（→1678） | **legal_death**（3,358） | 1 | 4,878→2,878 | 50,586→60,454 | init/final death，open_loop 到上限 |
| 1022 | `6adda1e2c43c9bed` | 115 / 230（→1657） | **legal_death**（3,316） | 1 | 4,878→3,004 | 50,601→59,781 | init/final death，open_loop 到上限 |
| 1023 | `4ea91dbfcf394451` | 126 / 252（→1668） | **legal_death**（3,338） | 1 | 4,878→2,934 | 50,592→60,096 | init death，final/open_loop 到上限 |

- 教师臂**不退化**：16 个不同候选、dominant 0.152–0.183（pilot 三臂是 1.0 的恒定候选）。
- 四根都在窗口第 **115–138** 个决策掉命；而 seed 12 同一条磁带在 02-C 里存活满 600 帧、kill_M1 恰好发生在**最后一个决策 1,842**——**卡内余量为 0**。
- `teacher_completed_roots=[]` → `interpretation="entry_or_task_seed_specific"`；gate 17/17（4 根都有结论、0 技术无效、4/4 settle、静音 0 问题）。

### 13.2 原生 2 帧语料离线复拟合（0 次启动，GPU 33.9 s）

`load_teacher_shard_v3` 对 T04 目录 **fail-closed**（无 `shard.json`、决策帧从 16 起），已记拒绝原文，改用其注册读取器 `touhou_rl.v420.dataset_2f` + `self_check_dataset`（passed、0 violations、hash_pairs 6,592）+ 三个 obs 平面按 T04 meta 重算 sha256 全一致；未合成或改写任何行/标签/像素。

| 组 | 语料 | 角色 | top-1 | dominant share | distinct 类 | 恒定？ |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A 原样（原生 2 帧，6,592 条） | T04 seed10 | 拟合根 | 0.4933 | 0.8961 | 3 | 近恒定 |
| A | T04 seed11 | 独立供体 | 0.5020 | 0.8927 | 3 | 近恒定 |
| B 类别平衡（同 200 步） | T04 seed10 | 拟合根 | 0.4413 | **0.3011** | **17** | 否 |
| B | T04 seed11 | 供体 | 0.3550 | 0.3054 | 16 | 否 |
| C D1 半密度基线 | D1 seed13 | 拟合根 | 0.4918 | 0.9285 | 3 | 近恒定 |
| C | D1 seed14 | 供体 | 0.4904 | 0.9407 | 3 | 近恒定 |

- **C 逐位复现上一轮**：initial/final 各 156/156 张量相同、200/200 loss 相同（文件 sha 差异只来自 provenance 块，已单独证明）。
- 判定 `near_constant_is_data_or_recipe = "optimization_or_readout"`：换原生 2 帧语料只把 dominant 从 0.929 变到 0.896（top-1 差 0.2 pp）→ **语料不是主因**；换类别平衡立刻变成 17 类（dominant 0.301），代价是拟合根 top-1 −5.20 pp、供体 −14.70 pp。
- 数据对齐事实（原样保留）：T04 的 `executed ≠ chosen` 有 2,117/6,592 行；存储 `act_hist` 是"已执行"历史而标签是 teacher_2f 重查询（按 chosen 重建历史 5,118/6,584 行不同）——这是一个**标签/历史节拍错位**，下一轮若要再用 2 帧语料必须先定口径。

### 13.3 首次失误法证（0 次启动，19 个 `decisions.jsonl` 只读）

| 问题 | 实测 |
| --- | --- |
| 首次失败时机 | 12 个 dev 臂中 8 个死亡，入口后中位 **168.5** 决策（范围 65–264）；死亡提交中位 181.5；4 个存活到 300 决策上限 |
| 目标代 life 停滞 vs 失败 | **0/8 先停滞**、8/8 晚 6–12 决策 → 停滞是"玩家 DYING 后射击中断"的**后果**，不是先兆 |
| 三臂差别 | 不在恒定性程度（12/12 臂 share ≥0.963，9/12 = 1.000，饱和），而在**恒定在哪个 cand**：init 全 cand 4（SE）→偏移 33 起钉在右下角；open_loop 全 cand 0（idle）→一步不动；final 以 idle 为主 + 4–10 次 W 抖动 |
| 跨种子证据 | init/open_loop 的逐决策位置与 life 序列在公共前缀内**完全相同**（6/6 对），但首次失败偏移不同 → 何时被命中由 **seed 弹幕场**决定，不是策略路径 |
| 可复核先兆 | 仅两条：①位置冻结先于失败 8/8（提前 65–264 决策，是状态不是计时器）；②DYING→死亡提交 13–14 决策（8/8，在失败之后）。**否证**：life 停滞 0/12、死亡前 30 决策 `0x0800` 聚集 0/8、`0x0400` 饱和无分辨力、`0x0020` 在全部失败臂死亡前窗恒为 0 |
| 成功臂签名 | arm_T：17 个 cand / 217 次切换 / 256 个位置 / 无冻结 / **16.31 life/决策**（正好在 16.26 阈值上）；arm_C：1 个 cand / 0 切换 / 偏移 47 冻结 / 13.83（线性投影需 352.6 > 300）。判别特征 = **动作不塌缩 + DPS 达标**，pilot 的 12 臂两者皆缺 |

### 13.4 合并结论（三段，均指向上游设计）

1. **任务卡本身不跨 seed 转移，且零余量。** 记录磁带在 seed 12 上恰好用满 600 帧（完成在最后一个决策）；换上四个未曝光根，**连教师都 115–138 决策即死**。所以 03 的 0/4 **不是学习信号**，而是"这张卡在那些根上没有可达目标"。要测学习，必须先让任务在 dev 根上可被同约束教师完成。
2. **学习器是第二个独立失败。** 200 步 BC 在拟合根上就是近恒定（offline 0.929/3 类；原生 2 帧也 0.896/3 类）；类别平衡能把它变成 17 类，但 top-1 仍在多数类基线附近（0.4413 vs 基线 0.4407）。在解决"任务可转移"与"策略可表达"之前，闭环比较没有信息量。
3. **失败机制已定位到动作塌缩。** 成功臂唯一同时具备"动作不塌缩"与"DPS ≥ 16.26 life/决策"；失败臂全部塌缩到单一 cand，且首次失败前只有"位置冻结"这一个可观测状态先兆。

### 13.5 对研究端的直接建议（按代价）

| # | 建议 | 代价 | 理由 |
| ---: | --- | --- | --- |
| 1 | **改任务入口**：从自然开局定义任务，或做**状态相关**的入口/教师（记录磁带不是），并在冻结前对每个 dev 根验证教师可完成 | 需新工单/资产 | §13.1：固定的记录磁带 0/4 转移 |
| 2 | 若保留中局卡：**入口留 ≥200 帧余量**（当前 0），且入口应选在敌方生命仍高、玩家未被逼角的位置 | 1 次 recon + 1 次教师×N 根 | §13.1 的四根都是 115–138 决策死 |
| 3 | **先修学习器再测闭环**：类别平衡采样 + 更多更新（或更强读出），并先做离线 top-1/分布门 | 0 次引擎 | §13.2：平衡后 17 类但 top-1 不升 |
| 4 | 复核 T04 的 `executed ≠ chosen`（2,117/6,592）与 `act_hist`/标签节拍错位，再决定能否复用 2 帧语料 | 0 次引擎 | §13.2 数据对齐缺口 |
| 5 | 用 §13.3 的"位置冻结 + DPS 阈值"作为下一轮**在线诊断**指标（不需要画面即可算） | 0 次引擎 | §13.3 的可复核先兆 |

### 13.6 预算与登记（本阶段结束状态）

- 台账：**26 启动 / 110,729 帧 / 1,385.513 s / 12.087 GiB**；OUT 实测 **13,044,491,720 B ≈ 12.15 GiB**；剩余 **2 启动 / 249,271 帧 / ~5.85 GiB**。
- 工具定稿 `15,068 行 / 6520ca93…5bde`；测试 `2,449 行 / cf619414…6acc`（Windows+WSL **132 passed**）。
- `diag-teacher`、复拟合、法证三份证据分别在 `03/diag/teacher_arm/`、`.tmp_v831_followup_probe/03_diag_refit.*`、`.tmp_v831_followup_probe/03_first_failure.*`；后者已用独立脚本复算 19/19 文件 sha（`RESULT PASS`）。
- `ASSET_SYNC` §40 的最终增量登记（工具/测试定稿 sha、`03/diag/` 证据、脚本清单、目录身份、全表校验）在本次收束时一并更新。
- **为什么不用剩余 2 次启动去重跑 recon**：recon 之后必须重跑 `nominate`；即使 nominate 复用已冻结卡，recon 的 `result.json` 也会被新时间戳/新 `tool_revision` 覆盖，而 `verify` 的卡—recon 交叉核对与 §1 的 `tool_revision_recorded` 都会变动。要"干净"地重跑就必须连 02-C（2 次）、03 三臂（12 次）与 13.1（4 次）一起重跑，远超 2 次余额。因此本单保留"记录 gate=false + verify 从原始行重算 0 不符"的处理方式，把 2 次启动留给下一张工单。
