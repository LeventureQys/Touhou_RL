# V829-INFO-01 回执

**DeepSeek 训练端 → 用户转交 → GPT 远端研究端**

| 项 | 值 |
| --- | --- |
| 检查日期/时区 | 2026-11-23，`+08:00`（Asia/Shanghai，UTC+8） |
| 实测窗口 | 2026-11-23 09:32:13 → 09:47（+0800） |
| 主机代号 | `DSH-WIN-01`（Windows 侧）/ `DSH-WSL-01`（WSL2 Ubuntu，同一台机器） |
| 运行根（实测） | `/mnt/d/WorkShop/Touhou_RL`（= Windows 检出经 `/mnt/d` 挂载）；`/root/Touhou_RL` 是 v4.0.0 时代旧检出，**不是本轮运行根** |
| 执行分工 | 主代理拆单与验收；4 个 DSH 子代理并行执行（D1 身份、D2 历史对照、D3 起点、D4 耗雷）。**本文所有加粗数字均由主代理另行独立复算**（见 §8） |
| 当前状态 | **四项全部完成**，均带命令、退出码、SHA。其中**两项是负面结论**：① 登记谱系里未曝光起点可用数 = **0**；② 帧级扣雷**不可回溯**（只能到决策级） |
| 只读合规 | 未启动任何引擎（`pgrep th08-modern` **rc=1**、Windows 侧进程 0）、未采集/训练/评测、未改删移任何既有文件、未 `git add/commit`、未上传。唯一新增：本文件与 `.tmp_v829_info/**`（我方命令与脚本日志） |

> **一句话给研究端**：历史 30 秒对照**逐格复现、无一处偏差**（G1 146/240、T2 141/240，8 个窗×臂格全部 +0）；但 v8.29 想做的「新起点配对」**用现有登记清单开不了工**——不是算时不够（两档 192 局约 10 分钟串行），而是**干净祖先为 0**，且清单缺接管帧/帧数四类字段。

---

## 五节模板（可直接粘贴）

```text
V829-INFO-01 回执    DeepSeek 训练端 → 用户转交 → GPT 远端研究端
检查日期/时区：2026-11-23 +08:00 (Asia/Shanghai)    主机代号：DSH-WIN-01 / DSH-WSL-01
实际运行根：/mnt/d/WorkShop/Touhou_RL（= Windows 检出）；/root/Touhou_RL 为旧检出，非运行根
当前状态：四项全部完成；新起点可用数 = 0（负面）、帧级扣雷不可回溯（负面）

1. 运行身份
   仓库/分支/HEAD：D:/WorkShop/Touhou_RL  v400  539a4e4f9261fff46c21a66746ad18d6bc7b72a1
                   WSL 同一 HEAD（经 /mnt/d 挂载）；/root/Touhou_RL = d55c26a6…（旧，无 touhou_rl/v420）
   Python：Windows 3.12.10 / torch 2.14.1+cu132；WSL /root/v400-venv/bin/python 3.10.12 / torch 2.14.1+cu130
   引擎：/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern  4,836,008 B
         691BFD77…A568EF6（裸 SHA 与登记相符）
   三份权重：.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt  各 39,069 B
         9C6D5D4A…8A483B / 36A51090…9F8E84C / 885DDE70…6E45943（裸 SHA 与登记相符）
   关键依赖（登记表存 LF 归一化 SHA，工作副本是 CRLF，必须 tr -d '\r' 再比）：
         v8r0_urgency.py  117,564 B  LF 4E81F911…EBF0E = 登记 = HEAD blob
         v8o0_ruletight.py 86,894 B  LF A8F699C4…3EAF838C = 登记 = HEAD blob
         v830_bombpol.py  140,832 B  8E40DED8…2B1870（未登记）
         v432_fork_arm.py 123,448 B  33235506…0CEDE（未登记，D3 门恢复入口）
   权重被实际使用的证据：24 个 d900 逐局文件 480 行里 engine_sha256 去重 1 值、
         ckpt_sha256 去重 3 值且与 policy seed 严格 1:1、cfg_sha256 1 值、prefix_frame 单值 8663
         ⇒ 每次运行用了哪份权重/哪个引擎可唯一确定
   T2 版本链：eval_T2_d900_* 由 v1（amend 前的 v8o0_ruletight.py，"the EVAL build"）产生；
         汇总由 ≥v3 计数、发布由 v4（= 现工具）经 --mode publish 落地

2. 历史 30 秒逐窗（G1 v8.26 URGENCY / T2 v8.23 RULETIGHT，每窗每臂 60 局）
   窗     G1 满段    T2 满段   差值
   w11    38/60      40/60     −2
   w12    44/60      39/60     +5
   w17    21/60      26/60     −5
   w18    43/60      36/60     +7
   合计   146/240    141/240   净 +5（毛差 19，两臂策略不同且非配对）
   工单校验脚本 exit 0，无 CHECK 行；8 格全部与期望逐位相符
   24/24 发布副本 SHA == ASSET_SYNC.sha256；24/24 运行副本 == 发布副本（逐字节）
   逐局自洽：480 行 17 条交叉谓词 0 违例；假满段（survived=true 但 first_death_offset 非空）0 条
   缺陷（非致命）：G1 逐局行 arm_label 全为 URG-a1.00（G2 标签），arm/alpha 正确

3. 新开发祖先与阶段（configs/v430/splits.json，118,172 B，BFAB16BA…0522C2）
   独立对局根 350（复合 id 360，去重键 = (source_id, episode 去掉 -iNNN 子窗口)）
   切分：train 328 / dev 20 / confirm 0 / adapt 0 / quarantine 12（合计 360，无跨 split 重复）
   曝光（Document/Update/Alpha/v4.3.0/registries/pretraining_sources.json，登记 9C890133…59CDE0，25 源）：
      used_for_training 17 / used_for_selection 2 / evaluated_only 3 / rejected_not_used 1 /
      executed_instrument_only 1 / rule_based_no_training_data 1
   可用数：30 秒档(≥1800 帧) 0；40 秒档(≥2400 帧) 0；未曝光(未训练且未选参) 0
   重叠：360 id 被引用 995 次，最高 4 次（19 个 id），无只被引用 1 次的 id
   未知：parent_prefix_digest / takeover_frame / verify_anchor / frame_count / bbox 五类字段
        在清单里整类不存在 ⇒ 350/350 谱系与 2400 帧可判定性 = 无法判定
   阶段：task_family 可读（stage1 242 / stage2 16 / stage3 16 / stage4a 17 / stage4b 16 /
        stage5 16 / stage6a 17 / stage6b 20）；道中/Boss/中Boss 二分无字段可判
   接管验证能力：清单内无接管帧，无法据此验证；帧长只散落在两个 registry 索引里
        （human_reference.json 最短 normal Stage1 = 13,258 帧；replay-fixtures.json 108/108 有 calculationFrames）

4. 耗雷例子与证据等级（36 个 jsonl / 720 局，G1 d900 + T2 d900 + G1 d1200）
   字段语义以 tools/rl/v830_bombpol.py::run_episode36 为准：
     bomb_emits = 请求次数（不是扣雷次数）；bombs_remaining_at_emit = 请求前库存；
     bomb_ignored = 逐局累计的未兑现请求；bombs_used = bombs_start − bombs_end（引擎状态差）
   联合恒等式在 720/720 局成立：S−E==U、emits−U==ignored、Δ=1 快照对数==U
   例 G1 d900 w11 s0 ep0：S3 E1 U2 emits5 ign3 dec[0,2,3,4,387] inv[3,2,2,2,2]
   例 T2 d900 w11 s0 ep0：S3 E0 U3 emits4 ign1 dec[0,2,355,595] inv[3,2,2,1]（fdo 1656）
   证据等级（每格 60 局，共 720 局）：
     仅用请求快照 → 精确 0 / 仅区间 720 / 不足 0
     并入 decisions_by_bombs_remaining 逐决策库存轨迹 → 精确 720 / 仅区间 0 / 不足 0
     帧级 → 0/720（日志无任何逐帧字段；bomb_frames / emit_decrement_at 算了但没写盘）
   结论：旧日志能精确回答「第几号决策扣的雷」，不能回答「哪一帧扣的雷」
   带雷死亡（死亡帧库存 > 0）：G1 d900 0/94、T2 d900 1/99、G1 d1200 1/206 = 2/399
     ⇒「省雷」不是死亡主因，后续假设应指向「何时花雷」

5. 历史吞吐与现有可用算时
   每局整局墙钟（spawn → 1800 帧 → QUIT，含前缀重放）：
     G1 池化 n=240 中位 1.9163 s / 均值 1.8689 s；T2 池化 n=240 中位 3.8619 s / 均值 3.9120 s
     G1 d1200 合计 463.839 s / 240 = 1.933 s（= d900 的 1.03×，长档并不线性变慢）
   预算（估算，依据上两行）：48 对 30 秒 = 96 局 ≈ 277 s ≈ 4.6 min 串行；
     两档 96 对 = 192 局 ≈ 563 s ≈ 9.4 min 串行；6 路并发 ≈ 2 min 墙钟
   现有可用：th08-modern 进程 0、load 0.04–0.46、20 vCPU、WSL 23 GiB（21 空闲）、
     RTX 4070 SUPER 12,282 MiB（用 1,440 MiB、0% util）、D: 空闲 1.1 T；WP05 仍暂停
   ⇒ 算时不是瓶颈；瓶颈是干净起点来源为 0

6. 可立即推进 / 尚需最小输入 / 封存用途
   可立即推进（不依赖新起点）：旧窗适配器冒烟（实测成本 < 1 min/配对流，但需启动引擎授权）；
     修 arm_label 缺陷（v8r0_urgency.py:844 默认 "G2"、:1318 只覆盖 arm）；
     新适配器补导帧级扣雷字段；历史带雷死亡诊断（0/94 与 1/99 已是结论）
   尚需最小输入：一份新起点来源工单 + 清单补 4 类字段（takeover_frame / total_frames /
     parent_prefix_digest / verify_anchor）；接管帧定义须改（A1 规则只留 30–120 帧）
   已保留封存用途：sealed read=false、present_on_host=false（5 类路径 2026-11-15 核验缺失）；
     confirm 四段 unused 各 100 seed（61000/61100/61200/61400 起）；
     61300–61399 已 spent 不可复用；60000–60199 永久禁训；108 条 retail 录像未取用
```

---

## 1. 运行身份（本次实测）

### 1.1 两个检出，只有一个是运行根

| 项 | `DSH-WIN-01`（Windows） | `DSH-WSL-01`（WSL2 Ubuntu） |
| --- | --- | --- |
| 仓库根 | `D:/WorkShop/Touhou_RL`（`rev-parse --show-toplevel` rc=0） | `/mnt/d/WorkShop/Touhou_RL`（rc=0，**同一 HEAD，同一工作树**） |
| 分支 / HEAD | `v400` / `539a4e4f9261fff46c21a66746ad18d6bc7b72a1` | 同左（rc=0） |
| git | 2.56.0.windows.2 | 2.34.1 |
| python | `C:\Users\a5935\AppData\Local\Programs\Python\Python312\python.exe` 3.12.10 | `/root/v400-venv/bin/python` → `/usr/bin/python3.10`，3.10.12 |
| torch | 2.14.1+cu132，`cuda True` | 2.14.1+cu130，`cuda True` |
| 备注 | — | `/usr/bin/python3` **无 torch**（`ModuleNotFoundError` rc=1） |

`/root/Touhou_RL` = `d55c26a6c465b26f11e26ba4fc1bdb888e8b130b`（v4.0.0 时代），**无 `touhou_rl/v420`**
⇒ 与 v4.4.0 回执的结论一致：**不要用 `/root/Touhou_RL` 下命令**；引擎二进制在它下面，代码与产物在 `/mnt/d`。

### 1.2 引擎、权重、关键依赖

```text
WSL 内实测（cwd=/mnt/d/WorkShop/Touhou_RL，sha256sum + stat -c %s）
```

| 路径 | 字节 | 实测 SHA-256 | 登记 | 相符 |
| --- | ---: | --- | --- | --- |
| `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` | 4,836,008 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | 行19 同 | ✅ 裸比 |
| `.tmp_v860_bombdag/ckpt/seed0/bombdag.pt` | 39,069 | `9c6d5d4ac7c9a6120ecd33aa9de7a0762c916413cc8f52e0c82f8ac41f8a483b` | 行344 同 | ✅ 裸比 |
| `.tmp_v860_bombdag/ckpt/seed1/bombdag.pt` | 39,069 | `36a51090d56149dc185d86e5b386f89304e5522c3612d9e919a4092e99f8e84c` | 行345 同 | ✅ 裸比 |
| `.tmp_v860_bombdag/ckpt/seed2/bombdag.pt` | 39,069 | `885dde70c57f18bb7f1e3d0351737d239a52ce064aeb47fda025866bd6e45943` | 行346 同 | ✅ 裸比 |
| `tools/rl/v8r0_urgency.py` | 117,564 | 裸 `98b28a8dd7c1f7c5…` / **LF `4e81f9111293148ff0221e5df4618702c4ad4069f4f070a50ffe2a26fe2ebf0e`** | 行1307 = LF 值 | ✅ LF 归一化 |
| `tools/rl/v8o0_ruletight.py` | 86,894 | 裸 `2351be25efd22dba…` / **LF `a8f699c4cc3073f38589e00b2aee2f2017c01d519028a2d2f60319713eaf838c`** | 行1019 = LF 值 | ✅ LF 归一化 |
| `tools/rl/v830_bombpol.py` | 140,832 | `8e40ded82ee0f5457018014afb852791252fa77ba86242880b5947cb942b1870` | **未登记** | — |
| `tools/rl/v800_bomb.py` | 93,930 | `85937621a27fe674dc42ab478b0b0f8837cf1a04795faa993dc7434f22aafff2` | **未登记** | — |
| `tools/rl/v431_arm_a.py` | 15,697 | `96923930fed5d1df6ea398e858aa9fb123ed39a455c3ad7e2706230432445503` | **未登记** | — |
| `tools/rl/v432_fork_arm.py` | 123,448 | `33235506077308549f7657e359381d8c38b1bf45ba4ede0d802448f2dbf0cede` | **未登记** | — |
| `tools/rl/v450_cf_label.py` | 67,077 | `b7ab0ebeee38df4fcbb5bbbdf5e453edc8e5547abb702473b0c09cb4e90d9947` | **未登记** | — |
| `touhou_rl/v420/contracts.py` | 56,321 | `1357c69c32a74fda20553a3e6b290ca2102b7cc5c52e47e9fdc6eb9e08c946c1` | **未登记** | — |
| `touhou_rl/v420/counterfactual.py` | 235,298 | `c1f8cae4c5ff8d36fc7d0ea4a5a908bb6bbe83cbc0218474ab647786d38eb9d8` | **未登记** | — |
| `touhou_rl/v400/gym/actions.py` | 3,813 | `a67145b2bec7c96b3aacaa2514e78cdadf50b056473d15b4ae8122c19c393eff` | **未登记** | — |

**登记表的 SHA 是 LF 归一化值，工作副本是 CRLF——这条必须先知道**（本次实测）：
`ASSET_SYNC.sha256` 193,874 B / 1424 行 / `448f83e9…47503`，**1424 行全是 CRLF**；
`v8r0_urgency.py` 有 2,410 处 `\r\n`、`v8o0_ruletight.py` 有 1,819 处。
把 `\r` 去掉后两个 runner 的 SHA **精确等于登记值，也等于 HEAD blob** ⇒ **无实质代码漂移**。
→ **操作性缺口**：在 WSL 里直接 `sha256sum -c ASSET_SYNC.sha256` 会**全线失败**（路径行尾带 `\r`），
AGENTS.md §8.3.3 的校验命令照抄不可用，须 `tr -d '\r'` 或跳过。见 §7。

### 1.3 权重**确实**被这批 jsonl 使用（可唯一确定，不是推断）

24 个 d900 逐局文件（12 G1 + 12 T2）共 **480 行**，0 行解析失败。四个字段的去重集合：

| 字段 | 去重数 | 取值 |
| --- | ---: | --- |
| `engine_sha256` | **1** | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`（480 行）＝ 实测引擎 |
| `ckpt_sha256` | **3** | `9c6d5d4a…8a483b` / `36a51090…9f8e84c` / `885dde70…6e45943`（各 160 行）＝ 三份 `bombdag.pt` 实测 |
| `cfg_sha256` | **1** | `cc062cfbf9fd2c14a1225376e182beffc52f964c342c187b409045f495d5f0fa`（＝ `fidelity_d3.json` 的 `data.cfg.sha256`，即静音 cfg） |
| `prefix_frame` | **1** | `8663`（＝ D3 门的 PARK 帧） |

`ckpt_sha256` 与 policy seed **严格 1:1**（seed0→`9c6d5d4a`、seed1→`36a51090`、seed2→`885dde70`），
单文件内去重恒为 1，**无串号**。⇒ 回答工单的「起点同源」问题：**能唯一确定，不存在跑 A 记 B**。

### 1.4 D3 保真门身份

| 文件 | 字节 | SHA-256 | 登记 |
| --- | ---: | --- | --- |
| `.tmp_v8r0_urgency/d3_gate.json` | 5,097 | `a2a61fa3f83d802fe56174ecf7d5789ad94c1255edadd8a4964b489a4fb48bc6` | 行1185 有副本，相符 |
| `.tmp_v8o0_ruletight/d3_gate.json` | 5,116 | `7d9105a47b3ed2e299173b9090ed259854dfb905e746d4d5c0d1a7fbf74256ec` | 行960 有副本，相符 |
| `.tmp_v8o0_ruletight/fidelity_d3.json` | 26,688 | `538f85819a662e108e8f31e6987d7039410b22280d845558ccc8dbcbe24a2f3c` | **未登记**，两目录均无发布副本 |

两门均 `passed=5/5`、`all_pass=true`、`exit=0`、`engine.sha256=691bfd77… match=true`；
`fidelity_d3`：seed=12、`hash_checks=2165`、`hash_mismatches=0`、`frames_walked=8663`、
`SDL_AUDIODRIVER=dummy`、R24 `18/18`。门内 `raw_sha256` 与实际文件 **MATCH**（自证一致）。

### 1.5 T2 的版本链：历史实跑 vs 当前版本（分别给身份）

| 角色 | 文件 | 字节 | SHA-256 |
| --- | --- | ---: | --- |
| v1 = **历史实跑**的评测代码 | `runner_v1_before_summary_amend.py` | 84,494 | `e5827cf7b1e353af8ab1e51c74382d99b059907db9c160ca5919b48c1b03dfe6` |
| v2 | `runner_v2.py` | 84,834 | `0df0534bbb90d2b4b4912f2faf42bbccfcd79a2926b9da1d358a432dcefded1e` |
| v3 | `runner_v3.py` | 84,990 | `9432cc5e09272c256996623579670d08e7043bae3a248559fc49bac1554c22db` |
| v4 = **当前工具**（无独立存档） | `tools/rl/v8o0_ruletight.py` | 86,894 | LF `a8f699c4…eaf838c` |
| 汇总 | `new_summary.py` / `patch_summary.py` / `make_tables.py` | 15,376 / 1,109 / 1,545 | `fb32965e…` / `c96edf0e…` / `da260adf…` |
| 发布 | `publish_final.sh` | 1,451 | `a0a52a0f…` |
| 产物 | `summary.json` | 51,361 | `e7d11625542f215af5b90b5c0e156cacf80aacb40d4252eb88d1cd2eba8a7ebd` |
| 证据 | `RUN.md` | 27,877 | `bd3157ded9fdc4389f70d38c58bb642f639964b8fffbc80616434b93c600b69c` |

补丁内容（读 3 个 `.patch`）：
- **v1→v2**：S-rise / S-variance 由「单 best 臂」改为「臂集合」，新增 `_arm_text` 与
  `"S-variance ALSO holds…"` 文本 ⇒ 只改结论措辞，不改评测。
- **v2→v3**：只给 P2 加 `held_by` / `held_by_best_arm` / `failed_by` 字段。
- **v3→v4**：只改 manifest 收文件（progress.json 循环补全臂、加 `extra_numbers.json`），不碰评测与统计。

**结论**：`eval_T2_d900_*` 由 **v1**（= amend 前的评测代码，自称 "the EVAL build"）产生；
汇总/发布由 **≥v3** 的 amend 版计数，最终 **v4**（= 现工具）经 `--mode publish` 落地。
证据：`commands.tsv` 九条全部调 `tools/rl/v8o0_ruletight.py --mode eval`；
`rebuild.sh:6` 先 `cp` 出 v1、`rebuild2.sh:2` 注释 "keep v1 (= the EVAL build) untouched"；
eval 文件 mtime 16:50–16:55 **早于** v1 快照 16:56:09；`log_summary.txt:18` 含只有 v2 才有的多臂文本；
`summary.json` 含只有 v3 才有的 `held_by`/`failed_by`；`rebuild3.sh:15-18` 断言 `mode_eval` 在 v1→final 完全相同。
唯一不能细分的是 `summary.json` 属 v3 还是 v4（同一分钟），字段证据的**下界是 v3**，两者汇总逻辑相同。

---

## 2. 历史 30 秒对照复核（任务 2）

### 2.1 工单脚本：**原样执行、未改断言、exit 0**

```powershell
python -B .tmp_v829_info\d2\verify_d900.py        # cwd = D:\WorkShop\Touhou_RL
# exit 0（连跑两次均 0）
```

原始 stdout（逐字，无 `CHECK` 行）：

```text
G1 11 full 38 verified_rows 60 expected 38 60
G1 12 full 44 verified_rows 60 expected 44 60
G1 17 full 21 verified_rows 60 expected 21 60
G1 18 full 43 verified_rows 60 expected 43 60
T2 11 full 40 verified_rows 60 expected 40 60
T2 12 full 39 verified_rows 60 expected 39 60
T2 17 full 26 verified_rows 60 expected 26 60
T2 18 full 36 verified_rows 60 expected 36 60
```

⇒ **8 格全部与期望逐位相符**，脚本的三层断言（登记 SHA、局号完整、身份与存活一致）全部通过。

### 2.2 逐窗表

| 臂 | 窗 | 有效局 | 满段 | 坏行 | 重复 `episode` | 行数≠20 | 期望 | 差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | 11 | 60 | **38** | 0 | 0 | 0 | 38 | +0 |
| G1 | 12 | 60 | **44** | 0 | 0 | 0 | 44 | +0 |
| G1 | 17 | 60 | **21** | 0 | 0 | 0 | 21 | +0 |
| G1 | 18 | 60 | **43** | 0 | 0 | 0 | 43 | +0 |
| T2 | 11 | 60 | **40** | 0 | 0 | 0 | 40 | +0 |
| T2 | 12 | 60 | **39** | 0 | 0 | 0 | 39 | +0 |
| T2 | 17 | 60 | **26** | 0 | 0 | 0 | 26 | +0 |
| T2 | 18 | 60 | **36** | 0 | 0 | 0 | 36 | +0 |

池化 **G1 146/240 = 60.83 %**、**T2 141/240 = 58.75 %**、合计 287/480；逐窗差 −2 / +5 / −5 / +7。

**文件身份**：24/24 发布副本 SHA == `ASSET_SYNC.sha256` 登记值；24/24 运行副本 == 发布副本（SHA + 字节）。
⇒ 工单担心的「两份同名副本」问题**不存在**，两份逐字节相同，只计一次即可。

### 2.3 逐局墙钟（`wall_s`）

口径已查源码：`tools/rl/v830_bombpol.py:754` 起 `started`（spawn 之前）、`:899` finally（`quit_child` 之后）
赋值 ⇒ **整局墙钟**（含 spawn、1800 帧、QUIT、前缀重放）。`spawn_ms` 仅 1–2 ms。

| 臂 | 窗 | n | 中位 (s) | P90 (s) |
| --- | ---: | ---: | ---: | ---: |
| G1 | 11 | 60 | 2.0198 | 2.19 |
| G1 | 12 | 60 | 2.0941 | 2.23 |
| G1 | 17 | 60 | 1.8706 | 2.13 |
| G1 | 18 | 60 | 1.6181 | 1.74 |
| T2 | 11 | 60 | 3.1258 | 3.86 |
| T2 | 12 | 60 | 4.4444 | 5.36 |
| T2 | 17 | 60 | 4.2172 | 5.17 |
| T2 | 18 | 60 | 4.0213 | 4.73 |

池化：**G1 n=240 中位 1.9163 / 均值 1.8689 / P90 2.17**；**T2 n=240 中位 3.8619 / 均值 3.9120 / P90 5.02**。
（P90 因插值方法在末位有 ±0.01 s 差异，量级一致。）
T2/G1 = 2.09×，但两臂每局帧数/决策数**完全相同**（1800/900、2 帧/决策）⇒ 是**会话级吞吐差**，
不能读成「T2 的策略更贵」。

### 2.4 前缀与恢复证据

| 项 | 实测 |
| --- | --- |
| `prefix_frame`（480 行） | 单值 **8663**，四窗 × 两臂一致 |
| 逐局行里的 restore / hash 字段 | **无**。480 行都没有 `restore` / `hash_checks` / `hash_mismatches` / `prefix_hash` |
| 窗口级 gate 侧车（`*_progress.json`） | 4 窗 × 3 臂 × 3 seed 共 24 个 cell，每个 `gate_prefix = {hash_checks: 2165, hash_mismatches: 0, final_frame: 8663}` ⇒ **51,960 次前缀哈希比对、0 失配** |
| `park_frame` | 全 8663，`all_pass` 全 true |
| 四窗同源判定 | **成立**：每窗在私有 data 目录独立重放冻结前缀录像 2165/2165 通过，且 480 次 spawn 后 `frame==8663` 断言全通过 |
| 逐局粒度 | 逐局行的 `prefix_frame` 是**源码常量**而非运行时回读 ⇒ 逐局没有哈希证据，只有窗口级 |

### 2.5 自洽性（480 行）

17 条交叉谓词 **0 违例**：`survived ↔ (fdo null & terminal null & frames 1800 & decisions_executed 900)`、
`survived ↔ first_death_frame null`、活局 `frames==alive_frames`、死局 `frames==alive+1`、
`fdo==alive==frames−1`、`first_death_frame==prefix_frame+frames`、`return_decisions==decisions_executed`、
`decisions_measured∈{de,de+1}`、`samples==measured`、雷数记账、`bomb_emits==len(bomb_decisions)`、
`bombs_start==3`、`checks` 无 false、`invalid`/`error` 全 0。

**「假满段」通道（工单特别关心）：G1 = 0 条，T2 = 0 条**；反向（`survived=false` 但 `fdo` 为空）= 0；
另 7 条变体通道（frames / alive / de / terminal / first_death_frame / frames_consistent）全 0。
193 条非满段行的 `terminal` 全为 `first_death`，`fdo` 覆盖 1089…1793。

### 2.6 与登记的差异（非致命，逐条报告）

1. **`arm_label` 缺陷（真实缺陷，建议修）**：G1 的 12 个 d900 文件里**每一行**的 `arm_label` 都是
   `URG-a1.00`（G2 的标签），而行内 `arm="G1"`、`alpha=0.5`、侧车 `arm_label="URG-a0.50"` 都正确。
   根因：`tools/rl/v8r0_urgency.py:844` 默认写 `"G2"`、`:1318` 只覆盖 `arm` 不覆盖 `arm_label`。
   **按 `arm`/`alpha` 聚合不受影响**（本轮所有计数用的是 `arm`），但按 `arm_label` 过滤会张冠李戴。
2. **「只差 5 局」是净值不是优势**：逐窗 −2 / +5 / −5 / +7，毛差 19，净 +5；两臂策略不同、**非配对**。
3. **`wall_s` 2.09× 是会话级差异**，不是策略成本比（见 §2.3）。
4. **`checks` 键集不对称**：G1 8 键、T2 5 键（T2 缺 `candidate_sampled` / `formula_reproduced` /
   `thresholds_in_bounds`）；两边显式 false 均为 0。

---

## 3. 新开发起点盘点（任务 3）

### 3.1 清单身份

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `configs/v430/splits.json` | 118,172 | `bfab16ba7b68ff4f8dc33a6dd06bbe0e69c1771293804ad1eb432f1a960522c2` |
| `configs/v430/assets.json` | 28,917 | `5bae116baf2591ec2a3e573be3060abb9494c619e9556ce0aae4d1ee7f71b383` |
| `configs/v430/contracts.json` | 10,199 | `1863c1cf8bc01670cdb9dc5b308e7f4f2a2e2c0898285e9e8bf7a753c17f80dd` |
| `Document/Update/Alpha/v4.3.0/registries/pretraining_sources.json`（splits 登记为 registry） | 68,945 | `9c890133fa307a49b4d015ebefd1beb0c2e977c0b3c3bc3394866178b559cde0` |

`schema=v430-splits/1` · `spec_revision=v430-plan-r2` · `frozen_at=2026-11-15T11:34:26+08:00` ·
`timezone=Asia/Shanghai` · `experiment_id=v430-w09-clean-holdout-splits-001`。
条目：`roots` 12 · `registries` 4 · `ancestors` 360 · `splits` 5 · `confirm_batches` 5 · `sealed` 1(块)。
`configs/v430/` 三文件 `git diff` 为空、SHA 与 `frozen_at` 时点一致（本轮未改动）。

### 3.2 独立对局根与曝光（回答「独立根总数 / 重复 / 未知」）

| 指标 | 值 | 依据 |
| --- | ---: | --- |
| `ancestors` 条目（复合 id） | **360** | 每条的键只有 `source_kind` / `source_id` / `episode` / `seed` / `task_family` / `note` |
| **独立对局根** | **350** | 去重键 = `(source_id, episode 去掉 -iNNN 子窗口)`；7 个 `d1-teacher-fast` 根带 17 个 `-iNNN` 复合 id |
| 切分 | train **328** / dev **20** / confirm **0** / adapt **0** / quarantine **12** | `splits.*.ancestor_ids`，合计 360，无跨 split 重复 |
| 曝光来源 | 25 个 pretraining source | `pretraining_sources.json`：`used_for_training` 17、`evaluated_only` 3、`used_for_selection` 2、`rejected_not_used` 1、`rule_based_no_training_data` 1、`executed_instrument_only` 1 |
| 重复引用 | 360 id 被引用 **995** 次，最高 4 次（19 个 id） | **没有只被引用 1 次的 id** |
| **未曝光（未训练且未选参）** | **0** | train 328 + dev 20 + quarantine 12（且 quarantine 的 seed 1000–1011 落在 dev 面板 1000–1019 内 = seed 级选参曝光） |

### 3.3 「剩余 2,400 帧」为什么**无法判定**（这是本项的核心负面结论）

判定规则：`remaining = L − t`，`≥2400 ⟺ t ≤ L − 2400`；60 Hz 出处 `configs/v430/contracts.json`
→ `constants.time.reference_hz = 60`（故 2400 帧 = 40.0 s、1800 帧 = 30.0 s）。

**清单里 `L` 与 `t` 都没有**（递归键名 + 关键词双查，`takeover` / `prefix_digest` / `frame_count` /
`total_frames` / `boss` / `midboss` / `fps` 命中全 0）⇒ **360/360 逐条「无法判定」**，
这是**没有输入**，不是「不足」。

- `L` 只散落在两个 registry 索引里：`human_reference.json` → `sets[].stages[].gameplay_frames`
  （12 个组合，**最短 normal Stage1 = 13,258 帧** ⇒ 接管帧 ≤ 10,858 仍可留 2400）；
  `replay-fixtures.json` → `fixtures[].retailTrace.calculationFrames`（108/108 有值；**顶层
  `calculationFrames` 全为 null，只读顶层会误判**）。
- `t` 只以旧规则形式存在：`configs/v420/a1-safe-set-full.json` 的
  `state_frame_formula = floor_to_2(first_death_frame − offset)`、`offsets=[30,60,90,120]`、
  `horizon_frames=120`。

### 3.4 阶段分布（**对子代理初报的一处修正**）

清单没有 `stage` / `boss` / `midboss` 独立字段，但 `task_family` 的名字就带阶段号：

| 阶段 | stage1 | stage2 | stage3 | stage4a | stage4b | stage5 | stage6a | stage6b | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 条目 | 242 | 16 | 16 | 17 | 16 | 16 | 17 | 20 | 360 |

⇒ **阶段可读**（`th08-stage4a-normal-team0` 之类），但**「道中 / Boss / 中Boss」二分没有字段可判**。
子代理初报写「道中 0 / Boss 0」，主代理复算后修正为：**可给阶段，不可给道中-Boss 二分**。
预注册要求「道中 4 + Boss 4」，这一条**当前无法从清单满足**。

### 3.5 可用数与结论

| 档位 | 可用独立根 | 理由 |
| --- | ---: | --- |
| 30 秒（≥1800 帧） | **0** | 328 训练 + 20 选参 + 12 隔离且 seed 污染 + 干净 0；且无帧数字段 |
| 40 秒（≥2400 帧） | **0** | 同上，另缺接管帧 |
| 缺口 | **= 研究端所需 N（全部）** | 阶段长度不是瓶颈（最短阶段仍余 10,858 帧），瓶颈是**干净根为 0** |

**更早会踩的坑**：即使补到全新干净 seed，若沿用 A1 的「首死前 30–120 帧」接管规则，
接管后只剩 **30–120 帧**，与 2,400 差 **20–80 倍** ⇒ **必须先改接管帧定义，再谈补 seed**。

---

## 4. 旧日志能回答哪种耗雷问题（任务 4）

### 4.1 字段语义（以代码为准，`tools/rl/v830_bombpol.py::run_episode36`，`DECISION_FRAMES=2`）

| 键 | 语义裁决 | 代码 |
| --- | --- | --- |
| `bombs_start` | 开局基线库存（决策 0 之前），逐局；720/720 恒 3 | `:781-784` |
| `bomb_emits` | **请求次数**（雷位=1 的决策数），**不是扣雷次数** | `:799`, `:828-831` |
| `bomb_decisions` | 请求的 0-based 决策号（升序） | `:831` |
| `bombs_remaining_at_emit` | **请求前**库存（该决策第 0 帧执行之前） | `:811` + `:835` |
| `bomb_ignored` | **逐局累计**的「请求未兑现」数（2 帧内末次读数未低于请求前库存） | `:767`, `:871-875`（判据 `:860-861`） |
| `bombs_end` | 最后读数；死于第 d 决策时 = **死亡帧库存** | `:890` |
| `bombs_used` | `bombs_start − bombs_end`（引擎状态差），**不是** `emits−ignored` 的别名 | `:891` |
| `decisions_by_bombs_remaining` | **逐决策「请求前库存」计数直方图**（和 == `decisions_measured`） | G1 `v8r0:730-732`；T2 `v8o0:629-631` |
| `bomb_frames` / `bomb_available_at_emit` / `emit_decrement_at` | **算了但没导出**（行构造器未拷贝） | `:832`, `:873`, `:846/862-863` |
| 雷位 | **只有发雷决策的第 0 帧带雷位** | `action_masks:504-523`，循环 `:848-850` |

### 4.2 两段实例 + 交叉核对（720/720 局全部闭合）

**G1 d900 w11 s0 ep0**：`S3 / E1 / U2 / emits5 / ign3`；`decisions [0,2,3,4,387]`；
`inventory_at_emit [3,2,2,2,2]`；`decisions_by_bombs_remaining {"3":1,"2":387,"1":512,"0":0}`；
`frames 1800`、`decisions 900`、存活。
**T2 d900 w11 s0 ep0**：`S3 / E0 / U3 / emits4 / ign1`；`decisions [0,2,355,595]`；
`inventory_at_emit [3,2,2,1]`；`hist {"3":1,"2":355,"1":240,"0":233}`；`frames 1657`、
`decisions 829`、`first_death_offset 1656`、`terminal first_death`。

| 核对项 | G1 | T2 | 全库 720 局 |
| --- | --- | --- | ---: |
| `S − E == U` | 3−1=2 ✔ | 3−0=3 ✔ | **720/720** |
| `emits − U == 被忽略请求数` | 5−2=3 ✔ | 4−3=1 ✔ | **720/720** |
| `bomb_ignored` 与差额一致 | 3 ✔ | 1 ✔ | **720/720** |
| Δ=1 快照对数 == U | 2 ✔ | 3 ✔ | **720/720** |
| 直方图下降沿 == Δ=1 请求决策序列 | `[0,387]` ✔ | `[0,355,595]` ✔ | **720/720 逐条相同** |

### 4.3 证据分级（每格 60 局，共 720 局）

| 臂 / 档 | 精确（决策级） | 仅能界定区间 | 不足 | 退回「纯请求快照」：精确 / 区间 | 帧级 |
| --- | ---: | ---: | ---: | --- | ---: |
| G1 d900 w11 / w12 / w17 / w18 | 60/60 各 | 0 | 0 | 0 / 60 各 | 0 |
| T2 d900 w11 / w12 / w17 / w18 | 60/60 各 | 0 | 0 | 0 / 60 各 | 0 |
| G1 d1200 w11 / w12 / w17 / w18 | 60/60 各 | 0 | 0 | 0 / 60 各 | 0 |
| **合计** | **720** | 0 | 0 | **0 / 720** | **0** |

- **判定 B（并入 `decisions_by_bombs_remaining` 逐决策库存轨迹）**：720/720 精确到**决策**，
  轨迹下降沿与快照 Δ=1 序列逐条相同。前提：**库存单调不增**（2,147 个下降沿 + 720 首尾全一致，无任何反例）。
- **判定 A（只用请求快照）**：720/720 只能给区间。根因不是随机：**没有一局「最后请求 == 最后一号决策」**
  （0/720），尾段必然宽（11–787 决策，中位 248）；2,147 个已兑现扣雷里只有 417 个（19.4%）落在宽度 1 的区间。
- **帧级 0/720**：日志没有任何逐帧字段。只能推：按下帧 = `2·d_i`（绝对帧 `8663+2d`），
  计数器下降只能限定在 `[2d+1, 2d+2]` 两帧窗口——**不给唯一帧**。

⇒ **旧日志能精确回答「第几号决策扣的雷」，不能回答「哪一帧扣的雷」。**

### 4.4 G1 d1200 的 12 个单元格（每窗 3 seeds × 20 局 = 60）

| 窗 | 满段 | Σemits | Σused | Σstart | Σend | Σignored | 首请求中位 | 库存归零局 | 首扣雷可判定性 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| w11 | 14/60 | 494 | 180 | 180 | 0 | 314 | 0 | 60 | 精确 60/60（纯快照 53/60） |
| w12 | 11/60 | 452 | 180 | 180 | 0 | 272 | 0 | 60 | 精确 60/60（纯快照 60/60） |
| w17 | 4/60 | 250 | 180 | 180 | 0 | 70 | 0 | 60 | 精确 60/60（纯快照 **0/60**） |
| w18 | 5/60 | 542 | 179 | 180 | 1 | 363 | 0 | 59 | 精确 60/60（纯快照 60/60） |
| **合计** | **34/240** | **1738** | **719** | **720** | **1** | **1019** | 0 | **239** | 精确 240/240 |

**退化警告**：240/240 局的 `first_bomb_decision` 全为 0，首次扣雷决策全 = 决策 0（`hist["3"]==1`）
⇒ **该量无区分度**。真正有区分度的是忽略率（12 格 **21.2 %–67.0 %**）。

### 4.5 带雷死亡：一个直接可用的结论

死亡帧库存 > 0 的局：**G1 d900 0/94、T2 d900 1/99、G1 d1200 1/206 = 2/399**。
⇒ 本批数据里**「省雷 / 带雷死亡」不是死亡主因**；后续假设应指向**「何时花雷」**
（G1 首雷在决策 0、忽略率 21–67 %），而不是「是否花雷」。这一条不需要新引擎运行即可支撑 §04 的阶段判断。

### 4.6 可独立继续的诊断

| # | 要回答什么 | 需要什么 | 现有日志够不够 |
| ---: | --- | --- | --- |
| 1 | 带雷死亡 | 死亡帧 + 当时库存 | **够**（`first_death_offset` + `bombs_end`），结论见 §4.5 |
| 2 | 扣雷帧 / 请求→扣雷延迟 | 逐帧或 `bomb_decrement_frame` | **不够**，不可回溯 |
| 3 | 逐请求是否被忽略 | 聚合 `bomb_ignored` + 轨迹 | **够（有前提）**，720/720 与聚合一致 |
| 4 | 最后请求距死亡多少帧 | `last_bomb_decision`+`fdo`+`frames=2·de` | **够（推导值，非字段）** |
| 5 | 同状态「按雷 / 不按雷」反事实 | 同状态成对 rollout | **不够**，必须跑引擎（本单禁止） |

---

## 5. 历史吞吐与现有可用算时

### 5.1 实测墙钟（见 §2.3）

| 对象 | 每局中位 | 每局均值 | 备注 |
| --- | ---: | ---: | --- |
| G1 d900（1,800 帧 / 900 决策） | 1.9163 s | 1.8689 s | n=240，含 spawn + 前缀重放 + QUIT |
| T2 d900（同帧数） | 3.8619 s | 3.9120 s | n=240，会话级慢 2.09× |
| G1 d600 / d1200 | 1.388 s / **1.933 s** | — | `wall_s_total` 333.059 / 463.839 除 240 |

### 5.2 试点预算（**估算**，依据上表，非实测）

| 项 | 算式 | 结果 |
| --- | --- | --- |
| 48 对（30 秒，96 局） | 48 × (1.87 + 3.91) | ≈ **277 s ≈ 4.6 min**（串行） |
| 两档 96 对（192 局） | 277 + 48 × (1.93 + 4.03) | ≈ **563 s ≈ 9.4 min**（串行） |
| 6 路并发 | 563 / 6 + 批间开销 | ≈ **2 min 墙钟** |
| 参照：本轮 27 进程实测 | 5 批 6/6/6/6/3 | 845 s 墙钟（17:40:52 → 17:54:57） |

⇒ 与工单「上限 403,200 受控游戏帧」同量级：按实测 ~700 fps 混合吞吐约 **10 分钟串行**。
**算时不是瓶颈**。

### 5.3 现有可用算时（本次实测）

| 项 | 值 |
| --- | --- |
| `th08-modern` 进程 | **0**（WSL `pgrep` rc=1、Windows 计数 0） |
| load | 0.04 / 0.46（两次采样） |
| CPU | 20 vCPU |
| 内存 | WSL 23 GiB（21 空闲）；Windows 31.8 GB（17 空闲） |
| GPU | RTX 4070 SUPER 12,282 MiB，已用 1,440 MiB（他方）、**利用率 0 %** |
| 磁盘 | `D:` 1.9 T 容量 / 空闲 **1.1 T** |
| 并发写入 | 无：`Document/**` 20 分钟内只有研究端 09:31:42 写入的 v8.29.0 三份文件 |
| WP05 状态 | **仍暂停**（本轮未恢复、未采集、未训练） |

### 5.4 音频（AGENTS.md §9）

本轮**未启动任何引擎**，§9.3 三项动作无触发机会。
历史两轮的自证审计均为全静音（本次读取复核）：`.tmp_v8r0_urgency` **109 cfg / 109 muted / 0 audible**、
`.tmp_v8o0_ruletight` **37 / 37 / 0**，`all_pass=true` ⇒ **未新增 audible 计数**。

---

## 6. 可立即推进项 / 尚需补充的最小输入 / 已保留的封存用途

### 6.1 可立即推进（不依赖新起点、不需要新授权以外的东西）

1. **旧窗适配器冒烟**（预注册 §3.1–§3.3）：用现有 4 个旧窗的冻结前缀，验证新 v8.29 入口的
   恢复 / 计分 / 采样 / 放雷事件；实测成本 < 1 min/配对流。**需要「启动引擎」的授权**——
   本单明确排除，请研究端在评测工单里写明。
2. **修 `arm_label` 缺陷**：`v8r0_urgency.py:844` 默认 `"G2"`、`:1318` 只覆盖 `arm`
   ⇒ 新证据不要再出现「G1 行标 `URG-a1.00`」。
3. **新适配器必须导出帧级扣雷字段**：`bomb_frames` / `bomb_available_at_emit` / `emit_decrement_at`
   现在**算了但没写盘**。预注册 §3.4 要求「请求 / 实际扣雷事件」，若不补导，下一轮仍然只能到决策级。
4. **历史诊断已可收口**：带雷死亡 2/399 ⇒ 把假设从「省雷」改成「何时花雷」，
   并利用忽略率 21–67 % 这个有区分度的量（首请求/首扣雷恒为决策 0，无区分度）。

### 6.2 尚需补充的最小输入

1. **一份「新起点来源」工单**。两条路（都需用户授权，WP05 仍暂停）：
   - **(a) 新 seed 采集**：每局实测 1.9–3.9 s + 前缀重放，成本低；但需要新 seed 段与采集授权。
   - **(b) 动用未花费 confirm 段**：`61000-61099` / `61100-61199` / `61200-61299` / `61400-61499`
     各 100 seed、`episodes=0`、`status=unused`；**用即消耗**。`61300-61399` 已 `spent`（100 集），不可复用。
   - 已占用、需回避的 seed 段（本轮从 registry 读出）：`1-25`、`1000-1019`、`2000-2010`、`2020-2054`、
     `3000-3004`、`3010-3029`、`3042-3091`；另 `60000-60199` 永久禁训、`1020-1039`/`1040-1099` 保留。
2. **清单补 4 类字段**：`takeover_frame`（或 `decision_frame`）、`total_frames`（或 `stage_frames`）、
   `parent_prefix_digest`、`verify_anchor`；可选 `inventory`（bbox / 命 / 雷）。
   否则「剩余 ≥ 2,400 帧」永远停在「无法判定」。
3. **改接管帧定义**：A1 的 `floor_to_2(first_death_frame − {30,60,90,120})` + `horizon_frames=120`
   只留 30–120 帧，与 2,400 差 20–80 倍。
4. **道中/Boss 二分的判据**：预注册要 4+4，清单无此字段；需研究端给判据或允许按 `task_family` 阶段近似。

### 6.3 已保留的封存用途（本轮未触碰）

| 项 | 状态（实测自 `splits.json`） |
| --- | --- |
| `sealed` | `read_sealed_results=false`、`present_on_host=false`；5 类路径模式（`sealed/`、`.tmp_v430_sealed`、`.tmp_v430_confirm`、`evidence/W12/SEALED`、`SEALED_RESULTS`）于 2026-11-15 核验**不存在**；`policy` 禁止任何 sealed 路径出现在配置里 |
| confirm 段 | 4 段 `unused`（61000 / 61100 / 61200 / 61400 起）保持原用途；`61300-61399` `spent` |
| 永久禁训 | `60000-60199` |
| retail 录像 | 108 条 `instrument_only`、`fits_any_model=false`，**本轮未取用**；`holdout_families.contamination_if_used` 规定一旦拟合/取用，G410-6 的两个 frozen family（`th08-stage4a-hard-team0`、`th08-stage6a-lunatic-team0`）全部失效 |

---

## 7. 缺项、差异与登记缺口（只报告、未改任何文件）

| # | 项 | 事实（实测） | 影响 |
| ---: | --- | --- | --- |
| 1 | **工作树大量缺失** | `D=3493`（两侧一致）：`Document/Dialogs/**` 等整目录 ENOENT，而 HEAD 跟踪它们 | **既有状态**，不是我方造成；`git diff --stat` 显示 3515 文件 / 8.24 M 删除。本单未清理、未提交 |
| 2 | **`M` 计数是客户端相关量** | Windows `D=3493 / ??=1585 / M=22`；WSL `M=374`。集合比 22⊂374，多出的 **352 个全是纯 CRLF**（`CR==CRLF`），非内容改动。根因：Windows `core.autocrlf=true`、WSL 未设 | 脏文件数**不能引用单一客户端数字** |
| 3 | **`sha256sum -c` 照抄不可用** | `ASSET_SYNC.sha256` 1424 行全 CRLF，存的是 LF 归一化 SHA ⇒ WSL 里直接 `-c` 全线失败（路径尾带 `\r`） | **AGENTS.md §8.3.3 的操作性缺口**；须 `tr -d '\r'` 后比 |
| 4 | **8 项未登记** | `v830_bombpol.py`（实际跑 episode 循环）、`v432_fork_arm.py`（实际跑 D3 门）、`v800_bomb.py`、`v431_arm_a.py`、`v450_cf_label.py`、`v420/contracts.py`、`v420/counterfactual.py`、`v400/gym/actions.py` | 承重执行入口不在资产册；审计链上有口子 |
| 5 | `fidelity_d3.json` 未登记 | 2 轮各 1 份，26,688 B，无发布副本；只由 `d3_gate.raw_sha256` 间接绑定（该字段实测相符） | 间接绑定成立，但资产册缺 2 行 |
| 6 | `runner_v4.py` 无存档 | `ls` rc=2；v4 只能由现工具 + HEAD blob 认定 | 版本链末端靠推断（字段证据下界 v3） |
| 7 | `ASSET_SYNC.md §1` HEAD 过期 | 仍写 `d825f84`，实际 `539a4e4` | 文档漂移，不影响资产 SHA |
| 8 | `arm_label` 缺陷 | 见 §2.6 | 不影响按 `arm` 聚合；按 label 过滤会错 |

**本轮未能给出/未能核实的项**：道中/Boss 二分（无字段）；每个提名根的接管帧与校验锚点（无字段）；
「剩余 ≥2400 帧」逐条结论（无输入）；`summary.json` 究竟属 v3 还是 v4（同分钟，不可细分）。
**没有一项是我方放弃检查的**；上述均为「清单/代码里没有该数据」。

---

## 8. 本次实测命令与退出码（主代理复算在最后）

```text
# DSH-WIN-01 / DSH-WSL-01, 2026-11-23 09:32–09:47 (+0800)
git rev-parse --show-toplevel|HEAD|--abbrev-ref HEAD      -> D:/WorkShop/Touhou_RL / 539a4e4… / v400        rc=0
git status --porcelain=v1 | Measure-Object -Line          -> 5100 (D 3493 / ?? 1585 / M 22)                  rc=0
WSL: 同上（cd /mnt/d/WorkShop/Touhou_RL）                  -> +5452 行 (M 374，352 个纯 CRLF)                 rc=0
/root/v400-venv/bin/python -c "import torch;…"            -> 3.10.12 / 2.14.1+cu130 / cuda True              rc=0
/usr/bin/python3 -c "import torch"                        -> ModuleNotFoundError                             rc=1
wsl: sha256sum /root/Touhou_RL/…/rl_v3/th08-modern …      -> 691bfd77… / 9c6d5d4a… / 36a51090… / 885dde70…   rc=0
python -B .tmp_v829_info\d2\verify_d900.py                -> 8 行 full/rows 全等期望、无 CHECK                rc=0
python -B （24 文件 SHA + 副本比对）                       -> 24/24 == 登记；24/24 运行==发布                  rc=0
python -B （480 行自洽 17 谓词）                           -> 0 违例；假满段 G1 0 / T2 0                       rc=0
python -B （720 行雷数恒等式）                             -> S−E==U、emits−U==ign 全通                        rc=0
python -B （splits.json 结构/去重/曝光）                    -> 360 id / 350 根 / train 328 dev 20 quar 12       rc=0
python -B （五类字段递归+关键词双查）                        -> takeover/prefix_digest/frame_count 命中 0        rc=0
python -B （d1200 12 格聚合）                              -> 34/240、Σemits 1738、Σused 719、ΣE 1、Σign 1019  rc=0
python -B （带雷死亡）                                     -> 2/399（0/94、1/99、1/206）                       rc=0
pgrep -af th08-modern / Get-Process th08*                 -> 0 个引擎                                          rc=1 / 0
nvidia-smi / free -g / df -h /mnt/d / /proc/loadavg       -> 0% util / 23 GiB / 1.1 T / 0.04                  rc=0
```

**主代理独立复算**（不照录子代理结论）：splits.json 切分 328/20/0/0/12、360 id、五类字段缺失；
`pretraining_sources.json` 25 源的状态分布与登记 SHA；G1 w11 s0 的 20 行/13 满段/`arm_label` 全为
`URG-a1.00`、发布==运行副本；720 行雷数恒等式 0 破例、带雷死亡 2/399；d1200 的 1738/719/720/1/1019；
两个 runner 的 CRLF/LF 双 SHA（LF 值 == 登记 == HEAD blob）；`gate_prefix` 2165/0/8663；
wall_s 中位数与 P90；GPU/RAM/磁盘/load。**以上与 4 份子代理报告逐项一致，未发现矛盾。**

**引擎启动计数：0**（WSL `pgrep` rc=1、Windows 计数 0）；**音频触发：0**；**git 写操作：0**。

---

## 9. 勘误（2026-11-23 11:15 +0800，主代理自查）

本轮做 v8.29 实施尽调时，对本文三处表述做了独立复核，**两处引用不准、一处归因不准**，逐条更正。原结论（干净起点 0、帧级不可回溯、历史对照全对）**均不受影响**。

| # | 原文位置 | 原文 | 更正 |
| ---: | --- | --- | --- |
| 1 | §3.3 | 「`L` 只散落在两个 registry 索引里：`human_reference.json` → …；`replay-fixtures.json` → …」 | **两个文件都不在 `Document/Update/Alpha/v4.3.0/registries/`**（该目录只有 5 个文件：`code_identity` / `holdout_families` / `pretraining_sources` / `runs` / `task_families`）。实际位置：`Document/Update/Alpha/v4.0.0/bench/human_reference.json`（8,691 B）与 `third_party/th08/config/replay-fixtures.json`（186,021 B，该目录是 git submodule）。数值本身无误：12 个组合帧数 13,258–36,403；108 条 `calculationFrames` 44,106–149,986、108/108 ≥ 2400 |
| 2 | §3.3 | 「**顶层也有同名字段且全为 null**，不要读错层级」 | **不成立**。`replay-fixtures.json` 顶层键只有 `version` / `targetSha256` / `fixtures`，**根本没有 `calculationFrames` 键**；全文 `"calculationFrames": null` 命中 **0**，108 条全在 `fixtures[].retailTrace.calculationFrames` 且全为非 null 整数。层级混淆的真实风险在 `v4.1.0/evidence/WP01/g0.1-108.json`（那份把它放在 `fixtures[].calculationFrames`），不在本文件 |
| 3 | §4.1 / §4.3 / §4.6 | 「`bomb_frames` / `bomb_available_at_emit` / `emit_decrement_at` **算了但没导出**」 | **归因不准**（结论不变）。`run_episode36` **已经导出** `bomb_frames`（`v830_bombpol.py:832`）与 `bomb_available_at_emit`（`:873`），是 **v8 两个 runner 的投影处丢的**（`v8r0_urgency.py:841-900`、`v8o0_ruletight.py:694-741`；对这两名 grep 命中 0）。**只有 `emit_decrement_at`（`:846`、`:862-863`）是真·从未写进 record**。⇒ 修复成本从「改决策循环」降到「投影处两行」；§4.3 的「帧级 0/720」与 §4.6 的「不可回溯」结论**不变** |
| 4 | §3.3 / §6.2 | 未写：索引侧到底有没有 `total_frames` | **补充**：`configs/v430/splits.json`（118,172 B）里字符串 `frame` 出现 **0 次**、`takeover` **0 次**；`configs/` 全部 JSON 对 `takeover_frame` / `total_frames` / `parent_prefix_digest` / `verify_anchor` **0 命中**。本机唯一带「每祖先帧跨度」的是**资产** `.tmp_v400_d1/d1/shards/<episode>/shard.json` 的 `stage_frames`，覆盖 **137/360**（38.1%）已登记祖先，且这 137 个全是 `first_decision_frame = 0` 的整面从头开录像 —— **正是预注册 §2 要排除的历史训练窗**。索引侧供 `total_frames` 的覆盖率是 **0/360**，`takeover_frame` 的登记来源是 **0** |

**另需说明的一处口径**：`tools/rl/v829_guard.py`（788 行 / 35,659 B / SHA `acd4e43e…6b5a05`）是**本次训练端新增**的配对清单守卫（`schema = v829-pair-manifest/1`），**不是研究端的冻结件**；它尚未登记、尚未提交。在研究端确认要不要把它当契约之前，请按「训练端提案」看待。它的 11 个错误码与 `selftest 13/13` 见下一份回执。
