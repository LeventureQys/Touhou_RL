# 开发与交付手册｜从检出仓库到第一张配对表

## 1. 第一天可直接执行的检查

在 Windows PowerShell 进入仓库，以下命令只读已存在的文件。Git 的 commit、运行机实际 commit 和 v4.3.1 历史证据基点分别记录。训练机的源码可能与编辑端不同，重点核对当前实际导入路径与文件 SHA。

```powershell
Set-Location 'D:\Workshop\Touhou_RL'
git rev-parse HEAD
git status --short --branch
git show -s --format='%h %ad %s' --date=iso-strict 7ec7674
$env:PYTHONPATH=(Get-Location).Path
python -B -c "import sys; print(sys.executable, sys.version)"
Test-Path -LiteralPath 'touhou_rl/v420/counterfactual.py'
Test-Path -LiteralPath '.tmp_v431_seg001/full/raw'
Get-Content -Raw -Encoding UTF8 Document/Update/Alpha/v4.3.1/evidence/R38/n80.json
Get-Content -Raw -Encoding UTF8 Document/Update/Alpha/v4.3.1/evidence/R51/baseline_ladder.json
```

以下命令从两份入库证据复算，不依赖游戏或权重。第一条输出 `1 6 1 172.375`。第二条需 SciPy；以 1.15.3 为复算参考。记录顺序为 Fisher 单/双侧和 Boschloo 单/双侧，约为 0.05837、0.11675、0.04828、0.09657。列为组（A、随机），行为结果（成功、失败）。

```powershell
python -B -c "import json,pathlib,statistics as s; e=pathlib.Path('Document/Update/Alpha/v4.3.1/evidence'); a=json.loads((e/'R38/n80.json').read_bytes())['per_window']; f=json.loads((e/'R51/baseline_ladder.json').read_bytes())['baselines']['const_chosen_on_train']['per_window']; print(sum(r['arm_mean']>f[str(r['seed'])] for r in a),sum(r['arm_survivals'] for r in a),sum(r['cfair_survivals'] for r in a),s.mean(f.values()))"
python -B -c "import scipy; from scipy.stats import fisher_exact as f,boschloo_exact as b; t=[[6,1],[74,79]]; print(scipy.__version__,f(t,alternative='greater').pvalue,f(t).pvalue,b(t,alternative='greater',n=256).pvalue,b(t,n=256).pvalue)"
```

**资产端盘点。** 从 [ASSET_SYNC](../../../../ASSET_SYNC.md) 定位 Windows 与 WSL 两个检出，从 [R36 脚本](../../../../tools/rl/v431_budget_200.py) 和 [历史命令](../v4.3.1/evidence/COMMANDS.md) 追到实际使用的训练配置、引擎和 `armA-multi4-f8664-128.pt`。先读取清单再对明确路径做 `Get-FileHash -Algorithm SHA256 -LiteralPath <实际文件>` 或 `sha256sum -- <实际文件>`。只核对有业务含义的白名单，不扫描整个树。历史清单中的引擎身份与本轮配置分别核对；本轮无法找到的条目明确写 `missing_on_host`。必要依赖包括 v420 原源码/配置、游戏数据、前缀 mask 与参考哈希、旧权重及其特征构造、教师和渲染入口。命令成功读取到一个同名文件仍需核对身份和真实推理。

可在依赖齐备的环境先跑邻接契约回归。下列测试均是现有入口；选择本批未使用的 basetemp，避免并行运行争用。它们只用于软件回归。

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -p no:cacheprovider --basetemp .tmp_v440_pytest_existing_001 -q tests/test_v430_actions.py tests/test_v430_contracts.py tests/test_v430_session.py
```

## 2. 工作包、责任与派工范围

所有下表 `v440` 路径是**待实现的建议白名单**。实现者只写派发给自己的精确文件和专属新输出目录。每个包默认附本包对应的 `tests/test_v440_<模块>.py`，原 v431 文件作为只读参照。共享 CLI 由集成人串行合并。遇到可复用的函数优先导入，必要的适配保留差分测试。本计划不授权修改第三方、协议 ABI、旧资产账本、默认模型、提交或外发。

| 工作包 / 负责人 | 前置与输入 | 待交付的代码和证据 | 通过标准 / 失败处理 |
| --- | --- | --- | --- |
| W00 资产与谱系 / 运行机负责人 | 原检出、资产清单、历史配置 | `touhou_rl/v440/manifest.py`、`configs/v440/paths.example.json`、`tools/rl/v440.py` 的 preflight 入口；主机盘点 | 核验所有需要的 SHA、路径映射、真实加载；缺项明确阻塞哪些实验，统计和裁判继续 |
| W01 恢复与计分 / 引擎负责人 | W00 的真实前缀、合法控制器、历史失败记录 | `restore.py`、`judge.py`、`runner.py`（均在 `touhou_rl/v440/`）、`configs/v440/smoke.json`、两条原始轨迹和复算 | 同起点同动作复跑哈希一致；正/负各两次有效；第 600 帧死亡计 0。记录/进程异常返回对应退出码 |
| W02 动作对照与开发 / 策略负责人 | W01、冻结 A、训练供体、教师 | `policies.py`、`donors.py`、`configs/v440/feedback-dev.json`；库摘要、每窗四臂表、补齐率、成本 | 开环隔离、动作映射、教师可解性校验通过，两臂使用相同起点；质量门未过先修或扩供体 |
| W03 配对分析 / 独立评测负责人 | 清单与裁判契约；可在 W01 前离线准备 | `stats.py`、`configs/v440/feedback-heldout.json`、功效与统计脚本入口、含尝试次数的配对表 | 从原始轨迹复算与另一实现相符，主表一祖先一对，封存后只分析一次。样本不足出不完整报告 |
| W04 学习器 / RL 负责人 | W01/W02 的真实反馈、阶段性诊断 | `learn.py`、`configs/v440/learner-dev.json`；L0/L1/L2 权重、状态、更新统计 | 短轨迹梯度、预算与恢复一致，再完成三种子同预算比较；开发选模后登记独立复验需求 |
| W05 像素 / 视觉负责人 | W01、干净前缀、像素输入与训练谱系 | `pixels.py`、`configs/v440/pixels-dev.json`；A/B/C 起点/帧栈比对、B/C 权重和学习曲线 | 同起点同动作的真值一致，渲染历史可独立重建，像素 actor 不接触真值，B/C 同在线预算 |
| W06 连续课程与交付 / 研究负责人 | E3 的像素候选、冻结教师和会话控制器 | `curriculum.py`、`configs/v440/natural-dev.json`、分阶段存活表和决策记录 | 自然继承资源，课程达到推进门再提名 S1；实际能力回到原 G410 门。没有候选时照常交付证据与阻塞 |
| 可选性能包 / 引擎负责人 | W01 的慢路径真值和成本，另行获得第三方修改权限 | 默认仅 `touhou_rl/v440/benchmark.py`、`configs/v440/restore-bench.json`；新构建另列授权白名单 | 冷重放/恢复差分零未解释差异，量出实际成本收益。失败继续慢路径 |

每个包生成的新大文件和新资产目录仍须按仓库规则同步登记 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`，由同一负责人串行追加，引用 run 清单和文件 SHA；禁止覆盖历史资产或上传。包内小型可同步材料只写 `Document/Update/Alpha/v4.4.0/evidence/<新运行号>/`。新增资产前核定本机空间、训练机路径和相应权限。未来实现时才创建目录与文件，本次只交付这套文档。

**首张可派发工单。** Provider=DeepSeek，模型=DeepSeek Flash。目标：在资产主机按列名清单核对原 A 和重放输入，生成本机可复核的盘点。允许：只读 ASSET_SYNC、v431 的 R36/R38/R74、这些材料指向的精确源文件与资产路径；只新增 `Document/Update/Alpha/v4.4.0/evidence/W00-001/preflight.json`（小证据，无原始游戏数据）。禁止：全仓递归扫描、恢复 WP05、下载替代权重、改协议/第三方/历史文件、删除、提交或上传。验收：`python -m json.tool Document/Update/Alpha/v4.4.0/evidence/W00-001/preflight.json`，另附逐项文件 SHA 计算命令和独立新进程加载权重的输出；JSON 解析只验证格式。完成或阻塞即回报变更、实际命令、退出码、可运行项和缺项。W00 的程序化 preflight 可在该小工单之后另派。

## 3. 未来 CLI 契约与验收命令

**以下入口需要先按上表实现。** 由 CLI 读 config，把 W00 返回的本机路径与冻结清单合并；不把示例配置当作真实主机配置。`--run` 指定本机一个新运行目录，子命令各建专属子目录，已有封存输出拒绝重写。示例在 PowerShell 从仓库根运行；实际引擎进程应由资产机端适配器在获准的 WSL 运行环境启动。

```powershell
$env:PYTHONPATH=(Get-Location).Path
$run='.tmp_v440_runs/dev-001'
python tools/rl/v440.py preflight --config configs/v440/smoke.json --run $run
python tools/rl/v440.py smoke --config configs/v440/smoke.json --run $run
python tools/rl/v440.py donors --config configs/v440/feedback-dev.json --run $run
python tools/rl/v440.py paired-eval --config configs/v440/feedback-dev.json --run $run
python tools/rl/v440.py analyze --experiment feedback-dev --run $run
```

| 包 / 新运行目录 | 完整的后续验收命令示例 | 必须查看的实际输出 |
| --- | --- | --- |
| W03，独立验证 | `python tools/rl/v440.py paired-eval --config configs/v440/feedback-heldout.json --run .tmp_v440_runs/heldout-001`，随后 `python tools/rl/v440.py analyze --experiment feedback-heldout --run .tmp_v440_runs/heldout-001` | 配置引用已封存的 W02 库，预检必经，验签且不对外输出中间胜率。一次解封，输出四格计数、风险差、区间和不完整情形 |
| W04，三配方 | `python tools/rl/v440.py train-state --config configs/v440/learner-dev.json --run .tmp_v440_runs/learn-001` | 逐训练种子的每个配方、训练开销、梯度、checkpoints 和开发窗结果，可定位依赖的 W01/W02 运行 |
| W05，像素 | `python tools/rl/v440.py pixel-smoke --config configs/v440/pixels-dev.json --run .tmp_v440_runs/pixel-001`，随后 `python tools/rl/v440.py train-pixel --config configs/v440/pixels-dev.json --run .tmp_v440_runs/pixel-001` | A/B/C 接管哈希与历史、每个真实更新的样本来源、B/C 预算对照和分窗表现 |
| W06，自然运行 | `python tools/rl/v440.py natural-eval --config configs/v440/natural-dev.json --run .tmp_v440_runs/natural-001` | 预注册的课程层级、连续事件、抵达/完成分母、自然 S1 结果和完整动作契约 |
| 可选加速 | `python tools/rl/v440.py restore-bench --config configs/v440/restore-bench.json --run .tmp_v440_runs/bench-001` | 真实同动作的旧/新路径等价性、每小时有效配对、资源峰值、失败回退记录 |

每个接触引擎的子命令自行校验输入和前置门，不能依赖 shell 操作员记得先运行哪条命令。每次保留命令、工作目录、工具链、实际退出码及标准输出；旧脚本碰到异常后仍可能打印 checkpoint 并返回 0，消费方须同时检查完整记录数、错误和模型身份。[历史执行器](../../../../tools/rl/v431_arm_a.py)

**退出码沿用项目约定。** `0` 表示所请求操作成功执行，若请求了判据则判据已通过；`2` 表示协议/输入/环境无效；`3` 表示有效测量未达到所请求的门；`4` 表示中断、资源不足或批次不完整。将 `execution_status`、`protocol_validity`、`research_result`、`capability_gate` 写到结果中。仅生成描述性表格时退出 0 可以与“研究未定”共存。显式请求验证门时，完整有效但未达到支持条件返回 3，协议错误与合法负结果分开。版本研究默认 `capability_gate=not_run`。[退出码定义](../../../../touhou_rl/v430/contracts.py)

## 4. 测试与真实验收

建议先实现下面的定向测试。在独立测试文件出现之后才执行对应命令。运行前按批次选新 basetemp，明确绕开 `.pytest_cache`。

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -p no:cacheprovider --basetemp .tmp_v440_pytest_contract_001 -q tests/test_v440_manifest.py tests/test_v440_restore.py tests/test_v440_judge.py
python -m pytest -p no:cacheprovider --basetemp .tmp_v440_pytest_feedback_001 -q tests/test_v440_donors.py tests/test_v440_policies.py tests/test_v440_stats.py tests/test_v440_runner.py
python -m pytest -p no:cacheprovider --basetemp .tmp_v440_pytest_learning_001 -q tests/test_v440_learn.py tests/test_v440_pixels.py tests/test_v440_curriculum.py
python -m compileall -q touhou_rl/v440
```

| 测试对象 | 应覆盖的回归 | 真实运行另需验证 |
| --- | --- | --- |
| 起点与裁判 | 0 锚点、有锚点缺一帧、错一帧接管、最后一帧死亡、提前过面、空死亡偏移、同帧冲突、日志半写 | 独立重放同一前缀和动作，真值及裁判一致；至少一正一负可从真实事件重建 |
| 策略隔离 | open 在目标输入任意扰动后动作表不变；B/C 注入 trap 真值字段时拒绝；#12 与 18 候选逐键相符 | 开环从封存表发键，像素栈重建一致；核对教师没有实际发 bomb 或释放固定射击 |
| 配对分析 | 全平、全胜、零不一致、未知、重复 attempt、交叉祖先、顺序打乱、训练种子与环境种子混用 | 从原始帧重新求 Y/T；独立分析器从同一四格表得到相同单侧尾概率；发现缺对便拒绝正式判定 |
| 学习器 | L0/L1 在等长轨迹相差固定尺度、变长时差异方向正确；手工核对逐步回报；基线梯度隔离 | 同起点多次真实发键；检查奖励与实际死亡位置、预算和裁剪频率；断点续训与同配置连续运行对齐 |
| 重试与并发 | 两执行者同时领取同一臂、写盘中断、已有输出再启动、不同 manifest 下错误 resume | 刻意中断一个开发 run，保留无效尝试，恢复后每个 pair 只有一份有效得分；旧目录 SHA 不变 |
| 像素与课程 | 帧栈偏移、真实动作历史、跨局 GRU 清空、段间资源继承、训练/开发祖先交叉 | 像素策略在合法片段完整主导动作；自然起点必须从开局跑到事件，按实际到达统计 |

真实条件缺失时，软件验证可以继续；交付报告把“代码回归”“真实冒烟”“学习结果”“能力门”四项分别打勾。报告任何通过项均给出命令、退出码、原始材料和具体断言。涉及确认样本的正式分析只在封存后运行。

## 5. 异常处置与交班模板

| 观察 | 先查什么 | 本班可继续的工作 |
| --- | --- | --- |
| import v420 失败、找不到旧 A 或权重维度不匹配 | 记录主机、源码 SHA、解释器、实际导入路径、checkpoint 键/形状。核对原执行端的明确路径 | 继续裁判、历史数据复算、教师、像素重放设计；重训 A 另分配身份 |
| τ=末帧时不同臂成功数不一致 | 对同一逐帧原始轨迹用同一 judge；检查死亡帧先后和空 offset | 在统一计分前只交工程结果。缺失旧逐帧记录时并列展示旧口径与可判定范围 |
| 教师全部早死或固定动作轻松通关 | 先核对两帧 mask、窗口阶段、资源、入场哈希和教师动作能力 | 开发阶段调整任务并另立清单；已封存结果照原协议报告 |
| A 与开环接近或 A 早死 | 查看样本宽度、动作库补齐、危险物干预、裁剪与早死轨迹 | 区分未定与排除 5pp；优先做 E2 的低成本配方对照 |
| 像素重放进入密集区时报 split mask | 打印第一个分歧帧、已执行键、历史哈希、源磁带契约 | 尝试独立合法录像，复用同一真实前缀给 A/B/C；禁止对原磁带修补造出一致性 |
| 新祖先不足、存储不足、运行超预算 | 统计剩余独立单位、实际 P50/P95、每有效配对磁盘量、可复用的已完成对象 | 封存现有批次和完整缺口，转开发工作或换新预注册。保持确认集和既有资产完整 |

单个工作包完成或遇阻即返回以下五行。独立接手者用相同资产身份和命令能够恢复下一步，才算交割。

```text
范围与状态    W编号 / run_id / 源码和配置SHA / 运行主机 / 时间
变更与资产    本包文件清单 / 新资产本机位置和SHA / 已登记清单
实测与复算    完整命令 / 退出码 / 计划数、有效数、死亡数、无效数、缺失数 / 原始证据位置
研究判读      主要效应与区间 / 适用的祖先、权重、动作空间 / 支持、负结果、未定或未跑
下一班动作    下一条可运行命令 / 所需资产或权限 / 独立可做工作 / 负责人
```

首个小里程碑是 W01 能交出可信的真死、真活各一条。首个研究里程碑是 W02 能交出新开发窗的完整配对表并说明成功从何而来。版本收束时列出所有已跑、未跑和仍未判定的工作；长期能力只按[原验收门](../v4.1.0/README.md)逐项登记。
