# WP06 状态（像素回报/终局）

- 时间：2026-11-13T06:58:00+08:00
- 结论：**WP06.0 只读审计与设计完成；G410-3 未通过、像素事件 detector 尚未实现/验收。** WP06 为 in_progress。

## 1. WP06.0 审计结论

证据：[`PLAN.md`](PLAN.md)、[`audit.json`](audit.json)（schema `v410-wp06-audit/1`）。

### 可以复用

- HUD 几何/裁切 + 数字/图标模板；lives/bombs 读数 100%（Wilson 95% 下界 99.997%）。
- bomb 事件：436/436、LB 99.126%、延迟 0。
- 统计/匹配/fail-closed（Wilson、rule-of-three、一对一 greedy、零样本不冒充）。
- stage 过渡的流式状态机形态（eventFrame/alarmFrame、jump+sustain+refractory），锚点需改为“像素可见起点”。
- 像素时序特征（current/diff1/diff3/diff6/collapse/novelty）；玩家局部窗。
- 失败关闭/切分思想：切分单位改为原轨迹/祖先，不是 stage。

### 不能直接用于 v4.1 像素-only

- playfield death 头：学到“顶部弹幕密集”dive 域，still 48 个死亡 0 命中。
- 学习型因果事件头未收敛（test 1/270）。
- death 只靠 HUD 递减：342/530 可观测（64.5%）；188/530 在 lives==0 后无递减。
- 旧 death 延迟按引擎边沿 +10/+4 帧；v4.1 要按首次可见证据，口径不同。
- NCC 玩家模板依赖引擎 px/py；v4.1 禁止运行时真值。
- 全局 play96 death CNN test 8.38%/2.08%。
- stage_clear 旧门：像素过渡起点中位 −112 帧；旧窗仅 15/55。
- game_over 静态规则需 60 帧 lag，延迟 89 帧。
- 56×160 绿色 HUD 丢失颜色/图标/暂停菜单/完整客户区；v4.1 要保留原始 HUD 区域。

### 数据阻塞

- `.tmp_v400_d1` 的 `frame_event_flags` 是每 4 帧单帧快照；**469,216 行里 death=0、bomb=0**，而 45 个 shard 实际 `stop_reason=death` → **不能当事件真值**。
- 必须新建连续逐帧事件数据/标注，冻结“首次可见证据”锚点；v4.1 需要全分辨率 HUD + 玩家局部窗 + 显式状态机。

## 1.1 WP06.1 离线事件/状态机契约（完成）

证据：[`CONTRACT.md`](CONTRACT.md)、[`contract.json`](contract.json)；`pytest tests/test_v410_reward_events.py tests/test_v410_reward_state_machine.py` → **71 passed**；compileall exit 0。

- 7 状态 + 25 条显式转移；UNKNOWN 不自动补零；表外转移 fail-closed。
- VISIBLE 锚点（首次可见证据）与 ENGINE 锚点分离；death 证据优先级 sprite_vanish>death_effect>hud_lives_decrement>respawn_only；不可见必须 unobservable 留分母。
- death 主延迟 P95≤10、引擎 P95≤30/max≤60；事件匹配/计数/Wilson 已实现。
- 资源归因：reset 窗口、bomb stock move、score reset 等；逐帧 label 13 字段 + 轨迹/祖先隔离 + 一次性确认封存。
- 400/536/550 计数口径已固化；仍**无 detector、无真实连续流、无确认集**，G410-3 未通过。

## 2. 下一阶段

1. **WP06.1**：建立连续逐帧 HUD/像素事件数据（death/death-bomb/no-bomb death、资源增减、stage transition、game_over、menu、遮挡/慢速/暂停/长无事件），按原轨迹/祖先隔离。
2. **WP06.2**：全分辨率 HUD 模板/数字/图标读数 + 玩家局部窗/定位；显式状态机；先做可解释规则，再引入学习 detector。
3. **WP06.3**：G410-3 事件门（recall/precision Wilson 99% 下界、≥400 正/预测事件、延迟 P95≤10/30 帧、资源 99.5%）；确认集封存。
4. 未实现/未验收前，不能写 G410-3 通过；不能拿旧 G2 数字改名。
