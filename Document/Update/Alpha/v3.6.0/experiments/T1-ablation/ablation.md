# v3.6.0 T1 消融主表（预注册协议）

- 生成：2026-10-25T13:58:12
- 数据：`Document\Update\Alpha\v3.5.0\probe\s0-teacher`（9 局 / 21156 决策）
- 协议：LOEO 9 折；岭回归 λ=1；局内列置换 200 次
- 总判定：**不进入 T2**

## 1 逐策略判定（预注册门禁）

| 策略 | 主标签 | 绝对门 | 无块 | 有块 | 增量（门禁标签） | Δ | p | 判定 |
|---|---|---|---:|---:|---|---:|---:|---|
| LocalDodge | death_60 | auc 0.8 | 0.3170 | 0.9560 | （对照） (-) | - | - | **pass** |
| REGION | in_region_60 | auc 0.8 | 0.8372 | 0.8663 | death_60 auc +0.02 (and) | -0.0148 | 1 | **fail** |
| LURE | min_clearance_60 | spearman 0.3 | 0.8808 | 0.9011 | min_clearance_60 spearman +0.05 / death_60 auc +0.02 (or) | +0.0203 / -0.0041 | 0.005 / 1 | **fail** |
| ENEMY | bullet_delta_60 | spearman 0.5 | 0.3691 | 0.4390 | bullet_delta_60 spearman +0.05 (and) | +0.0699 | 0.005 | **fail** |
| POWER | p_collected_60 | auc 0.6 | 0.7848 | 0.8252 | p_collected_60 auc +0.02 / power_delta_60 spearman +0.05 (or) | +0.0405 / +0.1666 | 0.005 / 0.005 | **pass** |
| BOSS | boss_life_delta_60 | spearman 0.3 | 0.3796 | 0.3689 | boss_life_delta_60 spearman +0.05 (and) | -0.0107 | 0.990 | **unstable** |
| EMERGENCY | bomb_needed_60 | brier <= 0.2 (实测 0.0159) | 0.9551 | 0.9573 | bomb_needed_60 auc +0.02 (and) | +0.0022 | 0.005 | **fail** |
| HISTORY | death_60 | n/a | 0.9560 | 0.9556 | death_60 auc +0.02 (and) | -0.0004 | 0.990 | **fail** |

## 2 全部单元格

| 策略 | 标签 | 角色 | 指标 | 无块 | 有块 | Δ | 局均 Δ | p | 零散布 | n | 阳性 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| LocalDodge | death_60 | primary | auc | 0.3170 | 0.9560 | +0.6390 | +0.4649 | 0.005 | no | 20886 | 462 |
| LocalDodge::min_clearance_60 | min_clearance_60 | extra | spearman | -0.1125 | 0.8808 | +0.9933 | n/a | 0.005 | no | 20886 | - |
| REGION | in_region_60 | primary | auc | 0.8372 | 0.8663 | +0.0292 | +0.0311 | 0.005 | no | 20886 | 2750 |
| REGION::death_60 | death_60 | increment | auc | 0.9560 | 0.9412 | -0.0148 | -0.0115 | 1 | no | 20886 | 462 |
| LURE | min_clearance_60 | primary | spearman | 0.8808 | 0.9011 | +0.0203 | +0.0135 | 0.005 | no | 20886 | - |
| LURE::death_60 | death_60 | increment | auc | 0.9560 | 0.9520 | -0.0041 | -0.0038 | 1 | no | 20886 | 462 |
| ENEMY | bullet_delta_60 | primary | spearman | 0.3691 | 0.4390 | +0.0699 | +0.0939 | 0.005 | no | 20886 | - |
| POWER | p_collected_60 | primary | auc | 0.7848 | 0.8252 | +0.0405 | +0.0554 | 0.005 | no | 20886 | 12691 |
| POWER::power_delta_60 | power_delta_60 | increment | spearman | 0.0745 | 0.2410 | +0.1666 | +0.0951 | 0.005 | no | 20886 | - |
| BOSS | boss_life_delta_60 | primary | spearman | 0.3796 | 0.3689 | -0.0107 | +0.0914 | 0.990 | no | 7598 | - |
| EMERGENCY | bomb_needed_60 | primary | auc | 0.9551 | 0.9573 | +0.0022 | +0.0058 | 0.005 | no | 20886 | 432 |
| EMERGENCY::bomb_used_60 | bomb_used_60 | extra | auc | 0.8111 | 0.7702 | -0.0408 | +0.0236 | 1 | no | 20886 | 390 |
| HISTORY | death_60 | primary | auc | 0.9560 | 0.9556 | -0.0004 | -0.0011 | 0.990 | no | 20886 | 462 |

## 3 全局门

- **G-1** pass 策略数 >= 3：2（阈值 3）→ 未过
- **G-2** 通过块的增量点估计为正且 p<0.05：0（阈值 0）→ 通过
- **G-3** ALL vs BASE death_60 ΔAUC >= +0.05：-0.0210（阈值 0.05）→ 未过
- **G-4** 无块的置换 p 与主结果符号相反：0（阈值 0）→ 通过

### ALL（八块全上）vs BASE 的 death_60

- 无块 AUC 0.9560（局均 0.9649） → 有块 0.9350（局均 0.9509）
- Δ = -0.0210（局均 -0.0141），p = 1

## 4 生存标签与标签平衡

- 生存标签 `death_60` 阳性 462 ≥ 100，按原门禁用 AUC。

| 二值标签 | 阳性 | 样本 | 基准率 | 备注 |
|---|---:|---:|---:|---|
| death_60 | 462 | 20886 | 0.0221 |  |
| death_240 | 1810 | 20076 | 0.0902 |  |
| in_region_60 | 2750 | 20886 | 0.1317 |  |
| p_collected_60 | 12691 | 20886 | 0.6076 |  |
| bomb_needed_60 | 432 | 20886 | 0.0207 |  |
| bomb_used_60 | 390 | 20886 | 0.0187 |  |

## 5 诊断（读表前必读）

- **退化为常数预测器时的 pooled 指标**：生存标签空特征 = 0.3170（局均 0.5000）。
  pooled LOEO AUC 会把『每折基准率不同』混进排序：死亡集中的短局恰好是训练集基准率更低的折，常数预测器因此拿到 0.317 而不是 0.5。所有 Δ 都在同一协议下计算、两侧同受影响，所以**方向可信**；但 `death_60` 的绝对 AUC 不要读成『泛化到新局的 AUC』。局均列是该偏差的敏感性对照：常数预测器在局均口径下正好 0.5。
- **零散布置换**：[]；这些块的边缘分布在局内近乎常数，打乱后 Δ 恒为 0，p 会退化成 1/(N+1)，**不可当显著证据**，只用效应量判。
- **门禁映射修正**：首版代码把增量门错挂在主标签上（REGION 用 `in_region_60` 自评），已改为按执行计划 §6.1 的门禁标签（REGION 用 `death_60`）；阈值未动。
- **pooled / 局均符号矛盾（unstable）**：['BOSS']。这些单元格 pooled 与局均指向相反方向，按执行计划 §9 记 `unstable`，不判 pass、不进 T2；两个数值都保留在第 2 节，不做取舍。
- **置换 p 的饱和**：200 次置换落在 (1/201, 1]，所以强效应普遍显示 `0.005`。该数值只说明『置换后没出现同样大的 Δ』，不是效应量；门槛判定请以 Δ 为准，p 只作显著性下限。
- **pooled / 局均符号一致**：10/12 个单元格。不一致的单元格已在第 2 节可见，需单独读。

## 6 数据清点

```json
{
 "episodes": 9,
 "rows": 21156,
 "blocks": {
  "BASE": 11,
  "REGION": 40,
  "LURE": 13,
  "ENEMY": 17,
  "POWER": 12,
  "BOSS": 12,
  "EMERGENCY": 7,
  "HISTORY": 20
 },
 "columns": {
  "BASE": [
   "BASE_boss_hp",
   "BASE_bullets_near_120",
   "BASE_bullets_near_60",
   "BASE_clearance",
   "BASE_horizon_best",
   "BASE_horizon_mean",
   "BASE_horizon_worst",
   "BASE_n_bullets",
   "BASE_power",
   "BASE_stage_frames",
   "BASE_survivable_actions"
  ],
  "REGION": [
   "REGION_centroid_x_rel",
   "REGION_centroid_y_rel",
   "REGION_den8_t0",
   "REGION_den8_t10",
   "REGION_den8_t20",
   "REGION_den8_t40",
   "REGION_den8_t60",
   "REGION_dense16_t0",
   "REGION_dense16_t10",
   "REGION_dense16_t20",
   "REGION_dense16_t40",
   "REGION_dense16_t60",
   "REGION_exit_area",
   "REGION_exit_dx",
   "REGION_exit_dy",
   "REGION_exit_ring",
   "REGION_expand_norm",
   "REGION_expand_x",
   "REGION_expand_y",
   "REGION_in_region20",
   "REGION_in_region40",
   "REGION_occ16_t0",
   "REGION_occ16_t10",
   "REGION_occ16_t20",
   "REGION_occ16_t40",
   "REGION_occ16_t60",
   "REGION_occ8_t0",
   "REGION_occ8_t10",
   "REGION_occ8_t20",
   "REGION_occ8_t40",
   "REGION_occ8_t60",
   "REGION_path_density",
   "REGION_player_comp20",
   "REGION_player_comp40",
   "REGION_union_comp",
   "REGION_union_frac",
   "REGION_wall_bottom",
   "REGION_wall_left",
   "REGION_wall_min",
   "REGION_wall_right"
  ],
  "LURE": [
   "LURE_aim_left",
   "LURE_aim_point_dx",
   "LURE_aim_right",
   "LURE_aimed_60",
   "LURE_aimed_count",
   "LURE_aimed_ttc_mean",
   "LURE_aimed_ttc_min",
   "LURE_approach_count",
   "LURE_corridor_20",
   "LURE_crosstrack_rate",
   "LURE_exit_progress",
   "LURE_room_left",
   "LURE_room_right"
  ],
  "ENEMY": [
   "ENEMY_above",
   "ENEMY_attributed_spawns",
   "ENEMY_boss_damageable",
   "ENEMY_boss_slots",
   "ENEMY_column_threat",
   "ENEMY_count",
   "ENEMY_damageable_count",
   "ENEMY_item_drop",
   "ENEMY_life_frac_mean",
   "ENEMY_life_sum",
   "ENEMY_near_120",
   "ENEMY_nearest_dist",
   "ENEMY_nearest_life",
   "ENEMY_phase_frac_mean",
   "ENEMY_pressure",
   "ENEMY_shoot_interval_mean",
   "ENEMY_unique_indices"
  ],
  "POWER": [
   "POWER_deficit",
   "POWER_deficit_frac",
   "POWER_p_above_line",
   "POWER_p_count",
   "POWER_p_max_value",
   "POWER_p_mean_y",
   "POWER_p_nearest",
   "POWER_p_onscreen",
   "POWER_power",
   "POWER_rate",
   "POWER_up_clearance",
   "POWER_y_to_line"
  ],
  "BOSS": [
   "BOSS_accepts_damage",
   "BOSS_align_safe",
   "BOSS_align_safe_margin",
   "BOSS_count",
   "BOSS_damage_window",
   "BOSS_damageable",
   "BOSS_dx",
   "BOSS_life",
   "BOSS_life_log",
   "BOSS_phase_frac",
   "BOSS_present",
   "BOSS_shoot_interval"
  ],
  "EMERGENCY": [
   "EMERGENCY_all_blocked",
   "EMERGENCY_best_route",
   "EMERGENCY_bombs",
   "EMERGENCY_deaths",
   "EMERGENCY_lives",
   "EMERGENCY_player_state",
   "EMERGENCY_routes"
  ],
  "HISTORY": [
   "HISTORY_bullets1",
   "HISTORY_bullets2",
   "HISTORY_bullets3",
   "HISTORY_bullets4",
   "HISTORY_clear1",
   "HISTORY_clear2",
   "HISTORY_clear3",
   "HISTORY_clear4",
   "HISTORY_d_bullets",
   "HISTORY_d_clear",
   "HISTORY_d_power",
   "HISTORY_d_py",
   "HISTORY_power1",
   "HISTORY_power2",
   "HISTORY_power3",
   "HISTORY_power4",
   "HISTORY_py1",
   "HISTORY_py2",
   "HISTORY_py3",
   "HISTORY_py4"
  ]
 },
 "labels": {
  "death_60": "auc",
  "death_240": "auc",
  "min_clearance_60": "spearman",
  "min_clearance_240": "spearman",
  "danger_60": "spearman",
  "in_region_60": "auc",
  "bullet_delta_60": "spearman",
  "bullet_delta_240": "spearman",
  "power_delta_60": "spearman",
  "power_delta_240": "spearman",
  "p_collected_60": "auc",
  "boss_life_delta_60": "spearman",
  "boss_life_delta_240": "spearman",
  "bomb_needed_60": "auc",
  "bomb_used_60": "auc",
  "attributed_spawns_60": "spearman"
 },
 "speed_mode": "config",
 "measured_speeds": [
  null,
  null,
  null,
  null,
  null,
  null,
  null,
  null,
  null
 ],
 "per_episode": [
  {
   "episode": "collect-ep01",
   "split": "train",
   "decisions": 3000,
   "deaths": 2,
   "death_60_positives": 60,
   "bomb_used_60_positives": 120,
   "p_collected_60_positives": 2154,
   "boss_present_decisions": 1044
  },
  {
   "episode": "collect-ep02",
   "split": "train",
   "decisions": 834,
   "deaths": 3,
   "death_60_positives": 64,
   "bomb_used_60_positives": 0,
   "p_collected_60_positives": 114,
   "boss_present_decisions": 0
  },
  {
   "episode": "collect-ep03",
   "split": "train",
   "decisions": 1041,
   "deaths": 3,
   "death_60_positives": 64,
   "bomb_used_60_positives": 0,
   "p_collected_60_positives": 363,
   "boss_present_decisions": 0
  },
  {
   "episode": "collect-ep04",
   "split": "train",
   "decisions": 3000,
   "deaths": 1,
   "death_60_positives": 30,
   "bomb_used_60_positives": 60,
   "p_collected_60_positives": 1203,
   "boss_present_decisions": 1500
  },
  {
   "episode": "collect-ep05",
   "split": "train",
   "decisions": 3000,
   "deaths": 2,
   "death_60_positives": 60,
   "bomb_used_60_positives": 60,
   "p_collected_60_positives": 1976,
   "boss_present_decisions": 1238
  },
  {
   "episode": "collect-ep06",
   "split": "validation",
   "decisions": 3000,
   "deaths": 2,
   "death_60_positives": 60,
   "bomb_used_60_positives": 60,
   "p_collected_60_positives": 2035,
   "boss_present_decisions": 1162
  },
  {
   "episode": "collect-ep07",
   "split": "validation",
   "decisions": 3000,
   "deaths": 1,
   "death_60_positives": 30,
   "bomb_used_60_positives": 30,
   "p_collected_60_positives": 2306,
   "boss_present_decisions": 1174
  },
  {
   "episode": "collect-ep08",
   "split": "train",
   "decisions": 3000,
   "deaths": 1,
   "death_60_positives": 30,
   "bomb_used_60_positives": 60,
   "p_collected_60_positives": 2122,
   "boss_present_decisions": 1480
  },
  {
   "episode": "collect-ep09",
   "split": "train",
   "decisions": 1281,
   "deaths": 3,
   "death_60_positives": 64,
   "bomb_used_60_positives": 0,
   "p_collected_60_positives": 418,
   "boss_present_decisions": 0
  }
 ]
}
```
