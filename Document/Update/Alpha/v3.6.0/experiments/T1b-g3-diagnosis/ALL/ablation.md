# v3.6.0 T1 消融主表（预注册协议）

- 生成：2026-10-25T14:27:24
- 数据：`Document\Update\Alpha\v3.6.0\data\probe\d3-fields`（6 局 / 30000 决策）
- 协议：LOEO 6 折；岭回归 λ=1；局内列置换 60 次
- 总判定：**不进入 T2**

## 1 逐策略判定（预注册门禁）

| 策略 | 主标签 | 绝对门 | 无块 | 有块 | 增量（门禁标签） | Δ | p | 判定 |
|---|---|---|---:|---:|---|---:|---:|---|
| LocalDodge | min_clearance_60 | spearman 0.3 | -0.0777 | 0.8458 | （对照） (-) | - | - | **pass** |
| REGION | in_region_60 | auc 0.8 | 0.8495 | 0.8606 | min_clearance_60 spearman +0.05 (and) | -0.0024 | 1 | **fail** |
| LURE | min_clearance_60 | spearman 0.3 | 0.8458 | 0.8471 | min_clearance_60 spearman +0.05 (or) | +0.0013 | 0.016 | **fail** |
| ENEMY | attributed_spawns_60 | spearman 0.5 | 0.4412 | 0.5459 | attributed_spawns_60 spearman +0.05 (and) | +0.1047 | 0.016 | **pass** |
| POWER | p_collected_60 | auc 0.6 | 0.9929 | 0.9982 | p_collected_60 auc +0.02 / power_delta_60 spearman +0.05 (or) | +0.0053 / +0.1535 | 0.016 / 0.016 | **pass** |
| BOSS | boss_life_delta_60 | spearman 0.3 | 0.5550 | 0.4904 | boss_life_delta_60 spearman +0.05 (and) | -0.0645 | 1 | **fail** |
| EMERGENCY | bomb_needed_60 | brier <= 0.2 (实测 0) | n/a | n/a | bomb_needed_60 auc +0.02 (and) | n/a | n/a | **underpowered** |
| HISTORY | min_clearance_60 | n/a | 0.8458 | 0.8514 | min_clearance_60 spearman +0.05 (and) | +0.0056 | 0.016 | **fail** |

## 2 全部单元格

| 策略 | 标签 | 角色 | 指标 | 无块 | 有块 | Δ | 局均 Δ | p | 零散布 | n | 阳性 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| LocalDodge | min_clearance_60 | primary | spearman | -0.0777 | 0.8458 | +0.9235 | n/a | 0.016 | no | 29820 | - |
| REGION | in_region_60 | primary | auc | 0.8495 | 0.8606 | +0.0111 | +0.0108 | 0.016 | no | 29820 | 4383 |
| REGION::min_clearance_60 | min_clearance_60 | increment | spearman | 0.8458 | 0.8435 | -0.0024 | -0.0021 | 1 | no | 29820 | - |
| LURE | min_clearance_60 | primary | spearman | 0.8458 | 0.8471 | +0.0013 | +0.0023 | 0.016 | no | 29820 | - |
| ENEMY | attributed_spawns_60 | primary | spearman | 0.4412 | 0.5459 | +0.1047 | +0.1057 | 0.016 | no | 29820 | - |
| ENEMY::bullet_delta_60 | bullet_delta_60 | extra | spearman | 0.2608 | 0.3900 | +0.1292 | +0.1276 | 0.016 | no | 29820 | - |
| POWER | p_collected_60 | primary | auc | 0.9929 | 0.9982 | +0.0053 | +0.0033 | 0.016 | no | 29820 | 28303 |
| POWER::power_delta_60 | power_delta_60 | increment | spearman | 0.3262 | 0.4797 | +0.1535 | +0.1589 | 0.016 | no | 29820 | - |
| BOSS | boss_life_delta_60 | primary | spearman | 0.5550 | 0.4904 | -0.0645 | -0.0531 | 1 | no | 16741 | - |
| EMERGENCY | bomb_needed_60 | primary | auc | n/a | n/a | n/a | n/a | n/a | n/a | 29820 | 0 |
| EMERGENCY::bomb_used_60 | bomb_used_60 | extra | auc | 0.3329 | 0.3693 | +0.0364 | 0 | 0.016 | no | 29820 | 30 |
| HISTORY | min_clearance_60 | primary | spearman | 0.8458 | 0.8514 | +0.0056 | +0.0044 | 0.016 | no | 29820 | - |

## 3 全局门

- **G-1** pass 策略数 >= 3：3（阈值 3）→ 通过
- **G-2** 通过块的增量点估计为正且 p<0.05：0（阈值 0）→ 通过
- **G-3** ALL vs BASE death_60 ΔAUC >= +0.05：0.0070（阈值 0.05）→ 未过
- **G-4** 无块的置换 p 与主结果符号相反：0（阈值 0）→ 通过

## 4 生存标签与标签平衡

- **生存标签回退已触发**：`death_60` 阳性 0 < 100，按执行计划-D2补遗 §3 改用 `min_clearance_60`（绝对门 ['spearman', 0.3]，增量门 +0.05）。阈值方向未变，只换了判生存的标签。

| 二值标签 | 阳性 | 样本 | 基准率 | 备注 |
|---|---:|---:|---:|---|
| death_60 | 0 | 29820 | 0.0000 |  |
| death_240 | 0 | 29280 | 0.0000 |  |
| in_region_60 | 4383 | 29820 | 0.1470 |  |
| p_collected_60 | 28303 | 29820 | 0.9491 | ⚠ 近常量，AUC 不可与常规标签并列 |
| bomb_needed_60 | 0 | 29820 | 0.0000 |  |
| bomb_used_60 | 30 | 29820 | 0.0010 | ⚠ 近常量，AUC 不可与常规标签并列 |

## 5 诊断（读表前必读）

- **退化为常数预测器时的 pooled 指标**：生存标签空特征 = -0.0777（局均 n/a）。
  pooled LOEO AUC 会把『每折基准率不同』混进排序：死亡集中的短局恰好是训练集基准率更低的折，常数预测器因此拿到 0.317 而不是 0.5。所有 Δ 都在同一协议下计算、两侧同受影响，所以**方向可信**；但 `death_60` 的绝对 AUC 不要读成『泛化到新局的 AUC』。局均列是该偏差的敏感性对照：常数预测器在局均口径下正好 0.5。
- **零散布置换**：[]；这些块的边缘分布在局内近乎常数，打乱后 Δ 恒为 0，p 会退化成 1/(N+1)，**不可当显著证据**，只用效应量判。
- **门禁映射修正**：首版代码把增量门错挂在主标签上（REGION 用 `in_region_60` 自评），已改为按执行计划 §6.1 的门禁标签（REGION 用 `death_60`）；阈值未动。
- **pooled / 局均符号矛盾（unstable）**：[]。这些单元格 pooled 与局均指向相反方向，按执行计划 §9 记 `unstable`，不判 pass、不进 T2；两个数值都保留在第 2 节，不做取舍。
- **置换 p 的饱和**：200 次置换落在 (1/201, 1]，所以强效应普遍显示 `0.005`。该数值只说明『置换后没出现同样大的 Δ』，不是效应量；门槛判定请以 Δ 为准，p 只作显著性下限。
- **pooled / 局均符号一致**：10/10 个单元格。不一致的单元格已在第 2 节可见，需单独读。

## 6 数据清点

```json
{
 "episodes": 6,
 "rows": 30000,
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
 "speed_mode": "measure",
 "measured_speeds": [
  {
   "speed": 5.0,
   "focus_speed": 1.9,
   "speed_diag": 3.5355339059327373,
   "focus_speed_diag": 1.34350288425444,
   "samples_axis_normal": 1280,
   "samples_axis_focus": 1000,
   "samples_diag_normal": 343,
   "samples_diag_focus": 138,
   "fallback": []
  },
  {
   "speed": 5.0,
   "focus_speed": 1.9,
   "speed_diag": 3.5355339059327373,
   "focus_speed_diag": 1.34350288425444,
   "samples_axis_normal": 1399,
   "samples_axis_focus": 303,
   "samples_diag_normal": 395,
   "samples_diag_focus": 55,
   "fallback": []
  },
  {
   "speed": 4.0,
   "focus_speed": 2.0,
   "speed_diag": 2.82842712474619,
   "focus_speed_diag": 1.414213562373095,
   "samples_axis_normal": 1319,
   "samples_axis_focus": 479,
   "samples_diag_normal": 400,
   "samples_diag_focus": 162,
   "fallback": []
  },
  {
   "speed": 5.0,
   "focus_speed": 2.2,
   "speed_diag": 3.5355339059327373,
   "focus_speed_diag": 1.5556349186104046,
   "samples_axis_normal": 1278,
   "samples_axis_focus": 187,
   "samples_diag_normal": 479,
   "samples_diag_focus": 80,
   "fallback": []
  },
  {
   "speed": 4.0,
   "focus_speed": 2.3,
   "speed_diag": 2.82842712474619,
   "focus_speed_diag": 1.626345596729059,
   "samples_axis_normal": 1452,
   "samples_axis_focus": 396,
   "samples_diag_normal": 340,
   "samples_diag_focus": 198,
   "fallback": []
  },
  {
   "speed": 5.0,
   "focus_speed": 1.9,
   "speed_diag": 3.5355339059327373,
   "focus_speed_diag": 1.34350288425444,
   "samples_axis_normal": 1084,
   "samples_axis_focus": 243,
   "samples_diag_normal": 415,
   "samples_diag_focus": 23,
   "fallback": []
  }
 ],
 "per_episode": [
  {
   "episode": "replay-ep01",
   "split": "train",
   "decisions": 5000,
   "deaths": 0,
   "death_60_positives": 0,
   "bomb_used_60_positives": 30,
   "p_collected_60_positives": 4672,
   "boss_present_decisions": 2812
  },
  {
   "episode": "replay-ep02",
   "split": "train",
   "decisions": 5000,
   "deaths": 0,
   "death_60_positives": 0,
   "bomb_used_60_positives": 0,
   "p_collected_60_positives": 4761,
   "boss_present_decisions": 2662
  },
  {
   "episode": "replay-ep03",
   "split": "train",
   "decisions": 5000,
   "deaths": 0,
   "death_60_positives": 0,
   "bomb_used_60_positives": 0,
   "p_collected_60_positives": 4694,
   "boss_present_decisions": 2814
  },
  {
   "episode": "replay-ep04",
   "split": "train",
   "decisions": 5000,
   "deaths": 0,
   "death_60_positives": 0,
   "bomb_used_60_positives": 0,
   "p_collected_60_positives": 4697,
   "boss_present_decisions": 2807
  },
  {
   "episode": "replay-ep05",
   "split": "train",
   "decisions": 5000,
   "deaths": 0,
   "death_60_positives": 0,
   "bomb_used_60_positives": 0,
   "p_collected_60_positives": 4709,
   "boss_present_decisions": 2827
  },
  {
   "episode": "replay-ep06",
   "split": "train",
   "decisions": 5000,
   "deaths": 0,
   "death_60_positives": 0,
   "bomb_used_60_positives": 0,
   "p_collected_60_positives": 4770,
   "boss_present_decisions": 2819
  }
 ]
}
```
