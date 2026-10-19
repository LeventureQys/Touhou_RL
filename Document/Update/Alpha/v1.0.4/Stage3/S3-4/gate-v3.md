# S3-4 G-v3 门控判定（机器生成）

- 生成时间：2026-10-19T11:12:06+08:00
- 命令：`C:\Users\a5935\AppData\Local\Programs\Python\Python312\python.exe tools\rl\run_v104_experiments.py --phase gate --out Document\Update\Alpha\v1.0.4\Stage3\S3-4\evidence\gate --pytest-log Document\Update\Alpha\v1.0.4\Stage3\S3-4\evidence\pytest-final-s3-4.txt --pytest-exit-code 0 --execute`
- 结论路线：`candidate-v3-blocked`
- 计数：pass=11 fail=3 blocked=1

## 逐项判定

| 项 | 状态 | 理由 / 关键值 |
| --- | --- | --- |
| matrix | pass | six formal seed-7 configurations present |
| coverage | pass | every formal run visited the fixed 7241-frame validation split exactly once |
| teacher | pass | all formal runs share the bound teacher fingerprint |
| history | pass | all formal runs declare causal-segment-v1 and share one source checkpoint per depth |
| init | pass | depth-1 and depth-3 policy runs start from a same-depth state checkpoint byte for byte |
| baseline | pass | the frozen baselines were re-evaluated on the full 7241-frame split |
| overfit | pass | reduction_at_least_30pct |
| throughput | pass | three paths measured; measured epoch seconds recorded against the 60 s ceiling |
| budget | pass | measured per-epoch seconds recorded for every formal run, with the one-time cold-start epochs separated from the steady state (設計 §5.1); this item only asserts that the record exists and is honest - whether <=60 s was met is the separate v3_entry_throughput item |
| v3_entry_throughput | fail | depth-3 steady epochs peak at 73.9 s (>60 s by +23.2%); the `<=60 s` v3 entry condition is therefore not met on the letter of the contract；depth3_steady_max_seconds=73.9, depth3_runs=state-d3 69.98/72.0 s; E1 71.32/73.9 s; E2 71.34/73.5 s; E3a 72.01/72.3 s; E3b 71.97/73.4 s; E3c 72.22/73.7 s, depth1_steady_max_seconds=44.5, throughput_probe_forecast_seconds=90.97, s3_3_measured_seconds=71.15 |
| pytest | pass | 0 failed and at least 178 cases |
| a01_state | fail | state best misses at least one A-01 bar；val_player_pos_px=13.06, val_danger_corr_perframe_median=0.6198, val_danger_corr_perframe_valid=7241, epoch=55, selection_rule=position |
| a02_policy | fail | 0-15 bucket regret is above the A-02 line；bucket_0_15_regret_px=3.049, bucket_0_15_frames=1076, bucket_0_15_top3=0.3578, bucket_0_15_near_optimal_05=0.29, global_regret_px=2.635, regret_p90_px=7.727, epoch=9 |
| startup | pass | real CLI run -> summary -> strict checkpoint load -> finite scores / legal actions |
| route | blocked | E* does not clear the v2-skip routing condition, and the written v3 entry checklist is not complete (failed=['throughput_le_60s'], blocked=[]).  In particular the <=60 s/epoch condition is not met on the letter of 設計 §5.2; the Q5 ruling covered continuing the depth-3 run, not this criterion, so v3 needs a new user ruling.  No v3 run is started by this tool. |

## 路线

E* does not clear the v2-skip routing condition, and the written v3 entry checklist is not complete (failed=['throughput_le_60s'], blocked=[]).  In particular the <=60 s/epoch condition is not met on the letter of 設計 §5.2; the Q5 ruling covered continuing the depth-3 run, not this criterion, so v3 needs a new user ruling.  No v3 run is started by this tool.

- 预选 E*：E3c，0–15 桶后悔 3.049 px，全局后悔 2.635 px，epoch 9（与第二名相差 0.001 px）
- policy sha256：d29c7fa6cfcc2370d7bf7dee38e4927f94c219ffe264005e53c6c9a8118c37be
- source state sha256：dc5d41fce9a36583ecf8ca2dba5f7734d63597a6cc46cefd4f4b3e07e03a247a

| 排序 | 配置 | 0–15 桶后悔 px | 全局后悔 px | epoch |
| ---: | --- | ---: | ---: | ---: |
| 1 | E3c | 3.049 | 2.635 | 9 |
| 2 | E2 | 3.05 | 2.499 | 12 |
| 3 | E3b | 3.1 | 2.675 | 9 |
| 4 | E3a | 3.11 | 2.689 | 9 |
| 5 | E1 | 3.728 | 3.922 | 12 |
| 6 | E0 | 3.767 | 3.594 | 12 |

## v3 进入条件清单（逐条，按字面）

| 条件 | 状态 | 判据 | 实测 |
| --- | --- | --- | --- |
| sequence_labels_coverage | pass | 序列对齐 / 标签 / 覆盖正确（7241 帧全验证、teacher 指纹一致、causal-segment-v1） | coverage/teacher/history 三项见同名 item |
| small_set_regret_reduction_30pct | pass | 小集 rank 后悔相对初始化下降 >= 30%（含有效对与梯度） | reduction=0.4776444929116685, initial_regret_px=4.585, final_regret_px=2.395 |
| valid_pairs_gradient | pass | 有效对有限非零、梯度有限非零 | finite=True, nonzero=True, first=0.16136073096277012, last=0.7600331759160449, max=2.287544841438599 |
| throughput_le_60s | fail | 吞吐 <= 60 s/epoch（训练 + 全验证，排除一次性 cache/labels） | depth3_steady_max_seconds=73.9, limit_seconds=60.0 |

清单结论：**blocked**；未通过 ['throughput_le_60s']，未判定 []；需用户裁决=True。

本工具不会启动 v3；进入 v3 与否由 MainAgent 依此书面判定决定。
