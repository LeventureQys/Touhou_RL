# S3-4 G-v3 门控判定（机器生成）

- 生成时间：2026-10-19T07:15:26+08:00
- 命令：`C:\Users\a5935\AppData\Local\Programs\Python\Python312\python.exe tools\rl\run_v104_experiments.py --phase gate --out D:\WorkShop\Touhou_RL\Document\Update\Alpha\v1.0.4\Stage3\S3-4\evidence\gate-preflight-blocked`
- 结论路线：`blocked`
- 计数：pass=0 fail=0 blocked=12

## 逐项判定

| 项 | 状态 | 理由 / 关键值 |
| --- | --- | --- |
| matrix | blocked | no finished matrix evidence (experiment-table.json + matrix.json are required) |
| coverage | blocked | matrix evidence incomplete |
| teacher | blocked | matrix evidence incomplete |
| history | blocked | matrix evidence incomplete |
| init | blocked | matrix evidence incomplete |
| budget | blocked | matrix evidence incomplete |
| route | blocked | matrix evidence incomplete |
| baseline | blocked | matrix evidence incomplete |
| overfit | blocked | matrix evidence incomplete |
| throughput | blocked | matrix evidence incomplete |
| pytest | blocked | matrix evidence incomplete |
| startup | blocked | matrix evidence incomplete |

## 路线

matrix evidence is missing


本工具不会启动 v3；进入 v3 与否由 MainAgent 依此书面判定决定。
