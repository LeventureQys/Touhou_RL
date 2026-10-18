# Stage 1 · collect（采集）

把「游戏里发生的事」变成磁盘上的原始分片。**这是三段中唯一需要占用真实游戏的阶段**，
其余两个阶段都是纯离线。

## 支持的组合

| source | modality | 实现 | 标签从哪来 |
| --- | --- | --- | --- |
| `replay` | `privileged` | `touhou_rl.capture_dataset` | 回放喂给游戏的输入（`g_GuiMessageInputCurrent`） |
| `replay` | `vision` | `touhou_rl.vision_capture` | 同上，**外加**每帧画面 |
| `teacher` | `privileged` | `touhou_rl.collect_pilot` | 脚本教师在真实环境里行动 |
| `teacher` | `vision` | 未实现（会显式报错） | — |

`teacher + vision` 还没接：视觉采集目前只在**回放播放**时读帧，而教师采集要走环境循环。
在它落地之前，视觉策略学生只能从人类轨迹学习（见 `train/README.md` 的已知缺口）。

## 用法

```powershell
# 视觉数据：从一份回放里抽帧（30 Hz）
python -m touhou_rl.pipeline collect --modality vision `
    --replay th08_replay/Th8_ud3936.rpy `
    --out third_party/th08/build/rl/datasets/vision-lunatic --max-seconds 160

# 特权数据：同一份回放，只要数值状态
python -m touhou_rl.pipeline collect --modality privileged `
    --replay th08_replay/Th8_ud3936.rpy --out .../privileged-lunatic --max-seconds 160

# 特权数据：让脚本教师自己打，采它走到过的状态（DAgger 数据）
python -m touhou_rl.pipeline collect --modality privileged --source teacher `
    --out .../dagger --episodes 8 --max-seconds 1320 --epsilon 0.05
```

## 产出

```
<out>/
  <tag>-00000.npz ...        分片（视觉 512 帧/片，特权 2048 帧/片）
  <tag>-00000.npz.json       每片的旁车（帧数、时间、来源）
  vision-summary.json | dataset-summary.json
  capture-report.json        本阶段统一写的采集报告
  collect-report.json        ← 本阶段产出（含上面所有报告的引用）
```

## 成本（实测）

| 量 | 值 |
| --- | --- |
| 游戏逻辑帧率 | 60 fps（前台）；**失焦掉到 14 fps** |
| 视觉采集速率 | 24–28 帧/s（每 2 逻辑帧取 1；含抓帧 0.7 ms + 缩放 + 写盘） |
| 特权采集速率 | 58–60 帧/s |
| 视觉单帧体积 | 128×112×3 ≈ 43 KB（压缩后约 30 KB） |
| 2.7 分钟视觉采集 | 4,524 帧 / 9 分片 / 136 MB |

## 必须知道的坑

1. **必须保持前台**：失焦会让游戏降到 14 fps，采到的帧率直接掉一半以上。采集器内置
   2 秒一次的前台守卫，会抢焦点。
2. **同一时刻只能有一个 `th08.exe`**：游戏自带命名互斥体，残留进程会让下一次启动失败
   （表现为菜单宏超时）。
3. **Extra 回放只有一个面（第 8 面）**：`--stage` 默认 `-1` 会自动取回放自己的第一面，
   硬写 0 会一路走到超时。
4. **`--every 2` 不是随手定的**：策略每 2 帧才决策一次，按 60 Hz 采会让磁盘和数据量翻倍，
   而标签并没有变多。
5. **本阶段不做任何标签计算**：教师价值是特权观测的纯函数，放在 `process` 阶段算，
   这样换教师不用重采视频。
