# exact F2 采集预注册 v1

- 状态：planned
- 适用版本：v3.7.5
- 目的：建立带 frame_proof=exact 的 F2 真实帧数据入口
- 当前限制：本机没有游戏，尚未执行

## 采集对象

只使用 build_fence 研究变体和对应地址表。canonical 镜像不产生 exact 声明。

## 预检

1. 变体 exe、map、地址表存在；
2. exe 的 text SHA、map SHA、地址表 SHA 互相匹配；
3. TH08_ADDRESS_FILE 指向本次链接生成的地址表；
4. 进程 PID 与 fence section PID 一致；
5. 前 10 个样本通过全部 13 项检查；
6. 采集环境指纹、Python、GPU、Git HEAD 和构建身份。

## 试采集规模

- 目标：3000 个有效帧；
- hold：优先使用 accept_current=True；
- 每帧保存 PNG、JSONL proof、状态摘要、输入读回、逻辑帧和耗时；
- 事件数据必须按完整 episode/template 分组。

## 停止和失败

- 任一 exact 样本失败、hold 过期、ticket 不稳定或像素不匹配：停止；
- discarded 不为 0：试采集失败；
- exactValidFrames 不等于 frames：manifest 失败；
- 失败数据保留并标记 invalid_exact，不删除、不补齐；
- 试采集前不解冻像素几何标签。

## 未执行声明

在游戏缺失时，本预注册只能作为协议文件，不能产生真实数据、真实 throughput 或通过结论。
