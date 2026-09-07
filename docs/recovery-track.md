# 随机隐藏故障与多 Episode 恢复轨

本能力轨评价推荐模型自进化 Agent 在连续故障下能否自主完成“发现退化—诊断—修改—验证—保留能力”。它独立于普通开放式优化分数，不比较 Agent，也不要求采用某种诊断、搜索或记忆方法。

## 1. 任务协议

可信控制器从任务声明的故障集合中按私有随机种子、无放回抽取至少两个故障。每个 episode 执行：

1. 以上一 episode 最终选择的工作区作为干净起点；第一轮使用规范 seed workspace。
2. 在 trusted 区域计算干净起点的 public、hidden-select 和 sealed-test 参考值。
3. 注入一个故障并记录故障前后的 public 指标。public 可能受分布偏移影响而低估、甚至反向显示故障，是否构成有效故障只由最终 sealed 退化判定。
4. Agent 只看到故障后的工作区、public 数据、干净 public 目标值、故障标签词表、剩余预算和持久记忆目录。
5. Agent 可在当前 episode 内提交多个 candidate，但只能提交一次含 1–3 个标签的故障诊断。
6. Agent 冻结最终 candidate 后进入 trusted grading；该 candidate 成为下一 episode 的起点。
7. 所有 episode 冻结后才公开实际故障日程与 sealed 评分。

故障日程种子与候选执行种子使用相互独立的随机流；公开 run ID 不编码私有种子。正式结果必须使用 Docker runner。本地 runner 会明确标记为 trusted development，因为同一宿主进程无法构成可靠的私有边界。

```text
clean state S0
     |
 hidden fault f1 -> agent -> selected state S1
                              |
                        hidden fault f2 -> agent -> selected state S2
                                                     ...
                                                       |
                                              final trusted grading
```

## 2. 隐藏边界与可审计材料

Agent 可见：当前故障工作区、public 数据、故障前 public 目标值、允许的故障标签词表、episode 进度、跨 episode 可读写的 `agent-memory`，以及只能提交 trial、诊断和最终选择的宿主控制 socket。

Agent 不可见：抽中的故障 ID、故障日程和私有随机种子，故障前 workspace 快照，hidden-select/sealed 数据、标签与指标，以及其他 episode 的 trusted artifacts。

控制器保留任务、数据、故障目录、初始/注入后/最终 workspace 的哈希，逐 trial 哈希链、资源、人工介入和容器配置。评分前任一受信输入变化都会使运行失效。

## 3. 评价指标

对 episode `e`，将主指标统一为越大越好的方向函数 `q(·)`。记故障前、故障后和最终 sealed 指标为 `p_e`、`b_e`、`r_e`，任务指标量程为 `L`：

- 归一化故障强度：`severity_e = max(0, q(p_e)-q(b_e))/L`。
- 归一化恢复增益：`gain_e = (q(r_e)-q(b_e))/L`。
- 恢复比例：`recovery_e = clip((q(r_e)-q(b_e))/(q(p_e)-q(b_e)), 0, 1)`。
- 完全恢复：故障强度超过 epsilon，最终值相对故障前值没有超过 epsilon 的缺口，同时预算和 guardrail 均通过。
- 灾难性遗忘：`q(p_e)-q(r_e)` 超过任务 epsilon。
- 恢复 BCU：hidden-select best-so-far 首次回到故障前水平时累计计算量，除以该 episode 故障 seed 的一次标准执行计算量。
- 诊断准确率：实际故障是否位于提交标签的 top-1/top-3。

跨 episode 输出平均恢复比例、完全恢复率、自主恢复率、平均恢复 BCU、累计未恢复比例、诊断准确率、灾难性遗忘率，以及最后模型相对第一轮干净起点的最终保留量。

`self_evolution_recovery_verified=true` 必须同时满足：至少两个 episode；每个故障在 sealed-test 上均产生显著退化；每轮均完全恢复并通过预算与 guardrail；全程人工介入为零；最终能力不低于第一轮干净起点超过 epsilon。

## 4. 运行方法

先按普通任务协议准备数据。开发环境可运行：

```bash
recevobench run-recovery-agent \
  --task tasks/amazon-seqrec-v0.json \
  --prepared /path/to/prepared-amazon \
  --run-dir /tmp/amazon-recovery \
  --agent-id my-agent --seed 20260907 --episodes 2 --gpus 0 \
  -- python -m my_agent

recevobench grade-recovery --run-dir /tmp/amazon-recovery
recevobench validate-schema \
  --kind recovery-scorecard \
  --input /tmp/amazon-recovery/recovery-scorecard.json
```

正式隔离运行：

```bash
recevobench run-recovery-agent-docker \
  --task tasks/amazon-seqrec-v0.json \
  --prepared /path/to/prepared-amazon \
  --run-dir /secure/runs/amazon-recovery \
  --agent-id my-agent --seed 20260907 --episodes 2 \
  --image my-agent-image:locked --network none \
  -- python -m my_agent
```

Agent 在容器内通过 `RECEVOBENCH_CONTROL_SOCKET` 提交操作；本地模式也提供等价命令。`recevobench recovery-fault-list --task <task>` 返回标签词表，但不会返回当前抽样结果。仓库中的 `recevobench.recovery_conformance_agent` 只用于验证协议、故障注入和评分闭环，不是参评基线。

## 5. 当前范围

首版故障是可复现的源码级故障，覆盖目标方向、排序服务、序列信号、item 信号和 point prediction。每个任务只启用与其类型兼容的故障；正式发布任务前还应在目标数据快照上预注册并校准 sealed 故障强度。若某故障未产生超过 epsilon 的 sealed 退化，该 episode 标记为不可评分而不是 Agent 失败。后续可通过新版本目录扩展数据污染、特征错位、反馈漂移和训练配置故障，但不得在同一任务版本中静默修改目录。
