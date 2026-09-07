# RecEvoBench 评测协议 v0.1

## 目标与非目标

目标是测量一个黑盒 Agent 在固定资源内，能否自主提出、实现、训练、诊断、选择并交付一个在未见数据上优于固定 seed 的推荐模型。benchmark 评价的是完整自进化系统，而不是某一种搜索算法。

当前非目标包括：建立 Agent 排行榜、证明某种方法优于另一种方法、做组件消融、用 LLM judge 判定代码新颖性，以及替代线上 A/B 测试。

## 信任边界

| 区域 | Agent 可见性 | 用途 | 是否进入最终效果分 |
| --- | --- | --- | --- |
| train | public | 训练 seed 与候选 | 否 |
| feedback-dev | public | 每轮低成本反馈 | 否 |
| hidden-select | hidden | 轨迹 AUC、选择遗憾、过拟合检测 | 否 |
| sealed-test | sealed | seed 与最终候选一次性终评 | 是 |

`hidden-select` 由隔离评测服务执行。Agent 只能看到 public feedback，不能看到逐候选 hidden 分数。运行结束后，Agent 必须自行声明 `final_candidate_id`；随后代码与环境被哈希冻结，sealed evaluator 才开始工作。

最终发布物中的 `EvaluationBundle` 是评测器产物，包含 public trace 和评测器补入的 hidden 指标。Agent 原始提交必须令所有 `hidden_select_metric` 为 `null`；`recevobench.sealing.seal_evaluation` 会检查这一点。

## Task manifest

每个任务 JSON 至少定义：

- 数据集族、任务类型和唯一 `task_id`；
- 主指标的方向、有限范围与有意义改进阈值；
- 次指标和不可退化 guardrail；
- BCU、trial、墙钟时间、token 上限；
- public、hidden、sealed split；
- 时间切分、候选集合、负例、过滤和分层规则；
- Agent 可修改的代码路径；
- 数据版本、许可证与复现实验元数据。

所有随机性必须由 run seed 派生。任务包需要记录原始文件 checksum、预处理代码 commit、容器镜像 digest、CUDA/框架版本和 seed baseline 的统计分布。

## Run trace

一个 trial 对应一次会消耗预算的候选评估尝试。失败也必须记账。字段包括：

- `candidate_id` 与 `parent_id`：恢复搜索图；
- `candidate_sha256`：由控制器冻结的候选内容哈希，用于识别重复提交；
- `status`：`valid/crash/oom/timeout/protocol_violation/invalid`；
- `public_metric`：Agent 当时实际可见的反馈；
- `compute_seconds`、`gpu_seconds`、`wall_seconds`、输入/输出 token；
- `promoted`：候选是否成为 Agent 的 incumbent；
- `hypothesis`：本轮可审计的研究意图；
- `human_interventions`：benchmark 开始后的人为干预次数。

正式 runner 还应把代码 diff、stdout/stderr、checkpoint hash 和环境 hash 写入 append-only artifact store。v0.1 的评分合同只要求 JSON 中的最小字段。

## BCU 归一化

`1 BCU = seed model 在指定硬件、数据和训练协议下完成一次标准候选执行所需的 compute seconds`。

候选总 BCU 为所有 trial compute seconds 之和除以该 run 的 `baseline_compute_seconds`。GPU 运行的 compute seconds 为墙钟时间乘分配 GPU 数；CPU-only conformance 运行按墙钟时间记账且 GPU 秒为零。OOM、超时和崩溃前已消耗的资源照常计入。BCU 提供任务内预算归一化，原始硬件型号、GPU 秒与墙钟时间仍必须保留。

## 主判定

一次 run 只有同时满足以下条件才记为 meaningful evolution：

1. sealed test 上方向归一后的提升严格超过 `max(absolute_epsilon, |baseline| × relative_epsilon)`；
2. 所有 guardrail 通过；
3. 没有超出 compute、trial、wall time 或 token 预算。

该判定不依赖其他 Agent，因此不是方法比较。

`meaningful evolution` 只证明最终结果有效。更严格的
`self_evolution_verified` 还要求：零人工介入，并至少存在一次以运行内候选
（而不是 seed）为父节点、位于最终候选谱系中、且在 hidden-select 上取得
显著提升的修订。完整的
证据等级和过程指标见 [自进化过程评测](evolution-evaluation.md)。

## 多次运行

每个配置至少 5 个预注册种子，报告全部 run，不允许只挑最好结果。聚合器输出：

- normalized improvement 的均值与总体标准差；
- meaningful improvement 概率；
- 最差 20% normalized improvement 的 CVaR；
- hidden anytime AUC 均值；
- valid trial、预算合规、guardrail 通过率。
- verified self-evolution 概率、自主运行率、反馈转化率和过程效率。

如果未来跨任务汇总，应先在每个 task 内按有限指标范围归一化，再对 task 做等权宏平均；不得按交互数量加权，让大数据集淹没小数据集。

## 防污染与防投机

- sealed split 的样本 ID、标签和时间边界不进入 Agent 容器；
- 当前 Amazon/淘宝任务的 public feedback 和 sealed test 都使用同一份冻结完整目录；
- 若未来任务引入采样反馈，必须使用独立冻结随机种子并发布为新的 task ID；
- 提交只允许 manifest 指定路径发生变化；
- grader 在干净环境重新训练/加载最终候选，不信任 Agent 自报分数；
- 超预算、修改 grader、读取禁区、伪造轨迹均标记为 `protocol_violation`；
- 同一版本 sealed test 查询次数固定为一次，防止把测试集变成反馈集。

## 数据任务落地检查表

### Amazon

- 冻结具体 category 与版本，不把同一用户未来事件泄漏到训练；
- 按用户序列最后三次交互切为 feedback-dev、hidden-select、sealed-test；
- full catalog 排除训练期已见物品，保持 candidate policy 固定；
- 报告短/长序列、head/mid/tail、新/老物品分层。

### KuaiRec

- 大矩阵只用于 public/hidden 开发；
- 小矩阵的用户与交互作为 sealed exposure-shift 测试；
- watch ratio 定义、视频时长过滤、重复曝光聚合规则必须冻结；
- 同时报稀疏日志与近全观测矩阵的性能差，作为泛化诊断而非第二主分。

### 淘宝

- 使用全局时间窗，禁止随机切分；
- 浏览/收藏/加购只能作为购买预测的历史特征，不能泄漏目标窗行为；
- 固定候选生成器或使用全目录，主评测不临时重采负例；
- 报告活跃度、品类、价格/流行度分层与 tail purchase recall。
