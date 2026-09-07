# RecEvoBench

[English](README.en.md) | 中文

RecEvoBench 是一个面向**任意推荐模型自进化 Agent** 的协议优先 benchmark。它不规定 Agent 使用 LLM、进化搜索、树搜索、诊断工具或记忆系统；只统一 Agent 得到什么、能修改什么、拥有多少预算、必须留下什么轨迹，以及最终如何由不可见评测器评分。

当前版本是一套可执行的 reference implementation，而不是方法排行榜。它包含：

- 通用黑盒任务契约与严格数据校验；
- `public feedback -> hidden select -> sealed test` 三层评测边界；
- 随机隐藏故障与跨 episode 持久记忆的自主恢复能力轨；
- Amazon 序列推荐、KuaiRec 曝光偏移、淘宝多行为购买预测三个任务模板；
- 效果、自进化过程、anytime 效率、稳定性、泛化、资源合规、诊断/修复评分；
- 单次轨迹评分与多随机种子聚合命令；
- 一份可直接跑通的示例 sealed evaluation。

> 重要：这里的“比较”只指同一次运行中 evolved candidate 相对固定 seed model 的提升，这是判断是否发生自进化的必要对照。当前阶段不比较不同 Agent，也不做任何组件消融。

## 一、参考文献总结

### Self-EvolveRec

[Self-EvolveRec](https://arxiv.org/abs/2602.12612) 把推荐模型改进组织为带方向性反馈的闭环：用户模拟器给出定性批评，模型诊断工具提供内部定量验证，并让诊断工具与模型共同演进。它说明“最终 NDCG”不足以刻画自进化，benchmark 还应记录反馈是否被转化为有效实验、诊断是否正确，以及修复是否成功。

RecEvoBench 借鉴其“反馈—诊断—修改—验证”闭环，但不把 Self-EvolveRec 的组件写成参赛要求。诊断工具、用户模拟器和 LLM 都是可选 Agent 内部实现。

### FML-bench

[FML-bench](https://arxiv.org/abs/2605.17373) 强调只看最终性能和算力会遗漏研究过程，并从探索、泛化、可靠性、效率和成本评价 Agent 的研究轨迹。对本项目的启示是：必须保留逐 trial 轨迹，评分不仅看终点，还要看在固定预算下何时找到改进、反馈能否转化、无效实验比例和代码探索范围。

### RE-Bench 与 MLE-bench

[RE-Bench](https://arxiv.org/abs/2411.15114) 使用开放式 ML 研究工程环境、固定时间预算和完整轨迹评估长时程 Agent；[MLE-bench](https://arxiv.org/abs/2410.07095) 将问题描述、数据、局部评估和隐藏 grading 分离。两者共同支持本项目的两个设计：以预算归一化的 anytime 曲线评价研究效率，以及将 Agent 可见反馈与最终 sealed grader 物理分离。

### KuaiRec 与离线推荐评测

[KuaiRec](https://arxiv.org/abs/2202.10842) 提供接近全观测的 user-item 小矩阵，并显示数据密度与曝光偏差会改变离线评测结论。因此 KuaiRec 任务把大矩阵用于开发，把小矩阵作为 sealed exposure-shift 测试。对普通隐式反馈数据，[Revisiting Negative Sampling vs. Non-sampling](https://doi.org/10.1145/3522672) 指出只在随机采样负例上排名可能产生误导，因此当前 Amazon 和淘宝任务的主分数与公开反馈都使用冻结完整目录；只有新建且显式版本化的任务才可定义仅用于反馈的采样候选。

## 二、评价指标

设任务主指标为 `m`，方向统一后记为 `q(m)`（最大化指标为 `m`，最小化指标为 `-m`），seed 与最终模型在 sealed test 上的结果分别为 `m0`、`m*`。

| 维度 | 当前实现 | 作用 |
| --- | --- | --- |
| 最终效果 | `absolute_improvement = q(m*) - q(m0)` | 判断最终模型是否真的优于同任务 seed |
| 跨任务可比效果 | `normalized_improvement = improvement / metric_range` | 避免 NDCG、GAUC、MAE 的量纲混用 |
| 有意义改进 | 超过绝对/相对 epsilon，且 guardrail 和预算均通过 | 防止把噪声、小数点波动当作进化 |
| Anytime 效率 | hidden best-so-far improvement 对计算预算的 AUC | 奖励更早找到有效改进，而非只看终点 |
| 搜索速度 | 首次有效改进预算比例、达到最终 90% 增益的预算比例 | 描述 Agent 的收敛速度 |
| 搜索持续性 | 后半程贡献的增益比例 | 区分早熟停滞与持续探索 |
| 选择质量 | selection regret、promoted regression rate | 衡量 Agent 是否会选错最终模型或错误晋升 |
| 可靠性 | valid trial ratio、crash/OOM/timeout 计数 | 反映工程可执行性 |
| 稳定性 | 多 seed 的均值、标准差、成功概率、worst-20% CVaR | 防止单次幸运运行主导结论 |
| 资源 | baseline compute unit、GPU 秒、wall time、tokens、人工介入 | 使不同硬件和 Agent 开销可审计 |
| 约束 | 准确率以外的覆盖率、长尾、误差、时延 guardrail | 防止主指标提升以系统退化为代价 |
| 诊断与修复 | fault top-1/top-3、repair success rate | 可选能力轨，评价是否能定位并修复故障 |
| 连续故障恢复 | 恢复比例、完全恢复率、恢复 BCU、遗忘率、最终能力保留 | 判断 Agent 能否在多 episode 中自主恢复且不累积退化 |
| 自进化证据 | evidence level、成功的非 seed 迭代修订、自主运行率 | 区分一次性找到好结果与自主迭代改进 |
| 反馈利用 | feedback conversion、misleading feedback、public-hidden 相关 | 衡量公开反馈能否泛化到隐藏选择集 |
| 搜索过程 | productive ratio、失败/回退恢复、浪费算力、谱系深度 | 评价持续修订和恢复能力 |
| 代码探索 | reach、spread、结构唯一率 | 描述允许修改路径内的结构探索，不进入通过判定 |

主结论以 sealed test 上的 `normalized_improvement`、`meaningful_improvement` 和跨 seed 统计为准。公开反馈指标不能直接进入最终效果分；hidden-select 只用于过程、泛化和选择诊断；sealed-test 每次运行只评估 seed 与 Agent 最终提交一次。

## 三、具体方法

### 1. 黑盒交互协议

```text
task package + seed code + public data + budget
                    |
                    v
          arbitrary evolution agent
                    |
          candidate + public trace
                    |
                    v
         hidden-select evaluator
                    |
      freeze final candidate selected by agent
                    |
                    v
           sealed-test evaluator
                    |
              scorecard.json
```

Agent 可修改 manifest 中 `editable_paths` 指定的代码，反复读取 public feedback；它不能读取 hidden-select 标签、sealed-test 数据或 grader 源码。每个 trial 必须记录候选关系、状态、公开分数、GPU/墙钟时间、token、假设和是否晋升。评测器在 Agent 停止后补入 hidden 指标，并拒绝带有伪造 hidden 指标的 public trace。

### 2. 三类任务

- `amazon-seqrec-v0`：leave-last-three 的时序切分；主指标为 sealed full-catalog `NDCG@10`；同时检查 Recall、目录覆盖率和推理时延。协议允许后续任务版本把采样候选仅用于公开快速反馈，但当前版本全程使用完整目录。
- `kuairec-watchtime-v0`：在稀疏大矩阵上开发与选择，在近全观测小矩阵上 sealed 测试；主指标为 `xAUC`，同时检查 watch-ratio MAE、全观测 xAUC 和时延。
- `taobao-purchase-v0`：严格时间窗、多行为输入（浏览、收藏、加购、购买），在冻结完整目录上预测未来购买；主指标为 `PurchaseGAUC`，检查 LogLoss、购买召回、长尾召回和时延。

任务文件位于 [`tasks/`](tasks/)，字段解释见 [`docs/protocol.md`](docs/protocol.md)。

### 3. 推荐的运行单元

每个 `Agent × task` 至少运行 5 个预注册随机种子。每次运行使用相同 seed model、数据快照、基础环境和预算；GPU 用量除以 seed model 一次标准训练的 GPU 秒，得到 Baseline Compute Unit（BCU）。无论 trial 成功、崩溃、OOM 还是超时，资源都计入预算。

### 4. 快速开始

需要 Python 3.10+。

```bash
python -m pip install -e '.[dev]'
recevobench task-list
recevobench smoke --output /tmp/recevobench-smoke
recevobench validate \
  --task tasks/amazon-seqrec-v0.json \
  --evaluation examples/amazon-evaluation.example.json
recevobench validate-run \
  --task tasks/amazon-seqrec-v0.json \
  --run examples/amazon-public-run.example.json
recevobench score \
  --task tasks/amazon-seqrec-v0.json \
  --evaluation examples/amazon-evaluation.example.json \
  --artifacts runs/amazon-seed-1 \
  --output scorecard.json
recevobench aggregate scorecard-seed-1.json scorecard-seed-2.json
recevobench recovery-fault-list --task tasks/amazon-seqrec-v0.json
```

Python API：

```python
from recevobench.io import load_evaluation_bundle, load_manifest
from recevobench.scoring import score_bundle

task = load_manifest("tasks/amazon-seqrec-v0.json")
evaluation = load_evaluation_bundle("examples/amazon-evaluation.example.json")
scorecard = score_bundle(task, evaluation)
```

### 5. 开源运行边界

仓库不重新分发第三方原始数据；使用者从官方来源下载后运行 `recevobench prepare`，生成带 checksum 的标准切分。开发时可使用本地 runner，正式评测应使用 `run-agent-docker`：Agent 只挂载 public data，通过宿主控制 socket 提交 trial；候选代码在独立、断网、只读容器中运行，private split 只在可信 `grade-campaign` 阶段挂载。

完整说明见 [隐藏故障多 Episode 恢复轨](docs/recovery-track.md)、[自进化过程评测](docs/evolution-evaluation.md)、[数据准备](docs/datasets.md)、[指标定义](docs/metrics.md)、[Agent 协议](docs/agent-protocol.md) 和 [复现政策](docs/reproducibility.md)。不同 Agent 对比、组件消融和公开排行榜仍不属于当前阶段。
