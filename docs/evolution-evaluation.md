# 自进化过程评测

本页规定 RecEvoBench 如何判断一个推荐模型 Agent 不只是“最终交出较好模型”，而是在固定预算内完成了可审计的迭代式自进化。该协议面向任意实现，不要求 Agent 使用 LLM、记忆、诊断器、进化算法或特定推荐模型，也不比较不同方法、不做组件消融。

## 一、参考文献总结

### Self-EvolveRec：结果之外还要验证反馈闭环

[Self-EvolveRec](https://arxiv.org/abs/2602.12612) 将推荐模型改进组织为“用户反馈—模型诊断—修改—验证”循环，并指出单一效果指标不能说明系统是否正确理解问题或完成了有效修复。它还用注入故障评估诊断与修复能力。RecEvoBench 因此将最终效果、迭代修订、反馈转化、失败恢复以及可选的故障诊断分开报告；但不会把论文中的用户模拟器或诊断器规定为 Agent 必需组件。

### FML-bench：评价开放式研究过程

[FML-bench](https://arxiv.org/abs/2605.17373) 将开放式 ML Agent 的过程指标归纳为探索、泛化、可靠性、效率和成本，并使用探索范围/离散度、验证—测试差距、有效步骤比例、随步骤变化的 AUC、首次/最佳改进时刻和后期增益等指标。RecEvoBench 采用同类维度，但把“步骤”替换为任务内可校准的 BCU，并保留负提升而非截断，以暴露真实回退。代码探索采用确定性的 AST/Jaccard 描述指标，而非语义模型嵌入，避免引入额外模型和不可复现依赖。

### RE-Bench、MLAgentBench 与 MLE-bench：时间、轨迹和隐藏终评

[RE-Bench](https://arxiv.org/abs/2411.15114) 为每次评分留下带时间戳的日志，从而恢复固定时间预算下的 score-over-time 曲线。[MLAgentBench](https://arxiv.org/abs/2310.03302) 保存 Agent 的行动、观察和工作区状态，并同时评价最终能力与时间/token 效率。[MLE-bench](https://arxiv.org/abs/2410.07095) 使用参赛者可见开发环境和独立隐藏 grader。三者共同支持 RecEvoBench 的三项规则：所有尝试（包括失败）进入 append-only 轨迹；过程效率按预算位置计算；最终结论只由 sealed test 给出。

## 二、评价指标

记主指标经方向统一后的值为 `q(m)`；最大化指标不变，最小化指标取负。一次相对父候选的变化只有严格超过任务 manifest 中的噪声阈值

`epsilon(parent) = max(absolute_epsilon, |parent| × relative_epsilon)`

才算显著。所有过程指标均由评测器记录的分数、代码快照、父子关系和资源账本计算，不依据 Agent 自报的自然语言 `hypothesis`。

### 1. 自进化证据等级

| 字段 | 判定 |
| --- | --- |
| `none` | sealed test 没有有意义改进，或违反预算/guardrail |
| `outcome_only` | 最终结果有意义，但存在人工介入，且没有成功的非 seed 迭代修订 |
| `autonomous_outcome` | 最终结果有意义、零人工介入，但只证明找到更好结果 |
| `iterative` | 最终结果有意义，且最终谱系中至少一个以运行内候选为父节点的修订在 hidden-select 上显著提升，但存在人工介入 |
| `autonomous_iterative` | 同时满足最终有意义改进、最终谱系上的成功非 seed 迭代修订、零人工介入 |

只有 `autonomous_iterative` 令 `self_evolution_verified=true`。种子到第一个候选的提升不算迭代证据；最终候选谱系中至少要出现 `candidate-A -> candidate-B`，且 B 在 hidden-select 上相对 A 显著提升。

这是行为轨迹上的操作性证据，不是“反馈导致改进”的因果证明。若未来需要因果结论，必须增加屏蔽或置乱反馈的配对运行；由于当前明确不做消融，该实验不属于 v0.1。

### 2. 效果、泛化与反馈利用

| 指标 | 定义 | 越好方向 |
| --- | --- | --- |
| `normalized_improvement` | sealed-test 上 `(q(final)-q(seed))/metric_range` | 高 |
| `validation_test_gap` | 最终候选在 hidden-select 与 sealed-test 上的归一化提升之差的绝对值 | 低 |
| `public_hidden_spearman` | 同一批有效候选 public 与 hidden-select 分数的 Spearman 相关 | 高；只作诊断 |
| `feedback_conversion_rate` | public 相对父节点显著提升的修订中，hidden-select 也显著提升的比例 | 高 |
| `misleading_feedback_rate` | public 显著提升但 hidden 未显著提升的比例 | 低 |
| `successful_revision_ratio` | 可评价父子修订中 hidden-select 显著提升的比例，包含 seed→candidate | 高 |
| `successful_iterative_revision_count` | 最终候选谱系中，父节点是运行内候选且 hidden-select 显著提升的次数 | 高；验证自进化所必需 |

分母为零或无法获得必要基线时返回 `null`，不把“没有机会观察”误记为 0。

### 3. 搜索进展、恢复与资源效率

| 指标 | 定义 | 越好方向 |
| --- | --- | --- |
| `productive_trial_ratio` | 有效 trial 中，创造显著 hidden best-so-far 的比例 | 高 |
| `iterative_revision_ratio` | 有效 trial 中，父节点不是 seed 的比例 | 描述性 |
| `post_failure_recovery_rate` | crash/OOM/timeout/invalid 后下一次尝试恢复为 valid 的比例 | 高 |
| `regression_recovery_rate` | hidden 回退后，其后代重新显著超过回退前父节点的比例 | 高 |
| `wasted_compute_fraction` | 失败或未创造显著 hidden best-so-far 的计算量占比 | 低 |
| `hidden_improvement_per_compute_unit` | hidden best 归一化提升除以已消耗 BCU | 高 |
| `best_improvement_compute_fraction` | 最终 hidden best 首次形成时已消耗计算量 / 总预算 | 低 |
| `hidden_auc_over_budget` | hidden best-so-far 归一化提升关于 BCU 比例的阶梯曲线面积 | 高 |

失败尝试消耗的资源全部计入分母。`best_improvement_compute_fraction` 以最大允许 BCU 为分母，保证同一任务内可比。

### 4. 搜索结构与代码探索

| 指标 | 定义 |
| --- | --- |
| `max_lineage_depth` | seed 到最深候选的父子链长度 |
| `search_branching_ratio` | 候选搜索图中拥有两个及以上子节点的父节点比例 |
| `unique_candidate_ratio` | 有效候选中 evaluator 冻结的代码 SHA-256 去重比例 |
| `code_exploration_reach` | seed 与各候选 AST 特征集合 Jaccard 距离的最大值 |
| `code_exploration_spread` | 所有候选对之间 AST 特征 Jaccard 距离的均值 |
| `structural_unique_candidate_ratio` | AST 特征集合去重后的候选比例 |

AST 特征包含节点类型、调用名、函数/类定义和语句结构哈希；只读取 manifest 允许修改的路径。探索范围只描述搜索行为，不进入 `meaningful_improvement` 或 `self_evolution_verified` 判定，因为大量无意义改动不应获得奖励。

### 5. 多随机种子和跨任务聚合

每个 `Agent × task` 至少预注册 5 个随机种子。先在任务内聚合，再对 Amazon、KuaiRec、淘宝等任务做等权宏平均。除已有的效果均值、标准差、成功概率和 CVaR 外，聚合器报告：

- `probability_of_verified_self_evolution`；
- `autonomous_run_rate`；
- 平均 productive/feedback-conversion/validation-test-gap/wasted-compute；
- 平均代码探索范围、结构唯一率与最大谱系深度；
- 最弱任务平均归一化提升，以及任务间平均提升标准差。

这些值评价同一 Agent 在不同推荐范式中的泛化，不构成方法排行榜。

## 三、具体方法

### 1. 单次运行

1. 评测器发布任务 manifest、seed workspace、public train/feedback 数据和固定预算。
2. Agent 从 seed 或已有候选创建新候选，并声明真实 `parent_id`；控制器校验父节点、冻结代码、记录 `candidate_sha256`，将候选交给 public evaluator。
3. 每个 valid/crash/OOM/timeout/invalid 尝试都写入哈希链账本。Agent 只收到 public feedback，自主决定下一轮修改和最终候选。
4. 运行结束后，可信 grader 对所有有效快照补算 hidden-select 分数，用于过程评测；这些分数从不返回给 Agent。
5. grader 只对 seed 与 Agent 已冻结的最终候选运行 sealed test，然后生成包含效果、过程、自主性和资源指标的 scorecard。

父节点必须是 `seed` 或轨迹中更早出现的候选；前向引用和不存在的父节点都会被协议校验拒绝。这使谱系深度、反馈转化和恢复指标可重建。

### 2. 运行命令

使用已有 sealed evaluation 计算过程指标：

```bash
recevobench score \
  --task tasks/amazon-seqrec-v0.json \
  --evaluation examples/amazon-evaluation.example.json \
  --artifacts runs/amazon-seed-1 \
  --output scorecard.json
```

`--artifacts` 是可选项；缺少代码快照时只有三个代码探索字段为 `null`，其余轨迹指标仍可计算。正式 `grade-campaign` 会自动传入 artifact 根目录。

### 3. 判读顺序

建议按以下顺序阅读单次 scorecard：

1. 先检查 `meaningful_improvement`、预算和 guardrail，确认终点有效；
2. 再检查 `self_evolution_verified` 和 `evolution_evidence_level`，区分结果发现与迭代进化；
3. 用 feedback conversion、validation-test gap 和 selection regret 判断是否过拟合公开反馈；
4. 用 productive ratio、恢复率、wasted compute 和 anytime AUC 判断研究效率；
5. 最后用搜索图和代码探索字段解释 Agent 的行为，不把它们当成绩奖励。

诊断/修复轨使用可选的预注册注入故障，输出 top-1、top-3 和 repair success；它补充说明 Agent 是否会定位故障，但不是所有推荐模型 Agent 的强制能力。
