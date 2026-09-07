# Frozen metric definitions

All ties are deterministic. Item-ID ascending order is the final ranking
tie-break. Users without positives are excluded from ranking macro averages;
the eligible-user count must be reported by production grading infrastructure.

## Ranking

- `Recall@K`: per-user fraction of relevant items in the first K positions,
  macro-averaged over eligible users.
- `NDCG@K`: binary-gain DCG with discount `1/log2(rank+1)`, divided by the ideal
  DCG for that user's number of positives, then macro-averaged.
- `MRR@K`: reciprocal rank of the first relevant item in the first K positions,
  macro-averaged.
- `CatalogCoverage@K`: number of distinct items occurring in any top-K list,
  divided by frozen catalog size.
- `TailPurchaseRecall@10`: Recall@10 restricted to purchase positives belonging
  to the least-frequent 20 percent of training items. Frequency ties use item ID.

Amazon uses full-catalog ranking after filtering training interactions. The
current Taobao task keeps repeat-purchase candidates and evaluates the frozen
full catalog. A future sampled-retrieval variant requires a new task version.

## Point prediction

- `PurchaseGAUC`: ROC AUC within each user that has both positive and negative
  candidates, weighted by that user's candidate count. Score ties receive 0.5.
- `LogLoss`: mean binary cross entropy. A submitted probability is used when
  present; otherwise the raw score is passed through a sigmoid.
- `MAE`: mean absolute error of predicted and observed watch ratio.
- `xAUC`: pairwise concordance for continuous watch-ratio targets. Pairs with
  equal targets are excluded; correct predicted order scores 1 and a predicted
  tie scores 0.5.
- `UserSpearman`: Spearman correlation per eligible user, macro-averaged.
- `NDCG@10_HighWatch`: binary NDCG treating `watch_ratio >= 2.0` as relevant.

KuaiRec point predictions are joined to the frozen truth with an evaluator-owned
SQLite database. XAUC remains exact: score ranks and target-tie groups are
processed from disk, so the fully observed matrix does not require loading all
rows into RAM. Prediction keys must match truth keys exactly.

## Evolution and resource metrics

Direction-oriented improvement is `q(final)-q(seed)`. When finite metric bounds
exist, normalized improvement divides by their range. A run is meaningful only
when improvement exceeds both the absolute/relative noise threshold, every
guardrail passes, and all budgets are respected.

Anytime AUC integrates hidden best-so-far normalized improvement over consumed
BCU. One BCU is the calibrated seed execution's compute seconds. GPU runs use
wall seconds multiplied by allocated GPU count; CPU-only smoke runs use wall
seconds while retaining GPU seconds as zero.

Repeated runs are aggregated inside each task first, then macro-averaged across
tasks so unequal seed counts cannot change task weights.

Self-evolution is verified only when a run has a meaningful sealed-test gain,
zero human interventions and at least one successful hidden-select revision on
the final candidate's lineage whose parent is another in-run candidate rather
than the seed. This separates
iterative adaptation from a one-shot seed modification. Feedback conversion,
validation-test gap, recovery, productive-trial, wasted-compute, lineage and
AST/Jaccard exploration metrics are specified in
[evolution-evaluation.md](evolution-evaluation.md). Structural exploration is
descriptive and never contributes to the pass/fail decision.

`InferenceP95Ms` is the nearest-rank 95th percentile of twenty evaluator-owned
prediction subprocess measurements after one excluded warm-up. The frozen
micro-batch contains at most 128 input rows; ranking tasks retain rows from one
user so the measurement represents one full-catalog request. Process startup
is intentionally included. Every raw sample is stored in `candidate-result.json`.
The warm-up and measured calls count toward consumed compute.
