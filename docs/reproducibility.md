# Reproducibility and release policy

An official result is reproducible only when the following immutable identifiers
are available:

- task manifest hash and RecEvoBench release;
- raw-source and prepared-split checksums;
- seed workspace hash;
- Agent source/container digest and command;
- Python/framework/CUDA versions and hardware allocation;
- random seed, public feedback policy and all budgets;
- hash-chained trial ledger, candidate hashes and final-candidate ID;
- trusted evaluation bundle and scorecard.

Task IDs are immutable. Any change to split boundaries, candidate policy,
metric semantics, seed model or budget creates a new task ID. Bug fixes that
would change historical scores require a benchmark minor/major release and a
documented migration.

At least five pre-registered Agent seeds are recommended for research results.
All runs, including crashes and protocol violations, remain in the aggregate.
Cross-task summaries are task-macro averages.

The generated smoke suite is the minimum installation check:

```bash
recevobench smoke --output /tmp/recevobench-smoke
```

It is not evidence of scientific validity. Before declaring a task stable, its
maintainer must publish seed variance, evaluator test-retest behavior, hidden to
sealed rank consistency, runtime distribution, leakage probes, and deliberately
valid/invalid conformance cases. These checks validate the benchmark and do not
compare Agent methods or perform method ablations.

