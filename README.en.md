# RecEvoBench

RecEvoBench is a protocol-first benchmark for arbitrary self-evolving
recommender-system agents. It standardizes the task package, seed model, visible
feedback, budgets, trajectory artifacts and trusted grading boundary without
requiring any particular LLM, search strategy, memory or diagnostic tool.

The repository ships three executable task families:

- chronological Amazon next-item recommendation;
- KuaiRec watch-time prediction under exposure shift;
- Taobao multi-behavior next-purchase recommendation.

## Install and verify

```bash
python -m pip install -e '.[dev]'
recevobench task-list
recevobench smoke --output /tmp/recevobench-smoke
pytest
```

The smoke suite generates tiny raw data and executes the complete preparation,
seed calibration, public trial, hidden evaluation, sealed test and scorecard
flow for all three tasks.

## Trust boundary

Agents see training data and public feedback only. A host-owned controller
freezes every candidate and maintains a hash-chained ledger. Hidden selection
is never returned as feedback. The sealed test runs once after final selection.
Official runs use Docker isolation; the local executor is for trusted
development only.

See [datasets](docs/datasets.md), [metrics](docs/metrics.md),
[self-evolution evaluation](docs/evolution-evaluation.md),
[agent protocol](docs/agent-protocol.md), [reproducibility](docs/reproducibility.md)
and [contributing](CONTRIBUTING.md).

Dataset files are not redistributed. The source code is MIT licensed; upstream
dataset terms remain applicable.
