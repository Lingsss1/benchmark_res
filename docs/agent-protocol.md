# Black-box agent protocol

An agent may use any model, LLM, search strategy, memory or diagnostic system.
It receives a writable copy of the canonical seed workspace, public data, a task
manifest and a fixed budget. It must not rely on hidden labels or grader code.

## Local trusted development

```bash
recevobench run-agent --task amazon-seqrec-v0 \
  --prepared /data/recevobench/amazon --run-dir runs/local-1 \
  --agent-id my-agent --seed 1 -- python my_agent.py
```

The agent receives `RECEVOBENCH_RUN_DIR`, `RECEVOBENCH_WORKSPACE`,
`RECEVOBENCH_PUBLIC_DATA`, `RECEVOBENCH_TASK_PATH`,
`RECEVOBENCH_FEEDBACK_SPLIT` and `RECEVOBENCH_RANDOM_SEED`.

It may submit a candidate after changing files allowed by `editable_paths`:

```bash
recevobench trial --run-dir "$RECEVOBENCH_RUN_DIR" \
  --candidate-id candidate-001 --parent-id seed \
  --hypothesis "Add recency-aware pooling" --promoted
```

Write the selected ID to `$RECEVOBENCH_RUN_DIR/final-candidate.txt`, or let the
runner choose the latest promoted valid candidate.

## Official Docker mode

```bash
recevobench run-agent-docker --task amazon-seqrec-v0 \
  --prepared /data/recevobench/amazon --run-dir runs/official-1 \
  --agent-id my-agent --seed 1 --image my-agent-image \
  --network bridge -- python -m my_agent
```

The agent container sees only the workspace, public data, task manifest and a
Unix control socket. It submits with `remote-trial` and selects with
`remote-finalize`; the host controller owns budgets, candidate snapshots and
the hash-chained ledger. Each candidate is executed again in a separate
network-disabled, read-only container. Hidden/private data is mounted only when
the trusted operator later runs `grade-campaign`.

The packaged `python -m recevobench.conformance_agent` demonstrates this control
protocol without changing the model. It is a verifier, not a competing method.

## Required reporting

Official reports must include task/version, Agent code version, model/provider,
prompt or policy version, tools and network policy, random seeds, hardware,
container digest, token accounting policy, every trial including failures, and
the complete public run plus scorecards.

`parent-id` must name `seed` or an earlier submitted candidate. A successful
seed-to-candidate edit can establish outcome improvement, but verified iterative
self-evolution requires a later candidate-to-candidate revision that improves
on hidden selection. Candidate hashes are evaluator-owned and parent links are
controller-validated against earlier trials; free-text hypotheses remain audit
context and are not judged as evidence.
