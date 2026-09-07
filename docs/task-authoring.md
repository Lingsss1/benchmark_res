# Task-authoring guide

New recommendation tasks implement the same black-box contract rather than a
new Agent integration.

1. Add a versioned manifest under `tasks/` with public, hidden and sealed splits.
2. Implement a streaming `DatasetAdapter` and record source terms/checksums.
3. Normalize interactions to the canonical CSV columns.
4. Implement deterministic metrics in `recevobench.evaluation` and document all
   eligibility, filtering, tie and missing-prediction behavior.
5. Provide a runnable seed workspace with list-valued train/predict commands.
6. Add generated or redistributable tiny raw data for CI.
7. Add positive, degraded, malformed, over-budget and leakage tests.
8. Freeze calibration evidence before marking the task stable.

Third-party adapters can register a `DatasetAdapter` in Python. A future task
registry may expose entry points; until then, contributions add an explicit
mapping in `recevobench.data.registry.ADAPTERS` so task discovery remains
auditable.

