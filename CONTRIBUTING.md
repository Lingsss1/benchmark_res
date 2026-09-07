# Contributing to RecEvoBench

Contributions are welcome for correctness fixes, new task adapters, metrics,
documentation and portability. Method leaderboard submissions are outside the
current project phase.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest --cov=recevobench --cov-report=term-missing
recevobench smoke --output /tmp/recevobench-smoke
python -m build
```

## Pull requests

Every change must:

1. preserve the public/hidden/sealed trust boundary;
2. include tests for new behavior and failure cases;
3. avoid checking in raw or prepared third-party datasets;
4. update the JSON Schema and protocol documentation when an artifact changes;
5. keep existing task versions immutable, or introduce a new task/version ID;
6. pass the full CI workflow.

New tasks should follow [the task-authoring guide](docs/task-authoring.md) and
include a generated tiny fixture so CI never needs private or large data.

By contributing, you agree that your contribution is licensed under the MIT
License in this repository.

