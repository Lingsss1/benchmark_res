"""Packaged no-op agent for protocol and container conformance checks."""

from __future__ import annotations

import os

from .control import finalize_remote, submit_remote_trial


def main() -> int:
    socket_path = os.environ["RECEVOBENCH_CONTROL_SOCKET"]
    submit_remote_trial(
        socket_path,
        candidate_id="conformance-001",
        parent_id="seed",
        hypothesis="No-op protocol conformance check.",
        promoted=True,
    )
    finalize_remote(socket_path, "conformance-001")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
