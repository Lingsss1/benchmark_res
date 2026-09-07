"""Append-only, hash-chained trial ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class LedgerError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _event_hash(sequence: int, previous_hash: str, payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "sequence": sequence,
                "previous_hash": previous_hash,
                "payload": payload,
            }
        )
    ).hexdigest()


class TrialLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        previous_hash = "0" * 64
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerError(f"invalid ledger JSON at line {line_number}") from exc
                expected_sequence = len(events) + 1
                if event.get("sequence") != expected_sequence:
                    raise LedgerError(f"ledger sequence break at line {line_number}")
                if event.get("previous_hash") != previous_hash:
                    raise LedgerError(f"ledger hash-chain break at line {line_number}")
                expected_hash = _event_hash(
                    expected_sequence, previous_hash, event.get("payload", {})
                )
                if event.get("event_hash") != expected_hash:
                    raise LedgerError(f"ledger event hash mismatch at line {line_number}")
                events.append(event)
                previous_hash = expected_hash
        return events

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            lines = [line for line in handle if line.strip()]
            previous_hash = "0" * 64
            for line_number, line in enumerate(lines, start=1):
                event = json.loads(line)
                expected = _event_hash(line_number, previous_hash, event.get("payload", {}))
                if (
                    event.get("sequence") != line_number
                    or event.get("previous_hash") != previous_hash
                    or event.get("event_hash") != expected
                ):
                    raise LedgerError(f"cannot append to corrupted ledger at line {line_number}")
                previous_hash = expected
            sequence = len(lines) + 1
            event = {
                "sequence": sequence,
                "previous_hash": previous_hash,
                "payload": payload,
            }
            event["event_hash"] = _event_hash(sequence, previous_hash, payload)
            handle.seek(0, 2)
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return event

    def payloads(self) -> Iterator[dict[str, Any]]:
        for event in self.read():
            yield dict(event["payload"])
