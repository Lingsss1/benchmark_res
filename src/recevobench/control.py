"""Host-owned control socket for sandboxed agents."""

from __future__ import annotations

import json
import socket
import socketserver
import threading
from pathlib import Path
from typing import Any

from .campaign import evaluate_trial

MAX_REQUEST_BYTES = 1_000_000


class ControlError(ValueError):
    pass


class _Controller:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.lock = threading.Lock()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action")
        with self.lock:
            if action == "trial":
                event = evaluate_trial(
                    run_dir=self.run_dir,
                    candidate_id=str(request["candidate_id"]),
                    parent_id=(
                        None if request.get("parent_id") is None else str(request["parent_id"])
                    ),
                    hypothesis=str(request.get("hypothesis", "")),
                    promoted=bool(request.get("promoted", False)),
                    input_tokens=int(request.get("input_tokens", 0)),
                    output_tokens=int(request.get("output_tokens", 0)),
                    human_interventions=int(request.get("human_interventions", 0)),
                )
                return {"ok": True, "result": event}
            if action == "finalize":
                candidate_id = str(request["candidate_id"])
                (self.run_dir / "final-candidate.txt").write_text(
                    candidate_id + "\n", encoding="utf-8"
                )
                return {"ok": True, "candidate_id": candidate_id}
            if action == "ping":
                return {"ok": True, "protocol": "recevobench-control-v1"}
            raise ControlError(f"unknown control action: {action!r}")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            response = {"ok": False, "error": "control request is too large"}
        else:
            try:
                request = json.loads(raw.decode("utf-8"))
                response = self.server.controller.handle(request)  # type: ignore[attr-defined]
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
        self.wfile.write(
            json.dumps(response, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        )


class ControlServer:
    def __init__(self, socket_path: str | Path, run_dir: str | Path) -> None:
        self.socket_path = Path(socket_path)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            raise ControlError(f"control socket already exists: {self.socket_path}")
        self.server = socketserver.ThreadingUnixStreamServer(str(self.socket_path), _Handler)
        self.server.controller = _Controller(Path(run_dir).resolve())  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> ControlServer:
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.socket_path.unlink(missing_ok=True)


def control_request(socket_path: str | Path, request: dict[str, Any]) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(
            json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        )
        response = b""
        while not response.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            response += chunk
            if len(response) > MAX_REQUEST_BYTES:
                raise ControlError("control response is too large")
    result = json.loads(response.decode("utf-8"))
    if not result.get("ok"):
        raise ControlError(str(result.get("error", "unknown controller error")))
    return result


def submit_remote_trial(
    socket_path: str | Path,
    *,
    candidate_id: str,
    parent_id: str | None,
    hypothesis: str,
    promoted: bool,
    input_tokens: int = 0,
    output_tokens: int = 0,
    human_interventions: int = 0,
) -> dict[str, Any]:
    return control_request(
        socket_path,
        {
            "action": "trial",
            "candidate_id": candidate_id,
            "parent_id": parent_id,
            "hypothesis": hypothesis,
            "promoted": promoted,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "human_interventions": human_interventions,
        },
    )


def finalize_remote(socket_path: str | Path, candidate_id: str) -> dict[str, Any]:
    return control_request(socket_path, {"action": "finalize", "candidate_id": candidate_id})
