"""Advance paper agents only when market data or selected runs change."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_bot.execution.agent_runtime import run_paper_agents


PAPER_AGENT_WATCH_SCHEMA_VERSION = "research-demo.paper-agent-watch.v1"
PAPER_AGENT_WATCH_STATUS_FILENAME = "_paper_agent_watch_status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input_signature(data_dir: Path) -> tuple[tuple[str, int, int], ...]:
    paths = {
        path.resolve()
        for pattern in ("*.csv", "agent_runs/**/*-walk-forward.json")
        for path in data_dir.glob(pattern)
        if path.is_file()
    }
    records = []
    for path in sorted(paths):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        records.append((
            str(path.relative_to(data_dir.resolve())),
            stat.st_size,
            stat.st_mtime_ns,
        ))
    return tuple(records)


def _write_status(data_dir: Path, payload: dict[str, Any]) -> Path:
    path = data_dir / PAPER_AGENT_WATCH_STATUS_FILENAME
    temporary = path.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _collector_running(data_dir: Path) -> bool:
    path = data_dir / "_collector_status.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "running"


def _prior_status(data_dir: Path) -> dict[str, Any]:
    path = data_dir / PAPER_AGENT_WATCH_STATUS_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": PAPER_AGENT_WATCH_SCHEMA_VERSION}
    return payload if isinstance(payload, dict) else {
        "schema_version": PAPER_AGENT_WATCH_SCHEMA_VERSION
    }


def watch(
    *,
    data_dir: Path,
    database: Path,
    repo_root: Path,
    poll_seconds: float = 30.0,
    once: bool = False,
) -> int:
    """Run the fleet on changed inputs; return nonzero for a failed once run."""
    if not math_is_positive_finite(poll_seconds):
        raise ValueError("poll_seconds must be positive and finite")
    data_dir = data_dir.resolve()
    repo_root = repo_root.resolve()
    database = database.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    previous_signature = None
    while True:
        signature = _input_signature(data_dir)
        if previous_signature is None or signature != previous_signature:
            if _collector_running(data_dir):
                payload = _prior_status(data_dir)
                payload.update({
                    "schema_version": PAPER_AGENT_WATCH_SCHEMA_VERSION,
                    "status": "waiting_for_collector",
                    "pid": os.getpid(),
                    "last_heartbeat_at": _now(),
                    "poll_seconds": poll_seconds,
                    "input_file_count": len(signature),
                    "message": (
                        "waiting for the collector cycle to finish before "
                        "loading checkpoints"
                    ),
                })
                _write_status(data_dir, payload)
                if once:
                    return 0
                time.sleep(poll_seconds)
                continue
            started_at = _now()
            try:
                result = run_paper_agents(
                    data_dir=data_dir,
                    database=database,
                    repo_root=repo_root,
                )
                status = "degraded" if result["failure_count"] else "running"
                payload = {
                    **result,
                    "schema_version": PAPER_AGENT_WATCH_SCHEMA_VERSION,
                    "runtime_schema_version": result.get("schema_version"),
                    "status": status,
                    "pid": os.getpid(),
                    "started_at": started_at,
                    "last_heartbeat_at": _now(),
                    "poll_seconds": poll_seconds,
                    "input_file_count": len(signature),
                }
                _write_status(data_dir, payload)
                previous_signature = signature
                if once:
                    return int(result["failure_count"] > 0)
            except Exception as error:
                _write_status(data_dir, {
                    "schema_version": PAPER_AGENT_WATCH_SCHEMA_VERSION,
                    "status": "error",
                    "pid": os.getpid(),
                    "started_at": started_at,
                    "last_heartbeat_at": _now(),
                    "poll_seconds": poll_seconds,
                    "input_file_count": len(signature),
                    "error_type": type(error).__name__,
                    "message": str(error),
                })
                if once:
                    return 1
        else:
            payload = _prior_status(data_dir)
            payload.update({
                "pid": os.getpid(),
                "last_heartbeat_at": _now(),
                "poll_seconds": poll_seconds,
                "input_file_count": len(signature),
            })
            _write_status(data_dir, payload)
        time.sleep(poll_seconds)


def math_is_positive_finite(value: float) -> bool:
    try:
        return float(value) > 0 and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--database", type=Path, default=Path("data/agent_paper.db"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    return watch(
        data_dir=args.data_dir,
        database=args.database,
        repo_root=args.repo_root,
        poll_seconds=args.poll_seconds,
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
