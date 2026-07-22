"""Watch market-data readiness and run one locked agent arena per session."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from zoneinfo import ZoneInfo

from trading_bot.training.arena import (
    DEFAULT_ARENA_SYMBOLS,
    arena_walk_forward_config,
    eligible_arena_dataset,
)
from trading_bot.training.dataset import SnapshotDataset


ARENA_WATCH_STATUS_SCHEMA_VERSION = "research-demo.arena-watch.status.v2"
ARENA_WATCH_RUN_CONTRACT_VERSION = "research-demo.arena-watch.run.v3"
ARENA_WATCH_STATUS_FILENAME = "_arena_watch_status.json"
ARENA_WATCH_LOCK_FILENAME = ".arena-watch.lock"
NEW_YORK = ZoneInfo("America/New_York")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat()


def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip())
    )
    if not normalized:
        raise ValueError("arena watcher requires at least one symbol")
    return normalized


def load_arena_watch_status(data_dir: Path) -> dict[str, Any] | None:
    """Load the watcher heartbeat, returning None before its first cycle."""
    path = data_dir / ARENA_WATCH_STATUS_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("arena watcher status must be a JSON object")
    return payload


def _write_status(data_dir: Path, payload: dict[str, Any]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / ARENA_WATCH_STATUS_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _base_status(
    *,
    now: datetime,
    symbols: tuple[str, ...],
    continuous: bool,
    poll_seconds: float,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    previous = previous or {}
    return {
        "schema_version": ARENA_WATCH_STATUS_SCHEMA_VERSION,
        "status": "checking",
        "last_heartbeat_at": _iso(now),
        "pid": os.getpid(),
        "continuous": continuous,
        "poll_seconds": poll_seconds,
        "symbols": list(symbols),
        "ticker_total": len(symbols),
        "ready_count": 0,
        "target_session_date": None,
        "last_completed_session_date": previous.get("last_completed_session_date"),
        "last_completed_symbols": previous.get("last_completed_symbols"),
        "last_completed_contract_version": previous.get(
            "last_completed_contract_version"
        ),
        "last_artifact": previous.get("last_artifact"),
        "readiness": [],
        "message": "checking eligible training/validation/test history",
    }


def inspect_arena_readiness(
    data_dir: Path,
    symbols: Sequence[str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Inspect the exact locked arena gate without engineering or training."""
    config = arena_walk_forward_config()
    readiness: list[dict[str, Any]] = []
    session_dates: list[str] = []
    for symbol in _normalize_symbols(symbols):
        try:
            dataset = SnapshotDataset.material_from_directory(data_dir, symbol)
            _, item = eligible_arena_dataset(dataset, config)
        except (FileNotFoundError, OSError, ValueError) as error:
            item = {
                "ready": False,
                "reason": "dataset_unavailable",
                "split": None,
                "validation": None,
                "test": None,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        readiness.append({"symbol": symbol, **item})
        if item.get("ready"):
            last_timestamp = (item.get("test") or {}).get("last_timestamp")
            try:
                timestamp = datetime.fromisoformat(str(last_timestamp))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                session_dates.append(timestamp.astimezone(NEW_YORK).date().isoformat())
            except (TypeError, ValueError):
                readiness[-1]["ready"] = False
                readiness[-1]["reason"] = "invalid_test_timestamp"

    if not readiness or not all(item.get("ready") for item in readiness):
        return readiness, None
    unique_dates = set(session_dates)
    if len(unique_dates) != 1:
        for item in readiness:
            item["ready"] = False
            item["reason"] = "ticker_session_dates_do_not_match"
        return readiness, None
    return readiness, session_dates[0]


def _arena_output_dir(data_dir: Path, now: datetime) -> Path:
    run_id = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return data_dir / "agent_runs" / "recurrent-arena" / run_id


def run_locked_arena(
    data_dir: Path,
    output_dir: Path,
    symbols: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    """Run the strict CLI contract in an isolated process and return its artifact."""
    command = [
        sys.executable,
        "-m",
        "trading_bot.training.arena",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
    ]
    for symbol in _normalize_symbols(symbols):
        command.extend(("--symbol", symbol))
    completed = runner(
        tuple(command),
        check=True,
        text=True,
        capture_output=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("agent arena produced no completion record")
    result = json.loads(lines[-1])
    artifact = Path(str(result["artifact"]))
    if not artifact.is_absolute():
        artifact = Path.cwd() / artifact
    if not artifact.is_file():
        raise RuntimeError(f"agent arena artifact is missing: {artifact}")
    if int(result.get("completed", 0)) != len(_normalize_symbols(symbols)):
        raise RuntimeError("agent arena did not complete every watched ticker")
    if int(result.get("failures", 0)):
        raise RuntimeError("agent arena reported ticker failures")
    return artifact.resolve()


def run_watch_cycle(
    data_dir: Path,
    symbols: Sequence[str] = DEFAULT_ARENA_SYMBOLS,
    *,
    continuous: bool = False,
    poll_seconds: float = 60.0,
    now: datetime | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run one readiness check and train only if this session is new."""
    timestamp = now or _utc_now()
    normalized = _normalize_symbols(symbols)
    try:
        previous = load_arena_watch_status(data_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        previous = None
    status = _base_status(
        now=timestamp,
        symbols=normalized,
        continuous=continuous,
        poll_seconds=poll_seconds,
        previous=previous,
    )
    readiness, session_date = inspect_arena_readiness(data_dir, normalized)
    status["readiness"] = readiness
    status["ready_count"] = sum(bool(item.get("ready")) for item in readiness)
    status["target_session_date"] = session_date

    if session_date is None:
        status["status"] = "waiting"
        status["message"] = (
            "waiting for every ticker to have thirteen regular, fresh, "
            "executable states for train/validation/test"
        )
        _write_status(data_dir, status)
        return status

    completed_for_contract = bool(
        status["last_completed_session_date"] == session_date
        and status["last_completed_symbols"] == list(normalized)
        and status["last_completed_contract_version"]
        == ARENA_WATCH_RUN_CONTRACT_VERSION
    )
    if completed_for_contract:
        status["status"] = "up_to_date"
        status["message"] = f"arena already completed for {session_date}"
        _write_status(data_dir, status)
        return status

    output_dir = _arena_output_dir(data_dir, timestamp)
    status["status"] = "running"
    status["message"] = f"training locked arena for {session_date}"
    _write_status(data_dir, status)
    try:
        artifact = run_locked_arena(
            data_dir.resolve(),
            output_dir.resolve(),
            normalized,
            runner=runner,
        )
    except (
        OSError,
        RuntimeError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        status["status"] = "error"
        status["last_heartbeat_at"] = _iso(_utc_now())
        status["error_type"] = type(error).__name__
        status["message"] = str(error)
        _write_status(data_dir, status)
        return status

    status["status"] = "complete"
    status["last_heartbeat_at"] = _iso(_utc_now())
    status["last_completed_session_date"] = session_date
    status["last_completed_symbols"] = list(normalized)
    status["last_completed_contract_version"] = ARENA_WATCH_RUN_CONTRACT_VERSION
    status["last_artifact"] = str(artifact)
    status["message"] = f"completed locked arena for {session_date}"
    _write_status(data_dir, status)
    return status


@contextmanager
def _exclusive_watch_lock(data_dir: Path) -> Iterator[None]:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / ARENA_WATCH_LOCK_FILENAME).open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another arena watcher already owns this data directory"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    symbols = tuple(args.symbol or DEFAULT_ARENA_SYMBOLS)
    try:
        with _exclusive_watch_lock(args.data_dir):
            while True:
                result = run_watch_cycle(
                    args.data_dir,
                    symbols,
                    continuous=not args.once,
                    poll_seconds=args.poll_seconds,
                )
                print(json.dumps(result, sort_keys=True), flush=True)
                if args.once:
                    return 1 if result["status"] == "error" else 0
                time.sleep(args.poll_seconds)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"arena watcher error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
