"""Inspect the collector heartbeat and fail when continuous collection is unhealthy."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_bot.market_data.collector import (
    COLLECTOR_STATUS_FILENAME,
    COLLECTOR_STATUS_SCHEMA_VERSION,
)


def load_collector_status(output_dir: Path) -> dict[str, Any]:
    path = output_dir / COLLECTOR_STATUS_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("collector status must be a JSON object")
    return payload


def process_is_alive(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def collector_health(
    payload: dict[str, Any],
    max_age_seconds: float,
    *,
    now: datetime | None = None,
) -> tuple[bool, tuple[str, ...], float]:
    """Validate schema, heartbeat freshness, failures, and process liveness."""
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive")
    issues: list[str] = []
    if payload.get("schema_version") != COLLECTOR_STATUS_SCHEMA_VERSION:
        issues.append("unsupported status schema")
    try:
        heartbeat = datetime.fromisoformat(str(payload["last_heartbeat_at"]))
        if heartbeat.tzinfo is None:
            heartbeat = heartbeat.replace(tzinfo=timezone.utc)
        age = max(
            0.0,
            ((now or datetime.now(timezone.utc)) - heartbeat).total_seconds(),
        )
    except (KeyError, TypeError, ValueError):
        age = float("inf")
        issues.append("invalid heartbeat timestamp")
    if age > max_age_seconds:
        issues.append(f"heartbeat is stale ({age:.0f}s)")
    try:
        failures = int(payload.get("failures", 0))
    except (TypeError, ValueError):
        failures = 0
        issues.append("invalid failure count")
    if failures:
        issues.append(f"last cycle has {failures} failure(s)")
    status = str(payload.get("status", "unknown"))
    continuous = bool(payload.get("continuous", False))
    if status not in {"running", "complete", "sleeping"}:
        issues.append(f"unexpected collector status: {status}")
    if continuous:
        try:
            pid = int(payload.get("pid", 0))
        except (TypeError, ValueError):
            issues.append("invalid collector PID")
        else:
            if not process_is_alive(pid):
                issues.append("continuous collector process is not running")
    return not issues, tuple(issues), age


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-age", type=float, default=1800.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = load_collector_status(args.output_dir)
        healthy, issues, age = collector_health(payload, args.max_age)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"collector status unavailable: {error}\n")
    report = {
        **payload,
        "healthy": healthy,
        "heartbeat_age_seconds": age,
        "health_issues": list(issues),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"collector {'healthy' if healthy else 'unhealthy'}: "
            f"{payload.get('status', 'unknown')}; heartbeat {age:.0f}s ago; "
            f"{payload.get('successes', 0)}/{payload.get('ticker_total', 0)} "
            "tickers succeeded"
        )
        for issue in issues:
            print(f"- {issue}")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
