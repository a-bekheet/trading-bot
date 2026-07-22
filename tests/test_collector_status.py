import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from trading_bot.market_data.collector import (
    COLLECTOR_STATUS_FILENAME,
    COLLECTOR_STATUS_SCHEMA_VERSION,
)
from trading_bot.market_data.status import (
    collector_health,
    load_collector_status,
)


class CollectorStatusTests(TestCase):
    def test_loads_status_and_accepts_fresh_live_continuous_collector(self):
        now = datetime(2026, 7, 22, 14, 0, tzinfo=timezone.utc)
        payload = {
            "schema_version": COLLECTOR_STATUS_SCHEMA_VERSION,
            "last_heartbeat_at": (now - timedelta(seconds=30)).isoformat(),
            "status": "sleeping",
            "continuous": True,
            "pid": 123,
            "failures": 0,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / COLLECTOR_STATUS_FILENAME
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_collector_status(Path(directory))

        with patch(
            "trading_bot.market_data.status.process_is_alive",
            return_value=True,
        ):
            healthy, issues, age = collector_health(loaded, 60, now=now)

        self.assertTrue(healthy)
        self.assertEqual(issues, ())
        self.assertEqual(age, 30)

    def test_reports_stale_failed_dead_collector(self):
        now = datetime(2026, 7, 22, 14, 0, tzinfo=timezone.utc)
        payload = {
            "schema_version": COLLECTOR_STATUS_SCHEMA_VERSION,
            "last_heartbeat_at": (now - timedelta(seconds=120)).isoformat(),
            "status": "sleeping",
            "continuous": True,
            "pid": 123,
            "failures": 2,
        }
        with patch(
            "trading_bot.market_data.status.process_is_alive",
            return_value=False,
        ):
            healthy, issues, age = collector_health(payload, 60, now=now)

        self.assertFalse(healthy)
        self.assertEqual(age, 120)
        self.assertIn("heartbeat is stale (120s)", issues)
        self.assertIn("last cycle has 2 failure(s)", issues)
        self.assertIn("continuous collector process is not running", issues)

    def test_malformed_counters_are_unhealthy_instead_of_raising(self):
        now = datetime(2026, 7, 22, 14, 0, tzinfo=timezone.utc)
        payload = {
            "schema_version": COLLECTOR_STATUS_SCHEMA_VERSION,
            "last_heartbeat_at": now.isoformat(),
            "status": "running",
            "continuous": True,
            "pid": "not-a-pid",
            "failures": "not-a-count",
        }

        healthy, issues, _ = collector_health(payload, 60, now=now)

        self.assertFalse(healthy)
        self.assertIn("invalid failure count", issues)
        self.assertIn("invalid collector PID", issues)
