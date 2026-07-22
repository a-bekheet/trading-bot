from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from trading_bot.interface.data import (
    available_tickers,
    load_latest_snapshot,
    market_data_freshness_status,
    market_session_status,
)


class InterfaceDataTests(TestCase):
    def test_loads_only_latest_snapshot(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            pd.DataFrame(
                [
                    {"collectedAt": "first", "symbol": "AAPL"},
                    {"collectedAt": "latest", "symbol": "AAPL"},
                    {"collectedAt": "latest", "symbol": "AAPL"},
                ]
            ).to_csv(data_dir / "AAPL.csv", index=False)

            tickers = available_tickers(data_dir)
            latest = load_latest_snapshot(data_dir, "AAPL")

        self.assertEqual(tickers, ["AAPL"])
        self.assertEqual(len(latest), 2)
        self.assertEqual(set(latest["collectedAt"]), {"latest"})

    def test_market_session_status_preserves_legacy_fallback(self):
        legacy = market_session_status(pd.DataFrame([{"symbol": "AAPL"}]))
        closed = market_session_status(pd.DataFrame([{
            "symbol": "AAPL", "marketState": "CLOSED",
        }]))
        regular = market_session_status(pd.DataFrame([{
            "symbol": "AAPL", "marketState": "REGULAR",
        }]))

        self.assertEqual(legacy, {
            "provider_state": "UNKNOWN",
            "regular": False,
            "coverage": 0.0,
            "trading_enabled": True,
        })
        self.assertFalse(closed["trading_enabled"])
        self.assertTrue(regular["trading_enabled"])

    def test_market_freshness_status_distinguishes_legacy_fresh_and_stale(self):
        legacy = market_data_freshness_status(pd.DataFrame([{
            "collectedAt": "2026-07-22T10:00:00Z",
        }]), max_age_seconds=1_200)
        fresh = market_data_freshness_status(pd.DataFrame([{
            "collectedAt": "2026-07-22T10:00:00Z",
            "underlyingQuoteTime": "2026-07-22T09:50:00Z",
            "underlyingQuoteTimeSource": "regularMarketTime",
            "underlyingPriceSource": "regularMarketPrice",
        }]), max_age_seconds=1_200)
        stale = market_data_freshness_status(pd.DataFrame([{
            "collectedAt": "2026-07-22T10:00:00Z",
            "underlyingQuoteTime": "2026-07-22T09:00:00Z",
        }]), max_age_seconds=1_200)

        self.assertEqual((legacy["coverage"], legacy["trading_enabled"]), (0, True))
        self.assertEqual((fresh["age_seconds"], fresh["trading_enabled"]), (600, True))
        self.assertEqual((stale["age_seconds"], stale["trading_enabled"]), (3_600, False))
