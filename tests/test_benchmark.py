from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from trading_bot.market_data.benchmark import fetch_benchmark_snapshot


class BenchmarkTests(TestCase):
    @patch("trading_bot.market_data.benchmark.yf.Ticker")
    def test_uses_latest_positive_timestamped_minute_close(self, ticker):
        ticker.return_value.history.return_value = pd.DataFrame(
            {"Close": [500.0, float("nan"), 501.5]},
            index=pd.to_datetime([
                "2026-07-22T13:58:00Z",
                "2026-07-22T13:59:00Z",
                "2026-07-22T14:00:00Z",
            ]),
        )

        snapshot = fetch_benchmark_snapshot(" spy ")

        ticker.assert_called_once_with("SPY")
        ticker.return_value.history.assert_called_once_with(
            period="1d",
            interval="1m",
            auto_adjust=False,
            prepost=True,
        )
        self.assertEqual(snapshot.symbol, "SPY")
        self.assertEqual(snapshot.price, 501.5)
        self.assertEqual(snapshot.quote_time, "2026-07-22T14:00:00+00:00")
        self.assertEqual(snapshot.price_source, "yfinance.history.1m.Close")

    @patch("trading_bot.market_data.benchmark.yf.Ticker")
    def test_refuses_untimestamped_fallback(self, ticker):
        ticker.return_value.history.return_value = pd.DataFrame()

        with self.assertRaisesRegex(ValueError, "No timestamped"):
            fetch_benchmark_snapshot("SPY")

    def test_rejects_empty_symbol(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            fetch_benchmark_snapshot(" ")
