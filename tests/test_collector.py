from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from trading_bot.market_data import collector
from trading_bot.market_data.universe import TOP_50_TICKERS


class CollectorTests(TestCase):
    def test_top_company_universe_has_50_unique_tickers(self):
        self.assertEqual(len(TOP_50_TICKERS), 50)
        self.assertEqual(len(set(TOP_50_TICKERS)), 50)

    @patch("trading_bot.market_data.collector.fetch_option_chain")
    def test_appends_greek_enriched_rows_and_migrates_old_csv(self, fetch):
        fetch.return_value = (
            "2026-08-21",
            SimpleNamespace(
                calls=pd.DataFrame(
                    [{"contractSymbol": "AAPL-C", "strike": 200, "impliedVolatility": 0.2}]
                ),
                puts=pd.DataFrame(
                    [{"contractSymbol": "AAPL-P", "strike": 200, "impliedVolatility": 0.2}]
                ),
            ),
            200.0,
            0.005,
        )
        captured_at = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            old_path = output_dir / "AAPL.csv"
            pd.DataFrame([{"collectedAt": "old", "symbol": "AAPL"}]).to_csv(
                old_path, index=False
            )
            path, row_count = collector.save_snapshot(
                "AAPL", output_dir, 0.05, captured_at
            )
            saved = pd.read_csv(path)

        self.assertEqual(row_count, 2)
        self.assertEqual(tuple(saved.columns), collector.CSV_COLUMNS)
        self.assertEqual(len(saved), 3)
        self.assertEqual(set(saved.iloc[1:]["optionType"]), {"call", "put"})
        self.assertTrue(saved.iloc[1:][["delta", "gamma", "theta", "vega"]].notna().all().all())
