import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from trading_bot.market_data import collector
from trading_bot.market_data.benchmark import BenchmarkSnapshot
from trading_bot.market_data.option_chain import OptionChainSnapshot
from trading_bot.market_data.universe import TOP_50_TICKERS


class CollectorTests(TestCase):
    def test_collection_interval_is_start_to_start_not_post_cycle_sleep(self):
        self.assertEqual(
            collector.next_cycle_delay(
                300.0,
                1_000.0,
                current_monotonic=1_102.5,
            ),
            197.5,
        )
        self.assertEqual(
            collector.next_cycle_delay(
                60.0,
                1_000.0,
                current_monotonic=1_102.5,
            ),
            0.0,
        )

    def test_top_company_universe_has_50_unique_tickers(self):
        self.assertEqual(len(TOP_50_TICKERS), 50)
        self.assertEqual(len(set(TOP_50_TICKERS)), 50)

    @patch("trading_bot.market_data.collector.fetch_option_chain_snapshot")
    def test_appends_greek_enriched_rows_and_migrates_old_csv(self, fetch):
        fetch.return_value = OptionChainSnapshot(
            chains=(
                (
                    "2026-08-21",
                    SimpleNamespace(
                        calls=pd.DataFrame([{
                            "contractSymbol": "AAPL-C1", "strike": 200,
                            "impliedVolatility": 0.2,
                        }]),
                        puts=pd.DataFrame([{
                            "contractSymbol": "AAPL-P1", "strike": 200,
                            "impliedVolatility": 0.2,
                        }]),
                    ),
                ),
                (
                    "2026-09-18",
                    SimpleNamespace(
                        calls=pd.DataFrame([{
                            "contractSymbol": "AAPL-C2", "strike": 200,
                            "impliedVolatility": 0.25,
                        }]),
                        puts=pd.DataFrame([{
                            "contractSymbol": "AAPL-P2", "strike": 200,
                            "impliedVolatility": 0.25,
                        }]),
                    ),
                ),
            ),
            spot=200.0,
            dividend_yield=0.005,
            market_state="REGULAR",
            underlying_price_source="regularMarketPrice",
            underlying_quote_time="2026-07-21T11:59:59+00:00",
            underlying_quote_time_source="regularMarketTime",
        )
        captured_at = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
        benchmark = BenchmarkSnapshot(
            symbol="SPY",
            price=625.0,
            price_source="yfinance.history.1m.Close",
            quote_time="2026-07-21T11:59:00+00:00",
            quote_time_source="yfinance.history.1m.index",
        )

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            old_path = output_dir / "AAPL.csv"
            pd.DataFrame([{"collectedAt": "old", "symbol": "AAPL"}]).to_csv(
                old_path, index=False
            )
            path, row_count = collector.save_snapshot(
                "AAPL",
                output_dir,
                0.05,
                captured_at,
                benchmark_snapshot=benchmark,
            )
            saved = pd.read_csv(path)

        fetch.assert_called_once_with("AAPL", 3)
        self.assertEqual(row_count, 4)
        self.assertEqual(tuple(saved.columns), collector.CSV_COLUMNS)
        self.assertEqual(len(saved), 5)
        self.assertEqual(set(saved.iloc[1:]["optionType"]), {"call", "put"})
        self.assertEqual(set(saved.iloc[1:]["marketState"]), {"REGULAR"})
        self.assertEqual(
            set(saved.iloc[1:]["underlyingPriceSource"]),
            {"regularMarketPrice"},
        )
        self.assertEqual(
            set(saved.iloc[1:]["underlyingQuoteTimeSource"]),
            {"regularMarketTime"},
        )
        self.assertEqual(set(saved.iloc[1:]["benchmarkSymbol"]), {"SPY"})
        self.assertEqual(set(saved.iloc[1:]["benchmarkPrice"]), {625.0})
        self.assertEqual(
            set(saved.iloc[1:]["benchmarkQuoteTimeSource"]),
            {"yfinance.history.1m.index"},
        )
        self.assertEqual(
            set(saved.iloc[1:]["expiration"]),
            {"2026-08-21", "2026-09-18"},
        )
        self.assertTrue(saved.iloc[1:][["delta", "gamma", "theta", "vega"]].notna().all().all())

    @patch("trading_bot.market_data.collector.fetch_option_chain_snapshot")
    def test_skips_unchanged_raw_surface_but_retains_rate_change(self, fetch):
        fetch.return_value = OptionChainSnapshot(
            chains=((
                "2026-08-21",
                SimpleNamespace(
                    calls=pd.DataFrame([{
                        "contractSymbol": "AAPL-C1",
                        "strike": 200,
                        "bid": 4.9,
                        "ask": 5.1,
                        "impliedVolatility": 0.2,
                    }]),
                    puts=pd.DataFrame([{
                        "contractSymbol": "AAPL-P1",
                        "strike": 200,
                        "bid": 4.8,
                        "ask": 5.2,
                        "impliedVolatility": 0.2,
                    }]),
                ),
            ),),
            spot=200.0,
            dividend_yield=0.005,
            market_state="REGULAR",
        )
        first = datetime(2026, 7, 21, 20, 0, tzinfo=timezone.utc)
        second = datetime(2026, 7, 21, 20, 15, tzinfo=timezone.utc)
        third = datetime(2026, 7, 21, 20, 30, tzinfo=timezone.utc)

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            path, first_rows = collector.save_snapshot(
                "AAPL", output_dir, 0.05, first, expiration_count=1
            )
            _, unchanged_rows = collector.save_snapshot(
                "AAPL", output_dir, 0.05, second, expiration_count=1
            )
            _, changed_rows = collector.save_snapshot(
                "AAPL", output_dir, 0.051, second, expiration_count=1
            )
            fetch.return_value = replace(fetch.return_value, market_state="CLOSED")
            _, session_rows = collector.save_snapshot(
                "AAPL", output_dir, 0.051, third, expiration_count=1
            )
            saved = pd.read_csv(path)
            state = collector._snapshot_state_path(output_dir, "AAPL")
            self.assertTrue(state.exists())

        self.assertEqual(
            (first_rows, unchanged_rows, changed_rows, session_rows),
            (2, 0, 2, 2),
        )
        self.assertEqual(len(saved), 6)
        self.assertEqual(saved["collectedAt"].nunique(), 3)
        self.assertEqual(saved["marketState"].iloc[-1], "CLOSED")

    def test_collector_lock_rejects_a_second_instance(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with collector.collector_lock(output_dir):
                with self.assertRaisesRegex(RuntimeError, "another collector"):
                    with collector.collector_lock(output_dir):
                        self.fail("second lock unexpectedly acquired")

    @patch("trading_bot.market_data.collector.fetch_option_chain_snapshot")
    def test_benchmark_change_is_a_material_snapshot(self, fetch):
        fetch.return_value = OptionChainSnapshot(
            chains=((
                "2026-08-21",
                SimpleNamespace(
                    calls=pd.DataFrame([{
                        "contractSymbol": "AAPL-C1",
                        "strike": 200,
                        "bid": 4.9,
                        "ask": 5.1,
                        "impliedVolatility": 0.2,
                    }]),
                    puts=pd.DataFrame([{
                        "contractSymbol": "AAPL-P1",
                        "strike": 200,
                        "bid": 4.8,
                        "ask": 5.2,
                        "impliedVolatility": 0.2,
                    }]),
                ),
            ),),
            spot=200.0,
            dividend_yield=0.005,
            market_state="REGULAR",
        )
        first = BenchmarkSnapshot(
            "SPY", 625.0, "history", "2026-07-21T20:00:00Z", "index"
        )
        second = replace(
            first,
            price=626.0,
            quote_time="2026-07-21T20:15:00Z",
        )

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            path, first_rows = collector.save_snapshot(
                "AAPL",
                output_dir,
                0.05,
                datetime(2026, 7, 21, 20, 1, tzinfo=timezone.utc),
                expiration_count=1,
                benchmark_snapshot=first,
            )
            _, second_rows = collector.save_snapshot(
                "AAPL",
                output_dir,
                0.05,
                datetime(2026, 7, 21, 20, 16, tzinfo=timezone.utc),
                expiration_count=1,
                benchmark_snapshot=second,
            )
            saved = pd.read_csv(path)

        self.assertEqual((first_rows, second_rows), (2, 2))
        self.assertEqual(saved["benchmarkPrice"].tolist(), [625.0] * 2 + [626.0] * 2)

    @patch("trading_bot.market_data.collector.fetch_risk_free_rate")
    def test_cycle_status_records_rate_failure(self, fetch_rate):
        fetch_rate.side_effect = RuntimeError("rate unavailable")
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result = collector.collect_cycle(output_dir, ticker_delay=0)
            status = json.loads(
                (output_dir / collector.COLLECTOR_STATUS_FILENAME).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result.failures, 50)
        self.assertEqual(result.errors, {"risk_free_rate": "rate unavailable"})
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["failures"], 50)

    @patch("trading_bot.market_data.collector.save_snapshot")
    @patch("trading_bot.market_data.collector.fetch_benchmark_snapshot")
    @patch("trading_bot.market_data.collector.fetch_risk_free_rate")
    def test_cycle_keeps_collecting_when_optional_benchmark_fails(
        self,
        fetch_rate,
        fetch_benchmark,
        save,
    ):
        fetch_rate.return_value = 0.05
        fetch_benchmark.side_effect = RuntimeError("benchmark unavailable")
        save.return_value = (Path("data/AAPL.csv"), 2)

        with TemporaryDirectory() as directory, patch.object(
            collector,
            "TOP_50_TICKERS",
            ("AAPL",),
        ):
            output_dir = Path(directory)
            result = collector.collect_cycle(output_dir, ticker_delay=0)

        self.assertEqual(result.successes, 1)
        self.assertEqual(result.failures, 0)
        self.assertEqual(
            result.errors,
            {"benchmark": "benchmark unavailable"},
        )
        save.assert_called_once_with(
            "AAPL",
            output_dir,
            0.05,
            expiration_count=3,
            benchmark_snapshot=None,
        )
