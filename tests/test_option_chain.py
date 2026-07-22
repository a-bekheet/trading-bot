from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from trading_bot.market_data.option_chain import (
    fetch_option_chain,
    fetch_option_chain_snapshot,
    fetch_option_chains,
)


class FakeTicker:
    options = ("2026-08-21", "2026-09-18")
    fast_info = {}
    requests = []

    def __init__(self, symbol):
        self.symbol = symbol

    def option_chain(self, expiration):
        self.requests.append(expiration)
        return SimpleNamespace(
            calls=f"calls-{expiration}",
            puts=f"puts-{expiration}",
            underlying={
                "regularMarketPrice": 200,
                "dividendYield": 0.5,
                "marketState": "REGULAR",
            },
        )


class FetchOptionChainTests(TestCase):
    @patch("trading_bot.market_data.option_chain.yf.Ticker", FakeTicker)
    def test_uses_nearest_expiration_and_returns_market_inputs(self):
        FakeTicker.requests.clear()
        expiration, chain, spot, dividend_yield = fetch_option_chain(" aapl ")

        self.assertEqual(expiration, "2026-08-21")
        self.assertEqual(chain.calls, "calls-2026-08-21")
        self.assertEqual(spot, 200)
        self.assertEqual(dividend_yield, 0.005)
        self.assertEqual(FakeTicker.requests, ["2026-08-21"])

    @patch("trading_bot.market_data.option_chain.yf.Ticker", FakeTicker)
    def test_fetches_multiple_expirations_with_shared_market_inputs(self):
        FakeTicker.requests.clear()

        chains, spot, dividend_yield = fetch_option_chains("AAPL", expiration_count=2)

        self.assertEqual([expiration for expiration, _ in chains], list(FakeTicker.options))
        self.assertEqual(FakeTicker.requests, list(FakeTicker.options))
        self.assertEqual(spot, 200)
        self.assertEqual(dividend_yield, 0.005)

    @patch("trading_bot.market_data.option_chain.yf.Ticker", FakeTicker)
    def test_exposes_normalized_provider_market_state(self):
        snapshot = fetch_option_chain_snapshot("AAPL", expiration_count=2)

        self.assertEqual(snapshot.market_state, "REGULAR")
        self.assertEqual(snapshot.spot, 200)
        self.assertEqual(len(snapshot.chains), 2)

    def test_rejects_invalid_expiration_count_before_network_access(self):
        with self.assertRaisesRegex(ValueError, "expiration_count"):
            fetch_option_chains("AAPL", expiration_count=0)

    @patch("trading_bot.market_data.option_chain.yf.Ticker")
    def test_rejects_ticker_without_options(self, ticker):
        ticker.return_value.options = ()

        with self.assertRaisesRegex(ValueError, "No listed options"):
            fetch_option_chain("INVALID")
