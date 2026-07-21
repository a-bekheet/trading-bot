from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from trading_bot.market_data.option_chain import fetch_option_chain


class FakeTicker:
    options = ("2026-08-21", "2026-09-18")
    fast_info = {}

    def __init__(self, symbol):
        self.symbol = symbol

    def option_chain(self, expiration):
        return SimpleNamespace(
            calls="calls",
            puts="puts",
            underlying={"regularMarketPrice": 200, "dividendYield": 0.5},
        )


class FetchOptionChainTests(TestCase):
    @patch("trading_bot.market_data.option_chain.yf.Ticker", FakeTicker)
    def test_uses_nearest_expiration_and_returns_market_inputs(self):
        expiration, chain, spot, dividend_yield = fetch_option_chain(" aapl ")

        self.assertEqual(expiration, "2026-08-21")
        self.assertEqual(chain.calls, "calls")
        self.assertEqual(spot, 200)
        self.assertEqual(dividend_yield, 0.005)

    @patch("trading_bot.market_data.option_chain.yf.Ticker")
    def test_rejects_ticker_without_options(self, ticker):
        ticker.return_value.options = ()

        with self.assertRaisesRegex(ValueError, "No listed options"):
            fetch_option_chain("INVALID")
