from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from trading_bot.execution import PaperBroker, PaperTradeError
from trading_bot.execution.valuation import mark_positions


ORDER = {
    "contract_symbol": "AAPL260821C00330000",
    "symbol": "AAPL",
    "expiration": "2026-08-21",
    "option_type": "call",
    "strike": 330.0,
}


class PaperBrokerTests(TestCase):
    def test_buy_and_sell_update_cash_position_and_ledger(self):
        with TemporaryDirectory() as directory:
            broker = PaperBroker(Path(directory) / "portfolio.db", starting_cash=10_000)
            broker.buy(**ORDER, quantity=2, price=5.00)
            sell = broker.sell(**ORDER, quantity=1, price=6.00)

            account = broker.account()
            positions = broker.positions()
            trades = broker.trades()

        self.assertEqual(account["cash"], 9_600)
        self.assertEqual(positions[0]["quantity"], 1)
        self.assertEqual(sell["realized_pnl"], 100)
        self.assertEqual([trade["side"] for trade in trades], ["sell", "buy"])

    def test_rejects_overselling_and_insufficient_cash(self):
        with TemporaryDirectory() as directory:
            broker = PaperBroker(Path(directory) / "portfolio.db", starting_cash=100)
            with self.assertRaisesRegex(PaperTradeError, "insufficient cash"):
                broker.buy(**ORDER, quantity=1, price=5.00)
            with self.assertRaisesRegex(PaperTradeError, "insufficient position"):
                broker.sell(**ORDER, quantity=1, price=5.00)

    def test_marks_position_at_quote_midpoint(self):
        positions = [{**ORDER, "quantity": 2, "average_price": 5.0}]
        quotes = pd.DataFrame(
            [{"contractSymbol": ORDER["contract_symbol"], "bid": 5.5, "ask": 6.5}]
        )

        marked = mark_positions(positions, quotes)

        self.assertEqual(marked.iloc[0]["mark_price"], 6.0)
        self.assertEqual(marked.iloc[0]["market_value"], 1_200)
        self.assertEqual(marked.iloc[0]["unrealized_pnl"], 200)
