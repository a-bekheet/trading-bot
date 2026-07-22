from unittest import TestCase

from trading_bot.market_data.freshness import (
    select_underlying_quote,
    underlying_quote_age,
)


class MarketFreshnessTests(TestCase):
    def test_selects_session_matched_price_and_timestamp(self):
        underlying = {
            "preMarketPrice": 101,
            "preMarketTime": 1_784_710_000,
            "regularMarketPrice": 99,
            "regularMarketTime": 1_784_700_000,
            "postMarketPrice": 100,
            "postMarketTime": 1_784_705_000,
        }

        pre = select_underlying_quote(underlying, "PRE")
        regular = select_underlying_quote(underlying, "REGULAR")

        self.assertEqual((pre.price, pre.price_source), (101, "preMarketPrice"))
        self.assertEqual(pre.quote_time_source, "preMarketTime")
        self.assertEqual(
            (regular.price, regular.price_source),
            (99, "regularMarketPrice"),
        )
        self.assertEqual(regular.quote_time_source, "regularMarketTime")

    def test_unknown_state_uses_latest_complete_pair(self):
        quote = select_underlying_quote({
            "regularMarketPrice": 99,
            "regularMarketTime": 100,
            "postMarketPrice": 100,
            "postMarketTime": 110,
        }, "NEW_PROVIDER_STATE")

        self.assertEqual(quote.price, 100)
        self.assertEqual(quote.quote_time_source, "postMarketTime")

    def test_preferred_price_without_time_is_not_mismatched(self):
        quote = select_underlying_quote({
            "preMarketPrice": 101,
            "regularMarketPrice": 99,
            "regularMarketTime": 100,
        }, "PRE")

        self.assertEqual(quote.price_source, "preMarketPrice")
        self.assertIsNone(quote.quote_time)
        self.assertIsNone(quote.quote_time_source)

    def test_missing_session_price_falls_back_to_a_complete_pair(self):
        quote = select_underlying_quote({
            "regularMarketPrice": 99,
            "regularMarketTime": 100,
        }, "PRE")

        self.assertEqual(quote.price_source, "regularMarketPrice")
        self.assertEqual(quote.quote_time_source, "regularMarketTime")

    def test_quote_age_rejects_missing_and_future_timestamps(self):
        self.assertEqual(underlying_quote_age("2026-07-22T10:00:00Z", None), (0, 0))
        self.assertEqual(
            underlying_quote_age(
                "2026-07-22T10:00:00Z",
                "2026-07-22T10:01:00Z",
            ),
            (0, 0),
        )
        self.assertEqual(
            underlying_quote_age(
                "2026-07-22T10:00:00Z",
                "2026-07-22T09:45:00Z",
            ),
            (900, 1),
        )
