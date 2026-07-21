import math
from datetime import datetime, timezone
from unittest import TestCase

from trading_bot.analytics.greeks import black_scholes_greeks, years_to_expiration


class GreeksTests(TestCase):
    def test_known_call_and_put_values(self):
        call = black_scholes_greeks("call", 100, 100, 1, 0.05, 0.20, 0.02)
        put = black_scholes_greeks("put", 100, 100, 1, 0.05, 0.20, 0.02)

        self.assertAlmostEqual(call["delta"], 0.5868511461, places=9)
        self.assertAlmostEqual(put["delta"], -0.3933475272, places=9)
        self.assertAlmostEqual(call["gamma"], 0.01895057876, places=9)
        self.assertAlmostEqual(call["theta"], -0.01394333949, places=9)
        self.assertAlmostEqual(put["theta"], -0.006283751063, places=9)
        self.assertAlmostEqual(call["vega"], 0.3790115751, places=9)

    def test_invalid_time_returns_nan_values(self):
        values = black_scholes_greeks("call", 100, 100, 0, 0.05, 0.20)
        self.assertTrue(all(math.isnan(value) for value in values.values()))

    def test_expiration_uses_new_york_market_close(self):
        captured = datetime(2026, 7, 21, 20, 0, tzinfo=timezone.utc)
        years = years_to_expiration("2026-07-22", captured)

        self.assertAlmostEqual(years, 1 / 365)
