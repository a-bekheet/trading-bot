from unittest import TestCase

from trading_bot.market_data.market_state import (
    market_state_features,
    normalize_market_state,
)


class MarketStateTests(TestCase):
    def test_normalizes_recognized_states_without_clock_inference(self):
        self.assertEqual(normalize_market_state(" regular "), "REGULAR")
        self.assertEqual(normalize_market_state("closed"), "CLOSED")
        self.assertEqual(market_state_features("REGULAR"), (1.0, 1.0))
        self.assertEqual(market_state_features("POST"), (0.0, 1.0))

    def test_unknown_provider_values_have_zero_coverage(self):
        self.assertEqual(normalize_market_state(None), "UNKNOWN")
        self.assertEqual(normalize_market_state(float("nan")), "UNKNOWN")
        self.assertEqual(market_state_features("new-provider-state"), (0.0, 0.0))
