from unittest import TestCase

import numpy as np
import pandas as pd

from trading_bot.training.features import ENGINEERED_FEATURES, engineer_snapshot
from trading_bot.training.schemas import Observation
from trading_bot.training.sequence import build_windows, observation_vector


class FeatureSequenceTests(TestCase):
    def test_features_are_finite_and_use_previous_snapshot_only(self):
        previous = pd.DataFrame([{
            "collectedAt": "2026-07-21T14:00:00Z", "contractSymbol": "C1",
            "expiration": "2026-08-21", "bid": 1, "ask": 1.2,
            "lastPrice": 1.1, "strike": 100, "underlyingPrice": 100,
            "impliedVolatility": .2, "volume": 10, "openInterest": 20,
            "lastTradeDate": "2026-07-21T13:59:00Z",
        }])
        current = previous.copy()
        current["collectedAt"] = "2026-07-21T14:01:00Z"
        current["underlyingPrice"] = 101
        current["impliedVolatility"] = .25

        engineered = engineer_snapshot(current, previous)

        self.assertEqual(set(ENGINEERED_FEATURES) - set(engineered.columns), set())
        self.assertAlmostEqual(engineered.iloc[0]["underlyingReturn"], .01)
        self.assertAlmostEqual(engineered.iloc[0]["ivChange"], .05)
        self.assertTrue(np.isfinite(engineered[list(ENGINEERED_FEATURES)].to_numpy()).all())

    def test_sequence_windows_are_chronological_and_fixed_shape(self):
        observations = [
            Observation(str(i), np.ones(2) * i, np.ones((2, 3)) * i, np.ones(3), np.ones(2, bool), np.ones((2, 2), bool), ("a", "b"))
            for i in range(3)
        ]
        windows = build_windows(observations, window=2)

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].features.shape, (2, observation_vector(observations[0]).size))
        self.assertEqual(windows[0].features[0, 0], 0)
        self.assertEqual(windows[1].features[0, 0], 1)
