from unittest import TestCase

import numpy as np
import pandas as pd

from trading_bot.training.features import ENGINEERED_FEATURES, engineer_snapshot
from trading_bot.training.env import CONTRACT_FEATURES
from trading_bot.training.schemas import FEATURE_VECTOR_SCHEMA_VERSION, Observation
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

    def test_surface_features_link_strikes_sides_and_expirations(self):
        rows = []
        for expiration, years, call_iv, put_iv in (
            ("2026-08-21", 0.1, 0.20, 0.22),
            ("2026-11-20", 0.35, 0.25, 0.27),
        ):
            for option_type, volatility, mid in (
                ("call", call_iv, 5.5),
                ("put", put_iv, 4.8),
            ):
                rows.append({
                    "collectedAt": "2026-07-21T14:00:00Z",
                    "contractSymbol": f"{expiration}-{option_type}",
                    "expiration": expiration,
                    "optionType": option_type,
                    "bid": mid - 0.1,
                    "ask": mid + 0.1,
                    "lastPrice": mid,
                    "strike": 100,
                    "underlyingPrice": 101,
                    "riskFreeRate": 0.04,
                    "dividendYield": 0.01,
                    "timeToExpiryYears": years,
                    "impliedVolatility": volatility,
                    "volume": 10,
                    "openInterest": 20,
                    "lastTradeDate": "2026-07-21T13:59:00Z",
                })

        engineered = engineer_snapshot(pd.DataFrame(rows))
        front = engineered[engineered["expiration"].eq("2026-08-21")]
        back = engineered[engineered["expiration"].eq("2026-11-20")]

        self.assertTrue((back["atmTermSlope"] > 0).all())
        self.assertTrue((front["atmTermSlope"] == 0).all())
        self.assertTrue(np.allclose(engineered["ivSkew"], 0))
        self.assertTrue(np.allclose(engineered["putCallIvSpread"], -0.02))
        self.assertEqual(front["parityResidual"].nunique(), 1)
        self.assertTrue((engineered["extrinsicValuePct"] >= 0).all())
        self.assertTrue(np.isfinite(engineered[list(ENGINEERED_FEATURES)]).all().all())

    def test_dimensionless_vector_is_bounded_and_price_scale_invariant(self):
        def make_observation(scale: float) -> Observation:
            contracts = np.zeros((1, len(CONTRACT_FEATURES)), dtype=float)
            values = {
                "strike": 100 * scale,
                "lastPrice": 2 * scale,
                "bid": 1.9 * scale,
                "ask": 2.1 * scale,
                "impliedVolatility": 0.2,
                "delta": 0.5,
                "gamma": 0.01 / scale,
                "theta": -0.1 * scale,
                "vega": 0.2 * scale,
                "midPrice": 2 * scale,
                "spread": 0.2 * scale,
                "dteDays": 30,
                "volumeLog": 5,
                "openInterestLog": 6,
                "quoteAgeSeconds": 60,
            }
            for name, value in values.items():
                contracts[0, CONTRACT_FEATURES.index(name)] = value
            return Observation(
                "now",
                np.array([100 * scale, 0.04]),
                contracts,
                np.array([
                    80_000 * scale,
                    20_000 * scale,
                    100_000 * scale,
                    50,
                    1 / scale,
                    -10 * scale,
                    20 * scale,
                ]),
                np.ones(1, dtype=bool),
                np.ones((1, 3), dtype=bool),
                ("C1",),
            )

        first = observation_vector(make_observation(1.0))
        second = observation_vector(make_observation(2.0))
        contract_end = 2 + len(CONTRACT_FEATURES)

        np.testing.assert_allclose(first[:contract_end], second[:contract_end])
        np.testing.assert_allclose(first[contract_end:contract_end + 7], second[contract_end:contract_end + 7])
        self.assertLessEqual(float(np.abs(first).max()), 10.0)
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(FEATURE_VECTOR_SCHEMA_VERSION, "dimensionless.v1")
        self.assertNotIn("volume", CONTRACT_FEATURES)
        self.assertNotIn("openInterest", CONTRACT_FEATURES)
        self.assertIn("volumeLog", CONTRACT_FEATURES)
