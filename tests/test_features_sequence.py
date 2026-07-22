from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from trading_bot.training.dataset import SnapshotDataset
from trading_bot.training.features import (
    ENGINEERED_FEATURES,
    MARKET_ENGINEERED_FEATURES,
    engineer_snapshot,
    realized_volatility_features,
    snapshot_gap_features,
)
from trading_bot.training.env import CONTRACT_FEATURES, MARKET_FEATURES, OptionsEnv
from trading_bot.training.schemas import FEATURE_VECTOR_SCHEMA_VERSION, Observation
from trading_bot.training.sequence import (
    AUXILIARY_TARGET_FEATURES,
    FEATURE_ABLATION_GROUPS,
    auxiliary_market_change_targets,
    build_windows,
    feature_ablation_indices,
    multi_horizon_auxiliary_targets,
    observation_vector,
)


def surface_snapshot(
    timestamp: str,
    atm_levels: tuple[float, float, float],
    *,
    front_call_wing: float,
    front_put_wing: float,
) -> pd.DataFrame:
    rows = []
    expirations = ("2026-08-21", "2026-10-16", "2027-01-15")
    for expiration, atm_level in zip(expirations, atm_levels, strict=True):
        for side, delta, volatility in (
            ("call", 0.5, atm_level - 0.01),
            ("put", -0.5, atm_level + 0.01),
        ):
            rows.append({
                "collectedAt": timestamp,
                "contractSymbol": f"{expiration}-ATM-{side}",
                "expiration": expiration,
                "optionType": side,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "strike": 100,
                "underlyingPrice": 100,
                "riskFreeRate": 0.0,
                "impliedVolatility": volatility,
                "delta": delta,
                "gamma": 0.01,
                "theta": -0.1,
                "vega": 0.2,
            })
        if expiration == expirations[0]:
            for side, strike, delta, volatility in (
                ("call", 110, 0.25, front_call_wing),
                ("put", 90, -0.25, front_put_wing),
            ):
                rows.append({
                    **rows[-1],
                    "contractSymbol": f"{expiration}-WING-{side}",
                    "optionType": side,
                    "strike": strike,
                    "delta": delta,
                    "impliedVolatility": volatility,
                })
    return pd.DataFrame(rows)


class FeatureSequenceTests(TestCase):
    def test_dataset_drops_consecutive_stale_quote_surfaces(self):
        base = {
            "contractSymbol": "AAPL-C1",
            "symbol": "AAPL",
            "expiration": "2026-08-21",
            "optionType": "call",
            "strike": 100,
            "bid": 1.0,
            "ask": 1.2,
            "lastPrice": 1.1,
            "impliedVolatility": 0.2,
            "underlyingPrice": 100,
            "riskFreeRate": 0.04,
            "greekModel": "black-scholes-merton",
            "delta": 0.5,
            "gamma": 0.01,
            "theta": -0.1,
            "vega": 0.2,
        }
        rows = [
            {**base, "collectedAt": "2026-07-21T14:00:00Z"},
            {
                **base,
                "collectedAt": "2026-07-21T14:01:00Z",
                "theta": -0.11,
                "delta": 0.51,
            },
            {
                **base,
                "collectedAt": "2026-07-21T14:02:00Z",
                "underlyingPrice": 101,
                "bid": 1.1,
            },
        ]

        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            pd.DataFrame(rows).to_csv(data_dir / "AAPL.csv", index=False)
            dataset = SnapshotDataset.from_directory(data_dir, "AAPL")

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            [snapshot.timestamp for snapshot in dataset.snapshots],
            [
                "2026-07-21T14:00:00+00:00",
                "2026-07-21T14:02:00+00:00",
            ],
        )
        self.assertEqual(
            dataset.snapshots[-1].frame.iloc[0]["snapshotGapSeconds"],
            120,
        )

    @staticmethod
    def auxiliary_observation(
        timestamp: str,
        *,
        spot: float,
        atm: float,
        risk_reversal: float,
        butterfly: float,
        term_slope: float,
        wing_coverage: float = 1.0,
    ) -> Observation:
        market = np.zeros(len(MARKET_FEATURES), dtype=float)
        for name, value in {
            "underlyingPrice": spot,
            "frontAtmIv": atm,
            "frontAtmIvCoverage": 1.0,
            "front25DeltaRiskReversal": risk_reversal,
            "front25DeltaButterfly": butterfly,
            "front25DeltaCoverage": wing_coverage,
            "atmTermStructureSlope": term_slope,
            "atmTermSlopeCoverage": 1.0,
        }.items():
            market[MARKET_FEATURES.index(name)] = value
        return Observation(
            timestamp,
            market,
            np.zeros((1, len(CONTRACT_FEATURES))),
            np.zeros(8),
            np.ones(1, dtype=bool),
            np.ones((2, 3), dtype=bool),
            ("C1",),
        )

    def test_auxiliary_market_targets_use_endpoint_scaling_and_coverage(self):
        current = self.auxiliary_observation(
            "current",
            spot=100,
            atm=0.20,
            risk_reversal=-0.02,
            butterfly=0.01,
            term_slope=-0.03,
        )
        future = self.auxiliary_observation(
            "future",
            spot=101,
            atm=0.23,
            risk_reversal=0.0,
            butterfly=0.0,
            term_slope=-0.13,
            wing_coverage=0.5,
        )

        values, available = auxiliary_market_change_targets(current, future)

        self.assertEqual(len(values), len(AUXILIARY_TARGET_FEATURES))
        np.testing.assert_allclose(available, (1, 1, 0, 0, 1))
        self.assertAlmostEqual(values[0], np.log1p(1.0))
        self.assertAlmostEqual(values[1], np.log1p(0.03))
        self.assertAlmostEqual(values[-1], -np.log1p(0.1))

    def test_multi_horizon_auxiliary_targets_are_cumulative_and_tail_masked(self):
        observations = [
            self.auxiliary_observation(
                str(index),
                spot=100 + 2 * index,
                atm=0.20 + 0.01 * index,
                risk_reversal=-0.02 + 0.005 * index,
                butterfly=0.01 + 0.002 * index,
                term_slope=-0.03 + 0.004 * index,
                wing_coverage=0.5 if index == 2 else 1.0,
            )
            for index in range(5)
        ]

        values, available = multi_horizon_auxiliary_targets(
            observations,
            (1, 4),
        )
        direct, direct_mask = auxiliary_market_change_targets(
            observations[0],
            observations[4],
        )

        self.assertEqual(values.shape, (4, 2 * len(AUXILIARY_TARGET_FEATURES)))
        np.testing.assert_allclose(values[0, 5:], direct)
        np.testing.assert_allclose(available[0, 5:], direct_mask)
        np.testing.assert_allclose(available[1:, 5:], 0.0)
        self.assertAlmostEqual(
            values[0, 5],
            np.log1p(abs(108 / 100 - 1) * 100),
        )
        # Both endpoints have full wing coverage for horizon four, while the
        # one-step target ending at the sparse middle snapshot is unavailable.
        self.assertEqual(available[0, 7], 1.0)
        self.assertEqual(available[1, 2], 0.0)

        for invalid in ((), (0,), (2, 1), (1, 1)):
            with self.assertRaisesRegex(ValueError, "horizons"):
                multi_horizon_auxiliary_targets(observations, invalid)

    def test_feature_ablation_groups_map_to_stable_non_overlapping_inputs(self):
        wing_indices = feature_ablation_indices(("surface_wings",), 2)
        quality_indices = feature_ablation_indices(("data_quality",), 2)
        contract_indices = feature_ablation_indices(
            ("derived_contract_surface",),
            2,
        )
        term_indices = feature_ablation_indices(("term_structure",), 2)
        dynamics_indices = feature_ablation_indices(("surface_dynamics",), 2)
        identity_indices = feature_ablation_indices(("slot_identity",), 2)
        position_indices = feature_ablation_indices(("position_state",), 2)
        time_indices = feature_ablation_indices(("time_context",), 2)
        trend_indices = feature_ablation_indices(("price_trend",), 2)
        volatility_indices = feature_ablation_indices(("volatility_regime",), 2)

        self.assertEqual(len(wing_indices), 3)
        self.assertEqual(len(quality_indices), 2)
        self.assertEqual(len(term_indices), 4)
        self.assertEqual(len(dynamics_indices), 7)
        self.assertEqual(len(identity_indices), 2)
        self.assertEqual(len(position_indices), 6)
        self.assertEqual(len(time_indices), 2)
        self.assertEqual(len(trend_indices), 2)
        self.assertEqual(
            len(contract_indices),
            2 * len(FEATURE_ABLATION_GROUPS["derived_contract_surface"]),
        )
        self.assertFalse(set(wing_indices) & set(quality_indices))
        self.assertFalse(set(term_indices) & set(dynamics_indices))
        self.assertFalse(set(time_indices) & set(dynamics_indices))
        self.assertFalse(set(trend_indices) & set(time_indices))
        self.assertFalse(set(trend_indices) & set(term_indices))
        self.assertFalse(set(position_indices) & set(identity_indices))
        for window in (4, 16):
            coverage_index = MARKET_FEATURES.index(f"realizedVol{window}Coverage")
            self.assertNotIn(coverage_index, trend_indices)
            self.assertNotIn(coverage_index, volatility_indices)
        self.assertTrue(all(index < len(MARKET_FEATURES) for index in time_indices))
        self.assertTrue(all(index < len(MARKET_FEATURES) for index in trend_indices))
        self.assertTrue(all(index < len(MARKET_FEATURES) for index in wing_indices))
        self.assertTrue(all(index >= len(MARKET_FEATURES) for index in contract_indices))
        self.assertTrue(all(index >= len(MARKET_FEATURES) for index in identity_indices))
        seen: set[int] = set()
        for group in FEATURE_ABLATION_GROUPS:
            indices = set(feature_ablation_indices((group,), 2))
            self.assertFalse(seen & indices, group)
            seen.update(indices)
        with self.assertRaisesRegex(ValueError, "unknown"):
            feature_ablation_indices(("future_leak",), 2)

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
        self.assertEqual(engineered.iloc[0]["snapshotGapSeconds"], 60)
        self.assertEqual(engineered.iloc[0]["snapshotGapCoverage"], 1)
        self.assertTrue(np.isfinite(engineered[list(ENGINEERED_FEATURES)].to_numpy()).all())

    def test_snapshot_gap_requires_a_valid_positive_causal_interval(self):
        current = pd.DataFrame({"collectedAt": ["2026-07-21T14:15:00Z"]})
        previous = pd.DataFrame({"collectedAt": ["2026-07-21T14:00:00Z"]})

        self.assertEqual(
            snapshot_gap_features(current, previous),
            {"snapshotGapSeconds": 900.0, "snapshotGapCoverage": 1.0},
        )
        for invalid_previous in (
            None,
            pd.DataFrame(),
            pd.DataFrame({"collectedAt": ["not-a-time"]}),
            pd.DataFrame({"other": [1]}),
            current,
            pd.DataFrame({"collectedAt": ["2026-07-21T14:30:00Z"]}),
        ):
            self.assertEqual(
                snapshot_gap_features(current, invalid_previous),
                {"snapshotGapSeconds": 0.0, "snapshotGapCoverage": 0.0},
            )
        self.assertEqual(
            snapshot_gap_features(pd.DataFrame({"other": [1]}), previous),
            {"snapshotGapSeconds": 0.0, "snapshotGapCoverage": 0.0},
        )

    def test_price_history_trend_uses_only_valid_observed_intervals(self):
        start = pd.Timestamp("2026-07-21T14:00:00Z")
        history = tuple(zip(
            (start + pd.Timedelta(minutes=15 * index) for index in range(5)),
            (100.0, 101.0, np.nan, 102.0, 103.0),
            strict=True,
        ))

        features = realized_volatility_features(history)

        self.assertEqual(features["realizedVol4Coverage"], 0.5)
        self.assertEqual(features["realizedVol16Coverage"], 0.125)
        self.assertAlmostEqual(
            features["underlyingLogReturn4"],
            np.log(101 / 100) + np.log(103 / 102),
        )
        self.assertEqual(
            features["underlyingLogReturn4"],
            features["underlyingLogReturn16"],
        )
        self.assertGreater(features["realizedVol4"], 0)

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
        self.assertTrue(np.allclose(engineered["frontAtmIv"], 0.21))
        self.assertTrue(np.allclose(engineered["frontAtmIvCoverage"], 1))
        self.assertTrue(np.allclose(engineered["front25DeltaCoverage"], 0))
        self.assertTrue(np.allclose(engineered["executableQuoteCoverage"], 1))
        self.assertTrue(np.allclose(engineered["greekCoverage"], 0))
        self.assertTrue(np.allclose(engineered["atmIvMinusRealizedVol4"], 0))
        self.assertEqual(front["parityResidual"].nunique(), 1)
        self.assertTrue((engineered["extrinsicValuePct"] >= 0).all())
        self.assertTrue(np.isfinite(engineered[list(ENGINEERED_FEATURES)]).all().all())

    def test_term_structure_and_surface_changes_require_explicit_coverage(self):
        previous = engineer_snapshot(surface_snapshot(
            "2026-07-21T14:00:00Z",
            (0.20, 0.24, 0.31),
            front_call_wing=0.24,
            front_put_wing=0.28,
        ))
        current = engineer_snapshot(
            surface_snapshot(
                "2026-07-21T14:01:00Z",
                (0.22, 0.255, 0.34),
                front_call_wing=0.27,
                front_put_wing=0.33,
            ),
            previous,
        )
        prior = previous.iloc[0]
        now = current.iloc[0]

        self.assertEqual(prior["atmTermSlopeCoverage"], 1)
        self.assertEqual(prior["atmTermCurvatureCoverage"], 1)
        self.assertNotEqual(prior["atmTermStructureSlope"], 0)
        self.assertNotEqual(prior["atmTermStructureCurvature"], 0)
        self.assertEqual(prior["frontAtmIvChangeCoverage"], 0)
        self.assertEqual(prior["frontWingChangeCoverage"], 0)
        self.assertEqual(prior["atmTermSlopeChangeCoverage"], 0)
        self.assertAlmostEqual(now["frontAtmIvChange"], 0.02)
        self.assertEqual(now["frontAtmIvChangeCoverage"], 1)
        self.assertAlmostEqual(
            now["front25DeltaRiskReversalChange"],
            -0.02,
        )
        self.assertAlmostEqual(
            now["front25DeltaButterflyChange"],
            0.02,
        )
        self.assertEqual(now["frontWingChangeCoverage"], 1)
        self.assertAlmostEqual(
            now["atmTermStructureSlopeChange"],
            now["atmTermStructureSlope"] - prior["atmTermStructureSlope"],
        )
        self.assertEqual(now["atmTermSlopeChangeCoverage"], 1)
        self.assertTrue(
            np.isfinite(current[list(MARKET_ENGINEERED_FEATURES)]).all().all()
        )

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
                "positionQuantity": 2,
                "positionAveragePrice": 1.2 * scale,
                "positionUnrealizedReturn": 0.25,
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
                    10,
                ]),
                np.ones(1, dtype=bool),
                np.ones((1, 3), dtype=bool),
                ("C1",),
            )

        first = observation_vector(make_observation(1.0))
        second = observation_vector(make_observation(2.0))
        contract_end = 2 + len(CONTRACT_FEATURES)

        np.testing.assert_allclose(first[:contract_end], second[:contract_end])
        np.testing.assert_allclose(
            first[contract_end:contract_end + 8],
            second[contract_end:contract_end + 8],
        )
        self.assertLessEqual(float(np.abs(first).max()), 10.0)
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(FEATURE_VECTOR_SCHEMA_VERSION, "dimensionless.v10")
        self.assertNotIn("volume", CONTRACT_FEATURES)
        self.assertNotIn("openInterest", CONTRACT_FEATURES)
        self.assertIn("volumeLog", CONTRACT_FEATURES)

    def test_volatility_regime_market_features_have_fixed_transform(self):
        market = np.zeros(len(MARKET_FEATURES), dtype=float)
        market[MARKET_FEATURES.index("underlyingPrice")] = 100
        market[MARKET_FEATURES.index("riskFreeRate")] = 0.04
        market[MARKET_FEATURES.index("snapshotGapSeconds")] = 900
        market[MARKET_FEATURES.index("snapshotGapCoverage")] = 1
        market[MARKET_FEATURES.index("realizedVol4")] = 0.2
        market[MARKET_FEATURES.index("underlyingLogReturn4")] = 0.01
        market[MARKET_FEATURES.index("underlyingLogReturn16")] = -0.02
        market[MARKET_FEATURES.index("frontAtmIv")] = 0.25
        market[MARKET_FEATURES.index("frontAtmIvCoverage")] = 1
        market[MARKET_FEATURES.index("front25DeltaRiskReversal")] = -0.04
        market[MARKET_FEATURES.index("front25DeltaButterfly")] = 0.05
        market[MARKET_FEATURES.index("front25DeltaCoverage")] = 1
        market[MARKET_FEATURES.index("atmTermStructureSlope")] = 0.2
        market[MARKET_FEATURES.index("atmTermStructureCurvature")] = -0.1
        market[MARKET_FEATURES.index("frontAtmIvChange")] = -0.03
        market[
            MARKET_FEATURES.index("front25DeltaRiskReversalChange")
        ] = 0.02
        market[MARKET_FEATURES.index("atmTermSlopeChangeCoverage")] = 1
        market[MARKET_FEATURES.index("executableQuoteCoverage")] = 0.8
        market[MARKET_FEATURES.index("greekCoverage")] = 0.75
        market[MARKET_FEATURES.index("atmIvMinusRealizedVol4")] = 0.05
        observation = Observation(
            "now",
            market,
            np.zeros((1, len(CONTRACT_FEATURES))),
            np.array([100_000, 0, 100_000, 0, 0, 0, 0, 0]),
            np.ones(1, dtype=bool),
            np.ones((2, 3), dtype=bool),
            ("C1",),
        )

        vector = observation_vector(observation)

        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("snapshotGapSeconds")],
            np.log1p(900) / 10,
        )
        self.assertEqual(
            vector[MARKET_FEATURES.index("snapshotGapCoverage")],
            1,
        )
        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("underlyingLogReturn4")],
            np.log1p(1.0),
        )
        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("underlyingLogReturn16")],
            -np.log1p(2.0),
        )

        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("frontAtmIv")],
            np.log1p(0.25),
        )
        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("atmIvMinusRealizedVol4")],
            np.log1p(0.05),
        )
        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("front25DeltaRiskReversal")],
            -np.log1p(0.04),
        )
        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("front25DeltaButterfly")],
            np.log1p(0.05),
        )
        self.assertEqual(
            vector[MARKET_FEATURES.index("front25DeltaCoverage")],
            1,
        )
        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("atmTermStructureSlope")],
            np.log1p(0.2),
        )
        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("atmTermStructureCurvature")],
            -np.log1p(0.1),
        )
        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("frontAtmIvChange")],
            -np.log1p(0.03),
        )
        self.assertAlmostEqual(
            vector[MARKET_FEATURES.index("executableQuoteCoverage")],
            0.8,
        )
        self.assertTrue(np.isfinite(vector).all())

    def test_front_wing_surface_factors_exclude_unexecutable_quotes(self):
        rows = []
        for symbol, side, strike, delta, volatility, bid, vega in (
            ("ATM-C", "call", 100, 0.50, 0.20, 1.0, 0.2),
            ("WING-C", "call", 110, 0.25, 0.24, 1.0, 0.2),
            ("BAD-C", "call", 111, 0.249, 0.99, 0.0, np.nan),
            ("ATM-P", "put", 100, -0.50, 0.22, 1.0, 0.2),
            ("WING-P", "put", 90, -0.25, 0.28, 1.0, 0.2),
        ):
            rows.append({
                "collectedAt": "2026-07-21T14:00:00Z",
                "contractSymbol": symbol,
                "expiration": "2026-08-21",
                "optionType": side,
                "bid": bid,
                "ask": 1.2,
                "lastPrice": 1.1,
                "strike": strike,
                "underlyingPrice": 100,
                "riskFreeRate": 0.0,
                "dividendYield": 0.0,
                "timeToExpiryYears": 31 / 365,
                "impliedVolatility": volatility,
                "delta": delta,
                "gamma": 0.01,
                "theta": -0.1,
                "vega": vega,
            })

        engineered = engineer_snapshot(pd.DataFrame(rows))
        first = engineered.iloc[0]

        self.assertAlmostEqual(first["frontAtmIv"], 0.21)
        self.assertEqual(first["frontAtmIvCoverage"], 1)
        self.assertAlmostEqual(first["front25DeltaRiskReversal"], -0.04)
        self.assertAlmostEqual(first["front25DeltaButterfly"], 0.05)
        self.assertEqual(first["front25DeltaCoverage"], 1)
        self.assertAlmostEqual(first["executableQuoteCoverage"], 0.8)
        self.assertAlmostEqual(first["greekCoverage"], 0.8)

    def test_sparse_atm_only_surface_does_not_masquerade_as_wings(self):
        rows = []
        for symbol, side, delta, volatility in (
            ("ATM-C", "call", 0.50, 0.20),
            ("ATM-P", "put", -0.50, 0.22),
        ):
            rows.append({
                "collectedAt": "2026-07-21T14:00:00Z",
                "contractSymbol": symbol,
                "expiration": "2026-08-21",
                "optionType": side,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "strike": 100,
                "underlyingPrice": 100,
                "impliedVolatility": volatility,
                "delta": delta,
            })

        engineered = engineer_snapshot(pd.DataFrame(rows))

        self.assertTrue((engineered["frontAtmIvCoverage"] == 1).all())
        self.assertTrue((engineered["front25DeltaCoverage"] == 0).all())
        self.assertTrue((engineered["front25DeltaRiskReversal"] == 0).all())
        self.assertTrue((engineered["front25DeltaButterfly"] == 0).all())

    def test_realized_volatility_is_backward_only_with_explicit_coverage(self):
        rows = []
        for index in range(18):
            rows.append({
                "collectedAt": pd.Timestamp("2026-07-21T14:00:00Z")
                + pd.Timedelta(minutes=15 * index),
                "contractSymbol": "TEST-C",
                "symbol": "TEST",
                "expiration": "2026-09-18",
                "optionType": "call",
                "strike": 100,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "impliedVolatility": 0.2,
                "underlyingPrice": 100 * 1.001**index,
                "riskFreeRate": 0.04,
                "greekModel": "black-scholes-merton",
            })
        rows[-1]["underlyingPrice"] = 1_000  # future shock

        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            pd.DataFrame(rows).to_csv(data_dir / "FULL.csv", index=False)
            pd.DataFrame(rows[:-1]).to_csv(data_dir / "PREFIX.csv", index=False)
            full = SnapshotDataset.from_directory(data_dir, "FULL")
            prefix = SnapshotDataset.from_directory(data_dir, "PREFIX")

        for index in range(len(prefix)):
            for name in MARKET_ENGINEERED_FEATURES:
                self.assertAlmostEqual(
                    full.snapshots[index].frame.iloc[0][name],
                    prefix.snapshots[index].frame.iloc[0][name],
                )
        self.assertEqual(full.snapshots[0].frame.iloc[0]["realizedVol4Coverage"], 0)
        self.assertEqual(full.snapshots[0].frame.iloc[0]["underlyingLogReturn4"], 0)
        self.assertEqual(full.snapshots[4].frame.iloc[0]["realizedVol4Coverage"], 1)
        self.assertEqual(full.snapshots[16].frame.iloc[0]["realizedVol16Coverage"], 1)
        self.assertGreater(full.snapshots[16].frame.iloc[0]["realizedVol16"], 0)
        self.assertAlmostEqual(
            full.snapshots[4].frame.iloc[0]["underlyingLogReturn4"],
            4 * np.log(1.001),
        )
        self.assertAlmostEqual(
            full.snapshots[16].frame.iloc[0]["underlyingLogReturn16"],
            16 * np.log(1.001),
        )
        fourth = full.snapshots[4].frame.iloc[0]
        self.assertAlmostEqual(
            fourth["atmIvMinusRealizedVol4"],
            fourth["frontAtmIv"] - fourth["realizedVol4"],
        )

        env = OptionsEnv(full, slot_count=1)
        observation, info = env.reset(options={"start_index": 16})
        self.assertEqual(observation.market.size, len(MARKET_FEATURES))
        self.assertEqual(info["market_features"], MARKET_FEATURES)
        self.assertNotIn("underlyingReturn", CONTRACT_FEATURES)
        self.assertNotIn("frontAtmIv", CONTRACT_FEATURES)
        self.assertIn("atmIvMinusRealizedVol16", MARKET_FEATURES)
        self.assertIn("front25DeltaRiskReversal", MARKET_FEATURES)
        self.assertIn("front25DeltaButterfly", MARKET_FEATURES)
        self.assertIn("snapshotGapSeconds", MARKET_FEATURES)
        self.assertIn("snapshotGapCoverage", MARKET_FEATURES)
        self.assertIn("underlyingLogReturn4", MARKET_FEATURES)
        self.assertIn("underlyingLogReturn16", MARKET_FEATURES)
        self.assertIn("executableQuoteCoverage", MARKET_FEATURES)
        self.assertTrue(np.isfinite(observation_vector(observation)).all())
