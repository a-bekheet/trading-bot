from dataclasses import replace
from unittest import TestCase

import numpy as np
import pandas as pd

from trading_bot.training.baselines import (
    LongVolatilityConfig,
    ShortVolatilityConfig,
    UnderlyingTrendConfig,
    buy_first_then_delta_hedge,
    cash_secured_short_put_delta_hedge,
    delta_neutral,
    long_volatility_delta_hedge,
    underlying_trend,
)
from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.env import MARKET_FEATURES, OptionsEnv
from trading_bot.training.evaluation import (
    CostScenario,
    cost_stressed_environment,
    run_episode,
)
from tests.test_training_env import three_snapshot_dataset


class BaselineTests(TestCase):
    @staticmethod
    def _long_volatility_dataset() -> SnapshotDataset:
        snapshots = []
        for index in range(3):
            rows = []
            for option_type, delta in (("call", 0.6), ("put", -0.4)):
                rows.append({
                    "collectedAt": f"2026-07-21T14:0{index}:00Z",
                    "contractSymbol": f"TEST-{option_type}",
                    "symbol": "TEST",
                    "expiration": "2026-08-21",
                    "optionType": option_type,
                    "strike": 100,
                    "bid": 1.0 + index * 0.1,
                    "ask": 1.2 + index * 0.1,
                    "lastPrice": 1.1 + index * 0.1,
                    "impliedVolatility": 0.2,
                    "underlyingPrice": 100,
                    "riskFreeRate": 0.04,
                    "delta": delta,
                    "gamma": 0.01,
                    "theta": -0.1,
                    "vega": 0.2,
                    "dteDays": 30 - index / 1_440,
                    "logMoneyness": 0.0,
                    "spreadPct": 0.18,
                    "openInterestLog": 5.0,
                    "realizedVol4": 0.45,
                    "realizedVol4Coverage": 1.0,
                    "realizedVol16": 0.50,
                    "realizedVol16Coverage": 1.0,
                    "frontAtmIv": 0.20,
                    "frontAtmIvCoverage": 1.0,
                    "atmIvMinusRealizedVol4": -0.25,
                    "atmIvMinusRealizedVol16": -0.30,
                })
            frame = pd.DataFrame(rows)
            snapshots.append(Snapshot(str(index), frame))
        return SnapshotDataset(tuple(snapshots), "TEST")

    @classmethod
    def _short_volatility_dataset(cls) -> SnapshotDataset:
        source = cls._long_volatility_dataset()
        snapshots = []
        for snapshot in source.snapshots:
            frame = snapshot.frame.copy()
            frame["realizedVol4"] = 0.20
            frame["realizedVol16"] = 0.20
            frame["frontAtmIv"] = 0.50
            frame["atmIvMinusRealizedVol4"] = 0.30
            frame["atmIvMinusRealizedVol16"] = 0.30
            snapshots.append(Snapshot(snapshot.timestamp, frame))
        return SnapshotDataset(tuple(snapshots), source.symbol)

    def test_delta_neutral_uses_underlying_slot_to_offset_option_delta(self):
        env = OptionsEnv(
            three_snapshot_dataset(),
            slot_count=2,
            max_quantity=2,
            starting_cash=1_000,
            underlying_lot_size=25,
        )
        env.reset()
        observation, _, _, _, _ = env.step(np.array([1, 0]))

        action = delta_neutral(observation)

        np.testing.assert_array_equal(action, np.array([0, 0, 4]))
        hedged, _, _, truncated, info = env.step(action)
        self.assertTrue(truncated)
        self.assertAlmostEqual(hedged.portfolio[3], 0)
        self.assertEqual(info["executions"][0]["instrument"], "underlying")
        self.assertEqual(info["executions"][0]["side"], "sell")
        self.assertEqual(info["executions"][0]["quantity"], 50)

        report = run_episode(
            OptionsEnv(
                three_snapshot_dataset(),
                slot_count=2,
                max_quantity=2,
                starting_cash=1_000,
                underlying_lot_size=25,
            ),
            buy_first_then_delta_hedge(),
        )
        self.assertEqual(report.executions, 2)
        self.assertAlmostEqual(report.final_delta, 0)

    def test_long_volatility_pair_waits_for_signal_then_delta_hedges(self):
        env = OptionsEnv(
            self._long_volatility_dataset(),
            slot_count=2,
            max_quantity=1,
            starting_cash=1_000,
            underlying_lot_size=25,
        )
        observation, _ = env.reset()
        policy = long_volatility_delta_hedge(
            LongVolatilityConfig(
                realized_window=16,
                min_coverage=0.75,
                min_volatility_edge=0.05,
            )
        )
        oversized = long_volatility_delta_hedge(
            LongVolatilityConfig(quantity=2)
        )
        no_history = observation.market.copy()
        no_history[MARKET_FEATURES.index("realizedVol16Coverage")] = 0
        no_edge = observation.market.copy()
        no_edge[MARKET_FEATURES.index("realizedVol16")] = 0.24

        np.testing.assert_array_equal(
            policy(replace(observation, market=no_history)),
            np.array([0, 0, 0]),
        )
        np.testing.assert_array_equal(
            policy(replace(observation, market=no_edge)),
            np.array([0, 0, 0]),
        )
        np.testing.assert_array_equal(
            oversized(observation),
            np.array([0, 0, 0]),
        )
        entry = policy(observation)
        np.testing.assert_array_equal(entry, np.array([1, 1, 0]))
        observation, _, _, _, info = env.step(entry)
        self.assertEqual(len(info["executions"]), 2)
        self.assertAlmostEqual(observation.portfolio[3], 20)

        hedge = policy(observation)
        np.testing.assert_array_equal(hedge, np.array([0, 0, 2]))
        observation, _, _, truncated, info = env.step(hedge)
        self.assertTrue(truncated)
        self.assertEqual(info["executions"][0]["instrument"], "underlying")
        self.assertAlmostEqual(observation.portfolio[3], -5)

        report = run_episode(
            OptionsEnv(
                self._long_volatility_dataset(),
                slot_count=2,
                max_quantity=1,
                starting_cash=1_000,
                underlying_lot_size=25,
            ),
            long_volatility_delta_hedge(),
        )
        self.assertEqual(report.executions, 3)
        self.assertAlmostEqual(report.final_delta, -5)

    def test_long_volatility_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "realized_window"):
            LongVolatilityConfig(realized_window=8)
        with self.assertRaisesRegex(ValueError, "min_coverage"):
            LongVolatilityConfig(min_coverage=1.1)
        with self.assertRaisesRegex(ValueError, "min_volatility_edge"):
            LongVolatilityConfig(min_volatility_edge=-0.01)
        with self.assertRaisesRegex(ValueError, "min_volatility_edge"):
            LongVolatilityConfig(min_volatility_edge=np.nan)

    def test_cash_secured_short_put_waits_for_edge_then_delta_hedges(self):
        dataset = self._short_volatility_dataset()
        long_only = OptionsEnv(
            dataset,
            slot_count=2,
            max_quantity=2,
            starting_cash=20_000,
        )
        legacy_observation, _ = long_only.reset()
        legacy_policy = cash_secured_short_put_delta_hedge()
        np.testing.assert_array_equal(
            legacy_policy(legacy_observation),
            np.array([0, 0, 0]),
        )

        env = OptionsEnv(
            dataset,
            slot_count=2,
            max_quantity=2,
            starting_cash=20_000,
            underlying_lot_size=25,
            allow_collateralized_option_shorts=True,
        )
        observation, _ = env.reset()
        policy = cash_secured_short_put_delta_hedge(
            ShortVolatilityConfig(
                realized_window=16,
                min_coverage=0.75,
                min_volatility_edge=0.05,
            )
        )
        no_history = observation.market.copy()
        no_history[MARKET_FEATURES.index("realizedVol16Coverage")] = 0
        no_edge = observation.market.copy()
        no_edge[MARKET_FEATURES.index("frontAtmIv")] = 0.24

        np.testing.assert_array_equal(
            policy(replace(observation, market=no_history)),
            np.array([0, 0, 0]),
        )
        np.testing.assert_array_equal(
            policy(replace(observation, market=no_edge)),
            np.array([0, 0, 0]),
        )
        entry = policy(observation)
        np.testing.assert_array_equal(entry, np.array([0, 3, 0]))
        observation, _, _, _, info = env.step(entry)
        self.assertEqual(len(info["executions"]), 1)
        self.assertEqual(info["executions"][0]["side"], "sell")
        self.assertEqual(observation.portfolio[8], 10_000)
        self.assertAlmostEqual(observation.portfolio[3], 40)

        hedge = policy(observation)
        np.testing.assert_array_equal(hedge, np.array([0, 0, 4]))
        observation, _, _, truncated, info = env.step(hedge)
        self.assertTrue(truncated)
        self.assertEqual(info["executions"][0]["instrument"], "underlying")
        self.assertAlmostEqual(observation.portfolio[3], -10)

    def test_short_volatility_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "realized_window"):
            ShortVolatilityConfig(realized_window=8)
        with self.assertRaisesRegex(ValueError, "min_coverage"):
            ShortVolatilityConfig(min_coverage=-0.1)
        with self.assertRaisesRegex(ValueError, "min_volatility_edge"):
            ShortVolatilityConfig(min_volatility_edge=np.nan)
        with self.assertRaisesRegex(ValueError, "quantity"):
            ShortVolatilityConfig(quantity=0)

    def test_short_put_baseline_uses_assignment_and_cost_stress(self):
        source = self._short_volatility_dataset()
        snapshots = []
        for timestamp, spot, snapshot in zip(
            (
                "2026-07-20T14:00:00Z",
                "2026-07-21T14:00:00Z",
                "2026-07-22T14:00:00Z",
            ),
            (100, 100, 80),
            source.snapshots,
            strict=True,
        ):
            frame = snapshot.frame.copy()
            frame["collectedAt"] = timestamp
            frame["expiration"] = "2026-07-21"
            frame["underlyingPrice"] = spot
            snapshots.append(Snapshot(timestamp, frame))
        dataset = SnapshotDataset(tuple(snapshots), source.symbol)
        env = OptionsEnv(
            dataset,
            slot_count=2,
            max_quantity=2,
            starting_cash=20_000,
            underlying_lot_size=25,
            allow_collateralized_option_shorts=True,
        )
        policy = cash_secured_short_put_delta_hedge()
        observation, _ = env.reset()
        observation, _, _, _, _ = env.step(policy(observation))
        self.assertEqual(observation.portfolio[8], 10_000)

        observation, _, _, truncated, info = env.step(policy(observation))

        self.assertTrue(truncated)
        self.assertEqual(observation.portfolio[8], 0)
        self.assertEqual(observation.portfolio[7], 50)
        self.assertEqual(
            info["option_settlements"][0]["style"],
            "physical_assignment",
        )

        base = run_episode(
            OptionsEnv(
                self._short_volatility_dataset(),
                slot_count=2,
                max_quantity=2,
                starting_cash=20_000,
                underlying_lot_size=25,
                allow_collateralized_option_shorts=True,
            ),
            cash_secured_short_put_delta_hedge(),
        )
        stressed_env = cost_stressed_environment(
            OptionsEnv(
                self._short_volatility_dataset(),
                slot_count=2,
                max_quantity=2,
                starting_cash=20_000,
                underlying_lot_size=25,
                allow_collateralized_option_shorts=True,
            ),
            CostScenario(
                "double",
                spread_multiplier=2,
                commission_multiplier=2,
            ),
        )
        stressed = run_episode(
            stressed_env,
            cash_secured_short_put_delta_hedge(),
        )

        self.assertGreater(stressed.fees, base.fees)
        self.assertLess(stressed.final_nav, base.final_nav)

    def test_underlying_trend_targets_direction_without_repeated_buying(self):
        env = OptionsEnv(
            self._long_volatility_dataset(),
            slot_count=2,
            max_quantity=2,
            starting_cash=100_000,
            underlying_lot_size=25,
        )
        observation, _ = env.reset()
        policy = underlying_trend(UnderlyingTrendConfig(
            return_window=16,
            min_coverage=0.75,
            min_abs_log_return=0.01,
            quantity=1,
        ))
        covered = observation.market.copy()
        covered[MARKET_FEATURES.index("realizedVol16Coverage")] = 1
        positive = covered.copy()
        positive[MARKET_FEATURES.index("underlyingLogReturn16")] = 0.02
        negative = covered.copy()
        negative[MARKET_FEATURES.index("underlyingLogReturn16")] = -0.02

        np.testing.assert_array_equal(policy(observation), np.array([0, 0, 0]))
        unavailable = underlying_trend(UnderlyingTrendConfig(quantity=3))
        np.testing.assert_array_equal(
            unavailable(replace(observation, market=positive)),
            np.array([0, 0, 0]),
        )
        entry = policy(replace(observation, market=positive))
        np.testing.assert_array_equal(entry, np.array([0, 0, 1]))
        observation, _, _, _, _ = env.step(entry)
        self.assertEqual(observation.portfolio[7], 25)
        np.testing.assert_array_equal(
            policy(replace(observation, market=positive)),
            np.array([0, 0, 0]),
        )
        reversal = policy(replace(observation, market=negative))
        np.testing.assert_array_equal(reversal, np.array([0, 0, 4]))
        observation, _, _, truncated, info = env.step(reversal)
        self.assertTrue(truncated)
        self.assertEqual(observation.portfolio[7], -25)
        self.assertEqual(info["executions"][0]["quantity"], 50)

    def test_underlying_trend_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "return_window"):
            UnderlyingTrendConfig(return_window=8)
        with self.assertRaisesRegex(ValueError, "min_coverage"):
            UnderlyingTrendConfig(min_coverage=-0.1)
        with self.assertRaisesRegex(ValueError, "min_abs_log_return"):
            UnderlyingTrendConfig(min_abs_log_return=np.nan)
        with self.assertRaisesRegex(ValueError, "quantity"):
            UnderlyingTrendConfig(quantity=0)
