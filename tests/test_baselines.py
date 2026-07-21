from dataclasses import replace
from unittest import TestCase

import numpy as np
import pandas as pd

from trading_bot.training.baselines import (
    LongVolatilityConfig,
    buy_first_then_delta_hedge,
    delta_neutral,
    long_volatility_delta_hedge,
)
from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.env import MARKET_FEATURES, OptionsEnv
from trading_bot.training.evaluation import run_episode
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
                    "atmIvMinusRealizedVol4": -0.25,
                    "atmIvMinusRealizedVol16": -0.30,
                })
            frame = pd.DataFrame(rows)
            snapshots.append(Snapshot(str(index), frame))
        return SnapshotDataset(tuple(snapshots), "TEST")

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
