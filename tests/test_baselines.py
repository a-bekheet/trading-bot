from unittest import TestCase

import numpy as np

from trading_bot.training.baselines import (
    buy_first_then_delta_hedge,
    delta_neutral,
)
from trading_bot.training.env import OptionsEnv
from trading_bot.training.evaluation import run_episode
from tests.test_training_env import three_snapshot_dataset


class BaselineTests(TestCase):
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
