from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from trading_bot.training.baselines import first_feasible, no_op
from trading_bot.training.evaluation import evaluate_cost_stress, evaluate_policy
from trading_bot.training.env import OptionsEnv


class EvaluationTests(TestCase):
    def test_no_op_evaluation_is_reproducible(self):
        rows = []
        for timestamp in ("2026-07-21T14:00:00Z", "2026-07-21T14:01:00Z"):
            rows.append(
                {
                    "collectedAt": timestamp, "contractSymbol": "TEST-C",
                    "symbol": "TEST", "expiration": "2026-08-21", "optionType": "call",
                    "strike": 100, "bid": 1, "ask": 1.1, "lastPrice": 1,
                    "impliedVolatility": .2, "underlyingPrice": 100,
                    "riskFreeRate": .04, "greekModel": "black-scholes-merton",
                }
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "TEST.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            factory = lambda: OptionsEnv.from_directory(Path(directory), "TEST", slot_count=2)
            first = evaluate_policy(factory, no_op, seeds=(4,))[0].to_dict()
            second = evaluate_policy(factory, no_op, seeds=(4,))[0].to_dict()

        self.assertEqual(first, second)
        self.assertEqual(first["invalid_actions"], 0)
        self.assertEqual(first["steps"], 1)
        self.assertEqual(first["total_return"], 0)
        self.assertTrue(all(
            np.isfinite(value)
            for key, value in first.items()
            if key != "seed"
        ))

    def test_doubled_costs_reduce_identical_policy_nav(self):
        rows = []
        for timestamp, bid, ask in (
            ("2026-07-21T14:00:00Z", 1.0, 1.2),
            ("2026-07-21T14:01:00Z", 1.5, 1.7),
        ):
            rows.append({
                "collectedAt": timestamp,
                "contractSymbol": "TEST-C",
                "symbol": "TEST",
                "expiration": "2026-08-21",
                "optionType": "call",
                "strike": 100,
                "bid": bid,
                "ask": ask,
                "lastPrice": bid,
                "impliedVolatility": 0.2,
                "underlyingPrice": 100,
                "riskFreeRate": 0.04,
                "delta": 0.5,
                "gamma": 0.01,
                "theta": -0.1,
                "vega": 0.2,
                "greekModel": "black-scholes-merton",
            })
        with TemporaryDirectory() as directory:
            path = Path(directory) / "TEST.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            env = OptionsEnv.from_directory(
                Path(directory), "TEST", slot_count=1, starting_cash=1_000
            )
            reports = evaluate_cost_stress(env, first_feasible, seeds=(7,))

        base = reports["base"][0]
        stressed = reports["double_costs"][0]
        self.assertGreater(base.final_nav, stressed.final_nav)
        self.assertGreater(stressed.fees, base.fees)
        self.assertGreater(base.turnover, 0)
        self.assertGreater(base.max_abs_delta, 0)
