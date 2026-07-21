from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from trading_bot.training.baselines import no_op
from trading_bot.training.evaluation import evaluate_policy
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
