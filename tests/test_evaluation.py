from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from trading_bot.training.baselines import first_feasible, no_op
from trading_bot.training.evaluation import (
    CostScenario,
    cost_stressed_environment,
    evaluate_cost_stress,
    evaluate_policy,
    paired_moving_block_bootstrap,
    run_episode_trace,
)
from trading_bot.training.env import OptionsEnv
from tests.test_training_env import three_snapshot_dataset


class EvaluationTests(TestCase):
    def test_episode_trace_retains_aligned_returns_without_changing_report(self):
        trace = run_episode_trace(
            OptionsEnv(three_snapshot_dataset(), slot_count=2),
            first_feasible,
            seed=9,
        )

        self.assertEqual(len(trace.timestamps), trace.report.steps)
        self.assertEqual(len(trace.step_returns), trace.report.steps)
        self.assertAlmostEqual(
            np.prod(1 + np.asarray(trace.step_returns)) - 1,
            trace.report.total_return,
        )

    def test_paired_block_bootstrap_is_deterministic_and_detects_lift(self):
        baseline = np.sin(np.arange(64) / 4) * 0.002
        candidate = baseline + 0.001

        first = paired_moving_block_bootstrap(
            candidate,
            baseline,
            samples=1_000,
            seed=17,
        )
        second = paired_moving_block_bootstrap(
            candidate,
            baseline,
            samples=1_000,
            seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.status, "ok")
        self.assertEqual(first.block_length, 8)
        self.assertGreater(first.ci_lower, 0)
        self.assertEqual(first.bootstrap_fraction_positive, 1.0)
        self.assertTrue(first.supports_improvement)
        self.assertAlmostEqual(
            first.point_estimate,
            float(np.sum(np.log1p(candidate) - np.log1p(baseline))),
        )

    def test_paired_block_bootstrap_refuses_false_precision(self):
        comparison = paired_moving_block_bootstrap(
            np.array([0.01, -0.01, 0.0]),
            np.zeros(3),
            samples=100,
            min_observations=8,
        )

        self.assertEqual(comparison.status, "insufficient_history")
        self.assertIsNone(comparison.ci_lower)
        self.assertIsNone(comparison.bootstrap_fraction_positive)
        self.assertFalse(comparison.supports_improvement)
        with self.assertRaisesRegex(ValueError, "equal length"):
            paired_moving_block_bootstrap([0.1], [0.1, 0.2])

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
            def factory():
                return OptionsEnv.from_directory(
                    Path(directory), "TEST", slot_count=2
                )
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

    def test_cost_scenario_stresses_underlying_slippage_and_commission(self):
        rows = []
        for index, timestamp in enumerate((
            "2026-07-21T14:00:00Z",
            "2026-07-21T14:01:00Z",
        )):
            rows.append({
                "collectedAt": timestamp,
                "contractSymbol": "TEST-C",
                "symbol": "TEST",
                "expiration": "2026-08-21",
                "optionType": "call",
                "strike": 100,
                "bid": 1,
                "ask": 1.2,
                "lastPrice": 1.1,
                "impliedVolatility": 0.2,
                "underlyingPrice": 100 + index,
                "riskFreeRate": 0.04,
            })
        with TemporaryDirectory() as directory:
            path = Path(directory) / "TEST.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            base = OptionsEnv.from_directory(
                Path(directory),
                "TEST",
                slot_count=1,
                slot_assignment="ranked",
                max_quantity=1,
                allow_collateralized_option_shorts=True,
                reward_drawdown_penalty=2.0,
                reward_downside_penalty=3.0,
                underlying_lot_size=25,
                underlying_commission_per_share=0.01,
                underlying_slippage_bps=10,
            )
            stressed = cost_stressed_environment(
                base,
                CostScenario(
                    "double",
                    spread_multiplier=2,
                    commission_multiplier=2,
                ),
            )
            base.reset()
            stressed.reset()
            _, _, _, _, base_info = base.step(np.array([0, 1]))
            _, _, _, _, stressed_info = stressed.step(np.array([0, 1]))

        self.assertAlmostEqual(base_info["executions"][0]["price"], 100.1)
        self.assertAlmostEqual(stressed_info["executions"][0]["price"], 100.2)
        self.assertEqual(stressed.slot_assignment, "ranked")
        self.assertEqual(stressed.manifest.slot_assignment, "ranked")
        self.assertTrue(stressed.allow_collateralized_option_shorts)
        self.assertTrue(
            stressed.manifest.allow_collateralized_option_shorts
        )
        self.assertEqual(stressed.reward_drawdown_penalty, 2.0)
        self.assertEqual(stressed.reward_downside_penalty, 3.0)
        self.assertEqual(stressed.manifest.reward_drawdown_penalty, 2.0)
        self.assertEqual(stressed.manifest.reward_downside_penalty, 3.0)
        self.assertAlmostEqual(base_info["fees"], 0.25)
        self.assertAlmostEqual(stressed_info["fees"], 0.5)
