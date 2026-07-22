import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from trading_bot.interface.results import (
    agent_leaderboard,
    arena_overview,
    discover_agent_runs,
    equity_curve,
    evidence_summary,
    heldout_results,
    promotion_assessment,
    trade_ledger,
)


def result_summary():
    candidates = [
        {
            "model_id": "gru",
            "model": {"kind": "gru", "encoder": "flat", "algorithm": "ppo"},
            "selection": {
                "robust_training_seed_validation_score": 0.02,
                "training_seed_mean_validation_reward": 0.03,
            },
            "inference_latency": {"median_microseconds": 90},
            "parameter_count": 100,
            "episodes_completed": 2,
        },
        {
            "model_id": "lstm",
            "model": {"kind": "lstm", "encoder": "flat", "algorithm": "ppo"},
            "selection": {
                "robust_training_seed_validation_score": 0.01,
                "training_seed_mean_validation_reward": 0.015,
            },
            "inference_latency": {"median_microseconds": 110},
            "parameter_count": 120,
            "episodes_completed": 2,
        },
    ]
    report = {
        "total_return": 0.01,
        "final_nav": 1_010,
        "max_drawdown": 0.002,
        "executions": 1,
        "turnover": 0.1,
        "fees": 0.65,
        "step_sharpe": 0.5,
        "return_beta_to_underlying": 0.1,
        "mean_abs_delta_notional_weight": 0.2,
        "steps": 2,
    }
    agent_trace = {
        "report": report,
        "timestamps": ["2026-01-01T15:00:00Z", "2026-01-01T15:01:00Z"],
        "step_returns": [0.0, 0.01],
        "navs": [1_000, 1_000, 1_010],
        "decisions": [
            {
                "decision_timestamp": "2026-01-01T14:59:00Z",
                "arrival_timestamp": "2026-01-01T15:00:00Z",
                "orders": [1, 0],
                "invalid_actions": 0,
                "reward": 0.0,
                "nav": 1_000,
                "executions": [
                    {
                        "instrument": "option",
                        "side": "buy",
                        "contract_symbol": "TEST-C",
                        "quantity": 1,
                        "price": 1.0,
                        "fee": 0.65,
                    }
                ],
            }
        ],
    }
    no_op_trace = {
        **agent_trace,
        "navs": [1_000, 1_000, 1_000],
        "decisions": [],
    }
    return {
        "schema_version": "research-demo.walk-forward.v59",
        "symbol": "TEST",
        "folds": [
            {
                "fold": 0,
                "model_selection": {
                    "selected_model_id": "gru",
                    "candidates": candidates,
                },
                "test": [report],
                "heldout_traces": {
                    "agent": [agent_trace],
                    "baselines": {"no_op": [no_op_trace]},
                },
                "statistical_comparisons": {
                    "no_op": [{"status": "insufficient_history"}],
                },
            }
        ],
    }


class InterfaceResultTests(TestCase):
    def test_discovers_only_walk_forward_artifacts(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            run_dir = data_dir / "agent_runs" / "demo"
            run_dir.mkdir(parents=True)
            valid = run_dir / "TEST-walk-forward.json"
            valid.write_text(json.dumps(result_summary()), encoding="utf-8")
            (run_dir / "broken-walk-forward.json").write_text("{", encoding="utf-8")
            (run_dir / "other.json").write_text("{}", encoding="utf-8")

            runs = discover_agent_runs(data_dir)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["symbol"], "TEST")
        self.assertEqual(runs[0]["_run_name"], "demo")

    def test_projects_leaderboard_heldout_curve_and_fills(self):
        summary = result_summary()

        leaderboard = agent_leaderboard(summary)
        heldout = heldout_results(summary)
        curve = equity_curve(summary, 0)
        ledger = trade_ledger(summary, 0)
        evidence = evidence_summary(summary)

        self.assertEqual(leaderboard["Agent"].tolist(), ["GRU Agent", "LSTM Agent"])
        self.assertEqual(
            leaderboard["Action policy"].tolist(),
            ["Factorized multi-leg", "Factorized multi-leg"],
        )
        self.assertEqual(leaderboard["Selected folds"].tolist(), [1, 0])
        self.assertEqual(heldout.iloc[0]["Agent"], "GRU Agent")
        self.assertEqual(heldout.iloc[0]["Encoder"], "Flat")
        self.assertEqual(heldout.iloc[0]["Architecture"], "Flat / Gru")
        self.assertEqual(heldout.iloc[0]["Action policy"], "Factorized multi-leg")
        self.assertAlmostEqual(heldout.iloc[0]["Test return"], 0.01)
        self.assertEqual(heldout.iloc[0]["Fills / decision"], 0.5)
        self.assertEqual(set(curve["Series"]), {"Selected agent", "No Op"})
        self.assertEqual(len(curve[curve["Series"] == "Selected agent"]), 2)
        self.assertEqual(ledger.iloc[0]["Contract"], "TEST-C")
        self.assertEqual(ledger.iloc[0]["Side"], "buy")
        self.assertEqual(evidence["grade"], "Exploratory")
        self.assertFalse(evidence["can_claim_improvement"])

    def test_missing_trace_data_is_an_empty_projection(self):
        summary = result_summary()
        summary["folds"][0].pop("heldout_traces")

        self.assertTrue(equity_curve(summary, 0).empty)
        self.assertTrue(trade_ledger(summary, 0).empty)

    def test_arena_overview_keeps_newest_run_per_ticker(self):
        aapl_new = result_summary()
        aapl_new["symbol"] = "AAPL"
        aapl_new["_run_name"] = "new"
        aapl_old = deepcopy(aapl_new)
        aapl_old["_run_name"] = "old"
        aapl_old["folds"][0]["test"][0]["total_return"] = -0.5
        nvda = deepcopy(aapl_new)
        nvda["symbol"] = "NVDA"
        nvda["_run_name"] = "arena"
        nvda["folds"][0]["model_selection"]["selected_model_id"] = "lstm"
        nvda["folds"][0]["model_selection"]["candidates"][1]["model"][
            "action_decoder"
        ] = "single_leg"
        nvda["folds"][0]["model_selection"]["candidates"][1]["model"]["encoder"] = (
            "surface_graph_set"
        )

        overview = arena_overview([aapl_new, aapl_old, nvda])

        self.assertEqual(overview["Ticker"].tolist(), ["AAPL", "NVDA"])
        self.assertEqual(overview.iloc[0]["Experiment"], "new")
        self.assertAlmostEqual(overview.iloc[0]["Held-out return"], 0.01)
        self.assertEqual(overview.iloc[1]["Selected agent"], "LSTM Agent")
        self.assertEqual(overview.iloc[1]["Selected encoder"], "Surface Graph Set")
        self.assertEqual(overview.iloc[1]["Architecture"], "Surface Graph Set / Lstm")
        self.assertEqual(overview.iloc[1]["Action policy"], "Sparse single-leg")
        self.assertEqual(overview.iloc[1]["Promotion"], "Research only")

    def test_promotion_gate_requires_all_deployment_evidence(self):
        summary = result_summary()

        rejected = promotion_assessment(summary)

        self.assertEqual(rejected["status"], "Research only")
        self.assertIn("held-out history is too short", rejected["failed_reasons"])

        fold = summary["folds"][0]
        fold["test"][0]["steps"] = 100
        fold["baselines"] = {"no_op": [{"total_return": 0.0}]}
        fold["cost_stress"] = {
            "double_costs": [{"total_return": 0.005}],
        }
        fold["statistical_comparisons"]["no_op"] = [
            {
                "status": "ok",
                "supports_improvement": True,
            }
        ]
        fold["test_data_quality"] = {
            "execution_provenance": "provider_confirmed_regular",
        }

        accepted = promotion_assessment(summary)

        self.assertEqual(accepted["status"], "Promotion ready")
        self.assertTrue(all(accepted["checks"].values()))
        self.assertAlmostEqual(accepted["mean_excess_vs_no_op"], 0.01)
