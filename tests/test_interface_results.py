import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from trading_bot.execution.agent_store import AgentPaperStore
from trading_bot.interface.results import (
    agent_decision_tape,
    agent_leaderboard,
    agent_roster,
    arena_readiness_overview,
    arena_overview,
    discover_agent_arena_manifests,
    discover_agent_runs,
    equity_curve,
    evidence_summary,
    feature_ablation_results,
    heldout_results,
    load_arena_watch_status,
    load_paper_agent_watch_status,
    paper_agent_decisions,
    paper_agent_equity_curve,
    paper_agent_overview,
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
            "model": {
                "kind": "lstm",
                "encoder": "flat",
                "algorithm": "ppo",
                "disabled_feature_groups": ("contract_smile_residual",),
            },
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
    def test_loads_only_versioned_paper_agent_heartbeat(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            path = data_dir / "_paper_agent_watch_status.json"
            path.write_text("{}", encoding="utf-8")
            self.assertIsNone(load_paper_agent_watch_status(data_dir))
            expected = {
                "schema_version": "research-demo.paper-agent-watch.v2",
                "status": "running",
            }
            path.write_text(json.dumps(expected), encoding="utf-8")

            self.assertEqual(load_paper_agent_watch_status(data_dir), expected)

    def test_projects_persistent_paper_agent_state_and_guarded_decision(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            store = AgentPaperStore(data_dir / "agent_paper.db")
            store.commit_cycle(
                {
                    "deployment_id": "dep-1",
                    "agent_id": "AAPL-surface-gru",
                    "symbol": "AAPL",
                    "model_id": "surface-gru",
                    "topology": "surface_gnn",
                    "checkpoint_path": "/tmp/aapl.pt",
                    "checkpoint_sha256": "a" * 64,
                    "activated": False,
                    "activation_reason": "edge below threshold",
                    "status": "guarded",
                    "message": "processed 1 new decision(s)",
                    "last_observation_timestamp": "2026-07-22T14:00:00Z",
                    "last_decision_timestamp": "2026-07-22T14:00:00Z",
                    "last_cash": 100_000.0,
                    "last_nav": 100_000.0,
                    "environment_state": {
                        "environment_contract": {"starting_cash": 100_000.0},
                        "positions": {},
                    },
                    "recurrent_state": {"steps": 4},
                },
                [{
                    "snapshot_timestamp": "2026-07-22T14:00:00Z",
                    "activated": False,
                    "research_orders": [1, 0],
                    "sandbox_orders": [0, 0],
                    "executions": [],
                    "reward": 0.0,
                    "cash": 100_000.0,
                    "nav": 100_000.0,
                    "decision_cash": 100_000.0,
                    "decision_nav": 100_000.0,
                    "outcome_status": "pending",
                    "outcome_timestamp": None,
                    "outcome_nav": None,
                    "outcome_return": None,
                    "action_confidence": 0.72,
                    "normalized_action_entropy": 0.41,
                    "explorable_action_factor_count": 2,
                    "decision_factor_count": 2,
                    "invalid_action_count": 0,
                }],
            )

            overview = paper_agent_overview(data_dir)
            decisions = paper_agent_decisions(data_dir)
            curve = paper_agent_equity_curve(data_dir)

        self.assertEqual(overview.iloc[0]["Topology"], "Surface Gnn")
        self.assertEqual(overview.iloc[0]["Activation"], "Guarded")
        self.assertEqual(overview.iloc[0]["Recurrent steps"], 4)
        self.assertEqual(overview.iloc[0]["Finalized outcomes"], 0)
        self.assertEqual(overview.iloc[0]["Pending outcomes"], 1)
        self.assertEqual(overview.iloc[0]["Latest action confidence"], 0.72)
        self.assertEqual(overview.iloc[0]["Latest action entropy"], 0.41)
        self.assertEqual(decisions.iloc[0]["Research action"], "UNFILLED")
        self.assertEqual(decisions.iloc[0]["Sandbox action"], "HOLD")
        self.assertEqual(decisions.iloc[0]["Executions"], 0)
        self.assertEqual(decisions.iloc[0]["Outcome status"], "Pending")
        self.assertEqual(decisions.iloc[0]["Action confidence"], 0.72)
        self.assertEqual(decisions.iloc[0]["Action entropy"], 0.41)
        self.assertEqual(len(curve), 1)
        self.assertEqual(curve.iloc[0]["Stage"], "First decision")

    def test_loads_valid_arena_watch_status_and_ignores_invalid_files(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            path = data_dir / "_arena_watch_status.json"
            self.assertIsNone(load_arena_watch_status(data_dir))
            path.write_text("{", encoding="utf-8")
            self.assertIsNone(load_arena_watch_status(data_dir))
            path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
            self.assertIsNone(load_arena_watch_status(data_dir))
            expected = {
                "schema_version": "research-demo.arena-watch.status.v2",
                "status": "waiting",
            }
            path.write_text(json.dumps(expected), encoding="utf-8")

            self.assertEqual(load_arena_watch_status(data_dir), expected)

    def test_projects_tangible_agent_roster_and_guarded_decision_tape(self):
        summary = result_summary()
        summary["_run_name"] = "latest-arena"
        fold = summary["folds"][0]
        fold["checkpoint"] = "TEST-fold-000-gru.pt"
        fold["test_data_quality"] = {
            "first_timestamp": "2026-01-01T15:00:00Z",
            "last_timestamp": "2026-01-01T15:01:00Z",
        }
        fold["model_selection"]["activation_gate"] = {
            "activated": False,
            "score_advantage": -0.0002,
        }

        tape = agent_decision_tape(summary, 0)
        roster = agent_roster([summary])

        self.assertEqual(len(tape), 1)
        self.assertEqual(tape.iloc[0]["Research action"], "BUY · 1 fill")
        self.assertEqual(tape.iloc[0]["Sandbox action"], "HOLD (guard)")
        self.assertEqual(tape.iloc[0]["Requested legs"], 1)
        self.assertEqual(tape.iloc[0]["Activation"], "Guarded")
        self.assertEqual(len(roster), 1)
        self.assertEqual(roster.iloc[0]["Agent ID"], "TEST-gru")
        self.assertEqual(roster.iloc[0]["State"], "Guarded / no-op")
        self.assertEqual(roster.iloc[0]["Topology"], "Flat vector")
        self.assertEqual(roster.iloc[0]["Last research action"], "BUY · 1 fill")
        self.assertEqual(roster.iloc[0]["Last sandbox action"], "HOLD (guard)")
        self.assertEqual(roster.iloc[0]["Checkpoint"], "TEST-fold-000-gru.pt")
        self.assertAlmostEqual(
            roster.iloc[0]["Validation edge vs no-op (bp)"],
            -2.0,
        )

    def test_discovers_and_projects_readiness_only_arena_manifest(self):
        manifest = {
            "schema_version": "research-demo.agent-arena.v10",
            "preflight": [
                {
                    "symbol": "AAPL",
                    "ready": False,
                    "reason": "regular_fresh_executable_tail_required",
                    "source_snapshot_count": 20,
                    "eligible_snapshot_count": 6,
                    "required_eligible_snapshot_count": 13,
                    "excluded_snapshot_count": 14,
                    "training": {
                        "snapshot_count": 6,
                        "regular_snapshot_count": 6,
                        "fresh_underlying_quote_count": 6,
                        "executable_option_quote_count": 6,
                    },
                    "validation": {
                        "snapshot_count": 3,
                        "regular_snapshot_count": 0,
                        "fresh_underlying_quote_count": 3,
                        "executable_option_quote_count": 3,
                    },
                    "test": {
                        "snapshot_count": 4,
                        "regular_snapshot_count": 1,
                        "fresh_underlying_quote_count": 4,
                        "executable_option_quote_count": 4,
                        "first_timestamp": "2026-07-22T13:00:00Z",
                        "last_timestamp": "2026-07-22T13:45:00Z",
                    },
                }
            ],
        }
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            run_dir = data_dir / "agent_runs" / "arena" / "run-id"
            run_dir.mkdir(parents=True)
            (run_dir / "agent-arena.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            manifests = discover_agent_arena_manifests(data_dir)
            readiness = arena_readiness_overview(manifests[0])

        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0]["_run_name"], "run-id")
        self.assertEqual(readiness.iloc[0]["Ready"], "Waiting")
        self.assertEqual(readiness.iloc[0]["Source snapshots"], 20)
        self.assertEqual(readiness.iloc[0]["Eligible snapshots"], 6)
        self.assertEqual(readiness.iloc[0]["Required eligible"], 13)
        self.assertEqual(readiness.iloc[0]["Excluded snapshots"], 14)
        self.assertEqual(readiness.iloc[0]["Training snapshots"], 6)
        self.assertEqual(readiness.iloc[0]["Training regular"], 6)
        self.assertEqual(readiness.iloc[0]["Validation regular"], 0)
        self.assertEqual(readiness.iloc[0]["Test regular"], 1)
        self.assertEqual(readiness.iloc[0]["Test end"], "2026-07-22T13:45:00Z")

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
        self.assertEqual(leaderboard["Smile residual"].tolist(), ["Enabled", "Ablated"])
        self.assertEqual(leaderboard["Competitive folds"].tolist(), [1, 1])
        self.assertEqual(leaderboard["Training seeds"].tolist(), [1, 1])
        self.assertEqual(heldout.iloc[0]["Agent"], "GRU Agent")
        self.assertEqual(heldout.iloc[0]["Encoder"], "Flat")
        self.assertEqual(heldout.iloc[0]["Architecture"], "Flat / Gru")
        self.assertEqual(heldout.iloc[0]["Feature set"], "Full")
        self.assertEqual(heldout.iloc[0]["Smile residual"], "Enabled")
        self.assertEqual(heldout.iloc[0]["Competitive candidates"], 1)
        self.assertEqual(heldout.iloc[0]["Activation"], "Active")
        self.assertEqual(heldout.iloc[0]["Sandbox policy"], "GRU Agent")
        self.assertAlmostEqual(heldout.iloc[0]["Sandbox return"], 0.01)
        self.assertEqual(heldout.iloc[0]["Sandbox executions"], 1)
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
        nvda["folds"][0]["model_selection"]["activation_gate"] = {
            "activated": False,
        }
        nvda["folds"][0]["test_data_quality"] = {
            "first_timestamp": "2026-07-22T14:00:00Z",
            "last_timestamp": "2026-07-22T14:15:00Z",
            "execution_provenance": "provider_confirmed_regular",
        }
        nvda["folds"][0]["baselines"] = {
            "no_op": [{"total_return": 0.0, "executions": 0, "fees": 0.0}],
        }

        overview = arena_overview([aapl_new, aapl_old, nvda])

        self.assertEqual(overview["Ticker"].tolist(), ["AAPL", "NVDA"])
        self.assertEqual(overview.iloc[0]["Experiment"], "new")
        self.assertAlmostEqual(overview.iloc[0]["Held-out return"], 0.01)
        self.assertEqual(overview.iloc[1]["Selected agent"], "LSTM Agent")
        self.assertEqual(overview.iloc[1]["Selected encoder"], "Surface Graph Set")
        self.assertEqual(overview.iloc[1]["Architecture"], "Surface Graph Set / Lstm")
        self.assertEqual(
            overview.iloc[1]["Feature set"],
            "Without Contract Smile Residual",
        )
        self.assertEqual(overview.iloc[1]["Smile residual"], "Ablated")
        self.assertEqual(overview.iloc[1]["Action policy"], "Sparse single-leg")
        self.assertEqual(overview.iloc[1]["Activation"], "Abstain")
        self.assertEqual(overview.iloc[1]["Sandbox policy"], "No Op")
        self.assertEqual(overview.iloc[1]["Sandbox return"], 0.0)
        self.assertAlmostEqual(overview.iloc[1]["Sandbox lift"], -0.01)
        self.assertEqual(overview.iloc[1]["Sandbox executions"], 0)
        self.assertEqual(overview.iloc[1]["Test start"], "2026-07-22T14:00:00Z")
        self.assertEqual(overview.iloc[1]["Test end"], "2026-07-22T14:15:00Z")
        self.assertEqual(overview.iloc[1]["Promotion"], "Research only")

    def test_feature_ablation_results_expose_validation_only_lift(self):
        summary = result_summary()
        ablated = summary["folds"][0]["model_selection"]["candidates"][1]
        ablated["validation_score_lift_vs_full"] = -0.0002
        ablated["training_seed_aggregate"] = {"training_seed_count": 3}
        summary["_run_name"] = "arena"

        results = feature_ablation_results([summary])

        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["Feature"], "Contract Smile Residual")
        self.assertAlmostEqual(results.iloc[0]["Feature lift (bp)"], 2.0)
        self.assertEqual(results.iloc[0]["Feature helped"], "Yes")
        self.assertEqual(results.iloc[0]["Training seeds"], 3)

        ablated["model"]["disabled_feature_groups"] = ("surface_velocity",)
        velocity = feature_ablation_results([summary], "surface_velocity")

        self.assertEqual(velocity.iloc[0]["Feature"], "Surface Velocity")
        self.assertAlmostEqual(velocity.iloc[0]["Feature lift (bp)"], 2.0)

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
        fold["model_selection"]["activation_gate"] = {"activated": True}

        accepted = promotion_assessment(summary)

        self.assertEqual(accepted["status"], "Promotion ready")
        self.assertTrue(all(accepted["checks"].values()))
        self.assertAlmostEqual(accepted["mean_excess_vs_no_op"], 0.01)
