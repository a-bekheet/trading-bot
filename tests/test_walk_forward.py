import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.env import OptionsEnv
from trading_bot.training.recurrent import build_recurrent_actor_critic
from trading_bot.training.sequence import feature_ablation_indices
from trading_bot.training.trainer import TrainingConfig, load_checkpoint
from trading_bot.training.walk_forward import (
    ModelSpec,
    WALK_FORWARD_SCHEMA_VERSION,
    WalkForwardConfig,
    _model_specs_from_args,
    _parser,
    resolve_recurrent_config,
    run_walk_forward_training,
)


def walk_forward_dataset() -> SnapshotDataset:
    snapshots = []
    for index in range(7):
        frame = pd.DataFrame([{
            "collectedAt": f"2026-07-21T14:{index:02d}:00Z",
            "contractSymbol": "TEST-C",
            "symbol": "TEST",
            "expiration": "2026-08-21",
            "optionType": "call",
            "strike": 100,
            "bid": 1.0 + index * 0.05,
            "ask": 1.2 + index * 0.05,
            "lastPrice": 1.1 + index * 0.05,
            "impliedVolatility": 0.2,
            "underlyingPrice": 100 + index * 0.1,
            "riskFreeRate": 0.04,
            "delta": 0.5,
            "gamma": 0.01,
            "theta": -0.1,
            "vega": 0.2,
            "volumeLog": 2.0,
            "openInterestLog": 3.0,
        }])
        snapshots.append(Snapshot(str(index), frame))
    return SnapshotDataset(tuple(snapshots), "TEST")


class WalkForwardTrainingTests(TestCase):
    def test_cli_builds_matched_auxiliary_loss_ablation(self):
        args = _parser().parse_args([
            "--auxiliary-coefficient",
            "0.05",
            "--auxiliary-ablation",
        ])

        specs = _model_specs_from_args(args)

        self.assertEqual(len(specs), 2)
        self.assertIsNone(specs[0].auxiliary_coefficient)
        self.assertEqual(specs[1].auxiliary_coefficient, 0.0)
        with self.assertRaisesRegex(ValueError, "requires"):
            _model_specs_from_args(_parser().parse_args([
                "--auxiliary-ablation",
            ]))
        with self.assertRaisesRegex(ValueError, "auxiliary_coefficient"):
            ModelSpec(auxiliary_coefficient=float("nan"))

    def test_cli_builds_one_step_auxiliary_horizon_ablation(self):
        args = _parser().parse_args([
            "--auxiliary-coefficient",
            "0.05",
            "--auxiliary-horizon",
            "1",
            "--auxiliary-horizon",
            "4",
            "--auxiliary-horizon-ablation",
        ])

        specs = _model_specs_from_args(args)

        self.assertEqual(
            [spec.auxiliary_horizons for spec in specs],
            [(1, 4), (1,)],
        )
        with self.assertRaisesRegex(ValueError, "horizon beyond one"):
            _model_specs_from_args(_parser().parse_args([
                "--auxiliary-horizon-ablation",
            ]))
        with self.assertRaisesRegex(ValueError, "auxiliary_horizons"):
            ModelSpec(auxiliary_horizons=(2, 1))

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_runner_selects_on_validation_then_reports_held_out_test(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(31,),
                    latency_warmup_iterations=1,
                    latency_measured_iterations=3,
                ),
                ModelSpec(
                    kind="hybrid",
                    encoder="graph",
                    hidden_size=8,
                    graph_hidden_size=4,
                    graph_layers=1,
                    graph_neighbors=1,
                    parameter_budget=5_000,
                ),
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                    seed=17,
                ),
                output_dir,
                env_kwargs={"slot_count": 1, "starting_cash": 1_000},
            )
            fold = summary["folds"][0]
            checkpoint_path = Path(fold["checkpoint"])
            _, checkpoint_manifest = load_checkpoint(checkpoint_path)
            written_summary = json.loads(
                (output_dir / "TEST-walk-forward.json").read_text()
            )

        self.assertEqual(summary["schema_version"], WALK_FORWARD_SCHEMA_VERSION)
        self.assertEqual(len(summary["folds"]), 1)
        self.assertEqual(fold["selection"]["scope"], "validation_research_demo")
        self.assertEqual(fold["test"][0]["steps"], 1)
        candidate = fold["model_selection"]["candidates"][0]
        self.assertLessEqual(candidate["parameter_count"], 5_000)
        self.assertEqual(
            candidate["parameter_budget_headroom"],
            5_000 - candidate["parameter_count"],
        )
        self.assertEqual(candidate["model"]["hidden_size"], 8)
        self.assertEqual(candidate["resolved_model"]["hidden_size"], 8)
        self.assertEqual(
            candidate["inference_latency"]["scope"],
            "streaming_batch_1_training_observation",
        )
        self.assertEqual(
            candidate["inference_latency"]["measured_iterations"],
            3,
        )
        self.assertGreater(
            candidate["inference_latency"]["median_microseconds"],
            0,
        )
        self.assertTrue(candidate["deployment_eligible"])
        self.assertEqual(candidate["slot_churn_rate"], 0.0)
        self.assertIsNone(candidate["ineligibility_reason"])
        self.assertFalse(
            fold["model_selection"]["eligibility_constraint"]["enabled"]
        )
        self.assertEqual(
            set(fold["baselines"]),
            {
                "no_op",
                "first_feasible",
                "buy_first_then_delta_hedge",
                "long_volatility_delta_hedge",
                "underlying_trend",
            },
        )
        self.assertEqual(set(fold["cost_stress"]), {"base", "double_costs"})
        self.assertEqual(
            set(fold["statistical_comparisons"]),
            {
                "no_op",
                "first_feasible",
                "buy_first_then_delta_hedge",
                "long_volatility_delta_hedge",
                "underlying_trend",
            },
        )
        self.assertEqual(
            fold["baseline_configuration"]["long_volatility_delta_hedge"],
            {
                "realized_window": 16,
                "min_coverage": 0.75,
                "min_volatility_edge": 0.02,
                "quantity": 1,
            },
        )
        self.assertEqual(
            fold["baseline_configuration"]["underlying_trend"],
            {
                "return_window": 16,
                "min_coverage": 0.75,
                "min_abs_log_return": 0.0,
                "quantity": 1,
            },
        )
        no_op_comparison = fold["statistical_comparisons"]["no_op"][0]
        self.assertEqual(no_op_comparison["status"], "insufficient_history")
        self.assertEqual(no_op_comparison["observations"], 1)
        self.assertIsNone(no_op_comparison["ci_lower"])
        self.assertEqual(no_op_comparison["first_arrival_timestamp"], "6")
        self.assertEqual(no_op_comparison["last_arrival_timestamp"], "6")
        self.assertFalse(no_op_comparison["supports_improvement"])
        self.assertEqual(
            checkpoint_manifest["selection"]["scope"],
            "validation_research_demo",
        )
        provenance = checkpoint_manifest["provenance"]
        fingerprints = provenance["environment_fingerprints"]
        self.assertEqual(len(set(fingerprints.values())), 3)
        self.assertEqual(written_summary["folds"][0]["test"], fold["test"])

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_tournament_selects_one_candidate_before_held_out_evaluation(self):
        candidates = (
            ModelSpec(kind="gru", encoder="flat", hidden_size=8),
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=8,
                disabled_feature_groups=("surface_wings",),
            ),
            ModelSpec(kind="lstm", encoder="flat", hidden_size=8),
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=8,
                algorithm="reinforce",
            ),
            ModelSpec(
                kind="hybrid",
                encoder="graph",
                hidden_size=8,
                graph_hidden_size=4,
                graph_layers=1,
                graph_neighbors=1,
            ),
            ModelSpec(
                kind="gru",
                encoder="graph_set",
                hidden_size=8,
                graph_hidden_size=4,
                graph_layers=1,
                graph_neighbors=1,
            ),
        )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(41,),
                    latency_warmup_iterations=1,
                    latency_measured_iterations=3,
                ),
                candidates,
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                    seed=23,
                ),
                output_dir,
                env_kwargs={"slot_count": 1, "starting_cash": 1_000},
            )
            fold = summary["folds"][0]
            selection = fold["model_selection"]
            candidate_results = selection["candidates"]
            expected = min(
                candidate_results,
                key=lambda result: (
                    -result["selection"]["validation_selection_score"],
                    result["parameter_count"],
                    result["active_input_count"],
                    result["optimizer_updates"],
                    result["model_id"],
                ),
            )
            checkpoint_files = list(output_dir.glob("*.pt"))
            _, manifest = load_checkpoint(checkpoint_files[0])

        self.assertIsNone(summary["model"])
        self.assertEqual(len(summary["candidate_models"]), 6)
        self.assertEqual(len(candidate_results), 6)
        self.assertEqual(len(checkpoint_files), 1)
        self.assertTrue(
            all(result["episodes_completed"] == 1 for result in candidate_results)
        )
        self.assertTrue(
            all(not result["stopped_early"] for result in candidate_results)
        )
        self.assertTrue(
            all(result["resolved_model"] for result in candidate_results)
        )
        self.assertTrue(
            all(
                result["parameter_budget_headroom"] is None
                for result in candidate_results
            )
        )
        self.assertEqual(
            selection["tie_break"],
            [
                "parameter_count",
                "active_input_count",
                "optimizer_updates",
                "model_id",
            ],
        )
        self.assertEqual(selection["criterion"], "validation_selection_score")
        self.assertEqual(
            selection["score_definition"],
            {
                "reward": "validation_total_reward",
                "drawdown_penalty": 0.0,
                "downside_penalty": 0.0,
                "turnover_penalty": 0.0,
                "cross_ticker_std_penalty": 0.0,
                "worst_ticker_weight": 0.0,
            },
        )
        self.assertEqual(selection["selected_model_id"], expected["model_id"])
        self.assertEqual(fold["selection"]["model_id"], expected["model_id"])
        self.assertEqual(
            manifest["provenance"]["model_selection"]["selected_model_id"],
            expected["model_id"],
        )
        self.assertEqual(manifest["model"]["kind"], expected["model"]["kind"])
        self.assertEqual(
            manifest["model"]["encoder"],
            expected["model"]["encoder"],
        )
        self.assertEqual(
            manifest["training"]["algorithm"],
            expected["model"]["algorithm"],
        )
        self.assertEqual(
            manifest["algorithm"],
            (
                "stateful_factorized_ppo"
                if expected["model"]["algorithm"] == "ppo"
                else "stateful_factorized_reinforce_baseline"
            ),
        )
        self.assertEqual(
            tuple(manifest["model"]["masked_input_indices"]),
            feature_ablation_indices(
                tuple(expected["model"]["disabled_feature_groups"]),
                1,
            ),
        )
        ablated = next(
            result
            for result in candidate_results
            if result["model"]["disabled_feature_groups"]
        )
        full = next(
            result
            for result in candidate_results
            if result["model"]["kind"] == "gru"
            and result["model"]["encoder"] == "flat"
            and result["model"]["algorithm"] == ablated["model"]["algorithm"]
            and not result["model"]["disabled_feature_groups"]
        )
        self.assertAlmostEqual(
            ablated["validation_reward_lift_vs_full"],
            ablated["selection"]["validation_total_reward"]
            - full["selection"]["validation_total_reward"],
        )
        self.assertAlmostEqual(
            ablated["validation_score_lift_vs_full"],
            ablated["selection"]["validation_selection_score"]
            - full["selection"]["validation_selection_score"],
        )
        self.assertEqual(
            ablated["active_input_count"] + ablated["masked_input_count"],
            full["active_input_count"],
        )
        self.assertTrue(all("test" not in result for result in candidate_results))

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_auxiliary_ablation_reports_validation_only_lift(self):
        candidates = (
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=4,
                auxiliary_horizons=(1, 2),
            ),
            ModelSpec(kind="gru", encoder="flat", hidden_size=4),
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=4,
                auxiliary_coefficient=0.0,
                auxiliary_horizons=(1, 2),
            ),
        )
        with TemporaryDirectory() as directory:
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(51,),
                    latency_warmup_iterations=1,
                    latency_measured_iterations=2,
                ),
                candidates,
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                    evaluation_interval=1,
                    auxiliary_coefficient=0.05,
                    auxiliary_horizons=(1, 2),
                    seed=29,
                ),
                Path(directory),
                env_kwargs={"slot_count": 1, "starting_cash": 1_000},
            )

        results = summary["folds"][0]["model_selection"]["candidates"]
        enabled, one_step, disabled = results
        self.assertEqual(enabled["effective_auxiliary_coefficient"], 0.05)
        self.assertEqual(enabled["effective_auxiliary_horizons"], [1, 2])
        self.assertEqual(one_step["effective_auxiliary_horizons"], [1])
        self.assertEqual(disabled["effective_auxiliary_coefficient"], 0.0)
        self.assertIsNone(
            enabled["validation_score_lift_vs_auxiliary_enabled"]
        )
        self.assertAlmostEqual(
            disabled["validation_score_lift_vs_auxiliary_enabled"],
            disabled["selection"]["validation_selection_score"]
            - enabled["selection"]["validation_selection_score"],
        )
        self.assertAlmostEqual(
            disabled["validation_reward_lift_vs_auxiliary_enabled"],
            disabled["selection"]["validation_total_reward"]
            - enabled["selection"]["validation_total_reward"],
        )
        self.assertAlmostEqual(
            one_step["validation_score_lift_vs_configured_horizons"],
            one_step["selection"]["validation_selection_score"]
            - enabled["selection"]["validation_selection_score"],
        )
        self.assertAlmostEqual(
            one_step["validation_reward_lift_vs_configured_horizons"],
            one_step["selection"]["validation_total_reward"]
            - enabled["selection"]["validation_total_reward"],
        )

    def test_cli_expands_each_architecture_with_requested_ablation(self):
        args = _parser().parse_args([
            "--candidate",
            "flat:gru",
            "--candidate",
            "graph:hybrid:reinforce",
            "--candidate",
            "graph_set:gru:ppo:0",
            "--ablation",
            "surface_wings",
            "--parameter-budget",
            "5000",
        ])

        specs = _model_specs_from_args(args)

        self.assertEqual(len(specs), 6)
        self.assertEqual(
            sum(bool(spec.disabled_feature_groups) for spec in specs),
            3,
        )
        self.assertEqual(
            {spec.algorithm for spec in specs},
            {"ppo", "reinforce"},
        )
        self.assertEqual({spec.parameter_budget for spec in specs}, {5000})
        graph_set_specs = [
            spec for spec in specs
            if spec.encoder == "graph_set"
        ]
        self.assertEqual({spec.graph_neighbors for spec in graph_set_specs}, {0})

    def test_cli_validates_candidate_graph_neighbor_override(self):
        args = _parser().parse_args([
            "--candidate",
            "graph_set:gru:ppo:not-an-integer",
        ])
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            _model_specs_from_args(args)

        args = _parser().parse_args([
            "--candidate",
            "graph_set:gru:ppo:-1",
        ])
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            _model_specs_from_args(args)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_parameter_budget_resolves_widest_fitting_recurrent_state(self):
        env = OptionsEnv(walk_forward_dataset(), slot_count=1)
        specs = (
            ModelSpec("gru", "flat", hidden_size=32, parameter_budget=5_000),
            ModelSpec("lstm", "flat", hidden_size=32, parameter_budget=5_000),
            ModelSpec("hybrid", "flat", hidden_size=32, parameter_budget=5_000),
            ModelSpec(
                "gru",
                "graph",
                hidden_size=32,
                graph_hidden_size=4,
                graph_layers=1,
                graph_neighbors=1,
                parameter_budget=5_000,
            ),
            ModelSpec(
                "hybrid",
                "graph",
                hidden_size=32,
                graph_hidden_size=4,
                graph_layers=1,
                graph_neighbors=1,
                parameter_budget=5_000,
            ),
            ModelSpec(
                "gru",
                "graph_set",
                hidden_size=32,
                graph_hidden_size=4,
                graph_layers=1,
                graph_neighbors=1,
                parameter_budget=5_000,
            ),
        )

        resolved_sizes = set()
        for spec in specs:
            config, count = resolve_recurrent_config(spec, env)
            actual_count = sum(
                parameter.numel()
                for parameter in build_recurrent_actor_critic(
                    config
                ).parameters()
            )
            self.assertEqual(count, actual_count)
            self.assertLessEqual(count, 5_000)
            self.assertLessEqual(config.hidden_size, spec.hidden_size)
            resolved_sizes.add(config.hidden_size)
            if config.hidden_size < spec.hidden_size:
                next_model = build_recurrent_actor_critic(
                    replace(config, hidden_size=config.hidden_size + 1)
                )
                next_count = sum(
                    parameter.numel() for parameter in next_model.parameters()
                )
                self.assertGreater(next_count, 5_000)

        self.assertGreater(len(resolved_sizes), 1)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_declared_latency_ceiling_excludes_slow_candidate(self):
        candidates = (
            ModelSpec("gru", "flat", hidden_size=4),
            ModelSpec("lstm", "flat", hidden_size=4),
        )

        def fake_benchmark(model, *_args, **_kwargs):
            latency = 50.0 if model.config.kind == "gru" else 500.0
            return {"median_microseconds": latency}

        with TemporaryDirectory() as directory, patch(
            "trading_bot.training.walk_forward.benchmark_recurrent_inference",
            side_effect=fake_benchmark,
        ):
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    3,
                    2,
                    2,
                    test_seeds=(43,),
                    max_median_inference_latency_us=100.0,
                ),
                candidates,
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                ),
                Path(directory),
                env_kwargs={"slot_count": 1},
            )

        selection = summary["folds"][0]["model_selection"]
        results = {
            result["model"]["kind"]: result
            for result in selection["candidates"]
        }
        self.assertTrue(selection["eligibility_constraint"]["enabled"])
        self.assertEqual(selection["eligibility_constraint"]["maximum"], 100.0)
        self.assertEqual(selection["selected_model_id"], candidates[0].identifier)
        self.assertTrue(results["gru"]["deployment_eligible"])
        self.assertFalse(results["lstm"]["deployment_eligible"])
        self.assertEqual(
            results["lstm"]["ineligibility_reason"],
            "median_inference_latency_exceeded",
        )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_latency_ceiling_fails_when_every_candidate_is_too_slow(self):
        with TemporaryDirectory() as directory, patch(
            "trading_bot.training.walk_forward.benchmark_recurrent_inference",
            return_value={"median_microseconds": 50.0},
        ), self.assertRaisesRegex(ValueError, "no model candidate satisfies"):
            run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    3,
                    2,
                    2,
                    max_median_inference_latency_us=10.0,
                ),
                ModelSpec(hidden_size=4),
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                ),
                Path(directory),
                env_kwargs={"slot_count": 1},
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_rejects_parameter_budget_below_minimum_model(self):
        env = OptionsEnv(walk_forward_dataset(), slot_count=1)
        with self.assertRaisesRegex(ValueError, "below the minimum"):
            resolve_recurrent_config(
                ModelSpec(hidden_size=8, parameter_budget=1),
                env,
            )

    def test_model_candidates_have_stable_unique_identifiers(self):
        first = ModelSpec(kind="gru", encoder="flat", hidden_size=8)
        same = ModelSpec(kind="gru", encoder="flat", hidden_size=8)
        different = ModelSpec(kind="lstm", encoder="flat", hidden_size=8)

        self.assertEqual(first.identifier, same.identifier)
        self.assertNotEqual(first.identifier, different.identifier)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique"):
                run_walk_forward_training(
                    walk_forward_dataset(),
                    WalkForwardConfig(3, 2, 2),
                    (first, same),
                    TrainingConfig(episodes=1),
                    Path(directory),
                )

    def test_rejects_invalid_model_candidate(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            ModelSpec(kind="transformer")
        with self.assertRaisesRegex(ValueError, "algorithm"):
            ModelSpec(algorithm="q_learning")
        with self.assertRaisesRegex(ValueError, "parameter_budget"):
            ModelSpec(parameter_budget=0)

    def test_rejects_invalid_latency_benchmark_lengths(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            WalkForwardConfig(
                2,
                2,
                2,
                latency_warmup_iterations=-1,
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            WalkForwardConfig(
                2,
                2,
                2,
                latency_measured_iterations=0,
            )
        for invalid in (0.0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "finite and positive",
            ):
                WalkForwardConfig(
                    2,
                    2,
                    2,
                    max_median_inference_latency_us=invalid,
                )

    def test_rejects_insufficient_history(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "too short"):
                run_walk_forward_training(
                    walk_forward_dataset(),
                    WalkForwardConfig(5, 2, 2),
                    ModelSpec(hidden_size=8),
                    TrainingConfig(episodes=1),
                    Path(directory),
                )
