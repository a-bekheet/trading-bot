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
from trading_bot.training.trainer import (
    TrainingConfig,
    _environment_kwargs_from_args,
    load_checkpoint,
)
from trading_bot.training.walk_forward import (
    ModelSpec,
    WALK_FORWARD_SCHEMA_VERSION,
    WalkForwardConfig,
    _model_specs_from_args,
    _parser,
    _select_seed_robust_group,
    _training_seed_aggregate,
    _walk_forward_config_from_args,
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


def short_volatility_walk_forward_dataset() -> SnapshotDataset:
    snapshots = []
    for index in range(7):
        timestamp = f"2026-07-21T14:{index:02d}:00Z"
        frame = pd.DataFrame([{
            "collectedAt": timestamp,
            "contractSymbol": "TEST-P",
            "symbol": "TEST",
            "expiration": "2026-08-21",
            "optionType": "put",
            "strike": 100,
            "bid": 1.0 + index * 0.05,
            "ask": 1.2 + index * 0.05,
            "lastPrice": 1.1 + index * 0.05,
            "impliedVolatility": 0.5,
            "underlyingPrice": 100,
            "riskFreeRate": 0.04,
            "delta": -0.4,
            "gamma": 0.01,
            "theta": -0.1,
            "vega": 0.2,
            "dteDays": 30 - index / 1_440,
            "logMoneyness": 0.0,
            "spreadPct": 0.18,
            "openInterestLog": 5.0,
            "realizedVol16": 0.2,
            "realizedVol16Coverage": 1.0,
            "frontAtmIv": 0.5,
            "frontAtmIvCoverage": 1.0,
            "atmIvMinusRealizedVol16": 0.3,
        }])
        snapshots.append(Snapshot(timestamp, frame))
    return SnapshotDataset(tuple(snapshots), "TEST")


class WalkForwardTrainingTests(TestCase):
    def test_training_seed_aggregate_penalizes_worst_case_and_dispersion(self):
        config = WalkForwardConfig(
            3,
            2,
            2,
            training_seed_offsets=(0, 10, 20),
            training_seed_worst_weight=0.25,
            training_seed_dispersion_penalty=0.5,
        )
        runs = [
            {
                "training_seed": seed,
                "validation_selection_score": score,
                "validation_total_reward": reward,
                "optimizer_updates": updates,
            }
            for seed, score, reward, updates in (
                (7, 1.0, 2.0, 3),
                (17, 0.0, 1.0, 2),
                (27, -2.0, -1.0, 1),
            )
        ]

        aggregate = _training_seed_aggregate(runs, config)

        self.assertEqual(aggregate["training_seed_count"], 3)
        self.assertEqual(aggregate["training_seeds"], [7, 17, 27])
        self.assertEqual(aggregate["representative_training_seed"], 17)
        self.assertIs(aggregate["representative_run"], runs[1])
        self.assertLess(
            aggregate["robust_training_seed_validation_score"],
            aggregate["validation_selection_score_mean"],
        )

    def test_training_seed_offsets_must_be_unique_non_negative_integers(self):
        for offsets in ((), (0, 0), (False,), (-1,)):
            with self.subTest(offsets=offsets), self.assertRaisesRegex(
                ValueError,
                "unique non-negative integers",
            ):
                WalkForwardConfig(3, 2, 2, training_seed_offsets=offsets)

    def test_seed_robust_selection_does_not_choose_best_single_run(self):
        spec = ModelSpec(hidden_size=4)

        def group(
            model_id,
            robust_score,
            representative_score,
            *,
            latency=100.0,
            parameter_count=100,
        ):
            representative = {
                "model_id": model_id,
                "model_spec": spec,
                "parameter_count": parameter_count,
                "active_input_count": 10,
                "optimizer_updates": 2,
                "validation_selection_score": representative_score,
                "inference_latency": {"median_microseconds": latency},
            }
            return {
                "representative": representative,
                "aggregate": {
                    "robust_training_seed_validation_score": robust_score,
                },
                "replicates": [representative],
                "latency_eligible": True,
            }

        unstable = group("unstable", -1.0, 10.0)
        stable = group("stable", 1.0, 1.0)

        selected = _select_seed_robust_group((unstable, stable))

        self.assertIs(selected, stable)

        smaller_slow = group(
            "smaller-slow",
            1.0,
            1.0,
            latency=200.0,
            parameter_count=50,
        )
        larger_fast = group(
            "larger-fast",
            1.0,
            1.0,
            latency=100.0,
            parameter_count=200,
        )
        self.assertIs(
            _select_seed_robust_group((smaller_slow, larger_fast)),
            larger_fast,
        )

    def test_deterministic_heldout_rejects_seed_pseudoreplication(self):
        with self.assertRaisesRegex(ValueError, "not independent"):
            WalkForwardConfig(
                min_train_size=3,
                validation_size=2,
                test_size=2,
                test_seeds=(1, 2),
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_runner_trains_seed_replicates_but_deploys_one_checkpoint(self):
        with TemporaryDirectory() as directory:
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    3,
                    2,
                    2,
                    training_seed_offsets=(0, 100),
                    test_seeds=(29,),
                    latency_warmup_iterations=1,
                    latency_measured_iterations=2,
                ),
                ModelSpec(hidden_size=4),
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                    seed=11,
                ),
                Path(directory),
                env_kwargs={"slot_count": 1, "starting_cash": 1_000},
            )
            fold = summary["folds"][0]
            candidate = fold["model_selection"]["candidates"][0]
            checkpoint_files = list(Path(directory).glob("*.pt"))
            _, manifest = load_checkpoint(checkpoint_files[0])

        self.assertEqual(len(checkpoint_files), 1)
        self.assertEqual(
            [
                replicate["training_seed"]
                for replicate in candidate["training_seed_replicates"]
            ],
            [11, 111],
        )
        self.assertEqual(
            candidate["training_seed_aggregate"]["training_seed_count"],
            2,
        )
        selected_seed = candidate["training_seed_aggregate"][
            "representative_training_seed"
        ]
        self.assertEqual(fold["selection"]["training_seed"], selected_seed)
        self.assertEqual(manifest["training"]["seed"], selected_seed)
        self.assertEqual(
            fold["heldout_evaluation_contract"]["training_seeds"],
            [11, 111],
        )
        self.assertEqual(len(fold["test"]), 1)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_short_volatility_baseline_trades_and_is_cost_stressed(self):
        with TemporaryDirectory() as directory:
            summary = run_walk_forward_training(
                short_volatility_walk_forward_dataset(),
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(37,),
                    bootstrap_samples=100,
                    latency_warmup_iterations=1,
                    latency_measured_iterations=2,
                ),
                ModelSpec(kind="gru", encoder="flat", hidden_size=4),
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                ),
                Path(directory),
                env_kwargs={
                    "slot_count": 1,
                    "max_quantity": 1,
                    "starting_cash": 20_000,
                    "allow_collateralized_option_shorts": True,
                },
            )

        fold = summary["folds"][0]
        name = "cash_secured_short_put_delta_hedge"
        report = fold["baselines"][name][0]
        stress = fold["baseline_cost_stress"][name]

        self.assertEqual(report["executions"], 1)
        self.assertEqual(report["invalid_actions"], 0)
        self.assertGreater(stress["double_costs"][0]["fees"], report["fees"])
        self.assertLess(
            stress["double_costs"][0]["final_nav"],
            stress["base"][0]["final_nav"],
        )
        self.assertIn(name, fold["statistical_comparisons"])

    def test_cli_maps_all_baseline_configuration(self):
        args = _parser().parse_args([
            "--burn-in-steps", "12",
            "--short-volatility-window", "4",
            "--short-volatility-min-coverage", "0.6",
            "--short-volatility-min-edge", "0.03",
            "--short-volatility-quantity", "2",
            "--trend-window", "4",
            "--trend-min-coverage", "0.5",
            "--trend-min-abs-log-return", "0.01",
            "--trend-quantity", "2",
            "--training-seed-offset", "0",
            "--training-seed-offset", "1000",
            "--training-seed-worst-weight", "0.4",
            "--training-seed-dispersion-penalty", "0.6",
            "--reward-drawdown-penalty", "3",
            "--reward-downside-penalty", "4",
        ])

        config = _walk_forward_config_from_args(args)

        self.assertEqual(args.burn_in_steps, 12)
        self.assertEqual(config.short_volatility_window, 4)
        self.assertEqual(config.short_volatility_min_coverage, 0.6)
        self.assertEqual(config.short_volatility_min_edge, 0.03)
        self.assertEqual(config.short_volatility_quantity, 2)
        self.assertEqual(config.trend_window, 4)
        self.assertEqual(config.trend_min_coverage, 0.5)
        self.assertEqual(config.trend_min_abs_log_return, 0.01)
        self.assertEqual(config.trend_quantity, 2)
        self.assertEqual(config.training_seed_offsets, (0, 1000))
        self.assertEqual(config.training_seed_worst_weight, 0.4)
        self.assertEqual(config.training_seed_dispersion_penalty, 0.6)
        environment = _environment_kwargs_from_args(args)
        self.assertEqual(environment["reward_drawdown_penalty"], 3)
        self.assertEqual(environment["reward_downside_penalty"], 4)
        self.assertEqual(environment["portfolio_valuation"], "liquidation")
        self.assertEqual(
            _environment_kwargs_from_args(_parser().parse_args([
                "--portfolio-valuation", "midpoint",
            ]))["portfolio_valuation"],
            "midpoint",
        )

    def test_cli_makes_collateralized_option_shorts_explicitly_opt_in(self):
        self.assertFalse(
            _parser().parse_args([]).allow_collateralized_option_shorts
        )
        self.assertTrue(
            _parser().parse_args([
                "--allow-collateralized-option-shorts",
            ]).allow_collateralized_option_shorts
        )

    def test_cli_builds_matched_fixed_step_discount_ablation(self):
        args = _parser().parse_args(["--fixed-step-discount-ablation"])

        specs = _model_specs_from_args(args)

        self.assertEqual(len(specs), 2)
        self.assertIsNone(specs[0].time_aware_discounting)
        self.assertFalse(specs[1].time_aware_discounting)
        with self.assertRaisesRegex(ValueError, "requires"):
            _model_specs_from_args(_parser().parse_args([
                "--no-time-aware-discounting",
                "--fixed-step-discount-ablation",
            ]))
        with self.assertRaisesRegex(ValueError, "time_aware_discounting"):
            ModelSpec(time_aware_discounting=1)

    def test_cli_builds_matched_recurrent_burn_in_ablation(self):
        args = _parser().parse_args([
            "--burn-in-steps", "12",
            "--burn-in-ablation",
        ])

        specs = _model_specs_from_args(args)

        self.assertEqual(len(specs), 2)
        self.assertIsNone(specs[0].burn_in_steps)
        self.assertEqual(specs[1].burn_in_steps, 0)
        with self.assertRaisesRegex(ValueError, "requires"):
            _model_specs_from_args(_parser().parse_args([
                "--burn-in-steps", "0",
                "--burn-in-ablation",
            ]))
        with self.assertRaisesRegex(ValueError, "burn_in_steps"):
            ModelSpec(burn_in_steps=True)

    def test_cli_builds_matched_start_sampling_ablation(self):
        args = _parser().parse_args([
            "--start-sampling", "volatility_stratified",
            "--volatility-regime-window", "4",
            "--volatility-regime-bins", "4",
            "--start-sampling-ablation",
        ])

        specs = _model_specs_from_args(args)

        self.assertEqual(args.volatility_regime_bins, 4)
        self.assertEqual(args.volatility_regime_window, 4)
        self.assertIsNone(specs[0].start_sampling)
        self.assertEqual(specs[1].start_sampling, "uniform")
        with self.assertRaisesRegex(ValueError, "requires"):
            _model_specs_from_args(_parser().parse_args([
                "--start-sampling-ablation",
            ]))
        with self.assertRaisesRegex(ValueError, "start_sampling"):
            ModelSpec(start_sampling="future_aware")

    def test_cli_builds_matched_factorized_ppo_objective_ablation(self):
        args = _parser().parse_args(["--factorized-objective-ablation"])

        specs = _model_specs_from_args(args)

        self.assertEqual(len(specs), 2)
        self.assertIsNone(specs[0].factorized_ppo_objective)
        self.assertEqual(specs[1].factorized_ppo_objective, "dimensionwise")
        with self.assertRaisesRegex(ValueError, "requires"):
            _model_specs_from_args(_parser().parse_args([
                "--factorized-ppo-objective", "dimensionwise",
                "--factorized-objective-ablation",
            ]))
        with self.assertRaisesRegex(ValueError, "requires a factorized PPO"):
            _model_specs_from_args(_parser().parse_args([
                "--candidate", "flat:gru:reinforce",
                "--factorized-objective-ablation",
            ]))
        with self.assertRaisesRegex(ValueError, "factorized_ppo_objective"):
            ModelSpec(factorized_ppo_objective="marginal")
        with self.assertRaisesRegex(ValueError, "requires factorized PPO"):
            ModelSpec(
                action_decoder="single_leg",
                factorized_ppo_objective="dimensionwise",
            )

    def test_cli_builds_matched_entropy_objective_ablation(self):
        args = _parser().parse_args(["--entropy-objective-ablation"])

        specs = _model_specs_from_args(args)

        self.assertEqual(len(specs), 2)
        self.assertIsNone(specs[0].entropy_objective)
        self.assertEqual(specs[1].entropy_objective, "raw_mean")
        with self.assertRaisesRegex(ValueError, "requires"):
            _model_specs_from_args(_parser().parse_args([
                "--entropy-objective", "raw_mean",
                "--entropy-objective-ablation",
            ]))
        with self.assertRaisesRegex(ValueError, "requires"):
            _model_specs_from_args(_parser().parse_args([
                "--entropy-coefficient", "0",
                "--entropy-objective-ablation",
            ]))
        with self.assertRaisesRegex(ValueError, "entropy_objective"):
            ModelSpec(entropy_objective="all_slots")

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

    def test_cli_builds_factorized_and_single_leg_candidates(self):
        args = _parser().parse_args([
            "--candidate",
            "flat:gru:ppo:factorized",
            "--candidate",
            "graph_set:mixture:ppo:0:single_leg",
            "--candidate",
            "attention_set:gru:ppo:factorized",
            "--attention-heads",
            "2",
        ])

        specs = _model_specs_from_args(args)

        self.assertEqual(
            [spec.action_decoder for spec in specs],
            ["factorized", "single_leg", "factorized"],
        )
        self.assertEqual(specs[1].graph_neighbors, 0)
        self.assertEqual(specs[2].encoder, "attention_set")
        self.assertEqual(specs[2].attention_heads, 2)
        self.assertEqual(specs[2].graph_neighbors, 0)
        self.assertNotEqual(specs[0].identifier, specs[1].identifier)

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
        self.assertEqual(
            summary["environment"]["portfolio_valuation"],
            "liquidation",
        )
        self.assertNotIn("data_hash", summary["environment"])
        self.assertEqual(written_summary["environment"], summary["environment"])
        self.assertEqual(len(summary["folds"]), 1)
        self.assertEqual(fold["selection"]["scope"], "validation_research_demo")
        self.assertEqual(fold["test"][0]["steps"], 1)
        self.assertEqual(fold["heldout_evaluation_contract"], {
            "deterministic_policy": True,
            "path_count": 1,
            "seed_repetitions": 1,
            "test_seed": 31,
            "training_seed_count": 1,
            "training_seeds": [17],
            "selected_training_seed": 17,
            "bootstrap_independence_unit": "arrival_time_block",
        })
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
                "cash_secured_short_put_delta_hedge",
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
                "cash_secured_short_put_delta_hedge",
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
            fold["baseline_configuration"][
                "cash_secured_short_put_delta_hedge"
            ],
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
        self.assertEqual(
            set(fold["baseline_cost_stress"][
                "cash_secured_short_put_delta_hedge"
            ]),
            {"base", "double_costs"},
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
                    -result["selection"][
                        "robust_training_seed_validation_score"
                    ],
                    max(
                        replicate["inference_latency"]["median_microseconds"]
                        for replicate in result["training_seed_replicates"]
                    ),
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
        self.assertTrue(all(
            result["effective_burn_in_steps"] == 8
            for result in candidate_results
        ))
        self.assertTrue(
            all(
                result["parameter_budget_headroom"] is None
                for result in candidate_results
            )
        )
        self.assertEqual(
            selection["tie_break"],
            [
                "dimensionwise_factorized_objective_ablation",
                "raw_mean_entropy_objective_ablation",
                "worst_training_seed_median_inference_latency",
                "parameter_count",
                "active_input_count",
                "optimizer_updates",
                "burn_in_ablation",
                "fixed_step_discount_ablation",
                "uniform_start_sampling_ablation",
                "model_id",
            ],
        )
        self.assertEqual(
            selection["criterion"],
            "robust_training_seed_validation_score",
        )
        self.assertEqual(
            selection["score_definition"],
            {
                "reward": "validation_total_reward",
                "drawdown_penalty": 0.0,
                "downside_penalty": 0.0,
                "turnover_penalty": 0.0,
                "cross_ticker_std_penalty": 0.0,
                "worst_ticker_weight": 0.0,
                "training_seed_worst_weight": 0.25,
                "training_seed_dispersion_penalty": 0.25,
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

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_discount_ablation_reports_validation_only_lift(self):
        candidates = (
            ModelSpec(kind="gru", encoder="flat", hidden_size=4),
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=4,
                time_aware_discounting=False,
            ),
        )
        with TemporaryDirectory() as directory:
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(53,),
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
                    seed=31,
                ),
                Path(directory),
                env_kwargs={"slot_count": 1, "starting_cash": 1_000},
            )

        time_aware, fixed = summary["folds"][0]["model_selection"][
            "candidates"
        ]
        self.assertTrue(time_aware["effective_time_aware_discounting"])
        self.assertFalse(fixed["effective_time_aware_discounting"])
        self.assertAlmostEqual(
            fixed["validation_score_lift_vs_time_aware_discounting"],
            fixed["selection"]["validation_selection_score"]
            - time_aware["selection"]["validation_selection_score"],
        )
        if (
            fixed["selection"]["validation_selection_score"]
            == time_aware["selection"]["validation_selection_score"]
        ):
            selected_id = summary["folds"][0]["model_selection"][
                "selected_model_id"
            ]
            selected = next(
                result
                for result in (time_aware, fixed)
                if result["model_id"] == selected_id
            )
            other = fixed if selected is time_aware else time_aware
            self.assertLessEqual(
                max(
                    item["inference_latency"]["median_microseconds"]
                    for item in selected["training_seed_replicates"]
                ),
                max(
                    item["inference_latency"]["median_microseconds"]
                    for item in other["training_seed_replicates"]
                ),
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_burn_in_ablation_reports_validation_only_lift(self):
        candidates = (
            ModelSpec(kind="gru", encoder="flat", hidden_size=4),
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=4,
                burn_in_steps=0,
            ),
        )
        with TemporaryDirectory() as directory:
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(59,),
                    latency_warmup_iterations=1,
                    latency_measured_iterations=2,
                ),
                candidates,
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    burn_in_steps=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                    evaluation_interval=1,
                    seed=37,
                ),
                Path(directory),
                env_kwargs={"slot_count": 1, "starting_cash": 1_000},
            )

        enabled, disabled = summary["folds"][0]["model_selection"][
            "candidates"
        ]
        self.assertEqual(enabled["effective_burn_in_steps"], 2)
        self.assertEqual(disabled["effective_burn_in_steps"], 0)
        self.assertAlmostEqual(
            disabled["validation_score_lift_vs_burn_in"],
            disabled["selection"]["validation_selection_score"]
            - enabled["selection"]["validation_selection_score"],
        )
        if (
            disabled["selection"]["validation_selection_score"]
            == enabled["selection"]["validation_selection_score"]
        ):
            selected_id = summary["folds"][0]["model_selection"][
                "selected_model_id"
            ]
            selected = next(
                result
                for result in (enabled, disabled)
                if result["model_id"] == selected_id
            )
            other = disabled if selected is enabled else enabled
            self.assertLessEqual(
                max(
                    item["inference_latency"]["median_microseconds"]
                    for item in selected["training_seed_replicates"]
                ),
                max(
                    item["inference_latency"]["median_microseconds"]
                    for item in other["training_seed_replicates"]
                ),
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_training_objective_ablations_report_validation_only_lift(self):
        candidates = (
            ModelSpec(kind="gru", encoder="flat", hidden_size=4),
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=4,
                factorized_ppo_objective="dimensionwise",
            ),
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=4,
                entropy_objective="raw_mean",
            ),
        )
        with TemporaryDirectory() as directory:
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(61,),
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
                    seed=41,
                ),
                Path(directory),
                env_kwargs={"slot_count": 2, "starting_cash": 1_000},
            )

        joint, dimensionwise, raw_entropy = summary["folds"][0]["model_selection"][
            "candidates"
        ]
        self.assertEqual(joint["effective_factorized_ppo_objective"], "joint")
        self.assertEqual(
            dimensionwise["effective_factorized_ppo_objective"],
            "dimensionwise",
        )
        self.assertAlmostEqual(
            dimensionwise[
                "validation_score_lift_vs_joint_factorized_objective"
            ],
            dimensionwise["selection"]["validation_selection_score"]
            - joint["selection"]["validation_selection_score"],
        )
        self.assertIsNone(
            joint["validation_score_lift_vs_joint_factorized_objective"]
        )
        self.assertEqual(
            joint["effective_entropy_objective"],
            "feasible_normalized",
        )
        self.assertEqual(raw_entropy["effective_entropy_objective"], "raw_mean")
        for candidate in (joint, dimensionwise, raw_entropy):
            for replicate in candidate["training_seed_replicates"]:
                entropy = replicate["entropy"]
                self.assertEqual(
                    entropy["objective"],
                    candidate["effective_entropy_objective"],
                )
                self.assertGreaterEqual(
                    entropy["minimum_feasible_normalized_entropy"],
                    0.0,
                )
                self.assertLessEqual(
                    entropy["maximum_feasible_normalized_entropy"],
                    1.0,
                )
        self.assertAlmostEqual(
            raw_entropy[
                "validation_score_lift_vs_feasible_normalized_entropy"
            ],
            raw_entropy["selection"]["validation_selection_score"]
            - joint["selection"]["validation_selection_score"],
        )
        if (
            joint["selection"]["validation_selection_score"]
            == dimensionwise["selection"]["validation_selection_score"]
            == raw_entropy["selection"]["validation_selection_score"]
        ):
            self.assertEqual(
                summary["folds"][0]["model_selection"]["selected_model_id"],
                joint["model_id"],
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
            ModelSpec("mixture", "flat", hidden_size=32, parameter_budget=5_000),
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
            ModelSpec(
                "mixture",
                "graph_set",
                hidden_size=32,
                graph_hidden_size=4,
                graph_layers=1,
                graph_neighbors=0,
                parameter_budget=5_000,
            ),
            ModelSpec(
                "gru",
                "attention_set",
                hidden_size=32,
                graph_hidden_size=4,
                graph_layers=1,
                attention_heads=2,
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
            "training_seed_inference_latency_exceeded",
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
        original_identifier = first.identifier
        with patch(
            "trading_bot.training.walk_forward.FEATURE_VECTOR_SCHEMA_VERSION",
            "dimensionless.future",
        ):
            self.assertNotEqual(original_identifier, ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=8,
            ).identifier)
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
        with self.assertRaisesRegex(ValueError, "action_decoder"):
            ModelSpec(action_decoder="beam_search")
        with self.assertRaisesRegex(ValueError, "attention_heads"):
            ModelSpec(attention_heads=0)
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelSpec(
                encoder="attention_set",
                graph_hidden_size=6,
                attention_heads=4,
            )

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
