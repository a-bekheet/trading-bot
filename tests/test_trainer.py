import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.env import OptionsEnv
from trading_bot.training.recurrent import (
    RecurrentConfig,
    build_recurrent_actor_critic,
)
from trading_bot.training.sequence import AUXILIARY_TARGET_FEATURES, observation_vector
from trading_bot.training.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainingConfig,
    _discounted_returns,
    _duration_adjusted_factors,
    _elapsed_seconds,
    _generalized_advantages,
    _parser,
    _sample_rollout_bounds,
    _symbols_from_args,
    aggregate_selection_scores,
    benchmark_recurrent_inference,
    evaluate_recurrent_policy,
    load_checkpoint,
    save_checkpoint,
    selection_score,
    train_actor_critic,
)
from tests.test_training_env import demo_dataset, three_snapshot_dataset


class TrainerTests(TestCase):
    @skipUnless(torch is not None, "install the optional ml extra")
    def test_graph_set_policy_trains_and_round_trips_checkpoint(self):
        env = OptionsEnv(three_snapshot_dataset(), slot_count=2)
        observation, _ = env.reset(seed=43)
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_slot_count=env.action_shape[0],
            action_count=env.action_shape[1],
            hidden_size=4,
            kind="mixture",
            encoder="graph_set",
            contract_feature_count=observation.contracts.shape[1],
            market_feature_count=observation.market.size,
            portfolio_feature_count=observation.portfolio.size,
            graph_hidden_size=4,
            graph_neighbors=0,
            auxiliary_target_count=len(AUXILIARY_TARGET_FEATURES),
        )
        training = TrainingConfig(
            episodes=1,
            sequence_length=2,
            ppo_epochs=1,
            minibatch_size=2,
            evaluation_interval=1,
            auxiliary_coefficient=0.05,
            seed=43,
        )

        model, metrics = train_actor_critic(env, recurrent, training)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph-set.pt"
            save_checkpoint(path, model, env, recurrent, training, metrics)
            restored, manifest = load_checkpoint(path)

        self.assertEqual(restored.config.encoder, "graph_set")
        self.assertEqual(restored.config.kind, "mixture")
        self.assertEqual(restored.config.graph_neighbors, 0)
        self.assertIsNotNone(restored.mixture_gate)
        self.assertTrue(math.isfinite(metrics[0]["auxiliary_loss"]))
        self.assertTrue(all(
            torch.isfinite(parameter).all()
            for parameter in restored.parameters()
        ))

    def test_cli_selects_single_or_top50_training_universe(self):
        single = _parser().parse_args(["--symbol", "msft"])
        universe = _parser().parse_args(["--universe", "top50"])
        sparse = _parser().parse_args(["--action-decoder", "single_leg"])

        self.assertEqual(_symbols_from_args(single), ("MSFT",))
        self.assertEqual(single.slot_assignment, "stable")
        self.assertEqual(single.action_decoder, "factorized")
        self.assertEqual(sparse.action_decoder, "single_leg")
        self.assertEqual(
            _parser().parse_args(["--slot-assignment", "ranked"]).slot_assignment,
            "ranked",
        )
        self.assertEqual(len(_symbols_from_args(universe)), 50)
        self.assertEqual(len(set(_symbols_from_args(universe))), 50)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_benchmarks_streaming_policy_inference_without_changing_mode(self):
        env = OptionsEnv(three_snapshot_dataset(), slot_count=1)
        observation, _ = env.reset(seed=3)
        model = build_recurrent_actor_critic(RecurrentConfig(
            input_size=observation_vector(observation).shape[0],
            slot_count=1,
            action_slot_count=env.action_shape[0],
            action_count=env.action_shape[1],
            hidden_size=4,
        ))
        model.train()

        report = benchmark_recurrent_inference(
            model,
            observation,
            2,
            warmup_iterations=1,
            measured_iterations=5,
        )

        self.assertTrue(model.training)
        self.assertEqual(
            report["schema_version"],
            "research-demo.inference-latency.v1",
        )
        self.assertEqual(report["measured_iterations"], 5)
        self.assertGreater(report["median_microseconds"], 0)
        self.assertGreaterEqual(
            report["p95_microseconds"],
            report["median_microseconds"],
        )
        self.assertGreater(report["torch_threads"], 0)

        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            benchmark_recurrent_inference(
                model,
                observation,
                2,
                warmup_iterations=-1,
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            benchmark_recurrent_inference(
                model,
                observation,
                2,
                measured_iterations=0,
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_train_and_save_auditable_checkpoint(self):
        env = OptionsEnv(demo_dataset(), slot_count=2, starting_cash=1_000)
        observation, _ = env.reset(seed=11)
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).shape[0],
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=8,
            kind="hybrid",
            encoder="graph",
            contract_feature_count=observation.contracts.shape[1],
            market_feature_count=observation.market.size,
            portfolio_feature_count=observation.portfolio.size,
            graph_hidden_size=8,
            masked_input_indices=(0,),
            auxiliary_target_count=len(AUXILIARY_TARGET_FEATURES),
        )
        training = TrainingConfig(
            episodes=2,
            sequence_length=2,
            evaluation_interval=1,
            seed=11,
            auxiliary_coefficient=0.05,
        )

        torch.manual_seed(training.seed)
        initial_auxiliary_weight = (
            build_recurrent_actor_critic(recurrent)
            .auxiliary.weight.detach().clone()
        )

        model, metrics = train_actor_critic(env, recurrent, training)

        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(math.isfinite(item["loss"]) for item in metrics))
        self.assertTrue(all(item["ppo_updates"] >= 1 for item in metrics))
        self.assertTrue(all(0 <= item["clip_fraction"] <= 1 for item in metrics))
        self.assertTrue(all(math.isfinite(item["approx_kl"]) for item in metrics))
        self.assertTrue(all(math.isfinite(item["auxiliary_loss"]) for item in metrics))
        self.assertTrue(all(item["auxiliary_loss"] >= 0 for item in metrics))
        self.assertTrue(all(
            item["auxiliary_target_coverage"]["t+1"]["underlyingReturn"] == 1
            for item in metrics
        ))
        self.assertTrue(all(math.isfinite(item["evaluation_total_reward"]) for item in metrics))
        self.assertTrue(all(math.isfinite(item["evaluation_selection_score"]) for item in metrics))
        self.assertTrue(all(0 <= item["requested_action_rate"] <= 1 for item in metrics))
        self.assertTrue(all(0 <= item["slot_churn_rate"] <= 1 for item in metrics))
        self.assertTrue(all(item["transition_seconds_mean"] == 60 for item in metrics))
        self.assertTrue(all(item["effective_gamma_mean"] > 0.99 for item in metrics))
        self.assertTrue(all(
            math.isclose(
                item["entropy_bonus"],
                training.entropy_coefficient * item["entropy"],
            )
            for item in metrics
        ))
        self.assertEqual(sum(item["selected_checkpoint"] for item in metrics), 1)
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        self.assertFalse(torch.equal(
            initial_auxiliary_weight,
            model.auxiliary.weight.detach(),
        ))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, env, recurrent, training, metrics)
            sidecar = json.loads(path.with_suffix(".pt.json").read_text())
            checkpoint = torch.load(path, weights_only=True)
            restored, restored_manifest = load_checkpoint(path)

        self.assertEqual(sidecar["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(sidecar["mode"], "research_demo")
        self.assertEqual(sidecar["algorithm"], "stateful_factorized_ppo")
        self.assertEqual(sidecar["action_policy"], {
            "factorization": "independent_masked_rows",
            "initial_hold_bias": 5.0,
            "hard_order_cap": None,
        })
        self.assertEqual(
            sidecar["temporal_training"],
            {
                "mode": "stateful_tbptt",
                "chunk_length": 2,
                "padding": "right_only_ignored",
                "discounting": {
                    "mode": "elapsed_wall_clock",
                    "gamma_per_reference_interval": 0.99,
                    "gae_lambda_per_reference_interval": 0.95,
                    "reference_seconds": 900.0,
                },
            },
        )
        self.assertEqual(sidecar["auxiliary_prediction"], {
            "enabled": True,
            "coefficient": 0.05,
            "targets": list(AUXILIARY_TARGET_FEATURES),
            "horizons": [1],
            "target_semantics": "cumulative_change_from_policy_state",
            "availability": "endpoint_point_in_time_coverage_mask",
            "inference_path": "excluded_from_policy_inference",
        })
        self.assertEqual(sidecar["selection"]["scope"], "in_sample_research_demo")
        self.assertEqual(sidecar["selection"]["metric"], "evaluation_selection_score")
        self.assertEqual(
            sidecar["selection"]["score_definition"],
            {
                "reward": "evaluation_total_reward",
                "drawdown_penalty": 0.0,
                "downside_penalty": 0.0,
                "turnover_penalty": 0.0,
                "cross_ticker_std_penalty": 0.0,
                "worst_ticker_weight": 0.0,
            },
        )
        self.assertEqual(
            sidecar["selection"]["early_stopping"],
            {
                "enabled": True,
                "patience": 3,
                "min_delta": 0.0,
                "completed_episodes": 2,
                "stopped_early": False,
            },
        )
        self.assertEqual(sidecar["model"]["kind"], "hybrid")
        self.assertEqual(sidecar["model"]["encoder"], "graph")
        self.assertEqual(sidecar["model"]["portfolio_feature_count"], 8)
        self.assertEqual(sidecar["model"]["action_slot_count"], 3)
        self.assertEqual(sidecar["model"]["initial_hold_bias"], 5.0)
        self.assertEqual(sidecar["model"]["masked_input_indices"], [0])
        self.assertEqual(sidecar["training"]["entropy_coefficient"], 1e-4)
        self.assertEqual(sidecar["environment"]["schema_version"], "research-demo.v13")
        self.assertEqual(sidecar["environment"]["starting_cash"], 1_000)
        self.assertEqual(sidecar["environment"]["slot_assignment"], "stable")
        self.assertEqual(sidecar["environment"]["spread_multiplier"], 1.0)
        self.assertEqual(sidecar["feature_vector_schema"], "dimensionless.v10")
        self.assertEqual(sidecar["provenance"], {})
        self.assertEqual(
            checkpoint["manifest"]["environment_fingerprint"],
            env.manifest.fingerprint,
        )
        self.assertEqual(len(sidecar["training_environments"]), 1)
        self.assertEqual(
            sidecar["training_environment_fingerprints"],
            {env.dataset.symbol: env.manifest.fingerprint},
        )
        self.assertIn("state_dict", checkpoint)
        self.assertEqual(restored_manifest["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(tuple(restored.config.masked_input_indices), (0,))
        for expected, actual in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, actual)

        reports = evaluate_recurrent_policy(
            env,
            restored,
            training.sequence_length,
            seeds=(21, 21),
        )
        self.assertEqual(reports[0], reports[1])

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_stateful_ppo_trains_contiguous_recurrent_chunks(self):
        env = OptionsEnv(three_snapshot_dataset(), slot_count=2)
        observation, _ = env.reset()
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=8,
            kind="hybrid",
        )

        model, metrics = train_actor_critic(
            env,
            recurrent,
            TrainingConfig(
                episodes=1,
                sequence_length=1,
                ppo_epochs=2,
                minibatch_size=2,
            ),
        )

        self.assertEqual(metrics[0]["steps"], 2)
        self.assertEqual(metrics[0]["recurrent_chunks"], 2)
        self.assertEqual(metrics[0]["ppo_updates"], 2)
        self.assertIn("mean_requested_orders_per_step", metrics[0])

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_training_integrates_seeded_random_regime_windows(self):
        source = demo_dataset().snapshots[0].frame
        snapshots = []
        for index in range(10):
            frame = source.copy()
            timestamp = f"2026-07-21T14:{index:02d}:00+00:00"
            frame["collectedAt"] = timestamp
            snapshots.append(Snapshot(timestamp, frame))
        env = OptionsEnv(
            SnapshotDataset(tuple(snapshots), "AAPL"),
            slot_count=2,
        )
        observation, _ = env.reset()
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=8,
        )
        config = TrainingConfig(
            episodes=5,
            sequence_length=2,
            max_steps=2,
            random_start=True,
            ppo_epochs=1,
            evaluation_interval=5,
            seed=41,
        )

        _, metrics = train_actor_critic(env, recurrent, config)
        expected_rng = np.random.default_rng(config.seed)
        expected_starts = [
            _sample_rollout_bounds(10, 2, True, expected_rng)[0]
            for _ in range(config.episodes)
        ]

        self.assertEqual(
            [item["rollout_start_index"] for item in metrics],
            expected_starts,
        )
        self.assertGreater(len(set(expected_starts)), 1)
        self.assertTrue(all(item["steps"] == 2 for item in metrics))
        self.assertTrue(all(
            item["rollout_end_index"] - item["rollout_start_index"] == 2
            for item in metrics
        ))

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_rejects_model_environment_shape_mismatch(self):
        env = OptionsEnv(demo_dataset(), slot_count=2)
        recurrent = RecurrentConfig(input_size=1, slot_count=2, action_count=7)
        with self.assertRaisesRegex(ValueError, "environment emits"):
            train_actor_critic(env, recurrent, TrainingConfig(episodes=1))

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_validation_environment_sets_selection_scope(self):
        source = demo_dataset()
        train_env = OptionsEnv(source, slot_count=2)
        validation_env = OptionsEnv(source, slot_count=2)
        observation, _ = train_env.reset()
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=train_env.action_shape[0],
            hidden_size=8,
        )

        _, metrics = train_actor_critic(
            train_env,
            recurrent,
            TrainingConfig(episodes=1, sequence_length=2),
            selection_env=validation_env,
        )

        self.assertEqual(metrics[0]["evaluation_scope"], "validation_research_demo")

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_selection_patience_stops_stalled_training(self):
        env = OptionsEnv(demo_dataset(), slot_count=2)
        observation, _ = env.reset()
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=8,
        )
        config = TrainingConfig(
            episodes=10,
            sequence_length=2,
            ppo_epochs=1,
            evaluation_interval=1,
            selection_patience=2,
        )

        fixed_report = SimpleNamespace(
            total_reward=1.0,
            max_drawdown=0.0,
            downside_deviation=0.0,
            turnover=0.0,
        )
        with patch(
            "trading_bot.training.trainer.evaluate_recurrent_policy",
            return_value=[fixed_report],
        ):
            _, metrics = train_actor_critic(env, recurrent, config)

        self.assertEqual(len(metrics), 3)
        self.assertEqual(
            [item["selection_improved"] for item in metrics],
            [1, 0, 0],
        )
        self.assertEqual(
            [item["selection_evaluations_without_improvement"] for item in metrics],
            [0, 1, 2],
        )
        self.assertEqual(metrics[-1]["early_stop_selection"], 1)
        self.assertEqual(metrics[0]["selected_checkpoint"], 1)

    def test_rejects_invalid_selection_stopping_configuration(self):
        with self.assertRaisesRegex(ValueError, "selection_patience"):
            TrainingConfig(selection_patience=0)
        with self.assertRaisesRegex(ValueError, "selection_min_delta"):
            TrainingConfig(selection_min_delta=float("nan"))
        with self.assertRaisesRegex(ValueError, "auxiliary_coefficient"):
            TrainingConfig(auxiliary_coefficient=-0.1)
        for invalid in ((), (0,), (2, 1), (1, 1)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "auxiliary_horizons",
            ):
                TrainingConfig(auxiliary_horizons=invalid)
        with self.assertRaisesRegex(ValueError, "max_steps"):
            TrainingConfig(
                max_steps=3,
                auxiliary_coefficient=0.05,
                auxiliary_horizons=(1, 4),
            )
        with self.assertRaisesRegex(ValueError, "risk penalties"):
            TrainingConfig(selection_drawdown_penalty=-1)
        with self.assertRaisesRegex(ValueError, "risk penalties"):
            TrainingConfig(selection_cross_ticker_std_penalty=-1)
        with self.assertRaisesRegex(ValueError, "time_aware_discounting"):
            TrainingConfig(time_aware_discounting=1)
        with self.assertRaisesRegex(ValueError, "discount_reference_seconds"):
            TrainingConfig(discount_reference_seconds=0)
        for invalid in (-0.1, 1.1, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "worst_ticker_weight",
            ):
                TrainingConfig(selection_worst_ticker_weight=invalid)

    def test_selection_score_combines_declared_validation_risks(self):
        report = SimpleNamespace(
            total_reward=0.02,
            max_drawdown=0.01,
            downside_deviation=0.002,
            turnover=0.5,
        )
        config = TrainingConfig(
            selection_drawdown_penalty=1.0,
            selection_downside_penalty=2.0,
            selection_turnover_penalty=0.1,
        )

        score = selection_score(report, config)

        self.assertAlmostEqual(score, -0.044)
        report.max_drawdown = -0.01
        with self.assertRaisesRegex(ValueError, "risk metrics"):
            selection_score(report, config)

    def test_aggregate_selection_score_penalizes_fragile_ticker_results(self):
        aggregate = aggregate_selection_scores(
            (1.0, -1.0),
            TrainingConfig(
                selection_cross_ticker_std_penalty=0.5,
                selection_worst_ticker_weight=0.25,
            ),
        )

        self.assertEqual(aggregate["mean"], 0.0)
        self.assertEqual(aggregate["worst"], -1.0)
        self.assertEqual(aggregate["standard_deviation"], 1.0)
        self.assertEqual(aggregate["score"], -0.75)
        with self.assertRaisesRegex(ValueError, "at least one"):
            aggregate_selection_scores((), TrainingConfig())
        with self.assertRaisesRegex(ValueError, "finite"):
            aggregate_selection_scores((float("nan"),), TrainingConfig())

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_shared_policy_balances_isolated_ticker_episodes_and_validation(self):
        base = three_snapshot_dataset()
        training_envs = (
            OptionsEnv(SnapshotDataset(base.snapshots, "AAA"), slot_count=2),
            OptionsEnv(SnapshotDataset(base.snapshots, "BBB"), slot_count=2),
        )
        selection_envs = (
            OptionsEnv(SnapshotDataset(base.snapshots, "AAA"), slot_count=2),
            OptionsEnv(SnapshotDataset(base.snapshots, "BBB"), slot_count=2),
        )
        observation, _ = training_envs[0].reset(seed=5)
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_slot_count=training_envs[0].action_shape[0],
            action_count=training_envs[0].action_shape[1],
            hidden_size=4,
        )
        config = TrainingConfig(
            episodes=2,
            sequence_length=2,
            ppo_epochs=1,
            minibatch_size=4,
            evaluation_interval=2,
            selection_patience=None,
            selection_cross_ticker_std_penalty=0.5,
            selection_worst_ticker_weight=0.25,
            seed=5,
        )

        model, metrics = train_actor_critic(
            training_envs,
            recurrent,
            config,
            selection_env=selection_envs,
        )

        self.assertEqual(
            {metric["training_symbol"] for metric in metrics},
            {"AAA", "BBB"},
        )
        self.assertEqual(
            metrics[-1]["evaluation_scope"],
            "validation_universe_research_demo",
        )
        self.assertEqual(
            set(metrics[-1]["evaluation_by_symbol"]),
            {"AAA", "BBB"},
        )
        per_ticker = [
            item["selection_score"]
            for item in metrics[-1]["evaluation_by_symbol"].values()
        ]
        expected = aggregate_selection_scores(per_ticker, config)
        self.assertAlmostEqual(
            metrics[-1]["evaluation_selection_score"],
            expected["score"],
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            save_checkpoint(
                path,
                model,
                training_envs,
                recurrent,
                config,
                metrics,
            )
            manifest = json.loads(
                path.with_suffix(".pt.json").read_text()
            )
        self.assertEqual(
            set(manifest["training_environment_fingerprints"]),
            {"AAA", "BBB"},
        )
        self.assertEqual(len(manifest["training_environments"]), 2)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_shared_policy_rejects_partial_ticker_coverage(self):
        base = three_snapshot_dataset()
        envs = (
            OptionsEnv(SnapshotDataset(base.snapshots, "AAA"), slot_count=2),
            OptionsEnv(SnapshotDataset(base.snapshots, "BBB"), slot_count=2),
        )
        observation, _ = envs[0].reset()
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_slot_count=envs[0].action_shape[0],
            action_count=envs[0].action_shape[1],
            hidden_size=4,
        )
        with self.assertRaisesRegex(ValueError, "one episode per ticker"):
            train_actor_critic(
                envs,
                recurrent,
                TrainingConfig(episodes=1),
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_auxiliary_training_rejects_missing_prediction_head(self):
        env = OptionsEnv(three_snapshot_dataset(), slot_count=2)
        observation, _ = env.reset()
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=4,
        )

        with self.assertRaisesRegex(ValueError, "auxiliary loss"):
            train_actor_critic(
                env,
                recurrent,
                TrainingConfig(episodes=1, auxiliary_coefficient=0.05),
            )
        incompatible = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=4,
            auxiliary_target_count=1,
        )
        with self.assertRaisesRegex(ValueError, "auxiliary target layout"):
            train_actor_critic(
                env,
                incompatible,
                TrainingConfig(episodes=1),
            )
        mismatched_horizons = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=4,
            auxiliary_target_count=len(AUXILIARY_TARGET_FEATURES),
            auxiliary_horizons=(1, 2),
        )
        with self.assertRaisesRegex(ValueError, "horizons do not match"):
            train_actor_critic(
                env,
                mismatched_horizons,
                TrainingConfig(episodes=1),
            )
        unavailable_horizon = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=4,
            auxiliary_target_count=2 * len(AUXILIARY_TARGET_FEATURES),
            auxiliary_horizons=(1, 4),
        )
        with self.assertRaisesRegex(ValueError, "shorter than"):
            train_actor_critic(
                env,
                unavailable_horizon,
                TrainingConfig(
                    episodes=1,
                    max_steps=None,
                    auxiliary_coefficient=0.05,
                    auxiliary_horizons=(1, 4),
                ),
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_risk_score_can_select_safer_lower_reward_checkpoint(self):
        env = OptionsEnv(demo_dataset(), slot_count=2)
        observation, _ = env.reset()
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=8,
        )
        risky = SimpleNamespace(
            total_reward=1.0,
            max_drawdown=0.9,
            downside_deviation=0.0,
            turnover=0.0,
        )
        safer = SimpleNamespace(
            total_reward=0.5,
            max_drawdown=0.0,
            downside_deviation=0.0,
            turnover=0.0,
        )
        with patch(
            "trading_bot.training.trainer.evaluate_recurrent_policy",
            side_effect=([risky], [safer]),
        ):
            _, metrics = train_actor_critic(
                env,
                recurrent,
                TrainingConfig(
                    episodes=2,
                    sequence_length=2,
                    ppo_epochs=1,
                    evaluation_interval=1,
                    selection_patience=None,
                    selection_drawdown_penalty=1.0,
                ),
            )

        self.assertAlmostEqual(metrics[0]["evaluation_selection_score"], 0.1)
        self.assertAlmostEqual(metrics[1]["evaluation_selection_score"], 0.5)
        self.assertEqual(metrics[1]["selected_checkpoint"], 1)

    def test_rejects_unknown_training_algorithm(self):
        with self.assertRaisesRegex(ValueError, "algorithm"):
            TrainingConfig(algorithm="q_learning")

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_reinforce_with_baseline_updates_recurrent_policy(self):
        env = OptionsEnv(three_snapshot_dataset(), slot_count=2)
        observation, _ = env.reset()
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).size,
            slot_count=2,
            action_count=7,
            action_slot_count=env.action_shape[0],
            hidden_size=8,
        )

        training = TrainingConfig(
            episodes=1,
            sequence_length=1,
            ppo_epochs=4,
            algorithm="reinforce",
        )
        model, metrics = train_actor_critic(
            env,
            recurrent,
            training,
        )

        self.assertEqual(metrics[0]["algorithm"], "reinforce")
        self.assertEqual(metrics[0]["ppo_updates"], 0)
        self.assertEqual(metrics[0]["reinforce_updates"], 1)
        self.assertEqual(
            metrics[0]["optimizer_updates"],
            metrics[0]["reinforce_updates"],
        )
        self.assertTrue(math.isfinite(metrics[0]["policy_loss"]))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reinforce.pt"
            save_checkpoint(
                path,
                model,
                env,
                recurrent,
                training,
                metrics,
            )
            _, manifest = load_checkpoint(path)

        self.assertEqual(
            manifest["algorithm"],
            "stateful_factorized_reinforce_baseline",
        )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_multi_horizon_training_supports_all_recurrent_learners_and_decoders(self):
        for kind in ("gru", "lstm", "hybrid", "mixture"):
            for algorithm in ("ppo", "reinforce"):
                for action_decoder in ("factorized", "single_leg"):
                    with self.subTest(
                        kind=kind,
                        algorithm=algorithm,
                        action_decoder=action_decoder,
                    ):
                        env = OptionsEnv(three_snapshot_dataset(), slot_count=2)
                        observation, _ = env.reset(seed=71)
                        recurrent = RecurrentConfig(
                            input_size=observation_vector(observation).size,
                            slot_count=2,
                            action_count=7,
                            action_slot_count=env.action_shape[0],
                            hidden_size=4,
                            kind=kind,
                            action_decoder=action_decoder,
                            auxiliary_target_count=(
                                2 * len(AUXILIARY_TARGET_FEATURES)
                            ),
                            auxiliary_horizons=(1, 2),
                        )
                        training = TrainingConfig(
                            episodes=1,
                            sequence_length=2,
                            ppo_epochs=1,
                            minibatch_size=2,
                            evaluation_interval=1,
                            auxiliary_coefficient=0.05,
                            auxiliary_horizons=(1, 2),
                            algorithm=algorithm,
                            seed=71,
                        )

                        model, metrics = train_actor_critic(
                            env,
                            recurrent,
                            training,
                        )

                        self.assertEqual(model.auxiliary.out_features, 10)
                        self.assertTrue(math.isfinite(metrics[0]["auxiliary_loss"]))
                        self.assertEqual(metrics[0]["action_decoder"], action_decoder)
                        self.assertEqual(
                            metrics[0]["action_likelihood_factors"],
                            1 if action_decoder == "single_leg" else 3,
                        )
                        self.assertEqual(metrics[0]["invalid_actions"], 0)
                        if action_decoder == "single_leg":
                            self.assertLessEqual(
                                metrics[0]["requested_option_orders"]
                                + metrics[0]["requested_underlying_orders"],
                                metrics[0]["steps"],
                            )
                        self.assertEqual(
                            metrics[0]["auxiliary_target_coverage"]["t+2"][
                                "underlyingReturn"
                            ],
                            0.5,
                        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "multi-horizon.pt"
            save_checkpoint(
                path,
                model,
                env,
                recurrent,
                training,
                metrics,
            )
            restored, manifest = load_checkpoint(path)
        self.assertEqual(restored.config.auxiliary_horizons, (1, 2))
        self.assertEqual(manifest["auxiliary_prediction"]["horizons"], [1, 2])
        self.assertEqual(manifest["training"]["algorithm"], "reinforce")
        self.assertEqual(manifest["algorithm"], "stateful_single_leg_joint_reinforce_baseline")
        self.assertEqual(manifest["action_policy"]["maximum_orders_per_step"], 1)
        self.assertEqual(manifest["action_policy"]["likelihood"], "exact_joint_categorical")

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_rejects_checkpoint_with_incompatible_feature_transform(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({
                "state_dict": {},
                "manifest": {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "feature_vector_schema": "raw.v0",
                    "model": {},
                },
            }, path)

            with self.assertRaisesRegex(ValueError, "feature-vector schema"):
                load_checkpoint(path)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_generalized_advantage_matches_terminal_return_vector(self):
        advantages, returns = _generalized_advantages(
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.5, 0.25]),
            next_value=99.0,
            terminal=True,
            discounts=torch.ones(2),
            trace_discounts=torch.ones(2),
            torch=torch,
        )

        torch.testing.assert_close(advantages, torch.tensor([1.5, 0.75]))
        torch.testing.assert_close(returns, torch.tensor([2.0, 1.0]))

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_discounted_monte_carlo_returns_handle_terminal_and_bootstrap(self):
        rewards = torch.tensor([1.0, 1.0])

        discounts = torch.full((2,), 0.5)
        terminal = _discounted_returns(
            rewards,
            99.0,
            True,
            discounts,
            torch,
        )
        bounded = _discounted_returns(
            rewards,
            2.0,
            False,
            discounts,
            torch,
        )

        torch.testing.assert_close(terminal, torch.tensor([1.5, 1.0]))
        torch.testing.assert_close(bounded, torch.tensor([2.0, 2.0]))

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_generalized_advantage_uses_each_transition_duration(self):
        advantages, returns = _generalized_advantages(
            torch.tensor([1.0, 1.0]),
            torch.zeros(2),
            next_value=99.0,
            terminal=True,
            discounts=torch.tensor([0.5, 0.25]),
            trace_discounts=torch.ones(2),
            torch=torch,
        )

        torch.testing.assert_close(advantages, torch.tensor([1.5, 1.0]))
        torch.testing.assert_close(returns, torch.tensor([1.5, 1.0]))

    def test_duration_adjusted_discounting_composes_in_physical_time(self):
        adjusted = _duration_adjusted_factors(
            [450.0, 900.0, 1_800.0],
            base=0.9,
            reference_seconds=900.0,
            time_aware=True,
        )
        fixed = _duration_adjusted_factors(
            [450.0, 900.0, 1_800.0],
            base=0.9,
            reference_seconds=900.0,
            time_aware=False,
        )

        np.testing.assert_allclose(adjusted, [0.9**0.5, 0.9, 0.9**2])
        np.testing.assert_allclose(fixed, [0.9, 0.9, 0.9])
        self.assertEqual(
            _elapsed_seconds(
                "2026-07-22T14:00:00+00:00",
                "2026-07-22T14:15:00+00:00",
            ),
            900.0,
        )
        self.assertEqual(_elapsed_seconds("1", "3.5"), 2.5)

    def test_rollout_segments_are_seeded_bounded_and_regime_diverse(self):
        first_rng = np.random.default_rng(31)
        second_rng = np.random.default_rng(31)
        first = [
            _sample_rollout_bounds(100, 30, True, first_rng)
            for _ in range(20)
        ]
        second = [
            _sample_rollout_bounds(100, 30, True, second_rng)
            for _ in range(20)
        ]

        self.assertEqual(first, second)
        self.assertGreater(len({start for start, _ in first}), 1)
        self.assertTrue(all(0 <= start <= 69 for start, _ in first))
        self.assertTrue(all(steps == 30 for _, steps in first))
        self.assertEqual(
            _sample_rollout_bounds(100, None, True, first_rng),
            (0, 99),
        )
        self.assertEqual(
            _sample_rollout_bounds(100, 30, False, first_rng),
            (0, 30),
        )
