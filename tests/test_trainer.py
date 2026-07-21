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
from trading_bot.training.recurrent import RecurrentConfig
from trading_bot.training.sequence import observation_vector
from trading_bot.training.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainingConfig,
    _discounted_returns,
    _generalized_advantages,
    _sample_rollout_bounds,
    evaluate_recurrent_policy,
    load_checkpoint,
    save_checkpoint,
    selection_score,
    train_actor_critic,
)
from tests.test_training_env import demo_dataset, three_snapshot_dataset


class TrainerTests(TestCase):
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
        )
        training = TrainingConfig(
            episodes=2,
            sequence_length=2,
            evaluation_interval=1,
            seed=11,
        )

        model, metrics = train_actor_critic(env, recurrent, training)

        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(math.isfinite(item["loss"]) for item in metrics))
        self.assertTrue(all(item["ppo_updates"] >= 1 for item in metrics))
        self.assertTrue(all(0 <= item["clip_fraction"] <= 1 for item in metrics))
        self.assertTrue(all(math.isfinite(item["approx_kl"]) for item in metrics))
        self.assertTrue(all(math.isfinite(item["evaluation_total_reward"]) for item in metrics))
        self.assertTrue(all(math.isfinite(item["evaluation_selection_score"]) for item in metrics))
        self.assertTrue(all(0 <= item["requested_action_rate"] <= 1 for item in metrics))
        self.assertTrue(all(
            math.isclose(
                item["entropy_bonus"],
                training.entropy_coefficient * item["entropy"],
            )
            for item in metrics
        ))
        self.assertEqual(sum(item["selected_checkpoint"] for item in metrics), 1)
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
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
            },
        )
        self.assertEqual(sidecar["selection"]["scope"], "in_sample_research_demo")
        self.assertEqual(sidecar["selection"]["metric"], "evaluation_selection_score")
        self.assertEqual(
            sidecar["selection"]["score_definition"],
            {
                "reward": "evaluation_total_reward",
                "drawdown_penalty": 0.0,
                "downside_penalty": 0.0,
                "turnover_penalty": 0.0,
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
        self.assertEqual(sidecar["environment"]["schema_version"], "research-demo.v7")
        self.assertEqual(sidecar["environment"]["starting_cash"], 1_000)
        self.assertEqual(sidecar["environment"]["spread_multiplier"], 1.0)
        self.assertEqual(sidecar["feature_vector_schema"], "dimensionless.v5")
        self.assertEqual(sidecar["provenance"], {})
        self.assertEqual(
            checkpoint["manifest"]["environment_fingerprint"],
            env.manifest.fingerprint,
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
        with self.assertRaisesRegex(ValueError, "risk penalties"):
            TrainingConfig(selection_drawdown_penalty=-1)

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
        self.assertEqual(manifest["training"]["algorithm"], "reinforce")

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
        config = TrainingConfig(episodes=1, gamma=1.0, gae_lambda=1.0)
        advantages, returns = _generalized_advantages(
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.5, 0.25]),
            next_value=99.0,
            terminal=True,
            config=config,
            torch=torch,
        )

        torch.testing.assert_close(advantages, torch.tensor([1.5, 0.75]))
        torch.testing.assert_close(returns, torch.tensor([2.0, 1.0]))

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_discounted_monte_carlo_returns_handle_terminal_and_bootstrap(self):
        rewards = torch.tensor([1.0, 1.0])

        terminal = _discounted_returns(rewards, 99.0, True, 0.5, torch)
        bounded = _discounted_returns(rewards, 2.0, False, 0.5, torch)

        torch.testing.assert_close(terminal, torch.tensor([1.5, 1.0]))
        torch.testing.assert_close(bounded, torch.tensor([2.0, 2.0]))

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
