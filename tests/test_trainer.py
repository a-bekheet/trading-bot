import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

try:
    import torch
except ImportError:
    torch = None

from trading_bot.training.env import OptionsEnv
from trading_bot.training.recurrent import RecurrentConfig
from trading_bot.training.sequence import observation_vector
from trading_bot.training.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainingConfig,
    _generalized_advantages,
    evaluate_recurrent_policy,
    load_checkpoint,
    save_checkpoint,
    train_actor_critic,
)
from tests.test_training_env import demo_dataset


class TrainerTests(TestCase):
    @skipUnless(torch is not None, "install the optional ml extra")
    def test_train_and_save_auditable_checkpoint(self):
        env = OptionsEnv(demo_dataset(), slot_count=2, starting_cash=1_000)
        observation, _ = env.reset(seed=11)
        recurrent = RecurrentConfig(
            input_size=observation_vector(observation).shape[0],
            slot_count=2,
            action_count=7,
            hidden_size=8,
            kind="hybrid",
            encoder="graph",
            contract_feature_count=observation.contracts.shape[1],
            market_feature_count=observation.market.size,
            portfolio_feature_count=observation.portfolio.size,
            graph_hidden_size=8,
        )
        training = TrainingConfig(episodes=2, sequence_length=2, seed=11)

        model, metrics = train_actor_critic(env, recurrent, training)

        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(math.isfinite(item["loss"]) for item in metrics))
        self.assertTrue(all(item["ppo_updates"] >= 1 for item in metrics))
        self.assertTrue(all(0 <= item["clip_fraction"] <= 1 for item in metrics))
        self.assertTrue(all(math.isfinite(item["approx_kl"]) for item in metrics))
        self.assertTrue(all(math.isfinite(item["evaluation_total_reward"]) for item in metrics))
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
        self.assertEqual(sidecar["algorithm"], "factorized_ppo")
        self.assertEqual(sidecar["selection"]["scope"], "in_sample_research_demo")
        self.assertEqual(sidecar["model"]["kind"], "hybrid")
        self.assertEqual(sidecar["model"]["encoder"], "graph")
        self.assertEqual(sidecar["model"]["portfolio_feature_count"], 7)
        self.assertEqual(sidecar["environment"]["schema_version"], "research-demo.v4")
        self.assertEqual(sidecar["environment"]["starting_cash"], 1_000)
        self.assertEqual(sidecar["environment"]["spread_multiplier"], 1.0)
        self.assertEqual(sidecar["feature_vector_schema"], "dimensionless.v2")
        self.assertEqual(sidecar["provenance"], {})
        self.assertEqual(
            checkpoint["manifest"]["environment_fingerprint"],
            env.manifest.fingerprint,
        )
        self.assertIn("state_dict", checkpoint)
        self.assertEqual(restored_manifest["schema_version"], CHECKPOINT_SCHEMA_VERSION)
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
