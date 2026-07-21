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
            graph_hidden_size=8,
        )
        training = TrainingConfig(episodes=2, sequence_length=2, seed=11)

        model, metrics = train_actor_critic(env, recurrent, training)

        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(math.isfinite(item["loss"]) for item in metrics))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, env, recurrent, training, metrics)
            sidecar = json.loads(path.with_suffix(".pt.json").read_text())
            checkpoint = torch.load(path, weights_only=True)

        self.assertEqual(sidecar["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(sidecar["mode"], "research_demo")
        self.assertEqual(sidecar["model"]["kind"], "hybrid")
        self.assertEqual(sidecar["model"]["encoder"], "graph")
        self.assertEqual(
            checkpoint["manifest"]["environment_fingerprint"],
            env.manifest.fingerprint,
        )
        self.assertIn("state_dict", checkpoint)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_rejects_model_environment_shape_mismatch(self):
        env = OptionsEnv(demo_dataset(), slot_count=2)
        recurrent = RecurrentConfig(input_size=1, slot_count=2, action_count=7)
        with self.assertRaisesRegex(ValueError, "environment emits"):
            train_actor_critic(env, recurrent, TrainingConfig(episodes=1))
