"""Reinforcement-learning research surfaces."""

from trading_bot.training.env import OptionsEnv
from trading_bot.training.evaluation import EpisodeReport, evaluate_policy, run_episode
from trading_bot.training.manifest import EnvManifest
from trading_bot.training.schemas import Action, Observation, Transition
from trading_bot.training.trainer import (
    TrainingConfig,
    evaluate_recurrent_policy,
    load_checkpoint,
    train_actor_critic,
)

__all__ = [
    "Action", "EpisodeReport", "EnvManifest", "Observation", "OptionsEnv",
    "TrainingConfig", "Transition", "evaluate_policy",
    "evaluate_recurrent_policy", "load_checkpoint", "run_episode",
    "train_actor_critic",
]
