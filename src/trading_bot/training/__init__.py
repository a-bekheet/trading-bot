"""Reinforcement-learning research surfaces."""

from trading_bot.training.baselines import (
    LongVolatilityConfig,
    ShortVolatilityConfig,
    buy_first_then_delta_hedge,
    cash_secured_short_put_delta_hedge,
    delta_neutral,
    first_feasible,
    long_volatility_delta_hedge,
    no_op,
)
from trading_bot.training.env import OptionsEnv
from trading_bot.training.evaluation import (
    BootstrapComparison,
    CostScenario,
    EpisodeReport,
    EpisodeTrace,
    cost_stressed_environment,
    evaluate_cost_stress,
    evaluate_policy,
    paired_moving_block_bootstrap,
    run_episode,
    run_episode_trace,
)
from trading_bot.training.manifest import EnvManifest
from trading_bot.training.schemas import Action, Observation, Transition
from trading_bot.training.sequence import (
    FEATURE_ABLATION_GROUPS,
    feature_ablation_indices,
)
from trading_bot.training.splits import WalkForwardSplit, walk_forward_splits
from trading_bot.training.trainer import (
    TrainingConfig,
    aggregate_selection_scores,
    benchmark_recurrent_inference,
    evaluate_recurrent_policy,
    load_checkpoint,
    selection_score,
    train_actor_critic,
)
from trading_bot.training.universe_walk_forward import (
    UNIVERSE_WALK_FORWARD_SCHEMA_VERSION,
    run_universe_walk_forward_training,
)
from trading_bot.training.walk_forward import (
    ModelSpec,
    WalkForwardConfig,
    resolve_recurrent_config,
    run_walk_forward_training,
)

__all__ = [
    "Action", "BootstrapComparison", "CostScenario", "EpisodeReport",
    "EpisodeTrace", "EnvManifest", "FEATURE_ABLATION_GROUPS",
    "LongVolatilityConfig", "ShortVolatilityConfig", "Observation",
    "ModelSpec", "OptionsEnv", "TrainingConfig", "Transition",
    "WalkForwardConfig", "WalkForwardSplit",
    "UNIVERSE_WALK_FORWARD_SCHEMA_VERSION",
    "benchmark_recurrent_inference",
    "aggregate_selection_scores",
    "buy_first_then_delta_hedge", "cash_secured_short_put_delta_hedge",
    "cost_stressed_environment",
    "delta_neutral", "evaluate_cost_stress", "evaluate_policy",
    "evaluate_recurrent_policy", "load_checkpoint",
    "feature_ablation_indices", "first_feasible",
    "long_volatility_delta_hedge", "no_op",
    "paired_moving_block_bootstrap", "run_episode", "run_episode_trace",
    "resolve_recurrent_config", "run_walk_forward_training", "selection_score",
    "run_universe_walk_forward_training",
    "train_actor_critic",
    "walk_forward_splits",
]
