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
    delta_notional_weight,
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
    BATCHED_RECURRENT_POLICY_STATE_SCHEMA_VERSION,
    CRITIC_BALANCE_DIAGNOSTIC_SCHEMA_VERSION,
    RECURRENT_POLICY_STATE_SCHEMA_VERSION,
    BatchedRecurrentPolicyState,
    BatchedStreamingRecurrentPolicy,
    RecurrentPolicyState,
    StreamingRecurrentPolicy,
    TrainingConfig,
    aggregate_selection_scores,
    batched_recurrent_policy,
    benchmark_batched_recurrent_inference,
    benchmark_recurrent_inference,
    critic_balance_diagnostics,
    evaluate_recurrent_policy,
    load_checkpoint,
    recurrent_policy,
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
from trading_bot.training.arena import (
    AGENT_ARENA_SCHEMA_VERSION,
    DEFAULT_ARENA_SYMBOLS,
    recurrent_arena_models,
    run_agent_arena,
)

__all__ = [
    "Action", "AGENT_ARENA_SCHEMA_VERSION",
    "BATCHED_RECURRENT_POLICY_STATE_SCHEMA_VERSION",
    "CRITIC_BALANCE_DIAGNOSTIC_SCHEMA_VERSION",
    "BatchedRecurrentPolicyState", "BatchedStreamingRecurrentPolicy",
    "BootstrapComparison", "CostScenario", "EpisodeReport",
    "EpisodeTrace", "EnvManifest", "FEATURE_ABLATION_GROUPS",
    "DEFAULT_ARENA_SYMBOLS", "LongVolatilityConfig", "ShortVolatilityConfig",
    "Observation",
    "ModelSpec", "OptionsEnv", "RecurrentPolicyState",
    "RECURRENT_POLICY_STATE_SCHEMA_VERSION", "StreamingRecurrentPolicy",
    "TrainingConfig", "Transition",
    "WalkForwardConfig", "WalkForwardSplit",
    "UNIVERSE_WALK_FORWARD_SCHEMA_VERSION",
    "batched_recurrent_policy", "benchmark_batched_recurrent_inference",
    "benchmark_recurrent_inference",
    "critic_balance_diagnostics",
    "aggregate_selection_scores",
    "buy_first_then_delta_hedge", "cash_secured_short_put_delta_hedge",
    "cost_stressed_environment",
    "delta_neutral", "delta_notional_weight", "evaluate_cost_stress",
    "evaluate_policy",
    "evaluate_recurrent_policy", "load_checkpoint",
    "feature_ablation_indices", "first_feasible",
    "long_volatility_delta_hedge", "no_op",
    "paired_moving_block_bootstrap", "recurrent_policy", "run_episode",
    "run_episode_trace",
    "recurrent_arena_models", "resolve_recurrent_config", "run_agent_arena",
    "run_walk_forward_training", "selection_score",
    "run_universe_walk_forward_training",
    "train_actor_critic",
    "walk_forward_splits",
]
