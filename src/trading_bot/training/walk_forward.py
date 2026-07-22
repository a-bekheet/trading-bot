"""Executable train/validation/test workflow for recurrent option policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from trading_bot.market_data.freshness import (
    DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS,
    underlying_quote_age,
)
from trading_bot.market_data.market_state import market_state_features
from trading_bot.training.baselines import (
    LongVolatilityConfig,
    ShortVolatilityConfig,
    UnderlyingTrendConfig,
    buy_first_then_delta_hedge,
    cash_secured_short_put_delta_hedge,
    first_feasible,
    long_volatility_delta_hedge,
    no_op,
    underlying_trend,
)
from trading_bot.training.dataset import SnapshotDataset
from trading_bot.training.env import CONTRACT_FEATURES, OptionsEnv
from trading_bot.training.evaluation import (
    DEFAULT_COST_SCENARIOS,
    cost_stressed_environment,
    paired_moving_block_bootstrap,
    run_episode,
    run_episode_trace,
)
from trading_bot.training.features import REALIZED_VOL_WINDOWS
from trading_bot.training.recurrent import (
    RecurrentConfig,
    build_recurrent_actor_critic,
)
from trading_bot.training.schemas import FEATURE_VECTOR_SCHEMA_VERSION
from trading_bot.training.sequence import (
    AUXILIARY_TARGET_FEATURES,
    FEATURE_ABLATION_GROUPS,
    feature_ablation_indices,
    normalize_auxiliary_target_exclusions,
    observation_vector,
)
from trading_bot.training.splits import walk_forward_splits
from trading_bot.training.trainer import (
    TrainingConfig,
    _environment_kwargs_from_args,
    benchmark_recurrent_inference,
    critic_balance_diagnostics,
    recurrent_policy,
    save_checkpoint,
    train_actor_critic,
)


WALK_FORWARD_SCHEMA_VERSION = "research-demo.walk-forward.v60"


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_size: int
    validation_size: int
    test_size: int
    embargo: int = 0
    step_size: int | None = None
    max_train_size: int | None = None
    training_seed_offsets: tuple[int, ...] = (0,)
    training_seed_worst_weight: float = 0.25
    training_seed_dispersion_penalty: float = 0.25
    selection_score_tolerance: float = 0.0
    test_seeds: tuple[int, ...] = (20_001,)
    bootstrap_samples: int = 2_000
    bootstrap_block_length: int | None = None
    bootstrap_confidence: float = 0.95
    bootstrap_min_observations: int = 20
    bootstrap_seed: int = 70_001
    long_volatility_window: int = 16
    long_volatility_min_coverage: float = 0.75
    long_volatility_min_edge: float = 0.02
    long_volatility_quantity: int = 1
    short_volatility_window: int = 16
    short_volatility_min_coverage: float = 0.75
    short_volatility_min_edge: float = 0.02
    short_volatility_quantity: int = 1
    trend_window: int = 16
    trend_min_coverage: float = 0.75
    trend_min_abs_log_return: float = 0.0
    trend_quantity: int = 1
    latency_warmup_iterations: int = 10
    latency_measured_iterations: int = 100
    max_median_inference_latency_us: float | None = None

    def __post_init__(self) -> None:
        if min(self.min_train_size, self.validation_size, self.test_size) < 2:
            raise ValueError("walk-forward partitions require at least two snapshots")
        if (
            not self.training_seed_offsets
            or len(set(self.training_seed_offsets))
            != len(self.training_seed_offsets)
            or any(
                not isinstance(offset, int) or isinstance(offset, bool)
                for offset in self.training_seed_offsets
            )
            or any(offset < 0 for offset in self.training_seed_offsets)
        ):
            raise ValueError(
                "training_seed_offsets must be unique non-negative integers"
            )
        if not 0 <= self.training_seed_worst_weight <= 1:
            raise ValueError(
                "training_seed_worst_weight must be between zero and one"
            )
        if (
            not math.isfinite(self.training_seed_dispersion_penalty)
            or self.training_seed_dispersion_penalty < 0
        ):
            raise ValueError(
                "training_seed_dispersion_penalty must be finite and non-negative"
            )
        if (
            not math.isfinite(self.selection_score_tolerance)
            or self.selection_score_tolerance < 0
        ):
            raise ValueError(
                "selection_score_tolerance must be finite and non-negative"
            )
        if len(self.test_seeds) != 1:
            raise ValueError(
                "deterministic held-out evaluation requires exactly one seed; "
                "repeated seeds are not independent paths"
            )
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if (
            self.bootstrap_block_length is not None
            and self.bootstrap_block_length < 1
        ):
            raise ValueError("bootstrap_block_length must be positive")
        if not 0 < self.bootstrap_confidence < 1:
            raise ValueError("bootstrap_confidence must be between zero and one")
        if self.bootstrap_min_observations < 2:
            raise ValueError("bootstrap_min_observations must be at least two")
        if self.latency_warmup_iterations < 0:
            raise ValueError("latency_warmup_iterations cannot be negative")
        if self.latency_measured_iterations < 1:
            raise ValueError("latency_measured_iterations must be positive")
        if self.max_median_inference_latency_us is not None and (
            not math.isfinite(self.max_median_inference_latency_us)
            or self.max_median_inference_latency_us <= 0
        ):
            raise ValueError(
                "max_median_inference_latency_us must be finite and positive"
            )
        LongVolatilityConfig(
            realized_window=self.long_volatility_window,
            min_coverage=self.long_volatility_min_coverage,
            min_volatility_edge=self.long_volatility_min_edge,
            quantity=self.long_volatility_quantity,
        )
        ShortVolatilityConfig(
            realized_window=self.short_volatility_window,
            min_coverage=self.short_volatility_min_coverage,
            min_volatility_edge=self.short_volatility_min_edge,
            quantity=self.short_volatility_quantity,
        )
        UnderlyingTrendConfig(
            return_window=self.trend_window,
            min_coverage=self.trend_min_coverage,
            min_abs_log_return=self.trend_min_abs_log_return,
            quantity=self.trend_quantity,
        )


@dataclass(frozen=True)
class ModelSpec:
    kind: str = "gru"
    encoder: str = "flat"
    hidden_size: int = 128
    layers: int = 1
    dropout: float = 0.0
    graph_hidden_size: int = 32
    graph_layers: int = 2
    graph_neighbors: int = 3
    attention_heads: int = 4
    initial_hold_bias: float = 5.0
    action_decoder: str = "factorized"
    disabled_feature_groups: tuple[str, ...] = ()
    algorithm: str = "ppo"
    parameter_budget: int | None = None
    auxiliary_coefficient: float | None = None
    auxiliary_horizons: tuple[int, ...] = (1,)
    auxiliary_target_exclusions: tuple[str, ...] = ()
    delta_neutrality_coefficient: float | None = None
    time_aware_discounting: bool | None = None
    burn_in_steps: int | None = None
    start_sampling: str | None = None
    factorized_ppo_objective: str | None = None
    entropy_objective: str | None = None
    critic_layer_norm: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"gru", "lstm", "hybrid", "mixture"}:
            raise ValueError(
                "model kind must be gru, lstm, hybrid, or mixture"
            )
        if self.encoder not in {
            "flat", "graph", "graph_set", "surface_graph_set", "attention_set",
        }:
            raise ValueError(
                "model encoder must be flat, graph, graph_set, "
                "surface_graph_set, or attention_set"
            )
        if min(
            self.hidden_size,
            self.layers,
            self.graph_hidden_size,
            self.graph_layers,
        ) < 1:
            raise ValueError("model sizes and layers must be positive")
        if self.graph_neighbors < 0:
            raise ValueError("model graph neighbors cannot be negative")
        if self.encoder == "attention_set":
            object.__setattr__(self, "graph_neighbors", 0)
        if self.attention_heads < 1:
            raise ValueError("model attention_heads must be positive")
        if (
            self.encoder == "attention_set"
            and self.graph_hidden_size % self.attention_heads
        ):
            raise ValueError(
                "attention_set graph_hidden_size must be divisible by attention_heads"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("model dropout must be in [0, 1)")
        if self.algorithm not in {"ppo", "reinforce"}:
            raise ValueError("model algorithm must be ppo or reinforce")
        if self.action_decoder not in {"factorized", "single_leg"}:
            raise ValueError(
                "model action_decoder must be factorized or single_leg"
            )
        if self.parameter_budget is not None and self.parameter_budget < 1:
            raise ValueError("parameter_budget must be positive when provided")
        if self.auxiliary_coefficient is not None and (
            not math.isfinite(self.auxiliary_coefficient)
            or self.auxiliary_coefficient < 0
        ):
            raise ValueError(
                "model auxiliary_coefficient must be finite and non-negative"
            )
        if self.delta_neutrality_coefficient is not None and (
            not math.isfinite(self.delta_neutrality_coefficient)
            or self.delta_neutrality_coefficient < 0
        ):
            raise ValueError(
                "model delta_neutrality_coefficient must be finite and "
                "non-negative"
            )
        if self.time_aware_discounting is not None and not isinstance(
            self.time_aware_discounting,
            bool,
        ):
            raise ValueError(
                "model time_aware_discounting must be a boolean or None"
            )
        if self.burn_in_steps is not None and (
            not isinstance(self.burn_in_steps, int)
            or isinstance(self.burn_in_steps, bool)
            or self.burn_in_steps < 0
        ):
            raise ValueError(
                "model burn_in_steps must be a non-negative integer or None"
            )
        if self.start_sampling not in {
            None,
            "uniform",
            "volatility_stratified",
        }:
            raise ValueError(
                "model start_sampling must be uniform, "
                "volatility_stratified, or None"
            )
        if self.factorized_ppo_objective not in {
            None,
            "joint",
            "dimensionwise",
        }:
            raise ValueError(
                "model factorized_ppo_objective must be joint, "
                "dimensionwise, or None"
            )
        if self.factorized_ppo_objective == "dimensionwise" and (
            self.algorithm != "ppo" or self.action_decoder != "factorized"
        ):
            raise ValueError(
                "dimensionwise objective requires factorized PPO"
            )
        if self.entropy_objective not in {
            None,
            "feasible_normalized",
            "raw_mean",
        }:
            raise ValueError(
                "model entropy_objective must be feasible_normalized, "
                "raw_mean, or None"
            )
        if not isinstance(self.critic_layer_norm, bool):
            raise ValueError("model critic_layer_norm must be a boolean")
        if (
            self.critic_layer_norm
            and self.kind != "hybrid"
            and self.hidden_size < 2
        ):
            raise ValueError(
                "model critic_layer_norm requires a critic width of at "
                "least two"
            )
        normalized_horizons = tuple(self.auxiliary_horizons)
        if (
            not normalized_horizons
            or any(
                not isinstance(horizon, int) or isinstance(horizon, bool)
                for horizon in normalized_horizons
            )
            or any(horizon < 1 for horizon in normalized_horizons)
            or tuple(sorted(set(normalized_horizons))) != normalized_horizons
        ):
            raise ValueError(
                "model auxiliary_horizons must be unique positive increasing integers"
            )
        object.__setattr__(self, "auxiliary_horizons", normalized_horizons)
        object.__setattr__(
            self,
            "auxiliary_target_exclusions",
            normalize_auxiliary_target_exclusions(
                self.auxiliary_target_exclusions
            ),
        )
        feature_ablation_indices(self.disabled_feature_groups, 1)

    @property
    def identifier(self) -> str:
        """Return a stable identifier for artifact joins."""
        canonical = json.dumps(
            {
                "feature_vector_schema": FEATURE_VECTOR_SCHEMA_VERSION,
                "model": asdict(self),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()[:10]
        return f"{self.encoder}-{self.kind}-{self.action_decoder}-{digest}"

    def build(self, env: OptionsEnv) -> RecurrentConfig:
        observation, _ = env.reset(seed=0)
        return RecurrentConfig(
            input_size=observation_vector(observation).shape[0],
            slot_count=env.slot_count,
            action_count=env.action_shape[1],
            action_slot_count=env.action_shape[0],
            hidden_size=self.hidden_size,
            layers=self.layers,
            kind=self.kind,
            dropout=self.dropout,
            encoder=self.encoder,
            contract_feature_count=observation.contracts.shape[1],
            market_feature_count=observation.market.size,
            portfolio_feature_count=observation.portfolio.size,
            graph_hidden_size=self.graph_hidden_size,
            graph_layers=self.graph_layers,
            graph_neighbors=self.graph_neighbors,
            attention_heads=self.attention_heads,
            initial_hold_bias=self.initial_hold_bias,
            action_decoder=self.action_decoder,
            auxiliary_target_count=(
                len(AUXILIARY_TARGET_FEATURES) * len(self.auxiliary_horizons)
            ),
            auxiliary_horizons=self.auxiliary_horizons,
            critic_layer_norm=self.critic_layer_norm,
            masked_input_indices=feature_ablation_indices(
                self.disabled_feature_groups,
                env.slot_count,
            ),
            graph_relation_indices=tuple(
                CONTRACT_FEATURES.index(name)
                for name in (
                    ("forwardLogMoneyness", "dteDays")
                    if self.encoder == "surface_graph_set"
                    else (
                        "impliedVolatility",
                        "delta",
                        "logMoneyness",
                        "dteDays",
                    )
                )
            ),
            graph_option_side_index=(
                CONTRACT_FEATURES.index("delta")
                if self.encoder == "surface_graph_set"
                else None
            ),
        )


def resolve_recurrent_config(
    model_spec: ModelSpec,
    env: OptionsEnv,
) -> tuple[RecurrentConfig, int]:
    """Resolve the widest recurrent state within a train-layout-only budget."""
    requested = model_spec.build(env)

    def parameter_count(config: RecurrentConfig) -> int:
        model = build_recurrent_actor_critic(config)
        return sum(parameter.numel() for parameter in model.parameters())

    if model_spec.parameter_budget is None:
        return requested, parameter_count(requested)

    minimum_hidden_size = (
        2
        if model_spec.critic_layer_norm and model_spec.kind != "hybrid"
        else 1
    )
    minimum = replace(requested, hidden_size=minimum_hidden_size)
    minimum_count = parameter_count(minimum)
    if minimum_count > model_spec.parameter_budget:
        raise ValueError(
            f"parameter_budget={model_spec.parameter_budget} is below the "
            f"minimum {minimum_count} for {model_spec.encoder}:{model_spec.kind}"
        )

    low = minimum_hidden_size
    high = model_spec.hidden_size
    best = minimum
    best_count = minimum_count
    while low <= high:
        hidden_size = (low + high) // 2
        candidate = replace(requested, hidden_size=hidden_size)
        count = parameter_count(candidate)
        if count <= model_spec.parameter_budget:
            best = candidate
            best_count = count
            low = hidden_size + 1
        else:
            high = hidden_size - 1
    return best, best_count


def _reports_to_dict(reports) -> list[dict[str, Any]]:
    return [report.to_dict() for report in reports]


def _traces_to_dict(traces) -> list[dict[str, Any]]:
    """Serialize auditable paths without coupling the UI to Torch objects."""
    return [
        {
            "report": trace.report.to_dict(),
            "timestamps": list(trace.timestamps),
            "step_returns": list(trace.step_returns),
            "navs": list(trace.navs),
            "decisions": list(trace.decisions),
        }
        for trace in traces
    ]


def _partition_data_quality(dataset: SnapshotDataset) -> dict[str, Any]:
    """Persist the execution-provenance coverage behind a result path."""
    session_coverage = []
    regular = []
    quote_time_coverage = []
    for snapshot in dataset.snapshots:
        first = snapshot.frame.iloc[0]
        is_regular, has_session = market_state_features(
            first.get("marketState")
        )
        _, has_quote_time = underlying_quote_age(
            snapshot.timestamp,
            first.get("underlyingQuoteTime"),
        )
        session_coverage.append(float(has_session))
        regular.append(float(is_regular))
        quote_time_coverage.append(float(has_quote_time))
    count = len(dataset)
    session_rate = sum(session_coverage) / count
    regular_rate = sum(regular) / count
    quote_time_rate = sum(quote_time_coverage) / count
    if session_rate == 0:
        execution_provenance = "legacy_unknown_session_fallback"
    elif regular_rate == 1:
        execution_provenance = "provider_confirmed_regular"
    else:
        execution_provenance = "provider_nonregular_present"
    return {
        "snapshot_count": count,
        "market_state_coverage": session_rate,
        "regular_session_fraction": regular_rate,
        "underlying_quote_time_coverage": quote_time_rate,
        "execution_provenance": execution_provenance,
        "first_timestamp": dataset.snapshots[0].timestamp,
        "last_timestamp": dataset.snapshots[-1].timestamp,
    }


def _normalize_model_specs(
    model_spec: ModelSpec | Sequence[ModelSpec],
) -> tuple[ModelSpec, ...]:
    specs = (
        (model_spec,)
        if isinstance(model_spec, ModelSpec)
        else tuple(model_spec)
    )
    if not specs:
        raise ValueError("at least one model candidate is required")
    if not all(isinstance(spec, ModelSpec) for spec in specs):
        raise TypeError("model candidates must be ModelSpec instances")
    identifiers = [spec.identifier for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("model candidates must be unique")
    return specs


def _selected_metric(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next(item for item in metrics if item["selected_checkpoint"])
    score = selected["evaluation_selection_score"]
    if score is None or not math.isfinite(float(score)):
        raise ValueError("candidate validation selection score must be finite")
    return selected


def _start_sampling_evidence(
    metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize training-only window-start coverage for one replicate."""
    effective = Counter(
        str(item["rollout_start_sampling_effective"])
        for item in metrics
    )
    regime_bins = Counter(
        int(item["rollout_start_regime_bin"])
        for item in metrics
        if item["rollout_start_regime_bin"] is not None
    )
    fallbacks = Counter(
        str(item["rollout_start_sampling_fallback_reason"])
        for item in metrics
        if item["rollout_start_sampling_fallback_reason"] is not None
    )
    return {
        "requested": metrics[0]["rollout_start_sampling_requested"],
        "effective_episode_counts": dict(sorted(effective.items())),
        "regime_bin_episode_counts": {
            str(key): value for key, value in sorted(regime_bins.items())
        },
        "fallback_episode_counts": dict(sorted(fallbacks.items())),
        "available_regime_bin_count": max(
            int(item["rollout_start_regime_bin_count"])
            for item in metrics
        ),
    }


def _entropy_evidence(
    metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize mask-aware exploration diagnostics for one replicate."""
    normalized = tuple(
        float(item["feasible_normalized_action_entropy"])
        for item in metrics
    )
    return {
        "objective": metrics[0]["entropy_objective"],
        "episode_mean_effective_entropy": statistics.fmean(
            float(item["entropy"]) for item in metrics
        ),
        "episode_mean_raw_entropy": statistics.fmean(
            float(item["raw_action_entropy"]) for item in metrics
        ),
        "episode_mean_feasible_normalized_entropy": statistics.fmean(
            normalized
        ),
        "minimum_feasible_normalized_entropy": min(normalized),
        "maximum_feasible_normalized_entropy": max(normalized),
        "episode_mean_explorable_factor_fraction": statistics.fmean(
            float(item["explorable_action_factor_fraction"])
            for item in metrics
        ),
    }


def _training_seed_aggregate(
    runs: Sequence[dict[str, Any]],
    config: WalkForwardConfig,
) -> dict[str, Any]:
    """Aggregate validation-only seed replicates without cherry-picking one."""
    if not runs:
        raise ValueError("at least one training-seed replicate is required")
    scores = tuple(float(run["validation_selection_score"]) for run in runs)
    rewards = tuple(float(run["validation_total_reward"]) for run in runs)
    if not all(math.isfinite(value) for value in (*scores, *rewards)):
        raise ValueError("training-seed validation metrics must be finite")
    score_mean = statistics.fmean(scores)
    score_worst = min(scores)
    score_std = statistics.pstdev(scores)
    score_standard_error = (
        statistics.stdev(scores) / math.sqrt(len(scores))
        if len(scores) > 1
        else 0.0
    )
    aggregate_score = (
        (1.0 - config.training_seed_worst_weight) * score_mean
        + config.training_seed_worst_weight * score_worst
        - config.training_seed_dispersion_penalty * score_std
    )
    median_score = statistics.median(scores)
    representative = min(
        runs,
        key=lambda run: (
            abs(float(run["validation_selection_score"]) - median_score),
            run["optimizer_updates"],
            run["training_seed"],
        ),
    )
    return {
        "training_seed_count": len(runs),
        "training_seeds": [run["training_seed"] for run in runs],
        "validation_selection_score_mean": score_mean,
        "validation_selection_score_worst": score_worst,
        "validation_selection_score_std": score_std,
        "validation_selection_score_standard_error": score_standard_error,
        "validation_total_reward_mean": statistics.fmean(rewards),
        "robust_training_seed_validation_score": aggregate_score,
        "worst_weight": config.training_seed_worst_weight,
        "dispersion_penalty": config.training_seed_dispersion_penalty,
        "representative_rule": "closest_to_median_validation_score",
        "representative_training_seed": representative["training_seed"],
        "representative_run": representative,
    }


def _select_seed_robust_group(
    groups: Sequence[dict[str, Any]],
    *,
    minimum_score_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Apply a one-standard-error rule, then prefer simpler deployment."""
    if not math.isfinite(minimum_score_tolerance) or minimum_score_tolerance < 0:
        raise ValueError("minimum_score_tolerance must be finite and non-negative")

    def delta_neutrality_enabled(group: dict[str, Any]) -> bool:
        representative = group["representative"]
        training = representative.get("training_config")
        coefficient = (
            training.delta_neutrality_coefficient
            if training is not None
            else representative["model_spec"].delta_neutrality_coefficient
        )
        return bool(coefficient is not None and coefficient > 0)

    eligible = [group for group in groups if group["latency_eligible"]]
    if not eligible:
        raise ValueError("at least one latency-eligible candidate is required")

    def tie_break(group: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(
                group["representative"][
                    "model_spec"
                ].factorized_ppo_objective == "dimensionwise"
            ),
            int(
                group["representative"]["model_spec"].entropy_objective
                == "raw_mean"
            ),
            int(delta_neutrality_enabled(group)),
            int(
                group["representative"]["model_spec"].critic_layer_norm
            ),
            int(bool(
                group["representative"][
                    "model_spec"
                ].auxiliary_target_exclusions
            )),
            max(
                run["inference_latency"]["median_microseconds"]
                for run in group["replicates"]
            ),
            group["representative"]["parameter_count"],
            group["representative"]["active_input_count"],
            sum(
                run["optimizer_updates"] for run in group["replicates"]
            ),
            int(group["representative"]["model_spec"].burn_in_steps == 0),
            int(
                group["representative"][
                    "model_spec"
                ].time_aware_discounting is False
            ),
            int(
                group["representative"]["model_spec"].start_sampling
                == "uniform"
            ),
            group["representative"]["model_id"],
        )

    raw_best = min(
        eligible,
        key=lambda group: (
            -group["aggregate"]["robust_training_seed_validation_score"],
            *tie_break(group),
        ),
    )
    best_score = float(
        raw_best["aggregate"]["robust_training_seed_validation_score"]
    )
    uncertainty_tolerance = float(
        raw_best["aggregate"]["validation_selection_score_standard_error"]
    )
    effective_tolerance = max(
        minimum_score_tolerance,
        uncertainty_tolerance,
    )
    competitive = [
        group
        for group in eligible
        if float(
            group["aggregate"]["robust_training_seed_validation_score"]
        )
        >= best_score - effective_tolerance
    ]
    selected = min(competitive, key=tie_break)
    selected_score = float(
        selected["aggregate"]["robust_training_seed_validation_score"]
    )
    selected["selection_rule"] = {
        "rule": "one_standard_error_with_materiality_floor",
        "best_model_id": raw_best["representative"]["model_id"],
        "best_score": best_score,
        "minimum_score_tolerance": minimum_score_tolerance,
        "uncertainty_tolerance": uncertainty_tolerance,
        "effective_score_tolerance": effective_tolerance,
        "competitive_candidate_count": len(competitive),
        "competitive_model_ids": [
            group["representative"]["model_id"] for group in competitive
        ],
        "selected_model_id": selected["representative"]["model_id"],
        "selected_score": selected_score,
        "score_sacrificed_for_simplicity": best_score - selected_score,
    }
    return selected


def run_walk_forward_training(
    dataset: SnapshotDataset,
    walk_forward_config: WalkForwardConfig,
    model_spec: ModelSpec | Sequence[ModelSpec],
    training_config: TrainingConfig,
    output_dir: Path,
    *,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select architectures on validation, then evaluate only the winner on test."""
    model_specs = _normalize_model_specs(model_spec)
    folds = walk_forward_splits(
        len(dataset),
        min_train_size=walk_forward_config.min_train_size,
        validation_size=walk_forward_config.validation_size,
        test_size=walk_forward_config.test_size,
        embargo=walk_forward_config.embargo,
        step_size=walk_forward_config.step_size,
        max_train_size=walk_forward_config.max_train_size,
    )
    if not folds:
        raise ValueError("dataset is too short for the requested walk-forward split")

    output_dir.mkdir(parents=True, exist_ok=True)
    environment_options = dict(env_kwargs or {})
    long_volatility_config = LongVolatilityConfig(
        realized_window=walk_forward_config.long_volatility_window,
        min_coverage=walk_forward_config.long_volatility_min_coverage,
        min_volatility_edge=walk_forward_config.long_volatility_min_edge,
        quantity=walk_forward_config.long_volatility_quantity,
    )
    short_volatility_config = ShortVolatilityConfig(
        realized_window=walk_forward_config.short_volatility_window,
        min_coverage=walk_forward_config.short_volatility_min_coverage,
        min_volatility_edge=walk_forward_config.short_volatility_min_edge,
        quantity=walk_forward_config.short_volatility_quantity,
    )
    trend_config = UnderlyingTrendConfig(
        return_window=walk_forward_config.trend_window,
        min_coverage=walk_forward_config.trend_min_coverage,
        min_abs_log_return=walk_forward_config.trend_min_abs_log_return,
        quantity=walk_forward_config.trend_quantity,
    )
    fold_results = []
    resolved_configs: dict[str, tuple[RecurrentConfig, int]] = {}
    environment_contract: dict[str, Any] | None = None
    for fold in folds:
        train_data, validation_data, test_data = fold.apply(dataset)
        train_env = OptionsEnv(train_data, **environment_options)
        validation_env = OptionsEnv(validation_data, **environment_options)
        if environment_contract is None:
            environment_contract = train_env.manifest.to_dict()
            for partition_field in ("data_hash", "symbol", "seed"):
                environment_contract.pop(partition_field, None)
        fold_training = replace(
            training_config,
            seed=training_config.seed + fold.fold,
        )
        benchmark_observation, _ = train_env.reset(seed=fold_training.seed)
        candidate_runs = []
        for candidate in model_specs:
            resolved = resolved_configs.get(candidate.identifier)
            if resolved is None:
                resolved = resolve_recurrent_config(candidate, train_env)
                resolved_configs[candidate.identifier] = resolved
            recurrent_config, resolved_parameter_count = resolved
            candidate_training_base = replace(
                fold_training,
                algorithm=candidate.algorithm,
                auxiliary_horizons=candidate.auxiliary_horizons,
                auxiliary_target_exclusions=(
                    candidate.auxiliary_target_exclusions
                ),
                auxiliary_coefficient=(
                    fold_training.auxiliary_coefficient
                    if candidate.auxiliary_coefficient is None
                    else candidate.auxiliary_coefficient
                ),
                delta_neutrality_coefficient=(
                    fold_training.delta_neutrality_coefficient
                    if candidate.delta_neutrality_coefficient is None
                    else candidate.delta_neutrality_coefficient
                ),
                time_aware_discounting=(
                    fold_training.time_aware_discounting
                    if candidate.time_aware_discounting is None
                    else candidate.time_aware_discounting
                ),
                burn_in_steps=(
                    fold_training.burn_in_steps
                    if candidate.burn_in_steps is None
                    else candidate.burn_in_steps
                ),
                start_sampling=(
                    fold_training.start_sampling
                    if candidate.start_sampling is None
                    else candidate.start_sampling
                ),
                factorized_ppo_objective=(
                    candidate.factorized_ppo_objective
                    if candidate.factorized_ppo_objective is not None
                    else (
                        fold_training.factorized_ppo_objective
                        if candidate.algorithm == "ppo"
                        and candidate.action_decoder == "factorized"
                        else "joint"
                    )
                ),
                entropy_objective=(
                    fold_training.entropy_objective
                    if candidate.entropy_objective is None
                    else candidate.entropy_objective
                ),
            )
            for seed_offset in walk_forward_config.training_seed_offsets:
                candidate_training = replace(
                    candidate_training_base,
                    seed=candidate_training_base.seed + seed_offset,
                )
                model, metrics = train_actor_critic(
                    train_env,
                    recurrent_config,
                    candidate_training,
                    selection_env=validation_env,
                )
                actual_parameter_count = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                if actual_parameter_count != resolved_parameter_count:
                    raise RuntimeError(
                        "trained model parameter count does not match its "
                        "resolved configuration"
                    )
                inference_latency = benchmark_recurrent_inference(
                    model,
                    benchmark_observation,
                    candidate_training.sequence_length,
                    warmup_iterations=(
                        walk_forward_config.latency_warmup_iterations
                    ),
                    measured_iterations=(
                        walk_forward_config.latency_measured_iterations
                    ),
                )
                latency_eligible = (
                    walk_forward_config.max_median_inference_latency_us is None
                    or inference_latency["median_microseconds"]
                    <= walk_forward_config.max_median_inference_latency_us
                )
                selected = _selected_metric(metrics)
                slot_changed_count = sum(
                    item["slot_changed_count"] for item in metrics
                )
                slot_comparable_count = sum(
                    item["slot_comparable_count"] for item in metrics
                )
                candidate_runs.append({
                    "model_id": candidate.identifier,
                    "replicate_id": (
                        f"{candidate.identifier}-seed-{candidate_training.seed}"
                    ),
                    "training_seed": candidate_training.seed,
                    "model_spec": candidate,
                    "recurrent_config": recurrent_config,
                    "model": model,
                    "metrics": metrics,
                    "start_sampling_evidence": _start_sampling_evidence(
                        metrics
                    ),
                    "entropy_evidence": _entropy_evidence(metrics),
                    "critic_balance_evidence": critic_balance_diagnostics(
                        metrics
                    ),
                    "training_config": candidate_training,
                    "parameter_count": resolved_parameter_count,
                    "inference_latency": inference_latency,
                    "latency_eligible": latency_eligible,
                    "masked_input_count": len(
                        recurrent_config.masked_input_indices
                    ),
                    "active_input_count": (
                        recurrent_config.input_size
                        - len(recurrent_config.masked_input_indices)
                    ),
                    "selected_episode": selected["episode"],
                    "validation_total_reward": float(
                        selected["evaluation_total_reward"]
                    ),
                    "validation_selection_score": float(
                        selected["evaluation_selection_score"]
                    ),
                    "validation_max_drawdown": float(
                        selected["evaluation_max_drawdown"]
                    ),
                    "validation_downside_deviation": float(
                        selected["evaluation_downside_deviation"]
                    ),
                    "validation_turnover": float(
                        selected["evaluation_turnover"]
                    ),
                    "validation_abs_beta_to_underlying": float(
                        selected["evaluation_abs_beta_to_underlying"]
                    ),
                    "validation_mean_abs_delta_notional_weight": float(
                        selected[
                            "evaluation_mean_abs_delta_notional_weight"
                        ]
                    ),
                    "selection_scope": selected["evaluation_scope"],
                    "episodes_completed": len(metrics),
                    "stopped_early": bool(
                        metrics[-1]["early_stop_selection"]
                    ),
                    "optimizer_updates": sum(
                        item["optimizer_updates"] for item in metrics
                    ),
                    "slot_changed_count": slot_changed_count,
                    "slot_comparable_count": slot_comparable_count,
                    "slot_churn_rate": (
                        slot_changed_count / slot_comparable_count
                        if slot_comparable_count
                        else 0.0
                    ),
                    "full_model_id": replace(
                        candidate,
                        disabled_feature_groups=(),
                    ).identifier,
                    "auxiliary_reference_model_id": replace(
                        candidate,
                        auxiliary_coefficient=None,
                    ).identifier,
                    "auxiliary_horizon_reference_model_id": replace(
                        candidate,
                        auxiliary_horizons=fold_training.auxiliary_horizons,
                    ).identifier,
                    "auxiliary_target_reference_model_id": replace(
                        candidate,
                        auxiliary_target_exclusions=(),
                    ).identifier,
                    "delta_neutrality_reference_model_id": replace(
                        candidate,
                        delta_neutrality_coefficient=0.0,
                    ).identifier,
                    "discount_reference_model_id": replace(
                        candidate,
                        time_aware_discounting=None,
                    ).identifier,
                    "burn_in_reference_model_id": replace(
                        candidate,
                        burn_in_steps=None,
                    ).identifier,
                    "start_sampling_reference_model_id": replace(
                        candidate,
                        start_sampling=None,
                    ).identifier,
                    "factorized_objective_reference_model_id": replace(
                        candidate,
                        factorized_ppo_objective=None,
                    ).identifier,
                    "entropy_objective_reference_model_id": replace(
                        candidate,
                        entropy_objective=None,
                    ).identifier,
                    "critic_layer_norm_reference_model_id": replace(
                        candidate,
                        critic_layer_norm=False,
                    ).identifier,
                })

        grouped_runs = []
        for candidate in model_specs:
            replicates = [
                run
                for run in candidate_runs
                if run["model_id"] == candidate.identifier
            ]
            aggregate = _training_seed_aggregate(
                replicates,
                walk_forward_config,
            )
            representative = aggregate.pop("representative_run")
            grouped_runs.append({
                "representative": representative,
                "aggregate": aggregate,
                "replicates": replicates,
                "latency_eligible": all(
                    run["latency_eligible"] for run in replicates
                ),
            })
        eligible_groups = [
            group for group in grouped_runs if group["latency_eligible"]
        ]
        if not eligible_groups:
            observed = ", ".join(
                f"{group['representative']['model_id']}="
                f"{max(run['inference_latency']['median_microseconds'] for run in group['replicates']):.3f}us"
                for group in grouped_runs
            )
            raise ValueError(
                "no model candidate satisfies max_median_inference_latency_us="
                f"{walk_forward_config.max_median_inference_latency_us}; "
                f"observed {observed}"
            )
        winning_group = _select_seed_robust_group(
            eligible_groups,
            minimum_score_tolerance=(
                walk_forward_config.selection_score_tolerance
            ),
        )
        winning_run = winning_group["representative"]
        selection_rule = winning_group["selection_rule"]
        candidate_results = [
            {
                "model_id": run["model_id"],
                "model": asdict(run["model_spec"]),
                "resolved_model": asdict(run["recurrent_config"]),
                "effective_auxiliary_coefficient": run[
                    "training_config"
                ].auxiliary_coefficient,
                "effective_auxiliary_horizons": list(
                    run["training_config"].auxiliary_horizons
                ),
                "effective_auxiliary_target_exclusions": list(
                    run["training_config"].auxiliary_target_exclusions
                ),
                "effective_delta_neutrality_coefficient": run[
                    "training_config"
                ].delta_neutrality_coefficient,
                "effective_time_aware_discounting": run[
                    "training_config"
                ].time_aware_discounting,
                "effective_burn_in_steps": run[
                    "training_config"
                ].burn_in_steps,
                "requested_start_sampling": run[
                    "training_config"
                ].start_sampling,
                "effective_factorized_ppo_objective": run[
                    "training_config"
                ].factorized_ppo_objective
                if run["model_spec"].algorithm == "ppo"
                and run["model_spec"].action_decoder == "factorized"
                else None,
                "effective_entropy_objective": run[
                    "training_config"
                ].entropy_objective,
                "critic_balance_diagnostic": run[
                    "critic_balance_evidence"
                ],
                "parameter_count": run["parameter_count"],
                "inference_latency": run["inference_latency"],
                "deployment_eligible": group["latency_eligible"],
                "selection_competitive": (
                    run["model_id"]
                    in selection_rule["competitive_model_ids"]
                ),
                "score_gap_to_best": (
                    selection_rule["best_score"]
                    - group["aggregate"][
                        "robust_training_seed_validation_score"
                    ]
                ),
                "ineligibility_reason": (
                    None
                    if group["latency_eligible"]
                    else "training_seed_inference_latency_exceeded"
                ),
                "training_seed_aggregate": group["aggregate"],
                "training_seed_replicates": [
                    {
                        "replicate_id": replicate["replicate_id"],
                        "training_seed": replicate["training_seed"],
                        "selected_episode": replicate["selected_episode"],
                        "validation_total_reward": replicate[
                            "validation_total_reward"
                        ],
                        "validation_selection_score": replicate[
                            "validation_selection_score"
                        ],
                        "optimizer_updates": replicate["optimizer_updates"],
                        "inference_latency": replicate["inference_latency"],
                        "deployment_eligible": replicate["latency_eligible"],
                        "start_sampling": replicate[
                            "start_sampling_evidence"
                        ],
                        "entropy": replicate["entropy_evidence"],
                        "critic_balance_diagnostic": replicate[
                            "critic_balance_evidence"
                        ],
                    }
                    for replicate in group["replicates"]
                ],
                "parameter_budget_headroom": (
                    run["model_spec"].parameter_budget
                    - run["parameter_count"]
                    if run["model_spec"].parameter_budget is not None
                    else None
                ),
                "masked_input_count": run["masked_input_count"],
                "active_input_count": run["active_input_count"],
                "episodes_completed": sum(
                    replicate["episodes_completed"]
                    for replicate in group["replicates"]
                ),
                "stopped_early": any(
                    replicate["stopped_early"]
                    for replicate in group["replicates"]
                ),
                "optimizer_updates": sum(
                    replicate["optimizer_updates"]
                    for replicate in group["replicates"]
                ),
                "slot_changed_count": sum(
                    replicate["slot_changed_count"]
                    for replicate in group["replicates"]
                ),
                "slot_comparable_count": sum(
                    replicate["slot_comparable_count"]
                    for replicate in group["replicates"]
                ),
                "slot_churn_rate": (
                    sum(
                        replicate["slot_changed_count"]
                        for replicate in group["replicates"]
                    )
                    / sum(
                        replicate["slot_comparable_count"]
                        for replicate in group["replicates"]
                    )
                    if sum(
                        replicate["slot_comparable_count"]
                        for replicate in group["replicates"]
                    )
                    else 0.0
                ),
                "validation_reward_lift_vs_full": None,
                "validation_score_lift_vs_full": None,
                "validation_reward_lift_vs_auxiliary_enabled": None,
                "validation_score_lift_vs_auxiliary_enabled": None,
                "validation_reward_lift_vs_configured_horizons": None,
                "validation_score_lift_vs_configured_horizons": None,
                "validation_reward_lift_vs_full_auxiliary_targets": None,
                "validation_score_lift_vs_full_auxiliary_targets": None,
                "validation_reward_lift_vs_delta_neutrality_disabled": None,
                "validation_score_lift_vs_delta_neutrality_disabled": None,
                "validation_reward_lift_vs_time_aware_discounting": None,
                "validation_score_lift_vs_time_aware_discounting": None,
                "validation_reward_lift_vs_burn_in": None,
                "validation_score_lift_vs_burn_in": None,
                "validation_reward_lift_vs_stratified_starts": None,
                "validation_score_lift_vs_stratified_starts": None,
                "validation_reward_lift_vs_joint_factorized_objective": None,
                "validation_score_lift_vs_joint_factorized_objective": None,
                "validation_reward_lift_vs_feasible_normalized_entropy": None,
                "validation_score_lift_vs_feasible_normalized_entropy": None,
                "validation_reward_lift_vs_critic_layer_norm_disabled": None,
                "validation_score_lift_vs_critic_layer_norm_disabled": None,
                "selection": {
                    "scope": run["selection_scope"],
                    "episode": run["selected_episode"],
                    "training_seed": run["training_seed"],
                    "validation_total_reward": run[
                        "validation_total_reward"
                    ],
                    "validation_selection_score": run[
                        "validation_selection_score"
                    ],
                    "training_seed_mean_validation_reward": group[
                        "aggregate"
                    ]["validation_total_reward_mean"],
                    "robust_training_seed_validation_score": group[
                        "aggregate"
                    ]["robust_training_seed_validation_score"],
                    "max_drawdown": run["validation_max_drawdown"],
                    "downside_deviation": run[
                        "validation_downside_deviation"
                    ],
                    "turnover": run["validation_turnover"],
                    "abs_beta_to_underlying": run[
                        "validation_abs_beta_to_underlying"
                    ],
                    "mean_abs_delta_notional_weight": run[
                        "validation_mean_abs_delta_notional_weight"
                    ],
                },
            }
            for group in grouped_runs
            for run in (group["representative"],)
        ]
        validation_scores = {
            group["representative"]["model_id"]: group["aggregate"][
                "robust_training_seed_validation_score"
            ]
            for group in grouped_runs
        }
        validation_rewards = {
            group["representative"]["model_id"]: group["aggregate"][
                "validation_total_reward_mean"
            ]
            for group in grouped_runs
        }
        for result, group in zip(candidate_results, grouped_runs, strict=True):
            run = group["representative"]
            aggregate_score = group["aggregate"][
                "robust_training_seed_validation_score"
            ]
            aggregate_reward = group["aggregate"][
                "validation_total_reward_mean"
            ]
            full_score = validation_scores.get(run["full_model_id"])
            if run["model_spec"].disabled_feature_groups and full_score is not None:
                result["validation_score_lift_vs_full"] = (
                    aggregate_score - full_score
                )
                result["validation_reward_lift_vs_full"] = (
                    aggregate_reward
                    - validation_rewards[run["full_model_id"]]
                )
            auxiliary_reference = validation_scores.get(
                run["auxiliary_reference_model_id"]
            )
            if (
                run["model_spec"].auxiliary_coefficient == 0.0
                and auxiliary_reference is not None
            ):
                result["validation_score_lift_vs_auxiliary_enabled"] = (
                    aggregate_score - auxiliary_reference
                )
                result["validation_reward_lift_vs_auxiliary_enabled"] = (
                    aggregate_reward
                    - validation_rewards[run["auxiliary_reference_model_id"]]
                )
            horizon_reference = validation_scores.get(
                run["auxiliary_horizon_reference_model_id"]
            )
            if (
                run["model_spec"].auxiliary_horizons
                != fold_training.auxiliary_horizons
                and horizon_reference is not None
            ):
                result["validation_score_lift_vs_configured_horizons"] = (
                    aggregate_score - horizon_reference
                )
                result["validation_reward_lift_vs_configured_horizons"] = (
                    aggregate_reward
                    - validation_rewards[
                        run["auxiliary_horizon_reference_model_id"]
                    ]
                )
            auxiliary_target_reference = validation_scores.get(
                run["auxiliary_target_reference_model_id"]
            )
            if (
                run["model_spec"].auxiliary_target_exclusions
                and auxiliary_target_reference is not None
            ):
                result[
                    "validation_score_lift_vs_full_auxiliary_targets"
                ] = aggregate_score - auxiliary_target_reference
                result[
                    "validation_reward_lift_vs_full_auxiliary_targets"
                ] = (
                    aggregate_reward
                    - validation_rewards[
                        run["auxiliary_target_reference_model_id"]
                    ]
                )
            delta_neutrality_reference = validation_scores.get(
                run["delta_neutrality_reference_model_id"]
            )
            if (
                run["training_config"].delta_neutrality_coefficient > 0
                and delta_neutrality_reference is not None
            ):
                result[
                    "validation_score_lift_vs_delta_neutrality_disabled"
                ] = aggregate_score - delta_neutrality_reference
                result[
                    "validation_reward_lift_vs_delta_neutrality_disabled"
                ] = (
                    aggregate_reward
                    - validation_rewards[
                        run["delta_neutrality_reference_model_id"]
                    ]
                )
            discount_reference = validation_scores.get(
                run["discount_reference_model_id"]
            )
            if (
                run["model_spec"].time_aware_discounting is False
                and discount_reference is not None
            ):
                result[
                    "validation_score_lift_vs_time_aware_discounting"
                ] = aggregate_score - discount_reference
                result[
                    "validation_reward_lift_vs_time_aware_discounting"
                ] = (
                    aggregate_reward
                    - validation_rewards[run["discount_reference_model_id"]]
                )
            burn_in_reference = validation_scores.get(
                run["burn_in_reference_model_id"]
            )
            if (
                run["model_spec"].burn_in_steps == 0
                and burn_in_reference is not None
            ):
                result["validation_score_lift_vs_burn_in"] = (
                    aggregate_score - burn_in_reference
                )
                result["validation_reward_lift_vs_burn_in"] = (
                    aggregate_reward
                    - validation_rewards[run["burn_in_reference_model_id"]]
                )
            start_sampling_reference = validation_scores.get(
                run["start_sampling_reference_model_id"]
            )
            if (
                run["model_spec"].start_sampling == "uniform"
                and fold_training.start_sampling == "volatility_stratified"
                and start_sampling_reference is not None
            ):
                result["validation_score_lift_vs_stratified_starts"] = (
                    aggregate_score - start_sampling_reference
                )
                result["validation_reward_lift_vs_stratified_starts"] = (
                    aggregate_reward
                    - validation_rewards[
                        run["start_sampling_reference_model_id"]
                    ]
                )
            factorized_reference = validation_scores.get(
                run["factorized_objective_reference_model_id"]
            )
            if (
                run["model_spec"].factorized_ppo_objective == "dimensionwise"
                and factorized_reference is not None
            ):
                result[
                    "validation_score_lift_vs_joint_factorized_objective"
                ] = aggregate_score - factorized_reference
                result[
                    "validation_reward_lift_vs_joint_factorized_objective"
                ] = (
                    aggregate_reward
                    - validation_rewards[
                        run["factorized_objective_reference_model_id"]
                    ]
                )
            entropy_reference = validation_scores.get(
                run["entropy_objective_reference_model_id"]
            )
            if (
                run["model_spec"].entropy_objective == "raw_mean"
                and entropy_reference is not None
            ):
                result[
                    "validation_score_lift_vs_feasible_normalized_entropy"
                ] = aggregate_score - entropy_reference
                result[
                    "validation_reward_lift_vs_feasible_normalized_entropy"
                ] = (
                    aggregate_reward
                    - validation_rewards[
                        run["entropy_objective_reference_model_id"]
                    ]
                )
            critic_layer_norm_reference = validation_scores.get(
                run["critic_layer_norm_reference_model_id"]
            )
            if (
                run["model_spec"].critic_layer_norm
                and critic_layer_norm_reference is not None
            ):
                result[
                    "validation_score_lift_vs_critic_layer_norm_disabled"
                ] = aggregate_score - critic_layer_norm_reference
                result[
                    "validation_reward_lift_vs_critic_layer_norm_disabled"
                ] = (
                    aggregate_reward
                    - validation_rewards[
                        run["critic_layer_norm_reference_model_id"]
                    ]
                )
        model = winning_run["model"]
        metrics = winning_run["metrics"]
        recurrent_config = winning_run["recurrent_config"]
        selected_training = winning_run["training_config"]
        selected = _selected_metric(metrics)
        winning_seed_aggregate = dict(winning_group["aggregate"])
        candidate_runs.clear()
        grouped_runs.clear()
        del winning_group

        # Do not instantiate or evaluate the held-out range until validation
        # has fixed both the architecture and its restored checkpoint.
        test_env = OptionsEnv(test_data, **environment_options)

        test_traces = [
            run_episode_trace(
                test_env,
                recurrent_policy(model, selected_training.sequence_length),
                seed,
            )
            for seed in walk_forward_config.test_seeds
        ]
        test_reports = [trace.report for trace in test_traces]
        baseline_traces = {
            "no_op": [
                run_episode_trace(test_env, no_op, seed)
                for seed in walk_forward_config.test_seeds
            ],
            "first_feasible": [
                run_episode_trace(test_env, first_feasible, seed)
                for seed in walk_forward_config.test_seeds
            ],
            "buy_first_then_delta_hedge": [
                run_episode_trace(test_env, buy_first_then_delta_hedge(), seed)
                for seed in walk_forward_config.test_seeds
            ],
            "long_volatility_delta_hedge": [
                run_episode_trace(
                    test_env,
                    long_volatility_delta_hedge(long_volatility_config),
                    seed,
                )
                for seed in walk_forward_config.test_seeds
            ],
            "cash_secured_short_put_delta_hedge": [
                run_episode_trace(
                    test_env,
                    cash_secured_short_put_delta_hedge(
                        short_volatility_config
                    ),
                    seed,
                )
                for seed in walk_forward_config.test_seeds
            ],
            "underlying_trend": [
                run_episode_trace(
                    test_env,
                    underlying_trend(trend_config),
                    seed,
                )
                for seed in walk_forward_config.test_seeds
            ],
        }
        baseline_reports = {
            name: [trace.report for trace in traces]
            for name, traces in baseline_traces.items()
        }
        statistical_comparisons = {}
        for baseline_index, (name, traces) in enumerate(baseline_traces.items()):
            comparisons = []
            for seed_index, (candidate, baseline) in enumerate(
                zip(test_traces, traces, strict=True)
            ):
                if candidate.timestamps != baseline.timestamps:
                    raise ValueError(
                        "candidate and baseline test paths are not aligned"
                    )
                comparison = paired_moving_block_bootstrap(
                    candidate.step_returns,
                    baseline.step_returns,
                    samples=walk_forward_config.bootstrap_samples,
                    block_length=walk_forward_config.bootstrap_block_length,
                    confidence_level=walk_forward_config.bootstrap_confidence,
                    min_observations=(
                        walk_forward_config.bootstrap_min_observations
                    ),
                    seed=(
                        walk_forward_config.bootstrap_seed
                        + fold.fold * 10_000
                        + baseline_index * 1_000
                        + seed_index
                    ),
                )
                comparisons.append({
                    "test_seed": walk_forward_config.test_seeds[seed_index],
                    "first_arrival_timestamp": (
                        candidate.timestamps[0] if candidate.timestamps else None
                    ),
                    "last_arrival_timestamp": (
                        candidate.timestamps[-1] if candidate.timestamps else None
                    ),
                    **comparison.to_dict(),
                })
            statistical_comparisons[name] = comparisons
        cost_stress = {}
        for scenario in DEFAULT_COST_SCENARIOS:
            cost_stress[scenario.name] = [
                run_episode(
                    cost_stressed_environment(test_env, scenario),
                    recurrent_policy(model, selected_training.sequence_length),
                    seed,
                )
                for seed in walk_forward_config.test_seeds
            ]
        baseline_cost_stress = {
            "cash_secured_short_put_delta_hedge": {
                scenario.name: [
                    run_episode(
                        cost_stressed_environment(test_env, scenario),
                        cash_secured_short_put_delta_hedge(
                            short_volatility_config
                        ),
                        seed,
                    )
                    for seed in walk_forward_config.test_seeds
                ]
                for scenario in DEFAULT_COST_SCENARIOS
            }
        }

        fold_record = {
            "fold": fold.fold,
            "split": fold.to_dict(),
            "heldout_evaluation_contract": {
                "deterministic_policy": True,
                "path_count": 1,
                "seed_repetitions": 1,
                "test_seed": walk_forward_config.test_seeds[0],
                "training_seed_count": len(
                    walk_forward_config.training_seed_offsets
                ),
                "training_seeds": [
                    fold_training.seed + offset
                    for offset in walk_forward_config.training_seed_offsets
                ],
                "selected_training_seed": selected_training.seed,
                "bootstrap_independence_unit": "arrival_time_block",
            },
            "environment_fingerprints": {
                "train": train_env.manifest.fingerprint,
                "validation": validation_env.manifest.fingerprint,
                "test": test_env.manifest.fingerprint,
            },
            "test_data_quality": _partition_data_quality(test_data),
            "selection": {
                "scope": selected["evaluation_scope"],
                "episode": selected["episode"],
                "validation_total_reward": selected["evaluation_total_reward"],
                "validation_selection_score": selected[
                    "evaluation_selection_score"
                ],
                "max_drawdown": selected["evaluation_max_drawdown"],
                "downside_deviation": selected[
                    "evaluation_downside_deviation"
                ],
                "turnover": selected["evaluation_turnover"],
                "model_id": winning_run["model_id"],
                "training_seed": selected_training.seed,
                "robust_training_seed_validation_score": (
                    winning_seed_aggregate[
                        "robust_training_seed_validation_score"
                    ]
                ),
            },
            "model_selection": {
                "criterion": "robust_training_seed_validation_score",
                "direction": "maximize",
                "score_definition": {
                    "reward": "validation_total_reward",
                    "drawdown_penalty": (
                        selected_training.selection_drawdown_penalty
                    ),
                    "downside_penalty": (
                        selected_training.selection_downside_penalty
                    ),
                    "turnover_penalty": (
                        selected_training.selection_turnover_penalty
                    ),
                    "absolute_beta_penalty": (
                        selected_training.selection_abs_beta_penalty
                    ),
                    "delta_notional_penalty": (
                        selected_training.selection_delta_notional_penalty
                    ),
                    "cross_ticker_std_penalty": (
                        selected_training.selection_cross_ticker_std_penalty
                    ),
                    "worst_ticker_weight": (
                        selected_training.selection_worst_ticker_weight
                    ),
                    "training_seed_worst_weight": (
                        walk_forward_config.training_seed_worst_weight
                    ),
                    "training_seed_dispersion_penalty": (
                        walk_forward_config.training_seed_dispersion_penalty
                    ),
                },
                "simplicity_rule": selection_rule,
                "tie_break": [
                    "dimensionwise_factorized_objective_ablation",
                    "raw_mean_entropy_objective_ablation",
                    "delta_neutrality_training_ablation",
                    "critic_layer_norm_ablation",
                    "auxiliary_target_ablation",
                    "worst_training_seed_median_inference_latency",
                    "parameter_count",
                    "active_input_count",
                    "optimizer_updates",
                    "burn_in_ablation",
                    "fixed_step_discount_ablation",
                    "uniform_start_sampling_ablation",
                    "model_id",
                ],
                "eligibility_constraint": {
                    "metric": "median_inference_latency_us",
                    "training_seed_scope": "all_replicates",
                    "maximum": (
                        walk_forward_config.max_median_inference_latency_us
                    ),
                    "enabled": (
                        walk_forward_config.max_median_inference_latency_us
                        is not None
                    ),
                },
                "selected_model_id": winning_run["model_id"],
                "candidates": candidate_results,
            },
            "test": _reports_to_dict(test_reports),
            "heldout_traces": {
                "agent": _traces_to_dict(test_traces),
                "baselines": {
                    name: _traces_to_dict(traces)
                    for name, traces in baseline_traces.items()
                },
            },
            "baselines": {
                name: _reports_to_dict(reports)
                for name, reports in baseline_reports.items()
            },
            "baseline_configuration": {
                "long_volatility_delta_hedge": asdict(long_volatility_config),
                "cash_secured_short_put_delta_hedge": asdict(
                    short_volatility_config
                ),
                "underlying_trend": asdict(trend_config),
            },
            "statistical_comparisons": statistical_comparisons,
            "cost_stress": {
                name: _reports_to_dict(reports)
                for name, reports in cost_stress.items()
            },
            "baseline_cost_stress": {
                baseline: {
                    name: _reports_to_dict(reports)
                    for name, reports in scenarios.items()
                }
                for baseline, scenarios in baseline_cost_stress.items()
            },
        }
        checkpoint = output_dir / (
            f"{dataset.symbol}-fold-{fold.fold:03d}-"
            f"{winning_run['model_id']}.pt"
        )
        save_checkpoint(
            checkpoint,
            model,
            train_env,
            recurrent_config,
            selected_training,
            metrics,
            provenance={
                "walk_forward_schema": WALK_FORWARD_SCHEMA_VERSION,
                **fold_record,
            },
        )
        fold_record["checkpoint"] = str(checkpoint)
        fold_results.append(fold_record)

    summary = {
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "mode": "research_demo",
        "symbol": dataset.symbol,
        "walk_forward": asdict(walk_forward_config),
        "model": asdict(model_specs[0]) if len(model_specs) == 1 else None,
        "candidate_models": [
            {"model_id": spec.identifier, "model": asdict(spec)}
            for spec in model_specs
        ],
        "training": asdict(training_config),
        "environment": environment_contract,
        "folds": fold_results,
    }
    summary_path = output_dir / f"{dataset.symbol}-walk-forward.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/walk-forward"),
    )
    parser.add_argument("--min-train-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--embargo", type=int, default=8)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--max-train-size", type=int)
    parser.add_argument(
        "--training-seed-offset",
        action="append",
        type=int,
        help=(
            "repeat to train independent validation replicates relative to "
            "--seed; defaults to one offset of zero"
        ),
    )
    parser.add_argument(
        "--training-seed-worst-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--training-seed-dispersion-penalty",
        type=float,
        default=0.25,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-block-length", type=int)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-min-observations", type=int, default=20)
    parser.add_argument("--bootstrap-seed", type=int, default=70_001)
    parser.add_argument("--long-volatility-window", type=int, default=16)
    parser.add_argument(
        "--long-volatility-min-coverage", type=float, default=0.75
    )
    parser.add_argument("--long-volatility-min-edge", type=float, default=0.02)
    parser.add_argument("--long-volatility-quantity", type=int, default=1)
    parser.add_argument("--short-volatility-window", type=int, default=16)
    parser.add_argument(
        "--short-volatility-min-coverage", type=float, default=0.75
    )
    parser.add_argument("--short-volatility-min-edge", type=float, default=0.02)
    parser.add_argument("--short-volatility-quantity", type=int, default=1)
    parser.add_argument("--trend-window", type=int, choices=(4, 16), default=16)
    parser.add_argument("--trend-min-coverage", type=float, default=0.75)
    parser.add_argument("--trend-min-abs-log-return", type=float, default=0.0)
    parser.add_argument("--trend-quantity", type=int, default=1)
    parser.add_argument("--latency-warmup-iterations", type=int, default=10)
    parser.add_argument("--latency-measured-iterations", type=int, default=100)
    parser.add_argument("--max-median-inference-latency-us", type=float)
    parser.add_argument(
        "--selection-score-tolerance",
        type=float,
        default=0.0,
        help=(
            "minimum validation-score tolerance for the one-standard-error "
            "simplest-competitive selection rule"
        ),
    )
    parser.add_argument(
        "--kind",
        choices=("gru", "lstm", "hybrid", "mixture"),
        default="gru",
    )
    parser.add_argument(
        "--encoder",
        choices=(
            "flat", "graph", "graph_set", "surface_graph_set", "attention_set",
        ),
        default="flat",
    )
    parser.add_argument("--algorithm", choices=("ppo", "reinforce"), default="ppo")
    parser.add_argument(
        "--action-decoder",
        choices=("factorized", "single_leg"),
        default="factorized",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        metavar="ENCODER:KIND[:ALGORITHM[:GRAPH_NEIGHBORS][:ACTION_DECODER]]",
        help=(
            "repeat to select architectures, PPO/REINFORCE, and optionally "
            "the graph-neighbor count and action decoder using validation only"
        ),
    )
    parser.add_argument(
        "--ablation",
        action="append",
        choices=tuple(FEATURE_ABLATION_GROUPS),
        help=(
            "repeat to add a validation-only candidate with one named "
            "feature group disabled"
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--graph-hidden-size", type=int, default=32)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbors", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument(
        "--parameter-budget",
        type=int,
        help="maximum trainable parameters; hidden-size becomes the search cap",
    )
    parser.add_argument("--initial-hold-bias", type=float, default=5.0)
    parser.add_argument(
        "--critic-layer-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "normalize recurrent features only on the critic branch; "
            "actor inference is unchanged"
        ),
    )
    parser.add_argument(
        "--critic-layer-norm-ablation",
        action="store_true",
        help=(
            "add matched candidates with critic-only LayerNorm disabled"
        ),
    )
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--burn-in-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument(
        "--random-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--start-sampling",
        choices=("uniform", "volatility_stratified"),
        default="uniform",
        help=(
            "sample training starts uniformly or across causal realized-"
            "volatility strata; validation and test stay chronological"
        ),
    )
    parser.add_argument("--volatility-regime-bins", type=int, default=3)
    parser.add_argument(
        "--volatility-regime-window",
        type=int,
        choices=REALIZED_VOL_WINDOWS,
        default=16,
    )
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument(
        "--selection-patience",
        type=int,
        default=3,
        help="evaluations without improvement before stopping; 0 disables",
    )
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument("--selection-drawdown-penalty", type=float, default=0.0)
    parser.add_argument("--selection-downside-penalty", type=float, default=0.0)
    parser.add_argument("--selection-turnover-penalty", type=float, default=0.0)
    parser.add_argument("--selection-abs-beta-penalty", type=float, default=0.0)
    parser.add_argument(
        "--selection-delta-notional-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--selection-cross-ticker-std-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--selection-worst-ticker-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument(
        "--time-aware-discounting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scale gamma and GAE lambda by wall-clock transition duration",
    )
    parser.add_argument(
        "--discount-reference-seconds",
        type=float,
        default=900.0,
        help="interval at which configured gamma and GAE lambda apply",
    )
    parser.add_argument("--entropy-coefficient", type=float, default=1e-4)
    parser.add_argument(
        "--entropy-objective",
        choices=("feasible_normalized", "raw_mean"),
        default="feasible_normalized",
        help=(
            "normalize masked entropy by each feasible action set or use "
            "the unnormalized explorable-factor mean"
        ),
    )
    parser.add_argument(
        "--factorized-ppo-objective",
        choices=("joint", "dimensionwise"),
        default="joint",
        help=(
            "use the exact joint action ratio or the legacy per-dimension "
            "clipped PPO research objective"
        ),
    )
    parser.add_argument(
        "--delta-neutrality-coefficient",
        type=float,
        default=0.0,
        help=(
            "train-only penalty for absolute Delta notional divided by NAV; "
            "validation and test rewards remain unshaped"
        ),
    )
    parser.add_argument(
        "--delta-neutrality-ablation",
        action="store_true",
        help=(
            "add matched candidates with Delta-neutrality shaping disabled"
        ),
    )
    parser.add_argument(
        "--auxiliary-coefficient",
        type=float,
        default=0.0,
        help=(
            "weight for train-only future-market prediction loss; zero disables"
        ),
    )
    parser.add_argument(
        "--auxiliary-horizon",
        action="append",
        type=int,
        help=(
            "repeat for cumulative train-only prediction horizons; defaults "
            "to one step"
        ),
    )
    parser.add_argument(
        "--auxiliary-ablation",
        action="store_true",
        help=(
            "add a matched validation candidate with auxiliary loss disabled"
        ),
    )
    parser.add_argument(
        "--auxiliary-horizon-ablation",
        action="store_true",
        help=(
            "add matched one-step candidates for configured multi-horizon "
            "models"
        ),
    )
    parser.add_argument(
        "--auxiliary-target-ablation",
        action="append",
        choices=AUXILIARY_TARGET_FEATURES,
        help=(
            "repeat to add a matched validation candidate excluding one "
            "named train-only prediction target"
        ),
    )
    parser.add_argument(
        "--fixed-step-discount-ablation",
        action="store_true",
        help="add matched candidates that ignore elapsed transition duration",
    )
    parser.add_argument(
        "--burn-in-ablation",
        action="store_true",
        help="add matched candidates with recurrent burn-in disabled",
    )
    parser.add_argument(
        "--start-sampling-ablation",
        action="store_true",
        help=(
            "add matched uniform-start candidates for volatility-stratified "
            "training"
        ),
    )
    parser.add_argument(
        "--factorized-objective-ablation",
        action="store_true",
        help=(
            "add matched legacy dimension-wise PPO candidates for every "
            "factorized PPO candidate"
        ),
    )
    parser.add_argument(
        "--entropy-objective-ablation",
        action="store_true",
        help=(
            "add matched unnormalized explorable-entropy candidates for every model"
        ),
    )
    parser.add_argument("--slot-count", type=int, default=32)
    parser.add_argument(
        "--slot-assignment",
        choices=("stable", "ranked"),
        default="stable",
    )
    parser.add_argument("--max-quantity", type=int, default=3)
    parser.add_argument(
        "--allow-collateralized-option-shorts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "allow covered calls and cash-secured puts; naked shorts stay "
            "forbidden"
        ),
    )
    parser.add_argument("--reward-drawdown-penalty", type=float, default=0.0)
    parser.add_argument("--reward-downside-penalty", type=float, default=0.0)
    parser.add_argument(
        "--portfolio-valuation",
        choices=("liquidation", "midpoint"),
        default="liquidation",
        help=(
            "mark open positions at executable exits plus estimated closing "
            "fees, or use legacy midpoint accounting"
        ),
    )
    parser.add_argument("--underlying-lot-size", type=int, default=25)
    parser.add_argument("--max-abs-underlying-shares", type=int, default=500)
    parser.add_argument("--underlying-commission-per-share", type=float, default=0.005)
    parser.add_argument("--underlying-slippage-bps", type=float, default=1.0)
    parser.add_argument("--max-abs-delta", type=float)
    parser.add_argument("--max-abs-gamma", type=float)
    parser.add_argument("--max-abs-theta", type=float)
    parser.add_argument("--max-abs-vega", type=float)
    parser.add_argument(
        "--max-underlying-quote-age-seconds",
        type=float,
        default=DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS,
        help=(
            "mask simulated option and underlying fills when an explicitly "
            "timestamped provider quote is older than this threshold"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _walk_forward_config_from_args(
    args: argparse.Namespace,
) -> WalkForwardConfig:
    """Build shared split/baseline settings for single and universe CLIs."""
    return WalkForwardConfig(
        min_train_size=args.min_train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        embargo=args.embargo,
        step_size=args.step_size,
        max_train_size=args.max_train_size,
        training_seed_offsets=tuple(args.training_seed_offset or (0,)),
        training_seed_worst_weight=args.training_seed_worst_weight,
        training_seed_dispersion_penalty=(
            args.training_seed_dispersion_penalty
        ),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_length=args.bootstrap_block_length,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_min_observations=args.bootstrap_min_observations,
        bootstrap_seed=args.bootstrap_seed,
        long_volatility_window=args.long_volatility_window,
        long_volatility_min_coverage=args.long_volatility_min_coverage,
        long_volatility_min_edge=args.long_volatility_min_edge,
        long_volatility_quantity=args.long_volatility_quantity,
        short_volatility_window=args.short_volatility_window,
        short_volatility_min_coverage=args.short_volatility_min_coverage,
        short_volatility_min_edge=args.short_volatility_min_edge,
        short_volatility_quantity=args.short_volatility_quantity,
        trend_window=args.trend_window,
        trend_min_coverage=args.trend_min_coverage,
        trend_min_abs_log_return=args.trend_min_abs_log_return,
        trend_quantity=args.trend_quantity,
        latency_warmup_iterations=args.latency_warmup_iterations,
        latency_measured_iterations=args.latency_measured_iterations,
        max_median_inference_latency_us=(
            args.max_median_inference_latency_us
        ),
        selection_score_tolerance=args.selection_score_tolerance,
    )


def _model_specs_from_args(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    candidates = args.candidate or [f"{args.encoder}:{args.kind}"]
    auxiliary_horizons = tuple(args.auxiliary_horizon or (1,))
    specs = []
    for candidate in candidates:
        parts = candidate.split(":")
        if len(parts) not in {2, 3, 4, 5}:
            raise ValueError(
                "candidate must use ENCODER:KIND, ENCODER:KIND:ALGORITHM, "
                "ENCODER:KIND:ALGORITHM:GRAPH_NEIGHBORS, or append "
                "ACTION_DECODER"
            )
        encoder, kind = parts[:2]
        algorithm = parts[2] if len(parts) >= 3 else args.algorithm
        action_decoder = args.action_decoder
        try:
            graph_neighbors = int(parts[3]) if len(parts) >= 4 else args.graph_neighbors
        except ValueError as error:
            if len(parts) == 4 and parts[3] in {"factorized", "single_leg"}:
                graph_neighbors = args.graph_neighbors
                action_decoder = parts[3]
            else:
                raise ValueError(
                    "candidate graph neighbors must be an integer"
                ) from error
        if len(parts) == 5:
            action_decoder = parts[4]
        specs.append(
            ModelSpec(
                kind=kind,
                encoder=encoder,
                hidden_size=args.hidden_size,
                graph_hidden_size=args.graph_hidden_size,
                graph_layers=args.graph_layers,
                graph_neighbors=graph_neighbors,
                attention_heads=args.attention_heads,
                initial_hold_bias=args.initial_hold_bias,
                action_decoder=action_decoder,
                algorithm=algorithm,
                parameter_budget=args.parameter_budget,
                auxiliary_horizons=auxiliary_horizons,
                critic_layer_norm=args.critic_layer_norm,
            )
        )
    full_specs = tuple(specs)
    for group in args.ablation or ():
        specs.extend(
            replace(spec, disabled_feature_groups=(group,))
            for spec in full_specs
        )
    if args.auxiliary_horizon_ablation:
        if auxiliary_horizons == (1,):
            raise ValueError(
                "--auxiliary-horizon-ablation requires a horizon beyond one"
            )
        specs.extend(
            replace(spec, auxiliary_horizons=(1,))
            for spec in full_specs
        )
    if args.auxiliary_ablation:
        if args.auxiliary_coefficient <= 0:
            raise ValueError(
                "--auxiliary-ablation requires --auxiliary-coefficient > 0"
            )
        specs.extend(
            replace(spec, auxiliary_coefficient=0.0)
            for spec in full_specs
        )
    auxiliary_target_ablations = tuple(
        args.auxiliary_target_ablation or ()
    )
    if len(set(auxiliary_target_ablations)) != len(
        auxiliary_target_ablations
    ):
        raise ValueError("auxiliary target ablations must be unique")
    if auxiliary_target_ablations:
        if args.auxiliary_coefficient <= 0:
            raise ValueError(
                "--auxiliary-target-ablation requires "
                "--auxiliary-coefficient > 0"
            )
        specs.extend(
            replace(spec, auxiliary_target_exclusions=(target,))
            for target in auxiliary_target_ablations
            for spec in full_specs
        )
    if args.fixed_step_discount_ablation:
        if not args.time_aware_discounting:
            raise ValueError(
                "--fixed-step-discount-ablation requires "
                "--time-aware-discounting"
            )
        specs.extend(
            replace(spec, time_aware_discounting=False)
            for spec in full_specs
        )
    if args.burn_in_ablation:
        if args.burn_in_steps <= 0:
            raise ValueError(
                "--burn-in-ablation requires --burn-in-steps > 0"
            )
        specs.extend(
            replace(spec, burn_in_steps=0)
            for spec in full_specs
        )
    if args.start_sampling_ablation:
        if args.start_sampling != "volatility_stratified":
            raise ValueError(
                "--start-sampling-ablation requires "
                "--start-sampling volatility_stratified"
            )
        specs.extend(
            replace(spec, start_sampling="uniform")
            for spec in full_specs
        )
    if args.factorized_objective_ablation:
        if args.factorized_ppo_objective != "joint":
            raise ValueError(
                "--factorized-objective-ablation requires "
                "--factorized-ppo-objective joint"
            )
        references = tuple(
            spec
            for spec in full_specs
            if spec.algorithm == "ppo" and spec.action_decoder == "factorized"
        )
        if not references:
            raise ValueError(
                "--factorized-objective-ablation requires a factorized PPO "
                "candidate"
            )
        specs.extend(
            replace(spec, factorized_ppo_objective="dimensionwise")
            for spec in references
        )
    if args.entropy_objective_ablation:
        if args.entropy_objective != "feasible_normalized":
            raise ValueError(
                "--entropy-objective-ablation requires "
                "--entropy-objective feasible_normalized"
            )
        if args.entropy_coefficient <= 0:
            raise ValueError(
                "--entropy-objective-ablation requires "
                "--entropy-coefficient > 0"
            )
        specs.extend(
            replace(spec, entropy_objective="raw_mean")
            for spec in full_specs
        )
    if args.delta_neutrality_ablation:
        if args.delta_neutrality_coefficient <= 0:
            raise ValueError(
                "--delta-neutrality-ablation requires "
                "--delta-neutrality-coefficient > 0"
            )
        specs.extend(
            replace(spec, delta_neutrality_coefficient=0.0)
            for spec in full_specs
        )
    if args.critic_layer_norm_ablation:
        if not args.critic_layer_norm:
            raise ValueError(
                "--critic-layer-norm-ablation requires "
                "--critic-layer-norm"
            )
        specs.extend(
            replace(spec, critic_layer_norm=False)
            for spec in full_specs
        )
    return _normalize_model_specs(specs)


def main() -> None:
    args = _parser().parse_args()
    dataset = SnapshotDataset.from_directory(args.data_dir, args.symbol)
    try:
        model_specs = _model_specs_from_args(args)
        summary = run_walk_forward_training(
            dataset,
            _walk_forward_config_from_args(args),
            model_specs,
            TrainingConfig(
                episodes=args.episodes,
                sequence_length=args.sequence_length,
                burn_in_steps=args.burn_in_steps,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                ppo_epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_ratio=args.clip_ratio,
                value_clip=args.value_clip,
                target_kl=args.target_kl,
                value_coefficient=args.value_coefficient,
                gradient_clip=args.gradient_clip,
                time_aware_discounting=args.time_aware_discounting,
                discount_reference_seconds=args.discount_reference_seconds,
                max_steps=args.max_steps,
                random_start=args.random_start,
                start_sampling=args.start_sampling,
                volatility_regime_window=args.volatility_regime_window,
                volatility_regime_bins=args.volatility_regime_bins,
                evaluation_interval=args.evaluation_interval,
                selection_patience=(
                    None
                    if args.selection_patience == 0
                    else args.selection_patience
                ),
                selection_min_delta=args.selection_min_delta,
                selection_drawdown_penalty=(
                    args.selection_drawdown_penalty
                ),
                selection_downside_penalty=args.selection_downside_penalty,
                selection_turnover_penalty=args.selection_turnover_penalty,
                selection_abs_beta_penalty=args.selection_abs_beta_penalty,
                selection_delta_notional_penalty=(
                    args.selection_delta_notional_penalty
                ),
                selection_cross_ticker_std_penalty=(
                    args.selection_cross_ticker_std_penalty
                ),
                selection_worst_ticker_weight=(
                    args.selection_worst_ticker_weight
                ),
                entropy_coefficient=args.entropy_coefficient,
                entropy_objective=args.entropy_objective,
                factorized_ppo_objective=args.factorized_ppo_objective,
                delta_neutrality_coefficient=(
                    args.delta_neutrality_coefficient
                ),
                auxiliary_coefficient=args.auxiliary_coefficient,
                auxiliary_horizons=tuple(args.auxiliary_horizon or (1,)),
                algorithm=args.algorithm,
                seed=args.seed,
            ),
            args.output_dir,
            env_kwargs=_environment_kwargs_from_args(args),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "summary": str(args.output_dir / f"{dataset.symbol}-walk-forward.json"),
        "folds": len(summary["folds"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
