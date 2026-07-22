"""Executable train/validation/test workflow for recurrent option policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

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
from trading_bot.training.recurrent import (
    RecurrentConfig,
    build_recurrent_actor_critic,
)
from trading_bot.training.sequence import (
    AUXILIARY_TARGET_FEATURES,
    FEATURE_ABLATION_GROUPS,
    feature_ablation_indices,
    observation_vector,
)
from trading_bot.training.splits import walk_forward_splits
from trading_bot.training.trainer import (
    TrainingConfig,
    _environment_kwargs_from_args,
    benchmark_recurrent_inference,
    recurrent_policy,
    save_checkpoint,
    train_actor_critic,
)


WALK_FORWARD_SCHEMA_VERSION = "research-demo.walk-forward.v37"


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_size: int
    validation_size: int
    test_size: int
    embargo: int = 0
    step_size: int | None = None
    max_train_size: int | None = None
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
        if not self.test_seeds:
            raise ValueError("at least one held-out test seed is required")
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
    time_aware_discounting: bool | None = None
    burn_in_steps: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"gru", "lstm", "hybrid", "mixture"}:
            raise ValueError(
                "model kind must be gru, lstm, hybrid, or mixture"
            )
        if self.encoder not in {
            "flat", "graph", "graph_set", "attention_set",
        }:
            raise ValueError(
                "model encoder must be flat, graph, graph_set, or attention_set"
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
        feature_ablation_indices(self.disabled_feature_groups, 1)

    @property
    def identifier(self) -> str:
        """Return a stable identifier for artifact joins."""
        canonical = json.dumps(
            asdict(self),
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
            masked_input_indices=feature_ablation_indices(
                self.disabled_feature_groups,
                env.slot_count,
            ),
            graph_relation_indices=tuple(
                CONTRACT_FEATURES.index(name)
                for name in (
                    "impliedVolatility",
                    "delta",
                    "logMoneyness",
                    "dteDays",
                )
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

    minimum = replace(requested, hidden_size=1)
    minimum_count = parameter_count(minimum)
    if minimum_count > model_spec.parameter_budget:
        raise ValueError(
            f"parameter_budget={model_spec.parameter_budget} is below the "
            f"minimum {minimum_count} for {model_spec.encoder}:{model_spec.kind}"
        )

    low = 1
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
    for fold in folds:
        train_data, validation_data, test_data = fold.apply(dataset)
        train_env = OptionsEnv(train_data, **environment_options)
        validation_env = OptionsEnv(validation_data, **environment_options)
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
            candidate_training = replace(
                fold_training,
                algorithm=candidate.algorithm,
                auxiliary_horizons=candidate.auxiliary_horizons,
                auxiliary_coefficient=(
                    fold_training.auxiliary_coefficient
                    if candidate.auxiliary_coefficient is None
                    else candidate.auxiliary_coefficient
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
                "model_spec": candidate,
                "recurrent_config": recurrent_config,
                "model": model,
                "metrics": metrics,
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
                "selection_scope": selected["evaluation_scope"],
                "episodes_completed": len(metrics),
                "stopped_early": bool(metrics[-1]["early_stop_selection"]),
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
                "discount_reference_model_id": replace(
                    candidate,
                    time_aware_discounting=None,
                ).identifier,
                "burn_in_reference_model_id": replace(
                    candidate,
                    burn_in_steps=None,
                ).identifier,
            })

        eligible_runs = [
            run for run in candidate_runs if run["latency_eligible"]
        ]
        if not eligible_runs:
            observed = ", ".join(
                f"{run['model_id']}={run['inference_latency']['median_microseconds']:.3f}us"
                for run in candidate_runs
            )
            raise ValueError(
                "no model candidate satisfies max_median_inference_latency_us="
                f"{walk_forward_config.max_median_inference_latency_us}; "
                f"observed {observed}"
            )
        winning_run = min(
            eligible_runs,
            key=lambda run: (
                -run["validation_selection_score"],
                run["parameter_count"],
                run["active_input_count"],
                run["optimizer_updates"],
                int(run["model_spec"].burn_in_steps == 0),
                int(run["model_spec"].time_aware_discounting is False),
                run["model_id"],
            ),
        )
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
                "effective_time_aware_discounting": run[
                    "training_config"
                ].time_aware_discounting,
                "effective_burn_in_steps": run[
                    "training_config"
                ].burn_in_steps,
                "parameter_count": run["parameter_count"],
                "inference_latency": run["inference_latency"],
                "deployment_eligible": run["latency_eligible"],
                "ineligibility_reason": (
                    None
                    if run["latency_eligible"]
                    else "median_inference_latency_exceeded"
                ),
                "parameter_budget_headroom": (
                    run["model_spec"].parameter_budget
                    - run["parameter_count"]
                    if run["model_spec"].parameter_budget is not None
                    else None
                ),
                "masked_input_count": run["masked_input_count"],
                "active_input_count": run["active_input_count"],
                "episodes_completed": run["episodes_completed"],
                "stopped_early": run["stopped_early"],
                "optimizer_updates": run["optimizer_updates"],
                "slot_changed_count": run["slot_changed_count"],
                "slot_comparable_count": run["slot_comparable_count"],
                "slot_churn_rate": run["slot_churn_rate"],
                "validation_reward_lift_vs_full": None,
                "validation_score_lift_vs_full": None,
                "validation_reward_lift_vs_auxiliary_enabled": None,
                "validation_score_lift_vs_auxiliary_enabled": None,
                "validation_reward_lift_vs_configured_horizons": None,
                "validation_score_lift_vs_configured_horizons": None,
                "validation_reward_lift_vs_time_aware_discounting": None,
                "validation_score_lift_vs_time_aware_discounting": None,
                "validation_reward_lift_vs_burn_in": None,
                "validation_score_lift_vs_burn_in": None,
                "selection": {
                    "scope": run["selection_scope"],
                    "episode": run["selected_episode"],
                    "validation_total_reward": run[
                        "validation_total_reward"
                    ],
                    "validation_selection_score": run[
                        "validation_selection_score"
                    ],
                    "max_drawdown": run["validation_max_drawdown"],
                    "downside_deviation": run[
                        "validation_downside_deviation"
                    ],
                    "turnover": run["validation_turnover"],
                },
            }
            for run in candidate_runs
        ]
        validation_scores = {
            run["model_id"]: run["validation_selection_score"]
            for run in candidate_runs
        }
        validation_rewards = {
            run["model_id"]: run["validation_total_reward"]
            for run in candidate_runs
        }
        for result, run in zip(candidate_results, candidate_runs, strict=True):
            full_score = validation_scores.get(run["full_model_id"])
            if run["model_spec"].disabled_feature_groups and full_score is not None:
                result["validation_score_lift_vs_full"] = (
                    run["validation_selection_score"] - full_score
                )
                result["validation_reward_lift_vs_full"] = (
                    run["validation_total_reward"]
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
                    run["validation_selection_score"] - auxiliary_reference
                )
                result["validation_reward_lift_vs_auxiliary_enabled"] = (
                    run["validation_total_reward"]
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
                    run["validation_selection_score"] - horizon_reference
                )
                result["validation_reward_lift_vs_configured_horizons"] = (
                    run["validation_total_reward"]
                    - validation_rewards[
                        run["auxiliary_horizon_reference_model_id"]
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
                ] = run["validation_selection_score"] - discount_reference
                result[
                    "validation_reward_lift_vs_time_aware_discounting"
                ] = (
                    run["validation_total_reward"]
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
                    run["validation_selection_score"] - burn_in_reference
                )
                result["validation_reward_lift_vs_burn_in"] = (
                    run["validation_total_reward"]
                    - validation_rewards[run["burn_in_reference_model_id"]]
                )
        model = winning_run["model"]
        metrics = winning_run["metrics"]
        recurrent_config = winning_run["recurrent_config"]
        selected_training = winning_run["training_config"]
        selected = _selected_metric(metrics)
        candidate_runs.clear()

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
            "environment_fingerprints": {
                "train": train_env.manifest.fingerprint,
                "validation": validation_env.manifest.fingerprint,
                "test": test_env.manifest.fingerprint,
            },
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
            },
            "model_selection": {
                "criterion": "validation_selection_score",
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
                    "cross_ticker_std_penalty": (
                        selected_training.selection_cross_ticker_std_penalty
                    ),
                    "worst_ticker_weight": (
                        selected_training.selection_worst_ticker_weight
                    ),
                },
                "tie_break": [
                    "parameter_count",
                    "active_input_count",
                    "optimizer_updates",
                    "burn_in_ablation",
                    "fixed_step_discount_ablation",
                    "model_id",
                ],
                "eligibility_constraint": {
                    "metric": "median_inference_latency_us",
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
        "--kind",
        choices=("gru", "lstm", "hybrid", "mixture"),
        default="gru",
    )
    parser.add_argument(
        "--encoder",
        choices=("flat", "graph", "graph_set", "attention_set"),
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
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--burn-in-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument(
        "--random-start",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        "--fixed-step-discount-ablation",
        action="store_true",
        help="add matched candidates that ignore elapsed transition duration",
    )
    parser.add_argument(
        "--burn-in-ablation",
        action="store_true",
        help="add matched candidates with recurrent burn-in disabled",
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
    parser.add_argument("--underlying-lot-size", type=int, default=25)
    parser.add_argument("--max-abs-underlying-shares", type=int, default=500)
    parser.add_argument("--underlying-commission-per-share", type=float, default=0.005)
    parser.add_argument("--underlying-slippage-bps", type=float, default=1.0)
    parser.add_argument("--max-abs-delta", type=float)
    parser.add_argument("--max-abs-gamma", type=float)
    parser.add_argument("--max-abs-theta", type=float)
    parser.add_argument("--max-abs-vega", type=float)
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
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                time_aware_discounting=args.time_aware_discounting,
                discount_reference_seconds=args.discount_reference_seconds,
                max_steps=args.max_steps,
                random_start=args.random_start,
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
                selection_cross_ticker_std_penalty=(
                    args.selection_cross_ticker_std_penalty
                ),
                selection_worst_ticker_weight=(
                    args.selection_worst_ticker_weight
                ),
                entropy_coefficient=args.entropy_coefficient,
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
