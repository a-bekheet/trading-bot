"""Leak-safe walk-forward selection for one shared multi-ticker policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trading_bot.market_data.universe import TOP_50_TICKERS
from trading_bot.training.baselines import (
    LongVolatilityConfig,
    UnderlyingTrendConfig,
    buy_first_then_delta_hedge,
    first_feasible,
    long_volatility_delta_hedge,
    no_op,
    underlying_trend,
)
from trading_bot.training.dataset import SnapshotDataset
from trading_bot.training.env import OptionsEnv
from trading_bot.training.evaluation import (
    DEFAULT_COST_SCENARIOS,
    cost_stressed_environment,
    paired_moving_block_bootstrap,
    run_episode,
    run_episode_trace,
)
from trading_bot.training.splits import walk_forward_splits
from trading_bot.training.trainer import (
    TrainingConfig,
    benchmark_recurrent_inference,
    recurrent_policy,
    save_checkpoint,
    train_actor_critic,
)
from trading_bot.training.walk_forward import (
    ModelSpec,
    WalkForwardConfig,
    _model_specs_from_args,
    _normalize_model_specs,
    _parser as single_parser,
    _selected_metric,
    resolve_recurrent_config,
)


UNIVERSE_WALK_FORWARD_SCHEMA_VERSION = (
    "research-demo.universe-walk-forward.v13"
)


def _normalize_datasets(
    datasets: Sequence[SnapshotDataset],
) -> tuple[SnapshotDataset, ...]:
    normalized = tuple(datasets)
    if len(normalized) < 2:
        raise ValueError("universe walk-forward requires at least two tickers")
    if not all(isinstance(item, SnapshotDataset) for item in normalized):
        raise TypeError("universe datasets must be SnapshotDataset instances")
    symbols = [item.symbol for item in normalized]
    if len(set(symbols)) != len(symbols):
        raise ValueError("universe dataset symbols must be unique")
    return normalized


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("snapshot timestamps must include a timezone")
    return parsed


def _global_chronology(
    partitions: Sequence[
        tuple[SnapshotDataset, SnapshotDataset, SnapshotDataset]
    ],
) -> dict[str, str]:
    latest_train = max(
        _timestamp(train.snapshots[-1].timestamp)
        for train, _, _ in partitions
    )
    earliest_validation = min(
        _timestamp(validation.snapshots[0].timestamp)
        for _, validation, _ in partitions
    )
    latest_validation = max(
        _timestamp(validation.snapshots[-1].timestamp)
        for _, validation, _ in partitions
    )
    earliest_test = min(
        _timestamp(test.snapshots[0].timestamp)
        for _, _, test in partitions
    )
    if latest_train >= earliest_validation:
        raise ValueError(
            "universe training arrivals overlap validation arrivals"
        )
    if latest_validation >= earliest_test:
        raise ValueError(
            "universe validation arrivals overlap test arrivals"
        )
    return {
        "latest_train_arrival": latest_train.isoformat(),
        "earliest_validation_arrival": earliest_validation.isoformat(),
        "latest_validation_arrival": latest_validation.isoformat(),
        "earliest_test_arrival": earliest_test.isoformat(),
    }


def _universe_latency(
    model,
    environments: Sequence[OptionsEnv],
    training_config: TrainingConfig,
    walk_forward_config: WalkForwardConfig,
) -> dict[str, Any]:
    per_symbol = {}
    for index, environment in enumerate(environments):
        observation, _ = environment.reset(
            seed=training_config.seed + index
        )
        per_symbol[environment.dataset.symbol] = benchmark_recurrent_inference(
            model,
            observation,
            training_config.sequence_length,
            warmup_iterations=(
                walk_forward_config.latency_warmup_iterations
            ),
            measured_iterations=(
                walk_forward_config.latency_measured_iterations
            ),
        )
    medians = [item["median_microseconds"] for item in per_symbol.values()]
    p95_values = [item["p95_microseconds"] for item in per_symbol.values()]
    return {
        "schema_version": "research-demo.universe-inference-latency.v1",
        "scope": "worst_ticker_streaming_batch_1_training_observation",
        "aggregation": "maximum_per_ticker_median",
        "median_microseconds": max(medians),
        "mean_ticker_median_microseconds": float(np.mean(medians)),
        "maximum_ticker_p95_microseconds": max(p95_values),
        "per_symbol": per_symbol,
    }


def _heldout_symbol_evidence(
    environment: OptionsEnv,
    model,
    training_config: TrainingConfig,
    walk_forward_config: WalkForwardConfig,
    long_volatility_config: LongVolatilityConfig,
    trend_config: UnderlyingTrendConfig,
    *,
    fold_index: int,
    symbol_index: int,
) -> dict[str, Any]:
    test_traces = [
        run_episode_trace(
            environment,
            recurrent_policy(model, training_config.sequence_length),
            seed,
        )
        for seed in walk_forward_config.test_seeds
    ]
    baseline_traces = {
        "no_op": [
            run_episode_trace(environment, no_op, seed)
            for seed in walk_forward_config.test_seeds
        ],
        "first_feasible": [
            run_episode_trace(environment, first_feasible, seed)
            for seed in walk_forward_config.test_seeds
        ],
        "buy_first_then_delta_hedge": [
            run_episode_trace(
                environment,
                buy_first_then_delta_hedge(),
                seed,
            )
            for seed in walk_forward_config.test_seeds
        ],
        "long_volatility_delta_hedge": [
            run_episode_trace(
                environment,
                long_volatility_delta_hedge(long_volatility_config),
                seed,
            )
            for seed in walk_forward_config.test_seeds
        ],
        "underlying_trend": [
            run_episode_trace(
                environment,
                underlying_trend(trend_config),
                seed,
            )
            for seed in walk_forward_config.test_seeds
        ],
    }
    comparisons = {}
    for baseline_index, (name, traces) in enumerate(baseline_traces.items()):
        items = []
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
                    + fold_index * 1_000_000
                    + symbol_index * 10_000
                    + baseline_index * 1_000
                    + seed_index
                ),
            )
            items.append({
                "test_seed": walk_forward_config.test_seeds[seed_index],
                "first_arrival_timestamp": (
                    candidate.timestamps[0] if candidate.timestamps else None
                ),
                "last_arrival_timestamp": (
                    candidate.timestamps[-1] if candidate.timestamps else None
                ),
                **comparison.to_dict(),
            })
        comparisons[name] = items
    cost_stress = {
        scenario.name: [
            run_episode(
                cost_stressed_environment(environment, scenario),
                recurrent_policy(model, training_config.sequence_length),
                seed,
            ).to_dict()
            for seed in walk_forward_config.test_seeds
        ]
        for scenario in DEFAULT_COST_SCENARIOS
    }
    return {
        "agent": [trace.report.to_dict() for trace in test_traces],
        "baselines": {
            name: [trace.report.to_dict() for trace in traces]
            for name, traces in baseline_traces.items()
        },
        "statistical_comparisons": comparisons,
        "cost_stress": cost_stress,
    }


def _heldout_aggregate(
    by_symbol: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reports = [
        report
        for evidence in by_symbol.values()
        for report in evidence["agent"]
    ]
    returns = np.asarray(
        [report["total_return"] for report in reports],
        dtype=np.float64,
    )
    rewards = np.asarray(
        [report["total_reward"] for report in reports],
        dtype=np.float64,
    )
    drawdowns = np.asarray(
        [report["max_drawdown"] for report in reports],
        dtype=np.float64,
    )
    return {
        "scope": "descriptive_across_ticker_seed_reports",
        "report_count": len(reports),
        "symbol_count": len(by_symbol),
        "mean_total_return": float(returns.mean()),
        "median_total_return": float(np.median(returns)),
        "worst_total_return": float(returns.min()),
        "mean_total_reward": float(rewards.mean()),
        "mean_max_drawdown": float(drawdowns.mean()),
        "positive_return_fraction": float((returns > 0).mean()),
    }


def run_universe_walk_forward_training(
    datasets: Sequence[SnapshotDataset],
    walk_forward_config: WalkForwardConfig,
    model_spec: ModelSpec | Sequence[ModelSpec],
    training_config: TrainingConfig,
    output_dir: Path,
    *,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one shared policy on global validation, then open global test."""
    universe = _normalize_datasets(datasets)
    model_specs = _normalize_model_specs(model_spec)
    lengths = {dataset.symbol: len(dataset) for dataset in universe}
    common_length = min(lengths.values())
    folds = walk_forward_splits(
        common_length,
        min_train_size=walk_forward_config.min_train_size,
        validation_size=walk_forward_config.validation_size,
        test_size=walk_forward_config.test_size,
        embargo=walk_forward_config.embargo,
        step_size=walk_forward_config.step_size,
        max_train_size=walk_forward_config.max_train_size,
    )
    if not folds:
        raise ValueError(
            "universe datasets are too short for the requested walk-forward split"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_options = dict(env_kwargs or {})
    long_volatility_config = LongVolatilityConfig(
        realized_window=walk_forward_config.long_volatility_window,
        min_coverage=walk_forward_config.long_volatility_min_coverage,
        min_volatility_edge=walk_forward_config.long_volatility_min_edge,
        quantity=walk_forward_config.long_volatility_quantity,
    )
    trend_config = UnderlyingTrendConfig(
        return_window=walk_forward_config.trend_window,
        min_coverage=walk_forward_config.trend_min_coverage,
        min_abs_log_return=walk_forward_config.trend_min_abs_log_return,
        quantity=walk_forward_config.trend_quantity,
    )
    resolved_configs = {}
    fold_results = []
    for fold in folds:
        partitions = tuple(fold.apply(dataset) for dataset in universe)
        global_chronology = _global_chronology(partitions)
        train_envs = tuple(
            OptionsEnv(train, **environment_options)
            for train, _, _ in partitions
        )
        validation_envs = tuple(
            OptionsEnv(validation, **environment_options)
            for _, validation, _ in partitions
        )
        fold_training = replace(
            training_config,
            seed=training_config.seed + fold.fold,
        )
        candidate_runs = []
        for candidate in model_specs:
            resolved = resolved_configs.get(candidate.identifier)
            if resolved is None:
                resolved = resolve_recurrent_config(candidate, train_envs[0])
                resolved_configs[candidate.identifier] = resolved
            recurrent_config, parameter_count = resolved
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
            )
            model, metrics = train_actor_critic(
                train_envs,
                recurrent_config,
                candidate_training,
                selection_env=validation_envs,
            )
            actual_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            if actual_count != parameter_count:
                raise RuntimeError(
                    "trained model parameter count does not match its "
                    "resolved configuration"
                )
            inference_latency = _universe_latency(
                model,
                train_envs,
                candidate_training,
                walk_forward_config,
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
                "parameter_count": parameter_count,
                "inference_latency": inference_latency,
                "latency_eligible": latency_eligible,
                "masked_input_count": len(
                    recurrent_config.masked_input_indices
                ),
                "active_input_count": (
                    recurrent_config.input_size
                    - len(recurrent_config.masked_input_indices)
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
                "selected": selected,
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
                "no universe model candidate satisfies "
                "max_median_inference_latency_us="
                f"{walk_forward_config.max_median_inference_latency_us}; "
                f"observed {observed}"
            )
        winning_run = min(
            eligible_runs,
            key=lambda run: (
                -float(run["selected"]["evaluation_selection_score"]),
                run["parameter_count"],
                run["active_input_count"],
                run["optimizer_updates"],
                int(run["model_spec"].time_aware_discounting is False),
                run["model_id"],
            ),
        )
        candidate_results = []
        validation_scores = {
            run["model_id"]: float(
                run["selected"]["evaluation_selection_score"]
            )
            for run in candidate_runs
        }
        validation_rewards = {
            run["model_id"]: float(
                run["selected"]["evaluation_total_reward"]
            )
            for run in candidate_runs
        }
        for run in candidate_runs:
            selected = run["selected"]
            full_score = validation_scores.get(run["full_model_id"])
            result = {
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
                "parameter_count": run["parameter_count"],
                "parameter_budget_headroom": (
                    run["model_spec"].parameter_budget
                    - run["parameter_count"]
                    if run["model_spec"].parameter_budget is not None
                    else None
                ),
                "active_input_count": run["active_input_count"],
                "masked_input_count": run["masked_input_count"],
                "optimizer_updates": run["optimizer_updates"],
                "slot_changed_count": run["slot_changed_count"],
                "slot_comparable_count": run["slot_comparable_count"],
                "slot_churn_rate": run["slot_churn_rate"],
                "episodes_completed": len(run["metrics"]),
                "stopped_early": bool(
                    run["metrics"][-1]["early_stop_selection"]
                ),
                "inference_latency": run["inference_latency"],
                "deployment_eligible": run["latency_eligible"],
                "ineligibility_reason": (
                    None
                    if run["latency_eligible"]
                    else "worst_ticker_median_inference_latency_exceeded"
                ),
                "validation_score_lift_vs_full": None,
                "validation_reward_lift_vs_full": None,
                "validation_score_lift_vs_auxiliary_enabled": None,
                "validation_reward_lift_vs_auxiliary_enabled": None,
                "validation_score_lift_vs_configured_horizons": None,
                "validation_reward_lift_vs_configured_horizons": None,
                "validation_score_lift_vs_time_aware_discounting": None,
                "validation_reward_lift_vs_time_aware_discounting": None,
                "selection": {
                    "scope": selected["evaluation_scope"],
                    "episode": selected["episode"],
                    "validation_total_reward": selected[
                        "evaluation_total_reward"
                    ],
                    "validation_selection_score": selected[
                        "evaluation_selection_score"
                    ],
                    "mean_ticker_selection_score": selected[
                        "evaluation_selection_score_mean"
                    ],
                    "worst_ticker_selection_score": selected[
                        "evaluation_worst_ticker_selection_score"
                    ],
                    "ticker_selection_score_std": selected[
                        "evaluation_selection_score_std"
                    ],
                    "mean_max_drawdown": selected[
                        "evaluation_max_drawdown"
                    ],
                    "mean_downside_deviation": selected[
                        "evaluation_downside_deviation"
                    ],
                    "mean_turnover": selected["evaluation_turnover"],
                    "per_symbol": selected["evaluation_by_symbol"],
                },
            }
            if (
                run["model_spec"].disabled_feature_groups
                and full_score is not None
            ):
                result["validation_score_lift_vs_full"] = (
                    float(selected["evaluation_selection_score"])
                    - full_score
                )
                result["validation_reward_lift_vs_full"] = (
                    float(selected["evaluation_total_reward"])
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
                    float(selected["evaluation_selection_score"])
                    - auxiliary_reference
                )
                result["validation_reward_lift_vs_auxiliary_enabled"] = (
                    float(selected["evaluation_total_reward"])
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
                    float(selected["evaluation_selection_score"])
                    - horizon_reference
                )
                result["validation_reward_lift_vs_configured_horizons"] = (
                    float(selected["evaluation_total_reward"])
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
            candidate_results.append(result)

        model = winning_run["model"]
        selected_training = winning_run["training_config"]
        recurrent_config = winning_run["recurrent_config"]
        selected = winning_run["selected"]
        metrics = winning_run["metrics"]
        candidate_runs.clear()

        # The test environments are intentionally created only after the
        # validation aggregate has fixed the shared architecture/checkpoint.
        test_envs = tuple(
            OptionsEnv(test, **environment_options)
            for _, _, test in partitions
        )
        heldout_by_symbol = {
            environment.dataset.symbol: _heldout_symbol_evidence(
                environment,
                model,
                selected_training,
                walk_forward_config,
                long_volatility_config,
                trend_config,
                fold_index=fold.fold,
                symbol_index=index,
            )
            for index, environment in enumerate(test_envs)
        }
        environment_fingerprints = {
            dataset.symbol: {
                "train": train_envs[index].manifest.fingerprint,
                "validation": validation_envs[index].manifest.fingerprint,
                "test": test_envs[index].manifest.fingerprint,
            }
            for index, dataset in enumerate(universe)
        }
        fold_record = {
            "fold": fold.fold,
            "split": fold.to_dict(),
            "global_chronology": global_chronology,
            "environment_fingerprints": environment_fingerprints,
            "selection": {
                "scope": selected["evaluation_scope"],
                "episode": selected["episode"],
                "validation_total_reward": selected[
                    "evaluation_total_reward"
                ],
                "validation_selection_score": selected[
                    "evaluation_selection_score"
                ],
                "mean_ticker_selection_score": selected[
                    "evaluation_selection_score_mean"
                ],
                "worst_ticker_selection_score": selected[
                    "evaluation_worst_ticker_selection_score"
                ],
                "ticker_selection_score_std": selected[
                    "evaluation_selection_score_std"
                ],
                "mean_max_drawdown": selected[
                    "evaluation_max_drawdown"
                ],
                "mean_downside_deviation": selected[
                    "evaluation_downside_deviation"
                ],
                "mean_turnover": selected["evaluation_turnover"],
                "per_symbol": selected["evaluation_by_symbol"],
                "model_id": winning_run["model_id"],
            },
            "model_selection": {
                "criterion": "validation_selection_score",
                "direction": "maximize",
                "score_definition": {
                    "per_ticker": (
                        "reward - drawdown_penalty * max_drawdown - "
                        "downside_penalty * downside_deviation - "
                        "turnover_penalty * turnover"
                    ),
                    "aggregate": "(1-w) * mean + w * worst - d * std",
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
                    "fixed_step_discount_ablation",
                    "model_id",
                ],
                "latency_constraint": {
                    "metric": "worst_ticker_median_inference_latency_us",
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
            "heldout": {
                "aggregate": _heldout_aggregate(heldout_by_symbol),
                "per_symbol": heldout_by_symbol,
            },
            "baseline_configuration": {
                "long_volatility_delta_hedge": asdict(
                    long_volatility_config
                ),
                "underlying_trend": asdict(trend_config),
            },
        }
        checkpoint = output_dir / (
            f"universe-fold-{fold.fold:03d}-{winning_run['model_id']}.pt"
        )
        save_checkpoint(
            checkpoint,
            model,
            train_envs,
            recurrent_config,
            selected_training,
            metrics,
            provenance={
                "universe_walk_forward_schema": (
                    UNIVERSE_WALK_FORWARD_SCHEMA_VERSION
                ),
                **fold_record,
            },
        )
        fold_record["checkpoint"] = str(checkpoint)
        fold_results.append(fold_record)

    summary = {
        "schema_version": UNIVERSE_WALK_FORWARD_SCHEMA_VERSION,
        "mode": "universe_research_demo",
        "symbols": [dataset.symbol for dataset in universe],
        "dataset_lengths": lengths,
        "common_length": common_length,
        "ignored_tail_snapshots": {
            symbol: length - common_length
            for symbol, length in lengths.items()
        },
        "walk_forward": asdict(walk_forward_config),
        "candidate_models": [
            {"model_id": spec.identifier, "model": asdict(spec)}
            for spec in model_specs
        ],
        "training": asdict(training_config),
        "folds": fold_results,
    }
    summary_path = output_dir / "universe-walk-forward.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = single_parser()
    parser.description = __doc__
    parser.add_argument(
        "--universe-symbol",
        action="append",
        help="repeat to override the default top-50 universe",
    )
    parser.set_defaults(
        output_dir=Path("data/models/universe-walk-forward"),
        episodes=100,
    )
    return parser


def _symbols_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    symbols = (
        tuple(symbol.upper() for symbol in args.universe_symbol)
        if args.universe_symbol
        else TOP_50_TICKERS
    )
    if len(symbols) < 2:
        raise ValueError("universe walk-forward requires at least two symbols")
    if len(set(symbols)) != len(symbols):
        raise ValueError("universe symbols must be unique")
    return symbols


def main() -> None:
    args = _parser().parse_args()
    try:
        symbols = _symbols_from_args(args)
        datasets = tuple(
            SnapshotDataset.from_directory(args.data_dir, symbol)
            for symbol in symbols
        )
        summary = run_universe_walk_forward_training(
            datasets,
            WalkForwardConfig(
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
                long_volatility_min_coverage=(
                    args.long_volatility_min_coverage
                ),
                long_volatility_min_edge=args.long_volatility_min_edge,
                long_volatility_quantity=args.long_volatility_quantity,
                latency_warmup_iterations=args.latency_warmup_iterations,
                latency_measured_iterations=(
                    args.latency_measured_iterations
                ),
                max_median_inference_latency_us=(
                    args.max_median_inference_latency_us
                ),
            ),
            _model_specs_from_args(args),
            TrainingConfig(
                episodes=args.episodes,
                sequence_length=args.sequence_length,
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
                selection_downside_penalty=(
                    args.selection_downside_penalty
                ),
                selection_turnover_penalty=(
                    args.selection_turnover_penalty
                ),
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
            env_kwargs={
                "slot_count": args.slot_count,
                "slot_assignment": args.slot_assignment,
                "max_quantity": args.max_quantity,
                "underlying_lot_size": args.underlying_lot_size,
                "max_abs_underlying_shares": (
                    args.max_abs_underlying_shares
                ),
                "underlying_commission_per_share": (
                    args.underlying_commission_per_share
                ),
                "underlying_slippage_bps": args.underlying_slippage_bps,
                "max_abs_delta": args.max_abs_delta,
                "max_abs_gamma": args.max_abs_gamma,
                "max_abs_theta": args.max_abs_theta,
                "max_abs_vega": args.max_abs_vega,
            },
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "summary": str(args.output_dir / "universe-walk-forward.json"),
        "folds": len(summary["folds"]),
        "symbol_count": len(summary["symbols"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
