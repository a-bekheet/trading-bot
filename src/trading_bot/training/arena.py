"""Run one comparable recurrent-agent tournament across several tickers."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from trading_bot.market_data.freshness import (
    DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS,
    underlying_quote_age,
)
from trading_bot.market_data.market_state import market_state_features
from trading_bot.training.dataset import SnapshotDataset
from trading_bot.training.splits import walk_forward_splits
from trading_bot.training.trainer import TrainingConfig
from trading_bot.training.walk_forward import (
    ModelSpec,
    WalkForwardConfig,
    run_walk_forward_training,
)


AGENT_ARENA_SCHEMA_VERSION = "research-demo.agent-arena.v9"
DEFAULT_ARENA_SYMBOLS = ("AAPL", "NVDA", "MSFT", "AMZN", "GOOG")
DEFAULT_ARENA_TRAINING_SEED_OFFSETS = (0, 1, 2)
DEFAULT_ARENA_SELECTION_SCORE_TOLERANCE = 1e-4
DEFAULT_ARENA_ACTIVATION_MIN_SCORE_ADVANTAGE = 1e-4
DEFAULT_ARENA_LATEST_FOLD_ONLY = True
DEFAULT_ARENA_REQUIRE_READY_TAIL = True
DEFAULT_ARENA_OUTPUT_ROOT = Path("data/agent_runs/recurrent-arena")


def default_arena_output_dir(
    created_at: datetime | None = None,
) -> Path:
    """Return a timestamped run directory so prior evidence is never overwritten."""
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("arena output timestamp must be timezone-aware")
    run_id = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return DEFAULT_ARENA_OUTPUT_ROOT / run_id


def _has_executable_option_quote(frame) -> bool:
    if "bid" not in frame or "ask" not in frame:
        return False
    for bid, ask in zip(frame["bid"], frame["ask"], strict=True):
        try:
            bid_value = float(bid)
            ask_value = float(ask)
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(bid_value)
            and math.isfinite(ask_value)
            and bid_value > 0
            and ask_value > 0
            and bid_value <= ask_value
        ):
            return True
    return False


def _partition_readiness(
    dataset: SnapshotDataset,
    max_quote_age_seconds: float,
) -> dict[str, Any]:
    regular_count = 0
    fresh_quote_count = 0
    executable_quote_count = 0
    for snapshot in dataset.snapshots:
        first = snapshot.frame.iloc[0]
        regular, coverage = market_state_features(first.get("marketState"))
        regular_count += int(coverage >= 1.0 and regular >= 1.0)
        quote_age, quote_coverage = underlying_quote_age(
            snapshot.timestamp,
            first.get("underlyingQuoteTime"),
        )
        fresh_quote_count += int(
            quote_coverage >= 1.0 and quote_age <= max_quote_age_seconds
        )
        executable_quote_count += int(_has_executable_option_quote(snapshot.frame))
    count = len(dataset)
    return {
        "snapshot_count": count,
        "regular_snapshot_count": regular_count,
        "fresh_underlying_quote_count": fresh_quote_count,
        "executable_option_quote_count": executable_quote_count,
        "first_timestamp": dataset.snapshots[0].timestamp,
        "last_timestamp": dataset.snapshots[-1].timestamp,
        "ready": bool(
            count
            and regular_count == count
            and fresh_quote_count == count
            and executable_quote_count == count
        ),
    }


def arena_tail_readiness(
    dataset: SnapshotDataset,
    walk_forward_config: WalkForwardConfig,
    *,
    max_quote_age_seconds: float = DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS,
) -> dict[str, Any]:
    """Check whether the latest validation/test tail can support trading evidence."""
    if not walk_forward_config.latest_fold_only:
        raise ValueError("arena readiness requires latest_fold_only")
    if not math.isfinite(max_quote_age_seconds) or max_quote_age_seconds < 0:
        raise ValueError("max_quote_age_seconds must be finite and non-negative")
    folds = walk_forward_splits(
        len(dataset),
        min_train_size=walk_forward_config.min_train_size,
        validation_size=walk_forward_config.validation_size,
        test_size=walk_forward_config.test_size,
        embargo=walk_forward_config.embargo,
        step_size=walk_forward_config.step_size,
        max_train_size=walk_forward_config.max_train_size,
        latest_only=True,
    )
    if not folds:
        return {
            "ready": False,
            "reason": "dataset_too_short",
            "split": None,
            "validation": None,
            "test": None,
            "max_quote_age_seconds": max_quote_age_seconds,
        }
    split = folds[0]
    _, validation, test = split.apply(dataset)
    validation_readiness = _partition_readiness(validation, max_quote_age_seconds)
    test_readiness = _partition_readiness(test, max_quote_age_seconds)
    ready = validation_readiness["ready"] and test_readiness["ready"]
    return {
        "ready": ready,
        "reason": "ready" if ready else "regular_fresh_executable_tail_required",
        "split": split.to_dict(),
        "validation": validation_readiness,
        "test": test_readiness,
        "max_quote_age_seconds": max_quote_age_seconds,
    }


def recurrent_arena_models(
    *,
    hidden_size: int = 8,
    initial_hold_bias: float = 0.0,
) -> tuple[ModelSpec, ...]:
    """Return recurrent controls, surface GNNs, and matched signal ablations."""
    flat_controls = tuple(
        ModelSpec(
            kind=kind,
            encoder="flat",
            hidden_size=hidden_size,
            initial_hold_bias=initial_hold_bias,
            algorithm="ppo",
            action_decoder=action_decoder,
        )
        for kind in ("gru", "lstm", "mixture")
        for action_decoder in ("factorized", "single_leg")
    )
    surface_gnn_agents = tuple(
        ModelSpec(
            kind=kind,
            encoder="surface_graph_set",
            hidden_size=hidden_size,
            graph_hidden_size=hidden_size,
            graph_layers=1,
            graph_neighbors=1,
            initial_hold_bias=initial_hold_bias,
            algorithm="ppo",
            action_decoder="single_leg",
        )
        for kind in ("gru", "lstm", "mixture")
    )

    def sparse_feature_ablations(group: str) -> tuple[ModelSpec, ...]:
        return tuple(
            ModelSpec(
                kind=kind,
                encoder=encoder,
                hidden_size=hidden_size,
                graph_hidden_size=(
                    hidden_size if encoder == "surface_graph_set" else 32
                ),
                graph_layers=1 if encoder == "surface_graph_set" else 2,
                graph_neighbors=1 if encoder == "surface_graph_set" else 3,
                initial_hold_bias=initial_hold_bias,
                algorithm="ppo",
                action_decoder="single_leg",
                disabled_feature_groups=(group,),
            )
            for encoder in ("flat", "surface_graph_set")
            for kind in ("gru", "lstm", "mixture")
        )

    smile_residual_ablations = sparse_feature_ablations("contract_smile_residual")
    surface_velocity_ablations = sparse_feature_ablations("surface_velocity")
    return (
        flat_controls
        + surface_gnn_agents
        + smile_residual_ablations
        + surface_velocity_ablations
    )


def run_agent_arena(
    *,
    data_dir: Path,
    output_dir: Path,
    symbols: Sequence[str],
    walk_forward_config: WalkForwardConfig,
    model_specs: Sequence[ModelSpec],
    training_config: TrainingConfig,
    env_kwargs: dict[str, Any],
    require_ready_tail: bool = False,
) -> dict[str, Any]:
    """Run independent ticker tournaments and retain explicit failures."""
    normalized_symbols = tuple(
        dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip())
    )
    if not normalized_symbols:
        raise ValueError("agent arena requires at least one symbol")
    if len(model_specs) < 2:
        raise ValueError("agent arena requires at least two model candidates")

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    failures = []
    preflight = []
    max_quote_age_seconds = env_kwargs.get(
        "max_underlying_quote_age_seconds",
        DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS,
    )
    for symbol in normalized_symbols:
        try:
            if walk_forward_config.latest_fold_only:
                material_dataset = SnapshotDataset.material_from_directory(
                    data_dir,
                    symbol,
                )
                readiness = arena_tail_readiness(
                    material_dataset,
                    walk_forward_config,
                    max_quote_age_seconds=max_quote_age_seconds,
                )
                preflight.append({"symbol": symbol, **readiness})
                if require_ready_tail and not readiness["ready"]:
                    failures.append(
                        {
                            "symbol": symbol,
                            "error_type": "InsufficientRegularTail",
                            "message": readiness["reason"],
                        }
                    )
                    continue
                dataset = material_dataset.engineered()
            elif require_ready_tail:
                raise ValueError("require_ready_tail requires latest_fold_only")
            else:
                dataset = SnapshotDataset.from_directory(data_dir, symbol)
            summary = run_walk_forward_training(
                dataset,
                walk_forward_config,
                model_specs,
                training_config,
                output_dir,
                env_kwargs=env_kwargs,
            )
        except Exception as error:  # Continue the declared multi-ticker job.
            failures.append(
                {
                    "symbol": symbol,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue
        folds = summary.get("folds", [])
        completed.append(
            {
                "symbol": symbol,
                "summary": str(output_dir / f"{symbol}-walk-forward.json"),
                "folds": len(folds),
                "selected_model_ids": [
                    fold.get("model_selection", {}).get("selected_model_id")
                    for fold in folds
                ],
                "heldout_returns": [
                    report.get("total_return")
                    for fold in folds
                    for report in fold.get("test", [])
                ],
            }
        )

    artifact = {
        "schema_version": AGENT_ARENA_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "symbols": list(normalized_symbols),
        "walk_forward": asdict(walk_forward_config),
        "candidate_models": [asdict(spec) for spec in model_specs],
        "training": asdict(training_config),
        "environment": dict(env_kwargs),
        "require_ready_tail": require_ready_tail,
        "preflight": preflight,
        "completed": completed,
        "failures": failures,
    }
    (output_dir / "agent-arena.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readiness_only = bool(failures) and all(
        failure["error_type"] == "InsufficientRegularTail" for failure in failures
    )
    if not completed and not readiness_only:
        raise RuntimeError("agent arena produced no completed ticker runs")
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "exact run directory; defaults to a timestamped directory under "
            "data/agent_runs/recurrent-arena"
        ),
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help="repeat to select tickers; defaults to five representative leaders",
    )
    parser.add_argument("--min-train-size", type=int, default=6)
    parser.add_argument("--validation-size", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=4)
    parser.add_argument("--embargo", type=int, default=0)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=4)
    parser.add_argument("--slot-count", type=int, default=8)
    parser.add_argument("--max-quantity", type=int, default=1)
    parser.add_argument("--initial-hold-bias", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--training-seed-offset",
        action="append",
        type=int,
        help=("repeat to declare training-seed offsets; defaults to 0, 1, and 2"),
    )
    parser.add_argument(
        "--selection-score-tolerance",
        type=float,
        default=DEFAULT_ARENA_SELECTION_SCORE_TOLERANCE,
        help=(
            "validation-score materiality floor for simplest-competitive "
            "selection; defaults to one basis point"
        ),
    )
    parser.add_argument(
        "--activation-min-score-advantage",
        type=float,
        default=DEFAULT_ARENA_ACTIVATION_MIN_SCORE_ADVANTAGE,
        help=(
            "validation advantage over no-op required for sandbox activation; "
            "defaults to one basis point"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--bootstrap-min-observations", type=int, default=2)
    parser.add_argument("--latency-warmup-iterations", type=int, default=3)
    parser.add_argument("--latency-measured-iterations", type=int, default=20)
    parser.add_argument(
        "--allow-unready-tail",
        action="store_true",
        help=(
            "run plumbing experiments even when validation/test are not all "
            "provider-confirmed regular, fresh, and executable"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir or default_arena_output_dir()
    try:
        result = run_agent_arena(
            data_dir=args.data_dir,
            output_dir=output_dir,
            symbols=tuple(args.symbol or DEFAULT_ARENA_SYMBOLS),
            walk_forward_config=WalkForwardConfig(
                min_train_size=args.min_train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                embargo=args.embargo,
                step_size=args.step_size,
                latest_fold_only=DEFAULT_ARENA_LATEST_FOLD_ONLY,
                training_seed_offsets=tuple(
                    args.training_seed_offset or DEFAULT_ARENA_TRAINING_SEED_OFFSETS
                ),
                selection_score_tolerance=args.selection_score_tolerance,
                activation_min_score_advantage=(args.activation_min_score_advantage),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_min_observations=args.bootstrap_min_observations,
                latency_warmup_iterations=args.latency_warmup_iterations,
                latency_measured_iterations=args.latency_measured_iterations,
            ),
            model_specs=recurrent_arena_models(
                hidden_size=args.hidden_size,
                initial_hold_bias=args.initial_hold_bias,
            ),
            training_config=TrainingConfig(
                episodes=args.episodes,
                sequence_length=args.sequence_length,
                burn_in_steps=0,
                max_steps=args.max_steps,
                ppo_epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                evaluation_interval=1,
                selection_patience=None,
                seed=args.seed,
            ),
            env_kwargs={
                "slot_count": args.slot_count,
                "max_quantity": args.max_quantity,
            },
            require_ready_tail=(
                DEFAULT_ARENA_REQUIRE_READY_TAIL and not args.allow_unready_tail
            ),
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "artifact": str(output_dir / "agent-arena.json"),
                "completed": len(result["completed"]),
                "failures": len(result["failures"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
