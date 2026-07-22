"""Run one comparable recurrent-agent tournament across several tickers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from trading_bot.training.dataset import SnapshotDataset
from trading_bot.training.trainer import TrainingConfig
from trading_bot.training.walk_forward import (
    ModelSpec,
    WalkForwardConfig,
    run_walk_forward_training,
)


AGENT_ARENA_SCHEMA_VERSION = "research-demo.agent-arena.v7"
DEFAULT_ARENA_SYMBOLS = ("AAPL", "NVDA", "MSFT", "AMZN", "GOOG")
DEFAULT_ARENA_TRAINING_SEED_OFFSETS = (0, 1, 2)
DEFAULT_ARENA_SELECTION_SCORE_TOLERANCE = 1e-4
DEFAULT_ARENA_ACTIVATION_MIN_SCORE_ADVANTAGE = 1e-4
DEFAULT_ARENA_LATEST_FOLD_ONLY = True
DEFAULT_ARENA_OUTPUT_ROOT = Path("data/agent_runs/recurrent-arena")


def default_arena_output_dir(
    created_at: datetime | None = None,
) -> Path:
    """Return a timestamped run directory so prior evidence is never overwritten."""
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("arena output timestamp must be timezone-aware")
    run_id = timestamp.astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    return DEFAULT_ARENA_OUTPUT_ROOT / run_id


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
    flat_smile_residual_ablations = tuple(
        ModelSpec(
            kind=kind,
            encoder="flat",
            hidden_size=hidden_size,
            initial_hold_bias=initial_hold_bias,
            algorithm="ppo",
            action_decoder="single_leg",
            disabled_feature_groups=("contract_smile_residual",),
        )
        for kind in ("gru", "lstm", "mixture")
    )
    surface_smile_residual_ablations = tuple(
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
            disabled_feature_groups=("contract_smile_residual",),
        )
        for kind in ("gru", "lstm", "mixture")
    )
    return (
        flat_controls
        + surface_gnn_agents
        + flat_smile_residual_ablations
        + surface_smile_residual_ablations
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
    for symbol in normalized_symbols:
        try:
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
        "completed": completed,
        "failures": failures,
    }
    (output_dir / "agent-arena.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not completed:
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
        help=(
            "repeat to declare training-seed offsets; defaults to 0, 1, and 2"
        ),
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
                    args.training_seed_offset
                    or DEFAULT_ARENA_TRAINING_SEED_OFFSETS
                ),
                selection_score_tolerance=args.selection_score_tolerance,
                activation_min_score_advantage=(
                    args.activation_min_score_advantage
                ),
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
