"""Run selected recurrent policies against isolated persistent paper accounts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from trading_bot.execution.agent_store import AgentPaperStore
from trading_bot.market_data.freshness import (
    DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS,
)
from trading_bot.training.arena import snapshot_execution_readiness
from trading_bot.training.dataset import SnapshotDataset
from trading_bot.training.env import OptionsEnv
from trading_bot.training.trainer import (
    RecurrentPolicyState,
    StreamingRecurrentPolicy,
    load_checkpoint,
)


PAPER_AGENT_RUNTIME_SCHEMA_VERSION = "research-demo.paper-agent-runtime.v1"
DEFAULT_AGENT_DATABASE = Path("data/agent_paper.db")
ENVIRONMENT_OPTION_NAMES = (
    "slot_count",
    "slot_assignment",
    "max_quantity",
    "allow_collateralized_option_shorts",
    "starting_cash",
    "commission_per_contract",
    "spread_multiplier",
    "portfolio_valuation",
    "underlying_lot_size",
    "max_abs_underlying_shares",
    "underlying_commission_per_share",
    "underlying_slippage_bps",
    "invalid_action_penalty",
    "reward_drawdown_penalty",
    "reward_downside_penalty",
    "max_abs_delta",
    "max_abs_gamma",
    "max_abs_theta",
    "max_abs_vega",
    "max_underlying_quote_age_seconds",
)


@contextmanager
def agent_runtime_lock(database: Path) -> Iterator[None]:
    """Prevent a manual cycle and service cycle from racing one portfolio."""
    database.parent.mkdir(parents=True, exist_ok=True)
    path = database.with_suffix(database.suffix + ".lock")
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another paper-agent cycle already owns {path}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _timestamp(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid timestamp: {value!r}")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(value: Any, *, repo_root: Path, summary_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FileNotFoundError("selected checkpoint path is missing")
    candidate = Path(str(value))
    choices = (
        candidate,
        repo_root / candidate,
        summary_path.parent / candidate.name,
    )
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    raise FileNotFoundError(f"selected checkpoint is unavailable: {candidate}")


def _topology(model: dict[str, Any]) -> str:
    encoder = str(model.get("encoder", "unknown"))
    if "graph" in encoder:
        return "surface_gnn"
    if "attention" in encoder:
        return "surface_attention"
    return "flat_vector"


def discover_selected_deployments(
    data_dir: Path,
    *,
    repo_root: Path = Path("."),
) -> list[dict[str, Any]]:
    """Return the newest selected walk-forward checkpoint per ticker."""
    paths = sorted(
        {
            path.resolve()
            for pattern in (
                "agent_runs/**/*-walk-forward.json",
                "models/walk-forward/**/*-walk-forward.json",
            )
            for path in data_dir.glob(pattern)
            if path.is_file()
        },
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not str(summary.get("schema_version", "")).startswith(
            "research-demo.walk-forward."
        ):
            continue
        symbol = str(summary.get("symbol", "")).upper()
        folds = summary.get("folds")
        if not symbol or symbol in seen or not isinstance(folds, list) or not folds:
            continue
        fold = max(folds, key=lambda item: int(item.get("fold", -1)))
        selection = fold.get("model_selection", {})
        model_id = str(selection.get("selected_model_id", ""))
        candidate = next(
            (
                item
                for item in selection.get("candidates", [])
                if str(item.get("model_id", "")) == model_id
            ),
            None,
        )
        gate = selection.get("activation_gate", {})
        cutoff = fold.get("test_data_quality", {}).get("last_timestamp")
        if not model_id or not isinstance(candidate, dict) or not cutoff:
            continue
        discovery_error = None
        try:
            checkpoint = _checkpoint_path(
                fold.get("checkpoint"),
                repo_root=repo_root,
                summary_path=path,
            )
        except FileNotFoundError as error:
            discovery_error = str(error)
            declared = fold.get("checkpoint")
            relative = Path(str(declared)) if declared else Path("missing.pt")
            checkpoint = (
                relative if relative.is_absolute() else repo_root / relative
            ).resolve()
        selected.append({
            "agent_id": f"{symbol}-{model_id}",
            "symbol": symbol,
            "model_id": model_id,
            "model": candidate.get("model", {}),
            "topology": _topology(candidate.get("model", {})),
            "checkpoint_path": checkpoint,
            "activated": bool(gate.get("activated", False)),
            "activation_gate": gate,
            "deployment_cutoff": str(cutoff),
            "summary_path": path,
            "discovery_error": discovery_error,
        })
        seen.add(symbol)
    return selected


def _tensor_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {name: _tensor_payload(item) for name, item in value.items()},
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_tensor_payload(item) for item in value]}
    array = value.detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError("recurrent hidden state must be finite")
    return {
        "kind": "tensor",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.reshape(-1).tolist(),
    }


def _payload_tensor(value: Any, torch) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("kind") not in {
        "dict", "tuple", "tensor",
    }:
        raise ValueError("recurrent hidden-state payload is invalid")
    if value["kind"] == "dict":
        items = value.get("items")
        if not isinstance(items, dict):
            raise ValueError("recurrent dictionary state is invalid")
        return {name: _payload_tensor(item, torch) for name, item in items.items()}
    if value["kind"] == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("recurrent tuple state is invalid")
        return tuple(_payload_tensor(item, torch) for item in items)
    shape = value.get("shape")
    values = value.get("values")
    if (
        value.get("dtype") != "float32"
        or not isinstance(shape, list)
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape)
        or not isinstance(values, list)
        or math.prod(shape) != len(values)
    ):
        raise ValueError("recurrent tensor state is invalid")
    tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
    if not torch.isfinite(tensor).all():
        raise ValueError("recurrent tensor state must be finite")
    return tensor


def recurrent_state_to_dict(state: RecurrentPolicyState) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "recurrent_kind": state.recurrent_kind,
        "layers": state.layers,
        "hidden_size": state.hidden_size,
        "model_contract": state.model_contract,
        "steps": state.steps,
        "last_timestamp": state.last_timestamp,
        "hidden_state": _tensor_payload(state.hidden_state),
    }


def recurrent_state_from_dict(payload: dict[str, Any], torch) -> RecurrentPolicyState:
    if not isinstance(payload, dict):
        raise TypeError("recurrent policy state must be a dictionary")
    required = {
        "schema_version", "recurrent_kind", "layers", "hidden_size",
        "model_contract", "steps", "last_timestamp", "hidden_state",
    }
    if set(payload) != required:
        raise ValueError("recurrent policy state fields are incompatible")
    values = dict(payload)
    values["hidden_state"] = _payload_tensor(values["hidden_state"], torch)
    return RecurrentPolicyState(**values)


def _environment_options(manifest: dict[str, Any]) -> dict[str, Any]:
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("checkpoint is missing its environment contract")
    missing = set(ENVIRONMENT_OPTION_NAMES) - set(environment)
    if missing:
        raise ValueError(f"checkpoint environment is missing: {sorted(missing)}")
    return {name: environment[name] for name in ENVIRONMENT_OPTION_NAMES}


def _eligible_dataset(data_dir: Path, symbol: str, max_age: float) -> SnapshotDataset:
    source = SnapshotDataset.material_from_directory(data_dir, symbol)
    eligible = tuple(
        snapshot
        for snapshot in source.snapshots
        if snapshot_execution_readiness(snapshot, max_age)["eligible"]
    )
    if not eligible:
        raise ValueError("no regular, fresh, executable snapshots are available")
    return SnapshotDataset(eligible, symbol).engineered()


def _activation_reason(gate: dict[str, Any]) -> str:
    advantage = gate.get("score_advantage")
    minimum = gate.get("minimum_score_advantage")
    if isinstance(advantage, (int, float)) and isinstance(minimum, (int, float)):
        return f"validation advantage {advantage:.8f}; required > {minimum:.8f}"
    return "validation activation evidence unavailable"


def _deployment_identity(spec: dict[str, Any], checkpoint_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(PAPER_AGENT_RUNTIME_SCHEMA_VERSION.encode())
    digest.update(b"\0")
    digest.update(spec["agent_id"].encode())
    digest.update(b"\0")
    digest.update(checkpoint_sha256.encode())
    return digest.hexdigest()[:24]


def _zero_action(env: OptionsEnv) -> np.ndarray:
    return np.zeros(env.action_shape[0], dtype=np.int64)


def _decision(
    *,
    timestamp: str,
    activated: bool,
    research_orders: np.ndarray,
    sandbox_orders: np.ndarray,
    reward: float,
    reward_horizon: str,
    observation,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "snapshot_timestamp": timestamp,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "activated": activated,
        "research_orders": [int(value) for value in research_orders],
        "sandbox_orders": [int(value) for value in sandbox_orders],
        "executions": info.get("executions", []),
        "reward": float(reward),
        "reward_horizon": reward_horizon,
        "cash": float(observation.portfolio[0]),
        "nav": float(observation.portfolio[2]),
        "invalid_action_count": int(info.get("invalid_action_count", 0)),
    }


def run_selected_agent(
    spec: dict[str, Any],
    *,
    data_dir: Path,
    store: AgentPaperStore,
) -> dict[str, Any]:
    """Advance one exact selected checkpoint over every unseen eligible state."""
    if spec.get("discovery_error"):
        raise FileNotFoundError(spec["discovery_error"])
    checkpoint_path = Path(spec["checkpoint_path"])
    checkpoint_sha256 = _sha256(checkpoint_path)
    deployment_id = _deployment_identity(spec, checkpoint_sha256)
    model, manifest = load_checkpoint(checkpoint_path)
    provenance = manifest.get("provenance", {})
    checkpoint_selection = provenance.get("model_selection", {})
    if checkpoint_selection.get("selected_model_id") != spec["model_id"]:
        raise ValueError("checkpoint selected model does not match run summary")
    checkpoint_gate = checkpoint_selection.get("activation_gate", {})
    if bool(checkpoint_gate.get("activated", False)) != bool(spec["activated"]):
        raise ValueError("checkpoint activation gate does not match run summary")

    environment_options = _environment_options(manifest)
    max_age = environment_options.get("max_underlying_quote_age_seconds")
    if max_age is None:
        max_age = DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS
    dataset = _eligible_dataset(data_dir, spec["symbol"], float(max_age))
    timestamps = [_timestamp(snapshot.timestamp) for snapshot in dataset.snapshots]
    cutoff = _timestamp(spec["deployment_cutoff"])
    cutoff_matches = [index for index, value in enumerate(timestamps) if value == cutoff]
    if not cutoff_matches:
        raise ValueError("checkpoint held-out cutoff is absent from eligible history")
    cutoff_index = cutoff_matches[-1]

    # The identical terminal frame lets OptionsEnv execute an order at the
    # newest known quote without inventing a future price movement.
    runtime_dataset = SnapshotDataset(
        (*dataset.snapshots, dataset.snapshots[-1]),
        dataset.symbol,
    )
    env = OptionsEnv(runtime_dataset, **environment_options)
    training = manifest.get("training", {})
    sequence_length = int(training.get("sequence_length", 0))
    if sequence_length < 1:
        raise ValueError("checkpoint sequence length is invalid")
    policy = StreamingRecurrentPolicy(model, sequence_length)
    stored = store.deployment(deployment_id)
    decisions: list[dict[str, Any]] = []

    if (
        stored is None
        or stored.get("environment_state") is None
        or stored.get("recurrent_state") is None
    ):
        warm_start = max(0, cutoff_index - sequence_length + 1)
        observation, _ = env.reset(options={"start_index": warm_start})
        for index in range(warm_start, cutoff_index + 1):
            policy(observation)
            if index < cutoff_index:
                observation, _, _, _, _ = env.step(_zero_action(env))
        cursor_index = cutoff_index
    else:
        if stored["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("stored deployment checkpoint binding is incompatible")
        observation, _ = env.restore_state(stored["environment_state"])
        import torch

        policy.restore(recurrent_state_from_dict(stored["recurrent_state"], torch))
        cursor_timestamp = _timestamp(str(stored["last_observation_timestamp"]))
        cursor_matches = [
            index for index, value in enumerate(timestamps) if value == cursor_timestamp
        ]
        if not cursor_matches:
            raise ValueError("stored deployment cursor is absent from eligible history")
        cursor_index = cursor_matches[-1]

    new_indices = [
        index
        for index in range(cursor_index + 1, len(dataset))
        if timestamps[index] > cutoff
    ]
    if new_indices:
        observation, _, _, _, _ = env.step(_zero_action(env))
        if observation.timestamp != dataset.snapshots[new_indices[0]].timestamp:
            raise RuntimeError("paper-agent environment cursor did not advance causally")
        for sequence_index, dataset_index in enumerate(new_indices):
            decision_timestamp = observation.timestamp
            if decision_timestamp != dataset.snapshots[dataset_index].timestamp:
                raise RuntimeError("paper-agent decision cursor is misaligned")
            research_orders = np.asarray(policy(observation), dtype=np.int64)
            sandbox_orders = (
                research_orders.copy()
                if spec["activated"]
                else _zero_action(env)
            )
            next_observation, reward, _, _, info = env.step(sandbox_orders)
            decisions.append(_decision(
                timestamp=decision_timestamp,
                activated=bool(spec["activated"]),
                research_orders=research_orders,
                sandbox_orders=sandbox_orders,
                reward=reward,
                reward_horizon=(
                    "through_next_eligible_snapshot"
                    if sequence_index + 1 < len(new_indices)
                    else "same_snapshot_execution_only"
                ),
                observation=next_observation,
                info=info,
            ))
            observation = next_observation
            expected_next = (
                dataset.snapshots[new_indices[sequence_index + 1]].timestamp
                if sequence_index + 1 < len(new_indices)
                else dataset.snapshots[-1].timestamp
            )
            if observation.timestamp != expected_next:
                raise RuntimeError("paper-agent environment skipped an eligible state")

    last_timestamp = dataset.snapshots[
        new_indices[-1] if new_indices else cursor_index
    ].timestamp
    status = "active" if spec["activated"] else "guarded"
    if not new_indices:
        status = "up_to_date" if stored is not None else "waiting_for_new_snapshot"
    message = (
        f"processed {len(decisions)} new decision(s)"
        if decisions
        else "no eligible post-evaluation snapshot is waiting"
    )
    deployment = {
        "deployment_id": deployment_id,
        "agent_id": spec["agent_id"],
        "symbol": spec["symbol"],
        "model_id": spec["model_id"],
        "topology": spec["topology"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "activated": bool(spec["activated"]),
        "activation_reason": _activation_reason(spec["activation_gate"]),
        "status": status,
        "message": message,
        "last_observation_timestamp": last_timestamp,
        "last_decision_timestamp": (
            decisions[-1]["snapshot_timestamp"]
            if decisions
            else stored.get("last_decision_timestamp") if stored else None
        ),
        "last_cash": float(observation.portfolio[0]),
        "last_nav": float(observation.portfolio[2]),
        "environment_state": env.snapshot_state(),
        "recurrent_state": recurrent_state_to_dict(policy.snapshot()),
    }
    persisted = store.commit_cycle(deployment, decisions)
    return {
        "agent_id": persisted["agent_id"],
        "deployment_id": deployment_id,
        "symbol": persisted["symbol"],
        "topology": persisted["topology"],
        "status": persisted["status"],
        "activated": persisted["activated"],
        "new_decisions": len(decisions),
        "decision_count": persisted["decision_count"],
        "execution_count": persisted["execution_count"],
        "last_observation_timestamp": persisted["last_observation_timestamp"],
        "last_decision_timestamp": persisted["last_decision_timestamp"],
    }


def _run_paper_agents_unlocked(
    *,
    data_dir: Path = Path("data"),
    database: Path = DEFAULT_AGENT_DATABASE,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Advance every newest per-ticker selected policy without cross-account state."""
    store = AgentPaperStore(database)
    specs = discover_selected_deployments(data_dir, repo_root=repo_root)
    results = []
    failures = []
    for spec in specs:
        try:
            results.append(run_selected_agent(spec, data_dir=data_dir, store=store))
        except Exception as error:  # Keep one ticker from blocking the fleet.
            checkpoint_path = Path(spec["checkpoint_path"])
            checkpoint_sha256 = (
                _sha256(checkpoint_path)
                if checkpoint_path.is_file()
                else hashlib.sha256(str(checkpoint_path).encode()).hexdigest()
            )
            deployment_id = _deployment_identity(spec, checkpoint_sha256)
            existing = store.deployment(deployment_id)
            store.commit_cycle(
                {
                    "deployment_id": deployment_id,
                    "agent_id": spec["agent_id"],
                    "symbol": spec["symbol"],
                    "model_id": spec["model_id"],
                    "topology": spec["topology"],
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "activated": bool(spec["activated"]),
                    "activation_reason": _activation_reason(
                        spec["activation_gate"]
                    ),
                    "status": "error",
                    "message": f"{type(error).__name__}: {error}",
                    "last_observation_timestamp": (
                        existing.get("last_observation_timestamp")
                        if existing
                        else None
                    ),
                    "last_decision_timestamp": (
                        existing.get("last_decision_timestamp")
                        if existing
                        else None
                    ),
                    "last_cash": existing.get("last_cash") if existing else None,
                    "last_nav": existing.get("last_nav") if existing else None,
                    "environment_state": (
                        existing.get("environment_state") if existing else None
                    ),
                    "recurrent_state": (
                        existing.get("recurrent_state") if existing else None
                    ),
                },
                [],
            )
            failures.append({
                "agent_id": spec["agent_id"],
                "symbol": spec["symbol"],
                "error_type": type(error).__name__,
                "message": str(error),
            })
    return {
        "schema_version": PAPER_AGENT_RUNTIME_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(database),
        "selected_agent_count": len(specs),
        "completed_count": len(results),
        "failure_count": len(failures),
        "agents": results,
        "failures": failures,
    }


def run_paper_agents(
    *,
    data_dir: Path = Path("data"),
    database: Path = DEFAULT_AGENT_DATABASE,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Advance the fleet while holding one process-wide account lock."""
    with agent_runtime_lock(database):
        return _run_paper_agents_unlocked(
            data_dir=data_dir,
            database=database,
            repo_root=repo_root,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--database", type=Path, default=DEFAULT_AGENT_DATABASE)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_paper_agents(
        data_dir=args.data_dir,
        database=args.database,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
