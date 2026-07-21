"""Small, auditable actor-critic trainer for the research-demo environment."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trading_bot.training.env import CONTRACT_FEATURES, OptionsEnv
from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic
from trading_bot.training.sequence import observation_vector


CHECKPOINT_SCHEMA_VERSION = "research-demo.actor-critic.v1"


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 25
    sequence_length: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gradient_clip: float = 0.5
    seed: int = 7
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.sequence_length < 1:
            raise ValueError("episodes and sequence_length must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between zero and one")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("loss coefficients cannot be negative")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error
    return torch


def _sequence_tensor(history: deque[np.ndarray], length: int, torch):
    feature_count = history[-1].shape[0]
    padded = np.zeros((length, feature_count), dtype=np.float32)
    available = list(history)[-length:]
    padded[-len(available):] = np.stack(available)
    return torch.from_numpy(padded).unsqueeze(0)


def train_actor_critic(
    env: OptionsEnv,
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig | None = None,
):
    """Train one recurrent policy and return ``(model, episode metrics)``.

    This is an integration trainer for the deterministic research surface. It
    deliberately makes no claim about historical or live trading performance.
    """
    torch = _torch()
    config = training_config or TrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    observation, _ = env.reset(seed=config.seed)
    actual_input_size = observation_vector(observation).shape[0]
    if recurrent_config.input_size != actual_input_size:
        raise ValueError(
            f"model input_size={recurrent_config.input_size}, environment emits {actual_input_size}"
        )
    if (recurrent_config.slot_count, recurrent_config.action_count) != env.action_shape:
        raise ValueError("model action dimensions do not match the environment")

    model = build_recurrent_actor_critic(recurrent_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metrics: list[dict[str, float | int]] = []

    for episode in range(config.episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        history: deque[np.ndarray] = deque(maxlen=config.sequence_length)
        rewards = []
        log_probabilities = []
        entropies = []
        values = []
        invalid_actions = executions = steps = 0

        while True:
            history.append(observation_vector(observation))
            sequence = _sequence_tensor(history, config.sequence_length, torch)
            action_mask = torch.from_numpy(observation.action_mask).unsqueeze(0)
            logits, value, _ = model(sequence, action_mask)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()

            observation, reward, terminated, truncated, info = env.step(
                action.squeeze(0).detach().cpu().numpy()
            )
            rewards.append(float(reward))
            log_probabilities.append(distribution.log_prob(action).sum(dim=-1).squeeze(0))
            entropies.append(distribution.entropy().sum(dim=-1).squeeze(0))
            values.append(value.squeeze(0))
            invalid_actions += int(info["invalid_action_count"])
            executions += len(info["executions"])
            steps += 1
            reached_limit = config.max_steps is not None and steps >= config.max_steps
            if terminated or truncated or reached_limit:
                break

        discounted = 0.0
        returns = []
        for reward in reversed(rewards):
            discounted = reward + config.gamma * discounted
            returns.append(discounted)
        returns_tensor = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        values_tensor = torch.stack(values)
        advantages = returns_tensor - values_tensor.detach()
        policy_loss = -(torch.stack(log_probabilities) * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values_tensor, returns_tensor)
        entropy = torch.stack(entropies).mean()
        loss = (
            policy_loss
            + config.value_coefficient * value_loss
            - config.entropy_coefficient * entropy
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        metrics.append(
            {
                "episode": episode,
                "steps": steps,
                "total_reward": float(sum(rewards)),
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "gradient_norm": float(gradient_norm),
                "invalid_actions": invalid_actions,
                "executions": executions,
            }
        )

    return model, metrics


def checkpoint_manifest(
    env: OptionsEnv,
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "mode": "research_demo",
        "environment": env.manifest.to_dict(),
        "environment_fingerprint": env.manifest.fingerprint,
        "model": asdict(recurrent_config),
        "training": asdict(training_config),
        "metrics": metrics,
    }


def save_checkpoint(
    path: Path,
    model,
    env: OptionsEnv,
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig,
    metrics: list[dict[str, Any]],
) -> Path:
    """Save model weights plus a human-readable provenance sidecar."""
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint_manifest(env, recurrent_config, training_config, metrics)
    torch.save({"state_dict": model.state_dict(), "manifest": manifest}, path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--kind", choices=("gru", "lstm", "hybrid"), default="gru")
    parser.add_argument("--encoder", choices=("flat", "graph"), default="flat")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--slot-count", type=int, default=32)
    parser.add_argument("--max-quantity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    env = OptionsEnv.from_directory(
        args.data_dir,
        args.symbol,
        slot_count=args.slot_count,
        max_quantity=args.max_quantity,
    )
    observation, _ = env.reset(seed=args.seed)
    recurrent_config = RecurrentConfig(
        input_size=observation_vector(observation).shape[0],
        slot_count=env.action_shape[0],
        action_count=env.action_shape[1],
        hidden_size=args.hidden_size,
        kind=args.kind,
        encoder=args.encoder,
        contract_feature_count=observation.contracts.shape[1],
        graph_relation_indices=tuple(
            CONTRACT_FEATURES.index(name)
            for name in ("impliedVolatility", "delta", "logMoneyness", "dteDays")
        ),
    )
    training_config = TrainingConfig(
        episodes=args.episodes,
        sequence_length=args.sequence_length,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    model, metrics = train_actor_critic(env, recurrent_config, training_config)
    output = args.output or Path("data/models") / (
        f"{env.dataset.symbol}-{args.encoder}-{args.kind}.pt"
    )
    save_checkpoint(output, model, env, recurrent_config, training_config, metrics)
    print(json.dumps({"checkpoint": str(output), "last_episode": metrics[-1]}, sort_keys=True))


if __name__ == "__main__":
    main()
