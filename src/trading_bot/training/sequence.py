"""Fixed-shape sequence windows for recurrent policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_bot.training.schemas import Observation


def observation_vector(observation: Observation) -> np.ndarray:
    """Flatten one observation with a stable ordering for LSTM/GRU inputs."""
    return np.concatenate(
        (
            observation.market.ravel(),
            observation.contracts.ravel(),
            observation.portfolio.ravel(),
            observation.valid_mask.astype(np.float64).ravel(),
        )
    ).astype(np.float32)


@dataclass(frozen=True)
class SequenceWindow:
    features: np.ndarray
    actions: np.ndarray | None = None
    rewards: np.ndarray | None = None


def build_windows(
    observations: list[Observation],
    *,
    window: int,
    actions: list[np.ndarray] | None = None,
    rewards: list[float] | None = None,
) -> list[SequenceWindow]:
    """Build chronological, non-padded windows; no future rows are included."""
    if window < 1:
        raise ValueError("window must be positive")
    if len(observations) < window:
        return []
    vectors = np.stack([observation_vector(item) for item in observations])
    result = []
    for end in range(window, len(observations) + 1):
        start = end - window
        result.append(
            SequenceWindow(
                features=vectors[start:end],
                actions=np.stack(actions[start:end]) if actions is not None else None,
                rewards=np.asarray(rewards[start:end], dtype=np.float32) if rewards is not None else None,
            )
        )
    return result
