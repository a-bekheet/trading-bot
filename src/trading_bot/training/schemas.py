"""Versioned, serializable types shared by the research environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


SCHEMA_VERSION = "research-demo.v1"


@dataclass(frozen=True)
class Observation:
    """Fixed-shape state exposed to a future policy."""

    timestamp: str
    market: np.ndarray
    contracts: np.ndarray
    portfolio: np.ndarray
    valid_mask: np.ndarray
    action_mask: np.ndarray
    contract_ids: tuple[str | None, ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class Action:
    """Per-slot action: 0 hold, 1..Q buy, Q+1..2Q sell."""

    orders: np.ndarray


@dataclass(frozen=True)
class Transition:
    """Auditable environment transition."""

    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
