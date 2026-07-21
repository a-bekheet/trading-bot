"""Dataset and environment provenance manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trading_bot.training.schemas import SCHEMA_VERSION


@dataclass(frozen=True)
class EnvManifest:
    """All configuration needed to reproduce an environment instance."""

    schema_version: str = SCHEMA_VERSION
    mode: str = "research_demo"
    data_source: str = "local-csv-yahoo-snapshots"
    data_hash: str = ""
    symbol: str = ""
    slot_count: int = 32
    max_quantity: int = 3
    starting_cash: float = 100_000.0
    commission_per_contract: float = 0.65
    spread_multiplier: float = 1.0
    underlying_lot_size: int = 25
    max_abs_underlying_shares: int = 500
    underlying_commission_per_share: float = 0.005
    underlying_slippage_bps: float = 1.0
    invalid_action_penalty: float = 0.001
    max_abs_delta: float | None = None
    max_abs_gamma: float | None = None
    max_abs_theta: float | None = None
    max_abs_vega: float | None = None
    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_directory(cls, data_dir: Path, **kwargs: Any) -> "EnvManifest":
        digest = hashlib.sha256()
        symbol = str(kwargs.get("symbol", "")).upper()
        paths = (
            [data_dir / f"{symbol}.csv"]
            if symbol
            else sorted(data_dir.glob("*.csv"))
        )
        for path in paths:
            if not path.is_file():
                continue
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return cls(data_hash=digest.hexdigest(), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()
