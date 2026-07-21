"""Dataset and environment provenance manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvManifest:
    """All configuration needed to reproduce an environment instance."""

    schema_version: str = "research-demo.v1"
    mode: str = "research_demo"
    data_source: str = "local-csv-yahoo-snapshots"
    data_hash: str = ""
    symbol: str = ""
    slot_count: int = 32
    max_quantity: int = 3
    starting_cash: float = 100_000.0
    commission_per_contract: float = 0.65
    invalid_action_penalty: float = 0.001
    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_directory(cls, data_dir: Path, **kwargs: Any) -> "EnvManifest":
        digest = hashlib.sha256()
        for path in sorted(data_dir.glob("*.csv")):
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
