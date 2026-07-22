"""Atomic state and decision ledger for isolated paper trading agents."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


AGENT_STORE_SCHEMA_VERSION = "research-demo.paper-agent-store.v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentPaperStore:
    """Persist model-bound portfolios and decisions in one SQLite transaction."""

    def __init__(self, database: Path):
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_deployments (
                    deployment_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    topology TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    checkpoint_sha256 TEXT NOT NULL,
                    activated INTEGER NOT NULL CHECK (activated IN (0, 1)),
                    activation_reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_observation_timestamp TEXT,
                    last_decision_timestamp TEXT,
                    last_cash REAL,
                    last_nav REAL,
                    decision_count INTEGER NOT NULL CHECK (decision_count >= 0),
                    execution_count INTEGER NOT NULL CHECK (execution_count >= 0),
                    environment_state_json TEXT,
                    recurrent_state_json TEXT,
                    UNIQUE(agent_id, checkpoint_sha256)
                );

                CREATE TABLE IF NOT EXISTS agent_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id TEXT NOT NULL REFERENCES agent_deployments(deployment_id),
                    snapshot_timestamp TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    activated INTEGER NOT NULL CHECK (activated IN (0, 1)),
                    research_orders_json TEXT NOT NULL,
                    sandbox_orders_json TEXT NOT NULL,
                    executions_json TEXT NOT NULL,
                    reward REAL NOT NULL,
                    reward_horizon TEXT NOT NULL DEFAULT 'unknown',
                    cash REAL NOT NULL,
                    nav REAL NOT NULL,
                    invalid_action_count INTEGER NOT NULL,
                    UNIQUE(deployment_id, snapshot_timestamp)
                );

                CREATE INDEX IF NOT EXISTS agent_decisions_deployment_time
                    ON agent_decisions(deployment_id, snapshot_timestamp DESC);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(agent_deployments)"
                ).fetchall()
            }
            for name in ("last_cash", "last_nav"):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE agent_deployments ADD COLUMN {name} REAL"
                    )
            decision_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(agent_decisions)"
                ).fetchall()
            }
            if "reward_horizon" not in decision_columns:
                connection.execute(
                    "ALTER TABLE agent_decisions ADD COLUMN reward_horizon "
                    "TEXT NOT NULL DEFAULT 'unknown'"
                )

    def deployment(self, deployment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_deployments WHERE deployment_id = ?",
                (deployment_id,),
            ).fetchone()
        return self._decode_deployment(row) if row else None

    def deployments(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_deployments ORDER BY updated_at DESC"
            ).fetchall()
        return [self._decode_deployment(row) for row in rows]

    def decisions(
        self,
        *,
        deployment_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._connect() as connection:
            if deployment_id is None:
                rows = connection.execute(
                    "SELECT * FROM agent_decisions ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM agent_decisions WHERE deployment_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (deployment_id, limit),
                ).fetchall()
        return [self._decode_decision(row) for row in rows]

    def commit_cycle(
        self,
        deployment: dict[str, Any],
        decisions: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Atomically store the newest cursor and every new decision."""
        required = {
            "deployment_id", "agent_id", "symbol", "model_id", "topology",
            "checkpoint_path", "checkpoint_sha256", "activated",
            "activation_reason", "status", "message",
        }
        missing = required - set(deployment)
        if missing:
            raise ValueError(f"deployment is missing fields: {sorted(missing)}")
        timestamp = _now()
        environment_state = deployment.get("environment_state")
        recurrent_state = deployment.get("recurrent_state")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT started_at FROM agent_deployments WHERE deployment_id = ?",
                (deployment["deployment_id"],),
            ).fetchone()
            started_at = existing["started_at"] if existing else timestamp
            connection.execute(
                """
                INSERT INTO agent_deployments (
                    deployment_id, schema_version, agent_id, symbol, model_id,
                    topology, checkpoint_path, checkpoint_sha256, activated,
                    activation_reason, status, message, started_at, updated_at,
                    last_observation_timestamp, last_decision_timestamp,
                    last_cash, last_nav, decision_count, execution_count,
                    environment_state_json,
                    recurrent_state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
                ON CONFLICT(deployment_id) DO UPDATE SET
                    activated = excluded.activated,
                    activation_reason = excluded.activation_reason,
                    status = excluded.status,
                    message = excluded.message,
                    updated_at = excluded.updated_at,
                    last_observation_timestamp = excluded.last_observation_timestamp,
                    last_decision_timestamp = excluded.last_decision_timestamp,
                    last_cash = excluded.last_cash,
                    last_nav = excluded.last_nav,
                    environment_state_json = excluded.environment_state_json,
                    recurrent_state_json = excluded.recurrent_state_json
                """,
                (
                    deployment["deployment_id"],
                    AGENT_STORE_SCHEMA_VERSION,
                    deployment["agent_id"],
                    str(deployment["symbol"]).upper(),
                    deployment["model_id"],
                    deployment["topology"],
                    str(deployment["checkpoint_path"]),
                    deployment["checkpoint_sha256"],
                    int(bool(deployment["activated"])),
                    deployment["activation_reason"],
                    deployment["status"],
                    deployment["message"],
                    started_at,
                    timestamp,
                    deployment.get("last_observation_timestamp"),
                    deployment.get("last_decision_timestamp"),
                    deployment.get("last_cash"),
                    deployment.get("last_nav"),
                    _json(environment_state) if environment_state is not None else None,
                    _json(recurrent_state) if recurrent_state is not None else None,
                ),
            )
            for decision in decisions:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO agent_decisions (
                        deployment_id, snapshot_timestamp, processed_at,
                        activated, research_orders_json, sandbox_orders_json,
                        executions_json, reward, reward_horizon, cash, nav,
                        invalid_action_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        deployment["deployment_id"],
                        decision["snapshot_timestamp"],
                        decision.get("processed_at", timestamp),
                        int(bool(decision["activated"])),
                        _json(decision["research_orders"]),
                        _json(decision["sandbox_orders"]),
                        _json(decision["executions"]),
                        float(decision["reward"]),
                        str(decision.get("reward_horizon", "unknown")),
                        float(decision["cash"]),
                        float(decision["nav"]),
                        int(decision["invalid_action_count"]),
                    ),
                )
            connection.execute(
                """
                UPDATE agent_deployments SET
                    decision_count = (
                        SELECT COUNT(*) FROM agent_decisions
                        WHERE deployment_id = agent_deployments.deployment_id
                    ),
                    execution_count = (
                        SELECT COALESCE(SUM(json_array_length(executions_json)), 0)
                        FROM agent_decisions
                        WHERE deployment_id = agent_deployments.deployment_id
                    )
                WHERE deployment_id = ?
                """,
                (deployment["deployment_id"],),
            )
        stored = self.deployment(str(deployment["deployment_id"]))
        if stored is None:  # pragma: no cover - transaction invariant
            raise RuntimeError("paper-agent deployment was not persisted")
        return stored

    @staticmethod
    def _decode_deployment(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["activated"] = bool(value["activated"])
        for source, target in (
            ("environment_state_json", "environment_state"),
            ("recurrent_state_json", "recurrent_state"),
        ):
            raw = value.pop(source)
            value[target] = json.loads(raw) if raw else None
        return value

    @staticmethod
    def _decode_decision(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["activated"] = bool(value["activated"])
        for source, target in (
            ("research_orders_json", "research_orders"),
            ("sandbox_orders_json", "sandbox_orders"),
            ("executions_json", "executions"),
        ):
            value[target] = json.loads(value.pop(source))
        return value
