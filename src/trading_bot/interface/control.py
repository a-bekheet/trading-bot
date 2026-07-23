"""Serialized local process controls for the trading desk."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ControlJob:
    id: str
    kind: str
    label: str
    command: list[str]
    status: str = "queued"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: str | None = None
    completed_at: str | None = None
    return_code: int | None = None
    output: list[str] = field(default_factory=list)


SERVICE_DEFINITIONS = {
    "collector": {
        "label": "Market collector",
        "description": "Top-50 option surfaces, benchmark context, rates, and Greeks.",
        "service_module": "trading_bot.market_data.service",
        "once_module": "trading_bot.market_data.collector",
        "once_args": ("--once", "--output-dir", "{data_dir}"),
        "status_file": "_collector_status.json",
    },
    "training": {
        "label": "Training watcher",
        "description": "Readiness-aware recurrent and surface-GNN arena.",
        "service_module": "trading_bot.training.arena_service",
        "once_module": "trading_bot.training.arena_watch",
        "once_args": ("--once", "--data-dir", "{data_dir}"),
        "status_file": "_arena_watch_status.json",
    },
    "paper_agents": {
        "label": "Paper agents",
        "description": "Selected checkpoint decisions and isolated simulated accounts.",
        "service_module": "trading_bot.execution.agent_service",
        "once_module": "trading_bot.execution.agent_watch",
        "once_args": ("--once", "--data-dir", "{data_dir}"),
        "status_file": "_paper_agent_watch_status.json",
    },
}


class ControlPlane:
    """Run only allow-listed repository commands and expose auditable jobs."""

    def __init__(self, repo_root: Path, data_dir: Path):
        self.repo_root = repo_root.resolve()
        self.data_dir = data_dir.resolve()
        self.jobs: dict[str, ControlJob] = {}
        self._service_locks = {
            name: asyncio.Lock() for name in SERVICE_DEFINITIONS
        }
        self._training_lock = asyncio.Lock()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        try:
            value = int(pid)
            if value <= 0:
                return False
            os.kill(value, 0)
        except (TypeError, ValueError, ProcessLookupError, PermissionError):
            return False
        return True

    def services(self) -> list[dict[str, Any]]:
        records = []
        for name, definition in SERVICE_DEFINITIONS.items():
            status = self._read_json(self.data_dir / definition["status_file"])
            raw_status = str(status.get("status", "not_started"))
            pid = status.get("pid")
            alive = self._pid_alive(pid) if pid is not None else False
            healthy_states = {
                "complete",
                "up_to_date",
                "waiting",
                "running",
                "active",
                "ok",
            }
            records.append(
                {
                    "id": name,
                    "label": definition["label"],
                    "description": definition["description"],
                    "status": raw_status,
                    "healthy": raw_status in healthy_states,
                    "running": alive or raw_status == "running",
                    "pid": pid,
                    "last_heartbeat_at": status.get("last_heartbeat_at")
                    or status.get("updated_at")
                    or status.get("cycle_finished_at"),
                    "message": status.get("message", ""),
                    "completed_count": int(status.get("completed_count", 0) or 0),
                    "failure_count": int(status.get("failure_count", 0) or 0),
                }
            )
        return records

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            asdict(job)
            for job in sorted(
                self.jobs.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
        ]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        return asdict(job) if job else None

    def _new_job(self, kind: str, label: str, command: list[str]) -> ControlJob:
        job = ControlJob(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            label=label,
            command=command,
        )
        self.jobs[job.id] = job
        return job

    async def _execute(
        self,
        job: ControlJob,
        lock: asyncio.Lock,
    ) -> None:
        async with lock:
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
            try:
                process = await asyncio.create_subprocess_exec(
                    *job.command,
                    cwd=self.repo_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                assert process.stdout is not None
                while line := await process.stdout.readline():
                    job.output.append(
                        line.decode("utf-8", errors="replace").rstrip()
                    )
                    if len(job.output) > 500:
                        del job.output[:100]
                job.return_code = await process.wait()
                job.status = "complete" if job.return_code == 0 else "failed"
            except Exception as error:  # pragma: no cover - OS boundary
                job.status = "failed"
                job.return_code = -1
                job.output.append(f"{type(error).__name__}: {error}")
            finally:
                job.completed_at = datetime.now(timezone.utc).isoformat()

    def service_action(self, service: str, action: str) -> dict[str, Any]:
        if service not in SERVICE_DEFINITIONS:
            raise ValueError(f"unknown service: {service}")
        if action not in {"start", "stop", "restart", "run_once"}:
            raise ValueError(f"unsupported service action: {action}")
        definition = SERVICE_DEFINITIONS[service]
        if action == "run_once":
            arguments = [
                value.format(data_dir=str(self.data_dir))
                for value in definition["once_args"]
            ]
            command = [
                sys.executable,
                "-m",
                definition["once_module"],
                *arguments,
            ]
        else:
            service_action = "uninstall" if action == "stop" else "install"
            command = [
                sys.executable,
                "-m",
                definition["service_module"],
                service_action,
                "--repo-root",
                str(self.repo_root),
                "--data-dir" if service != "collector" else "--output-dir",
                str(self.data_dir),
            ]
        job = self._new_job(
            f"service:{service}",
            f"{definition['label']} · {action.replace('_', ' ')}",
            command,
        )
        asyncio.create_task(self._execute(job, self._service_locks[service]))
        if action == "restart":
            job.output.append(
                "Restart uses the service installer, which replaces the existing service."
            )
        return asdict(job)

    def start_training(
        self,
        *,
        symbols: list[str],
        episodes: int,
        hidden_size: int,
        sequence_length: int,
        max_steps: int,
    ) -> dict[str, Any]:
        if not symbols:
            raise ValueError("select at least one symbol")
        normalized = []
        for symbol in symbols:
            value = symbol.strip().upper()
            if not value.isalnum() or len(value) > 8:
                raise ValueError(f"invalid ticker: {symbol}")
            if value not in normalized:
                normalized.append(value)
        if not 1 <= episodes <= 100:
            raise ValueError("episodes must be between 1 and 100")
        if hidden_size not in {8, 16, 32, 64, 128}:
            raise ValueError("hidden size must be one of 8, 16, 32, 64, 128")
        if not 1 <= sequence_length <= 64:
            raise ValueError("sequence length must be between 1 and 64")
        if not 2 <= max_steps <= 512:
            raise ValueError("max steps must be between 2 and 512")
        command = [
            sys.executable,
            "-m",
            "trading_bot.training.arena",
            "--data-dir",
            str(self.data_dir),
            "--episodes",
            str(episodes),
            "--hidden-size",
            str(hidden_size),
            "--sequence-length",
            str(sequence_length),
            "--max-steps",
            str(max_steps),
        ]
        for symbol in normalized:
            command.extend(("--symbol", symbol))
        job = self._new_job(
            "training",
            f"Agent arena · {', '.join(normalized)}",
            command,
        )
        asyncio.create_task(self._execute(job, self._training_lock))
        return asdict(job)
