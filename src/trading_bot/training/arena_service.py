"""Install or remove the readiness-aware arena watcher as a macOS LaunchAgent."""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


LAUNCH_AGENT_LABEL = "io.github.a-bekheet.trading-bot.arena"


def launch_agent_payload(
    python: Path,
    repo_root: Path,
    data_dir: Path,
    *,
    poll_seconds: float,
) -> dict[str, Any]:
    """Build a deterministic LaunchAgent property list."""
    return {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [
            str(python),
            "-m",
            "trading_bot.training.arena_watch",
            "--data-dir",
            str(data_dir),
            "--poll-seconds",
            str(poll_seconds),
        ],
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
        "StandardOutPath": str(data_dir / "arena-watch.stdout.log"),
        "StandardErrorPath": str(data_dir / "arena-watch.stderr.log"),
    }


def _launchctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("launchctl", *arguments),
        check=check,
        text=True,
        capture_output=True,
    )


def _bootstrap_launch_agent(domain: str, path: Path) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for delay in (0.0, 0.25, 1.0):
        if delay:
            time.sleep(delay)
        try:
            _launchctl("bootstrap", domain, str(path))
            return
        except subprocess.CalledProcessError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def install(repo_root: Path, data_dir: Path, *, poll_seconds: float) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("arena-service currently supports macOS only")
    repo_root = repo_root.resolve()
    if not (repo_root / "pyproject.toml").is_file():
        raise ValueError(f"not a trading-bot repository: {repo_root}")
    data_dir = (data_dir if data_dir.is_absolute() else repo_root / data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable).absolute()
    payload = launch_agent_payload(
        python,
        repo_root,
        data_dir,
        poll_seconds=poll_seconds,
    )
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    path = launch_agents / f"{LAUNCH_AGENT_LABEL}.plist"
    temporary = path.with_suffix(".plist.tmp")
    temporary.write_bytes(plistlib.dumps(payload, sort_keys=True))
    temporary.replace(path)

    domain = f"gui/{os.getuid()}"
    _launchctl("bootout", f"{domain}/{LAUNCH_AGENT_LABEL}", check=False)
    _bootstrap_launch_agent(domain, path)
    return path


def uninstall() -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("arena-service currently supports macOS only")
    path = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    _launchctl("bootout", f"{domain}/{LAUNCH_AGENT_LABEL}", check=False)
    path.unlink(missing_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "uninstall"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    try:
        if args.action == "install":
            path = install(
                args.repo_root,
                args.data_dir,
                poll_seconds=args.poll_seconds,
            )
            print(f"installed and started {LAUNCH_AGENT_LABEL}: {path}")
        else:
            path = uninstall()
            print(f"stopped and removed {LAUNCH_AGENT_LABEL}: {path}")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"arena service error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
