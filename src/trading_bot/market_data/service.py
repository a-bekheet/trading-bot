"""Install or remove the continuous collector as a macOS LaunchAgent."""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


LAUNCH_AGENT_LABEL = "io.github.a-bekheet.trading-bot.collector"


def launch_agent_payload(
    python: Path,
    repo_root: Path,
    output_dir: Path,
    *,
    interval: int,
    ticker_delay: float,
    expirations: int,
) -> dict[str, Any]:
    """Build a deterministic LaunchAgent property list."""
    return {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [
            str(python),
            "-m",
            "trading_bot.market_data.collector",
            "--output-dir",
            str(output_dir),
            "--interval",
            str(interval),
            "--ticker-delay",
            str(ticker_delay),
            "--expirations",
            str(expirations),
        ],
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
        "StandardOutPath": str(output_dir / "collector.stdout.log"),
        "StandardErrorPath": str(output_dir / "collector.stderr.log"),
    }


def _launchctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("launchctl", *arguments),
        check=check,
        text=True,
        capture_output=True,
    )


def _bootstrap_launch_agent(domain: str, path: Path) -> None:
    """Retry the short macOS bootout/bootstrap teardown race."""
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


def install(
    repo_root: Path,
    output_dir: Path,
    *,
    interval: int,
    ticker_delay: float,
    expirations: int,
) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("collector-service currently supports macOS only")
    repo_root = repo_root.resolve()
    if not (repo_root / "pyproject.toml").is_file():
        raise ValueError(f"not a trading-bot repository: {repo_root}")
    output_dir = (
        output_dir if output_dir.is_absolute() else repo_root / output_dir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the virtual-environment launcher path. Resolving its symlink can
    # accidentally select the base interpreter without this project's packages.
    python = Path(sys.executable).absolute()
    payload = launch_agent_payload(
        python,
        repo_root,
        output_dir,
        interval=interval,
        ticker_delay=ticker_delay,
        expirations=expirations,
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
        raise RuntimeError("collector-service currently supports macOS only")
    path = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    _launchctl("bootout", f"{domain}/{LAUNCH_AGENT_LABEL}", check=False)
    path.unlink(missing_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "uninstall"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--ticker-delay", type=float, default=1.0)
    parser.add_argument("--expirations", type=int, default=3)
    args = parser.parse_args()
    if args.interval < 1 or args.ticker_delay < 0 or args.expirations < 0:
        parser.error(
            "--interval must be positive; delays and expiration count cannot be negative"
        )
    try:
        if args.action == "install":
            path = install(
                args.repo_root,
                args.output_dir,
                interval=args.interval,
                ticker_delay=args.ticker_delay,
                expirations=args.expirations,
            )
            print(f"installed and started {LAUNCH_AGENT_LABEL}: {path}")
        else:
            path = uninstall()
            print(f"stopped and removed {LAUNCH_AGENT_LABEL}: {path}")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"collector service error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
