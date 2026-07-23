"""Launch the local options control room."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path
from typing import Sequence

import uvicorn

from trading_bot.interface.api import create_app


def parser() -> argparse.ArgumentParser:
    launch_parser = argparse.ArgumentParser(
        description="Launch the local options research and paper-trading application."
    )
    launch_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Saved market, agent, and paper-account directory.",
    )
    launch_parser.add_argument(
        "--address",
        default="127.0.0.1",
        help="Interface address (default: 127.0.0.1).",
    )
    launch_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Interface port (default: 8765).",
    )
    launch_parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open the application in the default browser.",
    )
    launch_parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the API when Python source changes.",
    )
    return launch_parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the API and compiled web application."""
    arguments = parser().parse_args(argv)
    if not 1 <= arguments.port <= 65_535:
        parser().error("--port must be between 1 and 65535")
    repo_root = Path.cwd().resolve()
    if not (repo_root / "pyproject.toml").is_file():
        parser().error("run trading-desk from the trading-bot repository")
    data_dir = arguments.data_dir.expanduser().resolve()
    static_dir = Path(__file__).with_name("web_dist")
    if not (static_dir / "index.html").is_file():
        parser().error(
            "compiled web application is missing; run `npm run build --prefix frontend`"
        )
    application = create_app(
        repo_root=repo_root,
        data_dir=data_dir,
        static_dir=static_dir,
    )
    url = f"http://{arguments.address}:{arguments.port}"
    if not arguments.headless:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.run(
        application,
        host=arguments.address,
        port=arguments.port,
        reload=arguments.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
