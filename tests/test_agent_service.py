from pathlib import Path
from unittest import TestCase

from trading_bot.execution.agent_service import (
    LAUNCH_AGENT_LABEL,
    launch_agent_payload,
)


class AgentServiceTests(TestCase):
    def test_launch_agent_runs_change_aware_paper_watcher(self):
        payload = launch_agent_payload(
            Path("/repo/.venv/bin/python"),
            Path("/repo"),
            Path("/repo/data"),
            Path("/repo/data/agent_paper.db"),
            poll_seconds=30.0,
        )

        self.assertEqual(payload["Label"], LAUNCH_AGENT_LABEL)
        self.assertEqual(payload["WorkingDirectory"], "/repo")
        self.assertEqual(
            payload["ProgramArguments"],
            [
                "/repo/.venv/bin/python",
                "-m",
                "trading_bot.execution.agent_watch",
                "--repo-root",
                "/repo",
                "--data-dir",
                "/repo/data",
                "--database",
                "/repo/data/agent_paper.db",
                "--poll-seconds",
                "30.0",
            ],
        )
        self.assertTrue(payload["RunAtLoad"])
        self.assertEqual(payload["KeepAlive"], {"SuccessfulExit": False})
