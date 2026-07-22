from pathlib import Path
from unittest import TestCase
from unittest.mock import call, patch

from trading_bot.training.arena_service import (
    LAUNCH_AGENT_LABEL,
    _bootstrap_launch_agent,
    launch_agent_payload,
)


class ArenaServiceTests(TestCase):
    def test_launch_agent_runs_readiness_watcher_with_absolute_paths(self):
        payload = launch_agent_payload(
            Path("/repo/.venv/bin/python"),
            Path("/repo"),
            Path("/repo/data"),
            poll_seconds=60.0,
        )

        self.assertEqual(payload["Label"], LAUNCH_AGENT_LABEL)
        self.assertEqual(payload["WorkingDirectory"], "/repo")
        self.assertEqual(
            payload["ProgramArguments"],
            [
                "/repo/.venv/bin/python",
                "-m",
                "trading_bot.training.arena_watch",
                "--data-dir",
                "/repo/data",
                "--poll-seconds",
                "60.0",
            ],
        )
        self.assertTrue(payload["RunAtLoad"])
        self.assertEqual(payload["KeepAlive"], {"SuccessfulExit": False})
        self.assertEqual(
            payload["StandardErrorPath"],
            "/repo/data/arena-watch.stderr.log",
        )

    @patch("trading_bot.training.arena_service.time.sleep")
    @patch("trading_bot.training.arena_service._launchctl")
    def test_bootstrap_retries_transient_launchctl_failure(self, launchctl, sleep):
        from subprocess import CalledProcessError

        launchctl.side_effect = (
            CalledProcessError(5, ("launchctl", "bootstrap")),
            None,
        )

        _bootstrap_launch_agent("gui/501", Path("/tmp/arena.plist"))

        self.assertEqual(launchctl.call_count, 2)
        self.assertEqual(sleep.call_args_list, [call(0.25)])
