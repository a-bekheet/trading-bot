from pathlib import Path
from unittest import TestCase
from unittest.mock import call, patch

from trading_bot.market_data.service import (
    LAUNCH_AGENT_LABEL,
    _bootstrap_launch_agent,
    launch_agent_payload,
)


class CollectorServiceTests(TestCase):
    def test_launch_agent_runs_collector_with_absolute_paths_and_restart(self):
        payload = launch_agent_payload(
            Path("/repo/.venv/bin/python"),
            Path("/repo"),
            Path("/repo/data"),
            interval=900,
            ticker_delay=0.5,
            expirations=1,
        )

        self.assertEqual(payload["Label"], LAUNCH_AGENT_LABEL)
        self.assertEqual(payload["WorkingDirectory"], "/repo")
        self.assertEqual(
            payload["ProgramArguments"],
            [
                "/repo/.venv/bin/python",
                "-m",
                "trading_bot.market_data.collector",
                "--output-dir",
                "/repo/data",
                "--interval",
                "900",
                "--ticker-delay",
                "0.5",
                "--benchmark-symbol",
                "SPY",
                "--expirations",
                "1",
            ],
        )
        self.assertTrue(payload["RunAtLoad"])
        self.assertEqual(payload["KeepAlive"], {"SuccessfulExit": False})
        self.assertEqual(
            payload["StandardErrorPath"],
            "/repo/data/collector.stderr.log",
        )

    @patch("trading_bot.market_data.service.time.sleep")
    @patch("trading_bot.market_data.service._launchctl")
    def test_bootstrap_retries_transient_launchctl_failure(self, launchctl, sleep):
        from subprocess import CalledProcessError

        launchctl.side_effect = (
            CalledProcessError(5, ("launchctl", "bootstrap")),
            None,
        )

        _bootstrap_launch_agent("gui/501", Path("/tmp/collector.plist"))

        self.assertEqual(launchctl.call_count, 2)
        self.assertEqual(sleep.call_args_list, [call(0.25)])
