import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from trading_bot.training.arena_watch import (
    ARENA_WATCH_STATUS_FILENAME,
    ARENA_WATCH_STATUS_SCHEMA_VERSION,
    _exclusive_watch_lock,
    inspect_arena_readiness,
    load_arena_watch_status,
    run_locked_arena,
    run_watch_cycle,
)
from trading_bot.training.arena import arena_walk_forward_config


def ready_items(*symbols: str):
    return [
        {
            "symbol": symbol,
            "ready": True,
            "reason": "ready",
            "test": {"last_timestamp": "2026-07-22T15:30:00+00:00"},
        }
        for symbol in symbols
    ]


class ArenaWatchTests(TestCase):
    @patch("trading_bot.training.arena_watch.run_locked_arena")
    @patch("trading_bot.training.arena_watch.inspect_arena_readiness")
    def test_waiting_status_does_not_start_training(self, inspect, run_arena):
        inspect.return_value = (
            [{"symbol": "AAPL", "ready": False, "reason": "waiting"}],
            None,
        )
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            result = run_watch_cycle(data_dir, ("AAPL",), continuous=True)
            written = load_arena_watch_status(data_dir)

        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["ready_count"], 0)
        self.assertTrue(result["continuous"])
        self.assertEqual(written, result)
        self.assertEqual(written["schema_version"], ARENA_WATCH_STATUS_SCHEMA_VERSION)
        run_arena.assert_not_called()

    @patch("trading_bot.training.arena_watch.run_locked_arena")
    @patch("trading_bot.training.arena_watch.inspect_arena_readiness")
    def test_ready_session_trains_once_then_becomes_up_to_date(
        self,
        inspect,
        run_arena,
    ):
        inspect.return_value = (ready_items("AAPL", "NVDA"), "2026-07-22")
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            artifact = data_dir / "agent-arena.json"
            run_arena.return_value = artifact
            first = run_watch_cycle(
                data_dir,
                ("AAPL", "NVDA"),
                now=datetime(2026, 7, 22, 16, tzinfo=timezone.utc),
            )
            second = run_watch_cycle(
                data_dir,
                ("AAPL", "NVDA"),
                now=datetime(2026, 7, 22, 16, 1, tzinfo=timezone.utc),
            )

        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["last_completed_session_date"], "2026-07-22")
        self.assertEqual(first["last_artifact"], str(artifact))
        self.assertEqual(second["status"], "up_to_date")
        self.assertEqual(run_arena.call_count, 1)

    @patch("trading_bot.training.arena_watch.run_locked_arena")
    @patch("trading_bot.training.arena_watch.inspect_arena_readiness")
    def test_training_error_is_visible_and_retryable(self, inspect, run_arena):
        inspect.return_value = (ready_items("AAPL"), "2026-07-22")
        run_arena.side_effect = RuntimeError("training failed")
        with TemporaryDirectory() as directory:
            result = run_watch_cycle(Path(directory), ("AAPL",))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["message"], "training failed")
        self.assertIsNone(result["last_completed_session_date"])

    @patch("trading_bot.training.arena_watch.arena_tail_readiness")
    @patch("trading_bot.training.arena_watch.SnapshotDataset.material_from_directory")
    def test_readiness_uses_new_york_session_date(self, loader, tail):
        loader.return_value = object()
        tail.side_effect = [
            {
                "ready": True,
                "reason": "ready",
                "test": {"last_timestamp": "2026-07-23T00:30:00+00:00"},
            },
            {
                "ready": True,
                "reason": "ready",
                "test": {"last_timestamp": "2026-07-22T20:30:00-04:00"},
            },
        ]

        readiness, session_date = inspect_arena_readiness(
            Path("/data"),
            ("aapl", "nvda"),
        )

        self.assertEqual(session_date, "2026-07-22")
        self.assertTrue(all(item["ready"] for item in readiness))
        self.assertEqual(
            [call.args[1] for call in tail.call_args_list],
            [arena_walk_forward_config(), arena_walk_forward_config()],
        )
        self.assertEqual(loader.call_args_list[0].args[1], "AAPL")
        self.assertEqual(loader.call_args_list[1].args[1], "NVDA")

    def test_locked_arena_uses_strict_cli_and_validates_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            output.mkdir()
            artifact = output / "agent-arena.json"
            artifact.write_text("{}", encoding="utf-8")
            runner = Mock(
                return_value=subprocess.CompletedProcess(
                    args=(),
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "artifact": str(artifact),
                            "completed": 2,
                            "failures": 0,
                        }
                    ),
                    stderr="",
                )
            )

            result = run_locked_arena(root, output, ("aapl", "NVDA"), runner=runner)

        self.assertEqual(result, artifact.resolve())
        command = runner.call_args.args[0]
        self.assertEqual(
            command[:3], (sys.executable, "-m", "trading_bot.training.arena")
        )
        self.assertNotIn("--allow-unready-tail", command)
        self.assertEqual(command[-4:], ("--symbol", "AAPL", "--symbol", "NVDA"))

    def test_advisory_lock_rejects_second_watcher(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            with _exclusive_watch_lock(data_dir):
                with self.assertRaisesRegex(RuntimeError, "another arena watcher"):
                    with _exclusive_watch_lock(data_dir):
                        self.fail("second lock unexpectedly acquired")
            self.assertTrue((data_dir / ".arena-watch.lock").exists())

    def test_load_returns_none_before_first_status(self):
        with TemporaryDirectory() as directory:
            self.assertIsNone(load_arena_watch_status(Path(directory)))
            self.assertFalse((Path(directory) / ARENA_WATCH_STATUS_FILENAME).exists())
