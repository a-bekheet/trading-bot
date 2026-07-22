import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from trading_bot.execution.agent_watch import watch


class AgentWatchTests(TestCase):
    @patch("trading_bot.execution.agent_watch.run_paper_agents")
    def test_waits_for_atomic_collector_cycle_before_loading_models(
        self,
        run_paper_agents,
    ):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "_collector_status.json").write_text(
                json.dumps({"status": "running"}),
                encoding="utf-8",
            )

            result = watch(
                data_dir=data_dir,
                database=data_dir / "agent_paper.db",
                repo_root=data_dir,
                once=True,
            )
            status = json.loads(
                (data_dir / "_paper_agent_watch_status.json").read_text()
            )

        self.assertEqual(result, 0)
        self.assertEqual(status["status"], "waiting_for_collector")
        run_paper_agents.assert_not_called()

    @patch("trading_bot.execution.agent_watch.run_paper_agents")
    def test_once_writes_auditable_fleet_status(self, run_paper_agents):
        run_paper_agents.return_value = {
            "schema_version": "research-demo.paper-agent-runtime.v2",
            "generated_at": "2026-07-22T14:00:00Z",
            "database": "/tmp/agents.db",
            "selected_agent_count": 2,
            "completed_count": 2,
            "failure_count": 0,
            "agents": [],
            "failures": [],
        }
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            result = watch(
                data_dir=data_dir,
                database=data_dir / "agent_paper.db",
                repo_root=data_dir,
                once=True,
            )
            status = json.loads(
                (data_dir / "_paper_agent_watch_status.json").read_text()
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            status["schema_version"],
            "research-demo.paper-agent-watch.v2",
        )
        self.assertEqual(
            status["runtime_schema_version"],
            "research-demo.paper-agent-runtime.v2",
        )
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["completed_count"], 2)
        self.assertIn("last_heartbeat_at", status)
