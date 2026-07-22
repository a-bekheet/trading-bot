from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from trading_bot.execution.agent_store import AgentPaperStore


def deployment() -> dict:
    return {
        "deployment_id": "deployment-1",
        "agent_id": "AAPL-gru",
        "symbol": "AAPL",
        "model_id": "gru",
        "topology": "flat_vector",
        "checkpoint_path": "/tmp/model.pt",
        "checkpoint_sha256": "a" * 64,
        "activated": False,
        "activation_reason": "validation edge below threshold",
        "status": "guarded",
        "message": "processed 1 new decision(s)",
        "last_observation_timestamp": "2026-07-22T14:00:00+00:00",
        "last_decision_timestamp": "2026-07-22T14:00:00+00:00",
        "environment_state": {"cash": 100_000.0},
        "recurrent_state": {"steps": 1},
    }


def decision() -> dict:
    return {
        "snapshot_timestamp": "2026-07-22T14:00:00+00:00",
        "activated": False,
        "research_orders": [1, 0],
        "sandbox_orders": [0, 0],
        "executions": [],
        "reward": 0.0,
        "cash": 100_000.0,
        "nav": 100_000.0,
        "invalid_action_count": 0,
    }


class AgentPaperStoreTests(TestCase):
    def test_cycle_is_idempotent_and_keeps_research_action_under_guard(self):
        with TemporaryDirectory() as directory:
            store = AgentPaperStore(Path(directory) / "agents.db")
            first = store.commit_cycle(deployment(), [decision()])
            second = store.commit_cycle(deployment(), [decision()])
            decisions = store.decisions(deployment_id="deployment-1")

        self.assertEqual(first["decision_count"], 1)
        self.assertEqual(second["decision_count"], 1)
        self.assertEqual(second["execution_count"], 0)
        self.assertEqual(decisions[0]["research_orders"], [1, 0])
        self.assertEqual(decisions[0]["sandbox_orders"], [0, 0])
        self.assertFalse(decisions[0]["activated"])

    def test_separate_checkpoints_have_isolated_deployments(self):
        with TemporaryDirectory() as directory:
            store = AgentPaperStore(Path(directory) / "agents.db")
            first = deployment()
            second = {
                **deployment(),
                "deployment_id": "deployment-2",
                "checkpoint_sha256": "b" * 64,
            }
            store.commit_cycle(first, [decision()])
            store.commit_cycle(second, [])
            deployments = store.deployments()

        self.assertEqual(len(deployments), 2)
        self.assertEqual(
            {item["checkpoint_sha256"] for item in deployments},
            {"a" * 64, "b" * 64},
        )
