import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import AsyncMock, patch

import pandas as pd
from fastapi.testclient import TestClient

from trading_bot.interface.api import create_app
from trading_bot.interface.control import ControlPlane


def write_snapshot(data_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "collectedAt": "2026-07-23T14:00:00+00:00",
                "symbol": "AAPL",
                "contractSymbol": "AAPL260821C00200000",
                "expiration": "2026-08-21",
                "optionType": "call",
                "strike": 200.0,
                "bid": 4.8,
                "ask": 5.0,
                "underlyingPrice": 202.0,
                "underlyingQuoteTime": "2026-07-23T13:59:30+00:00",
                "underlyingQuoteTimeSource": "provider",
                "underlyingPriceSource": "regularMarketPrice",
                "marketState": "REGULAR",
                "impliedVolatility": 0.25,
                "delta": 0.55,
                "gamma": 0.03,
                "theta": -0.04,
                "vega": 0.08,
                "openInterest": 100,
            }
        ]
    ).to_csv(data_dir / "AAPL.csv", index=False)


class InterfaceApiTests(TestCase):
    def test_serves_health_overview_market_and_static_application(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            static_dir = root / "web"
            data_dir.mkdir()
            static_dir.mkdir()
            (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (static_dir / "index.html").write_text(
                "<main>control room</main>",
                encoding="utf-8",
            )
            write_snapshot(data_dir)
            client = TestClient(
                create_app(
                    repo_root=root,
                    data_dir=data_dir,
                    static_dir=static_dir,
                )
            )

            health = client.get("/api/health")
            overview = client.get("/api/overview")
            market = client.get("/api/market?symbol=AAPL")
            index = client.get("/")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["mode"], "paper")
        self.assertEqual(overview.json()["market"]["symbol"], "AAPL")
        self.assertEqual(len(overview.json()["services"]), 3)
        self.assertEqual(
            market.json()["contracts"][0]["contractSymbol"],
            "AAPL260821C00200000",
        )
        self.assertIn("control room", index.text)

    def test_training_endpoint_uses_typed_control_plane_request(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            with patch.object(
                ControlPlane,
                "start_training",
                return_value={"id": "job-1", "status": "queued"},
            ) as start:
                client = TestClient(create_app(repo_root=root, data_dir=data_dir))
                response = client.post(
                    "/api/training/runs",
                    json={
                        "symbols": ["AAPL", "NVDA"],
                        "episodes": 4,
                        "hidden_size": 32,
                        "sequence_length": 8,
                        "max_steps": 24,
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "job-1")
        start.assert_called_once_with(
            symbols=["AAPL", "NVDA"],
            episodes=4,
            hidden_size=32,
            sequence_length=8,
            max_steps=24,
        )

    def test_control_plane_constructs_allow_listed_training_command(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            control = ControlPlane(root, data_dir)

            async def create_job():
                with patch.object(
                    control,
                    "_execute",
                    new=AsyncMock(),
                ):
                    job = control.start_training(
                        symbols=["aapl", "NVDA"],
                        episodes=3,
                        hidden_size=16,
                        sequence_length=4,
                        max_steps=16,
                    )
                    await asyncio.sleep(0)
                    return job

            job = asyncio.run(create_job())

        self.assertEqual(job["kind"], "training")
        self.assertIn("trading_bot.training.arena", job["command"])
        self.assertEqual(job["command"].count("--symbol"), 2)
        self.assertNotIn("--allow-unready-tail", job["command"])
