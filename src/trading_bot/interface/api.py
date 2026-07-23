"""FastAPI application for the local options control room."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trading_bot import __version__
from trading_bot.execution import PaperBroker, PaperTradeError
from trading_bot.execution.valuation import mark_positions
from trading_bot.interface.control import ControlPlane
from trading_bot.interface.data import (
    available_tickers,
    load_latest_snapshot,
    market_data_freshness_status,
    market_session_status,
)
from trading_bot.interface.results import (
    agent_roster,
    arena_overview,
    arena_readiness_overview,
    discover_agent_arena_manifests,
    discover_agent_runs,
    load_arena_watch_status,
    model_structure,
    model_structure_candidates,
    paper_agent_decisions,
    paper_agent_equity_curve,
    paper_agent_overview,
)


class ServiceAction(BaseModel):
    action: Literal["start", "stop", "restart", "run_once"]


class TrainingRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=20)
    episodes: int = Field(default=3, ge=1, le=100)
    hidden_size: int = Field(default=16)
    sequence_length: int = Field(default=4, ge=1, le=64)
    max_steps: int = Field(default=16, ge=2, le=512)


class PaperOrder(BaseModel):
    side: Literal["buy", "sell"]
    symbol: str
    contract_symbol: str
    quantity: int = Field(ge=1, le=100)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.copy()
    for column in clean:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].map(
                lambda value: value.isoformat() if pd.notna(value) else None
            )
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def _latest_runs_by_symbol(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest = {}
    for run in runs:
        symbol = str(run.get("symbol", "")).upper()
        if symbol and symbol not in latest:
            latest[symbol] = run
    return latest


def create_app(
    *,
    repo_root: Path,
    data_dir: Path,
    static_dir: Path | None = None,
) -> FastAPI:
    """Create the local-only control room application."""
    repo_root = repo_root.resolve()
    data_dir = data_dir.resolve()
    paper_broker = PaperBroker(data_dir / "paper_portfolio.db")
    control = ControlPlane(repo_root, data_dir)
    application = FastAPI(
        title="Options Control Room",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
    )

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "mode": "paper"}

    @application.get("/api/overview")
    def overview() -> dict[str, Any]:
        tickers = available_tickers(data_dir)
        services = control.services()
        agents = paper_agent_overview(data_dir)
        account = paper_broker.account()
        latest_symbol = tickers[0] if tickers else None
        market = {}
        if latest_symbol:
            snapshot = load_latest_snapshot(data_dir, latest_symbol)
            if not snapshot.empty:
                market = {
                    "symbol": latest_symbol,
                    "underlying_price": float(snapshot.iloc[0]["underlyingPrice"]),
                    "collected_at": str(snapshot.iloc[0]["collectedAt"]),
                    "session": market_session_status(snapshot),
                    "freshness": market_data_freshness_status(snapshot),
                }
        return {
            "version": __version__,
            "mode": "paper",
            "tickers": tickers,
            "service_summary": {
                "healthy": sum(item["healthy"] for item in services),
                "total": len(services),
            },
            "services": services,
            "market": market,
            "account": account,
            "agents": {
                "total": len(agents),
                "active": (
                    int((agents["Runtime"] == "Active").sum())
                    if not agents.empty
                    else 0
                ),
                "decisions": (
                    int(agents["Decisions"].sum()) if not agents.empty else 0
                ),
                "executions": (
                    int(agents["Executions"].sum()) if not agents.empty else 0
                ),
            },
            "jobs": control.list_jobs()[:8],
        }

    @application.get("/api/services")
    def services() -> list[dict[str, Any]]:
        return control.services()

    @application.post("/api/services/{service}/actions")
    async def service_action(
        service: str,
        request: ServiceAction,
    ) -> dict[str, Any]:
        try:
            return control.service_action(service, request.action)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return control.list_jobs()

    @application.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        payload = control.get_job(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="job not found")
        return payload

    @application.get("/api/agents")
    def agents() -> dict[str, Any]:
        runs = discover_agent_runs(data_dir)
        return {
            "roster": _records(agent_roster(runs)),
            "deployments": _records(paper_agent_overview(data_dir)),
            "decisions": _records(paper_agent_decisions(data_dir, limit=250)),
            "equity": _records(paper_agent_equity_curve(data_dir)),
        }

    @application.get("/api/models")
    def models() -> dict[str, Any]:
        runs = discover_agent_runs(data_dir)
        records = []
        for symbol, run in _latest_runs_by_symbol(runs).items():
            structure = model_structure(run)
            if not structure:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "structure": structure,
                    "candidates": _records(model_structure_candidates(run)),
                }
            )
        return {"models": records}

    @application.get("/api/training")
    def training() -> dict[str, Any]:
        watch = load_arena_watch_status(data_dir) or {}
        manifests = discover_agent_arena_manifests(data_dir)
        readiness_source = (
            {"preflight": watch.get("readiness", [])}
            if watch
            else manifests[0]
            if manifests
            else {}
        )
        return {
            "defaults": {
                "symbols": ["AAPL", "NVDA", "MSFT", "AMZN", "GOOG"],
                "episodes": 3,
                "hidden_size": 16,
                "sequence_length": 4,
                "max_steps": 16,
                "candidate_count_per_ticker": 12,
                "training_seed_count": 3,
            },
            "watcher": watch,
            "readiness": _records(arena_readiness_overview(readiness_source)),
            "jobs": [
                item
                for item in control.list_jobs()
                if item["kind"] == "training"
            ],
        }

    @application.post("/api/training/runs")
    async def start_training(request: TrainingRequest) -> dict[str, Any]:
        try:
            return control.start_training(**request.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @application.get("/api/market")
    def market(symbol: str | None = None) -> dict[str, Any]:
        tickers = available_tickers(data_dir)
        selected = (symbol or (tickers[0] if tickers else "")).upper()
        if selected not in tickers:
            raise HTTPException(status_code=404, detail="ticker data not found")
        snapshot = load_latest_snapshot(data_dir, selected)
        return {
            "tickers": tickers,
            "symbol": selected,
            "session": market_session_status(snapshot),
            "freshness": market_data_freshness_status(snapshot),
            "contracts": _records(snapshot),
        }

    @application.get("/api/portfolio")
    def portfolio() -> dict[str, Any]:
        positions = paper_broker.positions()
        symbols = sorted({item["symbol"] for item in positions})
        frames = [load_latest_snapshot(data_dir, symbol) for symbol in symbols]
        quotes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        marked = mark_positions(positions, quotes)
        account = paper_broker.account()
        market_value = (
            float(marked["market_value"].sum()) if not marked.empty else 0.0
        )
        return {
            "account": account,
            "market_value": market_value,
            "total_equity": float(account["cash"]) + market_value,
            "positions": _records(marked),
            "trades": paper_broker.trades(),
        }

    @application.post("/api/orders")
    def order(request: PaperOrder) -> dict[str, Any]:
        symbol = request.symbol.strip().upper()
        snapshot = load_latest_snapshot(data_dir, symbol)
        matching = snapshot[
            snapshot["contractSymbol"].astype(str) == request.contract_symbol
        ]
        if matching.empty:
            raise HTTPException(status_code=404, detail="contract not found")
        session = market_session_status(snapshot)
        freshness = market_data_freshness_status(snapshot)
        if not session["trading_enabled"] or not freshness["trading_enabled"]:
            raise HTTPException(
                status_code=409,
                detail="paper execution is disabled for this snapshot",
            )
        contract = matching.iloc[0]
        price = float(contract["ask"] if request.side == "buy" else contract["bid"])
        if not math.isfinite(price) or price <= 0:
            raise HTTPException(status_code=409, detail="no executable quote")
        payload = {
            "contract_symbol": request.contract_symbol,
            "symbol": symbol,
            "expiration": str(contract["expiration"]),
            "option_type": str(contract["optionType"]),
            "strike": float(contract["strike"]),
            "quantity": request.quantity,
            "price": price,
        }
        try:
            return (
                paper_broker.buy(**payload)
                if request.side == "buy"
                else paper_broker.sell(**payload)
            )
        except PaperTradeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.get("/api/research")
    def research() -> dict[str, Any]:
        runs = discover_agent_runs(data_dir)
        return {
            "arena": _records(arena_overview(runs)),
            "run_count": len(runs),
            "runs": [
                {
                    "symbol": run.get("symbol"),
                    "name": run.get("_run_name"),
                    "schema_version": run.get("schema_version"),
                    "artifact": run.get("_artifact_path"),
                    "fold_count": len(run.get("folds", [])),
                }
                for run in runs[:50]
            ],
        }

    resolved_static = (
        static_dir.resolve()
        if static_dir is not None
        else Path(__file__).with_name("web_dist")
    )
    if resolved_static.is_dir() and (resolved_static / "index.html").is_file():
        assets = resolved_static / "assets"
        if assets.is_dir():
            application.mount(
                "/assets",
                StaticFiles(directory=assets),
                name="assets",
            )

        @application.get("/")
        def index() -> FileResponse:
            return FileResponse(resolved_static / "index.html")

    return application
