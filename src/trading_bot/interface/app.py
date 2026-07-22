"""Streamlit research workspace for agents, market data, and paper trading."""

import json
import math
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from trading_bot.execution import PaperBroker, PaperTradeError
from trading_bot.execution.valuation import mark_positions
from trading_bot.interface.data import (
    available_tickers,
    load_latest_snapshot,
    market_data_freshness_status,
    market_session_status,
)
from trading_bot.interface.results import (
    agent_decision_tape,
    agent_leaderboard,
    agent_roster,
    arena_readiness_overview,
    arena_overview,
    discover_agent_arena_manifests,
    discover_agent_runs,
    equity_curve,
    evidence_summary,
    feature_ablation_results,
    heldout_results,
    load_arena_watch_status,
    load_paper_agent_watch_status,
    paper_agent_decisions,
    paper_agent_equity_curve,
    paper_agent_overview,
    promotion_assessment,
    trade_ledger,
)


DATA_DIR = Path("data")
PAPER_DATABASE = DATA_DIR / "paper_portfolio.db"

st.set_page_config(
    page_title="Options Agent Desk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root {
        --ink: #152033;
        --muted: #667085;
        --line: #e4e9f1;
        --panel: #ffffff;
        --accent: #3157d5;
        --positive: #087b60;
    }
    .stApp {background: #f6f8fc; color: var(--ink);}
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
    [data-testid="stSidebar"] {
        background: #111a2e;
        border-right: 0;
    }
    [data-testid="stSidebar"] * {color: #eef3ff;}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [role="radiogroup"] {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background: #ffffff;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
        color: #152033 !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {
        color: #152033 !important;
        fill: #152033 !important;
    }
    [data-testid="stSidebar"] .stSelectbox [role="group"] {
        background: #ffffff;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] .stSelectbox input {
        color: #152033 !important;
        -webkit-text-fill-color: #152033 !important;
    }
    [data-testid="stSidebar"] .stSelectbox button {
        color: #152033 !important;
    }
    [data-testid="stSidebar"] hr {border-color: rgba(255,255,255,.12);}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: .4rem;
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stTabs"] button {
        border-radius: 9px 9px 0 0;
        padding-left: 1rem;
        padding-right: 1rem;
        font-weight: 650;
    }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 6px 24px rgba(20, 31, 54, 0.045);
    }
    .agent-hero {
        padding: 25px 28px;
        border-radius: 18px;
        background:
            radial-gradient(circle at 88% 15%, rgba(79, 209, 197, .2), transparent 24%),
            linear-gradient(120deg, #111a2e 0%, #263c74 68%, #235e70 100%);
        color: white;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,.08);
        box-shadow: 0 16px 45px rgba(21, 32, 51, .14);
    }
    .agent-hero .eyebrow {
        color: #93dfd4;
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .12em;
        margin-bottom: 8px;
    }
    .agent-hero h1 {font-size: 2rem; line-height: 1.1; margin: 0 0 7px 0;}
    .agent-hero p {margin: 0; color: #dce6ff; max-width: 680px;}
    .section-kicker {
        color: #3157d5;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .11em;
        margin: 1rem 0 .25rem;
    }
    .workspace-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 100%;
        box-shadow: 0 6px 24px rgba(20,31,54,.04);
    }
    .workspace-card h3 {margin: 0 0 5px; font-size: 1.05rem;}
    .workspace-card p {color: var(--muted); margin: 0; font-size: .9rem;}
    .status-row {display:flex; align-items:center; gap:8px; margin:7px 0;}
    .status-dot {width:8px; height:8px; border-radius:50%; background:#98a2b3;}
    .status-dot.good {background:#12a27d; box-shadow:0 0 0 4px #e5f7f2;}
    .status-dot.warn {background:#e6a700; box-shadow:0 0 0 4px #fff5d6;}
    .status-dot.bad {background:#d94b59; box-shadow:0 0 0 4px #fdecef;}
    .desk-banner {
        display:flex; justify-content:space-between; gap:20px; align-items:center;
        background:linear-gradient(105deg,#fff,#f0f4ff);
        border:1px solid #dce4f6; border-radius:14px; padding:18px 20px;
        margin:.25rem 0 1rem;
    }
    .desk-banner h2 {font-size:1.35rem; margin:0 0 4px;}
    .desk-banner p {margin:0; color:var(--muted);}
    .badge {
        display:inline-block; padding:5px 10px; border-radius:999px;
        font-size:.75rem; font-weight:750; white-space:nowrap;
        background:#e9eefb; color:#23345f;
    }
    .badge.good {background:#e3f6f0; color:#06745a;}
    .badge.warn {background:#fff3d2; color:#8b6200;}
    .badge.bad {background:#fde8eb; color:#a82e3a;}
    .scope-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #e9eefb;
        color: #23345f;
        font-size: 0.78rem;
        font-weight: 650;
        margin-right: 6px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    .stAlert {border-radius: 12px;}
    h1, h2, h3 {letter-spacing: -.025em;}
    </style>
    <div class="agent-hero">
      <div class="eyebrow">OPTIONS INTELLIGENCE · PAPER EXECUTION</div>
      <h1>Agent Trading Desk</h1>
      <p>Monitor trained policies, inspect decisions, and test options execution from one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("PAPER TRADING ONLY · NO LIVE BROKER CONNECTION")

tickers = available_tickers(DATA_DIR)
if not tickers:
    st.info("No CSV data found. Run `collect-options --once` first.")
    st.stop()

broker = PaperBroker(PAPER_DATABASE)
st.sidebar.markdown("## Agent Desk")
st.sidebar.caption("PAPER ENVIRONMENT")
st.sidebar.divider()
st.sidebar.markdown("#### Market context")
symbol = st.sidebar.selectbox("Ticker", tickers)
option_type = st.sidebar.radio("Option type", ("call", "put"), horizontal=True)
snapshot = load_latest_snapshot(DATA_DIR, symbol)
filtered = snapshot[snapshot["optionType"] == option_type].sort_values("strike")
session = market_session_status(snapshot)
freshness = market_data_freshness_status(snapshot)
execution_enabled = bool(session["trading_enabled"] and freshness["trading_enabled"])

runs = discover_agent_runs(DATA_DIR)
watch_status = load_arena_watch_status(DATA_DIR)
arena_manifests = discover_agent_arena_manifests(DATA_DIR)
readiness = (
    arena_readiness_overview({"preflight": watch_status.get("readiness", [])})
    if watch_status
    else arena_readiness_overview(arena_manifests[0])
    if arena_manifests
    else pd.DataFrame()
)
paper_agents = paper_agent_overview(DATA_DIR)
paper_decisions = paper_agent_decisions(DATA_DIR)
paper_equity = paper_agent_equity_curve(DATA_DIR)
paper_watch = load_paper_agent_watch_status(DATA_DIR)
roster = agent_roster(runs) if runs else pd.DataFrame()

overview_tab, desk_tab, market_tab, portfolio_tab, research_tab = st.tabs(
    (
        "Overview",
        "Agent Desk",
        "Trade",
        "Portfolio",
        "Research",
    )
)

with overview_tab:
    first = snapshot.iloc[0]
    account = broker.account()
    active_count = (
        int((paper_agents["Runtime"] == "Active").sum())
        if not paper_agents.empty
        else 0
    )
    deployment_count = len(paper_agents)
    finalized_count = (
        int(paper_agents["Finalized outcomes"].sum())
        if not paper_agents.empty
        else 0
    )
    fleet_return = (
        float(paper_agents["Paper return"].median())
        if not paper_agents.empty
        and paper_agents["Paper return"].notna().any()
        else 0.0
    )

    st.markdown('<div class="section-kicker">OPERATING OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown(f"## {symbol} workspace")
    st.caption(
        "A fast read of market eligibility, paper-agent health, and measured outcomes. "
        "Use Agent Desk for a policy-level view."
    )
    headline = st.columns(4)
    headline[0].metric("Underlying", f"${float(first['underlyingPrice']):,.2f}")
    headline[1].metric("Paper cash", f"${float(account['cash']) / 1_000:,.1f}K")
    headline[2].metric("Agents online", f"{active_count}/{deployment_count}")
    headline[3].metric(
        "Fleet return",
        f"{fleet_return:.2%}",
        help=f"Based on {finalized_count} finalized online outcomes.",
    )

    market_state = str(session["provider_state"])
    market_good = bool(session["trading_enabled"])
    freshness_good = bool(freshness["trading_enabled"])
    runtime_state = str((paper_watch or {}).get("status", "not started"))
    runtime_good = runtime_state in {"active", "complete", "up_to_date", "ok"}
    runtime_warn = runtime_state in {"waiting", "not started"}
    training_state = str((watch_status or {}).get("status", "not started"))
    training_good = training_state in {"complete", "up_to_date"}
    training_warn = training_state in {"waiting", "not started", "running"}
    quote_age = (
        f"{float(freshness['age_seconds']):.0f}s old"
        if freshness["coverage"]
        else "age unavailable"
    )
    status_columns = st.columns(3)
    with status_columns[0]:
        st.markdown(
            f"""<div class="workspace-card"><h3>Market data</h3>
            <div class="status-row"><span class="status-dot {'good' if market_good and freshness_good else 'warn'}"></span>
            <strong>{escape(market_state)}</strong></div>
            <p>{escape(quote_age)} · {len(filtered):,} {option_type} contracts</p></div>""",
            unsafe_allow_html=True,
        )
    with status_columns[1]:
        status_class = "good" if runtime_good else "warn" if runtime_warn else "bad"
        st.markdown(
            f"""<div class="workspace-card"><h3>Paper agents</h3>
            <div class="status-row"><span class="status-dot {status_class}"></span>
            <strong>{escape(runtime_state.replace('_', ' ').title())}</strong></div>
            <p>{active_count} active deployments · fail-closed execution</p></div>""",
            unsafe_allow_html=True,
        )
    with status_columns[2]:
        status_class = "good" if training_good else "warn" if training_warn else "bad"
        ready_count = int((readiness["Ready"] == "Yes").sum()) if not readiness.empty else 0
        st.markdown(
            f"""<div class="workspace-card"><h3>Training arena</h3>
            <div class="status-row"><span class="status-dot {status_class}"></span>
            <strong>{escape(training_state.replace('_', ' ').title())}</strong></div>
            <p>{ready_count}/{len(readiness)} tickers evidence-ready</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-kicker">LIVE RESULTS</div>', unsafe_allow_html=True)
    result_column, action_column = st.columns((2, 1), gap="large")
    with result_column:
        st.markdown("### Paper equity")
        ticker_equity = (
            paper_equity[paper_equity["Ticker"] == symbol]
            if not paper_equity.empty
            else pd.DataFrame()
        )
        if ticker_equity.empty:
            st.info(
                f"{symbol} has no finalized online outcome yet. The first decision "
                "remains pending until the next eligible market snapshot."
            )
        else:
            st.line_chart(
                ticker_equity.set_index("Timestamp")["NAV"],
                y_label="Paper NAV",
                height=275,
            )
    with action_column:
        st.markdown("### What needs attention")
        if not freshness_good:
            st.warning("The selected quote is stale. New paper orders are disabled.")
        elif not market_good:
            st.info("The provider marks the market outside its regular session.")
        elif active_count < deployment_count:
            st.warning(
                f"{deployment_count - active_count} paper deployment(s) are guarded or "
                "incompatible. Inspect Agent Desk before interpreting fleet results."
            )
        else:
            st.success("Market and paper-agent runtime are ready for eligible decisions.")
        st.caption(
            f"Latest collection: {first['collectedAt']}\n\n"
            f"Training heartbeat: {(watch_status or {}).get('last_heartbeat_at', 'not started')}"
        )

    if not paper_agents.empty:
        st.markdown("### Fleet at a glance")
        overview_columns = [
            "Ticker",
            "Runtime",
            "Activation",
            "Decisions",
            "Finalized outcomes",
            "Pending outcomes",
            "Paper return",
            "Last decision",
        ]
        st.dataframe(
            paper_agents[overview_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "Paper return": st.column_config.NumberColumn(format="percent"),
            },
        )

with desk_tab:
    st.markdown('<div class="section-kicker">POLICY WORKSPACE</div>', unsafe_allow_html=True)
    if roster.empty:
        st.markdown("## Agent Desk")
        st.info(
            "No trained policy artifacts are available yet. Open Research for the "
            "walk-forward training command and evidence requirements."
        )
    else:
        roster_tickers = tuple(roster["Ticker"].astype(str))
        default_ticker = roster_tickers.index(symbol) if symbol in roster_tickers else 0
        desk_ticker = st.selectbox(
            "Agent",
            roster_tickers,
            index=default_ticker,
            key="agent_desk_ticker",
            label_visibility="collapsed",
        )
        agent = roster[roster["Ticker"] == desk_ticker].iloc[0]
        live_rows = paper_agents[paper_agents["Ticker"] == desk_ticker]
        live = live_rows.iloc[0] if not live_rows.empty else None
        state = str(agent["State"])
        badge_class = "good" if state == "Paper active" else "warn"
        st.markdown(
            f"""<div class="desk-banner"><div><h2>{escape(desk_ticker)} · {escape(str(agent['Research policy']))}</h2>
            <p>{escape(str(agent['Architecture']))} · {escape(str(agent['Algorithm']))} · {escape(str(agent['Action policy']))}</p></div>
            <span class="badge {badge_class}">{escape(state)}</span></div>""",
            unsafe_allow_html=True,
        )

        desk_metrics = st.columns(4)
        desk_metrics[0].metric("Held-out return", f"{float(agent['Held-out return']):.2%}")
        desk_metrics[1].metric(
            "Online return",
            f"{float(live['Paper return']):.2%}"
            if live is not None and pd.notna(live["Paper return"])
            else "Pending",
        )
        desk_metrics[2].metric(
            "Outcomes",
            int(live["Finalized outcomes"]) if live is not None else 0,
            delta=(
                f"{int(live['Pending outcomes'])} pending"
                if live is not None
                else "runtime not started"
            ),
            delta_color="off",
        )
        desk_metrics[3].metric(
            "Inference", f"{float(agent['Median latency (us)']):.1f} µs"
        )

        chart_column, rationale_column = st.columns((2, 1), gap="large")
        with chart_column:
            st.markdown("### Online paper equity")
            selected_equity = (
                paper_equity[paper_equity["Ticker"] == desk_ticker]
                if not paper_equity.empty
                else pd.DataFrame()
            )
            if selected_equity.empty:
                st.info("No finalized online path yet. This is a measured zero, not a hidden backtest.")
            else:
                st.line_chart(
                    selected_equity.set_index("Timestamp")["NAV"],
                    y_label="Paper NAV",
                    height=285,
                )
        with rationale_column:
            st.markdown("### Execution gate")
            edge = float(agent["Validation edge vs no-op (bp)"])
            if state == "Paper active":
                st.success(f"Activated after a {edge:.2f} bp validation edge over no-op.")
            else:
                st.warning(
                    f"Guarded: validation edge is {edge:.2f} bp. The policy is visible, "
                    "but the runtime substitutes HOLD."
                )
            if live is not None and str(live["Runtime"]) == "Error":
                st.error(str(live["Message"]))

        st.markdown("### Recent decisions")
        live_decisions = (
            paper_decisions[paper_decisions["Ticker"] == desk_ticker]
            if not paper_decisions.empty
            else pd.DataFrame()
        )
        if live_decisions.empty:
            st.info("No eligible live decision has been recorded for this deployment.")
        else:
            decision_columns = [
                "Timestamp",
                "Proposed action",
                "Executed action",
                "Outcome status",
                "Outcome return",
                "NAV",
            ]
            st.dataframe(
                live_decisions[decision_columns].head(12),
                width="stretch",
                hide_index=True,
                column_config={
                    "Outcome return": st.column_config.NumberColumn(format="percent"),
                    "NAV": st.column_config.NumberColumn(format="$%.2f"),
                },
            )
        with st.expander("Model identity and checkpoint"):
            identity = {
                "agent_id": agent["Agent ID"],
                "temporal_core": agent["Temporal core"],
                "topology": agent["Topology"],
                "feature_set": agent["Feature set"],
                "training_seeds": int(agent["Training seeds"]),
                "checkpoint": agent["Checkpoint"],
                "experiment": agent["Experiment"],
            }
            st.json(identity)

with research_tab:
    st.markdown('<div class="section-kicker">RESEARCH & DIAGNOSTICS</div>', unsafe_allow_html=True)
    st.markdown("## Training evidence")
    st.caption(
        "Detailed readiness, model selection, feature ablations, and held-out "
        "provenance live here so they do not crowd the operating workflow."
    )
    if watch_status:
        automation_columns = st.columns(4)
        automation_state = str(watch_status.get("status", "unknown"))
        automation_columns[0].metric(
            "Training automation",
            automation_state.replace("_", " ").title(),
        )
        automation_columns[1].metric(
            "Ready tickers",
            f"{int(watch_status.get('ready_count', 0))}/"
            f"{int(watch_status.get('ticker_total', 0))}",
        )
        automation_columns[2].metric(
            "Target session",
            str(watch_status.get("target_session_date") or "Waiting"),
        )
        automation_columns[3].metric(
            "Last trained session",
            str(watch_status.get("last_completed_session_date") or "None"),
        )
        st.caption(
            f"Watcher heartbeat: {watch_status.get('last_heartbeat_at', 'Unknown')} · "
            f"{watch_status.get('message', 'No status message')}"
        )
        if automation_state == "error":
            st.error("Automated arena training failed; inspect arena-watch.stderr.log.")
        elif automation_state == "running":
            st.info("The locked GRU/LSTM/mixture and surface-GNN arena is training.")
        elif automation_state in {"complete", "up_to_date"}:
            st.success(
                "The recurrent and surface-GNN arena is current for this session."
            )
    if not readiness.empty:
        ready_count = int((readiness["Ready"] == "Yes").sum())
        readiness_columns = st.columns(4)
        required_eligible = int(readiness["Required eligible"].max())
        minimum_eligible = int(readiness["Eligible snapshots"].min())
        readiness_columns[0].metric(
            "Arena tails ready",
            f"{ready_count}/{len(readiness)}",
        )
        readiness_columns[1].metric(
            "Eligible states",
            f"{minimum_eligible}/{required_eligible}",
        )
        readiness_columns[2].metric(
            "Regular validation states",
            int(readiness["Validation regular"].sum()),
        )
        readiness_columns[3].metric(
            "Regular test states",
            int(readiness["Test regular"].sum()),
        )
        if ready_count < len(readiness):
            st.warning(
                "The arena is waiting for enough provider-confirmed regular, "
                "fresh, executable states to fill training, validation, and "
                "test. Expensive agent training is skipped until all thirteen "
                "eligible states exist for every ticker."
            )
        else:
            st.success(
                "Training, validation, and test have enough regular, fresh, "
                "executable states for the locked arena."
            )
        with st.expander("Arena evidence readiness"):
            st.dataframe(readiness, width="stretch", hide_index=True)
    paper_agents = paper_agent_overview(DATA_DIR)
    paper_decisions = paper_agent_decisions(DATA_DIR)
    paper_equity = paper_agent_equity_curve(DATA_DIR)
    paper_watch = load_paper_agent_watch_status(DATA_DIR)
    st.subheader("Running paper-agent loop")
    st.caption(
        "Each checkpoint has an isolated account and recurrent cursor. Proposed "
        "orders are recorded on every new eligible snapshot, but only policies "
        "that passed the validation gate may create simulated fills."
    )
    if paper_agents.empty:
        st.info(
            "No persistent paper-agent cycle has run yet. Start one with "
            "`paper-agents --data-dir data`."
        )
    else:
        if paper_watch:
            st.caption(
                f"Paper-agent heartbeat: "
                f"{paper_watch.get('last_heartbeat_at', 'Unknown')} · "
                f"{paper_watch.get('completed_count', 0)} completed · "
                f"{paper_watch.get('failure_count', 0)} failed"
            )
            if paper_watch.get("status") in {"degraded", "error"}:
                st.warning(
                    "The paper-agent service is fail-closed because one or more "
                    "selected checkpoints are incompatible or not yet eligible. "
                    "The arena watcher will replace them after a current run."
                )
        runtime_metrics = st.columns(6)
        runtime_metrics[0].metric("Deployments", len(paper_agents))
        runtime_metrics[1].metric(
            "Runtime active",
            int((paper_agents["Runtime"] == "Active").sum()),
        )
        runtime_metrics[2].metric(
            "Live decisions", int(paper_agents["Decisions"].sum())
        )
        runtime_metrics[3].metric(
            "Finalized outcomes",
            int(paper_agents["Finalized outcomes"].sum()),
        )
        runtime_metrics[4].metric(
            "Paper fills", int(paper_agents["Executions"].sum())
        )
        runtime_metrics[5].metric(
            "Median paper return",
            f"{float(paper_agents['Paper return'].median()):.3%}",
        )
        st.dataframe(
            paper_agents,
            width="stretch",
            hide_index=True,
            column_config={
                "Cash": st.column_config.NumberColumn(format="$%.2f"),
                "Equity": st.column_config.NumberColumn(format="$%.2f"),
                "Paper return": st.column_config.NumberColumn(format="percent"),
                "Outcome hit rate": st.column_config.NumberColumn(
                    format="percent"
                ),
                "Online max drawdown": st.column_config.NumberColumn(
                    format="percent"
                ),
            },
        )
        if not paper_equity.empty:
            equity_ticker = st.selectbox(
                "Paper-agent equity path",
                tuple(sorted(paper_equity["Ticker"].unique())),
                key="paper_agent_equity_ticker",
            )
            selected_equity = paper_equity[
                paper_equity["Ticker"] == equity_ticker
            ]
            st.line_chart(
                selected_equity.set_index("Timestamp")["NAV"],
                y_label="Paper NAV",
            )
            st.caption(
                "The curve advances only when the next real eligible snapshot "
                "finalizes a decision outcome; the newest action remains pending."
            )
        with st.expander("Live paper-agent decision ledger"):
            if paper_decisions.empty:
                st.info(
                    "Deployments are registered, but no eligible post-evaluation "
                    "snapshot has arrived yet."
                )
            else:
                st.dataframe(
                    paper_decisions,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Reward": st.column_config.NumberColumn(format="%.6f"),
                        "Cash": st.column_config.NumberColumn(format="$%.2f"),
                        "NAV": st.column_config.NumberColumn(format="$%.2f"),
                        "Decision NAV": st.column_config.NumberColumn(
                            format="$%.2f"
                        ),
                        "Outcome NAV": st.column_config.NumberColumn(
                            format="$%.2f"
                        ),
                        "Outcome return": st.column_config.NumberColumn(
                            format="percent"
                        ),
                    },
                )
    if not runs:
        st.subheader("No walk-forward agent run yet")
        st.write(
            "Run the GRU/LSTM/mixture tournament below, then refresh this page. "
            "The result will include validation rankings, a held-out equity "
            "path, risk, latency, and the actual simulated fill ledger."
        )
        st.code(
            "train-walk-forward --symbol AAPL --output-dir "
            "data/agent_runs/aapl-recurrent-tournament --min-train-size 6 "
            "--validation-size 2 --test-size 3 --step-size 100 --episodes 2 "
            "--hidden-size 8 --sequence-length 2 --burn-in-steps 0 "
            "--max-steps 4 --initial-hold-bias 0 "
            "--candidate flat:gru --candidate flat:lstm "
            "--candidate flat:mixture",
            language="bash",
        )
    else:
        roster = agent_roster(runs)
        st.subheader("Persisted trading agents")
        st.caption(
            "One validation-selected policy per ticker from the newest successful "
            "run. Guarded agents still exist and retain their checkpoints and "
            "research decisions; the sandbox executes no-op until their "
            "validation edge clears the activation gate."
        )
        if roster.empty:
            st.info("Saved runs were found, but none contains a selected policy.")
        else:
            latest_runs = {}
            for saved_run in runs:
                ticker = str(saved_run.get("symbol", "")).upper()
                if ticker and ticker not in latest_runs:
                    latest_runs[ticker] = saved_run
            candidate_frames = []
            for ticker, latest_run in latest_runs.items():
                candidates = agent_leaderboard(latest_run).copy()
                if candidates.empty:
                    continue
                candidates.insert(0, "Ticker", ticker)
                candidate_frames.append(candidates)
            candidate_fleet = (
                pd.concat(candidate_frames, ignore_index=True)
                if candidate_frames
                else pd.DataFrame()
            )
            gnn_candidates = (
                int(
                    candidate_fleet["Architecture"]
                    .str.contains("Graph", case=False, na=False)
                    .sum()
                )
                if not candidate_fleet.empty
                else 0
            )
            roster_metrics = st.columns(6)
            roster_metrics[0].metric("Persisted agents", len(roster))
            roster_metrics[1].metric(
                "Sandbox active",
                int((roster["State"] == "Paper active").sum()),
            )
            roster_metrics[2].metric("GNN challengers", gnn_candidates)
            roster_metrics[3].metric(
                "Recorded decisions", int(roster["Decisions"].sum())
            )
            roster_metrics[4].metric(
                "Research fills",
                int(roster["Research executions"].sum()),
            )
            roster_metrics[5].metric(
                "Median actor latency",
                f"{float(roster['Median latency (us)'].median()):.1f} us",
            )

            card_columns = st.columns(min(3, len(roster)))
            for index, (_, agent) in enumerate(roster.iterrows()):
                with card_columns[index % len(card_columns)]:
                    with st.container(border=True):
                        st.markdown(
                            f"#### {agent['Ticker']} · {agent['Research policy']}"
                        )
                        st.caption(str(agent["Agent ID"]))
                        st.write(f"**{agent['State']}** · {agent['Topology']}")
                        st.metric(
                            "Held-out return",
                            f"{float(agent['Held-out return']):.3%}",
                        )
                        st.write(
                            f"Last research decision: **{agent['Last research action']}**"
                        )
                        st.caption(
                            f"Sandbox: {agent['Last sandbox action']} · "
                            f"{float(agent['Median latency (us)']):.1f} us · "
                            f"{int(agent['Training seeds'])} training seeds"
                        )

            selected_agent_ticker = st.selectbox(
                "Inspect agent decision tape",
                tuple(roster["Ticker"]),
                key="agent_decision_tape_ticker",
            )
            selected_agent_run = latest_runs[selected_agent_ticker]
            selected_agent_fold = max(
                selected_agent_run.get("folds", []),
                key=lambda item: int(item.get("fold", -1)),
            )
            decisions = agent_decision_tape(
                selected_agent_run,
                int(selected_agent_fold.get("fold", 0)),
            )
            st.subheader(f"{selected_agent_ticker} agent decision tape")
            st.caption(
                "Every research-policy decision is retained, including HOLD. "
                "Sandbox action shows the order surface that was actually "
                "allowed to operate after the validation-only guard."
            )
            if decisions.empty:
                st.info("This older agent artifact does not contain a decision trace.")
            else:
                st.dataframe(
                    decisions.sort_values("Timestamp", ascending=False),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Reward": st.column_config.NumberColumn(format="%.6f"),
                        "NAV": st.column_config.NumberColumn(format="$%.2f"),
                    },
                )
            with st.expander("Agent registry and complete candidate fleet"):
                st.dataframe(
                    roster,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Held-out return": st.column_config.NumberColumn(
                            format="percent"
                        ),
                        "Sandbox return": st.column_config.NumberColumn(
                            format="percent"
                        ),
                        "Validation edge vs no-op (bp)": (
                            st.column_config.NumberColumn(format="%.3f")
                        ),
                        "Median latency (us)": st.column_config.NumberColumn(
                            format="%.1f"
                        ),
                    },
                )
                if not candidate_fleet.empty:
                    st.dataframe(
                        candidate_fleet,
                        width="stretch",
                        hide_index=True,
                    )

        arena = arena_overview(runs)
        if len(arena) > 1:
            st.subheader("Cross-ticker agent arena")
            st.caption(
                "Newest run per ticker. The default arena uses the latest "
                "available chronological fold, with all earlier eligible "
                "history assigned to training. Each row has an independent "
                "validation tournament and held-out policy. Timestamped run "
                "directories preserve earlier experiments for drill-down."
            )
            arena_columns = st.columns(8)
            arena_columns[0].metric("Tickers evaluated", len(arena))
            arena_columns[1].metric(
                "Agents activated",
                int((arena["Activation"] == "Active").sum()),
            )
            arena_columns[2].metric(
                "Mean sandbox return",
                f"{float(arena['Sandbox return'].mean()):.3%}",
            )
            arena_columns[3].metric(
                "Research GNN winners",
                int((arena["Selected encoder"] == "Surface Graph Set").sum()),
            )
            arena_columns[4].metric(
                "Smile-signal winners",
                int((arena["Smile residual"] == "Enabled").sum()),
            )
            arena_columns[5].metric(
                "Sandbox fills", int(arena["Sandbox executions"].sum())
            )
            arena_columns[6].metric(
                "Median sandbox latency",
                f"{float(arena['Sandbox latency (us)'].median()):.1f} us",
            )
            arena_columns[7].metric(
                "Promotion-ready paths",
                int((arena["Promotion"] == "Promotion ready").sum()),
            )
            arena_chart, winner_chart = st.columns((3, 2))
            with arena_chart:
                st.caption("Research winner versus activated sandbox return")
                st.bar_chart(
                    arena.set_index("Ticker")[
                        [
                            "Held-out return",
                            "Sandbox return",
                        ]
                    ],
                    height=260,
                )
            with winner_chart:
                st.caption("Validation-selected architecture count")
                winner_labels = (
                    arena["Selected encoder"]
                    + " / "
                    + arena["Selected agent"]
                    + " / "
                    + arena["Action policy"]
                    + " / "
                    + arena["Smile residual"]
                )
                winner_counts = winner_labels.value_counts().rename("Selections")
                st.bar_chart(winner_counts, height=260)
            st.dataframe(
                arena,
                width="stretch",
                hide_index=True,
                column_config={
                    "Held-out return": st.column_config.NumberColumn(format="percent"),
                    "Sandbox return": st.column_config.NumberColumn(format="percent"),
                    "Sandbox lift": st.column_config.NumberColumn(format="percent"),
                    "Final NAV": st.column_config.NumberColumn(format="$%.2f"),
                    "Max drawdown": st.column_config.NumberColumn(format="percent"),
                    "Fees": st.column_config.NumberColumn(format="$%.2f"),
                    "Sandbox fees": st.column_config.NumberColumn(format="$%.2f"),
                    "Excess vs no-op": st.column_config.NumberColumn(format="percent"),
                    "Double-cost return": st.column_config.NumberColumn(
                        format="percent"
                    ),
                },
            )
            ablations = pd.concat(
                [
                    feature_ablation_results(runs, "contract_smile_residual"),
                    feature_ablation_results(runs, "surface_velocity"),
                ],
                ignore_index=True,
            )
            if not ablations.empty:
                st.subheader("Engineered-feature experiments")
                st.caption(
                    "Matched validation comparisons remove exactly one causal "
                    "feature group. Positive feature lift means the signal "
                    "helped; held-out data is excluded."
                )
                grouped = ablations.groupby(
                    ["Feature", "Encoder", "Agent"],
                    as_index=False,
                ).agg(
                    **{
                        "Mean feature lift (bp)": ("Feature lift (bp)", "mean"),
                        "Helped tickers": (
                            "Feature helped",
                            lambda values: int((values == "Yes").sum()),
                        ),
                        "Evaluated tickers": ("Ticker", "nunique"),
                    }
                )
                grouped["Candidate"] = (
                    grouped["Feature"]
                    + " / "
                    + grouped["Encoder"]
                    + " / "
                    + grouped["Agent"]
                )
                ablation_chart, ablation_table = st.columns((2, 3))
                with ablation_chart:
                    st.bar_chart(
                        grouped.set_index("Candidate")["Mean feature lift (bp)"],
                        height=260,
                    )
                with ablation_table:
                    st.dataframe(grouped, width="stretch", hide_index=True)
                with st.expander("Per-ticker matched comparisons"):
                    st.dataframe(ablations, width="stretch", hide_index=True)
            st.divider()

        st.subheader("Experiment drill-down")
        run_options = {
            (
                f"{run['_run_name']} · {run.get('symbol', 'unknown')} · "
                f"{len(run.get('folds', []))} fold(s)"
            ): run
            for run in runs
        }
        run_label = st.selectbox("Experiment", tuple(run_options))
        run = run_options[run_label]
        leaderboard = agent_leaderboard(run)
        heldout = heldout_results(run)
        evidence = evidence_summary(run)
        promotion = promotion_assessment(run)
        selection_rules = [
            fold.get("model_selection", {}).get("simplicity_rule", {})
            for fold in run.get("folds", [])
        ]
        selection_rules = [rule for rule in selection_rules if rule]
        activation_gates = [
            fold.get("model_selection", {}).get("activation_gate", {})
            for fold in run.get("folds", [])
        ]
        activation_gates = [gate for gate in activation_gates if gate]

        st.markdown(
            '<span class="scope-pill">Validation-ranked agents</span>'
            '<span class="scope-pill">Selected winner tested once</span>'
            '<span class="scope-pill">Net-liquidation accounting</span>',
            unsafe_allow_html=True,
        )
        if evidence["grade"] == "Exploratory":
            st.warning(
                "Exploratory evidence only. The held-out path is too short "
                "for an alpha claim; results are shown to make agent behavior "
                "inspectable, not to imply profitability."
            )
        provenance = {
            fold.get("test_data_quality", {}).get("execution_provenance")
            for fold in run.get("folds", [])
        }
        if "legacy_unknown_session_fallback" in provenance:
            st.error(
                "Execution provenance warning: at least one held-out slice "
                "uses legacy snapshots without provider-confirmed market-session "
                "or quote-time coverage. Its fills are sandbox demonstrations, "
                "not evidence of executable market performance."
            )
        elif "provider_nonregular_present" in provenance:
            st.warning(
                "The held-out slice includes provider-confirmed non-regular "
                "snapshots. The environment masks non-hold actions there; "
                "keep this path in the sandbox-evidence category."
            )
        if promotion["status"] == "Promotion ready":
            st.success(
                "Promotion gate passed: this run cleared held-out, no-op, "
                "cost-stress, provenance, and action-validity checks."
            )
        else:
            st.error(
                "Research-only policy. Promotion blockers: "
                + "; ".join(promotion["failed_reasons"])
                + "."
            )
        if selection_rules:
            rule = selection_rules[0]
            st.info(
                "Simplest-competitive selection retained "
                f"{int(rule['competitive_candidate_count'])} candidates "
                f"within {float(rule['effective_score_tolerance']) * 10_000:.2f} "
                "bp of the raw validation leader, then used the existing "
                "ablation, latency, and complexity tie-breaks. "
                f"Score traded: {float(rule['score_sacrificed_for_simplicity']) * 10_000:.2f} bp."
            )
        if activation_gates:
            gate = activation_gates[0]
            advantage_bp = float(gate["score_advantage"]) * 10_000
            required_bp = float(gate["minimum_score_advantage"]) * 10_000
            if gate["activated"]:
                st.success(
                    "Sandbox activation passed: validation advantage over "
                    f"no-op was {advantage_bp:.2f} bp versus {required_bp:.2f} "
                    "bp required."
                )
            else:
                st.warning(
                    "Sandbox abstains and deploys no-op. The research winner's "
                    f"validation advantage was {advantage_bp:.2f} bp versus "
                    f"{required_bp:.2f} bp required."
                )

        selected_name = (
            (
                f"{heldout.iloc[0]['Agent']} · "
                f"{heldout.iloc[0]['Encoder']} · "
                f"{heldout.iloc[0]['Action policy']} · "
                f"smile residual {heldout.iloc[0]['Smile residual'].lower()}"
            )
            if not heldout.empty
            else "Unavailable"
        )
        mean_return = float(heldout["Test return"].mean()) if not heldout.empty else 0.0
        activation = (
            str(heldout.iloc[0]["Activation"]) if not heldout.empty else "Unavailable"
        )
        sandbox_return = (
            float(heldout["Sandbox return"].mean()) if not heldout.empty else 0.0
        )
        sandbox_executions = (
            int(heldout["Sandbox executions"].sum()) if not heldout.empty else 0
        )
        sandbox_latency = (
            float(heldout["Sandbox latency (us)"].mean()) if not heldout.empty else 0.0
        )
        metric_columns = st.columns(8)
        metric_columns[0].metric("Agents compared", len(leaderboard))
        metric_columns[1].metric("Research winner", selected_name)
        metric_columns[2].metric("Activation", activation)
        metric_columns[3].metric("Research return", f"{mean_return:.3%}")
        metric_columns[4].metric("Sandbox return", f"{sandbox_return:.3%}")
        metric_columns[5].metric("Sandbox fills", sandbox_executions)
        metric_columns[6].metric("Sandbox latency", f"{sandbox_latency:.1f} us")
        metric_columns[7].metric("Evidence", evidence["grade"])

        st.subheader("Agent leaderboard")
        st.caption(
            "Models are ranked only on validation evidence. Surface Graph Set "
            "agents pass messages over neighboring strike/expiry contracts and "
            "opposite-side counterparts. Matched feature ablations show whether "
            "the contract-level smile residual helps. Held-out data is opened "
            "after a winner is fixed."
        )
        st.dataframe(
            leaderboard,
            width="stretch",
            hide_index=True,
            column_config={
                "Validation score": st.column_config.NumberColumn(format="%.6f"),
                "Validation reward": st.column_config.NumberColumn(format="%.6f"),
                "Median latency (us)": st.column_config.NumberColumn(format="%.1f"),
                "Parameters": st.column_config.NumberColumn(format="%d"),
                "Score gap (bp)": st.column_config.NumberColumn(format="%.3f"),
            },
        )

        fold_numbers = [int(fold["fold"]) for fold in run.get("folds", [])]
        selected_fold = st.selectbox("Inspect held-out fold", fold_numbers)
        fold_result = heldout[heldout["Fold"] == selected_fold]
        if not fold_result.empty:
            report = fold_result.iloc[0]
            st.subheader("Research-winner held-out result")
            result_columns = st.columns(6)
            result_columns[0].metric("Final NAV", f"${report['Final NAV']:,.2f}")
            result_columns[1].metric("Return", f"{report['Test return']:.3%}")
            result_columns[2].metric("Max drawdown", f"{report['Max drawdown']:.3%}")
            result_columns[3].metric("Turnover", f"{report['Turnover']:.3f}x")
            result_columns[4].metric("Fees", f"${report['Fees']:,.2f}")
            result_columns[5].metric(
                "Executions",
                (
                    f"{int(report['Executions'])} "
                    f"({report['Fills / decision']:.2f}/decision)"
                ),
            )

        curve = equity_curve(run, selected_fold)
        if curve.empty:
            st.info(
                "This older artifact has aggregate metrics but no stored path. "
                "Re-run the tournament to unlock equity and trade inspection."
            )
        else:
            st.subheader("Held-out equity path")
            chart = curve.pivot_table(
                index="Timestamp",
                columns="Series",
                values="Equity",
                aggfunc="first",
            )
            st.line_chart(chart, height=340)

        ledger = trade_ledger(run, selected_fold)
        st.subheader("Agent fills")
        if ledger.empty:
            st.info(
                "The selected policy made no executable held-out trades in "
                "this fold. This is a real result, not a missing ledger."
            )
        else:
            st.dataframe(
                ledger,
                width="stretch",
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.4f"),
                    "Fee": st.column_config.NumberColumn(format="$%.2f"),
                    "Post-trade NAV": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

        with st.expander("Run provenance and raw held-out metrics"):
            st.write(
                f"Artifact: `{run['_artifact_path']}` · schema: "
                f"`{run.get('schema_version')}`"
            )
            st.dataframe(heldout, width="stretch", hide_index=True)
            st.json(
                json.loads(
                    json.dumps(
                        {
                            "walk_forward": run.get("walk_forward"),
                            "training": run.get("training"),
                            "candidate_models": run.get("candidate_models"),
                        }
                    )
                )
            )

with market_tab:
    st.markdown('<div class="section-kicker">OPTIONS EXECUTION</div>', unsafe_allow_html=True)
    st.markdown(f"## Trade {symbol} {option_type}s")
    st.caption(
        "Build a simulated order from the latest eligible chain. Nothing here "
        "connects to a live broker."
    )
    first = snapshot.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Underlying", f"${first['underlyingPrice']:,.2f}")
    col2.metric("Expiration", first["expiration"])
    col3.metric("Contracts", len(filtered))
    col4.metric(
        "Quote age",
        (f"{freshness['age_seconds']:.0f}s" if freshness["coverage"] else "unknown"),
    )
    st.caption(
        f"{session['provider_state']} session · risk-free rate "
        f"{float(first['riskFreeRate']):.3%} · collected {first['collectedAt']}"
    )

    st.markdown("### Order ticket")
    if not session["trading_enabled"]:
        st.warning(
            "Orders are disabled because the provider marks this snapshot "
            "outside the regular market session."
        )
    elif not freshness["trading_enabled"]:
        st.warning(
            "Orders are disabled because the provider's underlying quote is stale."
        )
    elif not session["coverage"] or not freshness["coverage"]:
        st.warning(
            "Market-session or quote-time provenance is incomplete. This paper "
            "order is a UI demonstration, not execution evidence."
        )

    controls, quote_panel = st.columns((3, 2), gap="large")
    with controls:
        contract_symbol = st.selectbox(
            "Contract",
            filtered["contractSymbol"].tolist(),
            format_func=lambda contract: (
                f"{contract} · strike "
                f"${filtered.loc[filtered['contractSymbol'] == contract, 'strike'].iloc[0]:,.2f}"
            ),
        )
        quantity = int(st.number_input("Contracts", min_value=1, value=1, step=1))
    selected = filtered[filtered["contractSymbol"] == contract_symbol].iloc[0]
    buy_price = float(selected["ask"])
    sell_price = float(selected["bid"])
    with quote_panel:
        quote_metrics = st.columns(2)
        quote_metrics[0].metric("Bid", f"${sell_price:,.2f}")
        quote_metrics[1].metric("Ask", f"${buy_price:,.2f}")
        st.caption(
            f"Strike ${float(selected['strike']):,.2f} · "
            f"Delta {float(selected['delta']):.3f} · "
            f"IV {float(selected['impliedVolatility']):.2%} · "
            f"open interest {int(selected['openInterest']):,}"
        )

    buy_col, sell_col = st.columns(2)
    order = {
        "contract_symbol": contract_symbol,
        "symbol": symbol,
        "expiration": str(selected["expiration"]),
        "option_type": option_type,
        "strike": float(selected["strike"]),
        "quantity": quantity,
    }
    if buy_col.button(
        f"Paper buy at ask ${buy_price:,.2f}",
        disabled=(
            not execution_enabled or not math.isfinite(buy_price) or buy_price <= 0
        ),
        width="stretch",
    ):
        try:
            fill = broker.buy(**order, price=buy_price)
            st.success(
                f"Bought {fill['quantity']} contract(s) for ${fill['notional']:,.2f}."
            )
        except PaperTradeError as error:
            st.error(str(error))

    if sell_col.button(
        f"Paper sell at bid ${sell_price:,.2f}",
        disabled=(
            not execution_enabled or not math.isfinite(sell_price) or sell_price <= 0
        ),
        width="stretch",
    ):
        try:
            fill = broker.sell(**order, price=sell_price)
            st.success(
                f"Sold {fill['quantity']} contract(s) for ${fill['notional']:,.2f}."
            )
        except PaperTradeError as error:
            st.error(str(error))

    st.markdown("### Explore the chain")
    table_columns = [
        "contractSymbol",
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
        "delta",
        "gamma",
        "theta",
        "vega",
    ]
    with st.expander("Options chain and Greeks", expanded=True):
        st.dataframe(
            filtered[table_columns],
            width="stretch",
            height=420,
            hide_index=True,
            column_config={
                "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                "lastPrice": st.column_config.NumberColumn("Last", format="$%.2f"),
                "bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                "ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                "impliedVolatility": st.column_config.NumberColumn("IV", format="percent"),
                "delta": st.column_config.NumberColumn("Delta", format="%.3f"),
                "gamma": st.column_config.NumberColumn("Gamma", format="%.4f"),
                "theta": st.column_config.NumberColumn("Theta", format="%.3f"),
                "vega": st.column_config.NumberColumn("Vega", format="%.3f"),
            },
        )

with portfolio_tab:
    st.markdown('<div class="section-kicker">PAPER ACCOUNT</div>', unsafe_allow_html=True)
    st.markdown("## Portfolio")
    st.caption("Current marked positions and a complete audit trail of simulated fills.")
    account = broker.account()
    positions = broker.positions()
    quote_frames = [
        load_latest_snapshot(DATA_DIR, ticker)
        for ticker in sorted({position["symbol"] for position in positions})
    ]
    quotes = (
        pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame()
    )
    marked = mark_positions(positions, quotes)
    market_value = float(marked["market_value"].sum()) if not marked.empty else 0.0
    unrealized = float(marked["unrealized_pnl"].sum()) if not marked.empty else 0.0

    cash_col, value_col, equity_col, pnl_col = st.columns(4)
    cash_col.metric("Cash", f"${account['cash']:,.2f}")
    value_col.metric("Option value", f"${market_value:,.2f}")
    equity_col.metric("Total equity", f"${account['cash'] + market_value:,.2f}")
    pnl_col.metric("Unrealized P&L", f"${unrealized:,.2f}")

    if marked.empty:
        st.info("No paper positions yet.")
    else:
        st.dataframe(marked, width="stretch", hide_index=True)

with portfolio_tab:
    st.markdown("### Trade activity")
    trades = broker.trades()
    if trades:
        st.dataframe(pd.DataFrame(trades), width="stretch", hide_index=True)
    else:
        st.info("No paper trades yet.")
