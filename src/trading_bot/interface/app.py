"""Streamlit research workspace for agents, market data, and paper trading."""

import json
import math
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
    promotion_assessment,
    trade_ledger,
)


DATA_DIR = Path("data")
PAPER_DATABASE = DATA_DIR / "paper_portfolio.db"

st.set_page_config(page_title="Options Sandbox", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: #f5f7fb;}
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e6e9f0;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 18px rgba(24, 33, 56, 0.04);
    }
    .agent-hero {
        padding: 24px 28px;
        border-radius: 20px;
        background: linear-gradient(125deg, #121b34 0%, #243866 64%, #245b79 100%);
        color: white;
        margin-bottom: 14px;
    }
    .agent-hero h1 {font-size: 2rem; margin: 0 0 6px 0;}
    .agent-hero p {margin: 0; color: #dce6ff;}
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
    </style>
    <div class="agent-hero">
      <h1>Options Agent Research Lab</h1>
      <p>Train, compare, inspect, and paper-test recurrent options agents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("RESEARCH DEMO · PAPER TRADING ONLY · NO LIVE BROKER CONNECTION")

tickers = available_tickers(DATA_DIR)
if not tickers:
    st.info("No CSV data found. Run `collect-options --once` first.")
    st.stop()

broker = PaperBroker(PAPER_DATABASE)
st.sidebar.header("Research workspace")
symbol = st.sidebar.selectbox("Ticker", tickers)
option_type = st.sidebar.radio("Option type", ("call", "put"), horizontal=True)
snapshot = load_latest_snapshot(DATA_DIR, symbol)
filtered = snapshot[snapshot["optionType"] == option_type].sort_values("strike")
session = market_session_status(snapshot)
freshness = market_data_freshness_status(snapshot)
execution_enabled = bool(session["trading_enabled"] and freshness["trading_enabled"])

agent_tab, market_tab, portfolio_tab, history_tab = st.tabs(
    (
        "Agent Results",
        "Market & Paper Order",
        "Paper Portfolio",
        "Trade History",
    )
)

with agent_tab:
    runs = discover_agent_runs(DATA_DIR)
    arena_manifests = discover_agent_arena_manifests(DATA_DIR)
    readiness = (
        arena_readiness_overview(arena_manifests[0])
        if arena_manifests
        else pd.DataFrame()
    )
    if not readiness.empty:
        ready_count = int((readiness["Ready"] == "Yes").sum())
        readiness_columns = st.columns(3)
        readiness_columns[0].metric(
            "Arena tails ready",
            f"{ready_count}/{len(readiness)}",
        )
        readiness_columns[1].metric(
            "Regular validation states",
            int(readiness["Validation regular"].sum()),
        )
        readiness_columns[2].metric(
            "Regular test states",
            int(readiness["Test regular"].sum()),
        )
        if ready_count < len(readiness):
            st.warning(
                "The newest arena preflight is waiting for every validation "
                "and test state to be provider-confirmed regular, carry a "
                "fresh underlying quote, and contain an executable option "
                "quote. Expensive agent training is skipped until then."
            )
        else:
            st.success(
                "The newest arena validation/test tails passed the regular, "
                "fresh, and executable pre-training gate."
            )
        with st.expander("Arena evidence readiness"):
            st.dataframe(readiness, width="stretch", hide_index=True)
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
            ablations = feature_ablation_results(runs)
            if not ablations.empty:
                st.subheader("Contract smile-residual experiment")
                st.caption(
                    "Matched validation comparison with only the causal "
                    "contract-level smile residual removed. Positive feature "
                    "lift means the signal helped; held-out data is excluded."
                )
                grouped = ablations.groupby(["Encoder", "Agent"], as_index=False).agg(
                    **{
                        "Mean feature lift (bp)": (
                            "Feature lift (bp)",
                            "mean",
                        ),
                        "Helped tickers": (
                            "Feature helped",
                            lambda values: int((values == "Yes").sum()),
                        ),
                        "Evaluated tickers": ("Ticker", "nunique"),
                    }
                )
                grouped["Candidate"] = grouped["Encoder"] + " / " + grouped["Agent"]
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
    first = snapshot.iloc[0]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Underlying", f"${first['underlyingPrice']:,.2f}")
    col2.metric("Expiration", first["expiration"])
    col3.metric("Risk-free rate", f"{first['riskFreeRate']:.3%}")
    col4.metric("Contracts", len(filtered))
    col5.metric("Market session", session["provider_state"])
    col6.metric(
        "Underlying quote age",
        (f"{freshness['age_seconds']:.0f}s" if freshness["coverage"] else "unknown"),
    )
    st.caption(f"Collected at {first['collectedAt']} · Greeks: Black-Scholes-Merton")

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
    st.dataframe(filtered[table_columns], width="stretch", hide_index=True)

    st.subheader("Paper order")
    if not session["trading_enabled"]:
        st.warning(
            "Paper orders are disabled because the provider marks this "
            "snapshot as outside the regular market session."
        )
    elif not freshness["trading_enabled"]:
        st.warning(
            "Paper orders are disabled because the provider's underlying "
            "quote timestamp is explicitly stale."
        )
    elif not session["coverage"] or not freshness["coverage"]:
        st.warning(
            "Market-session or quote-time provenance is unavailable in this "
            "legacy snapshot; research-demo orders remain enabled but are not "
            "execution evidence."
        )
    contract_symbol = st.selectbox(
        "Contract",
        filtered["contractSymbol"].tolist(),
        format_func=lambda contract: (
            f"{contract} · strike "
            f"${filtered.loc[filtered['contractSymbol'] == contract, 'strike'].iloc[0]:,.2f}"
        ),
    )
    selected = filtered[filtered["contractSymbol"] == contract_symbol].iloc[0]
    quantity = int(st.number_input("Contracts", min_value=1, value=1, step=1))
    buy_price = float(selected["ask"])
    sell_price = float(selected["bid"])
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

with portfolio_tab:
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

with history_tab:
    trades = broker.trades()
    if trades:
        st.dataframe(pd.DataFrame(trades), width="stretch", hide_index=True)
    else:
        st.info("No paper trades yet.")
