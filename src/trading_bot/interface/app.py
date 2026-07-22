"""Streamlit interface for option data and paper trading."""

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


DATA_DIR = Path("data")
PAPER_DATABASE = DATA_DIR / "paper_portfolio.db"

st.set_page_config(page_title="Options Sandbox", layout="wide")
st.title("Options Research & Paper-Trading Sandbox")
st.warning("Paper trading only. This application has no live broker connection.")

tickers = available_tickers(DATA_DIR)
if not tickers:
    st.info("No CSV data found. Run `collect-options --once` first.")
    st.stop()

broker = PaperBroker(PAPER_DATABASE)
symbol = st.sidebar.selectbox("Ticker", tickers)
option_type = st.sidebar.radio("Option type", ("call", "put"), horizontal=True)
snapshot = load_latest_snapshot(DATA_DIR, symbol)
filtered = snapshot[snapshot["optionType"] == option_type].sort_values("strike")
session = market_session_status(snapshot)
freshness = market_data_freshness_status(snapshot)
execution_enabled = bool(
    session["trading_enabled"] and freshness["trading_enabled"]
)

market_tab, portfolio_tab, history_tab = st.tabs(
    ("Market Data & Paper Order", "Paper Portfolio", "Trade History")
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
        (
            f"{freshness['age_seconds']:.0f}s"
            if freshness["coverage"]
            else "unknown"
        ),
    )
    st.caption(
        f"Collected at {first['collectedAt']} · Greeks: Black-Scholes-Merton"
    )

    table_columns = [
        "contractSymbol", "strike", "lastPrice", "bid", "ask", "volume",
        "openInterest", "impliedVolatility", "delta", "gamma", "theta", "vega",
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
            not execution_enabled
            or not math.isfinite(buy_price)
            or buy_price <= 0
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
            not execution_enabled
            or not math.isfinite(sell_price)
            or sell_price <= 0
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
    quotes = pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame()
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
