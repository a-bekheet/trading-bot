"""Mark paper option positions against collected quotes."""

import math

import pandas as pd

from trading_bot.execution.paper_broker import CONTRACT_MULTIPLIER


def mark_positions(positions: list[dict], quotes: pd.DataFrame) -> pd.DataFrame:
    """Return positions with current marks, value, and unrealized P&L."""
    if not positions:
        return pd.DataFrame()

    quote_lookup = quotes.drop_duplicates("contractSymbol", keep="last").set_index(
        "contractSymbol"
    )
    marked = []
    for position in positions:
        quote = (
            quote_lookup.loc[position["contract_symbol"]]
            if position["contract_symbol"] in quote_lookup.index
            else None
        )
        mark = math.nan
        if quote is not None:
            bid = float(quote.get("bid", 0) or 0)
            ask = float(quote.get("ask", 0) or 0)
            last = float(quote.get("lastPrice", 0) or 0)
            mark = (bid + ask) / 2 if bid > 0 and ask > 0 else last

        row = dict(position)
        row["mark_price"] = mark
        row["market_value"] = mark * position["quantity"] * CONTRACT_MULTIPLIER
        row["unrealized_pnl"] = (
            (mark - position["average_price"])
            * position["quantity"]
            * CONTRACT_MULTIPLIER
        )
        marked.append(row)
    return pd.DataFrame(marked)
