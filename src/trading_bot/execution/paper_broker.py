"""Persistent, long-only paper option broker for testing workflows."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


CONTRACT_MULTIPLIER = 100


class PaperTradeError(ValueError):
    """Raised when a paper order violates an account invariant."""


class PaperBroker:
    """Store paper cash, positions, and fills in a local SQLite database."""

    def __init__(self, database: Path, starting_cash: float = 100_000.0):
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize(starting_cash)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self, starting_cash: float) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL CHECK (cash >= 0),
                    initial_cash REAL NOT NULL CHECK (initial_cash > 0)
                );

                CREATE TABLE IF NOT EXISTS positions (
                    contract_symbol TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    expiration TEXT NOT NULL,
                    option_type TEXT NOT NULL CHECK (option_type IN ('call', 'put')),
                    strike REAL NOT NULL CHECK (strike > 0),
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    average_price REAL NOT NULL CHECK (average_price > 0),
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    executed_at TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
                    contract_symbol TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    expiration TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    notional REAL NOT NULL,
                    realized_pnl REAL NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO account (id, cash, initial_cash) VALUES (1, ?, ?)",
                (starting_cash, starting_cash),
            )

    def account(self) -> dict:
        with self._connect() as connection:
            return dict(connection.execute("SELECT * FROM account WHERE id = 1").fetchone())

    def positions(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM positions ORDER BY symbol, expiration, strike"
            ).fetchall()
        return [dict(row) for row in rows]

    def trades(self, limit: int = 200) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def buy(self, **order) -> dict:
        return self._execute("buy", **order)

    def sell(self, **order) -> dict:
        return self._execute("sell", **order)

    def _execute(
        self,
        side: str,
        contract_symbol: str,
        symbol: str,
        expiration: str,
        option_type: str,
        strike: float,
        quantity: int,
        price: float,
        executed_at: datetime | None = None,
    ) -> dict:
        if side not in {"buy", "sell"}:
            raise PaperTradeError(f"Unsupported side: {side}")
        if option_type not in {"call", "put"}:
            raise PaperTradeError(f"Unsupported option type: {option_type}")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise PaperTradeError("quantity must be a positive integer")
        if price <= 0 or strike <= 0:
            raise PaperTradeError("price and strike must be positive")

        timestamp = (executed_at or datetime.now(timezone.utc)).isoformat()
        notional = round(price * quantity * CONTRACT_MULTIPLIER, 2)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            account = connection.execute(
                "SELECT cash FROM account WHERE id = 1"
            ).fetchone()
            position = connection.execute(
                "SELECT * FROM positions WHERE contract_symbol = ?",
                (contract_symbol,),
            ).fetchone()
            realized_pnl = 0.0

            if side == "buy":
                if account["cash"] < notional:
                    raise PaperTradeError(
                        f"insufficient cash: need ${notional:,.2f}, "
                        f"have ${account['cash']:,.2f}"
                    )
                new_quantity = quantity + (position["quantity"] if position else 0)
                old_cost = (
                    position["quantity"] * position["average_price"]
                    if position
                    else 0.0
                )
                average_price = (old_cost + quantity * price) / new_quantity
                connection.execute(
                    "UPDATE account SET cash = cash - ? WHERE id = 1", (notional,)
                )
                connection.execute(
                    """
                    INSERT INTO positions (
                        contract_symbol, symbol, expiration, option_type, strike,
                        quantity, average_price, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(contract_symbol) DO UPDATE SET
                        quantity = excluded.quantity,
                        average_price = excluded.average_price,
                        updated_at = excluded.updated_at
                    """,
                    (
                        contract_symbol, symbol, expiration, option_type, strike,
                        new_quantity, average_price, timestamp,
                    ),
                )
            else:
                if not position or position["quantity"] < quantity:
                    owned = position["quantity"] if position else 0
                    raise PaperTradeError(
                        f"insufficient position: trying to sell {quantity}, own {owned}"
                    )
                remaining = position["quantity"] - quantity
                realized_pnl = round(
                    (price - position["average_price"])
                    * quantity
                    * CONTRACT_MULTIPLIER,
                    2,
                )
                connection.execute(
                    "UPDATE account SET cash = cash + ? WHERE id = 1", (notional,)
                )
                if remaining:
                    connection.execute(
                        "UPDATE positions SET quantity = ?, updated_at = ? "
                        "WHERE contract_symbol = ?",
                        (remaining, timestamp, contract_symbol),
                    )
                else:
                    connection.execute(
                        "DELETE FROM positions WHERE contract_symbol = ?",
                        (contract_symbol,),
                    )

            cursor = connection.execute(
                """
                INSERT INTO trades (
                    executed_at, side, contract_symbol, symbol, expiration,
                    option_type, strike, quantity, price, notional, realized_pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, side, contract_symbol, symbol, expiration,
                    option_type, strike, quantity, price, notional, realized_pnl,
                ),
            )
            trade_id = cursor.lastrowid

        return {
            "id": trade_id,
            "side": side,
            "contract_symbol": contract_symbol,
            "quantity": quantity,
            "price": price,
            "notional": notional,
            "realized_pnl": realized_pnl,
        }
