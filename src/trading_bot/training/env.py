"""Deterministic Gymnasium-style research-demo options environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.features import ENGINEERED_FEATURES
from trading_bot.training.manifest import EnvManifest
from trading_bot.training.schemas import Action, Observation


CONTRACT_FEATURES = (
    "strike", "lastPrice", "bid", "ask", "impliedVolatility", "delta", "gamma",
    "theta", "vega", "volume", "openInterest", *ENGINEERED_FEATURES,
)
GREEK_NAMES = ("delta", "gamma", "theta", "vega")
PORTFOLIO_FEATURES = (
    "cash", "investedCost", "netAssetValue", *GREEK_NAMES,
)


@dataclass
class Position:
    quantity: int
    average_price: float
    greeks: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


class OptionsEnv:
    """Long-only, fixed-slot environment over current CSV snapshots.

    This is explicitly a research demo. It is deterministic and useful for
    integration testing, not a historical-performance simulator.
    """

    def __init__(
        self,
        dataset: SnapshotDataset,
        manifest: EnvManifest | None = None,
        *,
        slot_count: int = 32,
        max_quantity: int = 3,
        starting_cash: float = 100_000.0,
        commission_per_contract: float = 0.65,
        invalid_action_penalty: float = 0.001,
        max_abs_delta: float | None = None,
        max_abs_gamma: float | None = None,
        max_abs_theta: float | None = None,
        max_abs_vega: float | None = None,
    ):
        if slot_count < 1 or max_quantity < 1:
            raise ValueError("slot_count and max_quantity must be positive")
        limits = {
            "delta": max_abs_delta,
            "gamma": max_abs_gamma,
            "theta": max_abs_theta,
            "vega": max_abs_vega,
        }
        if any(limit is not None and limit <= 0 for limit in limits.values()):
            raise ValueError("Greek risk limits must be positive when provided")
        self.dataset = dataset
        self.slot_count = slot_count
        self.max_quantity = max_quantity
        self.starting_cash = starting_cash
        self.commission_per_contract = commission_per_contract
        self.invalid_action_penalty = invalid_action_penalty
        self.risk_limits = limits
        self.manifest = manifest or EnvManifest(
            symbol=dataset.symbol,
            slot_count=slot_count,
            max_quantity=max_quantity,
            starting_cash=starting_cash,
            commission_per_contract=commission_per_contract,
            invalid_action_penalty=invalid_action_penalty,
            max_abs_delta=max_abs_delta,
            max_abs_gamma=max_abs_gamma,
            max_abs_theta=max_abs_theta,
            max_abs_vega=max_abs_vega,
        )
        self._rng = np.random.default_rng()
        self._index = 0
        self._cash = starting_cash
        self._positions: dict[str, Position] = {}
        self._cached_index = -1
        self._cached_observation: Observation | None = None
        self._cached_info: dict[str, Any] = {}
        self._cached_slots: list[pd.Series] = []

    @classmethod
    def from_directory(cls, data_dir: Path, symbol: str, **kwargs: Any) -> "OptionsEnv":
        dataset = SnapshotDataset.from_directory(data_dir, symbol)
        manifest = kwargs.pop("manifest", None)
        if manifest is None:
            manifest = EnvManifest.for_directory(data_dir, symbol=dataset.symbol, **{
                key: kwargs[key]
                for key in (
                    "slot_count", "max_quantity", "starting_cash",
                    "commission_per_contract", "invalid_action_penalty",
                    "max_abs_delta", "max_abs_gamma", "max_abs_theta",
                    "max_abs_vega",
                )
                if key in kwargs
            })
        return cls(dataset, manifest=manifest, **kwargs)

    @property
    def action_shape(self) -> tuple[int, int]:
        return self.slot_count, 2 * self.max_quantity + 1

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._rng = np.random.default_rng(seed)
        self._cash = self.starting_cash
        self._positions = {}
        options = options or {}
        start = int(options.get("start_index", 0))
        if not 0 <= start < len(self.dataset):
            raise ValueError("start_index is outside the dataset")
        self._index = start
        frame = self._current_frame()
        slots = self._slots(frame)
        observation, info = self._observation(frame, slots)
        self._cache_state(observation, info, slots)
        info.update({"seed": seed, "manifest_fingerprint": self.manifest.fingerprint})
        return observation, info

    def step(self, action: Action | np.ndarray):
        action = action if isinstance(action, Action) else Action(np.asarray(action))
        orders = np.asarray(action.orders, dtype=int)
        if orders.shape != (self.slot_count,):
            raise ValueError(f"action must have shape {(self.slot_count,)}")

        current_frame = self._current_frame()
        if self._cached_index == self._index and self._cached_observation is not None:
            current_slots = self._cached_slots
            current_observation = self._cached_observation
            pre_info = self._cached_info.copy()
        else:
            current_slots = self._slots(current_frame)
            current_observation, pre_info = self._observation(
                current_frame, current_slots
            )
        pre_nav = float(current_observation.portfolio[2])
        running_exposures = current_observation.portfolio[3:].copy()
        invalid_actions = 0
        executions: list[dict[str, Any]] = []
        fees = 0.0
        for slot, encoded in enumerate(orders):
            if encoded == 0:
                continue
            if encoded < 0 or encoded > 2 * self.max_quantity:
                invalid_actions += 1
                continue
            if not current_observation.action_mask[slot, encoded]:
                invalid_actions += 1
                continue
            contract = current_slots[slot]
            quantity = encoded if encoded <= self.max_quantity else encoded - self.max_quantity
            side = "buy" if encoded <= self.max_quantity else "sell"
            greek_change = self._contract_greeks(contract) * quantity * 100
            if side == "sell":
                greek_change = -greek_change
            if not self._risk_allowed(running_exposures, greek_change):
                invalid_actions += 1
                continue
            price = float(contract["ask"] if side == "buy" else contract["bid"])
            fee = quantity * self.commission_per_contract
            try:
                self._fill(side, contract, quantity, price, fee)
            except ValueError:
                # The mask describes the pre-step state; multiple buy orders in
                # one action can collectively exceed cash. Such an order is an
                # explicit invalid action, never a negative-cash transition.
                invalid_actions += 1
                continue
            fees += fee
            running_exposures += greek_change
            executions.append({"side": side, "contract_symbol": contract["contractSymbol"], "quantity": quantity, "price": price, "fee": fee})

        next_index = self._index + 1
        truncated = next_index >= len(self.dataset)
        if not truncated:
            self._index = next_index
        next_frame = self._current_frame()
        next_slots = self._slots(next_frame)
        next_observation, next_info = self._observation(next_frame, next_slots)
        self._cache_state(next_observation, next_info, next_slots)
        post_nav = float(next_observation.portfolio[2])
        pnl = post_nav - pre_nav
        if pre_nav > 0:
            gross_pnl_return = (pnl + fees) / pre_nav
            fee_return = -fees / pre_nav
        else:
            gross_pnl_return = -1.0
            fee_return = 0.0
        invalid_return = -invalid_actions * self.invalid_action_penalty
        reward = gross_pnl_return + fee_return + invalid_return
        terminated = post_nav <= 0
        info = {
            **pre_info,
            **next_info,
            "pnl": pnl,
            "fees": fees,
            "trade_notional": sum(item["price"] * item["quantity"] * 100 for item in executions),
            "invalid_action_count": invalid_actions,
            "executions": executions,
            "greek_exposures": {
                name: float(next_observation.portfolio[3 + index])
                for index, name in enumerate(GREEK_NAMES)
            },
            "reward_components": {
                "gross_pnl_return": gross_pnl_return,
                "fees": fee_return,
                "invalid_action": invalid_return,
            },
        }
        return next_observation, float(reward), terminated, truncated, info

    def _cache_state(
        self,
        observation: Observation,
        info: dict[str, Any],
        slots: list[pd.Series],
    ) -> None:
        """Cache exactly the slot state returned to the policy."""
        self._cached_index = self._index
        self._cached_observation = observation
        self._cached_info = info.copy()
        self._cached_slots = slots

    def _current_frame(self) -> pd.DataFrame:
        return self.dataset.snapshots[self._index].frame

    def _slots(self, frame: pd.DataFrame) -> list[pd.Series]:
        ranked = frame.copy()
        if "logMoneyness" in ranked:
            ranked["_moneynessDistance"] = pd.to_numeric(
                ranked["logMoneyness"], errors="coerce"
            ).abs().fillna(float("inf"))
        else:
            spot = pd.to_numeric(ranked["underlyingPrice"], errors="coerce")
            strike = pd.to_numeric(ranked["strike"], errors="coerce")
            ranked["_moneynessDistance"] = np.log(spot / strike).abs().replace(
                [np.inf, -np.inf], np.nan
            ).fillna(float("inf"))
        spread = (
            ranked["spreadPct"]
            if "spreadPct" in ranked
            else pd.Series(float("inf"), index=ranked.index)
        )
        open_interest = (
            ranked["openInterest"]
            if "openInterest" in ranked
            else pd.Series(0.0, index=ranked.index)
        )
        ranked["_spreadRank"] = pd.to_numeric(
            spread, errors="coerce"
        ).fillna(float("inf"))
        ranked["_openInterestRank"] = -pd.to_numeric(
            open_interest, errors="coerce"
        ).fillna(0.0)
        ordering = [
            "expiration", "optionType", "_moneynessDistance", "_spreadRank",
            "_openInterestRank", "strike", "contractSymbol",
        ]
        ranked = ranked.sort_values(ordering)
        held_ids = set(self._positions)
        held = ranked[ranked["contractSymbol"].isin(held_ids)]
        remainder = ranked[~ranked["contractSymbol"].isin(held_ids)].copy()
        remainder["_surfaceDepth"] = remainder.groupby(
            ["expiration", "optionType"], sort=False
        ).cumcount()
        remainder = remainder.sort_values(
            ["_surfaceDepth", "expiration", "optionType", *ordering[2:]]
        )
        selected = pd.concat((held, remainder.head(max(0, self.slot_count - len(held)))))
        return [row for _, row in selected.head(self.slot_count).iterrows()]

    def _observation(
        self,
        frame: pd.DataFrame | None = None,
        slots: list[pd.Series] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        frame = self._current_frame() if frame is None else frame
        slots = self._slots(frame) if slots is None else slots
        contracts = np.zeros((self.slot_count, len(CONTRACT_FEATURES)), dtype=np.float64)
        valid = np.zeros(self.slot_count, dtype=bool)
        ids: list[str | None] = [None] * self.slot_count
        for index, contract in enumerate(slots):
            ids[index] = str(contract["contractSymbol"])
            valid[index] = self._quote_valid(contract)
            for feature_index, name in enumerate(CONTRACT_FEATURES):
                contracts[index, feature_index] = float(contract.get(name, 0) or 0)
        nav, exposures = self._portfolio_metrics(frame)
        action_mask = np.zeros(self.action_shape, dtype=bool)
        for index, contract in enumerate(slots):
            if not valid[index]:
                continue
            action_mask[index, 0] = True
            held = self._positions.get(ids[index] or "", Position(0, 0)).quantity
            ask = float(contract["ask"])
            bid = float(contract["bid"])
            contract_greeks = self._contract_greeks(contract)
            for quantity in range(1, self.max_quantity + 1):
                fee = quantity * self.commission_per_contract
                greek_change = contract_greeks * quantity * 100
                action_mask[index, quantity] = (
                    self._cash >= ask * quantity * 100 + fee
                    and self._risk_allowed(exposures, greek_change)
                )
                action_mask[index, self.max_quantity + quantity] = (
                    held >= quantity
                    and bid > 0
                    and self._risk_allowed(exposures, -greek_change)
                )
        spot = float(frame["underlyingPrice"].iloc[0])
        rate = float(frame["riskFreeRate"].iloc[0])
        market = np.array([spot, rate], dtype=np.float64)
        invested = sum(position.quantity * position.average_price * 100 for position in self._positions.values())
        portfolio = np.concatenate(
            (np.array([self._cash, invested, nav], dtype=np.float64), exposures)
        )
        return Observation(
            self.dataset.snapshots[self._index].timestamp,
            market,
            contracts,
            portfolio,
            valid,
            action_mask,
            tuple(ids),
        ), {
            "index": self._index,
            "data_source": self.manifest.data_source,
            "portfolio_features": PORTFOLIO_FEATURES,
            "risk_limits": self.risk_limits.copy(),
        }

    @staticmethod
    def _contract_greeks(contract: pd.Series) -> np.ndarray:
        values = pd.to_numeric(
            pd.Series([contract.get(name, 0.0) for name in GREEK_NAMES]),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = 0.0
        return values

    def _risk_allowed(self, current: np.ndarray, change: np.ndarray) -> bool:
        projected = current + change
        for index, name in enumerate(GREEK_NAMES):
            limit = self.risk_limits[name]
            if limit is None:
                continue
            if (
                abs(projected[index]) > limit
                and abs(projected[index]) > abs(current[index]) + 1e-12
            ):
                return False
        return True

    @staticmethod
    def _quote_valid(contract: pd.Series) -> bool:
        try:
            return all(math.isfinite(float(contract[name])) and float(contract[name]) > 0 for name in ("bid", "ask", "lastPrice")) and float(contract["bid"]) <= float(contract["ask"])
        except (TypeError, ValueError):
            return False

    def _fill(self, side: str, contract: pd.Series, quantity: int, price: float, fee: float) -> None:
        symbol = str(contract["contractSymbol"])
        greeks = tuple(float(value) for value in self._contract_greeks(contract))
        notional = price * quantity * 100 + (fee if side == "buy" else -fee)
        if side == "buy":
            if self._cash < notional:
                raise ValueError("insufficient cash")
            self._cash -= notional
            position = self._positions.get(symbol)
            if position:
                total = position.quantity + quantity
                position.average_price = (position.quantity * position.average_price + quantity * price) / total
                position.quantity = total
                position.greeks = greeks
            else:
                self._positions[symbol] = Position(quantity, price, greeks)
        else:
            if symbol not in self._positions or self._positions[symbol].quantity < quantity:
                raise ValueError("insufficient position")
            self._cash += notional
            position = self._positions[symbol]
            position.greeks = greeks
            position.quantity -= quantity
            if position.quantity == 0:
                del self._positions[symbol]

    def _portfolio_metrics(self, frame: pd.DataFrame) -> tuple[float, np.ndarray]:
        quotes = frame.drop_duplicates("contractSymbol").set_index("contractSymbol")
        value = 0.0
        exposures = np.zeros(len(GREEK_NAMES), dtype=np.float64)
        for symbol, position in self._positions.items():
            if symbol not in quotes.index:
                value += position.quantity * position.average_price * 100
                exposures += np.asarray(position.greeks) * position.quantity * 100
                continue
            quote = quotes.loc[symbol]
            bid, ask = float(quote["bid"]), float(quote["ask"])
            mark = (bid + ask) / 2 if bid > 0 and ask > 0 else float(quote["lastPrice"])
            value += position.quantity * mark * 100
            exposures += self._contract_greeks(quote) * position.quantity * 100
        return self._cash + value, exposures

    def _nav(self, frame: pd.DataFrame) -> float:
        return self._portfolio_metrics(frame)[0]
