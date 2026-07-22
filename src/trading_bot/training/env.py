"""Deterministic Gymnasium-style research-demo options environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.training.dataset import SnapshotDataset
from trading_bot.training.features import (
    ENGINEERED_FEATURES,
    MARKET_ENGINEERED_FEATURES,
)
from trading_bot.training.manifest import EnvManifest
from trading_bot.training.schemas import Action, Observation


CONTRACT_ENGINEERED_FEATURES = tuple(
    name for name in ENGINEERED_FEATURES if name not in MARKET_ENGINEERED_FEATURES
)
POSITION_CONTRACT_FEATURES = (
    "positionQuantity",
    "positionAveragePrice",
    "positionUnrealizedReturn",
    "positionAgeSteps",
    "positionLastTradeAgeSteps",
)
ACTION_FEASIBILITY_CONTRACT_FEATURES = (
    "buyFeasibleFraction",
    "sellFeasibleFraction",
)
CONTRACT_FEATURES = (
    "strike", "lastPrice", "bid", "ask", "impliedVolatility", "delta", "gamma",
    "theta", "vega", *CONTRACT_ENGINEERED_FEATURES,
    *POSITION_CONTRACT_FEATURES, *ACTION_FEASIBILITY_CONTRACT_FEATURES,
    "slotContinuity",
)
BUY_FEASIBILITY_INDEX = CONTRACT_FEATURES.index("buyFeasibleFraction")
SELL_FEASIBILITY_INDEX = CONTRACT_FEATURES.index("sellFeasibleFraction")
MARKET_FEATURES = ("underlyingPrice", "riskFreeRate", *MARKET_ENGINEERED_FEATURES)
GREEK_NAMES = ("delta", "gamma", "theta", "vega")
PORTFOLIO_FEATURES = (
    "cash", "investedCost", "netAssetValue", *GREEK_NAMES,
    "underlyingShares", "reservedCashCollateral", "reservedCoveredShares",
    "underlyingBuyFeasibleFraction", "underlyingSellFeasibleFraction",
)
PORTFOLIO_GREEK_SLICE = slice(3, 7)
OPTION_CONTRACT_MULTIPLIER = 100
PORTFOLIO_VALUATION_MODES = ("liquidation", "midpoint")


@dataclass
class Position:
    quantity: int
    average_price: float
    greeks: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    option_type: str = ""
    strike: float = 0.0
    expiration: str = ""
    last_liquidation_price: float = 0.0
    opened_index: int = 0
    last_trade_index: int = 0


@dataclass(frozen=True, slots=True)
class _ContractRow:
    """Cheap scalar view over one immutable snapshot row."""

    frame: "_FrameRows"
    position: int

    def __getitem__(self, name: str) -> Any:
        return self.frame.columns[name][self.position]

    def get(self, name: str, default: Any = None) -> Any:
        values = self.frame.columns.get(name)
        return default if values is None else values[self.position]


class _FrameRows:
    """Column-array snapshot view without per-row pandas Series allocation."""

    __slots__ = ("columns", "positions")

    def __init__(self, frame: pd.DataFrame):
        self.columns = {
            str(name): values.to_numpy(copy=False)
            for name, values in frame.items()
        }
        self.positions: dict[str, int] = {}
        symbols = self.columns.get("contractSymbol", ())
        for position, symbol in enumerate(symbols):
            self.positions.setdefault(str(symbol), position)

    def get(self, symbol: str) -> _ContractRow | None:
        position = self.positions.get(symbol)
        return None if position is None else _ContractRow(self, position)

    def at(self, position: int) -> _ContractRow:
        return _ContractRow(self, position)


class OptionsEnv:
    """Bounded option and underlying trades over CSV snapshots.

    This is explicitly a research demo. It is deterministic and useful for
    integration testing, not a historical-performance simulator.
    """

    def __init__(
        self,
        dataset: SnapshotDataset,
        manifest: EnvManifest | None = None,
        *,
        slot_count: int = 32,
        slot_assignment: str = "stable",
        max_quantity: int = 3,
        allow_collateralized_option_shorts: bool = False,
        starting_cash: float = 100_000.0,
        commission_per_contract: float = 0.65,
        spread_multiplier: float = 1.0,
        portfolio_valuation: str = "liquidation",
        underlying_lot_size: int = 25,
        max_abs_underlying_shares: int = 500,
        underlying_commission_per_share: float = 0.005,
        underlying_slippage_bps: float = 1.0,
        invalid_action_penalty: float = 0.001,
        reward_drawdown_penalty: float = 0.0,
        reward_downside_penalty: float = 0.0,
        max_abs_delta: float | None = None,
        max_abs_gamma: float | None = None,
        max_abs_theta: float | None = None,
        max_abs_vega: float | None = None,
    ):
        if slot_count < 1 or max_quantity < 1:
            raise ValueError("slot_count and max_quantity must be positive")
        if slot_assignment not in {"stable", "ranked"}:
            raise ValueError("slot_assignment must be stable or ranked")
        if portfolio_valuation not in PORTFOLIO_VALUATION_MODES:
            raise ValueError(
                "portfolio_valuation must be liquidation or midpoint"
            )
        if (
            commission_per_contract < 0
            or spread_multiplier < 0
            or underlying_commission_per_share < 0
            or underlying_slippage_bps < 0
        ):
            raise ValueError("execution costs cannot be negative")
        reward_penalties = {
            "invalid_action": invalid_action_penalty,
            "drawdown": reward_drawdown_penalty,
            "downside": reward_downside_penalty,
        }
        if any(
            not math.isfinite(penalty) or penalty < 0
            for penalty in reward_penalties.values()
        ):
            raise ValueError("reward penalties must be finite and nonnegative")
        if underlying_lot_size < 1 or max_abs_underlying_shares < underlying_lot_size:
            raise ValueError("underlying position limits and lot size are invalid")
        limits = {
            "delta": max_abs_delta,
            "gamma": max_abs_gamma,
            "theta": max_abs_theta,
            "vega": max_abs_vega,
        }
        if any(limit is not None and limit <= 0 for limit in limits.values()):
            raise ValueError("Greek risk limits must be positive when provided")
        if manifest is not None and manifest.schema_version != EnvManifest().schema_version:
            raise ValueError("environment manifest schema is incompatible")
        self.dataset = dataset
        self.slot_count = slot_count
        self.slot_assignment = slot_assignment
        self.max_quantity = max_quantity
        self.allow_collateralized_option_shorts = bool(
            allow_collateralized_option_shorts
        )
        self.starting_cash = starting_cash
        self.commission_per_contract = commission_per_contract
        self.spread_multiplier = spread_multiplier
        self.portfolio_valuation = portfolio_valuation
        self.underlying_lot_size = underlying_lot_size
        self.max_abs_underlying_shares = max_abs_underlying_shares
        self.underlying_commission_per_share = underlying_commission_per_share
        self.underlying_slippage_bps = underlying_slippage_bps
        self.invalid_action_penalty = invalid_action_penalty
        self.reward_drawdown_penalty = reward_drawdown_penalty
        self.reward_downside_penalty = reward_downside_penalty
        self.risk_limits = limits
        manifest_values = {
            "symbol": dataset.symbol,
            "slot_count": slot_count,
            "slot_assignment": slot_assignment,
            "max_quantity": max_quantity,
            "allow_collateralized_option_shorts": (
                self.allow_collateralized_option_shorts
            ),
            "starting_cash": starting_cash,
            "commission_per_contract": commission_per_contract,
            "spread_multiplier": spread_multiplier,
            "portfolio_valuation": portfolio_valuation,
            "underlying_lot_size": underlying_lot_size,
            "max_abs_underlying_shares": max_abs_underlying_shares,
            "underlying_commission_per_share": underlying_commission_per_share,
            "underlying_slippage_bps": underlying_slippage_bps,
            "invalid_action_penalty": invalid_action_penalty,
            "reward_drawdown_penalty": reward_drawdown_penalty,
            "reward_downside_penalty": reward_downside_penalty,
            "max_abs_delta": max_abs_delta,
            "max_abs_gamma": max_abs_gamma,
            "max_abs_theta": max_abs_theta,
            "max_abs_vega": max_abs_vega,
        }
        self.manifest = (
            replace(manifest, **manifest_values)
            if manifest is not None
            else EnvManifest(data_hash=dataset.fingerprint, **manifest_values)
        )
        self._rng = np.random.default_rng()
        self._index = 0
        self._cash = starting_cash
        self._positions: dict[str, Position] = {}
        self._underlying_shares = 0
        self._cached_index = -1
        self._cached_observation: Observation | None = None
        self._cached_info: dict[str, Any] = {}
        self._cached_slots: list[_ContractRow | None] = []
        self._contract_home_slots: dict[str, int] = {}
        self._peak_nav = starting_cash
        self._max_drawdown = 0.0

    @classmethod
    def from_directory(cls, data_dir: Path, symbol: str, **kwargs: Any) -> "OptionsEnv":
        dataset = SnapshotDataset.from_directory(data_dir, symbol)
        manifest = kwargs.pop("manifest", None)
        if manifest is None:
            manifest = EnvManifest.for_directory(data_dir, symbol=dataset.symbol, **{
                key: kwargs[key]
                for key in (
                    "slot_count", "slot_assignment", "max_quantity",
                    "allow_collateralized_option_shorts", "starting_cash",
                    "commission_per_contract", "spread_multiplier",
                    "portfolio_valuation",
                    "underlying_lot_size", "max_abs_underlying_shares",
                    "underlying_commission_per_share",
                    "underlying_slippage_bps",
                    "invalid_action_penalty",
                    "reward_drawdown_penalty",
                    "reward_downside_penalty",
                    "max_abs_delta", "max_abs_gamma", "max_abs_theta",
                    "max_abs_vega",
                )
                if key in kwargs
            })
        return cls(dataset, manifest=manifest, **kwargs)

    @property
    def action_shape(self) -> tuple[int, int]:
        return self.slot_count + 1, 2 * self.max_quantity + 1

    @property
    def underlying_action_quantities(self) -> np.ndarray:
        positive = np.arange(1, self.max_quantity + 1) * self.underlying_lot_size
        return np.concatenate((np.array([0]), positive, -positive)).astype(np.int64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._rng = np.random.default_rng(seed)
        self._cash = self.starting_cash
        self._positions = {}
        self._underlying_shares = 0
        self._cached_index = -1
        self._cached_observation = None
        self._cached_info = {}
        self._cached_slots = []
        self._contract_home_slots = {}
        options = options or {}
        start = int(options.get("start_index", 0))
        if not 0 <= start < len(self.dataset):
            raise ValueError("start_index is outside the dataset")
        self._index = start
        frame = self._current_frame()
        slots = self._slots(frame)
        observation, info = self._observation(frame, slots)
        self._peak_nav = float(observation.portfolio[2])
        self._max_drawdown = 0.0
        self._cache_state(observation, info, slots)
        info.update({"seed": seed, "manifest_fingerprint": self.manifest.fingerprint})
        return observation, info

    def step(self, action: Action | np.ndarray):
        action = action if isinstance(action, Action) else Action(np.asarray(action))
        orders = np.asarray(action.orders, dtype=int)
        if orders.shape == (self.slot_count,):
            orders = np.concatenate((orders, np.array([0], dtype=int)))
        elif orders.shape != (self.slot_count + 1,):
            raise ValueError(
                f"action must have shape {(self.slot_count,)} or "
                f"{(self.slot_count + 1,)}"
            )

        if self._index >= len(self.dataset) - 1:
            frame = self._current_frame()
            if self._cached_index == self._index and self._cached_observation is not None:
                observation = self._cached_observation
                info = self._cached_info.copy()
            else:
                slots = self._slots(frame)
                observation, info = self._observation(frame, slots)
                self._cache_state(observation, info, slots)
            return observation, 0.0, False, True, {
                **info,
                "pnl": 0.0,
                "fees": 0.0,
                "trade_notional": 0.0,
                "invalid_action_count": 0,
                "executions": [],
                "option_settlements": [],
                "greek_exposures": {
                    name: float(observation.portfolio[3 + index])
                    for index, name in enumerate(GREEK_NAMES)
                },
                "path_risk": {
                    "current_drawdown": (
                        (self._peak_nav - float(observation.portfolio[2]))
                        / self._peak_nav
                        if self._peak_nav > 0
                        else 1.0
                    ),
                    "maximum_drawdown": self._max_drawdown,
                    "drawdown_increase": 0.0,
                    "downside_return": 0.0,
                },
                "reward_components": {
                    "gross_pnl_return": 0.0,
                    "fees": 0.0,
                    "invalid_action": 0.0,
                    "drawdown": 0.0,
                    "downside": 0.0,
                },
            }

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
        running_exposures = current_observation.portfolio[
            PORTFOLIO_GREEK_SLICE
        ].copy()
        invalid_actions = 0
        executions: list[dict[str, Any]] = []
        fees = 0.0
        underlying_encoded = int(orders[-1])
        if underlying_encoded:
            underlying_slot = self.slot_count
            if (
                underlying_encoded < 0
                or underlying_encoded >= self.action_shape[1]
                or not current_observation.action_mask[
                    underlying_slot,
                    underlying_encoded,
                ]
            ):
                invalid_actions += 1
            else:
                signed_quantity = int(
                    self.underlying_action_quantities[underlying_encoded]
                )
                greek_change = np.array(
                    [signed_quantity, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                )
                if not self._risk_allowed(running_exposures, greek_change):
                    invalid_actions += 1
                else:
                    side = "buy" if signed_quantity > 0 else "sell"
                    quantity = abs(signed_quantity)
                    spot = float(current_frame["underlyingPrice"].iloc[0])
                    price = self._underlying_execution_price(spot, side)
                    fee = quantity * self.underlying_commission_per_share
                    if not self._underlying_fill_allowed(
                        signed_quantity,
                        price,
                        fee,
                    ):
                        invalid_actions += 1
                    else:
                        cash_change = signed_quantity * price + fee
                        self._cash -= cash_change
                        self._underlying_shares += signed_quantity
                        fees += fee
                        running_exposures += greek_change
                        executions.append({
                            "instrument": "underlying",
                            "side": side,
                            "contract_symbol": self.dataset.symbol,
                            "quantity": quantity,
                            "price": price,
                            "fee": fee,
                            "multiplier": 1,
                        })

        for slot, encoded in enumerate(orders[:self.slot_count]):
            if encoded == 0:
                continue
            if encoded < 0 or encoded > 2 * self.max_quantity:
                invalid_actions += 1
                continue
            if not current_observation.action_mask[slot, encoded]:
                invalid_actions += 1
                continue
            contract = current_slots[slot]
            if contract is None:  # Defensive: the action mask must forbid this.
                invalid_actions += 1
                continue
            quantity = encoded if encoded <= self.max_quantity else encoded - self.max_quantity
            side = "buy" if encoded <= self.max_quantity else "sell"
            greek_change = (
                self._contract_greeks(contract)
                * quantity
                * OPTION_CONTRACT_MULTIPLIER
            )
            if side == "sell":
                greek_change = -greek_change
            if not self._risk_allowed(running_exposures, greek_change):
                invalid_actions += 1
                continue
            price = self._execution_price(contract, side)
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
            executions.append({
                "instrument": "option",
                "side": side,
                "contract_symbol": contract["contractSymbol"],
                "quantity": quantity,
                "price": price,
                "fee": fee,
                "multiplier": OPTION_CONTRACT_MULTIPLIER,
            })

        next_index = self._index + 1
        if next_index < len(self.dataset):
            self._index = next_index
        truncated = self._index >= len(self.dataset) - 1
        next_frame = self._current_frame()
        option_settlements = self._settle_expired_positions(next_frame)
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
        net_pnl_return = gross_pnl_return + fee_return
        self._peak_nav = max(self._peak_nav, post_nav)
        current_drawdown = (
            (self._peak_nav - post_nav) / self._peak_nav
            if self._peak_nav > 0
            else 1.0
        )
        next_max_drawdown = max(self._max_drawdown, current_drawdown)
        drawdown_increase = next_max_drawdown - self._max_drawdown
        self._max_drawdown = next_max_drawdown
        drawdown_return = (
            -self.reward_drawdown_penalty * drawdown_increase
        )
        downside_return = (
            -self.reward_downside_penalty * max(-net_pnl_return, 0.0)
        )
        reward = (
            net_pnl_return
            + invalid_return
            + drawdown_return
            + downside_return
        )
        terminated = post_nav <= 0
        info = {
            **pre_info,
            **next_info,
            "pnl": pnl,
            "fees": fees,
            "trade_notional": sum(
                item["price"] * item["quantity"] * item["multiplier"]
                for item in executions
            ),
            "invalid_action_count": invalid_actions,
            "executions": executions,
            "option_settlements": option_settlements,
            "greek_exposures": {
                name: float(next_observation.portfolio[3 + index])
                for index, name in enumerate(GREEK_NAMES)
            },
            "path_risk": {
                "current_drawdown": current_drawdown,
                "maximum_drawdown": self._max_drawdown,
                "drawdown_increase": drawdown_increase,
                "downside_return": max(-net_pnl_return, 0.0),
            },
            "reward_components": {
                "gross_pnl_return": gross_pnl_return,
                "fees": fee_return,
                "invalid_action": invalid_return,
                "drawdown": drawdown_return,
                "downside": downside_return,
            },
        }
        return next_observation, float(reward), terminated, truncated, info

    def _cache_state(
        self,
        observation: Observation,
        info: dict[str, Any],
        slots: list[_ContractRow | None],
    ) -> None:
        """Cache exactly the slot state returned to the policy."""
        self._cached_index = self._index
        self._cached_observation = observation
        self._cached_info = info.copy()
        self._cached_slots = slots
        for index, contract_id in enumerate(observation.contract_ids):
            if contract_id is not None:
                self._contract_home_slots.setdefault(contract_id, index)

    def _current_frame(self) -> pd.DataFrame:
        return self.dataset.snapshots[self._index].frame

    def _ranked_slots(
        self,
        frame: pd.DataFrame,
        rows: _FrameRows | None = None,
    ) -> list[_ContractRow]:
        rows = _FrameRows(frame) if rows is None else rows
        ranked = frame.drop_duplicates(
            "contractSymbol",
            keep="first",
        ).copy()
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
        return [
            row
            for symbol in selected["contractSymbol"].head(self.slot_count)
            if (row := rows.get(str(symbol))) is not None
        ]

    def _slots(self, frame: pd.DataFrame) -> list[_ContractRow | None]:
        """Assign ranked contracts while preserving prior per-slot identity."""
        rows = _FrameRows(frame)
        if self.slot_assignment == "ranked" or self._cached_observation is None:
            return self._ranked_slots(frame, rows)

        current_rows = rows.positions
        assigned: list[_ContractRow | None] = [None] * self.slot_count
        used: set[str] = set()
        previous_contract_ids = (
            self._cached_observation.contract_ids[:self.slot_count]
        )
        # A currently visible held contract keeps its immediately prior slot,
        # even if another held contract later reappears with the same old home.
        for index, contract_id in enumerate(previous_contract_ids):
            if (
                contract_id in self._positions
                and contract_id in current_rows
            ):
                assigned[index] = rows.get(contract_id)
                used.add(contract_id)
        # Reappearing held contracts reclaim their original home when it is
        # vacant, otherwise the first vacancy. They must remain sellable.
        for contract_id in sorted(self._positions):
            if contract_id in used or contract_id not in current_rows:
                continue
            home = self._contract_home_slots.get(contract_id)
            if (
                home is not None
                and 0 <= home < self.slot_count
                and assigned[home] is None
            ):
                target = home
            else:
                target = next(
                    (
                        index
                        for index, row in enumerate(assigned)
                        if row is None
                    ),
                    None,
                )
            if target is None:
                continue
            assigned[target] = rows.get(contract_id)
            used.add(contract_id)
        for index, contract_id in enumerate(previous_contract_ids):
            if (
                assigned[index] is not None
                or contract_id is None
                or contract_id in used
                or contract_id not in current_rows
            ):
                continue
            assigned[index] = rows.get(contract_id)
            used.add(contract_id)

        if (
            all(row is not None for row in assigned)
            or len(used) >= len(current_rows)
        ):
            return assigned

        candidates = self._ranked_slots(frame, rows)
        vacant = iter(index for index, row in enumerate(assigned) if row is None)
        for candidate in candidates:
            contract_id = str(candidate["contractSymbol"])
            if contract_id in used:
                continue
            try:
                index = next(vacant)
            except StopIteration:
                break
            assigned[index] = candidate
            used.add(contract_id)
        return assigned

    def _observation(
        self,
        frame: pd.DataFrame | None = None,
        slots: list[_ContractRow | None] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        frame = self._current_frame() if frame is None else frame
        slots = self._slots(frame) if slots is None else slots
        contracts = np.zeros((self.slot_count, len(CONTRACT_FEATURES)), dtype=np.float64)
        valid = np.zeros(self.slot_count, dtype=bool)
        ids: list[str | None] = [None] * self.slot_count
        previous_ids = (
            self._cached_observation.contract_ids
            if self._cached_observation is not None
            else None
        )
        for index, contract in enumerate(slots):
            if contract is None:
                continue
            ids[index] = str(contract["contractSymbol"])
            valid[index] = self._quote_valid(contract)
            position = self._positions.get(ids[index])
            average_price = position.average_price if position is not None else 0.0
            for feature_index, name in enumerate(CONTRACT_FEATURES):
                if name == "slotContinuity":
                    contracts[index, feature_index] = float(
                        previous_ids is not None
                        and previous_ids[index] == ids[index]
                    )
                elif name == "positionQuantity":
                    contracts[index, feature_index] = float(
                        position.quantity if position is not None else 0
                    )
                elif name == "positionAveragePrice":
                    contracts[index, feature_index] = float(average_price)
                elif name == "positionUnrealizedReturn":
                    liquidation_price = (
                        self._execution_price(
                            contract,
                            "sell" if position.quantity > 0 else "buy",
                        )
                        if (
                            valid[index]
                            and position is not None
                            and position.quantity != 0
                            and average_price > 0
                        )
                        else 0.0
                    )
                    contracts[index, feature_index] = (
                        np.sign(position.quantity)
                        * (liquidation_price / average_price - 1.0)
                        if (
                            position is not None
                            and average_price > 0
                            and liquidation_price > 0
                        )
                        else 0.0
                    )
                elif name == "positionAgeSteps":
                    contracts[index, feature_index] = float(
                        max(self._index - position.opened_index, 0)
                        if position is not None
                        else 0
                    )
                elif name == "positionLastTradeAgeSteps":
                    contracts[index, feature_index] = float(
                        max(self._index - position.last_trade_index, 0)
                        if position is not None
                        else 0
                    )
                else:
                    contracts[index, feature_index] = float(
                        contract.get(name, 0) or 0
                    )
        frame_rows = next(
            (contract.frame for contract in slots if contract is not None),
            None,
        )
        if frame_rows is None:
            frame_rows = _FrameRows(frame)
        nav, exposures = self._portfolio_metrics(frame, frame_rows)
        action_mask = np.zeros(self.action_shape, dtype=bool)
        for index, contract in enumerate(slots):
            if contract is None:
                continue
            if not valid[index]:
                continue
            action_mask[index, 0] = True
            ask = self._execution_price(contract, "buy")
            bid = self._execution_price(contract, "sell")
            contract_greeks = self._contract_greeks(contract)
            held = self._positions.get(
                ids[index] or "",
                Position(0, 0.0),
            ).quantity
            feasible_buys = 0
            feasible_sells = 0
            for quantity in range(1, self.max_quantity + 1):
                fee = quantity * self.commission_per_contract
                greek_change = (
                    contract_greeks
                    * quantity
                    * OPTION_CONTRACT_MULTIPLIER
                )
                if self.allow_collateralized_option_shorts:
                    buy_allowed = self._option_fill_allowed(
                        "buy", contract, quantity, ask, fee
                    )
                    sell_allowed = self._option_fill_allowed(
                        "sell", contract, quantity, bid, fee
                    )
                else:
                    buy_allowed = (
                        self._cash
                        >= ask
                        * quantity
                        * OPTION_CONTRACT_MULTIPLIER
                        + fee
                    )
                    sell_allowed = held >= quantity
                buy_feasible = (
                    buy_allowed
                    and self._risk_allowed(exposures, greek_change)
                )
                sell_feasible = (
                    bid > 0
                    and sell_allowed
                    and self._risk_allowed(exposures, -greek_change)
                )
                action_mask[index, quantity] = buy_feasible
                action_mask[index, self.max_quantity + quantity] = (
                    sell_feasible
                )
                feasible_buys += int(buy_feasible)
                feasible_sells += int(sell_feasible)
            contracts[
                index,
                BUY_FEASIBILITY_INDEX,
            ] = feasible_buys / self.max_quantity
            contracts[
                index,
                SELL_FEASIBILITY_INDEX,
            ] = feasible_sells / self.max_quantity
        underlying_slot = self.slot_count
        action_mask[underlying_slot, 0] = True
        spot = float(frame_rows.columns["underlyingPrice"][0])
        for encoded, signed_quantity in enumerate(
            self.underlying_action_quantities[1:],
            start=1,
        ):
            projected_position = self._underlying_shares + int(
                signed_quantity
            )
            side = "buy" if signed_quantity > 0 else "sell"
            price = self._underlying_execution_price(spot, side)
            fee = abs(signed_quantity) * self.underlying_commission_per_share
            greek_change = np.array(
                [signed_quantity, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            if self.allow_collateralized_option_shorts:
                fill_allowed = self._underlying_fill_allowed(
                    int(signed_quantity), price, fee
                )
            else:
                fill_allowed = (
                    abs(projected_position)
                    <= self.max_abs_underlying_shares
                    and (
                        signed_quantity < 0
                        or self._cash
                        >= signed_quantity * price + fee
                    )
                )
            action_mask[underlying_slot, encoded] = (
                price > 0
                and fill_allowed
                and self._risk_allowed(exposures, greek_change)
            )
        underlying_non_hold = self.underlying_action_quantities[1:]
        underlying_non_hold_mask = action_mask[underlying_slot, 1:]
        positive_underlying = underlying_non_hold > 0
        negative_underlying = underlying_non_hold < 0
        underlying_buy_feasible = float(
            underlying_non_hold_mask[positive_underlying].mean()
            if positive_underlying.any()
            else 0.0
        )
        underlying_sell_feasible = float(
            underlying_non_hold_mask[negative_underlying].mean()
            if negative_underlying.any()
            else 0.0
        )
        first = frame_rows.at(0)
        market = np.array(
            [float(first.get(name, 0.0) or 0.0) for name in MARKET_FEATURES],
            dtype=np.float64,
        )
        invested = sum(
            abs(position.quantity)
            * position.average_price
            * OPTION_CONTRACT_MULTIPLIER
            for position in self._positions.values()
        )
        reserved_cash, reserved_shares, _ = self._collateral_requirements()
        portfolio = np.concatenate((
            np.array([self._cash, invested, nav], dtype=np.float64),
            exposures,
            np.array([
                self._underlying_shares,
                reserved_cash,
                reserved_shares,
                underlying_buy_feasible,
                underlying_sell_feasible,
            ], dtype=np.float64),
        ))
        observation = Observation(
            self.dataset.snapshots[self._index].timestamp,
            market,
            contracts,
            portfolio,
            valid,
            action_mask,
            tuple(ids),
            underlying_action_quantities=self.underlying_action_quantities,
        )
        info = {
            "index": self._index,
            "data_source": self.manifest.data_source,
            "portfolio_features": PORTFOLIO_FEATURES,
            "market_features": MARKET_FEATURES,
            "risk_limits": self.risk_limits.copy(),
            "reward_objective": {
                "invalid_action_penalty": self.invalid_action_penalty,
                "drawdown_penalty": self.reward_drawdown_penalty,
                "downside_penalty": self.reward_downside_penalty,
            },
            "portfolio_valuation": self.portfolio_valuation,
            "allow_collateralized_option_shorts": (
                self.allow_collateralized_option_shorts
            ),
            "collateral": {
                "reserved_cash": reserved_cash,
                "reserved_covered_shares": reserved_shares,
                "available_cash": self._cash - reserved_cash,
            },
        }
        if previous_ids is None:
            info.update({
                "slot_identity_status": "no_prior_snapshot",
                "slot_retained_count": 0,
                "slot_changed_count": 0,
                "slot_comparable_count": 0,
                "slot_churn_rate": None,
            })
        else:
            comparable = sum(
                previous is not None or current is not None
                for previous, current in zip(previous_ids, ids, strict=True)
            )
            retained = sum(
                previous is not None and previous == current
                for previous, current in zip(previous_ids, ids, strict=True)
            )
            changed = sum(
                previous != current
                for previous, current in zip(previous_ids, ids, strict=True)
                if previous is not None or current is not None
            )
            info.update({
                "slot_identity_status": self.slot_assignment,
                "slot_retained_count": retained,
                "slot_changed_count": changed,
                "slot_comparable_count": comparable,
                "slot_churn_rate": changed / comparable if comparable else 0.0,
            })
        return observation, info

    def _execution_price(self, contract: _ContractRow, side: str) -> float:
        """Return a deterministic fill with configurable spread stress."""
        bid = float(contract["bid"])
        ask = float(contract["ask"])
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if self.spread_multiplier == 1.0:
            return ask if side == "buy" else bid
        midpoint = (bid + ask) / 2
        half_spread = max(ask - bid, 0.0) / 2
        if side == "buy":
            return midpoint + self.spread_multiplier * half_spread
        if side == "sell":
            return max(midpoint - self.spread_multiplier * half_spread, 0.0)
        raise AssertionError("unreachable side")

    def _underlying_execution_price(self, spot: float, side: str) -> float:
        """Apply the explicit synthetic spread assumed for underlying fills."""
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        slippage = self.underlying_slippage_bps / 10_000
        return spot * (1 + slippage if side == "buy" else 1 - slippage)

    @staticmethod
    def _contract_greeks(contract: _ContractRow) -> np.ndarray:
        values = np.empty(len(GREEK_NAMES), dtype=np.float64)
        for index, name in enumerate(GREEK_NAMES):
            try:
                value = float(contract.get(name, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            values[index] = value if math.isfinite(value) else 0.0
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
    def _position_metadata(
        contract: _ContractRow,
    ) -> tuple[str, float, str]:
        option_type = str(contract.get("optionType", "")).lower()
        try:
            strike = float(contract.get("strike", 0.0))
        except (TypeError, ValueError):
            strike = 0.0
        expiration = str(contract.get("expiration", ""))
        return option_type, strike, expiration

    @staticmethod
    def _valid_expiration(expiration: str) -> bool:
        try:
            parsed = pd.Timestamp(expiration)
        except (TypeError, ValueError):
            return False
        return not pd.isna(parsed)

    def _collateral_requirements(
        self,
        *,
        override_symbol: str | None = None,
        override_quantity: int | None = None,
        override_contract: _ContractRow | None = None,
    ) -> tuple[float, int, int]:
        """Return cash, covered shares, and put-assignment shares reserved."""
        cash = 0.0
        covered_shares = 0
        put_assignment_shares = 0
        symbols = set(self._positions)
        if override_symbol is not None:
            symbols.add(override_symbol)
        for symbol in symbols:
            position = self._positions.get(symbol)
            quantity = (
                override_quantity
                if symbol == override_symbol and override_quantity is not None
                else position.quantity if position is not None else 0
            )
            if quantity >= 0:
                continue
            if symbol == override_symbol and override_contract is not None:
                option_type, strike, expiration = self._position_metadata(
                    override_contract
                )
            elif position is not None:
                option_type = position.option_type
                strike = position.strike
                expiration = position.expiration
            else:
                return math.inf, 0, 0
            if (
                option_type not in {"call", "put"}
                or not math.isfinite(strike)
                or strike <= 0
                or not self._valid_expiration(expiration)
            ):
                return math.inf, 0, 0
            contracts = abs(quantity)
            if option_type == "put":
                cash += strike * contracts * OPTION_CONTRACT_MULTIPLIER
                put_assignment_shares += (
                    contracts * OPTION_CONTRACT_MULTIPLIER
                )
            else:
                covered_shares += contracts * OPTION_CONTRACT_MULTIPLIER
        return cash, covered_shares, put_assignment_shares

    def _option_fill_allowed(
        self,
        side: str,
        contract: _ContractRow,
        quantity: int,
        price: float,
        fee: float,
    ) -> bool:
        symbol = str(contract["contractSymbol"])
        current_quantity = self._positions.get(
            symbol,
            Position(0, 0.0),
        ).quantity
        signed_quantity = quantity if side == "buy" else -quantity
        projected_quantity = current_quantity + signed_quantity
        if projected_quantity < 0 and not self.allow_collateralized_option_shorts:
            return False
        cash_delta = (
            -price * quantity * OPTION_CONTRACT_MULTIPLIER - fee
            if side == "buy"
            else price * quantity * OPTION_CONTRACT_MULTIPLIER - fee
        )
        projected_cash = self._cash + cash_delta
        if projected_cash < -1e-12:
            return False
        cash_collateral, covered_shares, put_assignment_shares = (
            self._collateral_requirements(
                override_symbol=symbol,
                override_quantity=projected_quantity,
                override_contract=contract,
            )
        )
        return (
            projected_cash + 1e-12 >= cash_collateral
            and (
                covered_shares == 0
                or self._underlying_shares >= covered_shares
            )
            and abs(self._underlying_shares + put_assignment_shares)
            <= self.max_abs_underlying_shares
        )

    def _underlying_fill_allowed(
        self,
        signed_quantity: int,
        price: float,
        fee: float,
    ) -> bool:
        projected_shares = self._underlying_shares + signed_quantity
        if abs(projected_shares) > self.max_abs_underlying_shares:
            return False
        projected_cash = self._cash - signed_quantity * price - fee
        cash_collateral, covered_shares, put_assignment_shares = (
            self._collateral_requirements()
        )
        return (
            projected_cash + 1e-12 >= cash_collateral
            and (
                covered_shares == 0
                or projected_shares >= covered_shares
            )
            and abs(projected_shares + put_assignment_shares)
            <= self.max_abs_underlying_shares
        )

    @staticmethod
    def _bid_ask_valid(contract: _ContractRow) -> bool:
        try:
            bid = float(contract["bid"])
            ask = float(contract["ask"])
            return (
                math.isfinite(bid)
                and math.isfinite(ask)
                and bid > 0
                and ask > 0
                and bid <= ask
            )
        except (TypeError, ValueError):
            return False

    @classmethod
    def _quote_valid(cls, contract: _ContractRow) -> bool:
        """Return whether the saved top of book supports an executable fill."""
        return cls._bid_ask_valid(contract)

    def _fill(self, side: str, contract: _ContractRow, quantity: int, price: float, fee: float) -> None:
        symbol = str(contract["contractSymbol"])
        greeks = tuple(float(value) for value in self._contract_greeks(contract))
        if not self._option_fill_allowed(side, contract, quantity, price, fee):
            raise ValueError("option fill violates cash, position, or collateral")
        signed_quantity = quantity if side == "buy" else -quantity
        cash_delta = (
            -price * quantity * OPTION_CONTRACT_MULTIPLIER - fee
            if side == "buy"
            else price * quantity * OPTION_CONTRACT_MULTIPLIER - fee
        )
        self._cash += cash_delta
        position = self._positions.get(symbol)
        old_quantity = position.quantity if position is not None else 0
        new_quantity = old_quantity + signed_quantity
        if new_quantity == 0:
            self._positions.pop(symbol, None)
            return
        if old_quantity == 0 or (old_quantity > 0) == (signed_quantity > 0):
            old_cost = (
                abs(old_quantity) * position.average_price
                if position is not None
                else 0.0
            )
            average_price = (
                old_cost + abs(signed_quantity) * price
            ) / abs(new_quantity)
        elif (new_quantity > 0) == (old_quantity > 0):
            average_price = position.average_price
        else:
            average_price = price
        option_type, strike, expiration = self._position_metadata(contract)
        same_position_lifecycle = (
            position is not None
            and old_quantity != 0
            and (new_quantity > 0) == (old_quantity > 0)
        )
        self._positions[symbol] = Position(
            new_quantity,
            average_price,
            greeks,
            option_type,
            strike,
            expiration,
            self._execution_price(
                contract,
                "sell" if new_quantity > 0 else "buy",
            ),
            (
                position.opened_index
                if same_position_lifecycle
                else self._index
            ),
            self._index,
        )

    def _settle_expired_positions(
        self,
        frame: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Settle positions at the first observed date after expiration."""
        if not self._positions:
            return []
        timestamp = frame.iloc[0].get(
            "collectedAt",
            self.dataset.snapshots[self._index].timestamp,
        )
        parsed_timestamp = pd.to_datetime(
            timestamp,
            utc=True,
            errors="coerce",
        )
        if pd.isna(parsed_timestamp):
            return []
        snapshot_date = parsed_timestamp.date()
        spot = float(frame["underlyingPrice"].iloc[0])
        settlements: list[dict[str, Any]] = []
        for symbol, position in tuple(self._positions.items()):
            try:
                expiration_date = pd.Timestamp(position.expiration).date()
            except (TypeError, ValueError):
                continue
            if snapshot_date <= expiration_date:
                continue
            intrinsic = (
                max(spot - position.strike, 0.0)
                if position.option_type == "call"
                else max(position.strike - spot, 0.0)
                if position.option_type == "put"
                else 0.0
            )
            quantity = position.quantity
            settlement_style = "expired_worthless"
            if quantity < 0 and intrinsic > 0:
                contracts = abs(quantity)
                settlement_style = "physical_assignment"
                if position.option_type == "call":
                    self._underlying_shares -= (
                        contracts * OPTION_CONTRACT_MULTIPLIER
                    )
                    self._cash += (
                        position.strike
                        * contracts
                        * OPTION_CONTRACT_MULTIPLIER
                    )
                elif position.option_type == "put":
                    self._underlying_shares += (
                        contracts * OPTION_CONTRACT_MULTIPLIER
                    )
                    self._cash -= (
                        position.strike
                        * contracts
                        * OPTION_CONTRACT_MULTIPLIER
                    )
            elif quantity > 0 and intrinsic > 0:
                settlement_style = "cash_intrinsic"
                self._cash += (
                    intrinsic * quantity * OPTION_CONTRACT_MULTIPLIER
                )
            settlements.append({
                "instrument": "option",
                "contract_symbol": symbol,
                "position_quantity": quantity,
                "option_type": position.option_type,
                "strike": position.strike,
                "spot": spot,
                "intrinsic_value": intrinsic,
                "style": settlement_style,
            })
            del self._positions[symbol]
        return settlements

    def _portfolio_metrics(
        self,
        frame: pd.DataFrame,
        rows: _FrameRows | None = None,
    ) -> tuple[float, np.ndarray]:
        rows = _FrameRows(frame) if rows is None else rows
        spot = float(rows.columns["underlyingPrice"][0])
        if self.portfolio_valuation == "liquidation" and self._underlying_shares:
            exit_side = "sell" if self._underlying_shares > 0 else "buy"
            value = (
                self._underlying_shares
                * self._underlying_execution_price(spot, exit_side)
                - abs(self._underlying_shares)
                * self.underlying_commission_per_share
            )
        else:
            value = self._underlying_shares * spot
        exposures = np.zeros(len(GREEK_NAMES), dtype=np.float64)
        exposures[0] = self._underlying_shares
        for symbol, position in self._positions.items():
            quote = rows.get(symbol)
            if quote is None:
                mark = (
                    position.last_liquidation_price
                    if (
                        self.portfolio_valuation == "liquidation"
                        and position.last_liquidation_price > 0
                    )
                    else position.average_price
                )
                value += (
                    position.quantity
                    * mark
                    * OPTION_CONTRACT_MULTIPLIER
                )
                if self.portfolio_valuation == "liquidation":
                    value -= (
                        abs(position.quantity)
                        * self.commission_per_contract
                    )
                exposures += (
                    np.asarray(position.greeks)
                    * position.quantity
                    * OPTION_CONTRACT_MULTIPLIER
                )
                continue
            bid, ask = float(quote["bid"]), float(quote["ask"])
            if self.portfolio_valuation == "liquidation":
                if self._bid_ask_valid(quote):
                    mark = self._execution_price(
                        quote,
                        "sell" if position.quantity > 0 else "buy",
                    )
                    position.last_liquidation_price = mark
                else:
                    mark = (
                        position.last_liquidation_price
                        if position.last_liquidation_price > 0
                        else position.average_price
                    )
            else:
                mark = (
                    (bid + ask) / 2
                    if bid > 0 and ask > 0
                    else float(quote["lastPrice"])
                )
            value += (
                position.quantity * mark * OPTION_CONTRACT_MULTIPLIER
            )
            if self.portfolio_valuation == "liquidation":
                value -= (
                    abs(position.quantity) * self.commission_per_contract
                )
            exposures += (
                self._contract_greeks(quote)
                * position.quantity
                * OPTION_CONTRACT_MULTIPLIER
            )
        return self._cash + value, exposures

    def _nav(self, frame: pd.DataFrame) -> float:
        return self._portfolio_metrics(frame)[0]
