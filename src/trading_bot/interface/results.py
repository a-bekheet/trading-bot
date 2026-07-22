"""Read-only projections of walk-forward artifacts for the Streamlit UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


AGENT_LABELS = {
    "gru": "GRU Agent",
    "lstm": "LSTM Agent",
    "hybrid": "GRU + LSTM Agent",
    "mixture": "Gated Mixture Agent",
}


def discover_agent_runs(data_dir: Path) -> list[dict[str, Any]]:
    """Load valid walk-forward summaries, newest first."""
    patterns = (
        "agent_runs/**/*-walk-forward.json",
        "models/walk-forward/**/*-walk-forward.json",
    )
    paths = {
        path.resolve()
        for pattern in patterns
        for path in data_dir.glob(pattern)
        if path.is_file()
    }
    runs = []
    for path in paths:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _is_walk_forward_summary(summary):
            continue
        summary["_artifact_path"] = str(path)
        summary["_modified_at"] = path.stat().st_mtime
        summary["_run_name"] = path.parent.name
        runs.append(summary)
    return sorted(runs, key=lambda run: run["_modified_at"], reverse=True)


def agent_leaderboard(summary: dict[str, Any]) -> pd.DataFrame:
    """Aggregate validation-only candidate evidence across folds."""
    records: list[dict[str, Any]] = []
    for fold in summary.get("folds", []):
        selection = fold.get("model_selection", {})
        selected = selection.get("selected_model_id")
        for candidate in selection.get("candidates", []):
            model = candidate.get("model", {})
            kind = str(model.get("kind", "unknown"))
            latency = candidate.get("inference_latency", {})
            candidate_selection = candidate.get("selection", {})
            records.append({
                "model_id": candidate.get("model_id", "unknown"),
                "Agent": AGENT_LABELS.get(kind, kind.upper()),
                "Architecture": _architecture_label(model),
                "Algorithm": str(model.get("algorithm", "unknown")).upper(),
                "Validation score": _number(candidate_selection.get(
                    "robust_training_seed_validation_score"
                )),
                "Validation reward": _number(candidate_selection.get(
                    "training_seed_mean_validation_reward",
                    candidate_selection.get("validation_total_reward"),
                )),
                "Median latency (us)": _number(
                    latency.get("median_microseconds")
                ),
                "Parameters": _number(candidate.get("parameter_count")),
                "Episodes": _number(candidate.get("episodes_completed")),
                "Selected folds": int(candidate.get("model_id") == selected),
            })
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    stable = frame[["model_id", "Agent", "Architecture", "Algorithm"]]
    stable = stable.drop_duplicates("model_id").set_index("model_id")
    numeric = frame.groupby("model_id", sort=False).agg({
        "Validation score": "mean",
        "Validation reward": "mean",
        "Median latency (us)": "mean",
        "Parameters": "mean",
        "Episodes": "sum",
        "Selected folds": "sum",
    })
    result = stable.join(numeric).reset_index(drop=True)
    return result.sort_values(
        ["Validation score", "Median latency (us)"],
        ascending=[False, True],
    ).reset_index(drop=True)


def heldout_results(summary: dict[str, Any]) -> pd.DataFrame:
    """Return one row per genuinely held-out selected-agent path."""
    records = []
    for fold in summary.get("folds", []):
        winner = _selected_candidate(fold)
        label = _agent_label(winner)
        for report in fold.get("test", []):
            records.append({
                "Fold": int(fold.get("fold", len(records))),
                "Agent": label,
                "Test return": _number(report.get("total_return")),
                "Final NAV": _number(report.get("final_nav")),
                "Max drawdown": _number(report.get("max_drawdown")),
                "Executions": int(report.get("executions", 0)),
                "Turnover": _number(report.get("turnover")),
                "Fees": _number(report.get("fees")),
                "Step Sharpe": _number(report.get("step_sharpe")),
                "Market beta": _number(
                    report.get("return_beta_to_underlying")
                ),
                "Mean |Delta notional|": _number(
                    report.get("mean_abs_delta_notional_weight")
                ),
                "Steps": int(report.get("steps", 0)),
            })
    return pd.DataFrame(records)


def arena_overview(runs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Summarize the newest discovered run for each ticker."""
    newest_by_symbol: dict[str, dict[str, Any]] = {}
    for run in runs:
        symbol = str(run.get("symbol", "")).upper()
        if symbol and symbol not in newest_by_symbol:
            newest_by_symbol[symbol] = run
    records = []
    for symbol, run in newest_by_symbol.items():
        heldout = heldout_results(run)
        if heldout.empty:
            continue
        provenance = {
            fold.get("test_data_quality", {}).get(
                "execution_provenance", "unknown"
            )
            for fold in run.get("folds", [])
        }
        agents = sorted(set(heldout["Agent"]))
        records.append({
            "Ticker": symbol,
            "Selected agent": ", ".join(agents),
            "Held-out return": float(heldout["Test return"].mean()),
            "Final NAV": float(heldout["Final NAV"].mean()),
            "Max drawdown": float(heldout["Max drawdown"].max()),
            "Executions": int(heldout["Executions"].sum()),
            "Fees": float(heldout["Fees"].sum()),
            "Steps": int(heldout["Steps"].sum()),
            "Evidence": evidence_summary(run)["grade"],
            "Execution provenance": ", ".join(sorted(provenance)),
            "Experiment": run.get("_run_name", "unknown"),
        })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("Ticker").reset_index(drop=True)


def equity_curve(
    summary: dict[str, Any],
    fold_number: int,
    baseline_names: Iterable[str] = ("no_op", "first_feasible"),
) -> pd.DataFrame:
    """Project stored NAV paths into a long chart-ready frame."""
    fold = _fold(summary, fold_number)
    traces = fold.get("heldout_traces", {})
    series = [("Selected agent", traces.get("agent", []))]
    baselines = traces.get("baselines", {})
    series.extend(
        (_humanize(name), baselines.get(name, []))
        for name in baseline_names
    )
    records = []
    for label, paths in series:
        for trace_index, trace in enumerate(paths):
            timestamps = trace.get("timestamps", [])
            navs = trace.get("navs", [])
            if len(navs) == len(timestamps) + 1:
                navs = navs[1:]
            for timestamp, nav in zip(timestamps, navs, strict=False):
                records.append({
                    "Timestamp": timestamp,
                    "Series": label,
                    "Equity": _number(nav),
                    "Path": trace_index,
                })
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
    return frame


def trade_ledger(summary: dict[str, Any], fold_number: int) -> pd.DataFrame:
    """Flatten the selected agent's stored held-out fills."""
    fold = _fold(summary, fold_number)
    traces = fold.get("heldout_traces", {}).get("agent", [])
    records = []
    for trace_index, trace in enumerate(traces):
        for decision_index, decision in enumerate(trace.get("decisions", [])):
            for execution in decision.get("executions", []):
                records.append({
                    "Timestamp": decision.get("decision_timestamp"),
                    "Path": trace_index,
                    "Decision": decision_index,
                    "Instrument": execution.get("instrument"),
                    "Side": execution.get("side"),
                    "Contract": execution.get("contract_symbol"),
                    "Quantity": execution.get("quantity"),
                    "Price": execution.get("price"),
                    "Fee": execution.get("fee"),
                    "Post-trade NAV": decision.get("nav"),
                })
    return pd.DataFrame(records)


def evidence_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Describe how much evidence the artifact actually contains."""
    heldout = heldout_results(summary)
    comparisons = [
        comparison
        for fold in summary.get("folds", [])
        for values in fold.get("statistical_comparisons", {}).values()
        for comparison in values
    ]
    successful = [item for item in comparisons if item.get("status") == "ok"]
    supporting = [item for item in successful if item.get("supports_improvement")]
    steps = int(heldout["Steps"].sum()) if not heldout.empty else 0
    grade = "Exploratory"
    if successful and steps >= 100:
        grade = "Statistically evaluated"
    return {
        "grade": grade,
        "folds": len(summary.get("folds", [])),
        "heldout_steps": steps,
        "successful_comparisons": len(successful),
        "supporting_comparisons": len(supporting),
        "can_claim_improvement": bool(supporting and grade != "Exploratory"),
    }


def _is_walk_forward_summary(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and str(value.get("schema_version", "")).startswith(
            "research-demo.walk-forward."
        )
        and isinstance(value.get("folds"), list)
    )


def _fold(summary: dict[str, Any], fold_number: int) -> dict[str, Any]:
    for fold in summary.get("folds", []):
        if int(fold.get("fold", -1)) == fold_number:
            return fold
    raise ValueError(f"fold {fold_number} is not present in the run")


def _selected_candidate(fold: dict[str, Any]) -> dict[str, Any]:
    selection = fold.get("model_selection", {})
    selected_id = selection.get("selected_model_id")
    return next(
        (
            candidate
            for candidate in selection.get("candidates", [])
            if candidate.get("model_id") == selected_id
        ),
        {},
    )


def _agent_label(candidate: dict[str, Any]) -> str:
    kind = str(candidate.get("model", {}).get("kind", "unknown"))
    return AGENT_LABELS.get(kind, kind.upper())


def _architecture_label(model: dict[str, Any]) -> str:
    encoder = _humanize(str(model.get("encoder", "unknown")))
    kind = _humanize(str(model.get("kind", "unknown")))
    return f"{encoder} / {kind}"


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
