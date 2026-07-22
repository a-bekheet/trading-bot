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
            seed_aggregate = candidate.get("training_seed_aggregate", {})
            records.append(
                {
                    "model_id": candidate.get("model_id", "unknown"),
                    "Agent": AGENT_LABELS.get(kind, kind.upper()),
                    "Architecture": _architecture_label(model),
                    "Feature set": _feature_set_label(model),
                    "Smile residual": _feature_status(
                        model, "contract_smile_residual"
                    ),
                    "Action policy": _decoder_label(model),
                    "Algorithm": str(model.get("algorithm", "unknown")).upper(),
                    "Validation score": _number(
                        candidate_selection.get("robust_training_seed_validation_score")
                    ),
                    "Validation reward": _number(
                        candidate_selection.get(
                            "training_seed_mean_validation_reward",
                            candidate_selection.get("validation_total_reward"),
                        )
                    ),
                    "Median latency (us)": _number(latency.get("median_microseconds")),
                    "Parameters": _number(candidate.get("parameter_count")),
                    "Score gap (bp)": 10_000.0
                    * _number(candidate.get("score_gap_to_best")),
                    "Competitive folds": int(
                        candidate.get("selection_competitive", True)
                    ),
                    "Training seeds": int(
                        seed_aggregate.get("training_seed_count", 1)
                    ),
                    "Episodes": _number(candidate.get("episodes_completed")),
                    "Selected folds": int(candidate.get("model_id") == selected),
                }
            )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    stable = frame[
        [
            "model_id",
            "Agent",
            "Architecture",
            "Feature set",
            "Smile residual",
            "Action policy",
            "Algorithm",
        ]
    ]
    stable = stable.drop_duplicates("model_id").set_index("model_id")
    numeric = frame.groupby("model_id", sort=False).agg(
        {
            "Validation score": "mean",
            "Validation reward": "mean",
            "Median latency (us)": "mean",
            "Parameters": "mean",
            "Score gap (bp)": "mean",
            "Competitive folds": "sum",
            "Training seeds": "sum",
            "Episodes": "sum",
            "Selected folds": "sum",
        }
    )
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
        model = winner.get("model", {})
        action_policy = _decoder_label(model)
        feature_set = _feature_set_label(model)
        smile_residual = _feature_status(model, "contract_smile_residual")
        selection_rule = fold.get("model_selection", {}).get(
            "simplicity_rule", {}
        )
        activation_gate = fold.get("model_selection", {}).get(
            "activation_gate", {}
        )
        activated = bool(activation_gate.get("activated", True))
        no_op_reports = fold.get("baselines", {}).get("no_op", [])
        selected_latency = _number(
            winner.get("inference_latency", {}).get("median_microseconds")
        )
        for report_index, report in enumerate(fold.get("test", [])):
            steps = int(report.get("steps", 0))
            executions = int(report.get("executions", 0))
            sandbox_report = report
            if not activated and report_index < len(no_op_reports):
                sandbox_report = no_op_reports[report_index]
            sandbox_executions = int(sandbox_report.get("executions", 0))
            research_return = _number(report.get("total_return"))
            sandbox_return = _number(sandbox_report.get("total_return"))
            records.append(
                {
                    "Fold": int(fold.get("fold", len(records))),
                    "Agent": label,
                    "Encoder": _humanize(str(model.get("encoder", "unknown"))),
                    "Architecture": _architecture_label(model),
                    "Feature set": feature_set,
                    "Smile residual": smile_residual,
                    "Action policy": action_policy,
                    "Selected latency (us)": selected_latency,
                    "Score traded (bp)": 10_000.0
                    * _number(
                        selection_rule.get(
                            "score_sacrificed_for_simplicity", 0.0
                        )
                    ),
                    "Competitive candidates": int(
                        selection_rule.get("competitive_candidate_count", 1)
                    ),
                    "Activation": "Active" if activated else "Abstain",
                    "Sandbox policy": label if activated else "No Op",
                    "Sandbox return": sandbox_return,
                    "Sandbox lift": sandbox_return - research_return,
                    "Sandbox executions": sandbox_executions,
                    "Sandbox fees": _number(sandbox_report.get("fees")),
                    "Sandbox latency (us)": (
                        selected_latency if activated else 0.0
                    ),
                    "Test return": research_return,
                    "Final NAV": _number(report.get("final_nav")),
                    "Max drawdown": _number(report.get("max_drawdown")),
                    "Executions": executions,
                    "Fills / decision": executions / steps if steps else 0.0,
                    "Turnover": _number(report.get("turnover")),
                    "Fees": _number(report.get("fees")),
                    "Invalid actions": int(report.get("invalid_actions", 0)),
                    "Step Sharpe": _number(report.get("step_sharpe")),
                    "Market beta": _number(report.get("return_beta_to_underlying")),
                    "Mean |Delta notional|": _number(
                        report.get("mean_abs_delta_notional_weight")
                    ),
                    "Steps": steps,
                }
            )
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
            fold.get("test_data_quality", {}).get("execution_provenance", "unknown")
            for fold in run.get("folds", [])
        }
        agents = sorted(set(heldout["Agent"]))
        encoders = sorted(set(heldout["Encoder"]))
        architectures = sorted(set(heldout["Architecture"]))
        feature_sets = sorted(set(heldout["Feature set"]))
        smile_residuals = sorted(set(heldout["Smile residual"]))
        action_policies = sorted(set(heldout["Action policy"]))
        activations = sorted(set(heldout["Activation"]))
        sandbox_policies = sorted(set(heldout["Sandbox policy"]))
        promotion = promotion_assessment(run)
        records.append(
            {
                "Ticker": symbol,
                "Selected agent": ", ".join(agents),
                "Selected encoder": ", ".join(encoders),
                "Architecture": ", ".join(architectures),
                "Feature set": ", ".join(feature_sets),
                "Smile residual": ", ".join(smile_residuals),
                "Action policy": ", ".join(action_policies),
                "Activation": ", ".join(activations),
                "Sandbox policy": ", ".join(sandbox_policies),
                "Sandbox return": float(heldout["Sandbox return"].mean()),
                "Sandbox lift": float(heldout["Sandbox lift"].mean()),
                "Sandbox executions": int(
                    heldout["Sandbox executions"].sum()
                ),
                "Sandbox fees": float(heldout["Sandbox fees"].sum()),
                "Sandbox latency (us)": float(
                    heldout["Sandbox latency (us)"].mean()
                ),
                "Selected latency (us)": float(
                    heldout["Selected latency (us)"].mean()
                ),
                "Score traded (bp)": float(
                    heldout["Score traded (bp)"].mean()
                ),
                "Competitive candidates": int(
                    heldout["Competitive candidates"].max()
                ),
                "Held-out return": float(heldout["Test return"].mean()),
                "Final NAV": float(heldout["Final NAV"].mean()),
                "Max drawdown": float(heldout["Max drawdown"].max()),
                "Executions": int(heldout["Executions"].sum()),
                "Fills / decision": (
                    float(heldout["Executions"].sum()) / float(heldout["Steps"].sum())
                    if heldout["Steps"].sum()
                    else 0.0
                ),
                "Fees": float(heldout["Fees"].sum()),
                "Steps": int(heldout["Steps"].sum()),
                "Evidence": evidence_summary(run)["grade"],
                "Excess vs no-op": promotion["mean_excess_vs_no_op"],
                "Double-cost return": promotion["worst_double_cost_return"],
                "Promotion": promotion["status"],
                "Execution provenance": ", ".join(sorted(provenance)),
                "Experiment": run.get("_run_name", "unknown"),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("Ticker").reset_index(drop=True)


def feature_ablation_results(
    runs: Sequence[dict[str, Any]],
    feature_group: str = "contract_smile_residual",
) -> pd.DataFrame:
    """Project newest per-ticker matched feature-ablation validation evidence."""
    newest_by_symbol: dict[str, dict[str, Any]] = {}
    for run in runs:
        symbol = str(run.get("symbol", "")).upper()
        if symbol and symbol not in newest_by_symbol:
            newest_by_symbol[symbol] = run
    records = []
    for symbol, run in newest_by_symbol.items():
        for fold in run.get("folds", []):
            for candidate in fold.get("model_selection", {}).get("candidates", []):
                model = candidate.get("model", {})
                disabled = tuple(model.get("disabled_feature_groups") or ())
                if feature_group not in disabled:
                    continue
                ablation_lift = _number(
                    candidate.get("validation_score_lift_vs_full")
                )
                if pd.isna(ablation_lift):
                    continue
                kind = str(model.get("kind", "unknown"))
                feature_lift_bp = -10_000.0 * ablation_lift
                records.append(
                    {
                        "Ticker": symbol,
                        "Fold": int(fold.get("fold", 0)),
                        "Encoder": _humanize(
                            str(model.get("encoder", "unknown"))
                        ),
                        "Agent": AGENT_LABELS.get(kind, kind.upper()),
                        "Feature": _humanize(feature_group),
                        "Feature lift (bp)": feature_lift_bp,
                        "Feature helped": "Yes" if feature_lift_bp > 0 else "No",
                        "Ablated latency (us)": _number(
                            candidate.get("inference_latency", {}).get(
                                "median_microseconds"
                            )
                        ),
                        "Ablated parameters": _number(
                            candidate.get("parameter_count")
                        ),
                        "Training seeds": int(
                            candidate.get("training_seed_aggregate", {}).get(
                                "training_seed_count", 1
                            )
                        ),
                    }
                )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        ["Ticker", "Encoder", "Agent"]
    ).reset_index(drop=True)


def promotion_assessment(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply conservative deployment gates to held-out research evidence."""
    agent_returns = []
    no_op_returns = []
    double_cost_returns = []
    comparisons = []
    provenances = []
    activation_gates = []
    invalid_actions = 0
    for fold in summary.get("folds", []):
        agent_reports = fold.get("test", [])
        no_op_reports = fold.get("baselines", {}).get("no_op", [])
        stressed_reports = fold.get("cost_stress", {}).get("double_costs", [])
        agent_returns.extend(
            _number(report.get("total_return")) for report in agent_reports
        )
        no_op_returns.extend(
            _number(report.get("total_return")) for report in no_op_reports
        )
        double_cost_returns.extend(
            _number(report.get("total_return")) for report in stressed_reports
        )
        invalid_actions += sum(
            int(report.get("invalid_actions", 0)) for report in agent_reports
        )
        comparisons.extend(fold.get("statistical_comparisons", {}).get("no_op", []))
        provenances.append(
            fold.get("test_data_quality", {}).get("execution_provenance", "unknown")
        )
        activation_gates.append(
            fold.get("model_selection", {}).get("activation_gate")
        )

    paired_count = min(len(agent_returns), len(no_op_returns))
    excess = [
        agent_returns[index] - no_op_returns[index] for index in range(paired_count)
    ]
    evidence = evidence_summary(summary)
    checks = {
        "positive_heldout_return": bool(agent_returns and min(agent_returns) > 0),
        "beats_no_op": bool(excess and min(excess) > 0),
        "statistical_support_vs_no_op": bool(
            comparisons
            and all(
                item.get("status") == "ok" and item.get("supports_improvement") is True
                for item in comparisons
            )
        ),
        "adequate_heldout_history": (evidence["grade"] == "Statistically evaluated"),
        "provider_confirmed_regular": bool(
            provenances
            and all(value == "provider_confirmed_regular" for value in provenances)
        ),
        "positive_under_double_costs": bool(
            double_cost_returns and min(double_cost_returns) > 0
        ),
        "no_invalid_actions": invalid_actions == 0,
        "validation_activation_passed": bool(
            activation_gates
            and all(
                isinstance(gate, dict) and gate.get("activated") is True
                for gate in activation_gates
            )
        ),
    }
    labels = {
        "positive_heldout_return": "held-out return is not positive",
        "beats_no_op": "agent does not beat no-op on every path",
        "statistical_support_vs_no_op": (
            "no statistically supported improvement over no-op"
        ),
        "adequate_heldout_history": "held-out history is too short",
        "provider_confirmed_regular": (
            "test data lacks fully confirmed regular-session provenance"
        ),
        "positive_under_double_costs": ("return is not positive under double costs"),
        "no_invalid_actions": "held-out path contains invalid actions",
        "validation_activation_passed": (
            "validation no-op activation gate did not pass"
        ),
    }
    failed = [labels[name] for name, passed in checks.items() if not passed]
    return {
        "status": "Promotion ready" if not failed else "Research only",
        "checks": checks,
        "failed_reasons": failed,
        "heldout_steps": evidence["heldout_steps"],
        "mean_heldout_return": (
            sum(agent_returns) / len(agent_returns) if agent_returns else float("nan")
        ),
        "mean_excess_vs_no_op": (sum(excess) / len(excess) if excess else float("nan")),
        "worst_double_cost_return": (
            min(double_cost_returns) if double_cost_returns else float("nan")
        ),
    }


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
    series.extend((_humanize(name), baselines.get(name, [])) for name in baseline_names)
    records = []
    for label, paths in series:
        for trace_index, trace in enumerate(paths):
            timestamps = trace.get("timestamps", [])
            navs = trace.get("navs", [])
            if len(navs) == len(timestamps) + 1:
                navs = navs[1:]
            for timestamp, nav in zip(timestamps, navs, strict=False):
                records.append(
                    {
                        "Timestamp": timestamp,
                        "Series": label,
                        "Equity": _number(nav),
                        "Path": trace_index,
                    }
                )
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
                records.append(
                    {
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
                    }
                )
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


def _feature_set_label(model: dict[str, Any]) -> str:
    disabled = tuple(model.get("disabled_feature_groups") or ())
    if not disabled:
        return "Full"
    return "Without " + ", ".join(_humanize(str(name)) for name in disabled)


def _feature_status(model: dict[str, Any], group: str) -> str:
    disabled = tuple(model.get("disabled_feature_groups") or ())
    return "Ablated" if group in disabled else "Enabled"


def _decoder_label(model: dict[str, Any]) -> str:
    decoder = str(model.get("action_decoder", "factorized"))
    if decoder == "single_leg":
        return "Sparse single-leg"
    if decoder == "factorized":
        return "Factorized multi-leg"
    return _humanize(decoder)


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
