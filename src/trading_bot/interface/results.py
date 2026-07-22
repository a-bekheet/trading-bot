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

ARENA_WATCH_STATUS_FILENAME = "_arena_watch_status.json"


def load_arena_watch_status(data_dir: Path) -> dict[str, Any] | None:
    """Load the latest training-automation heartbeat without ML imports."""
    path = data_dir / ARENA_WATCH_STATUS_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not str(
        payload.get("schema_version", "")
    ).startswith("research-demo.arena-watch.status."):
        return None
    return payload


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


def discover_agent_arena_manifests(data_dir: Path) -> list[dict[str, Any]]:
    """Load arena orchestration manifests, including readiness-only attempts."""
    paths = {
        path.resolve()
        for path in data_dir.glob("agent_runs/**/agent-arena.json")
        if path.is_file()
    }
    manifests = []
    for path in paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not str(manifest.get("schema_version", "")).startswith(
            "research-demo.agent-arena."
        ):
            continue
        manifest["_artifact_path"] = str(path)
        manifest["_modified_at"] = path.stat().st_mtime
        manifest["_run_name"] = path.parent.name
        manifests.append(manifest)
    return sorted(
        manifests,
        key=lambda manifest: manifest["_modified_at"],
        reverse=True,
    )


def arena_readiness_overview(manifest: dict[str, Any]) -> pd.DataFrame:
    """Project validation/test evidence readiness before expensive training."""
    records = []
    for item in manifest.get("preflight", []):
        validation = item.get("validation") or {}
        test = item.get("test") or {}
        records.append(
            {
                "Ticker": str(item.get("symbol", "unknown")).upper(),
                "Ready": "Yes" if item.get("ready") else "Waiting",
                "Reason": _humanize(str(item.get("reason", "unknown"))),
                "Validation snapshots": int(validation.get("snapshot_count", 0)),
                "Validation regular": int(validation.get("regular_snapshot_count", 0)),
                "Validation fresh": int(
                    validation.get("fresh_underlying_quote_count", 0)
                ),
                "Validation executable": int(
                    validation.get("executable_option_quote_count", 0)
                ),
                "Test snapshots": int(test.get("snapshot_count", 0)),
                "Test regular": int(test.get("regular_snapshot_count", 0)),
                "Test fresh": int(test.get("fresh_underlying_quote_count", 0)),
                "Test executable": int(test.get("executable_option_quote_count", 0)),
                "Test start": test.get("first_timestamp", "Unknown"),
                "Test end": test.get("last_timestamp", "Unknown"),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("Ticker").reset_index(drop=True)


def agent_decision_tape(
    summary: dict[str, Any],
    fold_number: int,
) -> pd.DataFrame:
    """Project every held-out policy decision, including explicit holds."""
    fold = _fold(summary, fold_number)
    winner = _selected_candidate(fold)
    model = winner.get("model", {})
    activation_gate = fold.get("model_selection", {}).get("activation_gate", {})
    activated = bool(activation_gate.get("activated", True))
    symbol = str(summary.get("symbol", "unknown")).upper()
    records = []
    traces = fold.get("heldout_traces", {}).get("agent", [])
    for path_index, trace in enumerate(traces):
        for decision_index, decision in enumerate(trace.get("decisions", [])):
            orders = tuple(int(value) for value in decision.get("orders", []))
            executions = decision.get("executions", [])
            research_action = _decision_label(
                orders,
                executions,
                int(decision.get("invalid_actions", 0)),
            )
            records.append(
                {
                    "Timestamp": decision.get("decision_timestamp"),
                    "Arrival": decision.get("arrival_timestamp"),
                    "Ticker": symbol,
                    "Agent": _agent_label(winner),
                    "Topology": _topology_label(model),
                    "Path": path_index,
                    "Decision": decision_index,
                    "Research action": research_action,
                    "Sandbox action": research_action if activated else "HOLD (guard)",
                    "Requested legs": sum(value != 0 for value in orders),
                    "Executions": len(executions),
                    "Invalid actions": int(decision.get("invalid_actions", 0)),
                    "Reward": _number(decision.get("reward")),
                    "NAV": _number(decision.get("nav")),
                    "Activation": "Active" if activated else "Guarded",
                }
            )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
        frame["Arrival"] = pd.to_datetime(frame["Arrival"], utc=True)
    return frame


def agent_roster(runs: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Describe the latest persisted selected policy for each ticker."""
    newest_by_symbol: dict[str, dict[str, Any]] = {}
    for run in runs:
        symbol = str(run.get("symbol", "")).upper()
        if symbol and symbol not in newest_by_symbol:
            newest_by_symbol[symbol] = run

    records = []
    for symbol, run in newest_by_symbol.items():
        folds = run.get("folds", [])
        if not folds:
            continue
        fold = max(folds, key=lambda item: int(item.get("fold", -1)))
        winner = _selected_candidate(fold)
        if not winner:
            continue
        model = winner.get("model", {})
        model_id = str(winner.get("model_id", "unknown"))
        gate = fold.get("model_selection", {}).get("activation_gate", {})
        activated = bool(gate.get("activated", True))
        reports = heldout_results(run)
        fold_number = int(fold.get("fold", 0))
        report = reports[reports["Fold"] == fold_number]
        activity = agent_decision_tape(run, fold_number)
        latest = activity.iloc[-1] if not activity.empty else None
        aggregate = report.iloc[0] if not report.empty else None
        data_quality = fold.get("test_data_quality", {})
        training_seed_count = int(
            winner.get("training_seed_aggregate", {}).get("training_seed_count", 1)
        )
        records.append(
            {
                "Agent ID": f"{symbol}-{model_id}",
                "Ticker": symbol,
                "State": "Paper active" if activated else "Guarded / no-op",
                "Research policy": _agent_label(winner),
                "Temporal core": _humanize(str(model.get("kind", "unknown"))),
                "Topology": _topology_label(model),
                "Architecture": _architecture_label(model),
                "Algorithm": str(model.get("algorithm", "unknown")).upper(),
                "Action policy": _decoder_label(model),
                "Feature set": _feature_set_label(model),
                "Training seeds": training_seed_count,
                "Validation edge vs no-op (bp)": 10_000.0
                * _number(gate.get("score_advantage")),
                "Held-out return": (
                    float(aggregate["Test return"])
                    if aggregate is not None
                    else float("nan")
                ),
                "Sandbox return": (
                    float(aggregate["Sandbox return"])
                    if aggregate is not None
                    else float("nan")
                ),
                "Decisions": len(activity),
                "Research executions": (
                    int(aggregate["Executions"]) if aggregate is not None else 0
                ),
                "Sandbox executions": (
                    int(aggregate["Sandbox executions"]) if aggregate is not None else 0
                ),
                "Last research action": (
                    str(latest["Research action"]) if latest is not None else "No trace"
                ),
                "Last sandbox action": (
                    str(latest["Sandbox action"]) if latest is not None else "No trace"
                ),
                "Median latency (us)": _number(
                    winner.get("inference_latency", {}).get("median_microseconds")
                ),
                "Checkpoint": str(fold.get("checkpoint", "Unavailable")),
                "Test start": data_quality.get("first_timestamp", "Unknown"),
                "Test end": data_quality.get("last_timestamp", "Unknown"),
                "Experiment": run.get("_run_name", "unknown"),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("Ticker").reset_index(drop=True)


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
                    "Smile residual": _feature_status(model, "contract_smile_residual"),
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
                    "Training seeds": int(seed_aggregate.get("training_seed_count", 1)),
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
        selection_rule = fold.get("model_selection", {}).get("simplicity_rule", {})
        activation_gate = fold.get("model_selection", {}).get("activation_gate", {})
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
                        selection_rule.get("score_sacrificed_for_simplicity", 0.0)
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
                    "Sandbox latency (us)": (selected_latency if activated else 0.0),
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
        test_qualities = [
            fold.get("test_data_quality", {}) for fold in run.get("folds", [])
        ]
        test_starts = sorted(
            str(item["first_timestamp"])
            for item in test_qualities
            if item.get("first_timestamp")
        )
        test_ends = sorted(
            str(item["last_timestamp"])
            for item in test_qualities
            if item.get("last_timestamp")
        )
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
                "Sandbox executions": int(heldout["Sandbox executions"].sum()),
                "Sandbox fees": float(heldout["Sandbox fees"].sum()),
                "Sandbox latency (us)": float(heldout["Sandbox latency (us)"].mean()),
                "Selected latency (us)": float(heldout["Selected latency (us)"].mean()),
                "Score traded (bp)": float(heldout["Score traded (bp)"].mean()),
                "Competitive candidates": int(heldout["Competitive candidates"].max()),
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
                "Test start": test_starts[0] if test_starts else "Unknown",
                "Test end": test_ends[-1] if test_ends else "Unknown",
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
                ablation_lift = _number(candidate.get("validation_score_lift_vs_full"))
                if pd.isna(ablation_lift):
                    continue
                kind = str(model.get("kind", "unknown"))
                feature_lift_bp = -10_000.0 * ablation_lift
                records.append(
                    {
                        "Ticker": symbol,
                        "Fold": int(fold.get("fold", 0)),
                        "Encoder": _humanize(str(model.get("encoder", "unknown"))),
                        "Agent": AGENT_LABELS.get(kind, kind.upper()),
                        "Feature": _humanize(feature_group),
                        "Feature lift (bp)": feature_lift_bp,
                        "Feature helped": "Yes" if feature_lift_bp > 0 else "No",
                        "Ablated latency (us)": _number(
                            candidate.get("inference_latency", {}).get(
                                "median_microseconds"
                            )
                        ),
                        "Ablated parameters": _number(candidate.get("parameter_count")),
                        "Training seeds": int(
                            candidate.get("training_seed_aggregate", {}).get(
                                "training_seed_count", 1
                            )
                        ),
                    }
                )
    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values(["Ticker", "Encoder", "Agent"])
        .reset_index(drop=True)
    )


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
        activation_gates.append(fold.get("model_selection", {}).get("activation_gate"))

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


def _topology_label(model: dict[str, Any]) -> str:
    encoder = str(model.get("encoder", "unknown"))
    if "graph" in encoder:
        return "Surface GNN"
    if "attention" in encoder:
        return "Surface attention"
    return "Flat vector"


def _decision_label(
    orders: Sequence[int],
    executions: Sequence[dict[str, Any]],
    invalid_actions: int,
) -> str:
    sides = {str(item.get("side", "unknown")).upper() for item in executions}
    if executions:
        side = next(iter(sides)) if len(sides) == 1 else "MULTI"
        suffix = "fill" if len(executions) == 1 else "fills"
        return f"{side} · {len(executions)} {suffix}"
    if any(value != 0 for value in orders):
        return "BLOCKED" if invalid_actions else "UNFILLED"
    return "HOLD"


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
