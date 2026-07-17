from __future__ import annotations

from cProfile import label
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import re


History = Mapping[str, Sequence[Any]]


@dataclass(frozen=True)
class MetricSpec:
    """Specification for resolving and plotting a metric from optimizer history."""

    name: str
    candidates: tuple[str, ...]
    label: str | None = None
    logy: bool = False
    ylabel: str | None = None

    @property
    def display_name(self) -> str:
        return self.label or self.name


DEFAULT_METRICS: dict[str, MetricSpec] = {
    "objective": MetricSpec(
        name="objective",
        candidates=("objective_val",),
        label="Objective value",
        logy=True,
        ylabel="Objective",
    ),
    "constraint_norm": MetricSpec(
        name="constraint_norm",
        candidates=("constraint_norm",),
        label="Constraint norm",
        logy=True,
        ylabel="Norm",
    ),
    "stationarity": MetricSpec(
        name="stationarity",
        candidates=("proj_gradient_norm", "primal_gradient_norm"),
        label="Stationarity",
        logy=True,
        ylabel="Norm",
    ),
    "train_accuracy": MetricSpec(
        name="train_accuracy",
        candidates=("train_accuracy",),
        label="Train accuracy",
        logy=False,
        ylabel="Accuracy",
    ),
    "test_accuracy": MetricSpec(
        name="test_accuracy",
        candidates=("test_accuracy",),
        label="Test accuracy",
        logy=False,
        ylabel="Accuracy",
    ),
    "train_loss": MetricSpec(
        name="train_loss",
        candidates=("train_mse", "train_loss"),
        label="Train loss",
        logy=True,
        ylabel="Loss",
    ),
    "test_loss": MetricSpec(
        name="test_loss",
        candidates=("test_mse", "test_loss"),
        label="Test loss",
        logy=True,
        ylabel="Loss",
    ),
    "test_l2_rel_error": MetricSpec(
        name="test_l2_rel_error",
        candidates=("test_l2_rel_error",),
        label="Test L2 Relative Error",
        logy=True,
        ylabel="Error",
    ),
    "test_linf_error": MetricSpec(
        name="test_linf_error",
        candidates=("test_linf_error",),
        label="Test L∞ Error",
        logy=True,
        ylabel="Error",
    ),
    "step_size": MetricSpec(
        name="step_size",
        candidates=("step_size",),
        label="Step size",
        logy=True,
        ylabel="Value",
    ),
    "globalization": MetricSpec(
        name="globalization",
        candidates=("merit_parameter", "penalty"),
        label="Globalization parameter",
        logy=True,
        ylabel="Value",
    ),
    "dual_norm": MetricSpec(
        name="dual_norm",
        candidates=("dual_norm",),
        label="Dual norm",
        logy=True,
        ylabel="Norm",
    ),
}


def _history_from_record(record: Any) -> History:
    if isinstance(record, Mapping):
        return record
    if hasattr(record, "history"):
        return record.history
    raise TypeError("Expected a history mapping or an object with a 'history' attribute.")


def _label_from_record(record: Any, default: str) -> str:
    if hasattr(record, "solver_name"):
        return str(record.solver_name)
    if hasattr(record, "name"):
        return str(record.name)
    return default


def _to_numpy_series(history: History, key: str) -> np.ndarray:
    if key not in history:
        raise KeyError(f"History does not contain key '{key}'.")

    values: list[float] = []
    for value in history[key]:
        if value is None:
            values.append(np.nan)
        elif isinstance(value, (bool, np.bool_)):
            values.append(float(value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            values.append(float(value))
        elif hasattr(value, "item"):
            values.append(float(value.item()))
        else:
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _present_keys(history: History, keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key in history]


def _metric_spec(metric: str | MetricSpec) -> MetricSpec:
    if isinstance(metric, MetricSpec):
        return metric
    if metric in DEFAULT_METRICS:
        return DEFAULT_METRICS[metric]
    return MetricSpec(name=metric, candidates=(metric,), label=metric)


def resolve_metric_key(history: History, metric: str | MetricSpec) -> str | None:
    spec = _metric_spec(metric)
    for key in spec.candidates:
        if key in history:
            return key
    return None


def get_metric_series(history: History, metric: str | MetricSpec) -> tuple[str, np.ndarray]:
    spec = _metric_spec(metric)
    key = resolve_metric_key(history, spec)
    if key is None:
        raise KeyError(
            f"Could not resolve metric '{spec.name}'. Tried keys {list(spec.candidates)}."
        )
    return key, _to_numpy_series(history, key)


def plot_metric_group(
    history: History,
    keys: Sequence[str],
    *,
    title: str,
    ylabel: str | None = None,
    logy: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    keys = _present_keys(history, keys)
    if not keys:
        raise ValueError("None of the requested keys are present in history.")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(1, len(next(iter(history.values()))) + 1)
    for key in keys:
        ax.plot(x, _to_numpy_series(history, key), label=key)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_metric_comparison(
    records: Sequence[Any],
    metric: str | MetricSpec,
    *,
    labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    logy: bool | None = None,
    logx: bool = False,
    title: str | None = None,
    time: bool = False,
) -> plt.Axes:
    spec = _metric_spec(metric)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if time:
        logx=False # override for time plots. 

    plotted_any = False
    for idx, record in enumerate(records):
        history = _history_from_record(record)
        key = resolve_metric_key(history, spec)
        if key is None:
            continue
        
        if time:
            x = _to_numpy_series(history, "train_time")
        else:
            x = np.arange(1, len(next(iter(history.values()))) + 1)
        y = _to_numpy_series(history, key)

        label = labels[idx] if labels is not None else _label_from_record(record, f"run_{idx}")
        ax.plot(x, y, label=label)
        plotted_any = True

    if not plotted_any:
        raise ValueError(
            f"None of the records contained metric '{spec.name}' with candidates {list(spec.candidates)}."
        )

    ax.set_title(title or spec.display_name)
    ax.set_xlabel("Time" if time else "Iteration")
    if spec.ylabel is not None:
        ax.set_ylabel(spec.ylabel)
    if logx:
        ax.set_xscale("log")
    if logy if logy is not None else spec.logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_comparison_panel(
    records: Sequence[Any],
    *,
    metrics: Sequence[str | MetricSpec] = (
        "objective",
        "constraint_norm",
        "stationarity",
        "test_accuracy",
    ),
    figsize: tuple[float, float] = (12, 8),
    time: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    metrics = list(metrics)
    n_plots = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for i, metric in enumerate(metrics):
        ax = axes.flat[i]
        try:
            plot_metric_comparison(records, metric, ax=ax, logx=True, time=time)
        except ValueError:
            spec = _metric_spec(metric)
            ax.set_title(spec.display_name)
            ax.text(0.5, 0.5, "Metric unavailable", ha="center", va="center")
            ax.axis("off")

    for j in range(n_plots, axes.size):
        axes.flat[j].axis("off")

    return fig, axes


def _overlay_pdn_step_markers(history: History, x: np.ndarray, y: np.ndarray, ax: plt.Axes) -> None:
    if "tan_inner_flag" not in history:
        return

    flag = _to_numpy_series(history, "tan_inner_flag")
    lpc_nc_mask = np.isfinite(y) & (flag == 1.0)

    if np.any(lpc_nc_mask):
        ax.scatter(x[lpc_nc_mask], y[lpc_nc_mask], marker="x", s=50, label="LPC/NC")


def _overlay_alm_step_markers(history: History, x: np.ndarray, y: np.ndarray, ax: plt.Axes) -> None:
    if "step_type" not in history:
        return

    step_type = _to_numpy_series(history, "step_type")
    type_1_mask = np.isfinite(y) & (step_type == 1)
    type_2_mask = np.isfinite(y) & (step_type == 2)

    if np.any(type_1_mask):
        ax.scatter(x[type_1_mask], y[type_1_mask], marker="x", s=45, label="Dual update")
    if np.any(type_2_mask):
        ax.scatter(x[type_2_mask], y[type_2_mask], marker="^", s=45, label="Dual + penalty update")


def plot_globalization_panel(
    history: History,
    ax: plt.Axes | None = None,
) -> tuple[plt.Axes, plt.Axes | None]:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(1, len(next(iter(history.values()))) + 1)

    step_size = None
    if "step_size" in history:
        step_size = _to_numpy_series(history, "step_size")
        ax.plot(x, step_size, label="step_size")
        _overlay_pdn_step_markers(history, x, step_size, ax)

    global_key = resolve_metric_key(history, "globalization")
    if global_key is not None:
        ax.plot(x, _to_numpy_series(history, global_key), label=global_key)

    ax.set_title("Globalization diagnostics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return ax, None


def plot_history(history: History) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    metric_keys = _present_keys(history, ["train_accuracy", "test_accuracy", "train_mse", "test_mse"])
    if metric_keys:
        plot_metric_group(
            history,
            metric_keys,
            title="Train / test metrics",
            ylabel="Value",
            logy=False,
            ax=axes[0, 0],
        )
    else:
        axes[0, 0].set_title("Train / test metrics")
        axes[0, 0].text(0.5, 0.5, "No train/test metrics", ha="center", va="center")
        axes[0, 0].axis("off")

    conv_keys = ["objective_val", "constraint_norm"]
    stationarity_key = resolve_metric_key(history, "stationarity")
    if stationarity_key is not None:
        conv_keys.append(stationarity_key)
    conv_keys = _present_keys(history, conv_keys)
    if conv_keys:
        plot_metric_group(
            history,
            conv_keys,
            title="Convergence diagnostics",
            ylabel="Value",
            logy=True,
            ax=axes[0, 1],
        )
    else:
        axes[0, 1].set_title("Convergence diagnostics")
        axes[0, 1].text(0.5, 0.5, "No convergence metrics", ha="center", va="center")
        axes[0, 1].axis("off")

    try:
        plot_globalization_panel(history, ax=axes[1, 0])
    except Exception:
        axes[1, 0].set_title("Globalization diagnostics")
        axes[1, 0].text(0.5, 0.5, "No globalization metrics", ha="center", va="center")
        axes[1, 0].axis("off")

    step_keys = _present_keys(
        history,
        ["normal_step_norm", "tangent_step_norm", "trial_step_norm", "dual_norm", "al_grad_norm", "subproblem_tol"],
    )
    if step_keys:
        plot_metric_group(
            history,
            step_keys,
            title="Step and auxiliary diagnostics",
            ylabel="Norm / value",
            logy=True,
            ax=axes[1, 1],
        )
    else:
        axes[1, 1].set_title("Step and auxiliary diagnostics")
        axes[1, 1].text(0.5, 0.5, "No step/auxiliary metrics", ha="center", va="center")
        axes[1, 1].axis("off")

    return fig, axes


def plot_pdn_diagnostics(history: History, title: str = "") -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title)
    
    conv_keys = _present_keys(history, ["objective_val", "constraint_norm", "proj_gradient_norm"])
    if conv_keys:
        plot_metric_group(
            history,
            conv_keys,
            title="PDN convergence diagnostics",
            ylabel="Value",
            logy=True,
            ax=axes[0, 0],
        )
    else:
        axes[0, 0].set_title("PDN convergence diagnostics")
        axes[0, 0].text(0.5, 0.5, "No PDN convergence metrics", ha="center", va="center")
        axes[0, 0].axis("off")

    try:
        plot_globalization_panel(history, ax=axes[0, 1])
    except Exception:
        axes[0, 1].set_title("Globalization diagnostics")
        axes[0, 1].text(0.5, 0.5, "No globalization metrics", ha="center", va="center")
        axes[0, 1].axis("off")

    if "tan_inner_iters" in history:
        x = np.arange(1, len(next(iter(history.values()))) + 1)
        tan_inner_iters = _to_numpy_series(history, "tan_inner_iters")
        axes[1, 0].plot(x, tan_inner_iters, label="tan_inner_iters")
        _overlay_pdn_step_markers(history, x, tan_inner_iters, axes[1, 0])
        axes[1, 0].set_title("PDN inner iterations")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Iterations")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    else:
        axes[1, 0].set_title("PDN inner iterations")
        axes[1, 0].text(0.5, 0.5, "No PDN inner-iteration metrics", ha="center", va="center")
        axes[1, 0].axis("off")

    step_keys = _present_keys(history, ["normal_step_norm", "tangent_step_norm", "trial_step_norm"])
    if step_keys:
        plot_metric_group(
            history,
            step_keys,
            title="PDN step norms",
            ylabel="Value",
            logy=True,
            ax=axes[1, 1],
        )
    else:
        axes[1, 1].set_title("PDN step norms")
        axes[1, 1].text(0.5, 0.5, "No PDN step metrics", ha="center", va="center")
        axes[1, 1].axis("off")

    return fig, axes


def plot_alm_diagnostics(history: History, title: str = "") -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title)

    conv_keys = _present_keys(history, ["objective_val", "constraint_norm", "primal_gradient_norm"])
    if conv_keys:
        plot_metric_group(
            history,
            conv_keys,
            title="ALM convergence diagnostics",
            ylabel="Value",
            logy=True,
            ax=axes[0, 0],
        )
    else:
        axes[0, 0].set_title("ALM convergence diagnostics")
        axes[0, 0].text(0.5, 0.5, "No ALM convergence metrics", ha="center", va="center")
        axes[0, 0].axis("off")

    try:
        plot_globalization_panel(history, ax=axes[0, 1])
    except Exception:
        axes[0, 1].set_title("Globalization diagnostics")
        axes[0, 1].text(0.5, 0.5, "No globalization metrics", ha="center", va="center")
        axes[0, 1].axis("off")

    keys = _present_keys(history, ["penalty", "subproblem_tol", "al_grad_norm"])
    if keys:
        plot_metric_group(
            history,
            keys,
            title="Penalty and subproblem diagnostics",
            ylabel="Value",
            logy=True,
            ax=axes[1, 0],
        )
        if "penalty" in history:
            x = np.arange(1, len(next(iter(history.values()))) + 1)
            penalty = _to_numpy_series(history, "penalty")
            _overlay_alm_step_markers(history, x, penalty, axes[1, 0])
            handles, labels = axes[1, 0].get_legend_handles_labels()
            keep_handles = []
            keep_labels = []
            for handle, label in zip(handles, labels):
                if label in {"Dual update", "Dual + penalty"} and label in keep_labels:
                    continue
                keep_handles.append(handle)
                keep_labels.append(label)
            axes[1, 0].legend(keep_handles, keep_labels)
    else:
        axes[1, 0].set_title("Penalty and subproblem diagnostics")
        axes[1, 0].text(0.5, 0.5, "No ALM penalty metrics", ha="center", va="center")
        axes[1, 0].axis("off")

    keys = _present_keys(history, ["step_size", "dual_norm"])
    if keys:
        plot_metric_group(
            history,
            keys,
            title="ALM step diagnostics",
            ylabel="Value",
            logy=True,
            ax=axes[1, 1],
        )
    else:
        axes[1, 1].set_title("ALM step diagnostics")
        axes[1, 1].text(0.5, 0.5, "No ALM step metrics", ha="center", va="center")
        axes[1, 1].axis("off")

    return fig, axes

def plot_sqp_diagnostics(history: History, title: str = "") -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title)

    conv_keys = _present_keys(history, ["objective_val", "constraint_norm", "primal_gradient_norm"])
    if conv_keys:
        plot_metric_group(
            history,
            conv_keys,
            title="SQP convergence diagnostics",
            ylabel="Value",
            logy=True,
            ax=axes[0, 0],
        )
    else:
        axes[0, 0].set_title("SQP convergence diagnostics")
        axes[0, 0].text(0.5, 0.5, "No SQP convergence metrics", ha="center", va="center")
        axes[0, 0].axis("off")

    try:
        plot_globalization_panel(history, ax=axes[0, 1])
    except Exception:
        axes[0, 1].set_title("Globalization diagnostics")
        axes[0, 1].text(0.5, 0.5, "No globalization metrics", ha="center", va="center")
        axes[0, 1].axis("off")

    inner_keys = _present_keys(history, ["inner_total_iters", "inner_restart_count"])
    if inner_keys:
        plot_metric_group(
            history,
            inner_keys,
            title="SQP inner solver diagnostics",
            ylabel="Count",
            logy=False,
            ax=axes[1, 0],
        )
        x = np.arange(1, len(next(iter(history.values()))) + 1)

        if "inner_flag" in history and "inner_total_iters" in history:
            inner_total_iters = _to_numpy_series(history, "inner_total_iters")
            inner_flag = _to_numpy_series(history, "inner_flag")

            type_ii_mask = np.isfinite(inner_total_iters) & (inner_flag == 2.0)
            if np.any(type_ii_mask):
                axes[1, 0].scatter(
                    x[type_ii_mask],
                    inner_total_iters[type_ii_mask],
                    marker="x",
                    s=50,
                    label="Type II step",
                )

        if "inner_restart_count" in history and "saw_cr_breakdown" in history:
            inner_restart_count = _to_numpy_series(history, "inner_restart_count")
            inner_saw_cr_breakdown = _to_numpy_series(history, "saw_cr_breakdown")

            cr_breakdown_mask = (
                np.isfinite(inner_restart_count) & (inner_saw_cr_breakdown == True)
            )
            if np.any(cr_breakdown_mask):
                axes[1, 0].scatter(
                    x[cr_breakdown_mask],
                    inner_restart_count[cr_breakdown_mask],
                    marker="o",
                    s=45,
                    facecolors="k",
                    label="CR breakdown restart detected",
                )

        if "inner_restart_count" in history and "saw_phase_limit_restart" in history:
            inner_restart_count = _to_numpy_series(history, "inner_restart_count")
            inner_saw_phase_limit_restart = _to_numpy_series(history, "saw_phase_limit_restart")

            phase_limit_mask = (
                np.isfinite(inner_restart_count) & (inner_saw_phase_limit_restart == True)
            )
            if np.any(phase_limit_mask):
                axes[1, 0].scatter(
                    x[phase_limit_mask],
                    inner_restart_count[phase_limit_mask],
                    marker="s",
                    s=40,
                    facecolors="k",
                    label="Phase-limit restart detected",
                )

        handles, labels = axes[1, 0].get_legend_handles_labels()
        keep_handles = []
        keep_labels = []
        for handle, label in zip(handles, labels):
            if label in keep_labels:
                continue
            keep_handles.append(handle)
            keep_labels.append(label)
        axes[1, 0].legend(keep_handles, keep_labels)
        
    else:
        axes[1, 0].set_title("SQP inner solver diagnostics")
        axes[1, 0].text(0.5, 0.5, "No SQP inner-solver metrics", ha="center", va="center")
        axes[1, 0].axis("off")

    has_primal = "primal_step_norm" in history
    has_dual = "dual_step_norm" in history

    if has_primal or has_dual:
        x = np.arange(1, len(next(iter(history.values()))) + 1)

        if has_primal:
            primal_step_norm = _to_numpy_series(history, "primal_step_norm")
            axes[1, 1].plot(x, primal_step_norm, label="primal_step_norm")
        else:
            primal_step_norm = None

        if has_dual:
            dual_step_norm = _to_numpy_series(history, "dual_step_norm")
            axes[1, 1].plot(x, dual_step_norm, label="dual_step_norm")
        else:
            dual_step_norm = None

        if has_primal and has_dual:
            total_step_norm = np.sqrt(primal_step_norm**2 + dual_step_norm**2)
            axes[1, 1].plot(x, total_step_norm, label="total_step_norm")

        axes[1, 1].set_title("SQP step norms")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Norm")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].set_title("SQP step norms")
        axes[1, 1].text(0.5, 0.5, "No SQP step metrics", ha="center", va="center")
        axes[1, 1].axis("off")

    return fig, axes

def plot_run_diagnostics(record: Any) -> tuple[plt.Figure, np.ndarray]:
    history = _history_from_record(record)
    solver_name = _label_from_record(record, "").lower()

    if "pdn" in solver_name or "primal" in solver_name:
        return plot_pdn_diagnostics(history, title=solver_name)
    if "alm" in solver_name or "augmented" in solver_name:
        return plot_alm_diagnostics(history, title=solver_name)
    if "sqp" in solver_name:
        return plot_sqp_diagnostics(history, title=solver_name)

    return plot_history(history)

def _final_method_label(record: Any, label_map: Mapping[str, str] | None = None) -> str:
    raw = _label_from_record(record, "run")

    # Manual override takes priority.
    if label_map is not None and raw in label_map:
        return label_map[raw]

    # Otherwise try to infer a good label automatically.
    return _infer_method_label(raw)


def _deduplicate_legend(handles, labels):
    keep_handles = []
    keep_labels = []
    for handle, label in zip(handles, labels):
        if label in keep_labels:
            continue
        keep_handles.append(handle)
        keep_labels.append(label)
    return keep_handles, keep_labels

def _finite_positive_values(*arrays: np.ndarray) -> np.ndarray:
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        mask = np.isfinite(arr) & (arr > 0.0)
        if np.any(mask):
            vals.append(arr[mask])
    if not vals:
        return np.asarray([], dtype=float)
    return np.concatenate(vals)

def _format_numeric_string(value: str) -> str:
    """Format numeric strings like '0.1000' -> '0.1', '1e-03' -> '0.001' or '1e-03' depending on g-format."""
    try:
        return f"{float(value):g}"
    except ValueError:
        return value


def _infer_method_label(raw: str) -> str:
    """
    Infer a publication-friendly label from a raw solver/file name.
    This is intentionally somewhat hardcoded for our final plotting use cases.
    """
    # PDN least-squares dual with tangent regularisation sweep
    match = re.fullmatch(r"pdn_ls_dual__tan_reg_([-+0-9.eE]+)", raw)
    if match is not None:
        reg = _format_numeric_string(match.group(1))
        return rf"PDN (least-squares $\lambda={reg}$)"

    # PDN zero-dual with tangent regularisation sweep
    match = re.fullmatch(r"pdn_zero_dual__tan_reg_([-+0-9.eE]+)", raw)
    if match is not None:
        reg = _format_numeric_string(match.group(1))
        return rf"PDN (zero dual, $\lambda={reg}$)"

    # Some common defaults from the earlier setup
    if raw == "alm_lbfgs_default":
        return "ALM-LBFGS"
    if raw == "sqp_default":
        return "SQP"
    if raw == "pdn_ls_dual":
        return "PDN (least-squares dual)"
    if raw == "pdn_zero_dual":
        return "PDN (zero dual)"

    # Generic fallback
    return raw.replace("_", " ")

def plot_final_time_comparison(
    records: Sequence[Any],
    *,
    figsize: tuple[float, float] = (14, 3.5),
    label_map: Mapping[str, str] | None = None,
    objective_logy: bool = True,
    constraint_tol: float | None = None,
    stationarity_tol: float | None = None,
    subsample_target_points: int | None = None,
    logx: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    metrics = ("objective", "constraint_norm", "stationarity")
    titles = ("Objective value", "Constraint norm", "Stationarity")
    ylabels = ("Objective", "Norm", "Norm")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes = np.atleast_1d(axes)

    line_handles = []
    line_labels = []
    constraint_values = []
    stationarity_values = []
    time_offset = 1.

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        spec = _metric_spec(metric)
        plotted_any = False

        for record in records:
            history = _history_from_record(record)
            key = resolve_metric_key(history, spec)
            if key is None or "train_time" not in history:
                continue
            
            if logx:
                x = _to_numpy_series(history, "train_time") + time_offset
            else:
                x = _to_numpy_series(history, "train_time")
            y = _to_numpy_series(history, key)
            if metric == "constraint_norm":
                constraint_values.append(y)
            elif metric == "stationarity":
                stationarity_values.append(y)

            label = _final_method_label(record, label_map=label_map)
            if subsample_target_points is not None:
                subsampling = max(1, len(x) // subsample_target_points)
                x_plot = x[::subsampling]
                y_plot = y[::subsampling]

                # Add back on the last point if it was missed and is different from the last plotted point.
                if x_plot[-1] != x[-1]:
                    x_plot = np.append(x_plot, x[-1])
                    y_plot = np.append(y_plot, y[-1])
            else:
                x_plot = x
                y_plot = y    

            (line,) = ax.plot(x_plot, y_plot, linewidth=1.5, alpha=0.6, label=label)
            plotted_any = True

            line_handles.append(line)
            line_labels.append(label)

        if not plotted_any:
            ax.set_title(title)
            ax.text(0.5, 0.5, "Metric unavailable", ha="center", va="center")
            ax.axis("off")
            continue

        if metric in {"constraint_norm", "stationarity"}:
            ax.set_yscale("log")
        elif metric == "objective" and objective_logy:
            ax.set_yscale("log")

        if metric == "constraint_norm" and constraint_tol is not None:
            ax.axhline(constraint_tol, linestyle="--", linewidth=1.2, color="k")

        if metric == "stationarity" and stationarity_tol is not None:
            ax.axhline(stationarity_tol, linestyle="--", linewidth=1.2, color="k")

        if logx:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    norm_values = _finite_positive_values(
    *constraint_values,
    *stationarity_values,
)

    if norm_values.size > 0 and axes[1].has_data() and axes[2].has_data():
        ymin = np.min(norm_values)
        ymax = np.max(norm_values)

        # Small padding for log-scale plots.
        ymin *= 0.8
        ymax *= 1.25

        axes[1].set_ylim(ymin, ymax)
        axes[2].set_ylim(ymin, ymax)

    handles, labels = _deduplicate_legend(line_handles, line_labels)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.83, 0.5),
            ncol=1,
            frameon=True,
        )

    fig.subplots_adjust(
        bottom=0.16,
        right=0.82,
        left=0.08,
        wspace=0.25,
    )

    return fig, axes