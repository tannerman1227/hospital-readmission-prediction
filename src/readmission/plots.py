"""Charts for the cohort description and the held-out evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_curve

NAVY = "#1F4E79"
TEAL = "#0E7C7B"
ORANGE = "#D55E00"
BLUE = "#0072B2"
SLATE = "#5C6B73"
GREEN = "#1B7F4E"
GOLD = "#B08900"
RED = "#A33B3B"
GRID = "#E6EEF2"
INK = "#1C2B33"

POLICY_COLORS = {
    "gradient_boosting": ORANGE,
    "logistic_regression": BLUE,
    "prior_inpatient": TEAL,
    "random": SLATE,
    "gb_calibrated": ORANGE,
    "lr_calibrated": BLUE,
    "gb_raw": GOLD,
}
POLICY_LABELS = {
    "gradient_boosting": "Gradient boosting",
    "logistic_regression": "Logistic regression",
    "prior_inpatient": "Prior inpatient stays",
    "random": "Random panel",
    "gb_calibrated": "Gradient boosting, calibrated",
    "lr_calibrated": "Logistic regression, calibrated",
    "gb_raw": "Gradient boosting, raw score",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _finish(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=INK, pad=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    return path


def plot_missingness(table: pd.DataFrame, path: Path) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    colors = [GOLD if kind.startswith("Test") else NAVY for kind in table["kind"]]
    ax.barh(table["field"], table["missing_rate"] * 100, color=colors, zorder=2)
    _finish(ax, "Fields that are mostly blank in the source extract", "Percent missing", "")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    for y, rate in enumerate(table["missing_rate"]):
        percent = rate * 100
        label = f"{percent:.1f}%" if percent >= 1 else f"{percent:.2f}%"
        ax.text(percent + 1.2, y, label, va="center", fontsize=8, color=SLATE)
    handles = [
        Line2D([0], [0], color=NAVY, lw=8, label="Unknown code ('?')"),
        Line2D([0], [0], color=GOLD, lw=8, label="Lab not recorded"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    return save_figure(fig, path) and fig


def plot_rate_bars(
    labels: list[str],
    rates: list[float],
    counts: list[int],
    path: Path,
    title: str,
    xlabel: str,
    horizontal: bool = False,
    base_rate: float | None = None,
) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.4), layout="constrained")
    percents = [rate * 100 for rate in rates]
    if horizontal:
        ax.barh(labels, percents, color=TEAL, zorder=2)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        span = max(percents) if percents else 1
        for y, (pct, count) in enumerate(zip(percents, counts)):
            ax.text(pct + span * 0.02, y, f"{pct:.1f}%  (n={count:,})", va="center", fontsize=8, color=SLATE)
        ax.set_xlim(0, span * 1.55)
    else:
        ax.bar(labels, percents, color=TEAL, zorder=2)
        for x, (pct, count) in enumerate(zip(percents, counts)):
            ax.text(x, pct + 0.3, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8, color=SLATE)
        ax.set_ylim(0, max(percents) * 1.18 if percents else 1)
    _finish(
        ax,
        title,
        "30-day readmission rate (%)" if horizontal else xlabel,
        "" if horizontal else "30-day readmission rate (%)",
    )
    if horizontal:
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID)
    if base_rate is not None:
        value = base_rate * 100
        if horizontal:
            ax.axvline(value, color=ORANGE, ls="--", lw=1, zorder=3)
            ax.set_xlabel(f"30-day readmission rate (%)    dashed line: cohort rate {value:.1f}%")
        else:
            ax.axhline(value, color=ORANGE, ls="--", lw=1, label="Cohort rate")
            ax.legend(frameon=False, loc="upper right")
    return save_figure(fig, path) and fig


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 6.2), layout="constrained")
    for name, (y, score) in curves.items():
        fpr, tpr, _ = roc_curve(y, score)
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y, score)
        ax.plot(fpr, tpr, color=POLICY_COLORS.get(name, NAVY), lw=2, label=f"{POLICY_LABELS.get(name, name)}  {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#C5CED6", lw=1)
    _finish(ax, "Discrimination on held-out patients", "False positive rate", "True positive rate")
    ax.legend(frameon=False, title="AUC", loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    return save_figure(fig, path) and fig


def plot_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], prevalence: float, path: Path) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 6.2), layout="constrained")
    for name, (y, score) in curves.items():
        precision, recall, _ = precision_recall_curve(y, score)
        from sklearn.metrics import average_precision_score

        ap = average_precision_score(y, score)
        ax.plot(recall, precision, color=POLICY_COLORS.get(name, NAVY), lw=2, label=f"{POLICY_LABELS.get(name, name)}  {ap:.3f}")
    ax.axhline(prevalence, color="#C5CED6", lw=1, label=f"Prevalence  {prevalence:.3f}")
    _finish(ax, "Precision–recall on held-out patients", "Recall", "Precision")
    ax.legend(frameon=False, title="Average precision", loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return save_figure(fig, path) and fig


def plot_calibration(
    y: np.ndarray,
    curves: dict[str, np.ndarray],
    path: Path,
    n_bins: int = 10,
) -> plt.Figure:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0), layout="constrained")
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#C5CED6", lw=1)
    for name, score in curves.items():
        order = np.argsort(score, kind="mergesort")
        chunks = np.array_split(order, n_bins)
        xs, ys = [], []
        for idx in chunks:
            if len(idx) == 0:
                continue
            xs.append(float(np.mean(score[idx])))
            ys.append(float(np.mean(y[idx])))
        ax.plot(xs, ys, marker="o", lw=1.8, color=POLICY_COLORS.get(name, NAVY), label=POLICY_LABELS.get(name, name))
    _finish(ax, "Do the probabilities match the outcomes?", "Mean predicted probability", "Observed readmission rate")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, max(0.45, max(float(np.max(s)) for s in curves.values()) * 1.05))
    ax.set_ylim(0, max(0.45, max(max(ys) for ys in [_bin_means(y, s, n_bins) for s in curves.values()]) * 1.15))

    ax = axes[1]
    for name, score in curves.items():
        ax.hist(
            score,
            bins=30,
            density=True,
            histtype="step",
            lw=1.8,
            color=POLICY_COLORS.get(name, NAVY),
            label=POLICY_LABELS.get(name, name),
        )
    _finish(ax, "Where the predicted risk sits", "Predicted probability", "Density")
    ax.legend(frameon=False)
    return save_figure(fig, path) and fig


def _bin_means(y: np.ndarray, score: np.ndarray, n_bins: int) -> list[float]:
    order = np.argsort(score, kind="mergesort")
    return [float(np.mean(y[idx])) for idx in np.array_split(order, n_bins) if len(idx)]


def plot_deciles(deciles: pd.DataFrame, base_rate: float, path: Path) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    labels = [f"D{int(d)}" for d in deciles["risk_decile"]]
    rates = deciles["readmission_rate"] * 100
    colors = [ORANGE if int(d) == 1 else TEAL for d in deciles["risk_decile"]]
    ax.bar(labels, rates, color=colors, zorder=2)
    ax.axhline(base_rate * 100, color=NAVY, ls="--", lw=1, label=f"Test-set rate {base_rate * 100:.1f}%")
    for x, (rate, lift) in enumerate(zip(rates, deciles["lift"])):
        ax.text(x, rate + 0.4, f"{rate:.1f}%", ha="center", va="bottom", fontsize=8, color=SLATE)
    _finish(ax, "Held-out readmission rate by predicted-risk decile", "Decile 1 is the highest predicted risk", "30-day readmission rate (%)")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(rates) * 1.2)
    return save_figure(fig, path) and fig


def plot_net_savings(scenarios: pd.DataFrame, primary_fraction: float, path: Path) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.3), layout="constrained")
    for policy, color in POLICY_COLORS.items():
        part = scenarios.loc[scenarios["policy"] == policy].sort_values("target_fraction")
        if part.empty:
            continue
        ax.plot(
            part["target_fraction"] * 100,
            part["net_savings_per_1000_discharges_usd"],
            marker="o",
            lw=2,
            color=color,
            label=POLICY_LABELS[policy],
        )
    ax.axhline(0, color="#C5CED6", lw=1)
    ax.axvline(primary_fraction * 100, color=GOLD, ls="--", lw=1, label="Pre-specified 10% panel")
    _finish(
        ax,
        "Net savings of a care-management bundle",
        "Percent of discharges enrolled, highest risk first",
        "Net savings per 1,000 discharges (USD)",
    )
    ax.legend(frameon=False, loc="best")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    return save_figure(fig, path) and fig


def pretty_feature(name: str) -> str:
    replacements = [
        ("medical_specialty_group_", "Specialty: "),
        ("admission_source_group_", "Admitted from: "),
        ("admission_type_", "Admission: "),
        ("discharge_group_", "Discharge: "),
        ("diag_1_group_", "Primary dx: "),
        ("diag_2_group_", "Second dx: "),
        ("diag_3_group_", "Third dx: "),
        ("a1c_result_", "HbA1c: "),
        ("max_glu_serum_", "Glucose: "),
        ("insulin_", "Insulin: "),
        ("metformin_status_", "Metformin: "),
        ("number_inpatient", "Prior inpatient stays"),
        ("number_emergency", "Prior emergency visits"),
        ("number_outpatient", "Prior outpatient visits"),
        ("number_diagnoses", "Diagnosis count"),
        ("num_lab_procedures", "Lab procedures"),
        ("num_procedures", "Procedures"),
        ("num_medications", "Medication count"),
        ("time_in_hospital", "Length of stay (days)"),
        ("n_diabetes_meds", "Diabetes medication count"),
        ("n_med_changes", "Dose changes during stay"),
        ("age_mid", "Age (band midpoint)"),
    ]
    label = name
    for source, target in replacements:
        if label.startswith(source) or label == source:
            label = target + label[len(source):]
            break
    return label.replace("_", " ")


def plot_shap(table: pd.DataFrame, path: Path, top_n: int = 15) -> plt.Figure:
    _apply_style()
    part = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 6.2), layout="constrained")
    colors = [ORANGE if direction > 0 else BLUE if direction < 0 else SLATE for direction in part["direction"]]
    ax.barh(part["feature"].map(pretty_feature), part["mean_abs_shap"], color=colors, zorder=2)
    _finish(ax, "What moves the gradient boosting score", "Mean absolute SHAP value on a test sample", "")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    handles = [
        Line2D([0], [0], color=ORANGE, lw=8, label="Higher value raises risk"),
        Line2D([0], [0], color=BLUE, lw=8, label="Higher value lowers risk"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    return save_figure(fig, path) and fig


def plot_fairness(audit: pd.DataFrame, path: Path) -> plt.Figure:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2), layout="constrained")
    for ax, group_type, title in (
        (axes[0], "race", "Race recorded in the extract"),
        (axes[1], "gender", "Sex recorded in the extract"),
    ):
        part = audit.loc[audit["group_type"] == group_type].sort_values("n", ascending=True)
        y = np.arange(len(part))
        ax.hlines(y, 0, part["recall"] * 100, color="#D0D7DE", lw=2)
        colors = [SLATE if flag else TEAL for flag in part["small_sample"]]
        ax.scatter(part["recall"] * 100, y, s=60, color=colors, zorder=2, label="Recall in group")
        ax.scatter(part["prevalence"] * 100, y, s=36, color=ORANGE, zorder=2, label="Group prevalence")
        ax.set_yticks(y, part["group"])
        _finish(ax, title, "Percent", "")
        ax.set_xlim(0, max(40, float(part["recall"].max() * 100) + 8))
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markersize=8, label="Share of the group's readmissions flagged"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=7, label="Group's readmission rate"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SLATE, markersize=8, label="Fewer than 150 test encounters"),
    ]
    fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=2)
    fig.suptitle("Who the top 10% panel reaches", x=0.0, ha="left", fontsize=14, fontweight="bold", color=INK)
    return save_figure(fig, path) and fig

