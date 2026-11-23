"""Ranking metrics, calibration error, deciles, and a patient-clustered AUC interval."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-frequency bins. Empty bins are skipped."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    y_sorted = y[order]
    p_sorted = p[order]
    folds = np.array_split(np.arange(len(p_sorted)), n_bins)
    total = 0.0
    for idx in folds:
        if len(idx) == 0:
            continue
        weight = len(idx) / len(p_sorted)
        total += weight * abs(y_sorted[idx].mean() - p_sorted[idx].mean())
    return float(total)


def discrimination_report(y: np.ndarray, p: np.ndarray, p_calibrated: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    report = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_raw": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "prevalence": float(y.mean()),
    }
    if p_calibrated is not None:
        calibrated = np.clip(np.asarray(p_calibrated, dtype=float), 0, 1)
        base = np.full(len(y), y.mean())
        brier = float(brier_score_loss(y, calibrated))
        brier_base = float(brier_score_loss(y, base))
        report.update(
            {
                "roc_auc_calibrated": float(roc_auc_score(y, calibrated)),
                "brier_calibrated": brier,
                "brier_prevalence": brier_base,
                "brier_skill": float(1 - brier / brier_base) if brier_base else None,
                "ece_calibrated": expected_calibration_error(y, calibrated),
            }
        )
    return report


def clustered_auc_interval(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile interval for AUC, resampling patients rather than encounters."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="mergesort")
    y = y[order]
    p = p[order]
    groups = groups[order]
    unique, starts = np.unique(groups, return_index=True)
    slices = list(zip(starts, list(starts[1:]) + [len(groups)]))
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    n_groups = len(unique)
    for _ in range(n_boot):
        drawn = rng.integers(0, n_groups, size=n_groups)
        parts_y = []
        parts_p = []
        for index in drawn:
            start, stop = slices[index]
            parts_y.append(y[start:stop])
            parts_p.append(p[start:stop])
        y_b = np.concatenate(parts_y)
        if y_b.min() == y_b.max():
            continue
        scores.append(roc_auc_score(y_b, np.concatenate(parts_p)))
    array = np.asarray(scores)
    lo, hi = np.percentile(array, [2.5, 97.5])
    return {
        "auc_boot_mean": float(array.mean()),
        "auc_ci95_low": float(lo),
        "auc_ci95_high": float(hi),
        "n_bootstrap": int(len(array)),
    }


def risk_deciles(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Decile 1 is the highest predicted risk."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(y) < 10:
        raise ValueError("Need at least 10 rows to form deciles.")
    order = pd.Series(score).rank(method="first", ascending=False)
    decile = pd.qcut(order, 10, labels=False) + 1
    frame = pd.DataFrame({"decile": decile.astype(int), "y": y, "score": score})
    base = float(y.mean())
    total_events = float(y.sum())
    rows = []
    for level, part in frame.groupby("decile"):
        rate = float(part["y"].mean())
        events = float(part["y"].sum())
        rows.append(
            {
                "risk_decile": int(level),
                "n": int(len(part)),
                "readmissions": int(events),
                "readmission_rate": rate,
                "lift": rate / base if base else None,
                "share_of_readmissions": events / total_events if total_events else None,
                "mean_score": float(part["score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("risk_decile").reset_index(drop=True)


def tie_broken_scores(score: np.ndarray, seed: int) -> np.ndarray:
    """Add a tiny seeded jitter so tied scores do not inherit row order."""
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(score)) * 1e-9
    return np.asarray(score, dtype=float) + jitter
