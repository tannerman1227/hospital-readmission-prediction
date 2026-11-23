"""Descriptive audit of the global outreach policy across demographic groups.

Race and gender are not model features. Gaps here can still appear through
correlated clinical variables. The audit describes the held-out test set; it
is not a determination that the model is fair or unfair to deploy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission.config import FAIRNESS_MIN_N, PRIMARY_TARGET_FRACTION
from readmission.evaluation import tie_broken_scores

AGE_BANDS = {
    "[0-10)": "Under 40",
    "[10-20)": "Under 40",
    "[20-30)": "Under 40",
    "[30-40)": "Under 40",
    "[40-50)": "40–49",
    "[50-60)": "50–59",
    "[60-70)": "60–69",
    "[70-80)": "70–79",
    "[80-90)": "80 and older",
    "[90-100)": "80 and older",
}

RACE_LABELS = {
    "AfricanAmerican": "African American",
    "Caucasian": "Caucasian",
    "Hispanic": "Hispanic",
    "Asian": "Asian",
    "Other": "Other",
    "Missing": "Missing",
}


def policy_flags(score: np.ndarray, fraction: float = PRIMARY_TARGET_FRACTION, seed: int = 0) -> np.ndarray:
    ranked = tie_broken_scores(score, seed=seed)
    k = max(1, int(round(len(score) * fraction)))
    chosen = np.argsort(-ranked, kind="mergesort")[:k]
    flags = np.zeros(len(score), dtype=bool)
    flags[chosen] = True
    return flags


def _group_rows(
    frame: pd.DataFrame,
    group_col: str,
    group_type: str,
    min_n: int,
) -> list[dict]:
    rows = []
    for key, part in frame.groupby(group_col, dropna=False):
        y = part["y"].to_numpy(dtype=float)
        flagged = part["flagged"].to_numpy(dtype=bool)
        events = float(y.sum())
        flagged_n = int(flagged.sum())
        rows.append(
            {
                "group_type": group_type,
                "group": str(key),
                "n": int(len(part)),
                "readmissions": int(events),
                "prevalence": float(y.mean()),
                "mean_predicted_risk": float(part["p"].mean()),
                "selection_rate": float(flagged.mean()),
                "recall": float(y[flagged].sum() / events) if events else None,
                "precision": float(y[flagged].sum() / flagged_n) if flagged_n else None,
                "small_sample": bool(len(part) < min_n),
            }
        )
    return rows


def fairness_audit(
    test: pd.DataFrame,
    y: np.ndarray,
    score: np.ndarray,
    calibrated: np.ndarray,
    fraction: float = PRIMARY_TARGET_FRACTION,
    seed: int = 0,
    min_n: int = FAIRNESS_MIN_N,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "race": test["race"].map(RACE_LABELS).fillna(test["race"].astype(str)).to_numpy(),
            "gender": test["gender"].astype(str).to_numpy(),
            "age_band": test["age"].map(AGE_BANDS).to_numpy(),
            "y": np.asarray(y, dtype=int),
            "p": np.asarray(calibrated, dtype=float),
            "flagged": policy_flags(score, fraction=fraction, seed=seed),
        }
    )
    rows: list[dict] = []
    rows.extend(_group_rows(frame, "race", "race", min_n))
    rows.extend(_group_rows(frame, "gender", "gender", min_n))
    rows.extend(_group_rows(frame, "age_band", "age", min_n))
    ordered = pd.DataFrame(rows)
    return ordered.sort_values(["group_type", "n"], ascending=[True, False]).reset_index(drop=True)


def recall_gap(audit: pd.DataFrame, group_type: str, exclude: set[str] | None = None) -> dict:
    """Max minus min recall among groups large enough to read."""
    part = audit.loc[(audit["group_type"] == group_type) & (~audit["small_sample"])].copy()
    if exclude:
        part = part.loc[~part["group"].isin(exclude)]
    part = part.dropna(subset=["recall"])
    if part.empty:
        return {"group_type": group_type, "gap": None, "highest": None, "lowest": None}
    high = part.loc[part["recall"].idxmax()]
    low = part.loc[part["recall"].idxmin()]
    return {
        "group_type": group_type,
        "gap": float(high["recall"] - low["recall"]),
        "highest_group": str(high["group"]),
        "highest_recall": float(high["recall"]),
        "lowest_group": str(low["group"]),
        "lowest_recall": float(low["recall"]),
        "groups_compared": int(len(part)),
    }
