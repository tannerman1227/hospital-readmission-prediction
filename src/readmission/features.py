"""Train-only specialty grouping and the model matrix.

Race and gender stay off the design matrix. They are used only in the
fairness audit. The target, identifiers, and raw ICD-9 strings are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from readmission.config import MIN_SPECIALTY_COUNT

NUMERIC_FEATURES = [
    "age_mid",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "n_diabetes_meds",
    "n_med_changes",
]

CATEGORICAL_FEATURES = [
    "admission_type",
    "discharge_group",
    "admission_source_group",
    "medical_specialty_group",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",
    "a1c_result",
    "insulin",
    "metformin_status",
]

FORBIDDEN_COLUMNS = {
    "race",
    "gender",
    "encounter_id",
    "patient_nbr",
    "readmitted_30d",
    "readmitted",
    "weight",
    "payer_code",
    "age",
    "medical_specialty",
}


@dataclass
class Design:
    matrices: list[pd.DataFrame]
    targets: list[np.ndarray]
    keep_specialties: list[str]
    encoder: OneHotEncoder


def _group_specialty(frame: pd.DataFrame, keep: set[str]) -> pd.DataFrame:
    out = frame.copy()
    specialty = out["medical_specialty"].astype(str)
    grouped = np.where(specialty.isin(keep) | specialty.eq("Missing"), specialty, "Other")
    out["medical_specialty_group"] = grouped
    return out


def _assert_safe(columns: list[str]) -> None:
    leaked = FORBIDDEN_COLUMNS.intersection(columns)
    if leaked:
        raise RuntimeError(f"Forbidden columns reached the model matrix: {sorted(leaked)}")
    banned_prefixes = ("race_", "gender_", "payer_", "weight_")
    bad = [col for col in columns if col.startswith(banned_prefixes)]
    if bad:
        raise RuntimeError(f"Protected or dropped fields were encoded: {bad}")


def fit_design(
    fit: pd.DataFrame,
    *others: pd.DataFrame,
    min_specialty_count: int = MIN_SPECIALTY_COUNT,
    scale_numeric: bool = False,
) -> Design:
    """Fit encoders on ``fit`` and transform ``fit`` plus each other frame."""
    counts = fit["medical_specialty"].astype(str).value_counts()
    keep = set(counts[counts >= min_specialty_count].index.tolist())
    keep.discard("Missing")

    frames = [_group_specialty(frame, keep) for frame in (fit, *others)]
    for frame in frames:
        for column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].astype(str)
        frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="raise")
        if frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().any().any():
            raise ValueError("Model columns contain nulls after cleaning.")

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )
    encoder.fit(frames[0][CATEGORICAL_FEATURES])
    cat_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    scaler = None
    if scale_numeric:
        scaler = StandardScaler()
        scaler.fit(frames[0][NUMERIC_FEATURES].astype(np.float64))

    matrices: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for frame in frames:
        numeric_frame = frame[NUMERIC_FEATURES].astype(np.float64)
        if scaler is not None:
            numeric = scaler.transform(numeric_frame)
        else:
            numeric = numeric_frame.to_numpy(dtype=np.float64)
        categorical = encoder.transform(frame[CATEGORICAL_FEATURES])
        matrix = pd.DataFrame(
            np.column_stack([numeric, categorical]),
            columns=NUMERIC_FEATURES + cat_names,
        )
        _assert_safe(matrix.columns.tolist())
        matrices.append(matrix)
        targets.append(frame["readmitted_30d"].to_numpy(dtype=int))

    return Design(
        matrices=matrices,
        targets=targets,
        keep_specialties=sorted(keep),
        encoder=encoder,
    )
