"""Cohort rules, missing-code handling, and ICD-9 grouping.

The prediction is made at discharge. In-stay labs, medications, and the
discharge plan are known then. Patients who died or went to hospice, and
encounters that never completed a discharge, are removed because they are
not in the risk set for a 30-day readmission.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# Discharge dispositions that leave the patient outside the readmission risk set.
# IDs follow UCI IDs_mapping.csv: expired, hospice, and still a patient.
EXCLUDED_DISCHARGE_IDS = {11, 12, 13, 14, 19, 20, 21}

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

AGE_MIDPOINT = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

ADMISSION_TYPE = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
}

# Admission-source IDs from IDs_mapping.csv, collapsed to four operational groups.
ADMISSION_SOURCE = {
    7: "Emergency room",
    1: "Referral",
    2: "Referral",
    3: "Referral",
    4: "Transfer",
    5: "Transfer",
    6: "Transfer",
    10: "Transfer",
    18: "Transfer",
    22: "Transfer",
    25: "Transfer",
    26: "Transfer",
}

HOME_DISCHARGE = {1}
HOME_HEALTH_DISCHARGE = {6, 8}
FACILITY_DISCHARGE = {3, 4, 23, 24}
TRANSFER_DISCHARGE = {2, 5, 9, 10, 15, 16, 17, 22, 27, 28, 29, 30}
AMA_DISCHARGE = {7}


def icd9_group(code: object) -> str:
    """Map an ICD-9 code to the clinical groups used by Strack et al. (2014)."""
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return "Missing"
    text = str(code).strip()
    if text in {"", "?", "nan", "None", "<NA>", "NaN"}:
        return "Missing"
    if text[0] in {"V", "E", "v", "e"}:
        return "Other"
    try:
        value = float(text)
    except ValueError:
        return "Other"
    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 140 <= value <= 239:
        return "Neoplasms"
    if 800 <= value <= 999:
        return "Injury"
    return "Other"


def _question_mark_rate(series: pd.Series) -> float:
    return float((series.astype(str) == "?").mean())


MISSINGNESS_FIELDS = [
    ("weight", "Weight", "Unknown code ('?')"),
    ("max_glu_serum", "Max serum glucose", "Test not recorded"),
    ("A1Cresult", "HbA1c", "Test not recorded"),
    ("medical_specialty", "Medical specialty", "Unknown code ('?')"),
    ("payer_code", "Payer code", "Unknown code ('?')"),
    ("race", "Race", "Unknown code ('?')"),
    ("diag_3", "Third diagnosis", "Unknown code ('?')"),
    ("diag_2", "Second diagnosis", "Unknown code ('?')"),
    ("diag_1", "Primary diagnosis", "Unknown code ('?')"),
]


def missingness_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Share of unusable codes in the published extract, before cohort filters."""
    rows = []
    for column, label, kind in MISSINGNESS_FIELDS:
        series = raw[column]
        if kind.startswith("Test"):
            rate = float(series.isna().mean())
        else:
            rate = _question_mark_rate(series)
        rows.append({"field": label, "missing_rate": rate, "kind": kind})
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=True).reset_index(drop=True)


def _discharge_group(disposition_id: int) -> str:
    if disposition_id in HOME_DISCHARGE:
        return "Home"
    if disposition_id in HOME_HEALTH_DISCHARGE:
        return "Home health"
    if disposition_id in FACILITY_DISCHARGE:
        return "Nursing facility"
    if disposition_id in TRANSFER_DISCHARGE:
        return "Transfer"
    if disposition_id in AMA_DISCHARGE:
        return "Left against advice"
    return "Unknown"


def _med_status(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    adjusted = text.isin(["Up", "Down"])
    out = np.where(adjusted, "Adjusted", text)
    out = np.where(pd.Series(out).isin(["No", "Steady", "Adjusted"]), out, "No")
    return pd.Series(out, index=series.index)


def _lab_result(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    text = series.astype(str)
    mapped = text.map(mapping)
    missing = series.isna() | text.isin(["?", "nan", "None", "<NA>", "NaN", ""])
    mapped = mapped.where(~missing, "NotMeasured")
    mapped = mapped.fillna("Other")
    return mapped


def prepare_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the analysis cohort and a count log of every exclusion."""
    required = {
        "encounter_id",
        "patient_nbr",
        "race",
        "gender",
        "age",
        "weight",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "payer_code",
        "medical_specialty",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "diag_1",
        "diag_2",
        "diag_3",
        "number_diagnoses",
        "max_glu_serum",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed",
        "readmitted",
        *MEDICATION_COLUMNS,
    }
    missing_cols = sorted(required - set(raw.columns))
    if missing_cols:
        raise ValueError(f"Raw extract is missing columns: {missing_cols}")

    frame = raw.copy()
    log: dict[str, Any] = {
        "raw_encounters": int(len(frame)),
        "raw_patients": int(frame["patient_nbr"].nunique()),
        "missing_rate_weight": _question_mark_rate(frame["weight"]),
        "missing_rate_payer_code": _question_mark_rate(frame["payer_code"]),
        "missing_rate_medical_specialty": _question_mark_rate(frame["medical_specialty"]),
        "missing_rate_race": _question_mark_rate(frame["race"]),
        "missing_rate_a1c": float(frame["A1Cresult"].isna().mean()),
        "missing_rate_glucose": float(frame["max_glu_serum"].isna().mean()),
    }

    excluded = frame["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)
    log["removed_expired_hospice_or_still_patient"] = int(excluded.sum())
    log["removed_excluded_but_coded_readmit_30d"] = int(
        (excluded & frame["readmitted"].astype(str).eq("<30")).sum()
    )
    frame = frame.loc[~excluded].copy()

    unknown_gender = frame["gender"].astype(str).isin(["Unknown/Invalid", "?", "nan"])
    log["removed_unknown_gender"] = int(unknown_gender.sum())
    frame = frame.loc[~unknown_gender].copy()

    frame["race"] = frame["race"].astype(str).replace({"?": "Missing"})
    frame["medical_specialty"] = frame["medical_specialty"].astype(str).replace({"?": "Missing"})
    frame["age_mid"] = frame["age"].map(AGE_MIDPOINT).astype(float)
    if frame["age_mid"].isna().any():
        unknown_ages = sorted(frame.loc[frame["age_mid"].isna(), "age"].astype(str).unique())
        raise ValueError(f"Unmapped age bands: {unknown_ages}")

    frame["admission_type"] = (
        frame["admission_type_id"].map(ADMISSION_TYPE).fillna("Other").astype(str)
    )
    frame["admission_source_group"] = (
        frame["admission_source_id"].map(ADMISSION_SOURCE).fillna("Other").astype(str)
    )
    frame["discharge_group"] = frame["discharge_disposition_id"].map(_discharge_group).astype(str)

    for source, target in (("diag_1", "diag_1_group"), ("diag_2", "diag_2_group"), ("diag_3", "diag_3_group")):
        frame[target] = frame[source].map(icd9_group)

    frame["a1c_result"] = _lab_result(
        frame["A1Cresult"], {">8": "Above8", ">7": "Above7", "Norm": "Normal"}
    )
    frame["max_glu_serum"] = _lab_result(
        frame["max_glu_serum"], {">300": "Above300", ">200": "Above200", "Norm": "Normal"}
    )
    frame["insulin"] = frame["insulin"].astype(str)
    frame["metformin_status"] = _med_status(frame["metformin"])

    meds = frame[MEDICATION_COLUMNS].astype(str)
    frame["n_diabetes_meds"] = meds.ne("No").sum(axis=1).astype(int)
    frame["n_med_changes"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(int)
    frame["readmitted_30d"] = frame["readmitted"].astype(str).eq("<30").astype(int)

    keep = [
        "encounter_id",
        "patient_nbr",
        "race",
        "gender",
        "age",
        "age_mid",
        "admission_type",
        "discharge_group",
        "admission_source_group",
        "time_in_hospital",
        "medical_specialty",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
        "number_diagnoses",
        "max_glu_serum",
        "a1c_result",
        "insulin",
        "metformin_status",
        "n_diabetes_meds",
        "n_med_changes",
        "readmitted_30d",
    ]
    cohort = frame[keep].copy()
    cohort = cohort.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)

    log["encounters"] = int(len(cohort))
    log["patients"] = int(cohort["patient_nbr"].nunique())
    log["readmissions_30d"] = int(cohort["readmitted_30d"].sum())
    log["readmission_rate"] = float(cohort["readmitted_30d"].mean())
    log["dropped_columns"] = ["weight", "payer_code"]
    log["exclusion_reason"] = (
        "Removed discharge dispositions 11, 13, 14, 19, 20, and 21 (expired or hospice) "
        "and 12 (still a patient), plus the three encounters with unknown gender. "
        "Weight and payer code are dropped because they are missing on most rows. "
        "Race and gender are retained for the fairness audit and are not model features."
    )
    return cohort, log
