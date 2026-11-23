"""Small encounter extract with the published column names."""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission.cleaning import MEDICATION_COLUMNS

AGES = ["[20-30)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]
RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]
SPECIALTIES = ["InternalMedicine", "Cardiology", "?", "Surgery-General", "Nephrology"]


def make_raw(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    patient = np.arange(n) // 2
    prior_inpatient = rng.integers(0, 5, size=n)
    readmit_prob = 0.08 + 0.12 * prior_inpatient
    readmit_30 = rng.random(n) < readmit_prob
    readmitted = np.where(readmit_30, "<30", np.where(rng.random(n) < 0.4, ">30", "NO"))
    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(1, n + 1),
            "patient_nbr": patient,
            "race": rng.choice(RACES, size=n),
            "gender": rng.choice(["Female", "Male"], size=n),
            "age": rng.choice(AGES, size=n),
            "weight": "?",
            "admission_type_id": rng.choice([1, 2, 3, 5], size=n),
            "discharge_disposition_id": rng.choice([1, 1, 3, 6, 7, 18], size=n),
            "admission_source_id": rng.choice([1, 7, 4], size=n),
            "time_in_hospital": rng.integers(1, 10, size=n),
            "payer_code": "?",
            "medical_specialty": rng.choice(SPECIALTIES, size=n),
            "num_lab_procedures": rng.integers(1, 80, size=n),
            "num_procedures": rng.integers(0, 6, size=n),
            "num_medications": rng.integers(1, 30, size=n),
            "number_outpatient": rng.integers(0, 4, size=n),
            "number_emergency": rng.integers(0, 3, size=n),
            "number_inpatient": prior_inpatient,
            "diag_1": rng.choice(["250.01", "410", "486", "428", "786", "V58"], size=n),
            "diag_2": rng.choice(["250.01", "401", "?", "780"], size=n),
            "diag_3": rng.choice(["250.02", "?", "585"], size=n),
            "number_diagnoses": rng.integers(1, 9, size=n),
            "max_glu_serum": rng.choice(["Norm", ">200", ">300", None], size=n),
            "A1Cresult": rng.choice(["Norm", ">7", ">8", None], size=n),
            "change": rng.choice(["No", "Ch"], size=n),
            "diabetesMed": rng.choice(["Yes", "No"], size=n),
            "readmitted": readmitted,
        }
    )
    for column in MEDICATION_COLUMNS:
        frame[column] = "No"
    frame.loc[frame.index[::3], "insulin"] = "Up"
    frame.loc[frame.index[1::4], "metformin"] = "Down"
    frame.loc[0, "discharge_disposition_id"] = 11
    frame.loc[1, "discharge_disposition_id"] = 12
    frame.loc[2, "discharge_disposition_id"] = 13
    frame.loc[3, "gender"] = "Unknown/Invalid"
    frame.loc[4, "race"] = "?"
    return frame
