"""Patient-level splits so the same person never sits in two sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from readmission.config import CALIB_SIZE, SEED, TEST_SIZE


def split_by_patient(
    cohort: pd.DataFrame,
    test_size: float = TEST_SIZE,
    calib_size: float = CALIB_SIZE,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Split encounters by ``patient_nbr``.

    Patients are stratified on whether any of their encounters was a 30-day
    readmission. ``calib_size`` is the share of the development patients used
    only to calibrate probabilities. The development frame (fit + calibration)
    is what cross-validation sees. The test frame is touched once.
    """
    patient_label = cohort.groupby("patient_nbr")["readmitted_30d"].max()
    patients = patient_label.index.to_numpy()
    labels = patient_label.to_numpy()
    dev_ids, test_ids = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    dev_labels = patient_label.loc[dev_ids].to_numpy()
    fit_ids, calib_ids = train_test_split(
        dev_ids,
        test_size=calib_size,
        random_state=seed,
        stratify=dev_labels,
    )

    def _take(ids: np.ndarray) -> pd.DataFrame:
        chosen = set(ids.tolist())
        part = cohort.loc[cohort["patient_nbr"].isin(chosen)]
        return part.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)

    fit = _take(np.asarray(fit_ids))
    calib = _take(np.asarray(calib_ids))
    test = _take(np.asarray(test_ids))
    dev = (
        pd.concat([fit, calib], ignore_index=True)
        .sort_values(["patient_nbr", "encounter_id"])
        .reset_index(drop=True)
    )
    sets = [set(fit["patient_nbr"]), set(calib["patient_nbr"]), set(test["patient_nbr"])]
    if not (sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])):
        raise RuntimeError("Patient overlap across fit, calibration, and test.")
    return {"fit": fit, "calib": calib, "dev": dev, "test": test}


def split_summary(parts: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    summary = {}
    for name in ("fit", "calib", "dev", "test"):
        frame = parts[name]
        summary[name] = {
            "encounters": int(len(frame)),
            "patients": int(frame["patient_nbr"].nunique()),
            "readmissions_30d": int(frame["readmitted_30d"].sum()),
            "prevalence": float(frame["readmitted_30d"].mean()),
        }
    return summary
