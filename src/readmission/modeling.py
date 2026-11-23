"""Logistic regression baseline, gradient boosting, calibration, and SHAP."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from readmission.config import CV_FOLDS, LGBM_PARAMS, SEED, SHAP_SAMPLE
from readmission.features import fit_design


def _logistic() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=SEED,
    )


def _lgbm(y: np.ndarray, params: dict | None = None) -> LGBMClassifier:
    positive = max(int(np.asarray(y).sum()), 1)
    negative = len(y) - positive
    settings = dict(LGBM_PARAMS)
    if params:
        settings.update(params)
    settings["scale_pos_weight"] = negative / positive
    return LGBMClassifier(**settings)


def fit_isotonic(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(scores, dtype=float), np.asarray(y, dtype=int))
    return model


def _predict(model, matrix: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(matrix)[:, 1]


def cross_validated_auc(
    dev: pd.DataFrame,
    kind: str,
    n_splits: int = CV_FOLDS,
    lgbm_params: dict | None = None,
) -> dict:
    """Out-of-fold AUC on development patients. Encoders refit inside each fold."""
    if kind not in {"logistic", "gradient_boosting"}:
        raise ValueError(kind)
    splitter = GroupKFold(n_splits=n_splits)
    y_all = dev["readmitted_30d"].to_numpy(dtype=int)
    groups = dev["patient_nbr"].to_numpy()
    oof = np.zeros(len(dev), dtype=float)
    fold_aucs: list[float] = []
    for train_idx, valid_idx in splitter.split(dev, y_all, groups):
        design = fit_design(
            dev.iloc[train_idx],
            dev.iloc[valid_idx],
            scale_numeric=(kind == "logistic"),
        )
        x_train, x_valid = design.matrices
        y_train, y_valid = design.targets
        model = _logistic() if kind == "logistic" else _lgbm(y_train, lgbm_params)
        model.fit(x_train, y_train)
        scores = _predict(model, x_valid)
        oof[valid_idx] = scores
        fold_aucs.append(float(roc_auc_score(y_valid, scores)))
    return {
        "fold_auc": fold_aucs,
        "mean": float(np.mean(fold_aucs)),
        "std": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "oof_auc": float(roc_auc_score(y_all, oof)),
    }


def fit_scored_model(
    fit: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    kind: str,
    lgbm_params: dict | None = None,
) -> dict:
    """Fit on the fit split, calibrate on calibration patients, score the test patients."""
    design = fit_design(fit, calib, test, scale_numeric=(kind == "logistic"))
    x_fit, x_calib, x_test = design.matrices
    y_fit, y_calib, y_test = design.targets
    model = _logistic() if kind == "logistic" else _lgbm(y_fit, lgbm_params)
    model.fit(x_fit, y_fit)
    raw_calib = _predict(model, x_calib)
    raw_test = _predict(model, x_test)
    calibrator = fit_isotonic(raw_calib, y_calib)
    return {
        "model": model,
        "design": design,
        "x_test": x_test,
        "y_fit": y_fit,
        "y_calib": y_calib,
        "y_test": y_test,
        "raw_calib": raw_calib,
        "raw_test": raw_test,
        "calibrated_calib": calibrator.predict(raw_calib),
        "calibrated_test": calibrator.predict(raw_test),
        "keep_specialties": design.keep_specialties,
    }


def _positive_class_shap(values) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[:, :, 1]
    if array.ndim == 3 and array.shape[0] == 2:
        return array[1]
    return array


def shap_table(model, matrix: pd.DataFrame, sample_size: int = SHAP_SAMPLE, seed: int = SEED) -> pd.DataFrame:
    """Mean absolute SHAP on a seeded sample of the test matrix."""
    rng = np.random.default_rng(seed)
    take = rng.choice(len(matrix), size=min(sample_size, len(matrix)), replace=False)
    sample = matrix.iloc[np.sort(take)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        values = _positive_class_shap(explainer.shap_values(sample))
    if values.shape != (len(sample), sample.shape[1]):
        raise RuntimeError(f"Unexpected SHAP shape {values.shape} for matrix {sample.shape}.")
    rows = []
    for index, feature in enumerate(sample.columns):
        column = sample.iloc[:, index].to_numpy(dtype=float)
        attribution = values[:, index]
        if np.nanstd(column) < 1e-8 or np.nanstd(attribution) < 1e-8:
            direction = 0
        else:
            direction = int(np.sign(np.corrcoef(column, attribution)[0, 1]))
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.abs(attribution).mean()),
                "direction": direction,
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return table
