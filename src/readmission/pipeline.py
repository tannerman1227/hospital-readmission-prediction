"""End-to-end cohort build, validation, charts, and the business case."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import duckdb
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.metrics import roc_auc_score

from readmission.business import (
    best_fraction_on_calibration,
    break_even_precision,
    scenario_table,
    targeting_outcome,
    write_workbook,
)
from readmission.cleaning import missingness_table, prepare_cohort
from readmission.config import (
    DATASET_CITATION,
    DATASET_NAME,
    DATA_PROCESSED,
    DATA_URL,
    FIGURES,
    INTERVENTION_COST_USD,
    METRICS_PATH,
    N_BOOTSTRAP,
    PRIMARY_TARGET_FRACTION,
    READMISSION_COST_USD,
    RELATIVE_RISK_REDUCTION,
    SEED,
    TABLES,
    WORKBOOK_PATH,
    ZIP_SHA256,
)
from readmission.download import load_raw_dataset
from readmission.evaluation import clustered_auc_interval, discrimination_report, risk_deciles
from readmission.fairness import fairness_audit, recall_gap
from readmission.modeling import cross_validated_auc, fit_scored_model, shap_table
from readmission.plots import (
    plot_calibration,
    plot_deciles,
    plot_fairness,
    plot_missingness,
    plot_net_savings,
    plot_pr,
    plot_rate_bars,
    plot_roc,
    plot_shap,
)
from readmission.splitting import split_by_patient, split_summary
from readmission.sql_analysis import run_cohort_queries


def _log(message: str) -> None:
    print(message, flush=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _age_label(band: str) -> str:
    low = int(str(band).split("-")[0].replace("[", ""))
    return f"{low}–{low + 9}"


def _records(frame: pd.DataFrame) -> list[dict]:
    return _json_ready(frame.to_dict(orient="records"))


def run_pipeline(save: bool = True) -> dict[str, Any]:
    """Download the extract if needed, fit the models, and write reports."""
    _log("Loading the UCI diabetes extract...")
    raw = load_raw_dataset()
    cohort, cohort_log = prepare_cohort(raw)
    sql = run_cohort_queries(cohort)
    parts = split_by_patient(cohort)
    splits = split_summary(parts)
    _log(
        "Cohort "
        f"{cohort_log['encounters']:,} encounters / {cohort_log['patients']:,} patients; "
        f"test prevalence {splits['test']['prevalence']:.3f}."
    )

    _log("Cross-validating on development patients...")
    cv_logistic = cross_validated_auc(parts["dev"], "logistic")
    cv_boosting = cross_validated_auc(parts["dev"], "gradient_boosting")
    _log(
        f"OOF AUC  logistic {cv_logistic['oof_auc']:.3f}  "
        f"gradient boosting {cv_boosting['oof_auc']:.3f}"
    )

    _log("Fitting the final models and calibrating probabilities...")
    logistic = fit_scored_model(parts["fit"], parts["calib"], parts["test"], "logistic")
    boosting = fit_scored_model(parts["fit"], parts["calib"], parts["test"], "gradient_boosting")
    y_test = boosting["y_test"]
    if not np.array_equal(y_test, parts["test"]["readmitted_30d"].to_numpy()):
        raise RuntimeError("Test labels drifted from the design matrix.")

    prior_score = parts["test"]["number_inpatient"].to_numpy(dtype=float)
    groups = parts["test"]["patient_nbr"].to_numpy()
    test_reports = {
        "logistic_regression": discrimination_report(
            y_test, logistic["raw_test"], logistic["calibrated_test"]
        ),
        "gradient_boosting": discrimination_report(
            y_test, boosting["raw_test"], boosting["calibrated_test"]
        ),
    }
    test_reports["logistic_regression"].update(
        clustered_auc_interval(y_test, logistic["raw_test"], groups, n_boot=N_BOOTSTRAP, seed=SEED)
    )
    test_reports["gradient_boosting"].update(
        clustered_auc_interval(y_test, boosting["raw_test"], groups, n_boot=N_BOOTSTRAP, seed=SEED)
    )
    prior_auc = float(roc_auc_score(y_test, prior_score))

    _log("Scoring outreach policies and SHAP...")
    score_map = {
        "gradient_boosting": boosting["raw_test"],
        "logistic_regression": logistic["raw_test"],
        "prior_inpatient": prior_score,
    }
    scenarios = scenario_table(y_test, score_map, seed=SEED)
    chosen_fraction = best_fraction_on_calibration(
        boosting["y_calib"], boosting["raw_calib"], seed=SEED
    )
    chosen_on_test = targeting_outcome(
        y_test,
        boosting["raw_test"],
        chosen_fraction,
        seed=SEED + int(chosen_fraction * 1000),
    )
    deciles = risk_deciles(y_test, boosting["raw_test"])
    importance = shap_table(boosting["model"], boosting["x_test"])
    audit = fairness_audit(
        parts["test"],
        y_test,
        boosting["raw_test"],
        boosting["calibrated_test"],
        fraction=PRIMARY_TARGET_FRACTION,
        seed=SEED + int(PRIMARY_TARGET_FRACTION * 1000),
    )
    gaps = {
        "race": recall_gap(audit, "race", exclude={"Missing"}),
        "gender": recall_gap(audit, "gender"),
        "age": recall_gap(audit, "age"),
    }

    assumptions = {
        "readmission_cost_usd": READMISSION_COST_USD,
        "intervention_cost_usd": INTERVENTION_COST_USD,
        "relative_risk_reduction": RELATIVE_RISK_REDUCTION,
        "primary_target_fraction": PRIMARY_TARGET_FRACTION,
        "break_even_precision": break_even_precision(),
        "test_encounters": int(len(y_test)),
        "test_readmissions": int(y_test.sum()),
        "calculator_rows": [
            {
                "policy": str(row.policy),
                "target_fraction": float(row.target_fraction),
                "n_targeted": int(row.n_targeted),
                "observed_readmissions": float(row.observed_readmissions),
            }
            for row in scenarios.itertuples(index=False)
        ],
    }

    metrics: dict[str, Any] = {
        "dataset": {
            "name": DATASET_NAME,
            "url": DATA_URL,
            "zip_sha256": ZIP_SHA256,
            "citation": DATASET_CITATION,
            "prediction_point": "At discharge, for a completed inpatient encounter.",
            "target": "readmitted == '<30' versus '>30' or 'NO'.",
        },
        "cohort": cohort_log,
        "split": splits,
        "validation": {
            "protocol": (
                "Patients are split 60% fit / 20% calibration / 20% test, stratified on "
                "whether the patient has any 30-day readmission. Five-fold GroupKFold AUC "
                "is computed on the development patients (fit + calibration), with encoders "
                "refit inside each fold. The test patients are scored once. Isotonic "
                "calibration is fit on the calibration patients. The outreach panel is the "
                "pre-specified top 10% by raw model score; a second capacity is chosen by "
                "maximizing net savings on the calibration set and then scored on the test set."
            ),
            "logistic_regression": cv_logistic,
            "gradient_boosting": cv_boosting,
        },
        "test": {
            "prior_inpatient_auc": prior_auc,
            "logistic_regression": test_reports["logistic_regression"],
            "gradient_boosting": test_reports["gradient_boosting"],
        },
        "assumptions": {key: value for key, value in assumptions.items() if key != "calculator_rows"},
        "calibration_chosen_fraction": chosen_fraction,
        "calibration_chosen_test": chosen_on_test,
        "deciles_gradient_boosting": _records(deciles),
        "scenarios": _records(scenarios),
        "shap_top": _records(importance.head(15)),
        "fairness": {"gaps": gaps, "groups": _records(audit)},
        "sql": {name: _records(frame) for name, frame in sql.items()},
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "shap": shap.__version__,
            "duckdb": duckdb.__version__,
        },
    }
    operating = {}
    primary = scenarios.loc[np.isclose(scenarios["target_fraction"], PRIMARY_TARGET_FRACTION)]
    for _, row in primary.iterrows():
        operating[str(row["policy"])] = _json_ready(row.drop(labels=["policy"]).to_dict())
    metrics["operating_point"] = {"fraction": PRIMARY_TARGET_FRACTION, "policies": operating}

    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        TABLES.mkdir(parents=True, exist_ok=True)
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        for name, frame in sql.items():
            frame.to_csv(TABLES / f"sql_{name}.csv", index=False)
        scenarios.to_csv(TABLES / "targeting_scenarios.csv", index=False)
        deciles.to_csv(TABLES / "risk_deciles.csv", index=False)
        audit.to_csv(TABLES / "fairness_audit.csv", index=False)
        importance.to_csv(TABLES / "shap_importance.csv", index=False)
        _save_figures(
            raw=raw,
            sql=sql,
            cohort_log=cohort_log,
            y_test=y_test,
            logistic=logistic,
            boosting=boosting,
            prior_score=prior_score,
            deciles=deciles,
            scenarios=scenarios,
            importance=importance,
            audit=audit,
        )
        model_rows = pd.DataFrame(
            [
                {"model": "Prior inpatient stays", "test_auc": prior_auc},
                {
                    "model": "Logistic regression",
                    "cv_auc": cv_logistic["oof_auc"],
                    "test_auc": test_reports["logistic_regression"]["roc_auc"],
                    "test_auc_ci95_low": test_reports["logistic_regression"]["auc_ci95_low"],
                    "test_auc_ci95_high": test_reports["logistic_regression"]["auc_ci95_high"],
                    "pr_auc": test_reports["logistic_regression"]["pr_auc"],
                    "brier_calibrated": test_reports["logistic_regression"]["brier_calibrated"],
                    "ece_calibrated": test_reports["logistic_regression"]["ece_calibrated"],
                },
                {
                    "model": "Gradient boosting",
                    "cv_auc": cv_boosting["oof_auc"],
                    "test_auc": test_reports["gradient_boosting"]["roc_auc"],
                    "test_auc_ci95_low": test_reports["gradient_boosting"]["auc_ci95_low"],
                    "test_auc_ci95_high": test_reports["gradient_boosting"]["auc_ci95_high"],
                    "pr_auc": test_reports["gradient_boosting"]["pr_auc"],
                    "brier_calibrated": test_reports["gradient_boosting"]["brier_calibrated"],
                    "ece_calibrated": test_reports["gradient_boosting"]["ece_calibrated"],
                },
            ]
        )
        write_workbook(
            WORKBOOK_PATH,
            scenarios=scenarios,
            deciles=deciles,
            fairness=audit,
            model_rows=model_rows,
            assumptions=assumptions,
        )
        METRICS_PATH.write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
        metrics["artifacts"] = {
            "metrics": str(METRICS_PATH),
            "workbook": str(WORKBOOK_PATH),
            "figures": str(FIGURES),
            "tables": str(TABLES),
        }
        _log(f"Wrote {METRICS_PATH}")
    return {
        "metrics": _json_ready(metrics),
        "cohort": cohort,
        "sql": sql,
        "scenarios": scenarios,
        "deciles": deciles,
        "audit": audit,
        "importance": importance,
    }


def _save_figures(
    raw: pd.DataFrame,
    sql: dict[str, pd.DataFrame],
    cohort_log: dict,
    y_test: np.ndarray,
    logistic: dict,
    boosting: dict,
    prior_score: np.ndarray,
    deciles: pd.DataFrame,
    scenarios: pd.DataFrame,
    importance: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    base = float(cohort_log["readmission_rate"])
    figures = [
        plot_missingness(missingness_table(raw), FIGURES / "01_missingness.png"),
    ]
    age = sql["readmission_by_age"].copy()
    age["label"] = age["age"].map(_age_label)
    figures.append(
        plot_rate_bars(
            age["label"].tolist(),
            age["readmission_rate"].tolist(),
            age["encounters"].astype(int).tolist(),
            FIGURES / "02_readmission_by_age.png",
            "Readmission peaks in the 20s, then edges up after 60",
            "Age band",
            base_rate=base,
        )
    )
    diagnosis = sql["readmission_by_primary_diagnosis"]
    diagnosis = diagnosis.loc[diagnosis["encounters"] >= 100].sort_values("readmission_rate")
    figures.append(
        plot_rate_bars(
            diagnosis["primary_diagnosis"].tolist(),
            diagnosis["readmission_rate"].tolist(),
            diagnosis["encounters"].astype(int).tolist(),
            FIGURES / "03_readmission_by_diagnosis.png",
            "Primary diagnosis group and 30-day readmission",
            "",
            horizontal=True,
            base_rate=base,
        )
    )
    prior = sql["readmission_by_prior_inpatient"].copy()
    prior["_order"] = prior["prior_inpatient_stays"].map({"0": 0, "1": 1, "2": 2, "3": 3, "4+": 4})
    prior = prior.sort_values("_order")
    figures.append(
        plot_rate_bars(
            prior["prior_inpatient_stays"].tolist(),
            prior["readmission_rate"].tolist(),
            prior["encounters"].astype(int).tolist(),
            FIGURES / "04_prior_inpatient.png",
            "A recent inpatient stay is the clearest risk marker",
            "Inpatient stays in the prior year",
            base_rate=base,
        )
    )
    a1c_order = ["NotMeasured", "Normal", "Above7", "Above8"]
    a1c_labels = {
        "NotMeasured": "Not measured",
        "Normal": "Normal",
        "Above7": "Above 7%",
        "Above8": "Above 8%",
    }
    a1c = sql["readmission_by_a1c"].copy()
    a1c["_order"] = a1c["a1c_result"].map({name: i for i, name in enumerate(a1c_order)})
    a1c = a1c.sort_values("_order")
    figures.append(
        plot_rate_bars(
            [a1c_labels.get(value, value) for value in a1c["a1c_result"]],
            a1c["readmission_rate"].tolist(),
            a1c["encounters"].astype(int).tolist(),
            FIGURES / "05_a1c.png",
            "Stays with a recorded HbA1c have a slightly lower raw rate",
            "",
            base_rate=base,
        )
    )
    ranking = {
        "prior_inpatient": (y_test, prior_score),
        "logistic_regression": (y_test, logistic["raw_test"]),
        "gradient_boosting": (y_test, boosting["raw_test"]),
    }
    figures.append(plot_roc(ranking, FIGURES / "06_roc.png"))
    figures.append(plot_pr(ranking, float(y_test.mean()), FIGURES / "07_precision_recall.png"))
    figures.append(
        plot_calibration(
            y_test,
            {
                "gb_raw": boosting["raw_test"],
                "gb_calibrated": boosting["calibrated_test"],
                "lr_calibrated": logistic["calibrated_test"],
            },
            FIGURES / "08_calibration.png",
        )
    )
    figures.append(
        plot_deciles(deciles, float(y_test.mean()), FIGURES / "09_risk_deciles.png")
    )
    figures.append(plot_shap(importance, FIGURES / "10_shap.png"))
    figures.append(
        plot_net_savings(scenarios, PRIMARY_TARGET_FRACTION, FIGURES / "11_net_savings.png")
    )
    figures.append(plot_fairness(audit, FIGURES / "12_fairness.png"))
    for fig in figures:
        plt.close(fig)
