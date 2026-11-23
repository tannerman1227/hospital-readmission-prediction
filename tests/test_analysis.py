"""Unit tests that do not download the UCI extract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission.business import (
    break_even_precision,
    random_targeting_outcome,
    targeting_outcome,
)
from readmission.cleaning import icd9_group, prepare_cohort
from readmission.config import SQL_PATH
from readmission.evaluation import risk_deciles
from readmission.fairness import fairness_audit
from readmission.features import FORBIDDEN_COLUMNS, fit_design
from readmission.modeling import cross_validated_auc, fit_scored_model
from readmission.splitting import split_by_patient
from readmission.sql_analysis import load_named_queries, run_cohort_queries
from tests.synthetic import make_raw

TINY_BOOSTING = {"n_estimators": 15, "num_leaves": 7, "min_child_samples": 5}


def test_icd9_groups_follow_the_strack_map():
    assert icd9_group("250.83") == "Diabetes"
    assert icd9_group("250") == "Diabetes"
    assert icd9_group(250.4) == "Diabetes"
    assert icd9_group("239") == "Neoplasms"
    assert icd9_group("240") == "Other"
    assert icd9_group("410") == "Circulatory"
    assert icd9_group("785") == "Circulatory"
    assert icd9_group("486") == "Respiratory"
    assert icd9_group("786") == "Respiratory"
    assert icd9_group("520") == "Digestive"
    assert icd9_group("787") == "Digestive"
    assert icd9_group("580") == "Genitourinary"
    assert icd9_group("788") == "Genitourinary"
    assert icd9_group("710") == "Musculoskeletal"
    assert icd9_group("800") == "Injury"
    assert icd9_group("V58.67") == "Other"
    assert icd9_group("E930") == "Other"
    assert icd9_group("?") == "Missing"
    assert icd9_group(None) == "Missing"
    assert icd9_group("") == "Missing"


def test_cohort_drops_death_hospice_still_patient_and_unknown_gender():
    raw = make_raw()
    cohort, log = prepare_cohort(raw)
    assert log["removed_expired_hospice_or_still_patient"] == 3
    assert log["removed_unknown_gender"] == 1
    assert len(cohort) == len(raw) - 4
    assert set(cohort["readmitted_30d"]).issubset({0, 1})
    assert "weight" not in cohort.columns
    assert "payer_code" not in cohort.columns
    assert (cohort["race"] == "Missing").any()
    assert not (cohort["race"] == "?").any()
    adjusted = cohort.loc[cohort["metformin_status"] == "Adjusted"]
    assert len(adjusted) > 0
    assert cohort["encounter_id"].is_unique


def test_patient_split_has_no_overlap_and_preserves_rows():
    cohort, _ = prepare_cohort(make_raw())
    parts = split_by_patient(cohort, seed=1)
    ids = [set(parts[name]["patient_nbr"]) for name in ("fit", "calib", "test")]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(parts["fit"]) + len(parts["calib"]) + len(parts["test"]) == len(cohort)
    assert len(parts["dev"]) == len(parts["fit"]) + len(parts["calib"])


def test_design_matrix_excludes_protected_attributes_and_the_target():
    cohort, _ = prepare_cohort(make_raw())
    parts = split_by_patient(cohort, seed=2)
    design = fit_design(parts["fit"], parts["test"], min_specialty_count=1)
    columns = set(design.matrices[0].columns)
    assert columns.isdisjoint(FORBIDDEN_COLUMNS)
    assert not any(column.startswith(("race_", "gender_")) for column in columns)
    assert "number_inpatient" in columns
    assert len(design.matrices[0]) == len(parts["fit"])
    assert len(design.matrices[1]) == len(parts["test"])
    assert design.targets[0].sum() > 0


def test_sql_file_runs_against_the_cohort():
    cohort, _ = prepare_cohort(make_raw(n=200, seed=3))
    names = [name for name, _ in load_named_queries(SQL_PATH)]
    assert names[:3] == ["cohort_overview", "readmission_by_age", "readmission_by_primary_diagnosis"]
    results = run_cohort_queries(cohort, SQL_PATH)
    overview = results["cohort_overview"].iloc[0]
    assert int(overview["encounters"]) == len(cohort)
    assert int(overview["patients"]) == cohort["patient_nbr"].nunique()
    assert abs(float(overview["readmission_rate"]) - round(cohort["readmitted_30d"].mean(), 4)) < 1e-9
    assert set(results) == set(names)


def test_business_math_matches_a_hand_calculation():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    result = targeting_outcome(
        y,
        score,
        fraction=0.2,
        readmission_cost=16_000,
        intervention_cost=500,
        relative_risk_reduction=0.2,
        seed=0,
    )
    assert result["n_targeted"] == 2
    assert result["observed_readmissions"] == 2
    assert result["readmissions_prevented"] == 0.4
    assert result["gross_savings_usd"] == 6_400
    assert result["program_cost_usd"] == 1_000
    assert result["net_savings_usd"] == 5_400
    assert result["net_savings_per_1000_discharges_usd"] == 540_000
    assert result["break_even_intervention_cost_usd"] == 3_200
    assert break_even_precision(16_000, 500, 0.2) == 500 / (0.2 * 16_000)
    random = random_targeting_outcome(y, 0.2, 16_000, 500, 0.2)
    assert random["observed_readmissions"] == y.mean() * 2


def test_highest_decile_has_the_highest_mean_score():
    rng = np.random.default_rng(0)
    score = rng.random(200)
    y = (score > 0.8).astype(int)
    table = risk_deciles(y, score)
    assert table.iloc[0]["risk_decile"] == 1
    assert table.iloc[0]["mean_score"] > table.iloc[-1]["mean_score"]
    assert abs(table["share_of_readmissions"].sum() - 1) < 1e-9


def test_models_train_and_race_stays_out_of_the_score():
    cohort, _ = prepare_cohort(make_raw(n=480, seed=4))
    parts = split_by_patient(cohort, seed=4)
    cv = cross_validated_auc(parts["dev"], "gradient_boosting", n_splits=2, lgbm_params=TINY_BOOSTING)
    assert 0.5 <= cv["oof_auc"] <= 1
    scored = fit_scored_model(
        parts["fit"], parts["calib"], parts["test"], "gradient_boosting", lgbm_params=TINY_BOOSTING
    )
    assert scored["calibrated_test"].min() >= 0
    assert scored["calibrated_test"].max() <= 1
    audit = fairness_audit(
        parts["test"],
        scored["y_test"],
        scored["raw_test"],
        scored["calibrated_test"],
        fraction=0.1,
        seed=4,
    )
    assert {"race", "gender", "age"} == set(audit["group_type"])
    assert audit["n"].sum() == 3 * len(parts["test"])
    logistic = fit_scored_model(parts["fit"], parts["calib"], parts["test"], "logistic")
    assert len(logistic["raw_test"]) == len(parts["test"])


def test_charts_write_files(tmp_path):
    from readmission.plots import plot_deciles, plot_missingness

    missing = pd.DataFrame(
        {
            "field": ["Weight", "Race"],
            "missing_rate": [0.97, 0.02],
            "kind": ["Unknown code ('?')", "Unknown code ('?')"],
        }
    )
    deciles = pd.DataFrame(
        {
            "risk_decile": list(range(1, 11)),
            "readmission_rate": np.linspace(0.3, 0.05, 10),
            "lift": np.linspace(2.5, 0.4, 10),
        }
    )
    plot_missingness(missing, tmp_path / "missing.png")
    plot_deciles(deciles, 0.11, tmp_path / "deciles.png")
    assert (tmp_path / "missing.png").stat().st_size > 1000
    assert (tmp_path / "deciles.png").stat().st_size > 1000
