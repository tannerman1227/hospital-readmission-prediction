"""Paths, locked modeling choices, and planning assumptions.

Cost and effectiveness figures are planning assumptions for the business case.
They are not estimated from this dataset. Change them here and rerun the
analysis, or edit the matching cells in ``reports/business_case.xlsx``.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
SQL_PATH = ROOT / "sql" / "cohort_queries.sql"
METRICS_PATH = REPORTS / "metrics.json"
WORKBOOK_PATH = REPORTS / "business_case.xlsx"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
ZIP_SHA256 = "66d9820c3bd5fd2abf7957bda2d9714b2de1e7bba1ac1f1ff8fa85e83c22cb33"
EXPECTED_ROWS = 101_766
EXPECTED_COLUMNS = 50

DATASET_NAME = "Diabetes 130-US Hospitals for Years 1999-2008"
DATASET_CITATION = (
    "Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, "
    "Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, "
    '"Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of '
    '70,000 Clinical Database Patient Records," BioMed Research International, '
    "2014. UCI Machine Learning Repository, DOI 10.24432/C5230J, CC BY 4.0. "
    "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
)

SEED = 42
TEST_SIZE = 0.20
# Share of development patients reserved to calibrate probabilities.
CALIB_SIZE = 0.25
CV_FOLDS = 5
N_BOOTSTRAP = 400
SHAP_SAMPLE = 2_000
MIN_SPECIALTY_COUNT = 300
FAIRNESS_MIN_N = 150

# Pre-specified outreach capacity. Chosen before looking at test results.
PRIMARY_TARGET_FRACTION = 0.10
TARGET_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

# Planning assumptions for a post-discharge care-management bundle.
READMISSION_COST_USD = 16_000
INTERVENTION_COST_USD = 500
RELATIVE_RISK_REDUCTION = 0.20

LGBM_PARAMS = {
    "n_estimators": 250,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "n_jobs": 1,
    "verbosity": -1,
    "force_row_wise": True,
}
