"""Cost of targeting the highest-risk discharges with a care-management bundle.

Net savings use observed 30-day readmissions inside the targeted group and a
constant relative-risk reduction. That reduction is an assumption, not an
effect estimated from these rows. The workbook repeats the arithmetic as
formulas so the costs can be edited without rerunning the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from readmission.config import (
    INTERVENTION_COST_USD,
    PRIMARY_TARGET_FRACTION,
    READMISSION_COST_USD,
    RELATIVE_RISK_REDUCTION,
    TARGET_FRACTIONS,
)
from readmission.evaluation import tie_broken_scores


def break_even_precision(
    readmission_cost: float = READMISSION_COST_USD,
    intervention_cost: float = INTERVENTION_COST_USD,
    relative_risk_reduction: float = RELATIVE_RISK_REDUCTION,
) -> float:
    """Positive predictive value at which the bundle pays for itself."""
    return float(intervention_cost / (relative_risk_reduction * readmission_cost))


def _panel_size(n: int, fraction: float) -> int:
    return max(1, int(round(n * fraction)))


def outcome_from_observed(
    n: int,
    k: int,
    observed_readmissions: float,
    total_readmissions: float,
    readmission_cost: float,
    intervention_cost: float,
    relative_risk_reduction: float,
) -> dict[str, float]:
    prevented = observed_readmissions * relative_risk_reduction
    gross = prevented * readmission_cost
    program = k * intervention_cost
    net = gross - program
    precision = observed_readmissions / k if k else None
    recall = observed_readmissions / total_readmissions if total_readmissions else None
    max_intervention = (
        precision * relative_risk_reduction * readmission_cost if precision is not None else None
    )
    return {
        "n": int(n),
        "n_targeted": int(k),
        "fraction_targeted": k / n if n else None,
        "observed_readmissions": float(observed_readmissions),
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "readmissions_prevented": float(prevented),
        "gross_savings_usd": float(gross),
        "program_cost_usd": float(program),
        "net_savings_usd": float(net),
        "net_savings_per_1000_discharges_usd": float(net / n * 1000) if n else None,
        "break_even_intervention_cost_usd": float(max_intervention) if max_intervention is not None else None,
    }


def targeting_outcome(
    y: np.ndarray,
    score: np.ndarray,
    fraction: float,
    readmission_cost: float = READMISSION_COST_USD,
    intervention_cost: float = INTERVENTION_COST_USD,
    relative_risk_reduction: float = RELATIVE_RISK_REDUCTION,
    seed: int = 0,
) -> dict[str, float]:
    """Rank by ``score`` and score the top ``fraction`` against observed outcomes."""
    y = np.asarray(y, dtype=float)
    ranked = tie_broken_scores(score, seed=seed)
    n = len(y)
    k = _panel_size(n, fraction)
    chosen = np.argsort(-ranked, kind="mergesort")[:k]
    return outcome_from_observed(
        n=n,
        k=k,
        observed_readmissions=float(y[chosen].sum()),
        total_readmissions=float(y.sum()),
        readmission_cost=readmission_cost,
        intervention_cost=intervention_cost,
        relative_risk_reduction=relative_risk_reduction,
    )


def random_targeting_outcome(
    y: np.ndarray,
    fraction: float,
    readmission_cost: float = READMISSION_COST_USD,
    intervention_cost: float = INTERVENTION_COST_USD,
    relative_risk_reduction: float = RELATIVE_RISK_REDUCTION,
) -> dict[str, float]:
    """Expected result of targeting a random panel of the same size."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    k = _panel_size(n, fraction)
    expected = float(y.mean() * k)
    return outcome_from_observed(
        n=n,
        k=k,
        observed_readmissions=expected,
        total_readmissions=float(y.sum()),
        readmission_cost=readmission_cost,
        intervention_cost=intervention_cost,
        relative_risk_reduction=relative_risk_reduction,
    )


def scenario_table(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    fractions: tuple[float, ...] = TARGET_FRACTIONS,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for fraction in fractions:
        random_row = random_targeting_outcome(y, fraction)
        rows.append({"policy": "random", "target_fraction": fraction, **random_row})
        for name, score in scores.items():
            rows.append(
                {
                    "policy": name,
                    "target_fraction": fraction,
                    **targeting_outcome(y, score, fraction, seed=seed + int(fraction * 1000)),
                }
            )
    return pd.DataFrame(rows)


def best_fraction_on_calibration(
    y: np.ndarray,
    score: np.ndarray,
    fractions: tuple[float, ...] = TARGET_FRACTIONS,
    seed: int = 0,
) -> float:
    """Pick the capacity with the highest net savings on the calibration set."""
    best_fraction = fractions[0]
    best_net = -np.inf
    for fraction in fractions:
        net = targeting_outcome(
            y,
            score,
            fraction,
            seed=seed + int(fraction * 1000),
        )["net_savings_usd"]
        # A tie goes to the smaller program.
        if net > best_net + 1e-6 or (abs(net - best_net) <= 1e-6 and fraction < best_fraction):
            best_net = net
            best_fraction = fraction
    return float(best_fraction)


def write_workbook(
    path: Path,
    scenarios: pd.DataFrame,
    deciles: pd.DataFrame,
    fairness: pd.DataFrame,
    model_rows: pd.DataFrame,
    assumptions: dict[str, float],
) -> None:
    """Write a value snapshot plus a formula sheet a reviewer can edit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assumption_frame = pd.DataFrame(
        [
            {"parameter": "readmission_cost_usd", "value": assumptions["readmission_cost_usd"],
             "meaning": "Planning cost of one unplanned 30-day readmission"},
            {"parameter": "intervention_cost_usd", "value": assumptions["intervention_cost_usd"],
             "meaning": "Planning cost of the post-discharge care-management bundle"},
            {"parameter": "relative_risk_reduction", "value": assumptions["relative_risk_reduction"],
             "meaning": "Assumed relative reduction in readmissions among patients who receive the bundle"},
            {"parameter": "primary_target_fraction", "value": assumptions["primary_target_fraction"],
             "meaning": "Pre-specified share of discharges enrolled, ranked by model score"},
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        assumption_frame.to_excel(writer, sheet_name="Assumptions", index=False)
        model_rows.to_excel(writer, sheet_name="Model metrics", index=False)
        deciles.to_excel(writer, sheet_name="Risk deciles", index=False)
        scenarios.to_excel(writer, sheet_name="Snapshot", index=False)
        fairness.to_excel(writer, sheet_name="Fairness audit", index=False)
        _write_calculator(writer.book, assumptions)

    _format_workbook(path)


def _write_calculator(book, assumptions: dict[str, float]) -> None:
    sheet = book.create_sheet("Calculator", 1)
    sheet["A1"] = "Edit the three green inputs. Dollar results recalculate from the test-set counts."
    sheet["A1"].font = Font(bold=True)
    labels = [
        (3, "Readmission cost (USD)", assumptions["readmission_cost_usd"]),
        (4, "Intervention cost (USD)", assumptions["intervention_cost_usd"]),
        (5, "Relative risk reduction", assumptions["relative_risk_reduction"]),
        (6, "Test discharges", assumptions["test_encounters"]),
        (7, "Test readmissions", assumptions["test_readmissions"]),
    ]
    for row, label, value in labels:
        sheet[f"A{row}"] = label
        sheet[f"B{row}"] = value
    sheet["A8"] = "Break-even precision"
    sheet["B8"] = "=B4/(B5*B3)"
    headers = [
        "Policy",
        "Target fraction",
        "N targeted",
        "Observed readmissions in panel",
        "Precision",
        "Recall",
        "Readmissions prevented",
        "Gross savings",
        "Program cost",
        "Net savings",
        "Net per 1,000 discharges",
    ]
    for col, header in enumerate(headers, start=1):
        cell = sheet.cell(10, col, header)
        cell.font = Font(bold=True)
    # The calculator is filled by the caller via rows stored on the book.
    rows = assumptions["calculator_rows"]
    for offset, row in enumerate(rows):
        excel_row = 11 + offset
        sheet.cell(excel_row, 1, row["policy"])
        sheet.cell(excel_row, 2, row["target_fraction"])
        sheet.cell(excel_row, 3, row["n_targeted"])
        sheet.cell(excel_row, 4, row["observed_readmissions"])
        sheet.cell(excel_row, 5, f"=D{excel_row}/C{excel_row}")
        sheet.cell(excel_row, 6, f"=D{excel_row}/$B$7")
        sheet.cell(excel_row, 7, f"=D{excel_row}*$B$5")
        sheet.cell(excel_row, 8, f"=G{excel_row}*$B$3")
        sheet.cell(excel_row, 9, f"=C{excel_row}*$B$4")
        sheet.cell(excel_row, 10, f"=H{excel_row}-I{excel_row}")
        sheet.cell(excel_row, 11, f"=J{excel_row}/$B$6*1000")
    sheet.auto_filter.ref = f"A10:K{10 + len(rows)}"
    sheet.freeze_panes = "A11"


def _format_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    book = load_workbook(path)
    for sheet in book.worksheets:
        for column in sheet.columns:
            letter = get_column_letter(column[0].column)
            width = min(max(len(str(cell.value)) if cell.value is not None else 0 for cell in column) + 2, 42)
            sheet.column_dimensions[letter].width = width
    calc = book["Calculator"]
    for row in range(3, 6):
        calc[f"B{row}"].font = Font(color="1B7F4E", bold=True)
    calc["B3"].number_format = '"$"#,##0'
    calc["B4"].number_format = '"$"#,##0'
    calc["B5"].number_format = "0%"
    calc["B8"].number_format = "0.0%"
    for row in range(11, calc.max_row + 1):
        calc[f"B{row}"].number_format = "0%"
        calc[f"E{row}"].number_format = "0.0%"
        calc[f"F{row}"].number_format = "0.0%"
        calc[f"G{row}"].number_format = "0.00"
        for col in ("H", "I", "J", "K"):
            calc[f"{col}{row}"].number_format = '"$"#,##0'
    book.save(path)


def primary_fraction() -> float:
    return PRIMARY_TARGET_FRACTION
