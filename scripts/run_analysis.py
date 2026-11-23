"""Run the full readmission analysis and write reports/."""

import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
matplotlib.use("Agg")

from readmission.pipeline import run_pipeline


def main() -> None:
    payload = run_pipeline(save=True)
    metrics = payload["metrics"]
    gb = metrics["test"]["gradient_boosting"]
    point = metrics["operating_point"]["policies"]["gradient_boosting"]
    print(
        "Test AUC {auc:.3f} (95% CI {lo:.3f}–{hi:.3f}); "
        "top 10% precision {precision:.3f}; "
        "net per 1,000 discharges ${net:,.0f}.".format(
            auc=gb["roc_auc"],
            lo=gb["auc_ci95_low"],
            hi=gb["auc_ci95_high"],
            precision=point["precision"],
            net=point["net_savings_per_1000_discharges_usd"],
        )
    )


if __name__ == "__main__":
    main()
