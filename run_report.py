# run_report.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd

from main import analyze_multiple_row


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def export_result_by_keys(
    df: pd.DataFrame,
    out_csv: str | None = None,
    anomalies_only: bool = False,
    output_dir: str = "reports",
    *,
    group_cols=("company", "account"),
    z_threshold: float = 2.5,
    robust_threshold: float = 3.5,
    value_col: str = "value",
) -> Path:
    """
    会社×科目×年×四半期ごとの詳細結果をCSV保存して、保存先Pathを返す。
    """

    result = analyze_multiple_row(
        df,
        group_cols=group_cols,
        z_threshold=z_threshold,
        robust_threshold=robust_threshold,
        value_col=value_col,
    )

    detail_cols = [
        "company", "account", "year", "quarter", "value",
        "seasonal_mean", "seasonal_std",
        "zscore", "robust_z",
        "yoy_diff", "qoq_diff",
        "group_n",
        "anomaly",
        "explanation",
    ]
    detail_cols = [c for c in detail_cols if c in result.columns]
    detail_df = result[detail_cols].copy()

    if anomalies_only:
        detail_df = detail_df[detail_df["anomaly"] == True]

    detail_df = detail_df.sort_values(["company", "account", "year", "quarter"])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_csv is None:
        name = "analysis_anomalies" if anomalies_only else "analysis_result"
        out_csv = f"{name}_{_ts()}.csv"

    path = out_dir / out_csv
    detail_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_summary_by_company_account(
    df: pd.DataFrame,
    out_csv: str | None = None,
    output_dir: str = "reports",
    *,
    group_cols=("company", "account"),
    z_threshold: float = 2.5,
    robust_threshold: float = 3.5,
    value_col: str = "value",
) -> Path:
    """
    会社×科目単位のサマリー(total/anomalies/risk_score)をCSV保存して、保存先Pathを返す。
    """

    result = analyze_multiple_row(
        df,
        group_cols=group_cols,
        z_threshold=z_threshold,
        robust_threshold=robust_threshold,
        value_col=value_col,
    )

    g = result.groupby(list(group_cols), dropna=False)
    summary = g.agg(
        total_rows=("anomaly", "size"),
        anomalies=("anomaly", "sum"),
    ).reset_index()
    summary["risk_score"] = (summary["anomalies"] / summary["total_rows"] * 100).round(2)
    summary = summary.sort_values(["risk_score", "anomalies"], ascending=[False, False])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_csv is None:
        out_csv = f"summary_by_group_{_ts()}.csv"

    path = out_dir / out_csv
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return path