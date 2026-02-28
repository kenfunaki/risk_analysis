# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 前処理: value を数値化（"1,234" もOK）
# -----------------------------
def normalize_value(df, value_col="value"):
    df = df.copy()
    df[value_col] = (
        df[value_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", np.nan)
    )
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df


# -----------------------------
# Robust Z (MAD)
# robust_z = (x - median) / (1.4826 * MAD)
# ※ MAD=0 のときは std にフォールバック（激しく動く系列でも潰れない）
# -----------------------------
def robust_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) == 0:
        return pd.Series([0.0] * len(series), index=series.index)

    med = valid.median()
    mad = np.median(np.abs(valid - med))

    denom = 1.4826 * mad if mad and not np.isnan(mad) and mad > 0 else valid.std(ddof=1)
    if denom is None or np.isnan(denom) or denom == 0:
        denom = 1e-9

    out = (s - med) / denom
    return out.fillna(0.0)


# -----------------------------
# KPI計算（グループ別・季節性対応）
# 期待列: company, account, year, quarter, value
# group_cols: 通常は ("company","account")
# -----------------------------
def calc_kpi(df, group_cols=("company", "account"), value_col="value"):
    df = df.copy()

    # グループ母数（欠損以外）
    df["group_n"] = df.groupby(list(group_cols))[value_col].transform("count")

    # 全期間の平均/標準偏差（参考）
    g = df.groupby(list(group_cols))[value_col]
    df["mean"] = g.transform("mean")
    df["std"] = g.transform("std")  # pandas標準 ddof=1

    # 季節性: 同一四半期内で比較する平均との差/標準偏差
    if "quarter" in df.columns:
        gq = df.groupby(list(group_cols) + ["quarter"])[value_col]
        df["seasonal_mean"] = gq.transform("mean")
        df["seasonal_std"] = gq.transform("std")
    else:
        # quarter が無い場合は全期間平均を代替（落ちないための保険）
        df["seasonal_mean"] = df["mean"]
        df["seasonal_std"] = df["std"]

    return df


# -----------------------------
# 変化量（YoY / QoQ）
# - YoY: 前年同四半期との差
# - QoQ: 直前四半期との差（同一会社×科目）
# -----------------------------
def add_deltas(df, group_cols=("company", "account"), value_col="value"):
    df = df.copy()

    needed = {"year", "quarter"}
    if not needed.issubset(df.columns):
        df["qoq_diff"] = np.nan
        df["yoy_diff"] = np.nan
        return df

    sort_cols = list(group_cols) + ["year", "quarter"]
    df = df.sort_values(sort_cols)

    # QoQ（同一会社×科目で1期前との差）
    df["qoq_diff"] = df.groupby(list(group_cols))[value_col].diff()

    # YoY（同一会社×科目×同一四半期で前年との差）
    df["yoy_diff"] = df.groupby(list(group_cols) + ["quarter"])[value_col].diff()

    return df


# -----------------------------
# 異常検知（季節性Z + RobustZ）
# 既存の「値の外れ値」検知
# -----------------------------
def detect_anomalies(
    df,
    z_threshold=1.5,
    robust_threshold=2.5,
    group_cols=("company", "account"),
    value_col="value",
):
    df = df.copy()

    # 季節性Z（四半期平均との差）
    invalid_std = df["seasonal_std"].isna() | (df["seasonal_std"] == 0)

    df["zscore"] = 0.0
    ok = ~invalid_std & df[value_col].notna() & df["seasonal_mean"].notna()
    df.loc[ok, "zscore"] = (
        (df.loc[ok, value_col] - df.loc[ok, "seasonal_mean"]) / df.loc[ok, "seasonal_std"]
    )

    # RobustZ（会社×科目の全期間）
    df["robust_z"] = df.groupby(list(group_cols))[value_col].transform(robust_zscore)

    df["anomaly_z"] = df["zscore"].abs() > z_threshold
    df["anomaly_robust"] = df["robust_z"].abs() > robust_threshold
    df["anomaly"] = df["anomaly_z"] | df["anomaly_robust"]

    return df


# -----------------------------
# Fraud-Grade Detection（実務版）
# 「値の外れ」ではなく「動きの不自然さ」を拾う
# - 変化率（QoQ%）のRobustZ
# - トレンド破断（rolling平均との差）
# - 符号反転（急落→急回復 など）
# - 総合スコア fraud_score と判定 fraud_anomaly
# -----------------------------
def add_fraud_grade_detection(
    df: pd.DataFrame,
    group_cols=("company", "account"),
    value_col="value",
    fraud_score_threshold=3.5,
):
    df = df.copy()

    # year/quarterが無いと時系列特徴が作れない
    if not {"year", "quarter"}.issubset(df.columns):
        df["qoq_pct"] = np.nan
        df["qoq_pct_rz"] = 0.0
        df["trend_break"] = np.nan
        df["trend_break_z"] = 0.0
        df["sign_flip"] = 0
        df["fraud_score"] = 0.0
        df["fraud_anomaly"] = False
        return df

    sort_cols = list(group_cols) + ["year", "quarter"]
    df = df.sort_values(sort_cols)
    grp = df.groupby(list(group_cols), dropna=False)

    # 1) QoQ変化率（急変検知）
    df["qoq_pct"] = grp[value_col].pct_change()

    # 変化率のRobustZ（グループ内での急変を拾う）
    df["qoq_pct_rz"] = grp["qoq_pct"].transform(lambda s: robust_zscore(s.fillna(0.0)))

    # 2) トレンド破断（Rolling meanとの差分）
    df["rolling_mean"] = grp[value_col].transform(
        lambda s: s.rolling(3, center=True, min_periods=1).mean()
    )
    df["trend_break"] = (df[value_col] - df["rolling_mean"]).abs()

    df["trend_break_z"] = grp["trend_break"].transform(
        lambda s: ((s - s.mean()) / (s.std(ddof=1) + 1e-9)).fillna(0.0)
    )

    # 3) 符号反転（戻し仕訳・期末調整の匂い）
    df["sign_flip"] = grp["qoq_pct"].transform(
        lambda s: ((s.shift(1) * s) < 0).astype(int)
    ).fillna(0).astype(int)

    # 4) 総合スコア（既存のz/robustも統合）
    z_abs = df["zscore"].abs() if "zscore" in df.columns else 0.0
    rz_abs = df["robust_z"].abs() if "robust_z" in df.columns else 0.0

    df["fraud_score"] = (
        rz_abs * 0.8                  # 分布外れ（値の異常）
        + z_abs * 0.5                 # 季節性の外れ
        + df["qoq_pct_rz"].abs() * 1.5   # 急変（最重要）
        + df["trend_break_z"].abs() * 1.2
        + df["sign_flip"] * 2.0          # 戻し/反転
    )

    df["fraud_anomaly"] = df["fraud_score"] > float(fraud_score_threshold)

    # 既存 anomaly に統合したい場合（任意：ここでOR統合）
    # df["anomaly"] = df.get("anomaly", False) | df["fraud_anomaly"]

        # =================================================
    # 6) Absolute Shock Rule（実務監査で必須）
    #    ボラが高い系列でも「急落・急騰」は強制的に拾う
    # =================================================
    df["shock_drop"] = df["qoq_pct"] < -0.70   # 70%以上の減少
    df["shock_spike"] = df["qoq_pct"] > 2.00   # 200%以上の増加
    df["shock_event"] = (df["shock_drop"] | df["shock_spike"]).fillna(False)

    # 判定に強制反映（OR）
    df["fraud_anomaly"] = df["fraud_anomaly"] | df["shock_event"]

    # スコアもブースト（順位付けが効く）
    df["fraud_score"] = df["fraud_score"] + df["shock_event"].astype(int) * 3.0

    df["anomaly"] = df.get("anomaly", False) | df["fraud_anomaly"]

    return df


# -----------------------------
# 説明生成（Fraud-Grade反映）
# -----------------------------
def feature_expl(row, z_th=1.5, robust_th=2.5, fraud_th=3.5):
    val = row.get("value", np.nan)
    sm = row.get("seasonal_mean", np.nan)
    zs = row.get("zscore", 0.0)
    rz = row.get("robust_z", 0.0)
    yoy = row.get("yoy_diff", np.nan)
    qoq = row.get("qoq_diff", np.nan)
    n = row.get("group_n", np.nan)

    qoq_pct = row.get("qoq_pct", np.nan)
    qoq_pct_rz = row.get("qoq_pct_rz", 0.0)
    trend_break_z = row.get("trend_break_z", 0.0)
    sign_flip = row.get("sign_flip", 0)
    fraud_score = row.get("fraud_score", 0.0)
    fraud_anom = row.get("fraud_anomaly", False)

    diff = val - sm if pd.notna(val) and pd.notna(sm) else np.nan

    # stdが成立しない場合（季節性Zが定義できない）を明示
    if pd.isna(row.get("seasonal_std")) or row.get("seasonal_std") == 0:
        z_note = "Z=NA(分散不足)"
    else:
        z_note = f"Z={zs:.2f}"

    base = (
        f"季節性平均との差={diff:.0f}, {z_note}(閾値={z_th}), "
        f"過去の傾向からの逸脱={rz:.2f}(閾値={robust_th}), "
        f"前年同期との差={yoy:.0f}, 前期との差={qoq:.0f}, 母数n={int(n) if pd.notna(n) else 'NA'}。"
    )

    # Fraud-Gradeの説明（激しく動く系列でも拾う）
    fraud_part = (
        f"不正リスク: 前期比={qoq_pct:.2%} , 前期比の超過={qoq_pct_rz:.2f}, "
        f"過去の傾向からの逸脱={trend_break_z:.2f}, リスク兆候={int(sign_flip)}, "
        f"リスクスコア={fraud_score:.2f}(閾値={fraud_th})."
    )

    reasons = []
    if row.get("anomaly_z", False):
        reasons.append("季節性変動からの逸脱(Z超過)")
    if row.get("anomaly_robust", False):
        reasons.append("過去の傾向からの逸脱(RobustZ超過)")
    if fraud_anom:
        reasons.append("リスク兆候のタイプ(急変/破断/反転)")

    if reasons:
        return base + fraud_part + " 異常の可能性: " + "・".join(reasons)

    return base + fraud_part + " 統計的な異常は検出されませんでした。"


# -----------------------------
# 全体分析（run_report.py から呼ばれる入口）
# 返却dfに以下が入る：
# mean/std/seasonal_mean/seasonal_std/zscore/robust_z/yoy_diff/qoq_diff/group_n/anomaly/explanation
# + Fraud-Grade: qoq_pct/qoq_pct_rz/trend_break_z/sign_flip/fraud_score/fraud_anomaly
# -----------------------------
def analyze_multiple_row(
    df,
    group_cols=("company", "account"),
    z_threshold=1.5,
    robust_threshold=2.5,
    value_col="value",
    fraud_score_threshold=3.5,
):
    df = df.copy()

    # 必須列チェック（落ちるより明確に）
    required = set(group_cols) | {value_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected at least {sorted(required)}")

    df = normalize_value(df, value_col=value_col)
    df = calc_kpi(df, group_cols=group_cols, value_col=value_col)
    df = add_deltas(df, group_cols=group_cols, value_col=value_col)

    df = detect_anomalies(
        df,
        z_threshold=z_threshold,
        robust_threshold=robust_threshold,
        group_cols=group_cols,
        value_col=value_col,
    )

    # ★ Fraud-Grade Detection を追加
    df = add_fraud_grade_detection(
        df,
        group_cols=group_cols,
        value_col=value_col,
        fraud_score_threshold=fraud_score_threshold,
    )

    df["explanation"] = df.apply(
        lambda r: feature_expl(
            r,
            z_th=z_threshold,
            robust_th=robust_threshold,
            fraud_th=fraud_score_threshold,
        ),
        axis=1
    )

    # 会社・科目・年・四半期で見たいので並べ替え（存在すれば）
    sort_cols = [c for c in ["company", "account", "year", "quarter"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df


# -----------------------------
# グラフ生成（表示しない）
# -----------------------------
def create_plot(df, x_col=None, value_col="value", title="Value Trend"):
    fig, ax = plt.subplots()
    x = df[x_col] if x_col else df.index
    ax.plot(x, df[value_col])
    ax.set_title(title)
    ax.set_xlabel(x_col or "index")
    ax.set_ylabel(value_col)
    return fig