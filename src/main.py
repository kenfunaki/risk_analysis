# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# KPI計算
# -----------------------------
def calc_kpi(df):
    df = df.copy()
    df["mean"] = df["value"].mean()
    df["std"] = df["value"].std()
    return df


# -----------------------------
# Zスコア異常検知
# -----------------------------
def zscore_anomalies(df, threshold=2.5):

    df = df.copy()

    mean = df["value"].mean()
    std = df["value"].std()

    if std == 0:
        df["zscore"] = 0
        df["anomaly"] = False
        return df

    df["zscore"] = (df["value"] - mean) / std
    df["anomaly"] = df["zscore"].abs() > threshold

    return df


# -----------------------------
# 説明生成
# -----------------------------
def feature_expl(row):

    if row["anomaly"]:
        return f"平均から {row['zscore']:.2f}σ 乖離しており異常値の可能性があります。"

    return "異常は検出されませんでした。"


# -----------------------------
# 全体分析
# -----------------------------
def analyze_multiple_row(df):

    df = calc_kpi(df)
    df = zscore_anomalies(df)

    df["explanation"] = df.apply(feature_expl, axis=1)

    return df


# -----------------------------
# グラフ生成（表示しない）
# -----------------------------
def create_plot(df):

    fig, ax = plt.subplots()

    ax.plot(df.index, df["value"])
    ax.set_title("Value Trend")

    return fig