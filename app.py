from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import streamlit as st
from pathlib import Path

from main import analyze_multiple_row
from run_report import export_result_by_keys, export_summary_by_company_account
from fraud_audit_report import export_audit_report_word, assess_fraud_risk_with_gpt5, build_llm_items

st.set_page_config(page_title="Risk Analysis", layout="wide")
st.title("財務データ リスク分析（季節性 + RobustZ + YoY/QoQ + Fraud-Grade）")

uploaded = st.file_uploader(
    "入力CSV（company, account, year, quarter, value を含む）",
    type=["csv"]
)

# -----------------------------
# 設定値は「必ず」先にデフォルト定義（NameError防止）
# -----------------------------
z_th_default = 2.5
robust_th_default = 3.5
fraud_th_default = 3.5
anomalies_only_default = False
output_dir_default = "reports"

with st.expander("設定", expanded=True):
    z_th = st.number_input("季節性Z閾値", value=float(z_th_default), step=0.1)
    robust_th = st.number_input("RobustZ(MAD)閾値", value=float(robust_th_default), step=0.1)
    fraud_th = st.number_input("FraudScore閾値", value=float(fraud_th_default), step=0.1)
    anomalies_only = st.checkbox("異常行のみ出力（詳細CSV）", value=anomalies_only_default)
    output_dir = st.text_input("保存先フォルダ（相対パス）", value=output_dir_default)

result = None  # まだアップロードされていない場合の保険

if uploaded:
    df = pd.read_csv(uploaded)

    required = {"company", "account", "year", "quarter", "value"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSVに必要な列がありません: {missing}（必要: {sorted(required)}）")
        st.stop()

    # 画面プレビュー（メモリ上で分析）
    result = analyze_multiple_row(
        df,
        group_cols=("company", "account"),
        z_threshold=float(z_th),
        robust_threshold=float(robust_th),
        value_col="value",
        fraud_score_threshold=float(fraud_th),  # ← main.pyの修正版に追加した引数
    )

    st.subheader("分析結果（プレビュー）")

    # main.py の列名に合わせる（z_scoreではなくzscore、robust_z_scoreではなくrobust_z）
    display_cols = [
        "company",
        "account",
        "year",
        "quarter",
        "value",
        "zscore",
        "robust_z",
        "anomaly_z",
        "anomaly_robust",
        "anomaly",
        "qoq_diff",
        "yoy_diff",
        "qoq_pct",
        "qoq_pct_rz",
        "trend_break_z",
        "sign_flip",
        "fraud_score",
        "fraud_anomaly",
        "explanation",
    ]
    display_cols = [c for c in display_cols if c in result.columns]
    display_df = result[display_cols]

    st.dataframe(display_df, use_container_width=True, height=420)

    st.subheader("CSV出力（ディスク保存→ダウンロード）")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("詳細CSVを生成して保存"):
            path = export_result_by_keys(
                df,
                out_csv=None,
                anomalies_only=anomalies_only,
                output_dir=output_dir,
                z_threshold=float(z_th),
                robust_threshold=float(robust_th),
            )
            st.success(f"保存しました: {path}")

            data = Path(path).read_bytes()
            st.download_button(
                label="📥 詳細CSVをダウンロード",
                data=data,
                file_name=Path(path).name,
                mime="text/csv",
            )

    with col2:
        if st.button("サマリーCSVを生成して保存"):
            path = export_summary_by_company_account(
                df,
                out_csv=None,
                output_dir=output_dir,
                z_threshold=float(z_th),
                robust_threshold=float(robust_th),
            )
            st.success(f"保存しました: {path}")

            data = Path(path).read_bytes()
            st.download_button(
                label="📥 サマリーCSVをダウンロード",
                data=data,
                file_name=Path(path).name,
                mime="text/csv",
            )

    st.divider()

    # -----------------------------
    # GPT-5 → Word 監査レポート
    # ※ result があるときだけボタンを出す（スコープ問題回避）
    # -----------------------------
    st.subheader("GPT-5による不正リスク評価（Wordレポート）")

    top_n = st.slider("LLMに渡す上位件数（fraud_score順）", min_value=10, max_value=80, value=30, step=5)

    if st.button("GPT-5で監査レポート(Word)生成"):
        with st.spinner("GPT-5で評価中..."):
            items = build_llm_items(result, top_n=int(top_n))
            assessment = assess_fraud_risk_with_gpt5(items, model="gpt-5.2")  # 必要に応じて model="gpt-5"
            out_path = export_audit_report_word(
                assessment,
                output_path="fraud_audit_report.docx"
            )

        data = Path(out_path).read_bytes()
        st.download_button(
            "📄 Wordをダウンロード",
            data=data,
            file_name=Path(out_path).name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

else:
    st.info("まずCSVをアップロードしてください。")