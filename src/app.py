# app.py
import streamlit as st
import pandas as pd

from main import analyze_multiple_row, create_plot
from reporting import repot_score


st.title("連結試算表 異常値リスク分析AI")

uploaded_file = st.file_uploader(
    "CSVファイルをアップロード",
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="cp932")

    st.subheader("入力データ")
    st.dataframe(df)

    # -----------------
    # 分析
    # -----------------
    result_df = analyze_multiple_row(df)

    st.subheader("分析結果")
    st.dataframe(result_df)

    # -----------------
    # グラフ
    # -----------------
    fig = create_plot(result_df)
    st.pyplot(fig)

    # -----------------
    # スコア
    # -----------------
    score = repot_score(result_df)

    st.subheader("リスク評価")

    col1, col2, col3 = st.columns(3)

    col1.metric("総件数", score["total_rows"])
    col2.metric("異常件数", score["anomalies"])
    col3.metric("リスク率(%)", score["risk_score"])