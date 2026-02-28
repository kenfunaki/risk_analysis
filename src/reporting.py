# reporting.py

def repot_score(df):

    total = len(df)
    anomaly_count = int(df["anomaly"].sum())

    risk_score = (anomaly_count / total) * 100 if total > 0 else 0

    return {
        "total_rows": total,
        "anomalies": anomaly_count,
        "risk_score": round(risk_score, 2),
    }