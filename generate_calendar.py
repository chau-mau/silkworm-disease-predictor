"""
Generate historical disease-calendar data for the static GitHub Pages site.

Reads results/merged_data_2025_2026.csv, aggregates to one record per date,
runs the saved RF+LR ensemble (app/models.pkl) for Virosis and Bacteriosis,
and writes docs/calendar_data.json plus a ready-to-use docs/calendar.html.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "app" / "models.pkl"
DATA_PATH = ROOT / "results" / "merged_data_2025_2026.csv"
OUT_JSON = ROOT / "docs" / "calendar_data.json"

DISEASES = ["Virosis", "Bacteriosis"]


def thi_nrc(tmax, tmin, rh):
    """NRC (1971) Temperature-Humidity Index."""
    t_mean = (tmax + tmin) / 2
    t_f = 1.8 * t_mean + 32
    return round(t_f - ((0.55 - 0.0055 * rh) * (t_f - 58)), 2)


def risk_level(prob):
    if prob < 0.2:
        return "Low"
    if prob < 0.4:
        return "Moderate"
    if prob < 0.6:
        return "High"
    return "Very High"


def risk_color(level):
    return {
        "Low": "#28a745",
        "Moderate": "#ffc107",
        "High": "#fd7e14",
        "Very High": "#dc3545",
    }[level]


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    return model_data["models"], model_data["scaler"], model_data["feature_cols"]


def generate_data():
    df = pd.read_csv(DATA_PATH)
    # Drop rows where the model cannot run
    df = df.dropna(subset=["Tmax", "Tmin", "Humidity", "Wind_Speed"]).copy()

    # Aggregate to one row per date (mean climate, any disease occurrence)
    agg = (
        df.groupby("Date")
        .agg(
            Tmax=("Tmax", "mean"),
            Tmin=("Tmin", "mean"),
            Humidity=("Humidity", "mean"),
            Wind_Speed=("Wind_Speed", "mean"),
            THI=("THI", "mean"),
            Virosis_Obs=("Virosis", "max"),
            Bacteriosis_Obs=("Bacteriosis", "max"),
            Year=("Year", "first"),
            Source=("Source", "first"),
        )
        .reset_index()
    )
    agg["Tmax"] = agg["Tmax"].round(1)
    agg["Tmin"] = agg["Tmin"].round(1)
    agg["Humidity"] = agg["Humidity"].round(1)
    agg["Wind_Speed"] = agg["Wind_Speed"].round(2)
    agg["THI"] = agg["THI"].round(2)

    models, scaler, feature_cols = load_model()

    records = []
    for _, row in agg.iterrows():
        features = {
            "Tmax": row["Tmax"],
            "Tmin": row["Tmin"],
            "Humidity": row["Humidity"],
            "THI": thi_nrc(row["Tmax"], row["Tmin"], row["Humidity"]),
            "Wind_Speed": row["Wind_Speed"],
        }
        x = np.array([[features[col] for col in feature_cols]])
        x_scaled = scaler.transform(x)

        preds = {}
        for disease in DISEASES:
            md = models[disease]
            rf_prob = md["random_forest"].predict_proba(x)[0][1]
            lr_prob = md["logistic_regression"].predict_proba(x_scaled)[0][1]
            prob = round((rf_prob + lr_prob) / 2, 4)
            preds[disease] = {
                "probability": prob,
                "risk": risk_level(prob),
                "observed": int(row[f"{disease}_Obs"]),
            }

        combined_level = "Low"
        for lvl in ["Very High", "High", "Moderate", "Low"]:
            if any(preds[d]["risk"] == lvl for d in DISEASES):
                combined_level = lvl
                break

        records.append(
            {
                "date": row["Date"],
                "year": int(row["Year"]),
                "source": row["Source"],
                "tmax": row["Tmax"],
                "tmin": row["Tmin"],
                "humidity": row["Humidity"],
                "thi": features["THI"],
                "wind_speed": row["Wind_Speed"],
                "virosis": preds["Virosis"],
                "bacteriosis": preds["Bacteriosis"],
                "combined_risk": combined_level,
            }
        )

    # Sort by date
    records.sort(key=lambda r: r["date"])

    summary = {
        "total_days": len(records),
        "virosis": {
            "high_days": sum(1 for r in records if r["virosis"]["risk"] in ("High", "Very High")),
            "very_high_days": sum(1 for r in records if r["virosis"]["risk"] == "Very High"),
            "observed_days": sum(r["virosis"]["observed"] for r in records),
        },
        "bacteriosis": {
            "high_days": sum(1 for r in records if r["bacteriosis"]["risk"] in ("High", "Very High")),
            "very_high_days": sum(1 for r in records if r["bacteriosis"]["risk"] == "Very High"),
            "observed_days": sum(r["bacteriosis"]["observed"] for r in records),
        },
        "combined_high_days": sum(
            1
            for r in records
            if r["combined_risk"] in ("High", "Very High")
        ),
        "location": {"name": "Ranchi, Jharkhand", "lat": 23.3441, "lon": 85.3096},
        "note": "Predictions from the weather-only RF+LR ensemble trained on 2025-2026 field data.",
    }

    payload = {"summary": summary, "records": records, "risk_colors": {lvl: risk_color(lvl) for lvl in ["Low", "Moderate", "High", "Very High"]}}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON} with {len(records)} daily records.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    generate_data()
