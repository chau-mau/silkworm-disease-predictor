"""
Generate the Tasar Silkworm Disease Calendar (Ranchi).

Fetches the last 5 full years of daily climate data from NASA POWER for Ranchi,
runs the saved RF+LR ensemble for Virosis and Bacteriosis, aggregates by
day-of-year, and writes docs/calendar_data.json plus docs/calendar.html.

Disclaimer: predictions are based purely on temperature/humidity/wind;
silkworm disease incidence is also subject to rearing season and management
practices.
"""

import json
import pickle
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "app" / "models.pkl"
OUT_JSON = ROOT / "docs" / "calendar_data.json"

DISEASES = ["Virosis", "Bacteriosis"]

# Ranchi, Jharkhand
LAT = 23.3441
LON = 85.3096

# Last 5 full years of daily data
END_YEAR = date.today().year - 1
START_YEAR = END_YEAR - 4


def thi_nrc(tmax, tmin, rh):
    """NRC (1971) Temperature-Humidity Index."""
    t_mean = (tmax + tmin) / 2
    t_f = 1.8 * t_mean + 32
    return t_f - ((0.55 - 0.0055 * rh) * (t_f - 58))


def risk_level(prob):
    if prob < 0.2:
        return "Low"
    if prob < 0.4:
        return "Moderate"
    if prob < 0.6:
        return "High"
    return "Very High"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    return model_data["models"], model_data["scaler"], model_data["feature_cols"]


def fetch_nasa_historical(lat, lon, start_year, end_year):
    """Fetch daily climate from NASA POWER for a year range."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M_MAX,T2M_MIN,RH2M,WS10M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    out = []
    for k in sorted(p["T2M_MAX"]):
        if p["T2M_MAX"][k] == -999:
            continue
        out.append(
            {
                "date": f"{k[:4]}-{k[4:6]}-{k[6:]}",
                "tmax": p["T2M_MAX"][k],
                "tmin": p["T2M_MIN"][k],
                "humidity": p["RH2M"][k],
                "wind_speed": p["WS10M"][k],
                "rainfall": p["PRECTOTCORR"][k],
            }
        )
    return out


def predict_day(features, models, scaler, feature_cols):
    x = np.array([[features[col] for col in feature_cols]])
    x_scaled = scaler.transform(x)
    preds = {}
    for disease in DISEASES:
        md = models[disease]
        rf_prob = md["random_forest"].predict_proba(x)[0][1]
        lr_prob = md["logistic_regression"].predict_proba(x_scaled)[0][1]
        prob = round((rf_prob + lr_prob) / 2, 4)
        preds[disease] = {"probability": prob, "risk": risk_level(prob)}
    return preds


def generate_data():
    print(f"Fetching NASA POWER climate for Ranchi ({START_YEAR}–{END_YEAR})...")
    raw = fetch_nasa_historical(LAT, LON, START_YEAR, END_YEAR)
    print(f"Retrieved {len(raw)} daily records.")

    models, scaler, feature_cols = load_model()

    # Predict for every historical day
    historical = []
    for row in raw:
        features = {
            "Tmax": row["tmax"],
            "Tmin": row["tmin"],
            "Humidity": row["humidity"],
            "THI": thi_nrc(row["tmax"], row["tmin"], row["humidity"]),
            "Wind_Speed": row["wind_speed"],
        }
        preds = predict_day(features, models, scaler, feature_cols)
        historical.append(
            {
                "date": row["date"],
                "doy": row["date"][5:],  # MM-DD
                "tmax": round(row["tmax"], 1),
                "tmin": round(row["tmin"], 1),
                "humidity": round(row["humidity"], 1),
                "thi": round(features["THI"], 2),
                "wind_speed": round(row["wind_speed"], 2),
                "virosis": preds["Virosis"],
                "bacteriosis": preds["Bacteriosis"],
            }
        )

    # Aggregate by day-of-year across all 5 years
    by_doy = defaultdict(list)
    for h in historical:
        by_doy[h["doy"]].append(h)

    # Build a typical-year calendar (use a non-leap year, e.g. 2025)
    typical_year = 2025
    records = []
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                d = date(typical_year, month, day)
            except ValueError:
                continue
            doy = f"{month:02d}-{day:02d}"
            days = by_doy.get(doy, [])
            if not days:
                continue

            v_probs = [d["virosis"]["probability"] for d in days]
            b_probs = [d["bacteriosis"]["probability"] for d in days]
            avg_v = sum(v_probs) / len(v_probs)
            avg_b = sum(b_probs) / len(b_probs)

            # Typical daily climate
            avg_tmax = sum(d["tmax"] for d in days) / len(days)
            avg_tmin = sum(d["tmin"] for d in days) / len(days)
            avg_hum = sum(d["humidity"] for d in days) / len(days)
            avg_thi = sum(d["thi"] for d in days) / len(days)
            avg_wind = sum(d["wind_speed"] for d in days) / len(days)

            combined_level = "Low"
            for lvl in ["Very High", "High", "Moderate"]:
                if risk_level(avg_v) == lvl or risk_level(avg_b) == lvl:
                    combined_level = lvl
                    break

            records.append(
                {
                    "date": d.isoformat(),
                    "tmax": round(avg_tmax, 1),
                    "tmin": round(avg_tmin, 1),
                    "humidity": round(avg_hum, 1),
                    "thi": round(avg_thi, 2),
                    "wind_speed": round(avg_wind, 2),
                    "virosis": {
                        "probability": round(avg_v, 4),
                        "risk": risk_level(avg_v),
                    },
                    "bacteriosis": {
                        "probability": round(avg_b, 4),
                        "risk": risk_level(avg_b),
                    },
                    "combined_risk": combined_level,
                    "years_sampled": len(days),
                }
            )

    summary = {
        "total_days": len(records),
        "historical_days": len(historical),
        "year_range": f"{START_YEAR}–{END_YEAR}",
        "typical_year": typical_year,
        "location": {"name": "Ranchi, Jharkhand", "lat": LAT, "lon": LON},
        "virosis": {
            "high_days": sum(1 for r in records if r["virosis"]["risk"] in ("High", "Very High")),
            "very_high_days": sum(1 for r in records if r["virosis"]["risk"] == "Very High"),
        },
        "bacteriosis": {
            "high_days": sum(1 for r in records if r["bacteriosis"]["risk"] in ("High", "Very High")),
            "very_high_days": sum(1 for r in records if r["bacteriosis"]["risk"] == "Very High"),
        },
        "combined_high_days": sum(
            1 for r in records if r["combined_risk"] in ("High", "Very High")
        ),
        "note": (
            "Predictions are from the weather-only RF+LR ensemble trained on 2025–2026 "
            f"field data, applied to {START_YEAR}–{END_YEAR} NASA POWER climate for Ranchi "
            "and averaged by day-of-year. Disease incidence is also subject to rearing season "
            "and management practices."
        ),
        "disclaimer": (
            "Disease incidence is subject to rearing season, crop stage, and management practices. "
            "This calendar indicates climate-based risk only and should be used as a decision-support tool, not a definitive forecast."
        ),
    }

    payload = {
        "summary": summary,
        "records": records,
        "risk_colors": {
            "Low": "#28a745",
            "Moderate": "#ffc107",
            "High": "#fd7e14",
            "Very High": "#dc3545",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON} with {len(records)} typical-year daily records.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    generate_data()
