"""
Build the 2026 silkworm disease dataset for Ranchi.

Sources:
  1. User-entered August 2026 field observations (typed records)
  2. 'Disease 2026.xlsx' - September 2026 larval disease counts
  3. NASA POWER API (authentic source) - daily climate for Ranchi (23.3441N, 85.3096E)

Outputs:
  results/disease_data_2026.csv   - daily disease counts (Aug-Sep 2026)
  results/climate_data_2026.csv   - daily NASA POWER climate (Aug-Sep 2026)
  results/merged_data_2025_2026.csv - combined training dataset (2025 + 2026 schema)
"""

import pandas as pd
import numpy as np
import requests
import json

RANCHI_LAT = 23.3441
RANCHI_LON = 85.3096
YEAR = 2026

# ---------------------------------------------------------------------------
# 1. User-entered August 2026 observations (typed records)
# ---------------------------------------------------------------------------
august_records = [
    ("2026-08-02", 0, 1),   # 1 larva - bacteriosis
    ("2026-08-03", 0, 1),   # 1 larva - bacteriosis
    ("2026-08-04", 2, 0),   # virosis in 2 larvae
    ("2026-08-11", 0, 1),   # 1 larva - bacteriosis
    ("2026-08-20", 0, 1),   # 1 larva - bacteriosis
    ("2026-08-22", 0, 2),   # bacteriosis in 2 larvae
    ("2026-08-24", 0, 10),  # bacteriosis in 10 larvae
]
aug = pd.DataFrame(august_records, columns=["Date", "Virosis", "Bacteriosis"])
aug["Source"] = "typed_august"

# ---------------------------------------------------------------------------
# 2. 'Disease 2026.xlsx' - September 2026 records
#    NOTE on dates: the sheet mixes Excel datetime values (2026-11-09,
#    2026-12-09) and text values (9/14/26). All dates are interpreted as
#    SEPTEMBER 2026 (day-first for datetimes, month-first for text), since
#    Nov/Dec 2026 dates would post-date the study period.
#    Each row is treated as one rearing tray/observation; rows are summed
#    per date.
# ---------------------------------------------------------------------------
raw = pd.read_excel("Disease 2026.xlsx", sheet_name="Sheet1", header=0)
raw.columns = ["Date_raw", "Blank", "Virosis", "Bacteriosis_note"]
raw = raw.dropna(subset=["Virosis"])
raw["Virosis"] = pd.to_numeric(raw["Virosis"], errors="coerce").fillna(0).astype(int)
# Bacteriosis column holds counts (symptom text ignored for counting)
raw["Bacteriosis"] = pd.to_numeric(raw["Bacteriosis_note"], errors="coerce").fillna(0).astype(int)


def parse_excel_date(v):
    """Interpret all dates as September 2026."""
    import datetime as _dt
    if isinstance(v, (_dt.datetime, pd.Timestamp)):
        # stored datetimes 2026-11-09 / 2026-12-09 are read day-first -> Sept
        return pd.Timestamp(year=YEAR, month=9, day=v.day)
    s = str(v).strip()
    m, d, y = s.split("/")
    return pd.Timestamp(year=2000 + int(y), month=int(m), day=int(d))


raw["Date"] = raw["Date_raw"].apply(parse_excel_date)
sep = raw.groupby("Date", as_index=False)[["Virosis", "Bacteriosis"]].sum()
sep["Source"] = "excel_september"

print("September 2026 daily totals (from Disease 2026.xlsx):")
print(sep.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Combine typed + excel records; fill zero-disease days across rearing span
# ---------------------------------------------------------------------------
dis = pd.concat([aug.assign(Date=pd.to_datetime(aug["Date"])), sep], ignore_index=True)
dis["Date"] = pd.to_datetime(dis["Date"])
first_day = pd.Timestamp("2026-08-01")       # start of August rearing
last_day = dis["Date"].max()                 # last recorded observation
full_range = pd.DataFrame({"Date": pd.date_range(first_day, last_day, freq="D")})
dis = full_range.merge(dis, on="Date", how="left")
for c in ["Virosis", "Bacteriosis"]:
    dis[c] = dis[c].fillna(0).astype(int)
dis["Pebrine"] = 0
dis["Muscardine"] = 0
dis["Source"] = dis["Source"].fillna("no_record_assumed_zero")
dis = dis[["Date", "Pebrine", "Virosis", "Bacteriosis", "Muscardine", "Source"]]
dis.to_csv("results/disease_data_2026.csv", index=False)
print(f"\nDisease data 2026: {len(dis)} days ({dis['Date'].min().date()} to {dis['Date'].max().date()})")
print(f"  Bacteriosis days: {(dis['Bacteriosis'] > 0).sum()}, Virosis days: {(dis['Virosis'] > 0).sum()}")

# ---------------------------------------------------------------------------
# 4. NASA POWER daily climate for Ranchi
# ---------------------------------------------------------------------------
start = first_day.strftime("%Y%m%d")
end = last_day.strftime("%Y%m%d")
url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": "T2M_MAX,T2M_MIN,RH2M,WS10M,PRECTOTCORR",
    "community": "AG",
    "longitude": RANCHI_LON,
    "latitude": RANCHI_LAT,
    "start": start,
    "end": end,
    "format": "JSON",
}
print(f"\nFetching NASA POWER daily data {start}-{end} for Ranchi...")
r = requests.get(url, params=params, timeout=120)
r.raise_for_status()
prop = r.json()["properties"]["parameter"]

climate = pd.DataFrame({
    "Date": pd.to_datetime(list(prop["T2M_MAX"].keys()), format="%Y%m%d"),
    "Tmax": list(prop["T2M_MAX"].values()),
    "Tmin": list(prop["T2M_MIN"].values()),
    "Humidity": list(prop["RH2M"].values()),
    "Wind_Speed": list(prop["WS10M"].values()),
    "Rainfall": list(prop["PRECTOTCORR"].values()),
})
# NASA uses -999 for missing
climate = climate.replace(-999, np.nan)
n_missing = climate[["Tmax", "Tmin", "Humidity", "Wind_Speed"]].isna().any(axis=1).sum()
print(f"Downloaded {len(climate)} days; days with missing values: {n_missing}")
if n_missing:
    print("Missing dates:", climate.loc[climate['Tmax'].isna(), 'Date'].dt.date.tolist())
# ---------------------------------------------------------------------------
# 4b. Backfill missing days (NASA POWER has ~5-day latency) from Open-Meteo
#     ERA5 archive (also authentic); wind in m/s at 10m, same as NASA WS10M
# ---------------------------------------------------------------------------
missing_dates = climate.loc[climate["Tmax"].isna(), "Date"]
if len(missing_dates):
    print(f"Backfilling {len(missing_dates)} days from Open-Meteo ERA5 archive...")
    om_url = "https://archive-api.open-meteo.com/v1/archive"
    om_params = {
        "latitude": RANCHI_LAT, "longitude": RANCHI_LON,
        "start_date": missing_dates.min().strftime("%Y-%m-%d"),
        "end_date": missing_dates.max().strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,wind_speed_10m_mean,precipitation_sum",
        "timezone": "Asia/Kolkata",
    }
    om = requests.get(om_url, params=om_params, timeout=120)
    om.raise_for_status()
    od = om.json()["daily"]
    om_df = pd.DataFrame({
        "Date": pd.to_datetime(od["time"]),
        "Tmax": od["temperature_2m_max"],
        "Tmin": od["temperature_2m_min"],
        "Humidity": od["relative_humidity_2m_mean"],
        "Wind_Speed": od["wind_speed_10m_mean"],
        "Rainfall": od["precipitation_sum"],
        "Climate_Source": "open-meteo_era5",
    })
    climate["Climate_Source"] = "nasa_power"
    for _, row in om_df.iterrows():
        mask = climate["Date"] == row["Date"]
        if mask.any():
            for c in ["Tmax", "Tmin", "Humidity", "Wind_Speed", "Rainfall"]:
                climate.loc[mask, c] = row[c]
            climate.loc[mask, "Climate_Source"] = row["Climate_Source"]
        else:  # date missing entirely from NASA response
            climate = pd.concat([climate, row.to_frame().T], ignore_index=True)
    climate = climate.sort_values("Date").reset_index(drop=True)
    still_missing = climate[["Tmax", "Tmin", "Humidity", "Wind_Speed"]].isna().any(axis=1).sum()
    print(f"After backfill - days still without climate: {still_missing}")
    print(climate.loc[climate['Tmax'].isna(), ['Date']].to_string(index=False))

climate.to_csv("results/climate_data_2026.csv", index=False)

# ---------------------------------------------------------------------------
# 5. THI (NRC 1971) - same formula as recalculate_thi_and_recreate_figure.py
# ---------------------------------------------------------------------------
def thi_nrc(tmax, tmin, rh):
    t_mean = (tmax + tmin) / 2
    t_f = 1.8 * t_mean + 32
    return t_f - ((0.55 - 0.0055 * rh) * (t_f - 58))

merged = dis.merge(climate, on="Date", how="left")
merged["THI"] = merged.apply(
    lambda r: thi_nrc(r["Tmax"], r["Tmin"], r["Humidity"]) if pd.notna(r["Tmax"]) else np.nan,
    axis=1)

# Training uses only days with available climate data
n_before = len(merged)
merged = merged.dropna(subset=["Tmax", "Tmin", "Humidity", "Wind_Speed"]).reset_index(drop=True)
print(f"\n2026 rows: {n_before} -> {len(merged)} after dropping days without climate data")

# ---------------------------------------------------------------------------
# 6. Merge with 2025 training data into one schema
#    2026 rows lack plot/spacing/instar/net-tech/pest info -> Unknown/0
# ---------------------------------------------------------------------------
df25 = pd.read_csv("results/cleaned_data_2025_corrected_thi.csv", low_memory=False)
df25["Date_parsed"] = pd.to_datetime(df25["Date_parsed"])

# one row per plot-date in 2025 (keep plot-level granularity)
df25_out = pd.DataFrame({
    "Year": 2025,
    "Plot_No": df25["Plot_No"],
    "Spacing": df25["Spacing"],
    "Instar": df25["Instar"],
    "Date": df25["Date_parsed"],
    "Tmax": df25["Tmax"],
    "Tmin": df25["Tmin"],
    "Humidity": df25["Humidity"],
    "Wind_Speed": df25["Wind_Speed"],
    "Rainfall": df25.get("Rainfall", np.nan),
    "THI": df25["THI_NRC"],          # corrected NRC THI for consistency
    "Net_Tech": df25["Net_Tech"],
    "Pests": df25["Pests"],
    "Has_Uzi": df25["Has_Uzi"],
    "Has_Mites": df25["Has_Mites"],
    "Has_Ants": df25["Has_Ants"],
    "Has_Spiders": df25["Has_Spiders"],
    "Has_Athropoda": df25["Has_Athropoda"],
    "Pebrine": df25["Pebrine"],
    "Virosis": df25["Virosis"],
    "Bacteriosis": df25["Bacteriosis"],
    "Muscardine": df25["Muscardine"],
    "Source": "field_2025",
})

df26_out = pd.DataFrame({
    "Year": 2026,
    "Plot_No": 0,
    "Spacing": "Unknown",
    "Instar": "Unknown",
    "Date": merged["Date"],
    "Tmax": merged["Tmax"],
    "Tmin": merged["Tmin"],
    "Humidity": merged["Humidity"],
    "Wind_Speed": merged["Wind_Speed"],
    "Rainfall": merged["Rainfall"],
    "THI": merged["THI"],
    "Net_Tech": "Unknown",
    "Pests": "",
    "Has_Uzi": 0,
    "Has_Mites": 0,
    "Has_Ants": 0,
    "Has_Spiders": 0,
    "Has_Athropoda": 0,
    "Pebrine": merged["Pebrine"],
    "Virosis": merged["Virosis"],
    "Bacteriosis": merged["Bacteriosis"],
    "Muscardine": merged["Muscardine"],
    "Source": merged["Source"],
})

combined = pd.concat([df25_out, df26_out], ignore_index=True)
combined.to_csv("results/merged_data_2025_2026.csv", index=False)

print("\n" + "=" * 70)
print("COMBINED DATASET SUMMARY")
print("=" * 70)
print(f"Total rows: {len(combined)}  (2025: {(combined['Year'] == 2025).sum()}, 2026: {(combined['Year'] == 2026).sum()})")
for d in ["Pebrine", "Virosis", "Bacteriosis", "Muscardine"]:
    sub = combined.groupby("Year")[d].apply(lambda s: (s > 0).sum())
    print(f"  {d}: 2025 positive rows={sub.get(2025, 0)}, 2026 positive rows={sub.get(2026, 0)}")
print("\nSaved:")
print("  results/disease_data_2026.csv")
print("  results/climate_data_2026.csv")
print("  results/merged_data_2025_2026.csv")
