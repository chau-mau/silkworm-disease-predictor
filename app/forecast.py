"""
Real-time weather forecast for disease risk prediction.

Data sources (public domain):
  1. Open-Meteo Forecast API (primary, free, no API key)
     https://api.open-meteo.com/v1/forecast
  2. NASA POWER daily endpoint (fallback)

Note on IMD/Mausam: the official IMD APIs (api.imd.gov.in, city.imd.gov.in,
mausam.imd.gov.in) require the caller's IP/domain to be whitelisted by IMD,
so they cannot be queried directly from an arbitrary server. If IMD access
is whitelisted for your deployment, add an IMD provider here as the first
entry in the fallback chain.

Responses are cached in memory for 30 minutes to avoid hammering the APIs.
"""

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests

# Ranchi, Jharkhand
DEFAULT_LAT = 23.3441
DEFAULT_LON = 85.3096

CACHE_TTL_SECONDS = 60 * 60  # 60 minutes

# Open-Meteo asks API consumers to identify themselves; this helps avoid blocks.
_REQUEST_HEADERS = {
    "User-Agent": "TasarSilkwormDiseasePredictor/1.0 (research; contact: nidhisukhija5@gmail.com)"
}

# Optional: set OPEN_METEO_API_KEY env var to use the higher-limit customer endpoint.
OPEN_METEO_API_KEY = os.environ.get("OPEN_METEO_API_KEY", "")

_cache = {"key": None, "timestamp": 0, "data": None}


def _load_historical_fallback():
    """Load the typical-year Ranchi climate calendar as a last-resort fallback."""
    try:
        calendar_path = Path(__file__).resolve().parent.parent / "docs" / "calendar_data.json"
        with open(calendar_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("records", [])
    except Exception:
        return []


def _historical_forecast(days, start_date=None):
    """Return the next N days from the typical-year calendar (5-year average)."""
    records = _load_historical_fallback()
    if not records:
        return None
    by_doy = {r["date"][5:]: r for r in records}
    start = start_date or date.today()
    out = []
    for i in range(days):
        d = start + timedelta(days=i)
        doy = d.strftime("%m-%d")
        rec = by_doy.get(doy)
        if rec:
            out.append({
                "date": d.isoformat(),
                "tmax": rec["tmax"],
                "tmin": rec["tmin"],
                "humidity": rec["humidity"],
                "wind_speed": rec["wind_speed"],
                "rainfall": None,
                "thi": rec["thi"],
                "source": "5-year typical climate (NASA POWER 2021-2025)",
            })
    return out


def thi_nrc(tmax, tmin, rh):
    """Temperature-Humidity Index, NRC (1971) formula.

    THI = (1.8*T + 32) - [(0.55 - 0.0055*RH) * (1.8*T - 58)], T = (Tmax+Tmin)/2
    """
    t_mean = (tmax + tmin) / 2
    t_f = 1.8 * t_mean + 32
    return round(t_f - ((0.55 - 0.0055 * rh) * (t_f - 58)), 2)


def _fetch_open_meteo(lat, lon, days):
    if OPEN_METEO_API_KEY:
        url = "https://customer-api.open-meteo.com/v1/forecast"
        params = {"apikey": OPEN_METEO_API_KEY}
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {}
    params.update({
        "latitude": lat,
        "longitude": lon,
        "daily": ("temperature_2m_max,temperature_2m_min,"
                  "relative_humidity_2m_mean,wind_speed_10m_mean,precipitation_sum"),
        "timezone": "Asia/Kolkata",
        "forecast_days": days,
    })
    r = requests.get(url, params=params, headers=_REQUEST_HEADERS, timeout=30)
    r.raise_for_status()
    payload = r.json()
    d = payload.get("daily") or {}
    if not d.get("time"):
        raise ValueError(f"Open-Meteo returned empty daily data: {payload.keys()}")
    return [
        {
            "date": d["time"][i],
            "tmax": d["temperature_2m_max"][i],
            "tmin": d["temperature_2m_min"][i],
            "humidity": d["relative_humidity_2m_mean"][i],
            "wind_speed": d["wind_speed_10m_mean"][i],
            "rainfall": d["precipitation_sum"][i],
            "source": "Open-Meteo Forecast",
        }
        for i in range(len(d["time"]))
    ]


def _fetch_nasa_power(lat, lon, days):
    """Fallback: NASA POWER daily endpoint (near-term days are forecast values)."""
    from datetime import date, timedelta
    start = date.today()
    end = start + timedelta(days=days - 1)
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M_MAX,T2M_MIN,RH2M,WS10M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    r = requests.get(url, params=params, headers=_REQUEST_HEADERS, timeout=30)
    r.raise_for_status()
    payload = r.json()
    p = payload.get("properties", {}).get("parameter", {})
    if not p or "T2M_MAX" not in p:
        raise ValueError(f"NASA POWER returned unexpected structure: {payload.keys()}")
    out = []
    for k in sorted(p["T2M_MAX"]):
        if p["T2M_MAX"][k] == -999:
            continue
        out.append({
            "date": f"{k[:4]}-{k[4:6]}-{k[6:]}",
            "tmax": p["T2M_MAX"][k],
            "tmin": p["T2M_MIN"][k],
            "humidity": p["RH2M"][k],
            "wind_speed": p["WS10M"][k],
            "rainfall": p["PRECTOTCORR"][k],
            "source": "NASA POWER",
        })
    return out


def get_forecast(lat=DEFAULT_LAT, lon=DEFAULT_LON, days=7):
    """Return list of daily forecast dicts (cached for 30 minutes)."""
    days = max(1, min(int(days), 16))
    key = (round(float(lat), 4), round(float(lon), 4), days)

    if _cache["key"] == key and time.time() - _cache["timestamp"] < CACHE_TTL_SECONDS:
        return _cache["data"]

    errors = []
    for fetcher in (_fetch_open_meteo, _fetch_nasa_power):
        try:
            data = fetcher(lat, lon, days)
            if not data:
                raise ValueError(f"{fetcher.__name__} returned empty forecast")
            for row in data:
                row["thi"] = thi_nrc(row["tmax"], row["tmin"], row["humidity"])
            result = {"success": True, "location": {"lat": lat, "lon": lon}, "daily": data}
            _cache.update(key=key, timestamp=time.time(), data=result)
            return result
        except Exception as e:
            errors.append(f"{fetcher.__name__}: {e}")

    # Last resort: typical-year historical climate so the UI still shows useful data.
    hist = _historical_forecast(days)
    if hist:
        result = {
            "success": True,
            "location": {"lat": lat, "lon": lon},
            "daily": hist,
            "warning": "Live weather APIs unavailable. Showing 5-year typical climate instead.",
        }
        _cache.update(key=key, timestamp=time.time(), data=result)
        return result

    return {"success": False, "error": " | ".join(errors)}
