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

import time
import requests

# Ranchi, Jharkhand
DEFAULT_LAT = 23.3441
DEFAULT_LON = 85.3096

CACHE_TTL_SECONDS = 30 * 60  # 30 minutes

_cache = {"key": None, "timestamp": 0, "data": None}


def thi_nrc(tmax, tmin, rh):
    """Temperature-Humidity Index, NRC (1971) formula.

    THI = (1.8*T + 32) - [(0.55 - 0.0055*RH) * (1.8*T - 58)], T = (Tmax+Tmin)/2
    """
    t_mean = (tmax + tmin) / 2
    t_f = 1.8 * t_mean + 32
    return round(t_f - ((0.55 - 0.0055 * rh) * (t_f - 58)), 2)


def _fetch_open_meteo(lat, lon, days):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ("temperature_2m_max,temperature_2m_min,"
                  "relative_humidity_2m_mean,wind_speed_10m_mean,precipitation_sum"),
        "timezone": "Asia/Kolkata",
        "forecast_days": days,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
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
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
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
            for row in data:
                row["thi"] = thi_nrc(row["tmax"], row["tmin"], row["humidity"])
            result = {"success": True, "location": {"lat": lat, "lon": lon}, "daily": data}
            _cache.update(key=key, timestamp=time.time(), data=result)
            return result
        except Exception as e:
            errors.append(f"{fetcher.__name__}: {e}")

    return {"success": False, "error": " | ".join(errors)}
