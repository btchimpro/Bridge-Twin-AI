"""
Weather Service Client for BridgeTwinAI.

Fetches real-time ambient weather for bridge geospatial coordinates using Open-Meteo
(free, no API key required, reliable for global meteorological observations).

Features:
- 15-minute in-memory caching to minimize external requests and maintain determinism.
- Graceful fallback to Mumbai climatological simulation if offline or on timeout.
- Explicit data provenance tracking ("public (live)" vs "simulated").
"""
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import httpx

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_EXPIRY_SECONDS = 900  # 15 minutes

# In-memory cache: key -> (timestamp, data)
_WEATHER_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}

def _climatological_fallback(lat: float, lon: float, now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Computes a realistic Mumbai climatological weather estimate based on time of day.
    Used as an offline/timeout fallback.
    """
    now = (now or datetime.now(IST)).astimezone(IST)
    hour = now.hour + now.minute / 60.0
    import math
    
    # Warmest around 15:00, coolest around 06:00
    phase = 2 * math.pi * (hour - 15.0) / 24.0
    temp = round(29.5 + 2.8 * math.cos(phase), 1)
    humidity = round(max(40.0, min(100.0, 76.0 - 6.0 * math.cos(phase))), 1)
    wind = round(max(1.0, 5.0 + 1.5 * math.cos(phase)), 1)
    
    return {
        "temperature_c": temp,
        "humidity_percent": humidity,
        "wind_speed_ms": wind,
        "precipitation_mm": 0.0,
        "provenance": "simulated",
        "source": "Mumbai Climatological Diurnal Model (Offline Fallback)",
        "is_live": False,
        "fetched_at": now.isoformat()
    }

def get_current_weather(lat: float, lon: float, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Retrieves current weather for a bridge location.
    Checks 15-minute cache first, then attempts live Open-Meteo API call,
    falling back to Mumbai climatology on any error.
    """
    cache_key = f"{round(lat, 3)}_{round(lon, 3)}"
    now_ts = time.time()
    
    if not force_refresh and cache_key in _WEATHER_CACHE:
        cached_time, cached_val = _WEATHER_CACHE[cache_key]
        if now_ts - cached_time < CACHE_EXPIRY_SECONDS:
            return cached_val

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        f"&timezone=Asia/Kolkata"
    )
    
    try:
        with httpx.Client(timeout=3.5) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json().get("current", {})
                result = {
                    "temperature_c": float(data.get("temperature_2m", 30.0)),
                    "humidity_percent": float(data.get("relative_humidity_2m", 65.0)),
                    "wind_speed_ms": float(data.get("wind_speed_10m", 4.0)),
                    "precipitation_mm": float(data.get("precipitation", 0.0)),
                    "provenance": "public (live)",
                    "source": "Open-Meteo Live API",
                    "is_live": True,
                    "fetched_at": datetime.now(IST).isoformat()
                }
                _WEATHER_CACHE[cache_key] = (now_ts, result)
                return result
    except Exception as exc:
        print(f"[WeatherClient] Live fetch failed ({exc}), activating climatological fallback.")

    fallback = _climatological_fallback(lat, lon)
    _WEATHER_CACHE[cache_key] = (now_ts, fallback)
    return fallback
