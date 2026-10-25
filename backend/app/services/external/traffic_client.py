"""
Traffic Service Client for BridgeTwinAI.

Estimates real-time traffic volume, congestion, and structural dynamic load multiplier
for bridge assets.

Features:
- Optional TomTom Traffic Flow API integration via TOMTOM_API_KEY environment variable.
- Calibrated Mumbai Metropolitan diurnal traffic model as high-fidelity fallback.
- Computes structural load multiplier (0.80 - 1.35x) to scale latent bridge strain & deflection.
- 15-minute in-memory caching.
- Explicit data provenance tracking.
"""
import os
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import httpx

from app.data.bridges import BridgeProfile

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_EXPIRY_SECONDS = 900  # 15 minutes

_TRAFFIC_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}

def _diurnal_mumbai_traffic(profile: BridgeProfile, now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Computes calibrated Mumbai traffic flow based on bridge ADT and Indian Standard Time rush hours.
    Morning peak: 09:00 - 11:30 (Business commute towards Mumbai island city / BKC / Navi Mumbai)
    Evening peak: 17:30 - 20:30 (Return commute)
    Midday: Moderate
    Night: Low flow
    """
    now = (now or datetime.now(IST)).astimezone(IST)
    hour = now.hour + now.minute / 60.0
    
    # Dual Gaussian peaks for Mumbai arterial bridges
    am_peak = math.exp(-((hour - 10.0) / 1.8) ** 2)
    pm_peak = math.exp(-((hour - 19.0) / 2.2) ** 2)
    night_trough = 0.15
    flow_intensity = min(1.0, night_trough + 0.85 * max(am_peak, pm_peak))
    
    # Calculate current vehicle volume
    peak_vph = profile.peak_direction_vph
    current_vph = round(peak_vph * flow_intensity, 1)
    
    # Congestion index & speed ratio
    if flow_intensity > 0.80:
        congestion_level = "Heavy"
        speed_ratio = 0.55
        load_multiplier = 1.25  # High bumper-to-bumper vehicle weight on deck
    elif flow_intensity > 0.60:
        congestion_level = "Moderate"
        speed_ratio = 0.75
        load_multiplier = 1.10
    elif flow_intensity > 0.30:
        congestion_level = "Normal"
        speed_ratio = 0.95
        load_multiplier = 1.00
    else:
        congestion_level = "Free Flow"
        speed_ratio = 1.00
        load_multiplier = 0.85
        
    return {
        "current_vph": current_vph,
        "peak_vph_capacity": peak_vph,
        "flow_intensity": round(flow_intensity, 3),
        "congestion_level": congestion_level,
        "speed_ratio": speed_ratio,
        "load_multiplier": load_multiplier,
        "provenance": "estimated from diurnal profile",
        "source": "Calibrated Mumbai Diurnal Arterial Model",
        "is_live": False,
        "timestamp": now.isoformat()
    }

def get_current_traffic(profile: BridgeProfile, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Retrieves current traffic conditions for a bridge profile.
    Checks cache first, then checks for live TomTom Flow API key,
    falling back to Mumbai diurnal model.
    """
    cache_key = profile.id
    now_ts = time.time()
    
    if not force_refresh and cache_key in _TRAFFIC_CACHE:
        cached_time, cached_val = _TRAFFIC_CACHE[cache_key]
        if now_ts - cached_time < CACHE_EXPIRY_SECONDS:
            return cached_val

    api_key = os.getenv("TOMTOM_API_KEY")
    if api_key:
        try:
            url = (
                f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
                f"?point={profile.lat},{profile.lon}&key={api_key}"
            )
            with httpx.Client(timeout=3.5) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    flow_data = resp.json().get("flowSegmentData", {})
                    curr_speed = float(flow_data.get("currentSpeed", 60))
                    free_speed = float(flow_data.get("freeFlowSpeed", 80))
                    speed_ratio = round(curr_speed / max(free_speed, 1.0), 2)
                    
                    if speed_ratio < 0.5:
                        congestion = "Severe"
                        load_mult = 1.30
                    elif speed_ratio < 0.7:
                        congestion = "Heavy"
                        load_mult = 1.20
                    elif speed_ratio < 0.85:
                        congestion = "Moderate"
                        load_mult = 1.08
                    else:
                        congestion = "Normal"
                        load_mult = 0.95
                        
                    current_vph = round(profile.peak_direction_vph * (1.1 - 0.3 * speed_ratio), 1)
                    
                    result = {
                        "current_vph": current_vph,
                        "peak_vph_capacity": profile.peak_direction_vph,
                        "speed_ratio": speed_ratio,
                        "current_speed_kmh": curr_speed,
                        "free_flow_speed_kmh": free_speed,
                        "congestion_level": congestion,
                        "load_multiplier": load_mult,
                        "provenance": "public (live)",
                        "source": "TomTom Flow API",
                        "is_live": True,
                        "timestamp": datetime.now(IST).isoformat()
                    }
                    _TRAFFIC_CACHE[cache_key] = (now_ts, result)
                    return result
        except Exception as exc:
            print(f"[TrafficClient] TomTom live query failed ({exc}), reverting to diurnal model.")

    fallback = _diurnal_mumbai_traffic(profile)
    _TRAFFIC_CACHE[cache_key] = (now_ts, fallback)
    return fallback
