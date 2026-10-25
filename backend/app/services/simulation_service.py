"""
Simulated sensor telemetry for the digital twin.

No real structural-health-monitoring feed is publicly available for the MVP
bridges, so the LSTM input window is **simulated** (and labelled as such in
the API/UI). The generator is designed to be physically plausible:

* Load-driven channels (strain, deflection, tilt, …) share a latent load
  signal L(t) with a realistic weekday diurnal traffic profile and AR(1)
  noise, so they are cross-correlated like real SHM data.
* L(t) is offset per bridge from its profile (age, traffic, maintenance),
  expressed in standard-deviation units of the LSTM training distribution.
* Environmental channels use Mumbai-climatology-informed values
  (warm, humid, tidal water level).

Windows are deterministic within a time bucket (default 15 min) so that a
refresh shows the same state, while the twin still evolves over time.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np

from app.data.bridges import BridgeProfile

IST = timezone(timedelta(hours=5, minutes=30))
BUCKET_MINUTES = 15

# Loading of each channel on the latent load signal (unitless, 0–1)
LOAD_BETAS: Dict[str, float] = {
    "Strain_microstrain": 0.85,
    "Deflection_mm": 0.75,
    "Vibration_ms2": 0.60,
    "Tilt_deg": 0.70,
    "Displacement_mm": 0.55,
    "Vehicle_Load_tons": 0.80,
    "Traffic_Volume_vph": 0.85,
}

MAINTENANCE_ENCODING = {"No-Maintainance": 0, "Annual": 1, "Bi-Annual": 2}


@dataclass
class SimulatedWindow:
    bridge_id: str
    generated_at: datetime
    timestamps: List[str]
    channels: Dict[str, List[float]]
    load_offset_sigma: float

    def records(self, features: List[str]) -> List[Dict[str, float]]:
        n = len(self.timestamps)
        return [{f: float(self.channels[f][i]) for f in features} for i in range(n)]

    def latest(self) -> Dict[str, float]:
        return {k: float(v[-1]) for k, v in self.channels.items()}

    def mean(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.channels.items()}


def current_bucket(now: datetime | None = None) -> datetime:
    now = (now or datetime.now(IST)).astimezone(IST)
    minute = (now.minute // BUCKET_MINUTES) * BUCKET_MINUTES
    return now.replace(minute=minute, second=0, microsecond=0)


def _diurnal_traffic(hour: float) -> float:
    """Normalised weekday traffic profile with AM and PM peaks (≈0–1)."""
    am = math.exp(-((hour - 9.5) / 2.0) ** 2)
    pm = math.exp(-((hour - 19.0) / 2.5) ** 2)
    night = 0.15
    return min(1.0, night + 0.85 * max(am, pm))


def bridge_load_offset(profile: BridgeProfile) -> float:
    """Per-bridge offset of the latent load signal, in training-σ units."""
    traffic_load_factor = min(profile.peak_direction_vph / 5000.0, 1.0)
    age_factor = min(profile.age_years / profile.design_life_years, 1.0)
    maint = MAINTENANCE_ENCODING.get(profile.maintenance_level, 0)
    return 1.6 * (traffic_load_factor - 0.5) + 2.0 * age_factor + 0.35 * (1 - maint / 2)


def simulate_window(
    profile: BridgeProfile,
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
    seq_length: int = 60,
    now: datetime | None = None,
    weather_data: Dict[str, Any] | None = None,
    traffic_data: Dict[str, Any] | None = None,
) -> SimulatedWindow:
    bucket = current_bucket(now)
    seed = zlib.crc32(f"{profile.id}|{bucket.isoformat()}".encode())
    rng = np.random.default_rng(seed)

    times = [bucket - timedelta(minutes=seq_length - 1 - i) for i in range(seq_length)]
    hours = np.array([t.hour + t.minute / 60.0 for t in times])

    offset = bridge_load_offset(profile)
    diurnal = np.array([_diurnal_traffic(h) for h in hours])

    # AR(1) noise for the latent load
    ar = np.zeros(seq_length)
    for i in range(1, seq_length):
        ar[i] = 0.9 * ar[i - 1] + rng.normal(0, 0.25)
    
    # Scale latent load if dynamic traffic multiplier is provided
    traffic_mult = float(traffic_data.get("load_multiplier", 1.0)) if traffic_data else 1.0
    latent = (offset + 1.4 * (diurnal - 0.45) + ar) * traffic_mult

    channels: Dict[str, np.ndarray] = {}
    for name, beta in LOAD_BETAS.items():
        mu, sd = feature_means[name], feature_stds[name]
        idio = rng.normal(0, math.sqrt(max(1 - beta**2, 0.05)), seq_length) * 0.6
        channels[name] = mu + sd * (beta * latent + idio)

    # Environmental channels – Live Weather if provided, otherwise Mumbai climatology
    day_phase = 2 * math.pi * (hours - 15.0) / 24.0  # warmest ≈ 15:00
    if weather_data:
        live_temp = float(weather_data.get("temperature_c", 30.0))
        live_hum = float(weather_data.get("humidity_percent", 65.0))
        live_wind = float(weather_data.get("wind_speed_ms", 4.0))
        channels["Temperature_C"] = live_temp + 0.4 * np.cos(day_phase) + rng.normal(0, 0.2, seq_length)
        channels["Humidity_percent"] = np.clip(
            live_hum - 1.5 * np.cos(day_phase) + rng.normal(0, 0.8, seq_length), 30, 100
        )
        channels["Wind_Speed_ms"] = np.clip(
            live_wind + 0.5 * np.cos(day_phase) + rng.normal(0, 0.4, seq_length), 0, None
        )
    else:
        channels["Temperature_C"] = 29.5 + 2.8 * np.cos(day_phase) + rng.normal(0, 0.3, seq_length)
        channels["Humidity_percent"] = np.clip(
            76.0 - 6.0 * np.cos(day_phase) + rng.normal(0, 1.2, seq_length), 40, 100
        )
        gust = np.abs(rng.normal(0, 0.8, seq_length))
        channels["Wind_Speed_ms"] = 5.0 + 1.5 * np.cos(day_phase) + gust

    tide_phase = 2 * math.pi * (hours / 12.42)  # semi-diurnal tide
    channels["Water_Level_m"] = feature_means["Water_Level_m"] + 0.35 * np.sin(tide_phase + seed % 7)
    channels["Seismic_Activity_ms2"] = rng.normal(0, feature_stds["Seismic_Activity_ms2"] * 0.5, seq_length)
    channels["Dynamic_Load_Distribution_percent"] = (
        feature_means["Dynamic_Load_Distribution_percent"]
        + feature_stds["Dynamic_Load_Distribution_percent"] * (0.3 * latent + rng.normal(0, 0.5, seq_length))
    )

    # Physical floor: no negative magnitudes for load quantities
    for name in ("Vehicle_Load_tons", "Traffic_Volume_vph", "Vibration_ms2", "Strain_microstrain"):
        channels[name] = np.clip(channels[name], 0, None)

    return SimulatedWindow(
        bridge_id=profile.id,
        generated_at=bucket,
        timestamps=[t.isoformat() for t in times],
        channels={k: [round(float(x), 4) for x in v] for k, v in channels.items()},
        load_offset_sigma=round(offset, 3),
    )
