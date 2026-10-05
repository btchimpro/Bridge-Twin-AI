"""
Physics-informed engineering indicators.

These are *engineering-inspired estimates for a student prototype*, NOT
certified structural calculations. Each indicator is normalised to 0–1 where
higher means *worse* (more degradation / more demand), and every formula is
documented in ``docs/ai-pipeline.md``.

Indicators
----------
age_factor                     age / design life
traffic_load_factor            peak-direction volume / training maximum (5,000 vph)
environmental_degradation      marine exposure + humidity + temperature
corrosion_factor               Tuutti-style initiation/propagation (sigmoid) × environment
fatigue_factor                 Palmgren–Miner damage with Basquin stress scaling (m = 3)
inspection_condition_factor    inverse of maintenance regime
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict

from app.data.bridges import HEAVY_VEHICLE_SHARE, BridgeProfile

MAINTENANCE_ENCODING = {"No-Maintainance": 0, "Annual": 1, "Bi-Annual": 2}
INSPECTION_FACTOR = {"No-Maintainance": 0.85, "Annual": 0.45, "Bi-Annual": 0.25}

# Corrosion initiation time (years) before maintenance benefit – assumed values
# typical of marine exposure: chloride-induced for concrete cover, coating life for steel.
CORROSION_INITIATION_YEARS = {"Concrete": 22.0, "Steel": 14.0}

# Reference number of heavy-vehicle stress cycles to fatigue "end of life" (assumed)
FATIGUE_REFERENCE_CYCLES = 2.0e8
BASQUIN_EXPONENT = 3.0

# Weights for the engineering composite (sum = 1)
ENGINEERING_WEIGHTS: Dict[str, float] = {
    "age_factor": 0.15,
    "traffic_load_factor": 0.20,
    "environmental_degradation": 0.20,
    "corrosion_factor": 0.20,
    "fatigue_factor": 0.15,
    "inspection_condition_factor": 0.10,
}


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


@dataclass
class EngineeringIndicators:
    age_factor: float
    traffic_load_factor: float
    environmental_degradation: float
    corrosion_factor: float
    fatigue_factor: float
    inspection_condition_factor: float
    engineering_composite: float
    annual_degradation_rate: float  # SHI points (0–1 scale) lost per year, no intervention
    intermediate: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def compute_indicators(
    profile: BridgeProfile,
    env_means: Dict[str, float],
    strain_mean: float,
    strain_reference: float,
) -> EngineeringIndicators:
    age = profile.age_years
    maint = MAINTENANCE_ENCODING.get(profile.maintenance_level, 0)

    age_factor = _clip01(age / profile.design_life_years)
    traffic_load_factor = _clip01(profile.peak_direction_vph / 5000.0)

    # Environmental degradation: marine exposure dominates for all MVP bridges
    exposure = 1.0 if "severe" in profile.exposure_class.lower() else 0.8 if "marine" in profile.exposure_class.lower() else 0.4
    rh = env_means.get("Humidity_percent", 70.0)
    temp = env_means.get("Temperature_C", 25.0)
    environmental = _clip01(0.55 * exposure + 0.30 * _clip01((rh - 55) / 40) + 0.15 * _clip01((temp - 15) / 20))

    # Corrosion: sigmoid around initiation time, extended by maintenance regime
    t_init = CORROSION_INITIATION_YEARS[profile.material_type] * (1 + 0.35 * maint)
    propagation = 1 / (1 + math.exp(-(age - t_init) / 5.0))
    corrosion = _clip01(environmental * (0.15 + 0.85 * propagation))

    # Fatigue: Miner damage from cumulative heavy-vehicle passages, scaled by
    # current stress-range ratio (Basquin S–N curve, m = 3)
    cycles = profile.adt_vehicles_per_day * HEAVY_VEHICLE_SHARE * 365 * age
    stress_ratio = max(strain_mean / strain_reference, 0.1)
    miner_damage = (cycles / FATIGUE_REFERENCE_CYCLES) * stress_ratio**BASQUIN_EXPONENT
    fatigue = _clip01(miner_damage)

    inspection = INSPECTION_FACTOR.get(profile.maintenance_level, 0.6)

    factors = {
        "age_factor": age_factor,
        "traffic_load_factor": traffic_load_factor,
        "environmental_degradation": environmental,
        "corrosion_factor": corrosion,
        "fatigue_factor": fatigue,
        "inspection_condition_factor": inspection,
    }
    composite = sum(ENGINEERING_WEIGHTS[k] * v for k, v in factors.items())

    # Annual SHI loss without intervention: base ageing + accelerated by mechanisms
    annual_rate = 0.004 * (1 + 1.5 * corrosion + 1.5 * fatigue + 0.8 * traffic_load_factor + 0.5 * inspection)

    return EngineeringIndicators(
        **{k: round(v, 4) for k, v in factors.items()},
        engineering_composite=round(composite, 4),
        annual_degradation_rate=round(annual_rate, 5),
        intermediate={
            "age_years": age,
            "corrosion_initiation_years": round(t_init, 1),
            "corrosion_propagation": round(propagation, 4),
            "heavy_vehicle_cycles": round(cycles),
            "stress_ratio": round(stress_ratio, 4),
            "miner_damage": round(miner_damage, 5),
            "exposure_coefficient": exposure,
        },
    )
