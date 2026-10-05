"""
Explainability layer: turns raw model attributions into labelled, ranked,
human-readable factors for the dashboard and the LLM narrative.
"""
from __future__ import annotations

from typing import Dict, List

XGB_FEATURE_LABELS: Dict[str, str] = {
    "maintenance_encoded": "Maintenance regime (encoded)",
    "Material_Type_Steel": "Material is steel",
    "age_factor": "Bridge age (years)",
    "traffic_load_factor": "Traffic volume (peak-dir vph)",
    "age_squared": "Age² (non-linear ageing)",
    "age_traffic_interaction": "Age × traffic interaction",
    "age_maintenance_interaction": "Age × maintenance interaction",
    "traffic_maintenance_interaction": "Traffic × maintenance interaction",
    "age_bin": "Age band (25-year bins)",
    "log_traffic": "log(Traffic volume)",
}

# How each engineered feature maps back to the four real-world inputs
XGB_FEATURE_GROUPS: Dict[str, Dict[str, float]] = {
    "maintenance_encoded": {"Maintenance": 1.0},
    "Material_Type_Steel": {"Material": 1.0},
    "age_factor": {"Age": 1.0},
    "age_squared": {"Age": 1.0},
    "age_bin": {"Age": 1.0},
    "traffic_load_factor": {"Traffic": 1.0},
    "log_traffic": {"Traffic": 1.0},
    "age_traffic_interaction": {"Age": 0.5, "Traffic": 0.5},
    "age_maintenance_interaction": {"Age": 0.5, "Maintenance": 0.5},
    "traffic_maintenance_interaction": {"Traffic": 0.5, "Maintenance": 0.5},
}

LSTM_FEATURE_LABELS: Dict[str, Dict[str, str]] = {
    "Strain_microstrain": {"label": "Strain", "unit": "µε"},
    "Deflection_mm": {"label": "Deflection", "unit": "mm"},
    "Vibration_ms2": {"label": "Vibration", "unit": "m/s²"},
    "Tilt_deg": {"label": "Tilt", "unit": "°"},
    "Displacement_mm": {"label": "Displacement", "unit": "mm"},
    "Temperature_C": {"label": "Temperature", "unit": "°C"},
    "Humidity_percent": {"label": "Humidity", "unit": "%"},
    "Wind_Speed_ms": {"label": "Wind speed", "unit": "m/s"},
    "Water_Level_m": {"label": "Water level", "unit": "m"},
    "Seismic_Activity_ms2": {"label": "Seismic activity", "unit": "m/s²"},
    "Vehicle_Load_tons": {"label": "Vehicle load", "unit": "t"},
    "Traffic_Volume_vph": {"label": "Traffic (sensor)", "unit": "vph"},
    "Dynamic_Load_Distribution_percent": {"label": "Dynamic load distribution", "unit": "%"},
}

ENGINEERING_LABELS: Dict[str, str] = {
    "age_factor": "Age factor",
    "traffic_load_factor": "Traffic / load factor",
    "environmental_degradation": "Environmental degradation",
    "corrosion_factor": "Corrosion / degradation",
    "fatigue_factor": "Fatigue",
    "inspection_condition_factor": "Inspection condition",
}


def explain_xgb(raw: Dict[str, object]) -> Dict[str, object]:
    contributions: Dict[str, float] = raw["contributions"]  # type: ignore[assignment]
    values: Dict[str, float] = raw["feature_values"]  # type: ignore[assignment]

    features: List[Dict[str, object]] = sorted(
        (
            {
                "feature": f,
                "label": XGB_FEATURE_LABELS.get(f, f),
                "value": round(values[f], 4),
                "shap_logodds": round(c, 4),
                "direction": "increases_risk" if c > 0 else "decreases_risk",
            }
            for f, c in contributions.items()
        ),
        key=lambda d: abs(d["shap_logodds"]),  # type: ignore[arg-type]
        reverse=True,
    )

    grouped: Dict[str, float] = {}
    for f, c in contributions.items():
        for group, share in XGB_FEATURE_GROUPS.get(f, {f: 1.0}).items():
            grouped[group] = grouped.get(group, 0.0) + c * share
    total_abs = sum(abs(v) for v in grouped.values()) or 1.0
    grouped_list = sorted(
        (
            {
                "group": g,
                "shap_logodds": round(v, 4),
                "share_percent": round(100 * abs(v) / total_abs, 1),
                "direction": "increases_risk" if v > 0 else "decreases_risk",
            }
            for g, v in grouped.items()
        ),
        key=lambda d: abs(d["shap_logodds"]),  # type: ignore[arg-type]
        reverse=True,
    )

    return {
        "method": "TreeSHAP (XGBoost native pred_contribs)",
        "base_value_logodds": round(float(raw["base_value_logodds"]), 4),  # type: ignore[arg-type]
        "features": features,
        "grouped": grouped_list,
    }


def explain_lstm(
    attribution: Dict[str, float],
    window_means: Dict[str, float],
    train_mean: Dict[str, float],
    train_std: Dict[str, float],
) -> Dict[str, object]:
    total_abs = sum(abs(v) for v in attribution.values()) or 1.0
    items = []
    for name, delta in attribution.items():
        meta = LSTM_FEATURE_LABELS.get(name, {"label": name, "unit": ""})
        value = window_means.get(name, 0.0)
        z = (value - train_mean[name]) / train_std[name] if train_std.get(name) else 0.0
        items.append(
            {
                "feature": name,
                "label": meta["label"],
                "unit": meta["unit"],
                "window_mean": round(value, 4),
                "training_mean": round(train_mean[name], 4),
                "z_score": round(z, 2),
                "shi_delta": round(delta, 5),
                "share_percent": round(100 * abs(delta) / total_abs, 1),
                "direction": "lowers_health" if delta < 0 else "raises_health",
            }
        )
    items.sort(key=lambda d: abs(d["shi_delta"]), reverse=True)
    return {
        "method": "Channel occlusion (replace channel with training mean, measure ΔSHI)",
        "features": items,
    }


def rank_engineering(factors: Dict[str, float], weights: Dict[str, float]) -> List[Dict[str, object]]:
    total = sum(weights[k] * factors[k] for k in weights) or 1.0
    ranked = [
        {
            "factor": k,
            "label": ENGINEERING_LABELS[k],
            "value": round(factors[k], 4),
            "weight": weights[k],
            "weighted_share_percent": round(100 * weights[k] * factors[k] / total, 1),
        }
        for k in weights
    ]
    ranked.sort(key=lambda d: d["weighted_share_percent"], reverse=True)  # type: ignore[arg-type,return-value]
    return ranked
