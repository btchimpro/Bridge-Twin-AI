"""
LLM narrative and predictive maintenance recommendation service for the dashboard.
Integrates Gemini 2.5 Flash for detailed civil engineering directives,
with a robust domain-grounded fallback generator and 15-minute response caching.
"""
import os
import json
import time
from typing import Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from app.data.bridges import BridgeProfile

# Cache: bridge_id + bucket_time -> response
_LLM_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
CACHE_EXPIRY_SECONDS = 900  # 15 mins

def _fallback_generate(
    bridge: BridgeProfile,
    risk: Dict[str, Any],
    shi: Dict[str, Any],
    xai: Dict[str, Any],
    eng: Dict[str, Any],
    fused: Optional[Dict[str, Any]] = None,
    weather: Optional[Dict[str, Any]] = None,
    traffic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Civil-engineering rule-based detailed generator when LLM is offline or unconfigured."""
    is_at_risk = risk.get("is_at_risk", False)
    shi_val = float(shi.get("predicted_shi", 0.85))
    priority = fused.get("priority", "High" if is_at_risk else "Low") if fused else ("High" if is_at_risk else "Low")
    overall_risk_pct = fused.get("overall_risk_score", 0.3) * 100.0 if fused else (risk.get("risk_probability", 0.1) * 100.0)
    
    top_shi_feat = [f["label"] for f in xai.get("lstm_explain", {}).get("features", []) if f.get("direction") == "lowers_health"]
    top_eng_factors = [ef.get("label", "") for ef in xai.get("engineering_factors", [])[:3]]
    primary_eng = top_eng_factors[0] if top_eng_factors else "Marine Environmental Degradation"
    
    temp = weather.get('temperature_c', 30) if weather else 30
    hum = weather.get('humidity_percent', 65) if weather else 65
    wind = weather.get('wind_speed_ms', 4.5) if weather else 4.5
    cong = traffic.get('congestion_level', 'Normal') if traffic else 'Normal'
    vph = traffic.get('current_vph', 2500) if traffic else 2500
    load_mult = traffic.get('load_multiplier', 1.0) if traffic else 1.0

    narrative = (
        f"Structural integrity evaluation for {bridge.name} ({bridge.structure_type}, {bridge.material_type}, age {bridge.age_years} yrs) "
        f"indicates an operational Structural Health Index (SHI) of {shi_val*100:.1f}/100 with an aggregate unified risk of {overall_risk_pct:.1f}% ({priority} Priority). "
        f"Current degradation dynamics are primarily governed by {primary_eng.lower()} under harsh {bridge.exposure_class} marine exposure. "
        f"Ambient atmospheric conditions ({temp}°C, {hum}% RH, {wind} m/s wind) combined with active {cong.lower()} vehicular flow ({vph:,} vph, {load_mult:.2f}x dynamic load factor) "
        f"are inducing cumulative micro-strain oscillations across primary load-bearing members. "
    )
    if top_shi_feat:
        narrative += f"Real-time sensor telemetry isolates {top_shi_feat[0]} as the most active stress concentration vector. "
    if is_at_risk:
        narrative += f"Immediate preventive remediation is advised to halt localized section loss and arrest accelerated fatigue crack nucleation."
    else:
        narrative += f"Overall structural response remains within safe design tolerances; continued continuous telemetry surveillance is recommended."

    root_cause = (
        f"The primary driver of structural degradation on {bridge.name} is the coupled interaction of {primary_eng.lower()} and "
        f"cyclic traffic excitation. Marine airborne chlorides penetrate micro-cracks in the concrete/steel interface, "
        f"accelerating depassivation of reinforcing steel at an estimated loss rate of {((eng.get('annual_degradation_rate', 0.005))*100):.2f}% per year. "
        f"During peak traffic hours ({vph:,} vph), dynamic strain amplitudes exacerbate micro-fissure propagation along critical spans."
    )

    actions = [
        f"Conduct non-destructive testing (NDT) focusing on {top_shi_feat[0] if top_shi_feat else 'critical girder junctions'} to quantify micro-strain anomalies.",
        f"Apply cathodic protection check and silane-based hydrophobic sealants to counteract {primary_eng.lower()} in splash zones.",
        f"Verify bearing pad alignment, thermal expansion joint clearances, and deck drainage efficiency.",
        f"Recalibrate real-time acoustic emission and vibration strain sensors to establish updated baseline thresholds.",
        f"Implement scheduled re-evaluation within {'7 days' if priority == 'Critical' else '30 days' if priority == 'High' else '90 days'}."
    ]

    inspection_protocol = (
        f"Execute ultrasonic pulse velocity (UPV) and half-cell potential mapping across piers and tidal splash zones. "
        f"Perform high-frequency (100 Hz) dynamic strain gauge diagnostics during transit peak hours to measure peak-to-peak deflection amplitudes."
    )

    equipment = [
        "Ultrasonic Flaw Detector & UPV Meter",
        "Half-Cell Corrosion Potential Meter",
        "High-Speed Dynamic Strain Data Logger",
        "Mobile Under-Bridge Inspection Platform (UBIP)",
        "Precision Laser Vibrometer"
    ]

    mitigations = [
        f"Enforce dynamic weight limit of 40 tonnes and reduce freight speed to 40 km/h during heavy congestion periods ({vph:,} vph).",
        "Establish 24/7 automated alert thresholds on strain gauges exceeding 250 microstrain.",
        "Restrict heavy axle transit across vulnerable lanes during adverse tidal and high-humidity episodes."
    ]

    timeframe = (
        "Within 7 calendar days (Emergency Interventions)" if priority == "Critical"
        else "Within 30 calendar days (Priority Action)" if priority == "High"
        else "Within 90 calendar days (Next Scheduled Quarter)" if priority == "Medium"
        else "Routine Annual Maintenance Cycle"
    )

    justification = (
        f"Unified assessment driven by {primary_eng.lower()} (accounting for dominant degradation share) "
        f"and real-time SHI telemetry ({shi_val*100:.1f}%), requiring coordinated preventative maintenance to guarantee long-term serviceability."
    )

    return {
        "analysis_narrative": narrative,
        "maintenance_plan": {
            "priority": priority,
            "estimated_timeframe": timeframe,
            "root_cause_analysis": root_cause,
            "recommended_actions": actions,
            "inspection_protocol": inspection_protocol,
            "required_equipment": equipment,
            "safety_mitigations": mitigations,
            "justification": justification
        }
    }


def generate_narrative_and_plan(
    bridge: BridgeProfile,
    risk: Dict[str, Any],
    shi: Dict[str, Any],
    xai: Dict[str, Any],
    eng: Dict[str, Any],
    fused: Optional[Dict[str, Any]] = None,
    weather: Optional[Dict[str, Any]] = None,
    traffic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate detailed dashboard narrative and predictive maintenance plan using Gemini 2.5 Flash if available,
    otherwise fallback to rule-based engineering generation. Includes 15-minute caching.
    """
    cache_key = f"{bridge.id}_{int(time.time() // CACHE_EXPIRY_SECONDS)}"
    if cache_key in _LLM_CACHE:
        _, cached_val = _LLM_CACHE[cache_key]
        return cached_val

    api_key = os.getenv("GEMINI_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "gemini")

    if not api_key or not HAS_GENAI or provider.lower() != "gemini":
        result = _fallback_generate(bridge, risk, shi, xai, eng, fused, weather, traffic)
        _LLM_CACHE[cache_key] = (time.time(), result)
        return result

    try:
        client = genai.Client(api_key=api_key)
        
        context = {
            "bridge": {
                "name": bridge.name,
                "type": bridge.structure_type,
                "age": bridge.age_years,
                "material": bridge.material_type,
                "environment": bridge.exposure_class,
                "length_m": bridge.length_m
            },
            "fused_risk_assessment": {
                "overall_risk_score": fused.get("overall_risk_score") if fused else risk.get("risk_probability"),
                "priority": fused.get("priority") if fused else ("High" if risk.get("is_at_risk") else "Low"),
                "confidence_score": fused.get("confidence_score", 0.85) if fused else 0.85,
                "dominant_stressor": fused.get("dominant_stressor") if fused else "Material & Environmental Ageing",
                "structural_health_index": shi.get("predicted_shi", 0.85),
                "shi_uncertainty_std": shi.get("mc_std", 0.02)
            },
            "live_ambient_environment": {
                "temperature_c": weather.get("temperature_c", 30) if weather else 30,
                "humidity_percent": weather.get("humidity_percent", 65) if weather else 65,
                "wind_speed_ms": weather.get("wind_speed_ms", 4.5) if weather else 4.5,
                "traffic_congestion": traffic.get("congestion_level", "Normal") if traffic else "Normal",
                "traffic_vph": traffic.get("current_vph", 2500) if traffic else 2500,
                "traffic_load_multiplier": traffic.get("load_multiplier", 1.0) if traffic else 1.0
            },
            "explainability_and_physics": {
                "dominant_engineering_factors": xai.get("engineering_factors", [])[:3],
                "top_sensor_stressors": xai.get("lstm_explain", {}).get("features", [])[:3],
                "annual_degradation_rate": eng.get("annual_degradation_rate", 0.005)
            }
        }
        
        prompt = (
            "You are a Senior Principal Civil and Structural Health Monitoring Engineer analyzing the real-time Digital Twin of an urban bridge in Mumbai.\n"
            "Based strictly on the provided multi-model data, physics indicators, and live environmental telemetry, produce a detailed engineering directive and predictive maintenance plan.\n\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Instructions:\n"
            "1. In 'analysis_narrative', provide a detailed, authoritative structural health diagnosis (3 to 5 comprehensive sentences). Cite the bridge asset type, age, Structural Health Index (SHI), aggregate risk score, dominant degradation mechanism, and environmental impact (temperature, humidity, traffic load).\n"
            "2. In 'root_cause_analysis', explain the precise chemical/physical mechanisms causing the degradation (e.g., chloride ingress in marine splash zone, cyclic fatigue under high dynamic traffic, thermal expansion micro-cracking).\n"
            "3. In 'recommended_actions', list 4 to 5 specific, prioritized, practical engineering work packages.\n"
            "4. In 'inspection_protocol', prescribe targeted non-destructive testing (NDT) techniques (e.g. ultrasonic flaw detection, half-cell potential mapping, dynamic strain monitoring).\n"
            "5. In 'required_equipment', list 4 to 5 specialized inspection tools and machinery needed.\n"
            "6. In 'safety_mitigations', provide 2 to 3 operational traffic management and safety constraints to minimize structural stress.\n"
            "7. Never fabricate arbitrary numbers; ground every claim strictly in the input JSON."
        )

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "analysis_narrative": {"type": "STRING"},
                "maintenance_plan": {
                    "type": "OBJECT",
                    "properties": {
                        "priority": {"type": "STRING", "enum": ["Low", "Medium", "High", "Critical"]},
                        "estimated_timeframe": {"type": "STRING"},
                        "root_cause_analysis": {"type": "STRING"},
                        "recommended_actions": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        },
                        "inspection_protocol": {"type": "STRING"},
                        "required_equipment": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        },
                        "safety_mitigations": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        },
                        "justification": {"type": "STRING"}
                    },
                    "required": [
                        "priority",
                        "estimated_timeframe",
                        "root_cause_analysis",
                        "recommended_actions",
                        "inspection_protocol",
                        "required_equipment",
                        "safety_mitigations",
                        "justification"
                    ]
                }
            },
            "required": ["analysis_narrative", "maintenance_plan"]
        }

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2,
            ),
        )
        
        parsed = json.loads(response.text)
        _LLM_CACHE[cache_key] = (time.time(), parsed)
        return parsed

    except Exception as e:
        print(f"[LLMService] Gemini generation failed ({e}). Using rule-based fallback.")
        result = _fallback_generate(bridge, risk, shi, xai, eng, fused, weather, traffic)
        _LLM_CACHE[cache_key] = (time.time(), result)
        return result
