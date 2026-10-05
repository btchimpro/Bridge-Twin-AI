"""
LLM narrative service for the dashboard.
If no API key is provided, falls back to a rule-based generator.
"""
import os
import json
from typing import Dict, Any, List

# Try to import the Google GenAI SDK; if not installed, we rely on the fallback.
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from app.data.bridges import BridgeProfile


def _fallback_generate(bridge: BridgeProfile, risk: Dict, shi: Dict, xai: Dict, eng: Dict) -> Dict:
    """Rule-based fallback if LLM is unavailable."""
    is_at_risk = risk["is_at_risk"]
    shi_val = shi["predicted_shi"]
    
    # Extract top contributors
    top_risk_groups = [g["group"] for g in xai["xgb_explain"]["grouped"] if g["direction"] == "increases_risk"]
    top_shi_feat = [f["label"] for f in xai["lstm_explain"]["features"] if f["direction"] == "lowers_health"]
    top_eng_factor = xai["engineering_factors"][0]["label"]

    narrative = (
        f"The {bridge.name} is currently assessed as {'High Risk' if is_at_risk else 'Low Risk'} "
        f"with a structural health index of {shi_val*100:.1f} / 100. "
    )
    if is_at_risk and top_risk_groups:
        narrative += f"Risk is primarily elevated by factors related to: {', '.join(top_risk_groups)}. "
    if top_shi_feat:
        narrative += f"Recent sensor telemetry shows {top_shi_feat[0]} acting as a leading stressor. "
    narrative += f"From an engineering perspective, {top_eng_factor.lower()} is the dominant degradation mechanism."

    # Simple rule-based maintenance plan
    plan_priority = "High" if is_at_risk else "Medium" if shi_val < 0.75 else "Low"
    plan_actions = [
        f"Immediate visual inspection focusing on {top_eng_factor.lower()}." if is_at_risk else "Continue routine monitoring.",
        f"Review sensor thresholds for {top_shi_feat[0]}." if top_shi_feat else "Ensure sensor calibration is up to date."
    ]
    
    return {
        "analysis_narrative": narrative,
        "maintenance_plan": {
            "priority": plan_priority,
            "recommended_actions": plan_actions,
            "estimated_timeframe": "Within 30 days" if is_at_risk else "Next scheduled cycle",
            "justification": f"Driven by {top_eng_factor.lower()} concerns and recent telemetry."
        }
    }


def generate_narrative_and_plan(bridge: BridgeProfile, risk: Dict, shi: Dict, xai: Dict, eng: Dict) -> Dict:
    """
    Generate the dashboard narrative using Gemini 2.5 Flash if available,
    otherwise fallback to rule-based generation.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "gemini")

    if not api_key or not HAS_GENAI or provider.lower() != "gemini":
        return _fallback_generate(bridge, risk, shi, xai, eng)

    try:
        client = genai.Client(api_key=api_key)
        
        # Prepare context for the prompt
        context = {
            "bridge": {
                "name": bridge.name,
                "type": bridge.structure_type,
                "age": bridge.age_years,
                "material": bridge.material_type,
                "environment": bridge.exposure_class
            },
            "ai_results": {
                "risk_probability": risk["risk_probability"],
                "is_at_risk": risk["is_at_risk"],
                "structural_health_index": shi["predicted_shi"],
            },
            "explainability": {
                "top_risk_drivers": xai["xgb_explain"]["grouped"][:3],
                "top_sensor_stressors": xai["lstm_explain"]["features"][:3],
                "dominant_engineering_factors": xai["engineering_factors"][:3],
                "annual_degradation_rate": eng["annual_degradation_rate"]
            }
        }
        
        prompt = (
            "You are an expert structural health AI analyzing a digital twin of a bridge. "
            "Based on the following AI models and engineering calculations, write a concise "
            "executive summary (narrative) and recommend a maintenance plan.\n\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Keep the narrative professional, specific to the factors provided, and under 4 sentences. "
            "For the maintenance plan, provide actionable steps justified by the data."
        )

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "analysis_narrative": {"type": "STRING"},
                "maintenance_plan": {
                    "type": "OBJECT",
                    "properties": {
                        "priority": {"type": "STRING", "enum": ["Low", "Medium", "High", "Critical"]},
                        "recommended_actions": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        },
                        "estimated_timeframe": {"type": "STRING"},
                        "justification": {"type": "STRING"}
                    },
                    "required": ["priority", "recommended_actions", "estimated_timeframe", "justification"]
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
                temperature=0.2, # Low temperature for more factual responses
            ),
        )
        
        return json.loads(response.text)

    except Exception as e:
        print(f"LLM Generation failed: {e}. Using fallback.")
        return _fallback_generate(bridge, risk, shi, xai, eng)
