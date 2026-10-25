from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
import json

from app.data.bridges import BRIDGES, get_bridge
from app.schemas.assessment import BridgeInfo, AssessmentResponse
from app.services.ai_service import ai_service
from app.services.simulation_service import simulate_window
from app.services.physics_service import compute_indicators, ENGINEERING_WEIGHTS
from app.services.explainability_service import explain_xgb, explain_lstm, rank_engineering
from app.services.llm_service import generate_narrative_and_plan
from app.services.fusion_service import compute_fused_risk
from app.services.external.weather_client import get_current_weather
from app.services.external.traffic_client import get_current_traffic

router = APIRouter()

@router.get("", response_model=List[BridgeInfo])
def list_bridges():
    """List all available bridges in the registry with multi-model consensus status."""
    res = []
    for b in BRIDGES:
        dual_risk = ai_service.predict_dual_risk(b.model_inputs()) if (ai_service.xgb_model or ai_service.svm_model) else {"is_at_risk": False, "consensus": "unanimous_healthy", "ensemble_probability": 0.05}
        is_risk = dual_risk.get("is_at_risk", False)
        consensus = dual_risk.get("consensus", "evaluated")
        prob = float(dual_risk.get("ensemble_probability", 0.0))
        
        status = "Critical" if (is_risk and consensus == "unanimous_at_risk") else ("Warning" if is_risk else "Healthy")
        risk_label = "Critical" if status == "Critical" else ("High" if is_risk else "Low")
        
        res.append(BridgeInfo(
            id=b.id,
            name=b.name,
            lat=b.lat,
            lon=b.lon,
            status=status,
            risk=risk_label,
            age_years=b.age_years,
            length_m=b.length_m,
            priority=risk_label,
            consensus=dual_risk.get("consensus_label", "Consensus Evaluated"),
            ensemble_risk_probability=prob
        ))
    return res

@router.get("/{bridge_id}/assessment", response_model=AssessmentResponse)
def get_bridge_assessment(bridge_id: str):
    """
    Get the full end-to-end AI Digital Twin assessment for a bridge.
    Pipeline:
    1. Real-time External Ingestion (Open-Meteo Weather + Mumbai Traffic flow)
    2. Telemetry Simulation (Cross-correlated physical channels scaled by live traffic & weather)
    3. Multi-Model AI Inference (LSTM temporal health + XGBoost & SVM dual-classifier)
    4. Physics-Informed Structural Engineering Indicators (Corrosion, Fatigue, Env, Load)
    5. Unified Risk Fusion Engine (Overall Health Index, Priority, Confidence)
    6. Explainable AI (SHAP TreeSHAP + LSTM Channel Attribution + Ranked Factors)
    7. LLM Narrative & Actionable Maintenance Plan (Gemini / deterministic rule engine)
    """
    bridge = get_bridge(bridge_id)
    if not bridge:
        raise HTTPException(status_code=404, detail="Bridge not found")

    if not ai_service.xgb_model or not ai_service.lstm_model:
        raise HTTPException(status_code=503, detail="AI models not fully loaded")

    # 1. External real-time context ingestion
    weather = get_current_weather(bridge.lat, bridge.lon)
    traffic = get_current_traffic(bridge)

    # 2. Simulate recent telemetry window with live environmental & traffic factors
    stats = ai_service.lstm_feature_stats
    window = simulate_window(
        bridge,
        stats["mean"],
        stats["std"],
        weather_data=weather,
        traffic_data=traffic
    )
    telemetry_records = window.records(ai_service.lstm_features)

    # 3. AI Inference
    shi_result = ai_service.predict_shi_with_uncertainty(telemetry_records)
    dual_risk = ai_service.predict_dual_risk(bridge.model_inputs())

    # 4. Physics-informed indicators
    eng = compute_indicators(
        bridge,
        window.mean(),
        strain_mean=window.mean().get("Strain_microstrain", 0),
        strain_reference=stats["std"].get("Strain_microstrain", 1) * 3,
        current_traffic_vph=traffic.get("current_vph")
    )

    # 5. Unified Risk Fusion Engine
    fused_risk = compute_fused_risk(dual_risk, shi_result, eng.as_dict())

    # 6. Explainability (XAI)
    xgb_xai = explain_xgb(ai_service.explain_risk(bridge.model_inputs()))
    lstm_xai = explain_lstm(
        ai_service.lstm_channel_attribution(telemetry_records),
        window.mean(),
        stats["mean"],
        stats["std"]
    )
    eng_ranked = rank_engineering(eng.as_dict(), ENGINEERING_WEIGHTS)

    xai_payload = {
        "xgb_explain": xgb_xai,
        "lstm_explain": lstm_xai,
        "engineering_factors": eng_ranked
    }

    # 7. LLM Synthesis
    llm_analysis = generate_narrative_and_plan(
        bridge,
        dual_risk,
        shi_result,
        xai_payload,
        eng.as_dict(),
        fused=fused_risk,
        weather=weather,
        traffic=traffic
    )

    # Assemble comprehensive assessment response
    return AssessmentResponse(
        bridge={
            "id": bridge.id,
            "name": bridge.name,
            "structure_type": bridge.structure_type,
            "material_type": bridge.material_type,
            "age_years": bridge.age_years,
            "length_m": bridge.length_m,
            "exposure_class": bridge.exposure_class,
            "telemetry": {
                "latest": window.latest(),
                "generated_at": window.generated_at.isoformat(),
                "provenance": "simulated"
            },
            "provenance": bridge.provenance
        },
        structural_health=shi_result,
        risk_assessment=dual_risk,
        fused_risk=fused_risk,
        engineering_indicators=eng.as_dict(),
        environment={
            "weather": weather,
            "traffic": traffic
        },
        explainability=xai_payload,
        llm_analysis=llm_analysis
    )
