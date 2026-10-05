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

router = APIRouter()

@router.get("", response_model=List[BridgeInfo])
def list_bridges():
    """List all available bridges in the MVP registry."""
    res = []
    for b in BRIDGES:
        # Quick dummy assessment to get status/risk for the map
        risk_res = ai_service.predict_risk(b.model_inputs()) if ai_service.xgb_model else {"is_at_risk": False}
        res.append(BridgeInfo(
            id=b.id,
            name=b.name,
            lat=b.lat,
            lon=b.lon,
            status="Warning" if risk_res.get("is_at_risk") else "Healthy",
            risk="High" if risk_res.get("is_at_risk") else "Low"
        ))
    return res

@router.get("/{bridge_id}/assessment", response_model=AssessmentResponse)
def get_bridge_assessment(bridge_id: str):
    """
    Get the full AI assessment for a bridge.
    Executes the entire end-to-end pipeline:
    Simulation -> Feature Engineering -> LSTM & XGBoost -> XAI -> LLM Analysis.
    """
    bridge = get_bridge(bridge_id)
    if not bridge:
        raise HTTPException(status_code=404, detail="Bridge not found")

    if not ai_service.xgb_model or not ai_service.lstm_model:
        raise HTTPException(status_code=503, detail="AI models not fully loaded")

    # 1. Simulate recent telemetry (deterministic per 15-min bucket)
    stats = ai_service.lstm_feature_stats
    window = simulate_window(bridge, stats["mean"], stats["std"])
    telemetry_records = window.records(ai_service.lstm_features)

    # 2. AI Inference
    shi_result = ai_service.predict_shi_with_uncertainty(telemetry_records)
    risk_result = ai_service.predict_risk(bridge.model_inputs())

    # 3. Physics-informed indicators
    eng = compute_indicators(
        bridge,
        window.mean(),
        strain_mean=window.mean().get("Strain_microstrain", 0),
        strain_reference=stats["std"].get("Strain_microstrain", 1) * 3
    )

    # 4. Explainability (XAI)
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

    # 5. LLM Synthesis
    llm_analysis = generate_narrative_and_plan(
        bridge, risk_result, shi_result, xai_payload, eng.as_dict()
    )

    # Package responses
    return AssessmentResponse(
        bridge={
            "id": bridge.id,
            "name": bridge.name,
            "structure_type": bridge.structure_type,
            "telemetry": {
                "latest": window.latest(),
                "generated_at": window.generated_at.isoformat(),
                "provenance": "simulated"
            },
            "provenance": bridge.provenance
        },
        structural_health=shi_result,
        risk_assessment=risk_result,
        engineering_indicators=eng.as_dict(),
        explainability=xai_payload,
        llm_analysis=llm_analysis
    )
