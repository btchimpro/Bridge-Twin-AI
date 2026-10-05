from fastapi import APIRouter, HTTPException, Depends
from app.schemas.ai import RiskPredictionRequest, RiskPredictionResponse, SHIPredictionRequest, SHIPredictionResponse
from app.services.ai_service import ai_service

router = APIRouter()

@router.post("/predict/risk", response_model=RiskPredictionResponse)
def predict_bridge_risk(request: RiskPredictionRequest):
    """
    Predict if a bridge is at risk using the XGBoost Classifier.
    """
    try:
        result = ai_service.predict_risk(request.model_dump())
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/shi", response_model=SHIPredictionResponse)
def predict_structural_health_index(request: SHIPredictionRequest):
    """
    Predict continuous Structural Health Index (SHI) using the PyTorch LSTM.
    Requires exactly 60 historical data points representing the sliding window.
    """
    try:
        result = ai_service.predict_shi(request.time_series_data)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
