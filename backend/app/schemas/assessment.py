from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class BridgeInfo(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    status: str
    risk: str
    priority: Optional[str] = None
    consensus: Optional[str] = None
    ensemble_risk_probability: Optional[float] = None

class AssessmentResponse(BaseModel):
    bridge: Dict[str, Any]
    structural_health: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    fused_risk: Dict[str, Any]
    engineering_indicators: Dict[str, Any]
    environment: Dict[str, Any]
    explainability: Dict[str, Any]
    llm_analysis: Dict[str, Any]
