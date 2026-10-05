from typing import Dict, Any, List
from pydantic import BaseModel

class BridgeInfo(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    status: str
    risk: str

class AssessmentResponse(BaseModel):
    bridge: Dict[str, Any]
    structural_health: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    engineering_indicators: Dict[str, Any]
    explainability: Dict[str, Any]
    llm_analysis: Dict[str, Any]
