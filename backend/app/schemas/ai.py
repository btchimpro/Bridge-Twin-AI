from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class RiskPredictionRequest(BaseModel):
    Age_of_Bridge: int = Field(..., description="Age of the bridge in years")
    Traffic_Volume: float = Field(..., description="Traffic volume (e.g., vehicles per hour)")
    Material_Type: str = Field(..., description="Material: 'Steel', 'Concrete', etc.")
    Maintenance_Level: str = Field(..., description="'No-Maintainance', 'Annual', 'Bi-Annual'")

class RiskPredictionResponse(BaseModel):
    is_at_risk: bool
    risk_probability: float
    threshold_used: float

class SHIPredictionRequest(BaseModel):
    time_series_data: List[Dict[str, float]] = Field(..., description="List of the last 60 minutes of sensor/environmental data.")

class SHIPredictionResponse(BaseModel):
    predicted_shi: float
