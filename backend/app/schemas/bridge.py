from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any
from uuid import UUID
from datetime import datetime

class SensorBase(BaseModel):
    sensor_type: str
    local_x: Optional[float] = None
    local_y: Optional[float] = None
    local_z: Optional[float] = None
    status: Optional[str] = "active"

class SensorCreate(SensorBase):
    pass

class Sensor(SensorBase):
    id: UUID
    bridge_id: UUID

    model_config = ConfigDict(from_attributes=True)

class BridgeBase(BaseModel):
    name: str
    length_m: Optional[float] = None
    built_year: Optional[int] = None
    material_type: Optional[str] = None
    traffic_volume: Optional[int] = None

class BridgeCreate(BridgeBase):
    lat: float
    lon: float

class Bridge(BridgeBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    sensors: List[Sensor] = []

    model_config = ConfigDict(from_attributes=True)
