import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from .base import Base

class Bridge(Base):
    __tablename__ = "bridges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    geometry = Column(Geometry('POINT', srid=4326))
    length_m = Column(Float)
    built_year = Column(Integer)
    material_type = Column(String)
    traffic_volume = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sensors = relationship("Sensor", back_populates="bridge")
    assessments = relationship("StructuralAssessment", back_populates="bridge")
    recommendations = relationship("MaintenanceRecommendation", back_populates="bridge")

class Sensor(Base):
    __tablename__ = "sensors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bridge_id = Column(UUID(as_uuid=True), ForeignKey("bridges.id"))
    sensor_type = Column(String, nullable=False)  # accelerometer, strain_gauge, displacement
    local_x = Column(Float)
    local_y = Column(Float)
    local_z = Column(Float)
    status = Column(String, default="active")

    bridge = relationship("Bridge", back_populates="sensors")
    readings = relationship("SensorReading", back_populates="sensor")

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sensor_id = Column(UUID(as_uuid=True), ForeignKey("sensors.id"), index=True)
    timestamp = Column(DateTime(timezone=True), index=True)
    value = Column(Float, nullable=False)
    unit = Column(String)

    sensor = relationship("Sensor", back_populates="readings")

class StructuralAssessment(Base):
    __tablename__ = "structural_assessments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bridge_id = Column(UUID(as_uuid=True), ForeignKey("bridges.id"))
    timestamp = Column(DateTime(timezone=True))
    health_score = Column(Float)  # 0.0 to 1.0
    anomaly_detected = Column(Boolean, default=False)
    details = Column(JSONB)

    bridge = relationship("Bridge", back_populates="assessments")
    recommendations = relationship("MaintenanceRecommendation", back_populates="assessment")

class MaintenanceRecommendation(Base):
    __tablename__ = "maintenance_recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("structural_assessments.id"))
    bridge_id = Column(UUID(as_uuid=True), ForeignKey("bridges.id"))
    recommended_action = Column(String)
    priority = Column(String)  # Low, Medium, High, Critical
    status = Column(String, default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    assessment = relationship("StructuralAssessment", back_populates="recommendations")
    bridge = relationship("Bridge", back_populates="recommendations")
