import os
import sys
from datetime import datetime
import time

# Add the backend directory to sys.path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app.core.database import engine, Base
from app.models.bridge import Bridge, Sensor
from sqlalchemy.orm import Session
from sqlalchemy import text

# Coordinates for the 3 MVP Bridges in Mumbai
BRIDGES = [
    {
        "name": "Vashi Road Bridge",
        "lat": 19.0664, 
        "lon": 72.9984,
        "length_m": 1837.0,
        "built_year": 1997,
        "material_type": "Concrete",
        "traffic_volume": 80000,
        "sensors": [
            {"sensor_type": "accelerometer", "local_x": 0, "local_y": 0, "local_z": 5},
            {"sensor_type": "strain_gauge", "local_x": 100, "local_y": 0, "local_z": 2}
        ]
    },
    {
        "name": "Atal Setu",
        "lat": 18.9667, 
        "lon": 72.9500,
        "length_m": 21800.0,
        "built_year": 2024,
        "material_type": "Concrete/Steel",
        "traffic_volume": 70000,
        "sensors": [
            {"sensor_type": "accelerometer", "local_x": 0, "local_y": 0, "local_z": 10},
            {"sensor_type": "accelerometer", "local_x": 1000, "local_y": 0, "local_z": 10},
            {"sensor_type": "strain_gauge", "local_x": 500, "local_y": 0, "local_z": 5}
        ]
    },
    {
        "name": "Bandra-Worli Sea Link",
        "lat": 19.0360, 
        "lon": 72.8170,
        "length_m": 5600.0,
        "built_year": 2009,
        "material_type": "Concrete/Steel",
        "traffic_volume": 37500,
        "sensors": [
            {"sensor_type": "accelerometer", "local_x": 0, "local_y": 0, "local_z": 50}, # Pylon
            {"sensor_type": "strain_gauge", "local_x": 200, "local_y": 0, "local_z": 0}
        ]
    }
]

def init_db():
    print("Creating tables...")
    
    # Ensure PostGIS is enabled
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

def seed_data():
    with Session(engine) as session:
        # Check if already seeded
        if session.query(Bridge).first():
            print("Database already seeded. Skipping...")
            return

        print("Seeding MVP bridges...")
        for b_data in BRIDGES:
            bridge = Bridge(
                name=b_data["name"],
                geometry=f"POINT({b_data['lon']} {b_data['lat']})",
                length_m=b_data["length_m"],
                built_year=b_data["built_year"],
                material_type=b_data["material_type"],
                traffic_volume=b_data["traffic_volume"]
            )
            session.add(bridge)
            session.flush() # To get the bridge.id

            print(f"Added {bridge.name}")

            for s_data in b_data["sensors"]:
                sensor = Sensor(
                    bridge_id=bridge.id,
                    sensor_type=s_data["sensor_type"],
                    local_x=s_data["local_x"],
                    local_y=s_data["local_y"],
                    local_z=s_data["local_z"]
                )
                session.add(sensor)
        
        session.commit()
        print("Seeding complete.")

import random
from app.models.bridge import SensorReading

def simulate_data():
    print("Starting simulation loop... Press Ctrl+C to stop.")
    try:
        while True:
            with Session(engine) as session:
                sensors = session.query(Sensor).all()
                for sensor in sensors:
                    # Generate a random value based on sensor type
                    base_val = 0.0
                    if sensor.sensor_type == "accelerometer":
                        base_val = 9.81 + random.gauss(0, 0.5) # Simulate gravity + vibration
                    elif sensor.sensor_type == "strain_gauge":
                        base_val = 200 + random.gauss(0, 10) # microstrain
                    
                    reading = SensorReading(
                        sensor_id=sensor.id,
                        timestamp=datetime.utcnow(),
                        value=base_val,
                        unit="m/s^2" if sensor.sensor_type == "accelerometer" else "microstrain"
                    )
                    session.add(reading)
                session.commit()
            print(f"Generated data for {len(sensors)} sensors at {datetime.utcnow()}")
            time.sleep(5) # Generate data every 5 seconds
    except KeyboardInterrupt:
        print("Simulation stopped.")

if __name__ == "__main__":
    init_db()
    seed_data()
    simulate_data()
