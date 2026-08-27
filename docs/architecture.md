# BridgeTwinAI - System Architecture

## 1. High-Level Architecture Overview
BridgeTwinAI is a digital twin system for structural health monitoring of bridges. It integrates 2D geographic information (GIS) with 3D bridge models, simulated sensor data, and AI-driven predictive maintenance.

**Core Components:**
- **Frontend (Web Application):** React.js + Three.js + Mapbox GL JS
- **Backend (API Service):** FastAPI (Python)
- **Database:** PostgreSQL with PostGIS extension (TimeScaleDB recommended for time-series sensor data)
- **AI/ML Service:** Python-based ML pipelines (PyTorch, Scikit-learn, XGBoost)
- **Data Simulation Engine:** Python-based simulation service to mock sensor readings

## 2. Repository Structure
The repository is designed as a monorepo to contain the frontend, backend, AI pipelines, and documentation.

```text
Bridge-Twin-AI/
├── .agents/                 # AI Assistant rules
├── dataset/                 # Static datasets (e.g., bridge_data.csv)
├── docs/                    # Architecture and design documentation
├── frontend/                # React.js web application
│   ├── public/              # 3D models (glTF/OBJ)
│   ├── src/
│   │   ├── components/      # UI components (Charts, Panels)
│   │   ├── three/           # Three.js 3D rendering logic
│   │   ├── map/             # Mapbox GIS integration
│   │   └── api/             # API client
├── backend/                 # FastAPI service
│   ├── app/
│   │   ├── api/             # API routes
│   │   ├── models/          # SQLAlchemy ORM models
│   │   ├── schemas/         # Pydantic validation schemas
│   │   └── core/            # Config, security, DB connections
├── ai_pipeline/             # ML models, training scripts, inference
│   ├── notebooks/           # Jupyter notebooks for EDA
│   ├── models/              # Saved model weights
│   └── src/                 # Feature extraction, PINNs, SHAP
├── simulator/               # Simulated sensor-data generator
├── docker-compose.yml       # Local execution setup
└── README.md
```

## 3. Frontend Architecture
The frontend provides a cohesive interface blending real-world geography with 3D visualization.
- **Framework:** React.js for component management and state.
- **GIS Layer:** Mapbox GL JS for rendering the 2D map, satellite imagery, and bridge markers.
- **3D Digital Twin Layer:** React Three Fiber (R3F) or Three.js overlaid on the GIS map. When a user clicks a bridge (e.g., Atal Setu), the camera zooms in, and the detailed 3D BIM/glTF model is rendered in context.
- **State Management:** Zustand or Context API for managing the active bridge and sensor data streams.

## 4. 2D GIS to 3D Bridge Digital Twin Connection
The integration between the 2D map and 3D models is critical for realism.
- **Coordinate Mapping:** Mapbox provides the geospatial context (Latitude/Longitude). The 3D models are georeferenced.
- **Custom WebGL Layer:** The Three.js scene is injected as a custom layer in Mapbox. The origin of the Three.js scene is anchored to the exact GPS coordinates of the bridge's center.
- **Scaling & Rotation:** The 3D model is scaled according to real-world meters and rotated to match the bridge's true bearing (azimuth).
- **Interactive Points:** Invisible hitboxes are placed on the 3D model corresponding to simulated sensor locations. Clicking them brings up real-time (simulated) data.

## 5. Simulated Sensor-Data Architecture
Since real-time data is unavailable, a robust simulation engine will mimic realistic structural behaviors.
- **Sensors:** Accelerometers, strain gauges, temperature sensors, and displacement sensors.
- **Simulation Engine:** A standalone Python process (or Celery task) generating time-series data.
- **Signal Generation:** 
  - Base signals: Sine waves (representing natural frequencies).
  - Noise: Gaussian white noise.
  - Environmental factors: Diurnal temperature variations affecting strain.
  - Anomalies: Simulated micro-cracks or load spikes injected randomly or predictably to trigger the AI alerts.
- **Data Delivery:** Data is pushed to the backend, which either saves it to the DB or streams it directly to the frontend via WebSockets.
