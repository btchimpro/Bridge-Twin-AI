# BridgeTwinAI - Development Roadmap

## Phase 1: Foundation & Infrastructure (MVP Base)
- [ ] Initialize repository structure (frontend, backend, ai_pipeline).
- [ ] Setup PostgreSQL with PostGIS locally using Docker Compose.
- [ ] Implement database schemas using SQLAlchemy.
- [ ] Develop the simulated sensor data generator script for the 3 target bridges:
  - Vashi Road Bridge
  - Atal Setu
  - Bandra-Worli Sea Link

## Phase 2: Core Backend & API
- [ ] Setup FastAPI server.
- [ ] Implement REST endpoints for retrieving bridge metadata.
- [ ] Implement WebSocket or Polling endpoints for streaming simulated sensor telemetry.
- [ ] Connect the simulation engine to feed data into the database continuously.

## Phase 3: 2D/3D Frontend Integration
- [ ] Scaffold React.js application.
- [ ] Integrate Mapbox GL JS for the 2D geospatial map.
- [ ] Plot the 3 bridges on the map with accurate GPS coordinates.
- [ ] Integrate Three.js / React Three Fiber.
- [ ] Load placeholder 3D models (glTF) for the bridges overlaid on the map.
- [ ] Build the UI dashboard for showing real-time sensor graphs.

## Phase 4: AI Pipeline & Intelligence
- [ ] Develop the Physics-Informed structural health estimation script.
- [ ] Build the Explainability (SHAP) module.
- [ ] Implement the Maintenance Recommendation engine.
- [ ] Create background tasks in FastAPI to periodically run the AI assessments.
- [ ] Surface AI insights, 3D highlights (anomalies), and recommendations to the frontend.

## Phase 5: Refinement & Polish
- [ ] Enhance 3D models and lighting.
- [ ] Optimize database queries for time-series data.
- [ ] Ensure everything runs seamlessly via `docker-compose up` for local demonstration.
- [ ] Finalize documentation.
