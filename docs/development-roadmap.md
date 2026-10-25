# BridgeTwinAI - Development Roadmap

## Phase 1: Foundation & Infrastructure (Completed)
- [x] Initialize repository structure (frontend, backend, ai_pipeline).
- [x] Configure Pydantic schemas and database architecture.
- [x] Develop physically plausible cross-correlated 13-channel sensor simulation engine.
- [x] Establish registry for the 3 target Mumbai bridges:
  - Vashi Road Bridge (1997, Concrete Girder)
  - Atal Setu / MTHL (2024, Steel Orthotropic)
  - Bandra-Worli Sea Link (2009, Cable-Stayed Concrete)

## Phase 2: Core Backend, APIs & External Ingestion (Completed)
- [x] Setup FastAPI server (`backend/main.py`).
- [x] Implement REST endpoints for retrieving bridge metadata (`/api/bridges`).
- [x] Integrate Open-Meteo live Weather API with 15-min caching and climatology fallback.
- [x] Integrate Mumbai diurnal traffic model and optional TomTom Traffic Flow API.
- [x] End-to-end assessment endpoint (`/api/bridges/{id}/assessment`).

## Phase 3: AI Pipeline, Dual Models & Fusion (Completed)
- [x] Train and optimize XGBoost condition classifier with Stratified 5-Fold CV.
- [x] Train and calibrate Support Vector Machine (SVM) with RBF kernel and balanced weights.
- [x] Deploy PyTorch LSTM sequence model (60-minute sliding window) with MC-dropout uncertainty.
- [x] Implement Explainable AI (TreeSHAP for XGBoost, Occlusion Attribution for LSTM).
- [x] Implement Physics-Informed structural engineering mechanics (Tuutti corrosion, Basquin-Miner fatigue).
- [x] Implement Unified Risk Fusion Engine ($0.35 \times \text{ML} + 0.35 \times \text{Health} + 0.30 \times \text{Physics}$).
- [x] Integrate Gemini 2.5 Flash for grounded executive summaries and maintenance plans with rule fallback.

## Phase 4: 2D/3D Frontend Digital Twin (Completed)
- [x] Build React + TypeScript + Vite web application with Tailwind CSS.
- [x] Leaflet 2D GIS map with bridge status markers colored by multi-model consensus.
- [x] Three.js / React Three Fiber interactive 3D digital twin viewport.
- [x] Real-time HUD showing live ambient weather, dynamic traffic, and sensor telemetry.
- [x] Multi-Model consensus and 3-way risk fusion breakdown cards.

## Phase 5: Refinement, Database Persistence & Deployment (Next Sprints)
- [ ] Connect PostgreSQL / PostGIS container to persist 15-minute historical assessment snapshots.
- [ ] Add interactive sensor hitboxes on the 3D twin with click-to-explain popups.
- [ ] Provision production TomTom Traffic API key.
- [ ] Full end-to-end integration test suite and docker-compose deployment.
