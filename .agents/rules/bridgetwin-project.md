---
trigger: always_on
---

# BridgeTwinAI Project Rules

## Project

We are building BridgeTwinAI, an AI-powered Digital Twin platform for urban bridge infrastructure.

The system is a student major project intended to demonstrate:

1. Urban infrastructure digital twins
2. GIS visualization
3. 3D bridge visualization
4. Physics-informed structural health estimation
5. Machine-learning-based risk prediction
6. Explainable AI
7. Predictive maintenance
8. Risk prioritization
9. Simulated real-time sensor data

The initial geographic focus is Mumbai.

The system should initially demonstrate three representative bridges:
- Vashi Road Bridge
- Atal Setu
- Bandra-Worli Sea Link

The architecture must be designed so additional bridges can be added later without rewriting the system.

## Core principle

This is NOT merely a dashboard.

The system must demonstrate an end-to-end pipeline:

Data
→ Feature Engineering
→ Physics-Informed Structural Health Estimation
→ ML Risk Prediction
→ Explainable AI
→ Maintenance Recommendation
→ Digital Twin Visualization

## Important data policy

Clearly distinguish between:
- real/public data
- estimated values
- simulated values

Never claim simulated structural measurements are real measurements.

Where structural sensor data is unavailable, use clearly labelled simulated data.

Never fabricate real inspection or engineering records and present them as genuine.

## Technology

Frontend:
- React
- TypeScript
- Vite
- Tailwind CSS
- Leaflet or MapLibre for 2D GIS
- Three.js / React Three Fiber for 3D
- Recharts for charts

Backend:
- Python
- FastAPI
- SQLAlchemy
- Pydantic

Database:
- PostgreSQL
- PostGIS

AI:
- Python
- NumPy
- Pandas
- scikit-learn
- XGBoost if useful
- SHAP

Data:
- GeoJSON
- CSV
- JSON
- simulated time-series data

Development:
- Docker where practical
- Git
- pytest
- ESLint
- TypeScript strict mode

## Architecture principles

Use modular architecture.

Do not place the entire application in one file.

Separate:
- API routes
- database models
- schemas
- services
- AI models
- feature engineering
- simulation
- frontend components
- pages
- map components
- 3D components

Avoid unnecessary technologies.

Do not introduce Kafka, Kubernetes, microservices, Redis, Celery, cloud infrastructure, or other infrastructure unless there is a demonstrated project requirement.

The MVP should run locally.

## AI principles

The AI system should never simply generate arbitrary risk numbers.

Risk should be derived from documented features.

The initial system should contain:

- structural health score
- environmental degradation score
- traffic/load factor
- age factor
- corrosion/degradation factor
- fatigue factor
- inspection condition factor
- overall risk score

The physics-informed component should generate meaningful engineering-inspired intermediate indicators.

These are estimates for the project prototype, not certified engineering calculations.

## Explainability

Every AI risk prediction should expose:
- prediction
- major contributing features
- feature values
- explanation
- confidence/uncertainty where appropriate

Use SHAP or feature importance where appropriate.

## UI principles

The interface should look like a professional infrastructure intelligence platform.

It should not look like a generic student CRUD application.

Use:
- clear hierarchy
- responsive layout
- cards
- charts
- maps
- status indicators
- professional typography
- useful animations only

The main dashboard should communicate infrastructure condition immediately.

## Development rules

Before implementing a major feature:
1. Inspect the existing repository.
2. Explain what files will change.
3. Implement the smallest coherent version.
4. Run tests/build/lint.
5. Verify the result.
6. Report what was changed.

Never rewrite working components unnecessarily.

Never delete functionality without explicit approval.

When a command fails:
- inspect the error
- diagnose the cause
- fix it
- rerun the command

Do not simply hide errors.

## Security

Never hardcode:
- API keys
- passwords
- database credentials
- secrets

Use .env files and provide .env.example.

## Documentation

Maintain:
- README.md
- docs/architecture.md
- docs/data-model.md
- docs/ai-pipeline.md
- docs/development-roadmap.md

Keep documentation synchronized with implementation.

## Scope control

Build in phases.

Do NOT attempt to implement the entire platform at once.

Prioritize a working MVP over excessive infrastructure.

The application must remain runnable after every major phase.