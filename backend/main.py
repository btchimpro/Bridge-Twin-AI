import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import ai, bridges
from app.core.database import engine
from app.models.base import Base
import app.models.bridge  # noqa: F401
from app.services.ai_service import ai_service

logger = logging.getLogger("uvicorn.info")

app = FastAPI(
    title="BridgeTwinAI API",
    description="Backend API for BridgeTwinAI Digital Twin Platform",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_db_client():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created successfully.")
    except Exception as e:
        logger.warning(f"Database connection deferred or not ready yet: {e}")

# Include Routers
app.include_router(ai.router, prefix="/api/ai", tags=["AI Models"])
app.include_router(bridges.router, prefix="/api/bridges", tags=["Bridges"])

@app.get("/")
def read_root():
    return {
        "message": "Welcome to BridgeTwinAI Backend API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": {
            "xgboost": ai_service.xgb_model is not None,
            "svm": ai_service.svm_model is not None,
            "lstm": ai_service.lstm_model is not None,
        }
    }
