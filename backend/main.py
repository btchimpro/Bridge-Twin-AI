from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import ai, bridges

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

# Include Routers
app.include_router(ai.router, prefix="/api/ai", tags=["AI Models"])
app.include_router(bridges.router, prefix="/api/bridges", tags=["Bridges"])

@app.get("/")
def read_root():
    return {"message": "Welcome to BridgeTwinAI Backend API"}
