"""
Root ASGI application entrypoint for cloud hosting (Render, Railway, Vercel, Fly.io).
Allows running:
  uvicorn main:app --host 0.0.0.0 --port $PORT
or
  uvicorn backend.main:app --host 0.0.0.0 --port $PORT
from the repository root directory.
"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(BASE_DIR, "backend")

# Ensure backend directory and its packages are on sys.path
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import the FastAPI instance from backend.main
from backend.main import app  # noqa: F401
