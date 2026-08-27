"""VisionGuard FastAPI application entry point.

Run with:
    uvicorn apps.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from apps.core.config import settings
from apps.db.init_db import init_db
from apps.api.routes import health, analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Ensure upload directory exists
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    # Create database tables
    init_db()

    yield


app = FastAPI(
    title="VisionGuard API",
    description="AI-powered image quality and defect detection service.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---- CORS ----
origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Routes ----
app.include_router(health.router)
app.include_router(analysis.router)

# ---- Static files: uploaded images ----
upload_path = Path(settings.upload_dir)
upload_path.mkdir(parents=True, exist_ok=True)
app.mount(settings.upload_url_base, StaticFiles(directory=str(upload_path)), name="uploads")
