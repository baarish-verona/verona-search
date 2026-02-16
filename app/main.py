"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import router
from .api.dependencies import get_vector_store, warmup_services
from .config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Verona AI Search service...")
    logger.info("Using OpenAI embeddings + ColBERT")

    # Ensure collection exists
    try:
        vector_store = get_vector_store()
        created = vector_store.create_collection(recreate=False)
        if created:
            logger.info("Created Qdrant collection")
        else:
            logger.info("Qdrant collection already exists")
    except Exception as e:
        logger.warning(f"Failed to ensure collection exists: {e}")

    # Warmup embedding providers
    try:
        logger.info("Warming up embedding providers...")
        warmup_services()
        logger.info("Embedding providers ready")
    except Exception as e:
        logger.warning(f"Failed to warmup providers: {e}")

    logger.info("=" * 50)
    logger.info("🚀 SERVER READY - http://localhost:3000")
    logger.info("📚 API Docs: http://localhost:3000/docs")
    logger.info("=" * 50)

    yield

    # Shutdown
    logger.info("Shutting down Verona AI Search service...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Multi-vector RAG search service for matrimonial profile matching",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api")

    # Custom validation error handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    # Root-level health check for Kubernetes probes
    @app.get("/health")
    @app.get("/")
    async def health():
        return {"status": "healthy"}

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=3000,
        reload=settings.debug,
    )
