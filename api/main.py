"""FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with all routers mounted.
    """
    application = FastAPI(
        title="Deliberative RAG API",
        description=(
            "Conflict-aware retrieval reasoning API. Supports deep research "
            "with web search, conflict detection, and multi-agent analysis."
        ),
        version="2.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)

    return application


app = create_app()
