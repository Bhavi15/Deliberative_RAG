"""FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI

from api.routes import router


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with all routers mounted.
    """
    pass


app = create_app()
