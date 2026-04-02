"""FastAPI route definitions.

Registers all API endpoints on a single APIRouter that is mounted by api/main.py.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class QueryRequest(BaseModel):
    """Incoming query payload."""

    query: str = Field(..., min_length=1, description="User query text.")


class QueryResponse(BaseModel):
    """Full answer payload returned to API callers."""

    query_id: str
    answer: str = Field(..., description="Generated answer text.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    has_unresolved_conflicts: bool
    reasoning_steps: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("ok")
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service liveness status.

    Returns:
        HealthResponse with status and version fields.
    """
    pass


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest) -> QueryResponse:
    """Accept a user query and return a deliberative RAG answer.

    Args:
        request: Validated QueryRequest payload.

    Returns:
        QueryResponse containing the answer, confidence, and reasoning steps.
    """
    pass
