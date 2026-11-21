from pydantic import BaseModel, Field
from typing import Any


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Question to ask the RAG system"
    )

    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Existing conversation session ID; omit to create a new session"
    )


class SourceItem(BaseModel):
    source: str
    chunk_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    question: str
    rewritten_question: str | None = None
    answer: str
    sources: list[Any] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    service: str