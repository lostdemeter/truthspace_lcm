"""
Pydantic models for OpenAI-compatible API.

These models match the OpenAI API specification for chat completions.
"""

import time
import uuid
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for /v1/chat/completions."""
    model: str = Field(default="geometric-lcm", description="Model to use")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream the response")
    top_p: Optional[float] = Field(default=1.0, description="Nucleus sampling parameter")
    n: int = Field(default=1, description="Number of completions to generate")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    presence_penalty: float = Field(default=0, description="Presence penalty")
    frequency_penalty: float = Field(default=0, description="Frequency penalty")


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """A single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"


class ChatResponse(BaseModel):
    """Response body for /v1/chat/completions."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "geometric-lcm"
    choices: List[Choice]
    usage: Usage


class StreamChoice(BaseModel):
    """A single streaming choice (delta format)."""
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    """Streaming response chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str = "geometric-lcm"
    choices: List[StreamChoice]


class ModelPermission(BaseModel):
    """Model permission info."""
    id: str = "modelperm-geometric"
    object: str = "model_permission"
    created: int
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = False
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelInfo(BaseModel):
    """Model information for /v1/models."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "geometric-lcm"
    permission: List[ModelPermission] = []
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response body for /v1/models."""
    object: str = "list"
    data: List[ModelInfo]
