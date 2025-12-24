"""
GeometricLCM API Server - OpenAI-Compatible Endpoints

This server exposes GeometricLCM as an OpenAI-compatible API,
allowing integration with tools like Open WebUI.

Endpoints:
- POST /v1/chat/completions - Chat completions (main endpoint)
- GET /v1/models - List available models
- GET /health - Health check

Author: Lesley Gushurst
License: GPLv3
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    Choice,
    Usage,
    StreamChoice,
    StreamResponse,
    ModelInfo,
    ModelsResponse,
)

# Initialize FastAPI app
app = FastAPI(
    title="GeometricLCM API",
    description="OpenAI-compatible API for GeometricLCM - Geometric Language Concept Model",
    version="1.0.0",
)

# Add CORS middleware for Open WebUI compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for GeometricLCM components (initialized on startup)
_lcm_state = {
    "orchestrator": None,
    "initialized": False,
}


def get_orchestrator():
    """Get the initialized orchestrator."""
    if not _lcm_state["initialized"]:
        raise HTTPException(status_code=503, detail="GeometricLCM not initialized")
    return _lcm_state["orchestrator"]


@app.on_event("startup")
async def startup_event():
    """Initialize GeometricLCM on server startup."""
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from truthspace_lcm import ConceptQA
    from truthspace_lcm.core.reasoning_engine import ReasoningEngine
    from truthspace_lcm.core.holographic_generator import HolographicGenerator
    from truthspace_lcm.core.code_generator import CodeGenerator
    from truthspace_lcm.core.planner import Planner
    from truthspace_lcm.training_data import train_model
    
    # Import orchestrator and handlers
    from core.orchestrator import Orchestrator
    from core.handlers import KnowledgeHandler, CodeHandler, ToolHandler, ChatHandler
    
    print("Initializing GeometricLCM...")
    
    # Initialize ConceptQA
    qa = ConceptQA()
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "truthspace_lcm",
        "concept_corpus.json"
    )
    qa.load_corpus(corpus_path)
    
    # Train with quality examples
    train_model(qa, verbose=True)
    
    # Initialize components
    reasoning = ReasoningEngine(qa.knowledge)
    hologen = HolographicGenerator(qa.knowledge)
    codegen = CodeGenerator()
    planner = Planner(codegen)
    
    # Create orchestrator and register handlers
    orchestrator = Orchestrator()
    orchestrator.register_handler(KnowledgeHandler(qa=qa, reasoning=reasoning, hologen=hologen))
    orchestrator.register_handler(CodeHandler(codegen=codegen))
    orchestrator.register_handler(ToolHandler(planner=planner, qa=qa))
    orchestrator.register_handler(ChatHandler())
    
    _lcm_state["orchestrator"] = orchestrator
    _lcm_state["initialized"] = True
    
    print("GeometricLCM initialized with modular handlers!")
    print(f"  Handlers: {[h.name for h in orchestrator.handlers]}")


@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "GeometricLCM API",
        "version": "1.0.0",
        "description": "OpenAI-compatible API for GeometricLCM",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
        },
    }


@app.get("/v1")
async def v1_root():
    """V1 API root."""
    return {
        "object": "api",
        "version": "v1",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/v1/openapi.json"],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if _lcm_state["initialized"] else "initializing",
        "model": "geometric-lcm",
        "version": "1.0.0",
    }


@app.get("/v1/openapi.json")
async def openapi_spec():
    """OpenAPI spec for Open WebUI compatibility."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "GeometricLCM API",
            "description": "OpenAI-compatible API for GeometricLCM",
            "version": "1.0.0",
        },
        "servers": [{"url": "/v1"}],
        "paths": {
            "/chat/completions": {
                "post": {
                    "summary": "Create chat completion",
                    "operationId": "createChatCompletion",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ChatRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ChatResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/models": {
                "get": {
                    "summary": "List models",
                    "operationId": "listModels",
                    "responses": {
                        "200": {
                            "description": "List of models",
                        }
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "ChatRequest": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "messages": {"type": "array"},
                        "temperature": {"type": "number"},
                        "max_tokens": {"type": "integer"},
                        "stream": {"type": "boolean"},
                    },
                },
                "ChatResponse": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "object": {"type": "string"},
                        "created": {"type": "integer"},
                        "model": {"type": "string"},
                        "choices": {"type": "array"},
                        "usage": {"type": "object"},
                    },
                },
            }
        },
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    from .models import ModelPermission
    
    created_time = int(time.time())
    perm = ModelPermission(created=created_time)
    
    return ModelsResponse(
        data=[
            ModelInfo(
                id="geometric-lcm",
                created=created_time,
                owned_by="geometric-lcm",
                permission=[perm],
                root="geometric-lcm",
            ),
        ]
    )


def process_message(messages: list[ChatMessage], orchestrator) -> str:
    """
    Process messages through the orchestrator.
    
    The orchestrator handles intent classification and routing to handlers.
    """
    # Get the last user message
    user_message = None
    system_prompt = None
    
    for msg in messages:
        if msg.role == "user":
            user_message = msg.content
        elif msg.role == "system":
            system_prompt = msg.content
    
    if not user_message:
        return "I didn't receive a message. How can I help you?"
    
    # Process through orchestrator
    return orchestrator.process(user_message, system_prompt)


async def generate_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Generate streaming response chunks."""
    orchestrator = get_orchestrator()
    
    # Generate the full response
    response_text = process_message(request.messages, orchestrator)
    
    # Create response ID
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    
    # Stream the response word by word (or chunk by chunk)
    words = response_text.split(" ")
    
    for i, word in enumerate(words):
        chunk = StreamResponse(
            id=response_id,
            created=created,
            choices=[
                StreamChoice(
                    index=0,
                    delta={"content": word + " " if i < len(words) - 1 else word},
                    finish_reason=None if i < len(words) - 1 else "stop",
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
    
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    Chat completions endpoint (OpenAI-compatible).
    
    This is the main endpoint for chat interactions.
    """
    orchestrator = get_orchestrator()
    
    if request.stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream",
        )
    
    # Generate response
    response_text = process_message(request.messages, orchestrator)
    
    # Calculate token counts (approximate)
    prompt_tokens = sum(len(m.content.split()) for m in request.messages)
    completion_tokens = len(response_text.split())
    
    return ChatResponse(
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# Legacy completions endpoint (for compatibility)
@app.post("/v1/completions")
async def completions(request: dict):
    """Legacy completions endpoint."""
    prompt = request.get("prompt", "")
    
    # Convert to chat format
    chat_request = ChatRequest(
        model=request.get("model", "geometric-lcm"),
        messages=[ChatMessage(role="user", content=prompt)],
        stream=request.get("stream", False),
    )
    
    return await chat_completions(chat_request)


# Routes without /v1 prefix (for compatibility with some clients)
@app.post("/chat/completions")
async def chat_completions_no_prefix(request: ChatRequest):
    """Chat completions without /v1 prefix."""
    return await chat_completions(request)


@app.get("/models")
async def list_models_no_prefix():
    """List models without /v1 prefix."""
    return await list_models()


@app.post("/completions")
async def completions_no_prefix(request: dict):
    """Legacy completions without /v1 prefix."""
    return await completions(request)
