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
    "qa": None,
    "reasoning": None,
    "hologen": None,
    "codegen": None,
    "planner": None,
    "memory": None,
    "initialized": False,
}


def get_lcm():
    """Get the initialized LCM components."""
    if not _lcm_state["initialized"]:
        raise HTTPException(status_code=503, detail="GeometricLCM not initialized")
    return _lcm_state


@app.on_event("startup")
async def startup_event():
    """Initialize GeometricLCM on server startup."""
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from truthspace_lcm import ConceptQA
    from truthspace_lcm.core.conversation_memory import ConversationMemory
    from truthspace_lcm.core.reasoning_engine import ReasoningEngine
    from truthspace_lcm.core.holographic_generator import HolographicGenerator
    from truthspace_lcm.core.code_generator import CodeGenerator
    from truthspace_lcm.core.planner import Planner
    from truthspace_lcm.training_data import train_model
    
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
    _lcm_state["qa"] = qa
    _lcm_state["reasoning"] = ReasoningEngine(qa.knowledge)
    _lcm_state["hologen"] = HolographicGenerator(qa.knowledge)
    _lcm_state["codegen"] = CodeGenerator()
    _lcm_state["planner"] = Planner(_lcm_state["codegen"])
    _lcm_state["memory"] = ConversationMemory(max_turns=20)
    _lcm_state["initialized"] = True
    
    print("GeometricLCM initialized and ready!")


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


def process_message(messages: list[ChatMessage], lcm: dict) -> str:
    """
    Process messages through GeometricLCM and generate a response.
    
    This is the core logic that routes requests to appropriate handlers.
    """
    qa = lcm["qa"]
    codegen = lcm["codegen"]
    planner = lcm["planner"]
    reasoning = lcm["reasoning"]
    hologen = lcm["hologen"]
    memory = lcm["memory"]
    
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
    
    # Classify intent
    intent = classify_intent(user_message)
    
    # Route to appropriate handler
    if intent == "greeting":
        return handle_greeting(user_message)
    
    elif intent == "meta":
        return handle_meta(user_message)
    
    elif intent == "code":
        return handle_code(user_message, codegen, planner)
    
    elif intent == "chart":
        return handle_chart(user_message, planner, qa)
    
    elif intent == "execute":
        return handle_execute(user_message, planner)
    
    elif intent == "question":
        return handle_question(user_message, qa, reasoning, hologen)
    
    else:
        # Default: try Q&A
        return handle_question(user_message, qa, reasoning, hologen)


def classify_intent(message: str) -> str:
    """Classify the intent of a message."""
    msg_lower = message.lower()
    
    # Greeting patterns
    greetings = ["hello", "hi ", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if any(g in msg_lower for g in greetings) or msg_lower in ["hi", "hey"]:
        return "greeting"
    
    # Meta patterns (asking about capabilities)
    meta_patterns = ["what can you do", "what are you", "who are you", "help me", "your capabilities", 
                     "what do you know", "tell me about yourself"]
    if any(p in msg_lower for p in meta_patterns):
        return "meta"
    
    # Code patterns
    code_patterns = ["write a function", "create a function", "generate code", "write code", 
                     "python function", "write a script", "code that", "function to", "function that"]
    if any(p in msg_lower for p in code_patterns):
        return "code"
    
    # Chart patterns
    chart_patterns = ["chart", "graph", "plot", "visualize", "visualization", "bar chart", 
                      "line chart", "pie chart", "histogram"]
    if any(p in msg_lower for p in chart_patterns):
        return "chart"
    
    # Execute patterns
    execute_patterns = ["calculate", "compute", "find the", "sum of", "average of", 
                        "sort", "filter", "count"]
    if any(p in msg_lower for p in execute_patterns):
        return "execute"
    
    # Question patterns
    question_words = ["who", "what", "where", "when", "why", "how", "is ", "are ", "does ", "did "]
    if any(msg_lower.startswith(q) or f" {q}" in msg_lower for q in question_words) or "?" in message:
        return "question"
    
    return "general"


def handle_greeting(message: str) -> str:
    """Handle greeting messages."""
    responses = [
        "Hello! I'm GeometricLCM, a geometric language model. I can answer questions, "
        "generate code, create charts, and execute tasks. How can I help you today?",
    ]
    return responses[0]


def handle_meta(message: str) -> str:
    """Handle meta questions about capabilities."""
    return """I'm **GeometricLCM**, a geometric language concept model. Unlike neural network LLMs, I use geometric operations in concept space to understand and respond.

**What I can do:**

• **Answer Questions** - Ask me about topics in my knowledge base (currently focused on Sherlock Holmes stories)
• **Generate Code** - I can write Python functions from natural language descriptions
• **Create Charts** - I can generate matplotlib visualizations
• **Execute Tasks** - I can plan and run calculations, filtering, sorting, and more

**My Philosophy:**
> "All semantic operations are geometric operations in concept space."

I don't have internet access or real-time information, but I'm fast, interpretable, and don't need a GPU!

What would you like to explore?"""


def handle_code(message: str, codegen, planner) -> str:
    """Handle code generation requests."""
    code = codegen.generate(message)
    
    if "pass  # TODO" in code:
        # Fallback for unrecognized patterns
        return f"""I'll help you write that code. Here's my attempt:

```python
{code}
```

This is a template - I recognized the request but don't have a specific implementation for it yet. Would you like me to help you fill in the logic?"""
    
    return f"""Here's the code you requested:

```python
{code}
```

Would you like me to explain how it works or make any modifications?"""


def handle_chart(message: str, planner, qa) -> str:
    """Handle chart generation requests."""
    return """I can help you create charts! To generate a visualization, I need:

1. **Data** - What data should I visualize?
2. **Chart type** - Bar, line, pie, scatter, etc.
3. **Labels** - Title, axis labels

For example, you could ask:
- "Create a bar chart of character appearances in Sherlock Holmes"
- "Plot the distribution of actions for Holmes"

What would you like to visualize?"""


def handle_execute(message: str, planner) -> str:
    """Handle task execution requests."""
    plan = planner.plan(message)
    result = planner.execute(plan)
    
    if result.success:
        # Format the response nicely
        steps_summary = "\n".join([f"  {i+1}. {s.description}" for i, s in enumerate(plan.steps)])
        return f"""**Task:** {message}

**Plan:**
{steps_summary}

**Result:** `{result.final_result}`

The task completed successfully!"""
    else:
        # Find the failed step
        failed = [s for s in plan.steps if s.error]
        error_msg = failed[0].error if failed else "Unknown error"
        return f"""I tried to execute that task but encountered an issue:

**Error:** {error_msg}

Could you rephrase the request or provide more details?"""


def handle_question(message: str, qa, reasoning, hologen) -> str:
    """Handle knowledge questions."""
    # Try standard Q&A first
    result = qa.ask_detailed(message)
    
    if result['answers'] and result['answers'][0]['confidence'] > 0.3:
        answer = result['answers'][0]['answer']
        
        # Check if it's a WHY/HOW question for reasoning
        msg_lower = message.lower()
        if msg_lower.startswith("why") or msg_lower.startswith("how"):
            path = reasoning.reason(message)
            if path.steps and len(path.steps) > 1:
                reasoning_chain = " → ".join([str(s) for s in path.steps[:4]])
                return f"{answer}\n\n**Reasoning path:** {reasoning_chain}"
        
        return answer
    
    # Try holographic generation for entity questions
    entities = ["holmes", "watson", "moriarty", "lestrade", "irene", "mycroft"]
    for entity in entities:
        if entity in message.lower():
            learnable = qa.projector.answer_generator.learnable
            output = hologen.generate(message, entity=entity, learnable=learnable)
            if output and len(output) > 20:
                return output
    
    # Fallback
    return f"""I don't have specific information about that in my current knowledge base.

My knowledge is primarily focused on Sherlock Holmes stories. You could try:
- "Who is Holmes?"
- "What is the relationship between Holmes and Watson?"
- "Why did Moriarty challenge Holmes?"

Or I can help with code generation, calculations, or charts instead!"""


async def generate_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Generate streaming response chunks."""
    lcm = get_lcm()
    
    # Generate the full response
    response_text = process_message(request.messages, lcm)
    
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
    lcm = get_lcm()
    
    if request.stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream",
        )
    
    # Generate response
    response_text = process_message(request.messages, lcm)
    
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
