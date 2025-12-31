"""
OpenAI-Compatible API Server for Emergent Conversational Chain

Provides a REST API compatible with OpenAI's chat completions endpoint,
allowing the emergent chat to be used with tools like Open WebUI.

Key feature: All responses are EMERGENT - no LLM during conversation.
The LLM is only used as a knowledge resource during corpus building.

Author: Lesley Gushurst
License: GPLv3
"""

import time
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from truthspace_lcm.gears.core import ConversationalChain


# Default configuration
DEFAULT_LLM_URL = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL = "qwen2:latest"

DEFAULT_SEED_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "programming",
    "python",
    "science",
    "philosophy",
    "mathematics",
]


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "emergent-chat"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "truthspace"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class EmergentChatEngine:
    """
    The emergent chat engine that powers the API.
    
    All responses are generated using emergent patterns only.
    No LLM calls during conversation.
    """
    
    def __init__(self, 
                 llm_url: str = DEFAULT_LLM_URL,
                 llm_model: str = DEFAULT_LLM_MODEL,
                 seed_topics: List[str] = None,
                 corpus_path: str = None):
        
        # Create conversational chain
        self.chain = ConversationalChain()
        self.chain.configure_llm(llm_url, llm_model)
        
        # Build or load corpus
        if corpus_path and Path(corpus_path).exists():
            logger.info(f"Loading corpus from {corpus_path}")
            self.chain.load_corpus(corpus_path)
        else:
            topics = seed_topics or DEFAULT_SEED_TOPICS
            logger.info(f"Building corpus from topics: {topics}")
            self.chain.build_corpus(topics, expand=True)
        
        stats = self.chain.get_stats()
        logger.info(f"Engine ready: {stats['topics']} topics, {stats['corpus_items']} items")
    
    def generate(self, messages: List[Message]) -> str:
        """
        Generate a response using emergent patterns only.
        NO LLM calls here.
        """
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            return "I need a question to answer."
        
        # Handle special commands
        if user_message.lower().startswith("learn about"):
            topic = user_message[11:].strip()
            if self.chain.learn_topic(topic):
                return f"I've learned about {topic}. You can now ask me questions about it."
            return f"I couldn't learn about {topic}. Please try again."
        
        if user_message.lower() == "what topics do you know?":
            topics = self.chain.list_topics()[:20]
            return f"I can discuss: {', '.join(topics)}"
        
        # Generate emergent response
        response = self.chain.chat(user_message)
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self.chain.get_stats()


def create_app(
    llm_url: str = DEFAULT_LLM_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    seed_topics: List[str] = None,
    corpus_path: str = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Emergent Chat API",
        description="OpenAI-compatible API for Emergent Conversational Chat. "
                    "All responses are generated using emergent patterns - no LLM during conversation.",
        version="1.0.0",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize engine
    engine = EmergentChatEngine(
        llm_url=llm_url,
        llm_model=llm_model,
        seed_topics=seed_topics,
        corpus_path=corpus_path,
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        stats = engine.get_stats()
        return {
            "status": "healthy",
            "engine": "emergent-chat",
            "topics": stats['topics'],
            "corpus_items": stats['corpus_items'],
            "llm_calls_during_chat": stats['conversation_calls'],
        }
    
    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        """List available models."""
        return ModelsResponse(
            data=[
                ModelInfo(
                    id="emergent-chat",
                    created=int(time.time()),
                    owned_by="truthspace",
                ),
                ModelInfo(
                    id="emergent-chat-expanded",
                    created=int(time.time()),
                    owned_by="truthspace",
                ),
            ]
        )
    
    @app.get("/stats")
    async def get_stats():
        """Get engine statistics."""
        return engine.get_stats()
    
    @app.get("/topics")
    async def list_topics():
        """List known topics."""
        return {"topics": engine.chain.list_topics()}
    
    @app.post("/learn")
    async def learn_topic(request: Request):
        """Learn about a new topic."""
        data = await request.json()
        topic = data.get("topic", "")
        if not topic:
            raise HTTPException(status_code=400, detail="Topic required")
        
        success = engine.chain.learn_topic(topic)
        if success:
            return {"status": "success", "topic": topic}
        raise HTTPException(status_code=500, detail="Failed to learn topic")
    
    @app.get("/books")
    async def list_books():
        """List available books from Project Gutenberg."""
        return {"books": engine.chain.get_available_books()}
    
    @app.post("/load_book")
    async def load_book(request: Request):
        """Load a literary work and build corpus from it."""
        data = await request.json()
        book_name = data.get("book_name")
        url = data.get("url")
        max_lines = data.get("max_lines")
        
        if not book_name and not url:
            raise HTTPException(status_code=400, detail="book_name or url required")
        
        logger.info(f"Loading book: {book_name or url}")
        
        success = engine.chain.load_book(
            book_name=book_name,
            url=url,
            max_lines=max_lines,
        )
        
        if success:
            stats = engine.chain.get_stats()
            return {
                "status": "success",
                "book": getattr(engine.chain, 'book_title', book_name),
                "topics": stats['topics'],
                "corpus_items": stats['corpus_items'],
            }
        raise HTTPException(status_code=500, detail="Failed to load book")
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completions endpoint (OpenAI-compatible)."""
        
        logger.info(f"Received request: model={request.model}, stream={request.stream}")
        logger.info(f"Messages: {[m.content[:50] for m in request.messages]}")
        
        try:
            response_text = engine.generate(request.messages)
            logger.info(f"Generated response: {response_text[:100]}")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # Handle streaming
        if request.stream:
            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                # Send the content in one chunk
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": response_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send finish chunk
                finish_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=sum(len(m.content.split()) for m in request.messages),
                completion_tokens=len(response_text.split()),
                total_tokens=sum(len(m.content.split()) for m in request.messages) + len(response_text.split()),
            ),
        )
    
    return app


def get_app():
    """Factory function for creating the app."""
    return create_app()


if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Emergent Chat API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--corpus", type=str, help="Path to corpus file")
    parser.add_argument("--topics", nargs="+", help="Seed topics for knowledge building")
    args = parser.parse_args()
    
    # Create app with settings
    app = create_app(
        corpus_path=args.corpus,
        seed_topics=args.topics,
    )
    
    uvicorn.run(app, host=args.host, port=args.port)
