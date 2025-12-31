"""
OpenAI-Compatible API Server for Emergent Conversational Chain

Provides a REST API compatible with OpenAI's chat completions endpoint,
allowing the emergent chat to be used with tools like Open WebUI.

Key feature: All responses are EMERGENT - no LLM during conversation.
The LLM is only used as a knowledge resource during corpus building.

Run with:
    cd /home/thorin/truthspace-lcm
    uvicorn truthspace_lcm.gears.practical_applications.chat.api_server:app --reload --port 8000

Load a book (starts with empty corpus):
    curl -X POST http://localhost:8000/load_book -H "Content-Type: application/json" -d '{"book_name": "moby_dick"}'

Chat (knowledge query):
    curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "emergent-chat",
        "messages": [{"role": "user", "content": "Who is Captain Ahab?"}]
    }'

Tool call (creates files/directories):
    curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "emergent-chat",
        "messages": [{"role": "user", "content": "Create a directory called test with a file in it"}]
    }'
    # Then send "yes" to confirm execution

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
from truthspace_lcm.gears.core.intent_detector import IntentDetectorGear, Intent
from truthspace_lcm.gears.core.gear_orchestrator import GearOrchestrator


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
    
    Now with gear-based routing for tool calls.
    """
    
    def __init__(self, 
                 llm_url: str = DEFAULT_LLM_URL,
                 llm_model: str = DEFAULT_LLM_MODEL,
                 seed_topics: List[str] = None,
                 corpus_path: str = None,
                 lazy_init: bool = False,
                 enable_tools: bool = True):
        
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.seed_topics = seed_topics
        self.corpus_path = corpus_path
        self.enable_tools = enable_tools
        
        # Create conversational chain
        self.chain = ConversationalChain()
        self.chain.configure_llm(llm_url, llm_model)
        
        # Intent detector for routing
        self.intent_detector = IntentDetectorGear()
        
        # Gear orchestrator for tool calls
        self.orchestrator: Optional[GearOrchestrator] = None
        if enable_tools:
            self.orchestrator = GearOrchestrator()
            self.orchestrator.configure_llm(llm_url, llm_model)
        
        # Pending commands awaiting confirmation
        self.pending_commands: List[str] = []
        
        # Build or load corpus (unless lazy)
        if not lazy_init:
            self._init_corpus()
    
    def _init_corpus(self):
        """Initialize corpus from path or topics."""
        if self.corpus_path and Path(self.corpus_path).exists():
            logger.info(f"Loading corpus from {self.corpus_path}")
            self.chain.load_corpus(self.corpus_path)
        elif self.seed_topics:
            logger.info(f"Building corpus from topics: {self.seed_topics}")
            self.chain.build_corpus(self.seed_topics, expand=True)
        
        stats = self.chain.get_stats()
        logger.info(f"Engine ready: {stats['topics']} topics, {stats['corpus_items']} items")
    
    def generate(self, messages: List[Message]) -> str:
        """
        Generate a response using smart routing.
        - Knowledge queries -> emergent chain
        - Tool calls -> gear orchestrator
        """
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            return "I need a question to answer."
        
        # Handle confirmation for pending commands
        if self.pending_commands and user_message.lower() in ('yes', 'y'):
            return self._execute_pending()
        elif self.pending_commands:
            self.pending_commands = []
            return "Cancelled."
        
        # Handle special commands
        if user_message.lower().startswith("learn about"):
            topic = user_message[11:].strip()
            if self.chain.learn_topic(topic):
                return f"I've learned about {topic}. You can now ask me questions about it."
            return f"I couldn't learn about {topic}. Please try again."
        
        if user_message.lower() == "what topics do you know?":
            topics = self.chain.list_topics()[:20]
            return f"I can discuss: {', '.join(topics)}"
        
        # Detect intent and route
        intent_result = self.intent_detector.detect(user_message)
        
        if intent_result.intent == Intent.CHAT:
            # Knowledge query - use emergent chain
            return self.chain.chat(user_message)
        
        elif intent_result.intent in (Intent.TOOL_CALL, Intent.ORCHESTRATOR):
            # Tool call - use orchestrator
            if not self.orchestrator:
                return "Tool calls are disabled."
            
            result = self.orchestrator.execute(user_message, dry_run=True)
            
            if result['commands']:
                self.pending_commands = result['commands']
                cmd_list = '\n'.join([f"  $ {cmd}" for cmd in result['commands']])
                return (
                    f"I'll need to run these commands:\n{cmd_list}\n\n"
                    f"Reply 'yes' to execute, or anything else to cancel."
                )
            else:
                return "I couldn't figure out what commands to run for that request."
        
        # Fallback to chat
        return self.chain.chat(user_message)
    
    def _execute_pending(self) -> str:
        """Execute pending commands."""
        import subprocess
        results = []
        
        for cmd in self.pending_commands:
            try:
                output = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=30
                )
                if output.returncode == 0:
                    results.append(f"✓ {cmd}")
                    if output.stdout.strip():
                        results.append(f"  {output.stdout.strip()[:100]}")
                else:
                    results.append(f"✗ {cmd}")
                    if output.stderr.strip():
                        results.append(f"  Error: {output.stderr.strip()[:100]}")
            except Exception as e:
                results.append(f"✗ {cmd}")
                results.append(f"  Error: {str(e)}")
        
        self.pending_commands = []
        return "Executed:\n" + '\n'.join(results)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = self.chain.get_stats()
        stats['tools_enabled'] = self.enable_tools
        return stats


def create_app(
    llm_url: str = DEFAULT_LLM_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    seed_topics: List[str] = None,
    corpus_path: str = None,
    lazy_init: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Emergent Chat API",
        description="OpenAI-compatible API for Emergent Conversational Chat. "
                    "All responses are generated using emergent patterns - no LLM during conversation. "
                    "Now with gear-based routing for tool calls.",
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
    
    # Initialize engine (lazy = don't build corpus yet)
    engine = EmergentChatEngine(
        llm_url=llm_url,
        llm_model=llm_model,
        seed_topics=seed_topics,
        corpus_path=corpus_path,
        lazy_init=lazy_init,
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
    
    @app.post("/refinement/enable")
    async def enable_refinement(request: Request):
        """Enable or disable automatic response refinement."""
        data = await request.json()
        enabled = data.get("enabled", True)
        threshold = data.get("threshold", 7.0)
        
        engine.chain.enable_refinement(enabled, threshold)
        
        return {
            "status": "success",
            "refinement_enabled": enabled,
            "threshold": threshold,
            "refinement_gear_available": engine.chain.refinement_gear is not None,
        }
    
    @app.post("/refinement/chat")
    async def chat_with_refinement(request: Request):
        """Chat with detailed refinement information."""
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            raise HTTPException(status_code=400, detail="message required")
        
        result = engine.chain.chat_with_details(message)
        
        return {
            "response": result['response'],
            "original": result.get('original'),
            "topics": result.get('topics', []),
            "refined": result.get('refined', False),
            "score_before": result.get('score_before'),
            "score_after": result.get('score_after'),
            "feedback": result.get('feedback'),
        }
    
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


# Module-level app instance for uvicorn (lazy init - use /load_book to load content)
app = create_app(lazy_init=True)


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
