#!/usr/bin/env python3
"""
Navigation Inference Server
============================

FastAPI server that replaces traditional LLM inference with geometric navigation
through the φ-lattice.

This is NOT steering - this IS inference, computed purely through:
1. Sign operations (XOR/multiplication) - INTEGER
2. Level operations (addition) - INTEGER
3. LUT lookups (φ^level) - TABLE
4. Accumulation - INTEGER

Usage:
    cd /home/thorin/truthspace-lcm
    source venv/bin/activate
    python src/phi_navigator/navigation_server.py

    # Or with uvicorn for development:
    uvicorn src.phi_navigator.navigation_server:app --host 0.0.0.0 --port 8009 --reload

API Endpoints:
    POST /v1/chat/completions - OpenAI-compatible chat endpoint
    POST /v1/completions - OpenAI-compatible completion endpoint
    GET /health - Health check
    GET /stats - Model statistics
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi_navigator.navigation_torch import TorchRoPEEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models (OpenAI-compatible)
# ============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "navigation-qwen2-7b"
    messages: List[Message]
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str = "navigation-qwen2-7b"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]

class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


# ============================================================================
# Navigation Server
# ============================================================================

class NavigationServer:
    """Server that uses geometric navigation for inference."""
    
    def __init__(self, cache_dir: str = None, max_layers: int = None, device: str = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/phi_navigation_rope")
        self.max_layers = max_layers
        self.device = device or 'cpu'  # Default to CPU for stability
        self.engine: Optional[TorchRoPEEngine] = None
        self.ready = False
        self.load_time = 0
        self.total_tokens_generated = 0
        self.total_requests = 0
    
    def initialize(self):
        """Initialize the navigation engine."""
        logger.info("Initializing Navigation Server...")
        start = time.time()
        
        self.engine = TorchRoPEEngine(cache_dir=self.cache_dir, device=self.device)
        
        # Check if cache exists
        config_path = os.path.join(self.cache_dir, 'config.npz')
        
        if os.path.exists(config_path):
            logger.info(f"Loading from cache: {self.cache_dir}")
            self.engine.load_from_cache(max_layers=self.max_layers)
        else:
            logger.info("Cache not found. Please run conversion first.")
            raise RuntimeError("No cached model found. Run navigation_inference.py to convert model first.")
        
        self.load_time = time.time() - start
        self.ready = True
        
        logger.info(f"Navigation Server ready!")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Layers: {len(self.engine.layers)}")
        logger.info(f"  Load time: {self.load_time:.1f}s")
    
    def _apply_chat_template(self, messages: List[Message]) -> str:
        """Apply Qwen2 chat template."""
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif msg.role == "user":
                formatted += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif msg.role == "assistant":
                formatted += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
        
        # Add assistant prefix for generation
        formatted += "<|im_start|>assistant\n"
        return formatted
    
    def _sample_token(self, logits: np.ndarray, temperature: float = 0.7, top_p: float = 0.9) -> int:
        """Sample next token from logits."""
        if temperature == 0:
            return int(np.argmax(logits))
        
        # Apply temperature
        logits = logits / temperature
        
        # Softmax
        logits_max = logits.max()
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum()
        
        # Top-p sampling
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        
        # Find cutoff
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = probs[top_indices]
        top_probs = top_probs / top_probs.sum()
        
        # Sample
        chosen_idx = np.random.choice(len(top_indices), p=top_probs)
        return int(top_indices[chosen_idx])
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[str, int, int]:
        """
        Generate text using navigation inference.
        
        Returns: (generated_text, prompt_tokens, completion_tokens)
        """
        if not self.ready:
            raise RuntimeError("Server not initialized")
        
        # Use the engine's generate method directly (handles chat_format=False since we already formatted)
        input_ids = self.engine.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tokens = len(input_ids)
        
        generated_ids = []
        
        for _ in range(max_tokens):
            # Forward pass
            all_ids = input_ids + generated_ids
            logits = self.engine.navigate_forward(all_ids)
            
            # Sample next token (logits is PyTorch tensor)
            next_logits = logits[0, -1, :]
            next_token = self.engine.sample_token(next_logits, temperature, top_p)
            
            # Check for EOS
            if next_token == self.engine.tokenizer.eos_token_id:
                break
            
            # Check for end of turn marker
            if next_token == 151645:  # <|im_end|>
                break
            
            generated_ids.append(next_token)
        
        # Decode
        generated_text = self.engine.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        self.total_tokens_generated += len(generated_ids)
        self.total_requests += 1
        
        return generated_text, prompt_tokens, len(generated_ids)
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Generate text with streaming."""
        if not self.ready:
            raise RuntimeError("Server not initialized")
        
        input_ids = self.engine.tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = []
        
        for _ in range(max_tokens):
            all_ids = input_ids + generated_ids
            logits = self.engine.navigate_forward(all_ids)
            
            next_logits = logits[0, -1, :]
            next_token = self.engine.sample_token(next_logits, temperature, top_p)
            
            if next_token == self.engine.tokenizer.eos_token_id:
                break
            if next_token == 151645:
                break
            
            generated_ids.append(next_token)
            
            # Yield token
            token_text = self.engine.tokenizer.decode([next_token])
            yield token_text
            
            # Allow other tasks to run
            await asyncio.sleep(0)
        
        self.total_tokens_generated += len(generated_ids)
        self.total_requests += 1


# ============================================================================
# FastAPI App
# ============================================================================

# Global server instance
server: Optional[NavigationServer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize server on startup."""
    global server
    
    # Get config from environment
    cache_dir = os.environ.get("NAV_CACHE_DIR", os.path.expanduser("~/.cache/phi_navigation_rope"))
    max_layers = int(os.environ.get("NAV_MAX_LAYERS", "0")) or None
    device = os.environ.get("NAV_DEVICE", None)
    
    server = NavigationServer(cache_dir=cache_dir, max_layers=max_layers, device=device)
    server.initialize()
    
    yield
    
    logger.info("Shutting down Navigation Server...")

app = FastAPI(
    title="Navigation Inference Server",
    description="Replaces LLM inference with geometric navigation through φ-lattice",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if server and server.ready else "initializing",
        "model": "navigation-qwen2-7b",
        "layers": len(server.engine.layers) if server and server.engine else 0,
    }


@app.get("/stats")
async def stats():
    """Server statistics."""
    if not server or not server.ready:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    return {
        "model": "navigation-qwen2-7b",
        "layers": len(server.engine.layers),
        "load_time_seconds": server.load_time,
        "total_requests": server.total_requests,
        "total_tokens_generated": server.total_tokens_generated,
        "cache_dir": server.cache_dir,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if not server or not server.ready:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    # Apply chat template
    prompt = server._apply_chat_template(request.messages)
    
    if request.stream:
        async def stream_response():
            request_id = f"chatcmpl-nav-{int(time.time())}"
            
            async for token in server.generate_stream(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ):
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {str(chunk)}\n\n"
            
            # Final chunk
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {str(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )
    
    # Non-streaming
    generated_text, prompt_tokens, completion_tokens = server.generate(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    
    return ChatCompletionResponse(
        id=f"chatcmpl-nav-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=generated_text),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""
    if not server or not server.ready:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    generated_text, prompt_tokens, completion_tokens = server.generate(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    return CompletionResponse(
        id=f"cmpl-nav-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionChoice(
                index=0,
                text=generated_text,
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Navigation Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8009, help="Port to bind to")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for φ-encoded weights")
    parser.add_argument("--max-layers", type=int, default=None, help="Max layers to load (for testing)")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--convert", action="store_true", help="Force re-conversion of model")
    
    args = parser.parse_args()
    
    # Set environment variables for lifespan
    if args.cache_dir:
        os.environ["NAV_CACHE_DIR"] = args.cache_dir
    if args.max_layers:
        os.environ["NAV_MAX_LAYERS"] = str(args.max_layers)
    if args.device:
        os.environ["NAV_DEVICE"] = args.device
    
    # Force re-conversion if requested
    if args.convert:
        cache_dir = args.cache_dir or os.path.expanduser("~/.cache/phi_navigation_rope")
        import shutil
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache: {cache_dir}")
    
    logger.info(f"Starting Navigation Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
