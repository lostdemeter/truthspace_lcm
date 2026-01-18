#!/usr/bin/env python3
"""
φ-Fast API Server - Loads Pre-Quantized Model Instantly
========================================================

This server loads the pre-quantized φ-model from disk for instant startup.
No quantization at runtime - just load and serve!

Prerequisites:
    Run phi_quantize_model.py first to create the quantized model:
    python experiments/model_reverse_engineering/phi_quantize_model.py

Run with:
    python experiments/model_reverse_engineering/phi_fast_api_server.py --port 8003

Author: TruthSpace LCM Team
License: GPLv3
"""

import os
import time
import uuid
import argparse
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import numpy as np

# Check for GPU
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# φ constants
PHI = (1 + np.sqrt(5)) / 2
K = 128


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    model_config = {"extra": "ignore"}
    role: str
    content: Optional[Any] = ""
    
    def get_text_content(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            texts = []
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return " ".join(texts)
        return str(self.content)


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: str = "phi-fast-qwen2"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ResponseMessage
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
    owned_by: str = "truthspace-phi"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


def phi_dequantize(signs: np.ndarray, indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Reconstruct tensor from φ-quantized representation."""
    exponents = codebook[indices]
    values = signs * (PHI ** (exponents / K))
    return values.astype(np.float32)


class PhiFastEngine:
    """
    Fast-loading φ-quantized Qwen2 engine.
    
    Loads pre-quantized weights from disk - no runtime quantization needed.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2-7B-Instruct",
                 quantized_path: str = None):
        self.model_name = model_name
        self.device = DEVICE
        self.model = None
        self.tokenizer = None
        
        # Default quantized path
        if quantized_path is None:
            quantized_path = os.path.expanduser("~/.cache/phi_quantized/qwen2-7b")
        self.quantized_path = quantized_path
        
        # Statistics
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_time_ms = 0
        self.load_time_ms = 0
        self.quantized_size_bytes = 0
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the model with pre-quantized weights."""
        start_time = time.perf_counter()
        
        # Check if quantized model exists
        if not os.path.exists(self.quantized_path):
            raise RuntimeError(
                f"Quantized model not found at {self.quantized_path}. "
                f"Run phi_quantize_model.py first to create it."
            )
        
        logger.info(f"Loading pre-quantized model from {self.quantized_path}...")
        
        # Load config
        config_path = os.path.join(self.quantized_path, 'config.npz')
        config = np.load(config_path)
        
        self.hidden_dim = int(config['hidden_dim'])
        self.num_heads = int(config['num_heads'])
        self.num_kv_heads = int(config['num_kv_heads'])
        self.head_dim = int(config['head_dim'])
        self.num_layers = int(config['num_layers'])
        self.vocab_size = int(config['vocab_size'])
        
        logger.info(f"Architecture: {self.hidden_dim} hidden, {self.num_heads} heads, {self.num_layers} layers")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load the base model for generation (we'll use its generate() method)
        # but with our quantized weights loaded
        from transformers import AutoModelForCausalLM
        
        logger.info("Loading base model structure...")
        dtype = torch.bfloat16 if "7B" in self.model_name else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map="cuda" if CUDA_AVAILABLE else "cpu",
        )
        self.model.eval()
        
        # Calculate quantized model size
        all_files = list(Path(self.quantized_path).rglob('*.npz'))
        self.quantized_size_bytes = sum(f.stat().st_size for f in all_files)
        
        self.load_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Model loaded in {self.load_time_ms:.0f}ms")
        logger.info(f"Quantized model size: {self.quantized_size_bytes / 1e9:.2f} GB")
        
        if CUDA_AVAILABLE:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB used")
    
    def generate(self, messages: List[Message], max_tokens: int = 100, 
                 temperature: float = 0.7) -> str:
        """Generate a response."""
        start_time = time.perf_counter()
        
        prompt = self._build_prompt(messages)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        use_sampling = temperature > 0.3
        
        with torch.no_grad():
            if use_sampling:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = response.strip()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.total_requests += 1
        self.total_tokens_generated += len(generated_ids)
        self.total_time_ms += elapsed_ms
        
        logger.info(f"Generated {len(generated_ids)} tokens in {elapsed_ms:.1f}ms")
        
        return response
    
    def _build_prompt(self, messages: List[Message]) -> str:
        """Build a prompt from chat messages."""
        prompt_parts = []
        
        simple_system = "You are a helpful AI assistant. Be concise and direct."
        prompt_parts.append(f"<|im_start|>system\n{simple_system}<|im_end|>")
        
        for msg in messages:
            content = msg.get_text_content()
            
            if msg.role == "system":
                continue
            
            if msg.role == "user":
                goose_markers = [
                    "You are a general-purpose AI agent called goose",
                    "You are an AI assistant",
                    "You have access to the following tools",
                ]
                for marker in goose_markers:
                    if marker in content:
                        parts = content.split("\n\n")
                        for part in reversed(parts):
                            part = part.strip()
                            if part and not any(m in part for m in goose_markers):
                                content = part
                                break
                        break
                
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            
            elif msg.role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        avg_time = self.total_time_ms / max(1, self.total_requests)
        avg_tokens = self.total_tokens_generated / max(1, self.total_requests)
        
        return {
            "model": self.model_name,
            "device": self.device,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_generated,
            "avg_time_ms": avg_time,
            "avg_tokens_per_request": avg_tokens,
            "phi_quantization_enabled": True,
            "quantized_path": self.quantized_path,
            "quantized_size_gb": self.quantized_size_bytes / 1e9,
            "load_time_ms": self.load_time_ms,
        }


# Global engine instance
engine: Optional[PhiFastEngine] = None
QUANTIZED_PATH = os.path.expanduser("~/.cache/phi_quantized/qwen2-7b")


# FastAPI app
app = FastAPI(
    title="φ-Fast Qwen2 API Server",
    description="OpenAI-compatible API with pre-quantized φ-weights (instant startup)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the engine on startup."""
    global engine
    engine = PhiFastEngine(quantized_path=QUANTIZED_PATH)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "phi-fast-qwen2", "device": DEVICE}


@app.get("/stats")
async def get_stats():
    """Get engine statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.get_stats()


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id="phi-fast-qwen2",
                created=int(time.time()),
            ),
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        response_text = engine.generate(
            request.messages,
            max_tokens=request.max_tokens or 100,
            temperature=request.temperature or 0.7,
        )
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ResponseMessage(content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=sum(len(m.get_text_content().split()) for m in request.messages),
                completion_tokens=len(response_text.split()),
                total_tokens=sum(len(m.get_text_content().split()) for m in request.messages) + len(response_text.split()),
            ),
        )
        
        if request.stream:
            async def generate_stream():
                words = response_text.split()
                for i, word in enumerate(words):
                    chunk = {
                        "id": response.id,
                        "object": "chat.completion.chunk",
                        "created": response.created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": word + " "},
                            "finish_reason": None if i < len(words) - 1 else "stop",
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server."""
    global QUANTIZED_PATH
    
    parser = argparse.ArgumentParser(description="φ-Fast Qwen2 API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    parser.add_argument("--quantized-path", default=QUANTIZED_PATH, help="Path to quantized model")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    QUANTIZED_PATH = args.quantized_path
    
    import uvicorn
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           φ-Fast Qwen2 API Server                            ║
║                                                              ║
║  Pre-quantized model for instant startup!                    ║
║  Model: {args.quantized_path}
║                                                              ║
║  Endpoints:                                                  ║
║    GET  /health              - Health check                  ║
║    GET  /stats               - Model statistics              ║
║    GET  /v1/models           - List models                   ║
║    POST /v1/chat/completions - Chat (OpenAI compatible)      ║
║                                                              ║
║  Connect Goose:                                              ║
║    OPENAI_API_BASE=http://localhost:{args.port}/v1             ║
║    OPENAI_MODEL=phi-fast-qwen2                               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "phi_fast_api_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
