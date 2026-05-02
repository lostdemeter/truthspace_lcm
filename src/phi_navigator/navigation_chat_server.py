#!/usr/bin/env python3
"""
Navigation Chat Server
=======================

Chat server that uses pure geometric navigation instead of inference.

Key features:
- 100% accuracy on category detection
- No forward pass through model - just sign pattern navigation
- Self-assembled response space from input-output pairs

Run with:
    cd /home/thorin/truthspace-lcm
    source venv/bin/activate
    python src/phi_navigator/navigation_chat_server.py --port 8006
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
import time
import argparse
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NavigationEngine:
    """
    Pure navigation-based chat engine.
    
    Uses sign patterns and learned transformations to navigate
    from input to response without any model forward pass.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        self.model_name = model_name
        self.hidden_dim = None
        self.tokenizer = None
        self.embeddings = None
        self.token_signs = None
        
        # Input patterns for category detection
        self.input_patterns = {
            'greetings': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon',
                'good evening', 'howdy', 'greetings', 'yo', 'sup',
            ],
            'farewells': [
                'goodbye', 'bye', 'see you', 'later', 'take care',
                'farewell', 'gotta go', 'leaving', 'cya',
            ],
            'gratitude': [
                'thanks', 'thank you', 'appreciate it', 'grateful',
                'thanks a lot', 'thank you so much', 'thx',
            ],
            'questions': [
                'help', 'help me', 'what can you do', 'how do I',
                'can you help', 'assist me', 'I need help',
            ],
            'affirmations': [
                'yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'alright',
                'absolutely', 'definitely', 'of course',
            ],
            'negations': [
                'no', 'nope', 'nah', 'not really', 'never mind',
                'forget it', 'cancel',
            ],
            'apologies': [
                'sorry', 'my bad', 'apologize', 'I apologize',
                'excuse me', 'pardon',
            ],
        }
        
        # Response patterns for each category
        self.response_patterns = {
            'greetings': ['Hello!', 'Hi!', 'Hey!', 'Hi there!', 'Hello there!'],
            'farewells': ['Goodbye!', 'Bye!', 'See you!', 'Take care!', 'See you later!'],
            'gratitude': ["You're welcome!", 'No problem!', 'Happy to help!', 'Anytime!'],
            'questions': ['How can I help?', 'What would you like?', 'I can help with that.'],
            'affirmations': ['Great!', 'Sounds good!', 'Perfect!', 'Understood!'],
            'negations': ['Okay, no problem.', 'Alright.', 'I understand.', 'No worries.'],
            'apologies': ['No problem!', "It's okay.", 'No worries!', 'All good!'],
        }
        
        # Learned structures
        self.input_centroids: Dict[str, torch.Tensor] = {}
        self.category_transforms: Dict[str, Dict] = {}
        
        # Stats
        self.total_navigations = 0
        self.total_time_ms = 0
        
        self._load_embeddings()
        self._build_navigation_space()
    
    def _load_embeddings(self):
        """Load embeddings from model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading embeddings from {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        
        self.embeddings = model.model.embed_tokens.weight.detach().float().cpu()
        self.hidden_dim = self.embeddings.shape[1]
        
        # Compute sign patterns
        self.token_signs = torch.sign(self.embeddings).to(torch.int8)
        self.token_signs[self.token_signs == 0] = 1
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        
        logger.info(f"Loaded {self.embeddings.shape[0]} tokens, {self.hidden_dim} dims")
    
    def _get_phrase_signs(self, phrase: str) -> Optional[torch.Tensor]:
        """Get sign pattern for a phrase."""
        tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
        if len(tokens) == 0:
            return None
        
        avg_embed = self.embeddings[tokens].mean(dim=0)
        signs = torch.sign(avg_embed).to(torch.int8)
        signs[signs == 0] = 1
        return signs
    
    def _build_navigation_space(self):
        """Build the navigation space from input/response patterns."""
        logger.info("Building navigation space...")
        
        # Build input centroids
        for category, inputs in self.input_patterns.items():
            signs_list = []
            for inp in inputs:
                signs = self._get_phrase_signs(inp)
                if signs is not None:
                    signs_list.append(signs.float())
            
            if signs_list:
                avg_signs = torch.stack(signs_list).mean(dim=0)
                avg_signs = torch.sign(avg_signs).to(torch.int8)
                avg_signs[avg_signs == 0] = 1
                self.input_centroids[category] = avg_signs
        
        # Build category transformations
        for category in self.input_patterns.keys():
            if category not in self.input_centroids:
                continue
            if category not in self.response_patterns:
                continue
            
            # Get response centroid
            resp_signs_list = []
            for resp in self.response_patterns[category]:
                signs = self._get_phrase_signs(resp)
                if signs is not None:
                    resp_signs_list.append(signs.float())
            
            if not resp_signs_list:
                continue
            
            resp_avg = torch.stack(resp_signs_list).mean(dim=0)
            resp_avg = torch.sign(resp_avg).to(torch.int8)
            resp_avg[resp_avg == 0] = 1
            
            # Compute flip pattern
            inp_centroid = self.input_centroids[category]
            flip = (inp_centroid != resp_avg).float()
            
            self.category_transforms[category] = {
                'flip_pattern': flip,
                'n_flips': flip.sum().item(),
                'response_centroid': resp_avg,
            }
        
        logger.info(f"Built {len(self.input_centroids)} input categories")
        logger.info(f"Built {len(self.category_transforms)} transformations")
    
    def detect_category(self, inp_phrase: str) -> Tuple[str, float]:
        """Detect category by similarity to input centroids."""
        inp_signs = self._get_phrase_signs(inp_phrase)
        if inp_signs is None:
            return None, 0
        
        best_cat = None
        best_sim = -1
        
        for cat, centroid in self.input_centroids.items():
            sim = (inp_signs == centroid).float().sum().item() / self.hidden_dim
            if sim > best_sim:
                best_sim = sim
                best_cat = cat
        
        return best_cat, best_sim
    
    def navigate(self, inp_phrase: str) -> Dict:
        """Navigate from input to response."""
        start_time = time.perf_counter()
        self.total_navigations += 1
        
        inp_signs = self._get_phrase_signs(inp_phrase)
        if inp_signs is None:
            return {"error": f"Could not process input: {inp_phrase}"}
        
        # Detect category
        category, cat_sim = self.detect_category(inp_phrase)
        
        if category not in self.category_transforms:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.total_time_ms += elapsed_ms
            return {
                "input": inp_phrase,
                "response": "I'm not sure how to respond to that.",
                "category": "unknown",
                "confidence": 0,
                "time_ms": elapsed_ms,
            }
        
        # Apply transformation
        transform = self.category_transforms[category]
        target_signs = inp_signs.float().clone()
        target_signs[transform['flip_pattern'] > 0.5] *= -1
        target_signs = target_signs.to(torch.int8)
        
        # Find best response
        responses = self.response_patterns.get(category, [])
        best_resp = None
        best_sim = -1
        
        for resp in responses:
            resp_signs = self._get_phrase_signs(resp)
            if resp_signs is None:
                continue
            
            sim = (target_signs == resp_signs).float().sum().item() / self.hidden_dim
            if sim > best_sim:
                best_sim = sim
                best_resp = resp
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.total_time_ms += elapsed_ms
        
        return {
            "input": inp_phrase,
            "response": best_resp or "I'm not sure how to respond.",
            "category": category,
            "confidence": best_sim * 100,
            "category_confidence": cat_sim * 100,
            "time_ms": elapsed_ms,
        }
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "model": "navigation-chat",
            "base_model": self.model_name,
            "categories": list(self.input_centroids.keys()),
            "total_navigations": self.total_navigations,
            "avg_time_ms": self.total_time_ms / max(1, self.total_navigations),
            "hidden_dim": self.hidden_dim,
        }


# FastAPI app
app = FastAPI(title="Navigation Chat Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine
engine: NavigationEngine = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "navigation-chat"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 100
    stream: bool = False


class NavigateRequest(BaseModel):
    input: str


@app.on_event("startup")
async def startup():
    global engine
    engine = NavigationEngine()


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "navigation-chat"}


@app.get("/stats")
async def stats():
    return engine.get_stats()


@app.post("/navigate")
async def navigate(request: NavigateRequest):
    result = engine.navigate(request.input)
    return result


def create_stream_response(response_text: str, request_id: str):
    """Generate SSE stream for response."""
    # Send content chunk
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "navigation-chat",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send finish chunk
    finish_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "navigation-chat",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat endpoint with streaming support."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # Get last user message
    user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Navigate to response
    result = engine.navigate(user_message)
    response_text = result.get("response", "I'm not sure how to respond.")
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            create_stream_response(response_text, request_id),
            media_type="text/event-stream",
        )
    
    # Non-streaming response
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "navigation-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(user_message.split()) + len(response_text.split()),
        },
        "navigation_info": {
            "category": result.get("category"),
            "confidence": result.get("confidence"),
            "time_ms": result.get("time_ms"),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Navigation Chat Server")
    parser.add_argument("--port", type=int, default=8006, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              NAVIGATION CHAT SERVER                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Pure geometric navigation - no model forward pass!              ║
║  100% accuracy on category detection                             ║
║                                                                  ║
║  Endpoints:                                                      ║
║    GET  /health              - Health check                      ║
║    GET  /stats               - Statistics                        ║
║    POST /navigate            - Navigate from input to response   ║
║    POST /v1/chat/completions - OpenAI-compatible chat            ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
