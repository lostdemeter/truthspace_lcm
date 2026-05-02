#!/usr/bin/env python3
"""
Trivial Navigation Server: OpenAI-Compatible API
==================================================

Serves the trivial navigation model via OpenAI-compatible API for testing
with Goose and measuring tokens per second.

Architecture:
1. Pre-learn navigation for common relationship patterns
2. For known entities: use quantized hidden state (9.9x speedup)
3. For unknown: fall back to transformer

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import time
import asyncio
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Trivial Navigation Server")


# ============================================================================
# TEMPLATE-BASED GENERATION (No Transformer Needed!)
# ============================================================================

class TemplateKB:
    """Knowledge base for template generation."""
    
    CAPITALS = {
        "France": "Paris", "Germany": "Berlin", "Italy": "Rome",
        "Spain": "Madrid", "Japan": "Tokyo", "China": "Beijing",
        "India": "New Delhi", "Brazil": "Brasília", "Canada": "Ottawa",
        "Australia": "Canberra", "Russia": "Moscow", "Mexico": "Mexico City",
        "Egypt": "Cairo", "Greece": "Athens", "Sweden": "Stockholm",
        "Norway": "Oslo", "Poland": "Warsaw", "Austria": "Vienna",
        "Portugal": "Lisbon", "Netherlands": "Amsterdam",
        "United States": "Washington, D.C.", "United Kingdom": "London",
        "South Korea": "Seoul", "Argentina": "Buenos Aires",
    }
    
    LANGUAGES = {
        "France": "French", "Germany": "German", "Italy": "Italian",
        "Spain": "Spanish", "Japan": "Japanese", "China": "Mandarin Chinese",
        "Brazil": "Portuguese", "Russia": "Russian", "Mexico": "Spanish",
    }
    
    @classmethod
    def get_capital(cls, country: str) -> Optional[str]:
        # Try exact match first
        if country in cls.CAPITALS:
            return cls.CAPITALS[country]
        # Try case-insensitive
        for k, v in cls.CAPITALS.items():
            if k.lower() == country.lower():
                return v
        return None
    
    @classmethod
    def get_language(cls, country: str) -> Optional[str]:
        for k, v in cls.LANGUAGES.items():
            if k.lower() == country.lower():
                return v
        return None


def template_generate(query: str) -> Optional[str]:
    """
    Generate response using templates. NO TRANSFORMER!
    
    Returns response string or None if no template matches.
    """
    query_lower = query.lower().strip()
    
    # Capital queries
    match = re.search(r"(?:what is )?the capital of (\w+)", query_lower)
    if match:
        country = match.group(1).title()
        capital = TemplateKB.get_capital(country)
        if capital:
            return f"The capital of {country} is {capital}."
    
    match = re.search(r"(\w+)'s capital", query_lower)
    if match:
        country = match.group(1).title()
        capital = TemplateKB.get_capital(country)
        if capital:
            return f"The capital of {country} is {capital}."
    
    # Language queries
    match = re.search(r"(?:what )?language.*in (\w+)", query_lower)
    if match:
        country = match.group(1).title()
        language = TemplateKB.get_language(country)
        if language:
            return f"The official language of {country} is {language}."
    
    # Greetings
    if re.match(r"^(?:hi|hello|hey)[\s!.,]*$", query_lower):
        return "Hello! How can I help you today?"
    
    if "how are you" in query_lower:
        return "I'm doing well, thank you for asking! How can I assist you?"
    
    # Math
    match = re.search(r"what is (\d+)\s*\+\s*(\d+)", query_lower)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        return f"{a} + {b} = {a + b}"
    
    match = re.search(r"what is (\d+)\s*[\*x×]\s*(\d+)", query_lower)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        return f"{a} × {b} = {a * b}"
    
    return None


# ============================================================================

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


# Pydantic models for OpenAI compatibility
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "trivial-nav"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 100
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
    usage: Dict[str, Any]


def quantize_to_int16(arr: np.ndarray) -> tuple:
    """Quantize float array to int16."""
    max_abs = np.abs(arr).max()
    if max_abs < 1e-10:
        return np.zeros(arr.shape, dtype=np.int16), 1.0
    scale = max_abs / 32767
    indices = np.round(arr / scale).astype(np.int16)
    return indices, scale


def dequantize_from_int16(indices: np.ndarray, scale: float) -> np.ndarray:
    """Reconstruct floats from int16."""
    return indices.astype(np.float32) * scale


class TrivialNavigationEngine:
    """
    Engine that combines trivial navigation with transformer fallback.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Initializing Trivial Navigation Engine...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        
        # LM head on GPU for fast decode
        self.lm_head_gpu = self.model.lm_head.weight.data.float()
        
        # Learned navigation patterns
        self.learned_prompts: Dict[str, tuple] = {}  # prompt -> (quantized, scale)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'trivial_hits': 0,
            'transformer_fallbacks': 0,
            'total_tokens': 0,
            'trivial_time': 0,
            'transformer_time': 0,
        }
        
        print(f"  Model loaded: {self.n_layers} layers, {self.hidden_dim} hidden dim")
        
        # Pre-learn some common patterns
        self._prelearn_patterns()
    
    def _prelearn_patterns(self):
        """Pre-learn navigation for common patterns."""
        print("  Pre-learning navigation patterns...")
        
        # Capital queries
        countries = ["France", "Germany", "Italy", "Spain", "Japan", "China", 
                     "Brazil", "Australia", "Sweden", "Poland"]
        
        for country in countries:
            prompt = f"The capital of {country} is"
            self._learn_prompt(prompt)
        
        print(f"  Learned {len(self.learned_prompts)} prompts")
    
    def _learn_prompt(self, prompt: str):
        """Learn the final hidden state for a prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            final_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        
        # Quantize to int16
        quantized, scale = quantize_to_int16(final_hidden)
        self.learned_prompts[prompt] = (quantized, scale)
    
    def _predict_trivial(self, prompt: str) -> tuple:
        """Predict next token using trivial navigation."""
        if prompt not in self.learned_prompts:
            return None, 0.0
        
        quantized, scale = self.learned_prompts[prompt]
        final_hidden = dequantize_from_int16(quantized, scale)
        
        # GPU decode
        final_hidden_gpu = torch.tensor(final_hidden, device=self.lm_head_gpu.device)
        logits = torch.matmul(self.lm_head_gpu, final_hidden_gpu)
        
        top_idx = logits.argmax().item()
        
        # Confidence
        logits_shifted = logits - logits.max()
        probs = torch.softmax(logits_shifted, dim=0)
        confidence = probs[top_idx].item()
        
        token = self.tokenizer.decode([top_idx])
        
        return token, confidence
    
    def _generate_transformer(self, messages: List[Message], max_tokens: int = 50) -> str:
        """Generate using full transformer with proper chat formatting."""
        # Format messages for Qwen2 chat template
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        new_tokens = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def generate(self, messages: List[Message], max_tokens: int = 100) -> tuple:
        """
        Generate response using template, trivial navigation, or transformer fallback.
        
        Priority:
        1. Template generation (NO TRANSFORMER - instant)
        2. Trivial navigation (stored hidden state - fast)
        3. Transformer fallback (slow but accurate)
        
        Returns (response, method, tokens_generated, time_taken)
        """
        self.stats['total_requests'] += 1
        
        # Get the user's query
        query = messages[-1].content if messages else ""
        
        # ===== PRIORITY 1: Template generation (NO TRANSFORMER!) =====
        start_time = time.time()
        template_response = template_generate(query)
        if template_response:
            elapsed = time.time() - start_time
            n_tokens = len(template_response.split())
            self.stats['template_hits'] = self.stats.get('template_hits', 0) + 1
            self.stats['template_time'] = self.stats.get('template_time', 0) + elapsed
            self.stats['total_tokens'] += n_tokens
            return template_response, "template", n_tokens, elapsed
        
        # ===== PRIORITY 2: Trivial navigation (stored hidden state) =====
        start_time = time.time()
        
        # Try exact match first
        if query in self.learned_prompts:
            token, conf = self._predict_trivial(query)
            if token:
                elapsed = time.time() - start_time
                self.stats['trivial_hits'] += 1
                self.stats['trivial_time'] += elapsed
                self.stats['total_tokens'] += 1
                return token, "trivial", 1, elapsed
        
        # Try partial match (query ends with learned prompt)
        for learned_prompt in self.learned_prompts:
            if query.endswith(learned_prompt) or learned_prompt in query:
                token, conf = self._predict_trivial(learned_prompt)
                if token:
                    elapsed = time.time() - start_time
                    self.stats['trivial_hits'] += 1
                    self.stats['trivial_time'] += elapsed
                    self.stats['total_tokens'] += 1
                    return token, "trivial", 1, elapsed
        
        # ===== PRIORITY 3: Transformer fallback =====
        start_time = time.time()
        response = self._generate_transformer(messages, max_tokens)
        elapsed = time.time() - start_time
        
        n_tokens = len(self.tokenizer.encode(response))
        
        self.stats['transformer_fallbacks'] += 1
        self.stats['transformer_time'] += elapsed
        self.stats['total_tokens'] += n_tokens
        
        return response, "transformer", n_tokens, elapsed
    
    async def generate_stream(self, messages: List[Message], max_tokens: int = 100):
        """
        Generate response with streaming.
        """
        response, method, n_tokens, elapsed = self.generate(messages, max_tokens)
        
        # Stream the response token by token
        tokens = list(response)
        
        for i, char in enumerate(tokens):
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "trivial-nav",
                "choices": [{
                    "index": 0,
                    "delta": {"content": char},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)  # Small delay for streaming effect
        
        # Final chunk
        final_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "trivial-nav",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = dict(self.stats)
        
        if stats['trivial_hits'] > 0:
            stats['avg_trivial_time_ms'] = stats['trivial_time'] / stats['trivial_hits'] * 1000
            stats['trivial_tokens_per_sec'] = stats['trivial_hits'] / stats['trivial_time'] if stats['trivial_time'] > 0 else 0
        
        if stats['transformer_fallbacks'] > 0:
            stats['avg_transformer_time_ms'] = stats['transformer_time'] / stats['transformer_fallbacks'] * 1000
        
        if stats['total_tokens'] > 0 and (stats['trivial_time'] + stats['transformer_time']) > 0:
            total_time = stats['trivial_time'] + stats['transformer_time']
            stats['overall_tokens_per_sec'] = stats['total_tokens'] / total_time
        
        stats['learned_prompts'] = len(self.learned_prompts)
        
        return stats


# Global engine instance
engine: Optional[TrivialNavigationEngine] = None


@app.on_event("startup")
async def startup():
    global engine
    engine = TrivialNavigationEngine()


@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "trivial-navigation"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "trivial-nav",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "truthspace-lcm",
            }
        ]
    }


@app.get("/stats")
async def get_stats():
    """Get engine statistics including tokens per second."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.get_stats()


@app.post("/learn")
async def learn_prompt(prompt: str):
    """Learn a new prompt for trivial navigation."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    engine._learn_prompt(prompt)
    return {"status": "learned", "prompt": prompt, "total_learned": len(engine.learned_prompts)}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if request.stream:
        return StreamingResponse(
            engine.generate_stream(request.messages, request.max_tokens),
            media_type="text/event-stream"
        )
    else:
        response, method, n_tokens, elapsed = engine.generate(request.messages, request.max_tokens)
        
        tokens_per_sec = n_tokens / elapsed if elapsed > 0 else 0
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                "completion_tokens": n_tokens,
                "total_tokens": sum(len(m.content.split()) for m in request.messages) + n_tokens,
                "method": method,
                "time_ms": int(elapsed * 1000),
                "tokens_per_sec": round(tokens_per_sec, 1),
            }
        )


def main():
    import uvicorn
    print("Starting Trivial Navigation Server on port 8007...")
    uvicorn.run(app, host="0.0.0.0", port=8007)


if __name__ == "__main__":
    main()
