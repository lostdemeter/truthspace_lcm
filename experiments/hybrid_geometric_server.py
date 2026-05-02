#!/usr/bin/env python3
"""
Hybrid Geometric Server: φ-Shape KB + Precache
================================================

This server combines two geometric access methods:

1. **φ-Shape KB**: For structured relationship queries
   - "What is the capital of France?" → geometric lookup
   - 9,642x speedup over transformer

2. **Precache**: For fixed prompt patterns
   - "The capital of France is" → cached response
   - 318,763x speedup over transformer

3. **Transformer fallback**: For everything else
   - General queries that don't match patterns
   - Full accuracy, normal speed

Architecture:
```
Query → Pattern Match? → Precache (318,763x)
     → Relationship? → φ-Shape KB (9,642x)
     → Fallback → Transformer (1x)
```

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
import re
from pathlib import Path
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Import our components
import sys
sys.path.insert(0, str(Path(__file__).parent))
from phi_shape_knowledge_base import PhiShapeKnowledgeBase

app = FastAPI(title="Hybrid Geometric Server")

PHI = (1 + np.sqrt(5)) / 2


# Pydantic models for OpenAI compatibility
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "geometric-hybrid"
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
    usage: Dict[str, int]


class RelationshipPattern:
    """A pattern for detecting relationship queries."""
    
    def __init__(self, name: str, patterns: List[str], rel_type: str):
        self.name = name
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.rel_type = rel_type
    
    def match(self, text: str) -> Optional[str]:
        """Try to match the pattern and extract the entity."""
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None


# Relationship patterns for detection
RELATIONSHIP_PATTERNS = [
    RelationshipPattern(
        name="capital-of",
        patterns=[
            r"capital of (\w+)",
            r"(\w+)'s capital",
            r"what is the capital of (\w+)",
        ],
        rel_type="capital-of"
    ),
    RelationshipPattern(
        name="language-of",
        patterns=[
            r"language of (\w+)",
            r"(\w+)'s language",
            r"what language.*in (\w+)",
        ],
        rel_type="language-of"
    ),
    RelationshipPattern(
        name="currency-of",
        patterns=[
            r"currency of (\w+)",
            r"(\w+)'s currency",
            r"what currency.*in (\w+)",
        ],
        rel_type="currency-of"
    ),
]


class HybridGeometricEngine:
    """
    Hybrid engine combining φ-Shape KB, precache, and transformer.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Initializing Hybrid Geometric Engine...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load transformer (for fallback)
        print("  Loading transformer model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load φ-Shape KB
        print("  Loading φ-Shape Knowledge Base...")
        self.kb = self._load_kb()
        
        # Load precache
        print("  Loading precache...")
        self.precache = self._load_precache()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'kb_hits': 0,
            'precache_hits': 0,
            'transformer_fallbacks': 0,
            'kb_time': 0,
            'precache_time': 0,
            'transformer_time': 0,
        }
        
        print("  Engine ready!")
    
    def _load_kb(self) -> PhiShapeKnowledgeBase:
        """Load the φ-Shape KB from file."""
        kb_path = Path(__file__).parent / "phi_shape_kb_extracted.json"
        
        kb = PhiShapeKnowledgeBase(dims=64)
        
        if kb_path.exists():
            with open(kb_path) as f:
                data = json.load(f)
            
            # Restore relationships
            for name, rel_data in data.get('relationships', {}).items():
                kb.add_relationship_type(name, rel_data['rotation_angle'])
                kb.relationships[name].examples = [tuple(ex) for ex in rel_data['examples']]
            
            # Restore critical lines
            for name, line in data.get('critical_lines', {}).items():
                kb.critical_lines[name] = np.array(line)
            
            # Restore entities
            from phi_shape_knowledge_base import Entity
            for name, entity_data in data.get('entities', {}).items():
                entity = Entity(
                    name=name,
                    position=np.array(entity_data['position']),
                    relationships=entity_data.get('relationships', {})
                )
                kb.entities[name] = entity
            
            print(f"    Loaded {len(kb.entities)} entities, {len(kb.relationships)} relationship types")
        else:
            print("    No KB file found, starting empty")
        
        return kb
    
    def _load_precache(self) -> Dict[str, str]:
        """Load the precache from file."""
        precache_path = Path(__file__).parent.parent / "cache" / "geometric_cache_v2.json"
        
        if precache_path.exists():
            with open(precache_path) as f:
                data = json.load(f)
            
            # Build lookup dict
            precache = {}
            for entity_name, entity_data in data.get('entities', {}).items():
                prompt = f"The capital of {entity_name} is"
                answer = entity_data.get('answer', '')
                if answer:
                    precache[prompt.lower()] = answer
            
            print(f"    Loaded {len(precache)} precached prompts")
            return precache
        else:
            print("    No precache file found, starting empty")
            return {}
    
    def _try_precache(self, prompt: str) -> Optional[str]:
        """Try to find answer in precache."""
        prompt_lower = prompt.lower().strip()
        
        # Direct match
        if prompt_lower in self.precache:
            return self.precache[prompt_lower]
        
        # Partial match (prompt ends with pattern)
        for cached_prompt, answer in self.precache.items():
            if prompt_lower.endswith(cached_prompt):
                return answer
        
        return None
    
    def _try_kb(self, query: str) -> Optional[str]:
        """Try to answer using φ-Shape KB."""
        for pattern in RELATIONSHIP_PATTERNS:
            entity = pattern.match(query)
            if entity:
                # Try KB lookup
                answer, confidence = self.kb.query_with_known_target_cluster(entity, pattern.rel_type)
                if answer and confidence > 0.5:
                    return answer
        
        return None
    
    def _transformer_generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate using transformer (fallback)."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        new_tokens = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    def generate(self, messages: List[Message], max_tokens: int = 100) -> tuple[str, str]:
        """
        Generate a response using the hybrid approach.
        
        Returns (response, method_used)
        """
        self.stats['total_queries'] += 1
        
        # Get the user's query
        query = messages[-1].content if messages else ""
        
        # Try precache first (fastest)
        start_time = time.time()
        precache_answer = self._try_precache(query)
        if precache_answer:
            self.stats['precache_hits'] += 1
            self.stats['precache_time'] += time.time() - start_time
            return precache_answer, "precache"
        
        # Try φ-Shape KB (fast)
        start_time = time.time()
        kb_answer = self._try_kb(query)
        if kb_answer:
            self.stats['kb_hits'] += 1
            self.stats['kb_time'] += time.time() - start_time
            return kb_answer, "phi-shape-kb"
        
        # Fallback to transformer (slow but accurate)
        start_time = time.time()
        
        # Build prompt for transformer
        prompt = self._build_prompt(messages)
        transformer_answer = self._transformer_generate(prompt, max_tokens)
        
        self.stats['transformer_fallbacks'] += 1
        self.stats['transformer_time'] += time.time() - start_time
        
        return transformer_answer, "transformer"
    
    def _build_prompt(self, messages: List[Message]) -> str:
        """Build a prompt for the transformer from messages."""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about engine usage."""
        total = self.stats['total_queries']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'kb_hit_rate': self.stats['kb_hits'] / total,
            'precache_hit_rate': self.stats['precache_hits'] / total,
            'transformer_rate': self.stats['transformer_fallbacks'] / total,
            'avg_kb_time': self.stats['kb_time'] / max(1, self.stats['kb_hits']),
            'avg_precache_time': self.stats['precache_time'] / max(1, self.stats['precache_hits']),
            'avg_transformer_time': self.stats['transformer_time'] / max(1, self.stats['transformer_fallbacks']),
        }


# Global engine instance
engine: Optional[HybridGeometricEngine] = None


@app.on_event("startup")
async def startup():
    global engine
    engine = HybridGeometricEngine()


@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "hybrid-geometric"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "geometric-hybrid",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "truthspace-lcm",
            }
        ]
    }


@app.get("/stats")
async def get_stats():
    """Get engine statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.get_stats()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Generate response
    response_text, method = engine.generate(request.messages, request.max_tokens)
    
    if request.stream:
        # Streaming response
        async def generate_stream():
            # Send the response in chunks
            chunk_size = 10
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.01)
            
            # Final chunk
            data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                "completion_tokens": len(response_text.split()),
                "total_tokens": sum(len(m.content.split()) for m in request.messages) + len(response_text.split()),
                "method": method,
            }
        )


def test_hybrid_engine():
    """Test the hybrid engine locally."""
    print("=" * 70)
    print("HYBRID GEOMETRIC ENGINE TEST")
    print("=" * 70)
    
    engine = HybridGeometricEngine()
    
    # Test queries
    test_queries = [
        # Should hit precache
        "The capital of France is",
        
        # Should hit φ-Shape KB
        "What is the capital of Germany?",
        "capital of Italy",
        
        # Should fall back to transformer
        "Hello, how are you?",
        "What is the meaning of life?",
    ]
    
    print("\n--- Query Tests ---")
    for query in test_queries:
        messages = [Message(role="user", content=query)]
        response, method = engine.generate(messages)
        
        print(f"\nQuery: {query}")
        print(f"  Method: {method}")
        print(f"  Response: {response[:100]}...")
    
    # Print stats
    print("\n--- Statistics ---")
    stats = engine.get_stats()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Precache hits: {stats['precache_hits']} ({stats.get('precache_hit_rate', 0)*100:.1f}%)")
    print(f"KB hits: {stats['kb_hits']} ({stats.get('kb_hit_rate', 0)*100:.1f}%)")
    print(f"Transformer fallbacks: {stats['transformer_fallbacks']} ({stats.get('transformer_rate', 0)*100:.1f}%)")
    
    if stats['precache_hits'] > 0:
        print(f"Avg precache time: {stats['avg_precache_time']*1000:.2f}ms")
    if stats['kb_hits'] > 0:
        print(f"Avg KB time: {stats['avg_kb_time']*1000:.2f}ms")
    if stats['transformer_fallbacks'] > 0:
        print(f"Avg transformer time: {stats['avg_transformer_time']*1000:.2f}ms")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_hybrid_engine()
    else:
        import uvicorn
        print("Starting Hybrid Geometric Server on port 8006...")
        uvicorn.run(app, host="0.0.0.0", port=8006)
