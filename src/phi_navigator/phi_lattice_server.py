#!/usr/bin/env python3
"""
φ-Lattice Navigation Server
============================

Chat server that uses φ-lattice steering for generation.

Key insight: We don't replace inference - we STEER it by injecting
φ-lattice positions into the embedding space.

The diffraction model (Doc 059):
- Knowledge Source: The φ-lattice position we inject
- Style Source: The model's learned layer transformations
- Interference: The model generates text influenced by both

Run with:
    cd /home/thorin/truthspace-lcm
    source venv/bin/activate
    python src/phi_navigator/phi_lattice_server.py --port 8008
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import time
import argparse
import uuid
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K_SCALE = 128


@dataclass
class ConceptPosition:
    """A position in φ-lattice space."""
    name: str
    levels: torch.Tensor
    signs: torch.Tensor
    keywords: List[str] = field(default_factory=list)


class PhiLatticeEngine:
    """
    φ-Lattice navigation engine for steered generation.
    
    Uses the φ-lattice rules (Doc 163) to navigate concept space
    and steer model generation via embedding injection.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Concept library: maps keywords to φ-lattice positions
        self.concepts: Dict[str, ConceptPosition] = {}
        
        # Intent patterns for routing
        self.intent_patterns = {
            'explain': ['explain', 'what is', 'describe', 'tell me about', 'how does'],
            'code': ['write code', 'code for', 'implement', 'function', 'program'],
            'compare': ['compare', 'difference between', 'vs', 'versus'],
            'list': ['list', 'enumerate', 'what are the', 'give me'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            'farewell': ['goodbye', 'bye', 'see you', 'take care'],
            'thanks': ['thanks', 'thank you', 'appreciate'],
        }
        
        # Stats
        self.total_requests = 0
        self.steered_requests = 0
        
        self._load_model()
        self._build_concept_library()
    
    def _load_model(self):
        """Load the model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        
        logger.info("Model loaded successfully")
    
    def _encode_to_phi_lattice(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode tensor to φ-lattice coordinates (levels, signs)."""
        signs = torch.sign(tensor)
        signs[signs == 0] = 1
        magnitudes = tensor.abs().clamp(min=1e-45)
        levels = torch.round(K_SCALE * torch.log(magnitudes) / LOG_PHI)
        return levels.to(torch.int16), signs.to(torch.int8)
    
    def _decode_from_phi_lattice(self, levels: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """Decode φ-lattice coordinates to tensor."""
        exponents = levels.float() / K_SCALE
        magnitudes = torch.exp(exponents * LOG_PHI)
        return signs.float() * magnitudes
    
    def _get_embedding_position(self, text: str) -> torch.Tensor:
        """Get the embedding position for a text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeds = self.model.model.embed_tokens(inputs['input_ids'])
            position = embeds.mean(dim=1).squeeze()
        return position.float().cpu()
    
    def _build_concept_library(self):
        """Build library of concept positions in φ-lattice space."""
        logger.info("Building concept library...")
        
        # Define concepts with their seed phrases
        concept_seeds = {
            # Programming languages
            'python': "Python programming language code script",
            'java': "Java programming language object-oriented",
            'javascript': "JavaScript web browser frontend",
            'rust': "Rust programming language memory safety",
            
            # Technical concepts
            'programming': "programming coding software development",
            'algorithm': "algorithm data structure complexity",
            'database': "database SQL query storage",
            'api': "API interface endpoint request response",
            'machine_learning': "machine learning AI neural network",
            
            # Explanation style
            'explanation': "explain describe definition meaning",
            'tutorial': "tutorial guide step by step how to",
            'comparison': "compare contrast difference similarity",
            
            # General topics
            'science': "science physics chemistry biology",
            'mathematics': "mathematics algebra calculus geometry",
            'history': "history historical events timeline",
            'philosophy': "philosophy ethics logic reasoning",
        }
        
        for name, seed_text in concept_seeds.items():
            position = self._get_embedding_position(seed_text)
            levels, signs = self._encode_to_phi_lattice(position)
            
            self.concepts[name] = ConceptPosition(
                name=name,
                levels=levels,
                signs=signs,
                keywords=seed_text.lower().split(),
            )
        
        logger.info(f"Built {len(self.concepts)} concept positions")
    
    def _detect_intent(self, query: str) -> Tuple[str, float]:
        """Detect the intent of a query."""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return intent, 1.0
        
        return 'general', 0.5
    
    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract the main topic from a query."""
        query_lower = query.lower()
        
        # Check for known concepts
        for concept_name, concept in self.concepts.items():
            if concept_name in query_lower:
                return concept_name
            for keyword in concept.keywords:
                if keyword in query_lower:
                    return concept_name
        
        return None
    
    def _interpolate_positions(
        self, 
        pos1: ConceptPosition, 
        pos2: ConceptPosition, 
        t: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interpolate between two φ-lattice positions (Rule 8)."""
        new_levels = torch.round(
            (1-t) * pos1.levels.float() + t * pos2.levels.float()
        ).to(torch.int16)
        
        # For signs, use probabilistic mixing
        new_signs = torch.where(
            torch.rand_like(pos1.signs.float()) < t,
            pos2.signs,
            pos1.signs
        )
        
        return new_levels, new_signs
    
    def _combine_positions(
        self, 
        positions: List[ConceptPosition],
        weights: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine multiple φ-lattice positions (Rule 10)."""
        if weights is None:
            weights = [1.0 / len(positions)] * len(positions)
        
        # Weighted average of levels
        combined_levels = torch.zeros_like(positions[0].levels.float())
        for pos, w in zip(positions, weights):
            combined_levels += w * pos.levels.float()
        combined_levels = torch.round(combined_levels).to(torch.int16)
        
        # Multiplicative combination of signs
        combined_signs = positions[0].signs.clone()
        for pos in positions[1:]:
            combined_signs = combined_signs * pos.signs
        
        return combined_levels, combined_signs
    
    def _generate_with_steering(
        self,
        prompt: str,
        steering_position: torch.Tensor,
        steering_strength: float = 0.1,
        max_tokens: int = 200,
    ) -> str:
        """Generate text with φ-lattice steering."""
        # Format as chat
        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Get original embeddings
            embeds = self.model.model.embed_tokens(inputs['input_ids'])
            
            # Normalize steering position and move to device
            steering_position = steering_position.to(self.device)
            pos_norm = steering_position / (steering_position.norm() + 1e-10)
            embed_scale = embeds[:, -1, :].norm()
            
            # Inject steering position as perturbation
            modified_embeds = embeds.clone()
            steering_vector = (steering_strength * pos_norm * embed_scale).to(self.model.dtype)
            modified_embeds[:, -1, :] = modified_embeds[:, -1, :] + steering_vector
            
            # Generate
            outputs = self.model.generate(
                inputs_embeds=modified_embeds,
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response
    
    def _generate_streaming(
        self,
        prompt: str,
        steering_position: Optional[torch.Tensor],
        steering_strength: float = 0.1,
        max_tokens: int = 200,
    ):
        """Generate text with streaming output."""
        # Format as chat
        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embeds = self.model.model.embed_tokens(inputs['input_ids'])
            
            if steering_position is not None:
                # Apply steering
                steering_position = steering_position.to(self.device)
                pos_norm = steering_position / (steering_position.norm() + 1e-10)
                embed_scale = embeds[:, -1, :].norm()
                steering_vector = (steering_strength * pos_norm * embed_scale).to(self.model.dtype)
                embeds[:, -1, :] = embeds[:, -1, :] + steering_vector
            
            # Generate token by token
            generated_ids = []
            past_key_values = None
            
            for _ in range(max_tokens):
                if past_key_values is None:
                    outputs = self.model(
                        inputs_embeds=embeds,
                        attention_mask=inputs['attention_mask'],
                        use_cache=True,
                    )
                else:
                    # Use last generated token
                    last_token_embed = self.model.model.embed_tokens(
                        torch.tensor([[generated_ids[-1]]], device=self.device)
                    )
                    outputs = self.model(
                        inputs_embeds=last_token_embed,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                
                # Sample
                probs = torch.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Check for EOS
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token)
                token_text = self.tokenizer.decode([next_token])
                
                yield token_text
    
    def navigate_and_generate(self, query: str, max_tokens: int = 200) -> Dict:
        """
        Main entry point: navigate φ-lattice and generate response.
        
        1. Detect intent
        2. Extract topic
        3. Compute steering position
        4. Generate with steering
        """
        self.total_requests += 1
        start_time = time.perf_counter()
        
        # Detect intent
        intent, intent_confidence = self._detect_intent(query)
        
        # Extract topic
        topic = self._extract_topic(query)
        
        # Compute steering position
        steering_position = None
        steering_info = {}
        
        if topic and topic in self.concepts:
            topic_concept = self.concepts[topic]
            
            # If we have an intent concept, combine them
            if intent == 'explain' and 'explanation' in self.concepts:
                intent_concept = self.concepts['explanation']
                levels, signs = self._interpolate_positions(
                    topic_concept, intent_concept, t=0.3
                )
            elif intent == 'code' and 'programming' in self.concepts:
                intent_concept = self.concepts['programming']
                levels, signs = self._interpolate_positions(
                    topic_concept, intent_concept, t=0.3
                )
            else:
                levels, signs = topic_concept.levels, topic_concept.signs
            
            steering_position = self._decode_from_phi_lattice(levels, signs)
            self.steered_requests += 1
            
            steering_info = {
                'topic': topic,
                'intent': intent,
                'steered': True,
                'mean_level': levels.float().mean().item(),
            }
        else:
            steering_info = {
                'topic': None,
                'intent': intent,
                'steered': False,
            }
        
        # Generate
        if steering_position is not None:
            response = self._generate_with_steering(
                query, steering_position, 
                steering_strength=0.1,
                max_tokens=max_tokens
            )
        else:
            # Fallback to unsteered generation
            response = self._generate_with_steering(
                query, 
                torch.zeros(self.model.config.hidden_size),
                steering_strength=0.0,
                max_tokens=max_tokens
            )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'query': query,
            'response': response,
            'steering': steering_info,
            'time_ms': elapsed_ms,
        }
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'model': self.model_name,
            'concepts': list(self.concepts.keys()),
            'total_requests': self.total_requests,
            'steered_requests': self.steered_requests,
            'steering_rate': self.steered_requests / max(1, self.total_requests),
        }


# FastAPI app
app = FastAPI(title="φ-Lattice Navigation Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine
engine: PhiLatticeEngine = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "phi-lattice"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 200
    stream: bool = False


@app.on_event("startup")
async def startup():
    global engine
    engine = PhiLatticeEngine()


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "phi-lattice"}


@app.get("/stats")
async def stats():
    return engine.get_stats()


@app.get("/concepts")
async def concepts():
    """List available concepts."""
    return {
        name: {
            'keywords': concept.keywords,
            'mean_level': concept.levels.float().mean().item(),
        }
        for name, concept in engine.concepts.items()
    }


@app.post("/navigate")
async def navigate(query: str):
    """Navigate and generate."""
    result = engine.navigate_and_generate(query)
    return result


def create_stream_response(engine: PhiLatticeEngine, query: str, max_tokens: int):
    """Generate SSE stream."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    # Detect intent and topic for steering
    intent, _ = engine._detect_intent(query)
    topic = engine._extract_topic(query)
    
    steering_position = None
    if topic and topic in engine.concepts:
        topic_concept = engine.concepts[topic]
        if intent == 'explain' and 'explanation' in engine.concepts:
            intent_concept = engine.concepts['explanation']
            levels, signs = engine._interpolate_positions(
                topic_concept, intent_concept, t=0.3
            )
        else:
            levels, signs = topic_concept.levels, topic_concept.signs
        steering_position = engine._decode_from_phi_lattice(levels, signs)
    
    # Stream tokens
    for token in engine._generate_streaming(query, steering_position, max_tokens=max_tokens):
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "phi-lattice",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send finish
    finish_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "phi-lattice",
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
    """OpenAI-compatible chat endpoint with φ-lattice steering."""
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
    
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            create_stream_response(engine, user_message, request.max_tokens),
            media_type="text/event-stream",
        )
    
    # Non-streaming
    result = engine.navigate_and_generate(user_message, max_tokens=request.max_tokens)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "phi-lattice",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result['response'],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(result['response'].split()),
            "total_tokens": len(user_message.split()) + len(result['response'].split()),
        },
        "steering_info": result['steering'],
    }


def main():
    parser = argparse.ArgumentParser(description="φ-Lattice Navigation Server")
    parser.add_argument("--port", type=int, default=8008, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              φ-LATTICE NAVIGATION SERVER                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Steered generation via φ-lattice positions                      ║
║                                                                  ║
║  The diffraction model:                                          ║
║    Knowledge Source = φ-lattice position                         ║
║    Style Source = model's layer transformations                  ║
║    Output = interference of both                                 ║
║                                                                  ║
║  Endpoints:                                                      ║
║    GET  /health              - Health check                      ║
║    GET  /stats               - Statistics                        ║
║    GET  /concepts            - List concept positions            ║
║    POST /navigate            - Navigate and generate             ║
║    POST /v1/chat/completions - OpenAI-compatible chat            ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
