"""
OpenAI-Compatible API Server for the Gear Chain System

Provides a REST API compatible with OpenAI's chat completions endpoint,
allowing the gear chain to be used with tools like Open WebUI.

Author: Lesley Gushurst
License: GPLv3
"""

import re
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from truthspace_lcm.gears.core import GearChain, GearState
from truthspace_lcm.gears.practical_applications.nlp.error_correction import ErrorCorrectionGear
from truthspace_lcm.gears.practical_applications.nlp import (
    RoleGear, ActionGear, TenseGear,
    DomainGear, StructureGear, OutputGear,
)
from truthspace_lcm.gears.corpus import get_corpus_path
from truthspace_lcm.core.geometric import GeometricQA


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gear-chain"
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


class GearChainEngine:
    """
    The gear chain engine that powers the API.
    """
    
    def __init__(self):
        # Load corpus
        self.qa = GeometricQA()
        corpus_path = get_corpus_path("experimental")
        self.qa.load_corpus(str(corpus_path))
        self.qa.set_output_lens('natural')
        
        # Build gear chain
        self.chain = GearChain("APIChain")
        self.chain.add(RoleGear())
        self.chain.add(ActionGear())
        self.chain.add(TenseGear(tense='present'))
        self.chain.add(ErrorCorrectionGear())
        self.chain.add(DomainGear())
        self.chain.add(StructureGear())
        self.chain.add(OutputGear())
        
        # Settings
        self.tense = 'present'
    
    def _parse_to_state(self, truth: str, concept: str) -> GearState:
        """Parse truth output into gear state."""
        truth_lower = truth.lower()
        
        state = GearState()
        state.entity = concept.title()
        
        # Role - handle "is someone" as well as "is a X"
        match = re.search(r'is (someone|a[n]? (\w+))', truth_lower)
        if match:
            if match.group(1) == 'someone':
                state.role = 'entity'
            else:
                state.role = match.group(2) or 'entity'
        
        # Actions - handle "who/that verbs" pattern
        match = re.search(r'(?:who|that)\s+(\w+)(?:,\s*(\w+))?\s+and\s+(\w+)', truth_lower)
        if match:
            state.actions = [a for a in match.groups() if a]
        else:
            # Try simpler pattern
            match = re.search(r'(?:who|that)\s+(\w+)', truth_lower)
            if match:
                state.actions = [match.group(1)]
        
        # Targets - handle "relates to X and Y"
        match = re.search(r'relates? to\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            state.targets = [t for t in match.groups() if t]
        
        return state
    
    def _extract_concept(self, text: str) -> str:
        """Extract the main concept from a query."""
        text_lower = text.lower().strip()
        
        # Handle "what is X" questions
        match = re.search(r'what (?:is|are) (?:a |an |the )?(\w+)', text_lower)
        if match:
            return match.group(1)
        
        # Handle "tell me about X"
        match = re.search(r'tell me about (?:a |an |the )?(\w+)', text_lower)
        if match:
            return match.group(1)
        
        # Handle "explain X"
        match = re.search(r'explain (?:a |an |the )?(\w+)', text_lower)
        if match:
            return match.group(1)
        
        # Handle "who is X"
        match = re.search(r'who (?:is|was) (?:a |an |the )?(\w+)', text_lower)
        if match:
            return match.group(1)
        
        # Default: last word
        words = text_lower.split()
        return words[-1].rstrip('?!.') if words else 'unknown'
    
    def generate(self, messages: List[Message], temperature: float = 0.7) -> str:
        """Generate a response using the gear chain."""
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            return "I need a question to answer."
        
        # Handle special commands in the message
        if user_message.lower().startswith("set tense"):
            match = re.search(r'set tense (?:to )?(\w+)', user_message.lower())
            if match:
                tense = match.group(1)
                if tense in ['present', 'past', 'future', 'perfect']:
                    self.chain.get("TenseGear").set_tense(tense)
                    return f"Tense set to {tense}."
            return "Invalid tense. Use: present, past, future, perfect"
        
        # Extract concept and query
        concept = self._extract_concept(user_message)
        
        # Get truth from corpus
        truth = self.qa.ask(f"What is {concept}?")
        
        if "don't know" in truth.lower():
            return f"I don't have information about '{concept}' in my knowledge base. Try asking about concepts like evolution, physics, Holmes, Watson, or biochemistry."
        
        # Parse to state
        state = self._parse_to_state(truth, concept)
        
        # Process through gear chain
        result = self.chain.process(state)
        
        return result


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Gear Chain API",
        description="OpenAI-compatible API for the TruthSpace Gear Chain System",
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
    engine = GearChainEngine()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "engine": "gear-chain"}
    
    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        """List available models."""
        return ModelsResponse(
            data=[
                ModelInfo(
                    id="gear-chain",
                    created=int(time.time()),
                    owned_by="truthspace",
                ),
                ModelInfo(
                    id="gear-chain-past",
                    created=int(time.time()),
                    owned_by="truthspace",
                ),
                ModelInfo(
                    id="gear-chain-future",
                    created=int(time.time()),
                    owned_by="truthspace",
                ),
            ]
        )
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completions endpoint (OpenAI-compatible)."""
        
        logger.info(f"Received request: model={request.model}, stream={request.stream}")
        logger.info(f"Messages: {[m.content[:50] for m in request.messages]}")
        
        # Handle model-based tense selection
        if request.model == "gear-chain-past":
            engine.chain.get("TenseGear").set_tense('past')
        elif request.model == "gear-chain-future":
            engine.chain.get("TenseGear").set_tense('future')
        else:
            engine.chain.get("TenseGear").set_tense('present')
        
        try:
            response_text = engine.generate(
                request.messages,
                temperature=request.temperature or 0.7,
            )
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


# Create the app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
