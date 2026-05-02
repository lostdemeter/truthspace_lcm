#!/usr/bin/env python3
"""
Zeta Barrier LOD Server
=======================

Uses the zeta barrier structure (light cone at n=80, 137/30 ratio) to 
determine LOD switching for token generation.

Key insights from fine_structure_in_zeta_zeros.md:
- Light cone barrier at n=80 is a quantum phase transition
- Ratio of slopes = 137/30 ≈ 1/α (fine structure constant)
- Maps to confidence threshold at 1/φ ≈ 0.618

LOD Selection:
- Confidence > 1/φ: Post-horizon (quantum regime) → LOW LOD (fast)
- Confidence near 1/φ: At barrier → BARRIER LOD
- Confidence < 1/φ²: Pre-horizon (classical regime) → HIGH LOD (precise)

Run with:
    python experiments/model_reverse_engineering/zeta_barrier_lod_server.py --port 8006

Author: TruthSpace LCM Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import uuid
import argparse
import gc
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import json
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fundamental constants
PHI = 1.6180339887498949
BARRIER = 1 / PHI  # 0.618 - the light cone threshold
FINE_STRUCTURE_RATIO = 137 / 30  # 4.567 - from zeta zero analysis


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
    model: str = "zeta-barrier-qwen2"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False


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


@dataclass 
class ZetaBarrierConfig:
    """Configuration based on zeta barrier structure."""
    # k values - increased for quality while testing barrier concept
    # The barrier determines WHEN to switch, not HOW aggressive
    k_low: int = 1200      # Post-horizon (quantum regime) - still fast
    k_barrier: int = 1800  # At the light cone
    k_high: int = 2800     # Pre-horizon (classical regime) - near full
    
    # Confidence thresholds from φ
    barrier_high: float = BARRIER           # 0.618 - above this, use low LOD
    barrier_low: float = BARRIER / PHI      # 0.382 - below this, use high LOD
    
    # Layers to patch (middle layers showed best tolerance)
    start_layer: int = 8
    end_layer: int = 20


class LODLinear:
    """LOD linear with precomputed SVD at multiple k levels."""
    
    def __init__(self, weight: torch.Tensor, config: ZetaBarrierConfig, name: str = ""):
        self.config = config
        self.device = weight.device
        self.dtype = weight.dtype
        self.name = name
        self.out_features, self.in_features = weight.shape
        
        # For small projections, just use full weight
        if min(weight.shape) <= 512:
            self.is_small = True
            self.weight = weight
            return
        
        self.is_small = False
        self._precompute(weight)
    
    def _precompute(self, weight: torch.Tensor):
        W = weight.detach().float().cpu()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        
        # Precompute for three LOD levels based on zeta barrier
        self.components = {}
        for name, k in [('low', self.config.k_low),
                        ('barrier', self.config.k_barrier),
                        ('high', self.config.k_high)]:
            k = min(k, len(S))
            Vt_k_T = Vt[:k].T.contiguous()
            US_k = (U[:, :k] * S[:k]).T.contiguous()
            self.components[name] = (
                Vt_k_T.to(device=self.device, dtype=self.dtype),
                US_k.to(device=self.device, dtype=self.dtype)
            )
        del U, S, Vt, W
    
    def __call__(self, x: torch.Tensor, lod: str = 'barrier') -> torch.Tensor:
        if self.is_small:
            return F.linear(x, self.weight)
        Vt_k, US_k = self.components[lod]
        return (x @ Vt_k) @ US_k


class PatchedQwen2MLP(nn.Module):
    """Patched MLP with zeta barrier LOD."""
    
    _current_lod = 'barrier'
    
    @classmethod
    def set_lod(cls, lod: str):
        cls._current_lod = lod
    
    def __init__(self, original_mlp, config: ZetaBarrierConfig, layer_idx: int):
        super().__init__()
        self.gate = LODLinear(original_mlp.gate_proj.weight, config, f"L{layer_idx}.gate")
        self.up = LODLinear(original_mlp.up_proj.weight, config, f"L{layer_idx}.up")
        self.down = LODLinear(original_mlp.down_proj.weight, config, f"L{layer_idx}.down")
        self.act_fn = original_mlp.act_fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lod = PatchedQwen2MLP._current_lod
        gate = self.gate(x, lod)
        up = self.up(x, lod)
        return self.down(self.act_fn(gate) * up, lod)


class LODLinearWrapper(nn.Module):
    """Wrapper for attention projections."""
    
    _current_lod = 'barrier'
    
    @classmethod
    def set_lod(cls, lod: str):
        cls._current_lod = lod
    
    def __init__(self, original_linear: nn.Linear, config: ZetaBarrierConfig, name: str = ""):
        super().__init__()
        self.lod_linear = LODLinear(original_linear.weight, config, name)
        self.bias = original_linear.bias if original_linear.bias is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lod_linear(x, LODLinearWrapper._current_lod)
        if self.bias is not None:
            out = out + self.bias
        return out


class ZetaBarrierEngine:
    """Engine using zeta barrier for LOD selection."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        self.model_name = model_name
        self.config = ZetaBarrierConfig()
        
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time_ms': 0,
            'lod_low': 0,
            'lod_barrier': 0,
            'lod_high': 0,
            'barrier_crossings': 0,
        }
        
        self._load_and_patch()
    
    def _load_and_patch(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        
        self.n_layers = self.model.config.num_hidden_layers
        logger.info(f"Model loaded: {self.n_layers} layers")
        
        # Patch layers with zeta barrier LOD
        start = self.config.start_layer
        end = min(self.config.end_layer, self.n_layers)
        logger.info(f"Patching layers {start}-{end-1} with Zeta Barrier LOD...")
        logger.info(f"  k_low={self.config.k_low} (post-horizon)")
        logger.info(f"  k_barrier={self.config.k_barrier} (at light cone)")
        logger.info(f"  k_high={self.config.k_high} (pre-horizon)")
        logger.info(f"  Barrier thresholds: high={self.config.barrier_high:.3f}, low={self.config.barrier_low:.3f}")
        
        for i in range(start, end):
            layer = self.model.model.layers[i]
            
            # Patch MLP
            original_mlp = layer.mlp
            layer.mlp = PatchedQwen2MLP(original_mlp, self.config, i)
            
            # Patch Attention Q and O projections
            layer.self_attn.q_proj = LODLinearWrapper(layer.self_attn.q_proj, self.config, f"L{i}.q")
            layer.self_attn.o_proj = LODLinearWrapper(layer.self_attn.o_proj, self.config, f"L{i}.o")
            
            if i % 4 == 0:
                logger.info(f"  Layer {i} patched")
            
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info(f"Patched {end - start} layers!")
        logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    
    def set_lod(self, lod: str):
        """Set LOD for all patched modules."""
        PatchedQwen2MLP.set_lod(lod)
        LODLinearWrapper.set_lod(lod)
        self.stats[f'lod_{lod}'] += 1
    
    def select_lod_from_confidence(self, confidence: float) -> str:
        """
        Select LOD based on zeta barrier structure.
        
        The light cone at n=80 maps to confidence = 1/φ ≈ 0.618
        - Above barrier: quantum regime, stable → low LOD
        - At barrier: transition zone → barrier LOD
        - Below barrier: classical regime, unstable → high LOD
        """
        if confidence > self.config.barrier_high:
            # Post-horizon: quantum regime
            return 'low'
        elif confidence > self.config.barrier_low:
            # At the light cone
            return 'barrier'
        else:
            # Pre-horizon: classical regime
            return 'high'
    
    def _build_prompt(self, messages: List[Message]) -> str:
        prompt_parts = []
        prompt_parts.append("<|im_start|>system\nYou are a helpful AI assistant. Be concise.<|im_end|>")
        
        for msg in messages:
            content = msg.get_text_content()
            if msg.role == "system":
                continue
            elif msg.role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif msg.role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
    
    @torch.no_grad()
    def generate(self, messages: List[Message], max_tokens: int = 100,
                 temperature: float = 0.7) -> Tuple[str, Dict]:
        start_time = time.perf_counter()
        
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        
        # Start at barrier LOD
        self.set_lod('barrier')
        
        generated_tokens = []
        past_key_values = None
        lod_decisions = []
        prev_lod = 'barrier'
        
        for step in range(max_tokens):
            if past_key_values is None:
                curr_input_ids = input_ids
                curr_attention_mask = attention_mask
            else:
                curr_input_ids = generated_tokens[-1].unsqueeze(0).unsqueeze(0)
                curr_attention_mask = torch.ones(
                    1, attention_mask.shape[1] + len(generated_tokens),
                    device='cuda', dtype=attention_mask.dtype
                )
            
            outputs = self.model(
                input_ids=curr_input_ids,
                attention_mask=curr_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Get confidence and select LOD for next token
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            confidence = probs.max().item()
            
            # Zeta barrier LOD selection
            lod = self.select_lod_from_confidence(confidence)
            lod_decisions.append((lod, confidence))
            
            # Track barrier crossings
            if lod != prev_lod:
                self.stats['barrier_crossings'] += 1
            prev_lod = lod
            
            self.set_lod(lod)
            
            # Sample token
            if temperature > 0.1:
                next_token = torch.multinomial(probs, num_samples=1).squeeze()
            else:
                next_token = logits.argmax(dim=-1).squeeze()
            
            generated_tokens.append(next_token)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode
        if generated_tokens:
            output_ids = torch.stack(generated_tokens)
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            text = ""
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        n_tokens = len(generated_tokens)
        
        # Update stats
        self.stats['total_requests'] += 1
        self.stats['total_tokens'] += n_tokens
        self.stats['total_time_ms'] += elapsed_ms
        
        # Compute LOD breakdown
        if lod_decisions:
            low_pct = sum(1 for l, _ in lod_decisions if l == 'low') / len(lod_decisions) * 100
            barrier_pct = sum(1 for l, _ in lod_decisions if l == 'barrier') / len(lod_decisions) * 100
            high_pct = sum(1 for l, _ in lod_decisions if l == 'high') / len(lod_decisions) * 100
            
            # Speedup estimate based on k values
            # k_low=300: ~12x, k_barrier=750: ~4.8x, k_high=2000: ~1.8x
            if low_pct + barrier_pct + high_pct > 0:
                speedup = 1 / ((low_pct/100)/12 + (barrier_pct/100)/4.8 + (high_pct/100)/1.8)
            else:
                speedup = 1
            
            avg_confidence = sum(c for _, c in lod_decisions) / len(lod_decisions)
        else:
            low_pct = barrier_pct = high_pct = speedup = avg_confidence = 0
        
        current_tps = n_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        
        stats = {
            'tokens': n_tokens,
            'time_ms': elapsed_ms,
            'tokens_per_sec': current_tps,
            'avg_confidence': avg_confidence,
            'lod_breakdown': {
                'low (post-horizon)': f'{low_pct:.1f}%',
                'barrier (light cone)': f'{barrier_pct:.1f}%',
                'high (pre-horizon)': f'{high_pct:.1f}%'
            },
            'estimated_speedup': f'{speedup:.1f}x',
            'projected_tps': current_tps * speedup,
            'barrier_crossings': self.stats['barrier_crossings'],
        }
        
        logger.info(f"Generated {n_tokens} tokens in {elapsed_ms:.0f}ms ({current_tps:.1f} TPS)")
        logger.info(f"Avg confidence: {avg_confidence:.3f}, Barrier (1/φ): {BARRIER:.3f}")
        logger.info(f"LOD: low={low_pct:.0f}%, barrier={barrier_pct:.0f}%, high={high_pct:.0f}%")
        logger.info(f"Projected: {stats['projected_tps']:.0f} TPS ({speedup:.1f}x)")
        
        return text, stats
    
    async def generate_stream(self, messages: List[Message], max_tokens: int = 100,
                              temperature: float = 0.7):
        """Streaming generation - yields tokens as they're generated."""
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        
        self.set_lod('barrier')
        
        generated_tokens = []
        past_key_values = None
        prev_lod = 'barrier'
        
        for step in range(max_tokens):
            with torch.no_grad():
                if past_key_values is None:
                    curr_input_ids = input_ids
                    curr_attention_mask = attention_mask
                else:
                    curr_input_ids = generated_tokens[-1].unsqueeze(0).unsqueeze(0)
                    curr_attention_mask = torch.ones(
                        1, attention_mask.shape[1] + len(generated_tokens),
                        device='cuda', dtype=attention_mask.dtype
                    )
                
                outputs = self.model(
                    input_ids=curr_input_ids,
                    attention_mask=curr_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
                confidence = probs.max().item()
                
                lod = self.select_lod_from_confidence(confidence)
                if lod != prev_lod:
                    self.stats['barrier_crossings'] += 1
                prev_lod = lod
                self.set_lod(lod)
                
                if temperature > 0.1:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze()
                else:
                    next_token = logits.argmax(dim=-1).squeeze()
                
                generated_tokens.append(next_token)
                
                # Decode just this token
                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                yield token_text
        
        self.stats['total_tokens'] += len(generated_tokens)
    
    def get_stats(self) -> Dict:
        total_lod = self.stats['lod_low'] + self.stats['lod_barrier'] + self.stats['lod_high']
        avg_tps = self.stats['total_tokens'] / (self.stats['total_time_ms'] / 1000) if self.stats['total_time_ms'] > 0 else 0
        
        return {
            'model': self.model_name,
            'method': 'Zeta Barrier LOD',
            'barrier_threshold': f'1/φ = {BARRIER:.4f}',
            'fine_structure_ratio': f'137/30 = {FINE_STRUCTURE_RATIO:.3f}',
            'total_requests': self.stats['total_requests'],
            'total_tokens': self.stats['total_tokens'],
            'avg_tokens_per_sec': avg_tps,
            'barrier_crossings': self.stats['barrier_crossings'],
            'lod_config': {
                'k_low': self.config.k_low,
                'k_barrier': self.config.k_barrier,
                'k_high': self.config.k_high,
            },
        }


engine: Optional[ZetaBarrierEngine] = None

app = FastAPI(
    title="Zeta Barrier LOD Server",
    description="LOD selection based on zeta barrier structure (1/φ threshold, 137/30 scaling)",
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
    global engine
    engine = ZetaBarrierEngine()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "zeta-barrier-qwen2", "barrier": f"1/φ = {BARRIER:.4f}"}


@app.get("/stats")
async def get_stats():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.get_stats()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "zeta-barrier-qwen2", "object": "model", "created": int(time.time()), "owned_by": "truthspace"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            stream_generate(request),
            media_type="text/event-stream"
        )
    
    try:
        response_text, stats = engine.generate(
            request.messages,
            max_tokens=request.max_tokens or 100,
            temperature=request.temperature or 0.7,
        )
        
        return ChatCompletionResponse(
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
                completion_tokens=stats["tokens"],
                total_tokens=sum(len(m.get_text_content().split()) for m in request.messages) + stats["tokens"],
            ),
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_generate(request: ChatCompletionRequest):
    """Stream tokens as SSE events."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    async for token_text in engine.generate_stream(
        request.messages,
        max_tokens=request.max_tokens or 100,
        temperature=request.temperature or 0.7,
    ):
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": token_text},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)  # Allow other tasks to run
    
    # Final chunk with finish_reason
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def main():
    parser = argparse.ArgumentParser(description="Zeta Barrier LOD Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8006)
    args = parser.parse_args()
    
    import uvicorn
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Zeta Barrier LOD Server                            ║
║                                                              ║
║  Based on fine structure in zeta zeros:                      ║
║    - Light cone barrier at n=80                              ║
║    - Ratio of slopes = 137/30 ≈ 1/α                          ║
║    - Maps to confidence threshold at 1/φ ≈ 0.618             ║
║                                                              ║
║  LOD Selection:                                              ║
║    conf > 1/φ (0.618):  LOW LOD (k={ZetaBarrierConfig.k_low})  post-horizon    ║
║    conf ~ 1/φ:          BARRIER LOD (k={ZetaBarrierConfig.k_barrier})           ║
║    conf < 1/φ² (0.382): HIGH LOD (k={ZetaBarrierConfig.k_high}) pre-horizon    ║
║                                                              ║
║  Server: http://localhost:{args.port}                           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
