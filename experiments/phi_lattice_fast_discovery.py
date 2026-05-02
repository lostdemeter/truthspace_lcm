#!/usr/bin/env python3
"""
φ-Lattice Fast Self-Discovery with Cached Prompts
==================================================

Combine cached prompts with self-discovery for faster axis finding.

The insight: The system prompt for generating word pairs is the SAME
for every axis. We can cache it once and reuse it.

Traditional:
  30 axes × (system_prompt + query) = 30 × full processing

Cached:
  1 × cache system_prompt
  30 axes × query_only = 30 × fast processing

Expected speedup: ~1.5-2x for the discovery process
"""

import torch
import torch.nn.functional as F
import math
import json
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K_SCALE = 128


def encode_phi(tensor):
    tensor = tensor.cpu().float()
    signs = torch.sign(tensor)
    signs[signs == 0] = 1
    magnitudes = tensor.abs().clamp(min=1e-45)
    levels = torch.round(K_SCALE * torch.log(magnitudes) / LOG_PHI)
    return levels.to(torch.int16), signs.to(torch.int8)


def decode_phi(levels, signs):
    exponents = levels.float() / K_SCALE
    magnitudes = torch.exp(exponents * LOG_PHI)
    return signs.float() * magnitudes


@dataclass
class DiscoveredAxis:
    name: str
    n_dimensions: int
    examples: List[Tuple[str, str]]
    validation_accuracy: float


class FastDiscoveryEngine:
    """Fast self-discovery using cached prompts."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.hidden_dim = model.config.hidden_size
        
        self.discovered_axes: Dict[str, DiscoveredAxis] = {}
        self.axis_dimensions: Dict[str, torch.Tensor] = {}
        
        # Cached prompt state
        self.cached_system = None
        self.cached_n_tokens = 0
    
    def cache_word_pair_prompt(self):
        """Cache the word pair generator system prompt."""
        system_prompt = """You are a helpful assistant that generates word pairs. 
When asked for pairs of words with a specific relationship, you output them in this exact format:
word1, word2

One pair per line. Use simple, common, single English words only. No explanations."""
        
        messages = [{"role": "system", "content": system_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        self.cached_n_tokens = inputs['input_ids'].shape[1]
        
        print(f"Caching word pair generator prompt ({self.cached_n_tokens} tokens)...")
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True,
            )
        
        self.cached_system = outputs.past_key_values
        print(f"  Cached!")
    
    def generate_with_cache(self, query: str, max_tokens: int = 300) -> str:
        """Generate using cached system prompt."""
        user_text = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        user_inputs = self.tokenizer(user_text, return_tensors="pt").to(self.device)
        user_len = user_inputs['input_ids'].shape[1]
        total_len = self.cached_n_tokens + user_len
        
        with torch.no_grad():
            # Run user tokens with cached KV
            user_outputs = self.model(
                input_ids=user_inputs['input_ids'],
                attention_mask=torch.ones(1, total_len, device=self.device),
                past_key_values=self.cached_system,
                use_cache=True,
                return_dict=True,
            )
            
            # Generate tokens
            current_past = user_outputs.past_key_values
            next_token = user_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [next_token]
            
            for _ in range(max_tokens - 1):
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=current_past,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated.append(next_token)
                current_past = outputs.past_key_values
        
        all_tokens = torch.cat(generated, dim=1)
        return self.tokenizer.decode(all_tokens[0], skip_special_tokens=True)
    
    def generate_without_cache(self, query: str, max_tokens: int = 300) -> str:
        """Generate without cache (baseline)."""
        system_prompt = """You are a helpful assistant that generates word pairs. 
When asked for pairs of words with a specific relationship, you output them in this exact format:
word1, word2

One pair per line. Use simple, common, single English words only. No explanations."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.model.model.embed_tokens.weight[ids[0]].detach()
    
    def find_nearest(self, embed: torch.Tensor, top_k: int = 5,
                     exclude_ids: Optional[List[int]] = None) -> List[Tuple[str, float]]:
        all_embeds = self.model.model.embed_tokens.weight.detach()
        sims = F.cosine_similarity(embed.unsqueeze(0).float().to(self.device),
                                   all_embeds.float())
        if exclude_ids:
            for idx in exclude_ids:
                sims[idx] = -1
        top_indices = sims.topk(top_k).indices
        return [(self.tokenizer.decode([idx.item()]).strip(), sims[idx].item()) 
                for idx in top_indices]
    
    def parse_pairs(self, response: str, n_pairs: int = 15) -> List[Tuple[str, str]]:
        """Parse word pairs from response."""
        pairs = []
        for line in response.strip().split('\n'):
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    w1 = re.sub(r'[^a-zA-Z]', '', parts[0]).strip().lower()
                    w2 = re.sub(r'[^a-zA-Z]', '', parts[1]).strip().lower()
                    if w1 and w2 and w1 != w2 and len(w1) > 1 and len(w2) > 1:
                        pairs.append((w1, w2))
        return pairs[:n_pairs]
    
    def discover_axis(self, pairs: List[Tuple[str, str]], 
                      threshold: float = 0.5) -> torch.Tensor:
        flip_counts = torch.zeros(self.hidden_dim)
        valid = 0
        
        for w1, w2 in pairs:
            e1, e2 = self.get_embedding(w1), self.get_embedding(w2)
            if e1 is None or e2 is None:
                continue
            _, s1 = encode_phi(e1)
            _, s2 = encode_phi(e2)
            flip_counts += (s1 != s2).float()
            valid += 1
        
        if valid == 0:
            return torch.tensor([])
        return (flip_counts / valid > threshold).nonzero().squeeze()
    
    def validate_axis(self, axis: torch.Tensor, 
                      pairs: List[Tuple[str, str]]) -> float:
        if len(axis) == 0:
            return 0.0
        
        correct = 0
        total = 0
        
        for w1, w2 in pairs:
            e1 = self.get_embedding(w1)
            if e1 is None:
                continue
            
            levels, signs = encode_phi(e1)
            signs[axis] *= -1
            navigated = decode_phi(levels, signs).to(e1.dtype).to(self.device)
            
            w1_id = self.tokenizer.encode(w1, add_special_tokens=False)[0]
            nearest = self.find_nearest(navigated, top_k=5, exclude_ids=[w1_id])
            
            if any(w2.lower() in t.lower() for t, _ in nearest):
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def discover_with_cache(self, name: str, query: str) -> Tuple[Optional[DiscoveredAxis], float]:
        """Discover an axis using cached prompt."""
        start = time.time()
        
        response = self.generate_with_cache(query)
        pairs = self.parse_pairs(response)
        
        if len(pairs) < 3:
            return None, time.time() - start
        
        axis = self.discover_axis(pairs)
        if len(axis) == 0:
            return None, time.time() - start
        
        accuracy = self.validate_axis(axis, pairs)
        
        discovered = DiscoveredAxis(
            name=name,
            n_dimensions=len(axis),
            examples=pairs,
            validation_accuracy=accuracy,
        )
        
        self.discovered_axes[name] = discovered
        self.axis_dimensions[name] = axis
        
        return discovered, time.time() - start
    
    def discover_without_cache(self, name: str, query: str) -> Tuple[Optional[DiscoveredAxis], float]:
        """Discover an axis without cache (baseline)."""
        start = time.time()
        
        response = self.generate_without_cache(query)
        pairs = self.parse_pairs(response)
        
        if len(pairs) < 3:
            return None, time.time() - start
        
        axis = self.discover_axis(pairs)
        if len(axis) == 0:
            return None, time.time() - start
        
        accuracy = self.validate_axis(axis, pairs)
        
        return DiscoveredAxis(
            name=name,
            n_dimensions=len(axis),
            examples=pairs,
            validation_accuracy=accuracy,
        ), time.time() - start


def main():
    print("="*70)
    print("FAST SELF-DISCOVERY WITH CACHED PROMPTS")
    print("="*70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    engine = FastDiscoveryEngine(model, tokenizer)
    
    # Cache the system prompt
    engine.cache_word_pair_prompt()
    
    # Define axes to discover
    axes_to_discover = [
        ("temperature", "List 15 pairs of temperature opposites like hot/cold, warm/cool."),
        ("size", "List 15 pairs of size opposites like big/small, huge/tiny."),
        ("speed", "List 15 pairs of speed opposites like fast/slow, quick/sluggish."),
        ("age", "List 15 pairs of age opposites like young/old, new/ancient."),
        ("valence", "List 15 pairs of positive/negative sentiment opposites like good/bad, happy/sad."),
        ("spatial", "List 15 pairs of spatial opposites like up/down, left/right, near/far."),
        ("temporal", "List 15 pairs of temporal opposites like before/after, early/late."),
        ("physical", "List 15 pairs of physical state opposites like solid/liquid, wet/dry."),
    ]
    
    print("\n" + "="*70)
    print("COMPARING CACHED vs UNCACHED DISCOVERY")
    print("="*70)
    
    cached_times = []
    uncached_times = []
    
    # Run cached discovery only (to avoid OOM)
    print("\n--- CACHED DISCOVERY ---")
    for name, query in axes_to_discover:
        axis, elapsed = engine.discover_with_cache(name, query)
        cached_times.append(elapsed)
        
        if axis:
            print(f"  {name:12s}: {axis.n_dimensions:4d} dims, {axis.validation_accuracy*100:3.0f}% acc, {elapsed:.2f}s")
        else:
            print(f"  {name:12s}: FAILED, {elapsed:.2f}s")
        
        # Clear some memory
        torch.cuda.empty_cache()
    
    # Clear cache and run uncached for comparison
    print("\n--- UNCACHED DISCOVERY (baseline) ---")
    engine.cached_system = None
    torch.cuda.empty_cache()
    
    for name, query in axes_to_discover:
        axis, elapsed = engine.discover_without_cache(name + "_uc", query)
        uncached_times.append(elapsed)
        
        if axis:
            print(f"  {name:12s}: {axis.n_dimensions:4d} dims, {axis.validation_accuracy*100:3.0f}% acc, {elapsed:.2f}s")
        else:
            print(f"  {name:12s}: FAILED, {elapsed:.2f}s")
        
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_cached = sum(cached_times)
    total_uncached = sum(uncached_times)
    overall_speedup = total_uncached / total_cached if total_cached > 0 else 0
    
    print(f"\nTotal cached time:   {total_cached:.2f}s")
    print(f"Total uncached time: {total_uncached:.2f}s")
    print(f"Overall speedup:     {overall_speedup:.2f}x")
    print(f"Time saved:          {total_uncached - total_cached:.2f}s")
    
    # Report discovered axes
    print("\n--- DISCOVERED AXES ---")
    for name, axis in engine.discovered_axes.items():
        print(f"  {name}: {axis.n_dimensions} dims, {axis.validation_accuracy*100:.0f}% accuracy")
        print(f"    Examples: {axis.examples[:2]}")


if __name__ == "__main__":
    main()
