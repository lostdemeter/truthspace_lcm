#!/usr/bin/env python3
"""
φ-Lattice Cached Prompts: Pre-computed Prompt Paths
====================================================

The idea: System prompts define a "starting position" in model state space.
We can pre-compute that position and cache it, then only process the query.

Traditional:
  [System: 100 tokens] + [Query: 10 tokens] → 110 tokens processed

Optimized:
  [System: 100 tokens] → Pre-compute ONCE → Cache KV state
  [Query: 10 tokens] → Inject cached state → Only 10 tokens processed

This gives us:
1. Faster inference (skip system prompt processing)
2. Consistent behavior (same cached state every time)
3. Memory efficiency (store compressed state, not full prompt)
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import json

@dataclass
class CachedPromptState:
    """Cached state from a pre-processed system prompt."""
    name: str
    prompt: str
    n_tokens: int
    past_key_values: Tuple  # The KV cache
    last_hidden_state: torch.Tensor  # Final hidden state


class CachedPromptEngine:
    """Engine for cached prompt inference."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.cached_prompts: Dict[str, CachedPromptState] = {}
    
    def cache_prompt(self, name: str, system_prompt: str) -> CachedPromptState:
        """
        Pre-compute and cache a system prompt's state.
        
        This runs the system prompt through the model ONCE and saves:
        1. The KV cache (past_key_values)
        2. The final hidden state
        
        Future queries can start from this cached state.
        """
        print(f"Caching prompt '{name}'...")
        
        # Format as chat
        messages = [{"role": "system", "content": system_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        n_tokens = inputs['input_ids'].shape[1]
        
        # Run through model to get KV cache
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Extract what we need
        past_key_values = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # Last token's hidden state
        
        # Create cached state
        cached = CachedPromptState(
            name=name,
            prompt=system_prompt,
            n_tokens=n_tokens,
            past_key_values=past_key_values,
            last_hidden_state=last_hidden,
        )
        
        self.cached_prompts[name] = cached
        print(f"  Cached {n_tokens} tokens")
        
        return cached
    
    def generate_with_cache(self, cache_name: str, user_query: str, 
                            max_new_tokens: int = 100) -> Tuple[str, float]:
        """
        Generate using a cached prompt state.
        
        Only processes the user query - the system prompt is already cached.
        """
        if cache_name not in self.cached_prompts:
            raise ValueError(f"No cached prompt named '{cache_name}'")
        
        cached = self.cached_prompts[cache_name]
        
        # Format user query - prepend the cached system tokens for position tracking
        user_text = f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
        user_inputs = self.tokenizer(user_text, return_tensors="pt").to(self.device)
        user_len = user_inputs['input_ids'].shape[1]
        
        # Create full input_ids (system + user) but we'll use cache for system part
        # We need to create position_ids and cache_position for the new API
        total_len = cached.n_tokens + user_len
        
        start_time = time.time()
        
        with torch.no_grad():
            # First, run user tokens through model with cached KV
            user_outputs = self.model(
                input_ids=user_inputs['input_ids'],
                attention_mask=torch.ones(1, total_len, device=self.device),
                past_key_values=cached.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Now generate from this state
            combined_past = user_outputs.past_key_values
            
            # Get the last token logits and start generating
            next_token_logits = user_outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated_tokens = [next_token]
            current_past = combined_past
            
            for _ in range(max_new_tokens - 1):
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=current_past,
                    use_cache=True,
                    return_dict=True,
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token)
                current_past = outputs.past_key_values
        
        elapsed = time.time() - start_time
        
        # Decode response
        all_tokens = torch.cat(generated_tokens, dim=1)
        response = self.tokenizer.decode(all_tokens[0], skip_special_tokens=True)
        
        return response, elapsed
    
    def generate_without_cache(self, system_prompt: str, user_query: str,
                               max_new_tokens: int = 100) -> Tuple[str, float]:
        """
        Generate WITHOUT cache (baseline for comparison).
        
        Processes the full system prompt + user query every time.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        elapsed = time.time() - start_time
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        return response, elapsed


def demo_bash_assistant():
    """Demo: Pre-cached bash command assistant."""
    
    print("="*70)
    print("CACHED PROMPT DEMO: BASH ASSISTANT")
    print("="*70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Create engine
    engine = CachedPromptEngine(model, tokenizer)
    
    # Define system prompts to cache
    prompts = {
        "bash": """You are a helpful assistant that is a programmer and you write bash commands like a professional. Reply only with the bash command related to the user's query. No explanations, just the command.""",
        
        "python": """You are a Python expert. Reply only with Python code. No explanations, just the code.""",
        
        "json": """You are a JSON formatter. Convert the user's input to valid JSON. Reply only with the JSON, nothing else.""",
        
        "translator": """You are a translator. Translate the user's text to French. Reply only with the translation.""",
    }
    
    # Cache all prompts
    print("\n--- CACHING PROMPTS ---")
    for name, prompt in prompts.items():
        engine.cache_prompt(name, prompt)
    
    # Test queries
    test_queries = [
        ("bash", "list all files in current directory"),
        ("bash", "find all python files"),
        ("bash", "count lines in a file"),
        ("python", "read a CSV file"),
        ("json", "name: John, age: 30, city: NYC"),
        ("translator", "Hello, how are you?"),
    ]
    
    print("\n--- TESTING CACHED GENERATION ---")
    print("-"*70)
    
    cached_times = []
    uncached_times = []
    
    for cache_name, query in test_queries:
        # With cache
        response_cached, time_cached = engine.generate_with_cache(
            cache_name, query, max_new_tokens=50
        )
        cached_times.append(time_cached)
        
        # Without cache (baseline)
        response_uncached, time_uncached = engine.generate_without_cache(
            prompts[cache_name], query, max_new_tokens=50
        )
        uncached_times.append(time_uncached)
        
        speedup = time_uncached / time_cached if time_cached > 0 else 0
        
        print(f"\n[{cache_name}] {query}")
        print(f"  Cached:   {response_cached[:60]}...")
        print(f"  Time:     {time_cached:.3f}s (cached) vs {time_uncached:.3f}s (uncached)")
        print(f"  Speedup:  {speedup:.2f}x")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    avg_cached = sum(cached_times) / len(cached_times)
    avg_uncached = sum(uncached_times) / len(uncached_times)
    avg_speedup = avg_uncached / avg_cached if avg_cached > 0 else 0
    
    print(f"\nAverage cached time:   {avg_cached:.3f}s")
    print(f"Average uncached time: {avg_uncached:.3f}s")
    print(f"Average speedup:       {avg_speedup:.2f}x")
    
    # Memory analysis
    print("\n--- MEMORY ANALYSIS ---")
    for name, cached in engine.cached_prompts.items():
        kv_size = sum(
            sum(t.numel() * t.element_size() for t in layer)
            for layer in cached.past_key_values
        )
        print(f"  {name}: {cached.n_tokens} tokens, KV cache = {kv_size/1e6:.1f} MB")


def demo_hardwired_paths():
    """Demo: Hardwired paths using φ-lattice navigation."""
    
    print("\n" + "="*70)
    print("HARDWIRED PATHS: φ-LATTICE + CACHED PROMPTS")
    print("="*70)
    
    print("""
The full pipeline:

1. CACHE: Pre-compute system prompt → KV cache
2. NAVIGATE: Use φ-lattice to find answer location
3. GENERATE: Only if navigation fails

For simple queries:
  "list files" → navigate(bash_context, "list") → "ls"
  
For complex queries:
  "find all python files modified today" → generate with cached prompt

This gives us:
  - O(1) for simple queries (navigation lookup)
  - O(query_length) for complex queries (cached prompt)
  - Never O(system_prompt + query) (traditional)
""")


if __name__ == "__main__":
    demo_bash_assistant()
    demo_hardwired_paths()
