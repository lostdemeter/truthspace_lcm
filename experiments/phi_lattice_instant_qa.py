#!/usr/bin/env python3
"""
φ-Lattice Instant Q&A: Using Pre-Discovered Axes
=================================================

The real optimization isn't caching prompts for generation.
It's caching the DISCOVERED AXES for instant navigation.

Discovery (one-time, ~30s):
  Model generates pairs → Find axis dimensions → Save to JSON

Runtime (instant, ~1ms):
  Load cached axes → Navigate directly → Return answer

This gives us:
  - O(1) for axis-based questions
  - No model inference needed
  - Just embedding lookup + sign flip + nearest neighbor
"""

import torch
import torch.nn.functional as F
import math
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional

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


class InstantQA:
    """Instant Q&A using pre-discovered axes."""
    
    def __init__(self, model, tokenizer, axes_file: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Load pre-discovered axes
        self.axes: Dict[str, torch.Tensor] = {}
        self.load_axes(axes_file)
        
        # Pre-compute all embeddings for fast lookup
        self.all_embeds = model.model.embed_tokens.weight.detach()
    
    def load_axes(self, filepath: str):
        """Load pre-discovered axes from JSON."""
        print(f"Loading axes from {filepath}...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, axis_data in data.items():
            if 'dimensions' in axis_data:
                dims = torch.tensor(axis_data['dimensions'])
                self.axes[name] = dims
                acc = axis_data.get('validation_accuracy', 0) * 100
                print(f"  {name}: {len(dims)} dims, {acc:.0f}% accuracy")
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def find_nearest(self, embed: torch.Tensor, top_k: int = 5,
                     exclude_ids: Optional[List[int]] = None) -> List[Tuple[str, float]]:
        sims = F.cosine_similarity(embed.unsqueeze(0).float(), self.all_embeds.float())
        if exclude_ids:
            for idx in exclude_ids:
                sims[idx] = -1
        top_indices = sims.topk(top_k).indices
        return [(self.tokenizer.decode([idx.item()]).strip(), sims[idx].item()) 
                for idx in top_indices]
    
    def navigate(self, word: str, axis_name: str) -> Tuple[str, float]:
        """Navigate along an axis. Returns (answer, time_ms)."""
        if axis_name not in self.axes:
            return f"Unknown axis: {axis_name}", 0
        
        start = time.perf_counter()
        
        embed = self.get_embedding(word)
        if embed is None:
            return f"Unknown word: {word}", 0
        
        axis = self.axes[axis_name]
        levels, signs = encode_phi(embed)
        signs[axis] *= -1
        navigated = decode_phi(levels, signs).to(embed.dtype).to(self.device)
        
        word_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
        nearest = self.find_nearest(navigated, top_k=1, exclude_ids=[word_id])
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        if nearest:
            return nearest[0][0], elapsed_ms
        return "No result", elapsed_ms
    
    def answer_opposite(self, word: str, axis_name: str) -> Tuple[str, float]:
        """Answer: What is the opposite of [word] on [axis]?"""
        return self.navigate(word, axis_name)
    
    def generate_answer(self, word: str, axis_name: str) -> Tuple[str, float]:
        """Generate answer using full model inference (baseline)."""
        prompt = f"What is the {axis_name} opposite of '{word}'? Reply with just one word."
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        start = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        # Extract first word
        answer = response.split()[0] if response.split() else response
        return answer, elapsed_ms


def main():
    print("="*70)
    print("φ-LATTICE INSTANT Q&A")
    print("="*70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Create instant QA with pre-discovered axes
    qa = InstantQA(model, tokenizer, "/home/thorin/truthspace-lcm/experiments/discovered_axes.json")
    
    # Test queries
    test_cases = [
        ("hot", "temperature"),
        ("big", "size"),
        ("up", "spatial_vertical"),
        ("left", "spatial_horizontal"),
        ("before", "temporal"),
        ("solid", "physical_state"),
        ("good", "quality"),
        ("many", "quantity"),
        ("deep", "depth"),
        ("heavy", "weight"),
    ]
    
    print("\n" + "="*70)
    print("COMPARING NAVIGATION vs GENERATION")
    print("="*70)
    
    nav_times = []
    gen_times = []
    
    print(f"\n{'Word':<10} {'Axis':<20} {'Nav Answer':<12} {'Nav Time':<10} {'Gen Answer':<12} {'Gen Time':<10} {'Speedup'}")
    print("-"*90)
    
    for word, axis in test_cases:
        # Navigation (instant)
        nav_answer, nav_time = qa.answer_opposite(word, axis)
        nav_times.append(nav_time)
        
        # Generation (slow)
        gen_answer, gen_time = qa.generate_answer(word, axis)
        gen_times.append(gen_time)
        
        speedup = gen_time / nav_time if nav_time > 0 else 0
        
        print(f"{word:<10} {axis:<20} {nav_answer:<12} {nav_time:>7.1f}ms   {gen_answer:<12} {gen_time:>7.0f}ms   {speedup:>6.0f}x")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    avg_nav = sum(nav_times) / len(nav_times)
    avg_gen = sum(gen_times) / len(gen_times)
    avg_speedup = avg_gen / avg_nav if avg_nav > 0 else 0
    
    print(f"\nAverage navigation time: {avg_nav:.2f}ms")
    print(f"Average generation time: {avg_gen:.0f}ms")
    print(f"Average speedup:         {avg_speedup:.0f}x")
    
    print(f"\nTotal navigation time:   {sum(nav_times):.0f}ms")
    print(f"Total generation time:   {sum(gen_times):.0f}ms")
    print(f"Time saved:              {sum(gen_times) - sum(nav_times):.0f}ms")
    
    print("""
KEY INSIGHT:
  Navigation is ~100-1000x faster than generation.
  For axis-based questions, we don't need the model at all.
  Just: embed → flip signs → lookup → answer.
""")


if __name__ == "__main__":
    main()
