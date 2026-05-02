#!/usr/bin/env python3
"""
Automated Dimension Discovery and Scaling
==========================================

Goal: Automate dimension discovery to achieve 100% accuracy.

Approach:
1. Use the model to discover semantic dimensions
2. For each dimension, get MANY word pairs (not just 2-3)
3. Build a clean axis from multiple pairs
4. Map words to their dimensions
5. Navigate using the correct dimension

The key: More pairs per dimension = cleaner axis = higher accuracy
"""

import torch
import torch.nn.functional as F
import math
import re
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/src')

from phi_navigator.coordinates import PhiCoordinates

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class Dimension:
    """A semantic dimension with its axis and word mappings."""
    name: str
    pairs: List[Tuple[str, str]]
    axis_vector: List[float]
    positive_words: List[str]  # Words on the positive end
    negative_words: List[str]  # Words on the negative end
    accuracy: float = 0.0


class AutoDimensionNavigator:
    """
    Automatically discover dimensions and navigate with high accuracy.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.all_embeds = model.model.embed_tokens.weight.detach()
        self.hidden_dim = self.all_embeds.shape[1]
        
        self.dimensions: Dict[str, Dimension] = {}
        self.word_to_dim: Dict[str, Tuple[str, str]] = {}  # word -> (dim_name, polarity)
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.all_embeds[ids[0]]
    
    def find_nearest(self, embed: torch.Tensor, top_k: int = 5,
                     exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        sims = F.cosine_similarity(embed.unsqueeze(0).float().to(self.device),
                                   self.all_embeds.float())
        if exclude:
            for word in exclude:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if ids:
                    sims[ids[0]] = -1
        top_indices = sims.topk(top_k).indices
        return [(self.tokenizer.decode([idx.item()]).strip(), sims[idx].item())
                for idx in top_indices]
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text from the model."""
        messages = [{"role": "user", "content": prompt}]
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
    
    def parse_pairs(self, response: str) -> List[Tuple[str, str]]:
        """Parse word pairs from model response."""
        pairs = []
        for line in response.strip().split('\n'):
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    w1 = re.sub(r'[^a-zA-Z]', '', parts[0]).strip().lower()
                    w2 = re.sub(r'[^a-zA-Z]', '', parts[1]).strip().lower()
                    if w1 and w2 and w1 != w2 and len(w1) > 1 and len(w2) > 1:
                        # Verify both words exist in vocabulary
                        if self.get_embedding(w1) is not None and self.get_embedding(w2) is not None:
                            pairs.append((w1, w2))
        return pairs
    
    # =========================================================================
    # AUTOMATED DIMENSION DISCOVERY
    # =========================================================================
    
    def discover_dimension(self, name: str, description: str, 
                           n_pairs: int = 30) -> Optional[Dimension]:
        """
        Discover a dimension by asking the model for many pairs.
        
        More pairs = cleaner axis = higher accuracy
        """
        print(f"  Discovering '{name}'...")
        
        prompt = f"""List {n_pairs} pairs of single English words that are {description}.
Use common, simple words. One pair per line, format: word1, word2
Examples of the format:
hot, cold
big, small"""
        
        response = self.generate(prompt)
        pairs = self.parse_pairs(response)
        
        if len(pairs) < 3:
            print(f"    Only got {len(pairs)} pairs, skipping")
            return None
        
        print(f"    Got {len(pairs)} valid pairs")
        
        # Compute axis from pairs
        directions = []
        positive_words = []
        negative_words = []
        
        for w1, w2 in pairs:
            e1 = self.get_embedding(w1)
            e2 = self.get_embedding(w2)
            
            if e1 is None or e2 is None:
                continue
            
            direction = (e2 - e1).float().cpu()
            direction = direction / direction.norm()
            directions.append(direction)
            
            negative_words.append(w1)
            positive_words.append(w2)
            
            # Map words to dimension
            self.word_to_dim[w1] = (name, "negative")
            self.word_to_dim[w2] = (name, "positive")
        
        if not directions:
            return None
        
        # Average direction (more pairs = more stable)
        axis_vector = torch.stack(directions).mean(dim=0)
        axis_vector = axis_vector / axis_vector.norm()
        
        dim = Dimension(
            name=name,
            pairs=pairs,
            axis_vector=axis_vector.tolist(),
            positive_words=positive_words,
            negative_words=negative_words,
        )
        
        self.dimensions[name] = dim
        return dim
    
    def discover_all_dimensions(self) -> Dict[str, Dimension]:
        """Discover all common semantic dimensions."""
        
        dimension_specs = [
            ("temperature", "temperature opposites like hot/cold, warm/cool"),
            ("size", "size opposites like big/small, large/tiny, huge/little"),
            ("speed", "speed opposites like fast/slow, quick/sluggish"),
            ("height", "height opposites like tall/short, high/low"),
            ("brightness", "brightness opposites like bright/dark, light/dim"),
            ("age", "age opposites like young/old, new/ancient"),
            ("wealth", "wealth opposites like rich/poor, wealthy/broke"),
            ("volume", "volume/loudness opposites like loud/quiet, noisy/silent"),
            ("thickness", "thickness opposites like thick/thin, fat/skinny"),
            ("depth", "depth opposites like deep/shallow"),
            ("hardness", "hardness opposites like hard/soft"),
            ("moisture", "moisture opposites like wet/dry, damp/arid"),
            ("valence", "positive/negative sentiment opposites like good/bad, happy/sad"),
            ("length", "length opposites like long/short"),
            ("width", "width opposites like wide/narrow, broad/thin"),
            ("weight", "weight opposites like heavy/light"),
            ("strength", "strength opposites like strong/weak"),
            ("cleanliness", "cleanliness opposites like clean/dirty"),
            ("fullness", "fullness opposites like full/empty"),
            ("openness", "openness opposites like open/closed"),
        ]
        
        print("="*70)
        print("AUTOMATED DIMENSION DISCOVERY")
        print("="*70)
        
        for name, description in dimension_specs:
            self.discover_dimension(name, description, n_pairs=30)
        
        return self.dimensions
    
    # =========================================================================
    # NAVIGATION
    # =========================================================================
    
    def navigate(self, word: str, scale: float = 2.0) -> Optional[Tuple[str, float, str]]:
        """
        Navigate to the opposite of a word.
        
        1. Find which dimension the word is on
        2. Move to the opposite end
        3. Return the nearest word
        """
        # Check if word is in a known dimension
        if word not in self.word_to_dim:
            return None
        
        dim_name, polarity = self.word_to_dim[word]
        dim = self.dimensions[dim_name]
        
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        axis = torch.tensor(dim.axis_vector, device=self.device)
        
        # Move to opposite end
        # If word is positive, move negative (subtract axis)
        # If word is negative, move positive (add axis)
        if polarity == "positive":
            new_embed = embed.float() - scale * axis
        else:
            new_embed = embed.float() + scale * axis
        
        nearest = self.find_nearest(new_embed, top_k=5, exclude=[word])
        
        if nearest:
            return (nearest[0][0], nearest[0][1], dim_name)
        return None
    
    def validate(self) -> float:
        """Validate navigation accuracy on all known pairs."""
        correct = 0
        total = 0
        
        for dim in self.dimensions.values():
            for w1, w2 in dim.pairs:
                # Test w1 -> w2
                result = self.navigate(w1)
                if result and w2.lower() in result[0].lower():
                    correct += 1
                total += 1
                
                # Test w2 -> w1
                result = self.navigate(w2)
                if result and w1.lower() in result[0].lower():
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def save(self, path: str):
        """Save dimensions to file."""
        data = {
            name: asdict(dim) for name, dim in self.dimensions.items()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.dimensions)} dimensions to {path}")
    
    def load(self, path: str):
        """Load dimensions from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for name, dim_data in data.items():
            dim = Dimension(**dim_data)
            self.dimensions[name] = dim
            
            for w in dim.negative_words:
                self.word_to_dim[w] = (name, "negative")
            for w in dim.positive_words:
                self.word_to_dim[w] = (name, "positive")
        
        print(f"Loaded {len(self.dimensions)} dimensions from {path}")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    nav = AutoDimensionNavigator(model, tokenizer)
    
    # Discover all dimensions
    nav.discover_all_dimensions()
    
    # Validate
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    accuracy = nav.validate()
    print(f"\nOverall accuracy: {accuracy*100:.1f}%")
    
    # Show per-dimension stats
    print("\nPer-dimension accuracy:")
    for name, dim in nav.dimensions.items():
        correct = 0
        total = 0
        for w1, w2 in dim.pairs:
            result = nav.navigate(w1)
            if result and w2.lower() in result[0].lower():
                correct += 1
            total += 1
        dim_acc = correct / total if total > 0 else 0
        print(f"  {name:15s}: {correct:2d}/{total:2d} ({dim_acc*100:.0f}%)")
    
    # Save for future use
    nav.save("/home/thorin/truthspace-lcm/src/phi_navigator/dimensions.json")
    
    # Test some examples
    print("\n" + "="*70)
    print("EXAMPLE NAVIGATIONS")
    print("="*70)
    
    test_words = ["hot", "big", "fast", "tall", "bright", "young", "rich", 
                  "loud", "thick", "deep", "hard", "wet", "good", "happy"]
    
    for word in test_words:
        result = nav.navigate(word)
        if result:
            got, conf, dim = result
            print(f"  {word:10s} --[{dim:12s}]--> {got}")
        else:
            print(f"  {word:10s} --> [not in any dimension]")


if __name__ == "__main__":
    main()
