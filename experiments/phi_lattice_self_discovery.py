#!/usr/bin/env python3
"""
φ-Lattice Self-Discovery
========================

The model discovers its own semantic axes.

Process:
1. ASK: Model generates word pairs for a relationship
2. DISCOVER: Find sign dimensions that encode the relationship
3. VALIDATE: Test the axis on new words
4. NAME: Model describes what the axis represents
5. EXPAND: Model generates more examples, refine axis

The model becomes its own cartographer.
"""

import torch
import torch.nn.functional as F
import math
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
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
    """A semantic axis discovered by the model."""
    name: str
    description: str
    dimensions: torch.Tensor
    examples: List[Tuple[str, str]]
    validation_accuracy: float = 0.0


class SelfDiscoveryEngine:
    """Engine for model self-discovery of semantic axes."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.hidden_dim = model.config.hidden_size
        self.discovered_axes: Dict[str, DiscoveredAxis] = {}
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
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
        # Extract just the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        """Get embedding for a word."""
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        return self.model.model.embed_tokens.weight[ids[0]].detach()
    
    def find_nearest(self, embed: torch.Tensor, top_k: int = 5,
                     exclude_ids: Optional[List[int]] = None) -> List[Tuple[str, float]]:
        """Find nearest tokens to an embedding."""
        all_embeds = self.model.model.embed_tokens.weight.detach()
        sims = F.cosine_similarity(embed.unsqueeze(0).float().to(self.device),
                                   all_embeds.float())
        
        if exclude_ids:
            for idx in exclude_ids:
                sims[idx] = -1
        
        top_indices = sims.topk(top_k).indices
        results = []
        for idx in top_indices:
            token = self.tokenizer.decode([idx.item()]).strip()
            sim = sims[idx].item()
            results.append((token, sim))
        return results
    
    # =========================================================================
    # STEP 1: ASK - Get word pairs from the model
    # =========================================================================
    
    def ask_for_pairs(self, relationship: str, n_pairs: int = 10) -> List[Tuple[str, str]]:
        """Ask the model to generate word pairs for a relationship."""
        prompt = f"""List {n_pairs} pairs of single English words that are {relationship}.
Format each pair as: word1, word2
Only output the pairs, one per line. Use simple, common words."""
        
        response = self.generate(prompt)
        
        # Parse pairs from response
        pairs = []
        for line in response.strip().split('\n'):
            # Try to extract two words
            line = line.strip()
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    w1 = re.sub(r'[^a-zA-Z]', '', parts[0]).strip().lower()
                    w2 = re.sub(r'[^a-zA-Z]', '', parts[1]).strip().lower()
                    if w1 and w2 and w1 != w2:
                        pairs.append((w1, w2))
        
        return pairs[:n_pairs]
    
    # =========================================================================
    # STEP 2: DISCOVER - Find the axis dimensions
    # =========================================================================
    
    def discover_axis(self, pairs: List[Tuple[str, str]], 
                      threshold: float = 0.5) -> torch.Tensor:
        """Discover which dimensions encode a relationship."""
        flip_counts = torch.zeros(self.hidden_dim)
        valid_pairs = 0
        
        for word1, word2 in pairs:
            embed1 = self.get_embedding(word1)
            embed2 = self.get_embedding(word2)
            
            if embed1 is None or embed2 is None:
                continue
            
            _, signs1 = encode_phi(embed1)
            _, signs2 = encode_phi(embed2)
            
            flip_counts += (signs1 != signs2).float()
            valid_pairs += 1
        
        if valid_pairs == 0:
            return torch.tensor([])
        
        flip_rate = flip_counts / valid_pairs
        return (flip_rate > threshold).nonzero().squeeze()
    
    # =========================================================================
    # STEP 3: VALIDATE - Test the axis
    # =========================================================================
    
    def validate_axis(self, axis: torch.Tensor, 
                      test_pairs: List[Tuple[str, str]]) -> float:
        """Validate an axis by testing navigation accuracy."""
        if len(axis) == 0:
            return 0.0
        
        correct = 0
        total = 0
        
        for word1, word2 in test_pairs:
            embed1 = self.get_embedding(word1)
            if embed1 is None:
                continue
            
            levels, signs = encode_phi(embed1)
            signs[axis] *= -1
            navigated = decode_phi(levels, signs).to(embed1.dtype).to(self.device)
            
            word1_id = self.tokenizer.encode(word1, add_special_tokens=False)[0]
            nearest = self.find_nearest(navigated, top_k=5, exclude_ids=[word1_id])
            
            # Check if word2 is in top 5
            if any(word2.lower() in t.lower() for t, _ in nearest):
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    # =========================================================================
    # STEP 4: NAME - Ask model to describe the axis
    # =========================================================================
    
    def name_axis(self, pairs: List[Tuple[str, str]]) -> str:
        """Ask the model to name/describe the relationship."""
        pairs_str = ", ".join(f"({a}, {b})" for a, b in pairs[:5])
        prompt = f"""These word pairs share a common relationship: {pairs_str}

In 2-3 words, what semantic dimension or relationship do these pairs represent?
Just give the name, nothing else."""
        
        response = self.generate(prompt, max_tokens=20)
        # Clean up response
        name = response.strip().split('\n')[0].strip()
        name = re.sub(r'[^a-zA-Z\s]', '', name).strip().lower()
        return name[:30] if name else "unknown"
    
    # =========================================================================
    # STEP 5: EXPAND - Get more examples to refine
    # =========================================================================
    
    def expand_axis(self, axis_name: str, existing_pairs: List[Tuple[str, str]],
                    n_more: int = 5) -> List[Tuple[str, str]]:
        """Ask for more examples to refine the axis."""
        pairs_str = ", ".join(f"({a}, {b})" for a, b in existing_pairs[:3])
        prompt = f"""Given these example pairs that represent {axis_name}: {pairs_str}

List {n_more} more pairs that follow the same pattern.
Format: word1, word2
Use simple, common words not already listed."""
        
        response = self.generate(prompt)
        
        # Parse new pairs
        new_pairs = []
        existing_words = set(w for pair in existing_pairs for w in pair)
        
        for line in response.strip().split('\n'):
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    w1 = re.sub(r'[^a-zA-Z]', '', parts[0]).strip().lower()
                    w2 = re.sub(r'[^a-zA-Z]', '', parts[1]).strip().lower()
                    if w1 and w2 and w1 != w2 and w1 not in existing_words:
                        new_pairs.append((w1, w2))
        
        return new_pairs[:n_more]
    
    # =========================================================================
    # MAIN: Full self-discovery loop
    # =========================================================================
    
    def discover_relationship(self, relationship: str) -> Optional[DiscoveredAxis]:
        """Full self-discovery loop for a relationship."""
        print(f"\n{'='*60}")
        print(f"DISCOVERING: {relationship}")
        print('='*60)
        
        # Step 1: Ask for pairs
        print("\n1. ASKING for word pairs...")
        pairs = self.ask_for_pairs(relationship, n_pairs=10)
        print(f"   Got {len(pairs)} pairs: {pairs[:5]}...")
        
        if len(pairs) < 3:
            print("   Not enough pairs, aborting.")
            return None
        
        # Step 2: Discover axis
        print("\n2. DISCOVERING axis dimensions...")
        axis = self.discover_axis(pairs)
        print(f"   Found {len(axis)} dimensions")
        
        if len(axis) == 0:
            print("   No axis found, aborting.")
            return None
        
        # Step 3: Validate
        print("\n3. VALIDATING axis...")
        accuracy = self.validate_axis(axis, pairs)
        print(f"   Accuracy on training pairs: {accuracy*100:.1f}%")
        
        # Step 4: Name
        print("\n4. NAMING the axis...")
        name = self.name_axis(pairs)
        print(f"   Name: '{name}'")
        
        # Step 5: Expand and refine
        print("\n5. EXPANDING with more examples...")
        more_pairs = self.expand_axis(name, pairs, n_more=5)
        print(f"   Got {len(more_pairs)} more pairs: {more_pairs}")
        
        # Re-discover with expanded set
        all_pairs = pairs + more_pairs
        refined_axis = self.discover_axis(all_pairs)
        print(f"   Refined axis: {len(refined_axis)} dimensions")
        
        # Final validation on new pairs
        if more_pairs:
            val_accuracy = self.validate_axis(refined_axis, more_pairs)
            print(f"   Validation accuracy (new pairs): {val_accuracy*100:.1f}%")
        else:
            val_accuracy = accuracy
        
        # Create discovered axis
        discovered = DiscoveredAxis(
            name=name,
            description=relationship,
            dimensions=refined_axis,
            examples=all_pairs,
            validation_accuracy=val_accuracy,
        )
        
        self.discovered_axes[name] = discovered
        return discovered
    
    def discover_all_relationships(self) -> Dict[str, DiscoveredAxis]:
        """Discover multiple semantic axes."""
        relationships = [
            "opposites (antonyms)",
            "male and female versions of the same concept",
            "singular and plural forms",
            "present and past tense verbs",
            "positive and negative sentiment",
            "concrete and abstract versions",
            "formal and informal versions",
            "big and small versions",
            "fast and slow versions",
            "old and young versions",
        ]
        
        for rel in relationships:
            try:
                self.discover_relationship(rel)
            except Exception as e:
                print(f"Error discovering '{rel}': {e}")
        
        return self.discovered_axes
    
    def navigate(self, word: str, axis_name: str) -> List[Tuple[str, float]]:
        """Navigate using a discovered axis."""
        if axis_name not in self.discovered_axes:
            return []
        
        axis = self.discovered_axes[axis_name]
        embed = self.get_embedding(word)
        if embed is None:
            return []
        
        levels, signs = encode_phi(embed)
        signs[axis.dimensions] *= -1
        navigated = decode_phi(levels, signs).to(embed.dtype).to(self.device)
        
        word_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
        return self.find_nearest(navigated, exclude_ids=[word_id])
    
    def report(self):
        """Print a report of discovered axes."""
        print("\n" + "="*70)
        print("SELF-DISCOVERY REPORT")
        print("="*70)
        
        for name, axis in self.discovered_axes.items():
            print(f"\n{name.upper()}")
            print(f"  Description: {axis.description}")
            print(f"  Dimensions: {len(axis.dimensions)}")
            print(f"  Validation accuracy: {axis.validation_accuracy*100:.1f}%")
            print(f"  Examples: {axis.examples[:3]}")


def main():
    print("="*70)
    print("φ-LATTICE SELF-DISCOVERY")
    print("="*70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Create self-discovery engine
    engine = SelfDiscoveryEngine(model, tokenizer)
    
    # Discover a few key relationships
    relationships = [
        "opposites (antonyms)",
        "male and female versions",
        "positive and negative sentiment",
    ]
    
    for rel in relationships:
        engine.discover_relationship(rel)
    
    # Report
    engine.report()
    
    # Test navigation with discovered axes
    print("\n" + "="*70)
    print("TESTING DISCOVERED AXES")
    print("="*70)
    
    for axis_name in engine.discovered_axes:
        print(f"\n--- {axis_name} ---")
        test_words = ["happy", "big", "king", "fast", "good"]
        for word in test_words:
            results = engine.navigate(word, axis_name)
            if results:
                print(f"  {word} → {results[0][0]} ({results[0][1]:.3f})")


if __name__ == "__main__":
    main()
