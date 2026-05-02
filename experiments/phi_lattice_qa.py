#!/usr/bin/env python3
"""
φ-Lattice Question Answering via Navigation
============================================

The hypothesis: Questions are navigation problems.
The answer exists at a specific location in embedding space.
We just need to navigate there.

Question Types:
1. OPPOSITE: "What is the opposite of X?" → Flip relevant axis
2. GENDER: "What is the female/male version of X?" → Flip gender axis
3. ANALOGY: "A is to B as C is to ?" → Extract A→B axis, apply to C
4. PROPERTY: "What is a [property] [thing]?" → Apply property axis to thing
5. RELATION: "Who is X's [relation]?" → Apply relation axes
"""

import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K_SCALE = 128


def encode_phi(tensor):
    """Encode tensor to φ-lattice (levels, signs)."""
    tensor = tensor.cpu().float()
    signs = torch.sign(tensor)
    signs[signs == 0] = 1
    magnitudes = tensor.abs().clamp(min=1e-45)
    levels = torch.round(K_SCALE * torch.log(magnitudes) / LOG_PHI)
    return levels.to(torch.int16), signs.to(torch.int8)


def decode_phi(levels, signs):
    """Decode φ-lattice (levels, signs) to tensor."""
    exponents = levels.float() / K_SCALE
    magnitudes = torch.exp(exponents * LOG_PHI)
    return signs.float() * magnitudes


@dataclass
class SemanticAxis:
    """A semantic axis defined by its sign dimensions."""
    name: str
    dimensions: torch.Tensor
    examples: List[Tuple[str, str]]  # (positive, negative) pairs


class PhiLatticeNavigator:
    """Navigate embedding space using φ-lattice coordinates."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.hidden_dim = model.config.hidden_size
        
        # Pre-computed axes (will be populated)
        self.axes: Dict[str, SemanticAxis] = {}
        
        # Discover standard axes
        self._discover_standard_axes()
    
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
    
    def discover_axis(self, word_pairs: List[Tuple[str, str]], 
                      threshold: float = 0.5) -> torch.Tensor:
        """Discover which dimensions encode a semantic relationship."""
        flip_counts = torch.zeros(self.hidden_dim)
        valid_pairs = 0
        
        for word1, word2 in word_pairs:
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
    
    def _discover_standard_axes(self):
        """Discover standard semantic axes."""
        print("Discovering semantic axes...")
        
        axis_definitions = {
            "gender": [
                ("king", "queen"), ("man", "woman"), ("boy", "girl"),
                ("father", "mother"), ("brother", "sister"), ("he", "she"),
                ("him", "her"), ("his", "hers"),
            ],
            "size": [
                ("big", "small"), ("large", "tiny"), ("huge", "little"),
            ],
            "temperature": [
                ("hot", "cold"), ("warm", "cool"), ("burning", "freezing"),
            ],
            "speed": [
                ("fast", "slow"), ("quick", "sluggish"), ("rapid", "gradual"),
            ],
            "age": [
                ("young", "old"), ("new", "ancient"), ("fresh", "stale"),
            ],
            "valence": [
                ("good", "bad"), ("happy", "sad"), ("love", "hate"),
            ],
        }
        
        for name, pairs in axis_definitions.items():
            dims = self.discover_axis(pairs)
            if len(dims) > 0:
                self.axes[name] = SemanticAxis(name=name, dimensions=dims, examples=pairs)
                print(f"  {name}: {len(dims)} dimensions")
    
    def navigate(self, start_word: str, axis_name: str) -> List[Tuple[str, float]]:
        """Navigate from a word along a semantic axis."""
        embed = self.get_embedding(start_word)
        if embed is None:
            return []
        
        if axis_name not in self.axes:
            return []
        
        axis = self.axes[axis_name]
        levels, signs = encode_phi(embed)
        
        # Flip the axis dimensions
        signs[axis.dimensions] *= -1
        
        # Decode and find nearest
        navigated = decode_phi(levels, signs).to(embed.dtype).to(self.device)
        
        start_id = self.tokenizer.encode(start_word, add_special_tokens=False)[0]
        return self.find_nearest(navigated, exclude_ids=[start_id])
    
    def extract_axis(self, word1: str, word2: str) -> torch.Tensor:
        """Extract the axis between two words."""
        embed1 = self.get_embedding(word1)
        embed2 = self.get_embedding(word2)
        
        if embed1 is None or embed2 is None:
            return torch.tensor([])
        
        _, signs1 = encode_phi(embed1)
        _, signs2 = encode_phi(embed2)
        
        # Dimensions that differ = the axis
        return (signs1 != signs2).nonzero().squeeze()
    
    def apply_axis(self, word: str, axis: torch.Tensor) -> List[Tuple[str, float]]:
        """Apply an axis transformation to a word."""
        embed = self.get_embedding(word)
        if embed is None:
            return []
        
        levels, signs = encode_phi(embed)
        signs[axis] *= -1
        
        navigated = decode_phi(levels, signs).to(embed.dtype).to(self.device)
        
        word_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
        return self.find_nearest(navigated, exclude_ids=[word_id])
    
    # =========================================================================
    # QUESTION ANSWERING METHODS
    # =========================================================================
    
    def answer_opposite(self, word: str, axis_name: str = None) -> str:
        """Answer: What is the opposite of [word]?"""
        if axis_name:
            results = self.navigate(word, axis_name)
        else:
            # Try to infer the axis from the word
            best_results = []
            for name in self.axes:
                results = self.navigate(word, name)
                if results:
                    best_results.append((name, results[0]))
            
            if not best_results:
                return f"Cannot find opposite of '{word}'"
            
            # Return the one with highest similarity
            best_results.sort(key=lambda x: x[1][1], reverse=True)
            return best_results[0][1][0]
        
        if results:
            return results[0][0]
        return f"Cannot find opposite of '{word}'"
    
    def answer_gender(self, word: str) -> str:
        """Answer: What is the male/female version of [word]?"""
        results = self.navigate(word, "gender")
        if results:
            return results[0][0]
        return f"Cannot find gender counterpart of '{word}'"
    
    def answer_analogy(self, a: str, b: str, c: str) -> str:
        """Answer: A is to B as C is to ?"""
        # Extract the axis from A → B
        axis = self.extract_axis(a, b)
        
        if len(axis) == 0:
            return f"Cannot extract relationship between '{a}' and '{b}'"
        
        # Apply to C
        results = self.apply_axis(c, axis)
        
        if results:
            return results[0][0]
        return f"Cannot complete analogy"
    
    def answer_property(self, thing: str, property_axis: str, 
                        direction: int = -1) -> str:
        """Answer: What is a [property] [thing]?
        
        direction: +1 for positive end, -1 for negative end
        e.g., "small dog" = dog + size axis (direction=-1)
        """
        embed = self.get_embedding(thing)
        if embed is None:
            return f"Cannot find '{thing}'"
        
        if property_axis not in self.axes:
            return f"Unknown property axis '{property_axis}'"
        
        axis = self.axes[property_axis]
        levels, signs = encode_phi(embed)
        
        if direction == -1:
            signs[axis.dimensions] *= -1
        # direction == +1 means keep as is (already on positive side)
        
        navigated = decode_phi(levels, signs).to(embed.dtype).to(self.device)
        
        thing_id = self.tokenizer.encode(thing, add_special_tokens=False)[0]
        results = self.find_nearest(navigated, exclude_ids=[thing_id])
        
        if results:
            return results[0][0]
        return f"Cannot find {property_axis} version of '{thing}'"


def run_qa_demo(navigator: PhiLatticeNavigator):
    """Run a demo of question-answering via navigation."""
    
    print("\n" + "="*70)
    print("QUESTION-ANSWERING VIA NAVIGATION")
    print("="*70)
    
    # Test 1: Opposites
    print("\n--- OPPOSITE QUESTIONS ---")
    opposite_tests = [
        ("hot", "temperature"),
        ("big", "size"),
        ("fast", "speed"),
        ("young", "age"),
        ("good", "valence"),
        ("happy", "valence"),
    ]
    
    for word, axis in opposite_tests:
        answer = navigator.answer_opposite(word, axis)
        print(f"Q: What is the opposite of '{word}'?")
        print(f"A: {answer}")
        print()
    
    # Test 2: Gender
    print("\n--- GENDER QUESTIONS ---")
    gender_tests = ["king", "man", "boy", "father", "brother", "actor"]
    
    for word in gender_tests:
        answer = navigator.answer_gender(word)
        print(f"Q: What is the female version of '{word}'?")
        print(f"A: {answer}")
        print()
    
    # Test 3: Analogies
    print("\n--- ANALOGY QUESTIONS ---")
    analogy_tests = [
        ("king", "queen", "man"),      # man → woman
        ("hot", "cold", "big"),        # big → small?
        ("good", "bad", "happy"),      # happy → sad?
        ("young", "old", "new"),       # new → old/ancient?
    ]
    
    for a, b, c in analogy_tests:
        answer = navigator.answer_analogy(a, b, c)
        print(f"Q: {a} is to {b} as {c} is to ?")
        print(f"A: {answer}")
        print()
    
    # Test 4: Property questions
    print("\n--- PROPERTY QUESTIONS ---")
    property_tests = [
        ("dog", "size", -1),      # small dog
        ("car", "speed", +1),     # fast car
        ("person", "age", -1),    # young person
    ]
    
    for thing, prop, direction in property_tests:
        prop_word = navigator.axes[prop].examples[0][1 if direction == -1 else 0]
        answer = navigator.answer_property(thing, prop, direction)
        print(f"Q: What is a {prop_word} {thing}?")
        print(f"A: {answer}")
        print()


def run_interactive_qa(navigator: PhiLatticeNavigator):
    """Interactive Q&A session."""
    
    print("\n" + "="*70)
    print("INTERACTIVE Q&A (type 'quit' to exit)")
    print("="*70)
    print("""
Commands:
  opposite <word> [axis]     - Find opposite (axis: temperature, size, speed, age, valence)
  gender <word>              - Find gender counterpart
  analogy <a> <b> <c>        - Complete: a is to b as c is to ?
  navigate <word> <axis>     - Navigate along axis
  extract <word1> <word2>    - Extract axis between words
  quit                       - Exit
""")
    
    while True:
        try:
            cmd = input("\n> ").strip()
        except EOFError:
            break
        
        if cmd.lower() == 'quit':
            break
        
        parts = cmd.split()
        if not parts:
            continue
        
        command = parts[0].lower()
        
        if command == "opposite" and len(parts) >= 2:
            word = parts[1]
            axis = parts[2] if len(parts) > 2 else None
            answer = navigator.answer_opposite(word, axis)
            print(f"→ {answer}")
        
        elif command == "gender" and len(parts) >= 2:
            word = parts[1]
            answer = navigator.answer_gender(word)
            print(f"→ {answer}")
        
        elif command == "analogy" and len(parts) >= 4:
            a, b, c = parts[1], parts[2], parts[3]
            answer = navigator.answer_analogy(a, b, c)
            print(f"→ {a} : {b} :: {c} : {answer}")
        
        elif command == "navigate" and len(parts) >= 3:
            word, axis = parts[1], parts[2]
            results = navigator.navigate(word, axis)
            if results:
                print(f"→ {', '.join(f'{t} ({s:.3f})' for t, s in results[:5])}")
            else:
                print("→ No results")
        
        elif command == "extract" and len(parts) >= 3:
            w1, w2 = parts[1], parts[2]
            axis = navigator.extract_axis(w1, w2)
            print(f"→ Axis has {len(axis)} dimensions")
            
            # Test by applying to a third word
            if len(parts) > 3:
                w3 = parts[3]
                results = navigator.apply_axis(w3, axis)
                if results:
                    print(f"  Applied to '{w3}': {results[0][0]}")
        
        else:
            print("Unknown command. Try: opposite, gender, analogy, navigate, extract")


def main():
    print("="*70)
    print("φ-LATTICE QUESTION ANSWERING")
    print("="*70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Create navigator
    navigator = PhiLatticeNavigator(model, tokenizer)
    
    # Run demo
    run_qa_demo(navigator)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
NAVIGATION-BASED Q&A:

1. OPPOSITE: Flip the relevant semantic axis
   - "opposite of hot?" → flip temperature → "cold"

2. GENDER: Flip the gender axis (7-839 dimensions)
   - "female king?" → flip gender → "queen"

3. ANALOGY: Extract axis from A→B, apply to C
   - "king:queen :: man:?" → extract king→queen axis → apply to man → "woman"

4. PROPERTY: Navigate along property axis
   - "small dog?" → dog + size axis (negative) → "puppy"?

KEY INSIGHT:
The answer already exists at a location in embedding space.
Questions tell us WHERE to navigate.
We don't compute - we LOOK UP.
""")


if __name__ == "__main__":
    main()
