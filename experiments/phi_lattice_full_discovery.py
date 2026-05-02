#!/usr/bin/env python3
"""
φ-Lattice Full Self-Discovery
==============================

Discover ALL semantic axes the model knows about.

Categories to explore:
1. Opposites/Antonyms (temperature, size, speed, age, valence, etc.)
2. Gender (male/female)
3. Number (singular/plural)
4. Tense (past/present/future)
5. Formality (formal/informal)
6. Concreteness (concrete/abstract)
7. Animacy (animate/inanimate)
8. Intensity (strong/weak)
9. Spatial (up/down, left/right, near/far)
10. Temporal (before/after, early/late)
11. Social (superior/inferior, friend/enemy)
12. Physical (solid/liquid, heavy/light)
13. Emotional (calm/excited, brave/afraid)
14. Cognitive (smart/dumb, know/unknown)
15. Moral (good/evil, honest/dishonest)
"""

import torch
import torch.nn.functional as F
import math
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import time

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
    description: str
    n_dimensions: int
    examples: List[Tuple[str, str]]
    validation_accuracy: float
    test_results: List[Tuple[str, str, str]]  # (input, expected, got)


class FullDiscoveryEngine:
    """Full discovery of all semantic axes."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.hidden_dim = model.config.hidden_size
        self.discovered_axes: Dict[str, DiscoveredAxis] = {}
        self.axis_dimensions: Dict[str, torch.Tensor] = {}
    
    def generate(self, prompt: str, max_tokens: int = 300) -> str:
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
    
    def ask_for_pairs(self, prompt: str, n_pairs: int = 15) -> List[Tuple[str, str]]:
        response = self.generate(prompt)
        pairs = []
        for line in response.strip().split('\n'):
            line = line.strip()
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
    
    def validate_axis(self, axis: torch.Tensor, 
                      test_pairs: List[Tuple[str, str]]) -> Tuple[float, List]:
        if len(axis) == 0:
            return 0.0, []
        
        results = []
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
            
            got = nearest[0][0] if nearest else ""
            found = any(word2.lower() in t.lower() for t, _ in nearest)
            
            results.append((word1, word2, got))
            if found:
                correct += 1
            total += 1
        
        return (correct / total if total > 0 else 0.0), results
    
    def discover_relationship(self, name: str, prompt: str) -> Optional[DiscoveredAxis]:
        print(f"\n{'─'*60}")
        print(f"  {name.upper()}")
        print(f"{'─'*60}")
        
        # Get pairs
        pairs = self.ask_for_pairs(prompt)
        print(f"  Pairs: {len(pairs)} - {pairs[:3]}...")
        
        if len(pairs) < 3:
            print(f"  ✗ Not enough pairs")
            return None
        
        # Discover axis
        axis = self.discover_axis(pairs)
        print(f"  Dimensions: {len(axis)}")
        
        if len(axis) == 0:
            print(f"  ✗ No axis found")
            return None
        
        # Validate
        accuracy, results = self.validate_axis(axis, pairs)
        print(f"  Accuracy: {accuracy*100:.0f}%")
        
        # Show some results
        for inp, exp, got in results[:3]:
            marker = "✓" if exp.lower() in got.lower() else "✗"
            print(f"    {inp} → {got} (expected: {exp}) {marker}")
        
        # Store
        self.axis_dimensions[name] = axis
        discovered = DiscoveredAxis(
            name=name,
            description=prompt[:50],
            n_dimensions=len(axis),
            examples=pairs,
            validation_accuracy=accuracy,
            test_results=results,
        )
        self.discovered_axes[name] = discovered
        
        return discovered
    
    def run_full_discovery(self):
        """Discover all semantic axes."""
        
        print("="*70)
        print("φ-LATTICE FULL SELF-DISCOVERY")
        print("="*70)
        
        # Define all relationships to discover
        relationships = [
            # Opposites by category
            ("temperature", "List 15 pairs of temperature-related opposites like hot/cold, warm/cool. Use common single words, one pair per line: word1, word2"),
            ("size", "List 15 pairs of size-related opposites like big/small, huge/tiny. Use common single words, one pair per line: word1, word2"),
            ("speed", "List 15 pairs of speed-related opposites like fast/slow, quick/sluggish. Use common single words, one pair per line: word1, word2"),
            ("age", "List 15 pairs of age-related opposites like young/old, new/ancient. Use common single words, one pair per line: word1, word2"),
            ("valence", "List 15 pairs of positive/negative sentiment opposites like good/bad, happy/sad. Use common single words, one pair per line: word1, word2"),
            ("brightness", "List 15 pairs of brightness-related opposites like bright/dark, light/dim. Use common single words, one pair per line: word1, word2"),
            ("weight", "List 15 pairs of weight-related opposites like heavy/light, dense/sparse. Use common single words, one pair per line: word1, word2"),
            ("height", "List 15 pairs of height-related opposites like tall/short, high/low. Use common single words, one pair per line: word1, word2"),
            ("width", "List 15 pairs of width-related opposites like wide/narrow, broad/thin. Use common single words, one pair per line: word1, word2"),
            ("depth", "List 15 pairs of depth-related opposites like deep/shallow, profound/superficial. Use common single words, one pair per line: word1, word2"),
            
            # Gender (with better prompt)
            ("gender", "List 15 pairs of common nouns where one is male and one is female, like king/queen, man/woman, boy/girl, father/mother. Do NOT use proper names. One pair per line: word1, word2"),
            
            # Number
            ("number", "List 15 pairs of singular/plural word forms like dog/dogs, child/children. One pair per line: singular, plural"),
            
            # Tense
            ("tense_past", "List 15 pairs of present/past tense verbs like run/ran, eat/ate, go/went. One pair per line: present, past"),
            
            # Formality
            ("formality", "List 15 pairs of formal/informal word equivalents like hello/hi, father/dad, mother/mom. One pair per line: formal, informal"),
            
            # Concreteness
            ("concreteness", "List 15 pairs of concrete/abstract word equivalents like book/knowledge, money/wealth, heart/love. One pair per line: concrete, abstract"),
            
            # Animacy
            ("animacy", "List 15 pairs of animate/inanimate equivalents like person/statue, dog/toy, bird/plane. One pair per line: animate, inanimate"),
            
            # Intensity
            ("intensity", "List 15 pairs of strong/weak intensity words like scream/whisper, sprint/walk, love/like. One pair per line: strong, weak"),
            
            # Spatial
            ("spatial_vertical", "List 15 pairs of up/down spatial opposites like up/down, above/below, top/bottom. One pair per line: word1, word2"),
            ("spatial_horizontal", "List 15 pairs of left/right or near/far spatial opposites like left/right, near/far, here/there. One pair per line: word1, word2"),
            
            # Temporal
            ("temporal", "List 15 pairs of temporal opposites like before/after, early/late, past/future. One pair per line: word1, word2"),
            
            # Emotional
            ("emotion_arousal", "List 15 pairs of calm/excited emotional opposites like calm/excited, relaxed/tense, peaceful/anxious. One pair per line: word1, word2"),
            ("emotion_courage", "List 15 pairs of brave/afraid emotional opposites like brave/afraid, bold/timid, confident/nervous. One pair per line: word1, word2"),
            
            # Cognitive
            ("intelligence", "List 15 pairs of smart/dumb cognitive opposites like smart/dumb, wise/foolish, clever/stupid. One pair per line: word1, word2"),
            ("knowledge", "List 15 pairs of known/unknown cognitive opposites like known/unknown, familiar/strange, clear/confusing. One pair per line: word1, word2"),
            
            # Moral
            ("morality", "List 15 pairs of moral opposites like good/evil, honest/dishonest, kind/cruel. One pair per line: word1, word2"),
            
            # Physical state
            ("physical_state", "List 15 pairs of physical state opposites like solid/liquid, wet/dry, clean/dirty. One pair per line: word1, word2"),
            
            # Social
            ("social_status", "List 15 pairs of social status opposites like rich/poor, powerful/weak, famous/unknown. One pair per line: word1, word2"),
            ("social_relation", "List 15 pairs of social relation opposites like friend/enemy, ally/rival, love/hate. One pair per line: word1, word2"),
            
            # Quantity
            ("quantity", "List 15 pairs of quantity opposites like many/few, more/less, full/empty. One pair per line: word1, word2"),
            
            # Quality
            ("quality", "List 15 pairs of quality opposites like good/bad, best/worst, perfect/flawed. One pair per line: word1, word2"),
        ]
        
        # Run discovery for each
        for name, prompt in relationships:
            try:
                self.discover_relationship(name, prompt)
            except Exception as e:
                print(f"  ✗ Error: {e}")
            time.sleep(0.5)  # Small delay to avoid overwhelming
        
        return self.discovered_axes
    
    def report(self):
        """Generate final report."""
        print("\n" + "="*70)
        print("DISCOVERY REPORT")
        print("="*70)
        
        # Sort by accuracy
        sorted_axes = sorted(
            self.discovered_axes.items(),
            key=lambda x: x[1].validation_accuracy,
            reverse=True
        )
        
        print(f"\n{'Axis':<20} {'Dims':>6} {'Accuracy':>10} {'Examples'}")
        print("-"*70)
        
        for name, axis in sorted_axes:
            examples = ", ".join(f"{a}/{b}" for a, b in axis.examples[:2])
            print(f"{name:<20} {axis.n_dimensions:>6} {axis.validation_accuracy*100:>9.0f}% {examples}")
        
        # Summary stats
        working = [a for a in self.discovered_axes.values() if a.validation_accuracy >= 0.5]
        partial = [a for a in self.discovered_axes.values() if 0.2 <= a.validation_accuracy < 0.5]
        failed = [a for a in self.discovered_axes.values() if a.validation_accuracy < 0.2]
        
        print(f"\n{'─'*70}")
        print(f"SUMMARY:")
        print(f"  Working (≥50%): {len(working)} axes")
        print(f"  Partial (20-50%): {len(partial)} axes")
        print(f"  Failed (<20%): {len(failed)} axes")
        print(f"  Total discovered: {len(self.discovered_axes)} axes")
        
        # Dimension overlap analysis
        print(f"\n{'─'*70}")
        print(f"DIMENSION OVERLAP:")
        
        working_names = [a.name for a in working]
        for i, name1 in enumerate(working_names[:5]):
            for name2 in working_names[i+1:6]:
                if name1 in self.axis_dimensions and name2 in self.axis_dimensions:
                    dims1 = set(self.axis_dimensions[name1].tolist())
                    dims2 = set(self.axis_dimensions[name2].tolist())
                    overlap = len(dims1 & dims2)
                    total = len(dims1 | dims2)
                    print(f"  {name1} ∩ {name2}: {overlap}/{total} ({overlap/total*100:.0f}%)")
    
    def save_results(self, filepath: str):
        """Save results to JSON."""
        results = {}
        for name, axis in self.discovered_axes.items():
            results[name] = {
                "name": axis.name,
                "description": axis.description,
                "n_dimensions": axis.n_dimensions,
                "examples": axis.examples,
                "validation_accuracy": axis.validation_accuracy,
                "test_results": axis.test_results,
            }
            if name in self.axis_dimensions:
                results[name]["dimensions"] = self.axis_dimensions[name].tolist()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filepath}")


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Run full discovery
    engine = FullDiscoveryEngine(model, tokenizer)
    engine.run_full_discovery()
    
    # Report
    engine.report()
    
    # Save results
    engine.save_results("/home/thorin/truthspace-lcm/experiments/discovered_axes.json")


if __name__ == "__main__":
    main()
