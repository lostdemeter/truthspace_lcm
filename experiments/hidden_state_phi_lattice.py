#!/usr/bin/env python3
"""
Hidden State φ-Lattice Analysis
================================

From Doc 162 (Tetromino Hypothesis):
- Weights exist on a constrained φ-lattice
- Only ~300 unique (level, sign_pattern) combinations
- 81.6% of deltas within ±2 levels

Hypothesis: If weights are on φ-lattice, hidden states should be too!
- Hidden state = linear combination of φ-lattice weights
- Therefore hidden state should also be on φ-lattice
- We might be able to store hidden states as (φ-level, sign) pairs

This could give us MUCH better compression than naive 8-bit quantization.

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


def to_phi_level(value: float, k: int = 32) -> Tuple[int, int]:
    """Convert float to (sign, φ-level)."""
    if abs(value) < 1e-10:
        return 0, 0
    sign = 1 if value > 0 else -1
    level = int(round(k * np.log(abs(value)) / LOG_PHI))
    return sign, level


def from_phi_level(sign: int, level: int, k: int = 32) -> float:
    """Convert (sign, φ-level) back to float."""
    if sign == 0:
        return 0.0
    return sign * (PHI ** (level / k))


class HiddenStatePhiAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        self.hidden_dim = self.model.config.hidden_size
        
        print(f"  Hidden dim: {self.hidden_dim}")
    
    def _get_hidden(self, prompt: str) -> np.ndarray:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    
    def _decode(self, hidden: np.ndarray) -> Tuple[str, int]:
        logits = np.dot(self.lm_head, hidden)
        idx = np.argmax(logits)
        return self.tokenizer.decode([idx]).strip(), idx
    
    def analyze_phi_structure(self, pairs: List[Tuple[str, str]], template: str):
        """Analyze φ-lattice structure of hidden states."""
        print(f"\nCollecting {len(pairs)} hidden states...")
        
        hiddens = []
        entities = []
        
        for entity, answer in pairs:
            hidden = self._get_hidden(template.format(entity=entity))
            hiddens.append(hidden)
            entities.append(entity)
        
        hiddens = np.array(hiddens)
        
        # 1. Convert all hidden state values to φ-levels
        print("\n--- φ-Level Distribution ---")
        
        all_levels = []
        all_signs = []
        
        for hidden in hiddens:
            for val in hidden:
                sign, level = to_phi_level(val)
                all_levels.append(level)
                all_signs.append(sign)
        
        all_levels = np.array(all_levels)
        all_signs = np.array(all_signs)
        
        print(f"  Level range: [{all_levels.min()}, {all_levels.max()}]")
        print(f"  Level mean: {all_levels.mean():.1f}")
        print(f"  Level std: {all_levels.std():.1f}")
        
        # How many unique levels?
        unique_levels = len(np.unique(all_levels))
        print(f"  Unique levels: {unique_levels}")
        
        # Level distribution
        level_counts = Counter(all_levels)
        top_levels = level_counts.most_common(10)
        print(f"\n  Top 10 levels:")
        for level, count in top_levels:
            pct = count / len(all_levels) * 100
            print(f"    Level {level}: {pct:.1f}%")
        
        # 2. Test φ-lattice reconstruction
        print("\n--- φ-Lattice Reconstruction ---")
        
        for k in [8, 16, 32, 64, 128]:
            correct = 0
            
            for i, hidden in enumerate(hiddens):
                # Convert to φ-lattice
                signs = np.sign(hidden)
                signs[np.abs(hidden) < 1e-10] = 0
                
                magnitudes = np.abs(hidden) + 1e-10
                levels = np.round(k * np.log(magnitudes) / LOG_PHI).astype(int)
                
                # Reconstruct
                reconstructed = signs * (PHI ** (levels / k))
                reconstructed[signs == 0] = 0
                
                # Decode
                _, orig_idx = self._decode(hidden)
                _, recon_idx = self._decode(reconstructed)
                
                if orig_idx == recon_idx:
                    correct += 1
            
            # Calculate bits needed
            level_range = all_levels.max() - all_levels.min()
            bits_per_level = int(np.ceil(np.log2(level_range + 1)))
            bits_per_value = bits_per_level + 1  # +1 for sign
            storage_per_entity = self.hidden_dim * bits_per_value // 8
            
            accuracy = correct / len(hiddens)
            print(f"  k={k}: {correct}/{len(hiddens)} = {accuracy*100:.0f}% accuracy, ~{storage_per_entity} bytes/entity")
        
        # 3. Test if φ-levels are predictable across entities
        print("\n--- φ-Level Correlation Across Entities ---")
        
        # Convert all hiddens to φ-levels
        hidden_levels = []
        for hidden in hiddens:
            signs = np.sign(hidden)
            magnitudes = np.abs(hidden) + 1e-10
            levels = np.round(32 * np.log(magnitudes) / LOG_PHI).astype(int)
            hidden_levels.append(levels)
        
        hidden_levels = np.array(hidden_levels)
        
        # Correlation between entities
        correlations = []
        for i in range(len(hidden_levels)):
            for j in range(i+1, len(hidden_levels)):
                corr = np.corrcoef(hidden_levels[i], hidden_levels[j])[0, 1]
                correlations.append(corr)
        
        print(f"  Mean level correlation between entities: {np.mean(correlations):.3f}")
        print(f"  Std level correlation: {np.std(correlations):.3f}")
        
        # 4. Test delta encoding (like tetromino)
        print("\n--- Delta Encoding (Tetromino Style) ---")
        
        # Compute mean level per dimension
        mean_levels = hidden_levels.mean(axis=0)
        
        # Compute deltas from mean
        deltas = hidden_levels - mean_levels
        
        # Delta distribution
        delta_flat = deltas.flatten()
        print(f"  Delta range: [{delta_flat.min():.0f}, {delta_flat.max():.0f}]")
        print(f"  Delta std: {delta_flat.std():.1f}")
        
        # What % of deltas are within ±N?
        for n in [1, 2, 3, 5, 10]:
            within = np.sum(np.abs(delta_flat) <= n) / len(delta_flat) * 100
            print(f"  |Δ| ≤ {n}: {within:.1f}%")
        
        # 5. Test delta-based reconstruction
        print("\n--- Delta-Based Reconstruction ---")
        
        # Store: mean_levels (shared) + deltas (per entity)
        # Reconstruct: levels = mean_levels + deltas
        
        for max_delta in [2, 5, 10, 20]:
            correct = 0
            
            for i, hidden in enumerate(hiddens):
                # Get original levels
                signs = np.sign(hidden)
                magnitudes = np.abs(hidden) + 1e-10
                orig_levels = np.round(32 * np.log(magnitudes) / LOG_PHI).astype(int)
                
                # Compute delta from mean
                delta = orig_levels - mean_levels
                
                # Clip delta
                clipped_delta = np.clip(delta, -max_delta, max_delta)
                
                # Reconstruct levels
                recon_levels = mean_levels + clipped_delta
                
                # Reconstruct hidden state
                reconstructed = signs * (PHI ** (recon_levels / 32))
                reconstructed[signs == 0] = 0
                
                # Decode
                _, orig_idx = self._decode(hidden)
                _, recon_idx = self._decode(reconstructed)
                
                if orig_idx == recon_idx:
                    correct += 1
            
            # Storage: mean_levels (shared) + clipped deltas (per entity)
            bits_per_delta = int(np.ceil(np.log2(2 * max_delta + 1)))
            delta_storage = self.hidden_dim * bits_per_delta // 8
            
            accuracy = correct / len(hiddens)
            print(f"  max_delta={max_delta}: {accuracy*100:.0f}% accuracy, {delta_storage} bytes/entity (deltas only)")
        
        return {
            'hiddens': hiddens,
            'hidden_levels': hidden_levels,
            'mean_levels': mean_levels,
        }


def main():
    print("=" * 70)
    print("HIDDEN STATE φ-LATTICE ANALYSIS")
    print("=" * 70)
    print("""
From Tetromino Hypothesis (Doc 162):
- Weights are on φ-lattice with ~300 unique patterns
- 81.6% of deltas within ±2 levels

Question: Are hidden states also on φ-lattice?
If so, we can store them MUCH more efficiently.
""")
    
    analyzer = HiddenStatePhiAnalyzer()
    
    pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("India", "Delhi"),
        ("Brazil", "Brasilia"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Russia", "Moscow"),
        ("Mexico", "Mexico"),
        ("Egypt", "Cairo"),
        ("Greece", "Athens"),
        ("Sweden", "Stockholm"),
    ]
    
    results = analyzer.analyze_phi_structure(pairs, "The capital of {entity} is")
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)


if __name__ == "__main__":
    main()
