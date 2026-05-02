#!/usr/bin/env python3
"""
Hidden State Sign Analysis
===========================

From Tetromino Hypothesis:
- Signs are the irreducible information (1 bit each)
- Magnitudes follow φ-lattice (compressible)

Hypothesis: The SIGNS of hidden state dimensions might be the key.
- Mean magnitude per dimension (shared)
- Sign pattern per entity (entity-specific)

This would give us: 3584 bits = 448 bytes per entity!

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class HiddenStateSignAnalyzer:
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
    
    def analyze_signs(self, pairs: List[Tuple[str, str]], template: str):
        """Analyze sign patterns in hidden states."""
        print(f"\nCollecting {len(pairs)} hidden states...")
        
        hiddens = []
        entities = []
        
        for entity, answer in pairs:
            hidden = self._get_hidden(template.format(entity=entity))
            hiddens.append(hidden)
            entities.append(entity)
        
        hiddens = np.array(hiddens)
        
        # 1. Extract signs and magnitudes
        print("\n--- Sign/Magnitude Decomposition ---")
        
        signs = np.sign(hiddens)
        magnitudes = np.abs(hiddens)
        
        # Mean magnitude per dimension
        mean_magnitudes = magnitudes.mean(axis=0)
        
        print(f"  Mean magnitude range: [{mean_magnitudes.min():.3f}, {mean_magnitudes.max():.3f}]")
        print(f"  Mean magnitude mean: {mean_magnitudes.mean():.3f}")
        
        # 2. Sign consistency across entities
        print("\n--- Sign Consistency ---")
        
        # For each dimension, what % of entities have the same sign?
        sign_consistency = []
        for d in range(self.hidden_dim):
            pos_count = np.sum(signs[:, d] > 0)
            neg_count = np.sum(signs[:, d] < 0)
            consistency = max(pos_count, neg_count) / len(hiddens)
            sign_consistency.append(consistency)
        
        sign_consistency = np.array(sign_consistency)
        
        print(f"  Mean sign consistency: {sign_consistency.mean():.3f}")
        print(f"  Dims with >90% consistency: {np.sum(sign_consistency > 0.9)}")
        print(f"  Dims with >80% consistency: {np.sum(sign_consistency > 0.8)}")
        print(f"  Dims with >70% consistency: {np.sum(sign_consistency > 0.7)}")
        
        # 3. Test: Use mean magnitude + entity signs
        print("\n--- Mean Magnitude + Entity Signs ---")
        
        correct = 0
        for i in range(len(hiddens)):
            # Reconstruct: mean_magnitude * entity_sign
            reconstructed = mean_magnitudes * signs[i]
            
            _, orig_idx = self._decode(hiddens[i])
            _, recon_idx = self._decode(reconstructed)
            if orig_idx == recon_idx:
                correct += 1
        
        storage = self.hidden_dim // 8  # 1 bit per sign
        print(f"  Accuracy: {correct}/{len(hiddens)} = {correct/len(hiddens)*100:.0f}%")
        print(f"  Storage: {storage} bytes/entity (signs only)")
        
        # 4. Test: Use mean hidden + sign flips
        print("\n--- Mean Hidden + Sign Flips ---")
        
        mean_hidden = hiddens.mean(axis=0)
        mean_signs = np.sign(mean_hidden)
        
        # For each entity, which signs differ from mean?
        sign_diffs = signs != mean_signs.reshape(1, -1)
        
        print(f"  Mean sign flips per entity: {sign_diffs.sum(axis=1).mean():.0f}")
        print(f"  Max sign flips: {sign_diffs.sum(axis=1).max()}")
        print(f"  Min sign flips: {sign_diffs.sum(axis=1).min()}")
        
        # Test reconstruction with sign flips
        correct = 0
        for i in range(len(hiddens)):
            # Reconstruct: flip signs of mean_hidden where entity differs
            reconstructed = mean_hidden.copy()
            reconstructed[sign_diffs[i]] *= -1
            
            _, orig_idx = self._decode(hiddens[i])
            _, recon_idx = self._decode(reconstructed)
            if orig_idx == recon_idx:
                correct += 1
        
        # Storage: just the indices of flipped signs
        avg_flips = sign_diffs.sum(axis=1).mean()
        storage_sparse = int(avg_flips * 2)  # 2 bytes per index (up to 65536 dims)
        
        print(f"  Accuracy: {correct}/{len(hiddens)} = {correct/len(hiddens)*100:.0f}%")
        print(f"  Storage (sparse): ~{storage_sparse} bytes/entity")
        
        # 5. Test: Mean hidden + magnitude scaling
        print("\n--- Mean Hidden + Magnitude Scaling ---")
        
        # For each entity, compute scaling factor per dimension
        scales = hiddens / (mean_hidden + 1e-10)
        
        print(f"  Scale range: [{scales.min():.3f}, {scales.max():.3f}]")
        print(f"  Scale mean: {scales.mean():.3f}")
        print(f"  Scale std: {scales.std():.3f}")
        
        # Quantize scales
        for bits in [8, 4, 2, 1]:
            if bits == 1:
                # Binary: scale > 1 or scale <= 1
                quantized = (scales > 1).astype(float) * 2 - 1  # -1 or 1
                reconstructed_all = mean_hidden * (1 + 0.5 * quantized)  # ±50% adjustment
            else:
                # Quantize to bits levels
                max_scale = 3.0  # Clip extreme scales
                clipped = np.clip(scales, -max_scale, max_scale)
                levels = 2 ** bits
                quantized = np.round((clipped + max_scale) / (2 * max_scale) * (levels - 1))
                dequantized = quantized / (levels - 1) * (2 * max_scale) - max_scale
                reconstructed_all = mean_hidden * dequantized
            
            correct = 0
            for i in range(len(hiddens)):
                _, orig_idx = self._decode(hiddens[i])
                _, recon_idx = self._decode(reconstructed_all[i])
                if orig_idx == recon_idx:
                    correct += 1
            
            storage = self.hidden_dim * bits // 8
            print(f"  {bits}-bit scales: {correct}/{len(hiddens)} = {correct/len(hiddens)*100:.0f}% accuracy, {storage} bytes/entity")
        
        # 6. The key test: What's the MINIMUM info needed?
        print("\n--- Minimum Information Test ---")
        
        # Just store the answer token ID
        print("  If we just store the answer token ID:")
        print(f"    Storage: 4 bytes/entity")
        print(f"    But we need to map entity → answer, which is the whole problem!")
        
        # Store the TOP-K dimensions that differ most
        print("\n  Top-K differing dimensions:")
        
        for k in [10, 50, 100, 200, 500]:
            correct = 0
            total_storage = 0
            
            for i in range(len(hiddens)):
                # Find dimensions that differ most from mean
                diff = np.abs(hiddens[i] - mean_hidden)
                top_k_dims = np.argsort(diff)[-k:]
                
                # Reconstruct: mean + corrections for top-k dims
                reconstructed = mean_hidden.copy()
                reconstructed[top_k_dims] = hiddens[i, top_k_dims]
                
                _, orig_idx = self._decode(hiddens[i])
                _, recon_idx = self._decode(reconstructed)
                if orig_idx == recon_idx:
                    correct += 1
                
                # Storage: k indices (2 bytes each) + k values (2 bytes each, int16)
                total_storage += k * 4
            
            avg_storage = total_storage // len(hiddens)
            print(f"    k={k}: {correct}/{len(hiddens)} = {correct/len(hiddens)*100:.0f}% accuracy, {avg_storage} bytes/entity")
        
        return {
            'hiddens': hiddens,
            'signs': signs,
            'magnitudes': magnitudes,
            'mean_magnitudes': mean_magnitudes,
            'sign_consistency': sign_consistency,
        }


def main():
    print("=" * 70)
    print("HIDDEN STATE SIGN ANALYSIS")
    print("=" * 70)
    print("""
From Tetromino Hypothesis:
- Signs are irreducible (1 bit each)
- Magnitudes follow φ-lattice

Question: Can we store hidden states as:
- Mean magnitude (shared) + signs (per entity)?
- This would be only 448 bytes per entity!
""")
    
    analyzer = HiddenStateSignAnalyzer()
    
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
    
    results = analyzer.analyze_signs(pairs, "The capital of {entity} is")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
