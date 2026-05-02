#!/usr/bin/env python3
"""
Hidden State Compression: Find Minimal Representation
=======================================================

Question: Can we store a SMALLER representation of hidden states
that still decodes correctly?

From DA2: 32 weights decoded the full depth map from 32 channels.
Maybe we can find a similar compression for LLM hidden states.

Approaches:
1. PCA: Find principal components that capture answer information
2. Answer-specific dimensions: Which dimensions matter for decoding?
3. Sparse representation: Most dimensions might be near-zero

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class HiddenStateCompressor:
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
        print(f"  LM head shape: {self.lm_head.shape}")
    
    def _get_hidden(self, prompt: str) -> np.ndarray:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    
    def _decode(self, hidden: np.ndarray) -> Tuple[str, int]:
        """Decode and return token and its index."""
        logits = np.dot(self.lm_head, hidden)
        idx = np.argmax(logits)
        return self.tokenizer.decode([idx]).strip(), idx
    
    def analyze_compression(self, pairs: List[Tuple[str, str]], template: str):
        """Analyze how much we can compress hidden states."""
        print(f"\nCollecting {len(pairs)} hidden states...")
        
        hiddens = []
        entities = []
        answer_ids = []
        
        for entity, answer in pairs:
            hidden = self._get_hidden(template.format(entity=entity))
            token, idx = self._decode(hidden)
            
            hiddens.append(hidden)
            entities.append(entity)
            answer_ids.append(idx)
        
        hiddens = np.array(hiddens)
        answer_ids = np.array(answer_ids)
        
        print(f"  Hidden states shape: {hiddens.shape}")
        
        # 1. PCA Analysis
        print("\n--- PCA Compression ---")
        
        for n_components in [10, 50, 100, 500, 1000]:
            if n_components > min(hiddens.shape):
                continue
                
            pca = PCA(n_components=n_components)
            compressed = pca.fit_transform(hiddens)
            reconstructed = pca.inverse_transform(compressed)
            
            # Test decoding
            correct = 0
            for i in range(len(hiddens)):
                _, orig_idx = self._decode(hiddens[i])
                _, recon_idx = self._decode(reconstructed[i])
                if orig_idx == recon_idx:
                    correct += 1
            
            accuracy = correct / len(hiddens)
            var_explained = pca.explained_variance_ratio_.sum()
            compression = self.hidden_dim / n_components
            
            print(f"  {n_components} components: {accuracy*100:.0f}% accuracy, {var_explained*100:.1f}% variance, {compression:.1f}x compression")
        
        # 2. Find answer-critical dimensions
        print("\n--- Answer-Critical Dimensions ---")
        
        # For each answer token, which dimensions have highest weight in LM head?
        unique_answers = np.unique(answer_ids)
        critical_dims = set()
        
        for ans_id in unique_answers:
            # Get LM head row for this answer
            lm_row = self.lm_head[ans_id]
            
            # Top dimensions by absolute weight
            top_dims = np.argsort(np.abs(lm_row))[-100:]
            critical_dims.update(top_dims)
        
        print(f"  Unique answers: {len(unique_answers)}")
        print(f"  Critical dimensions (top 100 per answer): {len(critical_dims)}")
        
        # Test decoding with only critical dimensions
        critical_dims_list = sorted(critical_dims)
        
        # Create sparse hidden states
        sparse_hiddens = np.zeros_like(hiddens)
        sparse_hiddens[:, critical_dims_list] = hiddens[:, critical_dims_list]
        
        correct = 0
        for i in range(len(hiddens)):
            _, orig_idx = self._decode(hiddens[i])
            _, sparse_idx = self._decode(sparse_hiddens[i])
            if orig_idx == sparse_idx:
                correct += 1
        
        print(f"  Accuracy with {len(critical_dims)} dims: {correct}/{len(hiddens)} = {correct/len(hiddens)*100:.0f}%")
        
        # 3. Quantization analysis
        print("\n--- Quantization Analysis ---")
        
        for bits in [16, 12, 8, 4]:
            # Quantize hidden states
            max_val = np.abs(hiddens).max()
            scale = max_val / (2 ** (bits - 1))
            
            quantized = np.round(hiddens / scale).astype(np.int16)
            dequantized = quantized.astype(np.float32) * scale
            
            correct = 0
            for i in range(len(hiddens)):
                _, orig_idx = self._decode(hiddens[i])
                _, quant_idx = self._decode(dequantized[i])
                if orig_idx == quant_idx:
                    correct += 1
            
            storage_per_entity = self.hidden_dim * bits // 8
            print(f"  {bits}-bit: {correct}/{len(hiddens)} = {correct/len(hiddens)*100:.0f}% accuracy, {storage_per_entity} bytes/entity")
        
        # 4. Combined: PCA + Quantization
        print("\n--- Combined Compression ---")
        
        best_n = min(10, len(hiddens) - 1)  # Can't have more components than samples
        pca = PCA(n_components=best_n)
        compressed = pca.fit_transform(hiddens)
        
        # Quantize compressed representation
        max_val = np.abs(compressed).max()
        scale = max_val / 32767
        quantized = np.round(compressed / scale).astype(np.int16)
        dequantized = quantized.astype(np.float32) * scale
        
        reconstructed = pca.inverse_transform(dequantized)
        
        correct = 0
        for i in range(len(hiddens)):
            _, orig_idx = self._decode(hiddens[i])
            _, recon_idx = self._decode(reconstructed[i])
            if orig_idx == recon_idx:
                correct += 1
        
        storage = best_n * 2  # int16
        original_storage = self.hidden_dim * 4  # float32
        compression_ratio = original_storage / storage
        
        print(f"  PCA({best_n}) + int16: {correct}/{len(hiddens)} = {correct/len(hiddens)*100:.0f}% accuracy")
        print(f"  Storage: {storage} bytes vs {original_storage} bytes = {compression_ratio:.1f}x compression")
        
        return {
            'hiddens': hiddens,
            'entities': entities,
            'answer_ids': answer_ids,
        }


def main():
    print("=" * 70)
    print("HIDDEN STATE COMPRESSION ANALYSIS")
    print("=" * 70)
    
    compressor = HiddenStateCompressor()
    
    # Use more pairs for better analysis
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
    
    compressor.analyze_compression(pairs, "The capital of {entity} is")
    
    print("\n" + "=" * 70)
    print("SCALABILITY ANALYSIS")
    print("=" * 70)
    
    # Calculate storage requirements
    print("""
Storage requirements for lookup table:

Original (float32): 3584 dims × 4 bytes = 14,336 bytes/entity
  - 1M entities = 14.3 GB
  - 10M entities = 143 GB

Int16 quantized: 3584 dims × 2 bytes = 7,168 bytes/entity
  - 1M entities = 7.2 GB
  - 10M entities = 72 GB

PCA(100) + int16: 100 dims × 2 bytes = 200 bytes/entity
  - 1M entities = 200 MB
  - 10M entities = 2 GB
  - 100M entities = 20 GB

This is VERY feasible for a lookup table approach!
""")


if __name__ == "__main__":
    main()
