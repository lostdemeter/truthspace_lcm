#!/usr/bin/env python3
"""
Test Factorized Embeddings
===========================

Test if we can use low-rank factorized embeddings without losing accuracy.

Embeddings: (152064, 3584) = 545M params
Factorized: (152064, k) @ (k, 3584) where k << 3584

From SVD analysis:
- k=351 captures 50% variance
- k=1425 captures 90% variance  
- k=2051 captures 95% variance
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main():
    print("=" * 70)
    print("FACTORIZED EMBEDDINGS TEST")
    print("=" * 70)
    
    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
    config._attn_implementation = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    device = next(model.parameters()).device
    
    # Get embeddings
    embeddings = model.model.embed_tokens.weight.data.float().cpu().numpy()
    V, d = embeddings.shape
    print(f"\nOriginal embeddings: {V} × {d} = {V*d:,} params")
    
    # Compute SVD
    print("\nComputing SVD...")
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
    
    # Test different ranks
    test_ranks = [100, 351, 500, 1000, 1425, 2000, 2051, 3000]
    
    print("\n--- Reconstruction Error ---")
    for k in test_ranks:
        # Reconstruct with rank k
        E_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        
        # Compute error
        error = np.linalg.norm(embeddings - E_k, 'fro') / np.linalg.norm(embeddings, 'fro')
        
        # Param count
        params_factorized = V * k + k * d
        compression = (V * d) / params_factorized
        
        print(f"  k={k:4d}: error={error:.4f}, params={params_factorized:,}, compression={compression:.2f}x")
    
    # Test token prediction accuracy with factorized embeddings
    print("\n--- Token Prediction Accuracy ---")
    
    np.random.seed(42)
    n_samples = 30
    
    for k in [351, 1425, 2051]:
        print(f"\n  Testing k={k}:")
        
        # Create factorized embedding
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]
        
        # E_factorized = U_k @ S_k @ Vt_k
        # But we want to factor as: A @ B where A is (V, k), B is (k, d)
        # So: A = U_k @ sqrt(S_k), B = sqrt(S_k) @ Vt_k
        sqrt_S = np.sqrt(S[:k])
        A = U_k * sqrt_S  # (V, k)
        B = sqrt_S[:, None] * Vt_k  # (k, d)
        
        # Convert to torch
        A_torch = torch.tensor(A, dtype=torch.bfloat16, device=device)
        B_torch = torch.tensor(B, dtype=torch.bfloat16, device=device)
        
        correct = 0
        
        for i in range(n_samples):
            token_A = np.random.randint(100, 10000)
            token_B = np.random.randint(100, 10000)
            
            # Original prediction
            ids = torch.tensor([[token_A, token_B]]).to(device)
            with torch.no_grad():
                out = model(ids)
            actual = torch.argmax(out.logits[0, 1]).item()
            
            # Factorized embedding
            emb_A_factorized = A_torch[token_A] @ B_torch
            emb_B_factorized = A_torch[token_B] @ B_torch
            
            # Original embedding
            emb_A_original = model.model.embed_tokens.weight.data[token_A]
            emb_B_original = model.model.embed_tokens.weight.data[token_B]
            
            # Check embedding similarity
            cos_A = torch.nn.functional.cosine_similarity(
                emb_A_factorized.unsqueeze(0), 
                emb_A_original.unsqueeze(0)
            ).item()
            
            # Replace embeddings temporarily and run
            # (This is a bit hacky but works for testing)
            original_embed_fn = model.model.embed_tokens.forward
            
            def factorized_embed(input_ids):
                batch_size, seq_len = input_ids.shape
                result = torch.zeros(batch_size, seq_len, d, dtype=torch.bfloat16, device=device)
                for b in range(batch_size):
                    for s in range(seq_len):
                        token_id = input_ids[b, s].item()
                        result[b, s] = A_torch[token_id] @ B_torch
                return result
            
            model.model.embed_tokens.forward = factorized_embed
            
            with torch.no_grad():
                out_factorized = model(ids)
            pred = torch.argmax(out_factorized.logits[0, 1]).item()
            
            model.model.embed_tokens.forward = original_embed_fn
            
            if actual == pred:
                correct += 1
        
        accuracy = correct / n_samples * 100
        print(f"    Accuracy: {correct}/{n_samples} = {accuracy:.1f}%")
        
        # Param savings
        original_params = V * d
        factorized_params = V * k + k * d
        savings = (1 - factorized_params / original_params) * 100
        print(f"    Param savings: {savings:.1f}%")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
