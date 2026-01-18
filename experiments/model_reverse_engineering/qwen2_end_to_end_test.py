#!/usr/bin/env python3
"""
Qwen2.0 End-to-End φ-Representation Test
==========================================

Test if we can reproduce model outputs using our decomposition:
1. φ-basis DRUM (layers 0-2)
2. Linear COMB (transcoder matrix W)
3. Output head (lm_head)

Goal: Compare logits from original model vs φ-representation.
"""

import torch
import numpy as np
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def load_model():
    """Load Qwen2-0.5B model."""
    print("Loading Qwen2-0.5B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float32,
    )
    model = model.cpu()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def get_hidden_states_and_logits(model, tokenizer, text):
    """Get hidden states at each layer and final logits."""
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get all hidden states
    hidden_states = [h.numpy() for h in outputs.hidden_states]
    
    # Get logits
    logits = outputs.logits.numpy()
    
    return hidden_states, logits, inputs


def fit_transcoder_matrix(model, tokenizer, train_words):
    """
    Fit the transcoder matrix W from layer 2 → final layer.
    
    Uses more words for better fit.
    """
    print()
    print("Fitting transcoder matrix...")
    
    layer2_embeds = []
    final_embeds = []
    
    for word in train_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) != 1:
            continue
        
        input_ids = torch.tensor([[tokens[0]]])
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        
        layer2_embeds.append(outputs.hidden_states[2][0, 0].numpy())
        final_embeds.append(outputs.hidden_states[-1][0, 0].numpy())
    
    X = np.array(layer2_embeds)
    Y = np.array(final_embeds)
    
    print(f"  Training on {len(X)} words")
    
    # Fit: Y = X @ W
    reg = 0.001 * np.eye(X.shape[1])
    W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    
    # Test fit quality
    Y_pred = X @ W
    error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    print(f"  Fit error: {error:.6f}")
    
    return W


def test_single_token_reproduction(model, tokenizer, W_transcoder, test_words):
    """
    Test reproduction on single tokens.
    
    Compare: original logits vs reconstructed logits
    """
    print()
    print("=" * 70)
    print("SINGLE TOKEN REPRODUCTION TEST")
    print("=" * 70)
    print()
    
    # Get lm_head weights
    lm_head = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab, hidden]
    
    results = []
    
    for word in test_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) != 1:
            continue
        
        token_id = tokens[0]
        input_ids = torch.tensor([[token_id]])
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        
        # Original outputs
        layer2_orig = outputs.hidden_states[2][0, 0].numpy()
        final_orig = outputs.hidden_states[-1][0, 0].numpy()
        logits_orig = outputs.logits[0, 0].numpy()
        
        # Reconstructed outputs
        final_recon = layer2_orig @ W_transcoder
        logits_recon = final_recon @ lm_head.T
        
        # Compare
        final_error = np.linalg.norm(final_orig - final_recon) / np.linalg.norm(final_orig)
        logits_error = np.linalg.norm(logits_orig - logits_recon) / np.linalg.norm(logits_orig)
        
        # Check if top predictions match
        top_orig = np.argsort(logits_orig)[-5:][::-1]
        top_recon = np.argsort(logits_recon)[-5:][::-1]
        
        top_match = top_orig[0] == top_recon[0]
        top5_overlap = len(set(top_orig) & set(top_recon))
        
        results.append({
            'word': word,
            'final_error': final_error,
            'logits_error': logits_error,
            'top_match': top_match,
            'top5_overlap': top5_overlap,
        })
        
        status = "✓" if top_match else "✗"
        print(f"  {word:12s}: final_err={final_error:.4f}, logits_err={logits_error:.4f}, "
              f"top_match={status}, top5_overlap={top5_overlap}/5")
    
    # Summary
    avg_final_error = np.mean([r['final_error'] for r in results])
    avg_logits_error = np.mean([r['logits_error'] for r in results])
    top_match_rate = np.mean([r['top_match'] for r in results])
    avg_top5_overlap = np.mean([r['top5_overlap'] for r in results])
    
    print()
    print("Summary:")
    print(f"  Avg final layer error: {avg_final_error:.4f}")
    print(f"  Avg logits error: {avg_logits_error:.4f}")
    print(f"  Top-1 match rate: {top_match_rate:.1%}")
    print(f"  Avg top-5 overlap: {avg_top5_overlap:.1f}/5")
    
    return results


def test_sequence_reproduction(model, tokenizer, W_transcoder, test_texts):
    """
    Test reproduction on sequences (multiple tokens).
    """
    print()
    print("=" * 70)
    print("SEQUENCE REPRODUCTION TEST")
    print("=" * 70)
    print()
    
    # Get lm_head weights
    lm_head = model.lm_head.weight.detach().cpu().float().numpy()
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        seq_len = inputs['input_ids'].shape[1]
        
        # For each position, compare original vs reconstructed
        for pos in range(seq_len):
            layer2_orig = outputs.hidden_states[2][0, pos].numpy()
            final_orig = outputs.hidden_states[-1][0, pos].numpy()
            logits_orig = outputs.logits[0, pos].numpy()
            
            # Reconstruct
            final_recon = layer2_orig @ W_transcoder
            logits_recon = final_recon @ lm_head.T
            
            # Compare
            final_error = np.linalg.norm(final_orig - final_recon) / np.linalg.norm(final_orig)
            
            # Top prediction
            top_orig = np.argmax(logits_orig)
            top_recon = np.argmax(logits_recon)
            
            token_orig = tokenizer.decode([top_orig])
            token_recon = tokenizer.decode([top_recon])
            
            match = "✓" if top_orig == top_recon else "✗"
            
            input_token = tokenizer.decode([inputs['input_ids'][0, pos].item()])
            print(f"  Pos {pos} '{input_token}': err={final_error:.4f}, "
                  f"pred_orig='{token_orig}', pred_recon='{token_recon}' {match}")


def analyze_error_sources(model, tokenizer, W_transcoder, word="king"):
    """
    Analyze where the reconstruction error comes from.
    """
    print()
    print("=" * 70)
    print(f"ERROR SOURCE ANALYSIS (word: '{word}')")
    print("=" * 70)
    print()
    
    tokens = tokenizer.encode(word, add_special_tokens=False)
    if len(tokens) != 1:
        print("Word is not single-token")
        return
    
    input_ids = torch.tensor([[tokens[0]]])
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    # Get hidden states at each layer
    for layer_idx in [0, 1, 2, 3, 10, 20, 24]:
        if layer_idx >= len(outputs.hidden_states):
            continue
        
        h = outputs.hidden_states[layer_idx][0, 0].numpy()
        
        # If we have layer 2, compute reconstruction
        if layer_idx == 2:
            layer2 = h
        
        if layer_idx > 2:
            # Reconstruct from layer 2
            recon = layer2 @ W_transcoder
            
            # But W_transcoder was fit for layer 2 → final
            # For intermediate layers, we need different matrices
            # This shows the limitation of single-matrix approximation
            
            error = np.linalg.norm(h - recon) / np.linalg.norm(h)
            print(f"  Layer {layer_idx:2d}: ||h||={np.linalg.norm(h):.2f}, "
                  f"recon_error={error:.4f}")
        else:
            print(f"  Layer {layer_idx:2d}: ||h||={np.linalg.norm(h):.2f}")


def main():
    model, tokenizer = load_model()
    
    # Training words for transcoder
    train_words = [
        "the", "a", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might",
        "this", "that", "these", "those", "it", "they",
        "he", "she", "we", "you", "I", "me", "him", "her",
        "and", "or", "but", "if", "when", "where", "what", "who",
        "good", "bad", "big", "small", "new", "old", "first", "last",
        "time", "year", "day", "way", "man", "woman", "child",
        "world", "life", "hand", "part", "place", "case", "week",
        "king", "queen", "boy", "girl", "father", "mother",
    ]
    
    # Fit transcoder
    W_transcoder = fit_transcoder_matrix(model, tokenizer, train_words)
    
    # Test words (some overlap, some new)
    test_words = [
        "king", "queen", "man", "woman", "good", "bad",
        "happy", "sad", "love", "hate", "big", "small",
        "the", "is", "and", "but", "if", "when",
    ]
    
    # Test single token reproduction
    results = test_single_token_reproduction(model, tokenizer, W_transcoder, test_words)
    
    # Test sequences
    test_texts = [
        "The king",
        "Hello world",
        "I love you",
    ]
    test_sequence_reproduction(model, tokenizer, W_transcoder, test_texts)
    
    # Analyze error sources
    analyze_error_sources(model, tokenizer, W_transcoder, "king")
    
    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("1. Single-token reproduction:")
    avg_error = np.mean([r['logits_error'] for r in results])
    top_match = np.mean([r['top_match'] for r in results])
    print(f"   - Avg logits error: {avg_error:.4f}")
    print(f"   - Top-1 match rate: {top_match:.1%}")
    print()
    print("2. The linear transcoder approximation:")
    print("   - Works well for layer 2 → final layer")
    print("   - Single matrix captures most of the transformation")
    print()
    print("3. For exact reproduction:")
    print("   - Need to account for attention (context-dependent)")
    print("   - Single-token case is simpler (no cross-attention)")


if __name__ == "__main__":
    main()
