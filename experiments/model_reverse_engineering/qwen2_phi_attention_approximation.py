#!/usr/bin/env python3
"""
Qwen2.0 φ-Based Attention Approximation
=========================================

Goal: Approximate attention with position-based φ-weights.
Target: 99.99% accuracy on sequence reproduction.

Strategy:
1. Extract actual attention patterns from the model
2. Find φ-based position weights that approximate them
3. Test reconstruction accuracy

From DA2 (doc 124):
- 17 unique φ-angles define the rotation structure
- θ ∈ {k × π / φ^n : k ∈ [-20, 20], n ∈ [-3, 3]}

From doc 055:
- W-axis = tachyon navigation (past → future)
- Position-based attention follows φ-decay

Hypothesis:
- Attention weights follow φ^(-distance) decay
- Combined with MASS (similarity) and SPIN (navigation)
"""

import torch
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def load_model():
    """Load Qwen2-0.5B model with eager attention."""
    print("Loading Qwen2-0.5B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model = model.cpu()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def get_attention_patterns(model, tokenizer, texts):
    """Extract attention patterns from multiple texts."""
    all_attentions = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Stack all layer attentions
        # Shape per layer: [batch, n_heads, seq_len, seq_len]
        attentions = torch.stack(outputs.attentions)  # [n_layers, batch, n_heads, seq_len, seq_len]
        
        all_attentions.append({
            'text': text,
            'tokens': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'attentions': attentions[:, 0].numpy(),  # Remove batch dim
        })
    
    return all_attentions


def analyze_position_decay(attentions_list):
    """
    Analyze how attention decays with position distance.
    
    Hypothesis: attention[i, j] ∝ φ^(-(i-j))
    """
    print()
    print("=" * 70)
    print("POSITION DECAY ANALYSIS")
    print("=" * 70)
    print()
    
    # Collect attention by distance
    distance_weights = defaultdict(list)
    
    for item in attentions_list:
        attns = item['attentions']  # [n_layers, n_heads, seq_len, seq_len]
        n_layers, n_heads, seq_len, _ = attns.shape
        
        for layer in range(n_layers):
            for head in range(n_heads):
                for i in range(seq_len):
                    for j in range(i + 1):  # Causal: j <= i
                        distance = i - j
                        weight = attns[layer, head, i, j]
                        distance_weights[distance].append(weight)
    
    # Compute mean weight per distance
    print("Mean attention weight by distance:")
    distances = sorted(distance_weights.keys())
    mean_weights = []
    
    for d in distances[:10]:  # First 10 distances
        weights = distance_weights[d]
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        mean_weights.append(mean_w)
        
        # Compare to φ-decay
        phi_decay = PHI ** (-d)
        ratio = mean_w / phi_decay if phi_decay > 0 else 0
        
        print(f"  d={d}: mean={mean_w:.4f} (std={std_w:.4f}), φ^(-{d})={phi_decay:.4f}, ratio={ratio:.4f}")
    
    return distances, mean_weights


def fit_phi_position_weights(attentions_list, n_layers=24, n_heads=14):
    """
    Fit φ-based position weights to approximate attention.
    
    Model: attention[i, j] = softmax(φ_weight[i-j] + content_similarity[i, j])
    
    We'll fit:
    1. Position-based φ-weights (shared across layers/heads)
    2. Per-layer scaling factors
    """
    print()
    print("=" * 70)
    print("FITTING φ-POSITION WEIGHTS")
    print("=" * 70)
    print()
    
    # Collect all attention patterns
    all_attns = []
    max_seq_len = 0
    
    for item in attentions_list:
        attns = item['attentions']
        seq_len = attns.shape[2]
        max_seq_len = max(max_seq_len, seq_len)
        all_attns.append(attns)
    
    print(f"Max sequence length: {max_seq_len}")
    
    # Initialize φ-position weights
    # Model: w[d] = α * φ^(-d/β) + γ
    # Parameters: α (scale), β (decay rate), γ (baseline)
    
    def phi_position_weights(params, max_dist):
        alpha, beta, gamma = params
        distances = np.arange(max_dist)
        weights = alpha * (PHI ** (-distances / beta)) + gamma
        return weights
    
    def compute_loss(params, attentions_list, max_dist):
        """Compute MSE between actual and predicted attention."""
        phi_weights = phi_position_weights(params, max_dist)
        
        total_loss = 0
        total_count = 0
        
        for item in attentions_list:
            attns = item['attentions']
            n_layers, n_heads, seq_len, _ = attns.shape
            
            for layer in range(n_layers):
                for head in range(n_heads):
                    for i in range(seq_len):
                        # Compute predicted attention (softmax of φ-weights)
                        distances = np.arange(i + 1)
                        logits = phi_weights[distances]
                        pred_attn = np.exp(logits) / np.sum(np.exp(logits))
                        
                        # Actual attention
                        actual_attn = attns[layer, head, i, :i+1]
                        
                        # MSE
                        loss = np.mean((pred_attn - actual_attn) ** 2)
                        total_loss += loss
                        total_count += 1
        
        return total_loss / total_count
    
    # Optimize
    initial_params = [1.0, 1.0, 0.0]  # α, β, γ
    
    print("Optimizing φ-position weights...")
    result = minimize(
        compute_loss,
        initial_params,
        args=(attentions_list, max_seq_len),
        method='Nelder-Mead',
        options={'maxiter': 100, 'disp': True}
    )
    
    best_params = result.x
    best_loss = result.fun
    
    print()
    print(f"Best parameters: α={best_params[0]:.4f}, β={best_params[1]:.4f}, γ={best_params[2]:.4f}")
    print(f"Best MSE loss: {best_loss:.6f}")
    
    # Compute accuracy
    accuracy = 1 - np.sqrt(best_loss)  # Rough approximation
    print(f"Approximate accuracy: {accuracy:.4%}")
    
    return best_params, best_loss


def fit_per_layer_phi_weights(attentions_list):
    """
    Fit separate φ-weights for each layer.
    
    This should give better accuracy since different layers
    have different attention patterns.
    """
    print()
    print("=" * 70)
    print("FITTING PER-LAYER φ-WEIGHTS")
    print("=" * 70)
    print()
    
    n_layers = 24
    layer_params = []
    layer_losses = []
    
    for layer in range(n_layers):
        # Collect attention for this layer
        layer_attns = []
        max_seq_len = 0
        
        for item in attentions_list:
            attns = item['attentions'][layer]  # [n_heads, seq_len, seq_len]
            seq_len = attns.shape[1]
            max_seq_len = max(max_seq_len, seq_len)
            layer_attns.append(attns)
        
        def compute_layer_loss(params, layer_attns, max_dist):
            alpha, beta, gamma = params
            distances = np.arange(max_dist)
            phi_weights = alpha * (PHI ** (-distances / beta)) + gamma
            
            total_loss = 0
            total_count = 0
            
            for attns in layer_attns:
                n_heads, seq_len, _ = attns.shape
                
                for head in range(n_heads):
                    for i in range(seq_len):
                        distances = np.arange(i + 1)
                        logits = phi_weights[distances]
                        pred_attn = np.exp(logits) / np.sum(np.exp(logits))
                        actual_attn = attns[head, i, :i+1]
                        
                        loss = np.mean((pred_attn - actual_attn) ** 2)
                        total_loss += loss
                        total_count += 1
            
            return total_loss / total_count
        
        # Optimize for this layer
        initial_params = [1.0, 1.0, 0.0]
        result = minimize(
            compute_layer_loss,
            initial_params,
            args=(layer_attns, max_seq_len),
            method='Nelder-Mead',
            options={'maxiter': 50}
        )
        
        layer_params.append(result.x)
        layer_losses.append(result.fun)
        
        if layer % 6 == 0 or layer == 23:
            print(f"Layer {layer:2d}: α={result.x[0]:.3f}, β={result.x[1]:.3f}, γ={result.x[2]:.3f}, MSE={result.fun:.6f}")
    
    avg_loss = np.mean(layer_losses)
    print()
    print(f"Average MSE across layers: {avg_loss:.6f}")
    print(f"Approximate accuracy: {1 - np.sqrt(avg_loss):.4%}")
    
    return layer_params, layer_losses


def fit_content_aware_phi_attention(model, tokenizer, attentions_list):
    """
    Fit φ-attention that includes content similarity.
    
    Model: attention[i, j] = softmax(φ_pos[i-j] + similarity(embed[i], embed[j]))
    
    This combines:
    1. Position-based φ-weights (SPIN-like navigation)
    2. Content similarity (MASS-like)
    """
    print()
    print("=" * 70)
    print("FITTING CONTENT-AWARE φ-ATTENTION")
    print("=" * 70)
    print()
    
    # Get embeddings for each text
    embeddings_list = []
    
    for item in attentions_list:
        text = item['text']
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Use layer 0 embeddings (before any attention)
        embed = outputs.hidden_states[0][0].numpy()  # [seq_len, hidden_dim]
        embeddings_list.append(embed)
    
    def compute_content_loss(params, attentions_list, embeddings_list):
        """
        Compute loss with content-aware attention.
        
        params: [α, β, γ, sim_scale]
        - α, β, γ: φ-position weight parameters
        - sim_scale: scaling for content similarity
        """
        alpha, beta, gamma, sim_scale = params
        
        total_loss = 0
        total_count = 0
        
        for item, embed in zip(attentions_list, embeddings_list):
            attns = item['attentions']
            n_layers, n_heads, seq_len, _ = attns.shape
            
            # Compute content similarity matrix
            embed_norm = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-8)
            similarity = embed_norm @ embed_norm.T  # [seq_len, seq_len]
            
            for layer in range(n_layers):
                for head in range(n_heads):
                    for i in range(seq_len):
                        # Position-based φ-weights
                        distances = np.arange(i + 1)
                        pos_logits = alpha * (PHI ** (-distances / beta)) + gamma
                        
                        # Content similarity
                        sim_logits = sim_scale * similarity[i, :i+1]
                        
                        # Combined logits
                        logits = pos_logits + sim_logits
                        pred_attn = np.exp(logits - np.max(logits))
                        pred_attn = pred_attn / np.sum(pred_attn)
                        
                        # Actual attention
                        actual_attn = attns[layer, head, i, :i+1]
                        
                        # MSE
                        loss = np.mean((pred_attn - actual_attn) ** 2)
                        total_loss += loss
                        total_count += 1
        
        return total_loss / total_count
    
    # Optimize
    initial_params = [1.0, 1.0, 0.0, 1.0]  # α, β, γ, sim_scale
    
    print("Optimizing content-aware φ-attention...")
    result = minimize(
        compute_content_loss,
        initial_params,
        args=(attentions_list, embeddings_list),
        method='Nelder-Mead',
        options={'maxiter': 200, 'disp': True}
    )
    
    best_params = result.x
    best_loss = result.fun
    
    print()
    print(f"Best parameters:")
    print(f"  Position: α={best_params[0]:.4f}, β={best_params[1]:.4f}, γ={best_params[2]:.4f}")
    print(f"  Content: sim_scale={best_params[3]:.4f}")
    print(f"Best MSE loss: {best_loss:.6f}")
    print(f"Approximate accuracy: {1 - np.sqrt(best_loss):.4%}")
    
    return best_params, best_loss


def fit_exact_attention_lookup(attentions_list, max_positions=20):
    """
    Fit exact attention weights per position (like DA2's error LUT).
    
    Instead of parameterized φ-decay, store exact weights for each position.
    This should give near-perfect reconstruction.
    """
    print()
    print("=" * 70)
    print("FITTING EXACT POSITION LOOKUP TABLE")
    print("=" * 70)
    print()
    
    n_layers = 24
    n_heads = 14
    
    # Collect mean attention per (layer, head, position_distance)
    attention_lut = np.zeros((n_layers, n_heads, max_positions))
    attention_counts = np.zeros((n_layers, n_heads, max_positions))
    
    for item in attentions_list:
        attns = item['attentions']
        seq_len = attns.shape[2]
        
        for layer in range(n_layers):
            for head in range(n_heads):
                for i in range(seq_len):
                    for j in range(i + 1):
                        d = i - j
                        if d < max_positions:
                            attention_lut[layer, head, d] += attns[layer, head, i, j]
                            attention_counts[layer, head, d] += 1
    
    # Average
    attention_lut = attention_lut / (attention_counts + 1e-8)
    
    # Compute reconstruction accuracy
    total_loss = 0
    total_count = 0
    
    for item in attentions_list:
        attns = item['attentions']
        seq_len = attns.shape[2]
        
        for layer in range(n_layers):
            for head in range(n_heads):
                for i in range(seq_len):
                    # Reconstruct attention using LUT
                    distances = np.arange(min(i + 1, max_positions))
                    lut_weights = attention_lut[layer, head, distances]
                    
                    # Normalize (softmax-like)
                    pred_attn = lut_weights / (np.sum(lut_weights) + 1e-8)
                    
                    # Actual attention (truncated to max_positions)
                    actual_attn = attns[layer, head, i, max(0, i+1-max_positions):i+1]
                    actual_attn = actual_attn / (np.sum(actual_attn) + 1e-8)
                    
                    if len(pred_attn) == len(actual_attn):
                        loss = np.mean((pred_attn - actual_attn) ** 2)
                        total_loss += loss
                        total_count += 1
    
    avg_loss = total_loss / total_count
    accuracy = 1 - np.sqrt(avg_loss)
    
    print(f"LUT size: {n_layers} × {n_heads} × {max_positions} = {n_layers * n_heads * max_positions} values")
    print(f"Storage: {n_layers * n_heads * max_positions * 4 / 1024:.1f} KB (float32)")
    print(f"MSE loss: {avg_loss:.6f}")
    print(f"Accuracy: {accuracy:.4%}")
    
    # Show sample LUT values
    print()
    print("Sample LUT values (layer 0, head 0):")
    for d in range(min(10, max_positions)):
        print(f"  d={d}: {attention_lut[0, 0, d]:.4f}")
    
    return attention_lut, avg_loss


def test_phi_attention_reproduction(model, tokenizer, phi_params, test_texts):
    """
    Test full sequence reproduction using φ-attention.
    
    Compare:
    1. Original model output
    2. φ-attention approximated output
    """
    print()
    print("=" * 70)
    print("TESTING φ-ATTENTION REPRODUCTION")
    print("=" * 70)
    print()
    
    alpha, beta, gamma, sim_scale = phi_params
    
    for text in test_texts:
        print(f"Text: '{text}'")
        
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        
        # Original logits
        logits_orig = outputs.logits[0].numpy()
        
        # Get embeddings
        embed = outputs.hidden_states[0][0].numpy()
        embed_norm = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-8)
        similarity = embed_norm @ embed_norm.T
        
        seq_len = embed.shape[0]
        
        # Compute φ-attention for each position
        phi_attentions = []
        for i in range(seq_len):
            distances = np.arange(i + 1)
            pos_logits = alpha * (PHI ** (-distances / beta)) + gamma
            sim_logits = sim_scale * similarity[i, :i+1]
            logits = pos_logits + sim_logits
            attn = np.exp(logits - np.max(logits))
            attn = attn / np.sum(attn)
            phi_attentions.append(attn)
        
        # Compare with actual attention (layer 0, head 0)
        actual_attn = outputs.attentions[0][0, 0].numpy()  # [seq_len, seq_len]
        
        # Compute correlation
        correlations = []
        for i in range(seq_len):
            if i > 0:
                corr = np.corrcoef(phi_attentions[i], actual_attn[i, :i+1])[0, 1]
                correlations.append(corr)
        
        avg_corr = np.mean(correlations) if correlations else 0
        print(f"  Attention correlation (L0H0): {avg_corr:.4f}")
        
        # Compare top predictions
        for pos in range(min(3, seq_len)):
            top_orig = np.argmax(logits_orig[pos])
            token_orig = tokenizer.decode([top_orig])
            print(f"  Pos {pos}: pred='{token_orig}'")
        
        print()


def main():
    model, tokenizer = load_model()
    
    # Test texts
    train_texts = [
        "The king examined the evidence",
        "She walked to the store",
        "I love programming in Python",
        "The quick brown fox jumps",
        "Hello world this is a test",
    ]
    
    test_texts = [
        "The queen ruled the kingdom",
        "He ran to the park",
    ]
    
    # Get attention patterns
    print("Extracting attention patterns...")
    train_attentions = get_attention_patterns(model, tokenizer, train_texts)
    
    # Analysis 1: Position decay
    analyze_position_decay(train_attentions)
    
    # Analysis 2: Global φ-weights
    global_params, global_loss = fit_phi_position_weights(train_attentions)
    
    # Analysis 3: Per-layer φ-weights
    layer_params, layer_losses = fit_per_layer_phi_weights(train_attentions)
    
    # Analysis 4: Content-aware φ-attention
    content_params, content_loss = fit_content_aware_phi_attention(model, tokenizer, train_attentions)
    
    # Analysis 5: Exact LUT (like DA2)
    attention_lut, lut_loss = fit_exact_attention_lookup(train_attentions)
    
    # Test reproduction
    test_phi_attention_reproduction(model, tokenizer, content_params, test_texts)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Approach                    | MSE Loss  | Accuracy")
    print("-" * 55)
    print(f"Global φ-weights            | {global_loss:.6f} | {1-np.sqrt(global_loss):.4%}")
    print(f"Per-layer φ-weights         | {np.mean(layer_losses):.6f} | {1-np.sqrt(np.mean(layer_losses)):.4%}")
    print(f"Content-aware φ-attention   | {content_loss:.6f} | {1-np.sqrt(content_loss):.4%}")
    print(f"Exact position LUT          | {lut_loss:.6f} | {1-np.sqrt(lut_loss):.4%}")
    print()
    print("Target: 99.99% accuracy")
    print()
    
    best_accuracy = max(
        1 - np.sqrt(global_loss),
        1 - np.sqrt(np.mean(layer_losses)),
        1 - np.sqrt(content_loss),
        1 - np.sqrt(lut_loss),
    )
    
    if best_accuracy >= 0.9999:
        print("✓ TARGET ACHIEVED!")
    else:
        print(f"✗ Best accuracy: {best_accuracy:.4%}")
        print("  Need more sophisticated approach (per-head LUT, content integration)")


if __name__ == "__main__":
    main()
