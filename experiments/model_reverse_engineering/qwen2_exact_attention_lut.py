#!/usr/bin/env python3
"""
Qwen2.0 Exact Attention LUT
============================

The key insight: To get 99.99% accuracy, we need to store the
EXACT attention patterns, not approximations.

Strategy:
1. For each (layer, head), store the attention weights directly
2. Index by token embedding similarity
3. Use nearest-neighbor lookup at inference

This is similar to how DA2 stored the exact error corrections.

The question is: How much storage do we need?
- 24 layers × 14 heads × N patterns × seq_len² values
- Can we compress using φ-structure?
"""

import torch
import numpy as np
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def load_model():
    """Load Qwen2-0.5B model."""
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


def extract_attention_database(model, tokenizer, texts):
    """
    Extract exact attention patterns for a set of texts.
    
    Returns a database indexed by token embeddings.
    """
    print()
    print("=" * 70)
    print("EXTRACTING ATTENTION DATABASE")
    print("=" * 70)
    print()
    
    # Database: for each (layer, head, query_pos), store (embedding, attention_row)
    database = defaultdict(list)
    
    for text_idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        
        # Get embeddings at each layer
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        
        seq_len = inputs['input_ids'].shape[1]
        
        # For layer 0
        layer_idx = 0
        hidden = hidden_states[layer_idx][0].numpy()  # [seq_len, 896]
        attn = attentions[layer_idx][0].numpy()  # [n_heads, seq_len, seq_len]
        
        n_heads = attn.shape[0]
        
        for h in range(n_heads):
            for i in range(seq_len):
                # Store: (query_embedding, attention_row)
                query_embed = hidden[i]
                attn_row = attn[h, i, :i+1]  # Only causal part
                
                database[(layer_idx, h, i)].append({
                    'embed': query_embed,
                    'attn': attn_row,
                    'text_idx': text_idx,
                })
        
        if text_idx % 10 == 0:
            print(f"  Processed {text_idx + 1}/{len(texts)} texts")
    
    print(f"Database entries: {sum(len(v) for v in database.values())}")
    
    return database


def lookup_attention(database, query_embed, layer, head, pos):
    """
    Look up attention pattern by finding nearest embedding.
    """
    key = (layer, head, pos)
    
    if key not in database or len(database[key]) == 0:
        return None
    
    entries = database[key]
    
    # Find nearest embedding
    best_sim = -1
    best_attn = None
    
    for entry in entries:
        sim = np.dot(query_embed, entry['embed']) / (
            np.linalg.norm(query_embed) * np.linalg.norm(entry['embed']) + 1e-8
        )
        
        if sim > best_sim:
            best_sim = sim
            best_attn = entry['attn']
    
    return best_attn, best_sim


def test_exact_lookup(model, tokenizer, database, test_texts):
    """
    Test attention reproduction using exact lookup.
    """
    print()
    print("=" * 70)
    print("TESTING EXACT LOOKUP")
    print("=" * 70)
    print()
    
    all_mses = []
    all_sims = []
    
    for text in test_texts:
        print(f"Text: '{text}'")
        
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        
        hidden = outputs.hidden_states[0][0].numpy()
        actual_attn = outputs.attentions[0][0].numpy()
        
        seq_len = hidden.shape[0]
        n_heads = actual_attn.shape[0]
        
        for h in range(min(3, n_heads)):
            head_mses = []
            head_sims = []
            
            for i in range(seq_len):
                result = lookup_attention(database, hidden[i], 0, h, i)
                
                if result is not None:
                    looked_up_attn, sim = result
                    
                    # Pad to match length
                    actual_row = actual_attn[h, i, :i+1]
                    
                    if len(looked_up_attn) == len(actual_row):
                        mse = np.mean((looked_up_attn - actual_row) ** 2)
                        head_mses.append(mse)
                        head_sims.append(sim)
            
            if head_mses:
                avg_mse = np.mean(head_mses)
                avg_sim = np.mean(head_sims)
                all_mses.append(avg_mse)
                all_sims.append(avg_sim)
                
                print(f"  Head {h}: mse={avg_mse:.6f}, avg_sim={avg_sim:.4f}")
        
        print()
    
    if all_mses:
        overall_mse = np.mean(all_mses)
        overall_sim = np.mean(all_sims)
        accuracy = 1 - np.sqrt(overall_mse)
        
        print(f"Overall:")
        print(f"  MSE: {overall_mse:.6f}")
        print(f"  Accuracy: {accuracy:.4%}")
        print(f"  Avg similarity: {overall_sim:.4f}")
        
        return accuracy
    
    return 0


def test_same_text_lookup(model, tokenizer, texts):
    """
    Test: If we look up the SAME text, do we get 100% accuracy?
    
    This verifies the lookup mechanism works.
    """
    print()
    print("=" * 70)
    print("TESTING SAME-TEXT LOOKUP (should be 100%)")
    print("=" * 70)
    print()
    
    # Build database from texts
    database = extract_attention_database(model, tokenizer, texts[:5])
    
    # Test on same texts
    all_mses = []
    
    for text in texts[:5]:
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        
        hidden = outputs.hidden_states[0][0].numpy()
        actual_attn = outputs.attentions[0][0].numpy()
        
        seq_len = hidden.shape[0]
        
        for h in range(3):
            for i in range(seq_len):
                result = lookup_attention(database, hidden[i], 0, h, i)
                
                if result is not None:
                    looked_up_attn, sim = result
                    actual_row = actual_attn[h, i, :i+1]
                    
                    if len(looked_up_attn) == len(actual_row):
                        mse = np.mean((looked_up_attn - actual_row) ** 2)
                        all_mses.append(mse)
    
    if all_mses:
        avg_mse = np.mean(all_mses)
        accuracy = 1 - np.sqrt(avg_mse)
        
        print(f"Same-text accuracy: {accuracy:.6%}")
        
        if accuracy > 0.9999:
            print("✓ Perfect lookup confirmed!")
        
        return accuracy
    
    return 0


def compute_storage_requirements(database):
    """
    Compute storage requirements for the attention database.
    """
    print()
    print("=" * 70)
    print("STORAGE REQUIREMENTS")
    print("=" * 70)
    print()
    
    total_entries = sum(len(v) for v in database.values())
    
    # Each entry: embedding (896 floats) + attention row (variable, avg ~10 floats)
    embed_storage = total_entries * 896 * 4 / 1024 / 1024  # MB
    attn_storage = total_entries * 10 * 4 / 1024 / 1024  # MB (estimate)
    
    print(f"Total entries: {total_entries}")
    print(f"Embedding storage: {embed_storage:.1f} MB")
    print(f"Attention storage: {attn_storage:.1f} MB")
    print(f"Total: {embed_storage + attn_storage:.1f} MB")
    
    # For full model (24 layers × 14 heads)
    full_storage = (embed_storage + attn_storage) * 24 * 14 / 14  # Scale from 14 heads to all
    print(f"Estimated full model: {full_storage:.1f} MB")
    
    return total_entries


def test_interpolated_lookup(model, tokenizer, database, test_texts, k=3):
    """
    Use k-nearest neighbors interpolation for lookup.
    """
    print()
    print("=" * 70)
    print(f"TESTING INTERPOLATED LOOKUP (k={k})")
    print("=" * 70)
    print()
    
    all_mses = []
    
    for text in test_texts:
        print(f"Text: '{text}'")
        
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        
        hidden = outputs.hidden_states[0][0].numpy()
        actual_attn = outputs.attentions[0][0].numpy()
        
        seq_len = hidden.shape[0]
        n_heads = actual_attn.shape[0]
        
        for h in range(min(3, n_heads)):
            head_mses = []
            
            for i in range(seq_len):
                key = (0, h, i)
                
                if key not in database or len(database[key]) == 0:
                    continue
                
                entries = database[key]
                
                # Find k nearest
                sims = []
                for entry in entries:
                    sim = np.dot(hidden[i], entry['embed']) / (
                        np.linalg.norm(hidden[i]) * np.linalg.norm(entry['embed']) + 1e-8
                    )
                    sims.append((sim, entry['attn']))
                
                sims.sort(key=lambda x: -x[0])
                top_k = sims[:k]
                
                # Weighted average
                total_weight = sum(s[0] for s in top_k)
                if total_weight > 0:
                    # Find common length
                    actual_row = actual_attn[h, i, :i+1]
                    target_len = len(actual_row)
                    
                    # Only use entries with matching length
                    matching = [(s, a) for s, a in top_k if len(a) == target_len]
                    
                    if matching:
                        total_weight = sum(s for s, a in matching)
                        interpolated = sum(s * a for s, a in matching) / total_weight
                        
                        mse = np.mean((interpolated - actual_row) ** 2)
                        head_mses.append(mse)
            
            if head_mses:
                avg_mse = np.mean(head_mses)
                all_mses.append(avg_mse)
                print(f"  Head {h}: mse={avg_mse:.6f}")
        
        print()
    
    if all_mses:
        overall_mse = np.mean(all_mses)
        accuracy = 1 - np.sqrt(overall_mse)
        
        print(f"Overall interpolated accuracy: {accuracy:.4%}")
        
        return accuracy
    
    return 0


def main():
    model, tokenizer = load_model()
    
    # Training texts
    train_texts = [
        "The king examined the evidence carefully",
        "She walked slowly to the old store",
        "Hello world this is a test message",
        "I love programming in Python language",
        "The quick brown fox jumps over",
        "A beautiful day in the park today",
        "He said that she was very happy",
        "They went to the beach yesterday",
        "The cat sat on the warm mat",
        "We need to find a good solution",
        "The queen ruled the kingdom wisely",
        "He ran quickly to the new park",
        "The dog barked at the mailman",
        "She sang a beautiful song today",
        "The sun was shining very brightly",
    ]
    
    # Test texts
    test_texts = [
        "The prince walked to the castle",
        "She danced gracefully on stage",
        "Goodbye world",
    ]
    
    # Test same-text lookup (should be 100%)
    test_same_text_lookup(model, tokenizer, train_texts)
    
    # Build database
    database = extract_attention_database(model, tokenizer, train_texts)
    
    # Compute storage
    compute_storage_requirements(database)
    
    # Test exact lookup
    accuracy_exact = test_exact_lookup(model, tokenizer, database, test_texts)
    
    # Test interpolated lookup
    accuracy_interp = test_interpolated_lookup(model, tokenizer, database, test_texts, k=3)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Same-text lookup: Should be 100% (verifies mechanism)")
    print(f"Exact lookup (nearest): {accuracy_exact:.4%}")
    print(f"Interpolated lookup (k=3): {accuracy_interp:.4%}")
    print()
    print("To achieve 99.99%:")
    print("1. Need more training data covering more patterns")
    print("2. Or: Store attention weights directly (large storage)")
    print("3. Or: Use AIG to compress the error patterns")


if __name__ == "__main__":
    main()
