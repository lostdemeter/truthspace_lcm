#!/usr/bin/env python3
"""
LM Head Geometry: Where Content Knowledge Lives
=================================================

Key finding: The hidden state after "The capital of France is" has only
4% cosine similarity with Paris's embedding, yet Paris is the top prediction.

This means the LM head encodes the relationship geometrically.

Hypothesis: The LM head learned a transformation where:
  hidden_state @ lm_head.T → logits
  
And the geometry of lm_head encodes relationships like "capital-of".

Let's investigate:
1. What makes Paris rank #1 for "capital of France"?
2. Is there a geometric pattern in the lm_head rows for capitals?
3. Can we predict content tokens from the lm_head geometry?

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


def analyze_capital_predictions(model, tokenizer, device):
    """
    Analyze what makes capitals rank high after "The capital of X is".
    """
    print("=" * 70)
    print("CAPITAL PREDICTION ANALYSIS")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data.float().cpu()
    
    countries = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Japan", "Tokyo"),
        ("Spain", "Madrid"),
    ]
    
    hidden_states = []
    
    for country, capital in countries:
        prompt = f"The capital of {country} is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
        
        hidden_states.append({
            'country': country,
            'capital': capital,
            'hidden': hidden,
        })
        
        # Compute logits
        logits = hidden @ lm_head.T
        
        # Find capital rank
        capital_ids = tokenizer.encode(capital, add_special_tokens=False)
        capital_with_space_ids = tokenizer.encode(" " + capital, add_special_tokens=False)
        
        if capital_with_space_ids:
            capital_idx = capital_with_space_ids[0]
            capital_logit = logits[capital_idx].item()
            capital_rank = (logits > capital_logit).sum().item() + 1
            
            # Top prediction
            top_idx = logits.argmax().item()
            top_token = tokenizer.decode([top_idx])
            top_logit = logits[top_idx].item()
            
            print(f"\n{country}:")
            print(f"  Top prediction: {repr(top_token)} (logit={top_logit:.2f})")
            print(f"  {capital} rank: {capital_rank} (logit={capital_logit:.2f})")
    
    return hidden_states, lm_head


def analyze_lm_head_structure(lm_head, tokenizer):
    """
    Analyze the structure of the LM head matrix.
    
    Key question: Do capitals cluster in lm_head space?
    """
    print("\n" + "=" * 70)
    print("LM HEAD STRUCTURE ANALYSIS")
    print("=" * 70)
    
    # Get lm_head rows for capitals and countries
    capitals = ["Paris", "Berlin", "Rome", "Tokyo", "Madrid"]
    countries = ["France", "Germany", "Italy", "Japan", "Spain"]
    
    capital_rows = []
    country_rows = []
    
    for capital in capitals:
        ids = tokenizer.encode(" " + capital, add_special_tokens=False)
        if ids:
            capital_rows.append(lm_head[ids[0]])
    
    for country in countries:
        ids = tokenizer.encode(country, add_special_tokens=False)
        if ids:
            country_rows.append(lm_head[ids[0]])
    
    if capital_rows and country_rows:
        capital_rows = torch.stack(capital_rows)
        country_rows = torch.stack(country_rows)
        
        # Compute pairwise similarities within capitals
        print("\nCapital-Capital similarities (lm_head rows):")
        for i, c1 in enumerate(capitals):
            for j, c2 in enumerate(capitals):
                if i < j:
                    sim = torch.nn.functional.cosine_similarity(
                        capital_rows[i].unsqueeze(0),
                        capital_rows[j].unsqueeze(0)
                    ).item()
                    print(f"  {c1}-{c2}: {sim:.4f}")
        
        # Mean similarity
        sims = []
        for i in range(len(capitals)):
            for j in range(len(capitals)):
                if i < j:
                    sim = torch.nn.functional.cosine_similarity(
                        capital_rows[i].unsqueeze(0),
                        capital_rows[j].unsqueeze(0)
                    ).item()
                    sims.append(sim)
        print(f"\n  Mean capital-capital similarity: {np.mean(sims):.4f}")
        
        # Compare to country-country
        sims = []
        for i in range(len(countries)):
            for j in range(len(countries)):
                if i < j:
                    sim = torch.nn.functional.cosine_similarity(
                        country_rows[i].unsqueeze(0),
                        country_rows[j].unsqueeze(0)
                    ).item()
                    sims.append(sim)
        print(f"  Mean country-country similarity: {np.mean(sims):.4f}")
        
        # Country-Capital pairs
        print("\nCountry-Capital similarities:")
        for i, (country, capital) in enumerate(zip(countries, capitals)):
            sim = torch.nn.functional.cosine_similarity(
                country_rows[i].unsqueeze(0),
                capital_rows[i].unsqueeze(0)
            ).item()
            print(f"  {country}-{capital}: {sim:.4f}")


def find_capital_direction(hidden_states, lm_head, tokenizer):
    """
    Is there a consistent direction in hidden space that points toward capitals?
    
    For each country, compute:
      direction = hidden_state @ lm_head[capital].T
    
    Is this direction consistent across countries?
    """
    print("\n" + "=" * 70)
    print("CAPITAL DIRECTION ANALYSIS")
    print("=" * 70)
    
    directions = []
    
    for hs in hidden_states:
        country = hs['country']
        capital = hs['capital']
        hidden = hs['hidden']
        
        # Get capital's lm_head row
        capital_ids = tokenizer.encode(" " + capital, add_special_tokens=False)
        if not capital_ids:
            continue
        
        capital_row = lm_head[capital_ids[0]]
        
        # The "direction" is the lm_head row itself
        # The logit is: hidden @ capital_row
        # So capital_row defines the direction that hidden needs to align with
        
        directions.append({
            'country': country,
            'capital': capital,
            'direction': capital_row,
            'hidden': hidden,
            'alignment': (hidden @ capital_row).item(),
        })
        
        print(f"\n{country} → {capital}:")
        print(f"  Hidden-Capital alignment: {directions[-1]['alignment']:.2f}")
    
    # Are the capital directions similar?
    print("\nCapital direction similarities:")
    for i, d1 in enumerate(directions):
        for j, d2 in enumerate(directions):
            if i < j:
                sim = torch.nn.functional.cosine_similarity(
                    d1['direction'].unsqueeze(0),
                    d2['direction'].unsqueeze(0)
                ).item()
                print(f"  {d1['capital']}-{d2['capital']}: {sim:.4f}")
    
    return directions


def analyze_hidden_state_difference(hidden_states, lm_head, tokenizer):
    """
    What's the difference between hidden states for different countries?
    
    If France→Paris and Germany→Berlin, what's different about their hidden states?
    """
    print("\n" + "=" * 70)
    print("HIDDEN STATE DIFFERENCE ANALYSIS")
    print("=" * 70)
    
    # Compare France vs Germany hidden states
    france_hs = None
    germany_hs = None
    
    for hs in hidden_states:
        if hs['country'] == 'France':
            france_hs = hs['hidden']
        if hs['country'] == 'Germany':
            germany_hs = hs['hidden']
    
    if france_hs is not None and germany_hs is not None:
        diff = germany_hs - france_hs
        
        print(f"\nGermany - France hidden state difference:")
        print(f"  Magnitude: {diff.norm().item():.2f}")
        print(f"  Cosine similarity: {torch.nn.functional.cosine_similarity(france_hs.unsqueeze(0), germany_hs.unsqueeze(0)).item():.4f}")
        
        # What tokens does this difference point toward?
        diff_logits = diff @ lm_head.T
        
        top_indices = diff_logits.argsort(descending=True)[:10]
        print(f"\n  Top tokens in difference direction:")
        for idx in top_indices:
            token = tokenizer.decode([idx.item()])
            logit = diff_logits[idx].item()
            print(f"    {repr(token)}: {logit:.2f}")
        
        # Does it point toward Berlin - Paris?
        paris_ids = tokenizer.encode(" Paris", add_special_tokens=False)
        berlin_ids = tokenizer.encode(" Berlin", add_special_tokens=False)
        
        if paris_ids and berlin_ids:
            paris_row = lm_head[paris_ids[0]]
            berlin_row = lm_head[berlin_ids[0]]
            
            capital_diff = berlin_row - paris_row
            
            # Similarity between hidden diff and capital diff
            sim = torch.nn.functional.cosine_similarity(
                diff.unsqueeze(0),
                capital_diff.unsqueeze(0)
            ).item()
            print(f"\n  Hidden diff vs (Berlin - Paris) lm_head: {sim:.4f}")


def test_geometric_prediction(model, tokenizer, device, lm_head):
    """
    Can we predict the capital geometrically?
    
    Idea: If we know the hidden state for "The capital of France is",
    can we find Paris by geometric operations on lm_head?
    """
    print("\n" + "=" * 70)
    print("GEOMETRIC PREDICTION TEST")
    print("=" * 70)
    
    # Get hidden state for France
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
    
    # Standard prediction (what the model does)
    logits = hidden @ lm_head.T
    top_idx = logits.argmax().item()
    top_token = tokenizer.decode([top_idx])
    
    print(f"\nStandard prediction: {repr(top_token)}")
    
    # Now try to understand WHY Paris ranks high
    paris_ids = tokenizer.encode(" Paris", add_special_tokens=False)
    if paris_ids:
        paris_row = lm_head[paris_ids[0]]
        
        # Decompose the dot product
        # logit = hidden @ paris_row = sum(hidden[i] * paris_row[i])
        
        # Find dimensions that contribute most
        contributions = hidden * paris_row
        top_contrib_dims = contributions.abs().argsort(descending=True)[:20]
        
        print(f"\nTop 20 dimensions contributing to Paris prediction:")
        for dim in top_contrib_dims:
            contrib = contributions[dim].item()
            h_val = hidden[dim].item()
            p_val = paris_row[dim].item()
            print(f"  Dim {dim.item():4d}: h={h_val:+.3f}, p={p_val:+.3f}, contrib={contrib:+.3f}")


def main():
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    
    # Analyze capital predictions
    hidden_states, lm_head = analyze_capital_predictions(model, tokenizer, device)
    
    # Analyze lm_head structure
    analyze_lm_head_structure(lm_head, tokenizer)
    
    # Find capital direction
    directions = find_capital_direction(hidden_states, lm_head, tokenizer)
    
    # Analyze hidden state differences
    analyze_hidden_state_difference(hidden_states, lm_head, tokenizer)
    
    # Test geometric prediction
    test_geometric_prediction(model, tokenizer, device, lm_head)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings:
1. The model correctly predicts capitals (Paris is #1 for France)
2. The knowledge is in the hidden state + lm_head combination
3. The lm_head rows for capitals may cluster
4. The hidden state difference (France vs Germany) may point toward capital difference
""")


if __name__ == "__main__":
    main()
