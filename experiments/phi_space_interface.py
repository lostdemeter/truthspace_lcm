#!/usr/bin/env python3
"""
φ-Space Interface Prototype

A terminal-based interface for navigating knowledge geometry.
Demonstrates the core concepts before building a full GUI.

Components:
1. Position Tracker - Shows current position in φ-space
2. Bottleneck Visualizer - Shows layer 27 convergence
3. Discovery Mode - Navigate to unexplored regions
4. Real-time Feedback - See geometry as you type
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
import json

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


class PhiSpaceInterface:
    """Terminal interface for φ-space navigation."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.position_history = []
        self.discoveries = []
        
        # Cache embeddings
        self.embeddings = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        self.lm_head = model.lm_head.weight.detach().float().cpu().numpy()
        
        # Current state
        self.current_position = None
        self.current_trajectory = None
        
    def get_phi_level(self, hidden: np.ndarray) -> float:
        """Compute φ-level of a hidden state."""
        mags = np.abs(hidden)
        mags = mags[mags > 1e-10]
        return float(np.mean(np.log(mags) / LOG_PHI))
    
    def get_trajectory(self, text: str) -> List[np.ndarray]:
        """Get hidden state trajectory for text."""
        inputs = self.tokenizer(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
        return [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    
    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    # =========================================================
    # CORE OPERATIONS
    # =========================================================
    
    def query(self, text: str) -> Dict:
        """
        QUERY: Navigate to a position and see what's there.
        
        Returns position info, bottleneck state, and nearby concepts.
        """
        # Get trajectory
        trajectory = self.get_trajectory(text)
        self.current_trajectory = trajectory
        self.current_position = trajectory[-1]  # Final layer
        
        # Compute φ-levels at key layers
        phi_levels = {
            'input': self.get_phi_level(trajectory[0]),
            'divergence': self.get_phi_level(trajectory[7]),
            'middle': self.get_phi_level(trajectory[14]),
            'bottleneck': self.get_phi_level(trajectory[27]),
            'output': self.get_phi_level(trajectory[28]),
        }
        
        # Find nearest concepts at output layer
        sims = [(i, self.cosine_sim(self.current_position, self.embeddings[i])) 
                for i in range(len(self.embeddings))]
        sims.sort(key=lambda x: -x[1])
        
        nearby = []
        seen = set()
        for idx, sim in sims[:50]:
            token = self.tokenizer.decode([idx]).strip()
            if token and len(token) > 1 and token not in seen:
                nearby.append({'token': token, 'similarity': sim})
                seen.add(token)
            if len(nearby) >= 5:
                break
        
        # Record position
        self.position_history.append({
            'query': text,
            'phi_levels': phi_levels,
            'nearby': nearby
        })
        
        return {
            'query': text,
            'phi_levels': phi_levels,
            'bottleneck_level': phi_levels['bottleneck'],
            'nearby_concepts': nearby,
            'position_norm': float(np.linalg.norm(self.current_position))
        }
    
    def explore(self, direction: str = 'random', steps: int = 5) -> List[Dict]:
        """
        EXPLORE: Move outward from current position.
        
        Directions: 'random', 'high_phi', 'low_phi', 'orthogonal'
        """
        if self.current_position is None:
            return [{'error': 'No current position. Run query first.'}]
        
        explorations = []
        pos = self.current_position.copy()
        
        for step in range(steps):
            # Choose direction
            if direction == 'random':
                delta = np.random.randn(len(pos))
            elif direction == 'high_phi':
                # Move toward higher magnitude
                delta = pos / (np.linalg.norm(pos) + 1e-10)
            elif direction == 'low_phi':
                # Move toward lower magnitude
                delta = -pos / (np.linalg.norm(pos) + 1e-10)
            elif direction == 'orthogonal':
                # Move perpendicular to current position
                delta = np.random.randn(len(pos))
                delta = delta - np.dot(delta, pos) * pos / (np.linalg.norm(pos)**2 + 1e-10)
            
            # Normalize and scale by φ
            delta = delta / (np.linalg.norm(delta) + 1e-10) * PHI
            pos = pos + delta
            
            # Find what's at new position
            sims = [(i, self.cosine_sim(pos, self.embeddings[i])) 
                    for i in range(len(self.embeddings))]
            sims.sort(key=lambda x: -x[1])
            
            best_token = None
            for idx, sim in sims[:20]:
                token = self.tokenizer.decode([idx]).strip()
                if token and len(token) > 2:
                    best_token = token
                    break
            
            explorations.append({
                'step': step + 1,
                'direction': direction,
                'phi_level': self.get_phi_level(pos),
                'nearest_concept': best_token,
                'similarity': sims[0][1] if sims else 0
            })
        
        return explorations
    
    def bridge(self, concept1: str, concept2: str, steps: int = 5) -> List[Dict]:
        """
        BRIDGE: Find the path between two concepts.
        """
        # Get embeddings
        tok1 = self.tokenizer.encode(concept1, add_special_tokens=False)
        tok2 = self.tokenizer.encode(concept2, add_special_tokens=False)
        
        if not tok1 or not tok2:
            return [{'error': 'Could not find embeddings for concepts'}]
        
        emb1 = self.embeddings[tok1[0]]
        emb2 = self.embeddings[tok2[0]]
        
        # Interpolate between them
        path = []
        for i in range(steps + 1):
            t = i / steps
            pos = (1 - t) * emb1 + t * emb2
            
            # Find nearest concept
            sims = [(j, self.cosine_sim(pos, self.embeddings[j])) 
                    for j in range(len(self.embeddings))]
            sims.sort(key=lambda x: -x[1])
            
            best_token = None
            for idx, sim in sims[:20]:
                token = self.tokenizer.decode([idx]).strip()
                if token and len(token) > 2 and token not in [concept1, concept2]:
                    best_token = token
                    break
            
            path.append({
                'step': i,
                't': t,
                'phi_level': self.get_phi_level(pos),
                'concept': best_token or (concept1 if t < 0.5 else concept2),
                'similarity': sims[0][1]
            })
        
        return path
    
    def discover(self, n_attempts: int = 10) -> List[Dict]:
        """
        DISCOVER: Find unexplored regions (unknown unknowns).
        
        Strategy: Navigate to regions far from known concept clusters.
        """
        # Sample random directions and find sparse regions
        discoveries = []
        
        # Compute centroid of known concepts
        centroid = np.mean(self.embeddings, axis=0)
        
        for attempt in range(n_attempts):
            # Generate random direction
            direction = np.random.randn(len(centroid))
            direction = direction / np.linalg.norm(direction)
            
            # Move away from centroid in this direction
            distance = PHI ** (attempt + 1)  # Increasing distance
            pos = centroid + direction * distance
            
            # Find nearest concept (to measure "sparseness")
            sims = [(i, self.cosine_sim(pos, self.embeddings[i])) 
                    for i in range(len(self.embeddings))]
            sims.sort(key=lambda x: -x[1])
            
            best_sim = sims[0][1]
            sparseness = 1 - best_sim  # Higher = more unexplored
            
            # Get nearest tokens
            nearby_tokens = []
            for idx, sim in sims[:5]:
                token = self.tokenizer.decode([idx]).strip()
                if token and len(token) > 1:
                    nearby_tokens.append(token)
            
            discoveries.append({
                'attempt': attempt + 1,
                'distance_from_centroid': distance,
                'sparseness': sparseness,
                'phi_level': self.get_phi_level(pos),
                'nearest_tokens': nearby_tokens[:3],
                'best_similarity': best_sim
            })
        
        # Sort by sparseness (most unexplored first)
        discoveries.sort(key=lambda x: -x['sparseness'])
        
        # Record top discoveries
        self.discoveries.extend(discoveries[:3])
        
        return discoveries
    
    def visualize_bottleneck(self, text: str) -> Dict:
        """
        BOTTLENECK VISUALIZER: Show how thought passes through layer 27.
        """
        trajectory = self.get_trajectory(text)
        
        # Get states at key layers
        layers = [0, 7, 14, 21, 27, 28]
        states = []
        
        for layer in layers:
            hidden = trajectory[layer]
            phi_level = self.get_phi_level(hidden)
            
            # Project to vocabulary to see "what it's thinking"
            logits = hidden @ self.lm_head.T
            top_idx = np.argsort(logits)[-5:][::-1]
            top_tokens = [self.tokenizer.decode([i]).strip() for i in top_idx]
            
            states.append({
                'layer': layer,
                'phi_level': phi_level,
                'top_predictions': top_tokens,
                'norm': float(np.linalg.norm(hidden))
            })
        
        # Compute convergence at bottleneck
        bottleneck_state = trajectory[27]
        
        return {
            'input': text,
            'layer_states': states,
            'bottleneck_phi': self.get_phi_level(bottleneck_state),
            'bottleneck_norm': float(np.linalg.norm(bottleneck_state)),
            'phi_target': PHI,
            'convergence_quality': 1 - abs(self.get_phi_level(bottleneck_state) - PHI) / PHI
        }
    
    def real_time_feedback(self, text: str) -> List[Dict]:
        """
        REAL-TIME FEEDBACK: Show position as each character is typed.
        """
        feedback = []
        
        # Build up text character by character (by words for efficiency)
        words = text.split()
        current = ""
        
        for i, word in enumerate(words):
            current = " ".join(words[:i+1])
            
            # Get trajectory for current text
            traj = self.get_trajectory(current)
            
            phi_27 = self.get_phi_level(traj[27])
            phi_7 = self.get_phi_level(traj[7])
            
            # Predict where we're heading
            logits = traj[-1] @ self.lm_head.T
            top_idx = np.argmax(logits)
            predicted_next = self.tokenizer.decode([top_idx]).strip()
            
            feedback.append({
                'text_so_far': current,
                'word_count': i + 1,
                'phi_divergence': phi_7,
                'phi_bottleneck': phi_27,
                'delta': phi_27 - phi_7,
                'predicted_next': predicted_next
            })
        
        return feedback
    
    # =========================================================
    # DISPLAY METHODS
    # =========================================================
    
    def display_query_result(self, result: Dict):
        """Pretty print query result."""
        print("\n" + "="*60)
        print(f"QUERY: {result['query']}")
        print("="*60)
        
        print("\nφ-LEVELS THROUGH LAYERS:")
        for layer, level in result['phi_levels'].items():
            bar = "█" * int((level + 12) * 2)  # Scale for display
            print(f"  {layer:12s}: {level:+7.3f} {bar}")
        
        print(f"\nBOTTLENECK (Layer 27): φ^{result['bottleneck_level']:.3f}")
        print(f"  Distance from φ: {abs(result['bottleneck_level'] - PHI):.4f}")
        
        print("\nNEARBY CONCEPTS:")
        for c in result['nearby_concepts']:
            print(f"  • {c['token']:20s} (sim={c['similarity']:.3f})")
    
    def display_bottleneck(self, result: Dict):
        """Pretty print bottleneck visualization."""
        print("\n" + "="*60)
        print(f"BOTTLENECK VISUALIZER: {result['input']}")
        print("="*60)
        
        print("\nLAYER PROGRESSION:")
        for state in result['layer_states']:
            layer = state['layer']
            phi = state['phi_level']
            preds = state['top_predictions'][:3]
            
            # Visual indicator
            if layer == 27:
                marker = " ← BOTTLENECK"
            elif layer == 7:
                marker = " ← DIVERGENCE"
            else:
                marker = ""
            
            print(f"  Layer {layer:2d}: φ^{phi:+.2f} → {preds}{marker}")
        
        print(f"\nCONVERGENCE QUALITY: {result['convergence_quality']*100:.1f}%")
    
    def display_discovery(self, discoveries: List[Dict]):
        """Pretty print discovery results."""
        print("\n" + "="*60)
        print("DISCOVERY MODE: UNKNOWN UNKNOWNS")
        print("="*60)
        
        print("\nMOST UNEXPLORED REGIONS:")
        for d in discoveries[:5]:
            sparseness_bar = "░" * int(d['sparseness'] * 20)
            print(f"\n  Region {d['attempt']}:")
            print(f"    Sparseness: {d['sparseness']:.3f} {sparseness_bar}")
            print(f"    φ-level: {d['phi_level']:.3f}")
            print(f"    Nearest concepts: {d['nearest_tokens']}")


def main():
    """Demo the interface."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    
    # Create interface
    interface = PhiSpaceInterface(model, tokenizer)
    
    print("\n" + "="*60)
    print("φ-SPACE INTERFACE DEMO")
    print("="*60)
    
    # Demo 1: Query
    print("\n[DEMO 1: QUERY]")
    result = interface.query("What is the nature of consciousness?")
    interface.display_query_result(result)
    
    # Demo 2: Bottleneck Visualizer
    print("\n[DEMO 2: BOTTLENECK VISUALIZER]")
    bottleneck = interface.visualize_bottleneck("The meaning of life is")
    interface.display_bottleneck(bottleneck)
    
    # Demo 3: Bridge
    print("\n[DEMO 3: BRIDGE]")
    print("\nBRIDGE: 'knowledge' → 'wisdom'")
    path = interface.bridge("knowledge", "wisdom", steps=5)
    for p in path:
        print(f"  t={p['t']:.1f}: {p['concept']:15s} (φ={p['phi_level']:.2f})")
    
    # Demo 4: Discovery Mode
    print("\n[DEMO 4: DISCOVERY MODE]")
    discoveries = interface.discover(n_attempts=10)
    interface.display_discovery(discoveries)
    
    # Demo 5: Real-time Feedback
    print("\n[DEMO 5: REAL-TIME FEEDBACK]")
    print("\nTyping: 'What is the nature of consciousness?'")
    feedback = interface.real_time_feedback("What is the nature of consciousness?")
    for f in feedback:
        print(f"  '{f['text_so_far']}'")
        print(f"    φ-bottleneck: {f['phi_bottleneck']:.3f}, Δ: {f['delta']:.3f}, next: '{f['predicted_next']}'")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("INTERFACE DEMO COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
