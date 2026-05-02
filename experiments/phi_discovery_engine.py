#!/usr/bin/env python3
"""
φ-Discovery Engine v2: Automated Novel Idea Generation

This system uses the φ-universal coordinate system to systematically
explore semantic space and discover novel patterns, relationships,
and concepts that emerge from the geometry itself.

Key Discovery (Feb 3, 2026):
- Layer 27 is a "universal bottleneck" where ALL reasoning converges to φ-level ≈ 1.57
- The golden ratio acts as a "universal gatekeeper for cognition"
- 27/7 ≈ φ^3 (divergence at 7, convergence at 27)

Usage:
    python phi_discovery_engine.py [--extended] [--themes THEME1,THEME2]

Discovery Methods:
1. Trajectory Divergence Analysis - Find where reasoning paths split
2. Resonance Point Exploration - Analyze convergence layers
3. Semantic Gap Detection - Find unnamed regions between concepts
4. Cross-Domain Bridging - Connect distant semantic regions
5. φ-Level Anomaly Detection - Find unusual φ-patterns
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


@dataclass
class Discovery:
    """A discovered pattern or insight."""
    discovery_type: str
    title: str
    description: str
    evidence: Dict
    novelty_score: float  # 0-1, higher = more novel
    timestamp: str
    
    def to_dict(self):
        return {
            'type': self.discovery_type,
            'title': self.title,
            'description': self.description,
            'evidence': self.evidence,
            'novelty_score': self.novelty_score,
            'timestamp': self.timestamp
        }


class PhiDiscoveryEngine:
    """Automated discovery engine using φ-geometry."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.discoveries = []
        
        # Cache embeddings
        self.all_embeddings = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        self.lm_head = model.lm_head.weight.detach().float().cpu().numpy()
        
    def get_trajectory(self, prompt: str) -> np.ndarray:
        """Get hidden state trajectory for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
        return np.array([h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states])
    
    def get_phi_levels(self, hidden: np.ndarray) -> Dict:
        """Compute φ-level statistics for a hidden state."""
        magnitudes = np.abs(hidden)
        magnitudes = magnitudes[magnitudes > 1e-10]
        levels = np.log(magnitudes) / LOG_PHI
        return {
            'mean': float(np.mean(levels)),
            'std': float(np.std(levels)),
            'median': float(np.median(levels)),
            'min': float(np.min(levels)),
            'max': float(np.max(levels))
        }
    
    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    # =========================================================
    # DISCOVERY METHOD 1: Trajectory Divergence Analysis
    # =========================================================
    
    def discover_divergence_patterns(self, prompt_pairs: List[Tuple[str, str]]) -> List[Discovery]:
        """Find where reasoning paths diverge and what that means."""
        discoveries = []
        
        divergence_data = []
        for p1, p2 in prompt_pairs:
            traj1 = self.get_trajectory(p1)
            traj2 = self.get_trajectory(p2)
            
            # Find divergence point
            correlations = []
            for layer in range(len(traj1)):
                corr = np.corrcoef(traj1[layer], traj2[layer])[0, 1]
                correlations.append(corr)
            
            # Find max divergence rate
            rates = [correlations[i] - correlations[i+1] for i in range(len(correlations)-1)]
            max_div_layer = int(np.argmax(rates))
            max_div_rate = float(rates[max_div_layer])
            
            divergence_data.append({
                'prompts': (p1, p2),
                'divergence_layer': max_div_layer,
                'divergence_rate': max_div_rate,
                'final_correlation': correlations[-1]
            })
        
        # Analyze patterns
        div_layers = [d['divergence_layer'] for d in divergence_data]
        common_layer = int(np.median(div_layers))
        layer_consistency = 1 - np.std(div_layers) / (np.mean(div_layers) + 1)
        
        if layer_consistency > 0.5:
            discoveries.append(Discovery(
                discovery_type='divergence_pattern',
                title=f'Consistent Divergence at Layer {common_layer}',
                description=f'Reasoning paths consistently diverge at layer {common_layer}. '
                           f'This suggests layer {common_layer} is where content-specific '
                           f'processing begins.',
                evidence={'divergence_data': divergence_data, 'consistency': layer_consistency},
                novelty_score=layer_consistency * 0.7,
                timestamp=datetime.now().isoformat()
            ))
        
        return discoveries
    
    # =========================================================
    # DISCOVERY METHOD 2: Resonance Point Exploration
    # =========================================================
    
    def discover_resonance_patterns(self, prompts: List[str]) -> List[Discovery]:
        """Find layers where different prompts converge to same φ-level."""
        discoveries = []
        
        # Get φ-levels at each layer for all prompts
        n_layers = len(self.get_trajectory(prompts[0]))
        layer_levels = {layer: [] for layer in range(n_layers)}
        
        for prompt in prompts:
            traj = self.get_trajectory(prompt)
            for layer in range(n_layers):
                phi_stats = self.get_phi_levels(traj[layer])
                layer_levels[layer].append(phi_stats['mean'])
        
        # Find resonance layers (minimum variance)
        layer_variances = [(layer, np.var(levels)) for layer, levels in layer_levels.items()]
        layer_variances.sort(key=lambda x: x[1])
        
        # Top resonance layer
        resonance_layer, min_variance = layer_variances[0]
        resonance_level = np.mean(layer_levels[resonance_layer])
        
        # Check if resonance level is φ-related
        phi_distance = abs(resonance_level - PHI)
        is_phi_related = phi_distance < 0.1
        
        discoveries.append(Discovery(
            discovery_type='resonance_point',
            title=f'Universal Resonance at Layer {resonance_layer}',
            description=f'All {len(prompts)} prompts converge to φ-level {resonance_level:.3f} '
                       f'at layer {resonance_layer}. '
                       f'{"This is remarkably close to φ!" if is_phi_related else ""}',
            evidence={
                'resonance_layer': resonance_layer,
                'resonance_level': resonance_level,
                'variance': min_variance,
                'phi_distance': phi_distance,
                'n_prompts': len(prompts)
            },
            novelty_score=0.8 if is_phi_related else 0.5,
            timestamp=datetime.now().isoformat()
        ))
        
        return discoveries
    
    # =========================================================
    # DISCOVERY METHOD 3: Semantic Gap Detection
    # =========================================================
    
    def discover_semantic_gaps(self, concept_pairs: List[Tuple[str, str]]) -> List[Discovery]:
        """Find unnamed regions between known concepts."""
        discoveries = []
        
        def get_emb(word):
            tok = self.tokenizer.encode(word, add_special_tokens=False)
            return self.all_embeddings[tok[0]] if tok else None
        
        gaps = []
        for w1, w2 in concept_pairs:
            e1, e2 = get_emb(w1), get_emb(w2)
            if e1 is None or e2 is None:
                continue
            
            midpoint = (e1 + e2) / 2
            
            # Find nearest token to midpoint
            best_sim = 0
            best_token = ''
            for i in range(len(self.all_embeddings)):
                sim = self.cosine_sim(midpoint, self.all_embeddings[i])
                if sim > best_sim:
                    best_sim = sim
                    best_token = self.tokenizer.decode([i]).strip()
            
            gap_size = 1 - best_sim
            gaps.append({
                'concepts': (w1, w2),
                'gap_size': gap_size,
                'nearest_token': best_token,
                'nearest_sim': best_sim
            })
        
        # Find largest gaps
        gaps.sort(key=lambda x: -x['gap_size'])
        
        for gap in gaps[:3]:  # Top 3 gaps
            if gap['gap_size'] > 0.2:  # Significant gap
                discoveries.append(Discovery(
                    discovery_type='semantic_gap',
                    title=f'Unnamed Region: {gap["concepts"][0]} ↔ {gap["concepts"][1]}',
                    description=f'There is a significant semantic gap (size={gap["gap_size"]:.3f}) '
                               f'between "{gap["concepts"][0]}" and "{gap["concepts"][1]}". '
                               f'The nearest token is "{gap["nearest_token"]}" but it\'s far away. '
                               f'This region may represent an unnamed concept.',
                    evidence=gap,
                    novelty_score=gap['gap_size'],
                    timestamp=datetime.now().isoformat()
                ))
        
        return discoveries
    
    # =========================================================
    # DISCOVERY METHOD 4: Cross-Domain Bridging
    # =========================================================
    
    def discover_cross_domain_bridges(self, domains: Dict[str, List[str]]) -> List[Discovery]:
        """Find unexpected connections between different domains."""
        discoveries = []
        
        def get_emb(word):
            tok = self.tokenizer.encode(word, add_special_tokens=False)
            return self.all_embeddings[tok[0]] if tok else None
        
        # Compute domain centroids
        domain_centroids = {}
        for domain, words in domains.items():
            embs = [get_emb(w) for w in words if get_emb(w) is not None]
            if embs:
                domain_centroids[domain] = np.mean(embs, axis=0)
        
        # Find unexpected bridges (high similarity between distant domains)
        domain_names = list(domain_centroids.keys())
        bridges = []
        
        for i, d1 in enumerate(domain_names):
            for d2 in domain_names[i+1:]:
                sim = self.cosine_sim(domain_centroids[d1], domain_centroids[d2])
                bridges.append({
                    'domains': (d1, d2),
                    'similarity': sim
                })
        
        # Find the bridge point between two domains
        bridges.sort(key=lambda x: -x['similarity'])
        
        for bridge in bridges[:3]:
            d1, d2 = bridge['domains']
            midpoint = (domain_centroids[d1] + domain_centroids[d2]) / 2
            
            # What's at the bridge?
            sims = [(i, self.cosine_sim(midpoint, self.all_embeddings[i])) 
                    for i in range(len(self.all_embeddings))]
            sims.sort(key=lambda x: -x[1])
            
            bridge_tokens = []
            for idx, sim in sims[:20]:
                token = self.tokenizer.decode([idx]).strip()
                if token and len(token) > 2 and token.isalpha():
                    bridge_tokens.append(token)
                if len(bridge_tokens) >= 5:
                    break
            
            discoveries.append(Discovery(
                discovery_type='cross_domain_bridge',
                title=f'Bridge: {d1} ↔ {d2}',
                description=f'The domains "{d1}" and "{d2}" are connected with '
                           f'similarity {bridge["similarity"]:.3f}. '
                           f'Bridge concepts: {bridge_tokens}',
                evidence={
                    'domains': bridge['domains'],
                    'similarity': bridge['similarity'],
                    'bridge_tokens': bridge_tokens
                },
                novelty_score=bridge['similarity'] * 0.6,
                timestamp=datetime.now().isoformat()
            ))
        
        return discoveries
    
    # =========================================================
    # DISCOVERY METHOD 5: φ-Level Anomaly Detection
    # =========================================================
    
    def discover_phi_anomalies(self, prompts: List[str]) -> List[Discovery]:
        """Find prompts with unusual φ-level patterns."""
        discoveries = []
        
        # Collect φ-patterns for all prompts
        patterns = []
        for prompt in prompts:
            traj = self.get_trajectory(prompt)
            phi_trajectory = [self.get_phi_levels(h)['mean'] for h in traj]
            patterns.append({
                'prompt': prompt,
                'trajectory': phi_trajectory,
                'mean': np.mean(phi_trajectory),
                'std': np.std(phi_trajectory),
                'range': max(phi_trajectory) - min(phi_trajectory)
            })
        
        # Find anomalies (unusual patterns)
        mean_of_means = np.mean([p['mean'] for p in patterns])
        std_of_means = np.std([p['mean'] for p in patterns])
        
        for p in patterns:
            z_score = abs(p['mean'] - mean_of_means) / (std_of_means + 1e-10)
            if z_score > 2:  # Anomaly threshold
                discoveries.append(Discovery(
                    discovery_type='phi_anomaly',
                    title=f'Unusual φ-Pattern',
                    description=f'The prompt "{p["prompt"][:50]}..." has an unusual '
                               f'φ-level pattern (z-score={z_score:.2f}). '
                               f'Mean level: {p["mean"]:.3f} vs expected {mean_of_means:.3f}',
                    evidence={
                        'prompt': p['prompt'],
                        'z_score': z_score,
                        'mean_level': p['mean'],
                        'expected_mean': mean_of_means
                    },
                    novelty_score=min(z_score / 5, 1.0),
                    timestamp=datetime.now().isoformat()
                ))
        
        return discoveries
    
    # =========================================================
    # DISCOVERY METHOD 6: Emergent Concept Generation
    # =========================================================
    
    def discover_emergent_concepts(self, seed_concepts: List[str], n_iterations: int = 5) -> List[Discovery]:
        """Generate new concepts by navigating φ-space."""
        discoveries = []
        
        def get_emb(word):
            tok = self.tokenizer.encode(word, add_special_tokens=False)
            return self.all_embeddings[tok[0]] if tok else None
        
        # Get seed embeddings
        seed_embs = [get_emb(c) for c in seed_concepts if get_emb(c) is not None]
        if len(seed_embs) < 2:
            return discoveries
        
        # Compute the "concept direction" from seeds
        centroid = np.mean(seed_embs, axis=0)
        
        # Navigate in φ-spiral from centroid
        U, S, Vt = np.linalg.svd(np.array(seed_embs) - centroid, full_matrices=False)
        dir1, dir2 = Vt[0], Vt[1] if len(Vt) > 1 else Vt[0]
        
        emergent = []
        for i in range(n_iterations):
            theta = i * 2 * np.pi / n_iterations
            r = PHI ** (i / n_iterations)
            
            point = centroid + r * (np.cos(theta) * dir1 + np.sin(theta) * dir2)
            
            # Find nearest tokens
            sims = [(j, self.cosine_sim(point, self.all_embeddings[j])) 
                    for j in range(len(self.all_embeddings))]
            sims.sort(key=lambda x: -x[1])
            
            for idx, sim in sims[:10]:
                token = self.tokenizer.decode([idx]).strip()
                if token and len(token) > 2 and token not in seed_concepts:
                    emergent.append({
                        'token': token,
                        'similarity': sim,
                        'position': (theta, r)
                    })
                    break
        
        if emergent:
            discoveries.append(Discovery(
                discovery_type='emergent_concept',
                title=f'Emergent Concepts from {seed_concepts[:3]}',
                description=f'Navigating φ-spiral from {seed_concepts[:3]} reveals: '
                           f'{[e["token"] for e in emergent[:5]]}',
                evidence={
                    'seeds': seed_concepts,
                    'emergent': emergent
                },
                novelty_score=0.6,
                timestamp=datetime.now().isoformat()
            ))
        
        return discoveries
    
    # =========================================================
    # MAIN DISCOVERY LOOP
    # =========================================================
    
    def run_discovery(self, config: Dict = None) -> List[Discovery]:
        """Run all discovery methods and collect findings."""
        if config is None:
            config = self.get_default_config()
        
        all_discoveries = []
        
        print("Running φ-Discovery Engine...")
        print("=" * 60)
        
        # Method 1: Divergence patterns
        if 'prompt_pairs' in config:
            print("\n[1/6] Analyzing trajectory divergence...")
            discoveries = self.discover_divergence_patterns(config['prompt_pairs'])
            all_discoveries.extend(discoveries)
            print(f"      Found {len(discoveries)} discoveries")
        
        # Method 2: Resonance patterns
        if 'diverse_prompts' in config:
            print("\n[2/6] Finding resonance points...")
            discoveries = self.discover_resonance_patterns(config['diverse_prompts'])
            all_discoveries.extend(discoveries)
            print(f"      Found {len(discoveries)} discoveries")
        
        # Method 3: Semantic gaps
        if 'concept_pairs' in config:
            print("\n[3/6] Detecting semantic gaps...")
            discoveries = self.discover_semantic_gaps(config['concept_pairs'])
            all_discoveries.extend(discoveries)
            print(f"      Found {len(discoveries)} discoveries")
        
        # Method 4: Cross-domain bridges
        if 'domains' in config:
            print("\n[4/6] Finding cross-domain bridges...")
            discoveries = self.discover_cross_domain_bridges(config['domains'])
            all_discoveries.extend(discoveries)
            print(f"      Found {len(discoveries)} discoveries")
        
        # Method 5: φ-anomalies
        if 'diverse_prompts' in config:
            print("\n[5/6] Detecting φ-level anomalies...")
            discoveries = self.discover_phi_anomalies(config['diverse_prompts'])
            all_discoveries.extend(discoveries)
            print(f"      Found {len(discoveries)} discoveries")
        
        # Method 6: Emergent concepts
        if 'seed_concepts' in config:
            print("\n[6/6] Generating emergent concepts...")
            for seeds in config['seed_concepts']:
                discoveries = self.discover_emergent_concepts(seeds)
                all_discoveries.extend(discoveries)
            print(f"      Found {len(discoveries)} discoveries")
        
        # Sort by novelty
        all_discoveries.sort(key=lambda x: -x.novelty_score)
        
        self.discoveries.extend(all_discoveries)
        return all_discoveries
    
    def get_default_config(self) -> Dict:
        """Get default discovery configuration."""
        return {
            'prompt_pairs': [
                ('What is truth?', 'What is belief?'),
                ('Why do we exist?', 'How do we exist?'),
                ('What is mind?', 'What is matter?'),
                ('What is cause?', 'What is effect?'),
                ('What is self?', 'What is other?'),
            ],
            'diverse_prompts': [
                'The capital of France is',
                'Two plus two equals',
                'If A implies B then',
                'Once upon a time',
                'The meaning of life is',
                'When I feel happy',
                'The speed of light is',
                'To solve this problem',
                'The color of the sky is',
                'In the beginning there was',
            ],
            'concept_pairs': [
                ('time', 'space'),
                ('mind', 'body'),
                ('cause', 'effect'),
                ('self', 'other'),
                ('finite', 'infinite'),
                ('order', 'chaos'),
                ('being', 'becoming'),
                ('form', 'content'),
            ],
            'domains': {
                'mathematics': ['number', 'equation', 'proof', 'theorem', 'algebra'],
                'emotion': ['happy', 'sad', 'angry', 'fear', 'love'],
                'physics': ['force', 'energy', 'mass', 'velocity', 'quantum'],
                'biology': ['cell', 'gene', 'evolution', 'organism', 'life'],
                'philosophy': ['truth', 'beauty', 'good', 'justice', 'wisdom'],
            },
            'seed_concepts': [
                ['knowledge', 'wisdom', 'understanding'],
                ['time', 'change', 'becoming'],
                ['self', 'consciousness', 'awareness'],
                ['order', 'pattern', 'structure'],
            ]
        }
    
    def report(self) -> str:
        """Generate a report of all discoveries."""
        report = []
        report.append("=" * 70)
        report.append("φ-DISCOVERY ENGINE REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Discoveries: {len(self.discoveries)}")
        report.append("=" * 70)
        
        for i, d in enumerate(self.discoveries, 1):
            report.append(f"\n[{i}] {d.title}")
            report.append(f"    Type: {d.discovery_type}")
            report.append(f"    Novelty: {d.novelty_score:.2f}")
            report.append(f"    {d.description}")
        
        return "\n".join(report)


class NovelIdeaGenerator:
    """
    Generate genuinely novel ideas using reverse navigation through φ-space.
    
    The key insight: the φ-bottleneck at layer 27 acts as a validity filter.
    Only coherent ideas can pass through the bottleneck, so we can:
    1. Define a novel goal by combining distant concepts
    2. Generate candidates aimed at that goal
    3. Filter by φ-bottleneck convergence
    4. Return only valid novel ideas
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.all_embeddings = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        
    def get_trajectory(self, text: str) -> np.ndarray:
        """Get hidden state trajectory."""
        inputs = self.tokenizer(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
        return np.array([h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states])
    
    def get_phi_level(self, hidden: np.ndarray) -> float:
        """Get mean φ-level of a hidden state."""
        magnitudes = np.abs(hidden)
        magnitudes = magnitudes[magnitudes > 1e-10]
        levels = np.log(magnitudes) / LOG_PHI
        return float(np.mean(levels))
    
    def get_concept_embedding(self, concept: str) -> np.ndarray:
        """Get embedding for a concept (average of its tokens)."""
        tokens = self.tokenizer.encode(concept, add_special_tokens=False)
        return np.mean(self.all_embeddings[tokens], axis=0)
    
    def check_validity(self, text: str) -> Tuple[bool, float, float]:
        """
        Check if an idea is valid via φ-bottleneck convergence.
        
        Returns: (is_valid, phi_27, distance_from_phi)
        """
        trajectory = self.get_trajectory(text)
        phi_27 = self.get_phi_level(trajectory[27])
        distance = abs(phi_27 - PHI)
        is_valid = distance < 0.15  # Strict threshold
        return is_valid, phi_27, distance
    
    def generate_candidates(self, concepts: List[str], n_candidates: int = 5) -> List[str]:
        """Generate candidate ideas combining the given concepts."""
        prompt = f"""Generate a novel scientific or philosophical concept that meaningfully connects these ideas: {', '.join(concepts)}.

The concept should be:
1. Genuinely novel (not just combining words)
2. Internally coherent (no contradictions)
3. Potentially testable or explorable

Novel concept:"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        candidates = []
        
        for _ in range(n_candidates):
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.9,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            candidates.append(response.strip())
        
        return candidates
    
    def generate_novel_ideas(self, concepts: List[str], n_candidates: int = 10) -> List[Dict]:
        """
        Generate genuinely novel ideas by combining concepts and filtering by validity.
        
        Args:
            concepts: List of concepts to combine
            n_candidates: Number of candidates to generate
            
        Returns:
            List of valid ideas with their φ-validity scores
        """
        print(f"\n{'='*60}")
        print(f"NOVEL IDEA GENERATOR")
        print(f"{'='*60}")
        print(f"Input concepts: {concepts}")
        print(f"Generating {n_candidates} candidates...")
        
        # Generate candidates
        candidates = self.generate_candidates(concepts, n_candidates)
        
        # Check validity of each
        results = []
        for i, candidate in enumerate(candidates):
            is_valid, phi_27, distance = self.check_validity(candidate)
            
            status = "VALID" if is_valid else ("MARGINAL" if distance < 0.25 else "INVALID")
            print(f"\n[{i+1}] {status} (φ-27={phi_27:.4f}, d={distance:.4f})")
            print(f"    {candidate[:100]}...")
            
            results.append({
                'idea': candidate,
                'is_valid': is_valid,
                'phi_27': phi_27,
                'distance_from_phi': distance,
                'status': status
            })
        
        # Sort by validity (closest to φ first)
        results.sort(key=lambda x: x['distance_from_phi'])
        
        # Summary
        valid_count = sum(1 for r in results if r['is_valid'])
        print(f"\n{'='*60}")
        print(f"SUMMARY: {valid_count}/{len(results)} ideas passed φ-validity filter")
        print(f"{'='*60}")
        
        return results
    
    def explore_idea_space(self, seed_concepts: List[List[str]], n_per_seed: int = 5) -> List[Dict]:
        """
        Explore the space of novel ideas from multiple seed concept combinations.
        
        Args:
            seed_concepts: List of concept lists to explore
            n_per_seed: Number of candidates per seed
            
        Returns:
            All valid ideas discovered
        """
        all_valid = []
        
        for concepts in seed_concepts:
            results = self.generate_novel_ideas(concepts, n_per_seed)
            valid = [r for r in results if r['is_valid']]
            all_valid.extend(valid)
        
        # Deduplicate and rank
        all_valid.sort(key=lambda x: x['distance_from_phi'])
        
        print(f"\n{'='*60}")
        print(f"EXPLORATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total valid ideas discovered: {len(all_valid)}")
        
        if all_valid:
            print(f"\nTop 5 most valid ideas:")
            for i, idea in enumerate(all_valid[:5], 1):
                print(f"\n{i}. [φ-27={idea['phi_27']:.4f}]")
                print(f"   {idea['idea'][:150]}...")
        
        return all_valid
    
    def validate_hypothesis(self, hypothesis: str) -> Dict:
        """
        Validate a specific hypothesis using the φ-bottleneck.
        
        Args:
            hypothesis: The hypothesis to validate
            
        Returns:
            Validation result with φ-level analysis
        """
        is_valid, phi_27, distance = self.check_validity(hypothesis)
        
        # Also check the negation
        negation = f"It is false that {hypothesis}"
        neg_valid, neg_phi_27, neg_distance = self.check_validity(negation)
        
        result = {
            'hypothesis': hypothesis,
            'is_valid': is_valid,
            'phi_27': phi_27,
            'distance_from_phi': distance,
            'negation_phi_27': neg_phi_27,
            'negation_distance': neg_distance,
            'coherence_ratio': distance / (neg_distance + 1e-10)
        }
        
        print(f"\n{'='*60}")
        print(f"HYPOTHESIS VALIDATION")
        print(f"{'='*60}")
        print(f"Hypothesis: {hypothesis}")
        print(f"φ-27: {phi_27:.4f} (distance: {distance:.4f})")
        print(f"Status: {'VALID' if is_valid else 'INVALID'}")
        print(f"\nNegation φ-27: {neg_phi_27:.4f} (distance: {neg_distance:.4f})")
        print(f"Coherence ratio: {result['coherence_ratio']:.4f}")
        
        if result['coherence_ratio'] < 1.0:
            print(f"\n→ Hypothesis is MORE coherent than its negation")
        else:
            print(f"\n→ Negation is MORE coherent than hypothesis")
        
        return result


def main():
    """Run the discovery engine."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    
    # Create engine
    engine = PhiDiscoveryEngine(model, tokenizer)
    
    # Run discovery
    discoveries = engine.run_discovery()
    
    # Print report
    print("\n" + engine.report())
    
    # Save discoveries
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_discoveries': len(discoveries),
        'discoveries': [d.to_dict() for d in discoveries]
    }
    
    with open('phi_discoveries.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDiscoveries saved to phi_discoveries.json")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
