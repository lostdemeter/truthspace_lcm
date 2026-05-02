#!/usr/bin/env python3
"""
Novel Idea Generator via Reverse φ-Navigation

Uses the φ-bottleneck as a validity filter to generate genuinely novel ideas
by combining distant concepts and filtering through geometric constraints.

Usage:
    python novel_idea_generator.py "concept1" "concept2" "concept3"
    python novel_idea_generator.py --interactive
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import argparse
import json
from dataclasses import dataclass
from datetime import datetime

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


@dataclass
class NovelIdea:
    """A generated novel idea with validity metrics."""
    concepts: List[str]
    idea: str
    phi_27: float
    distance_from_phi: float
    alignment_score: float
    is_valid: bool
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            'concepts': self.concepts,
            'idea': self.idea,
            'phi_27': self.phi_27,
            'distance_from_phi': self.distance_from_phi,
            'alignment_score': self.alignment_score,
            'is_valid': self.is_valid,
            'timestamp': self.timestamp
        }


class NovelIdeaGenerator:
    """Generate novel valid ideas via reverse φ-navigation."""
    
    def __init__(self, model, tokenizer, validity_threshold: float = 0.3):
        self.model = model
        self.tokenizer = tokenizer
        self.validity_threshold = validity_threshold
        
        # Cache embeddings
        self.embeddings = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        self.lm_head = model.lm_head.weight.detach().float().cpu().numpy()
        
        # Track generated ideas
        self.generated_ideas: List[NovelIdea] = []
    
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
    
    def get_concept_embedding(self, concept: str) -> Optional[np.ndarray]:
        """Get embedding for a concept."""
        tokens = self.tokenizer.encode(concept, add_special_tokens=False)
        if tokens:
            return self.embeddings[tokens[0]]
        return None
    
    def define_goal(self, concepts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Define a novel goal by combining concept embeddings."""
        valid_concepts = []
        embeddings = []
        
        for concept in concepts:
            emb = self.get_concept_embedding(concept)
            if emb is not None:
                embeddings.append(emb)
                valid_concepts.append(concept)
        
        if not embeddings:
            raise ValueError("No valid concept embeddings found")
        
        # Goal is the centroid
        goal = np.mean(embeddings, axis=0)
        return goal, valid_concepts
    
    def generate_candidate(self, concepts: List[str], temperature: float = 0.9) -> str:
        """Generate a candidate idea."""
        prompt = f"A genuinely novel idea connecting {', '.join(concepts)} would be:"
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def check_validity(self, text: str, goal: np.ndarray) -> Dict:
        """Check if an idea is valid via φ-bottleneck."""
        trajectory = self.get_trajectory(text[:500])  # Limit length
        
        phi_27 = self.get_phi_level(trajectory[27])
        distance_from_phi = abs(phi_27 - PHI)
        
        # Alignment with goal
        final_hidden = trajectory[-1]
        alignment = self.cosine_sim(final_hidden, goal)
        
        is_valid = distance_from_phi < self.validity_threshold
        
        return {
            'phi_27': phi_27,
            'distance_from_phi': distance_from_phi,
            'alignment': alignment,
            'is_valid': is_valid
        }
    
    def generate_novel_idea(
        self, 
        concepts: List[str], 
        n_candidates: int = 5,
        temperature: float = 0.9
    ) -> Optional[NovelIdea]:
        """
        Generate a novel valid idea by combining concepts.
        
        Args:
            concepts: List of concepts to combine
            n_candidates: Number of candidates to generate
            temperature: Generation temperature
            
        Returns:
            The best valid idea, or None if no valid ideas found
        """
        # Define the goal
        goal, valid_concepts = self.define_goal(concepts)
        
        # Generate and evaluate candidates
        candidates = []
        
        for i in range(n_candidates):
            # Generate candidate
            idea_text = self.generate_candidate(valid_concepts, temperature)
            
            # Build full text for evaluation
            full_text = f"A novel idea connecting {', '.join(valid_concepts)}: {idea_text}"
            
            # Check validity
            validity = self.check_validity(full_text, goal)
            
            candidates.append({
                'idea': idea_text,
                **validity
            })
        
        # Filter valid candidates
        valid_candidates = [c for c in candidates if c['is_valid']]
        
        if not valid_candidates:
            # Return best invalid candidate with warning
            best = max(candidates, key=lambda x: -x['distance_from_phi'])
            idea = NovelIdea(
                concepts=valid_concepts,
                idea=best['idea'],
                phi_27=best['phi_27'],
                distance_from_phi=best['distance_from_phi'],
                alignment_score=best['alignment'],
                is_valid=False,
                timestamp=datetime.now().isoformat()
            )
        else:
            # Return best valid candidate (highest alignment)
            best = max(valid_candidates, key=lambda x: x['alignment'])
            idea = NovelIdea(
                concepts=valid_concepts,
                idea=best['idea'],
                phi_27=best['phi_27'],
                distance_from_phi=best['distance_from_phi'],
                alignment_score=best['alignment'],
                is_valid=True,
                timestamp=datetime.now().isoformat()
            )
        
        self.generated_ideas.append(idea)
        return idea
    
    def explore_domain(
        self, 
        seed_concepts: List[str],
        n_ideas: int = 5
    ) -> List[NovelIdea]:
        """
        Explore a domain by generating multiple novel ideas.
        
        Varies the concept combinations to explore different regions.
        """
        ideas = []
        
        # Generate ideas with different temperatures
        for i in range(n_ideas):
            temp = 0.7 + (i * 0.1)  # Vary temperature
            idea = self.generate_novel_idea(seed_concepts, temperature=temp)
            if idea:
                ideas.append(idea)
        
        return ideas
    
    def find_nearest_concepts(self, goal: np.ndarray, n: int = 10) -> List[str]:
        """Find concepts nearest to a goal position."""
        sims = [(i, self.cosine_sim(goal, self.embeddings[i])) 
                for i in range(len(self.embeddings))]
        sims.sort(key=lambda x: -x[1])
        
        concepts = []
        seen = set()
        for idx, sim in sims[:100]:
            token = self.tokenizer.decode([idx]).strip()
            if token and len(token) > 2 and token.isalpha() and token.lower() not in seen:
                concepts.append(token)
                seen.add(token.lower())
            if len(concepts) >= n:
                break
        
        return concepts
    
    def save_ideas(self, filepath: str):
        """Save generated ideas to JSON."""
        data = {
            'generated_at': datetime.now().isoformat(),
            'validity_threshold': self.validity_threshold,
            'ideas': [idea.to_dict() for idea in self.generated_ideas]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def display_idea(self, idea: NovelIdea):
        """Pretty print an idea."""
        valid_marker = "✓ VALID" if idea.is_valid else "✗ INVALID"
        print(f"\n{'='*60}")
        print(f"NOVEL IDEA [{valid_marker}]")
        print(f"{'='*60}")
        print(f"Concepts: {', '.join(idea.concepts)}")
        print(f"Idea: {idea.idea}")
        print(f"φ-27: {idea.phi_27:.3f} (distance from φ: {idea.distance_from_phi:.3f})")
        print(f"Alignment: {idea.alignment_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Generate novel valid ideas')
    parser.add_argument('concepts', nargs='*', help='Concepts to combine')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--n-candidates', type=int, default=5, help='Candidates per idea')
    parser.add_argument('--temperature', type=float, default=0.9, help='Generation temperature')
    parser.add_argument('--output', type=str, default='novel_ideas.json', help='Output file')
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    
    generator = NovelIdeaGenerator(model, tokenizer)
    
    if args.interactive:
        print("\n" + "="*60)
        print("NOVEL IDEA GENERATOR - Interactive Mode")
        print("="*60)
        print("Enter concepts separated by commas, or 'quit' to exit.")
        print("Example: quantum, biology, consciousness")
        
        while True:
            try:
                user_input = input("\nConcepts: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                concepts = [c.strip() for c in user_input.split(',')]
                if len(concepts) < 2:
                    print("Please enter at least 2 concepts.")
                    continue
                
                print(f"\nGenerating novel idea from: {concepts}")
                idea = generator.generate_novel_idea(
                    concepts, 
                    n_candidates=args.n_candidates,
                    temperature=args.temperature
                )
                
                if idea:
                    generator.display_idea(idea)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Save all generated ideas
        if generator.generated_ideas:
            generator.save_ideas(args.output)
            print(f"\nSaved {len(generator.generated_ideas)} ideas to {args.output}")
    
    elif args.concepts:
        print(f"\nGenerating novel idea from: {args.concepts}")
        idea = generator.generate_novel_idea(
            args.concepts,
            n_candidates=args.n_candidates,
            temperature=args.temperature
        )
        
        if idea:
            generator.display_idea(idea)
            generator.save_ideas(args.output)
    
    else:
        # Demo mode
        print("\n" + "="*60)
        print("NOVEL IDEA GENERATOR - Demo")
        print("="*60)
        
        demo_combinations = [
            ['quantum', 'cooking', 'music'],
            ['mathematics', 'emotion', 'architecture'],
            ['time', 'memory', 'crystals'],
            ['gravity', 'language', 'dreams'],
            ['evolution', 'art', 'computation'],
        ]
        
        for concepts in demo_combinations:
            print(f"\n--- Combining: {concepts} ---")
            idea = generator.generate_novel_idea(concepts, n_candidates=3)
            if idea:
                generator.display_idea(idea)
        
        generator.save_ideas(args.output)
        print(f"\nSaved {len(generator.generated_ideas)} ideas to {args.output}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
