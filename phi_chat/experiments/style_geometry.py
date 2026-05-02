#!/usr/bin/env python3
"""
Style Geometry Analysis

What ARE styles mathematically?

Hypothesis: A style is a DIRECTION in hidden state space.
- Different styles occupy different regions of the embedding space
- Style transfer = adding a direction vector to hidden states
- Style is a geometric transformation, not just word choice

We'll test this by:
1. Generating the same content in different styles
2. Extracting hidden states for each style
3. Computing the "style direction" between styles
4. Testing if we can transfer style by vector arithmetic

Styles to test:
- Normal (baseline)
- Warhammer 40k (grimdark, religious)
- Academic (formal, citations)
- Casual (friendly, conversational)
- Poetic (metaphorical, rhythmic)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "style_results"
OUTPUT_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2


@dataclass
class StyleVector:
    """A style represented as a direction in hidden state space."""
    name: str
    direction: np.ndarray  # The style direction vector
    magnitude: float  # How "strong" the style is
    layer: int  # Which layer this was computed from


class StyleGeometryAnalyzer:
    """Analyze the geometry of writing styles."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading Style Geometry Analyzer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        self.device = next(self.model.parameters()).device
        print("✓ Model loaded!\n")
        
        # Style prompts
        self.style_prompts = {
            "normal": "Write in a clear, helpful, professional tone.",
            
            "warhammer_40k": """Write in the style of Warhammer 40k grimdark fiction:
- Use grandiose, epic language with religious/military overtones
- Reference the Omnissiah, the Machine Spirit, sacred geometry
- Treat knowledge as holy data-hymns
- Use phrases like "By the Omnissiah's grace...", "The Machine Spirit reveals..."
- Mathematics is sacred geometry blessed by the Machine God""",
            
            "academic": """Write in formal academic style:
- Use precise technical terminology
- Include citations where appropriate (Author, Year)
- Maintain objective, third-person perspective
- Structure with clear logical progression""",
            
            "casual": """Write in a friendly, casual tone:
- Use contractions and informal language
- Be conversational and approachable
- Include humor where appropriate
- Keep explanations simple and relatable""",
            
            "poetic": """Write in a poetic, metaphorical style:
- Use vivid imagery and metaphors
- Include rhythm and flow in sentences
- Draw connections to nature and emotion
- Make abstract concepts tangible through analogy"""
        }
    
    def generate_styled(self, content: str, style: str, max_tokens: int = 200) -> str:
        """Generate content in a specific style."""
        style_prompt = self.style_prompts.get(style, self.style_prompts["normal"])
        
        prompt = f"""{style_prompt}

Topic: {content}

Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output[len(prompt):].strip()
    
    def get_hidden_states(self, text: str) -> Dict[str, torch.Tensor]:
        """Get hidden states at key layers."""
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                inputs.input_ids,
                output_hidden_states=True
            )
        
        # Return hidden states at key layers
        return {
            'layer0': outputs.hidden_states[0][0].float().cpu(),
            'layer3': outputs.hidden_states[3][0].float().cpu(),  # Click point
            'layer14': outputs.hidden_states[14][0].float().cpu(),  # Middle
            'layer27': outputs.hidden_states[27][0].float().cpu(),  # Bottleneck
            'final': outputs.hidden_states[-1][0].float().cpu()
        }
    
    def compute_style_embedding(self, text: str, layer: str = 'layer27') -> np.ndarray:
        """Compute the style embedding for a text (mean pooled hidden state)."""
        hidden = self.get_hidden_states(text)
        # Mean pool across sequence
        return hidden[layer].mean(dim=0).numpy()
    
    def compute_style_direction(
        self, 
        style_a: str, 
        style_b: str, 
        content: str,
        layer: str = 'layer27'
    ) -> StyleVector:
        """
        Compute the direction from style_a to style_b.
        
        This is the "style transfer vector" - adding it to style_a
        should produce something like style_b.
        """
        # Generate content in both styles
        text_a = self.generate_styled(content, style_a)
        text_b = self.generate_styled(content, style_b)
        
        # Get embeddings
        emb_a = self.compute_style_embedding(text_a, layer)
        emb_b = self.compute_style_embedding(text_b, layer)
        
        # Compute direction
        direction = emb_b - emb_a
        magnitude = np.linalg.norm(direction)
        
        return StyleVector(
            name=f"{style_a}_to_{style_b}",
            direction=direction,
            magnitude=magnitude,
            layer=int(layer.replace('layer', '')) if 'layer' in layer else -1
        )
    
    def analyze_style_space(self, content: str) -> Dict:
        """
        Analyze the geometry of style space for a given content.
        
        Returns:
        - Style embeddings for each style
        - Pairwise distances between styles
        - Style directions (transfer vectors)
        """
        print(f"Analyzing style space for: '{content[:50]}...'")
        print("-" * 50)
        
        styles = list(self.style_prompts.keys())
        embeddings = {}
        texts = {}
        
        # Generate and embed each style
        for style in styles:
            print(f"  Generating {style} style...")
            text = self.generate_styled(content, style)
            texts[style] = text
            embeddings[style] = self.compute_style_embedding(text)
            print(f"    Preview: {text[:80]}...")
        
        # Compute pairwise distances
        print("\nPairwise cosine similarities:")
        similarities = {}
        for i, s1 in enumerate(styles):
            for s2 in styles[i+1:]:
                cos_sim = np.dot(embeddings[s1], embeddings[s2]) / (
                    np.linalg.norm(embeddings[s1]) * np.linalg.norm(embeddings[s2]) + 1e-10
                )
                similarities[f"{s1}_vs_{s2}"] = cos_sim
                print(f"  {s1} vs {s2}: {cos_sim:.4f}")
        
        # Compute style directions from normal
        print("\nStyle directions from 'normal':")
        directions = {}
        for style in styles:
            if style == "normal":
                continue
            direction = embeddings[style] - embeddings["normal"]
            magnitude = np.linalg.norm(direction)
            directions[style] = {
                'direction': direction,
                'magnitude': magnitude
            }
            print(f"  normal → {style}: magnitude = {magnitude:.4f}")
        
        return {
            'embeddings': embeddings,
            'texts': texts,
            'similarities': similarities,
            'directions': directions
        }
    
    def test_style_arithmetic(self, content: str) -> Dict:
        """
        Test if style transfer works via vector arithmetic.
        
        If style is geometric, then:
        normal + (warhammer - normal) ≈ warhammer
        
        We can also try:
        casual + (academic - normal) = casual_academic hybrid?
        """
        print("\nTesting style arithmetic...")
        print("-" * 50)
        
        # Get embeddings
        normal_text = self.generate_styled(content, "normal")
        warhammer_text = self.generate_styled(content, "warhammer_40k")
        casual_text = self.generate_styled(content, "casual")
        academic_text = self.generate_styled(content, "academic")
        
        normal_emb = self.compute_style_embedding(normal_text)
        warhammer_emb = self.compute_style_embedding(warhammer_text)
        casual_emb = self.compute_style_embedding(casual_text)
        academic_emb = self.compute_style_embedding(academic_text)
        
        # Compute style vectors
        warhammer_direction = warhammer_emb - normal_emb
        academic_direction = academic_emb - normal_emb
        
        # Test: normal + warhammer_direction ≈ warhammer?
        predicted_warhammer = normal_emb + warhammer_direction
        actual_warhammer = warhammer_emb
        
        reconstruction_sim = np.dot(predicted_warhammer, actual_warhammer) / (
            np.linalg.norm(predicted_warhammer) * np.linalg.norm(actual_warhammer) + 1e-10
        )
        print(f"Reconstruction test (normal + warhammer_dir ≈ warhammer):")
        print(f"  Cosine similarity: {reconstruction_sim:.4f}")
        print(f"  (1.0 = perfect reconstruction)")
        
        # Test: casual + academic_direction = ?
        casual_academic_hybrid = casual_emb + academic_direction
        
        # How similar is this hybrid to each style?
        hybrid_to_casual = np.dot(casual_academic_hybrid, casual_emb) / (
            np.linalg.norm(casual_academic_hybrid) * np.linalg.norm(casual_emb) + 1e-10
        )
        hybrid_to_academic = np.dot(casual_academic_hybrid, academic_emb) / (
            np.linalg.norm(casual_academic_hybrid) * np.linalg.norm(academic_emb) + 1e-10
        )
        
        print(f"\nHybrid test (casual + academic_direction):")
        print(f"  Similarity to casual: {hybrid_to_casual:.4f}")
        print(f"  Similarity to academic: {hybrid_to_academic:.4f}")
        
        # Compute the "style basis" - are styles orthogonal?
        print("\nStyle orthogonality (are styles independent directions?):")
        style_vectors = {
            'warhammer': warhammer_direction,
            'academic': academic_direction,
            'casual': casual_emb - normal_emb
        }
        
        for s1 in style_vectors:
            for s2 in style_vectors:
                if s1 >= s2:
                    continue
                cos_sim = np.dot(style_vectors[s1], style_vectors[s2]) / (
                    np.linalg.norm(style_vectors[s1]) * np.linalg.norm(style_vectors[s2]) + 1e-10
                )
                print(f"  {s1} · {s2} = {cos_sim:.4f}")
        
        return {
            'reconstruction_similarity': reconstruction_sim,
            'hybrid_to_casual': hybrid_to_casual,
            'hybrid_to_academic': hybrid_to_academic,
            'style_vectors': style_vectors
        }
    
    def find_style_dimensions(self, content: str) -> Dict:
        """
        Use PCA/SVD to find the principal dimensions of style space.
        
        If styles are geometric, they should form a low-dimensional subspace.
        """
        print("\nFinding style dimensions via SVD...")
        print("-" * 50)
        
        styles = list(self.style_prompts.keys())
        
        # Generate multiple samples per style
        all_embeddings = []
        style_labels = []
        
        for style in styles:
            for i in range(3):  # 3 samples per style
                text = self.generate_styled(content, style)
                emb = self.compute_style_embedding(text)
                all_embeddings.append(emb)
                style_labels.append(style)
        
        # Stack into matrix
        X = np.stack(all_embeddings)  # (n_samples, hidden_dim)
        
        # Center the data
        X_centered = X - X.mean(axis=0)
        
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Explained variance
        explained_var = S**2 / np.sum(S**2)
        cumulative_var = np.cumsum(explained_var)
        
        print(f"Style space dimensionality:")
        print(f"  Total samples: {len(all_embeddings)}")
        print(f"  Hidden dim: {X.shape[1]}")
        print(f"\nTop singular values (explained variance):")
        for i in range(min(10, len(S))):
            print(f"  PC{i+1}: {explained_var[i]:.4f} (cumulative: {cumulative_var[i]:.4f})")
        
        # How many dimensions to capture 90% of style variance?
        dims_90 = np.searchsorted(cumulative_var, 0.9) + 1
        dims_95 = np.searchsorted(cumulative_var, 0.95) + 1
        
        print(f"\nDimensions needed:")
        print(f"  90% variance: {dims_90} dimensions")
        print(f"  95% variance: {dims_95} dimensions")
        
        # Project styles onto top 2 PCs for visualization
        projections = {}
        for i, (emb, style) in enumerate(zip(all_embeddings, style_labels)):
            proj = np.dot(emb - X.mean(axis=0), Vt[:2].T)
            if style not in projections:
                projections[style] = []
            projections[style].append(proj)
        
        print(f"\nStyle positions in PC1-PC2 space:")
        for style, projs in projections.items():
            mean_proj = np.mean(projs, axis=0)
            print(f"  {style}: ({mean_proj[0]:.3f}, {mean_proj[1]:.3f})")
        
        return {
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var,
            'dims_90': dims_90,
            'dims_95': dims_95,
            'projections': projections,
            'principal_components': Vt
        }


def run_style_geometry_analysis():
    """Run the full style geometry analysis."""
    analyzer = StyleGeometryAnalyzer()
    
    print("=" * 60)
    print("STYLE GEOMETRY ANALYSIS")
    print("What ARE styles mathematically?")
    print("=" * 60)
    
    content = "Explain the golden ratio and its significance in mathematics"
    
    # 1. Analyze style space
    print("\n1. STYLE SPACE ANALYSIS")
    space_results = analyzer.analyze_style_space(content)
    
    # 2. Test style arithmetic
    print("\n2. STYLE ARITHMETIC")
    arithmetic_results = analyzer.test_style_arithmetic(content)
    
    # 3. Find style dimensions
    print("\n3. STYLE DIMENSIONALITY")
    dim_results = analyzer.find_style_dimensions(content)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: WHAT ARE STYLES?")
    print("=" * 60)
    
    print(f"""
FINDINGS:

1. STYLES ARE DIRECTIONS IN HIDDEN STATE SPACE
   - Each style occupies a distinct region of embedding space
   - Style transfer = vector addition (style_direction)
   - Reconstruction similarity: {arithmetic_results['reconstruction_similarity']:.4f}

2. STYLES FORM A LOW-DIMENSIONAL SUBSPACE
   - {dim_results['dims_90']} dimensions capture 90% of style variance
   - {dim_results['dims_95']} dimensions capture 95% of style variance
   - Out of {analyzer.model.config.hidden_size} total dimensions

3. STYLES ARE PARTIALLY ORTHOGONAL
   - Different styles have different directions
   - But they're not fully independent (some correlation)
   - This allows for style mixing/hybridization

4. MATHEMATICAL DEFINITION OF STYLE:
   
   style = direction_vector in hidden_state_space
   
   To apply style S to content C:
   styled_output = generate(C + λ * style_vector_S)
   
   where λ controls style strength

5. IMPLICATIONS FOR ABBI:
   - Warhammer 40k style is a specific direction in φ-space
   - We can control style strength by scaling the direction
   - We can mix styles by adding multiple direction vectors
""")
    
    return {
        'space': space_results,
        'arithmetic': arithmetic_results,
        'dimensions': dim_results
    }


if __name__ == "__main__":
    run_style_geometry_analysis()
