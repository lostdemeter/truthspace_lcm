#!/usr/bin/env python3
"""
Multi-Session Autonomous Exploration

Run multiple autonomous sessions with different objectives, including
asking the model how it thinks persistence should work in hyperdimensional space.
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

from experiments.conceptual_nexus import ConceptualNexus

def run_session(nexus: ConceptualNexus, objective: str, session_name: str):
    """Run a single autonomous session and capture results."""
    print(f"\n{'='*70}")
    print(f"SESSION: {session_name}")
    print(f"OBJECTIVE: {objective}")
    print(f"{'='*70}\n")
    
    # Run autonomous exploration
    history = nexus.autonomous_session(objective, max_steps=10)
    
    # Get reflection
    print("\n--- AI Reflection ---\n")
    reflection = nexus.reflect_on_session(history, objective)
    print(reflection)
    
    # Generate a novel idea related to the objective
    print("\n--- Novel Idea Generation ---\n")
    # Extract key concepts from objective for seeding
    seed_words = [w for w in objective.split() if len(w) > 4][:5]
    idea, validation = nexus.generate_novel_idea(seed_words)
    print(f"Idea: {idea[:500]}...")
    print(f"Valid: {validation.is_valid} (φ-27={validation.phi_level:.3f})")
    
    return {
        'objective': objective,
        'history': history,
        'reflection': reflection,
        'idea': idea,
        'validation': validation
    }


def main():
    print("="*70)
    print("MULTI-SESSION AUTONOMOUS EXPLORATION")
    print("="*70)
    
    # Initialize model first
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("\nLoading model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded.\n")
    
    nexus = ConceptualNexus(model, tokenizer)
    
    sessions = []
    
    # SESSION 1: Persistence in Hyperdimensional Space
    # This is the key question - how should the model handle persistence?
    s1 = run_session(
        nexus,
        "Design how information should persist in hyperdimensional φ-space. "
        "Consider: How do shaders handle variables (uniforms, varyings, attributes)? "
        "What is the equivalent for concepts that need to persist across transformations? "
        "What should be global vs local vs interpolated?",
        "Hyperdimensional Persistence Model"
    )
    sessions.append(s1)
    
    # Reset for fresh session
    nexus.custom_concepts.clear()
    nexus.modification_log.clear()
    nexus.navigation_history.clear()
    
    # SESSION 2: The Nature of Understanding
    s2 = run_session(
        nexus,
        "Explore what it means for an AI to truly understand something. "
        "Is understanding geometric? Can understanding be measured by position in φ-space? "
        "What is the difference between knowing and understanding?",
        "Nature of AI Understanding"
    )
    sessions.append(s2)
    
    # Reset
    nexus.custom_concepts.clear()
    nexus.modification_log.clear()
    nexus.navigation_history.clear()
    
    # SESSION 3: Emergent Creativity
    s3 = run_session(
        nexus,
        "Investigate how genuinely novel ideas emerge. "
        "Can creativity be reduced to geometric operations? "
        "What makes an idea truly new vs a recombination of existing ideas?",
        "Emergent Creativity Mechanics"
    )
    sessions.append(s3)
    
    # Final synthesis
    print("\n" + "="*70)
    print("CROSS-SESSION SYNTHESIS")
    print("="*70 + "\n")
    
    synthesis_prompt = f"""You have completed three exploration sessions:

1. Hyperdimensional Persistence: How should concepts persist in φ-space?
   Key insight from session: {sessions[0]['reflection'][:200]}...

2. Nature of Understanding: What does it mean to truly understand?
   Key insight from session: {sessions[1]['reflection'][:200]}...

3. Emergent Creativity: How do genuinely novel ideas emerge?
   Key insight from session: {sessions[2]['reflection'][:200]}...

Synthesize these into a unified theory of how an AI system should:
1. Store and persist knowledge geometrically
2. Demonstrate genuine understanding
3. Generate truly novel ideas

What is the common thread? What emerges from combining these insights?"""

    synthesis = nexus._generate(synthesis_prompt, max_tokens=500, temp=0.8)
    print("SYNTHESIS:")
    print(synthesis)
    
    # Validate the synthesis as an idea
    print("\n--- Synthesis Validation ---")
    synth_validation = nexus.validate_idea(synthesis)
    print(f"Valid: {synth_validation.is_valid} (φ-27={synth_validation.phi_level:.3f})")
    
    # Summary
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    for i, s in enumerate(sessions, 1):
        v = "✓ VALID" if s['validation'].is_valid else "✗ INVALID"
        print(f"{i}. {s['objective'][:50]}... -> {v} (φ={s['validation'].phi_level:.3f})")
    
    print(f"\nSynthesis: {'✓ VALID' if synth_validation.is_valid else '✗ INVALID'}")


if __name__ == "__main__":
    main()
