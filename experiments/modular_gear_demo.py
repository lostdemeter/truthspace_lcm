#!/usr/bin/env python3
"""
Modular Gear Chain Demo

Demonstrates the extensible, modular gear architecture.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.gears import (
    GearChain, GearState,
    RoleGear, ActionGear, TenseGear, 
    SignalGear, DomainGear, StructureGear, OutputGear,
    ErrorCorrectionGear,
)
from truthspace_lcm.core.geometric import GeometricQA


class ModularGearProjector:
    """
    A projector that uses the modular gear system.
    
    This demonstrates how to compose gears into a chain
    and use them for text transformation.
    """
    
    def __init__(self, truth_corpus_path: str, signal_corpus_path: str = None):
        # Load truth corpus
        self.qa = GeometricQA()
        self.qa.load_corpus(truth_corpus_path)
        self.qa.set_output_lens('natural')
        
        # Build gear chain
        self.chain = GearChain("ProjectionChain")
        self.chain.add(RoleGear())
        self.chain.add(ActionGear())
        self.chain.add(TenseGear(tense='present'))
        self.chain.add(ErrorCorrectionGear())  # NEW: Error correction
        self.chain.add(SignalGear(signal_corpus_path))
        self.chain.add(DomainGear())
        self.chain.add(StructureGear())
        self.chain.add(OutputGear())
        
        print(f"Modular gear chain: {self.chain}")
    
    def project(self, concept: str) -> str:
        """Project a concept through the gear chain."""
        truth = self.qa.ask(f"What is {concept}?")
        if "don't know" in truth.lower():
            return f"Information about {concept} is not available."
        
        # Parse truth into initial state
        state = self._parse_to_state(truth, concept)
        
        # Process through gear chain
        return self.chain.process(state)
    
    def _parse_to_state(self, truth: str, concept: str) -> GearState:
        """Parse truth into gear state."""
        truth_lower = truth.lower()
        
        state = GearState()
        state.entity = concept.title()
        
        # Role
        match = re.search(r'is a[n]? (\w+)', truth_lower)
        if match:
            state.role = match.group(1)
        
        # Actions
        match = re.search(r'who\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
        if match:
            state.actions = [a for a in match.groups() if a]
        else:
            match = re.search(r'is a \w+ that\s+(\w+)(?:,\s*(\w+))?(?:,?\s*and\s*(\w+))?', truth_lower)
            if match:
                state.actions = [a for a in match.groups() if a]
        
        # Targets
        match = re.search(r'(?:relates to|involving)\s+(\w+)(?:\s+and\s+(\w+))?', truth_lower)
        if match:
            state.targets = [t for t in match.groups() if t]
        
        return state


def demo():
    """Demonstrate the modular gear system."""
    print("=" * 70)
    print("MODULAR GEAR CHAIN DEMO")
    print("Extensible, composable gear architecture")
    print("=" * 70)
    
    truth_path = "truthspace_lcm/corpus_experimental.json"
    signal_path = "truthspace_lcm/corpus_signal_full.json"
    
    projector = ModularGearProjector(truth_path, signal_path)
    
    # Test concepts
    test_concepts = [
        'evolution', 'biochemistry', 'analysis', 'missions', 'reforms',
        'holmes', 'watson', 'physics', 'consciousness'
    ]
    
    print("\n" + "=" * 70)
    print("Basic projection:")
    print("=" * 70)
    
    for concept in test_concepts:
        output = projector.project(concept)
        print(f"\n{concept.upper()}: {output[:90]}..." if len(output) > 90 else f"\n{concept.upper()}: {output}")
    
    # Demo: Tense manipulation
    print("\n" + "=" * 70)
    print("Tense manipulation:")
    print("=" * 70)
    
    tense_gear = projector.chain.get("TenseGear")
    concept = 'evolution'
    
    for tense in ['present', 'past', 'future', 'perfect']:
        tense_gear.set_tense(tense)
        output = projector.project(concept)
        print(f"  {tense.upper():8} → {output}")
    
    tense_gear.set_tense('present')  # Reset
    
    # Demo: Error correction stats
    print("\n" + "=" * 70)
    print("Error correction stats:")
    print("=" * 70)
    
    error_gear = projector.chain.get("ErrorCorrectionGear")
    stats = error_gear.get_stats()
    print(f"  Verb rules: {stats['verb_rules']}")
    print(f"  Word rules: {stats['word_rules']}")
    print(f"  Pattern rules: {stats['pattern_rules']}")
    print(f"  Total corrections applied: {stats['total_corrections']}")
    
    # Demo: Adding custom corrections
    print("\n" + "=" * 70)
    print("Adding custom corrections:")
    print("=" * 70)
    
    # Add a custom word correction
    error_gear.add_word_correction('rigorizing', 'rigorously analyzing')
    print("  Added: rigorizing → rigorously analyzing")
    
    # Test it
    output = projector.project('analysis')
    print(f"  ANALYSIS: {output}")
    
    # Demo: Disabling gears
    print("\n" + "=" * 70)
    print("Disabling gears:")
    print("=" * 70)
    
    print(f"  Full chain: {projector.chain}")
    
    projector.chain.disable("DomainGear")
    print(f"  Without DomainGear: {projector.chain}")
    
    projector.chain.get("DomainGear").enable()
    print(f"  Re-enabled: {projector.chain}")
    
    # Demo: Creating a minimal chain
    print("\n" + "=" * 70)
    print("Creating a minimal chain:")
    print("=" * 70)
    
    minimal = GearChain("MinimalChain")
    minimal.add(RoleGear())
    minimal.add(ErrorCorrectionGear())
    minimal.add(OutputGear())
    
    print(f"  {minimal}")
    
    # Process with minimal chain
    state = projector._parse_to_state(
        projector.qa.ask("What is evolution?"), 
        "evolution"
    )
    result = minimal.process(state)
    print(f"  Result: {result}")


if __name__ == "__main__":
    demo()
