"""
Integration Test: Knowledge Store with Intent Detection

This test validates that the position-based knowledge architecture
works in practice with a real gear.

The flow:
1. IntentDetectorGear detects intent
2. Knowledge store learns from successful/failed detections
3. Concepts that are frequently correct persist
4. Concepts that are rarely used fade

Author: Lesley Gushurst
License: GPLv3
"""

import sys
sys.path.insert(0, '.')

from truthspace_lcm.core.knowledge import Concept, GeometricKnowledgeStore, CRITICAL_LINE
from truthspace_lcm.core.gears.intent_detector_gear import IntentDetectorGear, Intent


class KnowledgeAwareIntentDetector:
    """
    Intent detector that learns from usage patterns.
    
    Wraps IntentDetectorGear and uses GeometricKnowledgeStore
    to learn which patterns lead to successful intent detection.
    """
    
    def __init__(self):
        self.detector = IntentDetectorGear()
        self.store = GeometricKnowledgeStore(name="intent_patterns", dims=4)
        
        # Position anchors for each intent type
        # These define the "attractor basins" in the space
        self.intent_positions = {
            Intent.CHAT: (0.8, 0.0, 0.0, 0.0),
            Intent.TOOL_CALL: (0.0, 0.8, 0.0, 0.0),
            Intent.ORCHESTRATOR: (0.0, 0.0, 0.8, 0.0),
            Intent.CODE_GENERATION: (0.0, 0.0, 0.0, 0.8),
        }
    
    def detect(self, text: str) -> tuple:
        """
        Detect intent and return (intent, confidence, concept_id).
        
        Creates or updates a concept for this query pattern.
        """
        result = self.detector.detect(text)
        
        # Find or create concept for this query
        matches = self.store.query(text, top_k=1)
        
        if matches and matches[0][1] > 0.5:  # Good match
            concept = matches[0][0]
        else:
            # Create new concept for this pattern
            concept = self.store.add_from_text(text, source="query")
        
        return result.intent, result.confidence, concept.id
    
    def feedback(self, concept_id: str, detected_intent: Intent, was_correct: bool):
        """
        Provide feedback on whether the detection was correct.
        
        This is THE learning operation - moves concept toward/away
        from the intent's position anchor.
        """
        target_position = self.intent_positions.get(detected_intent)
        if target_position is None:
            return
        
        self.store.use(concept_id, target_position, success=was_correct)
    
    def get_stats(self) -> dict:
        """Get statistics about the knowledge store."""
        return {
            'total_concepts': len(self.store),
            'persisting': len(self.store.get_persisting_concepts()),
            'fading': len(self.store.get_fading_concepts()),
        }
    
    def prune(self) -> int:
        """Remove concepts that haven't learned enough."""
        return self.store.prune()


def run_integration_test():
    """
    Test the knowledge-aware intent detector.
    
    Simulates a series of queries with feedback to show
    how concepts learn and persist.
    """
    print("=" * 60)
    print("Integration Test: Knowledge Store + Intent Detection")
    print("=" * 60)
    print(f"Critical line: {CRITICAL_LINE}")
    print()
    
    detector = KnowledgeAwareIntentDetector()
    
    # Test queries with expected intents
    test_cases = [
        # (query, expected_intent, will_be_correct)
        ("Who is George Washington?", Intent.CHAT, True),
        ("What is the capital of France?", Intent.CHAT, True),
        ("Tell me about quantum physics", Intent.CHAT, True),
        ("Create a new directory called test", Intent.TOOL_CALL, True),
        ("Delete the file config.txt", Intent.TOOL_CALL, True),
        ("Run the build script", Intent.TOOL_CALL, True),
        ("Write a Python function to sort a list", Intent.CODE_GENERATION, True),
        ("Generate code for a bar chart", Intent.CODE_GENERATION, True),
        ("Set up a new project with git and requirements", Intent.ORCHESTRATOR, True),
        # Some that will be "wrong" (simulating user correction)
        ("Show me the files", Intent.TOOL_CALL, False),  # User wanted CHAT
        ("List the presidents", Intent.TOOL_CALL, False),  # User wanted CHAT
    ]
    
    print("Phase 1: Initial queries and feedback")
    print("-" * 40)
    
    for query, expected, correct in test_cases:
        intent, confidence, concept_id = detector.detect(query)
        detector.feedback(concept_id, intent, correct)
        
        status = "✓" if correct else "✗"
        print(f"{status} '{query[:40]}...' → {intent.name} (conf={confidence:.2f})")
    
    print()
    print(f"Stats after phase 1: {detector.get_stats()}")
    
    # Phase 2: Repeat successful patterns (reinforcement)
    print()
    print("Phase 2: Reinforcing successful patterns")
    print("-" * 40)
    
    reinforcement_queries = [
        ("Who is Abraham Lincoln?", Intent.CHAT),
        ("What is machine learning?", Intent.CHAT),
        ("Create a file called notes.txt", Intent.TOOL_CALL),
        ("Write Python code for fibonacci", Intent.CODE_GENERATION),
    ]
    
    # Repeat each 10 times with success
    for query, intent in reinforcement_queries:
        for _ in range(10):
            detected, conf, cid = detector.detect(query)
            detector.feedback(cid, detected, was_correct=True)
        
        concept = detector.store.get(cid)
        print(f"'{query[:35]}...' → mag={concept.magnitude:.3f}, persists={concept.persists}")
    
    print()
    print(f"Stats after phase 2: {detector.get_stats()}")
    
    # Phase 3: Prune fading concepts
    print()
    print("Phase 3: Pruning fading concepts")
    print("-" * 40)
    
    pruned = detector.prune()
    print(f"Pruned {pruned} concepts below critical line")
    print(f"Stats after pruning: {detector.get_stats()}")
    
    # Show surviving concepts
    print()
    print("Surviving concepts (past critical line):")
    for concept in detector.store.get_persisting_concepts():
        print(f"  {concept}")
    
    print()
    print("=" * 60)
    print("Integration test complete!")
    print()
    print("Key observations:")
    print("  - Concepts start at origin (magnitude 0)")
    print("  - Successful uses move concepts toward intent anchors")
    print("  - Frequently used patterns cross the critical line")
    print("  - Rarely used patterns stay near origin and get pruned")
    print("  - No statistical tracking needed - just position!")
    print("=" * 60)


if __name__ == '__main__':
    run_integration_test()
