"""
Tests for TruthSpace LCM Chat - Geometric Chat System

Tests the interactive chat functionality with GeometricLCM:
- Question answering
- Fact learning
- Analogical reasoning
- Commands
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.chat import GeometricChat


class TestGeometricChat:
    """Test GeometricChat with GeometricLCM backend."""
    
    def test_initialization(self):
        """Test chat initialization."""
        chat = GeometricChat()
        
        status = chat.lcm.status()
        assert status['facts'] > 0, "Should have bootstrap facts"
        assert status['entities'] > 0, "Should have entities"
        assert status['relations'] > 0, "Should have relations"
    
    def test_question_capital(self):
        """Test capital question."""
        chat = GeometricChat()
        
        response = chat.process("What is the capital of France?")
        
        assert "paris" in response.lower(), f"Should mention paris, got: {response}"
    
    def test_question_author(self):
        """Test author question."""
        chat = GeometricChat()
        
        response = chat.process("Who wrote Hamlet?")
        
        assert "shakespeare" in response.lower(), f"Should mention shakespeare, got: {response}"
    
    def test_question_location(self):
        """Test location question."""
        chat = GeometricChat()
        
        response = chat.process("Where is Japan?")
        
        assert "asia" in response.lower(), f"Should mention asia, got: {response}"
    
    def test_learn_new_fact(self):
        """Test learning a new fact."""
        chat = GeometricChat()
        
        initial_facts = chat.lcm.status()['facts']
        
        response = chat.process("Beijing is the capital of China.")
        
        assert "learned" in response.lower(), f"Should confirm learning, got: {response}"
        assert chat.lcm.status()['facts'] > initial_facts, "Should have more facts"
    
    def test_query_learned_fact(self):
        """Test querying a learned fact."""
        chat = GeometricChat()
        
        # Learn new fact
        chat.process("Beijing is the capital of China.")
        
        # Query it
        response = chat.process("What is the capital of China?")
        
        assert "beijing" in response.lower(), f"Should know beijing, got: {response}"
    
    def test_analogy_command(self):
        """Test /analogy command."""
        chat = GeometricChat()
        
        response = chat.process("/analogy france paris germany")
        
        assert "berlin" in response.lower(), f"Should find berlin, got: {response}"
    
    def test_analogy_accuracy(self):
        """Test analogy accuracy."""
        chat = GeometricChat()
        
        # Test multiple analogies
        analogies = [
            ("france", "paris", "germany", "berlin"),
            ("france", "paris", "japan", "tokyo"),
            ("melville", "moby_dick", "shakespeare", "hamlet"),
        ]
        
        correct = 0
        for a, b, c, expected in analogies:
            results = chat.lcm.analogy(a, b, c, k=1)
            if results and results[0][0] == expected:
                correct += 1
        
        accuracy = correct / len(analogies)
        assert accuracy >= 0.66, f"Analogy accuracy should be >= 66%, got {accuracy:.1%}"
    
    def test_status_command(self):
        """Test /status command."""
        chat = GeometricChat()
        
        response = chat.process("/status")
        
        assert "entities" in response.lower(), f"Should show entities, got: {response}"
        assert "relations" in response.lower(), f"Should show relations, got: {response}"
    
    def test_relations_command(self):
        """Test /relations command."""
        chat = GeometricChat()
        
        response = chat.process("/relations")
        
        assert "capital_of" in response.lower() or "wrote" in response.lower(), \
            f"Should list relations, got: {response}"
    
    def test_entities_command(self):
        """Test /entities command."""
        chat = GeometricChat()
        
        response = chat.process("/entities")
        
        assert "france" in response.lower() or "paris" in response.lower(), \
            f"Should list entities, got: {response}"
    
    def test_similar_command(self):
        """Test /similar command."""
        chat = GeometricChat()
        
        response = chat.process("/similar paris")
        
        assert "similar" in response.lower() or "berlin" in response.lower() or "rome" in response.lower(), \
            f"Should find similar entities, got: {response}"
    
    def test_help_command(self):
        """Test /help command."""
        chat = GeometricChat()
        
        response = chat.process("/help")
        
        assert "commands" in response.lower() or "help" in response.lower()
    
    def test_debug_command(self):
        """Test /debug command."""
        chat = GeometricChat()
        
        response = chat.process("/debug on")
        assert chat.debug_mode == True
        
        response = chat.process("/debug off")
        assert chat.debug_mode == False
    
    def test_quit_command(self):
        """Test /quit command."""
        chat = GeometricChat()
        
        response = chat.process("/quit")
        
        assert response == "QUIT"
    
    def test_unknown_command(self):
        """Test unknown command."""
        chat = GeometricChat()
        
        response = chat.process("/foobar")
        
        assert "unknown" in response.lower() or "help" in response.lower()
    
    def test_incremental_learning(self):
        """Test that learning new facts improves analogies."""
        chat = GeometricChat()
        
        # Learn new country-capital pair
        chat.process("Moscow is the capital of Russia.")
        
        # Test analogy with new entity
        results = chat.lcm.analogy("france", "paris", "russia", k=1)
        
        assert results, "Should have analogy results"
        assert results[0][0] == "moscow", f"Should find moscow, got {results[0][0]}"


class TestGeometricLCMCore:
    """Test GeometricLCM core functionality through chat."""
    
    def test_relation_consistency(self):
        """Test that relations have high consistency."""
        chat = GeometricChat()
        
        status = chat.lcm.status()
        
        for rel, cons in status['consistencies'].items():
            assert cons > 0.9, f"Relation {rel} should have >90% consistency, got {cons:.1%}"
    
    def test_query_accuracy(self):
        """Test query accuracy."""
        chat = GeometricChat()
        
        queries = [
            ("france", "capital_of", "paris"),
            ("germany", "capital_of", "berlin"),
            ("melville", "wrote", "moby_dick"),
        ]
        
        correct = 0
        for subj, rel, expected in queries:
            results = chat.lcm.query(subj, rel, k=1)
            if results and results[0][0] == expected:
                correct += 1
        
        accuracy = correct / len(queries)
        assert accuracy >= 0.9, f"Query accuracy should be >= 90%, got {accuracy:.1%}"
    
    def test_multi_hop(self):
        """Test multi-hop reasoning."""
        chat = GeometricChat()
        
        # france --capital_of--> paris --located_in--> ?
        # Note: paris is not in located_in facts, so this tests the structure
        results = chat.lcm.multi_hop("france", ["capital_of"], k=1)
        
        assert results, "Should have multi-hop results"
        assert results[0][0] == "paris", f"First hop should find paris, got {results[0][0]}"


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TruthSpace LCM Chat Test Suite")
    print("=" * 60)
    print()
    
    test_classes = [
        TestGeometricChat,
        TestGeometricLCMCore,
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        print(f"Testing {test_class.__name__}...")
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        passed = 0
        failed = 0
        
        for method_name in methods:
            try:
                getattr(instance, method_name)()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
        
        total_passed += passed
        total_failed += failed
        print()
    
    print("=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
