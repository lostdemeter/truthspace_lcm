"""
Tests for TruthSpace LCM Core - Geometric Language Model

Tests the core functionality:
- Vocabulary: hash-based word positions, IDF weighting, text encoding
- GeometricLCM: fact learning, queries, analogies, multi-hop reasoning
- Supporting modules: cosine similarity, tokenization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from truthspace_lcm.core import (
    Vocabulary,
    GeometricLCM,
    cosine_similarity,
    tokenize,
)


class TestVocabulary:
    """Test Vocabulary system."""
    
    def test_tokenize(self):
        """Test tokenization."""
        tokens = tokenize("Hello, World! How are you?")
        assert tokens == ['hello', 'world', 'how', 'are', 'you']
        
        tokens = tokenize("test123 foo_bar")
        assert tokens == ['test123', 'foo_bar']
    
    def test_word_position_deterministic(self):
        """Word positions should be deterministic."""
        vocab = Vocabulary(dim=64)
        
        pos1 = vocab.get_position("hello")
        pos2 = vocab.get_position("hello")
        
        assert np.allclose(pos1, pos2), "Same word should get same position"
    
    def test_word_position_normalized(self):
        """Word positions should be normalized."""
        vocab = Vocabulary(dim=64)
        
        pos = vocab.get_position("test")
        norm = np.linalg.norm(pos)
        
        assert abs(norm - 1.0) < 0.01, f"Position should be unit norm, got {norm}"
    
    def test_different_words_different_positions(self):
        """Different words should have different positions."""
        vocab = Vocabulary(dim=64)
        
        pos1 = vocab.get_position("cat")
        pos2 = vocab.get_position("dog")
        
        sim = cosine_similarity(pos1, pos2)
        assert sim < 0.9, f"Different words should have different positions, sim={sim}"
    
    def test_encode_text(self):
        """Test text encoding."""
        vocab = Vocabulary(dim=64)
        
        enc = vocab.encode("hello world")
        assert len(enc) == 64, f"Expected 64D encoding, got {len(enc)}"
        assert np.linalg.norm(enc) > 0, "Encoding should be non-zero"
    
    def test_encode_empty(self):
        """Empty text should return zero vector."""
        vocab = Vocabulary(dim=64)
        
        enc = vocab.encode("")
        assert np.allclose(enc, np.zeros(64)), "Empty text should encode to zero"
    
    def test_idf_weighting(self):
        """Rare words should have higher weight."""
        vocab = Vocabulary(dim=64)
        
        # Add common word many times
        for _ in range(100):
            vocab.add_text("the")
        
        # Add rare word once
        vocab.add_text("xyzzy")
        
        weight_common = vocab.idf_weight("the")
        weight_rare = vocab.idf_weight("xyzzy")
        
        assert weight_rare > weight_common, "Rare words should have higher weight"
    
    def test_similarity(self):
        """Test similarity computation."""
        vocab = Vocabulary(dim=64)
        
        sim = vocab.similarity("hello world", "hello world")
        assert sim > 0.99, f"Identical text should have sim ~1, got {sim}"
        
        sim = vocab.similarity("cat dog", "fish bird")
        assert sim < 0.9, f"Different text should have lower sim, got {sim}"


class TestCosine:
    """Test cosine similarity."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        v = np.array([1, 2, 3])
        assert abs(cosine_similarity(v, v) - 1.0) < 0.001
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        assert abs(cosine_similarity(v1, v2)) < 0.001
    
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])
        assert abs(cosine_similarity(v1, v2) + 1.0) < 0.001
    
    def test_zero_vector(self):
        """Zero vector should return 0 similarity."""
        v1 = np.array([1, 2, 3])
        v2 = np.zeros(3)
        assert cosine_similarity(v1, v2) == 0.0


class TestGeometricLCM:
    """Test GeometricLCM - the core dynamic geometric language model."""
    
    def test_initialization(self):
        """Test LCM initialization."""
        lcm = GeometricLCM(dim=256)
        
        assert lcm.dim == 256
        assert len(lcm.entities) == 0
        assert len(lcm.relations) == 0
        assert len(lcm.facts) == 0
    
    def test_add_fact(self):
        """Test adding facts."""
        lcm = GeometricLCM(dim=256)
        
        fact = lcm.add_fact("france", "capital_of", "paris")
        
        assert fact.subject == "france"
        assert fact.relation == "capital_of"
        assert fact.object == "paris"
        assert "france" in lcm.entities
        assert "paris" in lcm.entities
        assert "capital_of" in lcm.relations
    
    def test_ingest_text(self):
        """Test text ingestion."""
        lcm = GeometricLCM(dim=256)
        
        facts = lcm.ingest("Paris is the capital of France.")
        
        assert len(facts) > 0
        assert "france" in lcm.entities
        assert "paris" in lcm.entities
    
    def test_learn(self):
        """Test learning updates structure."""
        lcm = GeometricLCM(dim=256)
        
        lcm.add_fact("france", "capital_of", "paris")
        lcm.add_fact("germany", "capital_of", "berlin")
        lcm.add_fact("japan", "capital_of", "tokyo")
        
        consistency = lcm.learn(n_iterations=50)
        
        assert consistency > 0.9, f"Should achieve high consistency, got {consistency}"
    
    def test_query(self):
        """Test relational queries."""
        lcm = GeometricLCM(dim=256)
        
        lcm.add_fact("france", "capital_of", "paris")
        lcm.add_fact("germany", "capital_of", "berlin")
        lcm.learn(n_iterations=50)
        
        results = lcm.query("france", "capital_of", k=1)
        
        assert len(results) > 0
        assert results[0][0] == "paris", f"Expected paris, got {results[0][0]}"
    
    def test_inverse_query(self):
        """Test inverse relational queries."""
        lcm = GeometricLCM(dim=256)
        
        lcm.add_fact("france", "capital_of", "paris")
        lcm.add_fact("germany", "capital_of", "berlin")
        lcm.learn(n_iterations=50)
        
        results = lcm.inverse_query("paris", "capital_of", k=1)
        
        assert len(results) > 0
        assert results[0][0] == "france", f"Expected france, got {results[0][0]}"
    
    def test_analogy(self):
        """Test analogical reasoning."""
        lcm = GeometricLCM(dim=256)
        
        lcm.add_fact("france", "capital_of", "paris")
        lcm.add_fact("germany", "capital_of", "berlin")
        lcm.add_fact("japan", "capital_of", "tokyo")
        lcm.learn(n_iterations=50)
        
        # france:paris :: germany:?
        results = lcm.analogy("france", "paris", "germany", k=1)
        
        assert len(results) > 0
        assert results[0][0] == "berlin", f"Expected berlin, got {results[0][0]}"
    
    def test_analogy_cross_domain(self):
        """Test analogies don't cross domains incorrectly."""
        lcm = GeometricLCM(dim=256)
        
        # Geography
        lcm.add_fact("france", "capital_of", "paris")
        lcm.add_fact("germany", "capital_of", "berlin")
        
        # Literature
        lcm.add_fact("melville", "wrote", "moby_dick")
        lcm.add_fact("shakespeare", "wrote", "hamlet")
        
        lcm.learn(n_iterations=50)
        
        # Geography analogy should stay in geography
        results = lcm.analogy("france", "paris", "germany", k=1)
        assert results[0][0] == "berlin"
        
        # Literature analogy should stay in literature
        results = lcm.analogy("melville", "moby_dick", "shakespeare", k=1)
        assert results[0][0] == "hamlet"
    
    def test_similar(self):
        """Test finding similar entities."""
        lcm = GeometricLCM(dim=256)
        
        lcm.add_fact("france", "capital_of", "paris")
        lcm.add_fact("germany", "capital_of", "berlin")
        lcm.add_fact("italy", "capital_of", "rome")
        lcm.learn(n_iterations=50)
        
        results = lcm.similar("paris", k=3)
        
        assert len(results) > 0
        # Other capitals should be similar to paris
        similar_names = [r[0] for r in results]
        assert "berlin" in similar_names or "rome" in similar_names
    
    def test_multi_hop(self):
        """Test multi-hop reasoning."""
        lcm = GeometricLCM(dim=256)
        
        lcm.add_fact("france", "capital_of", "paris")
        lcm.add_fact("france", "located_in", "europe")
        lcm.learn(n_iterations=50)
        
        # france --capital_of--> ?
        results = lcm.multi_hop("france", ["capital_of"], k=1)
        
        assert len(results) > 0
        assert results[0][0] == "paris"
    
    def test_incremental_learning(self):
        """Test that new facts can be added incrementally."""
        lcm = GeometricLCM(dim=256)
        
        # Initial facts
        lcm.add_fact("france", "capital_of", "paris")
        lcm.add_fact("germany", "capital_of", "berlin")
        lcm.learn(n_iterations=30)
        
        # Add new fact
        lcm.add_fact("japan", "capital_of", "tokyo")
        lcm.learn(n_iterations=30)
        
        # Should work for new entity
        results = lcm.analogy("france", "paris", "japan", k=1)
        assert results[0][0] == "tokyo"
    
    def test_relation_consistency(self):
        """Test that relations achieve high consistency."""
        lcm = GeometricLCM(dim=256)
        
        # Add multiple instances of same relation
        pairs = [
            ("france", "paris"),
            ("germany", "berlin"),
            ("japan", "tokyo"),
            ("italy", "rome"),
            ("spain", "madrid"),
        ]
        
        for country, capital in pairs:
            lcm.add_fact(country, "capital_of", capital)
        
        lcm.learn(n_iterations=100, target_consistency=0.95)
        
        consistency = lcm.relations["capital_of"].consistency
        assert consistency > 0.9, f"Should have >90% consistency, got {consistency:.1%}"
    
    def test_natural_language_ask(self):
        """Test natural language question answering."""
        lcm = GeometricLCM(dim=256)
        
        lcm.ingest("Paris is the capital of France.")
        lcm.ingest("Berlin is the capital of Germany.")
        lcm.learn(n_iterations=30)
        
        answer = lcm.ask("What is the capital of France?")
        
        assert "paris" in answer.lower()
    
    def test_natural_language_tell(self):
        """Test natural language fact ingestion."""
        lcm = GeometricLCM(dim=256)
        
        result = lcm.tell("Tokyo is the capital of Japan.")
        
        assert "learned" in result.lower()
        assert "japan" in lcm.entities
        assert "tokyo" in lcm.entities
    
    def test_persistence(self):
        """Test save and load."""
        import tempfile
        import os
        
        lcm1 = GeometricLCM(dim=256)
        lcm1.add_fact("france", "capital_of", "paris")
        lcm1.add_fact("germany", "capital_of", "berlin")
        lcm1.learn(n_iterations=30)
        
        # Save
        filepath = tempfile.mktemp(suffix=".json")
        lcm1.save(filepath)
        
        # Load into new instance
        lcm2 = GeometricLCM()
        lcm2.load(filepath)
        
        # Verify
        assert len(lcm2.entities) == len(lcm1.entities)
        assert len(lcm2.relations) == len(lcm1.relations)
        
        results = lcm2.query("france", "capital_of", k=1)
        assert results[0][0] == "paris"
        
        # Cleanup
        os.remove(filepath)


class TestScaling:
    """Test scaling behavior."""
    
    def test_many_facts(self):
        """Test with many facts."""
        lcm = GeometricLCM(dim=256)
        
        # Add 50 country-capital pairs
        for i in range(50):
            lcm.add_fact(f"country_{i}", "capital_of", f"capital_{i}")
        
        lcm.learn(n_iterations=100, target_consistency=0.95)
        
        # Should still work
        results = lcm.query("country_0", "capital_of", k=1)
        assert results[0][0] == "capital_0"
        
        # Analogies should work
        results = lcm.analogy("country_0", "capital_0", "country_25", k=1)
        assert results[0][0] == "capital_25"
    
    def test_multiple_relations(self):
        """Test with multiple relation types."""
        lcm = GeometricLCM(dim=256)
        
        # Different relation types
        lcm.add_fact("france", "capital_of", "paris")
        lcm.add_fact("germany", "capital_of", "berlin")
        lcm.add_fact("melville", "wrote", "moby_dick")
        lcm.add_fact("shakespeare", "wrote", "hamlet")
        lcm.add_fact("paris", "located_in", "france")
        lcm.add_fact("berlin", "located_in", "germany")
        
        lcm.learn(n_iterations=50)
        
        # Each relation should work independently
        assert lcm.query("france", "capital_of", k=1)[0][0] == "paris"
        assert lcm.query("melville", "wrote", k=1)[0][0] == "moby_dick"


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TruthSpace LCM Core Test Suite")
    print("=" * 60)
    print()
    
    test_classes = [
        TestVocabulary,
        TestCosine,
        TestGeometricLCM,
        TestScaling,
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
