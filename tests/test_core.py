"""
Tests for TruthSpace LCM Core - Geometric Chat System

Tests the GCS-aligned core functionality:
- Vocabulary: hash-based word positions, IDF weighting, text encoding
- KnowledgeBase: facts, Q&A pairs, semantic search
- StyleEngine: style extraction, classification, transfer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from truthspace_lcm.core import (
    Vocabulary,
    KnowledgeBase,
    StyleEngine,
    Style,
    cosine_similarity,
    tokenize,
    detect_question_type,
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


class TestKnowledgeBase:
    """Test KnowledgeBase."""
    
    def test_add_fact(self):
        """Test adding facts."""
        vocab = Vocabulary(dim=64)
        kb = KnowledgeBase(vocab)
        
        fact = kb.add_fact("The sky is blue.", source="test")
        
        assert fact.content == "The sky is blue."
        assert fact.source == "test"
        assert len(fact.encoding) == 64
        assert fact.id in kb.facts
    
    def test_add_qa_pair(self):
        """Test adding Q&A pairs."""
        vocab = Vocabulary(dim=64)
        kb = KnowledgeBase(vocab)
        
        qa = kb.add_qa_pair("What is the color of the sky?", "The sky is blue.", source="test")
        
        assert qa.question == "What is the color of the sky?"
        assert qa.answer == "The sky is blue."
        assert qa.question_type == "WHAT"
        assert qa.id in kb.qa_pairs
    
    def test_search_qa(self):
        """Test Q&A search."""
        vocab = Vocabulary(dim=64)
        kb = KnowledgeBase(vocab)
        
        kb.add_qa_pair("What is Python?", "Python is a programming language.")
        kb.add_qa_pair("What is Java?", "Java is a programming language.")
        kb.add_qa_pair("Who is Einstein?", "Einstein was a physicist.")
        
        results = kb.search_qa("What is Python programming?", k=2)
        
        assert len(results) == 2
        best_qa, best_sim = results[0]
        assert "Python" in best_qa.question or "programming" in best_qa.answer
        assert best_sim > 0.2
    
    def test_search_facts(self):
        """Test fact search."""
        vocab = Vocabulary(dim=64)
        kb = KnowledgeBase(vocab)
        
        kb.add_fact("Cats are mammals.")
        kb.add_fact("Dogs are mammals.")
        kb.add_fact("Python is a programming language.")
        
        results = kb.search_facts("Tell me about cats", k=2)
        
        assert len(results) == 2
        best_fact, best_sim = results[0]
        assert "Cats" in best_fact.content or "mammals" in best_fact.content
    
    def test_ingest_text(self):
        """Test text ingestion."""
        vocab = Vocabulary(dim=64)
        kb = KnowledgeBase(vocab)
        
        text = "Captain Ahab is the captain of the Pequod. He is obsessed with Moby Dick."
        counts = kb.ingest_text(text, source="moby_dick")
        
        assert counts['facts'] >= 2
        assert len(kb.facts) >= 2


class TestQuestionType:
    """Test question type detection."""
    
    def test_who_questions(self):
        assert detect_question_type("Who is the president?") == "WHO"
        assert detect_question_type("Who was Einstein?") == "WHO"
    
    def test_what_questions(self):
        assert detect_question_type("What is Python?") == "WHAT"
        assert detect_question_type("What does this mean?") == "WHAT"
    
    def test_where_questions(self):
        assert detect_question_type("Where is Paris?") == "WHERE"
        assert detect_question_type("Where did it happen?") == "WHERE"
    
    def test_when_questions(self):
        assert detect_question_type("When did WWII end?") == "WHEN"
        assert detect_question_type("What year was it?") == "WHEN"
    
    def test_why_questions(self):
        assert detect_question_type("Why did he leave?") == "WHY"
        assert detect_question_type("Why is the sky blue?") == "WHY"
    
    def test_how_questions(self):
        assert detect_question_type("How does it work?") == "HOW"
        assert detect_question_type("How to cook pasta?") == "HOW"
    
    def test_unknown_questions(self):
        assert detect_question_type("Is this correct?") == "UNKNOWN"


class TestStyleEngine:
    """Test StyleEngine."""
    
    def test_extract_style(self):
        """Test style extraction."""
        vocab = Vocabulary(dim=64)
        engine = StyleEngine(vocab)
        
        exemplars = [
            "The methodology demonstrates significant improvements.",
            "One must consider the theoretical implications.",
            "Results indicate a statistically significant correlation.",
        ]
        
        style = engine.extract_style(exemplars, "formal")
        
        assert style.name == "formal"
        assert style.exemplar_count == 3
        assert len(style.centroid) == 64
        assert "formal" in engine.styles
    
    def test_classify(self):
        """Test style classification."""
        vocab = Vocabulary(dim=64)
        engine = StyleEngine(vocab)
        
        # Extract two styles
        engine.extract_style([
            "Hey, this is cool!",
            "So basically, it works.",
            "Pretty simple, right?",
        ], "casual")
        
        engine.extract_style([
            "The implementation demonstrates improvements.",
            "One must consider the implications.",
            "Results indicate significant findings.",
        ], "formal")
        
        # Classify casual text
        results = engine.classify("Hey, that's pretty neat!")
        assert results[0][0] == "casual", f"Expected casual, got {results[0][0]}"
        
        # Classify formal text
        results = engine.classify("The methodology indicates significant results.")
        assert results[0][0] == "formal", f"Expected formal, got {results[0][0]}"
    
    def test_transfer(self):
        """Test style transfer."""
        vocab = Vocabulary(dim=64)
        engine = StyleEngine(vocab)
        
        engine.extract_style([
            "The implementation demonstrates improvements.",
            "One must consider the implications.",
        ], "formal")
        
        styled_vec, words = engine.transfer("This is a test.", "formal", strength=0.5)
        
        assert len(styled_vec) == 64
        assert isinstance(words, list)
    
    def test_style_difference(self):
        """Test style difference computation."""
        vocab = Vocabulary(dim=64)
        engine = StyleEngine(vocab)
        
        engine.extract_style(["Hey, cool!", "Nice one!"], "casual")
        engine.extract_style(["The methodology is sound.", "Results are significant."], "formal")
        
        direction, toward_formal, toward_casual = engine.style_difference("casual", "formal")
        
        assert len(direction) == 64
        assert isinstance(toward_formal, list)
        assert isinstance(toward_casual, list)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TruthSpace LCM Core Test Suite")
    print("=" * 60)
    print()
    
    test_classes = [
        TestVocabulary,
        TestCosine,
        TestKnowledgeBase,
        TestQuestionType,
        TestStyleEngine,
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
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
        
        if failed == 0:
            print(f"  ✓ All {passed} tests passed")
        else:
            print(f"  {passed} passed, {failed} failed")
        
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
