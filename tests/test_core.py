"""
Tests for TruthSpace LCM Core - Concept Language System

Tests the core functionality:
- Vocabulary: hash-based word positions, IDF weighting, text encoding
- ConceptLanguage: concept frames, extraction, primitives
- ConceptKnowledge: knowledge storage, querying, holographic projection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from truthspace_lcm.core import (
    Vocabulary,
    tokenize,
    word_position,
    cosine_similarity,
    ConceptFrame,
    ConceptExtractor,
    ConceptStore,
    ConceptKnowledge,
    HolographicProjector,
    ConceptQA,
    ACTION_PRIMITIVES,
    SEMANTIC_ROLES,
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


class TestConceptFrame:
    """Test ConceptFrame data structure."""
    
    def test_create_frame(self):
        """Test creating a concept frame."""
        frame = ConceptFrame(
            agent="darcy",
            action="SPEAK",
            patient="elizabeth"
        )
        
        assert frame.agent == "darcy"
        assert frame.action == "SPEAK"
        assert frame.patient == "elizabeth"
    
    def test_frame_to_dict(self):
        """Test converting frame to dict."""
        frame = ConceptFrame(
            agent="holmes",
            action="THINK",
            location="london"
        )
        
        d = frame.to_dict()
        assert d['agent'] == "holmes"
        assert d['action'] == "THINK"
        assert d['location'] == "london"
    
    def test_frame_from_kwargs(self):
        """Test creating frame from kwargs."""
        frame = ConceptFrame(
            agent='alice',
            action='MOVE',
            goal='wonderland'
        )
        
        assert frame.agent == 'alice'
        assert frame.action == 'MOVE'
        assert frame.goal == 'wonderland'


class TestConceptExtractor:
    """Test ConceptExtractor."""
    
    def test_extract_simple_sentence(self):
        """Test extracting from simple sentence."""
        extractor = ConceptExtractor()
        
        frame = extractor.extract("Darcy walked to the garden.")
        
        assert frame is not None
        assert frame.agent == "darcy"
        assert frame.action == "MOVE"
    
    def test_extract_speech(self):
        """Test extracting speech action."""
        extractor = ConceptExtractor()
        
        frame = extractor.extract("Elizabeth said hello.")
        
        assert frame is not None
        assert frame.action == "SPEAK"
    
    def test_extract_thought(self):
        """Test extracting thought action."""
        extractor = ConceptExtractor()
        
        frame = extractor.extract("Holmes thought about the case.")
        
        assert frame is not None
        assert frame.action == "THINK"
    
    def test_action_primitives_exist(self):
        """Test that action primitives are defined."""
        assert 'MOVE' in ACTION_PRIMITIVES
        assert 'SPEAK' in ACTION_PRIMITIVES
        assert 'THINK' in ACTION_PRIMITIVES
        assert 'PERCEIVE' in ACTION_PRIMITIVES
        assert 'FEEL' in ACTION_PRIMITIVES
    
    def test_semantic_roles_exist(self):
        """Test that semantic roles are defined."""
        assert 'AGENT' in SEMANTIC_ROLES
        assert 'PATIENT' in SEMANTIC_ROLES
        assert 'LOCATION' in SEMANTIC_ROLES


class TestConceptStore:
    """Test ConceptStore."""
    
    def test_add_and_query(self):
        """Test adding and querying frames."""
        store = ConceptStore(dim=64)
        
        frame1 = ConceptFrame(agent="darcy", action="SPEAK")
        frame2 = ConceptFrame(agent="elizabeth", action="MOVE")
        
        store.add(frame1, "Darcy spoke")
        store.add(frame2, "Elizabeth walked")
        
        # Query for SPEAK action
        query = ConceptFrame(action="SPEAK")
        results = store.query(query, k=1)
        
        assert len(results) > 0
        assert results[0][0].agent == "darcy"
    
    def test_store_frames(self):
        """Test store frames list."""
        store = ConceptStore(dim=64)
        
        assert len(store.frames) == 0
        
        store.add(ConceptFrame(agent="test"), "test")
        assert len(store.frames) == 1


class TestConceptKnowledge:
    """Test ConceptKnowledge."""
    
    def test_add_frame(self):
        """Test adding frames to knowledge base."""
        kb = ConceptKnowledge(dim=64)
        
        frame = ConceptFrame(agent="darcy", action="SPEAK", patient="elizabeth")
        kb.add_frame(frame, "Darcy spoke to Elizabeth", "Test")
        
        assert len(kb.frames) == 1
        assert "darcy" in kb.entities
    
    def test_query_by_entity(self):
        """Test querying by entity."""
        kb = ConceptKnowledge(dim=64)
        
        frame1 = ConceptFrame(agent="darcy", action="SPEAK")
        frame2 = ConceptFrame(agent="darcy", action="MOVE")
        frame3 = ConceptFrame(agent="elizabeth", action="THINK")
        
        kb.add_frame(frame1, "Darcy spoke", "Test")
        kb.add_frame(frame2, "Darcy walked", "Test")
        kb.add_frame(frame3, "Elizabeth thought", "Test")
        
        results = kb.query_by_entity("darcy", k=10)
        assert len(results) == 2
    
    def test_query_by_action(self):
        """Test querying by action."""
        kb = ConceptKnowledge(dim=64)
        
        frame1 = ConceptFrame(agent="darcy", action="SPEAK")
        frame2 = ConceptFrame(agent="elizabeth", action="SPEAK")
        frame3 = ConceptFrame(agent="holmes", action="THINK")
        
        kb.add_frame(frame1, "Darcy spoke", "Test")
        kb.add_frame(frame2, "Elizabeth spoke", "Test")
        kb.add_frame(frame3, "Holmes thought", "Test")
        
        results = kb.query_by_action("SPEAK", k=10)
        assert len(results) == 2


class TestHolographicProjector:
    """Test HolographicProjector."""
    
    def test_detect_who_axis(self):
        """Test detecting WHO question axis."""
        kb = ConceptKnowledge(dim=64)
        projector = HolographicProjector(kb)
        
        axis, entity = projector.detect_question_axis("Who is Darcy?")
        assert axis == "WHO"
        assert entity == "darcy"
    
    def test_detect_what_axis(self):
        """Test detecting WHAT question axis."""
        kb = ConceptKnowledge(dim=64)
        projector = HolographicProjector(kb)
        
        axis, entity = projector.detect_question_axis("What did Holmes do?")
        assert axis == "WHAT"
        assert entity == "holmes"
    
    def test_detect_where_axis(self):
        """Test detecting WHERE question axis."""
        kb = ConceptKnowledge(dim=64)
        projector = HolographicProjector(kb)
        
        axis, entity = projector.detect_question_axis("Where is London?")
        assert axis == "WHERE"
        assert entity == "london"


class TestConceptQA:
    """Test ConceptQA end-to-end."""
    
    def test_ingest_and_ask(self):
        """Test ingesting text and asking questions."""
        qa = ConceptQA()
        
        # Ingest some text
        qa.ingest_text("Darcy walked to the garden. Elizabeth spoke to Jane.", "Test")
        
        # Ask a question
        result = qa.ask_detailed("Who is Darcy?")
        
        assert result['axis'] == "WHO"
        assert result['entity'] == "darcy"
    
    def test_ask_simple(self):
        """Test simple ask interface."""
        qa = ConceptQA()
        
        qa.ingest_text("Holmes thought about the mystery.", "Test")
        
        answer = qa.ask("Who is Holmes?")
        assert isinstance(answer, str)
        assert len(answer) > 0


def run_tests():
    """Run all tests."""
    test_classes = [
        TestVocabulary,
        TestConceptFrame,
        TestConceptExtractor,
        TestConceptStore,
        TestConceptKnowledge,
        TestHolographicProjector,
        TestConceptQA,
    ]
    
    total = 0
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        instance = test_class()
        
        for name in dir(instance):
            if name.startswith("test_"):
                total += 1
                try:
                    getattr(instance, name)()
                    print(f"  ✓ {name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {name}: {type(e).__name__}: {e}")
                    failed += 1
    
    print("\n" + "=" * 40)
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
