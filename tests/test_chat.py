"""
Tests for TruthSpace LCM Chat - Holographic Concept Q&A

Tests the chat functionality:
- ConceptQA initialization
- Question answering with holographic projection
- Entity and action queries
- Commands
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from truthspace_lcm.core import ConceptQA, ConceptKnowledge, HolographicProjector


class TestConceptQAChat:
    """Test ConceptQA chat functionality."""
    
    def test_initialization(self):
        """Test QA initialization."""
        qa = ConceptQA()
        
        assert qa.knowledge is not None
        assert qa.projector is not None
    
    def test_load_corpus(self):
        """Test loading corpus."""
        qa = ConceptQA()
        
        corpus_path = Path(__file__).parent.parent / 'truthspace_lcm' / 'concept_corpus.json'
        
        if corpus_path.exists():
            count = qa.load_corpus(str(corpus_path))
            assert count > 0, "Should load frames from corpus"
            assert len(qa.knowledge.frames) > 0
            assert len(qa.knowledge.entities) > 0
    
    def test_ask_who_question(self):
        """Test WHO question."""
        qa = ConceptQA()
        
        corpus_path = Path(__file__).parent.parent / 'truthspace_lcm' / 'concept_corpus.json'
        
        if corpus_path.exists():
            qa.load_corpus(str(corpus_path))
            
            result = qa.ask_detailed("Who is Darcy?")
            
            assert result['axis'] == "WHO"
            assert result['entity'] == "darcy"
            assert len(result['answers']) > 0
    
    def test_ask_what_question(self):
        """Test WHAT question."""
        qa = ConceptQA()
        
        corpus_path = Path(__file__).parent.parent / 'truthspace_lcm' / 'concept_corpus.json'
        
        if corpus_path.exists():
            qa.load_corpus(str(corpus_path))
            
            result = qa.ask_detailed("What did Holmes do?")
            
            assert result['axis'] == "WHAT"
            assert result['entity'] == "holmes"
    
    def test_ask_returns_string(self):
        """Test that ask() returns a string."""
        qa = ConceptQA()
        
        corpus_path = Path(__file__).parent.parent / 'truthspace_lcm' / 'concept_corpus.json'
        
        if corpus_path.exists():
            qa.load_corpus(str(corpus_path))
            
            answer = qa.ask("Who is Elizabeth?")
            
            assert isinstance(answer, str)
            assert len(answer) > 0
    
    def test_ingest_text(self):
        """Test ingesting new text."""
        qa = ConceptQA()
        
        count = qa.ingest_text(
            "The detective walked into the room. He examined the evidence carefully.",
            source="Test"
        )
        
        assert count > 0, "Should extract frames from text"
    
    def test_query_ingested_text(self):
        """Test querying ingested text."""
        qa = ConceptQA()
        
        qa.ingest_text(
            "Sherlock examined the clues. Watson wrote in his journal.",
            source="Test"
        )
        
        # Query for entities we just ingested
        result = qa.ask_detailed("Who is Sherlock?")
        assert result['entity'] == "sherlock"


class TestHolographicProjectorChat:
    """Test HolographicProjector in chat context."""
    
    def test_axis_detection_variations(self):
        """Test various question formats."""
        kb = ConceptKnowledge(dim=64)
        projector = HolographicProjector(kb)
        
        # WHO variations
        axis, entity = projector.detect_question_axis("Who is Darcy?")
        assert axis == "WHO"
        
        axis, entity = projector.detect_question_axis("Whom did Elizabeth meet?")
        assert axis == "WHO"
        
        # WHAT variations
        axis, entity = projector.detect_question_axis("What happened to Alice?")
        assert axis == "WHAT"
        
        # WHERE variations
        axis, entity = projector.detect_question_axis("Where did Holmes go?")
        assert axis == "WHERE"
    
    def test_entity_extraction_from_question(self):
        """Test entity extraction from questions."""
        kb = ConceptKnowledge(dim=64)
        projector = HolographicProjector(kb)
        
        # Should extract capitalized names, skipping titles like Mr/Mrs
        _, entity = projector.detect_question_axis("Who is Mr. Darcy?")
        assert entity == "darcy"  # Skip title, get actual name
        
        _, entity = projector.detect_question_axis("What did Sherlock Holmes do?")
        assert entity == "sherlock"
    
    def test_project_to_english(self):
        """Test projecting frame to English."""
        kb = ConceptKnowledge(dim=64)
        projector = HolographicProjector(kb)
        
        frame = {
            'agent': 'darcy',
            'action': 'SPEAK',
            'patient': 'elizabeth',
        }
        
        english = projector.project_to_english(frame, 'WHO')
        assert isinstance(english, str)
        assert 'darcy' in english.lower() or 'Darcy' in english


class TestChatIntegration:
    """Integration tests for chat system."""
    
    def test_full_pipeline(self):
        """Test full question-answer pipeline."""
        qa = ConceptQA()
        
        # Ingest some knowledge
        qa.ingest_text(
            "Captain Ahab hunted the white whale. Ishmael narrated the story.",
            source="Moby Dick Test"
        )
        
        # Ask about it
        answer = qa.ask("Who is Ahab?")
        
        assert isinstance(answer, str)
        # Should mention something about Ahab
    
    def test_cross_language_concepts(self):
        """Test that concept frames are language-agnostic."""
        qa = ConceptQA()
        
        # Ingest English
        qa.ingest_text("The knight walked to the castle.", source="English")
        
        # Ingest Spanish-like (simulated)
        qa.ingest_text("El caballero caminó al castillo.", source="Spanish")
        
        # Both should be in the knowledge base
        assert len(qa.knowledge.frames) >= 2


def run_tests():
    """Run all tests."""
    test_classes = [
        TestConceptQAChat,
        TestHolographicProjectorChat,
        TestChatIntegration,
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
