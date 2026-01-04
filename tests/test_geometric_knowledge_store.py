"""
Tests for GeometricKnowledgeStore

Verifies the Phase 1 implementation of geometric knowledge persistence.

Author: Lesley Gushurst
License: GPLv3
"""

import tempfile
import os
from pathlib import Path

from truthspace_lcm.core.knowledge import (
    Concept, 
    ConceptLevel, 
    GeometricKnowledgeStore
)


class TestConcept:
    """Tests for the Concept class."""
    
    def test_create_concept(self):
        """Test basic concept creation."""
        concept = Concept(
            words={'george', 'washington', 'president'},
            source='test'
        )
        
        assert concept.id is not None
        assert len(concept.words) == 3
        assert concept.temporary is True
        assert concept.use_count == 0
    
    def test_record_use(self):
        """Test recording usage statistics."""
        concept = Concept(words={'test'})
        
        concept.record_use(success=True)
        concept.record_use(success=True)
        concept.record_use(success=False)
        
        assert concept.use_count == 3
        assert concept.success_count == 2
        assert abs(concept.success_rate - 2/3) < 0.01
    
    def test_qualifies_for_promotion(self):
        """Test promotion qualification criteria (geometric)."""
        concept = Concept(words={'test'})
        
        # No uses yet - doesn't qualify
        assert not concept.qualifies_for_promotion(threshold=0.5)
        
        # Add uses with high success rate
        for _ in range(5):
            concept.record_use(success=True)
        
        assert concept.use_count == 5
        assert concept.success_rate == 1.0
        assert concept.stability == 1.0  # No drift yet
        assert concept.confidence == 1.0  # sqrt(1.0 * 1.0)
        
        # Qualifies because confidence (1.0) >= threshold (0.5)
        assert concept.qualifies_for_promotion(threshold=0.5)
        
        # Test with low success rate
        low_success = Concept(words={'test2'})
        for _ in range(5):
            low_success.record_use(success=False)
        
        assert low_success.confidence == 0.0  # sqrt(0.0 * 1.0)
        assert not low_success.qualifies_for_promotion(threshold=0.5)
    
    def test_promotion(self):
        """Test promoting a concept."""
        concept = Concept(words={'test'}, temporary=True)
        assert concept.temporary is True
        
        concept.promote()
        assert concept.temporary is False
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        original = Concept(
            words={'george', 'washington'},
            quaternion=(0.5, 0.5, 0.5, 0.5),
            level=ConceptLevel.FACT,
            source='test',
        )
        original.record_use(success=True)
        
        data = original.to_dict()
        restored = Concept.from_dict(data)
        
        assert restored.id == original.id
        assert restored.words == original.words
        assert restored.quaternion == original.quaternion
        assert restored.use_count == original.use_count
        assert restored.level == original.level


class TestGeometricKnowledgeStore:
    """Tests for the GeometricKnowledgeStore class."""
    
    def test_create_store(self):
        """Test basic store creation."""
        store = GeometricKnowledgeStore(name='test', dims=12)
        
        assert store.name == 'test'
        assert store.dims == 12
        assert len(store) == 0
    
    def test_extract_words(self):
        """Test word extraction from text (geometric stop word detection)."""
        store = GeometricKnowledgeStore(name='test')
        text = "George Washington was the first President of the United States."
        words = store.extract_words(text)
        
        # Should have content words (length >= 3)
        assert 'george' in words
        assert 'washington' in words
        assert 'president' in words
        assert 'united' in words
        assert 'states' in words
        
        # Should NOT have short words (< 3 chars)
        assert 'of' not in words
        
        # Note: 'the', 'was' are filtered by length (< 3) or geometric detection
        # In an empty store, only length filtering applies
    
    def test_word_overlap(self):
        """Test Jaccard similarity calculation."""
        words_a = {'george', 'washington', 'president'}
        words_b = {'george', 'washington', 'general'}
        words_c = {'thomas', 'jefferson', 'president'}
        
        # Same sets
        assert GeometricKnowledgeStore.word_overlap(words_a, words_a) == 1.0
        
        # Partial overlap
        overlap_ab = GeometricKnowledgeStore.word_overlap(words_a, words_b)
        assert 0 < overlap_ab < 1  # 2/4 = 0.5
        
        # Less overlap
        overlap_ac = GeometricKnowledgeStore.word_overlap(words_a, words_c)
        assert overlap_ac < overlap_ab  # 1/5 = 0.2
        
        # No overlap
        words_d = {'completely', 'different', 'words'}
        assert GeometricKnowledgeStore.word_overlap(words_a, words_d) == 0.0
    
    def test_add_concept(self):
        """Test adding concepts to the store."""
        store = GeometricKnowledgeStore(name='test')
        
        concept = Concept(words={'george', 'washington', 'president'})
        store.add(concept)
        
        assert len(store) == 1
        assert concept.id in store
        assert store.get(concept.id) == concept
    
    def test_add_from_text(self):
        """Test adding concepts from text."""
        store = GeometricKnowledgeStore(name='test')
        
        concept = store.add_from_text(
            "George Washington was the first President.",
            source='test'
        )
        
        assert len(store) == 1
        assert 'george' in concept.words
        assert 'washington' in concept.words
    
    def test_query(self):
        """Test querying the store."""
        store = GeometricKnowledgeStore(name='test')
        
        # Add some concepts
        store.add_from_text("George Washington was the first President.")
        store.add_from_text("Thomas Jefferson wrote the Declaration of Independence.")
        store.add_from_text("Benjamin Franklin was a diplomat and inventor.")
        
        # Query for Washington
        results = store.query("Who was George Washington?")
        
        assert len(results) > 0
        top_concept, top_similarity = results[0]
        assert 'washington' in top_concept.words
        assert top_similarity > 0
    
    def test_geometry_update(self):
        """Test that geometry is updated when adding concepts."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        
        store.add_from_text("George Washington was President.")
        store.add_from_text("Thomas Jefferson was President.")
        store.add_from_text("Cats and dogs are pets.")
        
        # Should have similarity matrix and positions
        assert store.similarity_matrix is not None
        assert store.positions is not None
        
        # Matrix should be 3x3
        assert store.similarity_matrix.shape == (3, 3)
        
        # Positions should be 3x4
        assert store.positions.shape == (3, 4)
        
        # Washington and Jefferson should be more similar than Washington and pets
        sim_wj = store.similarity_matrix[0, 1]
        sim_wp = store.similarity_matrix[0, 2]
        assert sim_wj > sim_wp
    
    def test_remove_concept(self):
        """Test removing concepts."""
        store = GeometricKnowledgeStore(name='test')
        
        c1 = store.add_from_text("George Washington")
        c2 = store.add_from_text("Thomas Jefferson")
        
        assert len(store) == 2
        
        store.remove(c1.id)
        
        assert len(store) == 1
        assert c1.id not in store
        assert c2.id in store
    
    def test_promote_qualifying(self):
        """Test promoting qualifying concepts (geometric criteria)."""
        store = GeometricKnowledgeStore(name='test')
        
        c1 = store.add_from_text("George Washington")
        c2 = store.add_from_text("Thomas Jefferson")
        c3 = store.add_from_text("Failed concept")
        
        # c1: high success rate - should qualify
        for _ in range(5):
            c1.record_use(success=True)
        
        # c2: also high success - should qualify
        for _ in range(3):
            c2.record_use(success=True)
        
        # c3: low success rate - should NOT qualify
        for _ in range(5):
            c3.record_use(success=False)
        
        # Check confidences
        assert c1.confidence == 1.0  # sqrt(1.0 * 1.0)
        assert c2.confidence == 1.0  # sqrt(1.0 * 1.0)
        assert c3.confidence == 0.0  # sqrt(0.0 * 1.0)
        
        # Threshold is 0.5 (critical line)
        assert store.promotion_threshold() == 0.5
        
        promoted = store.promote_qualifying()
        
        # c1 and c2 promoted (confidence >= 0.5)
        assert c1.id in promoted
        assert c2.id in promoted
        # c3 not promoted (confidence < 0.5)
        assert c3.id not in promoted
        
        assert not c1.temporary
        assert not c2.temporary
        assert c3.temporary
    
    def test_save_and_load(self):
        """Test saving and loading the store."""
        store = GeometricKnowledgeStore(name='test', dims=8)
        
        store.add_from_text("George Washington was President.")
        store.add_from_text("Thomas Jefferson wrote the Declaration.")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            store.save(temp_path)
            
            # Load it back
            loaded = GeometricKnowledgeStore.load(temp_path)
            
            assert loaded.name == store.name
            assert loaded.dims == store.dims
            assert len(loaded) == len(store)
            
            # Check concepts preserved
            for original in store.concepts:
                loaded_concept = loaded.get(original.id)
                assert loaded_concept is not None
                assert loaded_concept.words == original.words
            
            # Check geometry preserved
            assert loaded.similarity_matrix is not None
            assert loaded.positions is not None
            
        finally:
            os.unlink(temp_path)
    
    def test_merge_stores(self):
        """Test merging two stores."""
        store1 = GeometricKnowledgeStore(name='store1')
        store2 = GeometricKnowledgeStore(name='store2')
        
        c1 = store1.add_from_text("George Washington")
        c2 = store2.add_from_text("Thomas Jefferson")
        c3 = store2.add_from_text("Benjamin Franklin")
        
        # Merge store2 into store1
        count = store1.merge(store2)
        
        assert count == 2  # c2 and c3 added
        assert len(store1) == 3
        assert c2.id in store1
        assert c3.id in store1
    
    def test_clear_temporary(self):
        """Test clearing temporary concepts."""
        store = GeometricKnowledgeStore(name='test')
        
        c1 = store.add_from_text("George Washington")
        c2 = store.add_from_text("Thomas Jefferson")
        
        # Promote c1
        c1.promote()
        
        # Clear temporary
        removed = store.clear_temporary()
        
        assert removed == 1
        assert len(store) == 1
        assert c1.id in store
        assert c2.id not in store


class TestIntegration:
    """Integration tests for the knowledge system."""
    
    def test_full_workflow(self):
        """Test a complete workflow: add, query, use, promote, save, load."""
        store = GeometricKnowledgeStore(name='integration_test', dims=8)
        
        # Add knowledge
        store.add_from_text("George Washington was the first President of the United States.")
        store.add_from_text("Thomas Jefferson wrote the Declaration of Independence.")
        store.add_from_text("Benjamin Franklin was a diplomat, inventor, and Founding Father.")
        store.add_from_text("The American Revolution began in 1775.")
        
        assert len(store) == 4
        
        # Query
        results = store.query("Who was the first President?")
        assert len(results) > 0
        
        top_concept, similarity = results[0]
        assert 'washington' in top_concept.words or 'president' in top_concept.words
        
        # Record usage
        for _ in range(5):
            top_concept.record_use(success=True)
        
        # Promote
        promoted = store.promote_qualifying()
        assert top_concept.id in promoted
        
        # Save and reload
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            store.save(temp_path)
            loaded = GeometricKnowledgeStore.load(temp_path)
            
            # Verify loaded store works
            results2 = loaded.query("Who was the first President?")
            assert len(results2) > 0
            
            # Verify promotion persisted
            loaded_concept = loaded.get(top_concept.id)
            assert loaded_concept is not None
            assert not loaded_concept.temporary
            
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
