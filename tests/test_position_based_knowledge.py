"""
Tests for Position-Based Knowledge Architecture (Design 091)

The key insight: POSITION IS EVERYTHING.
- Concepts start at origin
- Success moves them toward query positions
- Failure moves them away
- Concepts past the critical line (0.5) persist
- Concepts inside the critical line fade and can be pruned

Author: Lesley Gushurst
License: GPLv3
"""

import sys
sys.path.insert(0, '.')

from truthspace_lcm.core.knowledge import Concept, GeometricKnowledgeStore, CRITICAL_LINE


class TestConcept:
    """Tests for the simplified Concept class."""
    
    def test_create_concept(self):
        """New concepts start at origin."""
        c = Concept(words={'test', 'concept'})
        assert c.magnitude == 0.0
        assert c.persists == False
        assert c.position == (0.0, 0.0, 0.0, 0.0)
    
    def test_magnitude(self):
        """Magnitude is the norm of the position vector."""
        c = Concept(words={'test'}, position=(0.3, 0.4, 0.0, 0.0))
        assert abs(c.magnitude - 0.5) < 0.001  # 3-4-5 triangle
    
    def test_persists(self):
        """Concepts past the critical line persist."""
        c1 = Concept(words={'test'}, position=(0.4, 0.0, 0.0, 0.0))
        c2 = Concept(words={'test'}, position=(0.6, 0.0, 0.0, 0.0))
        
        assert c1.persists == False  # 0.4 < 0.5
        assert c2.persists == True   # 0.6 >= 0.5
    
    def test_move_toward(self):
        """move_toward pulls concept toward target."""
        c = Concept(words={'test'}, position=(0.0, 0.0, 0.0, 0.0))
        target = (1.0, 0.0, 0.0, 0.0)
        
        c.move_toward(target, strength=0.5)
        
        assert abs(c.position[0] - 0.5) < 0.001
        assert c.magnitude > 0.0
    
    def test_move_away(self):
        """move_away pushes concept away from target."""
        c = Concept(words={'test'}, position=(0.5, 0.0, 0.0, 0.0))
        target = (1.0, 0.0, 0.0, 0.0)
        
        c.move_away(target, strength=0.5)
        
        # Should move away from target (toward origin in this case)
        assert c.position[0] < 0.5
    
    def test_serialization(self):
        """Concepts can be serialized and deserialized."""
        c = Concept(
            words={'test', 'concept'},
            position=(0.5, 0.3, 0.1, 0.0),
            source='test'
        )
        
        data = c.to_dict()
        c2 = Concept.from_dict(data)
        
        assert c2.words == c.words
        assert c2.position == c.position
        assert c2.source == c.source


class TestGeometricKnowledgeStore:
    """Tests for the GeometricKnowledgeStore."""
    
    def test_create_store(self):
        """Store can be created."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        assert len(store) == 0
        assert store.dims == 4
    
    def test_add_concept(self):
        """Concepts can be added to store."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        c = Concept(words={'test'})
        store.add(c)
        
        assert len(store) == 1
        assert store.get(c.id) == c
    
    def test_add_from_text(self):
        """Concepts can be created from text."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        c = store.add_from_text('George Washington was president')
        
        assert len(store) == 1
        assert 'washington' in c.words
        assert c.magnitude == 0.0  # Starts at origin
    
    def test_use_success(self):
        """Successful use moves concept toward query position."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        c = store.add_from_text('test concept')
        
        initial_mag = c.magnitude
        query_pos = (0.8, 0.0, 0.0, 0.0)
        
        store.use(c.id, query_pos, success=True)
        
        assert c.magnitude > initial_mag
    
    def test_use_failure(self):
        """Failed use moves concept away from query position."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        c = store.add_from_text('test concept')
        
        # First move it somewhere
        c.position = (0.5, 0.0, 0.0, 0.0)
        initial_mag = c.magnitude
        
        query_pos = (1.0, 0.0, 0.0, 0.0)
        store.use(c.id, query_pos, success=False)
        
        # Should move away from query (toward origin)
        assert c.magnitude < initial_mag
    
    def test_get_persisting_concepts(self):
        """Can filter concepts by persistence."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        
        c1 = Concept(words={'a'}, position=(0.6, 0.0, 0.0, 0.0))
        c2 = Concept(words={'b'}, position=(0.3, 0.0, 0.0, 0.0))
        
        store.add(c1)
        store.add(c2)
        
        persisting = store.get_persisting_concepts()
        fading = store.get_fading_concepts()
        
        assert len(persisting) == 1
        assert len(fading) == 1
        assert c1 in persisting
        assert c2 in fading
    
    def test_prune(self):
        """Prune removes concepts below critical line."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        
        c1 = Concept(words={'a'}, position=(0.6, 0.0, 0.0, 0.0))
        c2 = Concept(words={'b'}, position=(0.3, 0.0, 0.0, 0.0))
        
        store.add(c1)
        store.add(c2)
        
        pruned = store.prune()
        
        assert pruned == 1
        assert len(store) == 1
        assert store.get(c1.id) is not None
        assert store.get(c2.id) is None


class TestLearningDynamics:
    """Tests for the emergent learning dynamics."""
    
    def test_successful_concept_persists(self):
        """Concept with many successful uses crosses critical line."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        c = store.add_from_text('frequently used concept')
        
        query_pos = (0.8, 0.1, 0.0, 0.0)
        
        # Simulate many successful uses
        for _ in range(20):
            store.use(c.id, query_pos, success=True)
        
        assert c.persists == True
        assert c.magnitude >= CRITICAL_LINE
    
    def test_unused_concept_fades(self):
        """Concept with no uses stays at origin."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        c = store.add_from_text('unused concept')
        
        # No uses
        
        assert c.persists == False
        assert c.magnitude == 0.0
    
    def test_differential_persistence(self):
        """Used concepts persist, unused concepts fade."""
        store = GeometricKnowledgeStore(name='test', dims=4)
        
        c1 = store.add_from_text('frequently used')
        c2 = store.add_from_text('rarely used')
        c3 = store.add_from_text('never used')
        
        query_pos = (0.8, 0.0, 0.0, 0.0)
        
        # c1: 20 uses, c2: 5 uses, c3: 0 uses
        for _ in range(20):
            store.use(c1.id, query_pos, success=True)
        for _ in range(5):
            store.use(c2.id, query_pos, success=True)
        
        # Prune
        store.prune()
        
        # Only c1 should remain (crossed critical line)
        assert store.get(c1.id) is not None
        assert store.get(c2.id) is None  # Didn't quite make it
        assert store.get(c3.id) is None  # Never used


def run_all_tests():
    """Run all tests."""
    print("Running Position-Based Knowledge Tests...")
    
    tc = TestConcept()
    tc.test_create_concept()
    tc.test_magnitude()
    tc.test_persists()
    tc.test_move_toward()
    tc.test_move_away()
    tc.test_serialization()
    print("  ✓ TestConcept passed")
    
    ts = TestGeometricKnowledgeStore()
    ts.test_create_store()
    ts.test_add_concept()
    ts.test_add_from_text()
    ts.test_use_success()
    ts.test_use_failure()
    ts.test_get_persisting_concepts()
    ts.test_prune()
    print("  ✓ TestGeometricKnowledgeStore passed")
    
    tl = TestLearningDynamics()
    tl.test_successful_concept_persists()
    tl.test_unused_concept_fades()
    tl.test_differential_persistence()
    print("  ✓ TestLearningDynamics passed")
    
    print("\n✓ ALL TESTS PASSED!")
    print("  Position IS everything.")


if __name__ == '__main__':
    run_all_tests()
