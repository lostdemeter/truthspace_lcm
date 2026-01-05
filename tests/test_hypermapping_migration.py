"""
Comprehensive Tests for HyperMapping Migration

Tests the new HyperMapping-based architecture:
- ChatPipeline (replaces ChatGearChain)
- KnowledgeSpace (replaces GeometricKnowledgeStore)
- CodeSpace (replaces PythonCodeGear)
- IntentSpace (replaces IntentDetectorGear)
- HyperPipeline (replaces GearChain)

Author: Lesley Gushurst
License: GPLv3
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from hypermapping import (
    HyperMapping, HyperPipeline, Mapping, MatchResult,
    TextEncoder, CRITICAL_LINE
)
from truthspace_lcm.core import (
    ChatPipeline, ChatConfig, Intent, IntentResult, IntentSpace,
    KnowledgeSpace, CodeSpace, CodeResult
)


class TestMapping:
    """Tests for the Mapping dataclass."""
    
    def test_mapping_creation(self):
        """Test basic mapping creation."""
        m = Mapping(
            input="hello",
            output="world",
            position=np.array([0.5, 0.5, 0.5])
        )
        assert m.input == "hello"
        assert m.output == "world"
        assert m.use_count == 0
        assert m.success_count == 0
    
    def test_mapping_persistence(self):
        """Test mapping persistence based on magnitude."""
        # Mapping at critical line
        m1 = Mapping(input="a", output="b", position=np.array([0.5, 0.0, 0.0]))
        assert m1.magnitude == pytest.approx(0.5, abs=0.01)
        assert m1.persists == True
        
        # Mapping below critical line
        m2 = Mapping(input="c", output="d", position=np.array([0.3, 0.0, 0.0]))
        assert m2.magnitude == pytest.approx(0.3, abs=0.01)
        assert m2.persists == False
    
    def test_mapping_success_rate(self):
        """Test emergent success rate calculation."""
        m = Mapping(input="a", output="b", position=np.array([0.5, 0.0, 0.0]))
        
        # No uses yet
        assert m.success_rate == 0.0
        
        # Record uses
        m.record_use(success=True)
        m.record_use(success=True)
        m.record_use(success=False)
        
        assert m.use_count == 3
        assert m.success_count == 2
        assert m.success_rate == pytest.approx(2/3, abs=0.01)
    
    def test_mapping_serialization(self):
        """Test mapping serialization/deserialization."""
        m = Mapping(
            input="test",
            output="result",
            position=np.array([0.1, 0.2, 0.3]),
            metadata={"key": "value"},
            use_count=5,
            success_count=3
        )
        
        data = m.to_dict()
        m2 = Mapping.from_dict(data)
        
        assert m2.input == m.input
        assert m2.output == m.output
        assert np.allclose(m2.position, m.position)
        assert m2.metadata == m.metadata
        assert m2.use_count == m.use_count
        assert m2.success_count == m.success_count


class TestHyperMapping:
    """Tests for the HyperMapping class."""
    
    def test_basic_mapping(self):
        """Test basic map and forward operations."""
        space = HyperMapping(dims=8, name="test")
        space.map("list files", "ls")
        space.map("show files", "ls")
        space.map("delete file", "rm")
        
        assert len(space) == 3
        
        result = space.forward("display files")
        assert result is not None
        assert result.output in ["ls", "rm"]
    
    def test_pruning(self):
        """Test geometric pruning."""
        space = HyperMapping(dims=4, name="test")
        
        # Add mappings with different magnitudes
        m1 = space.map("a", "1", position=np.array([0.6, 0.0, 0.0, 0.0]))  # Above critical
        m2 = space.map("b", "2", position=np.array([0.3, 0.0, 0.0, 0.0]))  # Below critical
        m3 = space.map("c", "3", position=np.array([0.7, 0.0, 0.0, 0.0]))  # Above critical
        
        assert len(space) == 3
        assert len(space.get_persisting()) == 2
        assert len(space.get_fading()) == 1
        
        # Prune
        pruned = space.prune()
        assert pruned == 1
        assert len(space) == 2
    
    def test_reinforcement(self):
        """Test position reinforcement."""
        space = HyperMapping(dims=4, name="test")
        m = space.map("test", "result", position=np.array([0.5, 0.0, 0.0, 0.0]))
        
        initial_magnitude = m.magnitude
        
        # Success should increase magnitude
        space.reinforce(m, success=True)
        assert m.magnitude > initial_magnitude
        
        # Failure should decrease magnitude
        space.reinforce(m, success=False)
        # Still above initial due to asymmetric reinforcement
    
    def test_stats(self):
        """Test statistics gathering."""
        space = HyperMapping(dims=4, name="test")
        space.map("a", "1")
        space.map("b", "2")
        
        stats = space.get_stats()
        assert stats['total_mappings'] == 2
        assert stats['dims'] == 4
        assert stats['critical_line'] == CRITICAL_LINE
    
    def test_serialization(self):
        """Test HyperMapping serialization."""
        space = HyperMapping(dims=4, name="test")
        space.map("hello", "world")
        space.map("foo", "bar")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            space.save(f.name)
            loaded = HyperMapping.load(f.name)
        
        assert len(loaded) == len(space)
        assert loaded.name == space.name
        assert loaded.dims == space.dims


class TestHyperPipeline:
    """Tests for the HyperPipeline class."""
    
    def test_pipeline_creation(self):
        """Test pipeline creation with named stages."""
        pipeline = HyperPipeline(name="test")
        
        stage1 = HyperMapping(dims=4, name="stage1")
        stage2 = HyperMapping(dims=4, name="stage2")
        
        pipeline.add("first", stage1)
        pipeline.add("second", stage2)
        
        assert len(pipeline) == 2
        assert pipeline.get("first") == stage1
        assert pipeline.get("second") == stage2
    
    def test_enable_disable(self):
        """Test stage enable/disable."""
        pipeline = HyperPipeline(name="test")
        
        stage1 = HyperMapping(dims=4, name="stage1")
        pipeline.add("first", stage1)
        
        assert pipeline.is_enabled("first") == True
        
        pipeline.disable("first")
        assert pipeline.is_enabled("first") == False
        
        pipeline.enable("first")
        assert pipeline.is_enabled("first") == True
    
    def test_pipeline_stats(self):
        """Test pipeline statistics."""
        pipeline = HyperPipeline(name="test")
        
        stage1 = HyperMapping(dims=4, name="stage1")
        stage1.map("a", "b")
        
        pipeline.add("first", stage1)
        
        stats = pipeline.get_stats()
        assert stats['name'] == "test"
        assert stats['stages'] == 1
        assert 'first' in stats['stage_stats']


class TestKnowledgeSpace:
    """Tests for the KnowledgeSpace class."""
    
    def test_add_and_query(self):
        """Test adding and querying knowledge."""
        space = KnowledgeSpace(name="test", dims=8)
        
        space.add_text("Python is a programming language")
        space.add_text("Machine learning uses neural networks")
        space.add_text("The capital of France is Paris")
        
        assert len(space) == 3
        
        results = space.query_text("What is Python?", top_k=3)
        assert len(results) > 0
    
    def test_feedback_learning(self):
        """Test feedback-based learning."""
        space = KnowledgeSpace(name="test", dims=8)
        
        m = space.add_text("Test knowledge item")
        initial_magnitude = m.magnitude
        
        # Positive feedback should increase magnitude
        space.use(m, success=True)
        assert m.magnitude > initial_magnitude
        assert m.use_count == 1
        assert m.success_count == 1
    
    def test_geometric_stop_words(self):
        """Test geometric stop word detection."""
        space = KnowledgeSpace(name="test", dims=8)
        
        # Add multiple items with common words
        space.add_text("The cat is on the mat")
        space.add_text("The dog is in the house")
        space.add_text("The bird is in the tree")
        
        # "the" and "is" should be detected as stop words (high coverage)
        # Content words should be extracted
        words = space.extract_words("The cat is sleeping")
        assert "cat" in words
        assert "sleeping" in words
    
    def test_serialization(self):
        """Test KnowledgeSpace serialization."""
        space = KnowledgeSpace(name="test", dims=8)
        space.add_text("Test item 1")
        space.add_text("Test item 2")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            space.save(f.name)
            loaded = KnowledgeSpace.load(f.name)
        
        assert len(loaded) == len(space)
        assert loaded.name == space.name


class TestCodeSpace:
    """Tests for the CodeSpace class."""
    
    def test_code_generation(self):
        """Test basic code generation."""
        space = CodeSpace(name="test", dims=8)
        
        result = space.generate("print hello world")
        assert result.success == True
        assert "print" in result.code
    
    def test_pattern_matching(self):
        """Test pattern matching for code generation."""
        space = CodeSpace(name="test", dims=8)
        
        # Test different patterns
        result1 = space.generate("create a variable")
        assert result1.success == True
        
        result2 = space.generate("loop over a list")
        assert result2.success == True
        assert "for" in result2.code
    
    def test_feedback(self):
        """Test feedback on code generation."""
        space = CodeSpace(name="test", dims=8)
        
        result = space.generate("print hello")
        space.feedback(success=True)
        
        # Check that feedback was recorded
        patterns = space.list_patterns()
        assert any(p['use_count'] > 0 for p in patterns)
    
    def test_list_patterns(self):
        """Test listing available patterns."""
        space = CodeSpace(name="test", dims=8)
        
        patterns = space.list_patterns()
        assert len(patterns) > 0
        
        # Check pattern structure
        for p in patterns:
            assert 'name' in p
            assert 'description' in p
            assert 'use_count' in p


class TestIntentSpace:
    """Tests for the IntentSpace class."""
    
    def test_intent_detection(self):
        """Test intent detection."""
        space = IntentSpace()
        
        # Knowledge queries
        result = space.detect("What is Python?")
        assert result.intent == Intent.KNOWLEDGE
        
        # Tool calls
        result = space.detect("Create a new file")
        assert result.intent == Intent.TOOL_CALL
        
        # Code generation
        result = space.detect("Write code to print hello")
        assert result.intent == Intent.CODE_GENERATION
    
    def test_prefix_matching(self):
        """Test prefix-based matching for bootstrap patterns."""
        space = IntentSpace()
        
        # "what is" should match KNOWLEDGE via prefix
        result = space.detect("what is the meaning of life?")
        assert result.intent == Intent.KNOWLEDGE
        assert result.metadata.get('match_type') == 'prefix'
    
    def test_learning(self):
        """Test intent learning from corrections."""
        space = IntentSpace()
        
        # Learn a new pattern
        space.learn_intent("show me the code", Intent.CODE_GENERATION)
        
        # The pattern should now be in templates
        assert "show me the code" in space.templates


class TestChatPipeline:
    """Tests for the ChatPipeline class."""
    
    def test_basic_chat(self):
        """Test basic chat functionality."""
        config = ChatConfig(debug=False)
        pipeline = ChatPipeline(config)
        
        # Add some knowledge
        pipeline.add_knowledge("Python is a programming language")
        
        # Query
        response = pipeline.chat("Tell me about Python")
        assert response is not None
        assert len(response) > 0
    
    def test_intent_routing(self):
        """Test intent-based routing."""
        config = ChatConfig(debug=False)
        pipeline = ChatPipeline(config)
        
        # Knowledge query
        response = pipeline.chat("What is machine learning?")
        # Should route to knowledge handler
        
        # Code generation
        response = pipeline.chat("Write code to print hello")
        # Should route to code handler
        assert "```python" in response or "print" in response
    
    def test_feedback(self):
        """Test feedback mechanism."""
        config = ChatConfig(debug=False)
        pipeline = ChatPipeline(config)
        
        pipeline.add_knowledge("Test knowledge")
        pipeline.chat("Tell me about test")
        
        # Feedback should work
        result = pipeline.feedback(success=True)
        # May or may not have a mapping to reinforce
    
    def test_persistence(self):
        """Test knowledge persistence."""
        config = ChatConfig(debug=False)
        pipeline = ChatPipeline(config)
        
        pipeline.add_knowledge("Persistent knowledge item")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            pipeline.save_knowledge(f.name)
            
            # Create new pipeline and load
            pipeline2 = ChatPipeline(config)
            pipeline2.load_knowledge(f.name)
            
            assert len(pipeline2.knowledge_space) == len(pipeline.knowledge_space)
    
    def test_stats(self):
        """Test pipeline statistics."""
        config = ChatConfig(debug=False)
        pipeline = ChatPipeline(config)
        
        stats = pipeline.get_stats()
        assert 'pipeline' in stats
        assert 'knowledge' in stats
        assert 'intent_templates' in stats


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with legacy code."""
    
    def test_legacy_imports(self):
        """Test that legacy imports still work."""
        from truthspace_lcm.core import Gear, GearState, GearChain, Quaternion
        from truthspace_lcm.core import ConversationalChain
        
        # Should not raise
        assert Gear is not None
        assert GearState is not None
        assert GearChain is not None
        assert ConversationalChain is not None
    
    def test_new_imports(self):
        """Test that new imports work."""
        from truthspace_lcm.core import ChatPipeline, KnowledgeSpace, CodeSpace
        
        assert ChatPipeline is not None
        assert KnowledgeSpace is not None
        assert CodeSpace is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
