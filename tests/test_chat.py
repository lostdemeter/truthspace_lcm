"""
Tests for TruthSpace LCM Chat - Geometric Chat System

Tests the interactive chat functionality:
- Query processing
- Style commands
- Q&A resolution
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.chat import GeometricChat


class TestGeometricChat:
    """Test GeometricChat."""
    
    def test_initialization(self):
        """Test chat initialization."""
        chat = GeometricChat()
        
        assert len(chat.kb.qa_pairs) > 0, "Should have bootstrap Q&A pairs"
        assert len(chat.style_engine.styles) > 0, "Should have bootstrap styles"
    
    def test_query_known_question(self):
        """Test querying a known question."""
        chat = GeometricChat()
        
        answer, confidence, debug = chat.query("What is TruthSpace?")
        
        assert confidence > 0.5, f"Should have high confidence, got {confidence}"
        assert "geometric" in answer.lower() or "space" in answer.lower()
    
    def test_query_similar_question(self):
        """Test querying a similar but not exact question."""
        chat = GeometricChat()
        
        answer, confidence, debug = chat.query("Tell me about TruthSpace")
        
        assert confidence > 0.3, f"Should match similar question, got {confidence}"
    
    def test_query_unknown_question(self):
        """Test querying an unknown question."""
        chat = GeometricChat()
        
        answer, confidence, debug = chat.query("What is the capital of Mars?")
        
        assert confidence < 0.5, f"Should have low confidence for unknown, got {confidence}"
    
    def test_style_analysis(self):
        """Test style analysis."""
        chat = GeometricChat()
        
        results = chat.analyze_style("The methodology demonstrates significant improvements.")
        
        assert len(results) > 0
        assert results[0][0] in ['formal', 'casual', 'technical']
    
    def test_command_help(self):
        """Test /help command."""
        chat = GeometricChat()
        
        response = chat.process("/help")
        
        assert "Commands" in response or "help" in response.lower()
    
    def test_command_style_list(self):
        """Test /style list command."""
        chat = GeometricChat()
        
        response = chat.process("/style list")
        
        assert "formal" in response or "casual" in response
    
    def test_command_style_set(self):
        """Test /style <name> command."""
        chat = GeometricChat()
        
        response = chat.process("/style formal")
        
        assert "formal" in response.lower()
        assert chat.current_style == "formal"
    
    def test_command_style_off(self):
        """Test /style off command."""
        chat = GeometricChat()
        chat.current_style = "formal"
        
        response = chat.process("/style off")
        
        assert chat.current_style is None
    
    def test_command_debug(self):
        """Test /debug command."""
        chat = GeometricChat()
        
        response = chat.process("/debug on")
        assert chat.debug_mode == True
        
        response = chat.process("/debug off")
        assert chat.debug_mode == False
    
    def test_command_stats(self):
        """Test /stats command."""
        chat = GeometricChat()
        
        response = chat.process("/stats")
        
        assert "Vocab" in response or "Q&A" in response
    
    def test_command_analyze(self):
        """Test /analyze command."""
        chat = GeometricChat()
        
        response = chat.process("/analyze This is a formal methodology.")
        
        assert "formal" in response.lower() or "Style" in response
    
    def test_command_quit(self):
        """Test /quit command."""
        chat = GeometricChat()
        
        response = chat.process("/quit")
        
        assert response == "QUIT"
    
    def test_process_question(self):
        """Test processing a regular question."""
        chat = GeometricChat()
        
        response = chat.process("What is cosine similarity?")
        
        assert len(response) > 0
        assert "angle" in response.lower() or "vector" in response.lower() or "similarity" in response.lower()
    
    def test_debug_mode_output(self):
        """Test debug mode adds debug info."""
        chat = GeometricChat()
        chat.debug_mode = True
        
        response = chat.process("What is TruthSpace?")
        
        assert "[DEBUG" in response or "sim=" in response


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TruthSpace LCM Chat Test Suite")
    print("=" * 60)
    print()
    
    test_class = TestGeometricChat
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
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
