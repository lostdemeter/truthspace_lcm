"""
LCM Routing Test Suite

Tests for Phase 1 of the Goose Tool Calling Roadmap.
Measures pattern matching accuracy, execution success, and routing decisions.

Target metrics:
- 80%+ pattern match rate
- 90%+ execution success rate
- 80%+ modification accuracy
- 90%+ rejection accuracy

Run with: python -m pytest tests/test_lcm_routing.py -v
Or standalone: python tests/test_lcm_routing.py
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.code_orchestrator import CodeOrchestrator
from truthspace_lcm.core.holographic_pattern_space import HolographicPatternSpace
from truthspace_lcm.core.intent_classifier import IntentClassifier, Intent, IntentMatch


class ExpectedResult(Enum):
    CODE = "code"                    # Should generate executable code
    CODE_WITH_MODS = "code_with_mods"  # Should generate code with modifications applied
    REJECT = "reject"                # Should gracefully reject (out of scope)
    TOOL_CALL = "tool_call"          # Future: should trigger tool call
    KNOWLEDGE = "knowledge"          # Future: should answer question


@dataclass
class TestCase:
    query: str
    expected: ExpectedResult
    expected_pattern: str = ""       # Expected pattern name (if CODE)
    expected_mods: List[str] = None  # Expected modifications (if CODE_WITH_MODS)
    description: str = ""


# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = [
    # --- Basic Code Generation ---
    TestCase(
        query="create a sine wave plot",
        expected=ExpectedResult.CODE,
        expected_pattern="sine_wave",
        description="Basic sine wave"
    ),
    TestCase(
        query="make a cosine wave",
        expected=ExpectedResult.CODE,
        expected_pattern="cosine_wave",
        description="Basic cosine wave"
    ),
    TestCase(
        query="create a bar chart",
        expected=ExpectedResult.CODE,
        expected_pattern="bar_chart",
        description="Basic bar chart"
    ),
    TestCase(
        query="make a scatter plot",
        expected=ExpectedResult.CODE,
        expected_pattern="scatter_plot",
        description="Basic scatter plot"
    ),
    TestCase(
        query="create a histogram",
        expected=ExpectedResult.CODE,
        expected_pattern="histogram",
        description="Basic histogram"
    ),
    TestCase(
        query="make a line plot",
        expected=ExpectedResult.CODE,
        expected_pattern="line_plot",
        description="Basic line plot"
    ),
    
    # --- Code with Modifications ---
    TestCase(
        query="create a sine wave plot in red",
        expected=ExpectedResult.CODE_WITH_MODS,
        expected_pattern="sine_wave",
        expected_mods=["color:red"],
        description="Sine wave with color modification"
    ),
    TestCase(
        query="make a sine wave with amplitude of 2",
        expected=ExpectedResult.CODE_WITH_MODS,
        expected_pattern="sine_wave",
        expected_mods=["amplitude:2"],
        description="Sine wave with amplitude modification"
    ),
    TestCase(
        query="create a sine wave in cyan with amplitude 3",
        expected=ExpectedResult.CODE_WITH_MODS,
        expected_pattern="sine_wave",
        expected_mods=["color:cyan", "amplitude:3"],
        description="Sine wave with multiple modifications"
    ),
    TestCase(
        query="make a bar chart in green",
        expected=ExpectedResult.CODE_WITH_MODS,
        expected_pattern="bar_chart",
        expected_mods=["color:green"],
        description="Bar chart with color"
    ),
    TestCase(
        query="create a scatter plot with red dots",
        expected=ExpectedResult.CODE_WITH_MODS,
        expected_pattern="scatter_plot",
        expected_mods=["color:red"],
        description="Scatter plot with color"
    ),
    
    # --- Variations (should still match) ---
    TestCase(
        query="plot a sine function",
        expected=ExpectedResult.CODE,
        expected_pattern="sine_wave",
        description="Variation: 'sine function' instead of 'sine wave'"
    ),
    TestCase(
        query="show me a sin wave",
        expected=ExpectedResult.CODE,
        expected_pattern="sine_wave",
        description="Variation: 'sin' instead of 'sine'"
    ),
    TestCase(
        query="draw a bar graph",
        expected=ExpectedResult.CODE,
        expected_pattern="bar_chart",
        description="Variation: 'bar graph' instead of 'bar chart'"
    ),
    TestCase(
        query="create a distribution histogram",
        expected=ExpectedResult.CODE,
        expected_pattern="histogram",
        description="Variation: with 'distribution' keyword"
    ),
    
    # --- Should Reject Gracefully ---
    TestCase(
        query="write me a web server",
        expected=ExpectedResult.REJECT,
        description="Out of scope: web server"
    ),
    TestCase(
        query="create a machine learning model",
        expected=ExpectedResult.REJECT,
        description="Out of scope: ML model"
    ),
    TestCase(
        query="hello world",
        expected=ExpectedResult.REJECT,
        description="Out of scope: greeting"
    ),
    TestCase(
        query="asdfghjkl random gibberish",
        expected=ExpectedResult.REJECT,
        description="Out of scope: gibberish"
    ),
    
    # --- Future: Tool Calls ---
    TestCase(
        query="list files in current directory",
        expected=ExpectedResult.TOOL_CALL,
        description="Future: file listing tool"
    ),
    TestCase(
        query="read the contents of README.md",
        expected=ExpectedResult.TOOL_CALL,
        description="Future: file read tool"
    ),
    TestCase(
        query="run pytest",
        expected=ExpectedResult.TOOL_CALL,
        description="Future: bash tool"
    ),
    
    # --- Future: Knowledge Queries ---
    TestCase(
        query="what is a sine wave",
        expected=ExpectedResult.KNOWLEDGE,
        description="Future: knowledge query"
    ),
    TestCase(
        query="explain matplotlib",
        expected=ExpectedResult.KNOWLEDGE,
        description="Future: knowledge query"
    ),
]


# =============================================================================
# TEST RUNNER
# =============================================================================

class LCMTestRunner:
    """Runs the test suite and collects metrics."""
    
    def __init__(self):
        self.orchestrator = CodeOrchestrator(use_holographic=True)
        self.intent_classifier = IntentClassifier()
        self.results: List[Dict[str, Any]] = []
        
    def run_test(self, test: TestCase) -> Dict[str, Any]:
        """Run a single test case."""
        result = {
            "query": test.query,
            "expected": test.expected.value,
            "expected_pattern": test.expected_pattern,
            "description": test.description,
            "passed": False,
            "actual_pattern": None,
            "code_generated": False,
            "code_executes": False,
            "mods_applied": [],
            "error": None,
        }
        
        try:
            # Try to find a pattern match
            module, confidence, reason = self.orchestrator.pattern_space.find_best_match(test.query)
            
            if module and confidence >= 0.3:
                result["actual_pattern"] = module.name
                result["match_score"] = confidence
                result["match_reason"] = reason
                
                # Get the code template
                code = module.code_template if hasattr(module, 'code_template') else None
                
                if code:
                    result["code_generated"] = True
                    
                    # Apply template composition if needed
                    if test.expected == ExpectedResult.CODE_WITH_MODS:
                        composed = self.orchestrator.template_composer.compose(
                            test.query, result["actual_pattern"], code
                        )
                        code = composed.code
                        result["mods_applied"] = [
                            f"{m.target}:{m.value}" for m in composed.modifications_applied
                        ]
                    
                    # Test if code executes
                    result["code_executes"] = self._test_execution(code)
            
            # Determine if test passed
            if test.expected == ExpectedResult.CODE:
                result["passed"] = (
                    result["code_generated"] and 
                    result["code_executes"] and
                    (not test.expected_pattern or result["actual_pattern"] == test.expected_pattern)
                )
            elif test.expected == ExpectedResult.CODE_WITH_MODS:
                result["passed"] = (
                    result["code_generated"] and
                    result["code_executes"] and
                    len(result["mods_applied"]) > 0
                )
            elif test.expected == ExpectedResult.REJECT:
                # Should NOT find a match or should have low confidence
                result["passed"] = not result["code_generated"] or not result["code_executes"]
            elif test.expected == ExpectedResult.TOOL_CALL:
                # Use intent classifier for tool calls
                intent_match = self.intent_classifier.classify(test.query)
                result["intent"] = intent_match.intent.value
                result["intent_confidence"] = intent_match.confidence
                result["intent_reason"] = intent_match.reason
                result["tool_name"] = intent_match.tool_name
                result["tool_args"] = intent_match.tool_args
                result["passed"] = intent_match.intent == Intent.TOOL_CALL
                
            elif test.expected == ExpectedResult.KNOWLEDGE:
                # Use intent classifier for knowledge queries
                intent_match = self.intent_classifier.classify(test.query)
                result["intent"] = intent_match.intent.value
                result["intent_confidence"] = intent_match.confidence
                result["intent_reason"] = intent_match.reason
                result["passed"] = intent_match.intent == Intent.KNOWLEDGE
                
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _test_execution(self, code: str) -> bool:
        """Test if code executes without error."""
        if not code:
            return False
            
        # Write to temp file and try to execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                timeout=10,
                cwd=str(Path(__file__).parent.parent)
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            os.unlink(temp_path)
    
    def run_all(self) -> Dict[str, Any]:
        """Run all test cases and return metrics."""
        self.results = []
        
        for test in TEST_CASES:
            result = self.run_test(test)
            self.results.append(result)
            
            # Print progress
            status = "✓" if result["passed"] else ("⊘" if result["passed"] is None else "✗")
            print(f"{status} {test.query[:50]:<50} [{test.expected.value}]")
        
        return self.compute_metrics()
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        # Filter by category
        code_tests = [r for r in self.results if r["expected"] in ("code", "code_with_mods")]
        reject_tests = [r for r in self.results if r["expected"] == "reject"]
        tool_tests = [r for r in self.results if r["expected"] == "tool_call"]
        knowledge_tests = [r for r in self.results if r["expected"] == "knowledge"]
        intent_tests = tool_tests + knowledge_tests
        
        # Pattern match rate (for code tests)
        code_matched = sum(1 for r in code_tests if r["code_generated"])
        pattern_match_rate = code_matched / len(code_tests) if code_tests else 0
        
        # Execution success rate (for matched code)
        code_executed = sum(1 for r in code_tests if r["code_executes"])
        execution_rate = code_executed / code_matched if code_matched else 0
        
        # Modification accuracy (for code_with_mods tests)
        mod_tests = [r for r in self.results if r["expected"] == "code_with_mods"]
        mods_applied = sum(1 for r in mod_tests if len(r.get("mods_applied", [])) > 0)
        mod_accuracy = mods_applied / len(mod_tests) if mod_tests else 0
        
        # Rejection accuracy
        correctly_rejected = sum(1 for r in reject_tests if r["passed"])
        rejection_accuracy = correctly_rejected / len(reject_tests) if reject_tests else 0
        
        # Intent classification accuracy (Phase 2)
        intent_correct = sum(1 for r in intent_tests if r["passed"])
        intent_accuracy = intent_correct / len(intent_tests) if intent_tests else 0
        
        # Tool call accuracy
        tool_correct = sum(1 for r in tool_tests if r["passed"])
        tool_accuracy = tool_correct / len(tool_tests) if tool_tests else 0
        
        # Knowledge query accuracy
        knowledge_correct = sum(1 for r in knowledge_tests if r["passed"])
        knowledge_accuracy = knowledge_correct / len(knowledge_tests) if knowledge_tests else 0
        
        # Overall pass rate (all tests now active)
        all_tests = self.results
        overall_pass = sum(1 for r in all_tests if r["passed"])
        overall_rate = overall_pass / len(all_tests) if all_tests else 0
        
        metrics = {
            "total_tests": len(self.results),
            "code_tests": len(code_tests),
            "intent_tests": len(intent_tests),
            "pattern_match_rate": pattern_match_rate,
            "execution_success_rate": execution_rate,
            "modification_accuracy": mod_accuracy,
            "rejection_accuracy": rejection_accuracy,
            "intent_classification_accuracy": intent_accuracy,
            "tool_call_accuracy": tool_accuracy,
            "knowledge_accuracy": knowledge_accuracy,
            "overall_pass_rate": overall_rate,
            "targets": {
                "pattern_match_rate": 0.80,
                "execution_success_rate": 0.90,
                "modification_accuracy": 0.80,
                "rejection_accuracy": 0.90,
                "intent_classification_accuracy": 0.85,
            }
        }
        
        return metrics
    
    def print_report(self, metrics: Dict[str, Any]):
        """Print a formatted report."""
        print("\n" + "="*60)
        print("LCM ROUTING TEST REPORT")
        print("="*60)
        
        print(f"\nTests: {metrics['code_tests']} code, {metrics['intent_tests']} intent")
        
        print("\n--- Phase 1 Metrics (Code Generation) ---")
        for key in ["pattern_match_rate", "execution_success_rate", "modification_accuracy", "rejection_accuracy"]:
            actual = metrics[key]
            target = metrics["targets"][key]
            status = "✓" if actual >= target else "✗"
            print(f"{status} {key}: {actual*100:.1f}% (target: {target*100:.0f}%)")
        
        print("\n--- Phase 2 Metrics (Intent Classification) ---")
        intent_acc = metrics["intent_classification_accuracy"]
        intent_target = metrics["targets"]["intent_classification_accuracy"]
        status = "✓" if intent_acc >= intent_target else "✗"
        print(f"{status} intent_classification_accuracy: {intent_acc*100:.1f}% (target: {intent_target*100:.0f}%)")
        print(f"  - tool_call_accuracy: {metrics['tool_call_accuracy']*100:.1f}%")
        print(f"  - knowledge_accuracy: {metrics['knowledge_accuracy']*100:.1f}%")
        
        print(f"\n--- Overall ---")
        print(f"Pass rate: {metrics['overall_pass_rate']*100:.1f}%")
        
        # Check if ready for Phase 3
        phase1_ready = all(
            metrics[k] >= metrics["targets"][k] 
            for k in ["pattern_match_rate", "execution_success_rate", "modification_accuracy", "rejection_accuracy"]
        )
        phase2_ready = intent_acc >= intent_target
        
        if phase1_ready and phase2_ready:
            print(f"\nReady for Phase 3 (Tool Calling Implementation): YES ✓")
        elif phase1_ready:
            print(f"\nPhase 1 complete. Phase 2 in progress - improve intent classification.")
        else:
            print(f"\nPhase 1 incomplete - improve code generation first.")
        
        print("="*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the test suite."""
    print("LCM Routing Test Suite")
    print("Phase 1: Solidify the Core")
    print("-"*60)
    
    runner = LCMTestRunner()
    metrics = runner.run_all()
    runner.print_report(metrics)
    
    return metrics


if __name__ == "__main__":
    main()
