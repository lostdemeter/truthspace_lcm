"""
Gear Improvement Loop - Self-Improving Gear System

A meta-gear system that:
1. Identifies deficiencies in gear outputs
2. Creates gears to fix them
3. Tests and verifies the fixes
4. Iterates until quality threshold is met

This enables autonomous improvement of the gear chain.

Author: Lesley Gushurst
License: GPLv3
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from enum import Enum
import re
import json
from pathlib import Path

from truthspace_lcm.core.gear import Gear, GearState
from truthspace_lcm.core.utils.folding_deficiency import (
    FoldingDeficiencyDetector, FoldingStructure,
    ShapeDeficiency, ShapeDeficiencyType
)


class DeficiencyType(Enum):
    """Types of deficiencies that can be detected."""
    MISSING_CONTENT = "missing_content"      # Expected info not present
    WRONG_FORMAT = "wrong_format"            # Format doesn't match expectation
    TOO_VAGUE = "too_vague"                  # Response lacks specificity
    TOO_VERBOSE = "too_verbose"              # Response is too long
    IRRELEVANT = "irrelevant"                # Response doesn't address query
    FACTUAL_ERROR = "factual_error"          # Contains incorrect information
    INCOMPLETE = "incomplete"                # Partial answer only


@dataclass
class Deficiency:
    """A detected deficiency in gear output."""
    type: DeficiencyType
    description: str
    severity: float  # 0.0 to 1.0
    suggested_fix: str
    evidence: str = ""  # What in the output shows this deficiency


@dataclass
class TestCase:
    """A test case for evaluating gear quality."""
    input: str
    expected_contains: List[str] = field(default_factory=list)  # Must contain these
    expected_not_contains: List[str] = field(default_factory=list)  # Must not contain
    expected_format: Optional[str] = None  # Regex pattern for format
    quality_criteria: Dict[str, Any] = field(default_factory=dict)  # Custom criteria


@dataclass
class TestResult:
    """Result of running a test case."""
    test_case: TestCase
    output: str
    passed: bool
    score: float  # 0.0 to 1.0
    deficiencies: List[Deficiency] = field(default_factory=list)
    execution_time: float = 0.0


class DeficiencyDetectorGear:
    """
    Detects deficiencies in gear outputs by comparing against expectations.
    
    Uses pattern matching, semantic similarity, and heuristics to identify:
    - Missing expected content
    - Format mismatches
    - Vagueness/verbosity issues
    - Relevance problems
    
    Enhanced with semantic matching for better content detection.
    """
    
    def __init__(self):
        self.name = "DeficiencyDetector"
        
        # Vagueness indicators
        self.vague_phrases = [
            "is known to", "is associated with", "related to",
            "something about", "might be", "could be", "perhaps",
            "I don't have information", "I'm not sure"
        ]
        
        # Quality indicators (positive)
        self.quality_phrases = [
            "specifically", "characterized by", "described as",
            "notable for", "distinguished by", "known for"
        ]
        
        # Semantic similarity cache for word relationships
        self._synonym_cache: Dict[str, Set[str]] = {}
        self._init_basic_synonyms()
    
    def _init_basic_synonyms(self):
        """Initialize basic synonym groups for semantic matching."""
        synonym_groups = [
            {'captain', 'commander', 'leader', 'chief', 'master'},
            {'ship', 'vessel', 'boat', 'craft'},
            {'whale', 'leviathan', 'beast', 'creature'},
            {'hunt', 'chase', 'pursue', 'seek', 'track'},
            {'ocean', 'sea', 'water', 'deep'},
            {'crew', 'sailors', 'men', 'hands'},
            {'describe', 'explain', 'tell', 'show', 'depict'},
            {'appearance', 'look', 'features', 'characteristics'},
        ]
        for group in synonym_groups:
            for word in group:
                self._synonym_cache[word.lower()] = group
    
    def _semantic_match(self, expected: str, output: str) -> float:
        """
        Check if expected content is semantically present in output.
        
        Returns a score from 0.0 (not present) to 1.0 (exact match).
        """
        expected_lower = expected.lower()
        output_lower = output.lower()
        
        # Exact match
        if expected_lower in output_lower:
            return 1.0
        
        # Word-level match
        expected_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', expected_lower))
        output_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', output_lower))
        
        if not expected_words:
            return 0.0
        
        # Direct word overlap
        direct_overlap = len(expected_words & output_words)
        
        # Semantic overlap (synonyms)
        semantic_overlap = 0
        for exp_word in expected_words:
            if exp_word in output_words:
                continue  # Already counted
            synonyms = self._synonym_cache.get(exp_word, set())
            if synonyms & output_words:
                semantic_overlap += 0.8  # Partial credit for synonym
        
        total_match = (direct_overlap + semantic_overlap) / len(expected_words)
        return min(1.0, total_match)
    
    def detect(self, output: str, test_case: TestCase) -> List[Deficiency]:
        """Detect deficiencies in output against test case expectations."""
        deficiencies = []
        
        # Check for missing expected content (with semantic matching)
        for expected in test_case.expected_contains:
            match_score = self._semantic_match(expected, output)
            
            if match_score < 0.5:  # Less than 50% semantic match
                severity = 0.8 * (1.0 - match_score)  # Higher severity for lower match
                deficiencies.append(Deficiency(
                    type=DeficiencyType.MISSING_CONTENT,
                    description=f"Missing expected content: '{expected}'",
                    severity=severity,
                    suggested_fix=f"Add extraction for '{expected}' type content",
                    evidence=f"Semantic match score: {match_score:.2f}"
                ))
            elif match_score < 0.8:  # Partial match - lower severity
                deficiencies.append(Deficiency(
                    type=DeficiencyType.INCOMPLETE,
                    description=f"Partial match for '{expected}' (score: {match_score:.2f})",
                    severity=0.4,
                    suggested_fix=f"Enhance content about '{expected}'",
                    evidence=f"Semantic match score: {match_score:.2f}"
                ))
        
        # Check for unwanted content
        for unwanted in test_case.expected_not_contains:
            if unwanted.lower() in output.lower():
                deficiencies.append(Deficiency(
                    type=DeficiencyType.IRRELEVANT,
                    description=f"Contains unwanted content: '{unwanted}'",
                    severity=0.5,
                    suggested_fix=f"Filter out '{unwanted}' type content",
                    evidence=f"Output contains '{unwanted}'"
                ))
        
        # Check format if specified
        if test_case.expected_format:
            if not re.search(test_case.expected_format, output):
                deficiencies.append(Deficiency(
                    type=DeficiencyType.WRONG_FORMAT,
                    description=f"Format doesn't match expected pattern",
                    severity=0.6,
                    suggested_fix="Add format transformation gear",
                    evidence=f"Expected pattern: {test_case.expected_format}"
                ))
        
        # Check for vagueness
        vague_count = sum(1 for phrase in self.vague_phrases if phrase.lower() in output.lower())
        if vague_count >= 2:
            deficiencies.append(Deficiency(
                type=DeficiencyType.TOO_VAGUE,
                description=f"Response is too vague ({vague_count} vague phrases)",
                severity=0.7,
                suggested_fix="Add specificity extraction gear",
                evidence=f"Contains vague phrases like 'is known to', 'associated with'"
            ))
        
        # Check for verbosity (if output is very long relative to input)
        if len(output) > len(test_case.input) * 20:
            deficiencies.append(Deficiency(
                type=DeficiencyType.TOO_VERBOSE,
                description="Response is excessively long",
                severity=0.4,
                suggested_fix="Add summarization gear",
                evidence=f"Output length: {len(output)} chars"
            ))
        
        # Check for "I don't know" type responses
        if "don't have information" in output.lower() or "i'm not sure" in output.lower():
            deficiencies.append(Deficiency(
                type=DeficiencyType.INCOMPLETE,
                description="Response indicates lack of knowledge",
                severity=0.9,
                suggested_fix="Improve knowledge extraction or add fallback gear",
                evidence="Contains 'don't have information' or similar"
            ))
        
        return deficiencies
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gear interface for deficiency detection."""
        output = input_data.get('output', '')
        test_case = input_data.get('test_case')
        
        if not test_case:
            # Create minimal test case from input
            test_case = TestCase(
                input=input_data.get('input', ''),
                expected_contains=input_data.get('expected_contains', [])
            )
        
        deficiencies = self.detect(output, test_case)
        
        return {
            'deficiencies': deficiencies,
            'count': len(deficiencies),
            'max_severity': max((d.severity for d in deficiencies), default=0.0),
            'suggestions': [d.suggested_fix for d in deficiencies]
        }


class GearChainBuilder:
    """
    Dynamically composes gears into chains.
    
    Allows runtime construction of gear pipelines based on
    detected deficiencies and available fix gears.
    """
    
    def __init__(self):
        self.name = "GearChainBuilder"
        self.available_gears: Dict[str, Gear] = {}
        self.chains: Dict[str, List[str]] = {}  # Named chains
    
    def register_gear(self, name: str, gear: Gear):
        """Register a gear for use in chains."""
        self.available_gears[name] = gear
    
    def create_chain(self, name: str, gear_names: List[str]) -> bool:
        """Create a named chain from gear names."""
        # Verify all gears exist
        for gear_name in gear_names:
            if gear_name not in self.available_gears:
                return False
        
        self.chains[name] = gear_names
        return True
    
    def run_chain(self, chain_name: str, input_data: Any) -> Any:
        """Run a named chain on input data."""
        if chain_name not in self.chains:
            raise ValueError(f"Chain '{chain_name}' not found")
        
        result = input_data
        for gear_name in self.chains[chain_name]:
            gear = self.available_gears[gear_name]
            
            # Wrap result for gear processing if needed
            if not isinstance(result, dict):
                result = {'input': result, 'data': result}
            
            result = gear.process(result)
        
        return result
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gear interface for chain building."""
        action = input_data.get('action', 'run')
        
        if action == 'register':
            name = input_data['name']
            gear = input_data['gear']
            self.register_gear(name, gear)
            return {'success': True, 'registered': name}
        
        elif action == 'create':
            name = input_data['name']
            gears = input_data['gears']
            success = self.create_chain(name, gears)
            return {'success': success, 'chain': name}
        
        elif action == 'run':
            chain = input_data['chain']
            data = input_data['data']
            result = self.run_chain(chain, data)
            return {'result': result}
        
        elif action == 'list':
            return {
                'gears': list(self.available_gears.keys()),
                'chains': self.chains
            }
        
        return {'error': f"Unknown action: {action}"}


class GearTestHarness:
    """
    Test harness for evaluating gear quality.
    
    Runs test cases, measures quality, and reports results.
    """
    
    def __init__(self):
        self.name = "GearTestHarness"
        self.deficiency_detector = DeficiencyDetectorGear()
        self.test_history: List[TestResult] = []
    
    def run_test(self, gear: Gear, test_case: TestCase) -> TestResult:
        """Run a single test case against a gear."""
        import time
        
        start = time.time()
        
        # Run the gear
        try:
            if hasattr(gear, 'chat'):
                # ConversationalChain-like
                output = gear.chat(test_case.input)
            elif hasattr(gear, 'process'):
                result = gear.process({'input': test_case.input})
                output = result.get('output', str(result))
            else:
                output = str(gear(test_case.input))
        except Exception as e:
            output = f"Error: {str(e)}"
        
        execution_time = time.time() - start
        
        # Detect deficiencies
        deficiencies = self.deficiency_detector.detect(output, test_case)
        
        # Calculate score
        score = self._calculate_score(output, test_case, deficiencies)
        
        # Determine pass/fail
        passed = score >= 0.7 and not any(d.severity >= 0.8 for d in deficiencies)
        
        result = TestResult(
            test_case=test_case,
            output=output,
            passed=passed,
            score=score,
            deficiencies=deficiencies,
            execution_time=execution_time
        )
        
        self.test_history.append(result)
        return result
    
    def _calculate_score(self, output: str, test_case: TestCase, 
                         deficiencies: List[Deficiency]) -> float:
        """Calculate quality score for output."""
        score = 1.0
        
        # Deduct for deficiencies
        for d in deficiencies:
            score -= d.severity * 0.2
        
        # Bonus for expected content
        found = sum(1 for exp in test_case.expected_contains 
                    if exp.lower() in output.lower())
        if test_case.expected_contains:
            score += 0.2 * (found / len(test_case.expected_contains))
        
        return max(0.0, min(1.0, score))
    
    def run_suite(self, gear: Gear, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Run a suite of test cases."""
        results = [self.run_test(gear, tc) for tc in test_cases]
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_score = sum(r.score for r in results) / total if total > 0 else 0
        
        # Aggregate deficiencies
        all_deficiencies = []
        for r in results:
            all_deficiencies.extend(r.deficiencies)
        
        # Group by type
        by_type = {}
        for d in all_deficiencies:
            if d.type not in by_type:
                by_type[d.type] = []
            by_type[d.type].append(d)
        
        return {
            'passed': passed,
            'total': total,
            'pass_rate': passed / total if total > 0 else 0,
            'avg_score': avg_score,
            'results': results,
            'deficiencies_by_type': by_type,
            'most_common_deficiency': max(by_type.keys(), key=lambda k: len(by_type[k])) if by_type else None
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gear interface for test harness."""
        gear = input_data.get('gear')
        test_cases = input_data.get('test_cases', [])
        
        if not gear:
            return {'error': 'No gear provided'}
        
        if isinstance(test_cases, list) and len(test_cases) > 0:
            return self.run_suite(gear, test_cases)
        elif 'test_case' in input_data:
            result = self.run_test(gear, input_data['test_case'])
            return {
                'passed': result.passed,
                'score': result.score,
                'output': result.output,
                'deficiencies': result.deficiencies
            }
        
        return {'error': 'No test cases provided'}


class ShapeBasedTestHarness:
    """
    Test harness using shape-based (folding) deficiency detection.
    
    This is the geometric replacement for GearTestHarness.
    Uses fold patterns instead of pattern matching.
    """
    
    def __init__(self):
        self.name = "ShapeBasedTestHarness"
        self.deficiency_detector = FoldingDeficiencyDetector()
        self.test_history: List[TestResult] = []
    
    def run_test(self, gear: Gear, test_case: TestCase) -> TestResult:
        """Run a single test case against a gear using shape-based detection."""
        import time
        
        start = time.time()
        
        # Run the gear
        try:
            if hasattr(gear, 'chat'):
                output = gear.chat(test_case.input)
            elif hasattr(gear, 'process'):
                result = gear.process({'input': test_case.input})
                output = result.get('output', str(result))
            else:
                output = str(gear(test_case.input))
        except Exception as e:
            output = f"Error: {str(e)}"
        
        execution_time = time.time() - start
        
        # Detect deficiency using shape-based method
        # Use the input as the expected structure template
        shape_deficiency = self.deficiency_detector.detect(test_case.input, output)
        
        # Convert ShapeDeficiency to legacy Deficiency format for compatibility
        deficiencies = self._convert_shape_deficiency(shape_deficiency)
        
        # Calculate score based on shape similarity
        score = self._calculate_score(shape_deficiency)
        
        # Determine pass/fail
        passed = score >= 0.7 and shape_deficiency.severity < 0.5
        
        result = TestResult(
            test_case=test_case,
            output=output,
            passed=passed,
            score=score,
            deficiencies=deficiencies,
            execution_time=execution_time
        )
        
        self.test_history.append(result)
        return result
    
    def _convert_shape_deficiency(self, shape_def: ShapeDeficiency) -> List[Deficiency]:
        """Convert ShapeDeficiency to legacy Deficiency format."""
        if shape_def.type == ShapeDeficiencyType.NONE:
            return []
        
        # Map shape types to legacy types
        type_map = {
            ShapeDeficiencyType.INCOMPLETE: DeficiencyType.INCOMPLETE,
            ShapeDeficiencyType.MISSING_STRUCTURE: DeficiencyType.MISSING_CONTENT,
            ShapeDeficiencyType.WRONG_STRUCTURE: DeficiencyType.WRONG_FORMAT,
            ShapeDeficiencyType.PARTIAL: DeficiencyType.INCOMPLETE,
        }
        
        legacy_type = type_map.get(shape_def.type, DeficiencyType.INCOMPLETE)
        
        return [Deficiency(
            type=legacy_type,
            description=shape_def.description,
            severity=shape_def.severity,
            suggested_fix=shape_def.suggested_fix,
            evidence=f"Shape similarity: {shape_def.shape_similarity:.3f}, Fold ratio: {shape_def.fold_ratio:.2f}"
        )]
    
    def _calculate_score(self, shape_def: ShapeDeficiency) -> float:
        """Calculate quality score from shape deficiency."""
        # Score is primarily based on shape similarity
        base_score = shape_def.shape_similarity
        
        # Adjust for fold ratio
        if shape_def.fold_ratio < 0.5:
            base_score *= 0.8
        
        # Penalize for severity
        base_score -= shape_def.severity * 0.3
        
        return max(0.0, min(1.0, base_score))
    
    def run_suite(self, gear: Gear, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Run a suite of test cases."""
        results = [self.run_test(gear, tc) for tc in test_cases]
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_score = sum(r.score for r in results) / total if total > 0 else 0
        
        all_deficiencies = []
        for r in results:
            all_deficiencies.extend(r.deficiencies)
        
        by_type = {}
        for d in all_deficiencies:
            if d.type not in by_type:
                by_type[d.type] = []
            by_type[d.type].append(d)
        
        return {
            'passed': passed,
            'total': total,
            'pass_rate': passed / total if total > 0 else 0,
            'avg_score': avg_score,
            'results': results,
            'deficiencies_by_type': by_type,
            'most_common_deficiency': max(by_type.keys(), key=lambda k: len(by_type[k])) if by_type else None
        }


class GearImprovementLoop:
    """
    The main improvement loop that orchestrates:
    1. Testing a gear
    2. Detecting deficiencies
    3. Creating fix gears
    4. Composing improved chains
    5. Re-testing until quality threshold met
    
    This is the self-improving meta-gear.
    """
    
    def __init__(self, use_shape_based: bool = True):
        self.name = "GearImprovementLoop"
        
        # Use shape-based (folding) detection by default
        self.use_shape_based = use_shape_based
        if use_shape_based:
            self.test_harness = ShapeBasedTestHarness()
            self.shape_detector = FoldingDeficiencyDetector()
        else:
            self.test_harness = GearTestHarness()
        
        self.chain_builder = GearChainBuilder()
        self.deficiency_detector = DeficiencyDetectorGear()  # Keep for backward compatibility
        
        # Fix gear templates (patterns for creating fix gears)
        self.fix_templates: Dict[DeficiencyType, Callable] = {}
        self._register_default_fixes()
        
        # Improvement history
        self.iterations: List[Dict[str, Any]] = []
        self.max_iterations = 5
        self.quality_threshold = 0.8
        
        # === NEW: Emergent fix pattern learning ===
        # Tracks which fixes worked for which deficiency patterns
        self.successful_fixes: Dict[str, List[Dict[str, Any]]] = {}
        # Maps deficiency signature → fix that worked
        self.fix_memory: Dict[str, str] = {}
        # Tracks improvement deltas for each fix type
        self.fix_effectiveness: Dict[str, List[float]] = {}
    
    def _register_default_fixes(self):
        """Register default fix gear templates."""
        
        # Fix for vagueness - extract more specific content
        def create_specificity_gear():
            class SpecificityGear:
                def __init__(self):
                    self.name = "SpecificityExtractor"
                    self.specific_patterns = [
                        r"described as ([^.]+)",
                        r"characterized by ([^.]+)",
                        r"known for ([^.]+)",
                        r"notable for ([^.]+)",
                        r"has ([^.]+) appearance",
                        r"with ([^.]+) features",
                    ]
                
                def process(self, input_data):
                    text = input_data.get('output', input_data.get('input', ''))
                    
                    # Extract specific descriptions
                    specifics = []
                    for pattern in self.specific_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        specifics.extend(matches)
                    
                    if specifics:
                        return {'output': f"Specifically: {'; '.join(specifics)}"}
                    return {'output': text}
            
            return SpecificityGear()
        
        self.fix_templates[DeficiencyType.TOO_VAGUE] = create_specificity_gear
        
        # Fix for verbosity - summarize
        def create_summary_gear():
            class SummaryGear:
                def __init__(self):
                    self.name = "Summarizer"
                
                def process(self, input_data):
                    text = input_data.get('output', input_data.get('input', ''))
                    
                    # Simple extractive summary - take first sentence of each paragraph
                    paragraphs = text.split('\n\n')
                    summary_parts = []
                    for p in paragraphs[:3]:  # Max 3 paragraphs
                        sentences = p.split('. ')
                        if sentences:
                            summary_parts.append(sentences[0])
                    
                    return {'output': '. '.join(summary_parts)}
            
            return SummaryGear()
        
        self.fix_templates[DeficiencyType.TOO_VERBOSE] = create_summary_gear
        
        # Fix for missing content - try to extract from context
        def create_content_extractor():
            class ContentExtractorGear:
                def __init__(self):
                    self.name = "ContentExtractor"
                    self.description_patterns = [
                        r"([A-Z][a-z]+) is (?:a |an )?([^.]+)",
                        r"([A-Z][a-z]+), (?:a |an |the )?([^,]+)",
                        r"the ([a-z]+) ([A-Z][a-z]+)",
                    ]
                
                def process(self, input_data):
                    text = input_data.get('output', input_data.get('input', ''))
                    query = input_data.get('query', '')
                    
                    # Try to extract descriptive content
                    descriptions = []
                    for pattern in self.description_patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            if isinstance(match, tuple):
                                descriptions.append(' '.join(match))
                            else:
                                descriptions.append(match)
                    
                    if descriptions:
                        return {'output': f"Description: {'; '.join(descriptions[:3])}"}
                    return {'output': text}
            
            return ContentExtractorGear()
        
        self.fix_templates[DeficiencyType.MISSING_CONTENT] = create_content_extractor
    
    def register_fix(self, deficiency_type: DeficiencyType, 
                     gear_factory: Callable):
        """Register a fix gear factory for a deficiency type."""
        self.fix_templates[deficiency_type] = gear_factory
    
    def _get_deficiency_signature(self, deficiency: Deficiency) -> str:
        """Create a signature for a deficiency to enable pattern matching."""
        # Signature = type + key words from description
        key_words = re.findall(r'\b[a-zA-Z]{4,}\b', deficiency.description.lower())
        return f"{deficiency.type.value}:{':'.join(sorted(set(key_words[:3])))}"
    
    def _record_fix_result(self, deficiency: Deficiency, fix_name: str, 
                           score_before: float, score_after: float):
        """Record whether a fix was effective for learning."""
        sig = self._get_deficiency_signature(deficiency)
        delta = score_after - score_before
        
        # Track effectiveness
        if fix_name not in self.fix_effectiveness:
            self.fix_effectiveness[fix_name] = []
        self.fix_effectiveness[fix_name].append(delta)
        
        # If fix improved score, remember it
        if delta > 0.05:  # Meaningful improvement
            self.fix_memory[sig] = fix_name
            if sig not in self.successful_fixes:
                self.successful_fixes[sig] = []
            self.successful_fixes[sig].append({
                'fix': fix_name,
                'delta': delta,
                'deficiency': deficiency.description
            })
    
    def _get_best_fix_for_deficiency(self, deficiency: Deficiency) -> Optional[str]:
        """Check if we've seen this deficiency pattern before and know a good fix."""
        sig = self._get_deficiency_signature(deficiency)
        
        # Exact match
        if sig in self.fix_memory:
            return self.fix_memory[sig]
        
        # Partial match - check if any recorded signature shares the type
        for recorded_sig, fix_name in self.fix_memory.items():
            if recorded_sig.startswith(deficiency.type.value):
                # Check effectiveness of this fix
                if fix_name in self.fix_effectiveness:
                    avg_delta = sum(self.fix_effectiveness[fix_name]) / len(self.fix_effectiveness[fix_name])
                    if avg_delta > 0.05:
                        return fix_name
        
        return None
    
    def get_fix_stats(self) -> Dict[str, Any]:
        """Get statistics about fix effectiveness."""
        stats = {}
        for fix_name, deltas in self.fix_effectiveness.items():
            stats[fix_name] = {
                'uses': len(deltas),
                'avg_improvement': sum(deltas) / len(deltas) if deltas else 0,
                'max_improvement': max(deltas) if deltas else 0,
                'success_rate': sum(1 for d in deltas if d > 0.05) / len(deltas) if deltas else 0
            }
        return stats
    
    def save_learned_fixes(self, path: str):
        """Save learned fix patterns to JSON for persistence."""
        data = {
            'fix_memory': self.fix_memory,
            'fix_effectiveness': self.fix_effectiveness,
            'successful_fixes': {
                sig: [
                    {'fix': f['fix'], 'delta': f['delta'], 'deficiency': f['deficiency']}
                    for f in fixes
                ]
                for sig, fixes in self.successful_fixes.items()
            }
        }
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_learned_fixes(self, path: str) -> bool:
        """Load previously learned fix patterns."""
        filepath = Path(path)
        if not filepath.exists():
            return False
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            self.fix_memory = data.get('fix_memory', {})
            self.fix_effectiveness = data.get('fix_effectiveness', {})
            self.successful_fixes = data.get('successful_fixes', {})
            return True
        except Exception:
            return False
    
    def configure_llm(self, llm_url: str, llm_model: str):
        """Configure LLM for smart fix gear creation."""
        self.llm_url = llm_url
        self.llm_model = llm_model
    
    def create_smart_fix(self, deficiency: Deficiency, 
                         original_input: str, original_output: str,
                         source_gear) -> Optional[object]:
        """
        Create a smart fix gear using LLM to understand the deficiency
        and generate appropriate transformation logic.
        
        This is the key to self-improvement - using LLM as teacher
        to create fix gears that work without LLM at runtime.
        """
        if not hasattr(self, 'llm_url'):
            return None
        
        # For MISSING_CONTENT, we need to query the source more specifically
        if deficiency.type == DeficiencyType.MISSING_CONTENT:
            # Create a gear that re-queries with more specific prompts
            class SpecificQueryGear:
                def __init__(self, source, missing_content):
                    self.name = f"SpecificQuery:{missing_content}"
                    self.source = source
                    self.missing = missing_content
                
                def process(self, input_data):
                    original_input = input_data.get('input', '')
                    original_output = input_data.get('output', '')
                    
                    # Try more specific queries
                    specific_queries = [
                        f"What is the {self.missing} of {original_input.split()[-1]}?",
                        f"Describe the {self.missing} in the text",
                        f"Tell me about {self.missing}",
                    ]
                    
                    additional_info = []
                    for query in specific_queries:
                        if hasattr(self.source, 'chat'):
                            result = self.source.chat(query)
                            if self.missing.lower() in result.lower():
                                additional_info.append(result)
                                break
                    
                    if additional_info:
                        return {'output': f"{original_output}\n\nAdditional: {additional_info[0]}"}
                    return {'output': original_output}
                
                def chat(self, message):
                    result = self.process({'input': message})
                    return result.get('output', str(result))
            
            # Extract what's missing from the deficiency
            missing = deficiency.description.replace("Missing expected content: '", "").rstrip("'")
            return SpecificQueryGear(source_gear, missing)
        
        elif deficiency.type == DeficiencyType.TOO_VAGUE:
            # Create a gear that asks for more specific details
            class DetailEnhancerGear:
                def __init__(self, source):
                    self.name = "DetailEnhancer"
                    self.source = source
                
                def process(self, input_data):
                    original_input = input_data.get('input', '')
                    original_output = input_data.get('output', '')
                    
                    # Extract the subject and ask for specific details
                    words = original_input.split()
                    subject = words[-1] if words else ""
                    
                    detail_queries = [
                        f"What does {subject} look like?",
                        f"What are {subject}'s characteristics?",
                        f"Describe {subject}'s appearance",
                    ]
                    
                    details = []
                    for query in detail_queries:
                        if hasattr(self.source, 'chat'):
                            result = self.source.chat(query)
                            if "don't have" not in result.lower():
                                details.append(result)
                                break
                    
                    if details:
                        return {'output': f"{original_output}\n\nDetails: {details[0]}"}
                    return {'output': original_output}
                
                def chat(self, message):
                    result = self.process({'input': message})
                    return result.get('output', str(result))
            
            return DetailEnhancerGear(source_gear)
        
        elif deficiency.type == DeficiencyType.WRONG_FORMAT:
            # Create a gear that reformats the output
            class FormatFixerGear:
                def __init__(self):
                    self.name = "FormatFixer"
                
                def process(self, input_data):
                    output = input_data.get('output', '')
                    
                    # Clean up common format issues
                    # Remove excessive whitespace
                    output = re.sub(r'\n{3,}', '\n\n', output)
                    # Remove leading/trailing whitespace from lines
                    lines = [line.strip() for line in output.split('\n')]
                    output = '\n'.join(lines)
                    # Ensure proper sentence endings
                    output = re.sub(r'([a-z])([A-Z])', r'\1. \2', output)
                    
                    return {'output': output}
                
                def chat(self, message):
                    result = self.process({'input': message})
                    return result.get('output', str(result))
            
            return FormatFixerGear()
        
        elif deficiency.type == DeficiencyType.IRRELEVANT:
            # Create a gear that filters to relevant content
            class RelevanceFilterGear:
                def __init__(self, original_query):
                    self.name = "RelevanceFilter"
                    self.query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', original_query.lower()))
                
                def process(self, input_data):
                    output = input_data.get('output', '')
                    query = input_data.get('input', '')
                    
                    # Update query words if available
                    if query:
                        self.query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
                    
                    # Score sentences by relevance
                    sentences = re.split(r'[.!?]+', output)
                    scored = []
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower()))
                        overlap = len(words & self.query_words)
                        scored.append((overlap, sent))
                    
                    # Keep sentences with some relevance
                    relevant = [s for score, s in sorted(scored, reverse=True) if score > 0]
                    
                    if relevant:
                        return {'output': '. '.join(relevant[:5]) + '.'}
                    return {'output': output}
                
                def chat(self, message):
                    result = self.process({'input': message})
                    return result.get('output', str(result))
            
            return RelevanceFilterGear(original_input)
        
        elif deficiency.type == DeficiencyType.INCOMPLETE:
            # Create a gear that tries to complete the response
            class CompletionGear:
                def __init__(self, source):
                    self.name = "CompletionEnhancer"
                    self.source = source
                
                def process(self, input_data):
                    original_input = input_data.get('input', '')
                    original_output = input_data.get('output', '')
                    
                    # If response indicates lack of knowledge, try rephrasing
                    if "don't have" in original_output.lower() or "not sure" in original_output.lower():
                        # Try alternative phrasings
                        words = original_input.split()
                        subject = words[-1] if words else ""
                        
                        alt_queries = [
                            f"What can you tell me about {subject}?",
                            f"Describe {subject}",
                            f"Information about {subject}",
                        ]
                        
                        for query in alt_queries:
                            if hasattr(self.source, 'chat'):
                                result = self.source.chat(query)
                                if "don't have" not in result.lower() and len(result) > 20:
                                    return {'output': result}
                    
                    return {'output': original_output}
                
                def chat(self, message):
                    result = self.process({'input': message})
                    return result.get('output', str(result))
            
            return CompletionGear(source_gear)
        
        return None
    
    def improve(self, gear: Gear, test_cases: List[TestCase], 
                verbose: bool = False) -> Tuple[Gear, Dict[str, Any]]:
        """
        Run the improvement loop on a gear.
        
        Returns the improved gear (or chain) and improvement report.
        """
        current_gear = gear
        self.iterations = []
        
        for i in range(self.max_iterations):
            if verbose:
                print(f"\n=== Iteration {i+1} ===")
            
            # Test current gear
            suite_result = self.test_harness.run_suite(current_gear, test_cases)
            
            iteration_data = {
                'iteration': i + 1,
                'pass_rate': suite_result['pass_rate'],
                'avg_score': suite_result['avg_score'],
                'deficiencies': suite_result['deficiencies_by_type']
            }
            
            if verbose:
                print(f"Pass rate: {suite_result['pass_rate']:.1%}")
                print(f"Avg score: {suite_result['avg_score']:.2f}")
            
            # Check if we've reached quality threshold
            if suite_result['avg_score'] >= self.quality_threshold:
                iteration_data['status'] = 'threshold_met'
                self.iterations.append(iteration_data)
                if verbose:
                    print("Quality threshold met!")
                break
            
            # Find most impactful deficiency to fix
            if not suite_result['deficiencies_by_type']:
                iteration_data['status'] = 'no_deficiencies'
                self.iterations.append(iteration_data)
                break
            
            # Get the most common/severe deficiency type
            worst_type = suite_result['most_common_deficiency']
            worst_deficiencies = suite_result['deficiencies_by_type'].get(worst_type, [])
            
            if verbose:
                print(f"Most common deficiency: {worst_type.value}")
            
            # === NEW: Check learned fixes first ===
            fix_gear = None
            learned_fix_name = None
            if worst_deficiencies:
                learned_fix_name = self._get_best_fix_for_deficiency(worst_deficiencies[0])
                if learned_fix_name and verbose:
                    print(f"Found learned fix: {learned_fix_name}")
            
            # Try smart fix if no learned fix (uses source gear for re-querying)
            if not fix_gear and worst_deficiencies:
                # Get a sample test result to understand the context
                sample_result = suite_result['results'][0] if suite_result['results'] else None
                if sample_result:
                    fix_gear = self.create_smart_fix(
                        worst_deficiencies[0],
                        sample_result.test_case.input,
                        sample_result.output,
                        gear  # Pass original gear for re-querying
                    )
                    if fix_gear and verbose:
                        print(f"Created smart fix gear: {fix_gear.name}")
            
            # Fall back to template if no smart fix
            if not fix_gear and worst_type in self.fix_templates:
                fix_gear = self.fix_templates[worst_type]()
                if verbose:
                    print(f"Created template fix gear: {fix_gear.name}")
            
            if fix_gear:
                # Create a simple wrapper that chains base gear output through fix gear
                # Avoid using chain_builder to prevent recursion issues
                class SimpleChainedGear:
                    def __init__(self, base_gear, fix_gear, iteration):
                        self.name = f"Chain:iter{iteration}"
                        self.base = base_gear
                        self.fix = fix_gear
                    
                    def process(self, input_data):
                        # Run base gear first
                        if hasattr(self.base, 'chat'):
                            base_output = self.base.chat(input_data.get('input', ''))
                        elif hasattr(self.base, 'process'):
                            base_result = self.base.process(input_data)
                            base_output = base_result.get('output', str(base_result))
                        else:
                            base_output = str(input_data)
                        
                        # Run fix gear on base output
                        fix_input = {'input': input_data.get('input', ''), 'output': base_output}
                        if hasattr(self.fix, 'process'):
                            fix_result = self.fix.process(fix_input)
                            return fix_result
                        return {'output': base_output}
                    
                    def chat(self, message):
                        result = self.process({'input': message})
                        if isinstance(result, dict):
                            return result.get('output', str(result))
                        return str(result)
                
                current_gear = SimpleChainedGear(current_gear, fix_gear, i + 1)
                iteration_data['fix_applied'] = fix_gear.name
                iteration_data['status'] = 'fix_applied'
                
                # === NEW: Record fix result for learning ===
                # We'll record the result after next iteration's test
                iteration_data['_pending_fix'] = {
                    'deficiency': worst_deficiencies[0] if worst_deficiencies else None,
                    'fix_name': fix_gear.name,
                    'score_before': suite_result['avg_score']
                }
            else:
                iteration_data['status'] = 'no_fix_available'
                if verbose:
                    print(f"No fix available for {worst_type.value}")
            
            self.iterations.append(iteration_data)
            
            # === NEW: Record previous fix effectiveness ===
            if len(self.iterations) >= 2:
                prev = self.iterations[-2]
                if '_pending_fix' in prev and prev['_pending_fix']['deficiency']:
                    self._record_fix_result(
                        prev['_pending_fix']['deficiency'],
                        prev['_pending_fix']['fix_name'],
                        prev['_pending_fix']['score_before'],
                        iteration_data['avg_score']
                    )
        
        # Final report
        report = {
            'iterations': len(self.iterations),
            'initial_score': self.iterations[0]['avg_score'] if self.iterations else 0,
            'final_score': self.iterations[-1]['avg_score'] if self.iterations else 0,
            'improvement': (self.iterations[-1]['avg_score'] - self.iterations[0]['avg_score']) if self.iterations else 0,
            'fixes_applied': [it.get('fix_applied') for it in self.iterations if it.get('fix_applied')],
            'history': self.iterations
        }
        
        return current_gear, report
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gear interface for improvement loop."""
        gear = input_data.get('gear')
        test_cases = input_data.get('test_cases', [])
        verbose = input_data.get('verbose', False)
        
        if not gear:
            return {'error': 'No gear provided'}
        
        improved_gear, report = self.improve(gear, test_cases, verbose)
        
        return {
            'improved_gear': improved_gear,
            'report': report
        }


# Convenience function for quick testing
def quick_test(gear, input_text: str, expected_contains: List[str] = None,
               verbose: bool = True) -> TestResult:
    """Quick test a gear with a single input."""
    harness = GearTestHarness()
    test_case = TestCase(
        input=input_text,
        expected_contains=expected_contains or []
    )
    result = harness.run_test(gear, test_case)
    
    if verbose:
        print(f"Input: {input_text}")
        print(f"Output: {result.output[:200]}...")
        print(f"Score: {result.score:.2f}")
        print(f"Passed: {result.passed}")
        if result.deficiencies:
            print("Deficiencies:")
            for d in result.deficiencies:
                print(f"  - {d.type.value}: {d.description}")
    
    return result
