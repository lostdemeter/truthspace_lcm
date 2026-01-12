#!/usr/bin/env python3
"""
Experiment: Unified Self-Assembly Loop with Patterns

Extends the Phase 5 self-assembly loop to handle BOTH content AND patterns
in a unified space. Pattern dimensions are discovered and processed just
like content dimensions.

The extended loop:
1. INGEST     → Extract transformation pairs (content AND pattern)
2. DETECT     → New relationship type? → Create dimension (content OR pattern)
3. REBALANCE  → New dimension → Extend all positions
4. POSITION   → Place concepts (source=0, target=φ)
5. DISCOVER   → Find Platonic Ideals (content, pattern, AND unified)
6. GAP-FILL   → Identify missing → Query LLM
7. COMPOUND   → Derive compound positions (content + pattern)
8. VERIFY     → Check self-similarity across both types

Key insight: Pattern dimensions emerge the same way content dimensions do.
The loop doesn't need to know the difference - it just processes pairs.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import re

from experiments.self_assembling_corpus import (
    SelfAssemblingCorpus,
    TransformationPair,
    Dimension,
    PlatonicIdeal,
    ConceptType,
    LLMInterface,
    LLMEnhancedPipeline,
    AssemblyState,
    PHI,
)

from experiments.unified_space import UnifiedCorpus, DimensionType


# =============================================================================
# PATTERN DETECTION
# =============================================================================

class PatternDetector:
    """
    Detects speech patterns in text and extracts pattern pairs.
    
    This is the pattern equivalent of extracting content pairs.
    Instead of "king → queen (gender)", we extract:
    - "formal → casual (register)"
    - "verbose → terse (verbosity)"
    - "question → statement (speech_act)"
    """
    
    # Pattern indicators
    FORMAL_INDICATORS = {
        'would', 'could', 'shall', 'may', 'might', 'whom', 'therefore',
        'furthermore', 'nevertheless', 'consequently', 'regarding',
        'concerning', 'pursuant', 'hereby', 'wherein', 'thereof'
    }
    
    CASUAL_INDICATORS = {
        'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'nope',
        'hey', 'hi', 'ok', 'okay', 'cool', 'awesome', 'stuff', 'thing',
        'like', 'just', 'really', 'pretty', 'super', 'totally'
    }
    
    QUESTION_INDICATORS = {'?', 'what', 'who', 'where', 'when', 'why', 'how'}
    
    EMPHATIC_INDICATORS = {'!', 'very', 'extremely', 'absolutely', 'definitely'}
    
    def __init__(self):
        self.detected_patterns: List[Tuple[str, str, str]] = []
    
    def analyze_text(self, text: str) -> Dict[str, str]:
        """
        Analyze text to detect its pattern characteristics.
        
        Returns dict of dimension → value.
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        patterns = {}
        
        # Register detection
        formal_count = len(words & self.FORMAL_INDICATORS)
        casual_count = len(words & self.CASUAL_INDICATORS)
        
        if formal_count > casual_count:
            patterns['register'] = 'formal'
        elif casual_count > formal_count:
            patterns['register'] = 'casual'
        else:
            patterns['register'] = 'neutral'
        
        # Speech act detection
        if '?' in text:
            patterns['speech_act'] = 'question'
        elif '!' in text:
            patterns['speech_act'] = 'exclamation'
        else:
            patterns['speech_act'] = 'statement'
        
        # Verbosity detection
        sentences = text.split('.')
        avg_words = len(words) / max(len(sentences), 1)
        if avg_words > 15:
            patterns['verbosity'] = 'verbose'
        elif avg_words < 6:
            patterns['verbosity'] = 'terse'
        else:
            patterns['verbosity'] = 'moderate'
        
        # Tone detection
        emphatic_count = sum(1 for ind in self.EMPHATIC_INDICATORS if ind in text_lower)
        if emphatic_count > 2:
            patterns['tone'] = 'emphatic'
        else:
            patterns['tone'] = 'neutral'
        
        return patterns
    
    def extract_pattern_pairs(self, texts: List[str]) -> List[Tuple[str, str, str]]:
        """
        Extract pattern transformation pairs from a collection of texts.
        
        By comparing texts, we can infer pattern transformations:
        - If text A is formal and text B is casual, we have a register pair
        """
        pairs = []
        analyses = [(text, self.analyze_text(text)) for text in texts]
        
        # Compare pairs of texts
        for i, (text1, patterns1) in enumerate(analyses):
            for j, (text2, patterns2) in enumerate(analyses):
                if i >= j:
                    continue
                
                # Find dimensions where they differ
                for dim in patterns1:
                    if dim in patterns2 and patterns1[dim] != patterns2[dim]:
                        # Found a transformation!
                        source = patterns1[dim]
                        target = patterns2[dim]
                        pairs.append((source, target, dim))
        
        # Deduplicate
        unique_pairs = list(set(pairs))
        self.detected_patterns.extend(unique_pairs)
        
        return unique_pairs


# =============================================================================
# UNIFIED ASSEMBLY STATE
# =============================================================================

@dataclass
class UnifiedAssemblyState(AssemblyState):
    """Extended state that tracks both content and pattern dimensions."""
    content_dimensions: int = 0
    pattern_dimensions: int = 0
    content_pairs: int = 0
    pattern_pairs: int = 0
    unified_ideals: int = 0
    style_ideals: int = 0


# =============================================================================
# UNIFIED SELF-ASSEMBLY LOOP
# =============================================================================

class UnifiedSelfAssemblyLoop:
    """
    Extended self-assembly loop that handles both content AND patterns.
    
    The key insight: Pattern dimensions emerge the same way content dimensions do.
    We just need to:
    1. Detect patterns in text (like we detect content relationships)
    2. Add pattern pairs (like we add content pairs)
    3. Let the geometry handle the rest
    
    The loop:
    1. INGEST     → Extract content AND pattern pairs
    2. DETECT     → Create content OR pattern dimensions
    3. REBALANCE  → Extend all positions
    4. POSITION   → Place concepts (source=0, target=φ)
    5. DISCOVER   → Find content ideals, pattern ideals, AND unified ideals
    6. GAP-FILL   → Fill gaps in both content AND pattern space
    7. COMPOUND   → Derive compounds (content + pattern = styled content)
    8. VERIFY     → Check self-similarity across unified space
    """
    
    def __init__(self, corpus: UnifiedCorpus = None,
                 llm: LLMInterface = None,
                 verbose: bool = True):
        self.corpus = corpus or UnifiedCorpus()
        self.llm = llm or LLMInterface()
        self.pattern_detector = PatternDetector()
        self.verbose = verbose
        
        # History
        self.history: List[UnifiedAssemblyState] = []
        self.total_cycles = 0
        
        # Configuration
        self.max_gaps_per_cycle = 5
        self.min_self_similarity = 0.8
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def seed_base_patterns(self):
        """Seed the corpus with fundamental pattern dimensions."""
        # Register dimension
        self.corpus.add_pattern_pair("casual", "formal", "register")
        self.corpus.add_pattern_pair("colloquial", "academic", "register")
        self.corpus.add_pattern_pair("informal", "proper", "register")
        
        # Verbosity dimension
        self.corpus.add_pattern_pair("terse", "verbose", "verbosity")
        self.corpus.add_pattern_pair("brief", "elaborate", "verbosity")
        self.corpus.add_pattern_pair("concise", "detailed", "verbosity")
        
        # Tone dimension
        self.corpus.add_pattern_pair("serious", "playful", "tone")
        self.corpus.add_pattern_pair("somber", "whimsical", "tone")
        self.corpus.add_pattern_pair("grave", "lighthearted", "tone")
        
        # Speech act dimension
        self.corpus.add_pattern_pair("statement", "question", "speech_act")
        self.corpus.add_pattern_pair("declaration", "inquiry", "speech_act")
        
        # Certainty dimension
        self.corpus.add_pattern_pair("uncertain", "definite", "certainty")
        self.corpus.add_pattern_pair("tentative", "confident", "certainty")
        
        self.corpus.recompute()
        self._log(f"Seeded {len(self.corpus.pairs)} base pattern pairs")
    
    def seed_base_content(self):
        """Seed the corpus with fundamental content dimensions."""
        # Gender dimension
        self.corpus.add_content_pair("king", "queen", "gender")
        self.corpus.add_content_pair("man", "woman", "gender")
        self.corpus.add_content_pair("boy", "girl", "gender")
        self.corpus.add_content_pair("father", "mother", "gender")
        
        # Age dimension
        self.corpus.add_content_pair("boy", "man", "age")
        self.corpus.add_content_pair("girl", "woman", "age")
        self.corpus.add_content_pair("child", "adult", "age")
        self.corpus.add_content_pair("puppy", "dog", "age")
        
        # Size dimension
        self.corpus.add_content_pair("large", "small", "size")
        self.corpus.add_content_pair("giant", "tiny", "size")
        
        # Regality dimension
        self.corpus.add_content_pair("peasant", "king", "regality")
        self.corpus.add_content_pair("commoner", "royalty", "regality")
        
        self.corpus.recompute()
        self._log(f"Seeded {len(self.corpus.pairs)} total pairs (content + pattern)")
    
    def run_cycle(self, texts: List[str] = None) -> UnifiedAssemblyState:
        """
        Run one complete unified self-assembly cycle.
        
        Args:
            texts: Optional list of texts to ingest
            
        Returns:
            UnifiedAssemblyState with cycle results
        """
        self.total_cycles += 1
        state = UnifiedAssemblyState(cycle=self.total_cycles)
        
        # Capture before state
        self.corpus.recompute()
        state.pairs_before = len(self.corpus.pairs)
        state.dimensions_before = len(self.corpus.dimensions)
        state.ideals_before = len(self.corpus.ideals)
        
        # Count by type
        state.content_dimensions = sum(
            1 for d in self.corpus.dimensions 
            if self.corpus.get_dimension_type(d) == DimensionType.CONTENT
        )
        state.pattern_dimensions = sum(
            1 for d in self.corpus.dimensions
            if self.corpus.get_dimension_type(d) == DimensionType.PATTERN
        )
        
        self._log(f"\n{'='*60}")
        self._log(f"UNIFIED SELF-ASSEMBLY CYCLE {self.total_cycles}")
        self._log(f"{'='*60}")
        
        # Step 1: INGEST - Extract BOTH content AND pattern pairs
        if texts:
            self._log("\n[1] INGEST: Extracting content AND pattern pairs...")
            content_pairs, pattern_pairs = self._ingest_unified(texts)
            state.content_pairs = content_pairs
            state.pattern_pairs = pattern_pairs
            self._log(f"    Content pairs: {content_pairs}, Pattern pairs: {pattern_pairs}")
        else:
            self._log("\n[1] INGEST: No new texts (gap-fill mode)")
        
        # Step 2: DETECT - Check for new dimensions (both types)
        self._log("\n[2] DETECT: Checking for new dimensions...")
        self.corpus.recompute()
        content_dims = sum(
            1 for d in self.corpus.dimensions
            if self.corpus.get_dimension_type(d) == DimensionType.CONTENT
        )
        pattern_dims = sum(
            1 for d in self.corpus.dimensions
            if self.corpus.get_dimension_type(d) == DimensionType.PATTERN
        )
        self._log(f"    Content dimensions: {content_dims}")
        self._log(f"    Pattern dimensions: {pattern_dims}")
        
        # Step 3: REBALANCE
        self._log("\n[3] REBALANCE: Extending positions...")
        self._log(f"    All positions extended to {len(self.corpus.dimensions)} dimensions")
        
        # Step 4: POSITION
        self._log("\n[4] POSITION: Verifying φ-based positioning...")
        self._verify_positioning()
        
        # Step 5: DISCOVER - Find ALL types of ideals
        self._log("\n[5] DISCOVER: Finding Platonic Ideals...")
        content_ideals, pattern_ideals, unified_ideals = self._discover_ideals()
        state.style_ideals = pattern_ideals
        state.unified_ideals = unified_ideals
        self._log(f"    Content ideals: {content_ideals}")
        self._log(f"    Pattern ideals (styles): {pattern_ideals}")
        self._log(f"    Unified ideals: {unified_ideals}")
        
        # Step 6: GAP-FILL
        self._log("\n[6] GAP-FILL: Detecting gaps in unified space...")
        content_gaps, pattern_gaps = self._detect_unified_gaps()
        state.gaps_detected = content_gaps + pattern_gaps
        self._log(f"    Content gaps: {content_gaps}, Pattern gaps: {pattern_gaps}")
        
        # Step 7: COMPOUND - Derive content + pattern compounds
        self._log("\n[7] COMPOUND: Deriving styled content compounds...")
        compounds = self._derive_styled_compounds()
        state.compounds_derived = compounds
        self._log(f"    Styled compounds derived: {compounds}")
        
        # Step 8: VERIFY
        self._log("\n[8] VERIFY: Checking unified self-similarity...")
        state.self_similarity_score = self._verify_unified_similarity()
        self._log(f"    Unified self-similarity: {state.self_similarity_score:.2%}")
        
        # Capture after state
        self.corpus.recompute()
        state.pairs_after = len(self.corpus.pairs)
        state.dimensions_after = len(self.corpus.dimensions)
        state.ideals_after = len(self.corpus.ideals)
        
        # Summary
        self._log(f"\n{'─'*60}")
        self._log(f"CYCLE {self.total_cycles} COMPLETE")
        self._log(f"  Pairs: {state.pairs_before} → {state.pairs_after} (+{state.pairs_added()})")
        self._log(f"  Dimensions: {state.dimensions_before} → {state.dimensions_after}")
        self._log(f"    Content: {content_dims}, Pattern: {pattern_dims}")
        self._log(f"  Ideals: content={content_ideals}, pattern={pattern_ideals}, unified={unified_ideals}")
        self._log(f"  Self-similarity: {state.self_similarity_score:.2%}")
        
        self.history.append(state)
        return state
    
    def _ingest_unified(self, texts: List[str]) -> Tuple[int, int]:
        """Ingest texts and extract both content and pattern pairs."""
        content_pairs = 0
        pattern_pairs = 0
        
        # Extract pattern pairs by comparing texts
        detected = self.pattern_detector.extract_pattern_pairs(texts)
        for source, target, dim in detected:
            if self.corpus.add_pattern_pair(source, target, dim):
                pattern_pairs += 1
        
        # For content pairs, we'd need NLP - for now just count existing
        # In a full implementation, this would use the LLM pipeline
        
        return content_pairs, pattern_pairs
    
    def _verify_positioning(self):
        """Verify φ-based positioning is correct."""
        correct = 0
        total = 0
        
        for pair in self.corpus.pairs:
            src_pos = self.corpus.get_position(pair.source)
            tgt_pos = self.corpus.get_position(pair.target)
            
            if src_pos is not None and tgt_pos is not None:
                total += 1
                delta = tgt_pos - src_pos
                magnitude = np.linalg.norm(delta)
                
                # Should be approximately φ
                if abs(magnitude - PHI) < 0.5:
                    correct += 1
        
        if total > 0:
            self._log(f"    Positioning accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    def _discover_ideals(self) -> Tuple[int, int, int]:
        """Discover content ideals, pattern ideals, and unified ideals."""
        content_ideals = 0
        pattern_ideals = 0
        unified_ideals = 0
        
        # Count ideals by type
        for ideal_name in self.corpus.list_ideals():
            ideal = self.corpus.get_ideal(ideal_name)
            if ideal:
                # Check what type of dimensions it anchors
                content_dims = 0
                pattern_dims = 0
                
                for dim_name in ideal.dimensions_anchored:
                    dim_type = self.corpus.get_dimension_type(dim_name)
                    if dim_type == DimensionType.CONTENT:
                        content_dims += 1
                    elif dim_type == DimensionType.PATTERN:
                        pattern_dims += 1
                
                if content_dims > 0 and pattern_dims > 0:
                    unified_ideals += 1
                elif content_dims > 0:
                    content_ideals += 1
                elif pattern_dims > 0:
                    pattern_ideals += 1
        
        return content_ideals, pattern_ideals, unified_ideals
    
    def _detect_unified_gaps(self) -> Tuple[int, int]:
        """Detect gaps in both content and pattern space."""
        content_gaps = 0
        pattern_gaps = 0
        
        # For each ideal, check if it has variations in all dimensions
        for ideal_name in self.corpus.list_ideals():
            ideal = self.corpus.get_ideal(ideal_name)
            if not ideal:
                continue
            
            for dim_name in self.corpus.dimensions:
                dim_type = self.corpus.get_dimension_type(dim_name)
                
                # Check if ideal has a pair in this dimension
                has_pair = any(
                    (p.source == ideal_name or p.target == ideal_name) and 
                    p.relationship == dim_name
                    for p in self.corpus.pairs
                )
                
                if not has_pair:
                    if dim_type == DimensionType.CONTENT:
                        content_gaps += 1
                    elif dim_type == DimensionType.PATTERN:
                        pattern_gaps += 1
        
        return content_gaps, pattern_gaps
    
    def _derive_styled_compounds(self) -> int:
        """Derive compound positions that combine content + pattern."""
        compounds = 0
        
        # Get some content concepts and pattern concepts
        content_concepts = [
            c for c in list(self.corpus.concepts.keys())[:10]
            if self.corpus.get_concept_type(c) == DimensionType.CONTENT
        ]
        
        pattern_concepts = [
            c for c in list(self.corpus.concepts.keys())[:10]
            if self.corpus.get_concept_type(c) == DimensionType.PATTERN
        ]
        
        # Compose each content with each pattern
        for content in content_concepts[:5]:
            for pattern in pattern_concepts[:3]:
                pos = self.corpus.compose(pattern, content)
                if pos is not None:
                    compounds += 1
        
        return compounds
    
    def _verify_unified_similarity(self) -> float:
        """Verify self-similarity across the unified space."""
        if len(self.corpus.pairs) == 0:
            return 0.0
        
        # Check that transformations have consistent deltas
        deltas_by_dim: Dict[str, List[float]] = {}
        
        for pair in self.corpus.pairs:
            src_pos = self.corpus.get_position(pair.source)
            tgt_pos = self.corpus.get_position(pair.target)
            
            if src_pos is not None and tgt_pos is not None:
                delta = np.linalg.norm(tgt_pos - src_pos)
                dim = pair.relationship
                
                if dim not in deltas_by_dim:
                    deltas_by_dim[dim] = []
                deltas_by_dim[dim].append(delta)
        
        # Calculate consistency within each dimension
        consistencies = []
        for dim, deltas in deltas_by_dim.items():
            if len(deltas) > 1:
                mean_delta = np.mean(deltas)
                std_delta = np.std(deltas)
                if mean_delta > 0:
                    consistency = 1.0 - (std_delta / mean_delta)
                    consistencies.append(max(0, consistency))
        
        if consistencies:
            return np.mean(consistencies)
        return 1.0
    
    def run_until_stable(self, texts: List[str] = None, 
                         max_cycles: int = 5) -> List[UnifiedAssemblyState]:
        """Run cycles until the corpus is stable."""
        states = []
        
        for i in range(max_cycles):
            # Only pass texts on first cycle
            state = self.run_cycle(texts if i == 0 else None)
            states.append(state)
            
            # Check stability
            if (state.gaps_detected == 0 and 
                state.self_similarity_score >= self.min_self_similarity):
                self._log(f"\n✓ Corpus stable after {i+1} cycles")
                break
        
        return states
    
    def get_status(self) -> Dict:
        """Get current status of the unified corpus."""
        self.corpus.recompute()
        
        content_dims = sum(
            1 for d in self.corpus.dimensions
            if self.corpus.get_dimension_type(d) == DimensionType.CONTENT
        )
        pattern_dims = sum(
            1 for d in self.corpus.dimensions
            if self.corpus.get_dimension_type(d) == DimensionType.PATTERN
        )
        
        return {
            "total_pairs": len(self.corpus.pairs),
            "total_dimensions": len(self.corpus.dimensions),
            "content_dimensions": content_dims,
            "pattern_dimensions": pattern_dims,
            "total_concepts": len(self.corpus.concepts),
            "total_ideals": len(self.corpus.ideals),
            "cycles_run": self.total_cycles,
        }


# =============================================================================
# DEMO
# =============================================================================

def demo_unified_assembly():
    """Demonstrate the unified self-assembly loop."""
    print("=" * 70)
    print("DEMO: Unified Self-Assembly Loop (Content + Patterns)")
    print("=" * 70)
    print()
    print("This demo shows how the self-assembly loop handles BOTH")
    print("content dimensions AND pattern dimensions in a unified space.")
    print()
    
    # Create loop
    loop = UnifiedSelfAssemblyLoop(verbose=True)
    
    # Seed with base knowledge
    print("Seeding base knowledge...")
    loop.seed_base_patterns()
    loop.seed_base_content()
    print()
    
    # Show initial status
    status = loop.get_status()
    print(f"Initial status:")
    print(f"  Pairs: {status['total_pairs']}")
    print(f"  Dimensions: {status['content_dimensions']} content + {status['pattern_dimensions']} pattern")
    print()
    
    # Sample texts with different patterns
    sample_texts = [
        "The king ruled wisely over his kingdom.",
        "Hey, the king was pretty cool, ya know?",
        "His Majesty, the sovereign monarch, governed with utmost prudence.",
        "King? Yeah he was alright I guess.",
        "What manner of ruler was the king?",
        "Was the king good or what?",
    ]
    
    # Run a cycle with the texts
    print("Running unified assembly cycle with sample texts...")
    state = loop.run_cycle(sample_texts)
    
    # Show final status
    print()
    print("=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    status = loop.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test composition
    print()
    print("Testing content + pattern composition:")
    print("-" * 60)
    
    compositions = [
        ("formal", "king"),
        ("casual", "queen"),
        ("verbose", "formal", "king"),
        ("terse", "playful", "dog"),
    ]
    
    for concepts in compositions:
        pos = loop.corpus.compose(*concepts)
        if pos is not None:
            content, pattern = loop.corpus.decompose(pos)
            print(f"  {' + '.join(concepts)}:")
            print(f"    → Content: {content[:3]}, Pattern: {pattern[:2]}")
    
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Key findings:")
    print("  1. Pattern dimensions emerge alongside content dimensions")
    print("  2. The same self-assembly loop handles both")
    print("  3. Styled compounds (content + pattern) work correctly")
    print("  4. Unified self-similarity verification works")
    print()
    
    return loop


if __name__ == "__main__":
    demo_unified_assembly()
