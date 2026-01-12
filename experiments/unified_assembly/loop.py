#!/usr/bin/env python3
"""
Unified Self-Assembly Loop

This module provides the main self-assembly loop that handles ALL dimension
types (content, pattern, stylization) at ALL scales (character through document).

The loop:
1. INGEST     → Extract pairs at all scales
2. DETECT     → Create dimensions (any type, any scale)
3. REBALANCE  → Extend all positions
4. POSITION   → Place concepts (source=0, target=φ)
5. DISCOVER   → Find ideals (content, pattern, stylization, cross-scale)
6. GAP-FILL   → Fill gaps at all scales
7. COMPOUND   → Derive multi-scale compounds
8. VERIFY     → Check self-similarity across unified space

Key insight: The loop doesn't need to know what type or scale a dimension is.
It just processes pairs and lets structure emerge.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import re

from experiments.unified_assembly.core import (
    UnifiedCorpus,
    Scale,
    DimensionType,
    ScaledDimension,
    SCALE_DIMENSIONS,
    PHI,
)

from experiments.unified_assembly.stylization import (
    StylizationManager,
    detect_stylization,
    STYLIZATIONS,
)

from experiments.unified_assembly.scales import (
    ScaleDetector,
    FractalDimensionSpace,
)


# =============================================================================
# ASSEMBLY STATE
# =============================================================================

@dataclass
class UnifiedAssemblyState:
    """
    State of the unified self-assembly loop.
    
    Tracks what happened in each cycle across all dimension types and scales.
    """
    cycle: int = 0
    
    # Pair counts
    pairs_before: int = 0
    pairs_after: int = 0
    
    # Dimension counts by type
    content_dimensions: int = 0
    pattern_dimensions: int = 0
    stylization_dimensions: int = 0
    
    # Dimension counts by scale
    dimensions_by_scale: Dict[str, int] = field(default_factory=dict)
    
    # Ideal counts
    content_ideals: int = 0
    pattern_ideals: int = 0
    stylization_ideals: int = 0
    cross_scale_ideals: int = 0
    
    # Gap counts
    gaps_detected: int = 0
    gaps_filled: int = 0
    
    # Compounds
    compounds_derived: int = 0
    
    # Quality
    self_similarity_score: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def pairs_added(self) -> int:
        return self.pairs_after - self.pairs_before
    
    def total_dimensions(self) -> int:
        return self.content_dimensions + self.pattern_dimensions + self.stylization_dimensions


# =============================================================================
# PATTERN DETECTOR
# =============================================================================

class PatternDetector:
    """
    Detects speech patterns in text and extracts pattern pairs.
    """
    
    FORMAL_INDICATORS = {
        'would', 'could', 'shall', 'may', 'might', 'whom', 'therefore',
        'furthermore', 'nevertheless', 'consequently', 'regarding',
    }
    
    CASUAL_INDICATORS = {
        'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'nope',
        'hey', 'hi', 'ok', 'okay', 'cool', 'awesome', 'stuff',
    }
    
    def analyze_text(self, text: str) -> Dict[str, str]:
        """Analyze text to detect pattern characteristics."""
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
        
        # Verbosity
        sentences = text.split('.')
        avg_words = len(words) / max(len(sentences), 1)
        if avg_words > 15:
            patterns['verbosity'] = 'verbose'
        elif avg_words < 6:
            patterns['verbosity'] = 'terse'
        else:
            patterns['verbosity'] = 'moderate'
        
        return patterns
    
    def extract_pattern_pairs(self, texts: List[str]) -> List[Tuple[str, str, str]]:
        """Extract pattern transformation pairs from texts."""
        pairs = []
        analyses = [(text, self.analyze_text(text)) for text in texts]
        
        for i, (text1, patterns1) in enumerate(analyses):
            for j, (text2, patterns2) in enumerate(analyses):
                if i >= j:
                    continue
                
                for dim in patterns1:
                    if dim in patterns2 and patterns1[dim] != patterns2[dim]:
                        pairs.append((patterns1[dim], patterns2[dim], dim))
        
        return list(set(pairs))


# =============================================================================
# UNIFIED SELF-ASSEMBLY LOOP
# =============================================================================

class UnifiedSelfAssemblyLoop:
    """
    The unified self-assembly loop that handles ALL dimension types at ALL scales.
    
    This is the culmination of the Universal Dimension Principle:
    ANY transformation at ANY scale can be a dimension.
    
    The loop:
    1. INGEST     → Extract pairs (content, pattern, stylization) at all scales
    2. DETECT     → Create dimensions (auto-detect type and scale)
    3. REBALANCE  → Extend all positions for new dimensions
    4. POSITION   → Place concepts (source=0, target=φ)
    5. DISCOVER   → Find ideals (content, pattern, stylization, cross-scale)
    6. GAP-FILL   → Fill gaps at all scales
    7. COMPOUND   → Derive multi-scale compounds
    8. VERIFY     → Check self-similarity across unified space
    """
    
    def __init__(self, corpus: UnifiedCorpus = None, verbose: bool = True):
        self.corpus = corpus or UnifiedCorpus()
        self.verbose = verbose
        
        # Detectors
        self.pattern_detector = PatternDetector()
        self.stylization_manager = StylizationManager()
        self.scale_detector = ScaleDetector()
        
        # History
        self.history: List[UnifiedAssemblyState] = []
        self.total_cycles = 0
        
        # Configuration
        self.max_gaps_per_cycle = 5
        self.min_self_similarity = 0.8
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    # -------------------------------------------------------------------------
    # Seeding
    # -------------------------------------------------------------------------
    
    def seed_content(self):
        """Seed with fundamental content dimensions."""
        # Word-level content
        self.corpus.add_content_pair("king", "queen", "gender", Scale.WORD)
        self.corpus.add_content_pair("man", "woman", "gender", Scale.WORD)
        self.corpus.add_content_pair("boy", "girl", "gender", Scale.WORD)
        self.corpus.add_content_pair("father", "mother", "gender", Scale.WORD)
        
        self.corpus.add_content_pair("boy", "man", "age", Scale.WORD)
        self.corpus.add_content_pair("girl", "woman", "age", Scale.WORD)
        self.corpus.add_content_pair("child", "adult", "age", Scale.WORD)
        
        self.corpus.add_content_pair("small", "large", "size", Scale.WORD)
        self.corpus.add_content_pair("tiny", "huge", "size", Scale.WORD)
        
        self.corpus.add_content_pair("peasant", "king", "regality", Scale.WORD)
        
        self._log(f"  Seeded content dimensions")
    
    def seed_patterns(self):
        """Seed with fundamental pattern dimensions."""
        # Sentence-level patterns
        self.corpus.add_pattern_pair("casual", "formal", "register", Scale.SENTENCE)
        self.corpus.add_pattern_pair("colloquial", "academic", "register", Scale.SENTENCE)
        
        self.corpus.add_pattern_pair("terse", "verbose", "verbosity", Scale.SENTENCE)
        self.corpus.add_pattern_pair("brief", "elaborate", "verbosity", Scale.SENTENCE)
        
        self.corpus.add_pattern_pair("serious", "playful", "tone", Scale.SENTENCE)
        self.corpus.add_pattern_pair("somber", "whimsical", "tone", Scale.SENTENCE)
        
        self.corpus.add_pattern_pair("statement", "question", "speech_act", Scale.SENTENCE)
        
        self.corpus.add_pattern_pair("uncertain", "definite", "certainty", Scale.SENTENCE)
        
        # Document-level patterns
        self.corpus.add_pattern_pair("blog", "paper", "genre", Scale.DOCUMENT)
        self.corpus.add_pattern_pair("novice", "expert", "audience", Scale.DOCUMENT)
        
        self._log(f"  Seeded pattern dimensions")
    
    def seed_stylizations(self):
        """Seed with fundamental stylization dimensions."""
        # Character-level stylizations
        self.corpus.add_stylization_pair("plain", "vaporwave", "spacing", Scale.CHARACTER)
        self.corpus.add_stylization_pair("lowercase", "uppercase", "case", Scale.CHARACTER)
        self.corpus.add_stylization_pair("plain", "leetspeak", "substitution", Scale.CHARACTER)
        self.corpus.add_stylization_pair("plain", "mocking", "mockery", Scale.CHARACTER)
        self.corpus.add_stylization_pair("plain", "stutter", "hesitation", Scale.CHARACTER)
        self.corpus.add_stylization_pair("plain", "uwu", "cuteness", Scale.CHARACTER)
        
        self._log(f"  Seeded stylization dimensions")
    
    def seed_all(self):
        """Seed all dimension types."""
        self._log("Seeding unified corpus...")
        self.seed_content()
        self.seed_patterns()
        self.seed_stylizations()
        self.corpus.recompute()
        self._log(f"  Total: {len(self.corpus.pairs)} pairs, {len(self.corpus.dimensions)} dimensions")
    
    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    
    def run_cycle(self, texts: List[str] = None) -> UnifiedAssemblyState:
        """
        Run one complete unified self-assembly cycle.
        
        Args:
            texts: Optional list of texts to ingest
        """
        self.total_cycles += 1
        state = UnifiedAssemblyState(cycle=self.total_cycles)
        
        # Capture before state
        self.corpus.recompute()
        state.pairs_before = len(self.corpus.pairs)
        
        self._log(f"\n{'='*60}")
        self._log(f"UNIFIED SELF-ASSEMBLY CYCLE {self.total_cycles}")
        self._log(f"{'='*60}")
        
        # Step 1: INGEST
        if texts:
            self._log("\n[1] INGEST: Extracting pairs at all scales...")
            content, pattern, style = self._ingest_all(texts)
            self._log(f"    Content: {content}, Pattern: {pattern}, Stylization: {style}")
        else:
            self._log("\n[1] INGEST: No new texts (gap-fill mode)")
        
        # Step 2: DETECT
        self._log("\n[2] DETECT: Checking dimensions by type and scale...")
        self.corpus.recompute()
        state.content_dimensions = len(self.corpus.get_dimensions_by_type(DimensionType.CONTENT))
        state.pattern_dimensions = len(self.corpus.get_dimensions_by_type(DimensionType.PATTERN))
        state.stylization_dimensions = len(self.corpus.get_dimensions_by_type(DimensionType.STYLIZATION))
        
        for scale in Scale:
            dims = self.corpus.get_dimensions_at_scale(scale)
            if dims:
                state.dimensions_by_scale[scale.name] = len(dims)
        
        self._log(f"    Content: {state.content_dimensions}, Pattern: {state.pattern_dimensions}, Stylization: {state.stylization_dimensions}")
        self._log(f"    By scale: {state.dimensions_by_scale}")
        
        # Step 3: REBALANCE
        self._log("\n[3] REBALANCE: Extending positions...")
        self._log(f"    All positions extended to {len(self.corpus.dimensions)} dimensions")
        
        # Step 4: POSITION
        self._log("\n[4] POSITION: Verifying φ-based positioning...")
        accuracy = self._verify_positioning()
        self._log(f"    Positioning accuracy: {accuracy:.1%}")
        
        # Step 5: DISCOVER
        self._log("\n[5] DISCOVER: Finding ideals...")
        state.content_ideals, state.pattern_ideals, state.stylization_ideals, state.cross_scale_ideals = self._discover_ideals()
        self._log(f"    Content: {state.content_ideals}, Pattern: {state.pattern_ideals}, Stylization: {state.stylization_ideals}, Cross-scale: {state.cross_scale_ideals}")
        
        # Step 6: GAP-FILL
        self._log("\n[6] GAP-FILL: Detecting gaps...")
        state.gaps_detected = self._detect_gaps()
        self._log(f"    Gaps detected: {state.gaps_detected}")
        
        # Step 7: COMPOUND
        self._log("\n[7] COMPOUND: Deriving multi-scale compounds...")
        state.compounds_derived = self._derive_compounds()
        self._log(f"    Compounds derived: {state.compounds_derived}")
        
        # Step 8: VERIFY
        self._log("\n[8] VERIFY: Checking unified self-similarity...")
        state.self_similarity_score = self._verify_self_similarity()
        self._log(f"    Self-similarity: {state.self_similarity_score:.2%}")
        
        # Capture after state
        self.corpus.recompute()
        state.pairs_after = len(self.corpus.pairs)
        
        # Summary
        self._log(f"\n{'─'*60}")
        self._log(f"CYCLE {self.total_cycles} COMPLETE")
        self._log(f"  Pairs: {state.pairs_before} → {state.pairs_after} (+{state.pairs_added()})")
        self._log(f"  Dimensions: {state.total_dimensions()} (C:{state.content_dimensions} P:{state.pattern_dimensions} S:{state.stylization_dimensions})")
        self._log(f"  Self-similarity: {state.self_similarity_score:.2%}")
        
        self.history.append(state)
        return state
    
    # -------------------------------------------------------------------------
    # Step Implementations
    # -------------------------------------------------------------------------
    
    def _ingest_all(self, texts: List[str]) -> Tuple[int, int, int]:
        """Ingest texts and extract pairs at all levels."""
        content_pairs = 0
        pattern_pairs = 0
        style_pairs = 0
        
        # Extract pattern pairs
        detected_patterns = self.pattern_detector.extract_pattern_pairs(texts)
        for source, target, dim in detected_patterns:
            if self.corpus.add_pattern_pair(source, target, dim, Scale.SENTENCE):
                pattern_pairs += 1
        
        # Detect stylizations in texts
        for text in texts:
            style = self.stylization_manager.detect(text)
            if style != 'plain':
                # This text has a stylization
                # We could add pairs if we had the plain version
                pass
        
        return content_pairs, pattern_pairs, style_pairs
    
    def _verify_positioning(self) -> float:
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
                
                if abs(magnitude - PHI) < 0.5:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _discover_ideals(self) -> Tuple[int, int, int, int]:
        """Discover ideals by type."""
        content_ideals = 0
        pattern_ideals = 0
        style_ideals = 0
        cross_scale_ideals = 0
        
        for ideal_name in self.corpus.list_ideals():
            ideal = self.corpus.get_ideal(ideal_name)
            if not ideal:
                continue
            
            # Check dimension types
            types = set()
            scales = set()
            
            for dim_name in ideal.dimensions_anchored:
                dim_type = self.corpus.get_dimension_type(dim_name)
                dim_scale = self.corpus.get_dimension_scale(dim_name)
                types.add(dim_type)
                if dim_scale:
                    scales.add(dim_scale)
            
            # Categorize
            if len(scales) > 1:
                cross_scale_ideals += 1
            elif DimensionType.CONTENT in types:
                content_ideals += 1
            elif DimensionType.PATTERN in types:
                pattern_ideals += 1
            elif DimensionType.STYLIZATION in types:
                style_ideals += 1
        
        return content_ideals, pattern_ideals, style_ideals, cross_scale_ideals
    
    def _detect_gaps(self) -> int:
        """Detect gaps across all dimension types and scales."""
        gaps = 0
        
        for ideal_name in self.corpus.list_ideals():
            ideal = self.corpus.get_ideal(ideal_name)
            if not ideal:
                continue
            
            for dim_name in self.corpus.dimensions:
                has_pair = any(
                    (p.source == ideal_name or p.target == ideal_name) and
                    p.relationship == dim_name
                    for p in self.corpus.pairs
                )
                if not has_pair:
                    gaps += 1
        
        return gaps
    
    def _derive_compounds(self) -> int:
        """Derive multi-scale compound positions."""
        compounds = 0
        
        # Get concepts by type
        content = [c for c in list(self.corpus.concepts.keys())[:10]
                  if self.corpus.get_concept_type(c) == DimensionType.CONTENT]
        patterns = [c for c in list(self.corpus.concepts.keys())[:10]
                   if self.corpus.get_concept_type(c) == DimensionType.PATTERN]
        styles = [c for c in list(self.corpus.concepts.keys())[:10]
                 if self.corpus.get_concept_type(c) == DimensionType.STYLIZATION]
        
        # Compose across types
        for c in content[:3]:
            for p in patterns[:2]:
                pos = self.corpus.compose(c, p)
                if pos is not None:
                    compounds += 1
        
        # Compose across all three
        for c in content[:2]:
            for p in patterns[:2]:
                for s in styles[:2]:
                    pos = self.corpus.compose(c, p, s)
                    if pos is not None:
                        compounds += 1
        
        return compounds
    
    def _verify_self_similarity(self) -> float:
        """Verify self-similarity across the unified space."""
        if len(self.corpus.pairs) == 0:
            return 0.0
        
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
        
        consistencies = []
        for dim, deltas in deltas_by_dim.items():
            if len(deltas) > 1:
                mean_delta = np.mean(deltas)
                std_delta = np.std(deltas)
                if mean_delta > 0:
                    consistency = 1.0 - (std_delta / mean_delta)
                    consistencies.append(max(0, consistency))
        
        return np.mean(consistencies) if consistencies else 1.0
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def get_status(self) -> Dict:
        """Get current status of the unified corpus."""
        return self.corpus.get_status()
    
    def run_until_stable(self, texts: List[str] = None,
                         max_cycles: int = 5) -> List[UnifiedAssemblyState]:
        """Run cycles until stable."""
        states = []
        
        for i in range(max_cycles):
            state = self.run_cycle(texts if i == 0 else None)
            states.append(state)
            
            if (state.gaps_detected == 0 and
                state.self_similarity_score >= self.min_self_similarity):
                self._log(f"\n✓ Corpus stable after {i+1} cycles")
                break
        
        return states


# =============================================================================
# DEMO
# =============================================================================

def demo_unified_loop():
    """Demonstrate the unified self-assembly loop."""
    print("=" * 70)
    print("DEMO: Unified Self-Assembly Loop")
    print("=" * 70)
    print()
    print("This demo shows the unified loop handling ALL dimension types")
    print("(content, pattern, stylization) at ALL scales.")
    print()
    
    loop = UnifiedSelfAssemblyLoop(verbose=True)
    
    # Seed all dimension types
    loop.seed_all()
    
    # Sample texts
    sample_texts = [
        "The king ruled wisely over his kingdom.",
        "Hey, the king was pretty cool, ya know?",
        "His Majesty governed with utmost prudence.",
        "What manner of ruler was the king?",
    ]
    
    # Run a cycle
    state = loop.run_cycle(sample_texts)
    
    # Show status
    print()
    print("=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    status = loop.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test multi-scale composition
    print()
    print("Multi-scale composition test:")
    print("-" * 60)
    
    compositions = [
        ["king", "formal"],
        ["queen", "casual", "vaporwave"],
        ["king", "verbose", "uppercase"],
    ]
    
    for concepts in compositions:
        pos = loop.corpus.compose(*concepts)
        if pos is not None:
            content, pattern, style = loop.corpus.decompose(pos)
            print(f"  {' + '.join(concepts)}:")
            print(f"    → Content: {content}, Pattern: {pattern}, Style: {style}")
    
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    
    return loop


if __name__ == "__main__":
    demo_unified_loop()
