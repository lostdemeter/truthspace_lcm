#!/usr/bin/env python3
"""
Error-Driven Pattern Discovery for Bootstrap Knowledge

This module implements the principle: "Error = Where to Build"

When extraction fails on a sentence that likely contains a relation,
we analyze the sentence structure to discover new patterns that
could be added to bootstrap_knowledge.json.

Key insight from prior work:
- Errors don't measure accuracy, they tell us WHERE to add structure
- Each failed extraction points to a missing pattern
- The system discovers its own structure through use
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path


@dataclass
class PatternCandidate:
    """A candidate pattern discovered from text."""
    pattern: str
    regex: str
    example_sentences: List[str] = field(default_factory=list)
    extracted_entities: List[Tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    relation_type: str = ""


@dataclass
class ExtractionFailure:
    """A sentence where extraction failed but likely contains relations."""
    sentence: str
    entities_found: List[str]
    verbs_found: List[str]
    structure: str  # e.g., "ENTITY VERB ENTITY"


class PatternLearner:
    """
    Learns new extraction patterns from failed extractions.
    
    Uses error-driven construction: analyze sentences where extraction
    failed but entities were present, and discover the patterns that
    connect them.
    """
    
    def __init__(self, bootstrap_knowledge=None):
        from .bootstrap import get_bootstrap_knowledge
        self.bootstrap = bootstrap_knowledge or get_bootstrap_knowledge()
        
        # Track failures for pattern discovery
        self.failures: List[ExtractionFailure] = []
        self.discovered_patterns: List[PatternCandidate] = []
        
        # Common verbs that indicate relations
        self.relation_verbs = {
            # Existing patterns
            "said", "replied", "asked", "wrote", "killed", "attacked",
            # New potential patterns
            "loved", "hated", "feared", "wanted", "needed",
            "saw", "met", "found", "lost", "took", "gave",
            "told", "showed", "taught", "learned",
            "married", "divorced", "kissed", "embraced",
            "followed", "led", "helped", "saved", "rescued",
            "fought", "defeated", "captured", "escaped",
            "entered", "left", "arrived", "departed",
            "lived", "died", "born", "grew",
            "became", "remained", "seemed", "appeared",
            "believed", "thought", "knew", "understood",
            "remembered", "forgot", "recognized",
        }
        
        # Patterns for entity detection
        self.entity_pattern = re.compile(r'\b([A-Z][a-z]+)\b')
        self.verb_pattern = re.compile(r'\b([a-z]+ed|[a-z]+s)\b')
    
    def analyze_sentence(self, sentence: str, extracted_facts: List) -> Optional[ExtractionFailure]:
        """
        Analyze a sentence to see if extraction failed despite having entities.
        
        Returns an ExtractionFailure if the sentence likely contains
        unextracted relations.
        """
        # Find entities (capitalized words)
        entities = []
        for match in self.entity_pattern.finditer(sentence):
            word = match.group(1)
            if word.lower() not in {'the', 'a', 'an', 'i', 'he', 'she', 'it', 'they', 'we', 'you'}:
                entities.append(word)
        
        # Find verbs
        verbs = []
        for match in self.verb_pattern.finditer(sentence.lower()):
            word = match.group(1)
            # Check if it's a relation verb
            base = word.rstrip('ed').rstrip('s')
            if word in self.relation_verbs or base in self.relation_verbs:
                verbs.append(word)
        
        # If we have 2+ entities and a verb but no facts, this is a failure
        if len(entities) >= 2 and len(verbs) >= 1 and len(extracted_facts) == 0:
            # Determine structure
            structure = self._analyze_structure(sentence, entities, verbs)
            
            failure = ExtractionFailure(
                sentence=sentence,
                entities_found=entities,
                verbs_found=verbs,
                structure=structure
            )
            self.failures.append(failure)
            return failure
        
        return None
    
    def _analyze_structure(self, sentence: str, entities: List[str], verbs: List[str]) -> str:
        """Analyze the structure of a sentence."""
        # Simple structure detection
        words = sentence.split()
        structure_parts = []
        
        for word in words:
            clean = word.strip('.,!?;:"\'-')
            if clean in entities:
                structure_parts.append("ENTITY")
            elif clean.lower() in verbs or clean.lower().rstrip('ed').rstrip('s') in self.relation_verbs:
                structure_parts.append("VERB")
            elif clean.lower() in {'the', 'a', 'an'}:
                structure_parts.append("ART")
            elif clean.lower() in {'in', 'on', 'at', 'to', 'from', 'with', 'by'}:
                structure_parts.append("PREP")
            else:
                structure_parts.append("_")
        
        # Simplify consecutive underscores
        simplified = []
        prev = None
        for part in structure_parts:
            if part != "_" or prev != "_":
                simplified.append(part)
            prev = part
        
        return " ".join(simplified)
    
    def discover_patterns(self, min_occurrences: int = 2) -> List[PatternCandidate]:
        """
        Discover new patterns from accumulated failures.
        
        Groups failures by structure and proposes patterns for
        structures that appear multiple times.
        """
        # Group by structure
        structure_groups = defaultdict(list)
        for failure in self.failures:
            structure_groups[failure.structure].append(failure)
        
        candidates = []
        
        for structure, failures in structure_groups.items():
            if len(failures) >= min_occurrences:
                # This structure appears multiple times - likely a pattern
                candidate = self._create_pattern_candidate(structure, failures)
                if candidate:
                    candidates.append(candidate)
        
        self.discovered_patterns = candidates
        return candidates
    
    def _create_pattern_candidate(self, structure: str, failures: List[ExtractionFailure]) -> Optional[PatternCandidate]:
        """Create a pattern candidate from a structure and examples."""
        # Find the common verb
        verb_counts = defaultdict(int)
        for failure in failures:
            for verb in failure.verbs_found:
                verb_counts[verb] += 1
        
        if not verb_counts:
            return None
        
        common_verb = max(verb_counts, key=verb_counts.get)
        
        # Determine relation type from verb
        relation_type = self._verb_to_relation(common_verb)
        
        # Build regex pattern
        # Simple pattern: ENTITY verb ENTITY
        regex = f"([A-Z][a-z]+)\\s+{common_verb}\\s+([A-Z][a-z]+)"
        
        # Build human-readable pattern
        pattern = f"[ENTITY] {common_verb} [ENTITY2]"
        
        # Extract example entities
        examples = []
        entities = []
        for failure in failures[:5]:  # Limit examples
            examples.append(failure.sentence)
            if len(failure.entities_found) >= 2:
                entities.append((failure.entities_found[0], failure.entities_found[1]))
        
        return PatternCandidate(
            pattern=pattern,
            regex=regex,
            example_sentences=examples,
            extracted_entities=entities,
            confidence=len(failures) / len(self.failures) if self.failures else 0,
            relation_type=relation_type
        )
    
    def _verb_to_relation(self, verb: str) -> str:
        """Convert a verb to a relation name."""
        # Remove tense suffixes
        base = verb.rstrip('ed').rstrip('s')
        
        # Map to relation names
        verb_map = {
            "lov": "loves",
            "hat": "hates",
            "fear": "fears",
            "want": "wants",
            "need": "needs",
            "saw": "saw",
            "see": "saw",
            "met": "met",
            "meet": "met",
            "found": "found",
            "find": "found",
            "took": "took",
            "take": "took",
            "gave": "gave_to",
            "give": "gave_to",
            "told": "told",
            "tell": "told",
            "follow": "followed",
            "help": "helped",
            "save": "saved",
            "fight": "fought",
            "defeat": "defeated",
            "enter": "entered",
            "leav": "left",
            "arriv": "arrived",
            "liv": "lives_in",
            "di": "died",
            "born": "born_in",
            "becam": "became",
            "believ": "believes",
            "thought": "thinks",
            "think": "thinks",
            "knew": "knows",
            "know": "knows",
            "remember": "remembers",
            "recogniz": "recognized",
        }
        
        for prefix, relation in verb_map.items():
            if base.startswith(prefix):
                return relation
        
        # Default: use verb as relation
        return base + "s" if not base.endswith('s') else base
    
    def generate_json_patterns(self) -> List[Dict]:
        """Generate JSON patterns for bootstrap_knowledge.json."""
        patterns = []
        
        for candidate in self.discovered_patterns:
            pattern_dict = {
                "name": candidate.relation_type,
                "regex": candidate.regex,
                "groups": ["entity1", "entity2"],
                "fact": ["entity1", candidate.relation_type, "entity2"],
                "description": f"{candidate.pattern} -> (entity1, {candidate.relation_type}, entity2)",
                "discovered": True,
                "confidence": candidate.confidence,
                "examples": candidate.example_sentences[:3]
            }
            patterns.append(pattern_dict)
        
        return patterns
    
    def save_discovered_patterns(self, filepath: str = None):
        """Save discovered patterns to a JSON file."""
        if filepath is None:
            filepath = Path(__file__).parent.parent / "discovered_patterns.json"
        
        patterns = self.generate_json_patterns()
        
        with open(filepath, 'w') as f:
            json.dump({
                "version": "1.0",
                "description": "Patterns discovered through error-driven learning",
                "patterns": patterns,
                "total_failures_analyzed": len(self.failures),
            }, f, indent=2)
        
        return filepath
    
    def get_statistics(self) -> Dict:
        """Get statistics about pattern discovery."""
        structure_counts = defaultdict(int)
        verb_counts = defaultdict(int)
        
        for failure in self.failures:
            structure_counts[failure.structure] += 1
            for verb in failure.verbs_found:
                verb_counts[verb] += 1
        
        return {
            "total_failures": len(self.failures),
            "unique_structures": len(structure_counts),
            "top_structures": sorted(structure_counts.items(), key=lambda x: -x[1])[:10],
            "top_verbs": sorted(verb_counts.items(), key=lambda x: -x[1])[:10],
            "discovered_patterns": len(self.discovered_patterns),
        }
    
    def clear(self):
        """Clear accumulated failures."""
        self.failures = []
        self.discovered_patterns = []


def analyze_book_for_patterns(text: str, max_sentences: int = 1000) -> Dict:
    """
    Analyze a book to discover new patterns.
    
    Returns statistics and discovered patterns.
    """
    from .bootstrap import get_bootstrap_knowledge, BootstrapKnowledge
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10][:max_sentences]
    
    # Create fresh bootstrap and learner
    bootstrap = BootstrapKnowledge()
    learner = PatternLearner(bootstrap)
    
    # Process sentences
    for sentence in sentences:
        facts = bootstrap.extract_facts(sentence, use_coreference=False)
        learner.analyze_sentence(sentence, facts)
    
    # Discover patterns
    learner.discover_patterns(min_occurrences=2)
    
    return {
        "statistics": learner.get_statistics(),
        "patterns": learner.generate_json_patterns(),
    }
