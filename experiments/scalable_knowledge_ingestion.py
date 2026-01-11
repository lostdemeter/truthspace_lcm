"""
Experiment: Scalable Knowledge Ingestion via LLM-Assisted Concept Extraction

This experiment explores how to ingest large corpora of knowledge into the
geometric model by:

1. CONCEPT EXTRACTION: Parse input text to identify unknown concepts
2. LLM ENRICHMENT: Query local LLM to generate concept variations across dimensions
3. GEOMETRIC POSITIONING: Place concepts in semantic space based on their properties
4. STYLE AS POSITION: Treat styles (grimdark, formal, casual) as positions, not presets

The key insight: Styles aren't presets - they're regions in semantic space.
A "grimdark" version of a concept is just that concept shifted along certain dimensions.

Example:
  Input: "Linear algebra is the branch of mathematics..."
  
  For concept "linear algebra":
    - Base position: [0, 0.5, 0.8, 0]  (present, formal, technical, neutral)
    - Grimdark: [0, 0.3, 0.9, 0.8]    (present, dramatic, arcane, intense)
    - Casual: [0, -0.5, 0.5, 0]       (present, casual, semi-technical, neutral)
  
  The LLM generates the TEXT at each position, we store the mapping.

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
import re
import json
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from pathlib import Path
import hashlib
import time


PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# STYLE POSITIONS - Styles are regions in semantic space, not presets
# =============================================================================

@dataclass
class StylePosition:
    """A style is a position in semantic space."""
    name: str
    description: str
    position: np.ndarray  # [tense, formality, domain, intensity, drama, abstraction]
    
    def distance_to(self, other: np.ndarray) -> float:
        return np.linalg.norm(self.position - other)


# 6D semantic space for richer style representation
# Dimensions: [tense, formality, domain, intensity, drama, abstraction]
STYLE_POSITIONS = {
    "neutral": StylePosition(
        name="neutral",
        description="Standard, balanced expression",
        position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ),
    "formal": StylePosition(
        name="formal",
        description="Academic, professional tone",
        position=np.array([0.0, 0.8, 0.5, 0.0, 0.0, 0.3])
    ),
    "casual": StylePosition(
        name="casual", 
        description="Relaxed, conversational",
        position=np.array([0.0, -0.7, -0.3, 0.0, 0.0, -0.3])
    ),
    "grimdark": StylePosition(
        name="grimdark",
        description="Dark, dramatic, ominous",
        position=np.array([0.0, 0.3, 0.6, 0.9, 0.95, 0.5])
    ),
    "warhammer40k": StylePosition(
        name="warhammer40k",
        description="Gothic sci-fi, religious fervor",
        position=np.array([0.0, 0.6, 0.8, 0.85, 0.9, 0.4])
    ),
    "eli5": StylePosition(
        name="eli5",
        description="Explain like I'm 5 - simple, concrete",
        position=np.array([0.0, -0.5, -0.8, 0.0, 0.2, -0.8])
    ),
    "poetic": StylePosition(
        name="poetic",
        description="Lyrical, metaphorical",
        position=np.array([0.0, 0.4, 0.3, 0.5, 0.6, 0.7])
    ),
    "technical": StylePosition(
        name="technical",
        description="Precise, jargon-heavy",
        position=np.array([0.0, 0.5, 0.9, 0.0, 0.0, 0.6])
    ),
}


# =============================================================================
# CONCEPT EXTRACTOR - Identify concepts in text
# =============================================================================

class ConceptExtractor:
    """
    Extracts concepts from text for geometric positioning.
    
    A concept is a meaningful unit that can be positioned in semantic space.
    This includes: nouns, noun phrases, technical terms, named entities.
    """
    
    def __init__(self):
        # Common words to skip
        self._stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
            'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'what',
            'its', 'their', 'it', 'they', 'them', 'he', 'she', 'his', 'her',
            'including', 'using', 'between', 'studies', 'branch',
        }
    
    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract concepts from text.
        
        Returns list of concepts ordered by importance (longer phrases first).
        """
        concepts = set()
        
        # Extract noun phrases (simple heuristic: capitalized sequences, quoted terms)
        # Technical terms often have specific patterns
        
        # 1. Multi-word technical terms (2-4 words)
        words = text.split()
        for n in [4, 3, 2]:
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                # Clean punctuation
                phrase_clean = re.sub(r'[^\w\s-]', '', phrase).strip().lower()
                if self._is_valid_phrase(phrase_clean):
                    concepts.add(phrase_clean)
        
        # 2. Single important words
        for word in words:
            word_clean = re.sub(r'[^\w-]', '', word).strip().lower()
            if self._is_valid_word(word_clean):
                concepts.add(word_clean)
        
        # Sort by length (longer = more specific = higher priority)
        return sorted(concepts, key=lambda x: (-len(x), x))
    
    def _is_valid_phrase(self, phrase: str) -> bool:
        """Check if phrase is a valid concept."""
        words = phrase.split()
        if len(words) < 2:
            return False
        
        # At least one non-stopword
        non_stop = [w for w in words if w not in self._stopwords]
        if len(non_stop) < 1:
            return False
        
        # Not all stopwords
        if len(non_stop) == len(words):
            return True
        
        # Allow some stopwords if there's substance
        return len(non_stop) >= len(words) // 2
    
    def _is_valid_word(self, word: str) -> bool:
        """Check if word is a valid concept."""
        if len(word) < 3:
            return False
        if word in self._stopwords:
            return False
        if word.isdigit():
            return False
        return True


# =============================================================================
# LLM INTERFACE - Query local LLM for concept enrichment
# =============================================================================

class LocalLLM:
    """Interface to local Ollama LLM for concept enrichment."""
    
    def __init__(self, model: str = "qwen2.5:14b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._cache: Dict[str, str] = {}
    
    def query(self, prompt: str, cache_key: Optional[str] = None) -> Optional[str]:
        """Query the LLM with caching."""
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500,
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "")
                if cache_key:
                    self._cache[cache_key] = result
                return result
            else:
                return None
                
        except Exception as e:
            print(f"LLM query failed: {e}")
            return None
    
    def rewrite_in_style(self, text: str, style: StylePosition, context: str = "") -> Optional[str]:
        """Rewrite text in a specific style."""
        
        style_prompts = {
            "grimdark": "Rewrite in a dark, ominous, dramatic style with gothic horror undertones. Make it sound like forbidden knowledge.",
            "warhammer40k": "Rewrite in the style of Warhammer 40K lore - gothic sci-fi with religious fervor, treating technology as sacred and knowledge as holy scripture.",
            "eli5": "Rewrite as if explaining to a 5-year-old. Use simple words, concrete examples, and avoid jargon.",
            "formal": "Rewrite in formal academic style with precise terminology and professional tone.",
            "casual": "Rewrite in casual, conversational style as if chatting with a friend.",
            "poetic": "Rewrite in lyrical, poetic style with metaphors and evocative imagery.",
            "technical": "Rewrite with maximum technical precision, using proper jargon and formal definitions.",
        }
        
        style_instruction = style_prompts.get(style.name, f"Rewrite in {style.name} style.")
        
        prompt = f"""{style_instruction}

Original text: "{text}"

{f'Context: {context}' if context else ''}

Rewritten version (just the rewritten text, no explanation):"""
        
        cache_key = hashlib.md5(f"{text}:{style.name}".encode()).hexdigest()
        return self.query(prompt, cache_key)
    
    def define_concept(self, concept: str, context: str = "") -> Optional[str]:
        """Get a definition/explanation of a concept."""
        prompt = f"""Define the concept "{concept}" in one clear sentence.
{f'Context: {context}' if context else ''}

Definition:"""
        
        cache_key = hashlib.md5(f"define:{concept}".encode()).hexdigest()
        return self.query(prompt, cache_key)


# =============================================================================
# CONCEPT CORPUS - Storage for positioned concepts
# =============================================================================

@dataclass
class PositionedConcept:
    """A concept with its position and text at that position."""
    concept: str           # The base concept name
    position: np.ndarray   # Position in semantic space
    text: str              # The text representation at this position
    style: str             # Style name (for reference)
    source: str            # Where this came from (original, llm, etc.)


class ConceptCorpus:
    """
    Corpus of concepts positioned in semantic space.
    
    Each concept can have multiple positions (one per style).
    This enables fluid transformation between styles.
    """
    
    def __init__(self, dims: int = 6):
        self.dims = dims
        self._concepts: Dict[str, List[PositionedConcept]] = defaultdict(list)
        self._by_position: List[PositionedConcept] = []
    
    def add(self, concept: str, position: np.ndarray, text: str, 
            style: str = "neutral", source: str = "unknown"):
        """Add a positioned concept."""
        pc = PositionedConcept(
            concept=concept,
            position=position.copy(),
            text=text,
            style=style,
            source=source
        )
        self._concepts[concept].append(pc)
        self._by_position.append(pc)
    
    def get_concept_positions(self, concept: str) -> List[PositionedConcept]:
        """Get all positions for a concept."""
        return self._concepts.get(concept, [])
    
    def find_nearest(self, position: np.ndarray, concept: Optional[str] = None) -> Optional[PositionedConcept]:
        """Find nearest positioned concept to a position."""
        candidates = self._concepts.get(concept, []) if concept else self._by_position
        
        if not candidates:
            return None
        
        best = None
        best_dist = float('inf')
        
        for pc in candidates:
            dist = np.linalg.norm(pc.position - position)
            if dist < best_dist:
                best_dist = dist
                best = pc
        
        return best
    
    def transform_concept(self, concept: str, from_style: str, to_style: str) -> Optional[str]:
        """Transform a concept from one style to another."""
        from_pos = STYLE_POSITIONS.get(from_style)
        to_pos = STYLE_POSITIONS.get(to_style)
        
        if not from_pos or not to_pos:
            return None
        
        # Find the concept at target position
        target = self.find_nearest(to_pos.position, concept)
        return target.text if target else None
    
    def stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        return {
            "total_concepts": len(self._concepts),
            "total_positions": len(self._by_position),
            "by_style": {
                style: sum(1 for pc in self._by_position if pc.style == style)
                for style in STYLE_POSITIONS.keys()
            },
            "concepts": list(self._concepts.keys())[:20],
        }
    
    def save(self, path: Path):
        """Save corpus to file."""
        data = {
            "dims": self.dims,
            "concepts": [
                {
                    "concept": pc.concept,
                    "position": pc.position.tolist(),
                    "text": pc.text,
                    "style": pc.style,
                    "source": pc.source,
                }
                for pc in self._by_position
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ConceptCorpus':
        """Load corpus from file."""
        with open(path) as f:
            data = json.load(f)
        
        corpus = cls(dims=data.get("dims", 6))
        for item in data.get("concepts", []):
            corpus.add(
                concept=item["concept"],
                position=np.array(item["position"]),
                text=item["text"],
                style=item["style"],
                source=item["source"]
            )
        return corpus


# =============================================================================
# KNOWLEDGE INGESTER - Main ingestion pipeline
# =============================================================================

class KnowledgeIngester:
    """
    Ingests text into the geometric corpus.
    
    Pipeline:
    1. Extract concepts from input text
    2. For each unknown concept:
       a. Get base definition from LLM
       b. Generate style variations via LLM
       c. Position each variation in semantic space
    3. Store in corpus
    """
    
    def __init__(self, corpus: Optional[ConceptCorpus] = None, 
                 llm: Optional[LocalLLM] = None):
        self.corpus = corpus or ConceptCorpus(dims=6)
        self.llm = llm or LocalLLM()
        self.extractor = ConceptExtractor()
        self._known_concepts: Set[str] = set()
    
    def ingest(self, text: str, styles: Optional[List[str]] = None,
               context: str = "", verbose: bool = True) -> Dict[str, Any]:
        """
        Ingest text into the corpus.
        
        Args:
            text: Input text to ingest
            styles: List of styles to generate (default: all)
            context: Additional context for LLM
            verbose: Print progress
            
        Returns:
            Statistics about ingestion
        """
        styles = styles or list(STYLE_POSITIONS.keys())
        
        # Extract concepts
        concepts = self.extractor.extract_concepts(text)
        
        if verbose:
            print(f"Extracted {len(concepts)} concepts from text")
            print(f"Concepts: {concepts[:10]}{'...' if len(concepts) > 10 else ''}")
        
        # Filter to unknown concepts
        unknown = [c for c in concepts if c not in self._known_concepts]
        
        if verbose:
            print(f"Unknown concepts: {len(unknown)}")
        
        stats = {
            "input_length": len(text),
            "concepts_extracted": len(concepts),
            "concepts_unknown": len(unknown),
            "styles_generated": 0,
            "errors": [],
        }
        
        # Process each unknown concept
        for i, concept in enumerate(unknown):
            if verbose:
                print(f"\n[{i+1}/{len(unknown)}] Processing: {concept}")
            
            try:
                self._process_concept(concept, text, styles, context, verbose)
                self._known_concepts.add(concept)
                stats["styles_generated"] += len(styles)
            except Exception as e:
                stats["errors"].append(f"{concept}: {e}")
                if verbose:
                    print(f"  Error: {e}")
        
        return stats
    
    def _process_concept(self, concept: str, source_text: str,
                         styles: List[str], context: str, verbose: bool):
        """Process a single concept across all styles."""
        
        # Get base definition
        definition = self.llm.define_concept(concept, context)
        if not definition:
            definition = concept  # Fallback to concept name
        
        if verbose:
            print(f"  Definition: {definition[:100]}...")
        
        # Add neutral position
        neutral_pos = STYLE_POSITIONS["neutral"].position
        self.corpus.add(
            concept=concept,
            position=neutral_pos,
            text=definition,
            style="neutral",
            source="llm_definition"
        )
        
        # Generate style variations
        for style_name in styles:
            if style_name == "neutral":
                continue
            
            style = STYLE_POSITIONS.get(style_name)
            if not style:
                continue
            
            if verbose:
                print(f"  Generating {style_name}...")
            
            # Get styled version from LLM
            styled_text = self.llm.rewrite_in_style(definition, style, context)
            
            if styled_text:
                self.corpus.add(
                    concept=concept,
                    position=style.position,
                    text=styled_text.strip(),
                    style=style_name,
                    source="llm_style"
                )
                
                if verbose:
                    print(f"    → {styled_text[:80]}...")
            
            # Small delay to avoid overwhelming LLM
            time.sleep(0.1)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_concept_extraction():
    """Demonstrate concept extraction."""
    print("=" * 60)
    print("CONCEPT EXTRACTION")
    print("=" * 60)
    print()
    
    extractor = ConceptExtractor()
    
    test_text = """Linear algebra is the branch of mathematics that studies systems of 
    linear equations and linear transformations between vector spaces, including their 
    geometric interpretations and representations using matrices."""
    
    concepts = extractor.extract_concepts(test_text)
    
    print(f"Input text:\n{test_text}\n")
    print(f"Extracted {len(concepts)} concepts:")
    for c in concepts:
        print(f"  - {c}")
    print()


def demo_style_positions():
    """Demonstrate style as position concept."""
    print("=" * 60)
    print("STYLES AS POSITIONS")
    print("=" * 60)
    print()
    
    print("Style positions in 6D semantic space:")
    print("Dimensions: [tense, formality, domain, intensity, drama, abstraction]")
    print("-" * 60)
    
    for name, style in STYLE_POSITIONS.items():
        pos_str = "[" + ", ".join(f"{v:+.1f}" for v in style.position) + "]"
        print(f"{name:15} {pos_str}")
    
    print()
    
    # Show distances between styles
    print("Style distances (Euclidean):")
    print("-" * 60)
    
    styles = list(STYLE_POSITIONS.values())
    for i, s1 in enumerate(styles):
        for s2 in styles[i+1:]:
            dist = s1.distance_to(s2.position)
            print(f"  {s1.name} ↔ {s2.name}: {dist:.2f}")
    print()


def demo_ingestion_dry_run():
    """Demonstrate ingestion without LLM (dry run)."""
    print("=" * 60)
    print("INGESTION DRY RUN (No LLM)")
    print("=" * 60)
    print()
    
    corpus = ConceptCorpus(dims=6)
    
    # Manually add some positioned concepts to show the structure
    test_concepts = [
        ("linear algebra", "neutral", "Linear algebra is the study of linear equations and vector spaces."),
        ("linear algebra", "grimdark", "In the dark realm of mathematics, Linear Algebra lurks as the dread study of equations that bind dimensions together."),
        ("linear algebra", "eli5", "Linear algebra is like playing with arrows and boxes - you can stretch them, spin them, and stack them up!"),
        ("vector space", "neutral", "A vector space is a collection of vectors that can be added together and scaled."),
        ("vector space", "grimdark", "The Vector Space - a haunted void where mathematical entities drift eternally, bound by unholy laws of addition and scaling."),
        ("matrix", "neutral", "A matrix is a rectangular array of numbers arranged in rows and columns."),
        ("matrix", "warhammer40k", "The Sacred Matrix - a holy grid of numerals, each cell containing divine truth, arranged by the Emperor's own design."),
    ]
    
    for concept, style, text in test_concepts:
        pos = STYLE_POSITIONS[style].position
        corpus.add(concept, pos, text, style, "manual")
    
    print("Corpus statistics:")
    stats = corpus.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Demonstrate transformation
    print("Style transformation example:")
    print("-" * 60)
    
    # Find linear algebra in different styles
    for style_name in ["neutral", "grimdark", "eli5"]:
        style = STYLE_POSITIONS[style_name]
        result = corpus.find_nearest(style.position, "linear algebra")
        if result:
            print(f"\n{style_name.upper()}:")
            print(f"  {result.text}")
    print()


def demo_full_ingestion():
    """Demonstrate full ingestion with LLM."""
    print("=" * 60)
    print("FULL INGESTION (With LLM)")
    print("=" * 60)
    print()
    
    # Check if LLM is available
    llm = LocalLLM()
    test = llm.query("Say 'ok' if you can hear me.")
    
    if not test:
        print("LLM not available. Skipping full ingestion demo.")
        print("Start Ollama with: ollama serve")
        return
    
    print("LLM available. Starting ingestion...")
    print()
    
    ingester = KnowledgeIngester()
    
    test_text = """Linear algebra is the branch of mathematics that studies systems of 
    linear equations and linear transformations between vector spaces."""
    
    # Only do a few styles to keep demo short
    styles = ["neutral", "grimdark", "eli5"]
    
    stats = ingester.ingest(
        text=test_text,
        styles=styles,
        context="This is about mathematics and linear algebra.",
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Stats: {stats}")
    print()
    print("Corpus contents:")
    print(ingester.corpus.stats())


if __name__ == "__main__":
    demo_concept_extraction()
    demo_style_positions()
    demo_ingestion_dry_run()
    
    # Uncomment to run with LLM:
    demo_full_ingestion()
    
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print()
    print("Key insights:")
    print("1. Styles are POSITIONS in semantic space, not presets")
    print("2. Concepts can exist at multiple positions (one per style)")
    print("3. LLM generates text AT each position, we store the mapping")
    print("4. Transformation = find_nearest at target position")
    print("5. This enables fluid, continuous style interpolation")
