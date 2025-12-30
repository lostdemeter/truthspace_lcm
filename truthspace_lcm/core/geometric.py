#!/usr/bin/env python3
"""
Geometric Core: Fully Geometric Language Understanding

This module replaces hash-based and hard-coded approaches with pure geometry:
1. Position-based encoding (not hash-based)
2. Geometric stop word detection (no hard-coded lists)
3. Geometric morphology (learned from parallel structures)
4. Geometric conjugation (learned from parallel structures)

Core Principle: All semantic operations are geometric operations in concept space.

Mathematical Foundation:
- Position encodes semantic role (subject at 0, verb at 0.5, object at 1)
- Frequency distinguishes content from function words (Zipf's law)
- Parallel structure reveals morphological relationships

Author: Lesley Gushurst
License: GPLv3
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = 1.618034  # Golden ratio

# Bootstrap text for learning morphological patterns
# Format: "I [base]. He [3rd-singular]. I [past]."
MORPHOLOGY_BOOTSTRAP = """
I love. He loves. I loved.
I run. He runs. I ran.
I see. He sees. I saw.
I watch. He watches. I watched.
I go. He goes. I went.
I fall. He falls. I fell.
I speak. He speaks. I spoke.
I write. He writes. I wrote.
I read. He reads. I read.
I give. He gives. I gave.
I take. He takes. I took.
I make. He makes. I made.
I grow. He grows. I grew.
I know. He knows. I knew.
I think. He thinks. I thought.
I say. He says. I said.
I come. He comes. I came.
I find. He finds. I found.
I leave. He leaves. I left.
I begin. He begins. I began.
I examine. He examines. I examined.
I observe. He observes. I observed.
I assist. He assists. I assisted.
I question. He questions. I questioned.
I solve. He solves. I solved.
I end. He ends. I ended.
I kill. He kills. I killed.
I confront. He confronts. I confronted.
I reveal. He reveals. I revealed.
I drown. He drowns. I drowned.
I witness. He witnesses. I witnessed.
I ponder. He ponders. I pondered.
I poison. He poisons. I poisoned.
I propose. He proposes. I proposed.
I order. He orders. I ordered.
I explore. He explores. I explored.
I shout. He shouts. I shouted.
I smile. He smiles. I smiled.
I vanish. He vanishes. I vanished.
I ask. He asks. I asked.
I deduce. He deduces. I deduced.
I plot. He plots. I plotted.
I scheme. He schemes. I schemed.
I study. He studies. I studied.
I flee. He flees. I fled.
I wake. He wakes. I woke.
I shrink. He shrinks. I shrank.
I laugh. He laughs. I laughed.
I drink. He drinks. I drank.
I seek. He seeks. I sought.
I walk. He walks. I walked.
I talk. He talks. I talked.
I look. He looks. I looked.
I want. He wants. I wanted.
I need. He needs. I needed.
I feel. He feels. I felt.
I hear. He hears. I heard.
I try. He tries. I tried.
I use. He uses. I used.
I call. He calls. I called.
I tell. He tells. I told.
I show. He shows. I showed.
I move. He moves. I moved.
I live. He lives. I lived.
I believe. He believes. I believed.
I bring. He brings. I brought.
I happen. He happens. I happened.
I write. He writes. I wrote.
I sit. He sits. I sat.
I stand. He stands. I stood.
I lose. He loses. I lost.
I pay. He pays. I paid.
I meet. He meets. I met.
I include. He includes. I included.
I continue. He continues. I continued.
I set. He sets. I set.
I learn. He learns. I learned.
I change. He changes. I changed.
I lead. He leads. I led.
I understand. He understands. I understood.
I follow. He follows. I followed.
I stop. He stops. I stopped.
I create. He creates. I created.
I open. He opens. I opened.
I seem. He seems. I seemed.
I help. He helps. I helped.
I start. He starts. I started.
I hold. He holds. I held.
I remember. He remembers. I remembered.
I consider. He considers. I considered.
I appear. He appears. I appeared.
I buy. He buys. I bought.
I wait. He waits. I waited.
I serve. He serves. I served.
I die. He dies. I died.
I send. He sends. I sent.
I expect. He expects. I expected.
I build. He builds. I built.
I stay. He stays. I stayed.
I fall. He falls. I fell.
I cut. He cuts. I cut.
I reach. He reaches. I reached.
I remain. He remains. I remained.
I suggest. He suggests. I suggested.
I raise. He raises. I raised.
I pass. He passes. I passed.
I sell. He sells. I sold.
I require. He requires. I required.
I report. He reports. I reported.
I decide. He decides. I decided.
I pull. He pulls. I pulled.
"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GeometricConcept:
    """
    A concept with geometric properties.
    
    Mathematical representation:
    C = (p, f, r_vec, a_vec, t_vec)
    
    where:
    - p = mean position in [0, 1]
    - f = frequency count
    - r_vec = (initiator_count, mediator_count, receiver_count)
    - a_vec = action counts (what this concept does)
    - t_vec = target counts (what this concept acts upon)
    """
    word: str
    
    # Position statistics
    positions: List[float] = field(default_factory=list)
    sentence_count: int = 0
    
    # Role counts (the r_vec)
    initiator_count: int = 0
    mediator_count: int = 0
    receiver_count: int = 0
    
    # Relations
    actions: Counter = field(default_factory=Counter)  # a_vec
    targets: Counter = field(default_factory=Counter)  # t_vec
    
    @property
    def frequency(self) -> int:
        """Total occurrences."""
        return len(self.positions)
    
    @property
    def mean_position(self) -> float:
        """Mean position p̄(w) = (1/n) Σ p_i"""
        if not self.positions:
            return 0.5
        return sum(self.positions) / len(self.positions)
    
    @property
    def position_variance(self) -> float:
        """
        Position variance σ²(w) = (1/n) Σ (p_i - p̄)²
        
        High variance → appears everywhere (stop word candidate)
        Low variance → consistent position (content word)
        """
        if len(self.positions) < 2:
            return 0.0
        mean = self.mean_position
        return sum((p - mean) ** 2 for p in self.positions) / len(self.positions)
    
    @property
    def phi_direction(self) -> float:
        """
        φ-direction: measures if concept is primarily initiator or receiver.
        
        φ-dir(C) = (r_i - r_r) / (r_i + r_m + r_r + ε)
        
        > 0 → primarily initiator (subject-like)
        < 0 → primarily receiver (object-like)
        ≈ 0 → balanced or mediator
        """
        total = self.initiator_count + self.mediator_count + self.receiver_count
        if total == 0:
            return 0.0
        return (self.initiator_count - self.receiver_count) / total
    
    @property
    def phi_magnitude(self) -> float:
        """φ-magnitude: strength of the φ-direction."""
        total = self.initiator_count + self.mediator_count + self.receiver_count
        if total == 0:
            return 0.0
        return abs(self.initiator_count - self.receiver_count) / total
    
    @property
    def is_geometric_stop_word(self) -> bool:
        """
        Geometric stop word detection.
        
        A word is a stop word if:
        1. No semantic role (r_i + r_m + r_r = 0)
        2. OR: Short and frequent (len ≤ 4 and f ≥ 3)
        3. OR: Only receiver role and short (catches prepositions)
        """
        total_roles = self.initiator_count + self.mediator_count + self.receiver_count
        has_no_role = total_roles == 0
        is_short_frequent = len(self.word) <= 4 and self.frequency >= 3
        only_receiver = (self.receiver_count > 0 and 
                        self.initiator_count == 0 and 
                        self.mediator_count == 0 and
                        len(self.word) <= 5)
        return has_no_role or is_short_frequent or only_receiver
    
    @property
    def is_content_word(self) -> bool:
        """Inverse of stop word."""
        return not self.is_geometric_stop_word


@dataclass
class Frame:
    """
    Semantic frame: Initiator → Mediator → Receiver
    
    Extracted using position bands:
    - [0.0, 0.33) → Initiator
    - [0.33, 0.66) → Mediator  
    - [0.66, 1.0] → Receiver
    """
    initiator: str
    mediator: str
    receiver: Optional[str] = None
    source: str = ""
    text: str = ""


@dataclass
class VerbCluster:
    """A cluster of verb forms representing the same concept."""
    canonical: str
    forms: Dict[int, str] = field(default_factory=dict)
    
    def get_form(self, phase: int) -> str:
        return self.forms.get(phase, self.canonical)


# =============================================================================
# GEOMETRIC MORPHOLOGY
# =============================================================================

class GeometricMorphology:
    """
    Learn morphological equivalence from parallel structures.
    
    Key insight: Parallel sentences reveal morphological equivalence.
    "I love. He loves. I loved." → love ≡ loves ≡ loved
    """
    
    def __init__(self):
        self.words: Dict[str, Set[str]] = {}
        self.equivalence_classes: Dict[str, Set[str]] = {}
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _extract_mediator(self, sentence: str) -> Optional[str]:
        tokens = self._tokenize(sentence)
        if len(tokens) < 2:
            return None
        return tokens[1]
    
    def bootstrap(self, text: str):
        """Learn from parallel structure text."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_group: List[str] = []
        
        for sentence in sentences:
            mediator = self._extract_mediator(sentence)
            if not mediator:
                continue
            if mediator in {'will', 'would', 'could', 'should', 'may', 'might', 'can'}:
                continue
            
            current_group.append(mediator)
            
            if len(current_group) >= 3:
                self._create_equivalence(current_group)
                current_group = []
        
        if len(current_group) > 1:
            self._create_equivalence(current_group)
    
    def _create_equivalence(self, mediators: List[str]):
        filtered = [m for m in mediators 
                   if m not in {'will', 'would', 'could', 'should', 'may', 'might'}]
        
        if len(filtered) < 2:
            return
        
        canonical = filtered[0]
        equivalents = set(filtered)
        
        if canonical not in self.equivalence_classes:
            self.equivalence_classes[canonical] = set()
        
        self.equivalence_classes[canonical].update(equivalents)
        
        for word in filtered:
            if word not in self.words:
                self.words[word] = set()
            self.words[word].update(equivalents)
    
    def get_equivalents(self, word: str) -> Set[str]:
        return self.words.get(word.lower(), {word.lower()})
    
    def are_equivalent(self, word1: str, word2: str) -> bool:
        if word1.lower() == word2.lower():
            return True
        eq1 = self.get_equivalents(word1.lower())
        eq2 = self.get_equivalents(word2.lower())
        return bool(eq1 & eq2)
    
    def get_canonical(self, word: str) -> str:
        """Get canonical (base) form."""
        word = word.lower()
        equivalents = self.get_equivalents(word)
        if equivalents:
            # Return the shortest form as canonical
            return min(equivalents, key=len)
        return word


# =============================================================================
# GEOMETRIC CONJUGATION
# =============================================================================

class GeometricConjugation:
    """
    Learn verb conjugation from parallel structures.
    
    Position in parallel group encodes temporal phase:
    - Position 0: base form
    - Position 1: 3rd person singular
    - Position 2: past tense
    """
    
    def __init__(self):
        self.clusters: Dict[str, VerbCluster] = {}
        self.word_to_canonical: Dict[str, str] = {}
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _extract_mediator(self, sentence: str) -> Optional[str]:
        tokens = self._tokenize(sentence)
        return tokens[1] if len(tokens) > 1 else None
    
    def bootstrap(self, text: str):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_group: List[Tuple[str, int]] = []
        phase = 0
        
        for sentence in sentences:
            mediator = self._extract_mediator(sentence)
            if not mediator:
                continue
            if mediator in {'will', 'would', 'could', 'should', 'may', 'might', 'can'}:
                continue
            
            current_group.append((mediator, phase))
            phase += 1
            
            if phase >= 3:
                self._create_cluster(current_group)
                current_group = []
                phase = 0
        
        if len(current_group) > 1:
            self._create_cluster(current_group)
    
    def _create_cluster(self, group: List[Tuple[str, int]]):
        if not group:
            return
        
        canonical = group[0][0]
        cluster = VerbCluster(canonical=canonical)
        
        for mediator, phase in group:
            cluster.forms[phase] = mediator
            self.word_to_canonical[mediator] = canonical
        
        self.clusters[canonical] = cluster
    
    def get_canonical(self, word: str) -> str:
        return self.word_to_canonical.get(word.lower(), word.lower())
    
    # Irregular verb forms
    IRREGULAR_CONJUGATIONS = {
        # be forms
        'be': ('be', 'is', 'was'),
        'is': ('be', 'is', 'was'),
        'are': ('be', 'is', 'was'),
        'was': ('be', 'is', 'was'),
        'were': ('be', 'is', 'was'),
        'been': ('be', 'is', 'was'),
        # have forms
        'have': ('have', 'has', 'had'),
        'has': ('have', 'has', 'had'),
        'had': ('have', 'has', 'had'),
        # do forms
        'do': ('do', 'does', 'did'),
        'does': ('do', 'does', 'did'),
        'did': ('do', 'does', 'did'),
        # go forms
        'go': ('go', 'goes', 'went'),
        'goes': ('go', 'goes', 'went'),
        'went': ('go', 'goes', 'went'),
    }
    
    def conjugate(self, word: str, phase: int) -> str:
        """
        Conjugate word to given phase.
        Phase 0 = base, Phase 1 = 3rd singular, Phase 2 = past
        """
        w = word.lower()
        
        # Handle irregular verbs first
        if w in self.IRREGULAR_CONJUGATIONS:
            return self.IRREGULAR_CONJUGATIONS[w][phase]
        
        canonical = self.get_canonical(w)
        if canonical in self.clusters:
            return self.clusters[canonical].get_form(phase)
        
        # Repair truncated verbs (from bad extraction)
        base = self._repair_truncated_verb(w)
        
        # Apply conjugation rules
        if phase == 1:  # 3rd person singular
            if base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
                return base[:-1] + 'ies'
            elif base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
                return base + 'es'
            else:
                return base + 's'
        elif phase == 2:  # Past tense
            if base.endswith('e'):
                return base + 'd'
            elif base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
                return base[:-1] + 'ied'
            else:
                return base + 'ed'
        
        return base
    
    def _repair_truncated_verb(self, verb: str) -> str:
        """
        Repair truncated verbs that are missing their final 'e'.
        
        Common patterns from bad extraction:
        - emerg -> emerge
        - includ -> include
        - stat -> state
        """
        # Known truncated patterns that need 'e' added
        truncated_patterns = (
            'emerg', 'includ', 'provid', 'relat', 'creat', 'stat', 'debat',
            'locat', 'migrat', 'vibrat', 'generat', 'separ', 'demonstrat',
            'investigat', 'examin', 'observ', 'deduc', 'describ', 'explor',
            'produc', 'reduc', 'introduc', 'dominat', 'analyz', 'organiz',
            'recogniz', 'realiz', 'defin', 'combin', 'determin', 'imagin',
            'achiev', 'believ', 'receiv', 'perceiv', 'caus', 'paus', 'abus',
            'excus', 'diverg', 'converg', 'merg', 'purg', 'urg', 'surg',
            'involv', 'resolv', 'evolv', 'revolv', 'dissolv',
        )
        
        if verb.endswith(truncated_patterns):
            return verb + 'e'
        
        return verb


# =============================================================================
# GEOMETRIC KNOWLEDGE BASE
# =============================================================================

class GeometricKnowledge:
    """
    Fully geometric knowledge base.
    
    All components are geometric:
    1. Stop words detected by semantic role absence
    2. Frame slots assigned by position bands
    3. Morphology learned from parallel structures
    4. Conjugation learned from parallel structures
    """
    
    def __init__(self):
        self.concepts: Dict[str, GeometricConcept] = {}
        self.frames: List[Frame] = []
        self.total_sentences: int = 0
        self.entities: Dict[str, Dict] = {}  # For compatibility
        self.relations: Dict[str, Dict] = {}  # For compatibility
        
        # Initialize geometric morphology and conjugation
        self.morphology = GeometricMorphology()
        self.morphology.bootstrap(MORPHOLOGY_BOOTSTRAP)
        
        self.conjugation = GeometricConjugation()
        self.conjugation.bootstrap(MORPHOLOGY_BOOTSTRAP)
    
    def _get_or_create(self, word: str) -> GeometricConcept:
        word = word.lower()
        if word not in self.concepts:
            self.concepts[word] = GeometricConcept(word=word)
        return self.concepts[word]
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def learn(self, text: str, source: str = "", use_attention: bool = True):
        """
        Learn from text using geometric principles.
        
        Args:
            text: Text to learn from
            source: Source identifier
            use_attention: If True, use attention-based frame extraction (better for complex sentences)
        """
        sentences = re.split(r'[.!?]+', text)
        
        # Lazy-load attention extractor
        if use_attention and not hasattr(self, '_attention_extractor'):
            from .attention_extractor import AttentionExtractor
            self._attention_extractor = AttentionExtractor(self)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            self.total_sentences += 1
            tokens = self._tokenize(sentence)
            
            if len(tokens) < 2:
                continue
            
            # Record position statistics
            seen_in_sentence = set()
            for i, word in enumerate(tokens):
                if len(word) < 2:
                    continue
                
                c = self._get_or_create(word)
                pos = i / max(len(tokens) - 1, 1)
                c.positions.append(pos)
                
                if word not in seen_in_sentence:
                    c.sentence_count += 1
                    seen_in_sentence.add(word)
            
            # FRAME EXTRACTION
            initiator = None
            mediator = None
            receiver = None
            
            if use_attention:
                # Use attention-based extraction (better for complex sentences)
                frame_result = self._attention_extractor.extract_frame(sentence)
                if frame_result and frame_result.confidence > 0.3:
                    initiator = frame_result.initiator
                    mediator = frame_result.mediator
                    receiver = frame_result.receiver if frame_result.receiver else None
            
            # Fallback to position-based if attention fails or disabled
            if not initiator or not mediator:
                content_with_pos = []
                for i, w in enumerate(tokens):
                    if len(w) <= 3:
                        continue
                    if w.endswith('ly') and len(w) > 4:
                        continue
                    pos = i / max(len(tokens) - 1, 1)
                    content_with_pos.append((w, pos))
                
                if len(content_with_pos) < 2:
                    continue
                
                # Assign slots by position bands
                for word, pos in content_with_pos:
                    if pos < 0.33 and initiator is None:
                        initiator = word
                    elif pos < 0.66 and mediator is None:
                        mediator = word
                    elif receiver is None:
                        receiver = word
                
                if initiator is None and content_with_pos:
                    initiator = content_with_pos[0][0]
                if mediator is None and len(content_with_pos) > 1:
                    mediator = content_with_pos[1][0]
            
            if not initiator or not mediator:
                continue
            
            # Create frame
            frame = Frame(
                initiator=initiator, 
                mediator=mediator, 
                receiver=receiver,
                source=source,
                text=sentence
            )
            self.frames.append(frame)
            
            # Update role counts
            init_c = self._get_or_create(initiator)
            init_c.initiator_count += 1
            init_c.actions[mediator] += 1
            if receiver:
                init_c.targets[receiver] += 1
            
            med_c = self._get_or_create(mediator)
            med_c.mediator_count += 1
            
            if receiver:
                recv_c = self._get_or_create(receiver)
                recv_c.receiver_count += 1
            
            # Update entities dict for compatibility
            if initiator not in self.entities:
                self.entities[initiator] = {'actions': [], 'source': source}
            if mediator not in self.entities[initiator]['actions']:
                self.entities[initiator]['actions'].append(mediator)
    
    def encode(self, text: str) -> float:
        """Encode text to φ-space position."""
        tokens = self._tokenize(text)
        
        total = 0.0
        weight = 0.0
        
        for word in tokens:
            if word not in self.concepts:
                continue
            
            c = self.concepts[word]
            if c.is_geometric_stop_word:
                continue
            
            w = 1.0 / math.log(c.frequency + 2)
            total += c.mean_position * w
            weight += w
        
        return total / weight if weight > 0 else 0.5
    
    def query_by_entity(self, entity: str, k: int = 10) -> List[Dict]:
        """Query frames by entity."""
        entity = entity.lower()
        results = []
        
        for frame in self.frames:
            if frame.initiator == entity or frame.receiver == entity:
                results.append({
                    'agent': frame.initiator,
                    'action': frame.mediator,
                    'patient': frame.receiver,
                    'source': frame.source,
                    'text': frame.text,
                })
        
        return results[:k]
    
    def get_entity_info(self, entity: str) -> Optional[Dict]:
        """Get info about an entity."""
        entity = entity.lower()
        if entity in self.entities:
            return self.entities[entity]
        if entity in self.concepts:
            c = self.concepts[entity]
            return {
                'actions': list(c.actions.keys()),
                'targets': list(c.targets.keys()),
                'source': 'geometric',
            }
        return None


# =============================================================================
# GEOMETRIC Q&A
# =============================================================================

class GeometricQA:
    """
    Geometric Question-Answering system.
    
    Uses geometric principles for:
    - Question type detection
    - Entity matching (with morphological equivalence)
    - Response generation (with geometric conjugation)
    """
    
    def __init__(self):
        self.knowledge = GeometricKnowledge()
        
        # Dial settings for response style
        self.style_x = 0.0
        self.perspective_y = 0.0
        self.depth_z = 0.0
        self.certainty_w = 0.0
    
    def load_corpus(self, path: str) -> int:
        """Load corpus from JSON file."""
        import json
        from pathlib import Path
        
        corpus_path = Path(path)
        if not corpus_path.exists():
            return 0
        
        with open(corpus_path) as f:
            data = json.load(f)
        
        count = 0
        
        # Handle different corpus formats
        if isinstance(data, dict) and 'frames' in data:
            # Format: {"frames": [{"agent": ..., "text": ..., "source": ..., "count": N}, ...]}
            for item in data['frames']:
                if isinstance(item, dict):
                    text = item.get('text', '')
                    source = item.get('source', '')
                    # Support count field for deduplicated corpus
                    frame_count = item.get('count', 1)
                    if text:
                        for _ in range(frame_count):
                            self.knowledge.learn(text, source)
                        count += frame_count
        elif isinstance(data, list):
            # Format: [{"text": ..., "source": ..., "count": N}, ...]
            for item in data:
                if isinstance(item, dict):
                    text = item.get('text', '')
                    source = item.get('source', '')
                    frame_count = item.get('count', 1)
                    if text:
                        for _ in range(frame_count):
                            self.knowledge.learn(text, source)
                        count += frame_count
        
        # Generate Q&A pairs from loaded knowledge (templates emerge!)
        if hasattr(self, 'template_projector'):
            self._generate_qa_from_frames()
        
        return count
    
    def set_style(self, x: float):
        self.style_x = max(-1, min(1, x))
    
    def set_perspective(self, y: float):
        self.perspective_y = max(-1, min(1, y))
    
    def set_depth(self, z: float):
        self.depth_z = max(-1, min(1, z))
    
    def set_certainty(self, w: float):
        self.certainty_w = max(-1, min(1, w))
    
    def set_output_lens(self, lens_name: str = "natural"):
        """
        Set the output lens for more natural language output.
        
        Available lenses: natural, formal, casual, literary, scientific
        Set to None to disable the lens.
        """
        if lens_name is None:
            self._output_lens = None
        else:
            from .output_lens import OutputProjector, LENSES
            if lens_name not in LENSES:
                raise ValueError(f"Unknown lens: {lens_name}. Available: {list(LENSES.keys())}")
            self._output_lens = OutputProjector(LENSES[lens_name])
    
    def ask(self, query: str) -> str:
        """Answer a question."""
        result = self.ask_detailed(query)
        if result['answers']:
            raw_answer = result['answers'][0]['answer']
            # Apply output lens if set
            if hasattr(self, '_output_lens') and self._output_lens:
                answer_type = result.get('axis', 'describe').lower()
                if answer_type == 'who':
                    answer_type = 'who'
                elif answer_type == 'what':
                    answer_type = 'what'
                else:
                    answer_type = 'describe'
                return self._output_lens.project(raw_answer, answer_type)
            return raw_answer
        return "I don't have information about that."
    
    def ask_detailed(self, query: str) -> Dict:
        """Answer with detailed information."""
        tokens = self.knowledge._tokenize(query)
        
        # Detect question type
        axis = self._detect_axis(tokens)
        
        # Find content words using geometric morphology
        content = []
        for w in tokens:
            if w in self.knowledge.concepts and self.knowledge.concepts[w].is_content_word:
                content.append(w)
            else:
                for name, c in self.knowledge.concepts.items():
                    if c.is_content_word and self.knowledge.morphology.are_equivalent(name, w):
                        content.append(name)
                        break
        
        # Find entity and action
        entity = None
        action = None
        
        for word in content:
            if word not in self.knowledge.concepts:
                continue
            c = self.knowledge.concepts[word]
            if c.phi_direction > 0.3:
                entity = word
            elif c.mediator_count > 0:
                action = word
        
        # Check for action in query
        if not action:
            for w in tokens:
                equivalents = self.knowledge.morphology.get_equivalents(w)
                for name, c in self.knowledge.concepts.items():
                    if c.mediator_count > 0:
                        if name in equivalents or self.knowledge.morphology.are_equivalent(name, w):
                            action = name
                            break
                if action:
                    break
        
        # Generate response
        answers = []
        
        if axis == 'WHO' and action:
            answer = self._who_does(action)
            answers.append({
                'answer': answer,
                'confidence': 0.8,
                'source': 'geometric',
                'frame_count': 1,
            })
        elif axis == 'ACTION' and entity:
            answer = self._what_does(entity)
            answers.append({
                'answer': answer,
                'confidence': 0.8,
                'source': 'geometric',
                'frame_count': 1,
            })
        elif entity:
            answer = self._describe(entity)
            answers.append({
                'answer': answer,
                'confidence': 0.7,
                'source': 'geometric',
                'frame_count': 1,
            })
        elif content:
            answer = self._describe(content[0])
            answers.append({
                'answer': answer,
                'confidence': 0.5,
                'source': 'geometric',
                'frame_count': 1,
            })
        
        return {
            'axis': axis,
            'entity': entity,
            'action': action,
            'answers': answers,
        }
    
    def _detect_axis(self, tokens: List[str]) -> str:
        """Detect question type geometrically."""
        first_words = tokens[:3] if len(tokens) >= 3 else tokens
        
        for word in first_words:
            if word in {'who', 'whom', 'whose'}:
                return 'WHO'
            if word in {'what', 'which'}:
                return 'WHAT'
            if word in {'where'}:
                return 'WHERE'
            if word in {'when'}:
                return 'WHEN'
            if word in {'why'}:
                return 'WHY'
            if word in {'how'}:
                return 'HOW'
            if word in {'did', 'does', 'do'}:
                return 'ACTION'
        
        # Check for "What does X do?" pattern
        if 'does' in tokens or 'do' in tokens:
            return 'ACTION'
        
        return 'WHO'  # Default
    
    def _who_does(self, action: str) -> str:
        """Find who performs an action."""
        equivalents = self.knowledge.morphology.get_equivalents(action)
        
        actors = []
        for name, c in self.knowledge.concepts.items():
            if c.initiator_count == 0 or not c.is_content_word:
                continue
            for act in c.actions:
                if act in equivalents or self.knowledge.morphology.are_equivalent(act, action):
                    actors.append((name, c.actions[act]))
                    break
        
        if not actors:
            for frame in self.knowledge.frames:
                if self.knowledge.morphology.are_equivalent(frame.mediator, action):
                    if frame.initiator in self.knowledge.concepts:
                        c = self.knowledge.concepts[frame.initiator]
                        if c.is_content_word:
                            actors.append((frame.initiator, 1))
        
        if not actors:
            return f"I don't know who {action}s."
        
        actors.sort(key=lambda x: x[1], reverse=True)
        actor = actors[0][0]
        
        # Find target
        target = None
        for frame in self.knowledge.frames:
            if frame.initiator == actor and self.knowledge.morphology.are_equivalent(frame.mediator, action):
                if frame.receiver:
                    target = frame.receiver
                    break
        
        # Conjugate using geometric conjugation
        canonical = self.knowledge.conjugation.get_canonical(action)
        verb = self.knowledge.conjugation.conjugate(canonical, 1)
        
        if target:
            return f"{actor.title()} {verb} {target}."
        else:
            return f"{actor.title()} {verb}."
    
    def _what_does(self, entity: str) -> str:
        """Describe what an entity does."""
        if entity not in self.knowledge.concepts:
            return f"I don't know about {entity}."
        
        c = self.knowledge.concepts[entity]
        
        if c.actions:
            top_actions = c.actions.most_common(3)
            verbs = []
            for a, _ in top_actions:
                canonical = self.knowledge.conjugation.get_canonical(a)
                verb = self.knowledge.conjugation.conjugate(canonical, 1)
                verbs.append(verb)
            
            if len(verbs) == 1:
                action_desc = verbs[0]
            elif len(verbs) == 2:
                action_desc = f"{verbs[0]} and {verbs[1]}"
            else:
                action_desc = f"{', '.join(verbs[:-1])}, and {verbs[-1]}"
            
            return f"{entity.title()} {action_desc}."
        
        return f"I don't know what {entity} does."
    
    def _describe(self, entity: str) -> str:
        """Describe an entity."""
        if entity not in self.knowledge.concepts:
            return f"I don't know about {entity}."
        
        c = self.knowledge.concepts[entity]
        
        # Role: first check targets for category words (from "X is a Y" frames)
        # Only use if count >= 3 to avoid incidental mentions
        category_words = {'detective', 'doctor', 'scientist', 'teacher', 'writer',
                         'philosopher', 'artist', 'leader', 'hero', 'villain',
                         'science', 'field', 'discipline', 'study', 'branch',
                         'person', 'character', 'figure', 'companion', 'assistant'}
        
        role = None
        if c.targets:
            for target, count in c.targets.most_common(10):
                if target in category_words and count >= 3:  # Require multiple attestations
                    role = target
                    break
        
        # Fallback to φ-direction based role
        if not role:
            if c.phi_direction > 0.3:
                role = "protagonist"
            elif c.phi_direction < -0.3:
                role = "concept"
            else:
                role = "entity"
        
        # Actions using geometric conjugation
        if c.actions:
            top_actions = c.actions.most_common(3)
            verbs = []
            for a, _ in top_actions:
                # Skip non-verb words that might appear as actions
                if a in {'is', 'doctor', 'detective', 'science', 'field', 'cases', 'case', 
                         'holmes', 'watson', 'matter', 'energy', 'crimes', 'mysteries'}:
                    continue
                # If already ends in 's' (3rd person), use as-is to avoid double conjugation
                if a.endswith('s') and not a.endswith('ss'):
                    verbs.append(a)
                else:
                    canonical = self.knowledge.conjugation.get_canonical(a)
                    verb = self.knowledge.conjugation.conjugate(canonical, 1)
                    verbs.append(verb)
                # Limit to 3 verbs
                if len(verbs) >= 3:
                    break
            
            if len(verbs) == 0:
                response = f"{entity.title()} is a {role}"
            elif len(verbs) == 1:
                action_desc = verbs[0]
                response = f"{entity.title()} is a {role} who {action_desc}"
            elif len(verbs) == 2:
                action_desc = f"{verbs[0]} and {verbs[1]}"
                response = f"{entity.title()} is a {role} who {action_desc}"
            else:
                action_desc = f"{', '.join(verbs[:-1])}, and {verbs[-1]}"
                response = f"{entity.title()} is a {role} who {action_desc}"
        else:
            response = f"{entity.title()} is a {role}"
        
        # Targets (use most common, not insertion order)
        if c.targets:
            good_targets = []
            for t, _ in c.targets.most_common(10):
                if t in self.knowledge.concepts and self.knowledge.concepts[t].is_content_word:
                    good_targets.append(t)
                    if len(good_targets) >= 2:
                        break
            if good_targets:
                response += f", often involving {' and '.join(good_targets)}"
        
        return response + "."


# =============================================================================
# HOLOGRAPHIC ENHANCED QA
# =============================================================================

class HolographicGeometricQA(GeometricQA):
    """
    GeometricQA enhanced with holographic template projection and semantic quaternions.
    
    Instead of hard-coded response templates, this version:
    1. Learns response patterns from Q&A examples
    2. Projects templates dynamically via holographic interference
    3. Fills slots using geometric knowledge
    4. Uses semantic quaternions for concept similarity and analogies
    
    This combines:
    - Geometric understanding (position-based frames, morphology)
    - Holographic generation (interference-based templates)
    - Semantic quaternions (4D concept encoding for analogies)
    
    The Two Quaternions:
    - φ-dial (output): Style, Perspective, Depth, Certainty
    - Semantic (encoding): Gender, Age, Agency (φ-direction), Animacy
    """
    
    def __init__(self):
        super().__init__()
        
        # Import holographic templates
        from .holographic_templates import (
            HolographicTemplateProjector,
            HolographicResponseSynthesizer
        )
        
        # Import semantic quaternion
        from .semantic_quaternion import SemanticQuaternionNavigator
        
        self.template_projector = HolographicTemplateProjector(self.knowledge)
        self.response_synthesizer = HolographicResponseSynthesizer()
        self.semantic_navigator = SemanticQuaternionNavigator(self.knowledge)
        
        # Seed with default Q&A patterns
        self._seed_default_patterns()
    
    def _seed_default_patterns(self):
        """Seed with default Q&A patterns for template learning."""
        default_qa = [
            # WHO IS patterns
            ("Who is Watson?", "Watson is a loyal doctor who assists Holmes."),
            ("Who is Darcy?", "Darcy is a proud gentleman who loves Elizabeth."),
            ("Who is Moriarty?", "Moriarty is a cunning villain who opposes Holmes."),
            ("Who is Elizabeth?", "Elizabeth is a witty lady who challenges Darcy."),
            ("Who is Lestrade?", "Lestrade is a dedicated inspector who consults Holmes."),
            
            # WHAT DID patterns
            ("What did Holmes do?", "Holmes investigated the mysterious case carefully."),
            ("What did Watson do?", "Watson documented the important findings thoroughly."),
            ("What did Darcy do?", "Darcy proposed to Elizabeth unexpectedly."),
            
            # WHERE patterns
            ("Where is Holmes?", "Holmes is located in London."),
            ("Where is Darcy?", "Darcy is located in Derbyshire."),
            
            # WHY patterns
            ("Why did Holmes investigate?", "Holmes investigated because he noticed suspicious evidence."),
            ("Why did Darcy propose?", "Darcy proposed because he loved Elizabeth deeply."),
            
            # HOW patterns
            ("How did Holmes solve it?", "Holmes solved it by examining the evidence carefully."),
            ("How did Watson help?", "Watson helped by documenting the observations precisely."),
        ]
        
        self.template_projector.add_qa_pairs_from_corpus(default_qa)
        
        # Note: Q&A pairs from frames are generated after load_corpus() is called
        # via refresh_templates() - templates emerge from knowledge!
    
    def _generate_qa_from_frames(self, max_pairs: int = 50):
        """
        Generate Q&A pairs from frames to feed the template projector.
        
        This is how templates EMERGE from knowledge:
        - Frames encode relationships (initiator -> mediator -> receiver)
        - We generate natural Q&A pairs from these relationships
        - The holographic projector learns patterns via interference
        - Templates emerge as structure words align and content words become slots
        """
        # Skip function words and short words
        skip_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'he', 'she',
                      'they', 'we', 'you', 'i', 'is', 'are', 'was', 'were', 'be', 'been',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'can', 'and', 'or', 'but', 'if',
                      'then', 'so', 'because', 'while', 'although', 'however', 'therefore',
                      'further', 'unlike', 'such', 'each', 'every', 'some', 'any', 'all',
                      's', 't', 'd', 'm', 're', 've', 'll'}
        
        # Find high-agency entities (good subjects for WHO questions)
        entities = []
        for name, concept in self.knowledge.concepts.items():
            # Skip function words and short words
            if name in skip_words or len(name) < 4:
                continue
            # Need some initiator activity and actions
            if concept.initiator_count >= 2 and len(concept.actions) > 0:
                entities.append((name, concept))
        
        # Sort by initiator count (most active first)
        entities.sort(key=lambda x: x[1].initiator_count, reverse=True)
        
        qa_pairs = []
        
        for name, concept in entities[:max_pairs // 3]:
            # WHO IS pattern
            actions = list(concept.actions.keys())[:2]
            targets = list(concept.targets.keys())[:1]
            
            if actions:
                action_str = actions[0]
                if len(actions) > 1:
                    action_str = f"{actions[0]} and {actions[1]}"
                
                # Infer role from phi-direction
                if concept.phi_direction > 0.5:
                    role = "protagonist"
                elif concept.phi_direction > 0.2:
                    role = "character"
                else:
                    role = "figure"
                
                answer = f"{name.title()} is a {role} who {action_str}"
                if targets:
                    answer += f" {targets[0]}"
                answer += "."
                
                qa_pairs.append((f"Who is {name.title()}?", answer))
            
            # WHAT DOES pattern
            if actions and targets:
                qa_pairs.append(
                    (f"What does {name.title()} do?", 
                     f"{name.title()} {actions[0]} {targets[0]}.")
                )
        
        # Add the generated pairs
        for q, a in qa_pairs:
            self.template_projector.add_qa_pair(q, a)
    
    def refresh_templates(self):
        """
        Refresh templates from current knowledge.
        
        Call this after adding significant new knowledge to
        regenerate Q&A pairs and allow new templates to emerge.
        """
        # Clear existing generated pairs (keep default seeds)
        self.template_projector.qa_pairs = self.template_projector.qa_pairs[:14]  # Keep seeds
        self.template_projector.template_cache.clear()
        
        # Regenerate from frames
        self._generate_qa_from_frames()
    
    def add_qa_example(self, question: str, answer: str):
        """Add a Q&A example for template learning."""
        self.template_projector.add_qa_pair(question, answer)
    
    def ask(self, query: str) -> str:
        """Answer using holographic template projection."""
        result = self.ask_detailed(query)
        if result['answers']:
            return result['answers'][0]['answer']
        return "I don't have information about that."
    
    def ask_detailed(self, query: str) -> Dict:
        """Answer with detailed information using holographic templates."""
        tokens = self.knowledge._tokenize(query)
        
        # Detect question type
        axis = self._detect_axis(tokens)
        
        # Find content words using geometric morphology
        content = []
        for w in tokens:
            if w in self.knowledge.concepts and self.knowledge.concepts[w].is_content_word:
                content.append(w)
            else:
                for name, c in self.knowledge.concepts.items():
                    if c.is_content_word and self.knowledge.morphology.are_equivalent(name, w):
                        content.append(name)
                        break
        
        # Find entity and action
        entity = None
        action = None
        
        for word in content:
            if word not in self.knowledge.concepts:
                continue
            c = self.knowledge.concepts[word]
            # Entity detection: positive phi_direction OR high initiator count
            if c.phi_direction > 0.1 or c.initiator_count >= 5:
                entity = word
            elif c.mediator_count > 0:
                action = word
        
        # Check for action in query
        if not action:
            for w in tokens:
                equivalents = self.knowledge.morphology.get_equivalents(w)
                for name, c in self.knowledge.concepts.items():
                    if c.mediator_count > 0:
                        if name in equivalents or self.knowledge.morphology.are_equivalent(name, w):
                            action = name
                            break
                if action:
                    break
        
        # Generate response using holographic templates
        answers = []
        
        # Get slot values from geometric knowledge
        slot_values = self._get_slot_values(entity, action, axis)
        
        # Project template and fill
        template = self.template_projector.project_template(query)
        
        if template.confidence > 0.2 and slot_values:
            # Use holographic template
            response = self.template_projector.fill_template(
                template, 
                entity or content[0] if content else "it",
                slot_values
            )
            answers.append({
                'answer': response,
                'confidence': template.confidence,
                'source': 'holographic',
                'template': template.pattern,
            })
        else:
            # Fallback to geometric generation
            if axis == 'WHO' and action:
                answer = self._who_does(action)
            elif axis == 'ACTION' and entity:
                answer = self._what_does(entity)
            elif entity:
                answer = self._describe(entity)
            elif content:
                answer = self._describe(content[0])
            else:
                answer = "I don't have information about that."
            
            answers.append({
                'answer': answer,
                'confidence': 0.5,
                'source': 'geometric',
            })
        
        return {
            'axis': axis,
            'entity': entity,
            'action': action,
            'answers': answers,
            'template': template.pattern if template else None,
        }
    
    def _get_slot_values(self, entity: str, action: str, axis: str) -> Dict[str, str]:
        """Get slot values from geometric knowledge."""
        values = {}
        
        if entity and entity in self.knowledge.concepts:
            c = self.knowledge.concepts[entity]
            
            # Role: first check if concept has a category from "X is a Y" frames
            # These show up as targets when mediator is 'be' or 'is'
            # Only use if count >= 3 to avoid incidental mentions
            category_words = {'detective', 'doctor', 'scientist', 'teacher', 'writer',
                             'philosopher', 'artist', 'leader', 'hero', 'villain',
                             'science', 'field', 'discipline', 'study', 'branch',
                             'person', 'character', 'figure', 'companion', 'assistant'}
            
            # Look for category in top targets
            role_from_target = None
            if c.targets:
                for target, count in c.targets.most_common(10):
                    if target in category_words and count >= 3:  # Require multiple attestations
                        role_from_target = target
                        break
            
            if role_from_target:
                values['role'] = role_from_target
            elif c.phi_direction > 0.5:
                values['role'] = 'protagonist'
            elif c.phi_direction > 0.3:
                values['role'] = 'character'
            elif c.phi_direction < -0.3:
                values['role'] = 'concept'
            else:
                values['role'] = 'entity'
            
            # Adjective from common co-occurrences (simplified)
            if c.initiator_count > 3:
                values['adjective'] = 'notable'
            elif c.actions:
                # Infer adjective from actions
                top_action = c.actions.most_common(1)
                if top_action:
                    act = top_action[0][0]
                    # Simple mapping
                    if act in {'investigate', 'examine', 'deduce', 'observe'}:
                        values['adjective'] = 'analytical'
                    elif act in {'love', 'care', 'help', 'assist'}:
                        values['adjective'] = 'caring'
                    elif act in {'oppose', 'challenge', 'fight'}:
                        values['adjective'] = 'determined'
                    else:
                        values['adjective'] = 'notable'
            
            # Action from top actions
            if c.actions:
                top_actions = c.actions.most_common(3)
                verbs = []
                for a, _ in top_actions:
                    canonical = self.knowledge.conjugation.get_canonical(a)
                    verb = self.knowledge.conjugation.conjugate(canonical, 1)
                    verbs.append(verb)
                
                if len(verbs) == 1:
                    values['action'] = verbs[0]
                elif len(verbs) == 2:
                    values['action'] = f"{verbs[0]} and {verbs[1]}"
                else:
                    values['action'] = f"{', '.join(verbs[:-1])}, and {verbs[-1]}"
            
            # Target from top targets
            if c.targets:
                good_targets = [t for t in c.targets.keys() 
                              if t in self.knowledge.concepts 
                              and self.knowledge.concepts[t].is_content_word][:2]
                if good_targets:
                    values['target'] = ' and '.join(good_targets)
        
        return values
    
    def synthesize_from_sources(self, query: str, sources: List[str]) -> str:
        """
        Synthesize a response from multiple source texts.
        
        Uses holographic interference to find common elements.
        """
        return self.response_synthesizer.synthesize_with_structure(query, sources)
    
    # =========================================================================
    # SEMANTIC QUATERNION METHODS (with Lens Support)
    # =========================================================================
    
    def complete_analogy(self, a: str, b: str, c: str, k: int = 5, 
                         lens: str = "intrinsic_priority") -> List[Tuple[str, float]]:
        """
        Complete analogy: A is to B as C is to ?
        
        Uses lens-aware analogy solving:
        - "intrinsic": Semantic quaternion arithmetic (gender, age, agency, animacy)
        - "behavioral": φ-direction based matching
        - "relational": Connection/target based matching
        - "intrinsic_priority": Prefer intrinsic if high confidence, else weighted
        - "weighted": Combine all lenses with weights
        
        Example:
            complete_analogy("king", "queen", "man") -> [("woman", 1.0), ...]
        """
        # Try lens-aware approach first
        if lens in ("intrinsic_priority", "weighted", "behavioral", "relational"):
            result = self._lens_analogy(a, b, c, k, lens)
            if result:
                return result
        
        # Fallback to pure semantic quaternion
        return self.semantic_navigator.complete_analogy(a, b, c, k)
    
    def _lens_analogy(self, a: str, b: str, c: str, k: int, lens: str) -> List[Tuple[str, float]]:
        """
        Lens-aware analogy solving.
        
        Different lenses reveal different aspects of the analogy:
        - INTRINSIC: What has C's properties shifted by A→B delta?
        - BEHAVIORAL: What acts like C at the expected φ?
        - RELATIONAL: What connects to C like B connects to A?
        """
        from .semantic_quaternion import SemanticQuaternion, DEFAULT_SEMANTIC_FEATURES
        
        results = []
        
        # INTRINSIC lens: semantic quaternion arithmetic
        sq_a = DEFAULT_SEMANTIC_FEATURES.get(a.lower())
        sq_b = DEFAULT_SEMANTIC_FEATURES.get(b.lower())
        sq_c = DEFAULT_SEMANTIC_FEATURES.get(c.lower())
        
        intrinsic_results = []
        if sq_a and sq_b and sq_c:
            delta = sq_b - sq_a
            expected = sq_c + delta
            
            for word, sq in DEFAULT_SEMANTIC_FEATURES.items():
                if word in {a.lower(), b.lower(), c.lower()}:
                    continue
                distance = expected.distance(sq)
                if distance < 2.0:
                    similarity = 1.0 / (1.0 + distance)
                    intrinsic_results.append((word, similarity, "intrinsic"))
            
            intrinsic_results.sort(key=lambda x: -x[1])
        
        # If intrinsic_priority and we have high-confidence intrinsic match, use it
        if lens == "intrinsic_priority" and intrinsic_results:
            if intrinsic_results[0][1] >= 0.9:
                return [(w, s) for w, s, _ in intrinsic_results[:k]]
        
        # BEHAVIORAL lens: φ-direction matching
        behavioral_results = []
        concept_a = self.knowledge.concepts.get(a.lower())
        concept_b = self.knowledge.concepts.get(b.lower())
        concept_c = self.knowledge.concepts.get(c.lower())
        
        if concept_a and concept_b and concept_c:
            phi_delta = concept_b.phi_direction - concept_a.phi_direction
            expected_phi = concept_c.phi_direction + phi_delta
            expected_phi = max(-1, min(1, expected_phi))
            
            for word, concept in self.knowledge.concepts.items():
                if word in {a.lower(), b.lower(), c.lower()}:
                    continue
                if not concept.is_content_word:
                    continue
                
                phi_diff = abs(concept.phi_direction - expected_phi)
                if phi_diff < 0.5:
                    score = 1.0 - phi_diff
                    behavioral_results.append((word, score, "behavioral"))
            
            behavioral_results.sort(key=lambda x: -x[1])
        
        # RELATIONAL lens: target/connection matching
        relational_results = []
        if concept_c and concept_c.targets:
            for target, count in concept_c.targets.most_common(10):
                if target in {a.lower(), b.lower(), c.lower()}:
                    continue
                if target in self.knowledge.concepts:
                    score = min(1.0, count / 3)
                    relational_results.append((target, score, "relational"))
        
        # Combine based on lens strategy
        if lens == "behavioral":
            return [(w, s) for w, s, _ in behavioral_results[:k]]
        elif lens == "relational":
            return [(w, s) for w, s, _ in relational_results[:k]]
        elif lens in ("weighted", "intrinsic_priority"):
            # Weighted combination
            combined = {}
            weights = {"intrinsic": 1.5, "behavioral": 0.8, "relational": 0.7}
            
            for word, score, source in intrinsic_results[:k*2]:
                if word not in combined:
                    combined[word] = 0
                combined[word] += score * weights[source]
            
            for word, score, source in behavioral_results[:k*2]:
                if word not in combined:
                    combined[word] = 0
                combined[word] += score * weights[source]
            
            for word, score, source in relational_results[:k*2]:
                if word not in combined:
                    combined[word] = 0
                combined[word] += score * weights[source]
            
            sorted_combined = sorted(combined.items(), key=lambda x: -x[1])
            return sorted_combined[:k]
        
        # Default: intrinsic only
        return [(w, s) for w, s, _ in intrinsic_results[:k]]
    
    def semantic_similarity(self, word1: str, word2: str) -> float:
        """
        Compute semantic similarity between two concepts.
        
        Uses quaternion cosine similarity in 4D semantic space.
        """
        return self.semantic_navigator.similarity(word1, word2)
    
    def find_similar_relations(self, a: str, b: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Find pairs with similar relations to A:B.
        
        Example:
            find_similar_relations("king", "queen") -> 
                [("man", "woman", 1.0), ("actor", "actress", 1.0), ...]
        """
        return self.semantic_navigator.find_similar_relations(a, b, k)
    
    def add_semantic_concept(self, word: str, gender: float = 0.0, 
                             age: float = 0.0, animacy: float = 0.0):
        """
        Add a concept with semantic features.
        
        Args:
            word: The concept word
            gender: -1 (female) to +1 (male)
            age: -1 (young) to +1 (adult)
            animacy: -1 (abstract/place) to +1 (human)
        
        Note: Agency (z-axis) is automatically set from φ-direction if known.
        """
        from .semantic_quaternion import SemanticQuaternion
        
        # Get agency from geometric knowledge if available
        agency = 0.0
        if word.lower() in self.knowledge.concepts:
            agency = self.knowledge.concepts[word.lower()].phi_direction
        
        q = SemanticQuaternion(x=gender, y=age, z=agency, w=animacy)
        self.semantic_navigator.add_concept(word, q)
    
    # =========================================================================
    # GEODESIC GENERATION METHODS
    # =========================================================================
    
    def generate_about(self, concept: str, num_sentences: int = 3, style: str = None) -> str:
        """
        Generate free-form text about a concept using geodesic navigation.
        
        Instead of template-based generation, this navigates through
        concept space along geodesic paths, generating sentences at
        each waypoint.
        
        Args:
            concept: The concept to describe
            num_sentences: Number of sentences to generate
            style: Optional style preset ('hemingway', 'academic', 'journalistic', etc.)
        
        Returns:
            Free-form text about the concept
        """
        from .geodesic_generator import GeodesicGenerator
        
        if not hasattr(self, '_geodesic_generator'):
            self._geodesic_generator = GeodesicGenerator(self.knowledge)
        
        raw_text = self._geodesic_generator.generate_about(concept, num_sentences)
        
        # Apply style projection if requested
        if style:
            from .style_projector import StyleProjector
            if not hasattr(self, '_style_projector'):
                self._style_projector = StyleProjector(self.knowledge)
            self._style_projector.set_style(style)
            return self._style_projector.project(raw_text, concept)
        
        return raw_text
    
    def generate_story(self, concepts: List[str], max_sentences: int = 5, style: str = None) -> str:
        """
        Generate a short narrative connecting multiple concepts.
        
        Finds geodesic paths between concepts and generates
        sentences along the way.
        
        Args:
            concepts: List of concepts to connect
            max_sentences: Maximum sentences to generate
            style: Optional style preset ('hemingway', 'academic', 'journalistic', etc.)
        
        Returns:
            A coherent narrative connecting the concepts
        """
        from .geodesic_generator import GeodesicGenerator
        
        if not hasattr(self, '_geodesic_generator'):
            self._geodesic_generator = GeodesicGenerator(self.knowledge)
        
        raw_text = self._geodesic_generator.generate_story(concepts, max_sentences)
        
        # Apply style projection if requested
        if style:
            from .style_projector import StyleProjector
            if not hasattr(self, '_style_projector'):
                self._style_projector = StyleProjector(self.knowledge)
            self._style_projector.set_style(style)
            return self._style_projector.project(raw_text, concepts[0] if concepts else None)
        
        return raw_text
    
    def generate_comparison(self, concept_a: str, concept_b: str) -> str:
        """
        Generate text comparing two concepts.
        
        Args:
            concept_a: First concept
            concept_b: Second concept
        
        Returns:
            Comparison text
        """
        from .geodesic_generator import GeodesicGenerator
        
        if not hasattr(self, '_geodesic_generator'):
            self._geodesic_generator = GeodesicGenerator(self.knowledge)
        
        return self._geodesic_generator.generate_comparison(concept_a, concept_b)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PHI',
    'MORPHOLOGY_BOOTSTRAP',
    'GeometricConcept',
    'Frame',
    'VerbCluster',
    'GeometricMorphology',
    'GeometricConjugation',
    'GeometricKnowledge',
    'GeometricQA',
    'HolographicGeometricQA',
]
