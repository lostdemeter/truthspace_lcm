#!/usr/bin/env python3
"""
Attention-Based Frame Extractor

Improves frame extraction using spatial attention and hypothesis navigation
principles for better subject detection.

The Problem:
    Simple position-based extraction fails on complex sentences:
    "Recent advances in physics have transformed our understanding"
    → Position-based: I:recent M:advances R:physics (WRONG)
    → Should be: I:physics M:transformed R:understanding (or similar)

The Solution:
    Use bidirectional attention to find the TRUE subject:
    1. AGENCY HYPOTHESIS: The subject should have high agency (acts, not acted upon)
    2. VERB AGREEMENT: The subject agrees with the main verb
    3. SEMANTIC WEIGHT: Content words > function words (Zipf inverse)
    4. SYNTACTIC SIGNALS: Capitalization, position, determiners

This is "Tachyon extraction" - we hypothesize what the subject SHOULD be,
then gather evidence to confirm/refute.

Author: Lesley Gushurst
License: GPLv3
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
from collections import Counter
import math


PHI = 1.618034


@dataclass
class SubjectHypothesis:
    """A hypothesis about what the subject is."""
    word: str
    position: int
    evidence: List[Tuple[str, float]]  # (reason, weight)
    total_score: float = 0.0
    
    def add_evidence(self, reason: str, weight: float):
        self.evidence.append((reason, weight))
        self.total_score += weight


@dataclass 
class ExtractedFrame:
    """An extracted frame with confidence."""
    initiator: str
    mediator: str  # Stored in BASE FORM for clean generation
    receiver: str
    confidence: float
    source_sentence: str
    extraction_method: str


class AttentionExtractor:
    """
    Extract frames using attention-based subject detection.
    
    Key insight: The subject is the word with highest AGENCY evidence.
    We use bidirectional reasoning:
    - Forward: What words look like subjects? (position, capitalization)
    - Backward: If X is the subject, does the sentence make sense?
    """
    
    # Function words that are rarely subjects
    FUNCTION_WORDS = {
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'and', 'or', 'but', 'if', 'then', 'so', 'because',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'over',
        'who', 'what', 'where', 'when', 'why', 'how', 'which',
        'its', 'their', 'our', 'your', 'his', 'her', 'my',
        'it', 'they', 'we', 'you', 'he', 'she', 'i',
        'very', 'more', 'most', 'also', 'just', 'even', 'only',
        'not', 'no', 'yes', 'all', 'some', 'any', 'each', 'every',
        'such', 'other', 'another', 'both', 'few', 'many', 'much',
        'recent', 'new', 'old', 'first', 'last', 'next', 'same',
    }
    
    # Verb indicators (endings and common verbs)
    VERB_ENDINGS = ('ed', 'ing', 'es', 's', 'en')
    COMMON_VERBS = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        'go', 'goes', 'went', 'gone', 'going',
        'come', 'comes', 'came', 'coming',
        'make', 'makes', 'made', 'making',
        'take', 'takes', 'took', 'taken', 'taking',
        'get', 'gets', 'got', 'getting',
        'know', 'knows', 'knew', 'known', 'knowing',
        'think', 'thinks', 'thought', 'thinking',
        'see', 'sees', 'saw', 'seen', 'seeing',
        'say', 'says', 'said', 'saying',
        'use', 'uses', 'used', 'using',
        'find', 'finds', 'found', 'finding',
        'give', 'gives', 'gave', 'given', 'giving',
        'tell', 'tells', 'told', 'telling',
        'become', 'becomes', 'became', 'becoming',
        'show', 'shows', 'showed', 'shown', 'showing',
        'leave', 'leaves', 'left', 'leaving',
        'call', 'calls', 'called', 'calling',
        'include', 'includes', 'included', 'including',
        'continue', 'continues', 'continued', 'continuing',
        'provide', 'provides', 'provided', 'providing',
        'require', 'requires', 'required', 'requiring',
        'allow', 'allows', 'allowed', 'allowing',
        'develop', 'develops', 'developed', 'developing',
        'suggest', 'suggests', 'suggested', 'suggesting',
        'consider', 'considers', 'considered', 'considering',
        'describe', 'describes', 'described', 'describing',
        'examine', 'examines', 'examined', 'examining',
        'study', 'studies', 'studied', 'studying',
        'transform', 'transforms', 'transformed', 'transforming',
    }
    
    # Prepositions that often precede objects (not subjects)
    OBJECT_PREPOSITIONS = {'in', 'of', 'to', 'for', 'with', 'by', 'from', 'on', 'at'}
    
    def __init__(self, knowledge=None):
        """
        Initialize the extractor.
        
        Args:
            knowledge: Optional GeometricKnowledge for entity info
        """
        self.knowledge = knowledge
        self.learned_subjects: Counter = Counter()  # Track what we've seen as subjects
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize preserving case for analysis."""
        return re.findall(r'\b\w+\b', text)
    
    def _is_verb(self, word: str) -> bool:
        """Check if word is likely a verb."""
        w = word.lower()
        
        # Capitalized words at start are likely proper nouns, not verbs
        # (unless they're in COMMON_VERBS)
        if word[0].isupper() and w not in self.COMMON_VERBS:
            return False
        
        if w in self.COMMON_VERBS:
            return True
        
        # Check verb endings but exclude common noun patterns
        noun_exceptions = {'advances', 'physics', 'mathematics', 'ethics', 'politics',
                          'economics', 'statistics', 'linguistics', 'genetics',
                          'series', 'species', 'news', 'means', 'evidence', 'science',
                          'presence', 'absence', 'existence', 'essence', 'instance',
                          'holmes', 'watson', 'james', 'charles', 'thomas', 'jones'}
        if w in noun_exceptions:
            return False
        
        # Past tense verbs ending in 'ed' are verbs
        if w.endswith('ed') and len(w) > 3:
            return True
        # Present participle
        if w.endswith('ing') and len(w) > 4:
            return True
        # Third person singular ending in 'es'
        if w.endswith('es') and len(w) > 4:
            # explores -> explor + e = explore
            base = w[:-1]  # explores -> explore
            base2 = w[:-2]  # explores -> explor
            if base in self.COMMON_VERBS or base2 in self.COMMON_VERBS:
                return True
            # Common verb patterns: -ores, -ures, -ates, -izes
            if w.endswith(('ores', 'ures', 'ates', 'izes', 'ises', 'ases')):
                return True
        # Words ending in 's' - only if base is clearly a verb
        if w.endswith('s') and len(w) > 3 and w[-2] not in 'sxzh':
            base = w[:-1]
            if base in self.COMMON_VERBS or base + 'e' in self.COMMON_VERBS:
                return True
        return False
    
    def _is_content_word(self, word: str) -> bool:
        """Check if word is a content word (not function word)."""
        return word.lower() not in self.FUNCTION_WORDS
    
    # Irregular verb mappings for base form conversion
    IRREGULAR_VERBS = {
        'was': 'be', 'were': 'be', 'been': 'be',
        'had': 'have', 'has': 'have',
        'did': 'do', 'does': 'do',
        'went': 'go', 'gone': 'go', 'goes': 'go',
        'came': 'come', 'comes': 'come',
        'made': 'make', 'makes': 'make',
        'took': 'take', 'takes': 'take', 'taken': 'take',
        'got': 'get', 'gets': 'get',
        'knew': 'know', 'knows': 'know', 'known': 'know',
        'thought': 'think', 'thinks': 'think',
        'saw': 'see', 'sees': 'see', 'seen': 'see',
        'said': 'say', 'says': 'say',
        'told': 'tell', 'tells': 'tell',
        'found': 'find', 'finds': 'find',
        'gave': 'give', 'gives': 'give', 'given': 'give',
        'left': 'leave', 'leaves': 'leave',
        'became': 'become', 'becomes': 'become',
        'kept': 'keep', 'keeps': 'keep',
        'brought': 'bring', 'brings': 'bring',
        'began': 'begin', 'begins': 'begin', 'begun': 'begin',
        'wrote': 'write', 'writes': 'write', 'written': 'write',
        'ran': 'run', 'runs': 'run',
        'spoke': 'speak', 'speaks': 'speak', 'spoken': 'speak',
        'stood': 'stand', 'stands': 'stand',
        'understood': 'understand', 'understands': 'understand',
        'held': 'hold', 'holds': 'hold',
        'heard': 'hear', 'hears': 'hear',
        'met': 'meet', 'meets': 'meet',
        'led': 'lead', 'leads': 'lead',
        'felt': 'feel', 'feels': 'feel',
        'fell': 'fall', 'falls': 'fall', 'fallen': 'fall',
        'sent': 'send', 'sends': 'send',
        'built': 'build', 'builds': 'build',
        'lost': 'lose', 'loses': 'lose',
        'paid': 'pay', 'pays': 'pay',
        'spent': 'spend', 'spends': 'spend',
        'caught': 'catch', 'catches': 'catch',
        'taught': 'teach', 'teaches': 'teach',
        'bought': 'buy', 'buys': 'buy',
        'won': 'win', 'wins': 'win',
        'drew': 'draw', 'draws': 'draw', 'drawn': 'draw',
        'grew': 'grow', 'grows': 'grow', 'grown': 'grow',
        'threw': 'throw', 'throws': 'throw', 'thrown': 'throw',
        'drove': 'drive', 'drives': 'drive', 'driven': 'drive',
        'rose': 'rise', 'rises': 'rise', 'risen': 'rise',
        'chose': 'choose', 'chooses': 'choose', 'chosen': 'choose',
        'broke': 'break', 'breaks': 'break', 'broken': 'break',
        'ate': 'eat', 'eats': 'eat', 'eaten': 'eat',
    }
    
    def _to_base_form(self, verb: str) -> str:
        """Convert verb to base form (infinitive)."""
        v = verb.lower()
        
        # Check irregular verbs first
        if v in self.IRREGULAR_VERBS:
            return self.IRREGULAR_VERBS[v]
        
        # Handle -ed past tense
        if v.endswith('ed') and len(v) > 3:
            if v.endswith('ied'):
                return v[:-3] + 'y'  # studied -> study
            elif v.endswith('eed'):
                return v[:-2]  # agreed -> agree
            elif len(v) > 4 and v[-4] == v[-3]:  # doubled consonant
                return v[:-3]  # stopped -> stop
            else:
                base = v[:-2]
                # Handle -formed, -ferred patterns (these don't need 'e')
                if base.endswith(('form', 'fer')):
                    return base  # transform, transfer stay as-is
                # General rule: if base ends in consonant cluster that suggests
                # a dropped 'e', add it back. Common patterns:
                # -ize/-ise verbs, -ate verbs, -ude verbs, -ive verbs, etc.
                if base.endswith(('iz', 'is', 'at', 'ud', 'iv', 'ov', 'av',
                                 'uc', 'ac', 'ec', 'ic', 'oc', 'nc', 'rc',
                                 'ag', 'ng', 'rg', 'dg',
                                 'as', 'os', 'us',
                                 'in', 'an', 'on', 'un',
                                 'ir', 'ar', 'or', 'ur', 'er',
                                 'ab', 'ib', 'ob', 'ub',
                                 'ad', 'id', 'od', 'ud',
                                 'am', 'im', 'om', 'um',
                                 'ap', 'ip', 'op', 'up',
                                 'al', 'il', 'ol', 'ul',
                                 'av', 'ev', 'iv', 'ov', 'uv')):
                    return base + 'e'
                return base if base else v
        
        # Handle -ing forms
        if v.endswith('ing') and len(v) > 4:
            base = v[:-3]
            if len(base) > 1 and base[-1] == base[-2]:
                return base[:-1]  # running -> run
            # Use same general pattern as -ed forms for restoring 'e'
            if base.endswith(('iz', 'is', 'at', 'ud', 'iv', 'ov', 'av',
                             'uc', 'ac', 'ec', 'ic', 'oc', 'nc', 'rc',
                             'ag', 'ng', 'rg', 'dg',
                             'as', 'os', 'us',
                             'in', 'an', 'on', 'un',
                             'ir', 'ar', 'or', 'ur', 'er',
                             'ab', 'ib', 'ob', 'ub',
                             'ad', 'id', 'od', 'ud',
                             'am', 'im', 'om', 'um',
                             'ap', 'ip', 'op', 'up',
                             'al', 'il', 'ol', 'ul',
                             'av', 'ev', 'iv', 'ov', 'uv',
                             'ak', 'ok', 'ik')):
                return base + 'e'
            return base
        
        # Handle -s/-es forms
        if v.endswith('ies') and len(v) > 4:
            return v[:-3] + 'y'  # studies -> study
        if v.endswith('es') and len(v) > 3:
            base = v[:-2]
            if base.endswith(('ch', 'sh', 'ss', 'x', 'z')):
                return base  # watches -> watch
            return v[:-1]  # explores -> explore
        if v.endswith('s') and len(v) > 2 and not v.endswith('ss'):
            return v[:-1]  # walks -> walk
        
        return v
    
    def _find_main_verb(self, words: List[str]) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the main verb in the sentence.
        
        Returns (position, verb) or (None, None).
        """
        # Auxiliary verbs that precede main verbs
        aux_verbs = {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        
        # First pass: find main verb (skip auxiliaries)
        found_aux = False
        for i, word in enumerate(words):
            w = word.lower()
            
            if w in aux_verbs:
                found_aux = True
                continue
            
            # After auxiliary, next verb is likely the main verb
            if found_aux and self._is_verb(word):
                return i, word
            
            # Or a verb that's not auxiliary
            if self._is_verb(word) and w not in aux_verbs:
                return i, word
        
        # Fallback: any verb
        for i, word in enumerate(words):
            if self._is_verb(word):
                return i, word
        
        return None, None
    
    def _gather_subject_evidence(self, word: str, position: int, 
                                  words: List[str], verb_pos: Optional[int]) -> SubjectHypothesis:
        """
        Gather evidence for whether a word is the subject.
        
        Uses bidirectional reasoning:
        - Forward: syntactic signals (position, case, etc.)
        - Backward: semantic coherence (does it make sense as subject?)
        """
        hyp = SubjectHypothesis(word=word, position=position, evidence=[])
        w = word.lower()
        n = len(words)
        
        # === FORWARD EVIDENCE (syntactic signals) ===
        
        # 1. Position: Early words more likely to be subjects
        # But the LAST content word before verb is often the true subject
        if position == 0:
            hyp.add_evidence("sentence-initial", 0.2)
        elif position == 1 and words[0].lower() in {'the', 'a', 'an'}:
            hyp.add_evidence("after-determiner", 0.35)  # "The detective" - detective is subject
        
        # Bonus for being the last content word before verb
        if verb_pos is not None and position < verb_pos:
            # Check if there are any content words between this and verb
            is_last_before_verb = True
            for j in range(position + 1, verb_pos):
                if self._is_content_word(words[j]) and words[j].lower() not in self.OBJECT_PREPOSITIONS:
                    is_last_before_verb = False
                    break
            if is_last_before_verb:
                hyp.add_evidence("last-before-verb", 0.3)
        
        # 2. Capitalization: Proper nouns are often subjects
        if word[0].isupper() and position > 0:
            hyp.add_evidence("capitalized-mid-sentence", 0.2)
        
        # 3. Content word: Function words are rarely subjects
        if self._is_content_word(word):
            hyp.add_evidence("content-word", 0.2)
        else:
            hyp.add_evidence("function-word", -0.3)
        
        # 4. Not preceded by preposition: "in physics" → physics is object
        if position > 0:
            prev = words[position - 1].lower()
            if prev in self.OBJECT_PREPOSITIONS:
                hyp.add_evidence(f"preceded-by-{prev}", -0.5)  # Strong negative
        
        # 5. Before main verb: Subject usually precedes verb
        if verb_pos is not None:
            if position < verb_pos:
                hyp.add_evidence("before-verb", 0.2)
            else:
                hyp.add_evidence("after-verb", -0.2)
        
        # === BACKWARD EVIDENCE (semantic coherence) ===
        
        # 6. Known entity: If we've seen this as subject before
        if self.knowledge and w in self.knowledge.concepts:
            concept = self.knowledge.concepts[w]
            if concept.initiator_count > concept.receiver_count:
                hyp.add_evidence("known-initiator", 0.3)
            elif concept.receiver_count > concept.initiator_count * 2:
                hyp.add_evidence("known-receiver", -0.2)
        
        # 7. Learned subject: We've extracted this as subject before
        if self.learned_subjects[w] > 5:
            hyp.add_evidence("learned-subject", 0.2)
        
        # 8. φ-weighting: Rare words are more meaningful
        # (Zipf inverse - common words like "the" are structural, not content)
        word_freq = self.learned_subjects.get(w, 0) + 1
        phi_weight = PHI ** (-math.log1p(word_freq) * 0.5)
        if phi_weight > 0.5:
            hyp.add_evidence("phi-weighted-rare", phi_weight * 0.2)
        
        return hyp
    
    def _find_best_subject(self, words: List[str], verb_pos: Optional[int]) -> Tuple[Optional[str], int, float]:
        """
        Find the best subject candidate using hypothesis competition.
        
        Key insight: Subject must be BEFORE the verb and be a content word.
        
        Returns (subject, position, confidence).
        """
        if verb_pos is None:
            verb_pos = len(words)  # Assume verb at end if not found
        
        hypotheses = []
        
        # Only consider words BEFORE the verb as subject candidates
        for i, word in enumerate(words):
            if i >= verb_pos:
                break  # Subject must be before verb
            
            # Skip obvious non-subjects
            w = word.lower()
            if w in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'of', 'to', 'for', 'with', 'by'}:
                continue
            if self._is_verb(word):
                continue
            
            hyp = self._gather_subject_evidence(word, i, words, verb_pos)
            hypotheses.append(hyp)
        
        if not hypotheses:
            # Fallback: first content word
            for i, word in enumerate(words):
                if self._is_content_word(word) and not self._is_verb(word):
                    return word, i, 0.3
            return None, 0, 0.0
        
        # Find best hypothesis
        best = max(hypotheses, key=lambda h: h.total_score)
        
        # Confidence based on margin over second-best
        sorted_hyps = sorted(hypotheses, key=lambda h: h.total_score, reverse=True)
        if len(sorted_hyps) > 1:
            margin = best.total_score - sorted_hyps[1].total_score
            confidence = min(1.0, 0.5 + margin)
        else:
            confidence = 0.5
        
        return best.word, best.position, confidence
    
    def _find_object(self, words: List[str], subject_pos: int, verb_pos: int) -> Optional[str]:
        """Find the object (receiver) after the verb."""
        # Skip determiners and find first content word after verb
        skip_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'our', 'their', 'its'}
        
        for i in range(verb_pos + 1, len(words)):
            word = words[i]
            w = word.lower()
            
            if w in skip_words:
                continue
            
            if self._is_content_word(word) and not self._is_verb(word):
                return word
        
        return None
    
    def extract_frame(self, sentence: str) -> Optional[ExtractedFrame]:
        """
        Extract a frame from a sentence using attention-based subject detection.
        
        Returns ExtractedFrame or None if extraction fails.
        """
        words = self._tokenize(sentence)
        
        if len(words) < 3:
            return None
        
        # Find main verb
        verb_pos, verb = self._find_main_verb(words)
        
        if verb is None:
            return None
        
        # Find subject using hypothesis competition
        subject, subject_pos, confidence = self._find_best_subject(words, verb_pos)
        
        if subject is None:
            return None
        
        # Find object
        obj = self._find_object(words, subject_pos, verb_pos)
        
        # Learn from this extraction
        self.learned_subjects[subject.lower()] += 1
        
        # Convert verb to base form for clean storage
        base_verb = self._to_base_form(verb.lower())
        
        return ExtractedFrame(
            initiator=subject.lower(),
            mediator=base_verb,  # Store in base form
            receiver=obj.lower() if obj else '',
            confidence=confidence,
            source_sentence=sentence,
            extraction_method='attention'
        )
    
    def extract_frames(self, text: str, source: str = '') -> List[ExtractedFrame]:
        """
        Extract frames from text.
        
        Args:
            text: Text to extract from
            source: Source identifier
        
        Returns:
            List of ExtractedFrame objects
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+\s*', text)
        
        frames = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            frame = self.extract_frame(sentence)
            if frame and frame.confidence > 0.3:
                frames.append(frame)
        
        return frames


def compare_extractors(sentence: str, knowledge=None):
    """Compare attention-based vs position-based extraction."""
    from truthspace_lcm.core.geometric import GeometricKnowledge
    
    print(f"Sentence: {sentence}")
    print("-" * 60)
    
    # Position-based (current)
    words = re.findall(r'\b\w+\b', sentence.lower())
    n = len(words)
    if n >= 3:
        pos_i = words[0]
        pos_m = words[n // 2]
        pos_r = words[-1]
        print(f"Position-based: I:{pos_i} M:{pos_m} R:{pos_r}")
    
    # Attention-based (new)
    extractor = AttentionExtractor(knowledge)
    frame = extractor.extract_frame(sentence)
    if frame:
        print(f"Attention-based: I:{frame.initiator} M:{frame.mediator} R:{frame.receiver}")
        print(f"  Confidence: {frame.confidence:.2f}")
    else:
        print("Attention-based: Failed to extract")
    
    print()


def demo():
    """Demonstrate the attention extractor."""
    test_sentences = [
        "Recent advances in physics have transformed our understanding.",
        "Holmes examined the evidence carefully.",
        "The detective solved the mystery.",
        "Physics describes the fundamental nature of matter.",
        "Contemporary philosophy explores questions of existence.",
        "Watson assisted Holmes with the investigation.",
        "The study of biology reveals the complexity of life.",
        "Ancient Rome dominated the Mediterranean world.",
    ]
    
    print("=" * 70)
    print("  ATTENTION EXTRACTOR DEMO")
    print("=" * 70)
    print()
    
    for sentence in test_sentences:
        compare_extractors(sentence)


if __name__ == "__main__":
    demo()
