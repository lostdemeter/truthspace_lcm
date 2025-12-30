#!/usr/bin/env python3
"""
Curator LCM: Self-Improving Corpus Curation

Uses geometric features to score and improve sentence quality for ingestion.
The curator can:
1. Score sentences for frame extraction quality
2. Identify problematic frames
3. Suggest improvements or rewrites
4. Learn what makes a good frame from examples

Key Insight: The geometric properties we extract (φ-direction, role counts,
position variance) can ALSO tell us about sentence quality. A good sentence
for frame extraction has:
- Clear initiator (high φ-direction word at start)
- Clear mediator (verb-like word in middle)
- Clear receiver (content word at end)
- Low ambiguity (words have consistent roles)

This is ENCODE = DECODE applied to curation:
- Encoding a sentence reveals its geometric structure
- That structure tells us if the sentence is good for learning
- We can use this to filter, rank, or rewrite sentences

Author: Lesley Gushurst
License: GPLv3
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter


PHI = 1.618034


@dataclass
class SentenceScore:
    """Quality score for a sentence."""
    text: str
    overall: float  # 0-1, higher is better
    
    # Component scores
    structure_score: float = 0.0  # Has clear S-V-O structure
    role_clarity: float = 0.0     # Words have clear roles
    content_density: float = 0.0  # Ratio of content to function words
    length_score: float = 0.0     # Optimal length for frames
    
    # Issues found
    issues: List[str] = field(default_factory=list)
    
    # Suggested improvements
    suggestions: List[str] = field(default_factory=list)


@dataclass 
class FrameScore:
    """Quality score for an extracted frame."""
    initiator: str
    mediator: str
    receiver: str
    text: str
    overall: float
    
    # Component scores
    initiator_quality: float = 0.0  # Is initiator a good subject?
    mediator_quality: float = 0.0   # Is mediator a good verb?
    receiver_quality: float = 0.0   # Is receiver a good object?
    coherence: float = 0.0          # Do the parts make sense together?
    
    issues: List[str] = field(default_factory=list)


class CuratorLCM:
    """
    A curator that uses geometric features to improve corpus quality.
    
    The curator scores sentences and frames, identifies issues,
    and can suggest improvements using the geometric knowledge
    it has learned.
    """
    
    # Common function words (geometric stop words)
    FUNCTION_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'which', 'who', 'whom', 'whose',
    }
    
    # Good subject starters
    GOOD_SUBJECTS = {
        'i', 'he', 'she', 'it', 'we', 'they', 'the', 'a', 'an',
        'my', 'his', 'her', 'our', 'their', 'this', 'that',
    }
    
    # Common verbs (for mediator detection)
    COMMON_VERBS = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        'say', 'said', 'says', 'tell', 'told', 'tells',
        'go', 'goes', 'went', 'gone', 'going',
        'come', 'comes', 'came', 'coming',
        'see', 'sees', 'saw', 'seen', 'seeing',
        'know', 'knows', 'knew', 'known', 'knowing',
        'think', 'thinks', 'thought', 'thinking',
        'take', 'takes', 'took', 'taken', 'taking',
        'make', 'makes', 'made', 'making',
        'get', 'gets', 'got', 'gotten', 'getting',
        'give', 'gives', 'gave', 'given', 'giving',
        'find', 'finds', 'found', 'finding',
        'look', 'looks', 'looked', 'looking',
        'want', 'wants', 'wanted', 'wanting',
        'use', 'uses', 'used', 'using',
        'work', 'works', 'worked', 'working',
        'call', 'calls', 'called', 'calling',
        'try', 'tries', 'tried', 'trying',
        'ask', 'asks', 'asked', 'asking',
        'need', 'needs', 'needed', 'needing',
        'feel', 'feels', 'felt', 'feeling',
        'become', 'becomes', 'became', 'becoming',
        'leave', 'leaves', 'left', 'leaving',
        'put', 'puts', 'putting',
        'mean', 'means', 'meant', 'meaning',
        'keep', 'keeps', 'kept', 'keeping',
        'let', 'lets', 'letting',
        'begin', 'begins', 'began', 'begun', 'beginning',
        'seem', 'seems', 'seemed', 'seeming',
        'help', 'helps', 'helped', 'helping',
        'show', 'shows', 'showed', 'shown', 'showing',
        'hear', 'hears', 'heard', 'hearing',
        'play', 'plays', 'played', 'playing',
        'run', 'runs', 'ran', 'running',
        'move', 'moves', 'moved', 'moving',
        'live', 'lives', 'lived', 'living',
        'believe', 'believes', 'believed', 'believing',
        'hold', 'holds', 'held', 'holding',
        'bring', 'brings', 'brought', 'bringing',
        'write', 'writes', 'wrote', 'written', 'writing',
        'stand', 'stands', 'stood', 'standing',
        'sit', 'sits', 'sat', 'sitting',
        'lose', 'loses', 'lost', 'losing',
        'pay', 'pays', 'paid', 'paying',
        'meet', 'meets', 'met', 'meeting',
        'include', 'includes', 'included', 'including',
        'continue', 'continues', 'continued', 'continuing',
        'set', 'sets', 'setting',
        'learn', 'learns', 'learned', 'learning',
        'change', 'changes', 'changed', 'changing',
        'lead', 'leads', 'led', 'leading',
        'understand', 'understands', 'understood', 'understanding',
        'watch', 'watches', 'watched', 'watching',
        'follow', 'follows', 'followed', 'following',
        'stop', 'stops', 'stopped', 'stopping',
        'create', 'creates', 'created', 'creating',
        'speak', 'speaks', 'spoke', 'spoken', 'speaking',
        'read', 'reads', 'reading',
        'allow', 'allows', 'allowed', 'allowing',
        'add', 'adds', 'added', 'adding',
        'spend', 'spends', 'spent', 'spending',
        'grow', 'grows', 'grew', 'grown', 'growing',
        'open', 'opens', 'opened', 'opening',
        'walk', 'walks', 'walked', 'walking',
        'win', 'wins', 'won', 'winning',
        'offer', 'offers', 'offered', 'offering',
        'remember', 'remembers', 'remembered', 'remembering',
        'love', 'loves', 'loved', 'loving',
        'consider', 'considers', 'considered', 'considering',
        'appear', 'appears', 'appeared', 'appearing',
        'buy', 'buys', 'bought', 'buying',
        'wait', 'waits', 'waited', 'waiting',
        'serve', 'serves', 'served', 'serving',
        'die', 'dies', 'died', 'dying',
        'send', 'sends', 'sent', 'sending',
        'expect', 'expects', 'expected', 'expecting',
        'build', 'builds', 'built', 'building',
        'stay', 'stays', 'stayed', 'staying',
        'fall', 'falls', 'fell', 'fallen', 'falling',
        'cut', 'cuts', 'cutting',
        'reach', 'reaches', 'reached', 'reaching',
        'kill', 'kills', 'killed', 'killing',
        'remain', 'remains', 'remained', 'remaining',
    }
    
    def __init__(self, knowledge=None):
        """
        Initialize curator with optional GeometricKnowledge.
        
        If knowledge is provided, the curator can use learned
        geometric features to improve scoring.
        """
        self.knowledge = knowledge
        
        # Learn from knowledge if available
        self.learned_initiators: Set[str] = set()
        self.learned_mediators: Set[str] = set()
        self.learned_receivers: Set[str] = set()
        
        if knowledge:
            self._learn_from_knowledge(knowledge)
    
    def _learn_from_knowledge(self, knowledge):
        """Learn role patterns from geometric knowledge."""
        for name, concept in knowledge.concepts.items():
            if not concept.is_content_word:
                continue
            
            total = concept.initiator_count + concept.mediator_count + concept.receiver_count
            if total < 2:
                continue
            
            # Classify by dominant role
            if concept.initiator_count > total * 0.5:
                self.learned_initiators.add(name)
            elif concept.mediator_count > total * 0.5:
                self.learned_mediators.add(name)
            elif concept.receiver_count > total * 0.5:
                self.learned_receivers.add(name)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _is_content_word(self, word: str) -> bool:
        """Check if word is a content word (not function word)."""
        return word.lower() not in self.FUNCTION_WORDS
    
    def _is_verb(self, word: str) -> bool:
        """Check if word is likely a verb."""
        w = word.lower()
        
        # Check common verbs
        if w in self.COMMON_VERBS:
            return True
        
        # Check learned mediators
        if w in self.learned_mediators:
            return True
        
        # Check verb-like endings
        if w.endswith(('ed', 'ing', 'es')) and len(w) > 3:
            return True
        
        return False
    
    def _is_good_subject(self, word: str, position: int) -> bool:
        """Check if word is a good subject candidate."""
        w = word.lower()
        
        # Pronouns and determiners are good starters
        if w in self.GOOD_SUBJECTS:
            return True
        
        # Proper nouns (capitalized) are good subjects
        if word[0].isupper() and position == 0:
            return True
        
        # Learned initiators
        if w in self.learned_initiators:
            return True
        
        return False
    
    def score_sentence(self, text: str) -> SentenceScore:
        """
        Score a sentence for frame extraction quality.
        
        Returns a SentenceScore with overall quality and component scores.
        """
        words = self._tokenize(text)
        
        if not words:
            return SentenceScore(text=text, overall=0.0, issues=["Empty sentence"])
        
        issues = []
        suggestions = []
        
        # 1. Length score (optimal: 8-20 words)
        length = len(words)
        if length < 5:
            length_score = 0.2
            issues.append("Too short for good frame extraction")
        elif length < 8:
            length_score = 0.6
        elif length <= 20:
            length_score = 1.0
        elif length <= 30:
            length_score = 0.7
        else:
            length_score = 0.3
            issues.append("Too long - may have multiple clauses")
        
        # 2. Content density (ratio of content to function words)
        content_words = [w for w in words if self._is_content_word(w)]
        content_density = len(content_words) / len(words) if words else 0
        
        if content_density < 0.3:
            issues.append("Low content density - mostly function words")
        
        # 3. Structure score (S-V-O pattern)
        structure_score = 0.0
        
        # Check for subject at start
        has_subject = self._is_good_subject(words[0], 0) if words else False
        if has_subject:
            structure_score += 0.33
        else:
            issues.append("No clear subject at start")
            suggestions.append(f"Consider starting with a clear subject (he/she/the/a proper noun)")
        
        # Check for verb in middle third
        middle_start = len(words) // 3
        middle_end = 2 * len(words) // 3
        middle_words = words[middle_start:middle_end] if middle_end > middle_start else words[1:3]
        
        has_verb = any(self._is_verb(w) for w in middle_words)
        if has_verb:
            structure_score += 0.34
        else:
            # Check anywhere for verb
            has_verb_anywhere = any(self._is_verb(w) for w in words)
            if has_verb_anywhere:
                structure_score += 0.2
                issues.append("Verb not in expected position")
            else:
                issues.append("No clear verb found")
                suggestions.append("Add a clear action verb")
        
        # Check for object/receiver in last third
        last_third = words[2 * len(words) // 3:] if len(words) > 3 else words[-1:]
        has_object = any(self._is_content_word(w) for w in last_third)
        if has_object:
            structure_score += 0.33
        else:
            issues.append("No clear object at end")
        
        # 4. Role clarity (using learned knowledge)
        role_clarity = 0.5  # Default
        
        if self.knowledge:
            # Check if words have clear roles in our knowledge
            clear_roles = 0
            for w in content_words:
                if w in self.learned_initiators or w in self.learned_mediators or w in self.learned_receivers:
                    clear_roles += 1
            
            if content_words:
                role_clarity = clear_roles / len(content_words)
        
        # Calculate overall score
        overall = (
            length_score * 0.2 +
            content_density * 0.2 +
            structure_score * 0.4 +
            role_clarity * 0.2
        )
        
        return SentenceScore(
            text=text,
            overall=overall,
            structure_score=structure_score,
            role_clarity=role_clarity,
            content_density=content_density,
            length_score=length_score,
            issues=issues,
            suggestions=suggestions,
        )
    
    def score_frame(self, initiator: str, mediator: str, receiver: str, text: str) -> FrameScore:
        """
        Score an extracted frame for quality.
        """
        issues = []
        
        # Score initiator
        init_lower = initiator.lower() if initiator else ""
        if not initiator:
            initiator_quality = 0.0
            issues.append("Missing initiator")
        elif init_lower in self.FUNCTION_WORDS:
            initiator_quality = 0.2
            issues.append(f"Initiator '{initiator}' is a function word")
        elif init_lower in self.learned_initiators:
            initiator_quality = 1.0
        elif initiator[0].isupper():
            initiator_quality = 0.8  # Proper noun
        else:
            initiator_quality = 0.5
        
        # Score mediator
        med_lower = mediator.lower() if mediator else ""
        if not mediator:
            mediator_quality = 0.0
            issues.append("Missing mediator")
        elif med_lower in self.COMMON_VERBS or med_lower in self.learned_mediators:
            mediator_quality = 1.0
        elif self._is_verb(mediator):
            mediator_quality = 0.7
        else:
            mediator_quality = 0.3
            issues.append(f"Mediator '{mediator}' doesn't look like a verb")
        
        # Score receiver
        recv_lower = receiver.lower() if receiver else ""
        if not receiver:
            receiver_quality = 0.3  # Optional
        elif recv_lower in self.FUNCTION_WORDS:
            receiver_quality = 0.2
            issues.append(f"Receiver '{receiver}' is a function word")
        elif recv_lower in self.learned_receivers:
            receiver_quality = 1.0
        else:
            receiver_quality = 0.6
        
        # Coherence: do the parts make sense together?
        coherence = 0.5  # Default
        
        if self.knowledge:
            # Check if initiator has this action
            if init_lower in self.knowledge.concepts:
                concept = self.knowledge.concepts[init_lower]
                if med_lower in concept.actions:
                    coherence = 0.9
                elif concept.actions:
                    coherence = 0.6
        
        overall = (
            initiator_quality * 0.3 +
            mediator_quality * 0.4 +
            receiver_quality * 0.2 +
            coherence * 0.1
        )
        
        return FrameScore(
            initiator=initiator,
            mediator=mediator,
            receiver=receiver,
            text=text,
            overall=overall,
            initiator_quality=initiator_quality,
            mediator_quality=mediator_quality,
            receiver_quality=receiver_quality,
            coherence=coherence,
            issues=issues,
        )
    
    def suggest_rewrite(self, text: str) -> Optional[str]:
        """
        Suggest a rewrite of a sentence to improve frame extraction.
        
        Uses geometric knowledge to restructure the sentence.
        """
        score = self.score_sentence(text)
        
        if score.overall > 0.7:
            return None  # Good enough
        
        words = self._tokenize(text)
        if not words:
            return None
        
        # Find the key components
        subject = None
        verb = None
        obj = None
        
        for i, w in enumerate(words):
            if not subject and (self._is_good_subject(w, i) or w in self.learned_initiators):
                subject = w
            elif not verb and self._is_verb(w):
                verb = w
            elif subject and verb and self._is_content_word(w):
                obj = w
                break
        
        if subject and verb:
            if obj:
                return f"{subject.title()} {verb} {obj}."
            else:
                return f"{subject.title()} {verb}."
        
        return None
    
    def curate_sentences(self, sentences: List[str], min_score: float = 0.5) -> Tuple[List[str], List[str]]:
        """
        Curate a list of sentences, returning good ones and rejected ones.
        
        Args:
            sentences: List of sentences to curate
            min_score: Minimum score to accept (0-1)
        
        Returns:
            (accepted, rejected) tuple of sentence lists
        """
        accepted = []
        rejected = []
        
        for s in sentences:
            score = self.score_sentence(s)
            if score.overall >= min_score:
                accepted.append(s)
            else:
                rejected.append(s)
        
        return accepted, rejected
    
    def interactive_curate(self, sentence: str) -> Dict:
        """
        Interactively curate a sentence, returning analysis and suggestions.
        
        This is the main interface for the curator chatbot.
        """
        score = self.score_sentence(sentence)
        rewrite = self.suggest_rewrite(sentence)
        
        result = {
            'sentence': sentence,
            'score': score.overall,
            'verdict': 'good' if score.overall >= 0.7 else 'acceptable' if score.overall >= 0.5 else 'poor',
            'structure_score': score.structure_score,
            'content_density': score.content_density,
            'role_clarity': score.role_clarity,
            'issues': score.issues,
            'suggestions': score.suggestions,
        }
        
        if rewrite:
            result['suggested_rewrite'] = rewrite
        
        return result
    
    def explain_score(self, sentence: str) -> str:
        """
        Generate a natural language explanation of the sentence score.
        """
        result = self.interactive_curate(sentence)
        
        lines = [f"Sentence: \"{sentence}\""]
        lines.append(f"Overall score: {result['score']:.2f} ({result['verdict']})")
        lines.append("")
        
        lines.append("Component scores:")
        lines.append(f"  Structure (S-V-O): {result['structure_score']:.2f}")
        lines.append(f"  Content density: {result['content_density']:.2f}")
        lines.append(f"  Role clarity: {result['role_clarity']:.2f}")
        
        if result['issues']:
            lines.append("")
            lines.append("Issues found:")
            for issue in result['issues']:
                lines.append(f"  - {issue}")
        
        if result['suggestions']:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in result['suggestions']:
                lines.append(f"  - {suggestion}")
        
        if 'suggested_rewrite' in result:
            lines.append("")
            lines.append(f"Suggested rewrite: \"{result['suggested_rewrite']}\"")
        
        return "\n".join(lines)


def demo():
    """Demonstrate the curator."""
    print("=" * 70)
    print("CURATOR LCM DEMO")
    print("Self-improving corpus curation using geometric features")
    print("=" * 70)
    
    # Create curator (without knowledge first)
    curator = CuratorLCM()
    
    # Test sentences
    test_sentences = [
        "Holmes examined the evidence carefully.",
        "The detective solved the mystery.",
        "It was not that he felt any emotion.",
        "And yet there was but one woman.",
        "To Sherlock Holmes she is always the woman.",
        "I have seldom heard him mention her.",
        "Walking slowly through the garden.",
        "The.",
        "He ran.",
        "Watson assisted Holmes in the investigation of the crime scene.",
    ]
    
    print("\nScoring sentences:")
    print("-" * 70)
    
    for s in test_sentences:
        score = curator.score_sentence(s)
        verdict = "✓" if score.overall >= 0.7 else "~" if score.overall >= 0.5 else "✗"
        print(f"{verdict} [{score.overall:.2f}] {s[:50]}...")
        if score.issues:
            print(f"    Issues: {', '.join(score.issues[:2])}")
    
    # Test with knowledge
    print("\n" + "-" * 70)
    print("Testing with GeometricKnowledge:")
    print("-" * 70)
    
    from .geometric import GeometricQA
    
    qa = GeometricQA()
    qa.load_corpus('truthspace_lcm/sample_corpus_geometric.json')
    
    curator_with_knowledge = CuratorLCM(qa.knowledge)
    
    print(f"Learned {len(curator_with_knowledge.learned_initiators)} initiators")
    print(f"Learned {len(curator_with_knowledge.learned_mediators)} mediators")
    print(f"Learned {len(curator_with_knowledge.learned_receivers)} receivers")
    
    print("\nRe-scoring with knowledge:")
    for s in test_sentences[:5]:
        score = curator_with_knowledge.score_sentence(s)
        print(f"[{score.overall:.2f}] {s}")
    
    # Test explanation
    print("\n" + "-" * 70)
    print("Detailed explanation:")
    print("-" * 70)
    print(curator_with_knowledge.explain_score("Holmes examined the evidence carefully."))
    
    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
