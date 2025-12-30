#!/usr/bin/env python3
"""
Quality Verification for Self-Improvement

Measures the quality of the corpus and model to ensure self-improvement
is actually improving, not degrading.

Metrics:
1. **Answer Quality** - Can the model answer benchmark questions correctly?
2. **Generation Fluency** - Are generated sentences coherent?
3. **Concept Coverage** - How many domains are represented?
4. **Frame Quality** - Are frames well-formed (good I-M-R structure)?
5. **Role Clarity** - Do concepts have clear geometric roles?

The key insight: We need a HELD-OUT test set that doesn't change.
If performance on the test set improves, the self-improvement is working.
If it degrades, we're adding noise.

Usage:
    python scripts/quality_check.py                    # Full report
    python scripts/quality_check.py --quick            # Quick check
    python scripts/quality_check.py --compare old.json # Compare two corpora
    python scripts/quality_check.py --history          # Show improvement over time

Author: Lesley Gushurst
License: GPLv3
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.geometric import GeometricQA, HolographicGeometricQA
from truthspace_lcm.core.curator import CuratorLCM


# Benchmark questions - now focused on response quality rather than exact keywords
# We measure: relevance, coherence, and informativeness
BENCHMARK_QA = [
    # Literature domain - check if response mentions the entity and has meaningful content
    ("Who is Holmes?", {"entity": "holmes", "domain": "literature", "type": "who"}),
    ("Who is Watson?", {"entity": "watson", "domain": "literature", "type": "who"}),
    ("What does Holmes do?", {"entity": "holmes", "domain": "literature", "type": "what"}),
    
    # General knowledge
    ("What is physics?", {"entity": "physics", "domain": "science", "type": "what"}),
    ("What is philosophy?", {"entity": "philosophy", "domain": "philosophy", "type": "what"}),
    ("What is biology?", {"entity": "biology", "domain": "science", "type": "what"}),
    
    # Relationships
    ("Who works with Holmes?", {"entity": "holmes", "domain": "literature", "type": "who"}),
]

# Benchmark generation prompts
BENCHMARK_GENERATION = [
    ("holmes", 3),  # Generate 3 sentences about Holmes
    ("watson", 2),
    ("knowledge", 2),
    ("science", 2),
]

# Quality thresholds
THRESHOLDS = {
    'answer_accuracy': 0.5,      # At least 50% of benchmark questions answered
    'generation_fluency': 0.4,   # At least 40% fluency score
    'frame_quality': 0.6,        # At least 60% of frames are well-formed
    'role_clarity': 0.3,         # At least 30% of concepts have clear roles
    'min_concepts': 100,         # At least 100 concepts
    'min_frames': 50,            # At least 50 frames
}


class QualityChecker:
    """
    Verifies quality of corpus and model.
    """
    
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.qa = None
        self.curator = None
        self.metrics = {}
        
        if self.corpus_path.exists():
            self.qa = HolographicGeometricQA()
            self.qa.load_corpus(str(self.corpus_path))
            self.curator = CuratorLCM(self.qa.knowledge)
    
    def check_answer_quality(self) -> Tuple[float, List[Dict]]:
        """
        Test model answer quality using multiple criteria:
        - Relevance: Does the answer mention the queried entity?
        - Coherence: Is the answer grammatically structured?
        - Informativeness: Does it provide meaningful content?
        
        Returns (quality_score, details)
        """
        if not self.qa:
            return 0.0, []
        
        scores = []
        details = []
        
        for question, criteria in BENCHMARK_QA:
            answer = self.qa.ask(question).lower()
            entity = criteria.get('entity', '').lower()
            
            # Score components (0-1 each)
            score = 0.0
            issues = []
            
            # 1. Relevance (0.3): Does answer mention the entity?
            if entity and entity in answer:
                score += 0.3
            else:
                issues.append('entity_missing')
            
            # 2. Coherence (0.3): Is it a proper sentence?
            words = answer.split()
            if len(words) >= 5:  # Has enough words
                score += 0.15
            else:
                issues.append('too_short')
            
            if answer and answer[0].isalpha():  # Starts properly
                score += 0.15
            else:
                issues.append('bad_start')
            
            # 3. Informativeness (0.4): Has verbs and descriptive content
            verb_indicators = ('is', 'are', 'was', 'were', 'has', 'have', 'does', 'do',
                              'examines', 'investigates', 'studies', 'involves', 'includes',
                              'emerges', 'provides', 'states', 'deduces', 'observes')
            has_verb = any(v in answer for v in verb_indicators)
            if has_verb:
                score += 0.2
            else:
                issues.append('no_verb')
            
            # Has descriptive words (not just the entity repeated)
            unique_content = set(words) - {entity, 'is', 'a', 'the', 'an', 'who', 'and', 'or'}
            if len(unique_content) >= 3:
                score += 0.2
            else:
                issues.append('low_content')
            
            scores.append(score)
            details.append({
                'question': question,
                'answer': answer[:100],
                'entity': entity,
                'score': score,
                'issues': issues,
                'correct': score >= 0.5,  # Consider "correct" if score >= 50%
            })
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score, details
    
    def check_generation_fluency(self) -> Tuple[float, List[Dict]]:
        """
        Test if generated text is fluent.
        
        Fluency heuristics:
        - Has multiple words
        - Starts with capital
        - Has verb-like words
        - Not too repetitive
        """
        if not self.qa:
            return 0.0, []
        
        scores = []
        details = []
        
        for concept, num_sentences in BENCHMARK_GENERATION:
            try:
                text = self.qa.generate_about(concept, num_sentences)
            except:
                text = ""
            
            if not text:
                scores.append(0.0)
                details.append({'concept': concept, 'text': '', 'score': 0.0, 'issues': ['No output']})
                continue
            
            # Score the generation
            words = text.split()
            sentences = text.split('.')
            
            score = 0.0
            issues = []
            
            # Length check
            if len(words) >= num_sentences * 3:
                score += 0.25
            else:
                issues.append('Too short')
            
            # Capitalization check
            if text[0].isupper():
                score += 0.25
            else:
                issues.append('No capital')
            
            # Verb check
            verb_endings = ('ed', 'ing', 'es', 's')
            has_verb = any(w.endswith(verb_endings) for w in words)
            if has_verb:
                score += 0.25
            else:
                issues.append('No verbs')
            
            # Repetition check
            unique_words = set(w.lower() for w in words)
            repetition_ratio = len(unique_words) / len(words) if words else 0
            if repetition_ratio > 0.5:
                score += 0.25
            else:
                issues.append('Too repetitive')
            
            scores.append(score)
            details.append({
                'concept': concept,
                'text': text[:100],
                'score': score,
                'issues': issues,
            })
        
        avg_score = sum(scores) / len(scores) if scores else 0
        return avg_score, details
    
    def check_frame_quality(self) -> Tuple[float, Dict]:
        """
        Check quality of extracted frames.
        
        Good frames have:
        - Non-empty initiator, mediator, receiver
        - Initiator is a content word (not function word)
        - Mediator looks like a verb
        """
        if not self.qa:
            return 0.0, {}
        
        frames = self.qa.knowledge.frames
        if not frames:
            return 0.0, {'total': 0, 'good': 0}
        
        good_frames = 0
        issues = Counter()
        
        for frame in frames:
            is_good = True
            
            # Check initiator
            if not frame.initiator or frame.initiator.lower() in self.curator.FUNCTION_WORDS:
                is_good = False
                issues['bad_initiator'] += 1
            
            # Check mediator (should be verb-like)
            if not frame.mediator:
                is_good = False
                issues['missing_mediator'] += 1
            elif not self.curator._is_verb(frame.mediator):
                # Not necessarily bad, but note it
                issues['non_verb_mediator'] += 1
            
            # Check receiver
            if not frame.receiver:
                issues['missing_receiver'] += 1
                # Not necessarily bad
            
            if is_good:
                good_frames += 1
        
        quality = good_frames / len(frames)
        return quality, {
            'total': len(frames),
            'good': good_frames,
            'issues': dict(issues),
        }
    
    def check_role_clarity(self) -> Tuple[float, Dict]:
        """
        Check if concepts have clear geometric roles.
        
        A concept has clear role if one role (initiator, mediator, receiver)
        accounts for >50% of its usage. Only concepts with >=5 uses are
        evaluated to ensure statistical significance.
        """
        if not self.qa:
            return 0.0, {}
        
        concepts = self.qa.knowledge.concepts
        content_words = [c for c in concepts.values() if c.is_content_word]
        
        if not content_words:
            return 0.0, {'total': 0}
        
        clear_initiators = 0
        clear_receivers = 0
        clear_mediators = 0
        evaluated = 0  # Only count concepts with enough data
        
        for c in content_words:
            total_roles = c.initiator_count + c.mediator_count + c.receiver_count
            if total_roles < 5:  # Need at least 5 uses for statistical significance
                continue
            
            evaluated += 1
            
            # Check for dominant role
            if c.initiator_count > total_roles * 0.5:
                clear_initiators += 1
            elif c.receiver_count > total_roles * 0.5:
                clear_receivers += 1
            elif c.mediator_count > total_roles * 0.5:
                clear_mediators += 1
        
        total_clear = clear_initiators + clear_receivers + clear_mediators
        clarity = total_clear / evaluated if evaluated > 0 else 0.0
        
        return clarity, {
            'total_content_words': len(content_words),
            'evaluated': evaluated,
            'clear_initiators': clear_initiators,
            'clear_receivers': clear_receivers,
            'clear_mediators': clear_mediators,
            'total_clear': total_clear,
        }
    
    def check_concept_coverage(self) -> Dict:
        """
        Check domain coverage of concepts.
        """
        if not self.qa:
            return {}
        
        # Check for domain-specific keywords
        domains = {
            'literature': ['holmes', 'watson', 'darcy', 'elizabeth', 'novel', 'story'],
            'science': ['physics', 'biology', 'chemistry', 'energy', 'matter', 'cell'],
            'philosophy': ['philosophy', 'knowledge', 'truth', 'reason', 'logic'],
            'history': ['war', 'empire', 'revolution', 'century', 'king', 'queen'],
        }
        
        concepts = set(self.qa.knowledge.concepts.keys())
        
        coverage = {}
        for domain, keywords in domains.items():
            found = [kw for kw in keywords if kw in concepts]
            coverage[domain] = {
                'found': found,
                'coverage': len(found) / len(keywords),
            }
        
        return coverage
    
    def run_full_check(self) -> Dict:
        """
        Run all quality checks and return comprehensive report.
        """
        if not self.qa:
            return {'error': 'No corpus loaded'}
        
        print("Running quality checks...")
        
        # Basic stats
        stats = {
            'corpus_path': str(self.corpus_path),
            'timestamp': datetime.now().isoformat(),
            'total_frames': len(self.qa.knowledge.frames),
            'total_concepts': len(self.qa.knowledge.concepts),
            'content_words': len([c for c in self.qa.knowledge.concepts.values() if c.is_content_word]),
        }
        
        # Answer quality
        print("  Checking answer quality...")
        answer_acc, answer_details = self.check_answer_quality()
        stats['answer_accuracy'] = answer_acc
        stats['answer_details'] = answer_details
        
        # Generation fluency
        print("  Checking generation fluency...")
        gen_fluency, gen_details = self.check_generation_fluency()
        stats['generation_fluency'] = gen_fluency
        stats['generation_details'] = gen_details
        
        # Frame quality
        print("  Checking frame quality...")
        frame_quality, frame_details = self.check_frame_quality()
        stats['frame_quality'] = frame_quality
        stats['frame_details'] = frame_details
        
        # Role clarity
        print("  Checking role clarity...")
        role_clarity, role_details = self.check_role_clarity()
        stats['role_clarity'] = role_clarity
        stats['role_details'] = role_details
        
        # Domain coverage
        print("  Checking domain coverage...")
        stats['domain_coverage'] = self.check_concept_coverage()
        
        # Overall health score
        health_score = (
            answer_acc * 0.3 +
            gen_fluency * 0.2 +
            frame_quality * 0.3 +
            role_clarity * 0.2
        )
        stats['health_score'] = health_score
        
        # Check thresholds
        stats['passing'] = all([
            answer_acc >= THRESHOLDS['answer_accuracy'],
            gen_fluency >= THRESHOLDS['generation_fluency'],
            frame_quality >= THRESHOLDS['frame_quality'],
            role_clarity >= THRESHOLDS['role_clarity'],
            stats['total_concepts'] >= THRESHOLDS['min_concepts'],
            stats['total_frames'] >= THRESHOLDS['min_frames'],
        ])
        
        self.metrics = stats
        return stats
    
    def print_report(self, stats: Dict = None):
        """Print a human-readable report."""
        if stats is None:
            stats = self.metrics
        
        if not stats:
            print("No metrics available. Run check first.")
            return
        
        print()
        print("=" * 70)
        print("  QUALITY REPORT")
        print("=" * 70)
        print(f"  Corpus: {stats.get('corpus_path', 'N/A')}")
        print(f"  Time: {stats.get('timestamp', 'N/A')}")
        print()
        
        # Basic stats
        print("CORPUS STATS:")
        print(f"  Frames: {stats.get('total_frames', 0)}")
        print(f"  Concepts: {stats.get('total_concepts', 0)}")
        print(f"  Content words: {stats.get('content_words', 0)}")
        print()
        
        # Scores
        print("QUALITY SCORES:")
        
        def score_bar(score, threshold):
            filled = int(score * 20)
            bar = "█" * filled + "░" * (20 - filled)
            status = "✓" if score >= threshold else "✗"
            return f"{status} [{bar}] {score:.1%}"
        
        print(f"  Answer Accuracy:    {score_bar(stats.get('answer_accuracy', 0), THRESHOLDS['answer_accuracy'])}")
        print(f"  Generation Fluency: {score_bar(stats.get('generation_fluency', 0), THRESHOLDS['generation_fluency'])}")
        print(f"  Frame Quality:      {score_bar(stats.get('frame_quality', 0), THRESHOLDS['frame_quality'])}")
        print(f"  Role Clarity:       {score_bar(stats.get('role_clarity', 0), THRESHOLDS['role_clarity'])}")
        print()
        
        health = stats.get('health_score', 0)
        print(f"  OVERALL HEALTH:     {score_bar(health, 0.5)}")
        print()
        
        # Domain coverage
        print("DOMAIN COVERAGE:")
        for domain, info in stats.get('domain_coverage', {}).items():
            cov = info.get('coverage', 0)
            found = info.get('found', [])
            print(f"  {domain}: {cov:.0%} ({', '.join(found[:3])}{'...' if len(found) > 3 else ''})")
        print()
        
        # Answer details
        print("BENCHMARK ANSWERS:")
        for detail in stats.get('answer_details', [])[:5]:
            score = detail.get('score', 0)
            status = "✓" if detail.get('correct', False) else "✗"
            issues = detail.get('issues', [])
            print(f"  {status} Q: {detail['question']} [{score:.0%}]")
            print(f"      A: {detail['answer'][:60]}...")
            if issues:
                print(f"      Issues: {', '.join(issues)}")
        print()
        
        # Verdict
        if stats.get('passing', False):
            print("✓ QUALITY CHECK PASSED")
        else:
            print("✗ QUALITY CHECK FAILED - Review metrics above")
        
        print("=" * 70)


def save_history(stats: Dict, history_file: Path):
    """Save stats to history file for tracking over time."""
    history = []
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
    
    # Add new entry (simplified)
    entry = {
        'timestamp': stats['timestamp'],
        'frames': stats['total_frames'],
        'concepts': stats['total_concepts'],
        'answer_accuracy': stats['answer_accuracy'],
        'generation_fluency': stats['generation_fluency'],
        'frame_quality': stats['frame_quality'],
        'role_clarity': stats['role_clarity'],
        'health_score': stats['health_score'],
    }
    history.append(entry)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def show_history(history_file: Path):
    """Show improvement history."""
    if not history_file.exists():
        print("No history file found.")
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    if not history:
        print("History is empty.")
        return
    
    print()
    print("=" * 70)
    print("  IMPROVEMENT HISTORY")
    print("=" * 70)
    print()
    print(f"{'Time':<20} {'Frames':>8} {'Concepts':>10} {'Answer':>8} {'Health':>8}")
    print("-" * 70)
    
    for entry in history[-20:]:  # Last 20 entries
        ts = entry['timestamp'][:16].replace('T', ' ')
        print(f"{ts:<20} {entry['frames']:>8} {entry['concepts']:>10} "
              f"{entry['answer_accuracy']:>7.0%} {entry['health_score']:>7.0%}")
    
    print()
    
    # Show trend
    if len(history) >= 2:
        first = history[0]
        last = history[-1]
        
        print("TREND:")
        print(f"  Frames: {first['frames']} → {last['frames']} ({last['frames'] - first['frames']:+d})")
        print(f"  Health: {first['health_score']:.0%} → {last['health_score']:.0%} "
              f"({(last['health_score'] - first['health_score'])*100:+.1f}%)")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Quality verification for self-improvement')
    parser.add_argument('--corpus', type=str, default='truthspace_lcm/corpus_self_improved.json',
                        help='Corpus to check')
    parser.add_argument('--quick', action='store_true', help='Quick check (fewer tests)')
    parser.add_argument('--history', action='store_true', help='Show improvement history')
    parser.add_argument('--save-history', action='store_true', help='Save results to history')
    parser.add_argument('--compare', type=str, help='Compare with another corpus')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    history_file = Path('quality_history.json')
    
    if args.history:
        show_history(history_file)
        return 0
    
    # Run quality check
    checker = QualityChecker(args.corpus)
    stats = checker.run_full_check()
    
    if args.json:
        # Remove details for cleaner JSON
        clean_stats = {k: v for k, v in stats.items() if not k.endswith('_details')}
        print(json.dumps(clean_stats, indent=2))
    else:
        checker.print_report(stats)
    
    if args.save_history:
        save_history(stats, history_file)
        print(f"\nSaved to {history_file}")
    
    # Compare if requested
    if args.compare:
        print("\n" + "=" * 70)
        print("  COMPARISON")
        print("=" * 70)
        
        other_checker = QualityChecker(args.compare)
        other_stats = other_checker.run_full_check()
        
        print(f"\n{'Metric':<25} {'Current':>12} {'Other':>12} {'Diff':>12}")
        print("-" * 70)
        
        for key in ['total_frames', 'total_concepts', 'answer_accuracy', 
                    'generation_fluency', 'frame_quality', 'health_score']:
            curr = stats.get(key, 0)
            other = other_stats.get(key, 0)
            diff = curr - other
            
            if isinstance(curr, float):
                print(f"{key:<25} {curr:>11.1%} {other:>11.1%} {diff:>+11.1%}")
            else:
                print(f"{key:<25} {curr:>12} {other:>12} {diff:>+12}")
    
    return 0 if stats.get('passing', False) else 1


if __name__ == "__main__":
    sys.exit(main())
