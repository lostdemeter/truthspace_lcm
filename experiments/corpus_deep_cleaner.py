#!/usr/bin/env python3
"""
Corpus Deep Cleaner

Uses Qwen2 to identify and fix deeper quality issues in the corpus:
1. Incorrect roles (character for abstract concepts)
2. Awkward/nonsense verbs
3. Missing context
4. Grammatical issues
5. Semantic inconsistencies

This goes beyond simple pattern matching to use LLM judgment.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
import time
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.ollama_corpus_refiner import OllamaClient


class CleanAction(Enum):
    KEEP = "keep"
    REMOVE = "remove"
    REWRITE = "rewrite"


@dataclass
class FrameEvaluation:
    """Evaluation result for a frame."""
    index: int
    text: str
    source: str
    score: float  # 0-10
    issues: List[str]
    action: CleanAction
    rewritten_text: Optional[str] = None
    explanation: str = ""


class CorpusDeepCleaner:
    """
    Uses Qwen2 to deeply analyze and clean corpus frames.
    """
    
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.frames = []
        self.evaluations: List[FrameEvaluation] = []
        
        # Load corpus
        with open(corpus_path, 'r') as f:
            data = json.load(f)
            self.frames = data.get('frames', [])
        
        print(f"Loaded {len(self.frames)} frames from {corpus_path}")
        
        # Qwen2 client
        self.ollama = OllamaClient()
        if not self.ollama.is_available():
            raise RuntimeError("Ollama not available. Deep cleaning requires Qwen2.")
        
        # Stats
        self.stats = {
            'evaluated': 0,
            'keep': 0,
            'remove': 0,
            'rewrite': 0,
            'avg_score': 0.0,
        }
    
    def evaluate_frame(self, frame: Dict, index: int) -> FrameEvaluation:
        """
        Use Qwen2 to evaluate a single frame.
        """
        text = frame.get('text', '')
        source = frame.get('source', 'unknown')
        
        prompt = f"""Evaluate this knowledge base frame for quality:

"{text}"

Rate 0-10 on:
- Grammatical correctness
- Semantic clarity (does it make sense?)
- Factual plausibility
- Usefulness for a QA system

Identify any issues:
- ROLE: Wrong category (e.g., "character" for abstract concepts)
- VERB: Nonsense or awkward verbs
- GRAMMAR: Grammatical errors
- INCOMPLETE: Missing important context
- NONSENSE: Doesn't make logical sense

Respond in this EXACT format:
SCORE: [0-10]
ISSUES: [comma-separated list, or "none"]
ACTION: [KEEP/REMOVE/REWRITE]
REWRITE: [only if ACTION is REWRITE, provide corrected text]
EXPLANATION: [brief reason]"""

        response = self.ollama.generate(prompt, temperature=0.1)
        
        if not response:
            return FrameEvaluation(
                index=index, text=text, source=source,
                score=5.0, issues=[], action=CleanAction.KEEP,
                explanation="No response from evaluator"
            )
        
        # Parse response
        score = 5.0
        issues = []
        action = CleanAction.KEEP
        rewritten = None
        explanation = ""
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('SCORE:'):
                try:
                    score = float(re.search(r'[\d.]+', line).group())
                    score = min(10, max(0, score))
                except:
                    pass
            elif line.startswith('ISSUES:'):
                issues_str = line.replace('ISSUES:', '').strip()
                if issues_str.lower() != 'none':
                    issues = [i.strip() for i in issues_str.split(',') if i.strip()]
            elif line.startswith('ACTION:'):
                action_str = line.replace('ACTION:', '').strip().upper()
                if 'REMOVE' in action_str:
                    action = CleanAction.REMOVE
                elif 'REWRITE' in action_str:
                    action = CleanAction.REWRITE
            elif line.startswith('REWRITE:'):
                rewritten = line.replace('REWRITE:', '').strip().strip('"')
            elif line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
        
        return FrameEvaluation(
            index=index, text=text, source=source,
            score=score, issues=issues, action=action,
            rewritten_text=rewritten, explanation=explanation
        )
    
    def evaluate_batch(self, start: int = 0, limit: int = 100, 
                       delay: float = 0.3, score_threshold: float = 6.0) -> List[FrameEvaluation]:
        """
        Evaluate a batch of frames.
        
        Args:
            start: Starting index
            limit: Number of frames to evaluate
            delay: Delay between API calls
            score_threshold: Frames below this score are flagged
        """
        end = min(start + limit, len(self.frames))
        print(f"\nEvaluating frames {start} to {end}...")
        
        results = []
        total_score = 0
        
        for i in range(start, end):
            frame = self.frames[i]
            
            try:
                evaluation = self.evaluate_frame(frame, i)
                results.append(evaluation)
                self.evaluations.append(evaluation)
                
                total_score += evaluation.score
                self.stats['evaluated'] += 1
                
                if evaluation.action == CleanAction.KEEP:
                    self.stats['keep'] += 1
                elif evaluation.action == CleanAction.REMOVE:
                    self.stats['remove'] += 1
                elif evaluation.action == CleanAction.REWRITE:
                    self.stats['rewrite'] += 1
                
                # Progress
                if (i - start + 1) % 10 == 0:
                    avg = total_score / (i - start + 1)
                    print(f"  [{i - start + 1}/{end - start}] Avg score: {avg:.1f}")
                
            except Exception as e:
                print(f"  Error evaluating frame {i}: {e}")
            
            if delay > 0 and i < end - 1:
                time.sleep(delay)
        
        if results:
            self.stats['avg_score'] = total_score / len(results)
        
        return results
    
    def evaluate_by_source(self, source_filter: str, limit: int = 50) -> List[FrameEvaluation]:
        """
        Evaluate frames from a specific source.
        """
        matching = [(i, f) for i, f in enumerate(self.frames) 
                    if source_filter.lower() in f.get('source', '').lower()]
        
        print(f"\nFound {len(matching)} frames from source '{source_filter}'")
        
        results = []
        for i, (idx, frame) in enumerate(matching[:limit]):
            evaluation = self.evaluate_frame(frame, idx)
            results.append(evaluation)
            self.evaluations.append(evaluation)
            
            self.stats['evaluated'] += 1
            if evaluation.action == CleanAction.KEEP:
                self.stats['keep'] += 1
            elif evaluation.action == CleanAction.REMOVE:
                self.stats['remove'] += 1
            elif evaluation.action == CleanAction.REWRITE:
                self.stats['rewrite'] += 1
            
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{min(limit, len(matching))}]")
            
            time.sleep(0.3)
        
        return results
    
    def apply_cleaning(self, dry_run: bool = True) -> Dict:
        """
        Apply the cleaning decisions.
        """
        if not self.evaluations:
            print("No evaluations to apply.")
            return {}
        
        # Build index of evaluations
        eval_by_idx = {e.index: e for e in self.evaluations}
        
        kept_frames = []
        rewritten_frames = []
        removed_count = 0
        
        for i, frame in enumerate(self.frames):
            if i in eval_by_idx:
                evaluation = eval_by_idx[i]
                if evaluation.action == CleanAction.KEEP:
                    kept_frames.append(frame)
                elif evaluation.action == CleanAction.REWRITE and evaluation.rewritten_text:
                    new_frame = frame.copy()
                    new_frame['text'] = evaluation.rewritten_text
                    new_frame['source'] = frame.get('source', '') + '_cleaned'
                    rewritten_frames.append(new_frame)
                else:
                    removed_count += 1
            else:
                # Not evaluated, keep as-is
                kept_frames.append(frame)
        
        new_frames = kept_frames + rewritten_frames
        
        result = {
            'original_count': len(self.frames),
            'evaluated': len(self.evaluations),
            'kept': len(kept_frames),
            'rewritten': len(rewritten_frames),
            'removed': removed_count,
            'new_count': len(new_frames),
        }
        
        print(f"\n{'DRY RUN - ' if dry_run else ''}CLEANING RESULTS:")
        print(f"  Original: {result['original_count']:,} frames")
        print(f"  Evaluated: {result['evaluated']:,}")
        print(f"  Kept: {result['kept']:,}")
        print(f"  Rewritten: {result['rewritten']:,}")
        print(f"  Removed: {result['removed']:,}")
        print(f"  New total: {result['new_count']:,}")
        
        if not dry_run:
            backup_path = self.corpus_path.replace('.json', '_pre_clean.json')
            with open(backup_path, 'w') as f:
                json.dump({'frames': self.frames}, f, indent=2)
            print(f"\n  Backup saved to: {backup_path}")
            
            with open(self.corpus_path, 'w') as f:
                json.dump({'frames': new_frames}, f, indent=2)
            print(f"  Cleaned corpus saved to: {self.corpus_path}")
        
        return result
    
    def report(self, top_n: int = 10):
        """Print evaluation report."""
        print("\n" + "=" * 70)
        print("DEEP CLEANING REPORT")
        print("=" * 70)
        
        print(f"\nStatistics:")
        print(f"  Evaluated: {self.stats['evaluated']}")
        print(f"  Average score: {self.stats['avg_score']:.1f}")
        print(f"  Keep: {self.stats['keep']}")
        print(f"  Remove: {self.stats['remove']}")
        print(f"  Rewrite: {self.stats['rewrite']}")
        
        # Issue counts
        issue_counts = defaultdict(int)
        for e in self.evaluations:
            for issue in e.issues:
                issue_counts[issue] += 1
        
        if issue_counts:
            print(f"\nIssues found:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                print(f"  {issue}: {count}")
        
        # Low scoring frames
        low_scores = sorted([e for e in self.evaluations if e.score < 5], 
                           key=lambda x: x.score)
        if low_scores:
            print(f"\nLowest scoring frames:")
            for e in low_scores[:top_n]:
                print(f"  [{e.score:.1f}] {e.text[:60]}...")
                print(f"    Issues: {', '.join(e.issues) if e.issues else 'none'}")
                print(f"    Action: {e.action.value}")
        
        # Rewrites
        rewrites = [e for e in self.evaluations if e.action == CleanAction.REWRITE]
        if rewrites:
            print(f"\nSample rewrites:")
            for e in rewrites[:top_n]:
                print(f"  BEFORE: {e.text[:60]}...")
                print(f"  AFTER:  {e.rewritten_text[:60] if e.rewritten_text else 'N/A'}...")


def demo():
    """Demo the deep cleaner."""
    print("=" * 70)
    print("CORPUS DEEP CLEANER")
    print("Using Qwen2 for intelligent frame evaluation")
    print("=" * 70)
    
    corpus_path = "truthspace_lcm/corpus_experimental.json"
    
    cleaner = CorpusDeepCleaner(corpus_path)
    
    # Evaluate a sample
    results = cleaner.evaluate_batch(start=0, limit=50, delay=0.3)
    
    # Report
    cleaner.report()
    
    # Dry run
    cleaner.apply_cleaning(dry_run=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep clean corpus with Qwen2")
    parser.add_argument("--corpus", default="truthspace_lcm/corpus_experimental.json")
    parser.add_argument("--start", type=int, default=0, help="Starting frame index")
    parser.add_argument("--limit", type=int, default=100, help="Number of frames to evaluate")
    parser.add_argument("--source", type=str, help="Filter by source")
    parser.add_argument("--apply", action="store_true", help="Apply cleaning (default is dry run)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls")
    
    args = parser.parse_args()
    
    cleaner = CorpusDeepCleaner(args.corpus)
    
    if args.source:
        cleaner.evaluate_by_source(args.source, limit=args.limit)
    else:
        cleaner.evaluate_batch(start=args.start, limit=args.limit, delay=args.delay)
    
    cleaner.report()
    cleaner.apply_cleaning(dry_run=not args.apply)


if __name__ == "__main__":
    main()
