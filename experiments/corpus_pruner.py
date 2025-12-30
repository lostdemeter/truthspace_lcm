#!/usr/bin/env python3
"""
Corpus Pruner

Identifies and removes bad data from the corpus:
1. Very short frames (incomplete/garbage)
2. Excessive duplicates (keep max N copies)
3. Frames with typos/nonsense verbs
4. Frames with incorrect roles for abstract concepts

Uses Qwen2 to evaluate questionable frames and decide whether to keep/fix/remove.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import json
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from experiments.ollama_corpus_refiner import OllamaClient
except ImportError:
    OllamaClient = None


class PruneAction(Enum):
    KEEP = "keep"
    REMOVE = "remove"
    FIX = "fix"


@dataclass
class FrameAnalysis:
    """Analysis result for a frame."""
    index: int
    text: str
    source: str
    issues: List[str]
    action: PruneAction
    fixed_text: Optional[str] = None
    reason: str = ""


class CorpusPruner:
    """
    Analyzes and prunes bad data from the corpus.
    """
    
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.frames = []
        self.analyses: List[FrameAnalysis] = []
        
        # Load corpus
        with open(corpus_path, 'r') as f:
            data = json.load(f)
            self.frames = data.get('frames', [])
        
        print(f"Loaded {len(self.frames)} frames from {corpus_path}")
        
        # Qwen2 client for smart analysis
        self.ollama = OllamaClient() if OllamaClient else None
        if self.ollama and not self.ollama.is_available():
            print("WARNING: Ollama not available. Using rule-based pruning only.")
            self.ollama = None
        
        # Known typos and fixes (use word boundaries to avoid false matches)
        # These are regex patterns with word boundaries
        self.typo_fixes = {
            r'\bmonitores\b': 'monitors',
            r'\bfacilitats\b': 'facilitates',
            r'\bongoes\b': 'ongoing',
            # Note: "michels" is a proper name (Robert Michels), not a typo
            # Note: "emphasi" matches valid words like "emphasis", "emphasizing"
        }
        
        # Actual typos to detect (not fix, just flag)
        self.typo_patterns = ['monitores', 'facilitats', 'ongoes']
        
        # Abstract concept markers
        self.abstract_markers = ['ology', 'tion', 'ment', 'ness', 'ism', 'ics', 'istry', 'ure']
        
        # Stats
        self.stats = {
            'total': len(self.frames),
            'analyzed': 0,
            'keep': 0,
            'remove': 0,
            'fix': 0,
        }
    
    def analyze_all(self, max_duplicates: int = 10, min_length: int = 25) -> Dict:
        """
        Analyze all frames and determine actions.
        
        Args:
            max_duplicates: Maximum copies of identical frames to keep
            min_length: Minimum frame length (shorter = remove)
        """
        print(f"\nAnalyzing {len(self.frames)} frames...")
        print(f"  Max duplicates: {max_duplicates}")
        print(f"  Min length: {min_length}")
        
        # Track duplicates
        text_counts = Counter(f.get('text', '') for f in self.frames)
        text_seen = defaultdict(int)
        
        for i, frame in enumerate(self.frames):
            text = frame.get('text', '')
            source = frame.get('source', 'unknown')
            issues = []
            action = PruneAction.KEEP
            fixed_text = None
            reason = ""
            
            # 1. Check length
            if len(text) < min_length:
                issues.append("too_short")
                action = PruneAction.REMOVE
                reason = f"Frame too short ({len(text)} chars)"
            
            # 2. Check duplicates
            text_seen[text] += 1
            if text_seen[text] > max_duplicates:
                issues.append("duplicate")
                action = PruneAction.REMOVE
                reason = f"Duplicate #{text_seen[text]} (max {max_duplicates})"
            
            # 3. Check for typos (using regex with word boundaries)
            text_lower = text.lower()
            for typo_pattern, fix in self.typo_fixes.items():
                if re.search(typo_pattern, text_lower):
                    typo_word = typo_pattern.replace(r'\b', '')
                    issues.append(f"typo:{typo_word}")
                    if action != PruneAction.REMOVE:
                        action = PruneAction.FIX
                        fixed_text = re.sub(typo_pattern, fix, text, flags=re.IGNORECASE)
                        reason = f"Fixed typo: {typo_word} → {fix}"
            
            # 4. Check for "character" with abstract concepts
            if 'is a character' in text_lower or 'is a protagonist' in text_lower:
                for marker in self.abstract_markers:
                    if marker in text_lower:
                        issues.append("wrong_role")
                        if action != PruneAction.REMOVE:
                            action = PruneAction.FIX
                            fixed_text = re.sub(
                                r'is a (character|protagonist)',
                                'is a concept',
                                text,
                                flags=re.IGNORECASE
                            )
                            reason = "Fixed role: character → concept for abstract"
                        break
            
            # 5. Check for reinforcement bloat (too many from same source)
            if source == 'reinforcement' and text_seen[text] > 5:
                issues.append("reinforcement_bloat")
                if action == PruneAction.KEEP:
                    action = PruneAction.REMOVE
                    reason = "Excessive reinforcement frames"
            
            analysis = FrameAnalysis(
                index=i,
                text=text,
                source=source,
                issues=issues,
                action=action,
                fixed_text=fixed_text,
                reason=reason,
            )
            self.analyses.append(analysis)
            
            # Update stats
            self.stats['analyzed'] += 1
            if action == PruneAction.KEEP:
                self.stats['keep'] += 1
            elif action == PruneAction.REMOVE:
                self.stats['remove'] += 1
            elif action == PruneAction.FIX:
                self.stats['fix'] += 1
            
            # Progress
            if (i + 1) % 10000 == 0:
                print(f"  Analyzed {i + 1}/{len(self.frames)}...")
        
        return self.stats
    
    def analyze_with_qwen2(self, sample_size: int = 100) -> List[FrameAnalysis]:
        """
        Use Qwen2 to analyze questionable frames.
        
        Returns list of frames that Qwen2 recommends action on.
        """
        if not self.ollama:
            print("Qwen2 not available.")
            return []
        
        # Find questionable frames (have issues but not clear-cut)
        questionable = [a for a in self.analyses if a.issues and a.action == PruneAction.KEEP]
        
        if not questionable:
            print("No questionable frames to analyze.")
            return []
        
        sample = questionable[:sample_size]
        print(f"\nAnalyzing {len(sample)} questionable frames with Qwen2...")
        
        results = []
        for i, analysis in enumerate(sample):
            prompt = f"""Evaluate this corpus frame for a knowledge base:

"{analysis.text}"

Issues detected: {', '.join(analysis.issues) if analysis.issues else 'none'}

Should this frame be:
1. KEEP - It's valid and useful
2. REMOVE - It's garbage, incomplete, or wrong
3. FIX - It has fixable issues

Respond with ONLY one word: KEEP, REMOVE, or FIX
If FIX, add a second line with the corrected text."""

            response = self.ollama.generate(prompt, temperature=0.1)
            
            if response:
                lines = response.strip().split('\n')
                action_str = lines[0].strip().upper()
                
                if 'REMOVE' in action_str:
                    analysis.action = PruneAction.REMOVE
                    analysis.reason = "Qwen2: Remove"
                elif 'FIX' in action_str:
                    analysis.action = PruneAction.FIX
                    if len(lines) > 1:
                        analysis.fixed_text = lines[1].strip().strip('"')
                    analysis.reason = "Qwen2: Fix"
                else:
                    analysis.action = PruneAction.KEEP
                    analysis.reason = "Qwen2: Keep"
                
                results.append(analysis)
            
            if (i + 1) % 10 == 0:
                print(f"  Qwen2 analyzed {i + 1}/{len(sample)}...")
        
        return results
    
    def apply_pruning(self, dry_run: bool = True) -> Dict:
        """
        Apply the pruning decisions to create a cleaned corpus.
        
        Args:
            dry_run: If True, don't actually save (just report)
        """
        kept_frames = []
        fixed_frames = []
        removed_count = 0
        
        for analysis in self.analyses:
            if analysis.action == PruneAction.KEEP:
                kept_frames.append(self.frames[analysis.index])
            elif analysis.action == PruneAction.FIX and analysis.fixed_text:
                fixed_frame = self.frames[analysis.index].copy()
                fixed_frame['text'] = analysis.fixed_text
                fixed_frame['source'] = fixed_frame.get('source', '') + '_fixed'
                fixed_frames.append(fixed_frame)
            else:
                removed_count += 1
        
        new_frames = kept_frames + fixed_frames
        
        result = {
            'original_count': len(self.frames),
            'kept': len(kept_frames),
            'fixed': len(fixed_frames),
            'removed': removed_count,
            'new_count': len(new_frames),
            'reduction': len(self.frames) - len(new_frames),
            'reduction_pct': (len(self.frames) - len(new_frames)) / len(self.frames) * 100,
        }
        
        print(f"\n{'DRY RUN - ' if dry_run else ''}PRUNING RESULTS:")
        print(f"  Original: {result['original_count']:,} frames")
        print(f"  Kept: {result['kept']:,}")
        print(f"  Fixed: {result['fixed']:,}")
        print(f"  Removed: {result['removed']:,}")
        print(f"  New total: {result['new_count']:,}")
        print(f"  Reduction: {result['reduction']:,} ({result['reduction_pct']:.1f}%)")
        
        if not dry_run:
            # Backup original
            backup_path = self.corpus_path.replace('.json', '_pre_prune.json')
            with open(backup_path, 'w') as f:
                json.dump({'frames': self.frames}, f, indent=2)
            print(f"\n  Backup saved to: {backup_path}")
            
            # Save pruned corpus
            with open(self.corpus_path, 'w') as f:
                json.dump({'frames': new_frames}, f, indent=2)
            print(f"  Pruned corpus saved to: {self.corpus_path}")
        
        return result
    
    def report_issues(self, top_n: int = 10):
        """Print a report of issues found."""
        print("\n" + "=" * 70)
        print("CORPUS ANALYSIS REPORT")
        print("=" * 70)
        
        # Count by issue type
        issue_counts = defaultdict(int)
        for analysis in self.analyses:
            for issue in analysis.issues:
                issue_counts[issue] += 1
        
        print(f"\nIssues found:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue}: {count:,}")
        
        # Count by action
        print(f"\nActions:")
        print(f"  KEEP: {self.stats['keep']:,}")
        print(f"  REMOVE: {self.stats['remove']:,}")
        print(f"  FIX: {self.stats['fix']:,}")
        
        # Sample removals
        removals = [a for a in self.analyses if a.action == PruneAction.REMOVE]
        print(f"\nSample frames to REMOVE ({len(removals):,} total):")
        for analysis in removals[:top_n]:
            print(f"  [{analysis.source}] {analysis.text[:60]}...")
            print(f"    Reason: {analysis.reason}")
        
        # Sample fixes
        fixes = [a for a in self.analyses if a.action == PruneAction.FIX]
        print(f"\nSample frames to FIX ({len(fixes):,} total):")
        for analysis in fixes[:top_n]:
            print(f"  BEFORE: {analysis.text[:60]}...")
            print(f"  AFTER:  {analysis.fixed_text[:60] if analysis.fixed_text else 'N/A'}...")
            print(f"    Reason: {analysis.reason}")
        
        # By source
        source_actions = defaultdict(lambda: {'keep': 0, 'remove': 0, 'fix': 0})
        for analysis in self.analyses:
            source_actions[analysis.source][analysis.action.value] += 1
        
        print(f"\nBy source:")
        for source, actions in sorted(source_actions.items(), key=lambda x: -sum(x[1].values()))[:10]:
            total = sum(actions.values())
            remove_pct = actions['remove'] / total * 100 if total > 0 else 0
            print(f"  {source}: {total:,} frames ({remove_pct:.1f}% to remove)")


def demo():
    """Demo the corpus pruner."""
    print("=" * 70)
    print("CORPUS PRUNER")
    print("=" * 70)
    
    corpus_path = "truthspace_lcm/corpus_experimental.json"
    
    pruner = CorpusPruner(corpus_path)
    
    # Analyze all frames
    stats = pruner.analyze_all(max_duplicates=10, min_length=25)
    
    # Report
    pruner.report_issues(top_n=5)
    
    # Dry run
    result = pruner.apply_pruning(dry_run=True)
    
    print("\n" + "=" * 70)
    print("To apply pruning, run:")
    print("  python3 experiments/corpus_pruner.py --apply")
    print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prune bad data from corpus")
    parser.add_argument("--corpus", default="truthspace_lcm/corpus_experimental.json",
                       help="Path to corpus file")
    parser.add_argument("--apply", action="store_true",
                       help="Actually apply pruning (default is dry run)")
    parser.add_argument("--max-duplicates", type=int, default=10,
                       help="Max copies of identical frames to keep")
    parser.add_argument("--min-length", type=int, default=25,
                       help="Minimum frame length")
    parser.add_argument("--qwen2", action="store_true",
                       help="Use Qwen2 for smart analysis")
    parser.add_argument("--qwen2-sample", type=int, default=100,
                       help="Number of frames to analyze with Qwen2")
    
    args = parser.parse_args()
    
    pruner = CorpusPruner(args.corpus)
    
    # Analyze
    pruner.analyze_all(max_duplicates=args.max_duplicates, min_length=args.min_length)
    
    # Qwen2 analysis if requested
    if args.qwen2:
        pruner.analyze_with_qwen2(sample_size=args.qwen2_sample)
    
    # Report
    pruner.report_issues()
    
    # Apply
    pruner.apply_pruning(dry_run=not args.apply)


if __name__ == "__main__":
    main()
