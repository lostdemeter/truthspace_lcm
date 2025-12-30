#!/usr/bin/env python3
"""
Corpus Role Fixer

Fixes incorrect roles in the corpus by:
1. Identifying concepts with wrong roles (character for abstract concepts)
2. Adjusting role counts in the knowledge base
3. Optionally using Qwen2 to determine correct roles
4. Saving the fixed corpus

This modifies the underlying data, not just the projection.

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
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.geometric import GeometricQA, GeometricKnowledge

try:
    from experiments.ollama_corpus_refiner import OllamaClient
except ImportError:
    OllamaClient = None


@dataclass
class RoleFix:
    """A role fix to apply."""
    concept: str
    old_role: str
    new_role: str
    reason: str
    confidence: float


class CorpusRoleFixer:
    """
    Fixes incorrect roles in the corpus.
    """
    
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        
        # Load corpus via GeometricQA
        self.qa = GeometricQA()
        self.qa.load_corpus(corpus_path)
        self.qa.set_output_lens('natural')
        
        print(f"Loaded {len(self.qa.knowledge.concepts)} concepts")
        
        # Qwen2 client for smart role detection
        self.ollama = OllamaClient() if OllamaClient else None
        if self.ollama and not self.ollama.is_available():
            print("WARNING: Ollama not available. Using rule-based fixing only.")
            self.ollama = None
        
        # Role detection rules
        self.abstract_markers = ['ology', 'tion', 'ment', 'ness', 'ism', 'ics', 'istry', 'ure', 'ance', 'ence']
        self.person_names = {'holmes', 'watson', 'moriarty', 'lestrade', 'mycroft', 'irene', 
                            'darwin', 'einstein', 'newton', 'galileo', 'aristotle', 'plato'}
        
        # Fixes to apply
        self.fixes: List[RoleFix] = []
        
        # Stats
        self.stats = {
            'analyzed': 0,
            'wrong_role': 0,
            'fixed': 0,
        }
    
    def detect_correct_role(self, concept: str) -> Tuple[str, str, float]:
        """
        Detect the correct role for a concept.
        
        Returns (role, reason, confidence)
        """
        concept_lower = concept.lower()
        
        # Check if it's a known person
        if concept_lower in self.person_names:
            return 'character', 'known_person', 1.0
        
        # Check for person name patterns (capitalized, no abstract markers)
        if concept[0].isupper() and len(concept) > 2:
            # Likely a proper noun - could be person or place
            has_abstract = any(m in concept_lower for m in self.abstract_markers)
            if not has_abstract and not concept_lower.endswith('s'):
                # Could be a person name - use Qwen2 if available
                if self.ollama:
                    return self._qwen2_detect_role(concept)
                return 'entity', 'proper_noun', 0.6
        
        # Check for abstract concept markers
        for marker in self.abstract_markers:
            if concept_lower.endswith(marker) or marker in concept_lower:
                return 'concept', f'abstract_marker:{marker}', 0.9
        
        # Check for plural (likely not a character)
        if concept_lower.endswith('s') and concept_lower not in self.person_names:
            # Plurals are usually concepts or things, not characters
            if concept_lower.endswith('ies'):
                return 'concept', 'plural_ies', 0.8
            elif concept_lower.endswith('es'):
                return 'concept', 'plural_es', 0.7
            elif len(concept_lower) > 3:
                return 'concept', 'plural_s', 0.7
        
        # Default - use Qwen2 if available
        if self.ollama:
            return self._qwen2_detect_role(concept)
        
        return 'entity', 'default', 0.5
    
    def _qwen2_detect_role(self, concept: str) -> Tuple[str, str, float]:
        """Use Qwen2 to detect the correct role."""
        prompt = f"""What type of thing is "{concept}"?

Respond with ONLY one of these categories:
- PERSON: A human being (real or fictional)
- CONCEPT: An abstract idea, field of study, or process
- THING: A physical object or place
- EVENT: Something that happens

Just respond with the category name, nothing else."""

        response = self.ollama.generate(prompt, temperature=0.1)
        
        if not response:
            return 'entity', 'qwen2_no_response', 0.5
        
        response = response.strip().upper()
        
        if 'PERSON' in response:
            return 'character', 'qwen2:person', 0.85
        elif 'CONCEPT' in response:
            return 'concept', 'qwen2:concept', 0.85
        elif 'THING' in response:
            return 'entity', 'qwen2:thing', 0.85
        elif 'EVENT' in response:
            return 'concept', 'qwen2:event', 0.85
        else:
            return 'entity', f'qwen2:unknown:{response[:20]}', 0.5
    
    def analyze_concepts(self, limit: int = None) -> List[RoleFix]:
        """
        Analyze all concepts and identify role fixes needed.
        """
        fixes = []
        
        concepts = list(self.qa.knowledge.concepts.items())
        if limit:
            concepts = concepts[:limit]
        
        print(f"\nAnalyzing {len(concepts)} concepts...")
        
        for i, (name, concept) in enumerate(concepts):
            if not concept.is_content_word:
                continue
            
            self.stats['analyzed'] += 1
            
            # Get current role from answer
            answer = self.qa.ask(f"What is {name}?")
            current_role = self._extract_role(answer)
            
            # Detect correct role
            correct_role, reason, confidence = self.detect_correct_role(name)
            
            # Check if fix needed
            if current_role in ['character', 'someone', 'protagonist']:
                if correct_role in ['concept', 'entity']:
                    self.stats['wrong_role'] += 1
                    fixes.append(RoleFix(
                        concept=name,
                        old_role=current_role,
                        new_role=correct_role,
                        reason=reason,
                        confidence=confidence,
                    ))
            
            # Progress
            if (i + 1) % 500 == 0:
                print(f"  Analyzed {i + 1}/{len(concepts)}... ({len(fixes)} fixes needed)")
        
        self.fixes = fixes
        return fixes
    
    def _extract_role(self, answer: str) -> str:
        """Extract role from answer."""
        answer_lower = answer.lower()
        
        if 'is a character' in answer_lower:
            return 'character'
        elif 'is someone' in answer_lower:
            return 'someone'
        elif 'is a protagonist' in answer_lower:
            return 'protagonist'
        elif 'is a concept' in answer_lower:
            return 'concept'
        elif 'is a detective' in answer_lower:
            return 'detective'
        elif 'is a doctor' in answer_lower:
            return 'doctor'
        elif 'is an entity' in answer_lower:
            return 'entity'
        
        # Try to extract from "is a/an X"
        match = re.search(r'is a[n]? (\w+)', answer_lower)
        if match:
            return match.group(1)
        
        return 'unknown'
    
    def apply_fixes(self, dry_run: bool = True, min_confidence: float = 0.7) -> Dict:
        """
        Apply the role fixes to the knowledge base.
        """
        if not self.fixes:
            print("No fixes to apply.")
            return {}
        
        # Filter by confidence
        applicable = [f for f in self.fixes if f.confidence >= min_confidence]
        print(f"\nApplying {len(applicable)} fixes (confidence >= {min_confidence})...")
        
        applied = 0
        for fix in applicable:
            if fix.concept in self.qa.knowledge.concepts:
                concept = self.qa.knowledge.concepts[fix.concept]
                
                # Adjust role counts
                # Boost the correct role, reduce the incorrect one
                boost = 50  # Strong boost to override existing data
                
                if fix.new_role == 'concept':
                    concept.mediator_count += boost
                    concept.initiator_count = max(0, concept.initiator_count - boost // 2)
                elif fix.new_role == 'entity':
                    concept.receiver_count += boost
                    concept.initiator_count = max(0, concept.initiator_count - boost // 2)
                elif fix.new_role == 'character':
                    concept.initiator_count += boost
                
                applied += 1
                self.stats['fixed'] += 1
        
        result = {
            'total_fixes': len(self.fixes),
            'applicable': len(applicable),
            'applied': applied,
        }
        
        print(f"\n{'DRY RUN - ' if dry_run else ''}FIX RESULTS:")
        print(f"  Total fixes identified: {result['total_fixes']}")
        print(f"  Applicable (confidence >= {min_confidence}): {result['applicable']}")
        print(f"  Applied: {result['applied']}")
        
        if not dry_run and applied > 0:
            self._save_corpus()
        
        return result
    
    def _save_corpus(self):
        """Save the modified corpus."""
        # Convert knowledge to JSON format
        data = {
            'frames': [
                {
                    'text': f.text,
                    'source': getattr(f, 'source', 'unknown'),
                    'agent': getattr(f, 'initiator', ''),
                }
                for f in self.qa.knowledge.frames
            ]
        }
        
        # Backup
        backup_path = self.corpus_path.replace('.json', '_pre_rolefix.json')
        if os.path.exists(self.corpus_path):
            import shutil
            shutil.copy(self.corpus_path, backup_path)
            print(f"  Backup saved to: {backup_path}")
        
        # Save
        with open(self.corpus_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Fixed corpus saved to: {self.corpus_path}")
    
    def report(self, top_n: int = 20):
        """Print analysis report."""
        print("\n" + "=" * 70)
        print("ROLE FIX REPORT")
        print("=" * 70)
        
        print(f"\nStatistics:")
        print(f"  Analyzed: {self.stats['analyzed']}")
        print(f"  Wrong role: {self.stats['wrong_role']}")
        print(f"  Fixed: {self.stats['fixed']}")
        
        if self.fixes:
            # Group by reason
            by_reason = defaultdict(list)
            for fix in self.fixes:
                by_reason[fix.reason].append(fix)
            
            print(f"\nFixes by reason:")
            for reason, fixes in sorted(by_reason.items(), key=lambda x: -len(x[1])):
                print(f"  {reason}: {len(fixes)}")
            
            print(f"\nSample fixes:")
            for fix in self.fixes[:top_n]:
                print(f"  {fix.concept}: {fix.old_role} → {fix.new_role} ({fix.reason}, conf={fix.confidence:.2f})")


def demo():
    """Demo the role fixer."""
    print("=" * 70)
    print("CORPUS ROLE FIXER")
    print("=" * 70)
    
    corpus_path = "truthspace_lcm/corpus_experimental.json"
    
    fixer = CorpusRoleFixer(corpus_path)
    
    # Analyze (limit for demo)
    fixes = fixer.analyze_concepts(limit=500)
    
    # Report
    fixer.report()
    
    # Dry run
    fixer.apply_fixes(dry_run=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix incorrect roles in corpus")
    parser.add_argument("--corpus", default="truthspace_lcm/corpus_experimental.json")
    parser.add_argument("--limit", type=int, help="Limit number of concepts to analyze")
    parser.add_argument("--apply", action="store_true", help="Apply fixes (default is dry run)")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum confidence for fixes")
    parser.add_argument("--use-qwen2", action="store_true", help="Use Qwen2 for role detection")
    
    args = parser.parse_args()
    
    fixer = CorpusRoleFixer(args.corpus)
    
    if not args.use_qwen2:
        fixer.ollama = None  # Disable Qwen2
    
    fixer.analyze_concepts(limit=args.limit)
    fixer.report()
    fixer.apply_fixes(dry_run=not args.apply, min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
