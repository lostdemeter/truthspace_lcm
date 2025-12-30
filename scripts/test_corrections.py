#!/usr/bin/env python3
"""
Test Correction Learning

This script tests the correction learning mechanism by:
1. Recording current answers to test questions
2. Applying corrections
3. Saving the corpus
4. Reloading and verifying improvement

Author: Lesley Gushurst
License: GPLv3
"""

import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.correction_learner import CorrectionLearner


def test_correction_learning():
    """Test that corrections actually improve answers."""
    
    corpus_path = 'truthspace_lcm/corpus_self_improved.json'
    backup_path = 'truthspace_lcm/corpus_self_improved.backup.json'
    
    # Backup corpus first
    print("Backing up corpus...")
    shutil.copy(corpus_path, backup_path)
    
    try:
        learner = CorrectionLearner()
        learner.load_corpus(corpus_path)
        
        print("\n" + "=" * 70)
        print("CORRECTION LEARNING TEST")
        print("=" * 70)
        
        # Define test corrections
        # Format: (question, hypothetical_wrong, correct_answer)
        # The "wrong" answer is what we're correcting FROM (may not be current output)
        test_cases = [
            {
                'question': "Who is Holmes?",
                'wrong': "Holmes is a person",
                'correct': "Holmes is a consulting detective who solves mysteries",
            },
            {
                'question': "What does Watson do?",
                'wrong': "Watson does things",
                'correct': "Watson is a doctor who assists Holmes",
            },
        ]
        
        results = []
        
        for tc in test_cases:
            print(f"\n--- Testing: {tc['question']} ---")
            
            # Get current answer
            before = learner.test_answer(tc['question'])
            print(f"  BEFORE: {before[:80]}...")
            
            # Apply correction
            learner.correct(tc['question'], tc['wrong'], tc['correct'])
        
        # Save corpus with corrections
        print("\n--- Saving corpus with corrections ---")
        learner.save_corpus(corpus_path)
        
        # Reload and test
        print("\n--- Reloading corpus ---")
        learner.reload_qa()
        
        print("\n--- Results After Correction ---")
        for tc in test_cases:
            after = learner.test_answer(tc['question'])
            print(f"\n  Q: {tc['question']}")
            print(f"  TARGET: {tc['correct']}")
            print(f"  AFTER:  {after[:80]}...")
            
            # Check for improvement
            target_words = set(tc['correct'].lower().split()) - learner.SKIP_WORDS
            matches = sum(1 for w in target_words if w in after.lower())
            print(f"  Matches: {matches}/{len(target_words)} target words")
        
        # Save correction log
        learner.save_corrections()
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        print(f"\nCorrections saved to: {learner.correction_log_path}")
        print(f"Corpus updated: {corpus_path}")
        print(f"Backup at: {backup_path}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Restoring backup...")
        shutil.copy(backup_path, corpus_path)
        raise


def restore_backup():
    """Restore corpus from backup."""
    corpus_path = 'truthspace_lcm/corpus_self_improved.json'
    backup_path = 'truthspace_lcm/corpus_self_improved.backup.json'
    
    if Path(backup_path).exists():
        shutil.copy(backup_path, corpus_path)
        print(f"Restored {corpus_path} from backup")
    else:
        print("No backup found")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true', help='Restore from backup')
    args = parser.parse_args()
    
    if args.restore:
        restore_backup()
    else:
        test_correction_learning()
