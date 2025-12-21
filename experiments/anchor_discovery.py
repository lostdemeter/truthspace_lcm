#!/usr/bin/env python3
"""
Anchor Auto-Discovery Experiment

Research question: Can we automatically discover anchors from unlabeled Q&A text?

Hypothesis: High-frequency words that appear in response positions 
(after "Use", "Try", "Run") are likely anchors.

This is a key step toward fully automated domain ingestion.
"""

import re
import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_anchors_by_action_verbs(qa_pairs: List[Tuple[str, str]], 
                                      top_k: int = 20) -> List[Tuple[str, int]]:
    """
    Discover anchors by finding words that follow action verbs in responses.
    
    Pattern: "Use X", "Try X", "Run X" -> X is likely an anchor.
    """
    action_verbs = {'use', 'try', 'run', 'execute', 'apply', 'type', 'enter', 'call'}
    candidates = Counter()
    
    for q, a in qa_pairs:
        words = re.findall(r'\w+', a.lower())
        for i, word in enumerate(words):
            if word in action_verbs and i + 1 < len(words):
                next_word = words[i + 1]
                # Skip common articles and prepositions
                if next_word not in {'the', 'a', 'an', 'to', 'for', 'with', 'your'}:
                    candidates[next_word] += 1
    
    return candidates.most_common(top_k)


def discover_anchors_by_quotes(qa_pairs: List[Tuple[str, str]], 
                                top_k: int = 20) -> List[Tuple[str, int]]:
    """
    Discover anchors by finding words inside quotes in responses.
    
    Pattern: 'command' or `command` -> command is likely an anchor.
    """
    candidates = Counter()
    
    for q, a in qa_pairs:
        # Find quoted content
        quoted = re.findall(r"'([^']+)'", a) + re.findall(r"`([^`]+)`", a)
        for content in quoted:
            # Take first word of quoted content
            words = content.split()
            if words:
                first_word = words[0].lower()
                # Skip if it's a flag or path
                if not first_word.startswith('-') and not first_word.startswith('/'):
                    candidates[first_word] += 1
    
    return candidates.most_common(top_k)


def discover_anchors_by_frequency_ratio(qa_pairs: List[Tuple[str, str]], 
                                         top_k: int = 20) -> List[Tuple[str, float]]:
    """
    Discover anchors by finding words with high response/question frequency ratio.
    
    Anchors appear more in answers than questions.
    """
    q_counts = Counter()
    a_counts = Counter()
    
    for q, a in qa_pairs:
        q_words = set(re.findall(r'\w+', q.lower()))
        a_words = set(re.findall(r'\w+', a.lower()))
        
        for w in q_words:
            q_counts[w] += 1
        for w in a_words:
            a_counts[w] += 1
    
    # Compute ratio: answer_freq / (question_freq + 1)
    # High ratio = appears much more in answers than questions
    ratios = {}
    for word, a_count in a_counts.items():
        if a_count >= 2:  # Minimum frequency
            q_count = q_counts.get(word, 0)
            ratio = a_count / (q_count + 1)
            ratios[word] = ratio
    
    # Filter out common words
    stopwords = {'the', 'a', 'an', 'to', 'for', 'with', 'your', 'is', 'are', 
                 'or', 'and', 'of', 'in', 'on', 'it', 'you', 'can', 'will'}
    ratios = {w: r for w, r in ratios.items() if w not in stopwords}
    
    sorted_ratios = sorted(ratios.items(), key=lambda x: -x[1])
    return sorted_ratios[:top_k]


def discover_anchors_combined(qa_pairs: List[Tuple[str, str]], 
                               top_k: int = 15) -> List[str]:
    """
    Combine multiple discovery methods for robust anchor detection.
    
    Anchors that appear in multiple methods are more likely to be real.
    """
    # Get candidates from each method
    by_verbs = {w for w, c in discover_anchors_by_action_verbs(qa_pairs, 30)}
    by_quotes = {w for w, c in discover_anchors_by_quotes(qa_pairs, 30)}
    by_ratio = {w for w, r in discover_anchors_by_frequency_ratio(qa_pairs, 30)}
    
    # Score by how many methods found each word
    scores = Counter()
    for word in by_verbs | by_quotes | by_ratio:
        if word in by_verbs:
            scores[word] += 1
        if word in by_quotes:
            scores[word] += 1
        if word in by_ratio:
            scores[word] += 1
    
    # Return words found by at least 2 methods
    combined = [(w, s) for w, s in scores.most_common() if s >= 2]
    return [w for w, s in combined[:top_k]]


def evaluate_discovery(discovered: List[str], ground_truth: Set[str]) -> dict:
    """Evaluate discovered anchors against ground truth."""
    discovered_set = set(discovered)
    
    true_positives = discovered_set & ground_truth
    false_positives = discovered_set - ground_truth
    false_negatives = ground_truth - discovered_set
    
    precision = len(true_positives) / len(discovered_set) if discovered_set else 0
    recall = len(true_positives) / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }


def main():
    print("=" * 70)
    print("ANCHOR AUTO-DISCOVERY EXPERIMENT")
    print("=" * 70)
    print()
    
    # Test data: bash Q&A pairs
    bash_qa = [
        ("How do I list files?", "Use 'ls' to list files in a directory."),
        ("Show all files including hidden", "Use 'ls -la' to show all files including hidden ones."),
        ("How do I show disk space?", "Use 'df -h' to show disk space usage."),
        ("Check disk usage", "Use 'df -h' to check how much disk space is used."),
        ("What processes are running?", "Use 'ps aux' to see running processes."),
        ("Show running processes", "Use 'ps aux' or 'top' to view processes."),
        ("How do I check network connections?", "Use 'netstat -tuln' to check network connections."),
        ("Show network status", "Use 'netstat' or 'ss' to see network status."),
        ("Who is logged in?", "Use 'who' to see logged in users."),
        ("Show logged in users", "Use 'who' or 'w' to display logged in users."),
        ("What's my IP address?", "Use 'ip addr' to show your IP addresses."),
        ("Show network interfaces", "Use 'ip addr' or 'ifconfig' to show interfaces."),
        ("Find a file", "Use 'find' to search for files by name."),
        ("Search file contents", "Use 'grep' to search inside files."),
        ("Copy a file", "Use 'cp source dest' to copy files."),
        ("Move a file", "Use 'mv source dest' to move or rename files."),
        ("Delete a file", "Use 'rm filename' to delete files."),
        ("Create a directory", "Use 'mkdir dirname' to create a directory."),
        ("Show file contents", "Use 'cat filename' to display file contents."),
        ("Edit a file", "Use 'nano' or 'vim' to edit files."),
    ]
    
    # Ground truth anchors
    ground_truth = {'ls', 'df', 'ps', 'top', 'netstat', 'ss', 'who', 'w', 
                    'ip', 'ifconfig', 'find', 'grep', 'cp', 'mv', 'rm', 
                    'mkdir', 'cat', 'nano', 'vim'}
    
    print("Test data: 20 bash Q&A pairs")
    print(f"Ground truth anchors: {len(ground_truth)}")
    print()
    
    # Method 1: Action verbs
    print("-" * 70)
    print("METHOD 1: Action Verb Followers")
    print("-" * 70)
    by_verbs = discover_anchors_by_action_verbs(bash_qa, 15)
    print("Discovered:", [w for w, c in by_verbs])
    eval1 = evaluate_discovery([w for w, c in by_verbs], ground_truth)
    print(f"Precision: {eval1['precision']:.2f}, Recall: {eval1['recall']:.2f}, F1: {eval1['f1']:.2f}")
    print()
    
    # Method 2: Quoted content
    print("-" * 70)
    print("METHOD 2: Quoted Content")
    print("-" * 70)
    by_quotes = discover_anchors_by_quotes(bash_qa, 15)
    print("Discovered:", [w for w, c in by_quotes])
    eval2 = evaluate_discovery([w for w, c in by_quotes], ground_truth)
    print(f"Precision: {eval2['precision']:.2f}, Recall: {eval2['recall']:.2f}, F1: {eval2['f1']:.2f}")
    print()
    
    # Method 3: Frequency ratio
    print("-" * 70)
    print("METHOD 3: Answer/Question Frequency Ratio")
    print("-" * 70)
    by_ratio = discover_anchors_by_frequency_ratio(bash_qa, 15)
    print("Discovered:", [w for w, r in by_ratio])
    eval3 = evaluate_discovery([w for w, r in by_ratio], ground_truth)
    print(f"Precision: {eval3['precision']:.2f}, Recall: {eval3['recall']:.2f}, F1: {eval3['f1']:.2f}")
    print()
    
    # Combined method
    print("-" * 70)
    print("COMBINED METHOD (intersection of 2+ methods)")
    print("-" * 70)
    combined = discover_anchors_combined(bash_qa, 15)
    print("Discovered:", combined)
    eval_combined = evaluate_discovery(combined, ground_truth)
    print(f"Precision: {eval_combined['precision']:.2f}, Recall: {eval_combined['recall']:.2f}, F1: {eval_combined['f1']:.2f}")
    print()
    print(f"True positives: {eval_combined['true_positives']}")
    print(f"False positives: {eval_combined['false_positives']}")
    print(f"Missed: {eval_combined['false_negatives']}")
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The combined method achieves high precision by requiring agreement")
    print("between multiple heuristics. This is suitable for bootstrapping a")
    print("new domain without manual anchor specification.")
    print()
    print("For production use:")
    print("1. Start with auto-discovered anchors")
    print("2. Let the system learn from usage")
    print("3. Optionally refine with human review")


if __name__ == "__main__":
    main()
