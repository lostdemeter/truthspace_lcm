"""
Test the pure geometric translator with harder queries.

Categories of difficulty:
1. Novel phrasing (words not in bootstrap)
2. Indirect requests (implied actions)
3. Compound queries (multiple actions)
4. Ambiguous queries (could map to multiple commands)
5. Negation and edge cases

Author: Lesley Gushurst
License: GPLv3
"""

from pathlib import Path
from nl_to_bash_pure import PureGeometricTranslator


def test_category(translator, category_name: str, queries: list, expected: dict):
    """Test a category of queries and report results."""
    print(f"\n--- {category_name} ---")
    correct = 0
    total = len(queries)
    
    for query in queries:
        results = translator.translate(query, top_k=3)
        
        if results:
            best_cmd, confidence = results[0]
            base_cmd = best_cmd.split()[0]
            
            # Check correctness
            exp = expected.get(query, "")
            is_correct = base_cmd == exp or exp in best_cmd
            mark = "✓" if is_correct else "✗"
            if is_correct:
                correct += 1
            
            # Format alternatives
            alts = [f"{cmd}({c:.2f})" for cmd, c in results[1:3]]
            alt_str = f" | {', '.join(alts)}" if alts else ""
            
            print(f"  {mark} '{query}'")
            print(f"      → {best_cmd} ({confidence:.2f}){alt_str}")
        else:
            print(f"  ✗ '{query}' → NO MATCH")
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    return correct, total


def main():
    print("=== Harder Query Tests ===")
    print("Testing the pure geometric translator with challenging queries.")
    print()
    
    # Load translator
    bootstrap_path = Path(__file__).parent / "bootstrap" / "nl_bash_mappings.json"
    translator = PureGeometricTranslator(dims=12)
    translator.load_bootstrap(str(bootstrap_path))
    
    print(f"Loaded {len(translator.mappings)} mappings")
    
    total_correct = 0
    total_queries = 0
    
    # ==========================================================================
    # Category 1: Novel Phrasing
    # Words that don't appear in the bootstrap but mean similar things
    # ==========================================================================
    novel_queries = [
        "enumerate files",           # enumerate ≈ list
        "exhibit files",             # exhibit ≈ show
        "reveal hidden files",       # reveal ≈ show
        "terminate application",     # application ≈ process
        "erase file",                # erase ≈ delete
        "inspect memory",            # inspect ≈ check
        "examine disk",              # examine ≈ check
        "spawn directory",           # spawn ≈ create
        "fabricate folder",          # fabricate ≈ create
        "locate files",              # locate ≈ find
    ]
    novel_expected = {
        "enumerate files": "ls",
        "exhibit files": "ls",
        "reveal hidden files": "ls",
        "terminate application": "kill",
        "erase file": "rm",
        "inspect memory": "free",
        "examine disk": "df",
        "spawn directory": "mkdir",
        "fabricate folder": "mkdir",
        "locate files": "find",
    }
    c, t = test_category(translator, "Novel Phrasing", novel_queries, novel_expected)
    total_correct += c
    total_queries += t
    
    # ==========================================================================
    # Category 2: Indirect Requests
    # Implied actions without explicit verbs
    # ==========================================================================
    indirect_queries = [
        "files here",                # implied: list
        "directory contents",        # implied: show
        "what processes",            # implied: list
        "current memory",            # implied: show
        "available disk",            # implied: show
        "my ip",                     # implied: show
        "open ports",                # implied: list
        "file count",                # implied: count
        "folder structure",          # implied: list recursively
    ]
    indirect_expected = {
        "files here": "ls",
        "directory contents": "ls",
        "what processes": "ps",
        "current memory": "free",
        "available disk": "df",
        "my ip": "ip",
        "open ports": "ss",
        "file count": "wc",
        "folder structure": "ls",
    }
    c, t = test_category(translator, "Indirect Requests", indirect_queries, indirect_expected)
    total_correct += c
    total_queries += t
    
    # ==========================================================================
    # Category 3: Compound Queries
    # Multiple concepts in one query
    # ==========================================================================
    compound_queries = [
        "list files and show details",
        "find and delete files",
        "show processes and memory",
        "create directory and file",
        "check disk and memory usage",
    ]
    compound_expected = {
        "list files and show details": "ls",
        "find and delete files": "find",
        "show processes and memory": "ps",
        "create directory and file": "mkdir",
        "check disk and memory usage": "df",
    }
    c, t = test_category(translator, "Compound Queries", compound_queries, compound_expected)
    total_correct += c
    total_queries += t
    
    # ==========================================================================
    # Category 4: Conversational/Natural
    # How a human might actually ask
    # ==========================================================================
    conversational_queries = [
        "hey can you show me the files",
        "I need to see what's running",
        "how do I delete this folder",
        "what's using my memory",
        "is there enough disk space",
        "I want to create a new folder",
        "help me find a file",
        "can you kill that process",
        "what's my network address",
        "show me everything in this directory",
    ]
    conversational_expected = {
        "hey can you show me the files": "ls",
        "I need to see what's running": "ps",
        "how do I delete this folder": "rm",
        "what's using my memory": "free",
        "is there enough disk space": "df",
        "I want to create a new folder": "mkdir",
        "help me find a file": "find",
        "can you kill that process": "kill",
        "what's my network address": "ip",
        "show me everything in this directory": "ls",
    }
    c, t = test_category(translator, "Conversational/Natural", conversational_queries, conversational_expected)
    total_correct += c
    total_queries += t
    
    # ==========================================================================
    # Category 5: Minimal Queries
    # Very short, terse queries
    # ==========================================================================
    minimal_queries = [
        "files",
        "processes",
        "memory",
        "disk",
        "ports",
        "ip",
        "kill",
        "delete",
        "create",
        "find",
    ]
    minimal_expected = {
        "files": "ls",
        "processes": "ps",
        "memory": "free",
        "disk": "df",
        "ports": "ss",
        "ip": "ip",
        "kill": "kill",
        "delete": "rm",
        "create": "mkdir",
        "find": "find",
    }
    c, t = test_category(translator, "Minimal Queries", minimal_queries, minimal_expected)
    total_correct += c
    total_queries += t
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*60)
    overall_accuracy = 100 * total_correct / total_queries if total_queries > 0 else 0
    print(f"OVERALL: {total_correct}/{total_queries} ({overall_accuracy:.0f}%)")
    print("="*60)
    
    # Analysis
    print("\n--- Analysis ---")
    if overall_accuracy >= 80:
        print("✓ Good performance on harder queries!")
        print("  The geometric approach generalizes beyond exact matches.")
    elif overall_accuracy >= 50:
        print("△ Moderate performance on harder queries.")
        print("  Some generalization, but struggles with novel phrasing.")
    else:
        print("✗ Poor performance on harder queries.")
        print("  The system is too dependent on exact word matches.")
    
    print("\n--- Key Observations ---")
    print("1. Word overlap (Jaccard) requires SOME word match")
    print("2. Novel words with no overlap will fail")
    print("3. This is expected - we need a way to handle unknown words")
    print("4. Options: word embeddings, LLM bootstrap, or learned word positions")


if __name__ == "__main__":
    main()
