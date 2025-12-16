#!/usr/bin/env python3
"""
Test 12D Bash Knowledge
=======================

Comprehensive test of bash knowledge with 12D plastic-primary encoding.
Tests various natural language queries to verify correct resolution.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm import TruthSpace, Resolver


def test_bash_queries():
    """Test a comprehensive set of bash queries."""
    
    ts = TruthSpace()
    resolver = Resolver(ts)
    
    # Test cases: (query, expected_substring_in_output)
    test_cases = [
        # File operations
        ("list files in directory", "ls"),
        ("show files", "ls"),
        ("create a new file called test.txt", "touch"),
        ("create directory called mydir", "mkdir"),
        ("delete file test.txt", "rm"),
        ("remove directory mydir", "rm"),
        ("copy file.txt to backup.txt", "cp"),
        ("move file.txt to /tmp", "mv"),
        
        # File viewing
        ("view contents of file.txt", "cat"),
        ("show first 10 lines of log.txt", "head"),
        ("show last 20 lines of log.txt", "tail"),
        ("follow the log file", "tail -f"),
        
        # Search operations
        ("find all python files", "find"),
        ("search for error in log files", "grep"),
        ("find files named config", "find"),
        
        # Compression
        ("compress the logs folder", "tar"),
        ("extract archive.tar.gz", "tar"),
        
        # System info
        ("show disk space", "df"),
        ("directory size of /home", "du"),
        ("current directory", "pwd"),
        
        # Process management
        ("list running processes", "ps"),
        ("kill process 1234", "kill"),
        
        # Counting
        ("count lines in file.txt", "wc -l"),
        ("count words in document.txt", "wc -w"),
        
        # Network/Remote
        ("download file from http://example.com/file.zip", "curl"),
        ("ssh into server.com", "ssh"),
        ("copy file to remote server", "scp"),
        
        # Permissions
        ("make script.sh executable", "chmod"),
        
        # Sorting
        ("sort the file alphabetically", "sort"),
        ("get unique lines from file", "uniq"),
    ]
    
    print("=" * 70)
    print("12D BASH KNOWLEDGE TEST")
    print("=" * 70)
    print(f"\nTesting {len(test_cases)} queries...\n")
    
    passed = 0
    failed = 0
    results = []
    
    for query, expected in test_cases:
        try:
            result = resolver.resolve(query)
            output = result.output
            
            if expected in output:
                status = "✅"
                passed += 1
            else:
                status = "❌"
                failed += 1
            
            results.append({
                "query": query,
                "expected": expected,
                "got": output,
                "matched": result.knowledge.name,
                "passed": expected in output,
            })
            
            print(f"{status} '{query[:40]:<40}' → {output[:30]}")
            
        except Exception as e:
            status = "❌"
            failed += 1
            results.append({
                "query": query,
                "expected": expected,
                "got": str(e),
                "matched": "ERROR",
                "passed": False,
            })
            print(f"{status} '{query[:40]:<40}' → ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    print(f"  Failed: {failed}/{len(test_cases)}")
    print(f"  Success Rate: {passed/len(test_cases)*100:.1f}%")
    
    # Show failures
    if failed > 0:
        print("\n" + "-" * 70)
        print("FAILURES:")
        print("-" * 70)
        for r in results:
            if not r["passed"]:
                print(f"\n  Query: '{r['query']}'")
                print(f"  Expected: '{r['expected']}' in output")
                print(f"  Got: '{r['got']}'")
                print(f"  Matched: {r['matched']}")
    
    print("\n" + "=" * 70)
    
    return passed, failed, results


if __name__ == "__main__":
    passed, failed, results = test_bash_queries()
    sys.exit(0 if failed == 0 else 1)
