#!/usr/bin/env python3
"""
Test suite for compound phrase resolution and parameter extraction.

Tests various natural language patterns that users might type for
common operations, validating both concept extraction and parameter detection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm import TruthSpace


# =============================================================================
# TEST CASES
# =============================================================================

# Format: (query, expected_commands, expected_params)
# expected_commands: list of command bases (without params) - can include alternatives as tuples
# expected_params: list of lists of params for each command

COMPOUND_TESTS = [
    # ---------------------------------------------------------------------
    # BASIC COMPOUND QUERIES (two commands)
    # ---------------------------------------------------------------------
    (
        "create directory mydir and create file readme.txt",
        ["mkdir -p", "touch"],
        [["mydir"], ["readme.txt"]],
    ),
    (
        "list files and show disk space",
        ["ls", "df"],
        [[], []],
    ),
    (
        "show processes and show memory",
        [("ps", "top"), "free"],  # ps or top acceptable
        [[], []],
    ),
    
    # ---------------------------------------------------------------------
    # NATURAL LANGUAGE VARIATIONS
    # ---------------------------------------------------------------------
    (
        "create a directory called myproject",
        ["mkdir -p"],
        [["myproject"]],
    ),
    (
        "list files",
        ["ls"],
        [[]],
    ),
    (
        "show disk space",
        ["df"],
        [[]],
    ),
    (
        "show running processes",
        [("ps", "top")],
        [[]],
    ),
    
    # ---------------------------------------------------------------------
    # COMPOUND WITH PARAMETERS
    # ---------------------------------------------------------------------
    (
        "create a directory called src and create a file called main.py",
        ["mkdir -p", "touch"],
        [["src"], ["main.py"]],
    ),
    (
        "make directory backup and copy file data.txt",
        ["mkdir -p", "cp"],
        [["backup"], ["data.txt"]],
    ),
    (
        "create folder config and create file settings.json",
        ["mkdir -p", "touch"],
        [["config"], ["settings.json"]],
    ),
    
    # ---------------------------------------------------------------------
    # THREE OR MORE COMMANDS
    # ---------------------------------------------------------------------
    (
        "create directory src and create file app.py and list files",
        ["mkdir -p", "touch", "ls"],
        [["src"], ["app.py"], []],
    ),
    (
        "show disk space then show memory then show processes",
        ["df", "free", ("ps", "top")],
        [[], [], []],
    ),
    
    # ---------------------------------------------------------------------
    # FILE OPERATIONS
    # ---------------------------------------------------------------------
    (
        "copy file report.pdf",
        ["cp"],
        [["report.pdf"]],
    ),
    (
        "move file old.txt and delete file temp.log",
        ["mv", "rm"],
        [["old.txt"], ["temp.log"]],
    ),
    (
        "find file config.yaml",
        ["find"],
        [["config.yaml"]],
    ),
    
    # ---------------------------------------------------------------------
    # SYSTEM OPERATIONS
    # ---------------------------------------------------------------------
    (
        "check disk space and show system",
        ["df", "uname"],
        [[], []],
    ),
    (
        "show date and show uptime",
        ["date", "uptime"],
        [[], []],
    ),
    (
        "display hostname and show user",
        ["hostname", "whoami"],
        [[], []],
    ),
    
    # ---------------------------------------------------------------------
    # PROCESS OPERATIONS
    # ---------------------------------------------------------------------
    (
        "list processes and show memory",
        [("ps", "pstree"), "free"],
        [[], []],
    ),
    (
        "show processes and show disk space",
        [("ps", "top"), "df"],
        [[], []],
    ),
    
    # ---------------------------------------------------------------------
    # NETWORK OPERATIONS
    # ---------------------------------------------------------------------
    (
        "download url",
        [("curl", "wget")],
        [[]],
    ),
    (
        "test network and show network",
        ["ping", "ifconfig"],
        [[], []],
    ),
    
    # ---------------------------------------------------------------------
    # DATA OPERATIONS
    # ---------------------------------------------------------------------
    (
        "count file and sort output",
        ["wc", "sort"],
        [[], []],
    ),
    (
        "search text in files",
        ["grep"],
        [[]],
    ),
    
    # ---------------------------------------------------------------------
    # EDGE CASES
    # ---------------------------------------------------------------------
    (
        "create directory my-project",
        ["mkdir -p"],
        [["my-project"]],
    ),
    (
        "create file /tmp/output.txt",
        ["touch"],
        [["/tmp/output.txt"]],
    ),
]


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests():
    """Run all compound query tests."""
    ts = TruthSpace()
    
    print("=" * 70)
    print("COMPOUND QUERY & PARAMETER TEST SUITE")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    results = []
    
    for query, expected_cmds, expected_params in COMPOUND_TESTS:
        concepts = ts.resolve_with_params(query)
        
        # Extract actual commands and params
        actual_cmds = [c['command'] for c in concepts]
        actual_params = [c['params'] for c in concepts]
        
        # Check commands match (support alternatives as tuples)
        cmd_match = True
        if len(actual_cmds) != len(expected_cmds):
            cmd_match = False
        else:
            for actual, expected in zip(actual_cmds, expected_cmds):
                if isinstance(expected, tuple):
                    if actual not in expected:
                        cmd_match = False
                        break
                else:
                    if actual != expected:
                        cmd_match = False
                        break
        
        # Check params match (be lenient - just check expected params are present)
        param_match = True
        for i, exp_params in enumerate(expected_params):
            if i >= len(actual_params):
                if exp_params:  # Only fail if we expected params
                    param_match = False
                    break
            else:
                for exp_p in exp_params:
                    if exp_p not in actual_params[i]:
                        param_match = False
                        break
        
        success = cmd_match and param_match
        
        if success:
            passed += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"
        
        results.append({
            'query': query,
            'expected_cmds': expected_cmds,
            'actual_cmds': actual_cmds,
            'expected_params': expected_params,
            'actual_params': actual_params,
            'success': success,
            'cmd_match': cmd_match,
            'param_match': param_match,
        })
    
    # Print results
    for r in results:
        if r['success']:
            print(f"{status} \"{r['query'][:50]}...\"" if len(r['query']) > 50 else f"✓ \"{r['query']}\"")
        else:
            print(f"✗ \"{r['query']}\"")
            print(f"    Expected: {r['expected_cmds']} with {r['expected_params']}")
            print(f"    Got:      {r['actual_cmds']} with {r['actual_params']}")
            if not r['cmd_match']:
                print(f"    Issue: Command mismatch")
            if not r['param_match']:
                print(f"    Issue: Parameter mismatch")
        print()
    
    # Summary
    print("=" * 70)
    print(f"RESULTS: {passed}/{passed + failed} passed ({100*passed/(passed+failed):.0f}%)")
    print("=" * 70)
    
    return passed, failed


def run_single_test(query: str):
    """Run a single test query with detailed output."""
    ts = TruthSpace()
    
    print(f"Query: \"{query}\"")
    print("-" * 50)
    
    # Show tokenization
    tokens = ts._tokenize(query)
    print(f"Tokens: {tokens}")
    print()
    
    # Show geometric parameter detection
    params = ts.detect_parameters_geometric(query)
    print("Geometric parameters:")
    for idx, value, reason in params:
        print(f"  [{idx}] \"{value}\" ({reason})")
    print()
    
    # Show concept extraction
    concepts = ts.resolve_compound(query)
    print("Concepts:")
    for c in concepts:
        print(f"  [{c['start']}-{c['end']}] \"{c['window']}\" → {c['command']} ({c['similarity']:.2f})")
    print()
    
    # Show final result
    concepts = ts.resolve_with_params(query)
    commands = []
    for c in concepts:
        cmd = c['command']
        if c['params']:
            cmd += ' ' + ' '.join(c['params'])
        commands.append(cmd)
    
    print(f"Result: {' && '.join(commands)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        run_single_test(query)
    else:
        # Run all tests
        run_tests()
