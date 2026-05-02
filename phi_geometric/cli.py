#!/usr/bin/env python3
"""
φ-Geometric Transformation Engine — Command Line Interface

Discover transformation pipelines from example pairs and execute them.

Usage:
    # Discover from a TSV file (input<TAB>output per line)
    python -m phi_geometric discover pairs.tsv -o pipeline.json

    # Execute a saved pipeline on new inputs
    python -m phi_geometric execute pipeline.json -i "s h i p"

    # Discover and immediately test (no save)
    python -m phi_geometric discover pairs.tsv --test

    # Show info about a saved pipeline
    python -m phi_geometric info pipeline.json

Input format (TSV):
    Each line has input and output sequences separated by a tab.
    Tokens within each sequence are separated by spaces.

    Example pairs.tsv:
        c a t	k æ t
        s h i p	ʃ ɪ p
        t h i n	θ ɪ n

Author: TruthSpace LCM Project
Date: February 2026
"""

import argparse
import json
import sys
import os

# Ensure the package is importable when run as -m
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi_geometric.core.phase_discovery import PhaseDiscovery
from phi_geometric.core.serialization import save_pipeline, load_pipeline, navigator_to_dict


def parse_sequence(s: str) -> list:
    """Parse a space-separated token sequence."""
    return s.strip().split()


def read_pairs(path: str) -> list:
    """Read training pairs from a TSV file.

    Format: input_tokens<TAB>output_tokens
    Tokens separated by spaces within each column.
    Lines starting with # are comments. Blank lines skipped.
    """
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"WARNING: line {lineno}: expected 2 tab-separated columns, "
                      f"got {len(parts)} — skipping", file=sys.stderr)
                continue
            inp = parse_sequence(parts[0])
            out = parse_sequence(parts[1])
            if not inp or not out:
                print(f"WARNING: line {lineno}: empty sequence — skipping",
                      file=sys.stderr)
                continue
            pairs.append((inp, out))
    return pairs


def cmd_discover(args):
    """Discover a transformation pipeline from training pairs."""
    pairs = read_pairs(args.input)
    if not pairs:
        print("ERROR: No valid training pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(pairs)} training pairs from {args.input}")

    # Count unique tokens
    all_tokens = set()
    for inp, out in pairs:
        all_tokens.update(inp)
        all_tokens.update(out)
    print(f"Vocabulary: {len(all_tokens)} unique tokens")

    # Discover
    pd = PhaseDiscovery(
        context_window=args.context_window,
        geometric=args.geometric,
    )
    for inp, out in pairs:
        pd.add_pair(inp, out)

    result = pd.discover()
    print(f"\nDiscovered {result.n_phases} phase(s), {result.n_rules} rule(s)")
    print(f"Archetype: {result.archetype}")
    print()

    # Validate
    validation = result.validate()
    acc = validation['accuracy']
    print(f"Training accuracy: {validation['correct']}/{validation['total']} "
          f"({acc:.1%})")

    if validation['errors']:
        print(f"\nFirst errors (up to 5):")
        for err in validation['errors'][:5]:
            inp_str = ' '.join(str(t) for t in err['input'])
            print(f"  [{inp_str}] pos {err['position']}: "
                  f"expected '{err['expected']}', got '{err['actual']}'")

    # Describe pipeline
    if args.verbose:
        print(f"\n{'='*60}")
        print(result.describe())

    # Save
    nav = result.to_navigator()
    if args.output:
        save_pipeline(nav, args.output)
        size = os.path.getsize(args.output)
        print(f"\nPipeline saved to {args.output} ({size:,} bytes)")

    # Test mode: also show navigator description
    if args.test:
        print(f"\n{'='*60}")
        print("Navigator structure:")
        print(nav.describe())


def cmd_execute(args):
    """Execute a saved pipeline on input sequences."""
    nav = load_pipeline(args.pipeline)

    if args.input:
        # Single input from command line
        tokens = parse_sequence(args.input)
        trace = nav.execute(tokens)
        output = ' '.join(str(t) for t in trace.output_elements)
        print(output)
    elif args.file:
        # Batch: one input sequence per line (space-separated tokens)
        with open(args.file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                tokens = parse_sequence(line)
                trace = nav.execute(tokens)
                output = ' '.join(str(t) for t in trace.output_elements)
                if args.show_input:
                    inp_str = ' '.join(str(t) for t in tokens)
                    print(f"{inp_str}\t{output}")
                else:
                    print(output)
    else:
        # Interactive mode: read from stdin
        print("Enter space-separated tokens (Ctrl+D to quit):",
              file=sys.stderr)
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                tokens = parse_sequence(line)
                trace = nav.execute(tokens)
                print(' '.join(str(t) for t in trace.output_elements))
        except KeyboardInterrupt:
            pass


def cmd_info(args):
    """Show information about a saved pipeline."""
    with open(args.pipeline, 'r', encoding='utf-8') as f:
        d = json.load(f)

    print(f"Pipeline: {args.pipeline}")
    print(f"Version:  {d.get('version', 'unknown')}")
    print(f"Type:     {d.get('type', 'unknown')}")

    n_collapse = len(d.get('collapse_patterns', []))
    n_expand = len(d.get('expand_patterns', []))
    n_phases = len(d.get('phases', []))
    n_rules = sum(len(p.get('rules', [])) for p in d.get('phases', []))

    print(f"\nStructure:")
    if n_collapse:
        print(f"  Collapse patterns: {n_collapse}")
        for cp in d['collapse_patterns']:
            inp = ' '.join(cp['input'])
            out = ' '.join(cp['output'])
            f = ' [freeze]' if cp.get('freeze') else ''
            print(f"    {inp} → {out}{f}")
    if n_expand:
        print(f"  Expand patterns: {n_expand}")
        for ep in d['expand_patterns']:
            print(f"    {ep['input']} → {' '.join(ep['output'])}")
    print(f"  Phases: {n_phases} ({n_rules} rules total)")
    for i, p in enumerate(d.get('phases', [])):
        ctx = p.get('context_extractor', 'default')
        ctx_tag = f" [geometric]" if ctx == 'geometric' else ""
        flags = []
        if p.get('freeze_outputs'):
            flags.append('freeze')
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"    Phase {i}: {p['name']} "
              f"({len(p.get('rules', []))} rules){flag_str}{ctx_tag}")

    size = os.path.getsize(args.pipeline)
    print(f"\nFile size: {size:,} bytes")


def cmd_reverse(args):
    """Find inputs that produce a desired output."""
    nav = load_pipeline(args.pipeline)

    from phi_geometric.core.generation import ReverseEngine
    engine = ReverseEngine(nav)

    tokens = parse_sequence(args.output)
    results = engine.reverse(tokens, max_results=args.max_results)

    if not results:
        print("No valid inputs found.", file=sys.stderr)
        sys.exit(1)

    for inp in results:
        print(' '.join(str(t) for t in inp))


def cmd_navigate(args):
    """Generate novel valid pairs via lattice navigation."""
    pairs = read_pairs(args.training)

    pd = PhaseDiscovery(
        context_window=args.context_window,
        geometric=args.geometric,
    )
    for inp, out in pairs:
        pd.add_pair(inp, out)
    nav = pd.discover().to_navigator()

    from phi_geometric.core.generation import ReverseEngine
    engine = ReverseEngine(nav)

    novel = engine.navigate(
        seed_pairs=pairs,
        steps=args.steps,
        max_novel=args.max_novel,
    )

    print(f"Generated {len(novel)} novel pairs from {len(pairs)} seeds",
          file=sys.stderr)

    for inp, out in novel:
        inp_str = ' '.join(str(t) for t in inp)
        out_str = ' '.join(str(t) for t in out)
        print(f"{inp_str}\t{out_str}")


def main():
    parser = argparse.ArgumentParser(
        prog='phi_geometric',
        description='φ-Geometric Transformation Engine',
        epilog='https://github.com/truthspace-lcm/phi-geometric',
    )
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # discover
    p_disc = subparsers.add_parser(
        'discover',
        help='Discover transformation pipeline from training pairs',
    )
    p_disc.add_argument('input', help='TSV file with training pairs')
    p_disc.add_argument('-o', '--output', help='Save pipeline to JSON file')
    p_disc.add_argument('--geometric', action='store_true',
                        help='Use φ-decay geometric context (for long-range deps)')
    p_disc.add_argument('--context-window', type=int, default=1,
                        help='Context window size (default: 1)')
    p_disc.add_argument('--test', action='store_true',
                        help='Show detailed pipeline info after discovery')
    p_disc.add_argument('-v', '--verbose', action='store_true',
                        help='Show full phase discovery details')

    # execute
    p_exec = subparsers.add_parser(
        'execute',
        help='Execute a saved pipeline on new inputs',
    )
    p_exec.add_argument('pipeline', help='JSON pipeline file')
    p_exec.add_argument('-i', '--input',
                        help='Single input (space-separated tokens)')
    p_exec.add_argument('-f', '--file',
                        help='File with one input per line')
    p_exec.add_argument('--show-input', action='store_true',
                        help='Show input alongside output (TSV format)')

    # info
    p_info = subparsers.add_parser(
        'info',
        help='Show information about a saved pipeline',
    )
    p_info.add_argument('pipeline', help='JSON pipeline file')

    # reverse
    p_rev = subparsers.add_parser(
        'reverse',
        help='Find inputs that produce a desired output',
    )
    p_rev.add_argument('pipeline', help='JSON pipeline file')
    p_rev.add_argument('-o', '--output', dest='output', required=True,
                       help='Desired output (space-separated tokens)')
    p_rev.add_argument('-n', '--max-results', type=int, default=5,
                       help='Maximum results (default: 5)')

    # navigate
    p_nav = subparsers.add_parser(
        'navigate',
        help='Generate novel valid pairs via lattice navigation',
    )
    p_nav.add_argument('training', help='TSV file with training pairs')
    p_nav.add_argument('--steps', type=int, default=200,
                       help='Perturbation attempts (default: 200)')
    p_nav.add_argument('--max-novel', type=int, default=50,
                       help='Max novel pairs (default: 50)')
    p_nav.add_argument('--geometric', action='store_true',
                       help='Use φ-decay geometric context')
    p_nav.add_argument('--context-window', type=int, default=1,
                       help='Context window size (default: 1)')

    args = parser.parse_args()

    if args.command == 'discover':
        cmd_discover(args)
    elif args.command == 'execute':
        cmd_execute(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'reverse':
        cmd_reverse(args)
    elif args.command == 'navigate':
        cmd_navigate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
