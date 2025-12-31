#!/usr/bin/env python3
"""
Corpus Generation and Improvement Daemon

A long-running script that continuously builds and improves the chat corpus.
Features:
- Auto-save at configurable intervals
- Resume from saved state
- Graceful shutdown on SIGINT/SIGTERM
- Progress logging
- Configurable build intervals

Usage:
    python scripts/corpus_daemon.py [options]
    
Options:
    --corpus PATH       Path to corpus file (default: data/chat_corpus.json)
    --interval SECONDS  Build interval in seconds (default: 60)
    --save-interval N   Save every N iterations (default: 10)
    --max-iterations N  Maximum iterations (default: unlimited)
    --verbose           Enable verbose logging

Author: Lesley Gushurst
License: GPLv3
"""

import sys
import os
import time
import signal
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from truthspace_lcm.gears.core import ConversationalChain, SelfBuildingCorpusGear


class CorpusDaemon:
    """
    Daemon for continuous corpus generation and improvement.
    
    Features:
    - Runs build iterations at configurable intervals
    - Auto-saves progress periodically
    - Resumes from saved state on restart
    - Handles graceful shutdown
    """
    
    def __init__(self, corpus_path: str, build_interval: int = 60,
                 save_interval: int = 10, max_iterations: int = None,
                 verbose: bool = False):
        self.corpus_path = Path(corpus_path)
        self.build_interval = build_interval
        self.save_interval = save_interval
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # State
        self.running = False
        self.iteration = 0
        self.start_time = None
        self.last_save_time = None
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('CorpusDaemon')
        
        # Initialize chain
        self.chain = ConversationalChain()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def load_or_create(self):
        """Load existing corpus or create new one."""
        if self.corpus_path.exists():
            self.logger.info(f"Loading existing corpus from {self.corpus_path}")
            self.chain.load_corpus(str(self.corpus_path))
            
            if self.chain.default_corpus:
                items = len(self.chain.default_corpus.all_items)
                iterations = self.chain.default_corpus.build_stats.get('iterations', 0)
                self.logger.info(f"Loaded {items} items, {iterations} previous iterations")
        else:
            self.logger.info(f"Creating new corpus at {self.corpus_path}")
            # Ensure directory exists
            self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self):
        """Save current corpus state."""
        self.chain.save_corpus(str(self.corpus_path))
        self.last_save_time = datetime.now()
        
        if self.chain.default_corpus:
            items = len(self.chain.default_corpus.all_items)
            self.logger.info(f"Saved corpus: {items} items")
    
    def run_iteration(self):
        """Run one build iteration."""
        if not self.chain.default_corpus:
            self.logger.warning("No default corpus available")
            return None
        
        result = self.chain.default_corpus.build_iteration()
        self.iteration += 1
        
        if self.verbose or result['items_added'] > 0 or result['items_refined'] > 0:
            self.logger.info(
                f"Iteration {self.iteration}: "
                f"+{result['items_added']} added, "
                f"{result['items_refined']} refined, "
                f"{len(self.chain.default_corpus.all_items)} total"
            )
        
        return result
    
    def run(self):
        """Main daemon loop."""
        self.running = True
        self.start_time = datetime.now()
        self.last_save_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("Corpus Generation Daemon Starting")
        self.logger.info("=" * 60)
        self.logger.info(f"Corpus path: {self.corpus_path}")
        self.logger.info(f"Build interval: {self.build_interval}s")
        self.logger.info(f"Save interval: every {self.save_interval} iterations")
        if self.max_iterations:
            self.logger.info(f"Max iterations: {self.max_iterations}")
        self.logger.info("Press Ctrl+C to stop gracefully")
        self.logger.info("-" * 60)
        
        # Load or create corpus
        self.load_or_create()
        
        # Initial stats
        if self.chain.default_corpus:
            stats = self.chain.default_corpus.get_stats()
            self.logger.info(f"Starting with {stats['total_items']} items")
        
        iterations_since_save = 0
        
        try:
            while self.running:
                # Check max iterations
                if self.max_iterations and self.iteration >= self.max_iterations:
                    self.logger.info(f"Reached max iterations ({self.max_iterations})")
                    break
                
                # Run iteration
                result = self.run_iteration()
                iterations_since_save += 1
                
                # Auto-save
                if iterations_since_save >= self.save_interval:
                    self.save()
                    iterations_since_save = 0
                
                # Wait for next iteration
                if self.running:
                    time.sleep(self.build_interval)
        
        except Exception as e:
            self.logger.error(f"Error during iteration: {e}")
            raise
        
        finally:
            # Final save on shutdown
            self.logger.info("Performing final save...")
            self.save()
            
            # Print summary
            runtime = datetime.now() - self.start_time
            self.logger.info("-" * 60)
            self.logger.info("Daemon Stopped")
            self.logger.info(f"Total runtime: {runtime}")
            self.logger.info(f"Total iterations: {self.iteration}")
            if self.chain.default_corpus:
                self.logger.info(f"Final corpus size: {len(self.chain.default_corpus.all_items)} items")
            self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Corpus Generation and Improvement Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults (saves to data/chat_corpus.json)
    python scripts/corpus_daemon.py
    
    # Run with custom corpus path
    python scripts/corpus_daemon.py --corpus my_corpus.json
    
    # Run with faster iterations (every 10 seconds)
    python scripts/corpus_daemon.py --interval 10
    
    # Run for 100 iterations then stop
    python scripts/corpus_daemon.py --max-iterations 100
    
    # Run in background with nohup
    nohup python scripts/corpus_daemon.py > corpus.log 2>&1 &
"""
    )
    
    parser.add_argument(
        '--corpus', '-c',
        type=str,
        default='data/chat_corpus.json',
        help='Path to corpus file (default: data/chat_corpus.json)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Build interval in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--save-interval', '-s',
        type=int,
        default=10,
        help='Save every N iterations (default: 10)'
    )
    
    parser.add_argument(
        '--max-iterations', '-m',
        type=int,
        default=None,
        help='Maximum iterations (default: unlimited)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Resolve corpus path relative to project root
    corpus_path = args.corpus
    if not os.path.isabs(corpus_path):
        corpus_path = str(project_root / corpus_path)
    
    daemon = CorpusDaemon(
        corpus_path=corpus_path,
        build_interval=args.interval,
        save_interval=args.save_interval,
        max_iterations=args.max_iterations,
        verbose=args.verbose
    )
    
    daemon.run()


if __name__ == "__main__":
    main()
