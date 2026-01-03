#!/usr/bin/env python3
"""
Emergent Chat API Server

OpenAI-compatible API server for the emergent chat system.
Works with tools like Goose, Open WebUI, Continue, and any OpenAI-compatible client.

Usage:
    python run_api.py                    # Default: localhost:8001
    python run_api.py --port 8002        # Custom port
    python run_api.py --host 0.0.0.0     # Allow external connections

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent))

# Delegate to the actual run_api module
from truthspace_lcm.practical_applications.chat.run_api import main

if __name__ == "__main__":
    sys.exit(main())
