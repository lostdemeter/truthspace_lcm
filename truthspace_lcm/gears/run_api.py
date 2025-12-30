#!/usr/bin/env python3
"""
Gear Chain API Server

OpenAI-compatible API server for the gear chain system.
Works with tools like Open WebUI, Continue, and any OpenAI-compatible client.

Usage:
    python run_api.py                    # Default: localhost:8000
    python run_api.py --port 11434       # Ollama-compatible port
    python run_api.py --host 0.0.0.0     # Allow external connections

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from pathlib import Path

# Ensure the parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(
        description="Gear Chain API Server - OpenAI-Compatible"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("  Gear Chain API Server")
    print("  OpenAI-Compatible Endpoint")
    print("=" * 60)
    print()
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  URL:  http://{args.host}:{args.port}")
    print()
    print("  Endpoints:")
    print("    POST /v1/chat/completions  - Chat completions")
    print("    GET  /v1/models            - List models")
    print("    GET  /health               - Health check")
    print()
    print("  Available Models:")
    print("    gear-chain         - Present tense (default)")
    print("    gear-chain-past    - Past tense")
    print("    gear-chain-future  - Future tense")
    print()
    print("  For Open WebUI, use:")
    print(f"    http://{args.host}:{args.port}/v1")
    print()
    print("=" * 60)
    print()
    
    uvicorn.run(
        "truthspace_lcm.gears.practical_applications.nlp.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
