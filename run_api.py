#!/usr/bin/env python3
"""
GeometricLCM API Server Entry Point

Run this to start the OpenAI-compatible API server.

Usage:
    python run_api.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_api.py                    # Default: localhost:8000
    python run_api.py --port 11434       # Ollama-compatible port
    python run_api.py --host 0.0.0.0     # Allow external connections

Author: Lesley Gushurst
License: GPLv3
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="GeometricLCM API Server - OpenAI-Compatible"
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
    print("  GeometricLCM API Server")
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
    print("  For Open WebUI, use:")
    print(f"    http://{args.host}:{args.port}/v1")
    print()
    print("=" * 60)
    print()
    
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
