# TruthSpace Geometric LCM

**A geometric approach to language understanding - proving that structure IS information.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## What Is This?

TruthSpace Geometric LCM (Large Concept Model) is an experimental system exploring whether **LLMs can be replaced with a purely geometric approach**.

### The Hypothesis

> **LLMs are hyperdimensional transcoders** - they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights themselves, but in the **shape** those weights create.

We aim to prove this by building a system where:
- **Structure IS information** - No opaque weights or embeddings
- **Geometry IS computation** - Traversal through geometric space produces outputs
- **The shape IS the knowledge** - What an LLM "knows" is encoded in its geometric structure

## Quick Start

### Interactive Chat
```bash
python run.py --demo                    # Interactive chat with demo corpus
python run.py --demo "What is AI?"      # Single query with corpus
```

### API Server (OpenAI-Compatible)
```bash
python run_api.py --port 8002           # Builds corpus automatically
```

Works with any OpenAI-compatible client (Goose, Continue, Open WebUI):
```
http://localhost:8002/v1
```

## Project Structure

```
truthspace-lcm/
├── run.py                    # Interactive chat
├── run_api.py                # API server
├── truthspace_lcm/
│   ├── core/
│   │   ├── gear.py           # Base: Gear, GearState, Quaternion
│   │   ├── protocol.py       # GearProtocol, GearMessage
│   │   ├── gears/            # Gear implementations
│   │   ├── chains/           # Chain implementations (ConversationalChain)
│   │   ├── orchestrators/    # CodeOrchestrator, GearOrchestrator
│   │   ├── classifiers/      # Intent classification
│   │   └── utils/            # Utilities
│   ├── corpus/               # Knowledge corpuses
│   └── practical_applications/chat/
├── docs/
│   ├── papers/
│   └── design_considerations/  # 87+ design documents
└── temp/                     # Legacy code
```

## Core Concepts

- **Gears**: Composable transformation units
- **Chains**: Sequences of gears (main: `ConversationalChain`)
- **Emergent Classification**: Geometric intent detection via SVD
- **Fail-Fast**: No fallbacks - failures expose where geometry breaks down

## Requirements

```bash
pip install numpy requests beautifulsoup4 fastapi uvicorn
```

## License

GPLv3 - Lesley Gushurst

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for detailed philosophy and [ARCHITECTURE.md](ARCHITECTURE.md) for technical details.
