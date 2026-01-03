# TruthSpace Geometric LCM - Project Overview

**Author**: Lesley Gushurst  
**License**: GPLv3  
**Status**: Active Development

---

## Origin of the Name "TruthSpace"

The name "TruthSpace" originated from the initial concept of anchoring axes to **mathematical constants** rather than abstract concepts. Early experiments showed promise, but two challenges emerged:

1. **Axis-constant assignment** - It became increasingly difficult to determine which axes should be defined by which mathematical constants
2. **Dimensionality explosion** - Managing the sheer number of axes required became unwieldy

The **gears directory** and **quaternion paradigm** were introduced to resolve these issues. Quaternions provide a principled way to handle 4D rotations without manually assigning constants to individual axes, while gears provide composable transformations that can be chained dynamically.

---

## What Is This Project?

TruthSpace Geometric LCM (Large Concept Model) is an experimental system that seeks to **replace traditional Large Language Models (LLMs) with a purely geometric approach**.

### The Hypothesis

> **LLMs are hyperdimensional transcoders** - they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights themselves, but in the **shape** those weights create.

We aim to prove this hypothesis by building a system that:

- **Structure IS information** - There are no opaque weights or embeddings
- **Geometry IS computation** - Traversal through geometric space produces outputs
- **The shape IS the knowledge** - What an LLM "knows" is encoded in its geometric structure

If we can replicate LLM-like behavior using pure geometry, we validate the hypothesis. If we cannot, we learn where the hypothesis breaks down.

### Fail-Fast Development Philosophy

We adopt a **fail-fast** development strategy:

- **No graceful fallbacks** - If something fails, we want to see the error immediately
- **No hard-coded workarounds** - Hard-coding flies in the face of our hypothesis
- **Prove or disprove** - Every component must work emergently or expose why it can't

This means:
1. If intent detection fails geometrically, we don't fall back to pattern matching
2. If semantic understanding fails emergently, we don't use lookup tables
3. Errors are signals, not problems to hide

The goal is to **demonstrate the hypothesis is correct**, not to build a product that works by any means necessary. A working fallback would mask whether our geometric approach actually works.

---

## Core Philosophy

### 1. Structure Is Information

Every piece of information in our system has a geometric representation. We reject:
- Hard-coded morphology
- Pattern strings that don't transform to geometry
- Static lookup tables
- Graceful fallbacks that hide geometric failures

The only exception is **bootstrapped information** - initial seeds that are immediately transformed into geometry on program startup.

### 2. Emergent Geometry

We don't design the geometry top-down. Instead:
- Structure emerges from relationships in data
- Positions are constructed from similarity matrices
- The system discovers its own dimensions via SVD

### 3. ENCODE = DECODE

A fundamental insight: encoding and decoding are the **same operation in opposite directions**, like φ and 1/φ.

```
TEXT IN → φ-space → TEXT OUT
```

- When encoding words, we're decoding meaning
- When decoding response, we're encoding understanding
- "Thinking" isn't a step between - it IS the encode-decode

### 4. Self-Similarity

The system exhibits fractal self-similarity:
- The same transformations work identically at every scale
- Gender flip is always Δx = -2.0 (king→queen, man→woman, boy→girl)
- This self-similarity is self-verifying - no external validation needed

---

## Project Structure

```
truthspace-lcm/
├── truthspace_lcm/           # Main project code
│   ├── core/                 # Domain-agnostic core functionality
│   │   ├── gear.py           # Base Gear class
│   │   ├── conversational_chain.py  # Main chat chain
│   │   ├── emergent_classifier.py   # Intent classification
│   │   ├── code_orchestrator.py     # Code generation
│   │   └── ...               # Other core components
│   ├── practical_applications/
│   │   └── chat/             # Chat interface & API server
│   │       ├── api_server.py # OpenAI-compatible API
│   │       ├── chat.py       # EmergentChat class
│   │       └── run_api.py    # Server runner
│   ├── corpus/               # Knowledge corpuses (JSON)
│   └── tools/                # Corpus management tools
│
├── temp/                     # Deprecated/legacy code
│   └── legacy_truthspace_lcm/  # Old core and NLP gears
├── design_considerations/    # Design documentation (87+ documents)
├── data/                     # Runtime data storage
└── output/                   # Generated outputs (plots, etc.)
```

### Directory Philosophy

- **`core/`** - Contains domain-agnostic base classes and fundamental abstractions. Includes gear base classes, conversational chain, intent classification, and code orchestration.

- **`practical_applications/`** - Contains applications built on top of core. Currently focused on the chat application which provides an OpenAI-compatible API.

- **`temp/`** - Deprecated code and legacy implementations. Kept for reference but not actively used.

- **`design_considerations/`** - Living documentation of architectural decisions, discoveries, and research notes.

---

## Corpus: Training-Free Knowledge

A **corpus** is a flat file containing information that the Geometric LCM starts up with. Unlike LLM weights:

- **No training required** - Modify the corpus, restart, and the system has new knowledge
- **Human-readable** - Corpuses are JSON files you can edit directly
- **Geometry on load** - Corpus content is transformed into geometric structure at startup

### The Key Advantage

Traditional LLMs require expensive retraining to update knowledge. Our system:

```
Edit corpus file → Restart → New geometric structure → Updated behavior
```

This gives us a massive advantage in iteration speed and transparency.

### Corpus Location

Corpuses are stored in `truthspace_lcm/gears/corpus/` as JSON files. Each gear or gear chain can have its own corpus.

---

## The Gear Abstraction

We model transformations as a **chain of interlocking gears** where each gear handles one specific transformation.

### What Is a Gear?

A gear is a transformation unit that:
- Has its own corpus of information (information IS geometry)
- Transforms input state to output state
- Can be composed with other gears
- Maintains a quaternion signature of its transformation

```python
class Gear:
    def forward(self, state: GearState) -> GearState:
        """Transform state through this gear."""
        pass
```

### Gear Chains

Gears compose into chains:

```
Input → [Gear A] → [Gear B] → [Gear C] → Output
```

- Chains can be dynamic (assembled at runtime) or predefined
- Each gear adds dimensions to the transformation
- The quaternion accumulates through the chain, providing a geometric signature

### The Emergent Gear Pattern

Across the codebase, we implement a recurring 5-step pattern:

```
┌─────────────────────────────────────────────────────────────┐
│  1. STRUCTURE - Define what the space looks like            │
│     Patterns, signatures, templates, modules                │
│                                                             │
│  2. BOOTSTRAP - Seed with initial examples                  │
│     Use LLM to generate missing pieces                      │
│     Transform seeds into geometry immediately               │
│                                                             │
│  3. MATCH - Find the right structure for input              │
│     Project input into the space                            │
│     Find nearest/best matching structure                    │
│                                                             │
│  4. COMPOSE - Adapt structure to specific request           │
│     Extract parameters from input                           │
│     Modify the matched structure                            │
│                                                             │
│  5. LEARN - Self-improve from usage                         │
│     Record successes and failures                           │
│     Promote temporary structures to permanent               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Technical Concepts

### φ (Phi) Encoding

The golden ratio φ ≈ 1.618 is fundamental to our encoding:
- Positions are determined by powers of φ
- φ × 1/φ = 1 (self-inverse property)
- Provides natural spacing that avoids collisions

### Quaternions

4D rotations that represent transformations:
- Each gear transformation is a quaternion rotation
- Quaternions accumulate through the chain
- Provides geometric signature of transformation path

### Holographic Projection

Instead of discovering structure, we **construct** it:

```python
# Define desired similarity matrix
S[i,j] = word_overlap(module_i, module_j)

# Find positions that realize this similarity
eigenvalues, eigenvectors = eig(S)
P = eigenvectors @ diag(sqrt(eigenvalues))

# Now: dot(P[i], P[j]) ≈ S[i,j] by construction!
```

The geometry encodes relationships directly - no gates or patches needed.

### Contact Points (DNA-Inspired)

Gear chains communicate via shared anchor points, inspired by DNA mechanics:

| DNA Concept | Gear Chain Equivalent |
|-------------|----------------------|
| Zinc Fingers | Concept Fingers - recognize quaternion patterns |
| Major Groove | Semantic access (entity, role, actions) |
| Minor Groove | Geometric access (quaternion, position) |
| Supercoiling | Activation control via geometric tension |

### Temporary Module Injection

For unknown concepts:
1. Inject a temporary module into the holographic space
2. Let LLM handle the generation
3. If successful, promote to permanent
4. If failed, remove the temporary module

This allows the system to handle novel queries while learning from them.

---

## Self-Improvement Loop

One of the most powerful features is the **autonomous self-improvement loop** that can:

1. **Detect deficiencies** in gear outputs
2. **Create fix gears dynamically** to address issues
3. **Test and validate** the fixes
4. **Learn which fixes work** and remember them for future use

### The Improvement Cycle

```
┌─────────────────────────────────────────────────────────────┐
│  1. TEST - Run gear against test cases                      │
│     GearTestHarness / ShapeBasedTestHarness                 │
│                                                             │
│  2. DETECT - Identify deficiencies                          │
│     DeficiencyDetectorGear / FoldingDeficiencyDetector      │
│     Types: missing_content, wrong_format, too_vague,        │
│            too_verbose, irrelevant, factual_error           │
│                                                             │
│  3. FIX - Create fix gears dynamically                      │
│     GearImprovementLoop creates gears from templates        │
│     Can use LLM as "teacher" to generate fix logic          │
│                                                             │
│  4. COMPOSE - Build improved chain                          │
│     GearChainBuilder composes gears at runtime              │
│                                                             │
│  5. ITERATE - Re-test until quality threshold met           │
│     Default: 0.8 quality, max 5 iterations                  │
│                                                             │
│  6. LEARN - Remember what worked                            │
│     fix_memory: deficiency signature → fix that worked      │
│     fix_effectiveness: tracks improvement deltas            │
│     Can save/load to JSON for persistence                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Classes

| Class | Purpose |
|-------|--------|
| `GearImprovementLoop` | Main orchestrator - the self-improving meta-gear |
| `DeficiencyDetectorGear` | Pattern-based deficiency detection |
| `ShapeBasedTestHarness` | Geometric (folding) deficiency detection |
| `GearChainBuilder` | Dynamic gear composition at runtime |
| `GearTestHarness` | Test execution and quality measurement |

### Dynamic Gear Creation

The system can create new gears on the fly:

```python
# Fix templates registered for each deficiency type
fix_templates[DeficiencyType.TOO_VAGUE] = create_specificity_gear
fix_templates[DeficiencyType.TOO_VERBOSE] = create_summary_gear
fix_templates[DeficiencyType.MISSING_CONTENT] = create_content_extractor

# Smart fix creation using LLM as teacher
loop.create_smart_fix(deficiency, input, output, source_gear)
```

The LLM is used to **teach** the system how to create fixes, but the resulting fix gears run **without LLM** at runtime.

### Feedback Refinement

The `FeedbackRefinementGear` uses LLM for quality gating:
- Scores response quality (0-10)
- Suggests grammatical/clarity improvements
- Does NOT generate new content or knowledge

This maintains emergent nature while improving output quality.

---

## Current Capabilities

### 1. Standalone Chat

```bash
python truthspace_lcm/gears/run.py
```

Interactive chat using emergent geometry for response generation.

### 2. OpenAI-Compatible API Server

```bash
python truthspace_lcm/gears/run_api.py --port 8002
```

Implements `/v1/chat/completions` endpoint compatible with OpenAI clients.

### 3. Tool Calling (Agent Integration)

The API server supports tool calling for integration with agent frameworks like Goose:

- **File operations**: `list files`, `read file`
- **Shell commands**: `run pytest`
- **Knowledge acquisition**: Unknown topics trigger LLM lookup via tool calls

### 4. Code Generation

The `CodeOrchestrator` generates executable code:
- Pattern matching for common requests (plots, charts)
- Template composition with modifications
- Auto-execution and verification

### 5. Self-Improvement

The system can improve geometrically on the fly:
- Traversal paths are well-known
- Can train on ingested data
- Can work in reverse to update geometry from corrected outputs

---

## Technical Debt Resolution

### Resolved (Jan 2026)

Following the fail-fast philosophy, we removed hard-coded fallbacks:

| Component | Resolution |
|-----------|------------|
| `intent_detector.py` | **Removed from chat** - Using only emergent `IntentClassifier` |
| `semantic_chain.py` | **Labels moved to corpus** - `corpus/feature_labels.json` (editable without code changes) |
| `error_correction.py` | **Moved to nlp/** - Only used by legacy NLP module, not core |
| `data/` | **Moved to temp/** - Was only for demonstration purposes |
| `experiments/` | **Moved to temp/** - Kept for reference only |

### Architectural Direction

- **IntentClassifier** (emergent) replaces **IntentDetectorGear** (hard-coded patterns)
- **ConversationalChain** builds knowledge through corpus, uses LLM as knowledge resource (not generator)
- All feature labels now loaded from corpus files at startup (editable without retraining)

---

## What We're NOT Doing

To maintain focus and architectural integrity:

1. **No opaque neural networks** - Every transformation is explicit
2. **No hard-coded responses** - Everything derives from geometry
3. **No string matching as primary mechanism** - Geometry first, strings for bootstrapping only
4. **No monolithic models** - Composable, swappable gears

---

## Mathematical Foundation

### The Core Equation

```
ENCODE = DECODE
```

Both are the same operation through φ-space, just in opposite directions.

### Similarity Construction

Given modules M with relationships R:
```
S[i,j] = similarity(M[i], M[j])  # What we want
P = V @ sqrt(Λ)                   # Eigendecomposition
dot(P[i], P[j]) ≈ S[i,j]         # By construction
```

### Transformation Accumulation

Through a gear chain:
```
Q_total = Q_1 × Q_2 × Q_3 × ... × Q_n
```

The final quaternion encodes the complete transformation path.

---

## Design Considerations

We maintain extensive documentation in `/design_considerations/`:

| Range | Topics |
|-------|--------|
| 001-010 | Foundational geometry, φ-encoding, dimensionality |
| 011-020 | Knowledge expansion, disambiguation, holographic resolution |
| 021-030 | Attractor dynamics, Pareto bootstrap, semantic trees |
| 031-040 | Unified projection, VSA binding, spatial attention |
| 041-050 | φ-dial, quaternions, holographic bounds, geodesics |
| 051-060 | Roadmaps, tachyon hypothesis, diffraction, encode=decode |
| 061-070 | Morphology, templates, concept distillation |
| 071-080 | Self-similarity, gear chains, DNA parallels, emergent patterns |
| 081-087 | Folding, code generation, holographic projection, tool calling |

---

## Future Directions

1. **Corpus Update from LLM Responses** - Save acquired knowledge to geometric corpus
2. **Web Search Integration** - Expand knowledge acquisition beyond LLM
3. **Multi-Chain Orchestration** - Multiple gear chains working in parallel
4. **Geometric Reinforcement Learning** - Update geometry from feedback
5. **Distributed Geometry** - Shard geometric space across nodes

---

## Getting Started

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Chat

```bash
python truthspace_lcm/gears/run.py
```

### Run API Server

```bash
python -m truthspace_lcm.practical_applications.chat.api_server --port 8002
```

### Run Tests

```bash
python tests/test_lcm_routing.py
```

---

## Contributing

This is an experimental research project. Key principles for contributions:

1. **Structure over strings** - Prefer geometric solutions
2. **Emergent over designed** - Let patterns emerge from data
3. **Composable over monolithic** - Small, focused gears
4. **Document discoveries** - Add to design_considerations/

---

## License

GPLv3 - See LICENSE file for details.

---

*"φ is the whole thing."*
