# Python Code Gear Roadmap

## Project Goal

Create an emergent, geometric Python code generation system that:
1. Can be trained to learn Python patterns
2. Generates executable code without LLM at runtime
3. Integrates as a tool callable by the GearOrchestrator
4. Has its own corpus specific to Python
5. Can verify that generated code runs

## Success Criteria

The full chain of events should be observable:
1. **Intent Recognition**: Chat recognizes user wants to write code
2. **Planning**: GearOrchestrator plans the code creation
3. **Tool Call**: Orchestrator calls the PythonCodeGear
4. **Generation**: PythonCodeGear generates code from emergent patterns
5. **Verification**: Code compiles and runs without error
6. **Validation**: Code does what the user intended

## Scope

### In Scope (Simple Programs)
- `main` function structure
- Function definitions and calls
- Variables and assignments
- Strings and string operations
- Math operations (+, -, *, /, %, **)
- Print statements
- File read and write
- Basic control flow (if/else, for, while)
- Lists and basic operations
- User input

### Out of Scope (For Now)
- Advanced libraries (numpy, pandas, etc.)
- Classes and OOP
- Decorators
- Generators and iterators
- Async/await
- Exception handling beyond basic try/except
- Imports beyond builtins

## The Biological Model: Chromosomal Kissing (NHCCs)

Before diving into implementation, we discovered a powerful biological analogy that may guide our inter-gear communication design.

### Non-Homologous Chromosomal Contacts (NHCCs)

From the paper "Interchromosomal interactions: A genomic love story of kissing chromosomes" (Maass et al., 2019):

> "Nuclei require a precise three- and four-dimensional organization of DNA to establish cell-specific gene-expression programs."

Chromosomes occupy **distinct territories** in the nucleus, yet they can "kiss" - forming temporary contacts (NHCCs) to:
1. **Coordinate gene expression** across different chromosomes
2. **Form transcription factories** where related genes cluster
3. **Enable stochastic but specific** interactions

### Key Biological Insights

1. **Territories with Overlap**: Each chromosome has its own territory, but regions can extend to interact with others. *Our gears have their own corpora (territories) but can share concepts at contact points.*

2. **Transcription Factories**: Active genes from multiple chromosomes cluster in "factories" for coordinated expression. *Our gears could form "concept factories" where related intents cluster.*

3. **lncRNA as Scaffolds**: Long non-coding RNAs act as molecular scaffolds that bridge chromosomes. *We need a "scaffold" structure that bridges gears.*

4. **Phase Transitions**: NHCCs may emerge from phase transitions - molecules coalescing in physical proximity. *Concepts could "phase transition" into shared understanding.*

5. **Stochastic but Specific**: Interactions are variable but recurrent and important. *Not every gear needs to understand every concept - just the ones they "kiss" with.*

### The Kissing Model for Gears

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONCEPT SPACE                             │
│                                                                  │
│   ┌─────────────┐                      ┌─────────────┐          │
│   │ Orchestrator│                      │ PythonGear  │          │
│   │  Territory  │                      │  Territory  │          │
│   │             │    ╭─────────╮       │             │          │
│   │  [planning] │◄──►│ CONTACT │◄─────►│ [code gen]  │          │
│   │  [routing]  │    │  POINT  │       │ [patterns]  │          │
│   │  [commands] │    │         │       │ [verify]    │          │
│   │             │    │ Shared  │       │             │          │
│   │             │    │ Concepts│       │             │          │
│   └─────────────┘    ╰─────────╯       └─────────────┘          │
│                                                                  │
│   The "kiss" happens at contact points where both gears         │
│   understand a shared vocabulary of concepts                     │
└─────────────────────────────────────────────────────────────────┘
```

### What the Contact Point Contains

The contact point is NOT a full language - it's a **minimal shared vocabulary**:

```python
# Contact Point Vocabulary (the "kiss")
SHARED_CONCEPTS = {
    # Intent markers
    'CREATE': "make something new",
    'READ': "get existing data", 
    'TRANSFORM': "change data",
    'OUTPUT': "produce result",
    
    # Data type markers
    'TEXT': "string/text data",
    'NUMBER': "numeric data",
    'SEQUENCE': "list/iterable",
    'FILE': "file system object",
    
    # Structure markers
    'REPEAT': "iteration needed",
    'BRANCH': "conditional needed",
    'COMPOSE': "combine operations",
}
```

### How Kissing Works

1. **Orchestrator** encodes intent into shared concepts
2. **Contact point** is the geometric region where both gears' encodings overlap
3. **PythonGear** matches the shared concepts to its internal patterns
4. **Result** flows back through the same contact point

The key insight: **gears don't need to fully understand each other** - they just need to understand the contact point vocabulary. Like chromosomes that only interact at specific loci, not along their entire length.

### Geometric Interpretation

In φ-space, the contact point is where the **folding structures** of two gears' encodings have similar curvature:

```
Orchestrator's encoding:  ╭──╮    ╭──╮
                              ╰──╯    ╰──╯
                                 ▲
                                 │ KISS (similar fold)
                                 ▼
PythonGear's encoding:    ╭──╮    ╭──╮
                              ╰──╯    ╰──╯
```

When the shapes "kiss" (have matching fold patterns), information can transfer.

---

## The Fundamental Problem: Inter-Gear Communication

Before we can build the PythonCodeGear, we need to solve a fundamental problem:

**How do separate gear chains communicate concepts with each other?**

The chromosomal kissing model suggests: **through minimal shared contact points, not full mutual understanding.**

### Current State

The GearOrchestrator currently:
- Takes natural language goals
- Breaks them into steps (PlannerGear)
- Converts steps to bash commands (CommandGear)

The communication is **implicit** - each gear understands natural language and produces natural language or commands.

### The Challenge

For the PythonCodeGear to work, the Orchestrator needs to:
1. Recognize that Python code is needed
2. Describe WHAT the code should do
3. Pass this description to PythonCodeGear
4. Receive code back
5. Understand if the code succeeded

This requires a **shared conceptual language** between gears.

## Phase 0: Design the Concept Protocol

### What is a "Concept"?

A concept is a structured representation of intent that can be:
- Generated by one gear
- Understood by another gear
- Encoded geometrically (for emergent matching)

### Proposed Concept Structure

```python
@dataclass
class Concept:
    """A transferable unit of meaning between gears."""
    
    # What kind of thing is this?
    category: str  # e.g., "action", "data", "control_flow"
    
    # What specifically?
    type: str  # e.g., "create_function", "read_file", "loop"
    
    # Parameters/details
    params: Dict[str, Any]  # e.g., {"name": "greet", "args": ["name"]}
    
    # Geometric encoding for similarity matching
    encoding: Optional[np.ndarray] = None
```

### Concept Categories for Code Generation

```
ACTION
├── create_function(name, args, returns, body_intent)
├── create_variable(name, value_type, initial_value)
├── call_function(name, args)
├── print_value(what)
├── read_file(path)
├── write_file(path, content)
└── return_value(what)

CONTROL_FLOW
├── if_condition(condition, then_intent, else_intent)
├── for_loop(variable, iterable, body_intent)
├── while_loop(condition, body_intent)
└── sequence(intents...)

DATA
├── string(value)
├── number(value)
├── list(items)
├── expression(operation, operands)
└── variable_ref(name)

PROGRAM
├── main_function(body_intent)
├── script(intents...)
└── module(functions...)
```

### How Concepts Flow

```
User: "Write a Python program that reads a file and prints each line"

     ┌─────────────────┐
     │ IntentDetector  │
     └────────┬────────┘
              │ Intent.CODE_GENERATION
              ▼
     ┌─────────────────┐
     │ GearOrchestrator│
     └────────┬────────┘
              │ Concept: PROGRAM.script([
              │   ACTION.read_file(path="input.txt"),
              │   CONTROL_FLOW.for_loop(
              │     variable="line",
              │     iterable=variable_ref("file_content"),
              │     body=ACTION.print_value(variable_ref("line"))
              │   )
              │ ])
              ▼
     ┌─────────────────┐
     │ PythonCodeGear  │
     └────────┬────────┘
              │ Generated Code:
              │ with open("input.txt", "r") as f:
              │     for line in f:
              │         print(line)
              ▼
     ┌─────────────────┐
     │ CodeVerifier    │
     └────────┬────────┘
              │ Result: SUCCESS / FAILURE
              ▼
     ┌─────────────────┐
     │ Response to User│
     └─────────────────┘
```

## Phase 1: Concept Protocol Implementation

### 1.1 Define the Concept dataclass and categories
### 1.2 Create ConceptEncoder (geometric encoding of concepts)
### 1.3 Create ConceptMatcher (find similar concepts in corpus)
### 1.4 Update GearMessage to carry Concepts

## Phase 2: PythonCodeGear Core

### 2.1 Create PythonCodeCorpus
- Store (concept, code_template, test_case) tuples
- Support save/load for persistence
- Support similarity search

### 2.2 Create CodeGenerator
- Match concept to corpus entries
- Adapt templates with parameters
- Compose multiple patterns

### 2.3 Create CodeVerifier
- Syntax check with `ast.parse()`
- Execution check in sandbox
- Output capture

### 2.4 Create CorpusBuilder
- Seed with basic patterns
- Learn from successful generations
- Query LLM when stuck (corpus building mode)

## Phase 3: Integration

### 3.1 Update IntentDetector
- Add Intent.CODE_GENERATION
- Detect "write code", "create program", "Python script", etc.

### 3.2 Update GearOrchestrator
- Register PythonCodeGear as a tool
- Add ConceptTranslator (natural language → Concept)
- Handle code generation in execute()

### 3.3 Update Chat Interface
- Route code requests through orchestrator
- Display generated code
- Show verification results

## Phase 4: Self-Improvement

### 4.1 Success Tracking
- Record successful concept → code mappings
- Build confidence scores

### 4.2 Failure Recovery
- When verification fails, query LLM for fix
- Store successful fixes as new patterns

### 4.3 Pattern Generalization
- Identify common sub-patterns
- Create reusable components

## The Geometric Aspect

### How is this "geometric"?

1. **Concept Encoding**: Each concept gets a position in φ-space based on its category, type, and parameters

2. **Similarity Matching**: Finding the right code pattern is a nearest-neighbor search in concept space

3. **Composition**: Complex programs are compositions of simpler patterns, like fractal self-similarity

4. **Learning**: New patterns create new "attractors" in the space

### Example: Encoding a Concept

```python
def encode_concept(concept: Concept) -> np.ndarray:
    """Encode a concept into φ-space."""
    
    # Category dimension (coarse)
    category_phase = {
        'action': 0.0,
        'control_flow': φ,
        'data': φ**2,
        'program': φ**3,
    }[concept.category]
    
    # Type dimension (fine)
    type_encoding = hash(concept.type) % 1000 / 1000 * φ
    
    # Parameter encoding (details)
    param_encoding = encode_params(concept.params)
    
    return np.array([category_phase, type_encoding, *param_encoding])
```

## Milestones

### Milestone 1: Concept Protocol (Est: 1-2 sessions)
- [ ] Concept dataclass defined
- [ ] ConceptEncoder working
- [ ] Basic concept matching
- [ ] GearMessage updated

### Milestone 2: Basic Code Generation (Est: 2-3 sessions)
- [ ] PythonCodeCorpus with 20+ patterns
- [ ] CodeGenerator produces valid code
- [ ] CodeVerifier catches syntax errors
- [ ] Simple programs work (hello world, math)

### Milestone 3: Integration (Est: 1-2 sessions)
- [ ] IntentDetector routes code requests
- [ ] Orchestrator calls PythonCodeGear
- [ ] End-to-end flow works
- [ ] Chat can generate code

### Milestone 4: Self-Improvement (Est: 2-3 sessions)
- [ ] Learning from success
- [ ] LLM fallback for stuck cases
- [ ] Pattern generalization
- [ ] Corpus grows over time

## Open Questions

1. **Concept Granularity**: How detailed should concepts be?
   - Too coarse: Can't distinguish similar patterns
   - Too fine: Explosion of concept types

2. **Composition**: How do we compose concepts into programs?
   - Tree structure (AST-like)?
   - Sequence of concepts?
   - Nested concepts?

3. **Parameter Handling**: How do we encode variable names, values?
   - Placeholder tokens?
   - Semantic encoding?

4. **Error Messages**: When code fails, how do we communicate why?
   - Parse error messages?
   - Map back to concepts?

## Next Steps

1. **Create this roadmap** ✓
2. **Design Concept dataclass** - Start simple, iterate
3. **Implement basic ConceptEncoder** - Use existing φ-space encoding
4. **Create seed corpus** - 10-20 basic Python patterns
5. **Build minimal PythonCodeGear** - Just generation, no learning yet
6. **Test with hardcoded concepts** - Verify generation works
7. **Add verification** - Syntax + execution
8. **Integrate with orchestrator** - Full flow
9. **Add learning** - Self-improvement loop

## Files to Create

```
truthspace_lcm/gears/core/
├── concept.py              # Concept dataclass and encoding
├── concept_encoder.py      # Geometric encoding of concepts
├── python_code_gear.py     # Main code generation gear
├── python_corpus.py        # Python-specific corpus
└── code_verifier.py        # Syntax and execution verification

truthspace_lcm/gears/practical_applications/
└── python/
    ├── __init__.py
    ├── patterns/           # Seed patterns
    │   ├── basic.py
    │   ├── control_flow.py
    │   ├── file_io.py
    │   └── functions.py
    └── tests/              # Test cases for patterns
```

## References

- Design doc 082: Folding structure (shape-based matching)
- Design doc 072: Self-similar truthspace
- Existing: SelfBuildingCorpusGear (pattern for self-improvement)
- Existing: GearOrchestrator (integration target)
