# Research Proposal: Generalized φ-Based Knowledge Ingestion

## Abstract

We have demonstrated a working system that ingests domain knowledge (Linux commands) and achieves 100% accuracy on query matching using co-occurrence based cluster matching. This proposal outlines how to **abstract and generalize** this system for application to arbitrary domains—from cooking recipes to medical diagnosis to legal documents.

## Current System Analysis

### What We Built (Bash Domain)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training Data          Co-occurrence           Query Matching  │
│  ─────────────          ────────────           ──────────────  │
│                                                                 │
│  Q: "list files"        files→ls:14            Query: "show    │
│  A: "Use 'ls'"          disk→df:22              files"         │
│       ↓                 processes→ps:11              ↓          │
│  Ingest Q+A together    network→netstat:10     Sum affinities  │
│  (window=15)            users→who:17           → best command  │
│                                                                 │
│  DOMAIN-SPECIFIC:       DOMAIN-SPECIFIC:       DOMAIN-SPECIFIC:│
│  - Command words        - Command list         - Command list  │
│  - Q&A format           - Concept→cmd map      - Response has  │
│                                                  command       │
└─────────────────────────────────────────────────────────────────┘
```

### Domain-Specific Components (Must Abstract)

1. **Command list**: `['ls', 'df', 'ps', 'netstat', 'who']`
2. **Social keywords**: Hardcoded greeting/chitchat detection
3. **Response format**: Assumes response contains a command to execute
4. **Action type**: Bash execution

### Domain-Agnostic Components (Already Reusable)

1. **Co-occurrence tracking**: Window-based word co-occurrence
2. **Affinity calculation**: Sum co-occurrence counts for matching
3. **φ-based vocabulary**: Frequency-based level assignment
4. **Attractor dynamics**: Phase clustering from co-occurrence

## Proposed Generalization

### Core Abstraction: Domain Configuration

```python
@dataclass
class DomainConfig:
    """Configuration for a knowledge domain."""
    
    name: str                          # "bash", "cooking", "medical"
    
    # Anchor words: the "commands" of this domain
    anchors: List[str]                 # ["ls", "df"] or ["bake", "sauté", "boil"]
    
    # How to detect anchors in responses
    anchor_pattern: str                # r"'([^']+)'" or r"Recipe: (\w+)"
    
    # Social context (optional, can be shared)
    social_keywords: Dict[str, Set[str]]
    
    # Action handler (what to do when matched)
    action_handler: Callable[[str, str], str]  # (query, response) -> result
    
    # Response templates
    templates: Dict[str, str]          # {"not_found": "I don't know about {topic}"}
```

### Example Domain Configurations

#### Bash (Current)
```python
bash_domain = DomainConfig(
    name="bash",
    anchors=["ls", "df", "ps", "netstat", "who", "ip", "grep", "find"],
    anchor_pattern=r"'([^']+)'",
    action_handler=lambda q, r: subprocess.run(extract_cmd(r), shell=True, capture_output=True).stdout,
)
```

#### Cooking
```python
cooking_domain = DomainConfig(
    name="cooking",
    anchors=["bake", "sauté", "boil", "fry", "roast", "steam", "grill"],
    anchor_pattern=r"Recipe: (\w+)",
    action_handler=lambda q, r: r,  # Just return the recipe text
)
```

#### Medical (Symptom → Condition)
```python
medical_domain = DomainConfig(
    name="medical",
    anchors=["diabetes", "hypertension", "flu", "migraine", "arthritis"],
    anchor_pattern=r"Condition: (\w+)",
    action_handler=lambda q, r: f"⚠️ Consult a doctor. Possible: {r}",
)
```

### Generalized Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   GENERALIZED ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ DomainConfig │    │ DomainConfig │    │ DomainConfig │      │
│  │    (bash)    │    │  (cooking)   │    │  (medical)   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                  ┌─────────────────────┐                        │
│                  │  PhiKnowledgeEngine │                        │
│                  │  ─────────────────  │                        │
│                  │  - Co-occurrence    │                        │
│                  │  - Affinity calc    │                        │
│                  │  - Social detect    │                        │
│                  │  - Multi-domain     │                        │
│                  └──────────┬──────────┘                        │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ BashHandler  │    │ CookHandler  │    │ MedHandler   │      │
│  │ (execute)    │    │ (display)    │    │ (warn+show)  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Abstract Core Components

**Goal**: Extract domain-agnostic code into reusable classes.

```python
class CooccurrenceTracker:
    """Domain-agnostic co-occurrence tracking."""
    # Already implemented, just needs extraction

class PhiVocabulary:
    """Domain-agnostic vocabulary with φ-based levels."""
    # Already implemented, just needs extraction

class AffinityMatcher:
    """Domain-agnostic affinity-based matching."""
    
    def __init__(self, anchors: List[str], cooccurrence: CooccurrenceTracker):
        self.anchors = anchors
        self.cooccurrence = cooccurrence
    
    def match(self, query: str) -> Dict[str, float]:
        """Return affinity scores for each anchor."""
        words = tokenize(query)
        affinity = {anchor: 0.0 for anchor in self.anchors}
        
        for word in words:
            for anchor in self.anchors:
                affinity[anchor] += self.cooccurrence.get(word, anchor)
        
        return affinity
```

### Phase 2: Domain Configuration System

**Goal**: Define domains declaratively.

```python
# domains/bash.yaml
name: bash
description: Linux command line assistance

anchors:
  - ls
  - df
  - ps
  - netstat
  - who
  - grep
  - find
  - cat

anchor_extraction:
  patterns:
    - "'([^']+)'"
    - "`([^`]+)`"
  
action:
  type: shell_execute
  timeout: 10
  
social:
  inherit: default  # Use shared social keywords
```

```python
# domains/cooking.yaml
name: cooking
description: Recipe and cooking technique assistance

anchors:
  - bake
  - sauté
  - boil
  - fry
  - roast
  - steam
  - grill
  - simmer

anchor_extraction:
  patterns:
    - "Technique: (\\w+)"
    - "Method: (\\w+)"

action:
  type: display_text
  
social:
  inherit: default
```

### Phase 3: Data Generation Pipeline

**Goal**: Generalize LLM-based data generation.

```python
class DomainDataGenerator:
    """Generate training data for any domain using LLM."""
    
    def __init__(self, domain: DomainConfig, llm_client):
        self.domain = domain
        self.llm = llm_client
    
    def generate_qa_pairs(self, seed_examples: List[Tuple[str, str]], count: int) -> List[Tuple[str, str]]:
        """Generate Q&A pairs using few-shot prompting."""
        
        prompt = f"""
        Domain: {self.domain.name}
        
        Generate question-answer pairs where:
        - Questions ask about {self.domain.description}
        - Answers contain one of these key terms: {self.domain.anchors}
        
        Examples:
        {self._format_examples(seed_examples)}
        
        Generate {count} more pairs in the same format.
        """
        
        return self._parse_response(self.llm.generate(prompt))
```

### Phase 4: Multi-Domain Router

**Goal**: Handle queries across multiple domains.

```python
class MultiDomainEngine:
    """Route queries to appropriate domain handlers."""
    
    def __init__(self):
        self.domains: Dict[str, DomainConfig] = {}
        self.matchers: Dict[str, AffinityMatcher] = {}
    
    def add_domain(self, config: DomainConfig, training_data: List[Tuple[str, str]]):
        """Add a domain with its training data."""
        self.domains[config.name] = config
        
        # Build co-occurrence from training data
        cooccur = CooccurrenceTracker()
        for q, a in training_data:
            cooccur.track(tokenize(q + " " + a))
        
        self.matchers[config.name] = AffinityMatcher(config.anchors, cooccur)
    
    def query(self, text: str) -> Tuple[str, str, Any]:
        """Route query to best domain and return response."""
        
        # Check for social context first
        social = detect_social(text)
        if social['is_social']:
            return ("social", social['social_type'], self._social_response(social))
        
        # Find best domain by max affinity
        best_domain = None
        best_score = 0
        best_anchor = None
        
        for name, matcher in self.matchers.items():
            affinities = matcher.match(text)
            max_anchor = max(affinities, key=affinities.get)
            max_score = affinities[max_anchor]
            
            if max_score > best_score:
                best_score = max_score
                best_domain = name
                best_anchor = max_anchor
        
        if best_domain and best_score > 0:
            config = self.domains[best_domain]
            response = self._get_response(best_domain, best_anchor, text)
            result = config.action_handler(text, response)
            return (best_domain, best_anchor, result)
        
        return ("unknown", None, "I don't have information about that.")
```

## Research Questions

### Q1: Anchor Discovery
**Can we automatically discover anchors from unlabeled text?**

Hypothesis: High-frequency words that appear in response positions (after "Use", "Try", "Run") are likely anchors.

```python
def discover_anchors(corpus: List[Tuple[str, str]], top_k: int = 20) -> List[str]:
    """Discover anchor words from Q&A corpus."""
    # Find words that appear after action verbs in responses
    action_verbs = ["use", "try", "run", "execute", "apply", "make", "do"]
    candidates = Counter()
    
    for q, a in corpus:
        words = tokenize(a)
        for i, word in enumerate(words):
            if word in action_verbs and i + 1 < len(words):
                candidates[words[i + 1]] += 1
    
    return [w for w, _ in candidates.most_common(top_k)]
```

### Q2: Cross-Domain Transfer
**Do co-occurrence patterns transfer between related domains?**

Hypothesis: A model trained on "bash" might partially work for "PowerShell" because command concepts (files, processes, network) are shared.

Experiment:
1. Train on bash Q&A
2. Test on PowerShell queries (without PowerShell training)
3. Measure accuracy degradation

### Q3: Hierarchical Domains
**Can we nest domains for finer-grained matching?**

```
Computing
├── Operating Systems
│   ├── Linux (bash)
│   ├── Windows (PowerShell)
│   └── macOS (zsh)
├── Programming
│   ├── Python
│   ├── JavaScript
│   └── Rust
└── Databases
    ├── SQL
    └── NoSQL
```

Hypothesis: Hierarchical routing improves accuracy by first matching broad domain, then specific subdomain.

### Q4: Minimal Training Data
**What is the minimum training data needed for reliable matching?**

Experiment:
1. Start with 5 Q&A pairs per anchor
2. Measure accuracy
3. Add 5 more, repeat
4. Find the knee of the accuracy curve

Hypothesis: ~20-30 pairs per anchor achieves >90% accuracy.

### Q5: Dynamic Anchor Addition
**Can we add new anchors without retraining?**

Hypothesis: If we add a new anchor with a few examples, the co-occurrence matrix can be incrementally updated.

```python
def add_anchor(self, anchor: str, examples: List[Tuple[str, str]]):
    """Add a new anchor with minimal examples."""
    self.anchors.append(anchor)
    for q, a in examples:
        self.cooccurrence.track(tokenize(q + " " + a))
    # No full retrain needed
```

## Proposed File Structure

```
truthspace-lcm/
├── truthspace_lcm/
│   ├── core/
│   │   ├── cooccurrence.py      # Domain-agnostic co-occurrence
│   │   ├── vocabulary.py        # φ-based vocabulary
│   │   ├── matcher.py           # Affinity matching
│   │   └── social.py            # Social context detection
│   ├── domains/
│   │   ├── base.py              # DomainConfig base class
│   │   ├── bash.py              # Bash domain
│   │   ├── cooking.py           # Cooking domain (example)
│   │   └── loader.py            # YAML domain loader
│   ├── engine/
│   │   ├── single.py            # Single-domain engine
│   │   ├── multi.py             # Multi-domain router
│   │   └── chat.py              # Interactive chat interface
│   └── data/
│       ├── generator.py         # LLM-based data generation
│       └── validator.py         # Data quality checks
├── domains/
│   ├── bash.yaml                # Bash domain config
│   ├── cooking.yaml             # Cooking domain config
│   └── medical.yaml             # Medical domain config
├── data/
│   ├── bash/
│   │   └── qa_pairs.json
│   ├── cooking/
│   │   └── qa_pairs.json
│   └── shared/
│       └── social_keywords.json
└── experiments/
    ├── anchor_discovery.py
    ├── cross_domain_transfer.py
    └── minimal_training.py
```

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Domain addition time | < 1 hour | Time to add new domain with 50 Q&A pairs |
| Accuracy per domain | > 90% | Test set accuracy after training |
| Cross-domain accuracy | > 70% | Related domain without specific training |
| Minimal training | < 30 pairs/anchor | Pairs needed for 90% accuracy |
| Code reuse | > 80% | Shared code between domains |

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Extract domain-agnostic components from prototype |
| 2 | Implement DomainConfig and YAML loader |
| 3 | Build multi-domain router |
| 4 | Add cooking domain as proof of generalization |
| 5 | Run anchor discovery experiments |
| 6 | Run minimal training experiments |
| 7 | Documentation and API refinement |
| 8 | Release v1.0 of generalized system |

## Conclusion

The current bash-specific implementation demonstrates that **co-occurrence based cluster matching works**. The key insight—that clusters emerge from data through co-occurrence—is domain-agnostic.

To generalize:
1. **Abstract anchors**: Replace hardcoded commands with configurable anchor lists
2. **Abstract actions**: Replace bash execution with pluggable action handlers
3. **Abstract data generation**: Use LLM few-shot prompting with domain-specific seeds
4. **Build multi-domain routing**: Let affinity scores determine which domain handles a query

The φ-based geometric foundation (vocabulary levels, phase clustering, attractor dynamics) remains unchanged. Only the **surface configuration** changes per domain.

This positions TruthSpace-LCM as a **general-purpose knowledge ingestion framework** rather than a bash-specific tool.

## References

- Design Consideration 025: Co-occurrence Cluster Matching
- Design Consideration 022: Attractor/Repeller Dynamics
- `experiments/phi_ingestion_prototype.py` - Current implementation
- `experiments/llm_data_generator.py` - LLM data generation
