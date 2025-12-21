# Software Requirements Specification
# TruthSpace Geometric Chat System (GCS)

**Version:** 1.0  
**Date:** December 21, 2024  
**Project:** TruthSpace LCM Foundation  

---

## 1. Introduction

### 1.1 Purpose

This document specifies the software requirements for the **Geometric Chat System (GCS)**, a conversational AI system that replaces traditional neural network-based Large Language Models (LLMs) with a purely geometric approach based on holographic projection and semantic space navigation.

### 1.2 Scope

The GCS will provide:
- Natural language question answering
- Style-aware response generation
- Automatic knowledge ingestion from text sources
- Style extraction and transfer capabilities
- A generalizable, scalable framework for semantic understanding

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| **GCS** | Geometric Chat System |
| **LCM** | Language Concept Model (TruthSpace's geometric alternative to LLMs) |
| **Centroid** | Average position of exemplars in semantic space; represents a style or concept |
| **Holographic Projection** | Encoding method where information is distributed across the entire representation |
| **Semantic Space** | High-dimensional vector space where meaning is represented geometrically |
| **Style** | A position/region in semantic space characterized by linguistic patterns |
| **Gap-Filling** | Q&A method where questions define gaps and answers fill them |

### 1.4 Design Philosophy

The system adheres to these core principles:

1. **Geometric over Statistical**: All operations are vector arithmetic, not learned weights
2. **No Hardcoding**: Styles, knowledge, and behaviors emerge from data, not code
3. **Holographic Encoding**: Information distributed across representations
4. **Composable**: Small operations combine to create complex behaviors
5. **Interpretable**: Every decision can be traced to geometric relationships

---

## 2. Overall Description

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GEOMETRIC CHAT SYSTEM                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   INGEST    │    │   ENCODE    │    │   STORE     │             │
│  │             │───▶│             │───▶│             │             │
│  │ Gutenberg   │    │ Vocabulary  │    │ Knowledge   │             │
│  │ Files       │    │ Centroid    │    │ Base        │             │
│  │ Q&A Pairs   │    │ Style       │    │ Style DB    │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│         │                                     │                     │
│         │                                     │                     │
│         ▼                                     ▼                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   QUERY     │    │   MATCH     │    │  GENERATE   │             │
│  │             │───▶│             │───▶│             │             │
│  │ Parse       │    │ Geometric   │    │ Style       │             │
│  │ Encode      │    │ Similarity  │    │ Transfer    │             │
│  │ Classify    │    │ Gap-Fill    │    │ Response    │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

| Component | Responsibility |
|-----------|----------------|
| **Ingestion Engine** | Load and parse text from various sources |
| **Vocabulary System** | Word embeddings and text encoding |
| **Knowledge Base** | Store facts, triples, and Q&A pairs |
| **Style Engine** | Extract, store, and apply styles |
| **Query Processor** | Parse and encode user queries |
| **Matching Engine** | Geometric similarity and gap-filling |
| **Response Generator** | Compose and style responses |
| **Chat Interface** | User interaction layer |

### 2.3 User Classes

| User Class | Description |
|------------|-------------|
| **End User** | Interacts via chat interface to ask questions |
| **Content Admin** | Ingests new knowledge sources |
| **Style Admin** | Defines and manages styles |
| **Developer** | Extends system capabilities |

---

## 3. Functional Requirements

### 3.1 Ingestion System

#### FR-ING-001: Project Gutenberg Auto-Ingestion
- **Description**: System shall automatically download and ingest books from Project Gutenberg
- **Input**: Book ID or search query
- **Output**: Parsed knowledge stored in knowledge base
- **Priority**: High

#### FR-ING-002: Text File Ingestion
- **Description**: System shall ingest plain text files
- **Input**: File path or text content
- **Output**: Parsed knowledge stored in knowledge base
- **Priority**: High

#### FR-ING-003: Q&A Pair Ingestion
- **Description**: System shall ingest structured Q&A pairs
- **Input**: List of (question, answer) tuples or JSON/CSV file
- **Output**: Q&A pairs stored with encodings
- **Priority**: High

#### FR-ING-004: Automatic Chunking
- **Description**: System shall automatically split text into semantic chunks
- **Chunking modes**: Sentence, paragraph, chapter, custom regex
- **Priority**: Medium

#### FR-ING-005: Triple Extraction
- **Description**: System shall extract semantic triples (subject, predicate, object) from text
- **Output**: Structured triples with encodings
- **Priority**: Medium

#### FR-ING-006: Q&A Generation
- **Description**: System shall automatically generate Q&A pairs from ingested text
- **Question types**: WHO, WHAT, WHERE, WHEN, WHY, HOW
- **Priority**: High

### 3.2 Style System

#### FR-STY-001: Style Extraction
- **Description**: System shall extract style from any text source
- **Input**: Text content or file
- **Output**: Style object (centroid + metadata)
- **Priority**: High

#### FR-STY-002: Style Classification
- **Description**: System shall classify text against known styles
- **Input**: Text content
- **Output**: Ranked list of (style, similarity_score)
- **Priority**: High

#### FR-STY-003: Style Transfer
- **Description**: System shall apply style to content
- **Input**: Content text, target style, strength (0-1)
- **Output**: Styled content representation
- **Priority**: High

#### FR-STY-004: Style Persistence
- **Description**: System shall save and load styles
- **Format**: JSON with centroid vectors and metadata
- **Priority**: Medium

#### FR-STY-005: Style Comparison
- **Description**: System shall compute differences between styles
- **Output**: Direction vector, characteristic words for each direction
- **Priority**: Medium

#### FR-STY-006: Built-in Styles
- **Description**: System shall include predefined styles
- **Styles**: Q&A, Technical, Narrative, Formal, Casual
- **Priority**: Medium

### 3.3 Query Processing

#### FR-QRY-001: Natural Language Query
- **Description**: System shall accept natural language questions
- **Input**: Free-form text query
- **Output**: Encoded query vector with detected intent
- **Priority**: High

#### FR-QRY-002: Question Type Detection
- **Description**: System shall detect question type (WHO/WHAT/WHERE/WHEN/WHY/HOW)
- **Method**: Pattern matching + geometric classification
- **Priority**: High

#### FR-QRY-003: Query Encoding
- **Description**: System shall encode queries into semantic space
- **Method**: IDF-weighted word vector averaging
- **Priority**: High

#### FR-QRY-004: Context Tracking
- **Description**: System shall maintain conversation context
- **Scope**: Current session
- **Priority**: Medium

### 3.4 Matching and Retrieval

#### FR-MAT-001: Semantic Similarity Search
- **Description**: System shall find semantically similar content
- **Method**: Cosine similarity in semantic space
- **Priority**: High

#### FR-MAT-002: Gap-Filling Match
- **Description**: System shall match questions to answers via gap-filling
- **Method**: Question defines gap, answer fills gap
- **Priority**: High

#### FR-MAT-003: Multi-Source Retrieval
- **Description**: System shall retrieve from multiple knowledge sources
- **Sources**: Facts, triples, Q&A pairs, raw text
- **Priority**: Medium

#### FR-MAT-004: Confidence Scoring
- **Description**: System shall provide confidence scores for matches
- **Range**: 0.0 to 1.0
- **Priority**: Medium

### 3.5 Response Generation

#### FR-RES-001: Answer Composition
- **Description**: System shall compose answers from retrieved knowledge
- **Priority**: High

#### FR-RES-002: Style Application
- **Description**: System shall apply selected style to responses
- **Priority**: High

#### FR-RES-003: Source Attribution
- **Description**: System shall cite sources for answers
- **Priority**: Medium

#### FR-RES-004: Uncertainty Expression
- **Description**: System shall express uncertainty when confidence is low
- **Priority**: Medium

### 3.6 Chat Interface

#### FR-CHT-001: Interactive CLI
- **Description**: System shall provide command-line chat interface
- **Priority**: High

#### FR-CHT-002: Style Selection Command
- **Description**: User shall be able to select response style
- **Command**: `/style <style_name>` or `/style list`
- **Priority**: High

#### FR-CHT-003: Knowledge Source Command
- **Description**: User shall be able to select knowledge source
- **Command**: `/source <source_name>` or `/source list`
- **Priority**: Medium

#### FR-CHT-004: Ingest Command
- **Description**: User shall be able to ingest new content
- **Command**: `/ingest <url_or_path>`
- **Priority**: Medium

#### FR-CHT-005: Analyze Command
- **Description**: User shall be able to analyze text style
- **Command**: `/analyze <text>` or `/analyze file <path>`
- **Priority**: Medium

#### FR-CHT-006: Debug Mode
- **Description**: System shall provide debug output showing geometric operations
- **Command**: `/debug on|off`
- **Priority**: Low

---

## 4. Non-Functional Requirements

### 4.1 Performance

#### NFR-PRF-001: Query Response Time
- **Requirement**: Query response within 500ms for knowledge bases up to 10,000 facts
- **Priority**: High

#### NFR-PRF-002: Ingestion Speed
- **Requirement**: Ingest 100KB of text within 5 seconds
- **Priority**: Medium

#### NFR-PRF-003: Memory Efficiency
- **Requirement**: Base memory usage under 500MB
- **Priority**: Medium

### 4.2 Scalability

#### NFR-SCL-001: Knowledge Base Size
- **Requirement**: Support knowledge bases up to 1 million facts
- **Priority**: Medium

#### NFR-SCL-002: Style Count
- **Requirement**: Support unlimited concurrent styles
- **Priority**: Low

### 4.3 Extensibility

#### NFR-EXT-001: Plugin Architecture
- **Requirement**: Support custom ingestion plugins
- **Priority**: Low

#### NFR-EXT-002: Custom Encoders
- **Requirement**: Support custom text encoding methods
- **Priority**: Low

### 4.4 Documentation

#### NFR-DOC-001: API Documentation
- **Requirement**: All public APIs documented with docstrings
- **Priority**: High

#### NFR-DOC-002: Architecture Documentation
- **Requirement**: System architecture documented in design docs
- **Priority**: High

#### NFR-DOC-003: User Guide
- **Requirement**: End-user documentation for chat commands
- **Priority**: Medium

---

## 5. Data Requirements

### 5.1 Knowledge Base Schema

```python
@dataclass
class Fact:
    id: str
    content: str
    encoding: np.ndarray
    source: str
    metadata: Dict

@dataclass
class Triple:
    id: str
    subject: str
    predicate: str
    object: str
    modifiers: Dict[str, str]
    encoding: np.ndarray
    source: str

@dataclass
class QAPair:
    id: str
    question: str
    answer: str
    question_type: str  # WHO, WHAT, WHERE, WHEN, WHY, HOW
    question_encoding: np.ndarray
    answer_encoding: np.ndarray
    source: str
```

### 5.2 Style Schema

```python
@dataclass
class Style:
    name: str
    centroid: np.ndarray
    exemplar_count: int
    characteristic_words: List[Tuple[str, float]]
    metadata: Dict
```

### 5.3 Vocabulary Schema

```python
@dataclass
class Vocabulary:
    dim: int
    word_positions: Dict[str, np.ndarray]
    word_counts: Counter
```

### 5.4 Persistence

| Data | Format | Location |
|------|--------|----------|
| Knowledge Base | JSON + NumPy | `data/knowledge/` |
| Styles | JSON | `data/styles/` |
| Vocabulary | Pickle | `data/vocab.pkl` |
| Configuration | YAML | `config/` |

---

## 6. Interface Requirements

### 6.1 Command Line Interface

```
$ gcs chat
Welcome to Geometric Chat System v1.0
Type /help for commands, /quit to exit

You: Who is Captain Ahab?
GCS: Captain Ahab is the monomaniacal captain of the Pequod who is obsessed 
     with hunting the white whale Moby Dick. [Source: Moby Dick, confidence: 0.87]

You: /style Warhammer
Style set to: Warhammer

You: Who is Captain Ahab?
GCS: Captain Ahab is a zealous commander consumed by righteous fury against 
     the xenos leviathan. His obsession burns with the Emperor's own wrath.
     [Source: Moby Dick, style: Warhammer, confidence: 0.82]

You: /analyze The rain fell like tears on the dark city streets.
GCS: Style analysis:
     - Noir: 0.513 (best match)
     - Neutral: 0.104
     - Romance: 0.047
     Characteristic elements: atmospheric, metaphorical, melancholic

You: /ingest https://www.gutenberg.org/ebooks/2701
GCS: Ingesting "Moby Dick" from Project Gutenberg...
     Extracted 1,247 sentences, 342 triples, 856 Q&A pairs
     Style extracted: "MobyDick" (narrative, formal, descriptive)
     Done.

You: /quit
Goodbye!
```

### 6.2 Python API

```python
from gcs import GeometricChatSystem

# Initialize
gcs = GeometricChatSystem()

# Ingest knowledge
gcs.ingest_gutenberg(2701)  # Moby Dick
gcs.ingest_file("my_document.txt")
gcs.ingest_qa_pairs([("What is X?", "X is Y"), ...])

# Query
response = gcs.query("Who is Captain Ahab?")
print(response.answer)
print(response.confidence)
print(response.sources)

# Style operations
gcs.extract_style("path/to/text.txt", "MyStyle")
gcs.set_style("Warhammer")
analysis = gcs.analyze_style("Some text to analyze")

# Low-level access
encoding = gcs.encode("Some text")
similarity = gcs.similarity("text1", "text2")
```

---

## 7. System Constraints

### 7.1 Technical Constraints

| Constraint | Description |
|------------|-------------|
| **No Neural Networks** | System must not use trained neural networks |
| **No External APIs** | Core functionality must not require external AI APIs |
| **Pure Python** | Core system in Python with NumPy only |
| **Deterministic** | Same input must produce same output |

### 7.2 Design Constraints

| Constraint | Description |
|------------|-------------|
| **Geometric Operations** | All semantic operations must be expressible as vector arithmetic |
| **No Hardcoded Knowledge** | Knowledge must come from ingested data, not code |
| **Style Agnostic** | System must not favor any particular style |

---

## 8. Quality Attributes

### 8.1 Correctness Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Style Classification Accuracy | >80% | Test suite of labeled examples |
| Q&A Relevance | >70% | Human evaluation of top-1 answers |
| Source Attribution | 100% | Every answer cites source |

### 8.2 Code Quality

| Metric | Target |
|--------|--------|
| Test Coverage | >80% |
| Documentation Coverage | 100% public APIs |
| Type Hints | All function signatures |

---

## 9. Implementation Roadmap

### Phase 1: Core Foundation (MVP)
- [ ] Vocabulary system with encoding
- [ ] Basic knowledge base (facts)
- [ ] Semantic similarity search
- [ ] Simple CLI chat interface
- [ ] File ingestion

### Phase 2: Q&A System
- [ ] Triple extraction
- [ ] Q&A pair generation
- [ ] Gap-filling matching
- [ ] Question type detection
- [ ] Confidence scoring

### Phase 3: Style System
- [ ] Style extraction
- [ ] Style classification
- [ ] Style transfer
- [ ] Style persistence
- [ ] Built-in styles

### Phase 4: Gutenberg Integration
- [ ] Gutenberg API client
- [ ] Auto-download and parse
- [ ] Metadata extraction
- [ ] Batch ingestion

### Phase 5: Polish
- [ ] Full CLI with all commands
- [ ] Debug mode
- [ ] Performance optimization
- [ ] Comprehensive documentation
- [ ] Test suite

---

## 10. Appendices

### Appendix A: Geometric Foundations

The system is built on these geometric principles:

1. **Centroid = Style**: A style is the average position of its exemplars
   ```
   style_centroid = mean([encode(e) for e in exemplars])
   ```

2. **Similarity = Cosine**: Semantic similarity is cosine similarity
   ```
   similarity = dot(a, b) / (norm(a) * norm(b))
   ```

3. **Transfer = Interpolation**: Style transfer interpolates toward centroid
   ```
   styled = (1 - α) * content + α * style_centroid
   ```

4. **Direction = Difference**: Style difference is vector subtraction
   ```
   direction = style_b.centroid - style_a.centroid
   ```

5. **Gap-Filling = Projection**: Q&A matching projects onto question axis
   ```
   relevance = dot(answer, question_axis)
   ```

### Appendix B: File Structure

```
truthspace-lcm/
├── gcs/                          # Main package
│   ├── __init__.py
│   ├── core/
│   │   ├── vocabulary.py         # Word embeddings
│   │   ├── encoding.py           # Text encoding
│   │   └── similarity.py         # Geometric operations
│   ├── knowledge/
│   │   ├── base.py               # Knowledge base
│   │   ├── facts.py              # Fact storage
│   │   ├── triples.py            # Triple extraction
│   │   └── qa.py                 # Q&A pairs
│   ├── style/
│   │   ├── extractor.py          # Style extraction
│   │   ├── classifier.py         # Style classification
│   │   └── transfer.py           # Style transfer
│   ├── ingest/
│   │   ├── gutenberg.py          # Project Gutenberg
│   │   ├── text.py               # Plain text
│   │   └── qa_pairs.py           # Q&A pairs
│   ├── query/
│   │   ├── parser.py             # Query parsing
│   │   ├── matcher.py            # Matching engine
│   │   └── response.py           # Response generation
│   └── chat/
│       ├── cli.py                # CLI interface
│       └── commands.py           # Chat commands
├── data/
│   ├── knowledge/                # Persisted knowledge
│   ├── styles/                   # Persisted styles
│   └── vocab.pkl                 # Vocabulary
├── config/
│   └── default.yaml              # Configuration
├── tests/
│   └── ...                       # Test suite
├── docs/
│   ├── SRS_geometric_chat_system.md  # This document
│   └── ...                       # Other docs
└── examples/
    └── ...                       # Usage examples
```

### Appendix C: Related Design Documents

| Document | Description |
|----------|-------------|
| `design_considerations/030_geometric_qa_projection.md` | Q&A projection framework |
| `design_considerations/031_unified_projection_framework.md` | Unified style/Q&A theory |
| `design_considerations/019_holographic_resolution.md` | Holographic encoding principles |

### Appendix D: Prototype Implementations

| File | Description |
|------|-------------|
| `papers/style_centroid.py` | Style centroid approach (validated) |
| `papers/style_extractor.py` | On-the-fly style extraction |
| `papers/recursive_holographic_qa.py` | Recursive Q&A projection |
| `papers/holographic_qa_general.py` | Generalized Q&A system |
| `experiments/semantic_chatbot.py` | Early chatbot prototype |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-21 | TruthSpace Team | Initial SRS |
