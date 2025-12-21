# Software Design Specification
# TruthSpace Geometric Chat System (GCS)

**Version:** 1.0  
**Date:** December 21, 2024  
**Project:** TruthSpace LCM Foundation  
**Companion Document:** SRS_geometric_chat_system.md

---

## 1. Introduction

### 1.1 Purpose

This document provides the complete technical design for the Geometric Chat System (GCS). It specifies the exact mathematical methods, algorithms, and data structures required to implement a conversational AI system using purely geometric operations—no neural networks, no learned weights, no statistical inference.

This document is intended to be **self-contained**: a developer with no prior context should be able to implement the system from this specification alone.

### 1.2 Design Philosophy

The GCS is built on a single unifying principle:

> **All semantic operations are geometric operations in vector space.**

This means:
- **Meaning** = Position in semantic space
- **Similarity** = Distance/angle between positions
- **Style** = Region/centroid in semantic space
- **Understanding** = Projection onto relevant axes
- **Generation** = Movement through semantic space

### 1.3 Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| **v** | Vector (bold lowercase) |
| **M** | Matrix (bold uppercase) |
| ‖**v**‖ | Euclidean norm of vector |
| **v** · **w** | Dot product |
| cos(θ) | Cosine similarity |
| φ | Golden ratio ≈ 1.618 |
| dim | Dimensionality of semantic space |

---

## 2. Theoretical Foundation

### 2.1 The Holographic Principle

The system is based on **holographic encoding**, where information is distributed across the entire representation rather than localized.

**Key Properties:**
1. **Superposition**: Multiple concepts can be encoded in the same vector
2. **Interference**: Similar concepts reinforce; dissimilar concepts cancel
3. **Reconstruction**: Partial information can reconstruct the whole

**Analogy to Physical Holography:**
```
Physical:   I = I_L + I_R     (interference pattern stores both views)
            I_R - I_L = depth  (difference reveals hidden dimension)

Semantic:   centroid = Σ exemplars / n    (average stores essence)
            style_B - style_A = direction  (difference reveals style axis)
```

### 2.2 Semantic Space

All text exists as points in a high-dimensional **semantic space** ℝ^dim.

**Properties of Semantic Space:**
- Words that mean similar things are close together
- Words that mean different things are far apart
- Directions in space correspond to semantic relationships
- Distances correspond to semantic differences

**The Space is NOT Learned:**
Unlike word2vec or transformer embeddings, our semantic space is constructed deterministically from word co-occurrence and hash-based positioning. This ensures:
- Reproducibility (same input → same position)
- No training required
- Interpretable geometry

### 2.3 The Centroid Theorem

**Theorem:** A style (or concept, or category) is fully characterized by the centroid of its exemplars.

**Proof by Construction:**
1. Let S = {s₁, s₂, ..., sₙ} be exemplars of style S
2. Let encode(sᵢ) = **vᵢ** ∈ ℝ^dim
3. Define centroid: **c_S** = (1/n) Σᵢ **vᵢ**
4. For any new text t, style membership is: sim(t, S) = cos(**v_t**, **c_S**)

**Why This Works:**
- The centroid captures the "average" semantic position
- Outliers are averaged out
- Core characteristics are reinforced
- The centroid IS the style, geometrically

**Experimental Validation:**
- 8/8 correct style classifications (Warhammer, Romance, Noir, Technical)
- 6/6 correct author style detection (Hemingway, Lovecraft, Technical, Q&A)

### 2.4 The Gap-Filling Principle

**Principle:** A question defines a "gap" in semantic space. An answer "fills" that gap.

```
Question: "Who is Captain Ahab?"
         = encode("Captain Ahab") + GAP(identity)

Answer:   "Captain Ahab is the captain of the Pequod"
         = encode("Captain Ahab") + FILL(identity: "captain of the Pequod")

Match:    GAP ≈ FILL  →  Answer is relevant
```

**Question Types as Axes:**
| Type | Axis Direction | What It Seeks |
|------|----------------|---------------|
| WHO | Identity axis | Person/entity |
| WHAT | Definition axis | Thing/concept |
| WHERE | Location axis | Place |
| WHEN | Time axis | Temporal reference |
| WHY | Reason axis | Cause/motivation |
| HOW | Method axis | Process/mechanism |

---

## 3. Core Algorithms

### 3.1 Text Encoding

#### 3.1.1 Tokenization

```python
def tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r'\w+', text.lower())
```

#### 3.1.2 Word Position Assignment

Each word is assigned a deterministic position in ℝ^dim using a hash function:

```python
def word_position(word: str, dim: int) -> np.ndarray:
    """
    Assign deterministic position to word.
    
    Uses hash as random seed for reproducibility.
    Position is on unit hypersphere.
    """
    seed = hash(word) % (2**32)
    np.random.seed(seed)
    position = np.random.randn(dim)
    np.random.seed(None)  # Reset
    
    # Normalize to unit sphere
    return position / np.linalg.norm(position)
```

**Properties:**
- Same word always gets same position
- Different words get (almost certainly) different positions
- Positions are uniformly distributed on unit hypersphere

#### 3.1.3 Text Encoding with IDF Weighting

```python
def encode_text(text: str, vocab: Vocabulary) -> np.ndarray:
    """
    Encode text as IDF-weighted average of word positions.
    
    IDF weighting: rare words matter more than common words.
    
    Formula:
        v = Σᵢ wᵢ · pos(wordᵢ) / Σᵢ wᵢ
        where wᵢ = 1 / log(1 + count(wordᵢ))
    """
    words = tokenize(text)
    if not words:
        return np.zeros(vocab.dim)
    
    positions = []
    weights = []
    
    for word in words:
        pos = vocab.get_position(word)
        positions.append(pos)
        
        # IDF-like weight: rare words get higher weight
        count = vocab.word_counts.get(word, 1)
        weight = 1.0 / math.log(1 + count)
        weights.append(weight)
    
    positions = np.array(positions)
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    return np.average(positions, axis=0, weights=weights)
```

**Why IDF Weighting:**
- Common words ("the", "is", "a") contribute less
- Rare, meaningful words ("Ahab", "whale", "obsession") contribute more
- This focuses the encoding on semantically significant content

### 3.2 Similarity Computation

#### 3.2.1 Cosine Similarity

```python
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Formula:
        cos(θ) = (v1 · v2) / (‖v1‖ · ‖v2‖)
    
    Range: [-1, 1]
        1  = identical direction
        0  = orthogonal (unrelated)
        -1 = opposite direction
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    return np.dot(v1, v2) / (norm1 * norm2)
```

#### 3.2.2 Euclidean Distance

```python
def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Formula:
        d = ‖v1 - v2‖ = √(Σᵢ (v1ᵢ - v2ᵢ)²)
    """
    return np.linalg.norm(v1 - v2)
```

#### 3.2.3 When to Use Which

| Metric | Use Case | Why |
|--------|----------|-----|
| Cosine | Style classification | Direction matters, not magnitude |
| Cosine | Q&A matching | Semantic direction alignment |
| Euclidean | Style transfer | Actual distance to move |
| Euclidean | Nearest neighbor | Absolute proximity |

### 3.3 Style Operations

#### 3.3.1 Style Extraction (Centroid Computation)

```python
def extract_style(exemplars: List[str], vocab: Vocabulary) -> Style:
    """
    Extract style from exemplars.
    
    Algorithm:
    1. Encode each exemplar
    2. Compute centroid (mean position)
    3. Apply style bias to vocabulary (optional refinement)
    4. Recompute centroid with biased vocabulary
    5. Find characteristic words
    
    The centroid IS the style.
    """
    # First pass: encode all exemplars
    encodings = [encode_text(e, vocab) for e in exemplars]
    
    # Compute rough centroid
    rough_centroid = np.mean(encodings, axis=0)
    
    # Second pass: bias vocabulary toward style (refinement)
    for exemplar in exemplars:
        vocab.add_text(exemplar, style_bias=rough_centroid, bias_strength=0.15)
    
    # Recompute with biased vocabulary
    encodings = [encode_text(e, vocab) for e in exemplars]
    centroid = np.mean(encodings, axis=0)
    
    # Find characteristic words (nearest to centroid)
    char_words = vocab.nearest_words(centroid, k=15)
    
    return Style(
        centroid=centroid,
        exemplar_count=len(exemplars),
        characteristic_words=char_words
    )
```

**Style Bias Refinement:**

The optional bias step pulls word positions toward the style centroid:

```python
def apply_style_bias(word_pos: np.ndarray, 
                     style_centroid: np.ndarray, 
                     strength: float = 0.15) -> np.ndarray:
    """
    Bias word position toward style centroid.
    
    Formula:
        new_pos = (1 - α) · old_pos + α · centroid
    
    This makes words that appear in style exemplars
    cluster more tightly around the style centroid.
    """
    return (1 - strength) * word_pos + strength * style_centroid
```

#### 3.3.2 Style Classification

```python
def classify_style(text: str, styles: Dict[str, Style], vocab: Vocabulary) -> List[Tuple[str, float]]:
    """
    Classify text against known styles.
    
    Algorithm:
    1. Encode text
    2. Compute cosine similarity to each style centroid
    3. Rank by similarity
    
    Returns: [(style_name, similarity), ...] sorted descending
    """
    text_vec = encode_text(text, vocab)
    
    results = []
    for name, style in styles.items():
        sim = cosine_similarity(text_vec, style.centroid)
        results.append((name, sim))
    
    return sorted(results, key=lambda x: -x[1])
```

#### 3.3.3 Style Transfer

```python
def transfer_style(content: str, 
                   target_style: Style, 
                   vocab: Vocabulary,
                   strength: float = 0.5) -> Tuple[np.ndarray, List[str]]:
    """
    Transfer content toward target style.
    
    Algorithm:
    1. Encode content
    2. Interpolate toward style centroid
    3. Find nearest words to result (excluding original words)
    
    Formula:
        styled_vec = (1 - α) · content_vec + α · style_centroid
    
    The strength α controls how much style to apply:
        α = 0: No change (original content)
        α = 1: Pure style (lose content)
        α = 0.5: Balanced blend
    """
    content_vec = encode_text(content, vocab)
    styled_vec = (1 - strength) * content_vec + strength * target_style.centroid
    
    # Find words that characterize the styled result
    content_words = set(tokenize(content))
    nearest = vocab.nearest_words(styled_vec, k=15, exclude=content_words)
    
    return styled_vec, [word for word, sim in nearest]
```

#### 3.3.4 Style Difference (Direction)

```python
def style_difference(style_a: Style, style_b: Style, vocab: Vocabulary) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute what makes style_b different from style_a.
    
    Formula:
        direction = centroid_b - centroid_a
    
    Words aligned with direction → characteristic of B
    Words aligned with -direction → characteristic of A
    """
    direction = style_b.centroid - style_a.centroid
    
    toward_b = vocab.nearest_words(direction, k=10)
    toward_a = vocab.nearest_words(-direction, k=10)
    
    return direction, toward_b, toward_a
```

### 3.4 Knowledge Base Operations

#### 3.4.1 Fact Storage

```python
@dataclass
class Fact:
    id: str
    content: str
    encoding: np.ndarray
    source: str
    metadata: Dict = field(default_factory=dict)

class KnowledgeBase:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.facts: Dict[str, Fact] = {}
    
    def add_fact(self, content: str, source: str, metadata: Dict = None) -> Fact:
        """Add a fact to the knowledge base."""
        fact_id = hashlib.md5(content.encode()).hexdigest()[:12]
        encoding = encode_text(content, self.vocab)
        
        fact = Fact(
            id=fact_id,
            content=content,
            encoding=encoding,
            source=source,
            metadata=metadata or {}
        )
        self.facts[fact_id] = fact
        return fact
```

#### 3.4.2 Semantic Search

```python
def search(self, query: str, k: int = 5) -> List[Tuple[Fact, float]]:
    """
    Find k most similar facts to query.
    
    Algorithm:
    1. Encode query
    2. Compute similarity to all facts
    3. Return top k by similarity
    
    Complexity: O(n) where n = number of facts
    For large knowledge bases, use approximate nearest neighbor.
    """
    query_vec = encode_text(query, self.vocab)
    
    results = []
    for fact in self.facts.values():
        sim = cosine_similarity(query_vec, fact.encoding)
        results.append((fact, sim))
    
    results.sort(key=lambda x: -x[1])
    return results[:k]
```

### 3.5 Q&A System

#### 3.5.1 Question Type Detection

```python
QUESTION_PATTERNS = {
    'WHO': ['who is', 'who was', 'who are', 'who did', 'whose'],
    'WHAT': ['what is', 'what was', 'what are', 'what does', 'define'],
    'WHERE': ['where is', 'where was', 'where did', 'location of'],
    'WHEN': ['when did', 'when was', 'when is', 'what year', 'what time'],
    'WHY': ['why did', 'why does', 'why is', 'reason for', 'cause of'],
    'HOW': ['how did', 'how does', 'how do', 'how to', 'method of'],
}

def detect_question_type(question: str) -> str:
    """
    Detect question type from patterns.
    
    Returns: 'WHO', 'WHAT', 'WHERE', 'WHEN', 'WHY', 'HOW', or 'UNKNOWN'
    """
    question_lower = question.lower()
    
    for qtype, patterns in QUESTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in question_lower:
                return qtype
    
    return 'UNKNOWN'
```

#### 3.5.2 Triple Extraction

```python
@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    modifiers: Dict[str, str] = field(default_factory=dict)

def extract_triples(sentence: str) -> List[Triple]:
    """
    Extract semantic triples from a sentence.
    
    Pattern matching approach:
    1. Find subject (usually first noun phrase)
    2. Find predicate (main verb)
    3. Find object (noun phrase after verb)
    4. Extract modifiers (prepositional phrases)
    
    Example:
        "Captain Ahab hunted the white whale in the Pacific."
        → Triple(
            subject="Captain Ahab",
            predicate="hunted",
            object="the white whale",
            modifiers={"location": "the Pacific"}
        )
    """
    # Simplified extraction using regex patterns
    # Full implementation uses more sophisticated NLP
    
    # Pattern: Subject + Verb + Object
    pattern = r'^([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(is|was|are|were|has|had|did|does|[a-z]+ed|[a-z]+s)\s+(.+?)(?:\.|$)'
    
    match = re.match(pattern, sentence)
    if match:
        subject, predicate, rest = match.groups()
        
        # Extract location modifier
        loc_match = re.search(r'\b(in|at|on|near)\s+([^,\.]+)', rest)
        modifiers = {}
        if loc_match:
            modifiers['location'] = loc_match.group(2)
            rest = rest[:loc_match.start()].strip()
        
        return [Triple(
            subject=subject.strip(),
            predicate=predicate.strip(),
            object=rest.strip(),
            modifiers=modifiers
        )]
    
    return []
```

#### 3.5.3 Q&A Pair Generation

```python
def generate_qa_pairs(triple: Triple) -> List[Tuple[str, str, str]]:
    """
    Generate Q&A pairs from a triple.
    
    Returns: [(question, answer, question_type), ...]
    """
    pairs = []
    
    # WHO question
    if triple.subject:
        q = f"Who {triple.predicate} {triple.object}?"
        a = f"{triple.subject} {triple.predicate} {triple.object}."
        pairs.append((q, a, 'WHO'))
    
    # WHAT question
    if triple.object:
        q = f"What did {triple.subject} {triple.predicate}?"
        a = f"{triple.subject} {triple.predicate} {triple.object}."
        pairs.append((q, a, 'WHAT'))
    
    # WHERE question
    if 'location' in triple.modifiers:
        loc = triple.modifiers['location']
        q = f"Where did {triple.subject} {triple.predicate} {triple.object}?"
        a = f"{triple.subject} {triple.predicate} {triple.object} in {loc}."
        pairs.append((q, a, 'WHERE'))
    
    return pairs
```

#### 3.5.4 Gap-Filling Match

```python
def gap_fill_match(question: str, 
                   candidates: List[Tuple[str, str]], 
                   vocab: Vocabulary) -> List[Tuple[str, str, float]]:
    """
    Match question to candidate answers using gap-filling.
    
    Algorithm:
    1. Encode question
    2. For each candidate (q, a) pair:
       a. Encode candidate question
       b. Compute similarity between query and candidate question
       c. If similar, the candidate answer fills the gap
    3. Rank by similarity
    
    The intuition: Questions with similar structure have similar gaps.
    If my question matches a stored question, the stored answer fills my gap.
    """
    query_vec = encode_text(question, vocab)
    
    results = []
    for cand_q, cand_a in candidates:
        cand_q_vec = encode_text(cand_q, vocab)
        sim = cosine_similarity(query_vec, cand_q_vec)
        results.append((cand_q, cand_a, sim))
    
    results.sort(key=lambda x: -x[2])
    return results
```

### 3.6 Response Generation

#### 3.6.1 Answer Composition

```python
def compose_answer(query: str,
                   matches: List[Tuple[Fact, float]],
                   style: Optional[Style],
                   vocab: Vocabulary,
                   confidence_threshold: float = 0.3) -> Response:
    """
    Compose answer from matched facts.
    
    Algorithm:
    1. Filter matches by confidence threshold
    2. Select best match (or combine if multiple high-confidence)
    3. Apply style if specified
    4. Format response with source attribution
    """
    # Filter by confidence
    good_matches = [(f, s) for f, s in matches if s >= confidence_threshold]
    
    if not good_matches:
        return Response(
            answer="I don't have enough information to answer that question.",
            confidence=0.0,
            sources=[]
        )
    
    # Take best match
    best_fact, confidence = good_matches[0]
    answer = best_fact.content
    
    # Apply style if specified
    if style is not None:
        styled_vec, style_words = transfer_style(answer, style, vocab, strength=0.4)
        # In full implementation, use style_words to modify answer text
        # For now, we note the style was applied
    
    return Response(
        answer=answer,
        confidence=confidence,
        sources=[best_fact.source],
        style=style.name if style else None
    )
```

---

## 4. Data Structures

### 4.1 Vocabulary

```python
@dataclass
class Vocabulary:
    """
    Word embedding vocabulary.
    
    Stores word positions and counts for IDF weighting.
    """
    dim: int = 64  # Dimensionality of semantic space
    word_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    word_counts: Counter = field(default_factory=Counter)
    
    def get_position(self, word: str) -> np.ndarray:
        """Get or create position for word."""
        word = word.lower()
        if word not in self.word_positions:
            self.word_positions[word] = word_position(word, self.dim)
            self.word_counts[word] = 0
        self.word_counts[word] += 1
        return self.word_positions[word]
    
    def add_text(self, text: str, style_bias: np.ndarray = None, bias_strength: float = 0.15):
        """Add text to vocabulary, optionally with style bias."""
        for word in tokenize(text):
            pos = self.get_position(word)
            if style_bias is not None:
                self.word_positions[word] = apply_style_bias(pos, style_bias, bias_strength)
    
    def nearest_words(self, vector: np.ndarray, k: int = 10, exclude: Set[str] = None) -> List[Tuple[str, float]]:
        """Find k nearest words to vector."""
        if exclude is None:
            exclude = set()
        
        results = []
        vec_norm = np.linalg.norm(vector)
        if vec_norm < 1e-8:
            return []
        
        for word, pos in self.word_positions.items():
            if word in exclude:
                continue
            sim = np.dot(vector, pos) / (vec_norm * np.linalg.norm(pos))
            results.append((word, sim))
        
        return sorted(results, key=lambda x: -x[1])[:k]
```

### 4.2 Style

```python
@dataclass
class Style:
    """
    A style extracted from exemplars.
    
    The centroid IS the style - it captures the average
    semantic position of all exemplars.
    """
    name: str
    centroid: np.ndarray
    exemplar_count: int
    characteristic_words: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def similarity(self, text_vec: np.ndarray) -> float:
        """Compute similarity of text to this style."""
        return cosine_similarity(text_vec, self.centroid)
    
    def to_json(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            'name': self.name,
            'centroid': self.centroid.tolist(),
            'exemplar_count': self.exemplar_count,
            'characteristic_words': self.characteristic_words,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_json(cls, data: Dict) -> 'Style':
        """Deserialize from JSON dict."""
        return cls(
            name=data['name'],
            centroid=np.array(data['centroid']),
            exemplar_count=data['exemplar_count'],
            characteristic_words=data.get('characteristic_words', []),
            metadata=data.get('metadata', {})
        )
```

### 4.3 Knowledge Base

```python
@dataclass
class Fact:
    id: str
    content: str
    encoding: np.ndarray
    source: str
    metadata: Dict = field(default_factory=dict)

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
    question_type: str
    question_encoding: np.ndarray
    answer_encoding: np.ndarray
    source: str

class KnowledgeBase:
    """
    Stores facts, triples, and Q&A pairs with their encodings.
    """
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.facts: Dict[str, Fact] = {}
        self.triples: Dict[str, Triple] = {}
        self.qa_pairs: Dict[str, QAPair] = {}
    
    def add_fact(self, content: str, source: str) -> Fact:
        """Add a fact."""
        # Implementation as shown in 3.4.1
        pass
    
    def add_triple(self, triple: Triple, source: str) -> Triple:
        """Add a triple."""
        pass
    
    def add_qa_pair(self, question: str, answer: str, qtype: str, source: str) -> QAPair:
        """Add a Q&A pair."""
        pass
    
    def search_facts(self, query: str, k: int = 5) -> List[Tuple[Fact, float]]:
        """Search facts by semantic similarity."""
        pass
    
    def search_qa(self, question: str, k: int = 5) -> List[Tuple[QAPair, float]]:
        """Search Q&A pairs by question similarity."""
        pass
```

### 4.4 Response

```python
@dataclass
class Response:
    """
    A response from the chat system.
    """
    answer: str
    confidence: float  # 0.0 to 1.0
    sources: List[str]
    style: Optional[str] = None
    debug_info: Dict = field(default_factory=dict)
```

---

## 5. Ingestion Pipeline

### 5.1 Text Ingestion

```python
def ingest_text(text: str, source: str, kb: KnowledgeBase) -> IngestionResult:
    """
    Ingest raw text into knowledge base.
    
    Pipeline:
    1. Split into sentences
    2. For each sentence:
       a. Add as fact
       b. Extract triples
       c. Generate Q&A pairs
    3. Extract style from full text
    """
    sentences = split_sentences(text)
    
    facts_added = 0
    triples_added = 0
    qa_added = 0
    
    for sentence in sentences:
        # Add as fact
        kb.add_fact(sentence, source)
        facts_added += 1
        
        # Extract triples
        triples = extract_triples(sentence)
        for triple in triples:
            kb.add_triple(triple, source)
            triples_added += 1
            
            # Generate Q&A pairs from triple
            qa_pairs = generate_qa_pairs(triple)
            for q, a, qtype in qa_pairs:
                kb.add_qa_pair(q, a, qtype, source)
                qa_added += 1
    
    return IngestionResult(
        facts=facts_added,
        triples=triples_added,
        qa_pairs=qa_added,
        source=source
    )
```

### 5.2 Project Gutenberg Ingestion

```python
GUTENBERG_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"

def ingest_gutenberg(book_id: int, kb: KnowledgeBase) -> IngestionResult:
    """
    Ingest a book from Project Gutenberg.
    
    Algorithm:
    1. Download text from Gutenberg
    2. Strip header/footer boilerplate
    3. Ingest as text
    4. Extract book-specific style
    """
    # Download
    url = GUTENBERG_URL.format(id=book_id)
    response = requests.get(url)
    text = response.text
    
    # Strip Gutenberg boilerplate
    text = strip_gutenberg_header(text)
    text = strip_gutenberg_footer(text)
    
    # Ingest
    source = f"gutenberg:{book_id}"
    result = ingest_text(text, source, kb)
    
    # Extract style
    style = extract_style(split_sentences(text)[:100], kb.vocab)
    style.name = f"Gutenberg_{book_id}"
    style.metadata['book_id'] = book_id
    
    return result

def strip_gutenberg_header(text: str) -> str:
    """Remove Project Gutenberg header."""
    marker = "*** START OF"
    idx = text.find(marker)
    if idx != -1:
        # Find end of line
        end = text.find('\n', idx)
        return text[end+1:]
    return text

def strip_gutenberg_footer(text: str) -> str:
    """Remove Project Gutenberg footer."""
    marker = "*** END OF"
    idx = text.find(marker)
    if idx != -1:
        return text[:idx]
    return text
```

### 5.3 Q&A Pair Ingestion

```python
def ingest_qa_pairs(pairs: List[Tuple[str, str]], source: str, kb: KnowledgeBase) -> IngestionResult:
    """
    Ingest structured Q&A pairs.
    
    Each pair is (question, answer).
    Question type is auto-detected.
    """
    qa_added = 0
    
    for question, answer in pairs:
        qtype = detect_question_type(question)
        kb.add_qa_pair(question, answer, qtype, source)
        qa_added += 1
    
    return IngestionResult(
        facts=0,
        triples=0,
        qa_pairs=qa_added,
        source=source
    )
```

---

## 6. Query Processing Pipeline

### 6.1 Query Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Parse Query    │  → Tokenize, detect question type
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Encode Query   │  → Convert to vector in semantic space
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Search KB      │  → Find similar facts, triples, Q&A pairs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Rank Matches   │  → Score by similarity, filter by threshold
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Apply Style    │  → Transform response toward target style
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Format Output  │  → Add sources, confidence, metadata
└────────┬────────┘
         │
         ▼
    Response
```

### 6.2 Full Query Implementation

```python
def process_query(query: str, 
                  kb: KnowledgeBase, 
                  style: Optional[Style] = None,
                  debug: bool = False) -> Response:
    """
    Process a user query and generate response.
    """
    debug_info = {}
    
    # 1. Parse query
    qtype = detect_question_type(query)
    if debug:
        debug_info['question_type'] = qtype
    
    # 2. Encode query
    query_vec = encode_text(query, kb.vocab)
    if debug:
        debug_info['query_norm'] = float(np.linalg.norm(query_vec))
    
    # 3. Search knowledge base
    # Try Q&A pairs first (most structured)
    qa_matches = kb.search_qa(query, k=5)
    fact_matches = kb.search_facts(query, k=5)
    
    if debug:
        debug_info['qa_matches'] = [(qa.question, sim) for qa, sim in qa_matches[:3]]
        debug_info['fact_matches'] = [(f.content[:50], sim) for f, sim in fact_matches[:3]]
    
    # 4. Select best match
    best_answer = None
    best_confidence = 0.0
    best_source = None
    
    # Prefer Q&A matches (they're more precise)
    if qa_matches and qa_matches[0][1] > 0.5:
        best_qa, best_confidence = qa_matches[0]
        best_answer = best_qa.answer
        best_source = best_qa.source
    elif fact_matches and fact_matches[0][1] > 0.3:
        best_fact, best_confidence = fact_matches[0]
        best_answer = best_fact.content
        best_source = best_fact.source
    
    if best_answer is None:
        return Response(
            answer="I don't have enough information to answer that question.",
            confidence=0.0,
            sources=[],
            debug_info=debug_info if debug else {}
        )
    
    # 5. Apply style if specified
    if style is not None:
        styled_vec, style_words = transfer_style(best_answer, style, kb.vocab, strength=0.4)
        if debug:
            debug_info['style_words'] = style_words[:5]
    
    # 6. Format response
    return Response(
        answer=best_answer,
        confidence=best_confidence,
        sources=[best_source],
        style=style.name if style else None,
        debug_info=debug_info if debug else {}
    )
```

---

## 7. Chat Interface

### 7.1 Command Parser

```python
COMMANDS = {
    '/help': 'Show help message',
    '/quit': 'Exit the chat',
    '/style': 'Set or list styles. Usage: /style <name> or /style list',
    '/source': 'Set or list sources. Usage: /source <name> or /source list',
    '/ingest': 'Ingest content. Usage: /ingest <url_or_path>',
    '/analyze': 'Analyze text style. Usage: /analyze <text>',
    '/debug': 'Toggle debug mode. Usage: /debug on|off',
}

def parse_command(input_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse user input for commands.
    
    Returns: (command, args) or (None, None) if not a command
    """
    if not input_text.startswith('/'):
        return None, None
    
    parts = input_text.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ''
    
    return command, args
```

### 7.2 Chat Loop

```python
def chat_loop(gcs: GeometricChatSystem):
    """
    Main chat interaction loop.
    """
    print("Welcome to Geometric Chat System v1.0")
    print("Type /help for commands, /quit to exit")
    print()
    
    current_style = None
    debug_mode = False
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Check for command
        command, args = parse_command(user_input)
        
        if command == '/quit':
            print("Goodbye!")
            break
        
        elif command == '/help':
            print("\nAvailable commands:")
            for cmd, desc in COMMANDS.items():
                print(f"  {cmd}: {desc}")
            print()
            continue
        
        elif command == '/style':
            if args == 'list':
                print("\nAvailable styles:")
                for name in gcs.styles:
                    print(f"  - {name}")
                print()
            elif args:
                if args in gcs.styles:
                    current_style = gcs.styles[args]
                    print(f"Style set to: {args}")
                else:
                    print(f"Unknown style: {args}")
            else:
                print(f"Current style: {current_style.name if current_style else 'None'}")
            continue
        
        elif command == '/ingest':
            if args.startswith('http'):
                # URL ingestion
                if 'gutenberg.org' in args:
                    # Extract book ID
                    match = re.search(r'/ebooks/(\d+)', args)
                    if match:
                        book_id = int(match.group(1))
                        print(f"Ingesting from Project Gutenberg (ID: {book_id})...")
                        result = gcs.ingest_gutenberg(book_id)
                        print(f"Done. Added {result.facts} facts, {result.qa_pairs} Q&A pairs.")
                    else:
                        print("Could not parse Gutenberg URL")
            else:
                # File ingestion
                print(f"Ingesting from file: {args}...")
                result = gcs.ingest_file(args)
                print(f"Done. Added {result.facts} facts, {result.qa_pairs} Q&A pairs.")
            continue
        
        elif command == '/analyze':
            if args:
                analysis = gcs.analyze_style(args)
                print(f"\nStyle analysis:")
                for name, score in analysis[:3]:
                    print(f"  - {name}: {score:.3f}")
                print()
            continue
        
        elif command == '/debug':
            if args == 'on':
                debug_mode = True
                print("Debug mode enabled")
            elif args == 'off':
                debug_mode = False
                print("Debug mode disabled")
            continue
        
        # Regular query
        response = gcs.query(user_input, style=current_style, debug=debug_mode)
        
        print(f"\nGCS: {response.answer}")
        print(f"     [confidence: {response.confidence:.2f}, source: {response.sources[0] if response.sources else 'N/A'}]")
        
        if debug_mode and response.debug_info:
            print(f"     [debug: {response.debug_info}]")
        
        print()
```

---

## 8. Configuration

### 8.1 Default Configuration

```yaml
# config/default.yaml

# Vocabulary settings
vocabulary:
  dim: 64                    # Dimensionality of semantic space
  style_bias_strength: 0.15  # How much to bias words toward style

# Encoding settings
encoding:
  use_idf_weighting: true    # Weight rare words higher
  min_word_length: 2         # Ignore single-character words

# Search settings
search:
  default_k: 5               # Number of results to return
  confidence_threshold: 0.3  # Minimum confidence to return answer

# Style settings
style:
  default_strength: 0.5      # Default style transfer strength
  characteristic_words_k: 15 # Number of characteristic words to store

# Ingestion settings
ingestion:
  chunk_type: sentence       # How to split text (sentence, paragraph, line)
  extract_triples: true      # Whether to extract triples
  generate_qa: true          # Whether to generate Q&A pairs

# Persistence settings
persistence:
  knowledge_dir: data/knowledge
  styles_dir: data/styles
  vocab_file: data/vocab.pkl
```

---

## 9. Mathematical Summary

### 9.1 Core Formulas

| Operation | Formula | Description |
|-----------|---------|-------------|
| **Word Position** | `pos(w) = hash(w) → ℝ^dim` | Deterministic position from hash |
| **Text Encoding** | `enc(t) = Σᵢ wᵢ·pos(wordᵢ) / Σᵢ wᵢ` | IDF-weighted average |
| **IDF Weight** | `w = 1 / log(1 + count)` | Rare words weighted higher |
| **Cosine Similarity** | `sim(a,b) = (a·b) / (‖a‖·‖b‖)` | Semantic similarity |
| **Style Centroid** | `c = (1/n) Σᵢ enc(exemplarᵢ)` | Style = average position |
| **Style Transfer** | `styled = (1-α)·content + α·centroid` | Interpolate toward style |
| **Style Direction** | `dir = centroid_B - centroid_A` | What makes B different from A |
| **Style Bias** | `new = (1-α)·old + α·centroid` | Pull word toward style |

### 9.2 Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Tokenize | O(n) | O(n) |
| Encode text | O(w·d) | O(d) |
| Cosine similarity | O(d) | O(1) |
| Search k-nearest | O(n·d) | O(k) |
| Style extraction | O(e·w·d) | O(d) |
| Style transfer | O(d) | O(d) |

Where:
- n = text length
- w = number of words
- d = dimensionality (64)
- e = number of exemplars
- k = number of results

### 9.3 Geometric Interpretation

```
                    SEMANTIC SPACE (ℝ^64)
                    
        Style A                      Style B
        centroid                     centroid
           ●─────────────────────────────●
          /│\                           /│\
         / │ \                         / │ \
        /  │  \                       /  │  \
       ●   ●   ●                     ●   ●   ●
    exemplars of A                exemplars of B
    
    
    Query: "Who is Captain Ahab?"
           │
           ▼
           ●  ←── query position
          /|\
         / | \
        /  |  \
       ●   ●   ●  ←── similar Q&A pairs
       
    Best match = closest Q&A pair
    Answer = that pair's answer
    Confidence = cosine similarity
```

---

## 10. Validation Checklist

### 10.1 Unit Tests Required

- [ ] `test_tokenize`: Correct tokenization
- [ ] `test_word_position`: Deterministic, normalized
- [ ] `test_encode_text`: IDF weighting works
- [ ] `test_cosine_similarity`: Correct range [-1, 1]
- [ ] `test_style_extraction`: Centroid computed correctly
- [ ] `test_style_classification`: Correct style identified
- [ ] `test_style_transfer`: Interpolation works
- [ ] `test_question_type_detection`: All types detected
- [ ] `test_triple_extraction`: Triples extracted correctly
- [ ] `test_qa_generation`: Q&A pairs generated
- [ ] `test_search`: Relevant results returned
- [ ] `test_query_pipeline`: End-to-end query works

### 10.2 Integration Tests Required

- [ ] `test_gutenberg_ingestion`: Full book ingested
- [ ] `test_style_persistence`: Save/load works
- [ ] `test_chat_commands`: All commands work
- [ ] `test_multi_source`: Multiple sources queried

### 10.3 Acceptance Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Style classification accuracy | >80% | Test suite |
| Q&A relevance (top-1) | >70% | Human evaluation |
| Query response time | <500ms | Benchmark |
| Memory usage | <500MB | Profiling |

---

## 11. Appendices

### Appendix A: Prototype Code Locations

| Component | Prototype File | Status |
|-----------|----------------|--------|
| Style Centroid | `papers/style_centroid.py` | Validated (8/8) |
| Style Extractor | `papers/style_extractor.py` | Validated (6/6) |
| Q&A Projection | `papers/recursive_holographic_qa.py` | Working |
| General Q&A | `papers/holographic_qa_general.py` | Working |
| Semantic Chatbot | `experiments/semantic_chatbot.py` | Prototype |

### Appendix B: Design Documents

| Document | Content |
|----------|---------|
| `030_geometric_qa_projection.md` | Q&A as geometric projection |
| `031_unified_projection_framework.md` | Unified style/Q&A theory + validation |
| `019_holographic_resolution.md` | Holographic encoding principles |

### Appendix C: Golden Ratio Connection (Optional Enhancement)

The golden ratio φ ≈ 1.618 can be used for hierarchical encoding:

```python
def phi_encode(level: int) -> float:
    """Encode hierarchy level using golden ratio."""
    return PHI ** level
```

This creates self-similar spacing between levels, useful for:
- Hierarchical categories
- Nested concepts
- Multi-scale representations

### Appendix D: Future Enhancements

1. **Approximate Nearest Neighbor**: Use FAISS or Annoy for O(log n) search
2. **Incremental Learning**: Update centroids as new exemplars arrive
3. **Multi-modal**: Extend to images using same geometric framework
4. **Distributed**: Shard knowledge base across machines

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-21 | TruthSpace Team | Initial SDS |
