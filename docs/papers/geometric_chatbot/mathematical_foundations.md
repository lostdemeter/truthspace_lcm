# Mathematical Foundations of Geometric Language Understanding

This document provides detailed mathematical derivations for each component of the geometric chatbot system.

---

## 1. Position Encoding

### 1.1 Normalized Position

For a sentence with $n$ tokens, the normalized position of token $i$ is:

$$p_i = \frac{i}{n - 1}$$

This maps all positions to the interval $[0, 1]$, making positions comparable across sentences of different lengths.

### 1.2 Mean Position

For a word $w$ appearing at positions $\{p_1, p_2, ..., p_k\}$ across $k$ sentences:

$$\bar{p}(w) = \frac{1}{k} \sum_{j=1}^{k} p_j$$

**Interpretation:** Words with $\bar{p} \approx 0$ tend to appear at sentence beginnings (subjects). Words with $\bar{p} \approx 0.5$ tend to appear in the middle (verbs). Words with $\bar{p} \approx 1$ tend to appear at the end (objects).

### 1.3 Position Variance

$$\sigma^2(w) = \frac{1}{k} \sum_{j=1}^{k} (p_j - \bar{p})^2$$

**Interpretation:** 
- Low variance ($\sigma^2 \approx 0$): Word appears in consistent positions → likely content word
- High variance ($\sigma^2 > 0.1$): Word appears everywhere → likely stop word

### 1.4 φ-Weighted Encoding

For richer encoding, we can use φ-weighted positions:

$$\phi\text{-pos}(w) = \sum_{j=1}^{k} \phi^{-j} \cdot p_j$$

where $\phi = 1.618034...$ is the golden ratio. This gives more weight to earlier occurrences, creating a self-similar encoding.

---

## 2. Position Bands for Frame Extraction

### 2.1 The SVO Hypothesis

In Subject-Verb-Object (SVO) languages, the typical sentence structure maps to positions:

| Role | Typical Position | Position Band |
|------|------------------|---------------|
| Subject (Initiator) | 0.0 - 0.2 | [0.0, 0.33) |
| Verb (Mediator) | 0.3 - 0.5 | [0.33, 0.66) |
| Object (Receiver) | 0.6 - 1.0 | [0.66, 1.0] |

### 2.2 Mathematical Formulation

Given a sentence with content words at positions $\{(w_1, p_1), (w_2, p_2), ...\}$, we assign:

$$\text{role}(w_i) = \begin{cases}
\text{Initiator} & \text{if } p_i < \frac{1}{3} \text{ and no initiator assigned} \\
\text{Mediator} & \text{if } \frac{1}{3} \leq p_i < \frac{2}{3} \text{ and no mediator assigned} \\
\text{Receiver} & \text{if } p_i \geq \frac{2}{3} \text{ and no receiver assigned}
\end{cases}$$

### 2.3 Why Thirds?

The division into thirds is a first-order approximation. More sophisticated approaches could use:

**φ-based divisions:**
$$[0, \phi^{-2}), [\phi^{-2}, \phi^{-1}), [\phi^{-1}, 1]$$

where $\phi^{-1} \approx 0.618$ and $\phi^{-2} \approx 0.382$.

This creates self-similar bands that may better capture natural language structure.

---

## 3. Semantic Role Vectors

### 3.1 Role Vector Definition

For each concept $C$, we maintain a role vector:

$$\vec{r}(C) = (r_i, r_m, r_r)$$

where:
- $r_i$ = number of times $C$ appears as Initiator
- $r_m$ = number of times $C$ appears as Mediator
- $r_r$ = number of times $C$ appears as Receiver

### 3.2 φ-Direction

The φ-direction measures the "polarity" of a concept:

$$\phi\text{-dir}(C) = \frac{r_i - r_r}{r_i + r_m + r_r + \epsilon}$$

where $\epsilon$ is a small constant to avoid division by zero.

**Properties:**
- $\phi\text{-dir} \in [-1, 1]$
- $\phi\text{-dir} > 0$: Primarily initiator (agent-like)
- $\phi\text{-dir} < 0$: Primarily receiver (patient-like)
- $\phi\text{-dir} \approx 0$: Balanced or mediator

### 3.3 φ-Magnitude

$$|\phi|(C) = \frac{|r_i - r_r|}{r_i + r_m + r_r + \epsilon}$$

This measures the strength of the polarity.

---

## 4. Geometric Stop Word Detection

### 4.1 The Stop Word Criterion

A word $w$ is classified as a geometric stop word if:

$$\text{is\_stop}(w) = S_1(w) \lor S_2(w) \lor S_3(w)$$

where:

**S₁: No Semantic Role**
$$S_1(w) = (r_i + r_m + r_r = 0)$$

**S₂: Short and Frequent**
$$S_2(w) = (\text{len}(w) \leq 4) \land (f(w) \geq 3)$$

**S₃: Only Receiver and Short**
$$S_3(w) = (r_r > 0) \land (r_i = 0) \land (r_m = 0) \land (\text{len}(w) \leq 5)$$

### 4.2 Justification

**S₁:** Words with no semantic role are structural glue, not content carriers.

**S₂:** Short, frequent words follow Zipf's law for function words. The thresholds (4 chars, 3 occurrences) are empirically determined.

**S₃:** Prepositions often accidentally get receiver roles due to position. This criterion catches them.

### 4.3 Connection to Zipf's Law

Zipf's law states that word frequency is inversely proportional to rank:

$$f(r) \propto \frac{1}{r^\alpha}$$

where $\alpha \approx 1$. Stop words occupy the highest-frequency positions, which our geometric detection captures through the frequency threshold.

---

## 5. Morphological Equivalence

### 5.1 Parallel Structure Principle

**Theorem:** If sentences $s_1$ and $s_2$ have:
1. The same initiator
2. The same semantic structure
3. Different mediators $m_1$ and $m_2$

Then $m_1$ and $m_2$ are morphological variants of the same concept.

**Proof (informal):** The parallel structure constrains the semantic content to be identical. The only difference is the surface form of the verb, which must therefore represent the same underlying concept.

### 5.2 Equivalence Classes

We define an equivalence relation $\equiv$ on words:

$$w_1 \equiv w_2 \iff \exists \text{ parallel structure revealing equivalence}$$

This relation is:
- **Reflexive:** $w \equiv w$ (trivially)
- **Symmetric:** $w_1 \equiv w_2 \implies w_2 \equiv w_1$ (by definition)
- **Transitive:** $w_1 \equiv w_2 \land w_2 \equiv w_3 \implies w_1 \equiv w_3$ (by chaining)

The equivalence classes partition the vocabulary into morphological families.

### 5.3 Phase Encoding

Within an equivalence class, we assign phases based on position in the bootstrap:

$$\text{phase}(w) = \text{position of } w \text{ in parallel group}$$

| Phase | Meaning | Example |
|-------|---------|---------|
| 0 | Base form | love |
| 1 | 3rd person singular | loves |
| 2 | Past tense | loved |

### 5.4 Conjugation Function

Given a word $w$ and target phase $p$, conjugation is:

$$\text{conjugate}(w, p) = [w]_p$$

where $[w]$ is the equivalence class of $w$ and $[w]_p$ is the member at phase $p$.

---

## 6. Query Processing

### 6.1 Query Encoding

A query $Q$ is encoded to a position in concept space:

$$\text{encode}(Q) = \frac{\sum_{w \in Q} \text{weight}(w) \cdot \bar{p}(w)}{\sum_{w \in Q} \text{weight}(w)}$$

where the weight is Zipf-based:

$$\text{weight}(w) = \frac{1}{\log(f(w) + 2)}$$

Rarer words have higher weight, focusing on content.

### 6.2 Question Type Detection

Question type is detected geometrically:

| Pattern | Detection | Response Type |
|---------|-----------|---------------|
| "Who [action]?" | Action word with $r_m > 0$ | Find initiator |
| "What does X do?" | Entity X with $\phi\text{-dir} > 0$ | List actions |
| Entity mention | Entity with $r_i > 0$ | Describe entity |

### 6.3 Response Generation

For "Who [action]?" queries:

1. Find all concepts $C$ where $\exists a \in \text{actions}(C) : a \equiv \text{action}$
2. Select $C^* = \arg\max_C \text{count}(C, \text{action})$
3. Find target from frames: $t = \text{receiver}(C^*, \text{action})$
4. Conjugate: $v = \text{conjugate}(\text{action}, 1)$
5. Return: "$C^*$ $v$ $t$."

---

## 7. The φ Connection

### 7.1 Why φ?

The golden ratio appears throughout:

1. **Self-similarity:** $\phi = 1 + \frac{1}{\phi}$, creating fractal structure
2. **Optimal packing:** φ-based divisions minimize overlap
3. **Natural occurrence:** φ appears in language statistics (Zipf exponents)

### 7.2 φ and Zipf

Zipf's law can be written as:

$$f(r) = \frac{C}{r^\alpha}$$

where $\alpha \approx 1$. Interestingly, $\phi^{-1} + \phi^{-2} = 1$, suggesting a connection between φ-encoding and frequency distributions.

### 7.3 The Encode-Decode Duality

A key insight: **encoding and decoding are the same operation in opposite directions**.

$$\text{encode}: \text{words} \to \phi\text{-space}$$
$$\text{decode}: \phi\text{-space} \to \text{words}$$

Like $\phi$ and $\frac{1}{\phi} = \phi - 1$, they are inverses that share the same structure.

---

## 8. Complexity Analysis

### 8.1 Learning Complexity

- **Frame extraction:** $O(n \cdot m)$ where $n$ = sentences, $m$ = avg tokens
- **Role counting:** $O(n \cdot m)$
- **Stop word detection:** $O(v)$ where $v$ = vocabulary size

**Total:** $O(n \cdot m + v)$

### 8.2 Query Complexity

- **Tokenization:** $O(q)$ where $q$ = query length
- **Content word finding:** $O(q \cdot v)$ (can be optimized with hash lookup)
- **Response generation:** $O(v + f)$ where $f$ = number of frames

**Total:** $O(q \cdot v + f)$

### 8.3 Space Complexity

- **Concepts:** $O(v)$
- **Frames:** $O(n)$
- **Morphology:** $O(v)$

**Total:** $O(v + n)$

---

## 9. Theoretical Guarantees

### 9.1 Completeness

**Theorem:** If a word appears in a consistent semantic role across the corpus, it will be correctly classified as a content word.

**Proof:** Let $w$ be a word appearing $k$ times, always in role $R$. Then $r_R(w) = k > 0$, so $S_1(w) = \text{false}$. If $\text{len}(w) > 4$ or $k < 3$, then $S_2(w) = \text{false}$. If $R \neq \text{Receiver}$ or $\text{len}(w) > 5$, then $S_3(w) = \text{false}$. Thus $\text{is\_stop}(w) = \text{false}$.

### 9.2 Soundness

**Theorem:** If a word is classified as a content word, it has a meaningful semantic role.

**Proof:** By contrapositive. If $w$ has no semantic role, then $r_i = r_m = r_r = 0$, so $S_1(w) = \text{true}$, and $\text{is\_stop}(w) = \text{true}$.

### 9.3 Morphological Correctness

**Theorem:** If the bootstrap contains correct parallel structures, morphological equivalence is correctly learned.

**Proof:** By construction. Each group of 3 sentences in the bootstrap has the same semantic content with different verb forms. The equivalence class captures exactly these forms.

---

## 10. Extensions

### 10.1 Multi-Language Support

For SOV languages (Japanese, Korean, Turkish):

| Role | Position Band |
|------|---------------|
| Subject | [0.0, 0.33) |
| Object | [0.33, 0.66) |
| Verb | [0.66, 1.0] |

The position bands are simply reordered.

### 10.2 Rich Morphology

Languages with rich morphology (Finnish, Turkish) require larger bootstraps. The number of bootstrap sentences scales as:

$$|B| = O(v \cdot m)$$

where $v$ = number of verbs and $m$ = number of morphological forms per verb.

### 10.3 Continuous Representations

Instead of discrete equivalence classes, we could use continuous embeddings:

$$\vec{e}(w) = (\bar{p}(w), \sigma^2(w), \phi\text{-dir}(w), |\phi|(w), ...)$$

This would allow soft matching and interpolation between concepts.

---

## References

1. Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*.
2. Livio, M. (2002). *The Golden Ratio: The Story of Phi*.
3. Fillmore, C. J. (1976). Frame semantics and the nature of language.
4. Harris, Z. S. (1954). Distributional structure. *Word*, 10(2-3), 146-162.
