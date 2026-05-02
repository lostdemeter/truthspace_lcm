# Doc 280: The Superposition of Shapes

**Date:** March 3, 2026
**Status:** Theoretical Framework — Derived from F149–F150
**Prerequisites:** DC 277 (Geometric Instrument), DC 278 (Geometric Decomposition), DC 279 (Sign-Space Navigation)

---

## 1. The Discovery

We set out to understand what the COMB zone's weight shapes *do*.
What we found changes how we think about neural network weights entirely.

### 1.1 The Empirical Path

| Finding | What We Learned |
|:--------|:----------------|
| F149 | Weight signs carry the computation. Exponents are universal scale. |
| F150 | For a single structure class, the MLP collapses to a **rank-1 projector**. |

The full-rank weight matrix W_gate ∈ ℝ^(18944×3584) contains ~68 million
parameters. For the "capital of X" structure class, it reduces to a single
direction v₁ ∈ ℝ^3584 and a filter response f = W_gate · v₁ ∈ ℝ^18944.
That's 22,528 numbers — a **2960× compression**.

Both W_gate and W_up can be simultaneously reduced to rank-1, and the
model still produces correct answers.

### 1.2 The Implication

If the full-rank weight matrix is rank-1 *for each structure class
individually*, then the full matrix must be a **superposition** of
many rank-1 components:

```
W_gate = Σ_c  f_c ⊗ v_c^T
```

Where:
- c indexes structure classes (capital-of, color-of, located-in, ...)
- v_c ∈ ℝ^3584 is the **input direction** for structure class c
- f_c ∈ ℝ^18944 is the **filter response** — which neurons fire for class c
- ⊗ denotes the outer product

This is not an approximation. It is a restatement of linear algebra:
any matrix can be decomposed as a sum of rank-1 matrices (SVD).
What makes this *non-trivial* is the claim that the SVD components
correspond to **semantically meaningful structure classes**.

---

## 2. The Weight Matrix as a Dictionary of Shapes

### 2.1 What "Shape" Means

In the IPA converter, a shape is a RECT pair: a binary gate that
activates at a specific codepoint and adds a specific height.

```
IPA:  IF codepoint == X  THEN  output += H
```

In the transformer MLP, a shape is a rank-1 projector: a direction
that the gate responds to and a filter that determines which neurons fire.

```
MLP:  IF input · v_c > threshold  THEN  activate neurons per f_c
```

The weight matrix W_gate is a **dictionary of shapes** — one entry per
structure class. When an input arrives, it projects onto each v_c,
and the structure class with the highest projection "wins" the gate.

### 2.2 The IPA Converter Analogy

The IPA converter processes a sequence of codepoints. At each position:
1. Test each rule's activation condition (is the codepoint in range?)
2. Apply the matching rule's transformation (add height H)
3. Rules compose additively (residual accumulation)

The transformer MLP processes a hidden state. At each layer:
1. Test each structure class's activation condition (project onto v_c)
2. Apply the matching class's transformation (activate filter f_c, route through W_up, project through W_down)
3. Layers compose additively (residual stream)

The parallel is exact:

| IPA Converter | Transformer MLP |
|:--------------|:----------------|
| Codepoint x | Hidden state h |
| Rule condition: x ∈ [a, b] | Projection: h · v_c > τ |
| Rule output: height H | Filter response: f_c |
| Additive: y = x + Σ H_i | Residual: h' = h + Σ δ_c |
| Dictionary of RECT pairs | Dictionary of rank-1 projectors |

### 2.3 Why Full-Rank?

The weight matrix is full-rank because it encodes **all structure classes
simultaneously**. Each class contributes a rank-1 component. If there are
K structure classes and K ≤ min(18944, 3584) = 3584, the superposition
of K rank-1 matrices can fill the full rank.

This explains a puzzle from F149: the sign matrices are full-rank,
perfectly balanced (50% positive), and unique per layer. They MUST be,
because they encode the superposition of many different structure classes,
each with its own sign pattern.

---

## 3. The Geometry of Superposition

### 3.1 Near-Orthogonality

For the dictionary to work, the v_c directions must be **nearly orthogonal**
so that different structure classes don't interfere with each other.
In ℝ^3584, random unit vectors have expected |cos| ≈ 1/√3584 ≈ 0.017.
This means the space can support **thousands** of nearly orthogonal
structure classes.

We already know the "capital of X" structure class has rank-1 energy
of 93.5–96.3%. The ~4–7% residual energy is the projection of
other structure classes' v_c directions onto the capital-class inputs.

### 3.2 The Gate as a Classifier

The gate computation g = SiLU(W_gate · x) is, in superposition terms:

```
g_i = SiLU( Σ_c  f_c[i] · (v_c · x) )
```

For each neuron i, the gate value is the sum of contributions from
all structure classes, weighted by how much the input projects onto
each class's direction. When the input is a capital-of prompt,
the capital-class term dominates (because |v_capital · x| >> |v_other · x|),
and the gate pattern approximates f_capital.

This is why 98% of neurons fire for all capital prompts (F150 Inv 1):
the filter response f_capital has almost all positive entries.
The 2–7% of neurons that differ between entities reflect the
interference terms from other structure classes.

### 3.3 The Output Subspace Rotation

F150 showed that the MLP output δ is orthogonal to v₁ (cos ≈ 0.01).
In superposition terms, this makes sense: the MLP reads from the
v_capital direction and writes to a **different** direction, creating
a rotation in the hidden state space.

Each rank-1 projector maps:
```
v_c  →  (via gate × up × down)  →  u_c
```

Where u_c ⊥ v_c (approximately). This means the MLP is a **rotation**:
it takes information from the input manifold and projects it into a
new subspace where the extraction layers (L22–L27) can read it.

The extraction layers then perform their own geometric operations
(φ-lenses, head routing) on the u_c subspace to produce the final answer.

---

## 4. Implications

### 4.1 For Understanding

If the superposition hypothesis is correct, then:

1. **Every weight matrix is a dictionary.** Not just W_gate, but W_up, W_down,
   W_q, W_k, W_v, W_o — all are superpositions of rank-1 components,
   each corresponding to a semantic structure class.

2. **Training discovers the dictionary.** The training process finds the
   optimal set of {v_c, f_c} pairs that allow the model to handle
   all structure classes it encounters. This is not designed; it emerges.

3. **The "irreducible neural" layers are not irreducible.** The COMB zone
   appeared to be 6–7 layers of opaque neural computation. In reality,
   for any single structure class, they collapse to a sequence of
   6 rank-1 projectors — a total of ~113K parameters per class.

4. **The holistic barrier explained.** F148 showed that cross-entity
   navigation fails even with oracle deltas. This is because the
   entity identity is encoded in σ₁ — the scalar projection onto v_c —
   and this scalar is deeply entangled with the residual stream.
   You can't change σ₁ without changing the entire residual.

### 4.2 For Engineering

If we can identify the v_c directions for different structure classes:

1. **Read the dictionary:** Extract the filter responses for known classes.
   This gives us a complete map of what the model "knows" in each class.

2. **Write new entries:** Design a new v_c direction and filter response f_c.
   Add this rank-1 component to W_gate (and corresponding components to
   W_up, W_down). This would add a new "fact type" to the model without
   retraining.

3. **Edit existing entries:** Modify an existing f_c to change how the model
   handles a particular structure class. This is targeted knowledge editing
   without gradient descent.

4. **Compose dictionaries:** Two models' weight matrices are two dictionaries.
   If their v_c directions are sufficiently orthogonal, the dictionaries
   can be **superimposed** by matrix addition. This is model merging
   via geometric composition.

### 4.3 For the Hypothesis

"Structure IS information" — the weight matrix structure IS a dictionary
of geometric shapes. Each shape is a rank-1 projector that implements
one rule of the model's knowledge. The weight matrix is not opaque;
it is a transparently readable catalog of structure-class-specific
transformations.

"ENCODE = DECODE" — reading the dictionary (projecting onto v_c to get
f_c) and writing the dictionary (adding f_c ⊗ v_c^T to the matrix)
are the same operation in opposite directions.

---

## 5. Predictions

The superposition hypothesis makes specific, testable predictions:

### P1: Multi-class rank-1 test
Different structure classes ("capital of", "color of", "continent of")
should each have their own v_c direction with rank-1 energy > 90%.
The v_c directions should be nearly orthogonal (|cos| < 0.1).

### P2: Filter response uniqueness
The filter response f_c should be unique per structure class.
Two different classes at the same layer should have |cos(f_c1, f_c2)| < 0.5.

### P3: Cross-class interference
Inputs from class c1 should have small projection onto v_c2 (|v_c2 · x_c1| << |v_c1 · x_c1|).
This small projection is the "interference" that creates the 2–7% of
non-universal gate activations.

### P4: Dictionary editing
Adding a rank-1 component f_new ⊗ v_new^T to W_gate (with corresponding
edits to W_up, W_down) should allow the model to handle a new structure
class without disrupting existing classes.

### P5: SVD ↔ structure classes
The SVD of W_gate should have singular vectors that correspond to
identifiable structure classes. The singular values should reflect
the frequency or importance of each class in the training data.

---

## 6. The Shape of Knowledge

What does a neural network "know"?

The standard answer: it knows what its weights encode, but the encoding
is opaque — billions of floating-point numbers with no obvious structure.

Our answer: **the weights are a dictionary of geometric shapes.**
Each shape is a rank-1 projector that implements one rule.
The dictionary is readable, writable, and composable.

A weight matrix is not a black box. It is a library.

Each rank-1 component is a book on one shelf, containing one rule
about the world. The gate determines which book to open.
The up-projection reads the content. The down-projection writes
the result back into the stream of thought.

The transformer doesn't "think" in the way we imagine.
It opens the right book, reads the relevant page, and writes
the answer into the margin. The shapes of the books ARE the knowledge.

The language of light is the language of shapes.
And shapes can be translated.

---

## 7. Next Steps

1. **Test P1–P3:** Run the multi-class rank-1 test with diverse structure
   classes. This is Frontier 8.

2. **Extract the dictionary:** For a set of known structure classes, compute
   all {v_c, f_c} pairs. Visualize the dictionary.

3. **Test P4:** Attempt dictionary editing — add a new fact to the model
   by engineering a rank-1 component.

4. **Test P5:** Compare SVD components of W_gate with empirically discovered
   structure-class directions.

---

*"What an LLM 'knows' is encoded in its geometric structure."*
*Now we can read that structure. And write it.*
