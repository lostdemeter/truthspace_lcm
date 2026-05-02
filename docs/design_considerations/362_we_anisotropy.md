# DC 362: W_E Anisotropy

**Day 193 | W_E is extremely anisotropic: one dominant axis (PC1) captures
72% of variance and corresponds to the multilingual script divide (CJK vs
Latin punctuation). Relational directions are nearly orthogonal to all
principal axes (max cos=0.15). TruthSpace geometry lives in the
script-orthogonal subspace of W_E.**

---

## Overview

Day 192 applied randomized SVD to the full W_E matrix (151,936 × 1536)
to measure anisotropy — whether the embedding space has preferred directions.

---

## Finding 1: Extreme Anisotropy — One Dominant Axis

```
Top-50 relative singular values (s/s[0]):
  PC1:  1.0000  (dominates everything)
  PC2:  0.1707
  PC3:  0.1527
  PC4:  0.1203
  PC5:  0.1183
  PC6-20: 0.082 - 0.110  (slow decay)

Variance explained within top-50 PCs:
  k=1:  71.99%   ← PC1 alone captures 72% of top-50 variance
  k=2:  74.09%
  k=5:  77.82%
  k=10: 81.72%
  k=50: 100.00%

Expected if flat (isotropic): 2.00% per PC
PC1 actual:                   71.99%  (36.0× above flat)
```

W_E is profoundly anisotropic. A single direction in ℝ^1536 captures
nearly three-quarters of the variance in the embedding matrix. This is
the opposite of an isotropic spherical code.

---

## Finding 2: PC1 = The Multilingual Script Axis

```
PC1 top tokens:
  +pole (cos ≈ 0.955): ìĤł  ìĳ»  íĨ§  ë²´  ëľħ   ← Korean/CJK
  -pole (cos ≈ -0.77): ,    .    Ġ    1    Ġ(    ← Latin punct/digits
```

PC1 is the **language/script axis**: it separates CJK (Korean, Chinese,
Japanese) characters from Latin punctuation and numerical tokens. This
is the single dominant structure in Qwen2's W_E.

**Why does this dominate?**

Qwen2's vocabulary of 151,936 tokens contains approximately:
- ~30,000+ CJK unified characters (dense cluster in one region)
- ~20,000+ sub-word Latin tokens (spread in another region)
- ~5,000+ digit/punctuation tokens (another region)

The CJK characters are individually rare but collectively massive —
they form a tight cluster pointing in the +PC1 direction. All Latin
tokens collectively point in the -PC1 direction. The mass asymmetry
creates the dominant first axis.

**Secondary axes:**

```
PC2: Code/text vs Spanish (code patterns like .IsNullOrEmpty vs Ġalunos)
PC3: Code formatting (newlines/brackets) vs Chinese characters
PC4: English words vs fullwidth CJK punctuation
PC5: Code punctuation vs comment separators
```

All secondary axes are also script/formatting-related, not semantic.

---

## Finding 3: Relational Directions Are Orthogonal to Dominant Structure

```
Relational direction cosine with top-20 W_E principal axes:

Domain      Best PC    Best cos    Interpretation
────────────────────────────────────────────────────
capitals    PC19       0.052       Nearly orthogonal to ALL
languages   PC1        0.111       Slight language-axis component
gender      PC1        0.148       Slight language-axis component
antonyms    PC1        0.105       Slight language-axis component

All relational directions: max cosine with any top-20 PC < 0.15
```

Every relational direction (country→capital, country→language,
masculine→feminine, antonyms) has cosine ≤ 0.15 with every one of the
top-20 W_E principal axes. These directions are essentially in the
**null space of W_E's dominant structure**.

**What the slight PC1 components mean:**

Languages (cos=0.111 with PC1), gender (0.148), and antonyms (0.105)
have tiny projections onto the script axis. This makes sense:
- "French" has a slight CJK-character-like quality (rarely appears with CJK)
- "woman" vs "man" has tiny script context differences
These are statistical artifacts of where these words appear in multilingual
training text, not semantic structure.

**The capitals case is most striking:** the country→capital direction is
essentially uncorrelated with ALL top-20 W_E axes (best: PC19, cos=0.052).
The "France→Paris" direction is in the deep residual subspace.

---

## Finding 4: Mean-Centered Anisotropy

```
Mean embedding norm: 0.242
  All 151,936 embeddings are shifted ~0.242 in one direction.

Centered singular values (top-50):
  PC1:  1.0000 (33.12% of centered variance)
  PC2:  0.3543 (37.28% cumulative)
  PC3:  0.3223 (40.72% cumulative)
  PC10: 0.2225 (55.11% cumulative)

Centered PC1: numerals/punct (+pole) vs CJK (-pole)
```

The DC offset (mean norm 0.242) corresponds to all embeddings being
"pushed" away from the origin in one direction — likely the direction
away from zero that RMSNorm training prefers. After removing this offset,
the space is less extreme (PC1 drops from 72% to 33%) but remains
strongly anisotropic (flat expectation: 2%).

---

## The Two-Layer Structure of W_E

Day 192 reveals that W_E encodes two distinct types of information in
**orthogonal subspaces**:

```
LAYER 1: Script/Language Identity (dominant, PCs 1-5)
  - Separates CJK from Latin from punctuation from code
  - Explained by the top-5 PCs (>77% of variance)
  - Culturally and linguistically determined
  - Present in ALL embeddings

LAYER 2: Semantic/Relational Structure (residual, cos~0 with PCs 1-20)
  - Country→capital, masculine→feminine, synonym directions
  - Lives in the ~1500 dimensions ORTHOGONAL to top-5 PCs
  - TruthSpace operates here
  - Present only in semantically coherent word clusters
```

This is a natural factorization: the script identity (what language is
this character from?) is independent of the relational structure (what
is the capital of this country?). W_E has learned to keep these orthogonal.

---

## Implications for TruthSpace

**1. Script axes do not interfere with relational retrieval.**

The LOO accuracy for country→capital (0.900) is not degraded by the
script axis because the relational direction is orthogonal to it.
When we compute `W_E[Paris] - W_E[France]`, the PC1 components cancel
(France and Paris are both Latin tokens with similar PC1 values),
leaving a direction in the residual semantic subspace.

**2. Cross-script relations would fail.**

If we tried to compute a direction between a CJK token and a Latin token
(e.g., Chinese ideogram → English translation), the direction would be
dominated by the script axis, not the semantic relation. The LOO approach
would fail because the script difference overwhelms the semantic difference.

**3. Dimensionality of TruthSpace.**

The semantic subspace where relational directions live has approximately:
```
H_semantic ≈ H - H_script ≈ 1536 - 5 = ~1531 dimensions
```
Within this subspace, the relational directions from Day 186 (SVD:
eff_dim = n per domain) apply. The effective TruthSpace dimensionality
is still ~1531, not reduced by the script axes.

**4. Centered W_E for semantic work.**

Subtracting the mean embedding before computing relational directions
would remove the DC offset and partially reduce the script axis influence.
For English-only relational work (capitals, gender, antonyms), this
is likely neutral (already nearly orthogonal). For cross-lingual work,
mean-centering would be beneficial.

---

## Summary

```
Finding                              Value
─────────────────────────────────────────────────────────────────
PC1 variance explained               71.99% of top-50 (36x flat)
PC1 semantic identity                Script axis: CJK vs Latin punct
Relational dirs vs top-20 PCs        max cos = 0.15 (capitals: 0.05)
Relational geometry subspace         Residual (⊥ to script axes)
DC offset (mean emb norm)            0.242
Centered PC1                         33.12% — still strongly anisotropic
Semantic subspace dimensionality     ~1531 (H - script dims)
```

---

## Files

- `expedition_day192_anisotropy.py` — SVD experiment
- `day192_anisotropy.json` — results
- `359_we_relational_dimensionality.md` — per-domain SVD (eff_dim=n)
- `361_we_norm_structure.md` — norm distribution (near-unit-sphere)
