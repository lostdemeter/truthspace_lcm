# Doc 240: The Semantic Spectrometer — What the Mixer Actually Is

**Date:** February 8, 2026  
**Status:** Discovery  
**Prerequisites:** Doc 239 (The Two Frontiers)

## The Question

The pointwise convolutions (expand → GELU → compress) are the only non-φ-structured component in the encoder. They are full-rank, learned, and account for 99.4% of the encoder's parameters. What ARE they doing?

## What We Expected

A "channel mixer" — blending geometric features together, creating weighted combinations. Something like an audio mixing board, where each channel gets a volume slider.

## What We Found

Something radically different.

---

## The Three Revelations

### Revelation 1: Extreme Sparsity — This Is Not a Mixer

| Block | Expanded Channels | Active Per Pixel | Percentage |
|-------|------------------|-----------------|-----------|
| S0.B0 | 384 | 71 | **18.9%** |
| S1.B0 | 768 | 170 | 22.1% |
| S2.B0 | 1,536 | 149 | 9.7% |
| S2.B8 | 1,536 | 86 | **5.6%** |
| S3.B0 | 3,072 | 80 | **3.2%** |
| S3.B1 | 3,072 | 21 | **0.7%** |
| S3.B2 | 3,072 | 21 | **0.7%** |

At Stage 3, **97% of the expanded neurons are suppressed.** Only ~80 out of 3,072 fire at any given pixel. By Blocks 1-2, this drops to 21 — less than 1%.

This is not mixing. This is **identification**.

A mixer would have most channels active with varying weights. A 3% activation rate means the system is performing a needle-in-a-haystack search: "of 3,072 possible semantic features, which ~80 are present at THIS pixel?"

### Revelation 2: Each Pixel Gets a Unique Activation Fingerprint

| Stage | Always-On | Always-Off | Position-Dependent | Pairwise Jaccard |
|-------|-----------|-----------|-------------------|-----------------|
| S0.B0 | 0% | 29% | **71%** | 0.286 |
| S1.B0 | 0% | 0% | **100%** | 0.135 |
| S2.B0 | 0% | 47% | **53%** | 0.140 |
| S3.B0 | 0% | 87% | **13%** | 0.144 |

**Zero neurons are always-on.** Every single expanded channel can be switched off. And pairwise Jaccard similarity between pixel activation patterns is only 0.14 — meaning any two pixels share only 14% of their active neurons.

Each pixel receives a **unique sparse code** — a specific subset of neurons that fire for its particular combination of geometric features. Like a barcode.

### Revelation 3: The Net Transform Is Rotational, Not Identity-Like

The composition pw2 @ pw1 (ignoring GELU) reveals:

| Property | Stage 0 | Stage 3 |
|----------|---------|---------|
| **Diagonal fraction** | **5.3%** | **0.4%** |
| **Symmetry error** | 1.39 | 1.40 |
| **Complex eigenvalues** | 88% | **96%** |
| **Negative eigenvalues** | 52% | 51% |
| **Cross-block cosine** | **0.00** | **0.00** |

- **0.4% diagonal**: The transform has almost nothing to do with identity. It does not "keep each channel and adjust." It **completely rearranges** the feature vector.
- **96% complex eigenvalues**: The transform is dominated by **rotations** — it changes the coordinate system, not the magnitudes.
- **Cross-block cosine ≈ 0.00**: Every block performs a **completely different rotation**. Block 0's transform is orthogonal to Block 1's, which is orthogonal to Block 2's.

---

## The Machine Analogy: A Spectrometer

The semantic mixer is a **spectrometer for meaning**.

### How a Spectrometer Works

1. **Light enters** (input from geometric spatial filters)
2. **Passes through a dispersive element** (prism/grating → pw1 expansion to 4×dim)
3. **Only specific wavelengths register** (detector threshold → GELU gating at 3%)
4. **The emission spectrum identifies the substance** (sparse activation pattern → semantic fingerprint)
5. **Spectrum is read out** (pw2 compression → output feature)

### The Correspondence

| Spectrometer | Semantic Mixer |
|-------------|---------------|
| Light source | Geometric features from DW conv (edges, blobs, gradients) |
| Dispersive element (prism) | pw1: project dim → 4×dim (spread into spectral space) |
| Detector threshold | GELU: only 3% of channels activate (emission lines) |
| Emission spectrum | Sparse activation pattern (the semantic fingerprint) |
| Spectrum reader | pw2: compress 4×dim → dim (read out meaning) |
| Known spectral signatures | Learned weight matrices (the "catalog" of visual semantics) |

### Why "Spectrometer" and Not Other Analogies

| Analogy | Why It Fails |
|---------|-------------|
| **Audio mixer** | Mixers have all channels active with different volumes. Here 97% are OFF. |
| **Router/switchboard** | Routers send one signal to one destination. Here ~80 channels contribute simultaneously. |
| **Dictionary/lookup** | Lookups return one entry. Here the result is a PATTERN of ~80 activations. |
| **Filter** | Filters keep or remove — they don't create new information from the combination. |
| **Lens** | A lens focuses — it's a single transform. This is 18 orthogonal transforms in series. |

The spectrometer analogy is correct because:
1. **Extreme selectivity** (3% = spectral lines, not broadband)
2. **Identification through pattern** (which lines appear = which substance)
3. **The catalog is fixed** (weight matrices = spectral database)
4. **Different spectrometers for different properties** (orthogonal cross-block transforms = different spectral ranges)
5. **Content-dependent** at deep layers (different images produce different spectra)

---

## The Deeper Insight: Why φ Doesn't Apply Here

φ structures the spatial filters because spatial geometry IS φ-structured. Edges, blobs, gradients — these are geometric objects whose relative importance follows from the mathematics of image formation.

But the spectrometer catalogs are NOT geometric objects. They are associations:
- "This combination of edges + blob + gradient at this scale = sky"
- "This combination of edges + gradient + texture = grass"
- "This combination of smooth + warm + mid-frequency = skin"

These associations come from the **statistics of the visual world**, not from geometry. There is no first-principles reason why sky should be blue, grass should be green, or skin should be warm-toned. These are empirical facts about our particular universe.

The spectrometer learns: **given these geometric primitives, what MEANING do they have in Earth images?**

φ can't help here because φ governs structure (how things are organized), not content (what they mean). A spectrometer's design (optical path, grating equations) is geometric. Its spectral database (which lines mean which element) is empirical.

---

## The Complete Encoder Architecture — Revised

```
GRAYSCALE INPUT
     ↓
STEM: 13 geometric basis filters                      ← GEOMETRIC (φ-structured)
     ↓
×18 BLOCKS:
  ├─ DW 7×7: ~23 spatial basis filters per block       ← GEOMETRIC (φ-structured)
  │   "Illuminate the scene with geometric probes"
  │
  ├─ LayerNorm: normalize the geometric response        ← DETERMINISTIC
  │
  ├─ THE SPECTROMETER:                                  ← EMPIRICAL (NOT φ)
  │   ├─ pw1: fan out to 4×dim spectral channels
  │   │   "Disperse into meaning-space"
  │   │
  │   ├─ GELU: 97% suppressed, 3% fire                 
  │   │   "Only matching spectral lines register"
  │   │
  │   └─ pw2: read out the semantic fingerprint
  │       "The spectrum identifies the content"
  │
  └─ γ × residual + skip                               ← MIXED
     ↓
UNet DECODER → 256 features → color
```

Each block is a **geometric probe** followed by a **spectral reading**:
1. The DW conv asks: "what geometric patterns are here?" (φ-structured question)
2. The spectrometer answers: "these patterns mean ___" (empirical answer)

---

## Quantitative Summary

### Sparsity Progression (The Spectral Resolution Increases with Depth)

```
Stage 0: 18.9% active → coarse spectrum (rough category)
Stage 1: 21.2% active → still coarse
Stage 2:  5.6% active → fine spectrum (specific objects)
Stage 3:  0.7% active → ultra-fine spectrum (precise semantics)
```

As the network deepens, the spectrometer becomes more selective. Early layers fire broadly (is this an edge? a texture?). Deep layers fire on precise combinations (is this the boundary between sky and building at afternoon light?).

This mirrors how real spectrometers work: low-resolution spectrometers see broad bands (visible vs infrared), high-resolution spectrometers resolve individual atomic lines.

### Cross-Block Orthogonality (Each Block Is a Different Spectrometer)

All 18 blocks have pairwise Frobenius cosine ≈ 0.00. Each block performs a completely independent spectral analysis. This is like having 18 spectrometers, each tuned to a different wavelength range, collectively building a complete spectral profile.

### Cross-Image Behavior (Content-Dependent Spectra)

| Stage | Cross-Image Cosine | Interpretation |
|-------|-------------------|---------------|
| S0 | 0.90 | Similar spectra across images (structural features) |
| S3 | **0.32** | Very different spectra per image (semantic content) |

Early spectrometers give similar readings for all images (because basic geometry — edges, blobs — is universal). Deep spectrometers give image-specific readings (because semantic content differs).

---

## What This Means for the Project

### The Scaffolding/Content Boundary Is Now Precisely Defined

- **Scaffolding (geometric, φ-structured, derivable):**
  - Spatial basis filters (DW conv): which geometric patterns to probe
  - Importance hierarchy: φ-weighted singular values
  - Architecture: hierarchical multi-scale processing

- **Content (empirical, NOT φ, irreducible):**
  - Spectral catalogs (PW conv): what geometric patterns MEAN
  - These encode associations between visual patterns and semantic categories
  - They cannot be derived from geometry because meaning is empirical

### The Spectrometer Cannot Be Replaced

Unlike the spatial filters (which can be approximated by canonical geometry at 60% performance, or by rank-99% SVD at 99.9% fidelity), the spectrometer IS the learned knowledge. Replacing it would require either:

1. **A different training signal** (learning the meaning associations from data — which is what training does)
2. **A human-provided semantic dictionary** (explicitly listing "these features = sky, these = grass")
3. **Accepting the 60% ceiling** from geometry alone

### But the Spectrometer's STRUCTURE Is Interesting

Even though the content of the spectrometer is empirical, its FORM is revealing:
- Expand → gate → compress is the minimal architecture for sparse identification
- 3% activation = maximum selectivity with minimum ambiguity
- Orthogonal cross-block transforms = maximum information per block (no redundancy)
- 4× expansion factor = the "resolution" of the spectrometer

These structural choices may themselves be geometric necessities — the optimal design for a spectrometer given the information-theoretic constraints of the task. This is an open question.

---

## Connection to Doc 239

Doc 239 identified the boundary. Doc 240 characterizes what lives on the "content" side:

| Doc 239 Finding | Doc 240 Refinement |
|----------------|-------------------|
| "Pointwise = irreducible learned content" | It's not mixing — it's **spectral identification** |
| "99.4% of parameters" | These are the **spectral catalog** |
| "Full-rank, not φ" | Because meaning is empirical, not geometric |
| "The architecture follows φ, the room contents don't" | The spectrometer's **design** may be geometric; its **database** is empirical |

## Experimental Files

| File | Purpose |
|------|---------|
| `semantic_mixer_analysis.py` | Full instrumented analysis (GELU stats, spatial selectivity, net transform, cross-image routing) |

## Summary

The "semantic mixer" is misnamed. It is a **semantic spectrometer** — an extreme identifier that tests each pixel against a learned catalog of ~3,072 possible visual meanings, of which only ~80 (3%) match at any given location.

φ structures the light source (geometric spatial filters). The spectrometer's database (what patterns mean) is the irreducible empirical content that can only come from observing the visual world. The encoder is a **geometric instrument measuring empirical content** — a spectrometer built from φ-structured optics, reading a catalog written by nature.
