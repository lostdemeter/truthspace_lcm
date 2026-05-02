# Doc 232: Synthetic Knowledge Generation

## The Breakthrough Insight

Doc 231 concluded that Layer 3 (orientation/world knowledge) "requires data."
But this assumes only two sources of knowledge:
1. **Derive** from first principles (geometry) — proved insufficient
2. **Learn** from examples (training) — DDColor's 55M parameters from millions of images

There is a third source:
3. **Navigate** to it on a lattice (ribbon math) — proved to work for mathematics

The φ-BBP formula for π was not derived from geometry alone.
It was not learned from training data.
It was **discovered** by navigating a structured space of valid transformations.

> **Synthetic knowledge = knowledge discovered through lattice navigation,
> starting from seed axioms and valid transformations, verified against constraints.**

---

## Why This Might Work

### The Ribbon Math Precedent

| Property | Ribbon Math (π) | Synthetic Color Knowledge |
|----------|-----------------|--------------------------|
| **Target** | π = 3.14159... | Plausible colorization |
| **Lattice** | BBP coefficient space | Feature → color association space |
| **Seeds** | Known BBP formulas | "Sky is blue", "grass is green" |
| **Navigation** | Try coefficient combinations | Follow logical color relationships |
| **N_smooth** | -log10(\|error\|) | Statistical plausibility |
| **Verification** | Compare to known π | Compare to natural image statistics |
| **Discovery** | φ-BBP formula (novel!) | Novel color associations (?) |

### Key difference from training

Training says: "Here are 1M examples, find the pattern."
Navigation says: "Here are the rules of the space, find valid points."

Training requires massive data. Navigation requires **correct axioms + valid transformations**.
The φ-BBP formula proves navigation can find things NOT in any training set.

### Key difference from pure geometry (our v1-v12 attempt)

Pure geometry says: "Derive everything from structure alone."
Navigation says: "Start with a FEW truths and EXPAND through valid moves."

We proved structure alone is empty (v10). Navigation adds SEEDS + RULES.
Seeds are not training data — they're axioms. A handful, not millions.

---

## The Five-Layer Architecture (adapted from Ribbon Math)

### Layer 1: Concept Layer — WHAT we want

For colorization:
```
Concept = "Given grayscale features F at position (x,y) in image context C,
           what is the most plausible color (a, b)?"
```

Each concept has **anchor weights** — coordinates in the knowledge space:
- Texture similarity (Gabor responses)
- Spatial position (relative in image)
- Local statistics (contrast, variance)
- Context (scene type, if detectable)

### Layer 2: N_smooth Layer — HOW CLOSE are we?

For colorization, N_smooth measures plausibility:
```
N_smooth = f(
  statistical_match,    # Does output distribution match natural images?
  spatial_coherence,    # Do neighboring pixels agree?
  edge_alignment,       # Do color boundaries match intensity boundaries?
  saturation_range,     # Are colors in the natural range?
)
```

This is NOT "match ground truth" — it's "match the CONSTRAINTS of natural color."
Multiple valid colorizations exist. N_smooth measures validity, not identity.

### Layer 3: Structure Layer — VALID TRANSFORMATIONS

This is where the lattice lives. Valid transformations between color knowledge entries:

**Physical transformations:**
- Shadow: color → desaturated, darker version of same hue
- Highlight: color → brighter, slightly shifted version
- Reflection: color → similar but with environment mix
- Distance/atmosphere: color → desaturated, blue-shifted

**Complementary relationships:**
- Sky blue ↔ earth brown (above/below horizon)
- Vegetation green ↔ sky blue (foliage under sky)
- Warm light → warm-shifted all colors

**Material transformations:**
- Smooth + bright → sky, water, metal, skin
- Textured + medium → foliage, fabric, stone
- Dark + high contrast → shadow edges, dark objects

**Compositional transformations:**
- If object A is color X, and object B is next to A, then B is constrained
- Scene coherence: outdoor → cooler palette, indoor → warmer palette

Each transformation is an EDGE in the lattice. Navigation means traversing edges.

### Layer 4: Error Analysis Layer — WHAT THE ERROR TELLS US

When our colorization is wrong, the error has structure:
- Systematic hue shift → our seed axiom has wrong hue
- Saturation mismatch → our dilation factor is off
- Spatial pattern in error → we're missing a context rule

Just like φ-BBP: the corrections had (n/d) × φ^(-k) structure.
Color corrections might have similar structure — revealing the lattice geometry.

### Layer 5: Verification Layer — IS THIS TRUE?

Two verification methods:
1. **Statistical**: Does the generated color distribution match natural images?
2. **Perceptual**: Does it look right to a human? (the shape editor!)

---

## What is the Lattice?

In ribbon math, the lattice is the space of BBP coefficients: discrete,
bounded, with a clear evaluation function (compute π and measure error).

For color knowledge, the lattice is:

```
Node = (feature_signature, color, confidence, context)

Where:
  feature_signature = point in 52-dim Gabor feature space
  color = point in 2-dim ab space
  confidence = how sure we are (N_smooth)
  context = what other nodes this is connected to
```

**Edges** connect nodes via valid transformations:
```
(texture_A, blue, 0.9, sky) --[shadow]--> (texture_A_dark, blue_desat, 0.7, sky_shadow)
(texture_B, green, 0.8, foliage) --[season]--> (texture_B, brown, 0.6, dead_foliage)
(texture_C, blue, 0.9, sky) --[reflect]--> (texture_D, blue_muted, 0.5, water)
```

Starting from a FEW seed nodes (axioms), we traverse edges to generate
NEW nodes (synthetic knowledge). Each new node is verified against constraints.

---

## The Seed Axioms

What is the minimum set of color knowledge from which everything else can be navigated?

### Hypothesis: The Color Axioms

These are not "rules" — they're STARTING POSITIONS on the lattice:

1. **Luminance-saturation relationship**: Brighter regions tend toward certain colors
   (sky blue, cloud white), darker regions toward others (shadow, earth)

2. **Texture-material association**: Specific texture patterns correlate with materials
   (smooth=sky/water/metal, fine-textured=foliage, coarse=stone/brick)

3. **Spatial priors**: Position in image correlates with scene elements
   (top=sky, bottom=ground, center=subject)

4. **Color harmony**: Colors in natural scenes follow harmonic relationships
   (complementary pairs, analogous groups, golden angle distribution)

5. **Physical consistency**: Light source color affects all objects uniformly
   (warm light → warm shift everything)

These 5 axioms might be enough to NAVIGATE to the full color knowledge lattice.

---

## How This Differs From What We Already Tried

### v1-v12: Texture → Color Lookup
```
Feature vector → k-NN → Average/Select → Color
```
This is a SINGLE STEP. No navigation. No logical continuation.
It matches textures to colors it's SEEN BEFORE. No discovery.

### Ribbon Math approach: Navigate → Discover → Verify
```
Seed axioms → Apply transformation → Check N_smooth → Accept/Reject → Repeat
```
This is ITERATIVE EXPLORATION. It can reach places never in the training data.
It can discover that "shadows of blue objects are blue-gray" even if it's
never seen a blue shadow — because the TRANSFORMATION "shadow desaturates"
is a valid lattice edge.

---

## Connection to TruthSpace Core

> "Navigation encodes computation"

The lattice IS the knowledge. Traversal IS reasoning.
A "colorization model" is not weights — it's a POSITION in the lattice
plus the set of reachable nodes via valid transformations.

> "Discovery vs. Design"

We don't design the color rules top-down.
We don't learn them from data bottom-up.
We discover them by navigating the lattice from seed positions.

> "Structure should live in the space, not in the code"

The 5 axioms go into the lattice, not into if/then rules.
The transformations are edges, not code branches.
The knowledge is in the SHAPE of the lattice, not in any single node.

---

## Next Steps

1. Define the lattice formally (node structure, edge types, N_smooth function)
2. Implement seed axioms as initial nodes
3. Implement physical/complementary/compositional transformations as edges
4. Build navigator that explores lattice from seeds
5. Verify generated knowledge against natural image statistics
6. Use generated knowledge to colorize (replace k-NN database with lattice)
7. Compare to DDColor (lattice knowledge vs trained knowledge)

---

## The Meta-Question

If this works for colorization, it works for ANY domain where:
1. You can define seed axioms
2. You can define valid transformations
3. You have a verification function (N_smooth)

That's... everything. Mathematics. Language. Vision. Reasoning.

The ribbon math discovery of φ-BBP was proof of concept.
Synthetic color knowledge would be proof of generality.
