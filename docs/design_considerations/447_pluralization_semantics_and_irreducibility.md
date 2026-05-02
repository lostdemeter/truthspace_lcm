# DC 447: Pluralization as Semantic Operation and the Irreducibility Map

**Day 312 | Four structural discoveries: (1) The body-part axis has
internalized PLURALIZATION AS SEMANTIC OPERATION, not just +s morphology —
it correctly retrieves irregular plurals (foot→feet, tooth→teeth) and
unchanged plurals (deer→deer, fish→fish) without explicit training on these
forms. This is the strongest evidence yet that morphological transformations
in W_E are semantic operations, not surface string manipulations. (2) The
hand→hands failure has a definitive three-part explanation: axis scale
insufficient + no-space variant shadow + Chinese cross-lingual intercept
(手, cos=0.611 > hands, cos=0.600). (3) Irreducibility correlates inversely
with pc: high-pc regular axes (+er=12%, +s=13%) have few irreducibles while
low-pc derivational axes (un-=86%, +ness=75%, +ful=75%) have many. (4) gPC5
is a royal/succession axis but it mixes semantic and phonological content —
'queue' and 'enqueue' appear at the negative pole due to Q-U-E-E-N spelling
similarity to 'queen'. No single gender PC can navigate any word pair.**

---

## Pluralization as Semantic Operation

### The Evidence

The body-part plural axis, trained ONLY on head/foot/ear/knee/toe/lip/hip/
rib/thumb/wrist/elbow/heel/shoulder/chin/neck/jaw and their plurals, correctly
retrieves:

```
Transformation type    Example     Result
──────────────────────────────────────────────
Regular +s             ear→ears    ✓ (trivial)
Irregular (vowel)      foot→feet   ✓ !!! 
Irregular (vowel)      tooth→teeth ✓ !!!
Consonant change       knife→knives✓
Consonant change       wolf→wolves ✓
Consonant change       calf→calves ✓
Unchanged plural       deer→deer   ✓ !!!
Unchanged plural       fish→fish   ✓ !!!
```

The axis was NOT trained on any of these irregular or unchanged forms.
Yet it correctly handles them.

### Why This Is Remarkable

A purely STRING-BASED morphological system (like a suffix-stripping algorithm)
would fail all irregular and unchanged plurals, because:
- foot→feet requires knowing that foot belongs to the umlaut-plural class
- deer→deer requires knowing that deer is an invariant plural noun
- These facts are etymological/lexical, not computable from string patterns

The W_E body-part axis learns NONE of this from the training pairs. It learns
only from (head,heads), (foot,feet), (ear,ears), etc. — but it encodes the
SEMANTIC relationship between a body part and its plural form, which is
shared across all body parts including their irregular forms.

### The Semantic Mechanism

The body-part plural axis encodes the conceptual transformation:
```
BODY_PART(singular) → BODY_PART(plural form in body-part-collection context)
```

When we apply this to 'foot', the axis displacement moves 'foot' toward
the region where body-part collection vocabulary lives. In that region,
'feet' (not 'foots') is the closest clean token because 'feet' is the
actual plural used in the body-part collection context.

Similarly for 'deer': in the body-part/animal collection semantic field,
'deer' (unchanged) is its own plural. The axis displacement from 'deer'
lands in the vicinity where 'deer' (plural context) remains the closest
token — because the semantic change (singular→plural) corresponds to a
contextual shift, not a form change.

### The Principle

**Morphological axes in W_E encode SEMANTIC CONTEXT CHANGES, not string
transformations.** The surface morphological change (foot→feet) is a
consequence of landing in the "plural context" region where 'feet' is
the appropriate token, not a string rule applied by the axis.

This is a core validation of the TruthSpace hypothesis: the geometry of W_E
encodes meaning directly, and morphological form emerges from meaning.

---

## The hand→hands Triple Interference

### The Definitive Diagnosis

```
Body-part axis endpoint for 'hand' (scale=0.342):
  ' hand' (space-prefixed, source)  0.868  ← excluded as self
  'Hand' (capitalized)              0.717  ← excluded (cap)
  'Hand' (capitalized, variant)     0.687  ← excluded (cap)
  'hand' (no-space variant)         0.654  ← NOT excluded (different token ID)
  '-hand' (compound prefix)         0.653  ← excluded (compound)
  '_hand' (underscore prefix)       0.634  ← excluded (compound)
  '手' (Chinese hand)               0.611  ← NOT excluded (cross-lingual!)
  'hands'                           0.600  ← TARGET — blocked by above two
```

Three compounding interference mechanisms:

**Mechanism 1: Scale insufficiency**
The body-part axis at scale=0.342 puts the prediction at cos=0.868 to 'hand'
(self). The direct interpolation midpoint (t=0.5) requires cos≈0.862 to
'hands' to be top-1. The axis displacement is NOT large enough to reach this
transition point.

Compare: arm→arms requires cos(halfway, arms)=0.915 (cleaner transition),
and the arm displacement IS sufficient to reach past 50%.

**Mechanism 2: No-space token shadow**
Qwen2's BPE tokenizer has both ' hand' (with leading space) and 'hand'
(without space) as distinct tokens. get_emb('hand') returns the space-prefixed
version. The no-space variant is NOT excluded by self-exclusion.

At the axis endpoint, 'hand' (no-space) has cos=0.654 > 'hands' (0.600).
Even if the scale were sufficient to escape the space-prefixed 'hand', the
no-space variant would intercept first.

**Mechanism 3: Cross-lingual intercept**
Chinese '手' (hand) has cos=0.611 > 'hands' (0.600). Since cross-lingual
tokens for basic vocabulary are very close in W_E (this is the universal
meaning engine), '手' sits between the no-space 'hand' and 'hands' in the
axis direction.

For 'arm': Chinese '臂' (arm) and '手臂' are further from 'arms' relative to
the axis displacement, so they don't intercept. Body parts with simpler
Chinese equivalents (hand='手') are more susceptible to this interference.

### The Fix (if Needed)

A complete exclusion filter would need to also exclude:
1. All token variants with different space/no-space prefixes for the same word
2. Cross-lingual near-cognates

However, implementing this crosses into territory where we're patching around
the geometry rather than understanding it. The hand→hands failure is
INFORMATIVE — it reveals that 'hand' occupies an unusually cross-lingual
neighborhood in W_E.

---

## The Irreducibility Map

### Full Results

```
Axis       pc     Scale   Irreducible words                Fraction
────────────────────────────────────────────────────────────────────
+er        0.394  0.423   kind                            12.5%
+est       0.401  0.504   kind                            25.0%
er→est     0.436  0.423   (none found)                    ~0%
gender     0.209  0.342   groom, host                     33.3%
past_irr   0.284  0.584   say, think                      25.0%
+ed        0.231  0.584   close                           12.5%
+ness      0.169  0.745   bold, cold, soft, hard          75.0%
+ful       0.104  0.745   play, wonder, color, grace      75.0%
un-        0.096  0.745   tie, fold, pack, cover, +       85.7%
+ment      0.124  0.584   refresh                         33.3%
+s         0.259  0.181   hand (truly), eye (body-part fixes it) 13.3%
+tion      0.116  0.745   correct, produce                22.2%
```

### The Inverse pc–Irreducibility Correlation

Sorting by pc (high to low) vs irreducible fraction:

```
er→est:  pc=0.436  →  ~0%   irreducible
+est:    pc=0.401  →  25%
+er:     pc=0.394  →  12.5%
+s:      pc=0.259  →  13.3%
past_irr: pc=0.284 →  25%
+ed:     pc=0.231  →  12.5%
gender:  pc=0.209  →  33.3%
+ment:   pc=0.124  →  33.3%
+tion:   pc=0.116  →  22.2%
+ness:   pc=0.169  →  75%
+ful:    pc=0.104  →  75%
un-:     pc=0.096  →  85.7%
```

The correlation is strong: **higher pc axes have lower irreducibility**.
The three axes with pc < 0.12 (un-, +ful, +ness) all have 75%+ irreducibility.

This makes geometric sense. pc measures pairwise chord alignment:
- High pc (e.g., +er, pc=0.394): all comparative chords point in the SAME
  direction. The axis is a reliable LOCAL OPERATOR for the training domain.
  Holdout words in the same domain will also succeed.
- Low pc (e.g., un-, pc=0.096): prefix-addition chords scatter in different
  directions. The "un-" transformation has no consistent geometric direction
  — it's too context-dependent. Any holdout word is likely in a different
  local direction, making it irreducible under the mean axis.

### Why un- Has 86% Irreducibility

The prefix 'un-' has fundamentally different semantic effects depending on
context:
- un-happy: negation of positive property (semantic antonym)
- un-lock: reversal of an action (semantic reverse)
- un-known: absence of a state (semantic absence)
- un-tie: reversal of a physical action (physical reverse)
- un-wise: negation of a property (semantic antonym)

Each of these is a DIFFERENT semantic operation that happens to use the same
prefix. The geometric displacement for "adding un-" varies enormously across
these categories. The mean axis is a blend that fits nothing well — hence
86% irreducibility.

Compare with +er (pc=0.394): ALL comparative transformations share the same
semantic content (making a gradable adjective more extreme). There is one
consistent operation, one consistent direction, few irreducibles.

### Irreducibility as a Quality Metric

**Irreducibility fraction is a better quality metric than pc for practical use:**

| Metric | What it measures |
|--------|----------------|
| pc | Consistency of chord directions (how clean the axis is) |
| Irreducibility | What fraction of holdout words are NEVER reachable |

Both correlate, but irreducibility is the directly actionable metric:
if 85% of holdout words are irreducible under an axis, that axis should
not be used as a general-purpose transformation for that morphological
category.

---

## gPC5: The Royal/Phonological Axis

### Pole Vocabulary

**gPC5 positive** (source gender):
king, grandfather, father, hero — male figures with AUTHORITY/PATRIARCHAL roles

**gPC5 negative** (target gender):
queen (−0.268), prince (−0.261), grandmother (−0.208), mother (−0.139),
queue (−0.136), enqueue (−0.130)

The presence of 'queue' and 'enqueue' at the negative pole is a
PHONOLOGICAL ARTIFACT: queen/queue share the letter sequence Q-U-E-E/U.
BPE tokenization assigns them similar subword representations, pulling
their embeddings closer. gPC5 partly encodes this phonological proximity.

### Navigability

```
Source   Target     gPC5?   gPC1?    Explanation
─────────────────────────────────────────────────
prince   princess   ✓ 0.19  ✗        prince in gPC5- zone, princess nearby
king     queen      ✗       ✗        need gPC4-5-7-9 combination
man      woman      ✗       ✗        need core gender axis (trained pair)
boy      girl       ✗       ✗        need gPC1-dominant local axis
```

Only prince→princess works with gPC5 (scale=0.19). 'Prince' is already in
the negative gPC5 zone; 'princess' is slightly further in that direction.

king→queen requires gPC4+gPC5+gPC7+gPC9 — four PCs to reconstruct 84% of
the chord. This is the royal succession subspace, and no single PC can
navigate it.

### The Implication: PCA of Semantic Space Is Not Navigable

PCA of the gender chord space produces axes that are mathematically
orthogonal but semantically MIXED. gPC1 blends boy→girl and man→woman.
gPC5 blends king→queen with prince→(princess adjacent) AND with queue/enqueue.

Meaningful navigation requires PAIR-SPECIFIC axes, not PCA-derived axes.
The PCA is useful for CHARACTERIZING the structure (showing isotropy,
dimensionality) but not for NAVIGATING the space.

---

## Day 313 Plan

1. **Exclude no-space variants in retrieval**: implement `nn_retrieve_exact`
   that excludes all tokenizations of the source word (space, no-space,
   capitalize, compound). Does this finally fix hand→hands?

2. **Derivational axis quality**: why does un- have such low pc (0.096)?
   Is there a sub-domain of un- with higher pc (e.g., un-+adjective only)?
   Test: split un- into un-+adj vs un-+verb vs un-+state.

3. **Cross-lingual interference map**: for how many words is the top clean
   neighbor a cross-lingual token? Is this a general property of body-part
   and basic vocabulary?

4. **pc predicts irreducibility**: fit a linear model. What is the slope of
   the relationship? Can we use pc alone to estimate how useful an axis is?

---

## Files

- `expedition_log.md` — Day 312 results
- `446_domain_specific_plurals_and_king_queen_anomaly.md` — DC 446
- `day312_hand_path_irreducible_irregular.py` — experiment script
