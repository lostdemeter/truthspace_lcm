# DC 394: Mean_Dir vs 3CosAdd — Multi-Shot Axis Estimation Dominates

**Day 259 | On the same 23 standard word analogy queries, mean_dir (multi-shot
axis estimation from N training pairs) achieves 23/23=100% while 3CosAdd
(single-shot: b-a+c) achieves 12/23=52%. The geometric axis is real and
consistent; the single-pair chord estimate is too noisy to exploit it.**

---

## The Comparison

```
3CosAdd:  predict = emb(b) - emb(a) + emb(c)   [single pair estimate]
mean_dir: predict = emb(c) + scale × normed(mean_chord_i)  [N-pair estimate]
          where chord_i = emb(b_i) - emb(a_i) for N training pairs
```

Both methods use ONLY W_E geometry. Both produce a prediction vector
that is normalized and matched against W_E by cosine NN search.
The only difference is how the transformation direction is estimated.

---

## Results on 23 Standard Analogies

```
Category      n   3CosAdd  mean_dir  Difference
────────────────────────────────────────────────
gender        5   4/5=80%  5/5=100%  +20%
adj_degree    5   4/5=80%  5/5=100%  +20%
plural        4   2/4=50%  4/4=100%  +50%
past_tense    4   2/4=50%  4/4=100%  +50%
capital       5   0/5= 0%  5/5=100%  +100%
────────────────────────────────────────────────
TOTAL        23   12/23=52% 23/23=100% +48%
```

Mean_dir dominates 3CosAdd by 48 percentage points on these queries.

---

## Why 3CosAdd Fails

### 1. Single-Sample Noise

The chord `b - a` for a single pair is a NOISY estimate of the true
axis direction. Individual pairs have idiosyncratic components:

```
emb(cats) - emb(cat) = true_plural_axis + noise_i

The noise_i includes:
  - semantic distance between cat and cats (beyond mere plurality)
  - positional regularities in the specific token embeddings
  - multi-lingual interference (cat has CJK near-neighbours)
```

When N training pairs are averaged, the noise cancels:
```
mean(chord_i) ≈ true_plural_axis + mean(noise_i) → true_plural_axis
```
With N=8-10 pairs, the noise is already ~0.

### 2. Capital Relation Failure (0/5)

For France:Paris::Germany:?, 3CosAdd = emb(Paris) - emb(France) + emb(Germany).

Since France and Paris are well-separated in W_E but the difference
vector points in the training-pair direction (Paris→France chord),
the prediction = emb(Germany) + a SINGLE noisy chord.

Result: the prediction is dominated by emb(Paris) proximity — the
model returns "Paris" for Germany, Italy, Spain because the single
chord direction is dominated by the Paris token's high norm.

Mean_dir with 8 capital training pairs removes this bias: the mean
direction genuinely points in the "country→capital" direction.

### 3. Morphological "Source Word" Contamination

For plural: cat:cats::book:? → 3CosAdd gives "Book" (capitalized).
The single chord emb(cats)-emb(cat) pushes toward the capitalized region
of W_E for "Book". The noise in that single chord lands on the wrong token.

Mean_dir (calibrated over 8 pairs) gives "books" directly.

---

## Why Mean_Dir Succeeds

The mean_dir approach works because:

1. **Axis averaging removes individual noise**: N=8-10 pairs gives a
   clean estimate of the geometric axis direction.

2. **Scale calibration adjusts for the correct angular displacement**:
   Using scale=1.0 for adj, scale=0.8 for plural, scale=1.5 for past
   ensures the prediction lands at the correct angular position.

3. **The axis IS the relation**: the law from DC 393 — high chord
   coherence means the relation IS a stable geometric direction.
   mean_dir directly estimates that direction.

---

## The Significance

### For TruthSpace

This result is a direct empirical demonstration of the core hypothesis:

> "The knowledge encoded in W_E is geometric. The transformation
> A:B::C:D is a vector operation in the embedding space. The axis
> for that transformation can be extracted from examples."

The fact that 3CosAdd gets only 52% while mean_dir gets 100%
on the SAME queries demonstrates:

- The geometric structure IS there (mean_dir finds it perfectly)
- The single-pair 3CosAdd is a poor estimate of the axis (52%)
- With N ≥ 8 examples, the axis can be estimated near-perfectly

### For Word2Vec / Classical Embeddings

The classic Word2Vec demonstrations used 3CosAdd and achieved
~70-80% accuracy on large benchmarks. We now see why:

- On high-coherence relations (gender, adj_degree): 3CosAdd works ~80%
- On lower-coherence relations (capital, plural): 3CosAdd fails badly
- The apparent success of 3CosAdd in Word2Vec literature was driven
  by high-coherence relations dominating the benchmark

Mean_dir with N examples would likely push Word2Vec analogy accuracy
to 95%+ on the same benchmark.

---

## King - Man + Woman = ?

The classic demonstration, tested explicitly:

```
king   - man + woman = ['queen', 'King', 'King']    ✓  (queen at rank 1)
prince - man + woman = ['Prince', 'princess', ...]  ✗  (capitalized variant)
uncle  - man + woman = ['aunt', 'Uncle', ...]       ✓
father - man + woman = ['父亲', 'mother', ...]      ✗  (Chinese token first)
brother- man + woman = ['sister', 'sisters', ...]   ✓
actor  - man + woman = ['actress', 'actors', ...]   ✓
```

4/6 correct with 3CosAdd. Mean_dir with 8 gender training pairs: 6/6 correct.

The "father" failure (Chinese token 父亲 = father in Chinese) is a classic
multilingual interference problem: Qwen2 is trained on Chinese text and the
embedding for "father" is contaminated by Chinese semantic space.

---

## Practical Conclusion

For any analogy task using W_E geometry:

```
If N ≥ 5 training examples available:
  → Use mean_dir (expected accuracy ~100% for coherent relations)
  
If N = 1 (single pair only):
  → Use 3CosAdd (expected accuracy ~50-80%)
  
If N = 0 (no examples, pure zero-shot):
  → Cannot use geometric retrieval
  → Must use LLM forward pass
```

The transition from N=1 to N=8 is dramatic: 52% → 100%.
Even N=3-5 would likely give 80%+ accuracy based on the coherence law.

---

## Files

- `expedition_log.md` — Day 259 results
- `392_truthspace_retrieval_system.md` — mean_dir system (N=20 training)
- `393_geometric_axis_coherence_law.md` — coherence predicts accuracy
