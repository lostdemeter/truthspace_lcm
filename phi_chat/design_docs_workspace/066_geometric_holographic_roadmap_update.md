# Design Consideration 066: Geometric Holographic Roadmap Update

## Date: 2024-12-28

## Summary of Today's Work

We explored holographic template projection and discovered a key insight: **the holographic system was using hash-based encoding, not the geometric structure we built**.

### What We Built Today

| Component | File | Purpose |
|-----------|------|---------|
| Holographic Template Projector | `core/holographic_templates.py` | Dynamic templates via interference |
| Holographic Response Synthesizer | `core/holographic_templates.py` | Multi-source response synthesis |
| Holographic Concept Navigator | `core/holographic_templates.py` | Analogy completion (hash-based) |
| Geometric Holographic Navigator | `core/geometric_holographic.py` | Analogy completion (geometric) |
| HolographicGeometricQA | `core/geometric.py` | Integrated QA with holographic templates |

### Key Discovery: Hash vs Geometric Encoding

**The Problem:**
```
Hash-based encoding:
  phase = hash(word) % 10000 / 10000 * 2π
  
  → Different words get random phases
  → No semantic structure
  → Analogies fail (0% accuracy)
```

**The Solution:**
```
Geometric encoding:
  real = φ-direction (initiator vs receiver)
  imag = mediator_ratio (verb-ness)
  
  → Semantic roles create structure
  → Similar roles cluster together
  → Analogies work when roles differ
```

### What Geometric Encoding Captures vs Doesn't

| Captures | Doesn't Capture |
|----------|-----------------|
| Position in sentence | Semantic features (gender, age) |
| Semantic role (initiator/receiver) | Category membership |
| Verb-ness (mediator ratio) | Synonymy/antonymy |
| Actions and targets | Abstract relationships |

### Analogy Results

**Hash-based:** 0/11 correct (0%)

**Geometric-based:** Partial success
- `man : woman :: boy : ?` → **girl** ✓ (when roles differ)
- `king : queen :: man : ?` → fails (both have same role)

### The Fundamental Insight

**Analogies require FEATURE DIFFERENCES, not just role differences.**

When king and queen both appear as initiators at position 0.25, they get identical encodings. The relation vector is (0, 0) - no difference to apply!

For analogies to work geometrically, we need one of:

1. **Richer geometric features** - co-occurrence patterns, target patterns
2. **Explicit relation learning** - from parallel structures (like morphology)
3. **Semantic feature dimensions** - learned from context

---

## Updated Roadmap

### Completed Today

- [x] Holographic template projection (dynamic templates)
- [x] Multi-source response synthesis
- [x] Integration with GeometricQA
- [x] Geometric holographic navigator (proof of concept)
- [x] Identified hash vs geometric encoding issue

### Phase 1: Strengthen Foundation (Updated)

#### 1.1 Unify Holographic + Geometric
- [ ] **Replace all hash-based encoding** with geometric encoding
- [ ] **Extend geometric features** to include:
  - Co-occurrence patterns (who appears with whom)
  - Action patterns (what verbs each entity uses)
  - Target patterns (what each entity acts upon)
- [ ] **Learn relations from parallel structures** (like morphology)

#### 1.2 Improve Template Projection
- [ ] **More Q&A patterns** for different question types
- [ ] **Better slot inference** from geometric properties
- [ ] **Multi-sentence templates** for longer responses

#### 1.3 Analogy Completion
- [ ] **Feature-based encoding** for semantic dimensions
- [ ] **Relation learning** from example pairs
- [ ] **Benchmark** against word2vec/GloVe analogies

### Phase 2: Scale and Specialize (Unchanged)

- [ ] Domain specialization (literature, code, data)
- [ ] 1M+ frames from Wikipedia/books
- [ ] Efficient storage and retrieval
- [ ] 5+ hop reasoning

### Phase 3: Approach LLM Parity (Unchanged)

- [ ] Free-form generation via geodesic navigation
- [ ] Complex instruction following
- [ ] Long context (100+ turns)

---

## Technical Debt Identified

1. **Hash-based encoding in `holographic_templates.py`** - Should use geometric
2. **Separate navigator classes** - Should unify into one geometric system
3. **Template projector doesn't use knowledge** - Should fill slots from GeometricKnowledge

---

## Key Files Modified/Created Today

| File | Changes |
|------|---------|
| `core/holographic_templates.py` | New: template projection, synthesis, navigation |
| `core/geometric_holographic.py` | New: geometric-based holographic navigation |
| `core/geometric.py` | Added: HolographicGeometricQA class |
| `design_considerations/065_holographic_template_projection.md` | Design doc |

---

## Next Steps (Recommended)

1. **Unify encoding** - Make all holographic operations use geometric encoding
2. **Extend features** - Add co-occurrence and action patterns to GeometricConcept
3. **Learn relations** - Use parallel structure approach for semantic relations
4. **Test at scale** - Benchmark on larger vocabulary

---

## The Vision (Refined)

**Holographic operations on geometric structure:**

```
ENCODE (geometric):
  word → (φ-direction, position, mediator_ratio, actions, targets)
  
INTERFERENCE (holographic):
  Multiple encodings → constructive/destructive → template/synthesis
  
DECODE (geometric):
  encoding → word (via nearest neighbor in learned space)
```

The key insight: **ENCODE = DECODE** still holds, but the encoding must be **geometric** (learned from structure) not **hash-based** (arbitrary).

---

*"Structure is the new training. Geometry is the new statistics."*
