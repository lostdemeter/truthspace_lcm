# 219: The Knowledge Chemistry Guide

## Date: February 5, 2026

## Summary

We tested the "periodic table" metaphor on multiple AI problems and found it works **partially**. The refined framework uses the full chemistry metaphor:

| Level | What It Captures | Example |
|-------|------------------|---------|
| **Atoms** | Intrinsic properties | "sky is blue" |
| **Molecules** | Relationships | "sky is above ground" |
| **Reactions** | Transformations | "sunset changes sky color" |

---

## The Problem with Just Atoms

When we applied the periodic table to depth estimation, we found:

### What Works ✓
- **Position** → Depth range (near/mid/far/infinity)
- **Category** → Semantic category (sky/ground/person)
- **Surface** → Surface type (planar/curved/textured)

### What Doesn't Work ✗
- **Occlusion**: "Person is IN FRONT OF wall" - this is RELATIONAL
- **Scale**: "Person at 2m vs 20m" - this requires CONTEXT
- **Dynamics**: "Camera moves forward" - this is a TRANSFORMATION

The periodic table captures intrinsic properties but misses relationships and transformations.

---

## The Refined Framework: Knowledge Chemistry

### Level 1: Atoms (Intrinsic Properties)

Properties that belong to a single knowledge unit:

```python
@dataclass
class KnowledgeAtom:
    name: str           # Human-readable name
    symbol: str         # Short symbol
    position: tuple     # Location in feature space
    category: str       # Semantic category
    surface: str        # Texture/behavior type
    range: tuple        # Typical value range
```

**Examples:**
- Color: `SkyBlue(ab=(-5,-40), category=sky, surface=gradient)`
- Depth: `NearGround(depth=1.5m, category=ground, surface=planar)`
- Language: `NounPerson(embedding=..., category=noun, frequency=common)`

### Level 2: Molecules (Relational Properties)

Properties that describe relationships between atoms:

```python
@dataclass
class KnowledgeMolecule:
    atoms: List[KnowledgeAtom]
    relationship: str   # Type of relationship
    constraint: str     # How they relate
```

**Relationship Types:**
| Type | Description | Example |
|------|-------------|---------|
| `OCCLUSION` | A in front of B | person occludes wall |
| `ADJACENCY` | A next to B | sky above ground |
| `CONTAINMENT` | A inside B | person in room |
| `CAUSATION` | A causes B | light causes shadow |
| `SIMILARITY` | A like B | grass similar to foliage |

**Examples:**
- `PersonOnGround(person, ground, ADJACENCY, person.depth < ground.depth)`
- `SkyBehindAll(sky, *, OCCLUSION, sky.depth > all.depth)`
- `SubjectVerb(noun, verb, ADJACENCY, noun precedes verb)`

### Level 3: Reactions (Transformations)

How atoms and molecules change:

```python
@dataclass
class KnowledgeReaction:
    trigger: str        # What causes the change
    inputs: List        # Atoms/molecules before
    outputs: List       # Atoms/molecules after
    rule: str           # How the transformation works
```

**Reaction Types:**
| Type | Description | Example |
|------|-------------|---------|
| `LIGHTING` | Light changes appearance | sunset → warm colors |
| `VIEWPOINT` | Camera changes depth | zoom → scale depths |
| `TIME` | Temporal change | day → night |
| `GRAMMAR` | Linguistic transform | active → passive |

**Examples:**
- `Sunset(sky.color, LIGHTING, blue → orange)`
- `ZoomIn(all.depth, VIEWPOINT, depth *= 0.5)`
- `Passive(SVO, GRAMMAR, "dog bites man" → "man is bitten")`

---

## The Guide: How to Characterize Knowledge

### Step 1: Identify Atoms

Ask: What are the fundamental units of knowledge for this problem?

**For each atom, define:**
1. **Position**: Where in feature space?
2. **Category**: What semantic group?
3. **Surface**: What texture/behavior?
4. **Range**: What typical values?

**Organize into a periodic table** by category and properties.

### Step 2: Identify Molecules

Ask: How do atoms relate to each other?

**For each relationship, define:**
1. **Type**: Occlusion? Adjacency? Causation?
2. **Constraint**: What rule governs the relationship?
3. **Strength**: How strong is the relationship?

**Define molecular formulas** for common patterns.

### Step 3: Identify Reactions

Ask: How does knowledge transform?

**For each transformation, define:**
1. **Trigger**: What causes the change?
2. **Rule**: How do values change?
3. **Scope**: What atoms/molecules are affected?

**Define reaction equations** for dynamics.

### Step 4: Build the Knowledge Base

Compile:
1. **Periodic table** of all atoms
2. **Molecular catalog** of all relationships
3. **Reaction handbook** of all transformations

### Step 5: Use for Geometric AI

Apply:
1. **Project shapes** using atomic properties
2. **Enforce constraints** using molecular relationships
3. **Apply dynamics** using reaction rules

---

## Example: Colorization Knowledge Base

### Atoms (19 defined)
```
Sky: Sb, So, Og
Vegetation: Gg, Fg, Ao
Earth: Eb, Sd, Rg
Water: Ob, Rt
Skin: Sl, Sm, Sd
Wood: Wl, Wd
Light: Sc, Hw, Ng
```

### Molecules
```
SkyAboveGround(Sb, Gf): sky.y < ground.y
ShadowOnSurface(Sc, *): shadow.lum < surface.lum
SkinOnPerson(Sl, person): skin.region ⊂ person.region
```

### Reactions
```
Sunset: Sb → So (trigger: time=evening)
Shadow: * → Sc (trigger: occlusion)
Highlight: * → Hw (trigger: direct_light)
```

---

## Example: Depth Knowledge Base

### Atoms (13 defined)
```
Sky: Sk
Ground: Gn, Gf
Building: Wa, Fa
Vegetation: Bu, Tr
Person: Pn, Pf
Vehicle: Cn, Cf
Object: Tb, Ch
```

### Molecules
```
PersonOnGround(Pn, Gn): person.depth ≈ ground.depth
SkyBehind(Sk, *): sky.depth > all.depth
CarOnRoad(Cn, Gn): car.depth ≈ road.depth, car occludes road
```

### Reactions
```
ZoomIn: all.depth *= 0.5 (trigger: focal_length↑)
MoveForward: near.depth -= Δ, far.depth ≈ same
TiltDown: ground.gradient changes
```

---

## Example: Language Knowledge Base

### Atoms
```
Nouns: person, place, thing, idea
Verbs: action, state, motion, cognition
Adjectives: color, size, quality, quantity
```

### Molecules
```
SubjectVerb(noun, verb): noun precedes verb
VerbObject(verb, noun): verb precedes object
Modifier(adj, noun): adj precedes noun
```

### Reactions
```
Tense: verb.present → verb.past (trigger: time_reference)
Negation: verb → not + verb (trigger: negation_marker)
Passive: SVO → OVS + "by" (trigger: passive_voice)
```

---

## Validation: Does This Work?

### Test 1: Colorization
- Atoms: ✓ Defined 19 color atoms
- Molecules: ✓ Can define sky-ground, shadow-surface
- Reactions: ✓ Can define sunset, lighting changes
- **Result: Framework works**

### Test 2: Depth Estimation
- Atoms: ✓ Defined 13 depth atoms
- Molecules: ✓ Can define occlusion, adjacency
- Reactions: ✓ Can define viewpoint changes
- **Result: Framework works**

### Test 3: Language
- Atoms: ✓ Words have intrinsic properties
- Molecules: ✓ Syntax is relational
- Reactions: ✓ Grammar is transformational
- **Result: Framework works**

---

## Conclusion

The **Knowledge Chemistry** framework generalizes across domains:

1. **Atoms** = intrinsic properties (periodic table)
2. **Molecules** = relational properties (formulas)
3. **Reactions** = transformational properties (equations)

This is the guide for characterizing knowledge shapes in geometric AI.

---

## Next Steps

1. **Implement** the framework in code
2. **Build** knowledge bases for common tasks
3. **Test** whether molecular constraints improve results
4. **Explore** whether reactions enable dynamics

---

*Document created: February 5, 2026*
*Related: Doc 218 (Periodic Table), Doc 217 (Framework)*
*Implementation: `phi_geometric/evaluations/depth_periodic_table.py`*
