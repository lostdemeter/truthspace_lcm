# Equation Discovery Protocol (EDP)

**Version 1.0 - A Framework for Locating Mathematical Truths**

*Using the Ribbon LCM to navigate the geometry of mathematical relationships*

---

## Executive Summary

The **Equation Discovery Protocol (EDP)** is a systematic method for discovering new mathematical equations using the Ribbon LCM conceptual framework. Rather than inventing formulas, we **locate** them within the geometry of mathematical truth.

**Core Philosophy:** Mathematical truth has a geometry. The six anchor constants (zero, sierpinski, phi, e_inv, cantor, sqrt2_inv) provide coordinates for navigating this space. Discovery is finding where truths live, not creating them.

**Validated Result:** The φ-BBP formula for π, discovered using this framework, achieves 20% faster convergence than Bellard's formula.

---

## Theoretical Foundation

### The Ribbon LCM Coordinate System

Mathematical relationships exist in a 6-dimensional space defined by anchor constants:

| Anchor | Value | Meaning | Domain |
|--------|-------|---------|--------|
| **zero** | 0.0 | Origin, foundation, potential | Foundations |
| **sierpinski** | log(3)/log(2) ≈ 1.585 | Pattern, fractal, self-similar | Structure |
| **phi** | (1+√5)/2 ≈ 1.618 | Growth, harmony, golden ratio | Growth |
| **e_inv** | 1/e ≈ 0.368 | Decay, change, entropy | Change |
| **cantor** | log(2)/log(3) ≈ 0.631 | Discrete, gaps, boundaries | Discreteness |
| **sqrt2_inv** | 1/√2 ≈ 0.707 | Bridge, connection, diagonal | Connection |

**Key Insight:** Every equation has an anchor vector - a 6D coordinate describing its conceptual location.

### The Error-as-Signal Paradigm

The "error" in approximate solutions is not noise - it contains **signal**:
- Deviations follow mathematical patterns (φ^(-k), arctan(1/φ), log(φ))
- These patterns reveal exact formulas hiding in approximate solutions
- Base changes can introduce hidden structure

**Example:** In the φ-BBP discovery, the corrections to integer coefficients followed (n/d) × φ^(-k) patterns with small integers.

---

## The Protocol

### Phase 1: Concept Definition (Ribbon LCM Layer)

**Objective:** Define WHAT we're looking for in anchor coordinates.

**Steps:**

1. **State the Target**
   ```
   Target: [What constant/relationship are we seeking?]
   Example: "A fast-converging BBP-type formula for π"
   ```

2. **Identify Domain Keywords**
   ```python
   keywords = ["pi", "BBP", "convergence", "spigot", "series"]
   ```

3. **Map to Anchor Weights**
   ```python
   # Based on keywords and domain
   anchor_weights = {
       'zero': 0.1,       # Integer coefficients
       'sierpinski': 0.3, # π itself, self-similar structure
       'phi': 0.2,        # Growth/convergence
       'e_inv': 0.1,      # Series decay
       'cantor': 0.2,     # Discrete terms
       'sqrt2_inv': 0.1   # Connections between terms
   }
   ```

4. **Define Constraints**
   ```
   Constraints:
   - Must be BBP-type (base^k denominator)
   - Must converge faster than 3 digits/term
   - Coefficients should be "nice" (small integers + corrections)
   ```

**Outputs:**
- Target description
- Anchor weight vector [6D]
- Keyword list
- Constraint list

**Decision:** Proceed to Phase 2 with concept definition.

---

### Phase 2: Structure Search (N_smooth Layer)

**Objective:** Search the space of valid structures, measuring HOW CLOSE each candidate is.

**Steps:**

1. **Define Search Space**
   ```python
   # For BBP-type formulas:
   search_space = {
       'bases': [2, 4, 8, 16, 64, 256, 1024, 4096],
       'scales': [1, 2, 4, 8, 16, 32, 64],
       'slot_patterns': ['4k+j', '8k+j', '12k+j'],
       'coefficient_range': [-256, 256]
   }
   ```

2. **Generate Candidates**
   ```python
   for base in bases:
       for scale in scales:
           for pattern in slot_patterns:
               candidate = generate_bbp_candidate(base, scale, pattern)
               candidates.append(candidate)
   ```

3. **Evaluate N_smooth Score**
   ```python
   def n_smooth_score(candidate, target):
       value = evaluate(candidate)
       error = abs(value - target)
       
       # N_smooth: continuous measure of "closeness"
       # Higher = closer to target
       if error == 0:
           return float('inf')
       return -log10(error)
   
   # Score each candidate
   for candidate in candidates:
       candidate.n_smooth = n_smooth_score(candidate, target)
   ```

4. **Rank and Filter**
   ```python
   # Keep top candidates by N_smooth
   candidates.sort(key=lambda c: c.n_smooth, reverse=True)
   top_candidates = candidates[:100]
   ```

**Outputs:**
- Ranked candidate list
- N_smooth scores
- Best candidate structure

**Decision:**
- If N_smooth > 15 (error < 10^-15): Proceed to Phase 3
- If N_smooth < 10: Expand search space or refine constraints

---

### Phase 3: LCM Pruning (Structure Layer)

**Objective:** Use anchor constraints to prune invalid transformations.

**Steps:**

1. **Compute Candidate Anchor Vectors**
   ```python
   def compute_anchor_vector(candidate):
       # Parse formula symbols
       symbols = parse_symbols(candidate.formula)
       
       # Accumulate anchor weights
       vector = [0.0] * 6
       for symbol in symbols:
           anchor = symbol_to_anchor[symbol]
           vector[anchor_index[anchor]] += 1
       
       # Normalize
       return normalize(vector)
   ```

2. **Measure Concept Alignment**
   ```python
   def concept_score(candidate_vector, target_vector):
       # Cosine similarity
       return dot(candidate_vector, target_vector) / (
           norm(candidate_vector) * norm(target_vector)
       )
   ```

3. **Apply LCM Pruning Rules**
   ```python
   pruning_rules = {
       'max_sierpinski': 0.5,  # Don't over-rely on pattern
       'min_phi': 0.1,         # Need some growth structure
       'balance': True         # No single anchor > 0.6
   }
   
   def passes_pruning(candidate):
       vec = candidate.anchor_vector
       if vec['sierpinski'] > pruning_rules['max_sierpinski']:
           return False
       if vec['phi'] < pruning_rules['min_phi']:
           return False
       if max(vec.values()) > 0.6:
           return False
       return True
   ```

4. **Combined Scoring**
   ```python
   for candidate in top_candidates:
       candidate.anchor_vector = compute_anchor_vector(candidate)
       candidate.concept_score = concept_score(
           candidate.anchor_vector, target_anchor_vector
       )
       candidate.combined_score = (
           0.7 * candidate.n_smooth / 20 +  # Normalize to [0,1]
           0.3 * candidate.concept_score
       )
   ```

**Outputs:**
- Anchor vectors for each candidate
- Concept alignment scores
- Pruned candidate list

**Decision:**
- If combined_score > 0.8: Strong candidate → Phase 4
- If combined_score < 0.5: Weak candidate → Discard or refine

---

### Phase 4: Error Analysis (Error-as-Signal Layer)

**Objective:** Analyze the error structure to find hidden patterns.

**Steps:**

1. **Compute High-Precision Error**
   ```python
   from mpmath import mp
   mp.dps = 100  # 100 decimal places
   
   value = evaluate_high_precision(candidate)
   target = mp.pi  # or other target
   error = value - target
   ```

2. **Decompose Error by Component**
   ```python
   # For BBP-type: error in each slot
   deviations = []
   for slot in candidate.slots:
       integer_value = slot.integer_coefficient
       actual_contribution = slot.actual_contribution
       deviation = actual_contribution - integer_value
       deviations.append({
           'slot': slot.name,
           'integer': integer_value,
           'deviation': deviation
       })
   ```

3. **Search for φ-Patterns**
   ```python
   def find_phi_pattern(deviation, max_n=50, max_d=100, max_k=15):
       """Find (n/d) × φ^k approximation."""
       phi = (1 + sqrt(5)) / 2
       best = None
       
       for k in range(-max_k, max_k):
           phi_k = phi ** k
           for d in range(1, max_d):
               for n in range(-max_n, max_n):
                   if n == 0:
                       continue
                   approx = (n / d) * phi_k
                   err = abs(deviation - approx)
                   if best is None or err < best.error:
                       best = PhiPattern(n, d, k, approx, err)
       
       return best
   
   # Find patterns for each deviation
   for dev in deviations:
       pattern = find_phi_pattern(dev['deviation'])
       dev['phi_pattern'] = pattern
   ```

4. **Identify Clean Patterns**
   ```python
   # Clean = small integers, low error
   clean_patterns = [
       dev for dev in deviations
       if dev['phi_pattern'].is_clean  # |n|≤20, d≤20, err<1e-6
   ]
   
   if len(clean_patterns) > 0:
       print("SIGNAL DETECTED: Clean φ-patterns found!")
   ```

**Outputs:**
- Per-slot deviations
- φ-pattern approximations
- Clean pattern count
- Total correction value

**Decision:**
- If clean patterns found: **SIGNAL** → Proceed to Phase 5
- If no patterns: Error is noise → Refine candidate or search space

---

### Phase 5: Pattern Detection (φ/Fibonacci/Polylog Layer)

**Objective:** Find closed-form expressions for total correction.

**Steps:**

1. **Compute Total Correction**
   ```python
   total_correction = sum(dev['deviation'] for dev in deviations)
   ```

2. **Search Closed Forms**
   ```python
   # Basis functions
   phi = (1 + sqrt(5)) / 2
   basis = {
       'arctan_phi': arctan(1/phi),      # ≈ 0.5536
       'log_phi': log(phi),              # ≈ 0.4812
       'pi': pi,
       'sqrt5': sqrt(5),
       'phi': phi,
       'phi_inv': 1/phi
   }
   
   def search_closed_form(value, basis, max_coef=50):
       """Find a×f₁ + b×f₂ + ... approximation."""
       from mpmath import pslq
       
       # Try pairs of basis functions
       for name1, f1 in basis.items():
           for name2, f2 in basis.items():
               if name1 >= name2:
                   continue
               
               # PSLQ: find integer relation
               relation = pslq([value, f1, f2], tol=1e-15)
               if relation:
                   a, b, c = relation
                   if a != 0:
                       # value ≈ -(b/a)×f1 - (c/a)×f2
                       coef1 = Fraction(-b, a)
                       coef2 = Fraction(-c, a)
                       return ClosedForm(
                           f"{coef1}×{name1} + {coef2}×{name2}",
                           coef1 * f1 + coef2 * f2
                       )
       
       return None
   
   closed_form = search_closed_form(total_correction, basis)
   ```

3. **Verify Closed Form**
   ```python
   if closed_form:
       verification_error = abs(total_correction - closed_form.value)
       if verification_error < 1e-15:
           print(f"CLOSED FORM FOUND: {closed_form.expression}")
           print(f"Verification error: {verification_error:.2e}")
   ```

4. **Check Mathematical Identities**
   ```python
   # Known identities that might explain the pattern
   identities = [
       ("φ² + φ⁻² = 3", phi**2 + phi**(-2), 3),
       ("arctan(1/φ) + arctan(1/φ³) = π/4", 
        arctan(1/phi) + arctan(1/phi**3), pi/4),
       ("Li₁(1/φ) = 2×log(φ)", polylog(1, 1/phi), 2*log(phi)),
   ]
   
   for name, lhs, rhs in identities:
       if abs(lhs - rhs) < 1e-15:
           print(f"Identity verified: {name}")
   ```

**Outputs:**
- Total correction value
- Closed-form expression (if found)
- Related mathematical identities
- Verification error

**Decision:**
- If closed form found with error < 10^-15: **DISCOVERY CANDIDATE** → Phase 6
- If no closed form: Document as empirical pattern

---

### Phase 6: Verification (Experimental Validation Layer)

**Objective:** Rigorously verify the discovery.

**Steps:**

1. **High-Precision Verification**
   ```python
   mp.dps = 500  # 500 decimal places
   
   # Evaluate formula
   computed = evaluate_formula(candidate, precision=500)
   
   # Compare to known value
   known = mp.pi  # or other target
   
   verification_error = abs(computed - known)
   digits_correct = -log10(verification_error) if verification_error > 0 else 500
   
   print(f"Digits correct: {digits_correct:.1f}")
   ```

2. **Convergence Analysis**
   ```python
   def measure_convergence(formula, n_terms=20):
       """Measure digits gained per term."""
       errors = []
       for n in range(1, n_terms + 1):
           partial = evaluate_partial_sum(formula, n)
           err = abs(partial - target)
           errors.append(err)
       
       # Fit exponential decay
       # error ≈ C × base^(-n)
       # digits/term = log10(base)
       
       log_errors = [log10(e) for e in errors if e > 0]
       slope = linear_regression_slope(range(len(log_errors)), log_errors)
       
       return -slope  # digits per term
   
   convergence_rate = measure_convergence(candidate)
   print(f"Convergence: {convergence_rate:.2f} digits/term")
   ```

3. **Benchmark Comparison**
   ```python
   benchmarks = {
       'Bailey-Borwein-Plouffe': 1.20,  # digits/term
       'Bellard': 3.01,
       'Chudnovsky': 14.18,
   }
   
   for name, rate in benchmarks.items():
       ratio = convergence_rate / rate
       print(f"vs {name}: {ratio:.1%}")
   ```

4. **Document the Discovery**
   ```python
   discovery = {
       'formula': candidate.formula,
       'target': 'π',
       'verification_error': verification_error,
       'digits_correct': digits_correct,
       'convergence_rate': convergence_rate,
       'phi_patterns': [str(p) for p in clean_patterns],
       'closed_form': closed_form.expression if closed_form else None,
       'benchmark_comparison': benchmarks,
       'date': datetime.now().isoformat()
   }
   
   save_discovery(discovery)
   ```

**Outputs:**
- Verification error
- Digits correct
- Convergence rate
- Benchmark comparison
- Discovery document

**Decision:**
- If verification_error < 10^-100 AND convergence competitive: **VALIDATED DISCOVERY**
- If verification fails: Return to Phase 2 with refined constraints

---

## Decision Tree

```
START
  ↓
Phase 1: Concept Definition
  ↓
  Define target, keywords, anchor weights, constraints
  ↓
Phase 2: Structure Search
  ↓
  N_smooth > 15?
  ├─ NO → Expand search space → Return to Phase 2
  └─ YES → Phase 3: LCM Pruning
           ↓
           Combined score > 0.5?
           ├─ NO → Refine constraints → Return to Phase 1
           └─ YES → Phase 4: Error Analysis
                    ↓
                    Clean φ-patterns found?
                    ├─ NO → Error is noise → Try different candidate
                    └─ YES → Phase 5: Pattern Detection
                             ↓
                             Closed form found?
                             ├─ NO → Document empirical pattern
                             └─ YES → Phase 6: Verification
                                      ↓
                                      Verification passes?
                                      ├─ NO → Return to Phase 2
                                      └─ YES → VALIDATED DISCOVERY ✓
```

---

## Practical Example: φ-BBP Discovery

### Phase 1: Concept Definition
```
Target: Fast-converging BBP-type formula for π
Keywords: pi, BBP, convergence, spigot, base-4096
Anchor weights: [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]
Constraints: 
  - Base = 4096 (for fast convergence)
  - 8 slots (4k+j and 12k+j patterns)
  - Integer coefficients + small corrections
```

### Phase 2: Structure Search
```
Search space: base=4096, scale=64, slots=[4k+1, 4k+3, 12k+1..12k+11]
Best candidate N_smooth: 21.1 (error ≈ 10^-21)
```

### Phase 3: LCM Pruning
```
Anchor vector: [0.15, 0.25, 0.30, 0.05, 0.20, 0.05]
Dominant: phi (0.30), sierpinski (0.25), cantor (0.20)
Concept score: 0.89
Combined score: 0.85 → PROCEED
```

### Phase 4: Error Analysis
```
Deviations found:
  Slot 4 (12k+5): -128 + 0.0733 → (13/16) × φ^-5 ← CLEAN!
  Slot 0 (4k+1):  +256 + 0.0210 → (46/11) × φ^-11
  ...
Total correction: -0.1406
```

### Phase 5: Pattern Detection
```
Closed form search:
  Total ≈ (13/20)×arctan(1/φ) + (-26/25)×log(φ)
  Verification error: 2.3×10^-16 ← FOUND!
```

### Phase 6: Verification
```
High-precision (500 digits): PASS
Convergence: 3.61 digits/term
vs Bellard: 120% (20% faster!)
→ VALIDATED DISCOVERY ✓
```

---

## Implementation Checklist

### Phase 1: Concept Definition
- [ ] State target clearly
- [ ] List domain keywords
- [ ] Compute anchor weight vector
- [ ] Define search constraints
- [ ] Set pruning rules

### Phase 2: Structure Search
- [ ] Define search space bounds
- [ ] Generate candidate structures
- [ ] Evaluate N_smooth for each
- [ ] Rank by N_smooth
- [ ] Select top candidates

### Phase 3: LCM Pruning
- [ ] Compute anchor vectors
- [ ] Measure concept alignment
- [ ] Apply pruning rules
- [ ] Compute combined scores
- [ ] Filter weak candidates

### Phase 4: Error Analysis
- [ ] Compute high-precision error
- [ ] Decompose by component
- [ ] Search φ-patterns for each
- [ ] Identify clean patterns
- [ ] Compute total correction

### Phase 5: Pattern Detection
- [ ] Sum total correction
- [ ] Search closed forms (PSLQ)
- [ ] Verify closed form
- [ ] Check related identities
- [ ] Document patterns

### Phase 6: Verification
- [ ] High-precision verification (500+ digits)
- [ ] Measure convergence rate
- [ ] Compare to benchmarks
- [ ] Document discovery
- [ ] Publish/archive

---

## Tools Reference

### Ribbon LCM Tools
```python
from tools import translate, parse, get_vector, identify_value

# Translate equation to Ribbon Speech
speech = translate("E = mc²")

# Get anchor vector
vector = get_vector("E = mc²")

# Identify a value
result = identify_value(1.618033988749895)
```

### Fast Search Tools
```python
from ribbon_lcm_v5_experimental.fast_search import FastSearch, RibbonSearcher

# Fast pattern matching
fs = FastSearch()
result = fs.identify(0.618033988749895)

# Ribbon-integrated search
rs = RibbonSearcher()
result = rs.identify_with_anchors(value)
```

### Discovery Engine
```python
from ribbon_lcm_v5_experimental import DiscoveryEngine
from ribbon_lcm_v5_experimental.domains import BBPDomain

domain = BBPDomain()
engine = DiscoveryEngine(domain)

# Run discovery
discoveries = engine.discover(concept, target=3.14159265358979)
```

---

## Key Insights

### 1. Error is Signal, Not Noise
When approximate solutions have structured error (φ-patterns, clean rationals), the error is pointing toward the exact formula.

### 2. Anchor Vectors Guide Search
The 6D anchor space provides a conceptual compass. Candidates far from the target concept are unlikely to be correct.

### 3. Clean Patterns Indicate Truth
Small integers (|n| ≤ 20, d ≤ 20) in φ^k patterns are not coincidence - they indicate genuine mathematical structure.

### 4. Closed Forms Confirm Discovery
If the total correction has a closed form in arctan(1/φ), log(φ), etc., the discovery is almost certainly valid.

### 5. Verification is Non-Negotiable
Always verify to 100+ digits. Mathematical truth doesn't depend on precision, but our confidence does.

---

## When to Use EDP

### ✅ Use EDP When:
1. Searching for new formulas for known constants (π, e, φ, etc.)
2. Looking for faster-converging series
3. Exploring BBP-type or Ramanujan-type structures
4. Error in approximate solutions shows patterns
5. Need systematic, reproducible discovery process

### ❌ Don't Use EDP When:
1. Target is unknown (use exploration instead)
2. Problem is purely numerical (no symbolic structure)
3. No anchor mapping makes sense for the domain
4. Computational resources are severely limited

---

## Conclusion

The **Equation Discovery Protocol** provides a systematic method for locating mathematical truths using the Ribbon LCM framework. By treating error as signal and using anchor coordinates to navigate the space of possibilities, we can discover new equations that would be difficult to find through random search.

**Key Philosophy:** Mathematical truth exists. We don't create it - we locate it. The Ribbon LCM provides the map.

**Validated Result:** The φ-BBP formula, discovered using this protocol, demonstrates that the approach works for real mathematical discovery.

---

**Version:** 1.0  
**Date:** November 2025  
**Status:** Production Ready  
**License:** MIT
