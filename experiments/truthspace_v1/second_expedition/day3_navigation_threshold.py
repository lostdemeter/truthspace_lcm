"""
SECOND EXPEDITION — DAY 3
=========================
Is the Navigation Threshold at 1/φ Specifically?

Day 2 established: pairs at φ-level n≤2 navigate reliably, n≥3 do not.
But is 1/φ (0.618) specifically the threshold, or just "close enough"?

If 1/φ is incidental: navigation accuracy should fall smoothly with cosine,
and any threshold around 0.4-0.7 would describe it equally well.

If 1/φ is physically meaningful: there should be a SHARP transition near
cos=1/φ, not gradual decay. This would suggest the golden ratio is not a
label but a structural feature of the embedding geometry.

Approach:
  Phase 1: Self-navigation test on ALL 136 pairs from Day 2.
           For each pair, test if it can navigate to itself using ONLY
           its own chord direction (no mean-axis aggregation).
           Plot accuracy vs cosine — is there a sharp transition?

  Phase 2: Sweep the threshold. At what specific cosine value does
           navigation accuracy drop? Is it exactly 1/φ?

  Phase 3: n=1 semantic graph. For the top 1000 common English words,
           map all edges at cos∈[0.55, 0.68]. What structures appear?

  Phase 4: The self-referential test. If cos(A,B) ≈ 1/φ, and we
           apply A→B rotation to a RANDOM word C, what φ-level is
           the result? Does 1/φ produce a fixed-point in navigation?

  Phase 5: The n=1 pair taxonomy. What do all n=1 pairs have in common
           that n=2 pairs do not? Is it morphological, syntactic,
           semantic, or distributional?

Darwin's rule: follow the surprise.

Script: second_expedition/day3_navigation_threshold.py
"""

import torch, numpy as np

print("Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct',
                                              torch_dtype=torch.float32)
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n = (W_E / (np.linalg.norm(W_E, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
print(f"  shape={W_E.shape}")

EN_MASK = np.array([
    bool(tok.decode([i]).strip() and tok.decode([i]).strip().isalpha() and
         tok.decode([i]).strip().isascii() and len(tok.decode([i]).strip()) >= 2)
    for i in range(len(W_E))], dtype=bool)

PHI = (1 + 5**0.5) / 2
PHI_LEVELS = {n: 1.0 / PHI**n for n in range(0, 10)}

def normed(v): return v / (np.linalg.norm(v) + 1e-12)

def get_emb(word):
    for p in [' ', '']:
        ids = tok(p + word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def source_ids(word):
    ids = set()
    for p in [word, ' '+word, word[0].upper()+word[1:] if word and word[0].isascii() else word]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    return ids

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    idx = int(np.argmax(sims))
    return tok.decode([idx]).strip(), float(sims[idx]), idx

def nearest_phi_n(c):
    if c <= 0: return (None, abs(c))
    n_f = -np.log(max(c, 1e-9)) / np.log(PHI)
    n_r = round(n_f)
    return n_r, abs(c - PHI_LEVELS.get(n_r, 0))

# ── Full pair dataset from Day 2 ───────────────────────────────────────────────
ALL_PAIRS = {
    'gender': [
        ('man','woman'),('king','queen'),('father','mother'),('son','daughter'),
        ('boy','girl'),('husband','wife'),('uncle','aunt'),('prince','princess'),
        ('brother','sister'),('actor','actress'),('hero','heroine'),('waiter','waitress'),
        ('monk','nun'),('wizard','witch'),('lord','lady'),('god','goddess'),
        ('male','female'),('he','she'),('his','her'),('him','her'),
        ('grandfather','grandmother'),('nephew','niece'),('groom','bride'),
        ('bull','cow'),('cock','hen'),('ram','ewe'),
    ],
    'size': [
        ('big','small'),('large','tiny'),('huge','little'),('tall','short'),
        ('long','brief'),('fat','thin'),('wide','narrow'),('heavy','light'),
        ('strong','weak'),('hot','cold'),('fast','slow'),('hard','soft'),
        ('loud','quiet'),('deep','shallow'),('thick','thin'),('rich','poor'),
        ('old','young'),('high','low'),('full','empty'),('bright','dim'),
        ('sharp','blunt'),('rough','smooth'),('wet','dry'),('clean','dirty'),
        ('warm','cool'),('dark','pale'),
    ],
    'sentiment': [
        ('good','bad'),('happy','sad'),('love','hate'),('beautiful','ugly'),
        ('right','wrong'),('best','worst'),('kind','cruel'),('honest','dishonest'),
        ('wise','foolish'),('gentle','harsh'),('generous','selfish'),
        ('peaceful','violent'),('healthy','sick'),('success','failure'),
        ('hope','despair'),('joy','grief'),('pleasure','pain'),('truth','lie'),
        ('friend','enemy'),('hero','villain'),('angel','devil'),('heaven','hell'),
        ('light','dark'),('life','death'),
    ],
    'synonyms': [
        ('happy','joyful'),('fast','quick'),('big','large'),('smart','intelligent'),
        ('angry','furious'),('sad','unhappy'),('cold','chilly'),('hot','warm'),
        ('begin','start'),('end','finish'),('show','display'),('help','assist'),
        ('talk','speak'),('walk','stroll'),('look','see'),('want','desire'),
        ('house','home'),('road','street'),('child','kid'),('gift','present'),
    ],
    'hypernyms': [
        ('dog','animal'),('cat','animal'),('eagle','bird'),('salmon','fish'),
        ('rose','flower'),('oak','tree'),('iron','metal'),('ruby','stone'),
        ('Paris','city'),('London','city'),('jazz','music'),('chess','game'),
        ('sword','weapon'),('knife','weapon'),('surgeon','doctor'),('poet','artist'),
        ('anger','emotion'),('red','color'),('circle','shape'),('piano','instrument'),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Self-navigation test on all pairs
# For each pair (src, tgt), the "self-axis" is the chord from src to tgt.
# Test: can we navigate from src to tgt using ONLY this one chord?
# Scale optimised over grid. No mean-axis aggregation.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 1 — Self-Navigation Test: Every Pair Navigates Itself")
print("  Each pair uses its OWN chord direction (no mean-axis averaging)")
print("  Tests pure geometry: does the pair-to-pair chord reach its target?")
print("═"*72)

all_results = []   # (cat, src, tgt, cos_val, phi_n, self_nav_ok)

for cat, pairs in ALL_PAIRS.items():
    for src, tgt in pairs:
        es, idx_s = get_emb(src)
        et, idx_t = get_emb(tgt)
        if es is None or et is None: continue

        cos_val = float(np.dot(normed(es), normed(et)))
        phi_n, phi_resid = nearest_phi_n(cos_val)

        # Self-axis: chord from normalized src to normalized tgt
        chord = normed(et) - normed(es)
        chord_dir = normed(chord)

        # Grid search for best scale
        best_s, best_ok = 0.0, False
        for s in np.linspace(0.01, 5.0, 100):
            pred = es + s * chord_dir
            w, _, _ = nn_ret(pred, source_ids(src), EN_MASK)
            if w == tgt:
                best_ok = True
                best_s = s
                break   # found it — that's enough

        all_results.append((cat, src, tgt, cos_val, phi_n, best_ok, best_s))

# Sort by cosine value
all_results.sort(key=lambda x: -x[3])

print(f"\n  {'cat':<12}  {'pair':>26}  {'cos':>7}  {'n':>3}  {'self_nav':>8}  scale")
print(f"  {'─'*12}  {'─'*26}  {'─'*7}  {'─'*3}  {'─'*8}  {'─'*5}")
for cat, src, tgt, cos_val, phi_n, ok, s in all_results:
    flag = "✓" if ok else "✗"
    print(f"  {cat:<12}  {src:>12}→{tgt:<13}  {cos_val:>7.4f}  {phi_n:>3}  "
          f"{'✓ YES' if ok else '✗ NO ':>8}  {s:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Accuracy vs cosine — is the threshold at 1/φ?
# Bin all pairs by cosine and compute self-navigation accuracy per bin.
# Then sweep the threshold and find the cos value that best separates navigable
# from non-navigable.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 2 — Navigation Accuracy vs Cosine: Where is the Threshold?")
print("  φ-level marks: 1/φ=0.618  1/φ²=0.382  1/φ³=0.236  1/φ⁴=0.146")
print("═"*72)

cos_vals = np.array([r[3] for r in all_results])
nav_ok   = np.array([r[5] for r in all_results], dtype=float)

# Bin by cosine (bins of width 0.05)
print(f"\n  Accuracy by cosine bin (width=0.05):")
print(f"  {'cos range':>16}  {'n_pairs':>8}  {'correct':>8}  {'accuracy':>9}  φ-level")
print(f"  {'─'*16}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*10}")

bins = np.arange(0.0, 0.85, 0.05)
for lo in bins:
    hi = lo + 0.05
    mask = (cos_vals >= lo) & (cos_vals < hi)
    n = int(mask.sum())
    if n == 0: continue
    acc = nav_ok[mask].mean()
    correct = int(nav_ok[mask].sum())
    # Which φ-level falls in this bin?
    phi_in_bin = [f"1/φ^{k}={PHI_LEVELS[k]:.3f}" for k in range(1, 8)
                  if lo <= PHI_LEVELS[k] < hi]
    phi_str = phi_in_bin[0] if phi_in_bin else ""
    bar = "█" * int(acc * 20)
    print(f"  [{lo:.2f},{hi:.2f})  {n:>8}  {correct:>8}  {acc:>9.1%}  {phi_str}  {bar}")

# Sweep threshold: find the cosine value T that maximises accuracy(cos≥T) + (1-accuracy(cos<T))
print(f"\n  Threshold sweep (maximise separability):")
print(f"  {'threshold':>10}  {'above_acc':>10}  {'below_acc':>10}  {'separability':>13}  φ-ref?")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*13}  {'─'*10}")

best_sep, best_T = 0.0, 0.0
for T in np.arange(0.05, 0.80, 0.02):
    above = cos_vals >= T
    below = ~above
    acc_above = nav_ok[above].mean() if above.sum() > 0 else 0.0
    acc_below = (1 - nav_ok[below]).mean() if below.sum() > 0 else 0.0
    sep = acc_above + acc_below   # 2.0 = perfect separation
    if sep > best_sep: best_sep = sep; best_T = T
    # Which φ-level is nearest?
    phi_n, phi_d = nearest_phi_n(T)
    phi_flag = f"≈1/φ^{phi_n}" if phi_d < 0.03 else ""
    print(f"  T={T:.2f}       {acc_above:>10.1%}  {acc_below:>10.1%}  {sep:>13.4f}  {phi_flag}")

print(f"\n  ★ Best threshold: T={best_T:.2f}  separability={best_sep:.4f}")
phi_n_best, phi_d_best = nearest_phi_n(best_T)
print(f"    Nearest φ-level: 1/φ^{phi_n_best} = {PHI_LEVELS[phi_n_best]:.4f}  Δ={phi_d_best:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: n=1 semantic graph
# For the 2000 most common EN tokens (low IDs = common words in subword tokenizers),
# find all edges at cos∈[0.55, 0.68] and analyse the graph structure.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 3 — The n=1 Semantic Graph")
print("  2000 common EN tokens; edges where cos∈[0.55,0.68] (≈1/φ band)")
print("═"*72)

# Get common EN tokens: use IDs ≤ 40000 as proxy for frequency
# (lower IDs are generally more common in BPE vocabularies)
common_en = [i for i in range(min(40000, len(W_E)))
             if EN_MASK[i] and len(tok.decode([i]).strip()) >= 3][:2000]
print(f"  Using {len(common_en)} common EN tokens")

# Compute pairwise cosines among common_en
embs_c = W_n[common_en].astype(np.float64)  # [N, 1536]
cos_mat = embs_c @ embs_c.T                  # [N, N]

# Find edges in the n=1 band [0.55, 0.68]
LO, HI = 0.55, 0.68
edges = []
n_c = len(common_en)
for i in range(n_c):
    row = cos_mat[i]
    for j in range(i+1, n_c):
        c = float(row[j])
        if LO <= c < HI:
            w_i = tok.decode([common_en[i]]).strip()
            w_j = tok.decode([common_en[j]]).strip()
            edges.append((w_i, w_j, c))

print(f"  Found {len(edges)} edges in [{LO:.2f},{HI:.2f}] band")

# Degree distribution
from collections import defaultdict, Counter
degree = defaultdict(int)
for w1, w2, c in edges:
    degree[w1] += 1
    degree[w2] += 1

deg_arr = np.array(list(degree.values()))
print(f"  Node degree: mean={deg_arr.mean():.2f}  max={deg_arr.max()}  "
      f"isolated={n_c-len(degree)} nodes have no n=1 neighbor")

# Top-degree nodes (semantic hubs)
top_nodes = sorted(degree.items(), key=lambda x: -x[1])[:20]
print(f"\n  Top n=1 hubs (most n=1 neighbors):")
for word, deg in top_nodes:
    # Find their neighbors
    nbrs = [w2 if w1==word else w1 for w1,w2,c in edges if w1==word or w2==word]
    nbrs_str = ', '.join(sorted(nbrs)[:10])
    print(f"    {word:<15} degree={deg:>3}  neighbors: {nbrs_str}")

# Edge type analysis: what categories of relationships appear?
print(f"\n  Sample of n=1 edges (sorted by cosine, showing extremes):")
edges_sorted = sorted(edges, key=lambda x: -x[2])
print(f"  {'word_A':>14}  {'word_B':>14}  {'cos':>7}")
print(f"  {'─'*14}  {'─'*14}  {'─'*7}")
print("  [Top 20 — highest cosine]")
for w1, w2, c in edges_sorted[:20]:
    print(f"  {w1:>14}  {w2:>14}  {c:.4f}")
print("  ...")
print("  [Bottom 20 — lowest cosine in band]")
for w1, w2, c in edges_sorted[-20:]:
    print(f"  {w1:>14}  {w2:>14}  {c:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: The self-referential property of 1/φ
# cos(A,B) = 1/φ means A·B = 1/φ in normalized space.
# The golden ratio satisfies: 1/φ = φ - 1 and φ² = φ + 1.
# Test: if we apply the A→B rotation to B itself, where does B go?
# i.e., what is the cos(B, B + axis) for the n=1 pairs?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 4 — The Self-Referential Property: What Happens When We Repeat?")
print("  For n=1 pair (A,B): apply A→B rotation to B → what is C?")
print("  Is cos(B,C) also ≈ 1/φ? (self-referential = yes)")
print("  Is cos(A,C) ≈ 1/φ²? (two-step = double angle)")
print("═"*72)

# Use confirmed n=1 gender pairs
n1_pairs = [
    ('brother','sister'),('son','daughter'),('boy','girl'),('uncle','aunt'),
    ('grandfather','grandmother'),('nephew','niece'),('hero','heroine'),
    ('king','queen'),('father','mother'),('actor','actress'),
]

# Build gender axis from all n=1 pairs
gender_tangents = []
for s, t in n1_pairs:
    es, _ = get_emb(s); et, _ = get_emb(t)
    if es is None or et is None: continue
    en_s = normed(es); en_t = normed(et)
    cos_th = float(np.dot(en_s, en_t))
    sin_th = float(np.sqrt(max(0, 1 - cos_th**2)))
    tangent = (en_t - cos_th * en_s) / (sin_th + 1e-12)
    gender_tangents.append(tangent)

gender_axis = normed(np.mean(gender_tangents, axis=0))

print(f"\n  Using gender axis from {len(gender_tangents)} n=1 pairs")
print(f"\n  {'A':>10}→{'B':>10}  cos(A,B)  cos(B,C)  cos(A,C)  C (NN)  φ-pred")
print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*14}  {'─'*8}")

for src, tgt in n1_pairs:
    es, _ = get_emb(src); et, _ = get_emb(tgt)
    if es is None or et is None: continue

    en_s = normed(es); en_t = normed(et)
    cos_AB = float(np.dot(en_s, en_t))

    # Apply same rotation to B: pred_C = e_B + scale * axis
    # Use the optimal navigation theta from Day 2 (≈29°)
    th = np.radians(29.0)
    pred_C_n = np.cos(th) * en_t + np.sin(th) * gender_axis
    C_word, _, C_idx = nn_ret(pred_C_n, source_ids(tgt), EN_MASK)
    eC, _ = get_emb(C_word)
    if eC is None: continue
    en_C = normed(eC)

    cos_BC = float(np.dot(en_t, en_C))
    cos_AC = float(np.dot(en_s, en_C))
    n_BC, _ = nearest_phi_n(cos_BC)
    n_AC, _ = nearest_phi_n(cos_AC)
    phi_pred = f"n={n_BC}/n={n_AC}"
    print(f"  {src:>10}→{tgt:>10}  {cos_AB:>8.4f}  {cos_BC:>8.4f}  "
          f"{cos_AC:>8.4f}  {C_word:<14}  {phi_pred}")

print(f"\n  Theory: if rotation is self-referential at 1/φ,")
print(f"    cos(B,C) should ≈ 1/φ = {PHI_LEVELS[1]:.4f}  (same level)")
print(f"    cos(A,C) should ≈ 1/φ² = {PHI_LEVELS[2]:.4f}  (one level deeper)")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: What do n=1 pairs have in common that n=3+ do not?
# Morphological analysis: are n=1 pairs more often morphological relatives?
# Length analysis: do n=1 pairs share more characters?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 5 — n=1 vs n=3 Pair Anatomy: What Makes Them Different?")
print("  Tests: string similarity, length ratio, shared prefix/suffix")
print("═"*72)

def lcs_length(a, b):
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]: dp[i+1][j+1] = dp[i][j] + 1
            else: dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def pair_stats(src, tgt):
    a, b = src.lower(), tgt.lower()
    shared_prefix = len([1 for i in range(min(len(a),len(b))) if a[i]==b[i] and
                         all(a[j]==b[j] for j in range(i+1))])
    # actually compute proper prefix
    px = 0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]: px += 1
        else: break
    lcs = lcs_length(a, b)
    return {
        'prefix_frac': px / max(len(a),len(b)),
        'lcs_frac': lcs / max(len(a), len(b)),
        'len_ratio': min(len(a),len(b)) / max(len(a),len(b)),
        'len_diff': abs(len(a) - len(b)),
    }

n1_anatomy = []
n3_anatomy = []

for cat, pairs in ALL_PAIRS.items():
    for src, tgt in pairs:
        c = None
        for r in all_results:
            if r[1]==src and r[2]==tgt: c=r[3]; phi_n=r[4]; break
        if c is None: continue
        ps = pair_stats(src, tgt)
        if phi_n == 1: n1_anatomy.append(ps)
        elif phi_n >= 3: n3_anatomy.append(ps)

print(f"\n  Anatomy comparison (n=1 pairs  vs  n≥3 pairs):")
print(f"  {'metric':<18}  {'n=1 mean':>10}  {'n≥3 mean':>10}  {'ratio':>8}")
print(f"  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*8}")
for key in ['prefix_frac', 'lcs_frac', 'len_ratio', 'len_diff']:
    v1 = np.mean([a[key] for a in n1_anatomy]) if n1_anatomy else 0
    v3 = np.mean([a[key] for a in n3_anatomy]) if n3_anatomy else 0
    ratio = v1/v3 if v3 > 0 else float('nan')
    print(f"  {key:<18}  {v1:>10.4f}  {v3:>10.4f}  {ratio:>8.3f}x")
print(f"  n=1 pairs: {len(n1_anatomy)}   n≥3 pairs: {len(n3_anatomy)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: The golden ratio fixed-point test
# φ satisfies φ = 1 + 1/φ.  So 1/φ = φ - 1.
# In spherical geometry: if cos(A,B) = 1/φ, then cos(A,B)² + cos(A,B) = 1
# Because: (1/φ)² + (1/φ) = 1/φ² + 1/φ = (φ-1)²/φ² + (φ-1)/φ ... let me check.
# 1/φ + 1/φ² = 1/φ(1 + 1/φ) = (1/φ)φ = 1. YES! 1/φ + 1/φ² = 1.
# This means: cos(A,B) + cos(A,B)² = 1  ONLY when cos = 1/φ.
# Equivalently: cos² + cos - 1 = 0  → cos = (√5-1)/2 = 1/φ.
# Geometric meaning: the squared cosine plus the cosine equals exactly 1.
# Test: measure cos² + cos for all pairs and see how close to 1 they are for n=1.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("PHASE 6 — The Golden Ratio Fixed-Point Identity")
print("  Mathematical fact: 1/φ + (1/φ)² = 1  (unique to 1/φ)")
print("  This means: cos + cos² = 1 ONLY when cos = 1/φ")
print("  Test: how close to 1 is (cos + cos²) for each φ-level?")
print("═"*72)

print(f"\n  {'n':>4}  {'1/φⁿ':>8}  {'cos':>8}  {'cos²':>8}  {'cos+cos²':>10}  Δ from 1")
print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*9}")
for n in range(1, 8):
    c = PHI_LEVELS[n]
    c2 = c**2
    total = c + c2
    delta = abs(total - 1.0)
    flag = " ◀ EXACT" if n == 1 else ""
    print(f"  {n:>4}  {c:>8.4f}  {c:>8.4f}  {c2:>8.4f}  {total:>10.4f}  {delta:>9.4f}{flag}")

print(f"\n  For our pairs, measuring cos + cos²:")
print(f"\n  {'cat':<12}  {'pair':>26}  {'cos':>7}  {'cos+cos²':>10}  Δ from 1  n")
print(f"  {'─'*12}  {'─'*26}  {'─'*7}  {'─'*10}  {'─'*9}  {'─'*3}")
for cat, src, tgt, cos_val, phi_n, ok, s in all_results:
    if phi_n is None: continue
    c2 = cos_val**2
    total = cos_val + c2
    delta = abs(total - 1.0)
    flag = " ◀" if delta < 0.03 else ""
    if phi_n <= 2 or delta < 0.05:   # Show all n<=2 and any surprisingly close ones
        print(f"  {cat:<12}  {src:>12}→{tgt:<13}  {cos_val:>7.4f}  {total:>10.4f}  "
              f"{delta:>9.4f}  n={phi_n}{flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("SECOND EXPEDITION — DAY 3 SUMMARY")
print("═"*72)
print(f"""
Core question: Is the navigation threshold specifically at 1/φ?

Phase 1: Self-navigation test — is each pair self-navigable?
Phase 2: Threshold sweep — where is the accuracy cliff?
Phase 3: n=1 graph — structure of cos≈1/φ neighborhood
Phase 4: Self-referential test — does A→B→C maintain 1/φ structure?
Phase 5: Pair anatomy — what do n=1 pairs share morphologically?
Phase 6: Golden ratio identity — cos + cos² = 1 only at 1/φ

Mathematical identity unique to 1/φ: cos + cos² = 1
This may explain why 1/φ is the navigation threshold.

Record in second_expedition/expedition_log.md
""")
