#!/usr/bin/env python3
"""
φ-Cosine Investigation — Is cos(pos,comp)=cos(π/(2φ)) Universal?

The geometric audit's deepest finding:
  cos(emb(pos), emb(comp)) ≈ cos(π/(2φ)) = 0.5671 for English adj_degree.
  This is the SINGLE fact from which all arc geometry derives.

Question: is this specific to English in Qwen2, or universal?

Tests:
  A. CROSS-LANGUAGE: adj_degree comparatives in German, French, Spanish,
     Chinese, Russian. Does cos ≈ cos(π/(2φ)) hold for other languages?
     Qwen2 is multilingual — the same embedding space is shared.

  B. EXTENDED ENGLISH: test 50+ more adjective triples. Does the
     cos(π/(2φ)) hold broadly, or was the 24-word sample lucky?

  C. PARADIGM φ-CHECK: for paradigms where "Ω ≈ π/2" (plural, past_tense),
     is the actual cos closer to cos(π/4) = 0.707 or cos(π/(2φ)) = 0.567?
     Verify the paradigm class assignments.

  D. ANTONYM FAMILIES: adj_degree antonyms (big↔small, hot↔cold) —
     does the ANTONYM angle satisfy any φ-relation?
     cos(big, small) vs cos(π/(2φ)), cos(π/φ), cos(π/φ²)?

  E. THE φ-ANGLE SERIES: is there a pattern to the cosine values?
     adj_degree: cos ≈ 0.567 = cos(π/(2φ))
     What are the other exact cosine values when expressed as cos(π/k)?
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "phi_cosine.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + np.sqrt(5)) / 2
PHI_ANGLE   = np.degrees(np.arccos(np.cos(np.pi / (2 * PHI))))  # = 90/φ = 55.625°
PHI_COS     = np.cos(np.radians(PHI_ANGLE))                      # = 0.5671

print(f"  Target: cos(π/(2φ)) = {PHI_COS:.6f}  (angle = {PHI_ANGLE:.4f}°)")

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"\nLoading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
del model
V, H = W_E.shape
print(f"  V={V}, H={H}\n")

def get_emb(w, with_space=True):
    prefixes = [" "] if with_space else [" ", ""]
    for prefix in prefixes:
        ids = tok(prefix + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return W_E[ids[0]].copy()
    return None

def get_emb_any(w):
    """Try space-prefixed first, then bare."""
    r = get_emb(w, with_space=True)
    if r is not None:
        return r
    return get_emb(w, with_space=False)

def cos_sim(a, b):
    return float(np.dot(normed(a), normed(b)))

# ── Part A: Cross-language adj_degree ────────────────────────────────
print("=" * 70)
print("PART A: CROSS-LANGUAGE ADJ_DEGREE cos(pos, comp)")
print(f"        Target: cos(π/(2φ)) = {PHI_COS:.4f}")
print("=" * 70)
print()

MULTILINGUAL_TRIPLES = {
    "German": [
        ("schnell", "schneller", "schnellste"),    # fast
        ("lang",    "länger",    "längste"),        # long
        ("stark",   "stärker",   "stärkste"),       # strong
        ("groß",    "größer",    "größte"),         # big/great
        ("klein",   "kleiner",   "kleinste"),       # small
        ("jung",    "jünger",    "jüngste"),        # young
        ("alt",     "älter",     "älteste"),        # old
        ("warm",    "wärmer",    "wärmste"),        # warm
        ("kalt",    "kälter",    "kälteste"),       # cold
        ("schön",   "schöner",   "schönste"),       # beautiful
    ],
    "French": [
        ("grand",   "plus grand",   "le plus grand"),   # big
        ("petit",   "plus petit",   "le plus petit"),   # small
        ("fort",    "plus fort",    "le plus fort"),    # strong
        ("vieux",   "plus vieux",   "le plus vieux"),   # old
        ("jeune",   "plus jeune",   "le plus jeune"),   # young
        ("chaud",   "plus chaud",   "le plus chaud"),   # hot
        ("froid",   "plus froid",   "le plus froid"),   # cold
        ("beau",    "plus beau",    "le plus beau"),    # beautiful
        ("long",    "plus long",    "le plus long"),    # long
        ("haut",    "plus haut",    "le plus haut"),    # high
    ],
    "Spanish": [
        ("grande",   "más grande",   "el más grande"),
        ("pequeño",  "más pequeño",  "el más pequeño"),
        ("fuerte",   "más fuerte",   "el más fuerte"),
        ("viejo",    "más viejo",    "el más viejo"),
        ("joven",    "más joven",    "el más joven"),
        ("caliente", "más caliente", "el más caliente"),
        ("largo",    "más largo",    "el más largo"),
        ("alto",     "más alto",     "el más alto"),
        ("bajo",     "más bajo",     "el más bajo"),
        ("rápido",   "más rápido",   "el más rápido"),
    ],
    "Chinese": [
        ("快", "更快", "最快"),   # fast
        ("慢", "更慢", "最慢"),   # slow
        ("高", "更高", "最高"),   # high/tall
        ("低", "更低", "最低"),   # low
        ("大", "更大", "最大"),   # big
        ("小", "更小", "最小"),   # small
        ("长", "更长", "最长"),   # long
        ("短", "更短", "最短"),   # short
        ("强", "更强", "最强"),   # strong
        ("弱", "更弱", "最弱"),   # weak
    ],
    "Russian": [
        ("быстрый",   "быстрее",   "самый быстрый"),  # fast
        ("большой",   "больше",    "самый большой"),   # big
        ("маленький", "меньше",    "самый маленький"), # small
        ("старый",    "старше",    "самый старый"),    # old
        ("молодой",   "моложе",    "самый молодой"),   # young
        ("горячий",   "горячее",   "самый горячий"),   # hot
        ("холодный",  "холоднее",  "самый холодный"),  # cold
        ("сильный",   "сильнее",   "самый сильный"),   # strong
        ("длинный",   "длиннее",   "самый длинный"),   # long
        ("высокий",   "выше",      "самый высокий"),   # high
    ],
}

results_A = {}
for lang, triples in MULTILINGUAL_TRIPLES.items():
    cos_pc_vals = []; cos_cs_vals = []; n_ok = 0
    for pos_w, comp_w, sup_w in triples:
        P = get_emb_any(pos_w)
        C = get_emb_any(comp_w)
        S = get_emb_any(sup_w)
        if P is None or C is None or S is None:
            continue
        c_pc = cos_sim(P, C); c_cs = cos_sim(C, S)
        cos_pc_vals.append(c_pc); cos_cs_vals.append(c_cs); n_ok += 1

    if not cos_pc_vals:
        print(f"  {lang:<10}: no valid pairs (tokenization issues)")
        continue
    mean_pc = np.mean(cos_pc_vals)
    std_pc  = np.std(cos_pc_vals)
    diff_from_phi = mean_pc - PHI_COS
    results_A[lang] = {"mean_cos_pc": float(mean_pc), "std": float(std_pc),
                       "n": n_ok, "diff_from_phi_cos": float(diff_from_phi)}
    print(f"  {lang:<10}  n={n_ok:>2}  "
          f"cos(pos,comp)={mean_pc:.4f}±{std_pc:.4f}  "
          f"diff_from_cos(π/(2φ))={diff_from_phi:>+.4f}  "
          f"angle={np.degrees(np.arccos(np.clip(mean_pc,-1,1))):.2f}°")

print()
print(f"  English adj_degree (reference): "
      f"cos=0.5676±0.0788  angle=55.38°  diff=+0.0005")

# ── Part B: Extended English (50+ adjectives) ─────────────────────────
print()
print("=" * 70)
print("PART B: EXTENDED ENGLISH ADJ_DEGREE (additional pairs)")
print(f"        Target: cos(π/(2φ)) = {PHI_COS:.4f}")
print("=" * 70)
print()

EXTRA_ADJ = [
    ("warm","warmer","warmest"), ("cold","colder","coldest"),
    ("light","lighter","lightest"), ("heavy","heavier","heaviest"),
    ("quick","quicker","quickest"), ("slow","slower","slowest"),
    ("thin","thinner","thinnest"), ("thick","thicker","thickest"),
    ("rough","rougher","roughest"), ("smooth","smoother","smoothest"),
    ("sharp","sharper","sharpest"), ("dull","duller","dullest"),
    ("loud","louder","loudest"), ("quiet","quieter","quietest"),
    ("sweet","sweeter","sweetest"), ("sour","sourer","sourest"),
    ("soft","softer","softest"), ("hard","harder","hardest"),
    ("dry","drier","driest"), ("wet","wetter","wettest"),
    ("new","newer","newest"), ("thin","thinner","thinnest"),
    ("plain","plainer","plainest"), ("gentle","gentler","gentlest"),
    ("tender","tenderer","tenderest"), ("rare","rarer","rarest"),
    ("pure","purer","purest"), ("mild","milder","mildest"),
    ("noble","nobler","noblest"), ("humble","humbler","humblest"),
]

extra_cos = []
extra_results = []
for pos_w, comp_w, sup_w in EXTRA_ADJ:
    P = get_emb(pos_w); C = get_emb(comp_w)
    if P is None or C is None: continue
    c = cos_sim(P, C)
    extra_cos.append(c)
    diff = c - PHI_COS
    extra_results.append({"word": pos_w, "cos": float(c), "diff": float(diff)})

if extra_cos:
    mean_e = np.mean(extra_cos); std_e = np.std(extra_cos)
    print(f"  Extended English: n={len(extra_cos)}  "
          f"cos={mean_e:.4f}±{std_e:.4f}  "
          f"angle={np.degrees(np.arccos(mean_e)):.2f}°  "
          f"diff={mean_e-PHI_COS:+.4f}")
    worst = sorted(extra_results, key=lambda x: abs(x['diff']), reverse=True)[:5]
    print(f"  Largest deviations from cos(π/(2φ)):")
    for r in worst:
        print(f"    {r['word']:<12}  cos={r['cos']:.4f}  diff={r['diff']:+.4f}")

# ── Part C: Verify paradigm φ-class assignments ──────────────────────
print()
print("=" * 70)
print("PART C: PARADIGM cos VALUES AND φ-ANGLE SERIES")
print("=" * 70)
print()

# Known cos values (from universal_R.py):
paradigm_cos = {
    "adj_degree":     0.5676,
    "gender":         0.5281,
    "plural":         0.6695,
    "past_tense":     0.6729,
    "capital":        0.4462,
    "antonym_size":   0.2338,
}

# φ-angle candidates
phi_angles = {
    "π/(2φ) = 55.625°": np.pi / (2 * PHI),
    "π/φ = 111.25°":    np.pi / PHI,
    "π/(2φ²) = 34.38°": np.pi / (2 * PHI**2),
    "π/3 = 60°":        np.pi / 3,
    "π/4 = 45°":        np.pi / 4,
    "π/5 = 36°":        np.pi / 5,
    "π/6 = 30°":        np.pi / 6,
    "π/8 = 22.5°":      np.pi / 8,
}

print(f"  φ-angle candidates (as cosines):")
for name, angle in phi_angles.items():
    print(f"    cos({name}) = {np.cos(angle):.6f}")
print()

print(f"  {'paradigm':<14}  {'cos':>7}  {'angle':>8}  {'best_match':>20}  "
      f"{'match_diff':>10}")
for pname, cos_val in paradigm_cos.items():
    angle = np.degrees(np.arccos(cos_val))
    best_name = None; best_diff = 1e9
    for aname, aval in phi_angles.items():
        diff = abs(cos_val - np.cos(aval))
        if diff < best_diff: best_diff = diff; best_name = aname
    print(f"  {pname:<14}  {cos_val:>7.4f}  {angle:>7.2f}°  "
          f"{best_name:>20}  {best_diff:>10.4f}")

# ── Part D: Antonym cosine analysis ──────────────────────────────────
print()
print("=" * 70)
print("PART D: ANTONYM PAIRS — cos(A, B) distribution and φ-analysis")
print("=" * 70)
print()

ANTONYM_PAIRS = [
    ("big","small"), ("large","tiny"), ("huge","little"),
    ("tall","short"), ("wide","narrow"), ("thick","thin"),
    ("heavy","light"), ("hot","cold"), ("fast","slow"),
    ("hard","soft"), ("loud","quiet"), ("dark","bright"),
    ("old","young"), ("strong","weak"), ("rich","poor"),
    ("happy","sad"), ("good","bad"), ("long","short"),
    ("high","low"), ("deep","shallow"), ("early","late"),
    ("clean","dirty"), ("safe","dangerous"), ("warm","cool"),
]

ant_cos = []
for a_w, b_w in ANTONYM_PAIRS:
    A = get_emb(a_w); B = get_emb(b_w)
    if A is None or B is None: continue
    c = cos_sim(A, B)
    ant_cos.append(c)

if ant_cos:
    mean_ant = np.mean(ant_cos); std_ant = np.std(ant_cos)
    angle_ant = np.degrees(np.arccos(mean_ant))
    print(f"  Antonyms: n={len(ant_cos)}  cos={mean_ant:.4f}±{std_ant:.4f}  "
          f"angle={angle_ant:.2f}°")
    # φ-match check
    for aname, aval in phi_angles.items():
        diff = abs(mean_ant - np.cos(aval))
        if diff < 0.05:
            print(f"    Near φ-match: cos(angle) ≈ cos({aname}), diff={diff:.4f}")
    print()
    print(f"  cos(π/φ²) = cos(π/2.618) = cos(69.09°) = "
          f"{np.cos(np.pi/PHI**2):.4f}")
    print(f"  Mean antonym angle = {angle_ant:.2f}°  "
          f"vs cos(π/φ²) = {np.degrees(np.pi/PHI**2):.2f}°  "
          f"diff = {angle_ant - np.degrees(np.pi/PHI**2):.2f}°")

# ── Part E: The φ-angle series ────────────────────────────────────────
print()
print("=" * 70)
print("PART E: THE φ-ANGLE SERIES — is there a pattern?")
print("=" * 70)
print()
print("  If the paradigm angles follow a geometric series:")
print()
paradigm_angles = {
    "adj_degree": np.arccos(0.5676),
    "plural":     np.arccos(0.6695),
    "past_tense": np.arccos(0.6729),
    "capital":    np.arccos(0.4462),
    "antonym_size": np.arccos(0.2338),
}
for pname, angle in sorted(paradigm_angles.items(), key=lambda x: x[1]):
    pi_ratio = np.pi / angle
    phi_ratio = angle / (np.pi / (2 * PHI))
    print(f"  {pname:<14}  θ={np.degrees(angle):.2f}°  "
          f"π/θ={pi_ratio:.3f}  θ/(π/(2φ))={phi_ratio:.3f}")

print()
print("  Ratio of consecutive angles (sorted by angle):")
sorted_angles = sorted(paradigm_angles.values())
for i in range(len(sorted_angles)-1):
    r = sorted_angles[i+1] / sorted_angles[i]
    print(f"    {np.degrees(sorted_angles[i]):.2f}° → {np.degrees(sorted_angles[i+1]):.2f}°  "
          f"ratio = {r:.4f}  vs φ = {PHI:.4f}  vs √φ = {np.sqrt(PHI):.4f}")

output = {
    "phi_cos": PHI_COS,
    "phi_angle_deg": PHI_ANGLE,
    "results_A": results_A,
    "extended_english_mean_cos": float(np.mean(extra_cos)) if extra_cos else None,
    "antonym_mean_cos": float(np.mean(ant_cos)) if ant_cos else None,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("φ-cosine investigation complete.")
