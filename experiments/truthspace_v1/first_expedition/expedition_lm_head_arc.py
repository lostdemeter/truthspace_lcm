#!/usr/bin/env python3
"""
Day 249 — The LM Head and the Arc Endpoint

Day 248 found: cos(h_28, emb_D) ≈ 0.13-0.15 for all paradigms.
The NN of raw h_28 gives fill tokens. But RMSNorm(h_28) correctly
predicts the answer (rank=0 for "taller").

This script asks: is RMSNorm the operation that "extracts" the arc
endpoint from h_28? If yes, cos(RMSNorm(h_28), emb_D) >> cos(h_28, emb_D).

Parts:
  A. cos(h_28 raw, emb_D) vs cos(RMSNorm(h_28), emb_D)
     For successful completions and failures.

  B. The mean_dir probe on h_28:
     cos(RMSNorm(h_28), mean_dir_pc) — does h_28 align with the
     adj_degree transformation direction?
     Compare to control paradigms.

  C. Arc endpoint projection:
     For successful completions, does RMSNorm(h_28) point more toward
     emb_D than toward any other token in the vocab?
     (i.e., verify rank_D in the logit distribution via direct computation)

  D. Layer where arc endpoint first dominates:
     At which layer does cos(RMSNorm(h_l), emb_D) exceed 0.3?
     Compare to adj_degree vs control.

  E. Mean_dir in residual stream:
     For a successful analogy, decompose h_28 = c_D * emb_D + residual.
     How large is c_D relative to ||h_28||?
     This measures "how much of h_28 IS the answer embedding".
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "lm_head_arc.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

ANALOGY_PROMPTS = {
    "tall→taller(small)":    ("small is to smaller as tall is to",   "taller", "tall"),
    "short→shorter(long)":   ("long is to longer as short is to",    "shorter", "short"),
    "fast→faster(big)":      ("big is to bigger as fast is to",      "faster", "fast"),
    "cold→colder(hot)":      ("hot is to hotter as cold is to",      "colder", "cold"),
    "young→younger(old)":    ("old is to older as young is to",      "younger", "young"),
    "dark→darker(bright)":   ("bright is to brighter as dark is to", "darker", "dark"),
    "Germany→Berlin(France)":("France is to Paris as Germany is to", "Berlin", "Germany"),
    "dog→dogs(cat)":         ("cat is to cats as dog is to",         "dogs", "dog"),
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cos_sim(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
V, H = W_E.shape
N_LAYERS = len(model.model.layers)
print(f"  V={V}, H={H}, N_LAYERS={N_LAYERS}")

# Get the final RMSNorm weights
rms_scale = model.model.norm.weight.detach().numpy().astype(np.float64)
print(f"  RMSNorm scale: ||s||={np.linalg.norm(rms_scale):.4f}  "
      f"mean={rms_scale.mean():.4f}  std={rms_scale.std():.4f}\n")

Wn = np.array([normed(W_E[i]) for i in range(V)], dtype=np.float32)

def tid1(w):
    for pref in [" ", ""]:
        ids = tok(pref + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1: return ids[0]
    return None

def get_emb(w):
    t = tid1(w)
    return W_E[t].copy() if t is not None else None

def rms_norm(v, scale):
    """RMS normalization: v / rms(v) * scale"""
    rms = np.sqrt(np.mean(v**2) + 1e-6)
    return (v / rms) * scale

def run_forward(text):
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    logits = out.logits[0, -1, :].float().numpy()
    hs = [h[0, -1, :].float().numpy() for h in out.hidden_states]
    return logits, hs

# Load adj_degree mean_dir
ADJ_TRIPLES = [
    ("big","bigger"), ("fast","faster"), ("long","longer"), ("small","smaller"),
    ("hard","harder"), ("bright","brighter"), ("dark","darker"), ("rich","richer"),
    ("deep","deeper"), ("wide","wider"), ("high","higher"), ("low","lower"),
    ("old","older"), ("young","younger"), ("hot","hotter"), ("tall","taller"),
    ("strong","stronger"), ("weak","weaker"), ("short","shorter"), ("cool","cooler"),
    ("great","greater"), ("safe","safer"), ("cheap","cheaper"), ("clean","cleaner"),
]
diffs = []
for p, c in ADJ_TRIPLES:
    P = get_emb(p); C = get_emb(c)
    if P is not None and C is not None: diffs.append(C - P)
mean_dir_pc = np.mean(diffs, axis=0)
mean_dir_pc_n = normed(mean_dir_pc)
print(f"  Loaded {len(diffs)} adj_degree pairs for mean_dir\n")

# ── Part A: raw h_28 vs RMSNorm(h_28) alignment with emb_D ───────────
print("=" * 70)
print("PART A: cos(h_28 raw, emb_D) vs cos(RMSNorm(h_28), emb_D)")
print("        Is RMSNorm the operation that extracts the arc endpoint?")
print("=" * 70)
print()

print(f"  {'name':<28}  {'cos_raw':>8}  {'cos_norm':>9}  "
      f"{'ratio':>7}  {'rank_D':>7}  {'correct'}")
all_results = {}
for name, (prompt, answer_word, query_word) in ANALOGY_PROMPTS.items():
    logits, hs = run_forward(prompt)
    h28 = hs[-1]
    h28_norm = rms_norm(h28, rms_scale)

    t_D = tid1(answer_word)
    if t_D is None:
        print(f"  {name:<28}  MULTI-TOKEN ANSWER")
        continue
    emb_D = W_E[t_D]

    cos_raw  = cos_sim(h28, emb_D)
    cos_norm = cos_sim(h28_norm, emb_D)
    ratio    = cos_norm / (abs(cos_raw) + 1e-8)

    # Rank of answer in logits
    rank_D = int(np.sum(logits >= logits[t_D]))

    correct_mark = "✓" if rank_D == 0 else ("~" if rank_D <= 10 else "✗")
    print(f"  {name:<28}  {cos_raw:>8.4f}  {cos_norm:>9.4f}  "
          f"{ratio:>7.2f}  {rank_D:>7}  {correct_mark}")
    all_results[name] = {
        "cos_raw": float(cos_raw), "cos_norm": float(cos_norm),
        "rank_D": rank_D
    }

# ── Part B: mean_dir probe on h_28 vs control ─────────────────────────
print()
print("=" * 70)
print("PART B: cos(RMSNorm(h_28), mean_dir_pc)")
print("        Does h_28 align with the adj_degree transformation direction?")
print("=" * 70)
print()

mean_dir_n = normed(mean_dir_pc).astype(np.float64)
print(f"  {'name':<28}  {'cos_raw_dir':>12}  {'cos_norm_dir':>13}  "
      f"{'paradigm':>10}")
for name, (prompt, answer_word, query_word) in ANALOGY_PROMPTS.items():
    logits, hs = run_forward(prompt)
    h28 = hs[-1]
    h28_norm = rms_norm(h28, rms_scale)
    paradigm = "adj_degree" if name not in ["Germany→Berlin(France)", "dog→dogs(cat)"] else "control"
    c_raw  = cos_sim(h28, mean_dir_n)
    c_norm = cos_sim(h28_norm, mean_dir_n)
    print(f"  {name:<28}  {c_raw:>12.4f}  {c_norm:>13.4f}  {paradigm:>10}")

# ── Part C: Layer where arc endpoint first dominates ─────────────────
print()
print("=" * 70)
print("PART C: LAYER WHERE cos(RMSNorm(h_l), emb_D) FIRST EXCEEDS 0.3")
print("        Both adj_degree and control — any difference?")
print("=" * 70)
print()

THRESHOLD = 0.30
for name in ["tall→taller(small)", "short→shorter(long)",
             "Germany→Berlin(France)", "dog→dogs(cat)"]:
    prompt, answer_word, query_word = ANALOGY_PROMPTS[name]
    t_D = tid1(answer_word)
    if t_D is None: continue
    emb_D = W_E[t_D]

    logits, hs = run_forward(prompt)
    first_threshold_layer = None
    cos_by_layer = []
    for l, h in enumerate(hs):
        h_n = rms_norm(h, rms_scale)
        c = cos_sim(h_n, emb_D)
        cos_by_layer.append(c)
        if c >= THRESHOLD and first_threshold_layer is None:
            first_threshold_layer = l
    t_D_rank = int(np.sum(logits >= logits[t_D]))
    print(f"  {name:<28}  first_L≥{THRESHOLD:.2f}: {first_threshold_layer}  "
          f"final_cos={cos_by_layer[-1]:.4f}  rank={t_D_rank}")
    print(f"    cos progression: "
          + " ".join(f"L{l}={cos_by_layer[l]:.3f}" for l in [0,4,8,12,16,20,24,28]))

# ── Part D: h_28 decomposition onto emb_D ────────────────────────────
print()
print("=" * 70)
print("PART D: DECOMPOSE h_28 ONTO emb_D")
print("        h_28 = c_D * normed(emb_D) + residual")
print("        How much of h_28 IS the arc endpoint?")
print("=" * 70)
print()

print(f"  {'name':<28}  {'c_D':>8}  {'||h||':>7}  "
      f"{'frac_D':>8}  {'rank_D':>7}")
for name, (prompt, answer_word, query_word) in ANALOGY_PROMPTS.items():
    t_D = tid1(answer_word)
    if t_D is None: continue
    emb_D = W_E[t_D]
    emb_D_n = normed(emb_D)

    logits, hs = run_forward(prompt)
    h28 = hs[-1]
    h28_norm = rms_norm(h28, rms_scale)

    # Project h28 onto emb_D direction
    c_D = float(np.dot(h28, emb_D_n))
    h_norm_val = float(np.linalg.norm(h28))
    frac_D = c_D**2 / (h_norm_val**2 + 1e-8)

    t_D_rank = int(np.sum(logits >= logits[t_D]))
    print(f"  {name:<28}  {c_D:>8.4f}  {h_norm_val:>7.2f}  "
          f"{frac_D:>8.6f}  {t_D_rank:>7}")

# ── Part E: What are the top-1 W_E NN and LM head NN ─────────────────
print()
print("=" * 70)
print("PART E: nn(h_28 raw) vs nn(RMSNorm(h_28)) = LM head prediction")
print("=" * 70)
print()

print(f"  {'name':<28}  {'nn(raw)':>12}  {'nn(norm)':>12}  {'lm_head':>12}")
for name, (prompt, answer_word, query_word) in ANALOGY_PROMPTS.items():
    logits, hs = run_forward(prompt)
    h28 = hs[-1]
    h28_norm = rms_norm(h28, rms_scale)

    # NN from raw h28
    h28_f = normed(h28).astype(np.float32)
    sims_raw = Wn @ h28_f
    nn_raw = tok.decode([int(np.argmax(sims_raw))]).strip()

    # NN from normed h28 (should match LM head)
    h28_norm_f = normed(h28_norm).astype(np.float32)
    sims_norm = Wn @ h28_norm_f
    nn_norm = tok.decode([int(np.argmax(sims_norm))]).strip()

    # LM head top-1
    lm_top = tok.decode([int(np.argmax(logits))]).strip()
    print(f"  {name:<28}  {nn_raw:>12}  {nn_norm:>12}  {lm_top:>12}")

print()
print("  Note: nn(raw) ≠ lm_head because LM head uses RMSNorm(h_28),")
print("  not cosine-of-raw. Expect nn(norm) ≈ lm_head if W_E == W_LM.")

# ── Summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
cos_raw_list  = [all_results[n]["cos_raw"]  for n in all_results]
cos_norm_list = [all_results[n]["cos_norm"] for n in all_results]
print(f"  Mean cos(h_28 raw,  emb_D) = {np.mean(cos_raw_list):.4f}")
print(f"  Mean cos(RMSNorm,   emb_D) = {np.mean(cos_norm_list):.4f}")
print(f"  Ratio norm/raw = {np.mean(cos_norm_list)/np.mean(cos_raw_list):.2f}x")
print()
print(f"  If RMSNorm amplifies the answer direction by ~{np.mean(cos_norm_list)/np.mean(cos_raw_list):.1f}x,")
print(f"  then the final layer norm IS the arc-endpoint extraction operation.")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("LM head arc analysis complete.")
