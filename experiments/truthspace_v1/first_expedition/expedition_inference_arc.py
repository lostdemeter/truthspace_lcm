#!/usr/bin/env python3
"""
Day 248 — Does the W_E Arc Structure Get USED During Inference?

TruthSpace hypothesis: the intelligence is in the SHAPE of the weights.
The arc structures in W_E are not decorative — they are the MECHANISM
by which the transformer computes morphological relations.

If true: when processing "big : bigger :: fast : ___", the model should
TRAVERSE the arc from emb(fast) to emb(faster) internally, and the
hidden states should reflect this traversal.

Specific questions:
  A. Does the model CORRECTLY complete adj_degree analogies?
  B. At which layer does the last-token hidden state "point toward" the answer?
  C. Does the hidden state trajectory from layer 0 to layer N trace a
     path from emb(C) toward emb(D), following the arc direction?
  D. Do the attention heads attend specifically to the A and B tokens?
  E. If we PERTURB the query (replace C with a semantically different word),
     does the hidden state move to a different point on the same arc?
  F. Does the hidden state at the last position lie IN the private degree
     plane of the query word C?

Prompts:
  "big is to bigger as fast is to"       → "faster"
  "small is to smaller as tall is to"    → "taller"
  "hot is to hotter as cold is to"       → "colder"
  "long is to longer as short is to"     → "shorter"
  "old is to older as young is to"       → "younger"

Control (non-adj-degree):
  "France is to Paris as Germany is to"  → "Berlin" (capital)
  "cat is to cats as dog is to"          → "dogs" (plural)
  "walk is to walked as talk is to"      → "talked" (past_tense)
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "inference_arc.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

PHI = (1 + np.sqrt(5)) / 2

ANALOGY_PROMPTS = {
    # ADJ_DEGREE analogies
    "fast→faster(big)": ("big is to bigger as fast is to",   "faster", "fast"),
    "tall→taller(small)": ("small is to smaller as tall is to", "taller", "tall"),
    "cold→colder(hot)":  ("hot is to hotter as cold is to",   "colder", "cold"),
    "short→shorter(long)": ("long is to longer as short is to", "shorter", "short"),
    "young→younger(old)": ("old is to older as young is to",  "younger", "old"),
    "darker→darker(bright)": ("bright is to brighter as dark is to", "darker", "dark"),
    # CONTROL paradigms
    "Germany→Berlin(France)": ("France is to Paris as Germany is to", "Berlin", "Germany"),
    "dog→dogs(cat)": ("cat is to cats as dog is to",          "dogs", "dog"),
    "talk→talked(walk)": ("walk is to walked as talk is to",   "talked", "talk"),
    "woman→women(man)": ("man is to men as woman is to",       "women", "woman"),
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cos_sim(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
V, H = W_E.shape
N_LAYERS = len(model.model.layers)
print(f"  V={V}, H={H}, N_LAYERS={N_LAYERS}\n")
Wn = np.array([normed(W_E[i]) for i in range(V)], dtype=np.float32)
W_LM = model.lm_head.weight.detach().numpy().astype(np.float64)

def tid1(w):
    for pref in [" ", ""]:
        ids = tok(pref + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1: return ids[0]
    return None

def get_emb(w):
    t = tid1(w)
    return W_E[t].copy() if t is not None else None

def top_tokens(logits_vec, k=5):
    idx = np.argsort(logits_vec)[-k:][::-1]
    return [(tok.decode([i]).strip(), float(logits_vec[i])) for i in idx]

# ── Forward pass capturing all hidden states ─────────────────────────
def run_forward(text):
    """Returns (predicted_token, hidden_states[layer] at last position)."""
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    logits = out.logits[0, -1, :].float().numpy()  # last token logits
    hs     = [h[0, -1, :].float().numpy()           # hidden state at last pos
              for h in out.hidden_states]            # list of L+1 (incl. embedding)
    pred_tid = int(np.argmax(logits))
    pred_word = tok.decode([pred_tid]).strip()
    return pred_word, logits, hs

# ── Part A: Model analogy completion ─────────────────────────────────
print("=" * 70)
print("PART A: MODEL ANALOGY COMPLETION")
print("        Does the model correctly complete adj_degree analogies?")
print("=" * 70)
print()

results = {}
for name, (prompt, answer_word, query_word) in ANALOGY_PROMPTS.items():
    pred_word, logits, hs = run_forward(prompt)
    t_ans = tid1(answer_word)
    ans_logit = float(logits[t_ans]) if t_ans is not None else None
    top5 = top_tokens(logits, k=10)
    # Check if answer appears in top-10 (handles space-prefix tokens)
    top10_words = [w.lower() for w, _ in top5]
    in_top10 = (answer_word.lower() in top10_words)
    correct_top1 = (pred_word.lower().strip() == answer_word.lower())
    correct = in_top10
    ans_rank = next((i for i, (w, _) in enumerate(top5)
                     if w.lower() == answer_word.lower()), None)
    print(f"  {name:<28}  pred={pred_word:<12}  expected={answer_word:<12}  "
          f"rank={ans_rank if ans_rank is not None else 'miss':>4}  "
          f"{'✓' if correct else '✗'}")
    results[name] = {
        "prompt": prompt, "answer": answer_word, "pred": pred_word,
        "correct": correct, "top5": top5[:5], "ans_rank": ans_rank
    }

# ── Part B: Layer-by-layer hidden state cosine with answer ───────────
print()
print("=" * 70)
print("PART B: HIDDEN STATE cos(h_layer, emb(answer))")
print("        At which layer does the hidden state 'point toward' the answer?")
print("=" * 70)
print()

# Show progression for adj_degree and one control
selected = [
    ("fast→faster(big)", True),
    ("cold→colder(hot)", True),
    ("Germany→Berlin(France)", False),
    ("dog→dogs(cat)", False),
]
for name, is_adj in selected:
    prompt, answer_word, query_word = ANALOGY_PROMPTS[name]
    pred_word, logits, hs = run_forward(prompt)
    t_ans = tid1(answer_word)
    if t_ans is None: continue
    emb_ans = W_E[t_ans]
    emb_query = get_emb(query_word)

    # LM head normalization
    try:
        rms_scale = model.model.norm.weight.detach().numpy().astype(np.float64)
    except:
        rms_scale = np.ones(H)

    print(f"  {name}  (paradigm={'adj_degree' if is_adj else 'control'})")
    print(f"  {'Layer':<7}  {'cos(h,emb_ans)':>16}  {'cos(h,emb_query)':>18}  "
          f"{'nn1_word':>15}")
    hs_cos_ans   = []
    hs_cos_query = []
    for layer_idx, h in enumerate(hs):
        c_ans   = cos_sim(h, emb_ans)
        c_query = cos_sim(h, emb_query) if emb_query is not None else float('nan')
        hs_cos_ans.append(c_ans); hs_cos_query.append(c_query)
        # Only print every 4 layers + last
        if layer_idx % 4 == 0 or layer_idx == len(hs) - 1:
            # NN retrieval from this hidden state
            hn = normed(h).astype(np.float32)
            sims = Wn @ hn
            nn_tok = int(np.argmax(sims))
            nn_word = tok.decode([nn_tok]).strip()
            print(f"  L{layer_idx:<5}  {c_ans:>16.4f}  {c_query:>18.4f}  {nn_word:>15}")
    print(f"  L0→L{len(hs)-1}: cos_ans {hs_cos_ans[0]:.4f} → {hs_cos_ans[-1]:.4f}  "
          f"(Δ = {hs_cos_ans[-1]-hs_cos_ans[0]:+.4f})")
    print()

# ── Part C: Hidden state arc geometry ────────────────────────────────
print()
print("=" * 70)
print("PART C: ARC GEOMETRY OF HIDDEN STATES")
print("        Does h_last lie on the arc from emb(C) to emb(D)?")
print("=" * 70)
print()

# For adj_degree analogies: compute where h_last falls relative to the arc
print(f"  {'name':<28}  {'cos(h,C)':>9}  {'cos(h,D)':>9}  "
      f"{'arc_pos':>9}  {'correct':>8}")
for name, is_adj in selected:
    prompt, answer_word, query_word = ANALOGY_PROMPTS[name]
    _, _, hs = run_forward(prompt)
    h_last = hs[-1]  # last layer hidden state at last position
    emb_C = get_emb(query_word)
    t_D = tid1(answer_word)
    if emb_C is None or t_D is None: continue
    emb_D = W_E[t_D]

    c_C = cos_sim(h_last, emb_C)
    c_D = cos_sim(h_last, emb_D)
    # Arc position: if h_last is on the arc, where does it sit?
    # Parameterize: alpha=0 → emb_C, alpha=1 → emb_D
    # For simple interpolation on the arc: how much of the arc has been traversed?
    # As a simple proxy: arc_pos = cos(h,emb_C) vs cos(h,emb_D)
    arc_pos = c_D / (c_C + c_D + 1e-8)  # rough relative position
    correct = results[name]["correct"]
    print(f"  {name:<28}  {c_C:>9.4f}  {c_D:>9.4f}  {arc_pos:>9.4f}  "
          f"{'✓' if correct else '✗'}")

# ── Part D: Top-5 logit analysis for analogy prompts ────────────────
print()
print("=" * 70)
print("PART D: TOP-5 LOGITS FOR EACH ANALOGY")
print("        Where does the correct answer rank in the logit distribution?")
print("=" * 70)
print()

for name, (prompt, answer_word, query_word) in ANALOGY_PROMPTS.items():
    top5 = results[name]["top5"]
    ans_rank = results[name]["ans_rank"]
    top5_str = ", ".join(f"{w}({l:.2f})" for w, l in top5)
    print(f"  {name:<28}  ans={answer_word:<10}  rank={ans_rank}")
    print(f"    top-5: {top5_str}")

# ── Part E: Hidden state distance to arc ─────────────────────────────
print()
print("=" * 70)
print("PART E: DOES h_LAST LIE IN THE PRIVATE DEGREE PLANE OF QUERY C?")
print("        Project h_last onto the plane spanned by {emb(C), emb(D)}")
print("=" * 70)
print()

for name in ["fast→faster(big)", "cold→colder(hot)", "Germany→Berlin(France)"]:
    prompt, answer_word, query_word = ANALOGY_PROMPTS[name]
    _, _, hs = run_forward(prompt)
    h_last = hs[-1]
    emb_C = get_emb(query_word)
    t_D = tid1(answer_word)
    if emb_C is None or t_D is None: continue
    emb_D = W_E[t_D]

    # 2D basis for the plane of (emb_C, emb_D)
    e1 = normed(emb_C)
    e2 = normed(emb_D - float(np.dot(normed(emb_D), e1)) * e1)
    # Project h_last onto this plane
    c1 = float(np.dot(h_last, e1))
    c2 = float(np.dot(h_last, e2))
    h_in_plane = c1 * e1 + c2 * e2
    h_perp = h_last - h_in_plane
    frac_in_plane = float(np.linalg.norm(h_in_plane)**2 /
                          (np.linalg.norm(h_last)**2 + 1e-8))
    print(f"  {name:<28}  frac_in_plane={frac_in_plane:.4f}  "
          f"c1={c1:.4f}  c2={c2:.4f}  "
          f"||h_perp||={np.linalg.norm(h_perp):.4f}")

print()
print("  Note: frac_in_plane = fraction of ||h||² explained by the C-D plane.")
print("  Small value → h_last is NOT in the C-D plane (full 1536D component).")
print("  This is expected: the hidden state has H=1536 dimensions, and the")
print("  C-D plane is only 2D (< 0.0001% of total volume).")
print("  The relevant question is: which DIRECTION has the most weight?")

# ── Summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
total_correct = sum(1 for n in results if results[n]["correct"])
print(f"  Overall analogy completion (answer in top-10): {total_correct}/{len(results)}")
adj_names = [n for n in ANALOGY_PROMPTS
             if ANALOGY_PROMPTS[n][1] in
             ["faster","taller","colder","shorter","younger","darker"]]
adj_correct = sum(1 for n in adj_names if results[n]["correct"])
print(f"  adj_degree analogies: {adj_correct}/{len(adj_names)}")
print()
print(f"  Per-example answer ranks:")
for name in ANALOGY_PROMPTS:
    rank = results[name]["ans_rank"]
    print(f"    {name:<28}  rank={'✓ '+str(rank) if rank is not None else 'miss'}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "top5"}
               for k, v in results.items()}, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Inference arc investigation complete.")
