#!/usr/bin/env python3
"""
Day 126 — T2 vs Log-Probability Correlation

QUESTION: Does T2 ranking correlate with model log-probability?
  P(candidate | prompt) is the "ground truth" of what the model would predict.
  If T2 similarity ranking ≈ log-prob ranking, then T2 is a geometric proxy
  for model probability — the geometric structure encodes predictive information.

This is a DIRECT TEST of the TruthSpace hypothesis:
  "The shape IS the knowledge — what the LM knows is encoded in its
  geometric structure."

If T2 correlates strongly with log-prob:
  → Geometric traversal (T2) can predict what the model would generate
  → The T2 trie is a compressed representation of the model's knowledge

If T2 does NOT correlate:
  → T2 captures semantic category but NOT predictive ordering
  → The trie is useful for classification but not for generation/ranking

METRICS:
  1. Spearman rank correlation: T2 rank vs log-prob rank (per prompt)
  2. Mean correlation across all 12 prompts
  3. Per-category correlation
  4. MRR of log-prob ranking (upper bound for any geometric method)
  5. Whether T2 top-1 == log-prob top-1 (agreement on best candidate)

Also test: does combining T2+d_k (Day 125's best: α=0.9) correlate
better with log-prob than T2 alone?
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day126_t2_logprob_correlation.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DAY78_LAYERS = {
    "gender": 27, "comparative": 15, "hypernym": 28, "plural": 1,
    "synonym": 28, "concrete": 28, "past_tense": 28, "antonym": 28,
    "passive": 28, "causation": 28, "question": 28, "negation": 28,
}
AXIS_NAMES_12 = [
    "gender", "comparative", "hypernym", "plural",
    "synonym", "concrete", "past_tense", "antonym",
    "passive", "causation", "question", "negation",
]
AXIS_SENTENCE_PAIRS = {
    "gender": [
        ("The king ruled with great wisdom","The queen ruled with great wisdom"),
        ("A man walked through the forest","A woman walked through the forest"),
        ("The boy kicked the ball hard","The girl kicked the ball hard"),
        ("His brother arrived at the party","His sister arrived at the party"),
        ("The father worked to feed family","The mother worked to feed family"),
        ("The actor played a leading role","The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
        ("The old house","The older house"),("A bright star","A brighter star"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The car sped past the sign","The vehicle sped past the sign"),
        ("The eagle soared above the hill","The bird soared above the hill"),
        ("The ruby gleamed in the light","The gem gleamed in the light"),
        ("The hammer struck the nail","The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in field","Dogs played happily in field"),
        ("The cat sat quietly by window","The cats sat quietly by window"),
        ("A bird sang softly in mist","Birds sang softly in mist"),
        ("The tree fell down in storm","The trees fell down in storm"),
        ("A book sat open on desk","Books sat open on desk"),
        ("The car drove slowly down road","The cars drove slowly down road"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
        ("She is happy","She is joyful"),("He is old","He is aged"),
    ],
    "concrete": [
        ("The stone is too heavy","The burden is too heavy"),
        ("The long road leads","The long journey leads"),
        ("The high wall blocks","The high barrier blocks"),
        ("The flame fades away","The hope fades away"),
        ("The root grips soil","The base grips earth"),
        ("The bridge connects","The bond connects"),
    ],
    "past_tense": [
        ("I walk every morning","I walked every morning"),
        ("She runs through park","She ran through park"),
        ("He eats before leaving","He ate before leaving"),
        ("They build the wall","They built the wall"),
        ("We swim in lake","We swam in lake"),
        ("She writes a letter","She wrote a letter"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The news is good","The news is bad"),("She is happy","She is sad"),
        ("He is strong","He is weak"),("It is first","It is last"),
    ],
    "passive": [
        ("The cat chased mouse","The mouse was chased"),
        ("John broke window","The window was broken"),
        ("The chef cooked meal","The meal was cooked"),
        ("The storm destroyed house","The house was destroyed"),
        ("The artist painted picture","The picture was painted"),
        ("The king signed document","The document was signed"),
    ],
    "causation": [
        ("The rain falls down","The ground gets wet"),
        ("The fire burns long","The wood turns to ash"),
        ("The sun heats earth","The ice starts melting"),
        ("The wind blows branches","The leaves start falling"),
        ("The child cries loud","The mother comes running"),
        ("The glass breaks","The water spills out"),
    ],
    "question": [
        ("She is tired today","Is she tired today"),
        ("He can swim well","Can he swim well"),
        ("They went to market","Did they go to market"),
        ("The dog is hungry","Is the dog hungry"),
        ("She wrote the letter","Did she write letter"),
        ("He knows the answer","Does he know answer"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know answer"),
        ("The food is good","The food is not good"),
        ("They work hard","They do not work hard"),
        ("The water is cold","The water is not cold"),
    ],
}

RANKING_CASES = [
    {"category":"capitals","prompt":"The capital city of France is",
     "correct":["Paris"],"plausible":["London","Rome","Berlin","Madrid"],
     "wrong":["banana","running","very","because"]},
    {"category":"capitals","prompt":"The capital city of Japan is",
     "correct":["Tokyo"],"plausible":["Osaka","Beijing","Seoul","Bangkok"],
     "wrong":["quietly","ocean","third","green"]},
    {"category":"capitals","prompt":"The capital city of Germany is",
     "correct":["Berlin"],"plausible":["Frankfurt","Vienna","Warsaw","Amsterdam"],
     "wrong":["walked","stone","never","that"]},
    {"category":"languages","prompt":"The official language of Brazil is",
     "correct":["Portuguese"],"plausible":["Spanish","Italian","French","English"],
     "wrong":["purple","running","before","cold"]},
    {"category":"languages","prompt":"The official language of Egypt is",
     "correct":["Arabic"],"plausible":["Hebrew","Turkish","Persian","Urdu"],
     "wrong":["quickly","stone","later","because"]},
    {"category":"hypernyms","prompt":"A poodle is a type of",
     "correct":["dog","animal","mammal"],"plausible":["cat","rabbit","horse","pet"],
     "wrong":["quickly","stone","because","red"]},
    {"category":"hypernyms","prompt":"A rose is a type of",
     "correct":["flower","plant","bloom"],"plausible":["tree","bush","grass","weed"],
     "wrong":["capital","walked","never","cold"]},
    {"category":"hypernyms","prompt":"A hammer is a type of",
     "correct":["tool","instrument","implement"],"plausible":["machine","device","appliance","weapon"],
     "wrong":["flower","language","quickly","green"]},
    {"category":"antonyms","prompt":"The opposite of hot is",
     "correct":["cold","cool","frigid"],"plausible":["warm","lukewarm","mild","chilly"],
     "wrong":["Paris","running","because","stone"]},
    {"category":"antonyms","prompt":"The opposite of happy is",
     "correct":["sad","unhappy","miserable"],"plausible":["angry","bored","worried","tired"],
     "wrong":["capital","walked","flower","green"]},
    {"category":"tense","prompt":"Yesterday he",
     "correct":["walked","ran","ate","built","saw"],"plausible":["walk","run","eat","build","see"],
     "wrong":["Paris","flower","because","cold"]},
    {"category":"gender","prompt":"The king and",
     "correct":["queen","princess","duchess","empress"],"plausible":["prince","knight","lord","earl"],
     "wrong":["banana","quickly","walked","stone"]},
]

L_dk, H_dk = 23, 6

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_layers    = model.config.num_hidden_layers
vocab_size  = model.config.vocab_size
print(f"  hidden={hidden_size}  vocab={vocab_size}\n")

print("Extracting d_k ...")
layer_module = model.model.layers[L_dk]
W_k_full     = layer_module.self_attn.k_proj.weight.data.cpu().numpy()
head_dim     = hidden_size // model.config.num_attention_heads
n_kv_heads   = model.config.num_key_value_heads
kv_per_group = model.config.num_attention_heads // n_kv_heads
kv_head_idx  = H_dk // kv_per_group
W_k_head     = W_k_full[kv_head_idx*head_dim:(kv_head_idx+1)*head_dim, :]
_, _, Vt_k   = np.linalg.svd(W_k_head, full_matrices=False)
d_k = Vt_k[0, :].astype(np.float32)
print(f"  d_k extracted  ||d_k||={np.linalg.norm(d_k):.4f}\n")

print("Computing T2 axes ...")
t2_axes = {}
for ax_name in AXIS_NAMES_12:
    L = DAY78_LAYERS[ax_name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(ax_name, []):
        try:
            inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
            with torch.no_grad():
                o1 = model(**inp1, output_hidden_states=True)
                o2 = model(**inp2, output_hidden_states=True)
            h1 = o1.hidden_states[L][0,-1,:].numpy().astype(np.float32)
            h2 = o2.hidden_states[L][0,-1,:].numpy().astype(np.float32)
            d = h2-h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d/n)
        except: pass
    v = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, np.float32)
    nv = np.linalg.norm(v)
    t2_axes[ax_name] = (v/nv if nv > 1e-6 else v).astype(np.float32)
print("  Done.\n")

def get_candidate_logprob(prompt, candidate):
    """P(candidate_token | prompt) — log probability of first token of candidate."""
    inp = tok(prompt, return_tensors="pt")
    try:
        with torch.no_grad():
            logits = model(**inp).logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1).numpy()
        # Get token id of candidate (first subword token)
        cand_ids = tok(" " + candidate, add_special_tokens=False)["input_ids"]
        if not cand_ids: return float("-inf")
        return float(log_probs[cand_ids[0]])
    except:
        return float("-inf")

def get_t2_iso_sim(candidate, ctx_t2):
    """T2 similarity of isolated candidate to context anchor."""
    inp = tok(" " + candidate, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        t2v = np.zeros(12, np.float32)
        for k, ax_name in enumerate(AXIS_NAMES_12):
            L = DAY78_LAYERS[ax_name]
            h = normed(out.hidden_states[L][0,-1,:].numpy().astype(np.float32))
            t2v[k] = float(np.dot(h, t2_axes[ax_name]))
        ctx_n = np.linalg.norm(ctx_t2); t2v_n = np.linalg.norm(t2v)
        if ctx_n < 1e-8 or t2v_n < 1e-8: return 0.0
        return float(np.dot(ctx_t2, t2v) / (ctx_n * t2v_n))
    except:
        return 0.0

def get_dk_score(prompt, candidate):
    """d_k projection at last token of (prompt + candidate)."""
    text = prompt + " " + candidate
    inp  = tok(text, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = normed(out.hidden_states[L_dk][0,-1,:].numpy().astype(np.float32))
        return float(abs(np.dot(h, d_k)))
    except:
        return 0.0

def get_context_t2(prompt):
    inp = tok(prompt, return_tensors="pt")
    try:
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        t2v = np.zeros(12, np.float32)
        for k, ax_name in enumerate(AXIS_NAMES_12):
            L = DAY78_LAYERS[ax_name]
            h = normed(out.hidden_states[L][0, pos, :].numpy().astype(np.float32))
            t2v[k] = float(np.dot(h, t2_axes[ax_name]))
        return t2v
    except:
        return np.zeros(12, np.float32)

def rank_by(scores_dict, higher_is_better=True):
    s = sorted(scores_dict.keys(), key=lambda w: (-scores_dict[w] if higher_is_better else scores_dict[w]))
    return {w: i+1 for i, w in enumerate(s)}

print("=" * 72)
print("Day 126: T2 vs Log-Probability Correlation")
print("=" * 72)
print()

all_results = []
spearman_t2_logprob   = []
spearman_dk_logprob   = []
spearman_combo_logprob = []

for case in RANKING_CASES:
    prompt    = case["prompt"]
    correct   = case["correct"]
    plausible = case["plausible"]
    wrong     = case["wrong"]
    all_cands = correct + plausible + wrong

    # Context anchor T2
    ctx_t2 = get_context_t2(prompt)

    # Compute three scores for each candidate
    logprobs  = {w: get_candidate_logprob(prompt, w) for w in all_cands}
    t2_sims   = {w: get_t2_iso_sim(w, ctx_t2) for w in all_cands}
    dk_scores = {w: get_dk_score(prompt, w) for w in all_cands}

    # Combined score (Day 125 optimal α=0.9)
    t2_vals  = np.array([t2_sims[w]  for w in all_cands])
    dk_vals  = np.array([dk_scores[w] for w in all_cands])
    t2_n  = (t2_vals - t2_vals.min()) / (t2_vals.max()-t2_vals.min() + 1e-8)
    dk_n  = (dk_vals  - dk_vals.min())  / (dk_vals.max()-dk_vals.min()  + 1e-8)
    combo = 0.9 * t2_n + 0.1 * dk_n
    combo_scores = {w: float(combo[i]) for i, w in enumerate(all_cands)}

    # Spearman rank correlation between each method and log-prob
    lp_vec    = np.array([logprobs[w]    for w in all_cands])
    t2_vec    = np.array([t2_sims[w]     for w in all_cands])
    dk_vec    = np.array([dk_scores[w]   for w in all_cands])
    co_vec    = np.array([combo_scores[w] for w in all_cands])

    rho_t2, pval_t2 = spearmanr(t2_vec, lp_vec)
    rho_dk, pval_dk = spearmanr(dk_vec, lp_vec)
    rho_co, pval_co = spearmanr(co_vec, lp_vec)

    spearman_t2_logprob.append(float(rho_t2))
    spearman_dk_logprob.append(float(rho_dk))
    spearman_combo_logprob.append(float(rho_co))

    # MRR for each method on THIS prompt
    def mrr_here(scores, cands=all_cands, cor=correct, higher=True):
        ranked = sorted(cands, key=lambda w: (-scores[w] if higher else scores[w]))
        r = next((i+1 for i,w in enumerate(ranked) if w in cor), len(cands)+1)
        return 1.0/r

    mrr_lp   = mrr_here(logprobs)
    mrr_t2   = mrr_here(t2_sims)
    mrr_dk   = mrr_here(dk_scores)
    mrr_co   = mrr_here(combo_scores)

    # Log-prob top-1 candidate
    top1_lp = max(all_cands, key=lambda w: logprobs[w])
    top1_t2 = max(all_cands, key=lambda w: t2_sims[w])
    agree   = bool(top1_lp == top1_t2 or top1_t2 in correct)

    print(f"  [{case['category']}] {prompt!r}")
    print(f"    log-prob  top1={top1_lp!r:>14}  MRR={mrr_lp:.3f}")
    print(f"    T2_iso    top1={top1_t2!r:>14}  MRR={mrr_t2:.3f}  rho(T2,lp)={rho_t2:+.3f}")
    print(f"    d_k                          MRR={mrr_dk:.3f}  rho(dk,lp)={rho_dk:+.3f}")
    print(f"    T2+dk                        MRR={mrr_co:.3f}  rho(co,lp)={rho_co:+.3f}")
    print()

    all_results.append({
        "category": case["category"], "prompt": prompt,
        "top1_lp": top1_lp, "top1_t2": top1_t2,
        "mrr_lp": mrr_lp, "mrr_t2": mrr_t2, "mrr_dk": mrr_dk, "mrr_co": mrr_co,
        "rho_t2_lp": float(rho_t2), "pval_t2": float(pval_t2),
        "rho_dk_lp": float(rho_dk), "pval_dk": float(pval_dk),
        "rho_co_lp": float(rho_co), "pval_co": float(pval_co),
        "logprobs": {w: float(logprobs[w]) for w in all_cands},
        "t2_sims":  {w: float(t2_sims[w])  for w in all_cands},
    })

# ── Aggregate ────────────────────────────────────────────────────────────────
print("=" * 72)
print("Aggregate Spearman Correlation (T2/d_k/combo vs log-prob)")
print("=" * 72)

mean_rho_t2    = float(np.mean(spearman_t2_logprob))
mean_rho_dk    = float(np.mean(spearman_dk_logprob))
mean_rho_combo = float(np.mean(spearman_combo_logprob))

mrr_lp_all   = [r["mrr_lp"]  for r in all_results]
mrr_t2_all   = [r["mrr_t2"]  for r in all_results]
mrr_dk_all   = [r["mrr_dk"]  for r in all_results]
mrr_co_all   = [r["mrr_co"]  for r in all_results]
n_agree_top1 = sum(1 for r in all_results if r["top1_t2"] == r["top1_lp"])

print(f"""
  Method         Mean Spearman ρ    MRR
  ────────────────────────────────────────────
  log-prob        1.000 (oracle)   {float(np.mean(mrr_lp_all)):.4f}
  T2 (isolated)  {mean_rho_t2:+.4f}             {float(np.mean(mrr_t2_all)):.4f}
  d_k            {mean_rho_dk:+.4f}             {float(np.mean(mrr_dk_all)):.4f}
  T2+d_k (α=0.9) {mean_rho_combo:+.4f}            {float(np.mean(mrr_co_all)):.4f}
  Random baseline  0.000             0.3143

  log-prob MRR (upper bound for this candidate set): {float(np.mean(mrr_lp_all)):.4f}
  T2 top-1 agrees with log-prob top-1: {n_agree_top1}/{len(all_results)}
""")

print("Per-category Spearman ρ (T2 vs log-prob):")
print(f"  {'category':>12}  {'rho_mean':>10}  {'mrr_lp':>8}  {'mrr_t2':>8}  {'gap':>8}")
print(f"  {'-'*52}")
cats_seen = []
for r in all_results:
    if r["category"] not in cats_seen: cats_seen.append(r["category"])
for cat in cats_seen:
    cr = [r for r in all_results if r["category"] == cat]
    avg_rho = float(np.mean([r["rho_t2_lp"] for r in cr]))
    avg_lp  = float(np.mean([r["mrr_lp"]   for r in cr]))
    avg_t2  = float(np.mean([r["mrr_t2"]   for r in cr]))
    print(f"  {cat:>12}  {avg_rho:>10.4f}  {avg_lp:>8.4f}  {avg_t2:>8.4f}  "
          f"{avg_t2-avg_lp:>+8.4f}")

# ── Per-candidate log-prob profiles ─────────────────────────────────────────
print()
print("=" * 72)
print("Log-prob profiles — what does the model actually assign?")
print("=" * 72)
print()
for case, r in zip(RANKING_CASES, all_results):
    prompt    = case["prompt"]
    correct   = case["correct"]
    plausible = case["plausible"]
    wrong     = case["wrong"]
    lp = r["logprobs"]
    print(f"  {prompt!r}")
    sorted_cands = sorted(lp.keys(), key=lambda w: -lp[w])
    for i, w in enumerate(sorted_cands):
        tier = "C" if w in correct else ("P" if w in plausible else "W")
        print(f"    #{i+1:2d} [{tier}] {w:>20}  lp={lp[w]:+.3f}")
    print()

# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 126 Summary — T2 vs Log-Probability")
print("=" * 72)

strong_corr  = mean_rho_t2 > 0.5
moderate_corr = mean_rho_t2 > 0.3
lp_is_upper  = float(np.mean(mrr_lp_all)) > float(np.mean(mrr_t2_all))

print(f"""
  Mean Spearman ρ(T2, log-prob) = {mean_rho_t2:+.4f}
  Mean Spearman ρ(d_k, log-prob) = {mean_rho_dk:+.4f}
  Mean Spearman ρ(T2+d_k, log-prob) = {mean_rho_combo:+.4f}

  log-prob MRR (oracle):  {float(np.mean(mrr_lp_all)):.4f}
  T2+d_k MRR:             {float(np.mean(mrr_co_all)):.4f}
  MRR gap (T2 vs oracle): {float(np.mean(mrr_co_all)) - float(np.mean(mrr_lp_all)):+.4f}

  VERDICT:
  T2 correlation with log-prob: {'STRONG (ρ>0.5)' if strong_corr else 'MODERATE (ρ>0.3)' if moderate_corr else 'WEAK (ρ<0.3)'}

  TruthSpace hypothesis:
  {'→ SUPPORTED: T2 geometric ranking correlates with model probability. The geometric structure encodes predictive information.' if moderate_corr else
   '→ PARTIALLY SUPPORTED: T2 captures category membership but not fine-grained probability ordering.' if mean_rho_t2 > 0.1 else
   '→ NOT SUPPORTED: T2 ranking does not correlate with model probability. Geometric structure and probability are separate.'}

  Key insight: log-prob MRR = {float(np.mean(mrr_lp_all)):.4f} (oracle upper bound).
  T2+d_k captures {100*float(np.mean(mrr_co_all))/float(np.mean(mrr_lp_all)):.1f}% of the oracle performance.
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "all_results": all_results,
        "mean_rho_t2": mean_rho_t2,
        "mean_rho_dk": mean_rho_dk,
        "mean_rho_combo": mean_rho_combo,
        "mrr_logprob": float(np.mean(mrr_lp_all)),
        "mrr_t2": float(np.mean(mrr_t2_all)),
        "mrr_dk": float(np.mean(mrr_dk_all)),
        "mrr_combo": float(np.mean(mrr_co_all)),
        "n_agree_top1": n_agree_top1,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 126 complete.")
