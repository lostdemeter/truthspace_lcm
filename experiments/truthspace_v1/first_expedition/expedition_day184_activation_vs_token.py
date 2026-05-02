#!/usr/bin/env python3
"""
Day 184 — Activation Space vs Token Space

W_E is the STATIC token embedding space (layer 0).
The ACTIVATION space is the residual stream at layer L after processing a prompt.

QUESTION: Do the same directional encoding rules apply in activation space?
Can W_E directions (extracted from W_E) be applied to activation vectors?

THREE EXPERIMENTS:

  A. W_E direction applied to W_E query:
     Classical W_E experiment (baseline from Days 162+)
     direction = mean(W_E[capital] - W_E[country])
     query = W_E[France] + direction → snap → Paris?

  B. Activation direction applied to activation query:
     direction = mean(activation[capital] - activation[country])
     query = activation[France] + direction → snap in activation space → Paris?

  C. W_E direction applied to activation query (cross-space):
     direction = mean(W_E[capital] - W_E[country])
     query = activation[France] + direction → snap in activation space?

For retrieval in activation space, we compute activations for all target
candidates and find the nearest neighbor.

PROMPT FORMAT: single-word prompt " {word}" (same as token ID extraction).

DOMAINS: capitals (TYPE_BC), languages (TYPE_BC), antonyms (TYPE_A)
LAYERS: L0 (=W_E), L8, L16, L24, L27 (final)

PREDICTION:
  - Experiment A: 0.818 (known from Day 162)
  - Experiment B: better than A (activation space has more context-specific info)
  - Experiment C: depends on alignment between activation and token spaces
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day184_activation_vs_token.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

DOMAINS = {
    "capitals": [
        ("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
        ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
        ("Russia","Moscow"),("Greece","Athens"),("Sweden","Stockholm"),
        ("Korea","Seoul"),
    ],
    "languages": [
        ("France","French"),("Germany","German"),("Italy","Italian"),
        ("Spain","Spanish"),("Japan","Japanese"),("China","Chinese"),
        ("Russia","Russian"),("Greece","Greek"),
    ],
    "antonyms": [
        ("hot","cold"),("big","small"),("fast","slow"),
        ("hard","soft"),("light","dark"),("old","young"),("loud","quiet"),
    ],
}

LAYERS = [0, 8, 16, 24, 27]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                       normed(np.array(b,dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
print(f"  H={W_E.shape[1]}, L={len(model.model.layers)}\n")

def tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def get_activations(word, layers):
    t = tid(word)
    if t is None: return None
    inp = torch.tensor([[t]])
    with torch.no_grad():
        out = model.model(inp, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple: (embed, L1, ..., L28)
    acts = {}
    for L in layers:
        acts[L] = hidden_states[L][0, 0].numpy().astype(np.float32)
    return acts

print("Pre-computing activations for all domain words...")
all_words = set()
for pairs in DOMAINS.values():
    for a, b in pairs:
        all_words.update([a, b])

cache = {}
for w in sorted(all_words):
    if tid(w):
        cache[w] = get_activations(w, set(LAYERS))
        if cache[w] is None:
            del cache[w]

ok_words = set(cache.keys())
print(f"  Cached {len(ok_words)} words\n")

def loo_experiment(pairs, source_space, target_space, layer):
    """
    source_space: 'we' or 'act' — where to get the query embedding
    target_space: 'we' or 'act' — where to do the snap (retrieval)
    direction_space: always 'we' for Exp A and C, 'act' for Exp B
    """
    ok = [(a, b) for a, b in pairs if a in ok_words and b in ok_words]
    if len(ok) < 3: return 0.0

    def emb_src(w):
        return W_E[tid(w)] if source_space == 'we' else cache[w][layer]

    def emb_tgt(w):
        return W_E[tid(w)] if target_space == 'we' else cache[w][layer]

    tgt_vocab = {b: emb_tgt(b) for _, b in ok}

    nc = 0
    for a, b in ok:
        # LOO direction from same space as source
        loo_diffs = [normed(emb_src(bb) - emb_src(aa))
                     for aa, bb in ok if aa != a]
        if not loo_diffs: continue
        d = normed(np.mean(loo_diffs, axis=0))
        q = emb_src(a) + d
        cands = {w: cosine(q, tgt_vocab[w]) for w in tgt_vocab if w != a}
        if cands and max(cands, key=lambda w: cands[w]) == b:
            nc += 1
    return nc / len(ok)

def loo_proximity(pairs, space, layer):
    ok = [(a, b) for a, b in pairs if a in ok_words and b in ok_words]
    if len(ok) < 3: return 0.0
    def emb(w): return W_E[tid(w)] if space == 'we' else cache[w][layer]
    tgt_vocab = {b: emb(b) for _, b in ok}
    nc = 0
    for a, b in ok:
        cands = {w: cosine(emb(a), tgt_vocab[w]) for w in tgt_vocab if w != a}
        if cands and max(cands, key=lambda w: cands[w]) == b:
            nc += 1
    return nc / len(ok)

results = {}
print(f"{'Domain':>12}  {'Exp':>5}  {'L0(W_E)':>8}  {'L8':>8}  "
      f"{'L16':>8}  {'L24':>8}  {'L27':>8}")
print("-"*75)

for domain, pairs in DOMAINS.items():
    results[domain] = {}

    # Experiment A: W_E→W_E direction + W_E query + W_E snap
    row_a = {L: loo_experiment(pairs, 'we', 'we', L) for L in LAYERS}
    results[domain]["A_we_dir_we_snap"] = row_a
    vals = "  ".join(f"{row_a[L]:>8.3f}" for L in LAYERS)
    print(f"  {domain:>12}  {'A(W_E)':>5}  {vals}")

    # Experiment B: act→act direction + act query + act snap (per layer)
    row_b = {L: loo_experiment(pairs, 'act', 'act', L) for L in LAYERS}
    results[domain]["B_act_dir_act_snap"] = row_b
    vals = "  ".join(f"{row_b[L]:>8.3f}" for L in LAYERS)
    print(f"  {domain:>12}  {'B(act)':>5}  {vals}")

    # Experiment C: W_E direction + act query + act snap (cross-space)
    # We use W_E embeddings for direction computation but activation for query
    row_c = {}
    ok = [(a, b) for a, b in pairs if a in ok_words and b in ok_words]
    for L in LAYERS:
        tgt_vocab_act = {b: cache[b][L] for _, b in ok if b in cache}
        nc = 0
        for a, b in ok:
            loo_diffs = [normed(W_E[tid(bb)] - W_E[tid(aa)])
                         for aa, bb in ok if aa != a and tid(aa) and tid(bb)]
            if not loo_diffs: continue
            d = normed(np.mean(loo_diffs, axis=0))
            q = cache[a][L] + d  # activation query + W_E direction
            cands = {w: cosine(q, tgt_vocab_act[w]) for w in tgt_vocab_act if w != a}
            if cands and max(cands, key=lambda w: cands[w]) == b:
                nc += 1
        row_c[L] = nc / len(ok) if ok else 0.0
    results[domain]["C_we_dir_act_snap"] = row_c
    vals = "  ".join(f"{row_c[L]:>8.3f}" for L in LAYERS)
    print(f"  {domain:>12}  {'C(mix)':>5}  {vals}")

    # Proximity baselines: W_E and activation at each layer
    row_prox_we = {L: loo_proximity(pairs, 'we', L) for L in [0]}
    row_prox_act = {L: loo_proximity(pairs, 'act', L) for L in LAYERS}
    results[domain]["prox_we"] = row_prox_we
    results[domain]["prox_act"] = row_prox_act
    vals_act = "  ".join(f"{row_prox_act[L]:>8.3f}" for L in LAYERS)
    print(f"  {domain:>12}  {'prox':>5}  {row_prox_we[0]:>8.3f}  {vals_act[8:]}")
    print()

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 184 complete.")
