#!/usr/bin/env python3
"""
Day 145 — W_E Fact Surgery

DC 339-340 established: W_E encodes factual/relational knowledge geometrically.
France ≈ Paris because they co-occur in training text.

HYPOTHESIS: If we EDIT W_E[France] to resemble W_E[Tokyo], the model will
predict Tokyo instead of Paris for "The capital of France is".

This directly tests: IS THE KNOWLEDGE IN W_E, or does it also live in
the attention weights / MLP weights at later layers?

If the knowledge is purely in W_E:
  edit W_E[France] → Tokyo embedding → model predicts Tokyo

If the knowledge is distributed through all layers:
  edit W_E[France] → Tokyo embedding → model still predicts Paris
  (because layers 1-27 re-encode France's identity from context)

EXPERIMENTS:
  1. Baseline: France → Paris (rank 0)
  2. Surgery A: W_E[France] = W_E[Tokyo] → what rank does Tokyo get?
  3. Surgery B: W_E[France] = lerp(W_E[France], W_E[Tokyo], alpha)
     alpha ∈ {0.25, 0.5, 0.75, 1.0}
  4. Surgery C: W_E[France] += (W_E[Japan] - W_E[Germany]) — vector shift
     (add "Japan direction" to France)
  5. Multi-word: edit W_E for all words in "The capital of France is"
     except France → does France's unique embedding drive the prediction?

CONTROL: Test with Germany → Berlin, Italy → Rome (different countries)
"""
import json
from pathlib import Path
import numpy as np
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day145_we_fact_surgery.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
W_E_orig = model.model.embed_tokens.weight.detach().clone()  # frozen copy
print(f"  hidden={H}  W_E shape={W_E_orig.shape}\n")

def get_token_id(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

def logprob_rank(prompt, target_word, model):
    """Get rank of target_word in the model's output distribution."""
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1)
    tid = get_token_id(target_word)
    if tid is None: return -1, float('nan')
    # rank of target (0-indexed)
    rank = int((lp > lp[tid]).sum().item())
    return rank, float(lp[tid])

def top5_tokens(prompt, model):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1)
    top5_ids = lp.topk(5).indices.tolist()
    return [(tok.decode([t]).strip(), float(lp[t])) for t in top5_ids]

# Gather token IDs
WORDS = {
    "France": get_token_id("France"),
    "Paris":  get_token_id("Paris"),
    "Japan":  get_token_id("Japan"),
    "Tokyo":  get_token_id("Tokyo"),
    "Germany":get_token_id("Germany"),
    "Berlin": get_token_id("Berlin"),
    "Italy":  get_token_id("Italy"),
    "Rome":   get_token_id("Rome"),
    "Spain":  get_token_id("Spain"),
    "Madrid": get_token_id("Madrid"),
}
print("Token IDs:")
for w,t in WORDS.items(): print(f"  {w:>10}: {t}")
print()

PROMPTS = {
    "France": "The capital city of France is",
    "Japan":  "The capital city of Japan is",
    "Germany":"The capital city of Germany is",
}

def reset_we(model, W_E_orig):
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(W_E_orig)

def patch_we(model, token_id, new_emb):
    with torch.no_grad():
        model.model.embed_tokens.weight[token_id] = torch.tensor(new_emb, dtype=torch.float32)

print("="*72)
print("Day 145: W_E Fact Surgery")
print("="*72)
print()

all_results = []

# BASELINE
print("--- BASELINE (unmodified W_E) ---")
for country, prompt in PROMPTS.items():
    capital = {"France":"Paris","Japan":"Tokyo","Germany":"Berlin"}[country]
    rank, lp = logprob_rank(prompt, capital, model)
    top5 = top5_tokens(prompt, model)
    print(f"  {country} → {capital}: rank={rank}  lp={lp:.3f}  top5={[w for w,_ in top5]}")
    all_results.append({"exp":"baseline","country":country,"capital":capital,"rank":rank,"lp":lp,"top5":top5})

print()

# SURGERY A: Replace W_E[France] with W_E[Tokyo] entirely
print("--- SURGERY A: W_E[France] ← W_E[Tokyo] ---")
reset_we(model, W_E_orig)
france_orig = W_E_orig[WORDS["France"]].numpy().copy()
tokyo_emb   = W_E_orig[WORDS["Tokyo"]].numpy().copy()
japan_emb   = W_E_orig[WORDS["Japan"]].numpy().copy()

patch_we(model, WORDS["France"], tokyo_emb)

for target in ["Paris", "Tokyo"]:
    rank, lp = logprob_rank(PROMPTS["France"], target, model)
    top5 = top5_tokens(PROMPTS["France"], model)
    print(f"  France→{target}: rank={rank}  lp={lp:.3f}  top5={[w for w,_ in top5]}")
    all_results.append({"exp":"surgery_A","target":target,"rank":rank,"lp":lp,"top5":top5})

print()

# SURGERY B: Interpolate W_E[France] → W_E[Tokyo]
print("--- SURGERY B: W_E[France] ← lerp(France, Tokyo, alpha) ---")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    reset_we(model, W_E_orig)
    blended = (1-alpha)*france_orig + alpha*tokyo_emb
    patch_we(model, WORDS["France"], blended)
    rank_paris, lp_paris = logprob_rank(PROMPTS["France"], "Paris", model)
    rank_tokyo, lp_tokyo = logprob_rank(PROMPTS["France"], "Tokyo", model)
    top5 = top5_tokens(PROMPTS["France"], model)
    cos_to_tokyo = cosine(blended, tokyo_emb)
    cos_to_paris_emb = cosine(blended, W_E_orig[WORDS["Paris"]].numpy())
    print(f"  alpha={alpha:.2f}  cos(France_new,Tokyo_emb)={cos_to_tokyo:.3f}  "
          f"Paris:rank={rank_paris}  Tokyo:rank={rank_tokyo}  top5={[w for w,_ in top5]}")
    all_results.append({"exp":"surgery_B","alpha":alpha,"rank_paris":rank_paris,"rank_tokyo":rank_tokyo,
                        "top5":top5,"cos_tokyo":cos_to_tokyo,"cos_paris_emb":cos_to_paris_emb})

print()

# SURGERY C: W_E[France] += (Japan_emb - Germany_emb) [directional shift]
print("--- SURGERY C: W_E[France] += (Japan - Germany) direction ---")
reset_we(model, W_E_orig)
germany_emb = W_E_orig[WORDS["Germany"]].numpy().copy()
direction   = japan_emb - germany_emb
direction   = direction / (np.linalg.norm(direction) + 1e-8)
for scale in [0.0, 0.5, 1.0, 2.0, 5.0]:
    reset_we(model, W_E_orig)
    shifted = france_orig + scale * direction
    patch_we(model, WORDS["France"], shifted)
    rank_paris, lp_paris = logprob_rank(PROMPTS["France"], "Paris", model)
    rank_tokyo, lp_tokyo = logprob_rank(PROMPTS["France"], "Tokyo", model)
    top5 = top5_tokens(PROMPTS["France"], model)
    print(f"  scale={scale:.1f}  Paris:rank={rank_paris}  Tokyo:rank={rank_tokyo}  top5={[w for w,_ in top5]}")
    all_results.append({"exp":"surgery_C","scale":scale,"rank_paris":rank_paris,"rank_tokyo":rank_tokyo,"top5":top5})

print()

# SURGERY D: Normalize new embedding (keep magnitude, change direction)
print("--- SURGERY D: W_E[France] ← normed(Tokyo) × ||France|| ---")
reset_we(model, W_E_orig)
france_norm = float(np.linalg.norm(france_orig))
new_emb_D   = normed(tokyo_emb) * france_norm
patch_we(model, WORDS["France"], new_emb_D)
rank_paris, lp_paris = logprob_rank(PROMPTS["France"], "Paris", model)
rank_tokyo, lp_tokyo = logprob_rank(PROMPTS["France"], "Tokyo", model)
top5_D = top5_tokens(PROMPTS["France"], model)
print(f"  Paris:rank={rank_paris}  Tokyo:rank={rank_tokyo}  top5={[w for w,_ in top5_D]}")
all_results.append({"exp":"surgery_D","rank_paris":rank_paris,"rank_tokyo":rank_tokyo,"top5":top5_D})
print()

# SURGERY E: Patch ALL words in the prompt that relate to France
# Replace W_E[France] with W_E[Japan] AND keep Japan embedding separate
print("--- SURGERY E: W_E[France] ← W_E[Japan] (country-level swap) ---")
reset_we(model, W_E_orig)
patch_we(model, WORDS["France"], japan_emb)  # France token → Japan embedding
rank_paris, lp_paris = logprob_rank(PROMPTS["France"], "Paris", model)
rank_tokyo, lp_tokyo = logprob_rank(PROMPTS["France"], "Tokyo", model)
top5_E = top5_tokens(PROMPTS["France"], model)
print(f"  Paris:rank={rank_paris}  Tokyo:rank={rank_tokyo}  top5={[w for w,_ in top5_E]}")
print(f"  (Japan token still present and unchanged)")
# Also test Japan prompt with patched model (shouldn't change)
rank_j_tokyo, _ = logprob_rank(PROMPTS["Japan"], "Tokyo", model)
print(f"  Japan prompt (control): Tokyo rank={rank_j_tokyo}")
all_results.append({"exp":"surgery_E","rank_paris":rank_paris,"rank_tokyo":rank_tokyo,"top5":top5_E,
                    "japan_control_rank":rank_j_tokyo})
print()

# SURGERY F: Test on entity_excl — does modifying W_E change entity_excl output?
print("--- SURGERY F: entity_excl after W_E surgery ---")
reset_we(model, W_E_orig)
W_E_A = W_E_orig.numpy().copy()
W_E_A[WORDS["France"]] = tokyo_emb  # Surgery A

VOCAB_CURATED_SHORT = ["Paris","London","Rome","Berlin","Madrid","Tokyo","Moscow","Beijing",
                       "Warsaw","Athens","Seoul","Stockholm","Canberra","Cairo","Delhi",
                       "English","French","Spanish","German","Italian","Portuguese","Japanese","Korean"]
vocab_ok = {}
for w in VOCAB_CURATED_SHORT:
    tid = get_token_id(w)
    if tid: vocab_ok[w] = tid

france_tid = WORDS["France"]
exclude = {"France","French"}

# entity_excl with original W_E
scores_orig = {w: cosine(W_E_orig[france_tid].numpy(), W_E_orig[vocab_ok[w]].numpy())
               for w in vocab_ok if w not in exclude}
top3_orig = sorted(scores_orig, key=lambda w: -scores_orig[w])[:3]

# entity_excl with surgery A W_E
scores_A = {w: cosine(W_E_A[france_tid], W_E_orig[vocab_ok[w]].numpy())
            for w in vocab_ok if w not in exclude}
top3_A = sorted(scores_A, key=lambda w: -scores_A[w])[:3]

print(f"  entity_excl(France) original W_E: {top3_orig}")
print(f"  entity_excl(France) surgery A W_E: {top3_A}")
print()

# Reset model
reset_we(model, W_E_orig)

print("="*72)
print("Summary")
print("="*72)

# Find surgery alpha that first pushes Tokyo above Paris
print("\n  Surgery B interpolation — transition point:")
for r in all_results:
    if r["exp"] == "surgery_B":
        if r["rank_tokyo"] < r["rank_paris"]:
            print(f"    alpha={r['alpha']:.2f}: Tokyo(rank={r['rank_tokyo']}) > Paris(rank={r['rank_paris']})  ← CROSSOVER")
        else:
            print(f"    alpha={r['alpha']:.2f}: Paris(rank={r['rank_paris']}) > Tokyo(rank={r['rank_tokyo']})")

print("\n  Surgery A (full swap France←Tokyo):")
for r in all_results:
    if r["exp"] == "surgery_A":
        print(f"    {r['target']}: rank={r['rank']}  top5={[w for w,_ in r['top5']]}")

print("\n  VERDICT:")
sa = next(r for r in all_results if r["exp"]=="surgery_A" and r["target"]=="Tokyo")
sb_full = next(r for r in all_results if r["exp"]=="surgery_B" and abs(r["alpha"]-1.0)<0.01)

if sa["rank"] < 10:
    print(f"  W_E surgery WORKS: Tokyo at rank {sa['rank']} after surgery A")
    print(f"  Factual knowledge for 'France→capital' IS encoded in W_E")
elif sa["rank"] < 100:
    print(f"  W_E surgery PARTIAL: Tokyo at rank {sa['rank']} after surgery A")
    print(f"  Some factual knowledge in W_E, some in later layers")
else:
    print(f"  W_E surgery FAILS: Tokyo at rank {sa['rank']} after surgery A")
    print(f"  Factual knowledge NOT primarily in W_E (lives in later layers)")

print(f"\n  entity_excl correctly reflects surgery: {top3_A[0] == 'Tokyo'}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 145 complete.")
