#!/usr/bin/env python3
"""
Day 168 — Multi-Pole Routing for Type D Domains

Day 166 showed planets and colors_temp are "Type D" multi-pole domains:
  - Single direction fails (direction of rocky planets opposes gas planets)
  - Direction DEGRADES with more examples (more poles = more confusion)
  - Max accuracy ~43% (planets), ~32% (colors) before collapse

SOLUTION: Two-stage routing
  Stage 1: Classify which sub-category the instance belongs to
           (proximity-based, no supervision needed)
  Stage 2: Apply the sub-category-specific direction to get the label

PLANET EXAMPLE:
  Stage 1: Is Jupiter near {Mercury,Venus,Earth,Mars} or {Jupiter,Saturn,Uranus}?
           → Jupiter is near Saturn cluster → route to "gas direction"
  Stage 2: gas_direction = mean(Jupiter→gas, Saturn→gas)
           → Jupiter + gas_direction → nearest = "gas" ✓

COLOR EXAMPLE:
  Stage 1: Is "red" near {red,orange,yellow} or {blue,green,purple}?
           → red is near orange/yellow cluster → route to "warm direction"
  Stage 2: warm_direction = mean(red→warm, orange→warm, yellow→warm)
           → red + warm_direction → nearest = "warm" ✓

METHODS TESTED:
  M1: Oracle routing (know the sub-category, apply correct sub-direction)
      Upper bound on routing approach
  M2: k-NN clustering (proximity-based sub-category detection, leave-one-out)
      Unsupervised routing: which cluster is the query nearest to?
  M3: Single direction (Day 164/166 approach, baseline)
  M4: Random routing (lower bound)

ALSO TEST:
  Extension to NEW Type D domains:
    - Country → continent (Europe, Asia, Americas, Africa)
    - Season → type (hot/cold, wet/dry)
    - Number → parity (odd/even) — purely categorical
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day168_multipole_routing.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ─── Type D domain definitions (poles = sub-categories) ─────────
PLANETS = {
    "rocky": [("Mercury","rocky"),("Venus","rocky"),("Earth","rocky"),("Mars","rocky")],
    "gas":   [("Jupiter","gas"),("Saturn","gas"),("Uranus","gas"),("Neptune","gas")],
}

COLORS_TEMP = {
    "warm":    [("red","warm"),("orange","warm"),("yellow","warm")],
    "cool":    [("blue","cool"),("green","cool"),("purple","cool")],
    "neutral": [("white","neutral"),("black","neutral"),("gray","neutral")],
}

CONTINENTS = {
    "Europe":   [("France","Europe"),("Germany","Europe"),("Italy","Europe"),
                 ("Spain","Europe"),("Poland","Europe"),("Sweden","Europe")],
    "Asia":     [("Japan","Asia"),("China","Asia"),("India","Asia"),("Korea","Asia")],
    "Americas": [("Brazil","Americas"),("Mexico","Americas"),("Canada","Americas")],
    "Africa":   [("Egypt","Africa"),("Nigeria","Africa"),("Kenya","Africa")],
}

PARITY = {
    "odd":  [("one","odd"),("three","odd"),("five","odd"),("seven","odd"),("nine","odd")],
    "even": [("two","even"),("four","even"),("six","even"),("eight","even"),("ten","even")],
}

VOCAB = [
    # planets
    "Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune",
    "rocky","gas","inner","outer","planet","moon","solid","liquid",
    # colors
    "red","blue","yellow","green","orange","purple","white","black","gray",
    "warm","cool","neutral","primary","secondary","bright","dark",
    # countries/continents
    "France","Germany","Italy","Spain","Poland","Sweden","Japan","China",
    "India","Korea","Brazil","Mexico","Canada","Egypt","Nigeria","Kenya",
    "Europe","Asia","Americas","Africa","continent","country",
    # numbers
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "odd","even","prime","number",
    # general
    "hot","cold","big","small","fast","slow","good","bad",
    "metal","iron","copper","animal","bird","fish","insect",
    "Paris","Berlin","Tokyo","Beijing","Rome","Madrid",
    "French","German","Japanese","Chinese","Italian","Spanish",
    "king","queen","man","woman","boy","girl",
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                       normed(np.array(b,dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
print(f"  H={W_E.shape[1]}\n")

def tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

vocab_ok   = [w for w in dict.fromkeys(VOCAB) if tid(w)]
vocab_embs = {w: W_E[tid(w)] for w in vocab_ok}

def make_dir(pairs):
    ds = [normed(W_E[tid(b)] - W_E[tid(a)])
          for a, b in pairs if tid(a) and tid(b)]
    return normed(np.mean(ds, axis=0)) if ds else None

def entity_excl(src, direction, exclude):
    eid = tid(src)
    if eid is None: return None, 0.0
    e = W_E[eid].copy()
    if direction is not None: e = e + direction
    cands = [w for w in vocab_ok if w not in exclude]
    scores = {w: cosine(e, vocab_embs[w]) for w in cands}
    top1 = max(cands, key=lambda w: scores[w])
    return top1, scores[top1]

def run_multipole_domain(domain_name, poles_dict, verbose=True):
    """
    Test all 4 methods on a multi-pole domain.
    poles_dict: {pole_label: [(instance, label), ...]}
    """
    # Collect all valid pairs
    all_pairs = [(a,b,pole) for pole, pairs in poles_dict.items()
                 for a,b in pairs if tid(a) and tid(b)]
    if not all_pairs:
        print(f"  {domain_name}: no valid pairs")
        return {}
    pole_labels = list(poles_dict.keys())

    print(f"\n{'='*60}")
    print(f"Domain: {domain_name}  ({len(all_pairs)} valid pairs, {len(pole_labels)} poles)")
    print(f"Poles: {pole_labels}")
    print()

    # ── Method 1: Oracle routing (know sub-category) ──────────────
    # For each instance, use only its own pole's direction (leave-one-out)
    nc_oracle, n_oracle = 0, 0
    oracle_results = []
    for src, tgt, pole in all_pairs:
        # Train direction from all OTHER pairs in same pole
        same_pole = [(a,b) for a,b,p in all_pairs if p==pole and a!=src]
        if not same_pole: continue
        d = make_dir(same_pole)
        pred, score = entity_excl(src, d, {src})
        ok = pred == tgt
        if ok: nc_oracle += 1
        n_oracle += 1
        oracle_results.append({"src": src, "tgt": tgt, "pole": pole,
                                 "pred": pred, "ok": ok})
    acc_oracle = nc_oracle/n_oracle if n_oracle else 0
    print(f"  M1 (oracle routing LOO): {nc_oracle}/{n_oracle} = {acc_oracle:.3f}")
    if verbose:
        for r in oracle_results:
            print(f"    [{r['pole']:>8}] {r['src']:>10} → pred:{r['pred']:<10} "
                  f"target:{r['tgt']}  {'✓' if r['ok'] else '✗'}")

    # ── Method 2: k-NN routing (proximity-based sub-category) ────
    # For each instance, find which pole's centroid it is nearest to.
    # Then apply that pole's direction.
    pole_centroids = {}
    for pole, pairs in poles_dict.items():
        valid = [a for a,b in pairs if tid(a)]
        if valid:
            pole_centroids[pole] = normed(np.mean([W_E[tid(a)] for a in valid], axis=0))

    nc_knn, n_knn = 0, 0
    knn_results = []
    for src, tgt, true_pole in all_pairs:
        eid = tid(src)
        if eid is None: continue
        e_src = W_E[eid]
        # Find nearest pole centroid (excluding src from centroid calc)
        best_pole = None; best_sim = -999
        for pole, c in pole_centroids.items():
            # Recalculate centroid without src for fair eval
            valid_excl = [a for a,b in poles_dict[pole] if tid(a) and a!=src]
            if valid_excl:
                c_excl = normed(np.mean([W_E[tid(a)] for a in valid_excl], axis=0))
            else:
                c_excl = c
            sim = cosine(e_src, c_excl)
            if sim > best_sim:
                best_sim = sim; best_pole = pole
        # Apply best pole's direction (excl src)
        train = [(a,b) for a,b in poles_dict[best_pole] if a!=src and tid(a) and tid(b)]
        d = make_dir(train) if train else None
        pred, score = entity_excl(src, d, {src})
        ok = pred == tgt
        routed_correct = (best_pole == true_pole)
        if ok: nc_knn += 1
        n_knn += 1
        knn_results.append({"src": src, "tgt": tgt, "true_pole": true_pole,
                              "routed_to": best_pole, "route_ok": routed_correct,
                              "pred": pred, "ok": ok})
    acc_knn = nc_knn/n_knn if n_knn else 0
    route_acc = sum(1 for r in knn_results if r["route_ok"]) / len(knn_results) if knn_results else 0
    print(f"\n  M2 (k-NN routing): routing_acc={route_acc:.3f}, answer_acc={nc_knn}/{n_knn} = {acc_knn:.3f}")
    if verbose:
        for r in knn_results:
            route_marker = "✓" if r["route_ok"] else "✗"
            ans_marker   = "✓" if r["ok"]       else "✗"
            print(f"    [{r['true_pole']:>8}] {r['src']:>10} → route:{r['routed_to']:>8}{route_marker}  "
                  f"pred:{r['pred']:<10} target:{r['tgt']}  {ans_marker}")

    # ── Method 3: Single direction (all poles combined, baseline) ─
    all_dir_pairs = [(a,b) for a,b,p in all_pairs]
    nc_single, n_single = 0, 0
    for src, tgt, pole in all_pairs:
        train = [(a,b) for a,b in all_dir_pairs if a!=src]
        d = make_dir(train) if train else None
        pred, _ = entity_excl(src, d, {src})
        if pred == tgt: nc_single += 1
        n_single += 1
    acc_single = nc_single/n_single if n_single else 0
    print(f"\n  M3 (single dir LOO): {nc_single}/{n_single} = {acc_single:.3f}")

    # ── Method 4: No direction (proximity only) ───────────────────
    nc_prox, n_prox = 0, 0
    for src, tgt, pole in all_pairs:
        pred, _ = entity_excl(src, None, {src})
        if pred == tgt: nc_prox += 1
        n_prox += 1
    acc_prox = nc_prox/n_prox if n_prox else 0
    print(f"  M4 (proximity only): {nc_prox}/{n_prox} = {acc_prox:.3f}")

    print(f"\n  ── Summary: proxy={acc_prox:.3f} < single={acc_single:.3f} < knn={acc_knn:.3f} < oracle={acc_oracle:.3f}")
    return {
        "oracle": {"nc": nc_oracle, "n": n_oracle, "acc": acc_oracle},
        "knn":    {"nc": nc_knn,    "n": n_knn,    "acc": acc_knn,
                   "route_acc": route_acc, "details": knn_results},
        "single": {"nc": nc_single, "n": n_single, "acc": acc_single},
        "prox":   {"nc": nc_prox,   "n": n_prox,   "acc": acc_prox},
    }

# ─── Run all domains ─────────────────────────────────────────────
print("="*60)
print("MULTI-POLE ROUTING EXPERIMENT")
print("="*60)

results = {}
results["planets"]    = run_multipole_domain("Planets→Type", PLANETS)
results["colors_temp"]= run_multipole_domain("Colors→Temperature", COLORS_TEMP)
results["continents"] = run_multipole_domain("Country→Continent", CONTINENTS)
results["parity"]     = run_multipole_domain("Number→Parity", PARITY)

# ─── Summary table ────────────────────────────────────────────────
print()
print("="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Domain':>20}  {'prox':>6}  {'single':>6}  {'knn':>6}  {'oracle':>6}  {'route%':>6}")
print("-"*60)
for domain, r in results.items():
    if not r: continue
    print(f"  {domain:>18}  {r['prox']['acc']:>6.3f}  {r['single']['acc']:>6.3f}"
          f"  {r['knn']['acc']:>6.3f}  {r['oracle']['acc']:>6.3f}"
          f"  {r['knn']['route_acc']:>6.3f}")

print()
print("Key questions:")
print("  1. Does oracle routing unlock Type D domains (high oracle acc)?")
print("  2. Does k-NN routing achieve oracle accuracy (routing is correct)?")
print("  3. Is k-NN routing accuracy ≈ oracle × route_accuracy?")

with open(OUTPUT_FILE, "w") as f:
    json.dump({k: {m: {kk: vv for kk, vv in v.items() if kk != "details"}
                    for m, v in r.items()}
               for k, r in results.items() if r}, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 168 complete.")
