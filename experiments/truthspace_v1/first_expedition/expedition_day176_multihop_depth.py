#!/usr/bin/env python3
"""
Day 176 — Multi-Hop Chain Depth in W_E

DC 354 established: chain accuracy = product of individual step accuracies,
and the 'snap-to-nearest' operation is critical between hops.

QUESTIONS:
  Q1: Can we go 3 hops? 4 hops? When does the chain accuracy collapse?
  Q2: Does chain accuracy = step1_acc × step2_acc × ... (multiplicative)?
  Q3: Can we build DIVERSE chains (not just country→capital→language)?
      country → capital → currency?
      animal → sound → adjective?
      element → property → example?
  Q4: Do wrong snaps propagate catastrophically, or does the chain 
      recover if the next direction is strong enough?
  Q5: Can direction be inferred from the chain itself?
      (entity1→entity2) implies what relation? Can we use it to predict step3?

HYPOTHESIS: Chain accuracy follows a geometric degradation:
  P(chain_k correct) ≈ P(step1) × P(step2) × ... × P(stepk)
  Chains longer than ~3 hops degrade to chance level.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day176_multihop_depth.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ─── Relation datasets ───────────────────────────────────────────
# Chain: country → capital → language → language_family
COUNTRY_CAPITAL = [
    ("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
    ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing"),
    ("Russia","Moscow"),("Brazil","Brasilia"),("Greece","Athens"),
    ("Poland","Warsaw"),("Sweden","Stockholm"),("Korea","Seoul"),
]
CAPITAL_LANGUAGE = [
    ("Paris","French"),("Berlin","German"),("Rome","Italian"),
    ("Madrid","Spanish"),("Tokyo","Japanese"),("Beijing","Chinese"),
    ("Moscow","Russian"),("Athens","Greek"),("Warsaw","Polish"),
    ("Stockholm","Swedish"),("Seoul","Korean"),
]
LANGUAGE_FAMILY = [
    ("French","Romance"),("Italian","Romance"),("Spanish","Romance"),
    ("German","Germanic"),("Swedish","Germanic"),("English","Germanic"),
    ("Russian","Slavic"),("Polish","Slavic"),("Greek","Hellenic"),
    ("Japanese","Japonic"),("Chinese","Sinitic"),("Korean","Koreanic"),
]

# Chain: animal → sound → descriptor
ANIMAL_SOUND = [
    ("dog","bark"),("cat","meow"),("cow","moo"),("duck","quack"),
    ("lion","roar"),("bird","tweet"),("frog","croak"),("bee","buzz"),
]
SOUND_DESCRIPTOR = [
    ("bark","loud"),("meow","soft"),("moo","low"),("quack","flat"),
    ("roar","fierce"),("buzz","high"),("croak","rough"),
]

# Chain: metal → property → example_object  
METAL_PROPERTY = [
    ("iron","magnetic"),("copper","conductive"),("gold","malleable"),
    ("silver","reflective"),("aluminum","lightweight"),("lead","heavy"),
]
PROPERTY_USE = [
    ("magnetic","compass"),("conductive","wire"),("malleable","jewelry"),
    ("reflective","mirror"),("lightweight","aircraft"),("heavy","shield"),
]

# Chain: season → weather → activity
SEASON_WEATHER = [
    ("winter","snow"),("summer","heat"),("spring","rain"),("autumn","wind"),
]
WEATHER_ACTIVITY = [
    ("snow","skiing"),("heat","swimming"),("rain","reading"),("wind","sailing"),
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

def make_dir(pairs, excl_src=None):
    ds = [normed(W_E[tid(b)] - W_E[tid(a)])
          for a, b in pairs if tid(a) and tid(b) and a != excl_src]
    return normed(np.mean(ds, axis=0)) if ds else None

def snap(src_emb, direction, vocab_embs, exclude=None):
    excl = set(exclude or [])
    e = src_emb + direction if direction is not None else src_emb
    scores = {w: cosine(e, v) for w, v in vocab_embs.items() if w not in excl}
    return max(scores, key=lambda w: scores[w]) if scores else None

# ─── Helper: test a 2-hop chain ──────────────────────────────────
def test_chain(name, rel1_pairs, rel2_pairs, test_starts=None):
    """Test a 2-hop chain: A → B → C  (rel1: A→B, rel2: B→C)"""
    all_entities = set(a for a,b in rel1_pairs) | set(b for a,b in rel1_pairs)
    all_entities |= set(a for a,b in rel2_pairs) | set(b for a,b in rel2_pairs)
    vocab = {w: W_E[tid(w)] for w in all_entities if tid(w)}

    rel1_map = dict(rel1_pairs)
    rel2_map = dict(rel2_pairs)

    if test_starts is None:
        test_starts = [a for a,b in rel1_pairs if tid(a) and b in rel2_map and tid(b) and tid(rel2_map[b])]

    n_hop1 = 0; nc_hop1 = 0
    n_hop2 = 0; nc_hop2 = 0
    n_chain = 0; nc_chain = 0

    print(f"\n  Chain: {name}")
    for src in test_starts:
        eid = tid(src)
        if not eid: continue
        mid_tgt = rel1_map.get(src)
        if not mid_tgt or not tid(mid_tgt): continue
        end_tgt = rel2_map.get(mid_tgt)
        if not end_tgt or not tid(end_tgt): continue

        d1 = make_dir(rel1_pairs, excl_src=src)
        d2 = make_dir(rel2_pairs, excl_src=mid_tgt)
        if d1 is None or d2 is None: continue

        # Hop 1: src → mid
        pred1 = snap(W_E[eid], d1, vocab, {src})
        ok1 = (pred1 == mid_tgt)
        nc_hop1 += ok1; n_hop1 += 1

        # Hop 2 (from actual mid_tgt): mid → end
        mid_eid = tid(mid_tgt)
        pred2 = snap(W_E[mid_eid], d2, vocab, {mid_tgt}) if mid_eid else None
        ok2 = (pred2 == end_tgt)
        nc_hop2 += ok2; n_hop2 += 1

        # Chain: src → pred1 → pred3 (using pred1 as hop-2 input)
        pred1_eid = tid(pred1) if pred1 else None
        if pred1_eid:
            pred3 = snap(W_E[pred1_eid], d2, vocab, {pred1})
        else:
            pred3 = None
        ok_chain = (pred3 == end_tgt)
        nc_chain += ok_chain; n_chain += 1

        print(f"    {src:>12} →[{pred1 or '?':>10}]→ {pred3 or '?':>12}  "
              f"(target: {mid_tgt}→{end_tgt})  "
              f"{'✓✓' if ok_chain else ('✓✗' if ok1 else '✗?')}")

    a1 = nc_hop1/n_hop1 if n_hop1 else 0
    a2 = nc_hop2/n_hop2 if n_hop2 else 0
    ac = nc_chain/n_chain if n_chain else 0
    pred_chain = a1 * a2  # multiplicative prediction
    print(f"\n    Hop1={a1:.3f} Hop2={a2:.3f} Chain={ac:.3f} "
          f"(predicted={pred_chain:.3f}, {'matches' if abs(ac-pred_chain)<0.15 else 'deviates'})")
    return {"name": name, "hop1": a1, "hop2": a2, "chain": ac, "pred_chain": pred_chain}

# ─── Test 2-hop chains ────────────────────────────────────────────
print("="*64)
print("2-HOP CHAIN ACCURACY")
print("="*64)

results = []
r = test_chain("country→capital→language", COUNTRY_CAPITAL, CAPITAL_LANGUAGE)
results.append(r)
r = test_chain("animal→sound→descriptor", ANIMAL_SOUND, SOUND_DESCRIPTOR)
results.append(r)
r = test_chain("metal→property→use", METAL_PROPERTY, PROPERTY_USE)
results.append(r)
r = test_chain("season→weather→activity", SEASON_WEATHER, WEATHER_ACTIVITY)
results.append(r)

# ─── 3-hop chain ─────────────────────────────────────────────────
print()
print("="*64)
print("3-HOP CHAIN: country → capital → language → family")
print("="*64)

# Build full vocab
all_words = (set(a for a,b in COUNTRY_CAPITAL) | set(b for a,b in COUNTRY_CAPITAL) |
             set(a for a,b in CAPITAL_LANGUAGE) | set(b for a,b in CAPITAL_LANGUAGE) |
             set(a for a,b in LANGUAGE_FAMILY) | set(b for a,b in LANGUAGE_FAMILY))
vocab3 = {w: W_E[tid(w)] for w in all_words if tid(w)}

cap_map  = dict(COUNTRY_CAPITAL)
lang_map = dict(CAPITAL_LANGUAGE)
fam_map  = dict(LANGUAGE_FAMILY)

d_cap  = make_dir(COUNTRY_CAPITAL)
d_lang = make_dir(CAPITAL_LANGUAGE)
d_fam  = make_dir(LANGUAGE_FAMILY)

nc3 = 0; n3 = 0
for country in [a for a,b in COUNTRY_CAPITAL]:
    eid = tid(country)
    if not eid: continue
    cap_tgt  = cap_map.get(country)
    lang_tgt = lang_map.get(cap_tgt, "") if cap_tgt else ""
    fam_tgt  = fam_map.get(lang_tgt, "") if lang_tgt else ""
    if not all(tid(x) for x in [cap_tgt, lang_tgt, fam_tgt] if x): continue

    # Hop 1: country + cap_dir → capital
    d1 = make_dir(COUNTRY_CAPITAL, excl_src=country)
    pred1 = snap(W_E[eid], d1, vocab3, {country})

    # Hop 2: pred1 + lang_dir → language
    pred1_eid = tid(pred1) if pred1 else None
    pred2 = snap(W_E[pred1_eid], d_lang, vocab3, {pred1}) if pred1_eid else None

    # Hop 3: pred2 + fam_dir → family
    pred2_eid = tid(pred2) if pred2 else None
    pred3 = snap(W_E[pred2_eid], d_fam, vocab3, {pred2}) if pred2_eid else None

    ok = (pred3 == fam_tgt)
    nc3 += ok; n3 += 1
    print(f"  {country:>10} → {pred1 or '?':>10} → {pred2 or '?':>12} → {pred3 or '?':>12}  "
          f"(targets: {cap_tgt}→{lang_tgt}→{fam_tgt})  {'✓' if ok else '✗'}")

acc3 = nc3/n3 if n3 else 0
print(f"\n  3-hop accuracy: {nc3}/{n3} = {acc3:.3f}")
print(f"  (predicted from 2-hop data: ~{results[0]['hop1']*results[0]['hop2']:.3f})")

# ─── Error propagation test ───────────────────────────────────────
print()
print("="*64)
print("ERROR PROPAGATION: Does wrong snap recover?")
print("="*64)
print()
print("  Test: force hop1 to WRONG answer, does hop2 recover?")
print()

for country in ["France","Germany","Japan","Russia"][:4]:
    eid = tid(country)
    if not eid: continue
    cap_tgt = cap_map.get(country)
    lang_tgt = lang_map.get(cap_tgt, "") if cap_tgt else ""
    if not cap_tgt or not lang_tgt: continue

    # Wrong snap: use a DIFFERENT country's capital
    wrong_caps = [b for a,b in COUNTRY_CAPITAL if a != country and tid(b)]
    if not wrong_caps: continue
    wrong_cap = wrong_caps[0]
    wrong_lang_correct = lang_map.get(wrong_cap, "?")

    wrong_eid = tid(wrong_cap)
    if not wrong_eid: continue

    pred_lang = snap(W_E[wrong_eid], d_lang, vocab3, {wrong_cap})
    print(f"  {country}: correct chain={cap_tgt}→{lang_tgt}")
    print(f"    forced wrong: {wrong_cap}→{pred_lang}  (expected wrong: {wrong_lang_correct})")
    print(f"    Recovery? {'YES (got correct lang)' if pred_lang == lang_tgt else 'NO (got wrong lang)'}")
    print()

# ─── Summary ─────────────────────────────────────────────────────
print("="*64)
print("Summary")
print("="*64)
for r in results:
    deviation = abs(r['chain'] - r['pred_chain'])
    print(f"  {r['name'][:35]:>35}: chain={r['chain']:.3f}, "
          f"predicted={r['pred_chain']:.3f}, dev={deviation:.3f}")
print(f"\n  3-hop: {acc3:.3f}")
print(f"\n  Multiplicative prediction: {'HOLDS' if all(abs(r['chain']-r['pred_chain'])<0.15 for r in results) else 'DEVIATES'}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"chains_2hop": results, "3hop_acc": acc3}, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 176 complete.")
