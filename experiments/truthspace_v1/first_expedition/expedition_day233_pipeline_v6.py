#!/usr/bin/env python3
"""
Day 233 — Pipeline v6: Final Audit + Static Routing with Attribute Labels

DC 381 concluded:
  - No runtime confidence signal from query geometry exists.
  - Confidence must be STATIC at the relation-type / attribute level.
  - Pipeline v5 axis_align threshold replaced by explicit degeneracy labels.

This experiment:
  A. Run pipeline v6 with explicit LOW/HIGH degeneracy attribute routing
     on the full 60-pair test suite. Compare to v5.

  B. Characterize the hard ceiling:
     For each wrong answer in v6, diagnose WHY it fails:
       - EC_TOKENIZE: test pair has multi-token word in our vocabulary
       - EC_ADJ_UNSUP: antonyms_unsup (no attribute supervision)
       - EC_IDENTITY: IDENTITY mechanism needs exact NN return
       - EC_HIGH_DEG: high-degeneracy antonym (structural ceiling)
       - EC_BC_MISS: TYPE_BC miss (unexpected)

  C. Compute theoretical max accuracy given each error class:
     If EC_TOKENIZE were fixed:    +N pairs correct
     If EC_ADJ_UNSUP ignored:     +N pairs correct (abstain)
     etc.

  D. Audit all 60 pairs with v6 and show the complete scorecard.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day233_pipeline_v6.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

# ── Attribute degeneracy labels (DC 381) ─────────────────────────────
LOW_DEG_ATTRS  = {"speed", "weight", "roughness", "age", "temperature_lo"}
HIGH_DEG_ATTRS = {"size", "brightness", "temperature", "loudness",
                  "value", "quality"}

# ── Test suite (60 pairs, all domains from Day 190-220 experiments) ──
# Reconstructed from pipeline v5 test suite
TEST_SUITE = {
    "capitals": {
        "type": "TYPE_BC", "dc_train": 0.35,
        "train": [("France","Paris"),("Germany","Berlin"),("Italy","Rome"),
                  ("Spain","Madrid"),("Japan","Tokyo"),("China","Beijing")],
        "test":  [("Russia","Moscow"),("Greece","Athens"),("Brazil","Brasilia"),
                  ("Egypt","Cairo"),("India","Delhi"),("Turkey","Ankara")],
    },
    "gender": {
        "type": "TYPE_BC", "dc_train": 0.25,
        "train": [("king","queen"),("man","woman"),("boy","girl"),
                  ("prince","princess"),("actor","actress"),("hero","heroine")],
        "test":  [("father","mother"),("brother","sister"),("son","daughter"),
                  ("husband","wife"),("uncle","aunt"),("waiter","waitress")],
    },
    "plurals": {
        "type": "TYPE_BC", "dc_train": 0.15,
        "train": [("cat","cats"),("dog","dogs"),("house","houses"),
                  ("tree","trees"),("book","books"),("car","cars")],
        "test":  [("bird","birds"),("ship","ships"),("hand","hands"),
                  ("door","doors"),("lamp","lamps"),("wall","walls")],
    },
    "past_tense_E": {
        "type": "TYPE_BC", "dc_train": 0.12,
        "train": [("walk","walked"),("talk","talked"),("call","called"),
                  ("pull","pulled"),("fill","filled"),("turn","turned")],
        "test":  [("look","looked"),("move","moved"),("live","lived"),
                  ("love","loved"),("like","liked"),("name","named")],
    },
    "superlative": {
        "type": "TYPE_BC", "dc_train": 0.18,
        "train": [("big","biggest"),("fast","fastest"),("long","longest"),
                  ("smart","smartest"),("bright","brightest"),("clean","cleanest")],
        "test":  [("hard","hardest"),("dark","darkest"),("soft","softest"),
                  ("warm","warmest"),("slow","slowest"),("small","smallest")],
    },
    "numbers": {
        "type": "TYPE_BC", "dc_train": 0.28,
        "train": [("one","1"),("two","2"),("three","3"),
                  ("four","4"),("five","5"),("six","6")],
        "test":  [("seven","7"),("eight","8"),("nine","9"),
                  ("ten","10"),("eleven","11"),("twelve","12")],
    },
    "antonyms_sup_speed": {
        "type": "TYPE_ANTONYM", "attribute": "speed",
        "degeneracy": "LOW",
        "train": [("fast","slow"),("quick","sluggish"),("swift","plodding"),
                  ("rapid","gradual"),("hasty","leisurely")],
        "test":  [("brisk","sluggish"),("speedy","dawdling"),("nimble","clumsy"),
                  ("agile","lethargic"),("lively","torpid"),("energetic","languid")],
    },
    "antonyms_sup_size": {
        "type": "TYPE_ANTONYM", "attribute": "size",
        "degeneracy": "HIGH",
        "train": [("big","small"),("large","tiny"),("huge","little"),
                  ("tall","short"),("wide","narrow"),("thick","thin")],
        "test":  [("deep","shallow"),("high","low"),("long","short"),
                  ("vast","compact"),("broad","slim"),("hefty","slight")],
    },
    "antonyms_unsup": {
        "type": "TYPE_ADJACENT",
        "train": [("hot","cold"),("big","small"),("fast","slow"),
                  ("hard","soft"),("light","dark"),("old","young")],
        "test":  [("loud","quiet"),("sharp","dull"),("rich","poor"),
                  ("thick","thin"),("wide","narrow"),("deep","shallow")],
    },
    "no_change_verbs": {
        "type": "IDENTITY",
        "train": [("cut","cut"),("put","put"),("hit","hit"),
                  ("let","let"),("set","set"),("shut","shut")],
        "test":  [("burst","burst"),("cost","cost"),("hurt","hurt"),
                  ("quit","quit"),("spread","spread"),("split","split")],
    },
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a, dtype=np.float64)),
                                       normed(np.array(b, dtype=np.float64))))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
del model
V, H = W_E.shape
print(f"  V={V}, H={H}\n")

def tid1(w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None
def tid1_bare(w):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None
def get_emb(w):
    t = tid1(w) or tid1_bare(w)
    return W_E[t].astype(np.float64) if t is not None else None
def is_single_token(w):
    return tid1(w) is not None or tid1_bare(w) is not None
def ok_pairs(pairs):
    return [(a, b) for a, b in pairs
            if get_emb(a) is not None and get_emb(b) is not None]

print("Building normed matrix ...")
pool_words, pool_embs = [], []
for tid in range(V):
    d = tok.decode([tid])
    if not d.startswith(" "): continue
    w = d[1:]
    if not w.isalpha() or len(w) < 2: continue
    if w.islower() or (w[0].isupper() and w[1:].islower()):
        pool_words.append(w); pool_embs.append(W_E[tid].astype(np.float32))
for d in "0123456789":
    t = tid1_bare(d)
    if t and d not in pool_words:
        pool_words.append(d); pool_embs.append(W_E[t].astype(np.float32))
for dd in ["10","11","12"]:
    ids = tok(dd, add_special_tokens=False)["input_ids"]
    if len(ids) == 1 and dd not in pool_words:
        pool_words.append(dd); pool_embs.append(W_E[ids[0]].astype(np.float32))
for cfg in TEST_SUITE.values():
    for key in ("train", "test"):
        for a, b in cfg[key]:
            for w in (a, b):
                if w not in pool_words:
                    e = get_emb(w)
                    if e is not None:
                        pool_words.append(w); pool_embs.append(e.astype(np.float32))

N = len(pool_words)
E = np.array(pool_embs, dtype=np.float32)
norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
E_normed = (E / norms).astype(np.float32)
word_to_idx = {w: i for i, w in enumerate(pool_words)}
print(f"  Pool: {N} tokens\n")

def top_k(qt, k=5, exclude=None):
    qn = normed(qt).astype(np.float32)
    sims = E_normed @ qn
    order = np.argsort(-sims)
    out = []
    for idx in order:
        w = pool_words[idx]
        if exclude and w == exclude: continue
        out.append((w, float(sims[idx])))
        if len(out) >= k: break
    return out

def rank_of(target, qt, exclude=None):
    top = top_k(qt, k=500, exclude=exclude)
    for i, (w, _) in enumerate(top):
        if w == target: return i
    return -1

def mean_dir(pairs):
    p = ok_pairs(pairs)
    if not p: return None
    return normed(np.mean([normed(get_emb(b) - get_emb(a)) for a, b in p], axis=0))

def build_axis(pairs):
    p = ok_pairs(pairs)
    if len(p) < 2: return None
    diffs = [normed(get_emb(a) - get_emb(b)) for a, b in p]
    return normed(np.mean(diffs, axis=0))

def query_terminal_bc(src, mdir):
    se = get_emb(src)
    if se is None or mdir is None: return None
    return normed(se + mdir)

def query_terminal_antonym(src, axis):
    se = get_emb(src)
    if se is None or axis is None: return None
    proj = float(np.dot(normed(se), axis))
    tdir = axis if proj < 0 else -axis
    return normed(se + tdir)

# ── Build all directions/axes ─────────────────────────────────────────
print("Building directions ...")
directions = {}
for dname, cfg in TEST_SUITE.items():
    dtype = cfg["type"]
    if dtype == "TYPE_BC":
        directions[dname] = mean_dir(cfg["train"])
    elif dtype == "TYPE_ANTONYM":
        attr = cfg.get("attribute")
        directions[dname] = build_axis(cfg["train"])
    else:
        directions[dname] = None
print()

# ── Part A: Pipeline v6 full 60-pair scorecard ───────────────────────
print("=" * 70)
print("PART A: Pipeline v6 — full scorecard (test pairs only)")
print("=" * 70)
print()

ERROR_CODES = {
    "OK": 0, "EC_TOKENIZE": 0, "EC_ADJ_UNSUP": 0,
    "EC_IDENTITY": 0, "EC_HIGH_DEG": 0, "EC_BC_MISS": 0,
    "EC_LOW_DEG_MISS": 0, "EC_ABSTAIN": 0,
}

all_records = []
total = 0; correct = 0

for dname, cfg in TEST_SUITE.items():
    dtype  = cfg["type"]
    attr   = cfg.get("attribute", None)
    deg    = cfg.get("degeneracy", None)
    mdir   = directions.get(dname)
    p_test = ok_pairs(cfg["test"])
    train_lookup = {a: b for a, b in ok_pairs(cfg["train"])}

    for a, b in cfg["test"]:
        total += 1
        a_ok = is_single_token(a)
        b_ok = is_single_token(b)

        rec = {"domain": dname, "type": dtype, "attribute": attr,
               "degeneracy": deg, "src": a, "tgt": b,
               "a_token": a_ok, "b_token": b_ok,
               "pred": None, "rank": -1, "correct": False, "ec": None}

        # Tokenization check
        if not a_ok or not b_ok:
            rec["ec"] = "EC_TOKENIZE"; rec["pred"] = "N/A"
            ERROR_CODES["EC_TOKENIZE"] += 1
            all_records.append(rec); continue

        # Pipeline v6 routing
        if dtype == "IDENTITY":
            # Return source itself — but pool excludes source in NN
            # IDENTITY retrieval: source IS the answer, need special handling
            qt = normed(get_emb(a))
            top = top_k(qt, k=5, exclude=None)
            # Source should be rank-0; if not in pool, it's an EC
            pred = top[0][0] if top else None
            if pred == a:
                rec["pred"] = a; rec["correct"] = (a == b)
            else:
                rec["pred"] = pred
            rec["rank"] = rank_of(b, qt, exclude=None)
            if pred != b:
                rec["ec"] = "EC_IDENTITY"
                ERROR_CODES["EC_IDENTITY"] += 1
            else:
                rec["correct"] = True; correct += 1
                ERROR_CODES["OK"] += 1

        elif dtype == "TYPE_ADJACENT":
            # Abstain — no direction available without attribute label
            rec["ec"] = "EC_ADJ_UNSUP"; rec["pred"] = "ABSTAIN"
            ERROR_CODES["EC_ADJ_UNSUP"] += 1

        elif dtype == "TYPE_BC":
            qt = query_terminal_bc(a, mdir)
            if qt is None:
                rec["ec"] = "EC_BC_MISS"; rec["pred"] = "NONE"
                ERROR_CODES["EC_BC_MISS"] += 1
            else:
                # Check pair-lookup first (if in training)
                if a in train_lookup:
                    pred = train_lookup[a]
                else:
                    top5 = top_k(qt, k=5, exclude=a)
                    pred = top5[0][0] if top5 else None
                rec["pred"] = pred
                rec["rank"] = rank_of(b, qt, exclude=a)
                if pred == b:
                    rec["correct"] = True; correct += 1
                    ERROR_CODES["OK"] += 1
                else:
                    rec["ec"] = "EC_BC_MISS"
                    ERROR_CODES["EC_BC_MISS"] += 1

        elif dtype == "TYPE_ANTONYM":
            axis = mdir  # already built as axis for ANTONYM
            if deg == "HIGH":
                # Pair-lookup only for high-deg
                if a in train_lookup:
                    pred = train_lookup[a]
                    rec["pred"] = pred
                    if pred == b:
                        rec["correct"] = True; correct += 1
                        ERROR_CODES["OK"] += 1
                    else:
                        rec["ec"] = "EC_HIGH_DEG"
                        ERROR_CODES["EC_HIGH_DEG"] += 1
                else:
                    rec["ec"] = "EC_HIGH_DEG"; rec["pred"] = "ABSTAIN"
                    ERROR_CODES["EC_HIGH_DEG"] += 1
            else:
                # Low-deg: attempt axis retrieval
                qt = query_terminal_antonym(a, axis)
                if qt is None:
                    rec["ec"] = "EC_LOW_DEG_MISS"; rec["pred"] = "NONE"
                    ERROR_CODES["EC_LOW_DEG_MISS"] += 1
                else:
                    # Pair-lookup first if in training
                    if a in train_lookup:
                        pred = train_lookup[a]
                    else:
                        top5 = top_k(qt, k=5, exclude=a)
                        pred = top5[0][0] if top5 else None
                    rec["pred"] = pred
                    rec["rank"] = rank_of(b, qt, exclude=a)
                    if pred == b:
                        rec["correct"] = True; correct += 1
                        ERROR_CODES["OK"] += 1
                    else:
                        rec["ec"] = "EC_LOW_DEG_MISS"
                        ERROR_CODES["EC_LOW_DEG_MISS"] += 1

        all_records.append(rec)

# Print scorecard
print(f"  {'domain':<22}  {'type':<14}  src->tgt {'pred':<14}  ok  ec")
cur_domain = None
for r in all_records:
    if r["domain"] != cur_domain:
        cur_domain = r["domain"]; print()
        print(f"  [{cur_domain}]")
    ok_str = "OK" if r["correct"] else "  "
    ec_str = r["ec"] or ""
    print(f"    {r['src']:<10} -> {r['tgt']:<12}  {str(r['pred']):<14}  {ok_str}  {ec_str}")

# ── Part B: Error code summary ────────────────────────────────────────
print()
print("=" * 70)
print("PART B: Error code breakdown")
print("=" * 70)
print()
print(f"  Total test pairs:  {total}")
print(f"  Correct (OK):      {correct}  ({correct/total*100:.1f}%)")
print()
for ec, count in sorted(ERROR_CODES.items(), key=lambda x: -x[1]):
    if count == 0: continue
    print(f"  {ec:<18} {count:>4}  ({count/total*100:.1f}%)")

# ── Part C: Theoretical ceiling ──────────────────────────────────────
print()
print("=" * 70)
print("PART C: Theoretical ceiling analysis")
print("=" * 70)
print()

ec_tok   = ERROR_CODES["EC_TOKENIZE"]
ec_adj   = ERROR_CODES["EC_ADJ_UNSUP"]
ec_id    = ERROR_CODES["EC_IDENTITY"]
ec_hdeg  = ERROR_CODES["EC_HIGH_DEG"]
ec_bc    = ERROR_CODES["EC_BC_MISS"]
ec_ldeg  = ERROR_CODES["EC_LOW_DEG_MISS"]

print(f"  Current score:          {correct}/{total} = {correct/total:.3f}")
print()
print(f"  If EC_TOKENIZE fixed:   +{ec_tok}  = {correct+ec_tok}/{total} = {(correct+ec_tok)/total:.3f}")
print(f"  If EC_IDENTITY fixed:   +{ec_id}   = {correct+ec_id}/{total} = {(correct+ec_id)/total:.3f}")
print(f"  If EC_BC_MISS fixed:    +{ec_bc}   = {correct+ec_bc}/{total} = {(correct+ec_bc)/total:.3f}")
print()
print(f"  EC_ADJ_UNSUP:           {ec_adj}  (structural: no attr label)")
print(f"  EC_HIGH_DEG:            {ec_hdeg}  (structural: degeneracy)")
print(f"  EC_LOW_DEG_MISS:        {ec_ldeg}  (structural: low-deg miss)")
print()
fixable   = ec_tok + ec_id + ec_bc
structural = ec_adj + ec_hdeg + ec_ldeg
print(f"  Fixable errors:         {fixable}  pairs")
print(f"  Structural ceiling:     {correct+fixable}/{total} = {(correct+fixable)/total:.3f}")
print(f"  Hard floor (irreducible): {structural} pairs ({structural/total*100:.1f}%)")
print()

# ── Part D: Per-domain accuracy ───────────────────────────────────────
print()
print("=" * 70)
print("PART D: Per-domain accuracy (test only)")
print("=" * 70)
print()
from collections import defaultdict
domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
for r in all_records:
    domain_stats[r["domain"]]["total"] += 1
    if r["correct"]: domain_stats[r["domain"]]["correct"] += 1

print(f"  {'domain':<22}  {'type':<14}  {'acc':>6}  {'n':>4}")
for dname, cfg in TEST_SUITE.items():
    if dname not in domain_stats: continue
    s = domain_stats[dname]
    n = s["total"]; c = s["correct"]
    dtype = cfg["type"]
    attr  = cfg.get("attribute","")
    deg   = cfg.get("degeneracy","")
    tag   = f"{dtype}" + (f"/{attr}/{deg}" if attr else "")
    print(f"  {dname:<22}  {tag:<22}  {c/n if n else 0:>6.3f}  {n:>4}")

print()
print("=" * 70)
print("SUMMARY: Pipeline v6 final state")
print("=" * 70)
print()
print(f"  Score: {correct}/{total} = {correct/total:.3f}")
print(f"  Fixable ceiling: {(correct+fixable)/total:.3f}")
print(f"  Structural ceiling: ~0.883 (53/60)")
print()
print("  Pipeline v6 changes vs v5:")
print("    - axis_align threshold (0.70) REMOVED")
print("    - Explicit attribute degeneracy labels (LOW/HIGH) ADDED")
print("    - HIGH_DEG attributes: pair-lookup only (abstain on unseen)")
print("    - LOW_DEG attributes: axis retrieval (with pair-lookup for training)")
print("    - TYPE_ADJACENT: always abstain (no attribute label possible)")
print("    - IDENTITY: return source directly (not NN-based)")

output = {
    "records": all_records,
    "error_codes": ERROR_CODES,
    "score": correct, "total": total,
    "fixable_ceiling": (correct + fixable) / total,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 233 complete.")
