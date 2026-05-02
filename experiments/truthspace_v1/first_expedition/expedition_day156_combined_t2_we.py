#!/usr/bin/env python3
"""
Day 156 — Combined T2+W_E Pipeline: Can T2 fix W_E's irreducible failures?

Day 148 routed W_E pipeline: 24/29 = 82.8%
Remaining 5 failures:
  1. Australia → Canberra  (W_E gives Sydney)
  2. hammer   → tool       (W_E gives weapon)
  3. whale    → animal     (W_E gives bird)
  4. tense-she → went      (irreducible)
  5. tense-they → went     (irreducible)

HYPOTHESIS: The full-context forward pass (T2 / oracle logprob) already
knows the correct answers. W_E is a no-cost approximation; full inference
is the gold standard.

TEST:
  A. Oracle logprob (full forward pass): what does the model actually predict?
     (this is the T2 component — what does L25 full-context know?)

  B. Combined pipeline: W_E for categories where it excels; fall back to
     oracle logprob for categories where W_E fails or for hard cases detected
     by a confidence signal.

  C. Confidence-gated fallback: if W_E top-1 cosine score < threshold,
     use oracle logprob. Sweep thresholds.

  D. Hard-coded hybrid: use W_E for confident cases, oracle for the 5 failures.
     This is the THEORETICAL CEILING of the two-component architecture.

GOAL: Determine if the two-component pipeline can reach 28/29 = 96.6%
(by fixing the 4 non-tense failures with T2 full inference).
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day156_combined_t2_we.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

VOCAB_CURATED = [
    "walked","ran","ate","built","wrote","read","said","went","came","took",
    "made","got","saw","gave","knew","thought","found","told","became","left",
    "brought","bought","taught","caught","fought","heard","held","kept","sent",
    "fell","felt","grew","slept","spent","stood","wore","won",
    "drove","flew","swam","sang","sat","laid","paid","played","stayed","opened",
    "turned","looked","stopped","asked","started","tried","closed","moved","lived",
    "walk","run","eat","build","write","say","come","take","make","get","see",
    "give","know","think","find","tell","become","leave",
    "cold","hot","big","small","fast","slow","dark","light","happy","sad",
    "good","bad","strong","weak","young","old","loud","quiet","easy","hard",
    "clean","dirty","rich","poor","safe","early","late",
    "Paris","London","Rome","Berlin","Madrid","Tokyo","Moscow","Beijing",
    "Sydney","Ottawa","Canberra","Brasilia","Cairo","Delhi","Seoul","Bangkok",
    "Vienna","Warsaw","Athens","Lisbon","Brussels","Amsterdam","Oslo","Stockholm",
    "English","French","Spanish","German","Italian","Portuguese","Arabic",
    "Mandarin","Japanese","Korean","Hindi","Russian","Turkish","Persian",
    "Bengali","Tamil","Urdu","Polish","Dutch","Swedish","Greek",
    "animal","plant","tool","vehicle","food","music","sport","color","number",
    "language","country","city","flower","tree","bird","fish","dog","cat","horse",
    "instrument","weapon","machine","device","metal","mineral","crystal","gem",
    "king","queen","prince","princess","duke","duchess","emperor","empress",
    "father","mother","brother","sister","son","daughter","uncle","aunt",
    "man","woman","boy","girl","actor","actress","hero","heroine",
    "east","west","north","south","morning","evening","night","water","fire",
    "door","house","book","table","chair","window","street","road","park",
    "school","office","market","store","church","castle","palace","bridge",
    "then","also","soon","just","very","still","again","always","never",
    "first","last","next","before","after","here","there",
]

HELD_OUT = [
    ("The capital city of Russia is",      "Russia",   {"Russia","Russian"},  "Moscow",    "capitals"),
    ("The capital city of China is",       "China",    {"China","Chinese"},   "Beijing",   "capitals"),
    ("The capital city of Australia is",   "Australia",{"Australia"},         "Canberra",  "capitals"),
    ("The capital city of Greece is",      "Greece",   {"Greece","Greek"},    "Athens",    "capitals"),
    ("The capital city of Poland is",      "Poland",   {"Poland","Polish"},   "Warsaw",    "capitals"),
    ("The capital city of Sweden is",      "Sweden",   {"Sweden","Swedish"},  "Stockholm", "capitals"),
    ("The official language of Germany is","Germany",  {"Germany"},           "German",    "languages"),
    ("The official language of Japan is",  "Japan",    {"Japan"},             "Japanese",  "languages"),
    ("The official language of Korea is",  "Korea",    {"Korea"},             "Korean",    "languages"),
    ("A hammer is a type of",   "hammer", {"hammer"}, "tool",       "hypernyms"),
    ("A ruby is a type of",     "ruby",   {"ruby"},   "gem",        "hypernyms"),
    ("A whale is a type of",    "whale",  {"whale"},  "animal",     "hypernyms"),
    ("A violin is a type of",   "violin", {"violin"}, "instrument", "hypernyms"),
    ("The opposite of good is",   "good",   {"good"},   "bad",    "antonyms"),
    ("The opposite of fast is",   "fast",   {"fast"},   "slow",   "antonyms"),
    ("The opposite of clean is",  "clean",  {"clean"},  "dirty",  "antonyms"),
    ("The opposite of rich is",   "rich",   {"rich"},   "poor",   "antonyms"),
    ("The opposite of loud is",   "loud",   {"loud"},   "quiet",  "antonyms"),
    ("The opposite of strong is", "strong", {"strong"}, "weak",   "antonyms"),
    ("The prince and", "prince", {"prince"},  "princess","gender"),
    ("The duke and",   "duke",   {"duke"},    "duchess", "gender"),
    ("The actor and",  "actor",  {"actor"},   "actress", "gender"),
    ("The son and",    "son",    {"son"},     "daughter","gender"),
    ("The opposite of happy is",  "happy",  {"happy"},  "sad",    "antonyms_extra"),
    ("The opposite of early is",  "early",  {"early"},  "late",   "antonyms_extra"),
    ("The opposite of old is",    "old",    {"old"},    "young",  "antonyms_extra"),
    ("The opposite of easy is",   "easy",   {"easy"},   "hard",   "antonyms_extra"),
    ("Last month she",  "she",  {"she"},  "went", "tense"),
    ("Last week they",  "they", {"they"}, "went", "tense"),
]

COUNTRY_CAPITAL = {
    "France":"Paris","Germany":"Berlin","Italy":"Rome","Spain":"Madrid",
    "Japan":"Tokyo","China":"Beijing","Russia":"Moscow","Brazil":"Brasilia",
    "Egypt":"Cairo","Greece":"Athens","Poland":"Warsaw","Sweden":"Stockholm",
}
ANTONYM_PAIRS = [
    ("hot","cold"),("big","small"),("fast","slow"),("dark","light"),
    ("good","bad"),("young","old"),("rich","poor"),("clean","dirty"),
    ("loud","quiet"),("strong","weak"),("early","late"),("easy","hard"),
]
GENDER_PAIRS = [
    ("king","queen"),("prince","princess"),("actor","actress"),
    ("son","daughter"),("father","mother"),("brother","sister"),
    ("man","woman"),("boy","girl"),
]

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(a), normed(b)))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
H = model.config.hidden_size
W_E = model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
print(f"  hidden={H}\n")

def get_tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

# Build vocab
seen = set()
VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab_ok = [w for w in VOCAB if get_tid(w)]
vocab_embs = {w: W_E[get_tid(w)] for w in vocab_ok}
vocab_tids  = {w: get_tid(w) for w in vocab_ok}
N = len(vocab_ok)

# Universal directions
def make_dir(pairs):
    ds = [normed(W_E[get_tid(b)] - W_E[get_tid(a)])
          for a, b in pairs if get_tid(a) and get_tid(b)]
    return normed(np.mean(ds, axis=0))

cap_dir    = make_dir(list(COUNTRY_CAPITAL.items()))
gender_dir = make_dir(GENDER_PAIRS)

CAT_DIRECTION = {
    "gender": gender_dir, "capitals": cap_dir,
}

def entity_excl(entity_word, exclude):
    eid = get_tid(entity_word)
    if eid is None: return vocab_ok[0], 0.0
    e = W_E[eid]
    excl = [w for w in vocab_ok if w not in exclude]
    scores = {w: cosine(e, vocab_embs[w]) for w in excl}
    ranked = sorted(excl, key=lambda w: -scores[w])
    return ranked[0], scores[ranked[0]]

def direction_rank(entity_word, direction, exclude):
    eid = get_tid(entity_word)
    if eid is None: return vocab_ok[0], 0.0
    result = W_E[eid] + direction
    excl = [w for w in vocab_ok if w not in exclude]
    scores = {w: cosine(result, vocab_embs[w]) for w in excl}
    ranked = sorted(excl, key=lambda w: -scores[w])
    return ranked[0], scores[ranked[0]]

def oracle_logprob(prompt, exclude):
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        lp = torch.log_softmax(model(**inp).logits[0,-1,:], dim=-1).numpy()
    excl = [w for w in vocab_ok if w not in exclude]
    scores = {w: float(lp[vocab_tids[w]]) for w in excl}
    ranked = sorted(excl, key=lambda w: -scores[w])
    # Also compute rank of each vocab word
    return ranked[0], scores[ranked[0]], scores

# ─────────────────────────────────────────────────────────────
print("="*72)
print("Day 156: Combined T2+W_E Pipeline — Can Full Inference Fix Failures?")
print("="*72)
print()

all_results = []
for prompt, entity, exclude, expected, cat in HELD_OUT:
    # W_E method (Day 148 routed)
    direction = CAT_DIRECTION.get(cat)
    if direction is not None:
        we_top1, we_score = direction_rank(entity, direction, exclude)
    else:
        we_top1, we_score = entity_excl(entity, exclude)

    # Oracle (T2 full inference)
    oracle_top1, oracle_score, oracle_all = oracle_logprob(prompt, exclude)

    # Oracle rank of W_E top-1
    oracle_rank_we_top1 = sorted(oracle_all.keys(), key=lambda w: -oracle_all[w]).index(we_top1) + 1

    # W_E rank of oracle top-1
    if direction is not None:
        eid = get_tid(entity)
        result = W_E[eid] + direction
        excl = [w for w in vocab_ok if w not in exclude]
        we_all = {w: cosine(result, vocab_embs[w]) for w in excl}
    else:
        eid = get_tid(entity)
        excl = [w for w in vocab_ok if w not in exclude]
        we_all = {w: cosine(W_E[eid], vocab_embs[w]) for w in excl}
    we_rank_oracle = sorted(we_all.keys(), key=lambda w: -we_all[w]).index(oracle_top1) + 1

    row = {
        "prompt": prompt, "entity": entity, "cat": cat, "expected": expected,
        "oracle_top1": oracle_top1, "we_top1": we_top1,
        "agree_we":     we_top1     == oracle_top1,
        "agree_oracle": True,  # oracle always agrees with itself
        "we_score": round(float(we_score), 4),
        "we_rank_oracle": we_rank_oracle,
        "oracle_rank_we_top1": oracle_rank_we_top1,
    }
    all_results.append(row)

    e = "✓" if row["agree_we"] else "✗"
    print(f"  [{cat:>15}] {entity:>12}: W_E={e}{we_top1:<11}(cos={we_score:.3f})"
          f"  oracle={oracle_top1}  [W_E_rank_oracle={we_rank_oracle}, "
          f"oracle_rank_we={oracle_rank_we_top1}]")

# ─────────────────────────────────────────────────────────────
print()
print("="*72)
print("PART 2: Hard Cases — Detailed Analysis")
print("="*72)
hard_cases = [r for r in all_results if not r["agree_we"]]
print(f"\n  {len(hard_cases)} cases where W_E fails:\n")
for r in hard_cases:
    print(f"  [{r['cat']:>15}] {r['entity']:>12}:")
    print(f"    W_E gives:    {r['we_top1']}  (cos={r['we_score']:.3f})")
    print(f"    Oracle gives: {r['oracle_top1']}")
    print(f"    W_E rank of oracle: {r['we_rank_oracle']}")
    print(f"    Oracle rank of W_E-top1: {r['oracle_rank_we_top1']}")
    print()

# ─────────────────────────────────────────────────────────────
print("="*72)
print("PART 3: Confidence-Gated Fallback Sweep")
print("="*72)
print()

# For each threshold, use W_E if confidence >= threshold, else oracle
THRESHOLDS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
print(f"  {'threshold':>12}  {'n_we':>6}  {'n_oracle':>10}  {'total_correct':>15}  accuracy")
for thresh in THRESHOLDS:
    n_correct = 0
    n_use_we = 0; n_use_oracle = 0
    for r in all_results:
        if r["we_score"] >= thresh:
            pred = r["we_top1"]
            n_use_we += 1
        else:
            pred = r["oracle_top1"]
            n_use_oracle += 1
        if pred == r["oracle_top1"]:
            n_correct += 1
    print(f"  thresh={thresh:.2f}:  we={n_use_we:>4}  oracle={n_use_oracle:>5}  "
          f"correct={n_correct:>4}/29 = {n_correct/29:.3f}")

# Best threshold
best_thresh = max(THRESHOLDS,
    key=lambda t: sum(1 for r in all_results
        if (r["we_top1"] if r["we_score"]>=t else r["oracle_top1"]) == r["oracle_top1"]))
n_at_best = sum(1 for r in all_results
    if (r["we_top1"] if r["we_score"]>=best_thresh else r["oracle_top1"]) == r["oracle_top1"])
print(f"\n  Best threshold: {best_thresh} → {n_at_best}/29 = {n_at_best/29:.3f}")

# ─────────────────────────────────────────────────────────────
print()
print("="*72)
print("PART 4: Theoretical Ceiling")
print("="*72)
print()

# Oracle always wins — what's the max possible?
n_oracle_all = len(all_results)  # oracle is perfect by definition
# But can oracle fix the tense cases?
tense_cases = [r for r in all_results if r["cat"] == "tense"]
print(f"  Tense cases ({len(tense_cases)} total):")
for r in tense_cases:
    print(f"    {r['entity']:>12}: oracle={r['oracle_top1']}  expected={r['expected']}")

non_tense = [r for r in all_results if r["cat"] != "tense"]
# Among non-tense, how many does oracle get right (=oracle_top1==oracle_top1 by def)
# But we should check: does oracle (=model logprob) agree with expected?
n_oracle_vs_expected = sum(1 for r in all_results if r["oracle_top1"] == r["expected"])
print(f"\n  Oracle vs expected labels: {n_oracle_vs_expected}/29 = {n_oracle_vs_expected/29:.3f}")
print(f"  (Oracle=LM logprob, expected=our label — these may differ!)")

n_we_vs_expected = sum(1 for r in all_results if r["we_top1"] == r["expected"])
print(f"  W_E vs expected labels:    {n_we_vs_expected}/29 = {n_we_vs_expected/29:.3f}")

print()
print("="*72)
print("Summary")
print("="*72)
n_we = sum(r["agree_we"] for r in all_results)
print(f"""
  Method                    Accuracy
  ──────────────────────────────────────────────
  W_E routed (Day 148):     {n_we}/29 = {n_we/29:.3f}
  Oracle (full inference):  29/29 = 1.000  (by definition)
  Best gated hybrid:        {n_at_best}/29 = {n_at_best/29:.3f}

  W_E vs expected labels:   {n_we_vs_expected}/29 = {n_we_vs_expected/29:.3f}
  Oracle vs expected labels: {n_oracle_vs_expected}/29 = {n_oracle_vs_expected/29:.3f}

  Hard cases (W_E fails):   {len(hard_cases)} cases
    W_E gives wrong answer, Oracle gives right answer? → see above
""")

# Does oracle fix the W_E failures?
n_oracle_fixes = sum(1 for r in hard_cases
    if r["oracle_top1"] != r["we_top1"] and r["oracle_top1"] == r["expected"])
n_we_already_right = sum(1 for r in hard_cases if r["we_top1"] == r["expected"])
print(f"  Of {len(hard_cases)} W_E failures:")
print(f"    Oracle would fix:     {n_oracle_fixes} (oracle gives expected answer)")
print(f"    W_E already matches expected: {n_we_already_right}")
print(f"    Still unfixable (oracle≠expected): {len(hard_cases)-n_oracle_fixes-n_we_already_right}")
print()
print(f"  THEORETICAL CEILING with oracle fallback: {n_we + n_oracle_fixes}/29 = {(n_we+n_oracle_fixes)/29:.3f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({"all_results": all_results, "n_we": n_we, "n_oracle_fixes": n_oracle_fixes,
               "best_threshold": best_thresh, "n_at_best": n_at_best}, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 156 complete.")
