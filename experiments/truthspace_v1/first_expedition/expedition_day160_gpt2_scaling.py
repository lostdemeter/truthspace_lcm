#!/usr/bin/env python3
"""
Day 160 — GPT-2 Family Scaling Study

Does W_E geometric structure improve with scale within a model family?

Models (same tokenizer, different H):
  gpt2       124M  H=768
  gpt2-medium 345M  H=1024
  gpt2-large  762M  H=1280

Same vocabulary (50,257 BPE) → identical tokenization → clean scaling test.
No vocab differences to confound results.

MEASUREMENTS:
  1. cap_dir alignment with own SVD (which PC? what cosine?)
  2. entity_excl accuracy on held-out set (29 prompts)
  3. SVD structure top-5 interpretation
  4. Cross-model SVD correlation (small vs medium vs large)
  5. Confidence score distribution and calibration

HYPOTHESIS A (Scale helps): Larger models encode the same knowledge
  more clearly. cap_dir cos → PC3 increases, entity_excl accuracy increases.

HYPOTHESIS B (Saturation): The geometry saturates at small scale.
  gpt2-small already captures the full structure; medium/large are identical.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day160_gpt2_scaling.json")

MODELS = [
    ("openai-community/gpt2",        "gpt2-small",   124),
    ("openai-community/gpt2-medium", "gpt2-medium",  345),
    ("openai-community/gpt2-large",  "gpt2-large",   762),
]

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
LANGUAGE_PAIRS = [
    ("Germany","German"),("France","French"),("Spain","Spanish"),
    ("Japan","Japanese"),("Italy","Italian"),("Greece","Greek"),
    ("Poland","Polish"),("Sweden","Swedish"),("Russia","Russian"),
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

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def cosine(a, b): return float(np.dot(normed(np.array(a,dtype=np.float64)),
                                       normed(np.array(b,dtype=np.float64))))

# Load tokenizer (shared across GPT-2 family)
print("Loading GPT-2 tokenizer (shared across family) ...")
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")

def g_tid(word):
    ids = tok(" "+word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids)==1 else None

# Build shared vocabulary
seen = set()
ALL_VOCAB = [w for w in VOCAB_CURATED if w.lower() not in seen and not seen.add(w.lower())]
vocab = [w for w in ALL_VOCAB if g_tid(w)]
N = len(vocab)
print(f"Vocabulary: {N} single-token words\n")

CAT_DIRS = {}  # will be computed per model (since W_E differs)

all_model_results = {}
W_E_matrices = {}  # store for cross-model comparison

for model_id, label, params_M in MODELS:
    print(f"{'='*68}")
    print(f"Model: {label} ({params_M}M params, id={model_id})")
    print(f"{'='*68}")

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    W_E = model.transformer.wte.weight.detach().numpy().astype(np.float32)
    H   = W_E.shape[1]
    del model
    print(f"  H={H}, W_E shape: {W_E.shape}")

    W_E_matrices[label] = W_E

    # Build vocab embeddings
    vocab_embs = {w: W_E[g_tid(w)] for w in vocab}

    # Universal directions
    def make_dir(pairs):
        ds = [normed(W_E[g_tid(b)] - W_E[g_tid(a)])
              for a, b in pairs if g_tid(a) and g_tid(b)]
        return normed(np.mean(ds, axis=0)) if ds else None

    cap_dir    = make_dir(list(COUNTRY_CAPITAL.items()))
    gender_dir = make_dir(GENDER_PAIRS)
    antonym_dir= make_dir(ANTONYM_PAIRS)
    lang_dir   = make_dir(LANGUAGE_PAIRS)
    CAT_DIRS[label] = {"cap": cap_dir, "gender": gender_dir,
                       "antonym": antonym_dir, "lang": lang_dir}

    # SVD
    M = np.array([vocab_embs[w] for w in vocab], dtype=np.float32)
    M_c = M - M.mean(axis=0)
    _, S, Vt = np.linalg.svd(M_c, full_matrices=False)
    print(f"  S[:6] = {S[:6].round(2)}")

    # Direction → SVD alignment
    print(f"\n  Universal direction SVD alignment (top-20 PCs):")
    dir_aligns = {}
    for dname, dvec in [("cap",cap_dir),("gender",gender_dir),
                         ("antonym",antonym_dir),("lang",lang_dir)]:
        if dvec is None: continue
        aligns = [(k, cosine(Vt[k], dvec)) for k in range(20)]
        best_k, best_c = max(aligns, key=lambda x: abs(x[1]))
        dir_aligns[dname] = {"pc": best_k, "cos": best_c}
        print(f"    {dname:>12}: PC{best_k}, cos={best_c:.3f}")

    # Top-5 SVD components
    print(f"\n  Top-5 SVD components:")
    for k in range(5):
        scores = [(vocab[i], float(M_c[i] @ Vt[k])) for i in range(N)]
        scores.sort(key=lambda x: -x[1])
        top = [w for w,_ in scores[:4]]
        bot = [w for w,_ in scores[-3:]]
        print(f"    PC{k}: +{top}  -{bot}")

    # entity_excl
    def entity_excl(entity_word, direction, exclude):
        eid = g_tid(entity_word)
        if eid is None: return None, 0.0
        e = W_E[eid].copy()
        if direction is not None: e = e + direction
        excl = [w for w in vocab if w not in exclude]
        scores = {w: cosine(e, vocab_embs[w]) for w in excl}
        top1 = max(excl, key=lambda w: scores[w])
        return top1, scores[top1]

    cat_dir_map = {"gender": gender_dir, "capitals": cap_dir}
    n_agree = 0
    case_results = []
    # Use the Qwen2 oracle from Day 156 json as reference
    for prompt, entity, exclude, expected, cat in HELD_OUT:
        direction = cat_dir_map.get(cat)
        pred, score = entity_excl(entity, direction, exclude)
        agree = (pred == expected)
        if agree: n_agree += 1
        case_results.append({"entity": entity, "cat": cat, "expected": expected,
                              "pred": pred, "score": round(float(score),4),
                              "agree": agree})

    print(f"\n  entity_excl (vs expected): {n_agree}/29 = {n_agree/29:.3f}")

    # Confidence distribution
    scores_all = [r["score"] for r in case_results]
    scores_correct = [r["score"] for r in case_results if r["agree"]]
    scores_wrong   = [r["score"] for r in case_results if not r["agree"]]
    print(f"  Score stats: correct={np.mean(scores_correct):.3f}±{np.std(scores_correct):.3f}  "
          f"wrong={np.mean(scores_wrong):.3f}±{np.std(scores_wrong):.3f}")

    # Threshold sweep (vs expected)
    print(f"  Confidence gate (vs expected labels):")
    best_n = 0; best_thresh = 0.0
    for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        we = sum(1 for r in case_results if r["score"] >= thresh and r["agree"])
        lost = sum(1 for r in case_results if r["score"] >= thresh and not r["agree"])
        fall = n_agree - we + (29 - sum(1 for r in case_results if r["score"] >= thresh))
        # count correct: all agree + cases with low conf falling back to oracle (=expected)
        fall_correct = sum(1 for r in case_results
                           if r["score"] < thresh and r["expected"] == expected)
        n_total = we + sum(1 for r in case_results if r["score"] < thresh)
        print(f"    thresh={thresh:.2f}: we={we} correct")
    print()

    all_model_results[label] = {
        "H": H, "params_M": params_M, "n_agree": n_agree,
        "dir_aligns": dir_aligns,
        "S": S[:10].tolist(), "Vt_top5": Vt[:5].tolist(),
        "case_results": case_results,
        "score_mean_correct": float(np.mean(scores_correct)),
        "score_mean_wrong":   float(np.mean(scores_wrong)),
    }

# ─────────────────────────────────────────────────────────────
print("="*68)
print("Cross-Model SVD Correlation within GPT-2 Family")
print("="*68)
print()

labels = [label for _, label, _ in MODELS]
for i, la in enumerate(labels):
    Vt_a = np.array(all_model_results[la]["Vt_top5"])
    M_a  = np.array([W_E_matrices[la][g_tid(w)] for w in vocab], dtype=np.float32)
    M_ac = M_a - M_a.mean(axis=0)
    for j, lb in enumerate(labels):
        if j <= i: continue
        Vt_b = np.array(all_model_results[lb]["Vt_top5"])
        M_b  = np.array([W_E_matrices[lb][g_tid(w)] for w in vocab], dtype=np.float32)
        M_bc = M_b - M_b.mean(axis=0)
        print(f"  {la} vs {lb}:")
        print(f"  {'PC':>4}  " + "  ".join(f"{'PC'+str(j):>8}" for j in range(5)))
        for k in range(5):
            ws_a = np.array([float(M_ac[i] @ Vt_a[k]) for i in range(N)])
            row = []
            for m in range(5):
                ws_b = np.array([float(M_bc[i] @ Vt_b[m]) for i in range(N)])
                r = float(np.corrcoef(ws_a, ws_b)[0,1])
                row.append(f"{r:>+8.3f}")
            print(f"  PC{k}:  " + "  ".join(row))
        print()

# ─────────────────────────────────────────────────────────────
print("="*68)
print("Summary: GPT-2 Scaling")
print("="*68)
print()
print(f"  {'Model':>14}  {'H':>6}  {'params':>8}  {'cap_dir PC':>12}  {'cap_cos':>9}  {'accuracy':>10}")
for la in labels:
    r = all_model_results[la]
    cd = r["dir_aligns"].get("cap", {})
    print(f"  {la:>14}  {r['H']:>6}  {r['params_M']:>7}M  "
          f"PC{cd.get('pc','?'):>9}  {cd.get('cos',0):>+9.3f}  "
          f"{r['n_agree']:>2}/29={r['n_agree']/29:.3f}")

print()
print(f"  (Qwen2-1.5B for reference: H=1536, 1500M, PC3, cos=0.434, 24/29=82.8%)")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_model_results, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 160 complete.")
