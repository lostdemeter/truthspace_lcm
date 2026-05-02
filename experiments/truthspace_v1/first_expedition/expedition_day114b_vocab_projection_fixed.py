#!/usr/bin/env python3
"""
Day 114b — Corrected Vocabulary Projection (Per-Axis Layer)

Day 114 used a single canonical layer (L28) for all T2 axis projections,
but T2 axes are defined at different layers:
  gender=L27, comparative=L15, hypernym=L28, plural=L1, ...

This caused cross-layer mismatch. Day 114b projects each word's hidden
state at the CORRECT layer for each axis, exactly as classify_all() does.

CORRECTED METHOD:
  For each axis a at layer L_a:
    proj_a(word) = h_word(L_a) · t2_axis_a
  T2 12D coordinate = [proj_a for a in AXIS_NAMES_12]
  T2 magnitude = L2 norm of 12D coordinate
  d_k projection = h_word(L28) · d_k  (entity selector uses L23 output)

This gives the correct semantic signal per axis.

PREDICTION (two-structure theory):
  - gender_pair (king/queen/man/woman) → high gender-axis projection
  - verb_past (ran/walked) → high past_tense-axis projection
  - comparative (faster/bigger) → high comparative-axis projection
  - function words → low T2 magnitude (not defined by semantic axes)
  - proper-noun-like tokens → higher d_k projection
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day114b_vocab_projection_fixed.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

INV_PHI  = 1 / ((1 + math.sqrt(5)) / 2)
INV_PHI2 = INV_PHI ** 2

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
        ("A son was born in the winter","A daughter was born in the winter"),
        ("The prince rode across the land","The princess rode across the land"),
        ("The actor played a leading role","The actress played a leading role"),
    ],
    "comparative": [
        ("The fast car","The faster car"),("A big dog","A bigger dog"),
        ("The cold wind","The colder wind"),("A tall tree","A taller tree"),
        ("The old house","The older house"),("A bright star","A brighter star"),
        ("The dark room","The darker room"),("A hard rock","A harder rock"),
    ],
    "hypernym": [
        ("The dog ran away from danger","The animal ran away from danger"),
        ("A rose bloomed in the garden","A flower bloomed in the garden"),
        ("The oak crashed in the storm","The tree crashed in the storm"),
        ("The car sped past the sign","The vehicle sped past the sign"),
        ("The eagle soared above the hill","The bird soared above the hill"),
        ("The ruby gleamed in the light","The gem gleamed in the light"),
        ("The soldier marched into fight","The person marched into fight"),
        ("The hammer struck the nail","The tool struck the nail"),
    ],
    "plural": [
        ("A dog played happily in the open green field","Dogs played happily in the open green field"),
        ("The cat sat quietly by the rain-streaked window","The cats sat quietly by the rain-streaked window"),
        ("A bird sang softly in the still morning mist","Birds sang softly in the still morning mist"),
        ("The tree fell down hard in the terrible storm","The trees fell down hard in the terrible storm"),
        ("A book sat open on the old wooden desk","Books sat open on the old wooden desk"),
        ("The car drove slowly down the long empty road","The cars drove slowly down the long empty road"),
        ("A star shone brightly in the cold clear sky","Stars shone brightly in the cold clear sky"),
        ("The word appeared clearly in the printed text","The words appeared clearly in the printed text"),
    ],
    "synonym": [
        ("He is big","He is large"),("She is small","She is tiny"),
        ("He runs fast","He runs quick"),("It is cold","It is frigid"),
        ("She is happy","She is joyful"),("He spoke loudly","He spoke noisily"),
        ("It is hard","It is difficult"),("He is old","He is aged"),
    ],
    "concrete": [
        ("The stone is too heavy to lift","The burden is too heavy to lift"),
        ("The iron chain has broken now","The bond between them has broken"),
        ("The long road leads to the sea","The long journey leads to the sea"),
        ("The high wall blocks the view","The high barrier blocks the view"),
        ("The flame slowly fades away","The hope slowly fades away"),
        ("The strong root grips the soil","The strong base grips the earth"),
        ("The bridge connects two banks","The bond connects two communities"),
        ("The small key opens the door","The small answer opens the path"),
    ],
    "past_tense": [
        ("I walk to the market every single morning","I walked to the market every single morning"),
        ("She runs through the park after her long work","She ran through the park after her long work"),
        ("He eats breakfast before leaving the old house","He ate breakfast before leaving the old house"),
        ("They build a stone wall around the garden","They built a stone wall around the garden"),
        ("We swim in the lake on warm summer days","We swam in the lake on warm summer days"),
        ("She writes a letter to her dear old friend","She wrote a letter to her dear old friend"),
        ("He speaks quietly during the long weekly meeting","He spoke quietly during the long weekly meeting"),
        ("They sing together around the evening campfire","They sang together around the evening campfire"),
    ],
    "antonym": [
        ("It is hot","It is cold"),("He runs fast","He runs slow"),
        ("The light is on","The dark is on"),("The news is good","The news is bad"),
        ("It is hard","It is soft"),("She is happy","She is sad"),
        ("He is strong","He is weak"),("It is the first","It is the last"),
    ],
    "passive": [
        ("The cat chased the mouse","The mouse was chased by the cat"),
        ("John broke the window","The window was broken by John"),
        ("The chef cooked the meal","The meal was cooked by the chef"),
        ("The dog bit the man","The man was bitten by the dog"),
        ("The teacher helped the student","The student was helped by the teacher"),
        ("The storm destroyed the house","The house was destroyed by the storm"),
        ("The artist painted the picture","The picture was painted by the artist"),
        ("The king signed the document","The document was signed by the king"),
    ],
    "causation": [
        ("The heavy rain falls all day","The ground gets completely wet"),
        ("The fire burns for a long time","The wood turns to ash slowly"),
        ("The sun heats the cold earth","The ice melts quickly in spring"),
        ("The wind blows the tree branches","The leaves fall to the ground"),
        ("The child cries very loudly","The mother comes running in"),
        ("The ball rolls off the tall edge","The ball falls to the floor"),
        ("The teacher praises the student","The student feels very proud"),
        ("The glass breaks on hard stone","The water spills everywhere"),
    ],
    "question": [
        ("She is very tired today","Is she very tired today"),
        ("He can swim really well","Can he swim really well"),
        ("They went to the market","Did they go to the market"),
        ("The car broke down again","Did the car break down again"),
        ("The dog is hungry now","Is the dog hungry now"),
        ("She wrote the letter herself","Did she write the letter herself"),
        ("He knows the right answer","Does he know the right answer"),
        ("The house looks very old","Does the house look very old"),
    ],
    "negation": [
        ("The dog is fast","The dog is not fast"),
        ("She can swim well","She cannot swim well"),
        ("He knows the answer","He does not know the answer"),
        ("The food is good","The food is not good"),
        ("They work hard","They do not work hard"),
        ("The water is cold","The water is not cold"),
        ("The house looks old","The house does not look old"),
        ("It will rain today","It will not rain today"),
    ],
}

TOKEN_CATEGORIES = {
    "animal": ["dog","cat","bird","fish","horse","wolf","lion","tiger","elephant","mouse",
               "rabbit","deer","bear","fox","eagle","whale","shark","frog","snake","monkey"],
    "nature": ["tree","flower","rock","stone","wood","leaf","grass","root","river","mountain",
               "ocean","forest","cloud","rain","snow","wind","sun","moon","star","sky"],
    "artifact": ["house","door","window","table","chair","book","cup","key","car","road",
                 "bridge","boat","ship","plane","train","bike","knife","hammer","clock","lamp"],
    "verb_base": ["run","walk","jump","swim","eat","sleep","talk","write","read","build",
                  "break","open","close","think","know","see","hear","feel","love","hate"],
    "verb_past": ["ran","walked","jumped","flew","ate","saw","heard","broke","built","wrote"],
    "adjective": ["fast","slow","big","small","hot","cold","old","new","hard","soft",
                  "bright","dark","strong","weak","happy","sad","good","bad","right","wrong"],
    "comparative": ["faster","slower","bigger","smaller","better","worse",
                    "biggest","smallest","best","worst"],
    "function": ["the","a","and","or","not","is","was","in","on","of","to","from",
                 "with","for","he","she","it","they","we","I","you","his","her"],
    "gender_pair": ["king","queen","man","woman","boy","girl","brother","sister",
                    "father","mother","son","daughter","husband","wife","prince","princess"],
    "abstract": ["love","truth","beauty","freedom","power","time","space","mind","body",
                 "soul","life","death","hope","fear","joy","pain","trust","faith","peace"],
    "plural_noun": ["dogs","cats","trees","birds","horses","men","women","children","hands","eyes"],
    "hypernym": ["animal","vehicle","tool","gem","burden","barrier","journey","bond"],
}

PROBE_TOKENS = list(dict.fromkeys([
    "dog","cat","bird","fish","horse","wolf","lion","tiger","elephant","mouse",
    "rabbit","deer","bear","fox","eagle","whale","shark","frog","ant","bee",
    "snake","monkey","cow","pig","sheep","goat","duck","hen","crow","owl",
    "turtle","lizard","crab","lobster","octopus","beetle","butterfly","worm",
    "fly","mosquito","cricket","spider","salmon","tuna","herring","sparrow",
    "robin","finch","parrot","tree","flower","rock","stone","wood","leaf",
    "grass","root","river","mountain","ocean","forest","desert","cloud","rain",
    "snow","wind","sun","moon","star","sky","earth","soil","seed","branch",
    "bark","thorn","moss","mushroom","coral","house","door","window","table",
    "chair","book","cup","key","car","road","bridge","boat","ship","plane",
    "train","bike","knife","fork","spoon","plate","bowl","glass","bottle","box",
    "bag","rope","wire","nail","hammer","wheel","clock","lamp","pen","paper",
    "cloth","thread","button","ring","coin","mirror","hand","foot","eye","ear",
    "nose","mouth","arm","leg","head","heart","blood","bone","skin","hair",
    "finger","toe","back","chest","neck","shoulder","run","walk","jump","swim",
    "fly","eat","sleep","talk","write","read","build","break","open","close",
    "start","stop","think","know","see","hear","feel","love","hate","want",
    "give","take","make","find","lose","push","pull","turn","move","go","come",
    "fall","rise","grow","kill","help","ran","walked","jumped","flew","ate",
    "saw","heard","broke","built","wrote","fast","slow","big","small","hot",
    "cold","old","new","hard","soft","bright","dark","strong","weak","happy",
    "sad","good","bad","right","wrong","high","low","long","short","wide",
    "narrow","deep","shallow","thick","thin","heavy","light","clean","dirty",
    "sweet","bitter","sharp","dull","loud","quiet","faster","slower","bigger",
    "smaller","better","worse","biggest","smallest","best","worst","quickly",
    "slowly","often","never","always","very","quite","really","just","still",
    "the","a","and","or","not","is","was","in","on","of","to","from","with",
    "for","he","she","it","they","we","I","you","his","her","their","my","your",
    "its","our","but","if","one","two","three","four","five","six","seven",
    "eight","nine","ten","hundred","thousand","many","few","more","less","most",
    "least","all","some","king","queen","man","woman","boy","girl","child",
    "parent","brother","sister","father","mother","son","daughter","husband",
    "wife","prince","princess","actor","actress","red","blue","green","yellow",
    "white","black","brown","orange","purple","pink","gray","gold","love","hate",
    "truth","beauty","freedom","power","time","space","mind","body","soul",
    "life","death","hope","fear","joy","pain","trust","faith","peace","war",
    "law","right","duty","honor","shame","pride","guilt","anger","grief","city",
    "town","village","country","island","valley","cave","bridge","castle",
    "market","church","school","hospital","garden","field","park","lake",
    "coast","cliff","path","bread","meat","fruit","milk","water","fire","oil",
    "salt","sugar","coffee","wine","beer","tea","egg","cheese","dogs","cats",
    "trees","birds","horses","men","women","children","hands","eyes",
    "animal","vehicle","tool","gem","burden","barrier","journey","bond",
    "large","tiny","quick","frigid","joyful","difficult","aged","noisy",
    "oak","rose","ruby",
]))

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
n_heads     = model.config.num_attention_heads
n_kv_heads  = model.config.num_key_value_heads
head_dim    = hidden_size // n_heads
ALL_LAYERS  = sorted(set(DAY78_LAYERS.values()))
print(f"  hidden={hidden_size}\n")

def get_last_h(text, layer):
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    pos = inp["input_ids"].shape[1] - 1
    return out.hidden_states[layer][0, pos, :].numpy().astype(np.float32)

print("Computing T2 axes ...")
t2_axes = {}
for name in AXIS_NAMES_12:
    L = DAY78_LAYERS[name]
    diffs = []
    for s1, s2 in AXIS_SENTENCE_PAIRS.get(name, []):
        try:
            h1 = get_last_h(s1, L); h2 = get_last_h(s2, L)
            d  = h2 - h1; nrm = np.linalg.norm(d)
            if nrm > 1e-6: diffs.append(d / nrm)
        except: pass
    v  = np.mean(diffs, axis=0) if diffs else np.zeros(hidden_size, dtype=np.float32)
    nv = np.linalg.norm(v)
    t2_axes[name] = (v / nv if nv > 1e-6 else np.zeros(hidden_size)).astype(np.float32)

print("Computing d_k (H6 L23) entity selector direction ...")
L23     = model.model.layers[22]
W_k_L23 = L23.self_attn.k_proj.weight.data.float().numpy()
kv_grp  = n_heads // n_kv_heads
kvi     = 6 // kv_grp
h6k     = W_k_L23[kvi*head_dim : (kvi+1)*head_dim, :]
Uk,_,_  = np.linalg.svd(h6k, full_matrices=False)
d_k     = (h6k.T @ Uk[:, 0]).astype(np.float32)
d_k    /= np.linalg.norm(d_k)

print("Extracting probe token hidden states (all required layers) ...")
hs_by_layer = {L: [] for L in ALL_LAYERS}
hs_L23      = []
valid_words = []
for word in PROBE_TOKENS:
    try:
        inp = tok(" " + word.strip(), return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        pos = inp["input_ids"].shape[1] - 1
        for L in ALL_LAYERS:
            hs_by_layer[L].append(out.hidden_states[L][0, pos, :].numpy().astype(np.float32))
        hs_L23.append(out.hidden_states[23][0, pos, :].numpy().astype(np.float32))
        valid_words.append(word)
    except: pass
for L in ALL_LAYERS: hs_by_layer[L] = np.array(hs_by_layer[L], dtype=np.float32)
hs_L23 = np.array(hs_L23, dtype=np.float32)
N = len(valid_words)
word_idx = {w: i for i, w in enumerate(valid_words)}
print(f"  {N} tokens\n")

# Normalize per-layer hidden states
def normed(arr):
    nrm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(nrm, 1e-8)

hs_normed = {L: normed(hs_by_layer[L]) for L in ALL_LAYERS}
hs_L23_n  = normed(hs_L23)

# ── CORRECTED: Per-axis projection using correct layer ────────────────────────
# t2_projections[i, k] = h_word_i(L_axis_k) · t2_axis_k
t2_projections = np.zeros((N, 12), dtype=np.float32)
for k, ax_name in enumerate(AXIS_NAMES_12):
    L = DAY78_LAYERS[ax_name]
    t2_projections[:, k] = hs_normed[L] @ t2_axes[ax_name]

t2_magnitude = np.linalg.norm(t2_projections, axis=1)  # (N,)

# d_k projection using L23 (where entity selector operates)
dk_projection = np.abs(hs_L23_n @ d_k)  # (N,)

# ── Exp 1: Per-category corrected projection statistics ───────────────────────
print("=" * 72)
print("Exp 1: Category Projection (CORRECTED per-axis layer)")
print("=" * 72)
print(f"\n  {'category':>15}  {'n':>4}  {'T2_mag_mean':>12}  {'dk_abs_mean':>12}  {'T2/dk':>8}")
print(f"  {'-'*56}")

cat_results = {}
for cat_name, words in TOKEN_CATEGORIES.items():
    idxs = [word_idx[w] for w in words if w in word_idx]
    if not idxs: continue
    t2_m  = float(np.mean(t2_magnitude[idxs]))
    dk_m  = float(np.mean(dk_projection[idxs]))
    ratio = t2_m / max(dk_m, 1e-8)
    cat_results[cat_name] = {"t2_mag": t2_m, "dk_abs": dk_m, "ratio": ratio, "n": len(idxs)}
    print(f"  {cat_name:>15}  {len(idxs):>4}  {t2_m:>12.4f}  {dk_m:>12.4f}  {ratio:>8.2f}")

overall_t2 = float(np.mean(t2_magnitude))
overall_dk = float(np.mean(dk_projection))
print(f"  {'ALL':>15}  {N:>4}  {overall_t2:>12.4f}  {overall_dk:>12.4f}  "
      f"{overall_t2/max(overall_dk,1e-8):>8.2f}")

# ── Exp 2: Per-axis top words (CORRECTED) ─────────────────────────────────────
print()
print("=" * 72)
print("Exp 2: Per-Axis Top/Bottom 5 Words (CORRECTED)")
print("=" * 72)
print()

axis_top_words = {}
for k, ax_name in enumerate(AXIS_NAMES_12):
    projs_ax = t2_projections[:, k]
    top_pos  = np.argsort(-projs_ax)[:5]
    top_neg  = np.argsort(projs_ax)[:5]
    h_words  = [(valid_words[i], float(projs_ax[i])) for i in top_pos]
    l_words  = [(valid_words[i], float(projs_ax[i])) for i in top_neg]
    axis_top_words[ax_name] = {"H_pole": h_words, "L_pole": l_words}
    h_str = ", ".join(f"{w}({v:+.2f})" for w,v in h_words)
    l_str = ", ".join(f"{w}({v:+.2f})" for w,v in l_words)
    L_ax  = DAY78_LAYERS[ax_name]
    print(f"  {ax_name:>14} [L{L_ax}]  H: {h_str}")
    print(f"  {' '*14}         L: {l_str}")
    print()

# ── Exp 3: Top tokens by T2 vs d_k (CORRECTED) ────────────────────────────────
print("=" * 72)
print("Exp 3: Top 15 Tokens by T2 Magnitude and d_k Projection (CORRECTED)")
print("=" * 72)

sorted_t2 = np.argsort(-t2_magnitude)
sorted_dk = np.argsort(-dk_projection)
print(f"\n  {'T2_rank':>8}  {'word':>12}  {'T2_mag':>8}  |  {'dk_rank':>8}  {'word':>12}  {'dk_abs':>8}")
for i in range(15):
    it2 = sorted_t2[i]; idk = sorted_dk[i]
    print(f"  {i+1:>8}  {valid_words[it2]:>12}  {t2_magnitude[it2]:>8.4f}  |  "
          f"{i+1:>8}  {valid_words[idk]:>12}  {dk_projection[idk]:>8.4f}")

# ── Exp 4: Category alignment with specific axes ──────────────────────────────
print()
print("=" * 72)
print("Exp 4: Category Alignment with Specific Axes (Hypothesis Test)")
print("=" * 72)

AXIS_HYPOTHESES = {
    "gender":    ("gender_pair", "gender_pair should rank HIGH on gender axis"),
    "comparative": ("comparative", "comparative words should rank HIGH on comparative axis"),
    "past_tense": ("verb_past", "past tense verbs should rank HIGH on past_tense axis"),
    "plural":    ("plural_noun", "plural nouns should rank HIGH on plural axis"),
    "hypernym":  ("hypernym", "hypernym words (animal, tool) should rank HIGH on hypernym axis"),
}

print()
for ax_name, (cat_name, hypothesis) in AXIS_HYPOTHESES.items():
    if cat_name not in TOKEN_CATEGORIES: continue
    k = AXIS_NAMES_12.index(ax_name)
    L = DAY78_LAYERS[ax_name]
    projs_ax = t2_projections[:, k]
    # Cat words
    cat_idxs   = [word_idx[w] for w in TOKEN_CATEGORIES[cat_name] if w in word_idx]
    other_idxs = [i for i in range(N) if i not in set(cat_idxs)]
    if not cat_idxs or not other_idxs: continue
    cat_mean   = float(np.mean(projs_ax[cat_idxs]))
    other_mean = float(np.mean(projs_ax[other_idxs]))
    # Rank of category words among all tokens
    rank_arr   = np.argsort(-projs_ax)
    ranks      = {valid_words[idx]: int(np.where(rank_arr == idx)[0][0])+1 for idx in cat_idxs}
    top5_words = [(valid_words[rank_arr[i]], float(projs_ax[rank_arr[i]])) for i in range(5)]
    print(f"  {ax_name} axis [L{L}]:")
    print(f"    {cat_name} mean={cat_mean:+.4f}  other mean={other_mean:+.4f}  "
          f"delta={cat_mean-other_mean:+.4f}")
    top5_str = ", ".join(f"{w}({v:+.2f})" for w,v in top5_words)
    print(f"    Top 5 globally: {top5_str}")
    cat_ranks_str = ", ".join(f"{w}=#{r}" for w,r in sorted(ranks.items(), key=lambda x: x[1])[:6])
    print(f"    {cat_name} ranks: {cat_ranks_str}")
    print(f"    HYPOTHESIS: {'CONFIRMED' if cat_mean > other_mean + 0.01 else 'NOT CONFIRMED'}")
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 114b Summary — Corrected Vocabulary Projection")
print("=" * 72)

sorted_cats = sorted(cat_results.items(), key=lambda x: -x[1]["t2_mag"])
highest_t2_cat = sorted_cats[0][0]
lowest_t2_cat  = sorted_cats[-1][0]

confirmed = []
not_confirmed = []
for ax_name, (cat_name, hyp) in AXIS_HYPOTHESES.items():
    if cat_name not in TOKEN_CATEGORIES or cat_name not in cat_results: continue
    k = AXIS_NAMES_12.index(ax_name)
    projs_ax = t2_projections[:, k]
    cat_idxs   = [word_idx[w] for w in TOKEN_CATEGORIES[cat_name] if w in word_idx]
    other_idxs = [i for i in range(N) if i not in set(cat_idxs)]
    if not cat_idxs or not other_idxs: continue
    cat_mean   = float(np.mean(projs_ax[cat_idxs]))
    other_mean = float(np.mean(projs_ax[other_idxs]))
    if cat_mean > other_mean + 0.01:
        confirmed.append(ax_name)
    else:
        not_confirmed.append(ax_name)

print(f"""
  CORRECTED T2 projection (per-axis layer):
    Highest T2 magnitude category: {highest_t2_cat}
    Lowest  T2 magnitude category: {lowest_t2_cat}
    Overall T2 mean: {overall_t2:.4f}
    Overall d_k mean: {overall_dk:.4f}

  Axis-category hypothesis tests:
    CONFIRMED:     {', '.join(confirmed) if confirmed else 'none'}
    NOT CONFIRMED: {', '.join(not_confirmed) if not_confirmed else 'none'}

  Top token by T2 (corrected): {valid_words[sorted_t2[0]]} ({t2_magnitude[sorted_t2[0]]:.4f})
  Top token by d_k:            {valid_words[sorted_dk[0]]} ({dk_projection[sorted_dk[0]]:.4f})

  VERDICT:
  {'→ Axis hypotheses CONFIRMED: semantic categories project onto intended axes' if len(confirmed) >= 3 else
   '→ Axis hypotheses PARTIALLY CONFIRMED: some axes capture intended semantics' if len(confirmed) >= 1 else
   '→ Axis hypotheses NOT CONFIRMED even with corrected layer projection'}

  KEY FINDING:
  {'The T2 axes correctly capture semantic category membership when projected' if len(confirmed) >= 3 else
   'The T2 axes partially capture semantic categories at the correct layer' if len(confirmed) >= 1 else
   'The T2 axes do not cleanly separate semantic categories even at correct layers.'}
  {'This validates DC 331: T2 categorical subspace correctly characterizes token types.' if len(confirmed) >= 3 else ''}
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "cat_results": cat_results,
        "axis_top_words": axis_top_words,
        "confirmed_hypotheses": confirmed,
        "not_confirmed": not_confirmed,
        "top_t2_tokens": [(valid_words[i], float(t2_magnitude[i])) for i in sorted_t2[:20]],
        "top_dk_tokens": [(valid_words[i], float(dk_projection[i])) for i in sorted_dk[:20]],
        "overall_t2": overall_t2, "overall_dk": overall_dk,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 114b complete.")
