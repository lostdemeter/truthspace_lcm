#!/usr/bin/env python3
"""
Day 188 — W_E Coverage and Gaps

QUESTION: How complete is W_E as a knowledge store?

EXPERIMENT 1: Single-token fraction of the 151k vocabulary
  - How many unique English words (from a reference list) tokenize to 1 token?
  - What fraction of common English words are "first-class citizens" in W_E?
  - How does this scale with word frequency?

EXPERIMENT 2: Relation pair coverage
  - For common relational domains, what fraction of pairs have BOTH
    source and target as single tokens?
  - country→capital: "Philippines"→"Manila" — both single token?

EXPERIMENT 3: Multi-token composition
  - For words that tokenize to 2-3 tokens, can we compose the
    meaning from sub-token embeddings?
  - mean(sub-tokens) vs single-token for known synonyms?
  - Test: does mean(W_E["Ber"] + W_E["lin"]) ≈ W_E["Berlin"]?
  - Test: does the relation direction still work if source/target
    are averaged sub-token embeddings?

REFERENCE WORD LISTS:
  - Use the 5000 most common English words (from frequency data embedded here)
  - Country names, capital cities, common nouns, verbs
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day188_we_coverage.json")
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"

COMMON_WORDS_1000 = [
    "the","be","to","of","and","a","in","that","have","it","for","not","on","with",
    "he","as","you","do","at","this","but","his","by","from","they","we","say","her",
    "she","or","an","will","my","one","all","would","there","their","what","so",
    "up","out","if","about","who","get","which","go","me","when","make","can","like",
    "time","no","just","him","know","take","people","into","year","your","good","some",
    "could","them","see","other","than","then","now","look","only","come","its","over",
    "think","also","back","after","use","two","how","our","work","first","well","way",
    "even","new","want","because","any","these","give","day","most","us","great","between",
    "need","large","often","hand","high","place","hold","turn","were","much","before",
    "here","more","through","long","same","another","small","those","however","three",
    "number","come","again","point","city","play","small","world","still","own","where",
    "old","life","tell","write","become","show","leave","why","ask","change","men",
    "need","house","picture","try","again","animal","point","mother","word","answer",
    "found","study","still","learn","plant","cover","food","sun","four","between",
    "state","keep","never","last","let","thought","city","tree","cross","farm","hard",
    "start","might","story","far","sea","draw","left","late","run","dont","while",
    "press","close","night","real","few","north","near","open","seem","together","next",
    "white","children","begin","got","walk","example","ease","paper","group","always",
    "music","those","both","mark","book","letter","until","mile","river","car","feet",
    "care","second","enough","plain","girl","usual","young","ready","above","ever",
    "red","list","though","feel","talk","bird","soon","body","dog","family","direct",
    "pose","song","measure","door","product","black","short","numeral","class","wind",
    "question","happen","complete","ship","area","half","rock","order","fire","south",
    "problem","piece","told","knew","pass","since","top","whole","king","space","heard",
    "best","hour","better","true","during","hundred","five","remember","step","early",
    "hold","west","ground","interest","reach","fast","verb","sing","listen","six",
    "table","travel","less","morning","ten","simple","several","vowel","toward","war",
    "lay","against","pattern","slow","center","love","person","money","serve","appear",
    "road","map","rain","rule","govern","pull","cold","notice","voice","fall","power",
    "town","fine","drive","lead","cry","dark","machine","note","wait","plan","figure",
    "star","box","noun","field","rest","able","pound","done","beauty","drive","stood",
    "contain","front","teach","week","final","gave","green","quick","develop","ocean",
    "warm","free","minute","strong","special","mind","behind","clear","tail","produce",
    "fact","street","inch","multiply","nothing","course","stay","wheel","full","force",
    "blue","object","decide","surface","deep","moon","island","foot","system","busy",
    "test","record","boat","common","gold","possible","plane","age","dry","wonder",
    "laugh","thousand","ago","ran","check","game","shape","equate","hot","miss","bring",
    "heat","snow","tire","bring","yes","distant","fill","east","paint","language",
    "among","grand","ball","yet","wave","drop","heart","am","present","heavy","dance",
    "engine","position","arm","wide","sail","material","size","vary","settle","speak",
    "weight","general","ice","matter","circle","pair","include","divide","syllable",
    "felt","perhaps","pick","sudden","count","square","reason","length","represent",
    "heart","show","yes","clear","push","explain","sleep","knew","tall","sand","soil",
    "roll","temperature","finger","industry","value","fight","lie","beat","excite","natural",
    "view","sense","ear","else","quite","broke","case","middle","kill","son","lake",
    "moment","scale","loud","spring","observe","child","straight","consonant","nation",
    "dictionary","milk","speed","method","organ","pay","age","section","dress","cloud",
    "surprise","quiet","stone","tiny","climb","cool","design","poor","lot","experiment",
    "bottom","key","iron","single","stick","flat","twenty","skin","smile","crease",
    "hole","trade","melody","trip","office","receive","row","mouth","exact","symbol",
    "die","least","trouble","shout","except","wrote","seed","tone","join","suggest",
    "clean","break","lady","yard","rise","bad","blow","oil","blood","touch","grew",
    "cent","mix","team","wire","cost","lost","brown","wear","garden","equal","sent",
    "choose","fell","fit","flow","fair","bank","collect","save","control","decimal",
    "gentle","woman","captain","practice","separate","difficult","doctor","please","protect",
    "noon","whose","locate","ring","character","insect","caught","period","indicate",
    "radio","spoke","atom","human","history","effect","electric","expect","crop","modern",
    "element","hit","student","corner","party","supply","bone","rail","imagine","provide",
    "agree","thus","capital","chair","danger","fruit","rich","thick","soldier","process",
    "operate","guess","necessary","sharp","wing","create","neighbor","wash","bat","rather",
    "crowd","corn","compare","poem","string","bell","depend","meat","rub","tube","famous",
    "dollar","stream","fear","sight","thin","triangle","planet","hurry","chief","colony",
    "clock","mine","tie","enter","major","fresh","search","send","yellow","gun","allow",
    "print","dead","spot","desert","suit","current","lift","rose","continue","block",
    "chart","hat","sell","success","company","subtract","event","particular","deal",
    "swim","term","opposite","wife","shoe","shoulder","spread","arrange","camp","invent",
    "cotton","born","determine","quart","nine","truck","noise","level","chance","gather",
    "shop","stretch","throw","shine","property","column","molecule","select","wrong",
    "gray","repeat","require","broad","prepare","salt","nose","plural","anger","claim",
]

# Country→Capital pairs (broader than Day 182)
COUNTRY_CAPITAL_PAIRS = [
    ("France","Paris"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),
    ("Japan","Tokyo"),("China","Beijing"),("Russia","Moscow"),("Greece","Athens"),
    ("Poland","Warsaw"),("Sweden","Stockholm"),("Korea","Seoul"),("Brazil","Brasilia"),
    ("Canada","Ottawa"),("India","Delhi"),("Turkey","Ankara"),("Egypt","Cairo"),
    ("Mexico","Mexico"),("Argentina","Buenos"),("Nigeria","Abuja"),("Kenya","Nairobi"),
    ("Australia","Canberra"),("Indonesia","Jakarta"),("Pakistan","Islamabad"),
    ("Bangladesh","Dhaka"),("Philippines","Manila"),("Vietnam","Hanoi"),
    ("Thailand","Bangkok"),("Iran","Tehran"),("Iraq","Baghdad"),("Saudi","Riyadh"),
    ("Ukraine","Kyiv"),("Romania","Bucharest"),("Hungary","Budapest"),
    ("Portugal","Lisbon"),("Netherlands","Amsterdam"),("Belgium","Brussels"),
    ("Switzerland","Bern"),("Austria","Vienna"),("Denmark","Copenhagen"),
    ("Finland","Helsinki"),("Norway","Oslo"),
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
print(f"  Vocab size: {W_E.shape[0]}, H={W_E.shape[1]}\n")

def tok_ids(word):
    return tok(" "+word, add_special_tokens=False)["input_ids"]

def is_single(word):
    return len(tok_ids(word)) == 1

def emb(word):
    ids = tok_ids(word)
    if len(ids) == 1:
        return W_E[ids[0]]
    return np.mean([W_E[i] for i in ids], axis=0)

# ── Experiment 1: Single-token fraction ─────────────────────────────
print("Experiment 1: Single-token fraction of common English words")
print("-" * 60)
words = list(dict.fromkeys(COMMON_WORDS_1000))  # dedup while preserving order
n_total = len(words)
single_ids = [(w, tok_ids(w)[0]) for w in words if is_single(w)]
multi_words = [(w, tok_ids(w)) for w in words if not is_single(w)]
n_single = len(single_ids)
print(f"  Total words tested:   {n_total}")
print(f"  Single-token words:   {n_single} ({100*n_single/n_total:.1f}%)")
print(f"  Multi-token words:    {n_total-n_single} ({100*(n_total-n_single)/n_total:.1f}%)")
print()

# Show some multi-token examples
print("  Multi-token word examples (first 20):")
for w, ids in multi_words[:20]:
    toks = tok.convert_ids_to_tokens(ids)
    print(f"    {w:<20} → {toks}")
print()

# ── Experiment 2: Relation pair coverage ────────────────────────────
print("Experiment 2: Relation pair coverage (country→capital)")
print("-" * 60)
n_both = sum(1 for a, b in COUNTRY_CAPITAL_PAIRS
             if is_single(a) and is_single(b))
n_src  = sum(1 for a, b in COUNTRY_CAPITAL_PAIRS if is_single(a))
n_tgt  = sum(1 for a, b in COUNTRY_CAPITAL_PAIRS if is_single(b))
n_pairs = len(COUNTRY_CAPITAL_PAIRS)
print(f"  Total pairs:          {n_pairs}")
print(f"  Source single-token:  {n_src}  ({100*n_src/n_pairs:.1f}%)")
print(f"  Target single-token:  {n_tgt}  ({100*n_tgt/n_pairs:.1f}%)")
print(f"  Both single-token:    {n_both} ({100*n_both/n_pairs:.1f}%)")
print()
print("  Multi-token cases:")
for a, b in COUNTRY_CAPITAL_PAIRS:
    if not (is_single(a) and is_single(b)):
        ta = tok.convert_ids_to_tokens(tok_ids(a))
        tb = tok.convert_ids_to_tokens(tok_ids(b))
        print(f"    {a}→{b}: {ta}→{tb}")
print()

# ── Experiment 3: Sub-token composition ─────────────────────────────
print("Experiment 3: Sub-token composition accuracy")
print("-" * 60)

# Test: do multi-token cities approximate their W_E representation
# by comparing mean(sub-tokens) cosine similarity to single-token equivalent?
multi_token_cities = [(a, b) for a, b in COUNTRY_CAPITAL_PAIRS
                      if is_single(a) and not is_single(b)]
print(f"  Testing {len(multi_token_cities)} pairs where capital is multi-token")
print()

# Build vocabulary of single-token capitals + mean-composed multi-token capitals
ok_single_pairs = [(a, b) for a, b in COUNTRY_CAPITAL_PAIRS
                   if is_single(a) and is_single(b)]
ok_single_pairs = ok_single_pairs[:12]  # cap for LOO

print(f"  Single-token capital pairs available for direction: {len(ok_single_pairs)}")
if len(ok_single_pairs) >= 3:
    diffs = [normed(W_E[tok_ids(b)[0]] - W_E[tok_ids(a)[0]])
             for a, b in ok_single_pairs]
    mean_dir = normed(np.mean(diffs, axis=0))

    print("  Applying single-token capital direction to multi-token cases:")
    for a, b in multi_token_cities[:10]:
        if not is_single(a): continue
        sub_ids = tok_ids(b)
        composed = normed(np.mean([W_E[i] for i in sub_ids], axis=0))
        query = normed(W_E[tok_ids(a)[0]] + mean_dir.astype(np.float32))
        sim = cosine(query, composed)
        sub_toks = tok.convert_ids_to_tokens(sub_ids)
        print(f"    {a}→{b} (sub:{sub_toks}): cos(query,composed)={sim:.4f}")

print()

# How well does sub-token composition approximate single-token meaning?
print("  Sub-token mean vs single-token (for words that exist as both):")
# Test: token "ly" suffix words — "quickly" vs "quick"+"ly"
suffix_tests = [
    ("quickly",), ("slowly",), ("freely",), ("easily",),
    ("finally",), ("simply",), ("nearly",), ("clearly",),
]
for (word,) in suffix_tests:
    if is_single(word):
        single_emb = W_E[tok_ids(word)[0]]
        ids = tok_ids(word)
        print(f"  {word}: single-token (id={ids[0]})")
    else:
        ids = tok_ids(word)
        sub_toks = tok.convert_ids_to_tokens(ids)
        composed = np.mean([W_E[i] for i in ids], axis=0)
        # No ground truth to compare to, but check if sub-token average
        # is near the centroid of semantically similar words
        print(f"  {word}: multi-token {sub_toks} → composed emb norm={np.linalg.norm(composed):.3f}")

results = {
    "exp1": {
        "n_total": n_total,
        "n_single": n_single,
        "single_fraction": n_single/n_total,
        "multi_examples": [(w, tok.convert_ids_to_tokens(ids))
                           for w, ids in multi_words[:30]],
    },
    "exp2": {
        "n_pairs": n_pairs,
        "n_both_single": n_both,
        "coverage_fraction": n_both/n_pairs,
    }
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_FILE}")
print("Day 188 complete.")
