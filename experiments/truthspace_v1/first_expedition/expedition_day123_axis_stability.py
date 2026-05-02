#!/usr/bin/env python3
"""
Day 123 — T2 Axis Context-Stability Test

QUESTION: Is the T2 axis direction STABLE across different sentence contexts,
or does it vary depending on which sentence pairs are used?

Day 122 showed that the T2 method (sentence-pair causal ablation) is
superior to MD for semantic axes. But the T2 axes were computed from specific
sentence pairs (e.g., "The king ruled..." / "The queen ruled...").

If the semantic direction is a UNIVERSAL INTRINSIC PROPERTY of the LM,
then computing the same axis from DIFFERENT sentence pairs should yield
THE SAME DIRECTION.

EXPERIMENT: For each T2 axis, compute 10 independent axis estimates using
different non-overlapping sentence pairs. Measure:
  1. Pairwise cosine similarity of the 10 estimates
  2. Mean pairwise cosine = axis stability score
  3. Stability vs classification accuracy correlation

TruthSpace self-similarity prediction:
  High stability (cos > 0.5) for all axes — the direction is universal.

Null hypothesis (context-specific):
  Low stability (cos < 0.3) — direction varies with sentence choice.

Also test: does the mean of all 10 estimates outperform any single estimate?
  (If yes: more sentence pairs always helps — the axis is a fixed point)
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_FILE = str(SCRIPT_DIR / "day123_axis_stability.json")
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

# For each axis: 20 sentence pairs split into 10 groups of 2
# Group i uses pairs [2i, 2i+1]; this gives 10 independent single-pair estimates
AXIS_PAIRS_ALL = {
    "gender": [
        ("The king ruled with great wisdom","The queen ruled with great wisdom"),
        ("A man walked through the forest","A woman walked through the forest"),
        ("The boy kicked the ball hard","The girl kicked the ball hard"),
        ("His brother arrived at the party","His sister arrived at the party"),
        ("The father worked to feed family","The mother worked to feed family"),
        ("The actor played a leading role","The actress played a leading role"),
        ("The prince rode across the land","The princess rode across the land"),
        ("The uncle visited the old house","The aunt visited the old house"),
        ("The wizard cast a powerful spell","The witch cast a powerful spell"),
        ("The lord welcomed his dear guests","The lady welcomed her dear guests"),
        ("The monk meditated in the temple","The nun meditated in the temple"),
        ("The husband cooked the evening meal","The wife cooked the evening meal"),
        ("The grandfather told an old story","The grandmother told an old story"),
        ("The nephew helped with the harvest","The niece helped with the harvest"),
        ("His son won the swimming contest","His daughter won the swimming contest"),
        ("The duke presided at the banquet","The duchess presided at the banquet"),
        ("The emperor marched into the city","The empress marched into the city"),
        ("The hero saved the whole village","The heroine saved the whole village"),
        ("The stallion galloped across plains","The mare galloped across plains"),
        ("The bachelor lived alone in peace","The spinster lived alone in peace"),
    ],
    "comparative": [
        ("The fast car went down the road","The faster car went down the road"),
        ("A big dog sat by the fireplace","A bigger dog sat by the fireplace"),
        ("The cold wind blew from the north","The colder wind blew from the north"),
        ("A tall tree stood in the garden","A taller tree stood in the garden"),
        ("The old house sat on the corner","The older house sat on the corner"),
        ("A bright star shone in the sky","A brighter star shone in the sky"),
        ("The dark room was hard to see in","The darker room was hard to see in"),
        ("A hard rock lay on the ground","A harder rock lay on the ground"),
        ("The warm sun rose over the hills","The warmer sun rose over the hills"),
        ("A clean sheet lay on the table","A cleaner sheet lay on the table"),
        ("The soft breeze touched his cheek","The softer breeze touched his cheek"),
        ("A long path wound through the park","A longer path wound through the park"),
        ("The strong man lifted the heavy box","The stronger man lifted the heavy box"),
        ("A fresh apple sat on the counter","A fresher apple sat on the counter"),
        ("The short book was easy to read","The shorter book was easy to read"),
        ("A quick fox ran past the hedge","A quicker fox ran past the hedge"),
        ("The smart student answered first","The smarter student answered first"),
        ("A heavy stone blocked the gateway","A heavier stone blocked the gateway"),
        ("The thin ice cracked underfoot","The thinner ice cracked underfoot"),
        ("A rich dessert sat on the plate","A richer dessert sat on the plate"),
    ],
    "plural": [
        ("A dog played happily in the field","Dogs played happily in the field"),
        ("The cat sat quietly by the window","The cats sat quietly by the window"),
        ("A bird sang softly in the morning","Birds sang softly in the morning"),
        ("The tree fell down in the storm","The trees fell down in the storm"),
        ("A book sat open on the old desk","Books sat open on the old desk"),
        ("The car drove slowly down the road","The cars drove slowly down the road"),
        ("A star shone brightly in the sky","Stars shone brightly in the sky"),
        ("The word appeared clearly in text","The words appeared clearly in text"),
        ("A child ran out into the garden","Children ran out into the garden"),
        ("The house stood on a quiet hill","The houses stood on a quiet hill"),
        ("A boat sailed across the calm lake","Boats sailed across the calm lake"),
        ("The cup sat full on the table top","The cups sat full on the table top"),
        ("A leaf fell from the autumn tree","Leaves fell from the autumn tree"),
        ("The stone rolled down the hillside","The stones rolled down the hillside"),
        ("A hand reached out in the darkness","Hands reached out in the darkness"),
        ("The eye could see far in clear air","The eyes could see far in clear air"),
        ("A thought crossed his tired mind","Thoughts crossed his tired mind"),
        ("The year passed quickly and quietly","The years passed quickly and quietly"),
        ("A cloud drifted over the mountain","Clouds drifted over the mountain"),
        ("The key fit perfectly in the lock","The keys fit perfectly in the lock"),
    ],
    "past_tense": [
        ("I walk to the market every morning","I walked to the market every morning"),
        ("She runs through the park after work","She ran through the park after work"),
        ("He eats breakfast before leaving home","He ate breakfast before leaving home"),
        ("They build a wall around the garden","They built a wall around the garden"),
        ("We swim in the lake on summer days","We swam in the lake on summer days"),
        ("She writes a letter to her friend","She wrote a letter to her friend"),
        ("He speaks quietly at the meeting","He spoke quietly at the meeting"),
        ("They sing together at the campfire","They sang together at the campfire"),
        ("The bird flies south for the winter","The bird flew south for the winter"),
        ("The child breaks the old glass jar","The child broke the old glass jar"),
        ("She takes the bus to the office","She took the bus to the office"),
        ("He gives money to the old beggar","He gave money to the old beggar"),
        ("They make a cake for the birthday","They made a cake for the birthday"),
        ("She goes to the park every evening","She went to the park every evening"),
        ("He comes home late on most Fridays","He came home late on most Fridays"),
        ("The sun rises over the eastern hill","The sun rose over the eastern hill"),
        ("The rain falls hard against the roof","The rain fell hard against the roof"),
        ("She sees the answer right away","She saw the answer right away"),
        ("He knows the truth about the matter","He knew the truth about the matter"),
        ("The flower grows tall by the fence","The flower grew tall by the fence"),
    ],
    "antonym": [
        ("The weather is hot and humid today","The weather is cold and humid today"),
        ("He runs very fast around the track","He runs very slow around the track"),
        ("The news today is very good indeed","The news today is very bad indeed"),
        ("She feels extremely happy right now","She feels extremely sad right now"),
        ("The man is incredibly strong and fit","The man is incredibly weak and fit"),
        ("The road is very long and winding","The road is very short and winding"),
        ("The night is very dark and moonless","The night is very light and moonless"),
        ("The door is left open for visitors","The door is left closed for visitors"),
        ("The price is incredibly high right now","The price is incredibly low right now"),
        ("The surface feels very rough to touch","The surface feels very smooth to touch"),
        ("The water in the pool is very warm","The water in the pool is very cold"),
        ("The bag is quite heavy to carry around","The bag is quite light to carry around"),
        ("The answer was right on the first try","The answer was wrong on the first try"),
        ("The crowd was loud and full of energy","The crowd was quiet and full of energy"),
        ("The task turned out to be quite easy","The task turned out to be quite hard"),
        ("The room felt very clean and orderly","The room felt very dirty and orderly"),
        ("Her voice was soft and very gentle","Her voice was loud and very gentle"),
        ("The fruit tasted sweet and refreshing","The fruit tasted bitter and refreshing"),
        ("The child was brave and did not cry","The child was afraid and did not cry"),
        ("The stone wall stood firm and rigid","The wooden fence stood firm and rigid"),
    ],
    "negation": [
        ("The dog is fast and energetic","The dog is not fast and energetic"),
        ("She can swim very well in the pool","She cannot swim very well in the pool"),
        ("He knows the answer to the question","He does not know the answer to the question"),
        ("The food here is quite good today","The food here is not quite good today"),
        ("They work very hard every single day","They do not work very hard every day"),
        ("The water in the river is cold today","The water in the river is not cold today"),
        ("The house looks old and worn down","The house does not look old and worn down"),
        ("It will rain heavily this afternoon","It will not rain heavily this afternoon"),
        ("The plan worked out well in the end","The plan did not work out well in the end"),
        ("She arrived on time for the meeting","She did not arrive on time for the meeting"),
        ("He finished the long task yesterday","He did not finish the long task yesterday"),
        ("The car started on the first attempt","The car did not start on the first attempt"),
        ("They understood the complex problem","They did not understand the complex problem"),
        ("The door opened with a quiet click","The door did not open with a quiet click"),
        ("She remembered every single detail","She did not remember every single detail"),
        ("The bridge held under the great weight","The bridge did not hold under the great weight"),
        ("He passed the difficult final exam","He did not pass the difficult final exam"),
        ("The team won the important championship","The team did not win the important championship"),
        ("The sun shone brightly all afternoon","The sun did not shine brightly all afternoon"),
        ("The flower bloomed early in spring","The flower did not bloom early in spring"),
    ],
    "synonym": [
        ("He is a very big and strong man","He is a very large and strong man"),
        ("The cat is small and very quiet","The cat is tiny and very quiet"),
        ("She moves very fast on the track","She moves very quick on the track"),
        ("The air outside feels cold today","The air outside feels frigid today"),
        ("She felt very happy at the news","She felt very joyful at the news"),
        ("The problem was hard to solve","The problem was difficult to solve"),
        ("He looked sad after hearing that","He looked unhappy after hearing that"),
        ("The man was very old and tired","The man was very aged and tired"),
        ("The child was smart in the class","The child was intelligent in the class"),
        ("She was angry at what had happened","She was furious at what had happened"),
        ("He felt very tired after the run","He felt very exhausted after the run"),
        ("The room was very clean and tidy","The room was very spotless and tidy"),
        ("The family was rich and well known","The family was wealthy and well known"),
        ("The sky was dark and full of cloud","The sky was dim and full of cloud"),
        ("The stone was hard to the touch","The stone was solid to the touch"),
        ("The child was brave and did not cry","The child was courageous and did not cry"),
        ("She was kind to all she had met","She was gentle to all she had met"),
        ("The task was simple and took no time","The task was easy and took no time"),
        ("The sound was loud and hard to bear","The sound was noisy and hard to bear"),
        ("The gift was pretty and well wrapped","The gift was beautiful and well wrapped"),
    ],
    "hypernym": [
        ("The dog ran away from the danger","The animal ran away from the danger"),
        ("A rose bloomed in the spring garden","A flower bloomed in the spring garden"),
        ("The car sped past the traffic sign","The vehicle sped past the traffic sign"),
        ("The eagle soared above the tall hill","The bird soared above the tall hill"),
        ("The ruby gleamed under bright light","The gem gleamed under bright light"),
        ("The hammer struck the nail firmly","The tool struck the nail firmly"),
        ("The oak crashed down in the storm","The tree crashed down in the storm"),
        ("The salmon swam up the clear river","The fish swam up the clear river"),
        ("The rose grew tall by the old wall","The plant grew tall by the old wall"),
        ("The hawk circled high in the air","The raptor circled high in the air"),
        ("The wrench tightened the rusted bolt","The instrument tightened the rusted bolt"),
        ("The trout jumped over the low falls","The vertebrate jumped over the low falls"),
        ("The emerald sparkled on the table","The mineral sparkled on the table"),
        ("The van pulled up to the curb slowly","The transport pulled up to the curb slowly"),
        ("The tulip opened in the warm sun","The blossom opened in the warm sun"),
        ("The pine stood tall on the cold ridge","The conifer stood tall on the cold ridge"),
        ("The ant carried food to the colony","The insect carried food to the colony"),
        ("The cobra slid silently through grass","The reptile slid silently through grass"),
        ("The whale surfaced near the old boat","The mammal surfaced near the old boat"),
        ("The piano filled the room with sound","The instrument filled the room with sound"),
    ],
    "concrete": [
        ("The stone is too heavy to lift now","The burden is too heavy to lift now"),
        ("The long road leads down to the sea","The long journey leads down to the sea"),
        ("The high wall blocks the open view","The high barrier blocks the open view"),
        ("The flame slowly fades in the wind","The hope slowly fades in the wind"),
        ("The iron chain held the heavy gate","The bond held the heavy gate"),
        ("The bridge connects the two old banks","The connection spans the two old banks"),
        ("The strong root grips the dark soil","The strong base grips the dark soil"),
        ("The small key opens the locked door","The small answer opens the locked door"),
        ("The anchor held the ship in place","The foundation held the ship in place"),
        ("The thorn pierced the soft skin deep","The obstacle pierced the soft skin deep"),
        ("The wall kept out the winter cold","The boundary kept out the winter cold"),
        ("The lens focused the beam of light","The focus focused the beam of light"),
        ("The ladder reached up to the roof","The path reached up to the roof"),
        ("The thread connected the two beads","The link connected the two beads"),
        ("The mirror showed a clear reflection","The truth showed a clear reflection"),
        ("The dam blocked the flowing water","The constraint blocked the flowing water"),
        ("The rope tied the heavy cargo down","The agreement tied the heavy cargo down"),
        ("The needle threaded through the cloth","The idea threaded through the cloth"),
        ("The map guided them through the hills","The plan guided them through the hills"),
        ("The cage held the frightened bird","The system held the frightened bird"),
    ],
    "passive": [
        ("The cat chased the frightened mouse","The mouse was chased by the cat"),
        ("The chef cooked the evening meal","The meal was cooked by the chef"),
        ("The storm destroyed the old house","The house was destroyed by the storm"),
        ("The artist painted the large picture","The picture was painted by the artist"),
        ("The teacher helped the young student","The student was helped by the teacher"),
        ("The king signed the royal document","The document was signed by the king"),
        ("John broke the tall kitchen window","The window was broken by John"),
        ("The dog bit the postman on the leg","The postman was bitten by the dog"),
        ("The mechanic fixed the broken engine","The engine was fixed by the mechanic"),
        ("The farmer harvested the ripe wheat","The wheat was harvested by the farmer"),
        ("The police caught the fleeing thief","The thief was caught by the police"),
        ("The writer finished the long novel","The novel was finished by the writer"),
        ("The builder raised the stone tower","The tower was raised by the builder"),
        ("The director filmed the final scene","The scene was filmed by the director"),
        ("The nurse treated the injured child","The child was treated by the nurse"),
        ("The judge dismissed the weak case","The case was dismissed by the judge"),
        ("The wind scattered the dry leaves","The leaves were scattered by the wind"),
        ("The swimmer broke the world record","The record was broken by the swimmer"),
        ("The company launched the new product","The product was launched by the company"),
        ("The student solved the hard problem","The problem was solved by the student"),
    ],
    "causation": [
        ("The heavy rain falls all through day","The ground gets completely wet all day"),
        ("The fire burns for a very long time","The wood turns slowly to ash and dust"),
        ("The sun heats the cold frozen earth","The ice melts quickly away in spring"),
        ("The wind blows the thin tree branches","The leaves fall softly to the ground"),
        ("The child cries out very loudly now","The mother comes quickly running in"),
        ("The ball rolls off the sharp table edge","The ball falls hard to the wooden floor"),
        ("The teacher praises the hard work done","The student feels proud of the effort"),
        ("The glass breaks on the hard stone","The water spills out all over floor"),
        ("The match strikes against the rough box","The flame leaps up in the dark"),
        ("The wheel turns slowly on the dry axle","The cart moves forward along the road"),
        ("The acid touches the metal surface","The metal corrodes and turns dull green"),
        ("The seed falls into the moist earth","The plant grows tall toward the light"),
        ("The door slams in the strong wind","The picture falls from the cracked wall"),
        ("The battery drains in the cold night","The phone shuts down without warning"),
        ("The frost covers all the bare ground","The pipes freeze and can burst in spring"),
        ("The pressure builds in the sealed tank","The lid pops off with a loud snap"),
        ("The crowd cheers for the home team","The players feel energized on field"),
        ("The dye soaks into the dry fabric","The cloth changes to a bright new color"),
        ("The magnet pulls the iron filing","The filing moves toward the strong pole"),
        ("The wave hits the crumbling cliff face","The rock falls into the churning sea"),
    ],
    "question": [
        ("She is very tired after long work","Is she very tired after long work"),
        ("He can swim really well in the sea","Can he swim really well in the sea"),
        ("They went to the big market today","Did they go to the big market today"),
        ("The car broke down on the highway","Did the car break down on the highway"),
        ("The dog is very hungry right now","Is the dog very hungry right now"),
        ("She wrote the letter to her friend","Did she write the letter to her friend"),
        ("He knows the right answer to this","Does he know the right answer to this"),
        ("The house looks very old and worn","Does the house look very old and worn"),
        ("They have finished all their homework","Have they finished all their homework"),
        ("She will attend the big conference","Will she attend the big conference"),
        ("He was there at the exact right time","Was he there at the exact right time"),
        ("The rain had stopped before they left","Had the rain stopped before they left"),
        ("They are waiting by the front door","Are they waiting by the front door"),
        ("She has read the whole book before","Has she read the whole book before"),
        ("The train arrives at noon today","Does the train arrive at noon today"),
        ("He should rest before the long race","Should he rest before the long race"),
        ("They could hear the music outside","Could they hear the music outside"),
        ("She would like to try the new dish","Would she like to try the new dish"),
        ("The plant needs more water to grow","Does the plant need more water to grow"),
        ("He used to live near the old park","Did he use to live near the old park"),
    ],
}

# Use N_GROUPS groups of N_PER_GROUP pairs each
N_GROUPS    = 10
N_PER_GROUP = 2  # 10 groups × 2 pairs = 20 pairs total per axis

# Category words for classification test (from Day 122)
CATEGORY_WORDS_SMALL = {
    "gender":      {"A": ["king","man","boy","father","son","actor"],
                    "B": ["queen","woman","girl","mother","daughter","actress"]},
    "comparative": {"A": ["fast","big","old","cold","tall","bright"],
                    "B": ["faster","bigger","older","colder","taller","brighter"]},
    "plural":      {"A": ["dog","cat","tree","bird","book","car"],
                    "B": ["dogs","cats","trees","birds","books","cars"]},
    "past_tense":  {"A": ["walk","run","eat","see","build","swim"],
                    "B": ["walked","ran","ate","saw","built","swam"]},
    "antonym":     {"A": ["hot","fast","good","happy","strong","old"],
                    "B": ["cold","slow","bad","sad","weak","new"]},
    "negation":    {"A": ["fast","good","strong","happy","clean","loud"],
                    "B": ["slow","bad","weak","sad","dirty","quiet"]},
    "synonym":     {"A": ["big","small","fast","cold","happy","hard"],
                    "B": ["large","tiny","quick","frigid","joyful","difficult"]},
    "hypernym":    {"A": ["dog","rose","car","eagle","ruby","hammer"],
                    "B": ["animal","flower","vehicle","bird","gem","tool"]},
    "concrete":    {"A": ["stone","road","wall","flame","chain","bridge"],
                    "B": ["burden","journey","barrier","hope","bond","connection"]},
    "passive":     {"A": ["breaks","chases","cooks","destroys","helps","paints"],
                    "B": ["broken","chased","cooked","destroyed","helped","painted"]},
    "causation":   {"A": ["rain","fire","heat","wind","pressure","friction"],
                    "B": ["flood","ash","melt","fall","collapse","spark"]},
    "question":    {"A": ["is","can","does","was","will","has"],
                    "B": ["Is","Can","Does","Was","Will","Has"]},
}

def normed(v): return v / (np.linalg.norm(v) + 1e-8)

print(f"Loading {MODEL_ID} ...")
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()
hidden_size = model.config.hidden_size
print(f"  hidden={hidden_size}\n")

def compute_axis_from_pairs(pairs, layer):
    diffs = []
    for s1, s2 in pairs:
        try:
            inp1 = tok(s1, return_tensors="pt"); inp2 = tok(s2, return_tensors="pt")
            with torch.no_grad():
                o1 = model(**inp1, output_hidden_states=True)
                o2 = model(**inp2, output_hidden_states=True)
            h1 = o1.hidden_states[layer][0,-1,:].numpy().astype(np.float32)
            h2 = o2.hidden_states[layer][0,-1,:].numpy().astype(np.float32)
            d = h2-h1; n = np.linalg.norm(d)
            if n > 1e-6: diffs.append(d/n)
        except: pass
    if not diffs: return np.zeros(hidden_size, np.float32)
    v = np.mean(diffs, axis=0); nv = np.linalg.norm(v)
    return (v/nv if nv > 1e-6 else v).astype(np.float32)

def axis_classification_acc(axis, words_a, words_b, layer):
    a_projs = []; b_projs = []
    for w in words_a:
        inp = tok(" "+w, return_tensors="pt")
        try:
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h = normed(out.hidden_states[layer][0,-1,:].numpy().astype(np.float32))
            a_projs.append(float(np.dot(h, axis)))
        except: pass
    for w in words_b:
        inp = tok(" "+w, return_tensors="pt")
        try:
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h = normed(out.hidden_states[layer][0,-1,:].numpy().astype(np.float32))
            b_projs.append(float(np.dot(h, axis)))
        except: pass
    if not a_projs or not b_projs: return 0.0
    sign = 1 if np.mean(b_projs) > np.mean(a_projs) else -1
    n_correct = sum(1 for p in a_projs if sign*p < 0) + sum(1 for p in b_projs if sign*p > 0)
    return n_correct / (len(a_projs) + len(b_projs))

print("Computing 10 independent axis estimates per axis ...")
print("=" * 72)
print(f"  {'axis':>14}  {'opt_L':>5}  "
      f"{'stab_mean':>10}  {'stab_std':>10}  "
      f"{'full_acc':>10}  {'mean_acc':>10}  {'verdict':>14}")
print("  " + "-" * 80)

stability_results = {}
for ax_name in AXIS_NAMES_12:
    L     = DAY78_LAYERS[ax_name]
    pairs = AXIS_PAIRS_ALL.get(ax_name, [])
    if len(pairs) < N_GROUPS * N_PER_GROUP:
        stability_results[ax_name] = {"error": "insufficient_pairs"}
        print(f"  {ax_name:>14}  L{L:02d}  SKIP (only {len(pairs)} pairs)"); continue

    # 10 axis estimates (each from 2 pairs)
    group_axes = []
    for g in range(N_GROUPS):
        gp = pairs[g*N_PER_GROUP : (g+1)*N_PER_GROUP]
        ax = compute_axis_from_pairs(gp, L)
        group_axes.append(ax)

    # Full axis (all 20 pairs)
    full_axis = compute_axis_from_pairs(pairs, L)

    # Pairwise cosines between 10 estimates
    n_est = len(group_axes)
    pairwise = []
    for i in range(n_est):
        for j in range(i+1, n_est):
            c = float(abs(np.dot(group_axes[i], group_axes[j])))
            pairwise.append(c)
    stab_mean = float(np.mean(pairwise)); stab_std = float(np.std(pairwise))

    # Cosine of each estimate to full axis
    cos_to_full = [float(abs(np.dot(ax, full_axis))) for ax in group_axes]

    # Classification accuracy: full axis vs mean of 10 estimates
    cw = CATEGORY_WORDS_SMALL.get(ax_name, {"A": [], "B": []})
    full_acc = axis_classification_acc(full_axis, cw["A"], cw["B"], L)
    # Mean axis = mean of 10 group axes (already averaged in full_axis,
    # but test individual group mean separately)
    group_mean_axis = np.mean(group_axes, axis=0)
    nv = np.linalg.norm(group_mean_axis)
    group_mean_axis = (group_mean_axis/nv if nv > 1e-8 else group_mean_axis).astype(np.float32)
    mean_acc = axis_classification_acc(group_mean_axis, cw["A"], cw["B"], L)

    verdict = ("STABLE" if stab_mean > 0.5 else
               "PARTIAL" if stab_mean > 0.3 else
               "VARIABLE")
    stability_results[ax_name] = {
        "L": L, "stab_mean": stab_mean, "stab_std": stab_std,
        "cos_to_full_mean": float(np.mean(cos_to_full)),
        "pairwise": pairwise,
        "full_acc": full_acc, "mean_acc": mean_acc,
    }
    print(f"  {ax_name:>14}  L{L:02d}  "
          f"{stab_mean:>10.4f}  {stab_std:>10.4f}  "
          f"{100*full_acc:>9.1f}%  {100*mean_acc:>9.1f}%  {verdict:>14}")

# ── Per-axis stability profile ─────────────────────────────────────────────────
print()
print("=" * 72)
print("Stability Distribution: How consistent are pairwise estimates?")
print("=" * 72)
print()
for ax_name, r in stability_results.items():
    if "error" in r: continue
    pw  = r["pairwise"]
    v_str = ("STABLE" if r['stab_mean'] > 0.5 else
             "PARTIAL" if r['stab_mean'] > 0.3 else "VARIABLE")
    print(f"  {ax_name:>14}: min={min(pw):.3f}  median={np.median(pw):.3f}  "
          f"max={max(pw):.3f}  mean={r['stab_mean']:.3f}  ({v_str})")

# ── Does more pairs improve accuracy? ────────────────────────────────────────
print()
print("=" * 72)
print("Accuracy vs Number of Pairs Used (gender, plural, past_tense)")
print("=" * 72)
print()

n_pairs_test = [2, 4, 6, 8, 10, 14, 20]
for ax_name in ["gender", "plural", "past_tense", "antonym"]:
    if ax_name not in stability_results or "error" in stability_results[ax_name]: continue
    L  = DAY78_LAYERS[ax_name]
    cw = CATEGORY_WORDS_SMALL.get(ax_name, {"A": [], "B": []})
    pairs = AXIS_PAIRS_ALL.get(ax_name, [])
    print(f"  {ax_name} (L{L}):")
    accs = []
    for np_test in n_pairs_test:
        if np_test > len(pairs): break
        ax = compute_axis_from_pairs(pairs[:np_test], L)
        acc = axis_classification_acc(ax, cw["A"], cw["B"], L)
        accs.append((np_test, acc))
        print(f"    n={np_test:2d} pairs: acc={100*acc:.1f}%")
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("Day 123 Summary — T2 Axis Context-Stability Test")
print("=" * 72)

valid = {ax: r for ax, r in stability_results.items() if "error" not in r}
stab_means = [r["stab_mean"] for r in valid.values()]
overall_mean = float(np.mean(stab_means)) if stab_means else 0
n_stable  = sum(1 for r in valid.values() if r["stab_mean"] > 0.5)
n_partial = sum(1 for r in valid.values() if 0.3 < r["stab_mean"] <= 0.5)
n_var     = sum(1 for r in valid.values() if r["stab_mean"] <= 0.3)
best_ax  = max(valid, key=lambda ax: valid[ax]["stab_mean"]) if valid else "N/A"
worst_ax = min(valid, key=lambda ax: valid[ax]["stab_mean"]) if valid else "N/A"

print(f"""
  Axis stability summary (pairwise cosine between independent estimates):
    Overall mean stability: {overall_mean:.4f}
    Stable   (cos > 0.5): {n_stable}/12
    Partial  (0.3-0.5):   {n_partial}/12
    Variable (< 0.3):     {n_var}/12
    Most stable:  {best_ax} ({valid.get(best_ax,{}).get('stab_mean',0):.4f})
    Least stable: {worst_ax} ({valid.get(worst_ax,{}).get('stab_mean',0):.4f})

  VERDICT:
  {'→ T2 axes are STABLE: independent sentence-pair sets find the SAME direction' if overall_mean > 0.5 else
   '→ T2 axes are PARTIALLY STABLE: some consistency across sentence pairs' if overall_mean > 0.3 else
   '→ T2 axes are VARIABLE: different sentence pairs give different directions'}

  TruthSpace self-similarity: {'CONFIRMED' if overall_mean > 0.5 else 'PARTIAL' if overall_mean > 0.3 else 'NOT CONFIRMED'}
  (High stability = semantic directions are fixed geometric properties of the LM)
""")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "stability_results": {ax: {k: v for k,v in r.items() if k != "pairwise"}
                               for ax, r in stability_results.items()},
        "overall_mean_stability": overall_mean,
        "n_stable": n_stable,
        "n_partial": n_partial,
        "n_variable": n_var,
    }, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")
print("Day 123 complete.")
