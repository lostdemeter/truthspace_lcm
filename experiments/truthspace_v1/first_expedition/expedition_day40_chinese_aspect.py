#!/usr/bin/env python3
"""
Expedition Day 40 — Chinese Aspect Hypothesis

Hypothesis (from Day 39 + linguistic analysis):
  Qwen2-1.5B was trained by Chinese engineers on Chinese-dominant data.
  Mandarin has no verb conjugation — tense/aspect are standalone particles:
    走  = walk/walked/walking/walks (bare form, Zone A/B prediction)
    走着 = walking (着-aspect, ongoing)  → B001 rank-1 beam prediction
    走了 = walked  (了-aspect, completed) → B000 scatter prediction
    走过 = have walked (过-aspect, experiential) → ?

Tests:
  T1: Chinese bare verb forms (走/说/写/跑/吃/唱/做/来) — predict Zone A/B
  T2: Aspect forms — predict 着→B001, 了→B000
  T3: Aspect particles alone (着/了/过/的) — what zone?
  T4: Chinese nouns (猫/狗/树/人) — predict Zone C body (sanity check)
  T5: Cross-lingual — cos(English -ing φ, 走着 φ) vs cos(English -ing φ, 走了 φ)
      If B001 = 着-beam: English gerunds should be MORE similar to 走着 than to 走了
"""

import os, json, sys
import numpy as np

CACHE_FILE  = os.path.join(os.path.dirname(__file__), "day27_hs_cache.npz")
ATLAS_FILE  = os.path.join(os.path.dirname(__file__), "day27_atlas.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "day40_chinese_aspect.json")
MODEL_NAME  = "Qwen/Qwen2-1.5B-Instruct"
LAYER       = 14

KILLING_PAIRS = [
    ('cat','cats'),('dog','dogs'),('tree','trees'),('bird','birds'),
    ('house','houses'),('man','woman'),('king','queen'),('boy','girl'),
    ('big','bigger'),('fast','faster'),('old','older'),
]

# ── Chinese vocabulary to test ────────────────────────────────────────────────
# Format: (display_label, prompt_string, predicted_zone, description)
CHINESE_TESTS = [
    # Content verbs — bare form (predict Zone A/B)
    ("走",   "走",   "A/B", "walk — bare form (content verb)"),
    ("说",   "说",   "A/B", "say/speak — bare form"),
    ("写",   "写",   "A/B", "write — bare form"),
    ("跑",   "跑",   "A/B", "run — bare form"),
    ("吃",   "吃",   "A/B", "eat — bare form"),
    ("唱",   "唱",   "A/B", "sing — bare form"),
    ("看",   "看",   "A/B", "see/watch — bare form"),
    ("打",   "打",   "A/B", "hit/play — bare form"),
    # Grammatical verbs — bare form (predict B001 or A/B)
    ("做",   "做",   "B001", "do/make — grammatical verb"),
    ("来",   "来",   "B001", "come — grammatical verb"),
    ("去",   "去",   "B001", "go — grammatical verb"),
    ("用",   "用",   "B001", "use — grammatical verb"),
    ("给",   "给",   "B001", "give — grammatical verb"),
    # 着-aspect forms (ongoing, predict B001)
    ("走着", "走着", "B001", "walking (着-ongoing)"),
    ("说着", "说着", "B001", "saying (着-ongoing)"),
    ("写着", "写着", "B001", "writing (着-ongoing)"),
    ("跑着", "跑着", "B001", "running (着-ongoing)"),
    ("吃着", "吃着", "B001", "eating (着-ongoing)"),
    ("唱着", "唱着", "B001", "singing (着-ongoing)"),
    ("做着", "做着", "B001", "doing/making (着-ongoing)"),
    # 了-aspect forms (completed, predict B000)
    ("走了", "走了", "B000", "walked (了-completed)"),
    ("说了", "说了", "B000", "said (了-completed)"),
    ("写了", "写了", "B000", "wrote (了-completed)"),
    ("跑了", "跑了", "B000", "ran (了-completed)"),
    ("吃了", "吃了", "B000", "ate (了-completed)"),
    ("唱了", "唱了", "B000", "sang (了-completed)"),
    ("做了", "做了", "B000", "made/did (了-completed)"),
    # 过-aspect forms (experiential, predict ?)
    ("走过", "走过", "?", "have walked (过-experiential)"),
    ("说过", "说过", "?", "have said (过-experiential)"),
    ("吃过", "吃过", "?", "have eaten (过-experiential)"),
    # Aspect particles alone
    ("着",   "着",   "B001", "着 particle alone (ongoing marker)"),
    ("了",   "了",   "B000", "了 particle alone (completion marker)"),
    ("过",   "过",   "?",    "过 particle alone (experiential marker)"),
    ("的",   "的",   "?",    "的 particle alone (nominalizer/possessive)"),
    # Chinese nouns — sanity check (predict Zone C body)
    ("猫",   "猫",   "C",    "cat — should match animal body"),
    ("狗",   "狗",   "C",    "dog — should match animal body"),
    ("树",   "树",   "C",    "tree — should match plant/nature body"),
    ("人",   "人",   "C",    "person/people — should match person body"),
    ("山",   "山",   "C",    "mountain — should match nature body"),
    ("水",   "水",   "C",    "water — should match nature body"),
    ("书",   "书",   "C",    "book — should match knowledge body"),
    ("城市", "城市", "C",    "city — should match place body"),
]

# ── Load English atlas ────────────────────────────────────────────────────────
print("── Load English atlas ────────────────────────────────────────────────")
npz       = np.load(CACHE_FILE, allow_pickle=True)
words_all = list(npz['words'])
hs14_all  = npz['hs_14'].astype(np.float64)
w2i       = {w: i for i, w in enumerate(words_all)}

with open(ATLAS_FILE) as f:
    atlas = json.load(f)
wmap = atlas['word_map']

# Build Z2 axis from Killing pairs
deltas = []
for a, b in KILLING_PAIRS:
    for pfx in [' ', '']:
        wa, wb = pfx+a, pfx+b
        if wa in w2i and wb in w2i:
            d = hs14_all[w2i[wb]] - hs14_all[w2i[wa]]
            dm = np.linalg.norm(d)
            if dm > 1e-20:
                deltas.append(d / dm)
            break
D = np.stack(deltas)
_, _, Vt = np.linalg.svd(D, full_matrices=False)
z2 = Vt[0].astype(np.float64)

def batch_phi(hs, z2):
    H  = hs.astype(np.float64)
    nm = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (nm + 1e-20)
    proj = (Hn @ z2)[:, None] * z2
    perp = Hn - proj
    pm   = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / (pm + 1e-20)

def phi_single(h, z2):
    hn  = h.astype(np.float64) / (np.linalg.norm(h) + 1e-20)
    proj = float(hn @ z2)
    perp = hn - proj * z2
    pm   = np.linalg.norm(perp)
    return perp / (pm + 1e-20)

# Build zone centroids from English data
wmap_words = [w for w in wmap.keys() if w in w2i]
wmap_idx   = np.array([w2i[w] for w in wmap_words])
wmap_phi   = batch_phi(hs14_all[wmap_idx], z2)
w2l        = {w: i for i, w in enumerate(wmap_words)}

zone_c_idx  = [i for i, w in enumerate(wmap_words)
               if wmap[w]['phase']==2 and wmap[w].get('L14_body') not in ('B000','B001',None)]
b000_idx    = [i for i, w in enumerate(wmap_words)
               if wmap[w]['phase']==2 and wmap[w].get('L14_body') == 'B000']
b001_idx    = [i for i, w in enumerate(wmap_words)
               if wmap[w]['phase']==2 and wmap[w].get('L14_body') == 'B001']
ab_idx      = [i for i, w in enumerate(wmap_words) if wmap[w]['phase'] != 2]

c_C   = wmap_phi[zone_c_idx].mean(axis=0);  c_C  /= np.linalg.norm(c_C)
c_B0  = wmap_phi[b000_idx].mean(axis=0);    c_B0 /= np.linalg.norm(c_B0)
c_B1  = wmap_phi[b001_idx].mean(axis=0);    c_B1 /= np.linalg.norm(c_B1)
c_AB  = wmap_phi[ab_idx].mean(axis=0);      c_AB /= np.linalg.norm(c_AB)

print(f"  Loaded {len(wmap_words)} atlas words | "
      f"Zone C={len(zone_c_idx)} B000={len(b000_idx)} B001={len(b001_idx)} A/B={len(ab_idx)}")

# Build per-body centroids for Zone C (for noun sanity check)
body_centroids = {}
body_labels    = {}
for i, w in enumerate(wmap_words):
    v = wmap[w]
    if v['phase'] == 2 and v.get('L14_body') not in ('B000','B001',None):
        bd = v['L14_body']
        if bd not in body_centroids:
            body_centroids[bd] = []
            body_labels[bd]    = v.get('L14_label', bd)
        body_centroids[bd].append(wmap_phi[i])
body_centroid_vecs = {}
for bd, vecs in body_centroids.items():
    c = np.stack(vecs).mean(axis=0)
    cn = np.linalg.norm(c)
    if cn > 1e-20:
        body_centroid_vecs[bd] = c / cn

# English gerund centroid (from B001 — the 着-beam)
eng_gerund_phi = wmap_phi[b001_idx].mean(axis=0)
eng_gerund_phi /= np.linalg.norm(eng_gerund_phi)

# English past-tense centroid (B000 content verbs only — strip adjectives/nouns)
eng_past_sample_words = [
    w for w in ['walked','wrote','spoke','gave','broke','drove','sang',
                'kept','held','built','threw','found','ran','came']
    for key in [w, ' '+w] if key in w2l
]
if eng_past_sample_words:
    eng_past_phi = np.stack([wmap_phi[w2l[w]] for w in eng_past_sample_words if w in w2l]).mean(axis=0)
    eng_past_phi /= np.linalg.norm(eng_past_phi)
else:
    eng_past_phi = c_B0.copy()

print(f"  English gerund centroid ready ({len(b001_idx)} words)")
print(f"  English past centroid from {len(eng_past_sample_words)} sample words")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\n── Load {MODEL_NAME} ─────────────────────────────────────────────────")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, output_hidden_states=True,
    torch_dtype=torch.float32, device_map='cpu')
model.eval()
print(f"  Model loaded. Hidden dim = {model.config.hidden_size}")

# ── Extract hidden states for Chinese vocabulary ──────────────────────────────
def get_hs14(text):
    """Extract L14 hidden state for the LAST meaningful token of text."""
    inputs = tok(text, return_tensors='pt', add_special_tokens=True)
    ids    = inputs['input_ids'][0]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states  # list of (1, seq_len, hidden_dim)
    # For Chinese, find the position of the target token(s)
    # Use the last non-BOS token position for single-char, last token for multi-char
    # But since we're probing isolated words, take the last token position
    target_pos = ids.shape[0] - 1
    h14 = hs[LAYER][0, target_pos, :].numpy().astype(np.float64)
    # Also try the token right before EOS if applicable
    token_ids_str = [str(t.item()) for t in ids]
    return h14, token_ids_str

def assign_zone(phi_v, threshold=0.05):
    """Assign zone label based on cosine similarity to centroids."""
    sim_C  = float(phi_v @ c_C)
    sim_B0 = float(phi_v @ c_B0)
    sim_B1 = float(phi_v @ c_B1)
    sim_AB = float(phi_v @ c_AB)
    sims = {'C': sim_C, 'B000': sim_B0, 'B001': sim_B1, 'A/B': sim_AB}
    assigned = max(sims, key=sims.get)
    return assigned, sims

def top3_bodies(phi_v):
    """Find top-3 Zone C body matches."""
    scores = {bd: float(phi_v @ cv) for bd, cv in body_centroid_vecs.items()}
    top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
    return [(bd, body_labels.get(bd, bd), s) for bd, s in top3]

# ── Run tests ─────────────────────────────────────────────────────────────────
print(f"\n── Extracting Chinese hidden states ─────────────────────────────────")
results = {}
for label, prompt, predicted, desc in CHINESE_TESTS:
    h14, token_ids = get_hs14(prompt)
    phi_v  = phi_single(h14, z2)
    zone, sims = assign_zone(phi_v)
    top3 = top3_bodies(phi_v)
    correct = (zone == predicted) or (predicted == '?')
    results[label] = {
        'prompt': prompt,
        'predicted_zone': predicted,
        'assigned_zone': zone,
        'correct': correct,
        'sims': sims,
        'top3_bodies': top3,
        'n_tokens': len(token_ids),
        'token_ids': token_ids,
        'description': desc,
    }

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"T1/T2: Chinese Verb Forms — Zone Assignment vs Prediction")
print(f"{'='*65}")
print(f"\n  {'Form':<8s} {'Predict':<8s} {'Assigned':<8s} {'OK':<4s}  "
      f"{'cos(C)':<8s} {'cos(B0)':<8s} {'cos(B1)':<8s} {'cos(AB)':<8s}  Description")
print(f"  {'-'*100}")

sections = [
    ("Content verbs — bare form", [k for k in results if results[k]['description'].endswith("bare form (content verb)") or
                                                          results[k]['description'].endswith("bare form")]),
    ("Grammatical verbs — bare form", [k for k in results if "grammatical verb" in results[k]['description']]),
    ("着-aspect (ongoing)", [k for k in results if "着-ongoing" in results[k]['description']]),
    ("了-aspect (completed)", [k for k in results if "了-completed" in results[k]['description']]),
    ("过-aspect (experiential)", [k for k in results if "过-experiential" in results[k]['description']]),
    ("Particles alone", [k for k in results if "particle alone" in results[k]['description']]),
    ("Nouns (sanity check)", [k for k in results if "should match" in results[k]['description']]),
]

all_correct = 0
all_total   = 0
for section_name, keys in sections:
    if not keys:
        continue
    print(f"\n  [{section_name}]")
    for k in keys:
        r = results[k]
        ok = '✓' if r['correct'] else '✗'
        s  = r['sims']
        print(f"  {k:<8s} {r['predicted_zone']:<8s} {r['assigned_zone']:<8s} {ok:<4s}"
              f"  {s['C']:.3f}   {s['B000']:.3f}   {s['B001']:.3f}   {s['A/B']:.3f}"
              f"  {r['description']}")
        if r['predicted_zone'] != '?':
            all_total += 1
            if r['correct']:
                all_correct += 1
        if r['assigned_zone'] == 'C':
            top3 = r['top3_bodies']
            print(f"          → top bodies: {', '.join(f'{lbl}({s:.2f})' for _,lbl,s in top3)}")

print(f"\n  Prediction accuracy (excluding '?' targets): {all_correct}/{all_total}")

# ── T5: Cross-lingual similarity ──────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"T5 — Cross-Lingual Similarity: English φ vs Chinese aspect forms")
print(f"{'='*65}")

print(f"\n  Centroid similarity (English gerund beam vs Chinese forms):")
for label in ['走', '走着', '走了', '走过', '着', '了', '说着', '说了', '做着', '做了']:
    if label not in results:
        continue
    phi_zh = phi_single(
        get_hs14(results[label]['prompt'])[0], z2)
    sim_to_gerund = float(phi_zh @ eng_gerund_phi)
    sim_to_past   = float(phi_zh @ eng_past_phi)
    print(f"  {label:<6s}  cos(eng_gerund)={sim_to_gerund:.3f}  cos(eng_past)={sim_to_past:.3f}")

print(f"\n  Key question: for 走着 vs 走了, which is closer to the English gerund beam?")
if '走着' in results and '走了' in results:
    phi_zh_prog = phi_single(get_hs14('走着')[0], z2)
    phi_zh_past = phi_single(get_hs14('走了')[0], z2)
    sim_prog_gerund = float(phi_zh_prog @ eng_gerund_phi)
    sim_past_gerund = float(phi_zh_past @ eng_gerund_phi)
    sim_prog_past   = float(phi_zh_prog @ eng_past_phi)
    sim_past_past   = float(phi_zh_past @ eng_past_phi)
    print(f"  走着 (ongoing) ↔ English gerunds: cos={sim_prog_gerund:.3f}")
    print(f"  走了 (completed) ↔ English gerunds: cos={sim_past_gerund:.3f}")
    print(f"  走着 (ongoing) ↔ English past tense: cos={sim_prog_past:.3f}")
    print(f"  走了 (completed) ↔ English past tense: cos={sim_past_past:.3f}")
    diff_prog = sim_prog_gerund - sim_past_gerund
    diff_past = sim_past_past - sim_prog_past
    print(f"\n  Hypothesis confirmation:")
    print(f"  走着 closer to English gerunds: {diff_prog:+.3f} ({'+' if diff_prog>0 else '-'}CONFIRMED" + (f")" if diff_prog>0 else f" REFUTED)"))
    print(f"  走了 closer to English past:    {diff_past:+.3f} ({'+' if diff_past>0 else '-'}CONFIRMED" + (f")" if diff_past>0 else f" REFUTED)"))

# ── T5b: 着/了 particles vs English morphology centroids ──────────────────────
print(f"\n  Particles alone vs English zone centroids:")
for particle_label in ['着', '了', '过', '的']:
    if particle_label not in results:
        continue
    phi_p = phi_single(get_hs14(particle_label)[0], z2)
    print(f"  {particle_label}  cos(B001)={phi_p@c_B1:.3f}  cos(B000)={phi_p@c_B0:.3f}  "
          f"cos(Zone C)={phi_p@c_C:.3f}  cos(A/B)={phi_p@c_AB:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    'meta': {'experiment': 'Day 40 — Chinese Aspect Hypothesis'},
    'accuracy': f"{all_correct}/{all_total}",
    'results': {k: {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                    for kk, vv in v.items() if kk != 'sims'}
                for k, v in results.items()},
}
# Convert sims separately (dict of floats)
for k in results:
    output['results'][k]['sims'] = {kk: float(vv) for kk, vv in results[k]['sims'].items()}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"\nDay 40 complete.")
