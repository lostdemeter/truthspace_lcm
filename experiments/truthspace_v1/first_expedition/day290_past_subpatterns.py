import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

def normed(v): return v/(np.linalg.norm(v)+1e-8)
def get_emb(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None
def compute_axis(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if not chords: return None, 0.0, valid
    md = normed(np.mean(chords, axis=0))
    return md, float(np.mean([np.dot(normed(c), md) for c in chords])), valid
def nn_retrieve(pred_emb, exclude_ids, top_n=1):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n
    for eid in exclude_ids: sims[eid] = -1.0
    top = np.argsort(sims)[-top_n:][::-1]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]
def best_scale(axis, valid_pairs, lo=0.02, hi=4.0, n=50):
    best_s, best_acc = 1.0, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for s_,t_,sid,tid in valid_pairs
                if nn_retrieve(W_E[sid]+s*axis,[sid])[0][0]==t_)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc
def eval_axis(axis, scale, test_pairs):
    results = []
    for s_, t_, sid, tid in test_pairs:
        r = nn_retrieve(W_E[sid] + scale * axis, [sid])
        got = r[0][0] if r else '?'
        results.append((s_, t_, got, got == t_))
    return results
def valid_pairs_for(pair_list):
    vp = []
    for s, t in pair_list:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        vp.append((s, t, sid, tid))
    return vp

print("DAY 290: PAST TENSE SUB-PATTERN ANALYSIS")
print("="*65)
print("Why does past_reg fail at 58% on holdout?")
print("Hypothesis: the +ed suffix has sub-patterns that produce")
print("different displacement vectors in W_E.")
print()

# ====================================================================
# PAST TENSE SUB-PATTERNS
# ====================================================================
# Pattern 1: plain +ed (consonant-final verbs)
PAT_ED = [
    ('walk','walked'),('talk','talked'),('work','worked'),('call','called'),
    ('turn','turned'),('start','started'),('want','wanted'),('need','needed'),
    ('look','looked'),('ask','asked'),('seem','seemed'),('help','helped'),
    ('pass','passed'),('wait','waited'),('watch','watched'),('reach','reached'),
    ('touch','touched'),('search','searched'),('climb','climbed'),('learn','learned'),
]
PAT_ED_HOLD = [
    ('jump','jumped'),('kick','kicked'),('push','pushed'),('pull','pulled'),
    ('cook','cooked'),('wash','washed'),('print','printed'),('count','counted'),
    ('test','tested'),('rest','rested'),('land','landed'),('sound','sounded'),
    ('smell','smelled'),('spell','spelled'),('fill','filled'),('kill','killed'),
    ('burn','burned'),('earn','earned'),('rain','rained'),('clean','cleaned'),
]

# Pattern 2: silent-e + d  (ends in -e)
PAT_D = [
    ('move','moved'),('live','lived'),('love','loved'),('use','used'),
    ('hope','hoped'),('care','cared'),('smile','smiled'),('dance','danced'),
    ('change','changed'),('close','closed'),('raise','raised'),('place','placed'),
    ('face','faced'),('race','raced'),('force','forced'),('trace','traced'),
]
PAT_D_HOLD = [
    ('save','saved'),('wave','waved'),('name','named'),('blame','blamed'),
    ('flame','flamed'),('taste','tasted'),('waste','wasted'),('paste','pasted'),
    ('joke','joked'),('smoke','smoked'),('vote','voted'),('note','noted'),
    ('serve','served'),('curve','curved'),('nerve','nerved'),
]

# Pattern 3: doubled consonant + ed (CVC pattern)
PAT_DBL = [
    ('stop','stopped'),('drop','dropped'),('plan','planned'),('trip','tripped'),
    ('grab','grabbed'),('drag','dragged'),('slip','slipped'),('clip','clipped'),
    ('skip','skipped'),('whip','whipped'),('tap','tapped'),('wrap','wrapped'),
    ('knit','knitted'),('fit','fitted'),('pat','patted'),('bat','batted'),
]
PAT_DBL_HOLD = [
    ('clap','clapped'),('snap','snapped'),('slap','slapped'),('trap','trapped'),
    ('spin','spun'),('grin','grinned'),('pin','pinned'),('win','won'),
    ('beg','begged'),('log','logged'),('tag','tagged'),('bag','bagged'),
]

# Pattern 4: irregulars (for comparison)
PAT_IRR = [
    ('feel','felt'),('run','ran'),('go','went'),('get','got'),
    ('say','said'),('make','made'),('take','took'),('see','saw'),
    ('know','knew'),('come','came'),('give','gave'),('think','thought'),
    ('find','found'),('tell','told'),('keep','kept'),('leave','left'),
    ('stand','stood'),('lose','lost'),('hold','held'),('read','read'),
]

print("PART A: Sub-pattern axis coherence and self-accuracy")
print("-"*65)

patterns = [
    ('+ed (plain)',        PAT_ED,    PAT_ED_HOLD),
    ('+d (silent-e)',      PAT_D,     PAT_D_HOLD),
    ('+ped (doubled)',     PAT_DBL,   PAT_DBL_HOLD),
    ('irregular',          PAT_IRR,   []),
]

pat_axes = {}
for name, train, hold in patterns:
    vt = valid_pairs_for(train)
    if not vt: continue
    ax, coh, valid = compute_axis(train)
    if ax is None: continue
    s_opt, acc = best_scale(ax, valid)
    vh = valid_pairs_for(hold)
    acc_h = sum(1 for r in eval_axis(ax, s_opt, vh) if r[3]) if vh else 0
    pat_axes[name] = (ax, s_opt, coh, len(vt), acc, len(vh), acc_h, vh)
    print("  %-20s coh=%.4f scale=%.2f train=%d/%d(%.0f%%) hold=%d/%d(%.0f%%)" % (
        name, coh, s_opt, acc, len(vt), 100*acc/max(1,len(vt)),
        acc_h, len(vh), 100*acc_h/max(1,len(vh))))
print()

# ====================================================================
# PART B: INTER-SUB-AXIS SIMILARITY
# ====================================================================
print("PART B: Cosine similarity between sub-pattern axes")
print("-"*65)
ax_names = list(pat_axes.keys())
print("         " + "  ".join("%-16s" % n for n in ax_names))
for n1 in ax_names:
    row = "  %-18s" % n1
    for n2 in ax_names:
        sim = float(np.dot(pat_axes[n1][0].astype(np.float32),
                           pat_axes[n2][0].astype(np.float32)))
        row += " %+.4f        " % sim
    print(row)
print()

# ====================================================================
# PART C: CROSS-PATTERN GENERALISATION
# Can +ed axis predict +d forms? Can +d predict +ed?
# ====================================================================
print("PART C: Cross-pattern axis application")
print("-"*65)
print("Testing whether sub-pattern axes can substitute for each other.")
print()

for src_name, hold_name, hold in [
    ('+ed (plain)', '+d (silent-e)', valid_pairs_for(PAT_D_HOLD[:8])),
    ('+d (silent-e)', '+ed (plain)', valid_pairs_for(PAT_ED_HOLD[:8])),
    ('+ed (plain)', '+ped (doubled)', valid_pairs_for(PAT_DBL_HOLD[:6])),
]:
    if src_name not in pat_axes or not hold: continue
    ax, s_opt, coh, _, _, _, _, _ = pat_axes[src_name]
    acc = sum(1 for r in eval_axis(ax, s_opt, hold) if r[3])
    print("  Axis=%-20s tested on %-16s: %d/%d (%.0f%%)" % (
        src_name, hold_name, acc, len(hold), 100*acc/max(1,len(hold))))
print()

# ====================================================================
# PART D: COMBINED PAST-REG AXIS (mixing all sub-patterns)
# This is what Day 289 used — compare to individual sub-axes
# ====================================================================
print("PART D: Combined past-reg axis vs individual sub-axes")
print("-"*65)

ALL_REG_TRAIN = PAT_ED[:10] + PAT_D[:8] + PAT_DBL[:8]
ALL_REG_HOLD  = PAT_ED_HOLD[:5] + PAT_D_HOLD[:5] + PAT_DBL_HOLD[:5]

ax_combined, coh_combined, valid_combined = compute_axis(ALL_REG_TRAIN)
s_combined, acc_combined = best_scale(ax_combined, valid_combined)
vh_combined = valid_pairs_for(ALL_REG_HOLD)
acc_ch = sum(1 for r in eval_axis(ax_combined, s_combined, vh_combined) if r[3])

print("  Combined axis: coh=%.4f scale=%.2f train=%d/%d(%.0f%%) hold=%d/%d(%.0f%%)" % (
    coh_combined, s_combined, acc_combined, len(valid_combined),
    100*acc_combined/max(1,len(valid_combined)),
    acc_ch, len(vh_combined), 100*acc_ch/max(1,len(vh_combined))))
print()

# Show combined on mixed holdout
print("  Combined axis on mixed holdout (all sub-patterns):")
for s_, t_, got, hit in eval_axis(ax_combined, s_combined, vh_combined):
    print("  %-12s -> %-14s  got=%-14s [%s]" % (s_, t_, got, 'HIT' if hit else '---'))
print()

# ====================================================================
# PART E: CHORD DISPERSION ANALYSIS
# Why does +ed have lower coherence than +er (comparative)?
# ====================================================================
print("PART E: Chord dispersion analysis")
print("-"*65)
print("Measuring the spread of displacement vectors per pattern.")
print()

for name, pairs in [('+ed (plain)', PAT_ED), ('+d (silent-e)', PAT_D),
                    ('+ped (doubled)', PAT_DBL), ('comparative (+er)', [
    ('fast','faster'),('slow','slower'),('tall','taller'),('small','smaller'),
    ('large','larger'),('hard','harder'),('soft','softer'),('warm','warmer'),
    ('dark','darker'),('clean','cleaner'),('sharp','sharper'),('deep','deeper'),
    ('wide','wider'),('strong','stronger'),('long','longer'),('old','older'),
    ('thick','thicker'),('thin','thinner'),('smooth','smoother'),('quiet','quieter'),
])]:
    chords = []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        c = et - es
        chords.append(c)
    if len(chords) < 2: continue
    ax, coh, _ = compute_axis(pairs)
    if ax is None: continue
    # Pairwise cosine between chords
    chord_norms = [normed(c).astype(np.float32) for c in chords]
    sims = [float(np.dot(chord_norms[i], chord_norms[j]))
            for i in range(len(chords)) for j in range(i+1, len(chords))]
    mean_pairwise = float(np.mean(sims)) if sims else 0
    # Chord magnitude spread
    mags = [np.linalg.norm(c) for c in chords]
    print("  %-22s  coh=%.3f  pairwise_cos=%.3f  mag_mean=%.2f  mag_std=%.2f" % (
        name, coh, mean_pairwise, np.mean(mags), np.std(mags)))
print()

# ====================================================================
# PART F: MORPHOLOGICAL LINEARITY SPECTRUM (all axes)
# ====================================================================
print("PART F: Morphological linearity spectrum")
print("-"*65)
print("Ranking axes by linearity (pairwise cosine between chords).")
print()

ALL_AXES = [
    ('+er (comp)',  [('fast','faster'),('slow','slower'),('tall','taller'),('small','smaller'),
                    ('large','larger'),('hard','harder'),('soft','softer'),('warm','warmer'),
                    ('dark','darker'),('clean','cleaner'),('sharp','sharper'),('deep','deeper'),
                    ('wide','wider'),('strong','stronger'),('long','longer'),('old','older')]),
    ('+est (sup)',  [('fast','fastest'),('slow','slowest'),('tall','tallest'),('small','smallest'),
                    ('large','largest'),('hard','hardest'),('soft','softest'),('warm','warmest'),
                    ('dark','darkest'),('clean','cleanest'),('sharp','sharpest'),('deep','deepest'),
                    ('wide','widest'),('strong','strongest'),('long','longest'),('old','oldest')]),
    ('+s (plural)', [('cat','cats'),('dog','dogs'),('bird','birds'),('tree','trees'),
                    ('book','books'),('car','cars'),('hand','hands'),('eye','eyes'),
                    ('word','words'),('day','days'),('year','years'),('house','houses'),
                    ('arm','arms'),('leg','legs'),('door','doors'),('line','lines')]),
    ('+ed plain',   PAT_ED[:16]),
    ('+d silent-e', PAT_D[:16]),
    ('+ped doubled', PAT_DBL[:16]),
    ('gender',      [('king','queen'),('man','woman'),('boy','girl'),('son','daughter'),
                    ('brother','sister'),('father','mother'),('uncle','aunt'),('prince','princess'),
                    ('hero','heroine'),('actor','actress'),('waiter','waitress'),('god','goddess')]),
    ('past irr',    PAT_IRR[:16]),
]

spectrum = []
for name, pairs in ALL_AXES:
    chords = []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(normed(et - es).astype(np.float32))
    if len(chords) < 3: continue
    sims = [float(np.dot(chords[i], chords[j]))
            for i in range(len(chords)) for j in range(i+1, len(chords))]
    pc = float(np.mean(sims)) if sims else 0
    ax, coh, _ = compute_axis(pairs)
    spectrum.append((name, pc, coh, len(chords)))

spectrum.sort(key=lambda x: -x[1])
print("  %-22s  pairwise_cos  coherence  n_pairs  linearity")
print("  " + "-"*65)
for name, pc, coh, n in spectrum:
    lin = "HIGH" if pc > 0.4 else ("MEDIUM" if pc > 0.2 else "LOW")
    print("  %-22s  %.4f        %.4f     %2d       %s" % (name, pc, coh, n, lin))
print()
print("  Higher pairwise cosine = more linear = better generalisation")
print("  This is the geometric explanation for the Day 289 results.")
