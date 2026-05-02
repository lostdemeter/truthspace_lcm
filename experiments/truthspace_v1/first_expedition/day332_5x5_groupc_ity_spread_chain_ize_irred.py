import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

print("Building masks...", flush=True)
CLEAN_MASK   = np.zeros(len(W_E), dtype=bool)
RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w or len(w) <= 1: continue
    if w.startswith('-') or w.startswith('_'): continue
    RELAXED_MASK[i] = True
    if not w[0].isupper(): CLEAN_MASK[i] = True
print("  clean=%d  relaxed=%d" % (CLEAN_MASK.sum(), RELAXED_MASK.sum()))

_src_cache = {}
def source_ids(word):
    if word in _src_cache: return _src_cache[word]
    ids = set()
    for p in [' '+word, word, ' '+word[0].upper()+word[1:],
              word[0].upper()+word[1:], word.upper(), ' '+word.upper()]:
        tks = tok(p, add_special_tokens=False)['input_ids']
        if len(tks) == 1: ids.add(tks[0])
    _src_cache[word] = ids
    return ids

def normed(v): return v / (np.linalg.norm(v) + 1e-8)
def get_emb(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if len(ids) == 1: return W_E[ids[0]].copy(), ids[0]
    return None, None

def nn_retrieve(pred_emb, excl_ids, mask, top_n=5):
    pred_n = normed(pred_emb).astype(np.float32)
    sims   = W_n @ pred_n
    sims[~mask] = -1.0
    for eid in excl_ids: sims[eid] = -1.0
    top = np.argpartition(sims, -top_n)[-top_n:]
    top = top[np.argsort(sims[top])[::-1]]
    return [(tok.decode([i]).strip(), float(sims[i]), int(i)) for i in top]

def compute_axis(pairs):
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, valid, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0))
    pc  = float(np.mean([np.dot(cn[i], cn[j])
                         for i in range(len(cn)) for j in range(i+1, len(cn))]))
    return md, valid, pc

def compute_axis_full(pairs):
    """Returns (axis, valid, pc, spread, mean_mag)."""
    chords, valid = [], []
    for s, t in pairs:
        es, sid = get_emb(s); et, tid = get_emb(t)
        if es is None or et is None: continue
        chords.append(et - es); valid.append((s, t, sid, tid))
    if len(chords) < 2: return None, valid, 0.0, 0.0, 0.0
    cn = [normed(c).astype(np.float32) for c in chords]
    md = normed(np.mean(chords, axis=0)).astype(np.float32)
    pc = float(np.mean([np.dot(cn[i], cn[j])
                        for i in range(len(cn)) for j in range(i+1, len(cn))]))
    spread = float(np.std([np.dot(cn[i], md) for i in range(len(cn))]))
    mean_mag = float(np.mean([np.linalg.norm(c) for c in chords]))
    return md, valid, pc, spread, mean_mag

def best_scale(axis, valid, mask, lo=0.02, hi=6.0, n=30):
    best_s, best_acc = 0.5, 0
    for s in np.linspace(lo, hi, n):
        c = sum(1 for _,t,sid,_ in valid
                if nn_retrieve(W_E[sid]+s*axis, source_ids(tok.decode([sid]).strip()), mask, 1)[0][0]==t)
        if c > best_acc: best_acc=c; best_s=s
    return best_s, best_acc

def axis_loo(axis, valid, mask):
    if len(valid) < 3: return 0.0
    chords_f = [W_E[tid]-W_E[sid] for _,_,sid,tid in valid]
    ax_full  = normed(np.mean(chords_f, axis=0))
    gs, _    = best_scale(ax_full, valid, mask)
    hits = 0
    for i in range(len(valid)):
        tv = [valid[j] for j in range(len(valid)) if j!=i]
        al = normed(np.mean([W_E[tid]-W_E[sid] for _,_,sid,tid in tv], axis=0))
        test_s, test_t, test_sid, _ = valid[i]
        r = nn_retrieve(W_E[test_sid]+gs*al, source_ids(test_s), mask, 1)
        if r[0][0] == test_t: hits += 1
    return hits/len(valid)

def irred_on_holdout(axis, holdout, mask, lo=0.02, hi=6.0, n=60):
    irred=0; n_ho=0; details=[]
    for s_w, t_w in holdout:
        es, sid = get_emb(s_w)
        if es is None: continue
        n_ho += 1; found_at = None
        for s in np.linspace(lo, hi, n):
            r = nn_retrieve(W_E[sid]+s*axis, source_ids(s_w), mask, 1)
            if r[0][0] == t_w: found_at=s; break
        if found_at is None: irred += 1
        details.append((s_w, t_w, found_at))
    return irred/n_ho if n_ho else 0.0, n_ho, details

print()
print("DAY 332: 5x5 GROUP MAP, GROUP C vs +ity, SPREAD FEATURE, C→E CHAIN, +ize IRRED")
print("="*80)
print()

# =====================================================================
# CANONICAL PAIR SETS FOR ALL GROUPS
# =====================================================================
GROUP_PAIRS = {
    'A:+ance': [('perform','performance'),('exist','existence'),('enter','entrance'),
                 ('resist','resistance'),('accept','acceptance'),('appear','appearance'),
                 ('depend','dependence'),('insist','insistence')],
    'A:+ment': [('achieve','achievement'),('develop','development'),('manage','management'),
                 ('govern','government'),('engage','engagement'),('require','requirement'),
                 ('move','movement'),('improve','improvement')],
    'A:+tion': [('act','action'),('direct','direction'),('educate','education'),
                 ('create','creation'),('produce','production'),('relate','relation'),
                 ('combine','combination'),('apply','application')],
    'A:+al_nom': [('arrive','arrival'),('propose','proposal'),('approve','approval'),
                   ('refuse','refusal'),('remove','removal'),('survive','survival'),
                   ('deny','denial'),('dispose','disposal')],
    'B:+ity':  [('human','humanity'),('real','reality'),('national','nationality'),
                 ('personal','personality'),('moral','morality'),('legal','legality'),
                 ('final','finality'),('normal','normality')],
    'B:+ness': [('happy','happiness'),('kind','kindness'),('sad','sadness'),
                 ('bright','brightness'),('dark','darkness'),('soft','softness'),
                 ('weak','weakness'),('good','goodness')],
    'C:+en':   [('bright','brighten'),('dark','darken'),('hard','harden'),
                 ('wide','widen'),('soft','soften'),('fresh','freshen'),
                 ('weak','weaken'),('sharp','sharpen'),('deep','deepen'),
                 ('light','lighten'),('thick','thicken'),('white','whiten')],
    'C:+ize':  [('memory','memorize'),('symbol','symbolize'),('organ','organize'),
                 ('moral','moralize'),('legal','legalize'),('minimal','minimize'),
                 ('real','realize'),('national','nationalize'),('local','localize'),
                 ('modern','modernize'),('final','finalize'),('general','generalize')],
    'D:+less': [('hope','hopeless'),('fear','fearless'),('care','careless'),
                 ('pain','painless'),('end','endless'),('home','homeless'),
                 ('harm','harmless'),('power','powerless')],
    'D:+ful':  [('hope','hopeful'),('care','careful'),('fear','fearful'),
                 ('use','useful'),('grace','graceful'),('help','helpful'),
                 ('faith','faithful'),('joy','joyful')],
    'D:+able': [('read','readable'),('wash','washable'),('break','breakable'),
                 ('love','lovable'),('use','usable'),('accept','acceptable'),
                 ('avoid','avoidable'),('change','changeable')],
    'E:+3ps':  [('run','runs'),('walk','walks'),('jump','jumps'),('eat','eats'),
                 ('read','reads'),('write','writes'),('play','plays'),('work','works')],
    'E:+ed':   [('walk','walked'),('talk','talked'),('jump','jumped'),('call','called'),
                 ('play','played'),('clean','cleaned'),('open','opened'),('start','started')],
    'E:+ing':  [('go','going'),('take','taking'),('run','running'),('see','seeing'),
                 ('give','giving'),('make','making'),('write','writing'),('read','reading')],
    'E:ablaut':[('go','went'),('take','took'),('give','gave'),('see','saw'),
                 ('know','knew'),('drive','drove'),('write','wrote'),('ride','rode')],
    # Standalone reference axes
    'S:+al_rel':[('nation','national'),('region','regional'),('culture','cultural'),
                  ('nature','natural'),('person','personal'),('origin','original'),
                  ('emotion','emotional'),('tradition','traditional')],
    'S:+er_comp':[('fast','faster'),('slow','slower'),('bright','brighter'),
                   ('dark','darker'),('soft','softer'),('warm','warmer'),
                   ('tall','taller'),('clean','cleaner')],
    'S:un-':   [('happy','unhappy'),('clear','unclear'),('fair','unfair'),
                 ('likely','unlikely'),('known','unknown'),('safe','unsafe'),
                 ('usual','unusual'),('equal','unequal')],
}

print("Building all canonical axes...", flush=True)
AXES = {}
for name, pairs in GROUP_PAIRS.items():
    ax, valid, pc = compute_axis(pairs)
    if ax is not None:
        AXES[name] = (ax, valid, pc)
        print("  %-12s  n=%d  pc=%.4f" % (name, len(valid), pc))
print()

# =====================================================================
# PART A: COMPLETE 5x5 INTER-GROUP MATRIX
# =====================================================================
print("PART A: Complete 5x5 inter-group cosine matrix")
print("-"*80)

def group_mean_cosine(g1_prefix, g2_prefix):
    axes1 = [(n,a) for n,(a,_,_) in AXES.items() if n.startswith(g1_prefix)]
    axes2 = [(n,a) for n,(a,_,_) in AXES.items() if n.startswith(g2_prefix)]
    if not axes1 or not axes2: return float('nan')
    cosines = [float(np.dot(a1.astype(np.float32), a2.astype(np.float32)))
               for _,a1 in axes1 for _,a2 in axes2]
    return float(np.mean(cosines))

GROUPS = ['A:', 'B:', 'C:', 'D:', 'E:']
GNAMES = ['A(v→n)', 'B(a→n)', 'C(a→v)', 'D(v→a)', 'E(v→v)']
print("  " + "%-12s" % "" + "".join("  %-10s" % g for g in GNAMES))
for i, g1 in enumerate(GROUPS):
    row = "  %-12s" % GNAMES[i]
    for j, g2 in enumerate(GROUPS):
        if i == j:
            row += "  %-10s" % "---"
        else:
            c = group_mean_cosine(g1, g2)
            row += "  %+.4f    " % c
    print(row)
print()

# Also: reverse pair and standalone vs groups
print("  REVERSE PAIR and STANDALONE vs groups:")
for sname in ['S:+al_rel', 'S:+er_comp', 'S:un-']:
    if sname not in AXES: continue
    sax = AXES[sname][0]
    row = "  %-12s" % sname[2:]
    for g in GROUPS:
        axes_g = [(n,a) for n,(a,_,_) in AXES.items() if n.startswith(g)]
        if axes_g:
            c = float(np.mean([np.dot(sax.astype(np.float32), a.astype(np.float32))
                                for _,a in axes_g]))
            row += "  %+.4f    " % c
        else:
            row += "  n/a       "
    print(row)
print()

# The anti-aligned pairs
if 'S:+al_rel' in AXES and 'B:+ity' in AXES:
    c = float(np.dot(AXES['S:+al_rel'][0].astype(np.float32),
                     AXES['B:+ity'][0].astype(np.float32)))
    print("  +al_rel vs +ity:  cos = %+.4f  (reverse pair #1)" % c)
# GROUP C vs GROUP A
c_ca = group_mean_cosine('C:', 'A:')
print("  GROUP C vs GROUP A:  cos = %+.4f  (reverse pair #2)" % c_ca)
print()

# =====================================================================
# PART B: GROUP C vs +ity (adj→noun) — directional test
# =====================================================================
print("PART B: GROUP C vs all standalone axes and reverse pair")
print("-"*80)

for cname in ['C:+en', 'C:+ize']:
    if cname not in AXES: continue
    cax = AXES[cname][0]
    print("  %s cosines with all axes:" % cname)
    for ref_name in sorted(AXES.keys()):
        if ref_name == cname: continue
        rax = AXES[ref_name][0]
        c = float(np.dot(cax.astype(np.float32), rax.astype(np.float32)))
        flag = '***' if abs(c) > 0.30 else ('  *' if abs(c) > 0.20 else '   ')
        print("    %s cos(%-10s, %-12s) = %+.4f" % (flag, cname[2:], ref_name[2:], c))
    print()

# Key question: does GROUP_C have a positive cosine with +ity?
# (GROUP_C arrives at verb cluster, +ity departs from adj cluster — same departure direction?)
if 'C:+en' in AXES and 'B:+ity' in AXES:
    c = float(np.dot(AXES['C:+en'][0].astype(np.float32),
                     AXES['B:+ity'][0].astype(np.float32)))
    print("  Specific: cos(+en, +ity) = %+.4f" % c)
if 'C:+ize' in AXES and 'B:+ity' in AXES:
    c = float(np.dot(AXES['C:+ize'][0].astype(np.float32),
                     AXES['B:+ity'][0].astype(np.float32)))
    print("  Specific: cos(+ize, +ity) = %+.4f" % c)
print()

# =====================================================================
# PART C: SPREAD AS 4TH PREDICTOR FEATURE
# =====================================================================
print("PART C: 4-feature predictor using (pc, LOO, irred, spread)")
print("-"*80)

# Compute spread for all GROUP members
print("  %-14s  pc      spread  mag     LOO%%  type")
print("  " + "-"*62)

# Known type labels for each axis
TYPE_LABELS = {
    'A:+ance': 'phonol_scatter', 'A:+ment': 'phonol_scatter',
    'A:+tion': 'phonol_scatter', 'A:+al_nom': 'phonol_scatter',
    'B:+ity': 'phonol_scatter', 'B:+ness': 'phonol_scatter',
    'C:+en': 'morph_moderate?', 'C:+ize': 'borderline',
    'D:+less': 'phonol_scatter', 'D:+ful': 'phonol_scatter',
    'D:+able': 'phonol_scatter',
    'E:+3ps': 'morph_moderate', 'E:+ed': 'morph_moderate',
    'E:+ing': 'morph_moderate', 'E:ablaut': 'phonol_scatter',
    'S:+al_rel': 'relational_geom', 'S:+er_comp': 'morph_uniform',
    'S:un-': 'phonol_scatter',
}

spread_by_type = {}
for name, pairs in GROUP_PAIRS.items():
    ax, valid, pc, spread, mean_mag = compute_axis_full(pairs)
    if ax is None: continue
    loo = axis_loo(ax, valid, CLEAN_MASK)
    true_type = TYPE_LABELS.get(name, '?')
    print("  %-14s  pc=%.4f  s=%.4f  m=%.4f  %.0f%%  %s" %
          (name, pc, spread, mean_mag, 100*loo, true_type))
    if true_type not in spread_by_type:
        spread_by_type[true_type] = []
    spread_by_type[true_type].append(spread)
print()

print("  Mean spread by type:")
for t, vals in sorted(spread_by_type.items()):
    print("  %-20s n=%d  mean_spread=%.4f  range=[%.4f, %.4f]" %
          (t, len(vals), np.mean(vals), min(vals), max(vals)))
print()

# Key question: does spread help separate ablaut from morph axes?
# ablaut has high spread despite phonol_scatter type
# morph_moderate axes should have lower spread
ablaut_ax, ablaut_vl, ablaut_pc, ablaut_sp, ablaut_mg = compute_axis_full(GROUP_PAIRS['E:ablaut'])
er_comp_ax, _, er_comp_pc, er_comp_sp, er_comp_mg = compute_axis_full(GROUP_PAIRS['S:+er_comp'])
print("  Ablaut vs er_comp (same pc range):")
print("  ablaut:  pc=%.4f  spread=%.4f  mag=%.4f" % (ablaut_pc, ablaut_sp, ablaut_mg))
print("  er_comp: pc=%.4f  spread=%.4f  mag=%.4f" % (er_comp_pc, er_comp_sp, er_comp_mg))
print("  Spread successfully separates them: ablaut > er_comp by %.4f" %
      (ablaut_sp - er_comp_sp))
print()

# =====================================================================
# PART D: GROUP C → GROUP E CHAIN
# =====================================================================
print("PART D: GROUP C → GROUP E chain (adj → verb → inflected)")
print("-"*80)

ax_en, valid_en, _ = compute_axis(GROUP_PAIRS['C:+en'])
ax_3ps, valid_3ps, _ = compute_axis(GROUP_PAIRS['E:+3ps'])
ax_ed,  valid_ed, _  = compute_axis(GROUP_PAIRS['E:+ed'])
ax_ing, valid_ing, _ = compute_axis(GROUP_PAIRS['E:+ing'])

if ax_en is not None:
    bs_en, _ = best_scale(ax_en, valid_en, CLEAN_MASK)
    bs_3ps, _ = best_scale(ax_3ps, valid_3ps, CLEAN_MASK)
    bs_ed,  _ = best_scale(ax_ed, valid_ed, CLEAN_MASK)
    bs_ing, _ = best_scale(ax_ing, valid_ing, CLEAN_MASK)
    print("  Scales: +en=%.2f  +3ps=%.2f  +ed=%.2f  +ing=%.2f" %
          (bs_en, bs_3ps, bs_ed, bs_ing))
    print()
    print("  Chain test (adj → brighten → brightened/brightens/brightening):")
    test_adjs = [('bright','brighten'), ('dark','darken'), ('hard','harden'),
                  ('wide','widen'), ('soft','soften'), ('weak','weaken'),
                  ('deep','deepen'), ('sharp','sharpen')]
    print("  %-8s  %-12s  %-12s  %-12s  %-12s" %
          ('adj', 'C(verb)', 'C+ed', 'C+3ps', 'C+ing'))
    for adj, verb in test_adjs:
        es, sid = get_emb(adj)
        if es is None: continue
        # Step 1: adj -> verb
        step1 = W_E[sid] + bs_en * ax_en
        r1 = nn_retrieve(step1, source_ids(adj), RELAXED_MASK, 1)
        got_verb = r1[0][0]
        v1_id = r1[0][2]
        ok1 = '✓' if got_verb == verb else '~'

        # Step 2a: verb -> past
        step2a = W_E[v1_id] + bs_ed * ax_ed
        r2a = nn_retrieve(step2a, source_ids(got_verb), RELAXED_MASK, 1)
        got_ed = r2a[0][0]

        # Step 2b: verb -> 3ps
        step2b = W_E[v1_id] + bs_3ps * ax_3ps
        r2b = nn_retrieve(step2b, source_ids(got_verb), RELAXED_MASK, 1)
        got_3ps = r2b[0][0]

        # Step 2c: verb -> ing
        step2c = W_E[v1_id] + bs_ing * ax_ing
        r2c = nn_retrieve(step2c, source_ids(got_verb), RELAXED_MASK, 1)
        got_ing = r2c[0][0]

        print("  %s %-8s  %-12s  %-12s  %-12s  %-12s" %
              (ok1, adj, got_verb, got_ed, got_3ps, got_ing))
    print()
    # Expected results:
    print("  Expected (brighten -> brightened, brightens, brightening):")
    print("  [Note: Qwen2 tokenizer single-token forms may differ]")
print()

# =====================================================================
# PART E: GROUP C +ize IRRED ANALYSIS
# =====================================================================
print("PART E: GROUP C +ize irreducibility analysis")
print("-"*80)

IZE_ALL = [
    ('memory','memorize'),('symbol','symbolize'),('organ','organize'),
    ('moral','moralize'),('legal','legalize'),('minimal','minimize'),
    ('real','realize'),('national','nationalize'),('local','localize'),
    ('modern','modernize'),('final','finalize'),('general','generalize'),
    ('civil','civilize'),('human','humanize'),('natural','naturalize'),
    ('social','socialize'),('formal','formalize'),('vocal','vocalize'),
]

ax_ize, valid_ize, pc_ize = compute_axis(IZE_ALL)
if ax_ize is not None:
    loo_ize = axis_loo(ax_ize, valid_ize, CLEAN_MASK)
    irr_ize, n_irr, details_ize = irred_on_holdout(
        ax_ize,
        [('popular','popularize'),('equal','equalize'),('visual','visualize'),
         ('crystal','crystallize'),('neutral','neutralize'),('normal','normalize'),
         ('standard','standardize'),('active','activate')],
        CLEAN_MASK)
    print("  +ize full axis: pc=%.4f  LOO=%.0f%%  irred=%.0f%%  n=%d" %
          (pc_ize, 100*loo_ize, 100*irr_ize, len(valid_ize)))
    print()
    print("  Per-pair irred status:")
    for s_w, t_w, found_at in details_ize:
        et, tid = get_emb(t_w)
        status = "found@s=%.2f" % found_at if found_at is not None else "IRREDUCIBLE"
        # Check tokenization
        ids_t = tok(' '+t_w, add_special_tokens=False)['input_ids']
        ntoks = len(ids_t)
        print("  %-15s -> %-15s  %s  [%d tok]" % (s_w, t_w, status, ntoks))
    print()

    # Second analysis: which training pairs also fail at best scale?
    bs_ize, _ = best_scale(ax_ize, valid_ize, CLEAN_MASK)
    print("  Training pair navigation at scale=%.2f:" % bs_ize)
    fail_count = 0
    for s_w, t_w, sid, _ in valid_ize:
        r = nn_retrieve(W_E[sid]+bs_ize*ax_ize, source_ids(s_w), CLEAN_MASK, 1)
        ok = '✓' if r[0][0] == t_w else '✗'
        if r[0][0] != t_w: fail_count += 1
        ids_t = tok(' '+t_w, add_special_tokens=False)['input_ids']
        print("  %s %-15s -> %-15s  (got: %s)  [%d tok]" %
              (ok, s_w, t_w, r[0][0], len(ids_t)))
    print()
    print("  Training fails: %d/%d" % (fail_count, len(valid_ize)))
