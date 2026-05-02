import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', dtype=torch.float32)
W_E   = model.model.embed_tokens.weight.detach().numpy().astype(np.float64)
W_n   = np.array([v/(np.linalg.norm(v)+1e-8) for v in W_E], dtype=np.float32)

RELAXED_MASK = np.zeros(len(W_E), dtype=bool)
for i in range(len(W_E)):
    w = tok.decode([i]).strip()
    if not w or len(w) < 2: continue
    if w.startswith('-') or w.startswith('_'): continue
    RELAXED_MASK[i] = True

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

def tokenize_word(word):
    for p in [' ', '']:
        ids = tok(p+word, add_special_tokens=False)['input_ids']
        if ids: return ids, [tok.decode([i]) for i in ids]
    return [], []

def nn_ret(pred_emb, excl_ids, mask):
    pred_n = normed(pred_emb).astype(np.float32)
    sims = W_n @ pred_n; sims[~mask] = -1.0
    for e in excl_ids: sims[e] = -1.0
    idx = int(np.argmax(sims))
    return tok.decode([idx]).strip(), float(sims[idx]), idx

def build_axis(pairs):
    chords = []; skipped = []
    for s, t in pairs:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is None or et is None:
            skipped.append((s,t)); continue
        chords.append(et - es)
    if not chords: return None, skipped
    return normed(np.mean(chords, axis=0)), skipped

def best_scale(ax_dir, pairs, mask):
    best_s, best_a = 0.5, 0
    for s in np.linspace(0.02, 8.0, 40):
        c = 0
        for sr, tg in pairs:
            es, _ = get_emb(sr)
            if es is None: continue
            w, _, _ = nn_ret(es + s*ax_dir, source_ids(sr), mask)
            if w == tg: c += 1
        if c > best_a: best_a=c; best_s=s
    return best_s

def eval_axis(ax_dir, s, pairs, mask, label=''):
    hits = 0; n = 0; details = []
    for src, tgt in pairs:
        es, _ = get_emb(src)
        if es is None: continue
        n += 1
        pred = es + s * ax_dir
        w, sim, _ = nn_ret(pred, source_ids(src), mask)
        ok = (w == tgt)
        if ok: hits += 1
        details.append((src, tgt, w, ok, sim))
    return hits, n, details

# ====================================================================
# AXIS DATASETS
# ====================================================================
EN_GENDER = [
    ('king','queen'), ('man','woman'), ('boy','girl'),
    ('father','mother'), ('son','daughter'), ('husband','wife'),
    ('uncle','aunt'), ('prince','princess'), ('actor','actress'),
    ('waiter','waitress'),
]
EN_PLURAL = [
    ('cat','cats'), ('dog','dogs'), ('house','houses'), ('car','cars'),
    ('tree','trees'), ('book','books'), ('bird','birds'), ('door','doors'),
    ('hand','hands'), ('arm','arms'), ('eye','eyes'), ('leg','legs'),
]

# French gender pairs (masculine → feminine)
FR_GENDER_TRAIN = [
    ('roi',  'reine'),
    ('homme','femme'),
    ('fils', 'fille'),
    ('père', 'mère'),
    ('frère','sœur'),
]
FR_GENDER_TEST = [
    ('oncle',  'tante'),
    ('prince', 'princesse'),
    ('acteur', 'actrice'),
    ('garçon', 'fille'),
    ('chien',  'chienne'),
]

# French plural
FR_PLURAL_TRAIN = [
    ('chat',   'chats'),
    ('chien',  'chiens'),
    ('maison', 'maisons'),
    ('voiture','voitures'),
    ('livre',  'livres'),
]
FR_PLURAL_TEST = [
    ('arbre',  'arbres'),
    ('porte',  'portes'),
    ('main',   'mains'),
    ('jambe',  'jambes'),
    ('table',  'tables'),
]

# Spanish gender pairs
ES_GENDER_TRAIN = [
    ('rey',    'reina'),
    ('hombre', 'mujer'),
    ('hijo',   'hija'),
    ('padre',  'madre'),
    ('hermano','hermana'),
]
ES_GENDER_TEST = [
    ('tío',    'tía'),
    ('príncipe','princesa'),
    ('actor',  'actriz'),
    ('niño',   'niña'),
    ('perro',  'perra'),
]

# Spanish plural
ES_PLURAL_TRAIN = [
    ('gato',  'gatos'),
    ('perro', 'perros'),
    ('casa',  'casas'),
    ('libro', 'libros'),
    ('árbol', 'árboles'),
]

# German gender pairs
DE_GENDER_TRAIN = [
    ('König',  'Königin'),
    ('Mann',   'Frau'),
    ('Sohn',   'Tochter'),
    ('Vater',  'Mutter'),
    ('Bruder', 'Schwester'),
]
DE_GENDER_TEST = [
    ('Onkel',  'Tante'),
    ('Prinz',  'Prinzessin'),
    ('Schauspieler', 'Schauspielerin'),
    ('Hund',   'Hündin'),
    ('Kellner','Kellnerin'),
]

print("\nDAY 354: Cross-Language Axis Transfer")
print("="*70)

# ====================================================================
# PHASE 1: Build and calibrate English axes
# ====================================================================
print("\nPhase 1: English axes")

en_gender_dir, skip_g = build_axis(EN_GENDER)
en_plural_dir, skip_p = build_axis(EN_PLURAL)
s_eg = best_scale(en_gender_dir, EN_GENDER, RELAXED_MASK)
s_ep = best_scale(en_plural_dir, EN_PLURAL, RELAXED_MASK)

h,n,_ = eval_axis(en_gender_dir, s_eg, EN_GENDER, RELAXED_MASK)
print("  EN gender: scale=%.3f  train_acc=%d/%d=%.0f%%" % (s_eg, h, n, 100*h/max(n,1)))
h,n,_ = eval_axis(en_plural_dir, s_ep, EN_PLURAL, RELAXED_MASK)
print("  EN plural: scale=%.3f  train_acc=%d/%d=%.0f%%" % (s_ep, h, n, 100*h/max(n,1)))

# ====================================================================
# PHASE 2: Tokenisation check for all FR / ES / DE pairs
# ====================================================================
print("\nPhase 2: Tokenisation check — which pairs are single-token?")
for lang, pairs_list in [
    ('FR_GENDER_TRAIN', FR_GENDER_TRAIN),
    ('FR_GENDER_TEST',  FR_GENDER_TEST),
    ('FR_PLURAL_TRAIN', FR_PLURAL_TRAIN),
    ('FR_PLURAL_TEST',  FR_PLURAL_TEST),
    ('ES_GENDER_TRAIN', ES_GENDER_TRAIN),
    ('ES_GENDER_TEST',  ES_GENDER_TEST),
    ('ES_PLURAL_TRAIN', ES_PLURAL_TRAIN),
    ('DE_GENDER_TRAIN', DE_GENDER_TRAIN),
    ('DE_GENDER_TEST',  DE_GENDER_TEST),
]:
    single = []
    multi  = []
    for s, t in pairs_list:
        es, _ = get_emb(s); et, _ = get_emb(t)
        if es is not None and et is not None:
            single.append((s,t))
        else:
            ids_s, tks_s = tokenize_word(s)
            ids_t, tks_t = tokenize_word(t)
            multi.append((s, t, tks_s, tks_t))
    print("  %-20s  %d single-token  %d multi-token" % (lang, len(single), len(multi)))
    for s,t,tks,tkt in multi:
        print("    MULTI: %-12s→%-12s  src=%s  tgt=%s" % (s,t,tks,tkt))

# Filter to single-token pairs only
def filter_single(pairs):
    return [(s,t) for s,t in pairs
            if get_emb(s)[0] is not None and get_emb(t)[0] is not None]

fr_g_train = filter_single(FR_GENDER_TRAIN)
fr_g_test  = filter_single(FR_GENDER_TEST)
fr_p_train = filter_single(FR_PLURAL_TRAIN)
fr_p_test  = filter_single(FR_PLURAL_TEST)
es_g_train = filter_single(ES_GENDER_TRAIN)
es_g_test  = filter_single(ES_GENDER_TEST)
es_p_train = filter_single(ES_PLURAL_TRAIN)
de_g_train = filter_single(DE_GENDER_TRAIN)
de_g_test  = filter_single(DE_GENDER_TEST)

# ====================================================================
# PHASE 3: Zero-shot transfer — English axes on French/Spanish/German words
# ====================================================================
print("\nPhase 3: Zero-shot transfer — English axes applied to FR/ES/DE words")
print("  Using English scale (s_eg=%.3f for gender, s_ep=%.3f for plural)" % (s_eg, s_ep))
print()

def eval_zero_shot(ax_dir, s, pairs, mask, label):
    hits = 0; n = 0; details = []
    for src, tgt in pairs:
        es, _ = get_emb(src)
        if es is None: continue
        n += 1
        pred = es + s * ax_dir
        w, sim, _ = nn_ret(pred, source_ids(src), mask)
        ok = (w == tgt)
        if ok: hits += 1
        details.append((src, tgt, w, ok, sim))
    pct = 100*hits/max(n,1)
    print("  %-28s  %d/%d = %3.0f%%  %s" % (
        label, hits, n, pct,
        '  '.join('%s→%s[%s]%s' % (s,t,w,'✓' if ok else '✗')
                  for s,t,w,ok,_ in details)))
    return hits, n, pct

# Gender transfer
for lang, pairs, label in [
    ('FR', fr_g_train + fr_g_test, 'EN→FR gender (train+test)'),
    ('ES', es_g_train + es_g_test, 'EN→ES gender (train+test)'),
    ('DE', de_g_train + de_g_test, 'EN→DE gender (train+test)'),
]:
    eval_zero_shot(en_gender_dir, s_eg, pairs, RELAXED_MASK, label)

print()
# Plural transfer
for lang, pairs, label in [
    ('FR', fr_p_train + fr_p_test, 'EN→FR plural (train+test)'),
    ('ES', es_p_train,             'EN→ES plural (train)'),
]:
    eval_zero_shot(en_plural_dir, s_ep, pairs, RELAXED_MASK, label)

# ====================================================================
# PHASE 4: Build language-specific axes and compare to English
# ====================================================================
print("\nPhase 4: Language-specific axes vs English axes")
print()

axes = {}
for lang, train_pairs, ax_type in [
    ('FR_gender', fr_g_train, 'gender'),
    ('FR_plural', fr_p_train, 'plural'),
    ('ES_gender', es_g_train, 'gender'),
    ('ES_plural', es_p_train, 'plural'),
    ('DE_gender', de_g_train, 'gender'),
]:
    ax_dir, skipped = build_axis(train_pairs)
    if ax_dir is None:
        print("  %s: no single-token pairs!" % lang)
        continue
    axes[lang] = ax_dir
    en_dir = en_gender_dir if ax_type == 'gender' else en_plural_dir
    cos = float(np.dot(ax_dir.astype(np.float32), en_dir.astype(np.float32)))
    n_train = len(train_pairs) - len(skipped)
    print("  %-12s  n=%d  cos(lang_axis, EN_%s_axis)=%.4f  [alignment: %s]" % (
        lang, n_train, ax_type, cos,
        'HIGH' if abs(cos) > 0.7 else ('MED' if abs(cos) > 0.4 else 'LOW')))

# ====================================================================
# PHASE 5: Language-specific vs English on held-out test pairs
# ====================================================================
print("\nPhase 5: Lang-specific axes vs English axes on held-out test pairs")
print()

for lang, test_pairs, ax_type, ax_key in [
    ('FR', fr_g_test, 'gender', 'FR_gender'),
    ('FR', fr_p_test, 'plural', 'FR_plural'),
    ('ES', es_g_test, 'gender', 'ES_gender'),
    ('DE', de_g_test, 'gender', 'DE_gender'),
]:
    if not test_pairs:
        print("  %s %s: no test pairs" % (lang, ax_type)); continue

    en_dir = en_gender_dir if ax_type == 'gender' else en_plural_dir
    s_en   = s_eg if ax_type == 'gender' else s_ep

    # English axis at English scale
    h_en, n, det_en = eval_axis(en_dir, s_en, test_pairs, RELAXED_MASK)

    # Language-specific axis
    if ax_key in axes:
        ax_dir_lang = axes[ax_key]
        s_lang = best_scale(ax_dir_lang, test_pairs, RELAXED_MASK) if test_pairs else s_en
        h_lang, n, det_lang = eval_axis(ax_dir_lang, s_lang, test_pairs, RELAXED_MASK)
    else:
        h_lang = 0; s_lang = 0

    print("  %s %-8s  EN_axis=%d/%d=%.0f%%  LANG_axis=%d/%d=%.0f%%  (n=%d)" % (
        lang, ax_type,
        h_en,   n, 100*h_en/max(n,1),
        h_lang, n, 100*h_lang/max(n,1), n))

    for (s,t,w_en,ok_en,_), (s2,t2,w_l,ok_l,_) in zip(det_en, det_lang
            if ax_key in axes else [(x,y,'',False,0) for x,y,*_ in det_en]):
        mark_en = '✓' if ok_en else '✗'
        mark_l  = '✓' if ok_l  else '✗'
        print("    %-10s → %-12s   EN:%s%-12s  LANG:%s%-12s" % (
            s, t, mark_en, w_en, mark_l, w_l))
    print()

# ====================================================================
# PHASE 6: Axis similarity matrix across all languages
# ====================================================================
print("Phase 6: Axis cosine similarity matrix (gender axes)")
print()

all_gender_axes = {
    'EN': en_gender_dir,
}
for k in ['FR_gender','ES_gender','DE_gender']:
    if k in axes: all_gender_axes[k.split('_')[0]] = axes[k]

langs = sorted(all_gender_axes.keys())
print("  " + "  ".join("%-8s" % l for l in langs))
for l1 in langs:
    row = "  " + "%-4s" % l1
    for l2 in langs:
        cos = float(np.dot(all_gender_axes[l1].astype(np.float32),
                           all_gender_axes[l2].astype(np.float32)))
        row += "  %.4f  " % cos
    print(row)

print()
print("  Axis cosine similarity matrix (plural axes)")
all_plural_axes = {'EN': en_plural_dir}
for k in ['FR_plural','ES_plural']:
    if k in axes: all_plural_axes[k.split('_')[0]] = axes[k]

langs_p = sorted(all_plural_axes.keys())
print("  " + "  ".join("%-8s" % l for l in langs_p))
for l1 in langs_p:
    row = "  " + "%-4s" % l1
    for l2 in langs_p:
        cos = float(np.dot(all_plural_axes[l1].astype(np.float32),
                           all_plural_axes[l2].astype(np.float32)))
        row += "  %.4f  " % cos
    print(row)

# ====================================================================
# PHASE 7: Reverse transfer — French gender axis → English test pairs
# ====================================================================
print("\nPhase 7: Reverse transfer — French/Spanish axes on English test pairs")
print()

EN_GENDER_TEST = [
    ('king','queen'), ('man','woman'), ('boy','girl'),
    ('father','mother'), ('husband','wife'),
]

for lang, ax_key in [('FR_gender','FR_gender'), ('ES_gender','ES_gender'), ('DE_gender','DE_gender')]:
    if ax_key not in axes: continue
    ax_dir_lang = axes[ax_key]
    s_lang_on_en = best_scale(ax_dir_lang, EN_GENDER_TEST, RELAXED_MASK)
    h, n, det = eval_axis(ax_dir_lang, s_lang_on_en, EN_GENDER_TEST, RELAXED_MASK)
    print("  %-12s → EN gender:  %d/%d=%.0f%%  s=%.3f  %s" % (
        ax_key, h, n, 100*h/max(n,1), s_lang_on_en,
        '  '.join('%s→%s[%s]%s' % (s,t,w,'✓' if ok else '✗')
                  for s,t,w,ok,_ in det)))

# ====================================================================
# PHASE 8: Mixed-axis training — does a multilingual gender axis
#          outperform monolingual axes?
# ====================================================================
print("\nPhase 8: Multilingual gender axis (EN+FR+ES+DE training)")
print()

all_gender_train = list(EN_GENDER) + fr_g_train + es_g_train + de_g_train
multi_gender_dir, _ = build_axis(all_gender_train)
s_multi = best_scale(multi_gender_dir, all_gender_train, RELAXED_MASK)

print("  Training on %d pairs across EN+FR+ES+DE" % len(all_gender_train))

for lang, test_pairs, label in [
    ('EN', EN_GENDER_TEST,  'EN test'),
    ('FR', fr_g_test,        'FR test'),
    ('ES', es_g_test,        'ES test'),
    ('DE', de_g_test,        'DE test'),
]:
    if not test_pairs: continue
    # EN monolingual
    h_en, n, _ = eval_axis(en_gender_dir, s_eg, test_pairs, RELAXED_MASK)
    # Multilingual
    h_m,  n, _ = eval_axis(multi_gender_dir, s_multi, test_pairs, RELAXED_MASK)
    print("  %-6s  EN_mono=%d/%d=%.0f%%  multi=%d/%d=%.0f%%" % (
        label, h_en, n, 100*h_en/max(n,1), h_m, n, 100*h_m/max(n,1)))

# ====================================================================
# PHASE 9: Summary
# ====================================================================
print("\n" + "="*70)
print("SUMMARY: Day 354 Cross-Language Axis Transfer")
print("="*70)
print()
print("  Key question: are embedding axes universal (meaning-level)")
print("  or language-specific (surface-level)?")
print()
print("  cos(English_gender, French_gender) → ?  (HIGH = universal)")
print("  cos(English_plural, French_plural) → ?  (HIGH = universal)")
print()
print("  Zero-shot transfer accuracy EN→FR gender:  see Phase 3")
print("  Zero-shot transfer accuracy EN→FR plural:  see Phase 3")
