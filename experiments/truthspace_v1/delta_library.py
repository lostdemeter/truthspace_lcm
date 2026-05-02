#!/usr/bin/env python3
"""
delta_library.py — Expanded geometric delta library for TruthSpace LCM.

Defines 8 semantic relationships as learned delta vectors, each characterised
by (source, target) pairs from which we learn the mean phi4 displacement.

Usage:
    python delta_library.py              # full demo + LOO report
    from delta_library import DeltaLibrary, build_lcm
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc299_phase4_lcm_inference import LCMIndex

# ─── Relationship catalogue ───────────────────────────────────────────────────
#
# Each entry: pairs, keywords (for routing), direction label, description.
# Only use single-token or _get_proj-reachable words.

RELATIONSHIPS = {
    # ── Geographic / factual ──────────────────────────────────────────────────
    'capital_of': {
        'pairs': [
            ('France','Paris'),('Germany','Berlin'),('Japan','Tokyo'),
            ('China','Beijing'),('Italy','Rome'),('Spain','Madrid'),
            ('Russia','Moscow'),('Greece','Athens'),('Poland','Warsaw'),
            ('Sweden','Stockholm'),('Norway','Oslo'),('Austria','Vienna'),
            ('Belgium','Brussels'),('Netherlands','Amsterdam'),
        ],
        'keywords': {'capital','capitals','city'},
        'description': 'country → capital city',
        'direction': '→',
    },
    'city_country': {
        'pairs': [
            ('Paris','France'),('Berlin','Germany'),('Tokyo','Japan'),
            ('Rome','Italy'),('Madrid','Spain'),('Moscow','Russia'),
            ('Beijing','China'),('Athens','Greece'),('Warsaw','Poland'),
            ('Stockholm','Sweden'),('Oslo','Norway'),('Vienna','Austria'),
            ('Brussels','Belgium'),('Amsterdam','Netherlands'),
        ],
        'keywords': {'country','nation','located','belong','part'},
        'description': 'city → the country it is in',
        'direction': '→',
    },
    'country_language': {
        'pairs': [
            ('France','French'),('Germany','German'),('Japan','Japanese'),
            ('China','Chinese'),('Italy','Italian'),('Spain','Spanish'),
            ('Russia','Russian'),('Greece','Greek'),('Poland','Polish'),
            ('Sweden','Swedish'),('Norway','Norwegian'),
        ],
        'keywords': {'language','speak','speaks','spoken','tongue','official'},
        'description': 'country → official language',
        'direction': '→',
    },
    'country_currency': {
        'pairs': [
            ('France','euro'),('Japan','yen'),('China','yuan'),
            ('UK','pound'),('Mexico','peso'),('Korea','won'),
            ('Switzerland','franc'),('Brazil','real'),
        ],
        'keywords': {'currency','money','coin','cash','pay','denomination'},
        'description': 'country → national currency',
        'direction': '→',
    },
    # ── Gender ────────────────────────────────────────────────────────────────
    'male_female': {
        'pairs': [
            ('king','queen'),('man','woman'),('boy','girl'),
            ('father','mother'),('brother','sister'),('actor','actress'),
            ('prince','princess'),('hero','heroine'),('son','daughter'),
            ('husband','wife'),
        ],
        'keywords': {'female','woman','feminine','girl','sister','wife'},
        'description': 'male form → female form',
        'direction': '→',
        'bidirectional': True,
    },
    # ── Semantic opposites ────────────────────────────────────────────────────
    'antonym': {
        'pairs': [
            ('hot','cold'),('big','small'),('fast','slow'),
            ('light','dark'),('old','young'),('good','bad'),
            ('rich','poor'),('hard','soft'),('clean','dirty'),
            ('wet','dry'),('strong','weak'),('loud','quiet'),
            ('happy','sad'),('love','hate'),
        ],
        'keywords': {'opposite','antonym','contrary','reverse','not'},
        'description': 'word → semantic opposite',
        'direction': '↔',
        'bidirectional': True,
    },
    # ── Morphological: adjective → comparative ────────────────────────────────
    'comparative': {
        'pairs': [
            ('big','bigger'),('small','smaller'),('fast','faster'),
            ('old','older'),('young','younger'),('cold','colder'),
            ('hot','hotter'),('tall','taller'),('short','shorter'),
            ('long','longer'),('hard','harder'),('soft','softer'),
            ('dark','darker'),('bright','brighter'),
        ],
        'keywords': {'more','comparative','than','er','compare'},
        'description': 'adjective → comparative form',
        'direction': '→',
    },
    # ── Morphological: present → simple past ─────────────────────────────────
    # NOTE: no seed axis for verb tense → boost never fires → pure cosine.
    # Typical result: answer at rank 1-2 (71% top3, 14% top1 with 14 pairs).
    # More pairs reduce noise; 20+ pairs give stable top3 accuracy.
    'past_tense': {
        'pairs': [
            ('run','ran'),('eat','ate'),('see','saw'),
            ('make','made'),('give','gave'),('come','came'),
            ('take','took'),('know','knew'),('think','thought'),
            ('buy','bought'),('find','found'),('write','wrote'),
            ('speak','spoke'),('break','broke'),
            ('go','went'),('say','said'),('have','had'),
            ('do','did'),('stand','stood'),('hold','held'),
            ('keep','kept'),('leave','left'),('feel','felt'),
        ],
        'keywords': {'past','yesterday','ago','did','was','were','tense'},
        'description': 'verb present → simple past tense',
        'direction': '→',
    },
    # ── Antonym sub-deltas: domain-specific (universal antonym is incoherent) ──
    # Coherence analysis shows only 'speed' domain pairs align well (cos≈0.59).
    # Temperature and valence pairs point in conflicting directions.
    # Each sub-delta is limited to pairs that share a semantic axis.
    'antonym_speed': {
        'pairs': [
            ('fast','slow'),('quick','slow'),('rapid','slow'),
            ('swift','slow'),('fast','gradual'),('quick','gradual'),
        ],
        'keywords': {'speed','pace','rate','quick','fast','slow'},
        'description': 'speed-domain antonym (fast→slow)',
        'direction': '↔',
        'bidirectional': True,
    },
    'antonym_size': {
        'pairs': [
            ('big','small'),('large','small'),('huge','tiny'),
            ('tall','short'),('long','short'),('wide','narrow'),
            ('broad','narrow'),('great','little'),
        ],
        'keywords': {'size','big','large','huge','tall','long','wide'},
        'description': 'size-domain antonym (big→small)',
        'direction': '↔',
        'bidirectional': True,
    },
    'antonym_valence': {
        'pairs': [
            ('good','bad'),('happy','sad'),('love','hate'),
            ('joy','sorrow'),('beautiful','ugly'),('kind','cruel'),
            ('brave','coward'),('honest','dishonest'),
        ],
        'keywords': {'opposite','antonym','contrary','reverse','not'},
        'description': 'valence-domain antonym (good→bad)',
        'direction': '↔',
        'bidirectional': True,
    },
}

# ─── DeltaLibrary ─────────────────────────────────────────────────────────────

class DeltaLibrary:
    """Learned delta vectors for TruthSpace geometric Q&A."""

    def __init__(self, lcm: LCMIndex):
        self.lcm = lcm
        self._rel   = {}   # name → dict from RELATIONSHIPS
        self._delta = {}   # name → delta ndarray (float32)
        self._fp    = {}   # name → flip_prob ndarray
        self._pairs = {}   # name → filtered (valid) pairs
        self._build()

    # ── Construction ──────────────────────────────────────────────────────────

    def _word_ok(self, word):
        try:
            self.lcm._get_proj(word)
            return True
        except RuntimeError:
            return False

    def _valid_pairs(self, pairs):
        return [(s, t) for s, t in pairs if self._word_ok(s) and self._word_ok(t)]

    def _build(self):
        for name, info in RELATIONSHIPS.items():
            vp = self._valid_pairs(info['pairs'])
            if len(vp) < 3:
                print(f'  [SKIP] {name}: only {len(vp)} valid pairs')
                continue
            delta, _       = self.lcm.learn_delta(vp)
            _, _, fp, _    = self.lcm.learn_delta_v2(vp)
            self._rel[name]   = info
            self._delta[name] = delta
            self._fp[name]    = fp
            self._pairs[name] = vp

    # ── Inference ─────────────────────────────────────────────────────────────

    def answer(self, source, relationship, k=5, exclude=None,
               context_words=None):
        """Apply a named relationship delta to *source*, return top-k.

        context_words: optional list of contextually relevant words from the
        query.  When provided, *source*'s projection is shifted toward the
        context words via inverse-falloff gravity (DC 302) before delta
        application.  This resolves polysemy and reduces near-miss failures.
        """
        if relationship not in self._delta:
            raise KeyError(f'Unknown relationship: {relationship}')
        delta = self._delta[relationship]
        fp    = self._fp[relationship]
        info  = self._rel[relationship]

        # Bidirectional relationships (antonym, gender): choose delta direction
        # by checking whether source projects on the same side as the mean source
        # cluster.  mean_src is the centroid of all source projections.
        if info.get('bidirectional'):
            try:
                src_proj, _  = self.lcm._get_proj(source)
                src_vecs     = []
                for s, _ in self._pairs[relationship]:
                    try:
                        p, _ = self.lcm._get_proj(s)
                        src_vecs.append(p.astype(np.float64))
                    except RuntimeError:
                        pass
                if src_vecs:
                    mean_src  = np.mean(src_vecs, axis=0)
                    # If source is closer to the TARGET cluster than source cluster,
                    # flip so we head back toward source cluster.
                    tgt_vecs  = []
                    for _, t in self._pairs[relationship]:
                        try:
                            p, _ = self.lcm._get_proj(t)
                            tgt_vecs.append(p.astype(np.float64))
                        except RuntimeError:
                            pass
                    mean_tgt  = np.mean(tgt_vecs, axis=0) if tgt_vecs else mean_src
                    sp        = src_proj.astype(np.float64)
                    dist_src  = float(np.linalg.norm(sp - mean_src))
                    dist_tgt  = float(np.linalg.norm(sp - mean_tgt))
                    if dist_tgt < dist_src:   # source is nearer to target cluster
                        delta = -delta        # flip to head back toward source
            except RuntimeError:
                pass

        excl = [source] + (exclude or [])
        src_proj = None
        if context_words:
            src_proj = self.lcm.context_correct_proj(source, context_words)
        return self.lcm.apply_delta_phi_boost_v8(
            source, delta, fp, k=k,
            exclude_words=excl, boost_threshold=0.75,
            source_proj=src_proj)

    # ── LOO evaluation ────────────────────────────────────────────────────────

    def loo_test(self, name, boost_threshold=0.75):
        """Leave-one-out test: for each pair, learn delta from the rest.

        Returns list of (source, target, rank) where rank is 0-indexed position
        of target in the top-50 results (-1 if not found).

        NOTE: always tests in the FORWARD direction — pairs are defined
        src→tgt and the delta is applied to src expecting tgt.
        No bidirectional flip is applied (that is only for inference-time
        reverse queries).
        """
        vp = self._pairs[name]
        results = []
        for i, (src, tgt) in enumerate(vp):
            loo_pairs = [p for j, p in enumerate(vp) if j != i]
            if len(loo_pairs) < 2:
                results.append((src, tgt, -1))
                continue
            delta_loo, _    = self.lcm.learn_delta(loo_pairs)
            _, _, fp_loo, _ = self.lcm.learn_delta_v2(loo_pairs)
            top50 = self.lcm.apply_delta_phi_boost_v8(
                src, delta_loo, fp_loo, k=50,
                exclude_words=[src], boost_threshold=boost_threshold)
            rank = next(
                (r for r, (w, _) in enumerate(top50)
                 if w.lower() == tgt.lower()), -1)
            results.append((src, tgt, rank))
        return results

    # ── Keyword routing ───────────────────────────────────────────────────────

    def route(self, query_words):
        """Return relationship name that best matches query keywords, or None."""
        words = {w.lower() for w in query_words}
        for name, info in self._rel.items():
            if words & info['keywords']:
                return name
        return None

    # ── Report ────────────────────────────────────────────────────────────────

    def report(self):
        """Print LOO accuracy for every loaded relationship."""
        SEP = '─' * 70
        print(SEP)
        print(f'  {"relationship":<18s} {"pairs":>5}  {"top1":>5}  {"top3":>5}  '
              f'{"mrr":>5}  worst failures')
        print(SEP)
        for name in self._rel:
            loo = self.loo_test(name)
            n = len(loo)
            top1 = sum(r == 0 for _, _, r in loo)
            top3 = sum(0 <= r <= 2 for _, _, r in loo)
            mrr  = sum(1.0 / (r + 1) for _, _, r in loo if r >= 0) / n
            failures = [(s, t, r) for s, t, r in loo if r != 0]
            fail_str = ', '.join(f'{s}→{t}(r{r})' for s, t, r in
                                 sorted(failures, key=lambda x: -x[2])[:3])
            top1_pct = f'{top1/n:.0%}'
            top3_pct = f'{top3/n:.0%}'
            mrr_str  = f'{mrr:.2f}'
            print(f'  {name:<18s} {n:>5}  {top1_pct:>5}  {top3_pct:>5}  '
                  f'{mrr_str:>5}  {fail_str}')
        print(SEP)

    # ── Convenience ───────────────────────────────────────────────────────────

    def __contains__(self, name):
        return name in self._delta

    def __iter__(self):
        return iter(self._rel)

    @property
    def names(self):
        return list(self._rel)


# ─── Bootstrap helpers ────────────────────────────────────────────────────────

def build_lcm():
    """Build LCMIndex with all required word injections."""
    lcm = LCMIndex()
    lcm.add_word('Oslo',   57858, overwrite=True)
    lcm.add_word('Greek',  17860, overwrite=False)
    lcm.add_word('boy',     8171, overwrite=False)
    lcm.add_word('girl',    3743, overwrite=False)
    lcm.add_word('Polish', 31984, overwrite=True)
    return lcm


# ─── Demo / main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Building LCM index…')
    lcm = build_lcm()

    print('Learning delta library…')
    lib = DeltaLibrary(lcm)

    # ── LOO accuracy report ───────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('LEAVE-ONE-OUT ACCURACY REPORT')
    print('=' * 70)
    lib.report()

    # ── Per-relationship sample answers ──────────────────────────────────────
    STOPS = {'what','is','the','of','do','does','which','in','a','an','are',
             'speak','spoken','there','version','equivalent','word','for',
             'form','tense','comparative','past'}

    ALL_SOURCES = set()
    for name in lib._rel:
        for s, _ in lib._pairs[name]:
            ALL_SOURCES.add(s.lower())

    QA_TESTS = [
        # (query,                                  expected)
        ('What is the capital of France',          'Paris'),
        ('What is the capital of Norway',          'Oslo'),
        ('What is the capital of Japan',           'Tokyo'),
        ('What country is Paris in',               'France'),
        ('What country is Tokyo in',               'Japan'),
        ('What language is spoken in France',      'French'),
        ('What language is spoken in Japan',       'Japanese'),
        ('What is the currency of France',         'euro'),
        ('What is the currency of Japan',          'yen'),
        ('What is the currency of UK',             'pound'),
        ('What is the currency of Mexico',         'peso'),
        ('What is the female version of king',     'queen'),
        ('Female equivalent of hero',              'heroine'),
        ('What is the opposite of hot',            'cold'),
        ('What is the opposite of big',            'small'),
        ('What is the opposite of happy',          'sad'),
        ('Comparative form of big',                'bigger'),
        ('Comparative form of old',                'older'),
        ('Comparative form of fast',               'faster'),
        ('Past tense of run',                      'ran'),
        ('Past tense of eat',                      'ate'),
        ('Past tense of think',                    'thought'),
    ]

    print('\n' + '=' * 70)
    print('END-TO-END QA  (keyword routing + v8 geometric retrieval)')
    print('=' * 70)
    ok = 0
    for query, expected in QA_TESTS:
        words   = query.lower().split()
        rel     = lib.route(words)
        if rel is None:
            print(f'  ✗ "{query}" → NO ROUTE')
            continue
        # source = first word in ALL_SOURCES found in query
        source  = None
        for w in words:
            if w in ALL_SOURCES:
                source = w; break
        if source is None:
            # fallback: first non-stop content word
            for w in words:
                if w not in STOPS and lib._word_ok(w):
                    source = w; break
        if source is None:
            print(f'  ✗ "{query}" → no source found')
            continue
        top5    = lib.answer(source, rel, k=5)
        top5w   = [w for w, _ in top5]
        rank    = next((i for i, w in enumerate(top5w)
                        if w.lower() == expected.lower()), -1)
        mark    = '✓' if rank == 0 else ('~' if rank > 0 else '✗')
        ok     += (rank == 0)
        print(f'  {mark} "{query}"')
        print(f'     src={source}  rel={rel}  '
              f'top3={top5w[:3]}  '
              f'({"rank " + str(rank) if rank >= 0 else "missing"}  exp={expected})')

    print(f'\n  QA accuracy: {ok}/{len(QA_TESTS)}')

    # ── Delta geometry table ──────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('DELTA GEOMETRY — cosine similarity matrix between relationships')
    print('=' * 70)
    names = lib.names

    def unit(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-20)

    units = {n: unit(lib._delta[n].astype(np.float64)) for n in names}
    header = '  ' + ''.join(f'{n[:9]:>10s}' for n in names)
    print(header)
    for a in names:
        row = f'  {a[:9]:<9s}'
        for b in names:
            c = float(np.dot(units[a], units[b]))
            row += f'  {c:+.3f}  '[:-1]
        print(row)
