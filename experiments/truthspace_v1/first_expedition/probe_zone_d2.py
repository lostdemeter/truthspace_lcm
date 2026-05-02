import json
from collections import Counter

with open("day27_atlas.json") as f:
    atlas = json.load(f)
wmap = atlas['word_map']

# Proper zone split by phase AND body
zone_c  = {w: v for w, v in wmap.items() if v['phase']==2 and v.get('L14_body') not in ('B000','B001',None)}
body_b000 = {w: v for w, v in wmap.items() if v['phase']==2 and v.get('L14_body') == 'B000'}
body_b001 = {w: v for w, v in wmap.items() if v['phase']==2 and v.get('L14_body') == 'B001'}
no_body   = {w: v for w, v in wmap.items() if v['phase']==2 and v.get('L14_body') is None}
other     = {w: v for w, v in wmap.items() if v['phase']!=2}

print(f"Zone C (95 specific bodies):  {len(zone_c):5d}")
print(f"B000 'Verbs of Strong Impact': {len(body_b000):5d}")
print(f"B001 'data and operations':    {len(body_b001):5d}")
print(f"No body (L14_body=None):       {len(no_body):5d}")
print(f"Phase != 2 (Zone A/B/etc):     {len(other):5d}")

print(f"\nB000 sample (50):")
print([w.strip() for w in sorted(body_b000.keys())[:50]])
print(f"\nB001 sample (50):")
print([w.strip() for w in sorted(body_b001.keys())[:50]])

# Verb form lookup using all prefix variants
def find_in_wmap(word):
    for pfx in [' '+word, word, '▁'+word]:
        if pfx in wmap:
            v = wmap[pfx]
            bd = v.get('L14_body')
            phase = v.get('phase', '?')
            lbl = v.get('L14_label','?')
            if bd in ('B000','B001',None):
                zone = 'D'
            elif bd is None:
                zone = '?'
            else:
                zone = 'C'
            return pfx, zone, bd, lbl, phase
    return None, '?', '?', '?', '?'

print("\nVerb form zones (all forms):")
verb_groups = [
    ('walk','walked','walking','walks'),
    ('run','ran','running','runs'),
    ('eat','ate','eating','eats'),
    ('kill','killed','killing','kills'),
    ('write','wrote','writing','writes'),
    ('speak','spoke','speaking','speaks'),
    ('go','went','going','goes'),
    ('take','took','taking','takes'),
    ('make','made','making','makes'),
    ('think','thought','thinking','thinks'),
    ('give','gave','giving','gives'),
    ('come','came','coming','comes'),
    ('say','said','saying','says'),
    ('know','knew','knowing','knows'),
    ('get','got','getting','gets'),
    ('see','saw','seeing','sees'),
    ('use','used','using','uses'),
    ('find','found','finding','finds'),
    ('want','wanted','wanting','wants'),
    ('tell','told','telling','tells'),
]
for group in verb_groups:
    parts = []
    for v in group:
        key, zone, bd, lbl, phase = find_in_wmap(v)
        parts.append(f"{v}→{zone}({bd})")
    print(f"  {parts[0]:<20} {parts[1]:<20} {parts[2]:<20} {parts[3]}")

# How many words in B000/B001 are verb forms?
import re
print(f"\nB001 word-ending breakdown (clues to what's in it):")
endings = Counter()
for w in body_b001:
    ws = w.strip()
    if ws.endswith('ing'):  endings['gerund/pres-part'] += 1
    elif ws.endswith('ed'):  endings['past-part/-ed'] += 1
    elif ws.endswith('ion'): endings['-ion noun'] += 1
    elif ws.endswith('ment'):endings['-ment noun'] += 1
    elif ws.endswith('ness'):endings['-ness noun'] += 1
    elif ws.endswith('ly'):  endings['-ly adverb'] += 1
    elif ws.endswith('al'):  endings['-al adj'] += 1
    elif ws.endswith('ive'): endings['-ive adj'] += 1
    else:                    endings['other'] += 1
for ending, cnt in endings.most_common():
    print(f"  {cnt:4d}  {ending}")

print(f"\nB000 word-ending breakdown:")
endings2 = Counter()
for w in body_b000:
    ws = w.strip()
    if ws.endswith('ing'):   endings2['gerund/pres-part'] += 1
    elif ws.endswith('ed'):  endings2['past/-ed'] += 1
    elif ws.endswith('ion'): endings2['-ion noun'] += 1
    elif ws.endswith('ment'):endings2['-ment noun'] += 1
    elif ws.endswith('ness'):endings2['-ness noun'] += 1
    elif ws.endswith('ly'):  endings2['-ly adverb'] += 1
    else:                    endings2['other'] += 1
for ending, cnt in endings2.most_common():
    print(f"  {cnt:4d}  {ending}")
