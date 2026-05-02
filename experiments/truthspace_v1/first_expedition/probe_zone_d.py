import json
from collections import Counter

with open("day27_atlas.json") as f:
    atlas = json.load(f)
wmap = atlas['word_map']

zone_c = {w: v for w, v in wmap.items() if v['phase']==2 and v.get('L14_body') not in ('B000','B001',None)}
zone_d = {w: v for w, v in wmap.items() if v['phase']==2 and v.get('L14_body') == 'B000'}

print(f"Zone C words: {len(zone_c)}")
print(f"Zone D words: {len(zone_d)}")
print(f"\nSample Zone D words (stripped, first 80):")
print([w.strip() for w in sorted(zone_d.keys())[:80]])

test_verbs = [
    'walk','walked','walking','walks',
    'run','ran','running','runs',
    'eat','ate','eating','eats',
    'sing','sang','singing','sings',
    'kill','killed','killing','kills',
    'see','saw','seeing','sees',
    'write','wrote','writing','writes',
    'speak','spoke','speaking','speaks',
    'go','went','going','goes',
    'take','took','taking','takes',
    'make','made','making','makes',
    'come','came','coming','comes',
    'think','thought','thinking','thinks',
    'give','gave','giving','gives',
]
print(f"\nVerb form zone assignments:")
for v in test_verbs:
    for pfx in [' '+v, v]:
        if pfx in wmap:
            bd = wmap[pfx].get('L14_body','?')
            zone = 'C' if bd not in ('B000','B001',None) else 'D'
            lbl  = wmap[pfx].get('L14_label','?')
            print(f"  {pfx!r:<22s} Zone {zone}  body={bd}  ({lbl[:35]})")
            break
    else:
        print(f"  {v!r:<22s} NOT IN ATLAS")

print(f"\nZone D sample — more words:")
zd_words = sorted([w.strip() for w in zone_d.keys()])
print(zd_words[:120])
