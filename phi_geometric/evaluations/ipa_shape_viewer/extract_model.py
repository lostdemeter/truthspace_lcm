"""Extract simplified model data as JSON for the 3D viewer."""
import sys, json
sys.path.insert(0, '/home/thorin/truthspace-lcm')

from phi_geometric.evaluations.ipa_model_simplification import (
    SimplifiedExecutor, build_program
)

program = build_program()
exe = SimplifiedExecutor.simplify(program)

# Build JSON-serializable model
model = {
    'digraphs': {},
    'frozenDigraphs': [],
    'charMap': {},
    'contextTables': {},
    'magicETables': {},
}

for (c1, c2), repl in exe.digraphs.items():
    key = c1 + c2
    model['digraphs'][key] = repl if repl else ''
    if (c1, c2) in exe.frozen_digraphs:
        model['frozenDigraphs'].append(key)

model['charMap'] = dict(exe.char_map)

for char, table in exe.context_tables.items():
    t = dict(table)
    # Convert any non-string keys to strings
    if 'selector_map' in t:
        t['selector_map'] = {str(k): v for k, v in t['selector_map'].items()}
    if 'pure_map' in t:
        t['pure_map'] = {str(k): v for k, v in t['pure_map'].items()}
    if 'fine_gears' in t:
        fg = {}
        for cv, (fv, fm, zd) in t['fine_gears'].items():
            fg[str(cv)] = {
                'fine_var': fv,
                'fine_map': {str(fk): fvv for fk, fvv in fm.items()} if fm else {},
                'zone_default': zd
            }
        t['fine_gears'] = fg
    model['contextTables'][char] = t

for vowel, table in exe.magic_e_tables.items():
    t = dict(table)
    if 'pure_map' in t:
        t['pure_map'] = {str(k): v for k, v in t['pure_map'].items()}
    if 'fine_gears' in t:
        fg = {}
        for cv, (fv, fm, zd) in t['fine_gears'].items():
            fg[str(cv)] = {
                'fine_var': fv,
                'fine_map': {str(fk): fvv for fk, fvv in fm.items()} if fm else {},
                'zone_default': zd
            }
        t['fine_gears'] = fg
    model['magicETables'][vowel] = t

print(json.dumps(model, indent=2, ensure_ascii=False))
