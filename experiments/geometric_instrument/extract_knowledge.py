"""
Knowledge Extraction Pipeline
===============================

Extracts geometric vectors from the model for all (entity × fact_type) pairs
defined in knowledge_base.py. Saves everything to disk as a compact .npz bundle.

After extraction, the model is NOT needed for queries.

Usage:
    python extract_knowledge.py [--output PATH] [--fact-types capital,language,...]

The extraction is resumable — it saves progress incrementally.
"""

import sys, os, time, gc, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_matmul import phi_linear
from knowledge_base import (ENTITIES, FACT_TYPES, KNOWN_ANSWERS,
                            get_all_facts, get_intent_keywords)
from shapespace import ShapeSpace

MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'
DEFAULT_OUTPUT = 'experiments/geometric_instrument/geometric_knowledge.npz'
EXTRACTION_LAYER = 23  # extract through L22 (0-indexed, exclusive)


def extract_answer_direction(engine, tokenizer, answer_token_str, identity, chunk_size=512):
    """Extract answer direction from lm_head for a given answer token."""
    hidden_dim = engine.hidden_dim
    aid = tokenizer.encode(answer_token_str)[-1]
    ans_dir = np.zeros(hidden_dim, dtype=np.float32)
    for start in range(0, hidden_dim, chunk_size):
        end = min(start + chunk_size, hidden_dim)
        chunk = identity[start:end][np.newaxis, :, :]
        logits = phi_linear(engine.lm_head.weight, chunk)[0]
        ans_dir[start:end] = logits[:, aid]
    return ans_dir


def extract_lastpos_state(engine, tokenizer, prompt):
    """Run prompt through model to EXTRACTION_LAYER, return last-position state."""
    tids = tokenizer.encode(prompt)
    h = engine.embedding(tids)[np.newaxis, :, :]
    for li in range(EXTRACTION_LAYER):
        h = engine.layers[li](h)
    return h[0, -1, :].copy()


def auto_discover_answer(engine, tokenizer, prompt):
    """Run full forward pass to discover what the model actually predicts."""
    tids = tokenizer.encode(prompt)
    h = engine.embedding(tids)[np.newaxis, :, :]
    for li in range(len(engine.layers)):
        h = engine.layers[li](h)
    # Apply final norm + lm_head
    from phi_geometric.inference.phi_components import rms_norm
    h_normed = rms_norm(h, engine.final_norm.weight)
    logits = phi_linear(engine.lm_head.weight, h_normed)
    top_id = int(np.argmax(logits[0, -1, :]))
    return tokenizer.decode([top_id])


def build_shapespaces(fact_types, entities, lastpos_by_ft, answer_by_ft,
                      answer_str_by_ft, args):
    """Build ShapeSpaces from raw vectors and verify accuracy."""
    print(f"\n  Step 4: Build ShapeSpaces")
    shapespaces = {}
    for fact_type in fact_types:
        lp = lastpos_by_ft[fact_type]
        av = answer_by_ft[fact_type]
        if len(lp) < 2:
            print(f"    SKIP {fact_type}: only {len(lp)} entities")
            continue

        common = sorted(set(lp.keys()) & set(av.keys()))
        entity_vecs = {n: lp[n] for n in common}
        answer_vecs = {n: av[n] for n in common}

        kwargs = {}
        if args.min_d is not None:
            kwargs['min_d'] = args.min_d
        if args.variance_threshold != 0.999:
            kwargs['variance_threshold'] = args.variance_threshold

        space = ShapeSpace.from_vectors(
            entity_vecs=entity_vecs,
            answer_vecs=answer_vecs,
            **kwargs,
        )
        shapespaces[fact_type] = space
        print(f"    {fact_type}: {space}")

    print(f"\n  Step 5: Verify accuracy")
    for fact_type, space in shapespaces.items():
        gt = {n: n for n in space._entities}
        acc = space.accuracy(gt)
        n_ent = len(space._entities)

        n_correct_str = 0
        for name in space._entities:
            predicted, score = space.query(name)
            pred_ans = answer_str_by_ft[fact_type].get(predicted, predicted)
            true_ans = answer_str_by_ft[fact_type].get(name, name)
            if pred_ans == true_ans:
                n_correct_str += 1

        print(f"    {fact_type}: entity-match={acc*100:.0f}% "
              f"answer-match={n_correct_str}/{n_ent} ({n_correct_str/n_ent*100:.0f}%) "
              f"d={space.dimensionality} ops={space.ops_per_query}")

    return shapespaces, answer_str_by_ft


def save_final(shapespaces, answer_str_by_ft, entity_embeddings,
               all_keywords, entities, fact_types, hidden_dim, args):
    """Save ShapeSpaces + embeddings to final .npz."""
    print(f"\n  Step 6: Save to {args.output}")

    save_data = {}

    for name, emb in entity_embeddings.items():
        save_data[f"entity_emb_{name}"] = emb

    for key, emb in all_keywords.items():
        save_data[f"intent_emb_{key}"] = emb

    for ft, space in shapespaces.items():
        sd = space.to_dict()
        save_data[f"shapespace_{ft}_basis"] = np.array(sd['basis'])
        save_data[f"shapespace_{ft}_entity_mean"] = np.array(
            sd['entity_mean']) if sd['entity_mean'] else np.array([])
        save_data[f"shapespace_{ft}_singular_values"] = np.array(
            sd['singular_values'])
        for name in sd['entities']:
            save_data[f"shapespace_{ft}_entity_{name}"] = np.array(
                sd['entities'][name])
        for name in sd['answers']:
            save_data[f"shapespace_{ft}_answer_{name}"] = np.array(
                sd['answers'][name])
        if sd['bindings']:
            for name in sd['bindings']:
                save_data[f"shapespace_{ft}_binding_{name}"] = np.array(
                    sd['bindings'][name])

    answer_strs_json = json.dumps({
        f"{ft}:{name}": ans
        for ft in fact_types
        for name, ans in answer_str_by_ft[ft].items()
    })
    save_data['answer_strings'] = np.array([answer_strs_json])

    metadata = json.dumps({
        'entities': list(entities),
        'fact_types': fact_types,
        'intent_keywords': {ft: FACT_TYPES[ft]['keywords'] for ft in fact_types},
        'response_templates': {ft: FACT_TYPES[ft]['response'] for ft in fact_types},
        'extraction_layer': EXTRACTION_LAYER,
        'hidden_dim': hidden_dim,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    save_data['metadata'] = np.array([metadata])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **save_data)
    file_size = os.path.getsize(args.output)

    total_storage = sum(sp.storage_bytes for sp in shapespaces.values())
    print(f"\n  Saved: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"  ShapeSpace storage: {total_storage:,} bytes ({total_storage/1024:.1f} KB)")
    print(f"  Total facts: {sum(len(sp._entities) for sp in shapespaces.values())}")
    print(f"  Fact types: {len(shapespaces)}")

    print(f"\n  Done. Use geometric_engine.py to query without model.")


def rebuild_from_raw(raw_path, output_path, fact_types, entities, args):
    """Rebuild ShapeSpaces from saved raw vectors. No model needed."""
    print("=" * 80)
    print("  Rebuild Mode — Loading raw vectors")
    print("=" * 80)

    if not os.path.exists(raw_path):
        print(f"  ERROR: Raw vectors not found at {raw_path}")
        print(f"  Run extraction first (without --rebuild)")
        return

    data = np.load(raw_path, allow_pickle=True)
    meta = json.loads(str(data['metadata'][0]))
    hidden_dim = meta['hidden_dim']
    ans_json = json.loads(str(data['answer_strings'][0]))

    # Reconstruct per-fact-type dicts
    lastpos_by_ft = {ft: {} for ft in fact_types}
    answer_by_ft = {ft: {} for ft in fact_types}
    answer_str_by_ft = {ft: {} for ft in fact_types}

    for ft in fact_types:
        for entity in entities:
            lp_key = f"lastpos_{ft}_{entity}"
            ad_key = f"ansdir_{ft}_{entity}"
            as_key = f"{ft}:{entity}"
            if lp_key in data:
                lastpos_by_ft[ft][entity] = data[lp_key]
            if ad_key in data:
                answer_by_ft[ft][entity] = data[ad_key]
            if as_key in ans_json:
                answer_str_by_ft[ft][entity] = ans_json[as_key]

    n_facts = sum(len(v) for v in lastpos_by_ft.values())
    print(f"  Loaded {n_facts} raw vectors from {raw_path}")

    # Entity embeddings
    entity_embeddings = {}
    for name in entities:
        key = f"entity_emb_{name}"
        if key in data:
            entity_embeddings[name] = data[key]

    # Intent keywords
    all_keywords = {}
    for ft in fact_types:
        for kw in meta.get('intent_keywords', {}).get(ft, []):
            key = f"intent_emb_{ft}:{kw}"
            if key in data:
                all_keywords[f"{ft}:{kw}"] = data[key]

    # Build + verify
    shapespaces, answer_str_by_ft = build_shapespaces(
        fact_types, entities, lastpos_by_ft, answer_by_ft, answer_str_by_ft, args)

    # Save
    save_final(shapespaces, answer_str_by_ft, entity_embeddings,
               all_keywords, entities, fact_types, hidden_dim, args)


def main():
    parser = argparse.ArgumentParser(description='Extract geometric knowledge')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Output path for extracted knowledge')
    parser.add_argument('--fact-types', default=None,
                        help='Comma-separated fact types to extract (default: all)')
    parser.add_argument('--entities', default=None,
                        help='Comma-separated entities to extract (default: all)')
    parser.add_argument('--auto-discover', action='store_true',
                        help='Auto-discover answers from model (slow but accurate)')
    parser.add_argument('--resume', default=None,
                        help='Path to partial extraction to resume from')
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild ShapeSpaces from saved raw vectors (no model needed)')
    parser.add_argument('--min-d', type=int, default=None,
                        help='Minimum dimensionality for ShapeSpaces')
    parser.add_argument('--variance-threshold', type=float, default=0.999,
                        help='Variance threshold for auto-d (default 0.999)')
    args = parser.parse_args()

    # Filter entities/fact types if specified
    entities = args.entities.split(',') if args.entities else ENTITIES
    fact_types = args.fact_types.split(',') if args.fact_types else list(FACT_TYPES.keys())

    total_facts = len(entities) * len(fact_types)
    raw_path = args.output.replace('.npz', '_raw.npz')

    # ─── REBUILD MODE: skip extraction, load raw vectors ───
    if args.rebuild:
        return rebuild_from_raw(raw_path, args.output, fact_types, entities, args)

    print("=" * 80)
    print("  Knowledge Extraction Pipeline")
    print("=" * 80)
    print(f"  Entities:   {len(entities)}")
    print(f"  Fact types: {len(fact_types)} ({', '.join(fact_types)})")
    print(f"  Total facts: {total_facts}")
    print(f"  Output: {args.output}")
    est_time = total_facts * 25  # ~25s per fact
    print(f"  Estimated time: {est_time//60}m {est_time%60}s")

    # Load model
    gc.collect()
    print(f"\n  Loading model...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")

    hidden_dim = engine.hidden_dim
    identity = np.eye(hidden_dim, dtype=np.float32)

    # Load partial extraction if resuming
    partial = {}
    if args.resume and os.path.exists(args.resume):
        print(f"  Resuming from {args.resume}")
        partial = dict(np.load(args.resume, allow_pickle=True))

    # ═══════════════════════════════════════════════════════════
    # 1. Extract entity token embeddings
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Step 1: Entity token embeddings ({len(entities)} entities)")
    entity_embeddings = {}
    for entity_name in entities:
        tids = tokenizer.encode(entity_name)
        emb = engine.embedding(tids)
        entity_embeddings[entity_name] = emb[-1, :].copy()

    # ═══════════════════════════════════════════════════════════
    # 2. Extract intent keyword embeddings
    # ═══════════════════════════════════════════════════════════
    intent_kw = get_intent_keywords()
    all_keywords = {}
    for ft in fact_types:
        for kw in intent_kw.get(ft, []):
            tids = tokenizer.encode(kw)
            emb = engine.embedding(tids)
            all_keywords[f"{ft}:{kw}"] = emb[-1, :].copy()
    print(f"  Step 2: Intent keyword embeddings ({len(all_keywords)} keywords)")

    # ═══════════════════════════════════════════════════════════
    # 3. Extract last-pos states + answer directions per fact
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Step 3: Extract vectors ({total_facts} facts)")

    # Cache answer directions (same answer token reused across entities)
    answer_dir_cache = {}

    # Per-fact-type storage
    lastpos_by_ft = {ft: {} for ft in fact_types}
    answer_by_ft = {ft: {} for ft in fact_types}
    answer_str_by_ft = {ft: {} for ft in fact_types}

    n_done = 0
    n_skipped = 0
    t_start = time.time()

    for fi, fact_type in enumerate(fact_types):
        ft_info = FACT_TYPES[fact_type]
        template = ft_info['template']

        for ei, entity_name in enumerate(entity_name for entity_name in entities):
            fact_key = f"{entity_name}:{fact_type}"
            n_done += 1

            # Check if already extracted (resume)
            lp_key = f"lastpos_{fact_key}"
            if lp_key in partial:
                lastpos_by_ft[fact_type][entity_name] = partial[lp_key]
                ans_key = f"ansdir_{fact_key}"
                if ans_key in partial:
                    answer_by_ft[fact_type][entity_name] = partial[ans_key]
                n_skipped += 1
                continue

            prompt = template.format(entity=entity_name)

            # Get answer token
            answer_str = KNOWN_ANSWERS.get((entity_name, fact_type))
            if answer_str is None and args.auto_discover:
                answer_str = auto_discover_answer(engine, tokenizer, prompt)
                print(f"    AUTO: {fact_key} → '{answer_str}'")
            if answer_str is None:
                print(f"    SKIP: {fact_key} — no known answer")
                continue

            answer_str_by_ft[fact_type][entity_name] = answer_str.strip()

            # Extract last-position hidden state
            t0 = time.time()
            lastpos = extract_lastpos_state(engine, tokenizer, prompt)
            lastpos_by_ft[fact_type][entity_name] = lastpos

            # Extract answer direction (cached by token)
            if answer_str not in answer_dir_cache:
                answer_dir_cache[answer_str] = extract_answer_direction(
                    engine, tokenizer, answer_str, identity)
            answer_by_ft[fact_type][entity_name] = answer_dir_cache[answer_str]

            elapsed = time.time() - t0
            eta = (total_facts - n_done) * (time.time() - t_start) / max(n_done - n_skipped, 1)

            print(f"    [{n_done}/{total_facts}] {fact_key:<30s} "
                  f"→ '{answer_str.strip()}'  "
                  f"({elapsed:.1f}s, ETA {eta/60:.0f}m)", flush=True)

    print(f"\n  Extraction complete: {n_done} facts in {(time.time()-t_start)/60:.1f}m")
    if n_skipped:
        print(f"  ({n_skipped} resumed from partial)")

    # ═══════════════════════════════════════════════════════════
    # 3b. Save raw vectors (for rebuild without model)
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Saving raw vectors to {raw_path}")
    raw_data = {}
    for ft in fact_types:
        for name, vec in lastpos_by_ft[ft].items():
            raw_data[f"lastpos_{ft}_{name}"] = vec
        for name, vec in answer_by_ft[ft].items():
            raw_data[f"ansdir_{ft}_{name}"] = vec
        for name, ans in answer_str_by_ft[ft].items():
            pass  # saved in answer_strs_json below
    # Entity embeddings
    for name, emb in entity_embeddings.items():
        raw_data[f"entity_emb_{name}"] = emb
    # Intent keyword embeddings
    for key, emb in all_keywords.items():
        raw_data[f"intent_emb_{key}"] = emb
    # Answer strings + metadata
    raw_data['answer_strings'] = np.array([json.dumps({
        f"{ft}:{name}": ans
        for ft in fact_types
        for name, ans in answer_str_by_ft[ft].items()
    })])
    raw_data['metadata'] = np.array([json.dumps({
        'entities': list(entities),
        'fact_types': fact_types,
        'intent_keywords': {ft: FACT_TYPES[ft]['keywords'] for ft in fact_types},
        'response_templates': {ft: FACT_TYPES[ft]['response'] for ft in fact_types},
        'extraction_layer': EXTRACTION_LAYER,
        'hidden_dim': hidden_dim,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })])
    os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
    np.savez_compressed(raw_path, **raw_data)
    raw_size = os.path.getsize(raw_path)
    print(f"  Raw vectors: {raw_size:,} bytes ({raw_size/1024/1024:.1f} MB)")

    # ═══════════════════════════════════════════════════════════
    # 4. Build ShapeSpaces
    # ═══════════════════════════════════════════════════════════
    shapespaces, answer_str_by_ft_out = build_shapespaces(
        fact_types, entities, lastpos_by_ft, answer_by_ft, answer_str_by_ft, args)

    # ═══════════════════════════════════════════════════════════
    # 5. Verify + Save
    # ═══════════════════════════════════════════════════════════
    save_final(shapespaces, answer_str_by_ft, entity_embeddings,
               all_keywords, entities, fact_types, hidden_dim, args)


if __name__ == '__main__':
    main()
