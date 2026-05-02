"""
Geometric Engine — Query without the model
=============================================

Loads pre-extracted geometric knowledge from disk and answers
factual questions using ONLY geometric operations.

NO model loaded. NO neural forward pass. Just directions.

Usage:
    # Interactive
    python geometric_engine.py

    # With custom knowledge file
    python geometric_engine.py --knowledge path/to/geometric_knowledge.npz

    # Programmatic
    from geometric_engine import GeometricEngine
    engine = GeometricEngine.load('geometric_knowledge.npz')
    result = engine.answer('What is the capital of France?')
    print(result['response'])
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shapespace import ShapeSpace

DEFAULT_KNOWLEDGE = 'experiments/geometric_instrument/geometric_knowledge.npz'


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-20 else 0.0


class GeometricEngine:
    """A factual Q&A engine powered entirely by geometric operations.

    After loading, requires NO neural model. All queries are:
      1. Entity detection (cosine on embeddings)
      2. Intent detection (cosine on keyword embeddings)
      3. ShapeSpace lookup (project + dot + argmax)
    """

    def __init__(self):
        self.entity_embeddings = {}    # name → embedding
        self.entity_names = []         # ordered list
        self.intent_embeddings = {}    # (fact_type, keyword) → embedding
        self.fact_spaces = {}          # fact_type → ShapeSpace
        self.answer_strings = {}       # (fact_type, entity) → answer string
        self.response_templates = {}   # fact_type → template
        self.intent_keywords = {}      # fact_type → [keywords]
        self.metadata = {}

    @classmethod
    def load(cls, path):
        """Load geometric engine from extracted knowledge file."""
        engine = cls()
        t0 = time.time()

        data = np.load(path, allow_pickle=True)

        # Metadata
        engine.metadata = json.loads(str(data['metadata'][0]))
        engine.entity_names = engine.metadata['entities']
        engine.response_templates = engine.metadata['response_templates']
        engine.intent_keywords = engine.metadata['intent_keywords']
        fact_types = engine.metadata['fact_types']

        # Entity embeddings
        for name in engine.entity_names:
            key = f"entity_emb_{name}"
            if key in data:
                engine.entity_embeddings[name] = data[key].astype(np.float64)

        # Intent keyword embeddings
        for ft, keywords in engine.intent_keywords.items():
            for kw in keywords:
                key = f"intent_emb_{ft}:{kw}"
                if key in data:
                    engine.intent_embeddings[(ft, kw)] = data[key].astype(np.float64)

        # Answer strings
        ans_json = json.loads(str(data['answer_strings'][0]))
        for key, val in ans_json.items():
            ft, name = key.split(':', 1)
            engine.answer_strings[(ft, name)] = val

        # ShapeSpaces
        for ft in fact_types:
            basis_key = f"shapespace_{ft}_basis"
            if basis_key not in data:
                continue

            # Reconstruct ShapeSpace from saved components
            space = ShapeSpace()
            basis = data[basis_key]
            space._d = basis.shape[0]
            space._basis = basis.astype(np.float64)

            sv_key = f"shapespace_{ft}_singular_values"
            if sv_key in data:
                space._singular_values = data[sv_key].astype(np.float64)

            mean_key = f"shapespace_{ft}_entity_mean"
            if mean_key in data and data[mean_key].size > 0:
                space._entity_mean = data[mean_key].astype(np.float64)
                space._source_dim = len(space._entity_mean)
            else:
                space._entity_mean = np.zeros(basis.shape[1], dtype=np.float64)
                space._source_dim = basis.shape[1]

            # Load entities
            for name in engine.entity_names:
                ek = f"shapespace_{ft}_entity_{name}"
                if ek in data:
                    space._entities[name] = data[ek].astype(np.float64)

            # Load answers
            for name in engine.entity_names:
                ak = f"shapespace_{ft}_answer_{name}"
                if ak in data:
                    space._answers[name] = data[ak].astype(np.float64)

            # Load bindings (if any)
            for name in engine.entity_names:
                bk = f"shapespace_{ft}_binding_{name}"
                if bk in data:
                    space._bindings[name] = data[bk].astype(np.float64)
                elif name in space._entities:
                    space._bindings[name] = np.zeros(space._d, dtype=np.float64)

            engine.fact_spaces[ft] = space

        load_time = time.time() - t0
        total_facts = sum(len(sp._entities) for sp in engine.fact_spaces.values())
        total_bytes = sum(sp.storage_bytes for sp in engine.fact_spaces.values())

        print(f"  Geometric Engine loaded in {load_time*1000:.1f}ms")
        print(f"  {len(engine.fact_spaces)} fact types, "
              f"{len(engine.entity_names)} entities, "
              f"{total_facts} total facts")
        print(f"  Storage: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")

        return engine

    def detect_entity(self, words):
        """Detect entity from input words via string matching + embedding fallback."""
        # Fast path: exact string match
        for word in words:
            w = word.strip('?,.')
            if w in self.entity_embeddings:
                return w, 1.0

        # Slow path: embedding cosine (for partial matches, typos, etc.)
        # Only if we have a way to embed unknown tokens
        # For now, try case-insensitive match
        for word in words:
            w = word.strip('?,.')
            for name in self.entity_names:
                if w.lower() == name.lower():
                    return name, 0.95

        return None, 0.0

    def detect_intent(self, words):
        """Detect fact type from input words via keyword matching."""
        # Score each fact type by keyword matches
        scores = {ft: 0.0 for ft in self.fact_spaces}

        for word in words:
            w = word.strip('?,.').lower()
            for ft, keywords in self.intent_keywords.items():
                if ft in self.fact_spaces:
                    for kw in keywords:
                        if w == kw.lower():
                            scores[ft] += 1.0

        best_ft = max(scores, key=scores.get) if scores else None
        best_score = scores.get(best_ft, 0.0) if best_ft else 0.0

        # Default to first fact type if no keyword matched
        if best_score == 0.0 and self.fact_spaces:
            best_ft = list(self.fact_spaces.keys())[0]

        return best_ft, best_score

    def answer(self, user_input):
        """Answer a user question using only geometric operations."""
        t0 = time.perf_counter_ns()

        words = user_input.replace('?', ' ? ').replace('.', ' ').replace(',', ' ').split()

        entity, entity_conf = self.detect_entity(words)
        intent, intent_conf = self.detect_intent(words)

        if entity and intent and intent in self.fact_spaces and \
                entity in self.fact_spaces[intent]._entities:
            predicted, score = self.fact_spaces[intent].query(entity)
            answer_str = self.answer_strings.get((intent, predicted), predicted)

            template = self.response_templates.get(intent, '{entity} → {answer}')
            response = template.format(entity=entity, answer=answer_str)
            success = True
        elif entity is None:
            response = "I don't recognize that entity."
            answer_str = None
            score = 0
            success = False
        else:
            response = f"I don't have {intent or 'that'} information for {entity}."
            answer_str = None
            score = 0
            success = False

        elapsed_ns = time.perf_counter_ns() - t0

        return {
            'response': response,
            'success': success,
            'entity': entity,
            'entity_confidence': entity_conf,
            'intent': intent,
            'intent_confidence': intent_conf,
            'answer': answer_str,
            'score': score,
            'time_us': elapsed_ns / 1000,
        }

    def stats(self):
        """Return engine statistics."""
        total_facts = sum(len(sp._entities) for sp in self.fact_spaces.values())
        total_bytes = sum(sp.storage_bytes for sp in self.fact_spaces.values())
        return {
            'entities': len(self.entity_names),
            'fact_types': len(self.fact_spaces),
            'total_facts': total_facts,
            'storage_bytes': total_bytes,
            'fact_type_details': {
                ft: {
                    'd': sp.dimensionality,
                    'entities': len(sp._entities),
                    'ops_per_query': sp.ops_per_query,
                    'storage_bytes': sp.storage_bytes,
                }
                for ft, sp in self.fact_spaces.items()
            },
        }


def run_tests(engine):
    """Run automated tests."""
    from knowledge_base import KNOWN_ANSWERS

    print("\n" + "=" * 80)
    print("  Automated Tests")
    print("=" * 80)

    n_correct = 0
    n_total = 0
    n_entity_ok = 0
    n_intent_ok = 0
    failures = []

    for (entity, fact_type), expected_answer in sorted(KNOWN_ANSWERS.items()):
        if entity not in engine.entity_names:
            continue
        if fact_type not in engine.fact_spaces:
            continue
        if entity not in engine.fact_spaces[fact_type]._entities:
            continue

        n_total += 1

        # Build a natural query
        queries_by_ft = {
            'capital': f"What is the capital of {entity}?",
            'language': f"What language is spoken in {entity}?",
            'continent': f"What continent is {entity} on?",
            'currency': f"What is the currency of {entity}?",
        }
        query = queries_by_ft.get(fact_type,
                                  f"What is the {fact_type} of {entity}?")
        result = engine.answer(query)

        entity_ok = result['entity'] == entity
        intent_ok = result['intent'] == fact_type
        expected_clean = expected_answer.strip()
        answer_ok = (result['success'] and
                     expected_clean.lower() in result['response'].lower())

        if entity_ok:
            n_entity_ok += 1
        if intent_ok:
            n_intent_ok += 1
        if answer_ok:
            n_correct += 1
        else:
            failures.append((entity, fact_type, expected_clean,
                             result.get('answer', '?'),
                             result.get('response', '?')))

    print(f"\n  Results ({n_total} facts tested):")
    print(f"    Entity detection:  {n_entity_ok}/{n_total} "
          f"({n_entity_ok/max(n_total,1)*100:.0f}%)")
    print(f"    Intent detection:  {n_intent_ok}/{n_total} "
          f"({n_intent_ok/max(n_total,1)*100:.0f}%)")
    print(f"    Correct answer:    {n_correct}/{n_total} "
          f"({n_correct/max(n_total,1)*100:.0f}%)")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for entity, ft, expected, got, response in failures[:20]:
            print(f"    {entity}/{ft}: expected '{expected}', "
                  f"got '{got}' → {response[:60]}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")

    return n_correct, n_total


def interactive_mode(engine):
    """Interactive chat loop."""
    s = engine.stats()
    print("\n" + "=" * 80)
    print("  Geometric Engine — Interactive Mode")
    print(f"  {s['total_facts']} facts | "
          f"{s['entities']} entities | "
          f"{s['fact_types']} fact types | "
          f"{s['storage_bytes']/1024:.0f} KB")
    print("  Type 'quit' to exit, 'stats' for details, 'debug' for verbose.")
    print("=" * 80)

    verbose = False

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("  Goodbye!")
            break
        if user_input.lower() == 'stats':
            s = engine.stats()
            print(f"  Entities: {s['entities']}")
            print(f"  Fact types: {s['fact_types']}")
            print(f"  Total facts: {s['total_facts']}")
            print(f"  Storage: {s['storage_bytes']:,} bytes")
            for ft, d in s['fact_type_details'].items():
                print(f"    {ft}: d={d['d']}, {d['entities']} entities, "
                      f"{d['ops_per_query']} ops, {d['storage_bytes']:,}B")
            continue
        if user_input.lower() == 'debug':
            verbose = not verbose
            print(f"  Debug: {'ON' if verbose else 'OFF'}")
            continue

        result = engine.answer(user_input)

        if verbose:
            print(f"  [Entity: {result['entity']} "
                  f"(conf={result['entity_confidence']:.2f})]")
            print(f"  [Intent: {result['intent']} "
                  f"(conf={result['intent_confidence']:.2f})]")
            print(f"  [Score: {result['score']:.3f}  "
                  f"Time: {result['time_us']:.0f}μs]")

        print(f"  Bot: {result['response']}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Geometric Q&A Engine')
    parser.add_argument('--knowledge', default=DEFAULT_KNOWLEDGE,
                        help='Path to extracted knowledge file')
    parser.add_argument('--test', action='store_true',
                        help='Run automated tests')
    parser.add_argument('--query', default=None,
                        help='Single query (non-interactive)')
    args = parser.parse_args()

    print("=" * 80)
    print("  Geometric Engine")
    print("  Directions, not weights")
    print("=" * 80)

    engine = GeometricEngine.load(args.knowledge)

    if args.test:
        run_tests(engine)
    elif args.query:
        result = engine.answer(args.query)
        print(f"\n  {result['response']}")
        print(f"  [{result['time_us']:.0f}μs]")
    else:
        # Run tests first, then interactive
        if sys.stdin.isatty():
            run_tests(engine)
            interactive_mode(engine)
        else:
            run_tests(engine)


if __name__ == '__main__':
    main()
