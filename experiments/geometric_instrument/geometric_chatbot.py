"""
Geometric Chatbot — Proof of Concept
======================================

A chatbot that answers factual questions using ONLY geometric operations:
  - Parsing: token embeddings → cosine match (dot products)
  - Lookup: ShapeSpace query (62 ops)
  - No neural forward pass at query time

Architecture:
  User input → tokenize → embed tokens → {
      entity detection:  max cosine(token_emb, known_entity_embs) → entity
      intent detection:  max cosine(token_emb, known_intent_embs) → fact_type
  } → ShapeSpace[fact_type].query(entity) → answer → template → response

This is the fail-fast test: where does pure geometry work for chat,
and where does it break?

Bootstrapping:
  The model is loaded ONCE to extract:
    1. Token embeddings for known entities and intent keywords
    2. ShapeSpaces for each fact type (entity→answer mappings)
  After extraction, the model is not needed. All queries are geometric.
"""

import sys, os, time, gc, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shapespace import ShapeSpace
from phi_geometric.inference import PhiQwen2Engine
from phi_geometric.inference.tokenizer import Qwen2Tokenizer
from phi_geometric.inference.phi_matmul import phi_linear


MODEL_DIR = 'experiments/model_reverse_engineering_v2/phi_model'

# ═══════════════════════════════════════════════════════════════
# Knowledge base: entities × fact types
# ═══════════════════════════════════════════════════════════════

ENTITIES = {
    'France':  {'capital': ' Paris',    'language': ' French',    'continent': ' Europe'},
    'Germany': {'capital': ' Berlin',   'language': ' German',    'continent': ' Europe'},
    'Japan':   {'capital': ' Tokyo',    'language': ' Japanese',  'continent': ' Asia'},
    'Italy':   {'capital': ' Rome',     'language': ' Italian',   'continent': ' Europe'},
    'Spain':   {'capital': ' Madrid',   'language': ' Spanish',   'continent': ' Europe'},
    'Brazil':  {'capital': ' Bras',     'language': ' Portuguese','continent': ' South'},
    'China':   {'capital': ' Beijing',  'language': ' Mandarin',  'continent': ' Asia'},
}

FACT_TYPES = {
    'capital':   'The capital of {entity} is',
    'language':  'The primary language of {entity} is',
    'continent': 'The continent of {entity} is',
}

INTENT_KEYWORDS = {
    'capital':   ['capital', 'city', 'Capital'],
    'language':  ['language', 'speak', 'spoken', 'Language'],
    'continent': ['continent', 'located', 'Continent', 'where'],
}

RESPONSE_TEMPLATES = {
    'capital':   'The capital of {entity} is {answer}.',
    'language':  'The primary language of {entity} is {answer}.',
    'continent': '{entity} is located in {answer}.',
}


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-20 else 0.0


class GeometricChatbot:
    """A chatbot that answers questions using only geometric operations."""

    def __init__(self):
        self.tokenizer = None
        self.entity_embeddings = {}    # entity_name → embedding vector
        self.intent_embeddings = {}    # (fact_type, keyword) → embedding vector
        self.fact_spaces = {}          # fact_type → ShapeSpace
        self.answer_tokens = {}        # (entity, fact_type) → answer string
        self.embedding_matrix = None   # full vocab embedding for token lookup

    def bootstrap(self, engine, tokenizer):
        """Extract all geometric data from the model. One-time cost."""
        self.tokenizer = tokenizer
        hidden_dim = engine.hidden_dim

        print("  Bootstrapping geometric chatbot...")
        t_total = time.time()

        # ─── 1. Extract entity token embeddings ───
        print("    Extracting entity embeddings...", flush=True)
        for entity_name in ENTITIES:
            tids = tokenizer.encode(entity_name)
            emb = engine.embedding(tids)
            # Use the LAST token embedding (for multi-token names)
            self.entity_embeddings[entity_name] = emb[-1, :].copy()

        # ─── 2. Extract intent keyword embeddings ───
        print("    Extracting intent keyword embeddings...", flush=True)
        for fact_type, keywords in INTENT_KEYWORDS.items():
            for kw in keywords:
                tids = tokenizer.encode(kw)
                emb = engine.embedding(tids)
                self.intent_embeddings[(fact_type, kw)] = emb[-1, :].copy()

        # ─── 3. Build ShapeSpaces per fact type ───
        # Uses LAST-POSITION hidden state at L22 as entity vector.
        # This is where the model composes entity + relationship into
        # a single direction pointing toward the answer. No head-specific
        # binding needed — the geometry is fact-type agnostic.
        identity = np.eye(hidden_dim, dtype=np.float32)
        chunk_size = 512

        for fact_type, prompt_template in FACT_TYPES.items():
            print(f"    Building ShapeSpace for '{fact_type}'...", flush=True)

            lastpos_vecs = {}
            answer_vecs = {}

            for entity_name, facts in ENTITIES.items():
                answer_str = facts[fact_type]
                self.answer_tokens[(entity_name, fact_type)] = answer_str.strip()

                # Forward pass through L22, take LAST position
                prompt = prompt_template.format(entity=entity_name)
                tids = tokenizer.encode(prompt)

                h = engine.embedding(tids)[np.newaxis, :, :]
                for li in range(23):
                    h = engine.layers[li](h)

                lastpos_vecs[entity_name] = h[0, -1, :].copy()

                # Answer direction from lm_head
                aid = tokenizer.encode(answer_str)[-1]
                ans_dir = np.zeros(hidden_dim, dtype=np.float32)
                for start in range(0, hidden_dim, chunk_size):
                    end = min(start + chunk_size, hidden_dim)
                    chunk = identity[start:end][np.newaxis, :, :]
                    logits = phi_linear(engine.lm_head.weight, chunk)[0]
                    ans_dir[start:end] = logits[:, aid]
                answer_vecs[entity_name] = ans_dir

            # Build ShapeSpace: last-pos as entity, no binding needed
            space = ShapeSpace.from_vectors(
                entity_vecs=lastpos_vecs,
                answer_vecs=answer_vecs,
            )
            self.fact_spaces[fact_type] = space
            print(f"      {space}")

        elapsed = time.time() - t_total
        print(f"  Bootstrap complete in {elapsed:.1f}s")
        print(f"  {len(self.fact_spaces)} fact spaces, "
              f"{len(ENTITIES)} entities, "
              f"{len(ENTITIES) * len(FACT_TYPES)} total facts")

    def detect_entity(self, tokens_text: list) -> tuple:
        """Detect which entity the user is asking about.

        Pure geometry: cosine similarity between each input token's
        embedding and known entity embeddings.

        Returns (entity_name, confidence, token_index).
        """
        best_entity = None
        best_score = -np.inf
        best_idx = -1

        for i, token_text in enumerate(tokens_text):
            # Get this token's embedding
            tids = self.tokenizer.encode(token_text)
            if len(tids) == 0:
                continue
            emb = self._get_token_embedding(tids[-1])

            for entity_name, entity_emb in self.entity_embeddings.items():
                score = cosine(emb, entity_emb)
                if score > best_score:
                    best_score = score
                    best_entity = entity_name
                    best_idx = i

        return best_entity, best_score, best_idx

    def detect_intent(self, tokens_text: list) -> tuple:
        """Detect what fact type the user is asking about.

        Pure geometry: cosine similarity between input tokens and
        intent keyword embeddings.

        Returns (fact_type, confidence, matching_keyword).
        """
        best_type = None
        best_score = -np.inf
        best_keyword = None

        for i, token_text in enumerate(tokens_text):
            tids = self.tokenizer.encode(token_text)
            if len(tids) == 0:
                continue
            emb = self._get_token_embedding(tids[-1])

            for (fact_type, keyword), kw_emb in self.intent_embeddings.items():
                score = cosine(emb, kw_emb)
                if score > best_score:
                    best_score = score
                    best_type = fact_type
                    best_keyword = keyword

        return best_type, best_score, best_keyword

    def _get_token_embedding(self, token_id):
        """Get embedding for a single token ID."""
        tids = np.array([token_id], dtype=np.int64)
        # Use a cached version if available
        if not hasattr(self, '_emb_cache'):
            self._emb_cache = {}
        if token_id not in self._emb_cache:
            from phi_geometric.inference import PhiQwen2Engine
            # We need the engine for this... store it during bootstrap
            emb = self._engine.embedding(tids)
            self._emb_cache[token_id] = emb[0, :].copy()
        return self._emb_cache[token_id]

    def bootstrap_with_engine(self, engine, tokenizer):
        """Bootstrap and keep engine reference for token lookup."""
        self._engine = engine
        self.bootstrap(engine, tokenizer)

    def answer(self, user_input: str) -> dict:
        """Answer a user question using only geometric operations.

        Returns a dict with the answer and diagnostic information.
        """
        t0 = time.perf_counter_ns()

        # Tokenize: split into words (simple whitespace + punctuation)
        words = user_input.replace('?', ' ').replace('.', ' ').replace(',', ' ').split()

        # Detect entity (geometric)
        entity, entity_conf, entity_idx = self.detect_entity(words)

        # Detect intent (geometric)
        intent, intent_conf, intent_kw = self.detect_intent(words)

        # Lookup in ShapeSpace
        if intent in self.fact_spaces and entity in self.fact_spaces[intent]._entities:
            predicted, score = self.fact_spaces[intent].query(entity)
            # Map predicted entity → answer string
            # (handles shared answers: Germany→France in continent space
            #  means Germany matched France's "Europe" direction = correct)
            answer_str = self.answer_tokens.get((predicted, intent), predicted)

            # Format response
            template = RESPONSE_TEMPLATES.get(intent, '{entity} → {answer}')
            response = template.format(entity=entity, answer=answer_str)
            success = True
        else:
            response = f"I don't know how to answer that geometrically."
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
            'intent_keyword': intent_kw,
            'answer': answer_str,
            'score': score,
            'time_us': elapsed_ns / 1000,
        }

    def save(self, path):
        """Save the chatbot's geometric state (no model needed to load)."""
        data = {
            'entity_embeddings': {n: v.tolist()
                                  for n, v in self.entity_embeddings.items()},
            'intent_embeddings': {f"{ft}:{kw}": v.tolist()
                                  for (ft, kw), v in self.intent_embeddings.items()},
            'fact_spaces': {ft: sp.to_dict()
                           for ft, sp in self.fact_spaces.items()},
            'answer_tokens': {f"{e}:{ft}": a
                             for (e, ft), a in self.answer_tokens.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        return os.path.getsize(path)

    @classmethod
    def load(cls, path, engine):
        """Load from saved state. Engine needed only for token embedding lookup."""
        bot = cls()
        bot._engine = engine
        bot.tokenizer = Qwen2Tokenizer()

        with open(path) as f:
            data = json.load(f)

        bot.entity_embeddings = {n: np.array(v, dtype=np.float64)
                                 for n, v in data['entity_embeddings'].items()}
        bot.intent_embeddings = {
            tuple(k.split(':', 1)): np.array(v, dtype=np.float64)
            for k, v in data['intent_embeddings'].items()
        }
        bot.fact_spaces = {ft: ShapeSpace.from_dict(sd)
                          for ft, sd in data['fact_spaces'].items()}
        bot.answer_tokens = {
            tuple(k.split(':', 1)): v
            for k, v in data['answer_tokens'].items()
        }
        return bot


def run_tests(bot):
    """Structured tests before interactive mode."""
    print("\n" + "=" * 80)
    print("  Automated Tests")
    print("=" * 80)

    tests = [
        # Standard queries
        ("What is the capital of France?", "capital", "France", "Paris"),
        ("What is the capital of Germany?", "capital", "Germany", "Berlin"),
        ("What is the capital of Japan?", "capital", "Japan", "Tokyo"),
        ("What is the capital of Italy?", "capital", "Italy", "Rome"),
        ("What is the capital of Spain?", "capital", "Spain", "Madrid"),
        ("What is the capital of China?", "capital", "China", "Beijing"),
        # Language queries
        ("What language do they speak in France?", "language", "France", "French"),
        ("What language is spoken in Japan?", "language", "Japan", "Japanese"),
        ("What language do they speak in Germany?", "language", "Germany", "German"),
        ("What is the language of Italy?", "language", "Italy", "Italian"),
        # Continent queries
        ("What continent is France on?", "continent", "France", "Europe"),
        ("Where is Japan located?", "continent", "Japan", "Asia"),
        ("What continent is China on?", "continent", "China", "Asia"),
        # Informal
        ("capital France", "capital", "France", "Paris"),
        ("language Japan", "language", "Japan", "Japanese"),
        ("France capital?", "capital", "France", "Paris"),
        # Tricky
        ("Tell me about the capital city of Spain", "capital", "Spain", "Madrid"),
        ("Which language is spoken in Brazil?", "language", "Brazil", "Portuguese"),
        ("Is Italy in Europe?", "continent", "Italy", "Europe"),
    ]

    n_entity_correct = 0
    n_intent_correct = 0
    n_answer_correct = 0
    n_total = len(tests)

    print(f"\n  {'Query':<50s} {'Entity':>8s} {'Intent':>10s} {'Answer':>12s} {'Time':>8s}")
    print(f"  {'-'*50} {'-'*8} {'-'*10} {'-'*12} {'-'*8}")

    for query, expected_intent, expected_entity, expected_answer in tests:
        result = bot.answer(query)

        entity_ok = result['entity'] == expected_entity
        intent_ok = result['intent'] == expected_intent
        answer_ok = expected_answer.lower() in result['response'].lower() if result['success'] else False

        if entity_ok:
            n_entity_correct += 1
        if intent_ok:
            n_intent_correct += 1
        if answer_ok:
            n_answer_correct += 1

        e_mark = "✓" if entity_ok else f"✗({result['entity']})"
        i_mark = "✓" if intent_ok else f"✗({result['intent']})"
        a_mark = "✓" if answer_ok else "✗"

        print(f"  {query:<50s} {e_mark:>8s} {i_mark:>10s} {a_mark:>12s} "
              f"{result['time_us']:>6.0f}μs")

    print(f"\n  Results:")
    print(f"    Entity detection:  {n_entity_correct}/{n_total} "
          f"({n_entity_correct/n_total*100:.0f}%)")
    print(f"    Intent detection:  {n_intent_correct}/{n_total} "
          f"({n_intent_correct/n_total*100:.0f}%)")
    print(f"    Correct answer:    {n_answer_correct}/{n_total} "
          f"({n_answer_correct/n_total*100:.0f}%)")

    return n_entity_correct, n_intent_correct, n_answer_correct, n_total


def interactive_mode(bot):
    """Interactive chat loop."""
    print("\n" + "=" * 80)
    print("  Geometric Chatbot — Interactive Mode")
    print("  Ask factual questions about countries.")
    print("  Type 'quit' to exit, 'debug' to toggle verbose mode.")
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
        if user_input.lower() == 'debug':
            verbose = not verbose
            print(f"  Debug mode: {'ON' if verbose else 'OFF'}")
            continue

        result = bot.answer(user_input)

        if verbose:
            print(f"  [Entity: {result['entity']} "
                  f"(conf={result['entity_confidence']:.4f})]")
            print(f"  [Intent: {result['intent']} via '{result['intent_keyword']}' "
                  f"(conf={result['intent_confidence']:.4f})]")
            print(f"  [Score: {result['score']:.3f}  "
                  f"Time: {result['time_us']:.0f}μs]")

        print(f"  Bot: {result['response']}")


def main():
    print("=" * 80)
    print("  Geometric Chatbot")
    print("  A chatbot powered by directions, not weights")
    print("=" * 80)

    gc.collect()
    print("\n  Loading model for bootstrap...", end="", flush=True)
    t0 = time.time()
    engine = PhiQwen2Engine.load(MODEL_DIR, verbose=False)
    tokenizer = Qwen2Tokenizer()
    print(f" done in {time.time()-t0:.1f}s")

    # Bootstrap
    bot = GeometricChatbot()
    bot.bootstrap_with_engine(engine, tokenizer)

    # Save state
    save_path = '/tmp/geometric_chatbot_state.json'
    size = bot.save(save_path)
    print(f"\n  Saved chatbot state: {size:,} bytes ({size/1024:.1f} KB)")
    print(f"  (This is ALL that's needed for queries — no model required*)")
    print(f"  (*except embedding lookup for unknown tokens)")

    # Run automated tests
    n_ent, n_int, n_ans, n_tot = run_tests(bot)

    # Summary
    total_facts = len(ENTITIES) * len(FACT_TYPES)
    total_storage = sum(sp.storage_bytes for sp in bot.fact_spaces.values())
    print(f"\n" + "=" * 80)
    print(f"  Chatbot Summary")
    print(f"=" * 80)
    print(f"  Knowledge: {total_facts} facts "
          f"({len(ENTITIES)} entities × {len(FACT_TYPES)} fact types)")
    print(f"  Storage:   {total_storage:,} bytes ({total_storage/1024:.1f} KB)")
    print(f"  Accuracy:  {n_ans}/{n_tot} ({n_ans/n_tot*100:.0f}%) on test suite")
    print(f"  Per-query: ~2μs ShapeSpace + embedding lookup overhead")

    # Interactive mode if terminal
    if sys.stdin.isatty():
        interactive_mode(bot)
    else:
        print("\n  (Non-interactive mode — skipping chat loop)")


if __name__ == '__main__':
    main()
