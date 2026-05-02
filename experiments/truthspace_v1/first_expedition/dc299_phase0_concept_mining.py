#!/usr/bin/env python3
"""
DC299 Phase 0 — Concept Vocabulary Expansion
=============================================

Goal: Mine the full ~152K token vocabulary for clean, single-token concepts
suitable for use as anchor candidates and IRD subjects in Phase 1.

Filters applied (fail-fast, no fallbacks):
  1. Space-prefix marker (Ġ or ▁) — standalone word, not subword fragment
  2. All-alphabetic, ASCII — no punctuation, numbers, symbols
  3. Length 3–15 characters (post-stripping)
  4. Norm band: only tokens whose embedding norm sits within the
     [10th, 90th] percentile of norm-filtered candidates
  5. Dedup: if both bare and space-prefixed form exist, keep space-prefixed

Output:
  - dc299_phase0_concepts.json  — list of {word, token_id, token_str, norm}
  - dc299_phase0_notes.md       — field notes on counts and norm distribution
"""

import sys
import os
import json
import re
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = (PROJECT_ROOT / "experiments" / "model_reverse_engineering_v2"
             / "phi_model")

OUTPUT_JSON = SCRIPT_DIR / "dc299_phase0_concepts.json"
NOTES_PATH  = SCRIPT_DIR / "dc299_phase0_notes.md"

MIN_LEN = 3
MAX_LEN = 15
NORM_PERCENTILE_LO = 10
NORM_PERCENTILE_HI = 90


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Notes:
    def __init__(self, path):
        self.f = open(path, "w")
        self._w("# DC299 Phase 0 — Concept Mining Notes\n\n")

    def _w(self, text):
        self.f.write(text)
        self.f.flush()
        print(text, end="")

    def section(self, title):
        self._w(f"\n## {title}\n\n")

    def log(self, text):
        self._w(text + "\n")

    def finding(self, text):
        self._w(f"\n> **FINDING:** {text}\n\n")

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# Embedding + tokenizer loading
# ---------------------------------------------------------------------------

def load_data():
    from phi_geometric.inference.phi_types import PhiEncoded

    print("Loading embeddings …", flush=True)
    phi = PhiEncoded.load(str(MODEL_DIR / "embed_tokens.npz"))
    embeddings = phi.decode()                     # (vocab_size, 3584)

    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots"
    )
    snapshots = os.listdir(cache_dir)
    vocab_file = os.path.join(cache_dir, snapshots[0], "tokenizer.json")
    with open(vocab_file) as f:
        tokenizer_data = json.load(f)

    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    token_to_id = {tok: idx for tok, idx in vocab.items()}

    print(f"  vocab_size={len(vocab)}, embedding_shape={embeddings.shape}",
          flush=True)
    return embeddings, id_to_token, token_to_id


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

SPACE_PREFIX_RE = re.compile(r"^[Ġ▁ ](.*)")

def extract_candidates(embeddings, id_to_token, notes):
    """Apply all filters and return candidate concept list."""

    notes.section("Filter Pipeline")
    notes.log(f"Total vocab tokens: {len(id_to_token)}")

    # Pass 1: space-prefixed, alphabetic, length
    pass1 = []
    for tid, tok in id_to_token.items():
        m = SPACE_PREFIX_RE.match(tok)
        if not m:
            continue
        word = m.group(1)
        if not word.isalpha():
            continue
        if not word.isascii():
            continue
        if not (MIN_LEN <= len(word) <= MAX_LEN):
            continue
        pass1.append((tid, tok, word))

    notes.log(f"After space-prefix + alpha + length filter: {len(pass1)}")

    # Pass 2: norm band
    norms = np.array([np.linalg.norm(embeddings[tid]) for tid, _, _ in pass1])
    lo = np.percentile(norms, NORM_PERCENTILE_LO)
    hi = np.percentile(norms, NORM_PERCENTILE_HI)
    notes.log(f"Norm band [{lo:.2f}, {hi:.2f}]  "
              f"(p{NORM_PERCENTILE_LO}–p{NORM_PERCENTILE_HI})")

    pass2 = []
    for (tid, tok, word), norm in zip(pass1, norms):
        if lo <= norm <= hi:
            pass2.append((tid, tok, word, float(norm)))

    notes.log(f"After norm-band filter: {len(pass2)}")

    # Pass 3: dedup — if lowercase and capitalised both present, keep one
    # Prefer space-prefixed (Ġ/▁ already assured), prefer lowercase form
    seen_words: dict[str, tuple] = {}
    for tid, tok, word, norm in pass2:
        key = word.lower()
        if key not in seen_words:
            seen_words[key] = (tid, tok, word, norm)
        else:
            # Prefer the lowercase version
            existing = seen_words[key]
            if word == word.lower() and existing[2] != existing[2].lower():
                seen_words[key] = (tid, tok, word, norm)

    pass3 = list(seen_words.values())
    notes.log(f"After dedup: {len(pass3)}")

    return pass3


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def analyse_norm_distribution(candidates, notes):
    norms = np.array([c[3] for c in candidates])
    notes.section("Norm Distribution")
    notes.log(f"  min={norms.min():.2f}  max={norms.max():.2f}")
    notes.log(f"  mean={norms.mean():.2f}  std={norms.std():.2f}")

    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        notes.log(f"  p{p:2d} = {np.percentile(norms, p):.2f}")


def show_samples(candidates, notes, n=30):
    notes.section("Random Sample of Mined Concepts")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(candidates), size=min(n, len(candidates)),
                            replace=False)
    sample = [candidates[i] for i in sorted(sample_idx)]
    for tid, tok, word, norm in sample:
        notes.log(f"  tid={tid:6d}  norm={norm:.2f}  word={word}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    notes = Notes(NOTES_PATH)
    notes.section("Configuration")
    notes.log(f"  MIN_LEN={MIN_LEN}  MAX_LEN={MAX_LEN}")
    notes.log(f"  NORM_PERCENTILE_LO={NORM_PERCENTILE_LO}  "
              f"NORM_PERCENTILE_HI={NORM_PERCENTILE_HI}")

    embeddings, id_to_token, token_to_id = load_data()

    candidates = extract_candidates(embeddings, id_to_token, notes)

    if len(candidates) < 100:
        raise RuntimeError(
            f"FAIL-FAST: only {len(candidates)} candidates — "
            "filters too aggressive or embedding load failed."
        )

    analyse_norm_distribution(candidates, notes)
    show_samples(candidates, notes)

    # Serialise
    output = [
        {
            "word":      word,
            "token_id":  int(tid),
            "token_str": tok,
            "norm":      round(norm, 4),
        }
        for tid, tok, word, norm in candidates
    ]
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    notes.finding(
        f"Mined {len(output)} clean single-token concepts and saved to "
        f"{OUTPUT_JSON.name}"
    )
    notes.log(f"\nOutput: {OUTPUT_JSON}")
    notes.close()

    print(f"\n{'='*60}")
    print(f"Phase 0 complete: {len(output)} concepts → {OUTPUT_JSON}")
    print(f"Notes            → {NOTES_PATH}")


if __name__ == "__main__":
    main()
