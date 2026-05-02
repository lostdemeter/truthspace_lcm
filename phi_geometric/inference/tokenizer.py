"""
Lightweight Qwen2 tokenizer -- no HuggingFace dependency.

Loads tokenizer.json from the HuggingFace cache and provides
encode/decode. Uses the BPE merge rules for encoding.
"""

import os
import json
from typing import List, Optional, Dict, Tuple

# Qwen2 EOS token ID (known constant)
QWEN2_EOS_TOKEN_ID = 151643
QWEN2_PAD_TOKEN_ID = 151643

# The GPT2-style byte-to-unicode mapping used by Qwen2's BPE
def _bytes_to_unicode():
    """Returns mapping from bytes to unicode chars (GPT2-style)."""
    bs = (
        list(range(ord('!'), ord('~') + 1))
        + list(range(ord('\xa1'), ord('\xac') + 1))
        + list(range(ord('\xae'), ord('\xff') + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {bytes([b]): chr(c) for b, c in zip(bs, cs)}

# Precompute the reverse mapping
_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


def _find_tokenizer_json() -> Optional[str]:
    """Search HuggingFace cache for any Qwen2 tokenizer.json."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.exists(cache_dir):
        return None

    candidates = [
        "models--Qwen--Qwen2-7B",
        "models--Qwen--Qwen2-7B-Instruct",
        "models--Qwen--Qwen2-0.5B",
        "models--Qwen--Qwen2-1.5B-Instruct",
        "models--Qwen--Qwen2-1.5B",
    ]

    for model_dir_name in candidates:
        model_path = os.path.join(cache_dir, model_dir_name, "snapshots")
        if os.path.exists(model_path):
            snapshots = os.listdir(model_path)
            if snapshots:
                tok_path = os.path.join(model_path, snapshots[0], "tokenizer.json")
                if os.path.exists(tok_path):
                    return tok_path
    return None


class Qwen2Tokenizer:
    """
    Qwen2 BPE tokenizer loaded from tokenizer.json.

    Supports:
      - encode(text) -> List[int]
      - decode(token_ids) -> str
      - decode_token(token_id) -> str
    """

    def __init__(self, tokenizer_json_path: Optional[str] = None):
        if tokenizer_json_path is None:
            tokenizer_json_path = _find_tokenizer_json()

        if tokenizer_json_path is None:
            raise FileNotFoundError(
                "No Qwen2 tokenizer.json found in HuggingFace cache."
            )

        with open(tokenizer_json_path, 'r') as f:
            data = json.load(f)

        model_data = data.get('model', {})
        self.vocab: Dict[str, int] = model_data.get('vocab', {})
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # BPE merges
        merges_raw = model_data.get('merges', [])
        self.merges: Dict[Tuple[str, str], int] = {}
        for i, merge_str in enumerate(merges_raw):
            parts = merge_str.split(' ', 1)
            if len(parts) == 2:
                self.merges[(parts[0], parts[1])] = i

        # Added tokens (special tokens)
        self.added_tokens: Dict[str, int] = {}
        for tok_info in data.get('added_tokens', []):
            self.added_tokens[tok_info['content']] = tok_info['id']

        self.eos_token_id = QWEN2_EOS_TOKEN_ID

    def _bpe(self, token_str: str) -> List[str]:
        """Apply BPE merges to a single pre-tokenized word."""
        if len(token_str) <= 1:
            return [token_str]

        word = list(token_str)

        while len(word) > 1:
            # Find the pair with lowest merge priority
            best_pair = None
            best_rank = float('inf')
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merges.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None or best_rank == float('inf'):
                break

            # Merge all occurrences of the best pair
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and
                        word[i] == best_pair[0] and word[i + 1] == best_pair[1]):
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return word

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using BPE.

        Uses the GPT2-style byte encoding: each byte of the UTF-8
        representation maps to a unicode character, then BPE merges
        are applied.
        """
        # Convert text to byte-level unicode representation
        text_bytes = text.encode('utf-8')
        unicode_str = ''.join(_BYTE_ENCODER[bytes([b])] for b in text_bytes)

        # Apply BPE
        bpe_tokens = self._bpe(unicode_str)

        # Convert BPE tokens to IDs
        token_ids = []
        for tok in bpe_tokens:
            if tok in self.vocab:
                token_ids.append(self.vocab[tok])
            else:
                # Unknown token — encode byte by byte
                for ch in tok:
                    if ch in self.vocab:
                        token_ids.append(self.vocab[ch])

        return token_ids

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to its string representation."""
        if token_id not in self.id_to_token:
            return f'[{token_id}]'

        token_str = self.id_to_token[token_id]

        # Convert from byte-level unicode back to UTF-8 string
        try:
            byte_list = b''.join(_BYTE_DECODER.get(c, c.encode('utf-8'))
                                 for c in token_str)
            return byte_list.decode('utf-8', errors='replace')
        except Exception:
            return token_str

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs to text."""
        return ''.join(self.decode_token(tid) for tid in token_ids)
