"""Minimal byte-pair encoding (BPE) tokenizer."""


class BPETokenizer:
    """
    A simple BPE tokenizer for learning purposes.

    This trains on raw text by iteratively merging the most frequent
    adjacent pair of tokens.
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    def _get_pair_counts(self, token_ids: list[int]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for pair in zip(token_ids, token_ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text: str, vocab_size: int):
        """Train BPE merges from raw text."""
        assert vocab_size >= 256, "vocab_size must be >= 256 (byte-level base)"

        # Start with raw bytes as initial tokens
        token_ids = list(text.encode("utf-8"))

        # Initialize base vocab: one entry per byte value
        self.vocab = {i: bytes([i]) for i in range(256)}

        num_merges = vocab_size - 256
        for i in range(num_merges):
            counts = self._get_pair_counts(token_ids)
            if not counts:
                break

            # Find most frequent pair
            best_pair = max(counts, key=counts.get)
            new_id = 256 + i

            # Record the merge
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply the merge to the token list
            token_ids = self._apply_merge(token_ids, best_pair, new_id)

    def _apply_merge(
        self, token_ids: list[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        merged: list[int] = []
        i = 0
        while i < len(token_ids):
            if (
                i < len(token_ids) - 1
                and token_ids[i] == pair[0]
                and token_ids[i + 1] == pair[1]
            ):
                merged.append(new_id)
                i += 2
            else:
                merged.append(token_ids[i])
                i += 1
        return merged

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        token_ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            token_ids = self._apply_merge(token_ids, pair, new_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        raw_bytes = b"".join(self.vocab[t] for t in token_ids)
        return raw_bytes.decode("utf-8", errors="replace")
