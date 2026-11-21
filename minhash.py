"""Deterministic three-word shingling, MinHash, and banded LSH."""

from collections import defaultdict
from hashlib import blake2b
from random import Random
import re

HASH_COUNT = 128
SEED = 480580
# A prime just above the range of the 32-bit base hash.
PRIME = 4_294_967_311
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def words(text):
    return WORD.findall(text.casefold())


def shingles(text, *, mask_numbers=False):
    """Represent text as a set. Short nonempty inputs get one padded triple."""
    tokens = words(text)
    if mask_numbers:
        tokens = [re.sub(r"\d+", "0", token) for token in tokens]
    if not tokens:
        return frozenset()
    if len(tokens) < 3:
        tokens.extend(["\x00"] * (3 - len(tokens)))
    return frozenset(zip(tokens, tokens[1:], tokens[2:]))


class MinHash:
    def __init__(self, seed=SEED):
        rng = Random(seed)
        self.hashes = tuple(
            (rng.randrange(1, PRIME), rng.randrange(PRIME)) for _ in range(HASH_COUNT)
        )

    def signature(self, features):
        if not features:
            return ()
        values = [
            int.from_bytes(blake2b("\0".join(feature).encode(), digest_size=4).digest(), "big")
            for feature in features
        ]
        return tuple(
            min((multiplier * value + offset) % PRIME for value in values)
            for multiplier, offset in self.hashes
        )


def similarity(left, right):
    if not left or not right:
        return 0.0
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


class LSH:
    """Use equality of at least one band to retrieve candidate signatures."""

    def __init__(self, bands=32):
        if bands <= 0 or HASH_COUNT % bands:
            raise ValueError("bands must divide the signature length")
        self.rows = HASH_COUNT // bands
        self.buckets = defaultdict(list)
        self.signatures = {}

    def _keys(self, signature):
        for start in range(0, len(signature), self.rows):
            yield start, signature[start:start + self.rows]

    def add(self, key, signature):
        if key in self.signatures:
            raise ValueError(f"duplicate LSH key: {key!r}")
        if signature:
            self.signatures[key] = signature
            for band in self._keys(signature):
                self.buckets[band].append(key)

    def query(self, signature):
        candidates = set()
        for band in self._keys(signature):
            candidates.update(self.buckets.get(band, ()))
        return [(key, similarity(signature, self.signatures[key])) for key in sorted(candidates)]
