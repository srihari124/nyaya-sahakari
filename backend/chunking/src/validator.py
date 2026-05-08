import re
import hashlib
from typing import List, Set


class ChunkValidator:
    """Filter noisy or duplicate chunks before embedding."""

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 1000,
        deduplicate: bool = True
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.deduplicate = deduplicate
        self.seen_hashes: Set[str] = set()

    def is_valid_length(self, chunk: str) -> bool:
        words = len(chunk.split())
        return self.min_words <= words <= self.max_words

    def has_noise(self, chunk: str) -> bool:
        text = chunk.lower()

        if re.search(r"page \d+ of \d+", text):
            return True

        if len(set(text.split())) < 10:
            return True

        return False

    def has_meaningful_content(self, chunk: str) -> bool:
        text = chunk.lower()

        keywords = [
            "court",
            "accused",
            "bail",
            "judge",
            "section",
            "offence",
            "petition",
            "order"
        ]

        return any(k in text for k in keywords)

    def is_duplicate(self, chunk: str) -> bool:
        if not self.deduplicate:
            return False

        normalized = re.sub(r"\s+", " ", chunk).strip().lower()
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if h in self.seen_hashes:
            return True

        self.seen_hashes.add(h)
        return False

    def validate(self, chunk: str) -> bool:
        """Run the chunk validation checks."""
        if not chunk or not chunk.strip():
            return False

        if not self.is_valid_length(chunk):
            return False

        if self.has_noise(chunk):
            return False

        if not self.has_meaningful_content(chunk):
            return False

        if self.is_duplicate(chunk):
            return False

        return True


def filter_chunks(chunks: List[str], validator: ChunkValidator) -> List[str]:
    """Filter a list of chunks with the given validator."""

    valid_chunks = []

    for chunk in chunks:
        if validator.validate(chunk):
            valid_chunks.append(chunk)

    return valid_chunks
