import hashlib
import re
from typing import List

from src.adaptive import ChunkConfig


def split_large_paragraph(paragraph: str, max_words: int) -> List[str]:
    words = paragraph.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def normalize_for_hash(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_chunk_hash(text: str) -> str:
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_tiny_metadata_heavy(chunk: str) -> bool:
    """Filter tiny chunks that are mostly metadata."""
    words = chunk.split()
    if len(words) >= 90:
        return False

    text = chunk.lower()
    metadata_hits = 0
    markers = [
        "court:",
        "year:",
        "bail outcome:",
        "crime type:",
        "ipc sections:",
        "case no",
        "versus",
        "coram:",
        "appearance :",
    ]
    for marker in markers:
        if marker in text:
            metadata_hits += 1

    return metadata_hits >= 2


def normalize_paragraphs(paragraphs: List[str], chunk_config: ChunkConfig) -> List[str]:
    """Split oversized paragraphs before chunk assembly."""
    if chunk_config.chunk_size is None:
        return paragraphs

    max_para_words = int(chunk_config.chunk_size * 0.5)

    normalized = []

    for para in paragraphs:
        word_count = len(para.split())

        if word_count > max_para_words:
            normalized.extend(split_large_paragraph(para, max_para_words))
        else:
            normalized.append(para)

    return normalized


def build_chunks(
    paragraphs: List[str],
    chunk_config: ChunkConfig
) -> List[str]:
    """Build chunks with paragraph-based overlap."""

    if chunk_config.chunk_size is None:
        return [" ".join(paragraphs)]

    paragraphs = normalize_paragraphs(paragraphs, chunk_config)

    chunks = []
    seen_hashes = set()
    current_chunk = []
    current_length = 0

    chunk_size = chunk_config.chunk_size
    overlap_words = chunk_config.overlap

    for para in paragraphs:
        words = para.split()
        word_count = len(words)

        if current_length + word_count > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text and not is_tiny_metadata_heavy(chunk_text):
                chunk_hash = compute_chunk_hash(chunk_text)
                if chunk_hash not in seen_hashes:
                    seen_hashes.add(chunk_hash)
                    chunks.append(chunk_text)

            overlap_block = []
            overlap_len = 0

            for prev_para in reversed(current_chunk):
                overlap_block.insert(0, prev_para)
                overlap_len += len(prev_para.split())

                if overlap_len >= overlap_words:
                    break

            current_chunk = overlap_block
            current_length = overlap_len

        current_chunk.append(para)
        current_length += word_count

    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text and not is_tiny_metadata_heavy(chunk_text):
            chunk_hash = compute_chunk_hash(chunk_text)
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                chunks.append(chunk_text)

    return chunks


def build_chunks_from_text(
    text: str,
    paragraphs: List[str],
    chunk_config: ChunkConfig
) -> List[str]:
    """Build chunks from normalized document text."""

    if not paragraphs:
        return []

    if chunk_config.chunk_size is None:
        return [text]

    return build_chunks(paragraphs, chunk_config)
