import re
from typing import List


def clean_text(text: str) -> str:
    """
    Light normalization (safe for legal text).
    """

    if not text:
        return ""

    # normalize line endings
    text = text.replace("\r", "\n")

    # remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # remove page markers like "Page 1 of 10"
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)

    return text.strip()


def split_into_paragraph_blocks(text: str) -> List[str]:
    """
    Primary split using structural hints.
    """

    # split by multiple newlines
    blocks = re.split(r"\n{2,}", text)

    # fallback: if no structure, split by sentence boundaries
    if len(blocks) <= 2:
        blocks = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    return blocks


def split_large_block(block: str, max_words: int = 200) -> List[str]:
    """
    Split overly large paragraph into smaller segments.
    """

    sentences = re.split(r"(?<=[.!?])\s+", block)

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        wlen = len(words)

        if current_len + wlen > max_words:
            if current:
                chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(sent)
        current_len += wlen

    if current:
        chunks.append(" ".join(current))

    return chunks


def split_paragraphs(text: str) -> List[str]:
    """
    Main function:
    Convert raw text → meaningful paragraph units.
    """

    text = clean_text(text)

    if not text:
        return []

    blocks = split_into_paragraph_blocks(text)

    paragraphs = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        word_count = len(block.split())

        # split large blocks
        if word_count > 400:
            paragraphs.extend(split_large_block(block))
        else:
            paragraphs.append(block)

    # final filtering (remove very small noise)
    paragraphs = [
        p for p in paragraphs
        if len(p.split()) > 20
    ]

    return paragraphs