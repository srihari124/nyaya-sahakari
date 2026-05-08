import re


def remove_boilerplate_lines(text):
    """Remove known footer/header boilerplate commonly present in Indiankanoon PDFs."""
    patterns = [
        r"Indian Kanoon - http://indiankanoon\.org/doc/\d+/?",
        r"Page No\.#\s*\d+/\d+",
        r"^\s*=+\s*$",
    ]
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    kept = []
    for line in text.splitlines():
        if any(rx.search(line) for rx in compiled):
            continue
        kept.append(line)
    return "\n".join(kept)


def normalize_whitespace_and_hyphens(text):
    """Light normalization only: collapse spaces and fix split hyphen spacing."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*-\s+", "-", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_repeated_lines(text):
    """Remove exact repeated lines while preserving order."""
    seen = set()
    out = []
    for line in text.splitlines():
        key = line.strip()
        if not key:
            out.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out)


def truncate_on_new_case_header(text):
    """
    Keep only first case body if another case header appears later in the same text.
    Example header pattern: 'X vs Y on ... Indian Kanoon - http://indiankanoon.org/doc/...'
    """
    lines = text.splitlines()
    header_re = re.compile(
        r".+\bvs\b.+\bon\b.+Indian Kanoon - http://indiankanoon\.org/doc/\d+/?",
        re.IGNORECASE,
    )

    hits = [i for i, ln in enumerate(lines) if header_re.search(ln)]
    if len(hits) <= 1:
        return text
    return "\n".join(lines[:hits[1]]).strip()


def clean_for_rag(text):
    """Run all light cleanup operations in a safe order."""
    text = truncate_on_new_case_header(text)
    text = remove_boilerplate_lines(text)
    text = dedupe_repeated_lines(text)
    text = normalize_whitespace_and_hyphens(text)
    return text

