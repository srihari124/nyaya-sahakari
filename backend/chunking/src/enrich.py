from typing import Dict


def build_prefix(metadata: Dict) -> str:
    """
    Build contextual prefix for chunk text.
    """

    parts = []

    if metadata.get("court"):
        parts.append(f"Court: {metadata['court']}")

    if metadata.get("year"):
        parts.append(f"Year: {metadata['year']}")

    if metadata.get("outcome"):
        parts.append(f"Bail Outcome: {metadata['outcome']}")

    if metadata.get("crime_type"):
        parts.append(f"Crime Type: {metadata['crime_type']}")

    if metadata.get("ipc_sections"):
        ipc = ", ".join(metadata["ipc_sections"])
        parts.append(f"IPC Sections: {ipc}")

    return "\n".join(parts)


def enrich_chunk_text(chunk: str, metadata: Dict) -> str:
    """
    Add prefix context to chunk text.
    """

    prefix = build_prefix(metadata)

    if prefix:
        return prefix + "\n\n" + chunk

    return chunk


def enrich_metadata(base_metadata: Dict, chunk_index: int, total_chunks: int) -> Dict:
    """
    Add chunk-specific metadata.
    """

    return {
        **base_metadata,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "position": get_position(chunk_index, total_chunks)
    }


def get_position(index: int, total: int) -> str:
    """
    Tag chunk position in document.
    """

    if total == 0:
        return "unknown"

    ratio = index / total

    if ratio < 0.2:
        return "facts"
    elif ratio > 0.8:
        return "judgment"
    else:
        return "analysis"