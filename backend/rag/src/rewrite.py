from typing import Dict, Optional, Tuple


def rewrite_query(query: str) -> Tuple[str, Optional[Dict]]:
    """
    Minimal query rewrite hook.
    Returns:
      - rewritten query text
      - optional metadata filters for retriever
    """
    return query.strip(), None

