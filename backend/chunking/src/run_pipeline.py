import os
import json
import sys
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **kwargs):
        return items

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.splitter import split_paragraphs
from src.adaptive import get_chunk_config
from src.builder import build_chunks_from_text, compute_chunk_hash
from src.enrich import enrich_chunk_text, enrich_metadata
from src.validator import ChunkValidator

TEXT_DIR = os.path.join(PROJECT_ROOT, "..", "ingestion", "data", "processed", "text")
METADATA_PATH = os.path.join(PROJECT_ROOT, "..", "ingestion", "data", "processed", "metadata.jsonl")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "chunks.jsonl")


def load_metadata():
    metadata_map = {}

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta = json.loads(line)
            metadata_map[meta["id"]] = meta

    return metadata_map


def run():
    metadata_map = load_metadata()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    total_chunks = 0
    total_docs = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:

        for file in tqdm(os.listdir(TEXT_DIR), desc="Processing documents"):

            if not file.endswith(".txt"):
                continue

            case_id = file.replace(".txt", "")
            metadata = metadata_map.get(case_id)

            if not metadata:
                print(f"Missing metadata for {case_id}")
                continue

            total_docs += 1

            file_path = os.path.join(TEXT_DIR, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                continue

            paragraphs = split_paragraphs(text)
            config = get_chunk_config(text)
            chunks = build_chunks_from_text(text, paragraphs, config)

            if not chunks:
                continue

            validator = ChunkValidator(
                min_words=int(config.chunk_size * 0.3) if config.chunk_size else 50,
                max_words=int(config.chunk_size * 1.5) if config.chunk_size else 1000
            )

            valid_chunks = []

            for chunk in chunks:
                if validator.validate(chunk):
                    valid_chunks.append(chunk)

            if not valid_chunks:
                continue

            seen_chunk_ids = set()
            for i, chunk in enumerate(valid_chunks):
                enriched_text = enrich_chunk_text(chunk, metadata)
                enriched_meta = enrich_metadata(metadata, i, len(valid_chunks))
                enriched_meta["chunk_hash"] = compute_chunk_hash(chunk)
                chunk_id = f"{case_id}_{i}"
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)

                record = {
                    "chunk_id": chunk_id,
                    "text": enriched_text,
                    "metadata": enriched_meta
                }

                output_file.write(json.dumps(record) + "\n")
                total_chunks += 1

    print("\n Chunking completed")
    print(f" Documents processed: {total_docs}")
    print(f" Total chunks saved: {total_chunks}")


if __name__ == "__main__":
    run()
