import os

PDF_DIR = os.getenv("INGESTION_PDF_DIR", "data/raw/raw_data_pdfs")
JSON_PATH = os.getenv("INGESTION_JSON_PATH", "data/raw/raw_data.json")
TEXT_OUTPUT_DIR = os.getenv("INGESTION_TEXT_OUTPUT_DIR", "data/processed/text")
METADATA_OUTPUT = os.getenv("INGESTION_METADATA_OUTPUT", "data/processed/metadata.jsonl")
MIN_WORDS = int(os.getenv("INGESTION_MIN_WORDS", "100"))
