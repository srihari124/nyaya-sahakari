import os
import sys
import json
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items):
        return items

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import *
from src.extract_pdf import extract_pdf_text
from src.transform import build_metadata
from src.text_cleanup import clean_for_rag


def resolve_path(path_value, fallback=None):
    candidate = path_value
    if not os.path.isabs(candidate):
        candidate = os.path.join(PROJECT_ROOT, candidate)
    if os.path.exists(candidate):
        return candidate

    if fallback:
        alt = fallback
        if not os.path.isabs(alt):
            alt = os.path.join(PROJECT_ROOT, alt)
        if os.path.exists(alt):
            return alt

    return candidate


def main():
    pdf_dir = resolve_path(PDF_DIR)
    json_path = resolve_path(JSON_PATH, fallback="data/raw_data.json")
    text_output_dir = resolve_path(TEXT_OUTPUT_DIR)
    metadata_output = resolve_path(METADATA_OUTPUT)

    os.makedirs(text_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_output), exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    metadata_map = {}

    for case in data:
        case_id = case.get("case_id")
        if case_id:
            key = f"case{case_id.zfill(4)}"
            metadata_map[key] = case

    count = 0
    files = sorted(os.listdir(pdf_dir))

    with open(metadata_output, "w", encoding="utf-8") as metadata_file:
        for file in tqdm(files):
            if not file.lower().endswith(".pdf"):
                continue

            case_id = file.replace(".pdf", "").replace(".PDF", "")
            case = metadata_map.get(case_id, {})

            pdf_path = os.path.join(pdf_dir, file)
            text = extract_pdf_text(pdf_path)

            if not text.strip():
                continue

            cleaned = clean_for_rag(text)

            text_path = os.path.join(text_output_dir, f"{case_id}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            metadata = build_metadata(case_id, text_path, case)
            metadata_file.write(json.dumps(metadata) + "\n")

            count += 1

    print(f"\nDone: {count} documents processed")


if __name__ == "__main__":
    main()
