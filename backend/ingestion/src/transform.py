import re


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid(text, min_words=100):
    return len(text.split()) > min_words


def build_text(case):
    parts = []

    if case.get("facts"):
        parts.append("Facts: " + case["facts"])

    if case.get("legal_issues"):
        parts.append("Issues: " + "; ".join(case["legal_issues"]))

    if case.get("judgment_reason"):
        parts.append("Judgment: " + case["judgment_reason"])

    if case.get("summary"):
        parts.append("Summary: " + case["summary"])

    return "\n\n".join(parts)


def build_metadata(case_id, text_path, case):
    return {
        "id": case_id,
        "text_path": text_path,
        "court": case.get("court"),
        "year": case.get("date", "")[:4],
        "judge": case.get("judge"),
        "outcome": case.get("bail_outcome"),
        "ipc_sections": case.get("ipc_sections", []),
        "crime_type": case.get("crime_type"),
        "region": case.get("region"),
        "bail_type": case.get("bail_type"),
        "landmark": case.get("landmark_case", False)
    }
