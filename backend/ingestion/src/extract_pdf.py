from pypdf import PdfReader

def extract_pdf_text(path):
    try:
        reader = PdfReader(path)
        text = ""

        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + " "

        return text.strip()

    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""