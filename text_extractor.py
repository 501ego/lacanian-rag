import re
import os
import json
from pathlib import Path
from PyPDF2 import PdfReader

INPUT_PDF_DIR = Path("data_pdfs")
OUTPUT_DIR = Path("text_chunks_json")
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 1100
OVERLAP = 200

LECON_REGEX = re.compile(
    r"(Le(?:ç|c)on\s+\d+\s+\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE)


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'—|–', '-', text)
    text = re.sub(r'Page\s*\d+', '', text, flags=re.I)
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()


def chunk_text_with_pages(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    page_splits = re.split(r'\[PAGE_(\d+)\]', text)
    chunks = []
    buffer = []
    current_words = 0
    page_tracker = []
    current_page = None
    index = 0

    def flush_chunk(start_page, end_page, content, index):
        return {
            "chunk_id": f"chunk_{index:03d}",
            "chunk_index": index,
            "text": content.strip(),
            "pages": list(set(range(start_page, end_page + 1)))
        }

    for i in range(1, len(page_splits), 2):
        page_number = int(page_splits[i])
        page_text = clean_text(page_splits[i + 1])
        words = page_text.split()

        j = 0
        while j < len(words):
            needed = chunk_size - current_words
            to_add = words[j:j + needed]
            buffer.extend(to_add)
            current_words += len(to_add)
            if current_page is None:
                current_page = page_number
            page_tracker.append(page_number)
            j += needed
            if current_words >= chunk_size:
                content = ' '.join(buffer)
                chunks.append(flush_chunk(min(page_tracker),
                              max(page_tracker), content, index))
                index += 1
                buffer = buffer[chunk_size - overlap:]
                page_tracker = [page_number]
                current_words = len(buffer)

    if buffer:
        content = ' '.join(buffer)
        chunks.append(flush_chunk(min(page_tracker),
                      max(page_tracker), content, index))

    return chunks


def process_pdf_to_json(pdf_path):
    reader = PdfReader(pdf_path)
    seminar_id = pdf_path.stem.replace(" ", "_").replace(
        "'", "").replace("...", "").replace("__", "_")
    full_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n[PAGE_{i+1}]\n{text}"

    lecons = list(LECON_REGEX.finditer(full_text))
    if lecons:
        segments = []
        for idx, match in enumerate(lecons):
            start = match.start()
            end = lecons[idx + 1].start() if idx + \
                1 < len(lecons) else len(full_text)
            lecon_id = match.group(1).strip().replace(" ", "_")
            segments.append((lecon_id, full_text[start:end]))
    else:
        segments = [(None, full_text)]

    all_chunks = []
    for lecon_id, text_segment in segments:
        chunks = chunk_text_with_pages(text_segment)
        for chunk in chunks:
            all_chunks.append({
                "seminar": seminar_id,
                "lecon": lecon_id,
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "pages": chunk["pages"]
            })

    output_file = OUTPUT_DIR / f"{seminar_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"[OK] {len(all_chunks)} chunks saved from {pdf_path.name}")


# Run on all PDFs
for pdf_file in INPUT_PDF_DIR.glob("*.pdf"):
    process_pdf_to_json(pdf_file)
