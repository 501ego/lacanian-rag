"""PDF text extraction and chunking utilities."""

import json
import re
from pathlib import Path

from PyPDF2 import PdfReader

from ..core.config import AppConfig

LECON_REGEX = re.compile(
    r"(Le(?:ç|c)on\s+\d+\s+\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE)


class PDFChunker:
    """Extract, segment, and chunk PDFs into JSON files."""

    def __init__(self, input_dir: Path, output_dir: Path, chunk_size: int, overlap: int):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._output_dir.mkdir(exist_ok=True)

    @staticmethod
    def clean_text(text):
        """Normalize whitespace and remove page artifacts."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"—|–", "-", text)
        text = re.sub(r"Page\s*\d+", "", text, flags=re.I)
        text = re.sub(r"\[\d+\]", "", text)
        return text.strip()

    def chunk_text_with_pages(self, text):
        """Split text into overlapping chunks while tracking page ranges."""
        page_splits = re.split(r"\[PAGE_(\d+)\]", text)
        chunks = []
        word_buffer = []
        current_words = 0
        page_numbers = []
        chunk_index = 0

        def flush_chunk(start_page, end_page, content, chunk_index):
            """Build a chunk payload with page range metadata."""
            return {
                "chunk_id": f"chunk_{chunk_index:03d}",
                "chunk_index": chunk_index,
                "text": content.strip(),
                "pages": list(set(range(start_page, end_page + 1))),
            }

        for page_pos in range(1, len(page_splits), 2):
            page_number = int(page_splits[page_pos])
            page_text = self.clean_text(page_splits[page_pos + 1])
            words = page_text.split()

            word_index = 0
            while word_index < len(words):
                needed = self._chunk_size - current_words
                to_add = words[word_index:word_index + needed]
                word_buffer.extend(to_add)
                current_words += len(to_add)
                page_numbers.append(page_number)
                word_index += needed
                if current_words >= self._chunk_size:
                    content = " ".join(word_buffer)
                    chunks.append(flush_chunk(min(page_numbers),
                                              max(page_numbers), content, chunk_index))
                    chunk_index += 1
                    word_buffer = word_buffer[self._chunk_size - self._overlap:]
                    page_numbers = [page_number]
                    current_words = len(word_buffer)

        if word_buffer:
            content = " ".join(word_buffer)
            chunks.append(flush_chunk(min(page_numbers),
                                      max(page_numbers), content, chunk_index))

        return chunks

    def process_pdf_to_json(self, pdf_path: Path):
        """Extract PDF text and write chunked JSON output."""
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
            chunks = self.chunk_text_with_pages(text_segment)
            for chunk in chunks:
                all_chunks.append({
                    "seminar": seminar_id,
                    "lecon": lecon_id,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "pages": chunk["pages"],
                })

        output_file = self._output_dir / f"{seminar_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"[OK] {len(all_chunks)} chunks saved from {pdf_path.name}")

    def run(self):
        """Process all PDFs in the input directory."""
        for pdf_file in self._input_dir.glob("*.pdf"):
            self.process_pdf_to_json(pdf_file)


class TextChunker:
    """Chunk raw text into overlapping segments."""

    def __init__(self, chunk_size: int, overlap: int):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk_text(self, text: str):
        """Split text into chunks using word counts and overlap."""
        cleaned = PDFChunker.clean_text(text)
        words = cleaned.split()
        if not words:
            return []
        step = self._chunk_size - self._overlap
        if step <= 0:
            raise ValueError("chunk_size must be larger than overlap.")
        chunks = []
        chunk_index = 0
        start = 0
        while start < len(words):
            end = min(start + self._chunk_size, len(words))
            content = " ".join(words[start:end]).strip()
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_index:03d}",
                    "chunk_index": chunk_index,
                    "text": content,
                    "pages": [1],
                }
            )
            if end >= len(words):
                break
            start += step
            chunk_index += 1


if __name__ == "__main__":
    app_config = AppConfig()
    chunker = PDFChunker(
        input_dir=app_config.input_pdf_dir,
        output_dir=app_config.output_dir,
        chunk_size=app_config.chunk_size,
        overlap=app_config.overlap,
    )
    chunker.run()
