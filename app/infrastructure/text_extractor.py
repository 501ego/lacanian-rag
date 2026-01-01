"""PDF text extraction and chunking utilities."""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyPDF2 import PdfReader

from ..core.config import AppConfig


FRENCH_MONTHS = (
    "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    "septembre|octobre|novembre|décembre|decembre"
)

WEEKDAYS = (
    "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche"
)


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _normalize_month(m: str) -> str:
    raw = m.strip().lower()
    raw_no_acc = _strip_accents(raw)
    mapping = {
        "janvier": "janvier",
        "fevrier": "février",
        "mars": "mars",
        "avril": "avril",
        "mai": "mai",
        "juin": "juin",
        "juillet": "juillet",
        "aout": "août",
        "septembre": "septembre",
        "octobre": "octobre",
        "novembre": "novembre",
        "decembre": "décembre",
    }
    return mapping.get(raw_no_acc, raw)


def _fix_ocr_digits(s: str) -> str:
    s = re.sub(r"(?<=\d)[lI](?=\d)", "1", s)
    s = re.sub(r"(?<=\s)[lI](?=\d{2,4}\b)", "1", s)
    s = re.sub(r"(?<=\b)[lI](?=\d{2,4}\b)", "1", s)
    s = re.sub(r"\b0l\b", "01", s)
    s = re.sub(r"\b0I\b", "01", s)
    return s


def _date_key(day: str, month: str, year: int) -> str:
    d2 = day.zfill(2)
    m = _strip_accents(month.lower())
    return f"{d2}_{m}_{year}"


def _label_value(label: str, num: str, day: str, month: str, year: int) -> str:
    d2 = day.zfill(2)
    month_norm = _normalize_month(month)
    return f"{label}__{num}_____{d2}_{month_norm}_____{year}"


def _infer_year(year_str: str, base_start_year: Optional[int]) -> Optional[int]:
    year_str = _fix_ocr_digits(year_str).strip()
    if not year_str.isdigit():
        return None
    if len(year_str) == 4:
        return int(year_str)
    if len(year_str) == 2:
        if base_start_year is None:
            return None
        yy = int(year_str)
        century = (base_start_year // 100) * 100
        y = century + yy
        if y < base_start_year:
            y += 100
        return y
    return None


RANGE_YEAR_REGEX = re.compile(r"\b(?P<y1>\d{4})\s*-\s*(?P<y2>\d{2,4})\b")


def _extract_seminar_start_year(text: str) -> Optional[int]:
    t = _fix_ocr_digits(text)
    m = RANGE_YEAR_REGEX.search(t)
    if not m:
        return None
    try:
        return int(m.group("y1"))
    except ValueError:
        return None


TABLE_ENTRY_REGEX = re.compile(
    rf"""
    ^\s*
    (?:
        (?P<label>Le(?:ç|c)on|Séance)\s+(?P<num>\d+)
        |
        (?P<num_only>\d+)\s*[\)\.\-:]?
    )
    \s+
    (?P<day>\d{{1,2}})
    \s+
    (?P<month>[A-Za-zÀ-ÿ]+)
    \s+
    (?P<year>[0-9lI]{{2,4}})
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

BODY_DATE_REGEX = re.compile(
    rf"""
    (?:
        \b(?P<weekday>{WEEKDAYS})\b
        [^\S\r\n]+
    )?
    (?P<day>\d{{1,2}})
    [^\S\r\n]+
    (?P<month>{FRENCH_MONTHS})
    [^\S\r\n]+
    (?P<year>[0-9lI]{{4}})
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def build_date_to_label(full_text: str) -> Tuple[Dict[str, str], str]:
    txt = _fix_ocr_digits(full_text)
    base_start_year = _extract_seminar_start_year(txt)

    has_explicit_lecon = bool(
        re.search(r"\bLe(?:ç|c)on\b", txt, flags=re.IGNORECASE))
    default_label = "Leçon" if has_explicit_lecon else "Séance"

    date_to_label: Dict[str, str] = {}

    for m in TABLE_ENTRY_REGEX.finditer(txt):
        label = m.group("label")
        num = m.group("num") or m.group("num_only")
        day = m.group("day")
        month = _normalize_month(m.group("month"))
        year_raw = m.group("year")

        label_final = (label or default_label).strip()
        year = _infer_year(year_raw, base_start_year)
        if year is None:
            continue

        key = _date_key(day, month, year)
        date_to_label[key] = _label_value(
            label_final, str(int(num)), day, month, year)

    return date_to_label, default_label


def split_by_body_dates(full_text: str) -> List[Tuple[Optional[str], str]]:
    txt = _fix_ocr_digits(full_text)
    matches = list(BODY_DATE_REGEX.finditer(txt))
    if not matches:
        return [(None, txt)]

    segments: List[Tuple[Optional[str], str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)

        day = m.group("day")
        month = _normalize_month(m.group("month"))
        year_str = _fix_ocr_digits(m.group("year"))
        if not year_str.isdigit():
            date_key = None
        else:
            date_key = _date_key(day, month, int(year_str))

        segments.append((date_key, txt[start:end]))

    return segments


class PDFChunker:
    """Extract, segment, and chunk PDFs into JSON files."""

    def __init__(self, input_dir: Path, output_dir: Path, chunk_size: int, overlap: int):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._output_dir.mkdir(exist_ok=True)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\u2014\u2013]", "-", text)
        text = re.sub(r"Page\s*\d+", "", text, flags=re.I)
        text = re.sub(r"\[\d+\]", "", text)
        return text.strip()

    def chunk_text_with_pages(self, text: str) -> List[dict]:
        page_splits = re.split(r"\[PAGE_(\d+)\]", text)
        chunks: List[dict] = []
        word_buffer: List[str] = []
        current_words = 0
        page_numbers: List[int] = []
        chunk_index = 0

        def flush_chunk(start_page: int, end_page: int, content: str, idx: int) -> dict:
            return {
                "chunk_id": f"chunk_{idx:03d}",
                "chunk_index": idx,
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
                to_add = words[word_index: word_index + needed]
                word_buffer.extend(to_add)
                current_words += len(to_add)
                page_numbers.append(page_number)

                word_index += needed
                if current_words >= self._chunk_size:
                    content = " ".join(word_buffer)
                    chunks.append(flush_chunk(min(page_numbers), max(
                        page_numbers), content, chunk_index))
                    chunk_index += 1

                    word_buffer = word_buffer[self._chunk_size -
                                              self._overlap:]
                    current_words = len(word_buffer)
                    page_numbers = [page_number] if word_buffer else []

        if word_buffer:
            content = " ".join(word_buffer)
            if not page_numbers:
                page_numbers = [1]
            chunks.append(flush_chunk(min(page_numbers), max(
                page_numbers), content, chunk_index))

        return chunks

    def process_pdf_to_json(self, pdf_path: Path) -> None:
        reader = PdfReader(pdf_path)
        seminar_id = (
            pdf_path.stem.replace(" ", "_")
            .replace("'", "")
            .replace("...", "")
            .replace("__", "_")
        )

        full_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n[PAGE_{i + 1}]\n{text}"

        date_to_label, default_label = build_date_to_label(full_text)
        segments = split_by_body_dates(full_text)

        all_chunks: List[dict] = []
        unknown_counter = 0

        for date_key, text_segment in segments:
            lecon_id = date_to_label.get(date_key) if date_key else None

            if lecon_id is None:
                unknown_counter += 1
                lecon_id = f"{default_label}__UNKNOWN_{unknown_counter:03d}"

            chunks = self.chunk_text_with_pages(text_segment)
            for chunk in chunks:
                all_chunks.append(
                    {
                        "seminar": seminar_id,
                        "lecon": lecon_id,
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "pages": chunk["pages"],
                    }
                )

        output_file = self._output_dir / f"{seminar_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"[OK] {len(all_chunks)} chunks saved from {pdf_path.name}")

    def run(self) -> None:
        for pdf_file in self._input_dir.glob("*.pdf"):
            self.process_pdf_to_json(pdf_file)


class TextChunker:
    """Chunk raw text into overlapping segments."""

    def __init__(self, chunk_size: int, overlap: int):
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk_text(self, text: str) -> List[dict]:
        cleaned = PDFChunker.clean_text(text)
        words = cleaned.split()
        if not words:
            return []

        step = self._chunk_size - self._overlap
        if step <= 0:
            raise ValueError("chunk_size must be larger than overlap.")

        chunks: List[dict] = []
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

        return chunks


if __name__ == "__main__":
    app_config = AppConfig()
    chunker = PDFChunker(
        input_dir=app_config.input_pdf_dir,
        output_dir=app_config.output_dir,
        chunk_size=app_config.chunk_size,
        overlap=app_config.overlap,
    )
    chunker.run()
