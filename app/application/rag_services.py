"""Service classes for building prompts and processing responses."""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..domain.lacan_canon import LACAN_CANON
from ..domain.lecon_labels import get_lecon_label


@dataclass(frozen=True)
class SourceEntry:
    """Metadata for a retrieved source chunk."""

    source_index: int
    seminar_id: str
    seminar_title: str
    lesson_label: str
    lesson_raw: Optional[str]
    chunk_id: str
    chunk_index: int
    full_id: str
    pages: List[int]


class ChunkRepository:
    """Load and resolve chunk data from disk."""

    def __init__(self, chunks_dir: Path, ui_text: Dict[str, str]):
        self._chunks_dir = chunks_dir
        self._ui_text = ui_text
        self._cache: Dict[str, Tuple[Dict[str, Any], Dict[int, Any]]] = {}

    def _load_seminar(self, seminar: str):
        if seminar in self._cache:
            return
        path = self._chunks_dir / f"{seminar}.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            by_id = {c["chunk_id"]: c for c in all_chunks}
            by_index = {c["chunk_index"]: c for c in all_chunks}
            self._cache[seminar] = (by_id, by_index)
        except (OSError, json.JSONDecodeError) as exc:
            print(self._ui_text["file_error"].format(path=path, error=exc))
            self._cache[seminar] = ({}, {})

    def get_matching_chunk(self, seminar: str, raw_id: str):
        """Return the matching chunk payload for a given raw id."""
        self._load_seminar(seminar)
        by_id, by_index = self._cache.get(seminar, ({}, {}))
        if not by_id:
            return None

        if raw_id.startswith("chunk_"):
            chunk_id = raw_id
        elif "_chunk_" in raw_id:
            chunk_id = f"chunk_{raw_id.split('_chunk_', 1)[1]}"
        elif raw_id.startswith(f"{seminar}_"):
            chunk_id = raw_id[len(seminar) + 1:]
        else:
            chunk_id = raw_id

        matching_chunk = by_id.get(chunk_id)
        if not matching_chunk:
            match = re.search(r"(\d+)$", raw_id)
            if match:
                matching_chunk = by_index.get(int(match.group(1)))
        return matching_chunk


class LabelResolver:
    """Resolve seminar and lesson labels for output metadata."""

    def __init__(self, language_code: str, unknown_lecon: Dict[str, str]):
        self._language_code = language_code
        self._unknown_lecon = unknown_lecon

    def get_seminar_title(self, seminar_id: str) -> str:
        """Return localized seminar title."""
        canon = LACAN_CANON.get(self._language_code, {})
        return canon.get(seminar_id, {}).get("titulo", seminar_id)

    def get_lesson_label(self, lecon_raw: Optional[str]) -> str:
        """Return a friendly lesson label based on the raw lesson id."""
        return get_lecon_label(
            lecon_raw,
            self._language_code,
            self._unknown_lecon.get(self._language_code, self._unknown_lecon["en"]),
        )


class PromptBuilder:
    """Build prompts using retrieved chunks and JSON template."""

    def __init__(
        self,
        chunks_dir: Path,
        ui_text: Dict[str, str],
        language_names: Dict[str, str],
        structure_hint: str,
        json_keys: Dict[str, str],
        unknown_lecon: Dict[str, str],
    ):
        self._repo = ChunkRepository(chunks_dir, ui_text)
        self._ui_text = ui_text
        self._language_names = language_names
        self._structure_hint = structure_hint
        self._json_keys = json_keys
        self._unknown_lecon = unknown_lecon

    def build(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        language_code: str,
        detail_level: str = "concise",
    ) -> Tuple[Optional[str], List[SourceEntry], str]:
        """Build the full prompt and return sources and context text."""
        context_parts = []
        sources: List[SourceEntry] = []
        source_index = 1

        if not retrieved_chunks:
            return None, [], ""

        label_resolver = LabelResolver(language_code, self._unknown_lecon)

        for chunk in retrieved_chunks:
            seminar = chunk.get("seminar")
            if not seminar:
                continue

            matching_chunk = self._repo.get_matching_chunk(
                seminar, chunk.get("id", ""))
            if matching_chunk:
                context_parts.append(
                    f"[{source_index}] {matching_chunk['text']}")
                lecon_raw = matching_chunk.get("lecon") or chunk.get("lecon")
                pages = matching_chunk.get("pages") or chunk.get("pages") or []
                if isinstance(pages, list):
                    pages = sorted(set(pages))
                source_entry = SourceEntry(
                    source_index=source_index,
                    seminar_id=seminar,
                    seminar_title=label_resolver.get_seminar_title(seminar),
                    lesson_label=label_resolver.get_lesson_label(lecon_raw),
                    lesson_raw=lecon_raw,
                    chunk_id=matching_chunk.get("chunk_id"),
                    chunk_index=matching_chunk.get("chunk_index"),
                    full_id=chunk.get("id"),
                    pages=pages,
                )
                sources.append(source_entry)
                source_index += 1
            else:
                print(self._ui_text["missing_chunk"].format(
                    seminar=seminar, chunk_id=chunk.get("id", "")))

        if not context_parts:
            return None, [], ""

        context_text = "\n\n".join(context_parts)
        language_name = self._language_names[language_code]
        template_sources = []
        for source in sources:
            template_sources.append(
                {
                    self._json_keys["source_index"]: source.source_index,
                    self._json_keys["french_quotes"]: [],
                    self._json_keys["translations"]: [],
                    self._json_keys["translation_critique"]: "",
                    self._json_keys["context"]: "",
                    self._json_keys["lacanian_development"]: "",
                }
            )
        template_payload = {
            self._json_keys["language"]: language_code,
            self._json_keys["sources"]: template_sources,
            self._json_keys["comparative_trajectory"]: "",
        }
        template_text = json.dumps(
            template_payload, ensure_ascii=False, indent=2)
        detail_level = detail_level.lower().strip()
        if detail_level not in ("concise", "full"):
            detail_level = "concise"
        if detail_level == "full":
            quotes_instruction = "Provide 1-2 French quotes per source and a translation for each quote"
            comparative_instruction = (
                'Write "comparative_trajectory" as a long critical synthesis '
                "(not a summary): 2-3 paragraphs, at least 8 sentences total and "
                "at least 900 characters; use a Lacanian voice; highlight tensions, "
                "shifts, or stakes across sources; explain what changes in the "
                "conceptual position; avoid adding facts not in the context and "
                "mark speculation explicitly"
            )
            brevity_instruction = ""
        else:
            quotes_instruction = "Provide exactly 1 French quote per source and a translation for the quote"
            comparative_instruction = (
                'Write "comparative_trajectory" as a concise critical synthesis '
                "(not a summary): 3-4 sentences total; highlight the key tension "
                "or shift across sources; avoid adding facts not in the context and "
                "mark speculation explicitly"
            )
            brevity_instruction = (
                "- Keep translation_critique to 2 short sentences.\n"
                "- Keep context to 2-3 sentences.\n"
                "- Keep lacanian_development to 3-4 sentences.\n"
            )

        prompt = f"""
You are a helpful assistant specialized in Jacques Lacan's work. Use only the context below to answer the question. Do not make up information.

Context:
{context_text}

Question:
{question}

Instructions:
- Respond only in {language_name} for all string values, except the French quotes which must stay in French
- Use every source from [1] to [{len(sources)}] in order; do not skip any source
- The "sources" array must contain exactly {len(sources)} objects with source_index from 1 to {len(sources)}
- {quotes_instruction}
- French quotes must be exact substrings from the context (copy-paste, no edits)
- Set "language" to "{language_code}"
- The "translation_critique" must include at least two alternative rendering
- Write the "lacanian_development" in a Lacanian voice while avoiding facts not in the context; mark speculation explicitly
{brevity_instruction}- {comparative_instruction}
- If "language" is "es", avoid English words in string values (except French quotes and [n] citations)
- Cite claims with numbered references like [1], [2], etc. inside the relevant string values
- {self._structure_hint}

JSON template (fill in all empty strings, keep quotes exactly):
{template_text}

Begin your answer below:
"""
        return prompt, sources, context_text


class ResponseProcessor:
    """Parse, normalize, and validate model responses."""

    def __init__(
        self,
        language_names: Dict[str, str],
        json_keys: Dict[str, str],
        source_metadata_keys: Dict[str, str],
        translator: Callable[[str, str, str], str],
    ):
        self._language_names = language_names
        self._json_keys = json_keys
        self._source_metadata_keys = source_metadata_keys
        self._translator = translator

    def parse(self, response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Parse model output into JSON if possible."""
        cleaned = self._basic_clean_json_text(response_text)
        last_error: Optional[str] = None
        for candidate in (cleaned, self._repair_json_text(response_text)):
            try:
                return json.loads(candidate), None
            except json.JSONDecodeError as exc:
                last_error = str(exc)
            try:
                return json.loads(candidate, strict=False), None
            except json.JSONDecodeError as exc:
                last_error = str(exc)
        return None, last_error

    def normalize(
        self,
        response_json: Dict[str, Any],
        total_sources: int,
        language_code: str,
        context_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Normalize the response into a compliant JSON structure."""
        if not isinstance(response_json, dict):
            return None
        normalized = {
            self._json_keys["language"]: language_code,
            self._json_keys["sources"]: [],
            self._json_keys["comparative_trajectory"]: "",
        }
        trajectory = response_json.get(
            self._json_keys["comparative_trajectory"])
        if isinstance(trajectory, str):
            normalized[self._json_keys["comparative_trajectory"]
                       ] = trajectory.strip()
        sources = response_json.get(self._json_keys["sources"])
        if not isinstance(sources, list):
            sources = []
        segments = self._parse_context_segments(context_text)
        for idx in range(1, total_sources + 1):
            source = sources[idx - 1] if idx - 1 < len(sources) else {}
            if not isinstance(source, dict):
                source = {}
            segment = segments.get(idx, "")
            quotes = source.get(self._json_keys["french_quotes"])
            if not isinstance(quotes, list):
                quotes = []
            quotes = [q.strip()
                      for q in quotes if isinstance(q, str) and q.strip()]
            if segment:
                if quotes:
                    aligned = [self._snap_quote_to_context(
                        q, segment) for q in quotes]
                    quotes = [q for q in aligned if q in segment]
                if not quotes:
                    quotes = self._extract_default_quotes(
                        segment, max_quotes=2)
            translations = source.get(self._json_keys["translations"])
            if not isinstance(translations, list):
                translations = []
            translations = [
                t.strip() if isinstance(t, str) else "" for t in translations
            ]
            if len(translations) != len(quotes):
                translations = [""] * len(quotes)

            normalized_source = {
                self._json_keys["source_index"]: idx,
                self._json_keys["french_quotes"]: quotes,
                self._json_keys["translations"]: translations,
                self._json_keys["translation_critique"]: self._get_text(
                    source, self._json_keys["translation_critique"]),
                self._json_keys["context"]: self._get_text(
                    source, self._json_keys["context"]),
                self._json_keys["lacanian_development"]: self._get_text(
                    source, self._json_keys["lacanian_development"]),
            }
            normalized[self._json_keys["sources"]].append(normalized_source)
        return normalized

    def ensure_translations(self, response_json: Dict[str, Any], language_code: str):
        """Ensure each French quote has a translation in the target language."""
        sources = response_json.get(self._json_keys["sources"], [])
        target_language = self._language_names[language_code]
        for source in sources:
            quotes = source.get(self._json_keys["french_quotes"], [])
            translations = source.get(self._json_keys["translations"], [])
            if not isinstance(translations, list) or len(translations) != len(quotes):
                translations = []
            if len(translations) != len(quotes) or any(not t.strip() for t in translations):
                translations = [
                    self._translator(quote, "French", target_language)
                    for quote in quotes
                ]
            source[self._json_keys["translations"]] = translations

    def align_quotes_to_context(self, response_json: Dict[str, Any], context_text: str):
        """Replace quotes with exact context substrings when possible."""
        segments = self._parse_context_segments(context_text)
        sources = response_json.get(self._json_keys["sources"], [])
        for source in sources:
            source_index = source.get(self._json_keys["source_index"])
            segment = segments.get(source_index, "")
            quotes = source.get(self._json_keys["french_quotes"], [])
            aligned = [self._snap_quote_to_context(
                quote, segment) for quote in quotes]
            source[self._json_keys["french_quotes"]] = aligned

    def collect_validation_errors(
        self,
        response_json: Dict[str, Any],
        total_sources: int,
        language_code: str,
        context_text: Optional[str] = None,
    ) -> List[str]:
        """Validate response schema and content consistency."""
        errors = []
        if not isinstance(response_json, dict):
            return ["Response is not a JSON object."]
        language = response_json.get(self._json_keys["language"])
        if language not in ("en", "es"):
            errors.append(
                'Missing or invalid "language" field (must be "en" or "es").')
        sources = response_json.get(self._json_keys["sources"])
        if not isinstance(sources, list):
            return errors + ['"sources" must be an array.']
        if len(sources) != total_sources:
            errors.append(
                f'"sources" must contain exactly {total_sources} objects.')
        for idx, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                errors.append(f"Source {idx} is not an object.")
                continue
            if source.get(self._json_keys["source_index"]) != idx:
                errors.append(f"Source {idx} has incorrect source_index.")
            quotes = source.get(self._json_keys["french_quotes"])
            translations = source.get(self._json_keys["translations"])
            if not isinstance(quotes, list) or not quotes:
                errors.append(
                    f"Source {idx} must include french_quotes array with at least one item."
                )
            if not isinstance(translations, list) or len(translations) != len(quotes or []):
                errors.append(
                    f"Source {idx} must include translations matching french_quotes length."
                )
            if any(not isinstance(item, str) or not item.strip() for item in quotes or []):
                errors.append(f"Source {idx} has empty quote strings.")
            if any(not isinstance(item, str) or not item.strip() for item in translations or []):
                errors.append(f"Source {idx} has empty translation strings.")
            for key in (
                self._json_keys["translation_critique"],
                self._json_keys["context"],
                self._json_keys["lacanian_development"],
            ):
                value = source.get(key)
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"Source {idx} missing required field: {key}.")
            if language_code == "es":
                for key in (
                    self._json_keys["translation_critique"],
                    self._json_keys["context"],
                    self._json_keys["lacanian_development"],
                ):
                    if self._contains_english_tokens(source.get(key, "")):
                        errors.append(
                            f"Source {idx} contains English in {key}.")
        trajectory = response_json.get(
            self._json_keys["comparative_trajectory"])
        if not isinstance(trajectory, str) or not trajectory.strip():
            errors.append("comparative_trajectory must be a non-empty string.")
        if language_code == "es" and self._contains_english_tokens(trajectory):
            errors.append("comparative_trajectory contains English.")
        if context_text:
            segments = self._parse_context_segments(context_text)
            for source in sources:
                source_index = source.get(self._json_keys["source_index"])
                quotes = source.get(self._json_keys["french_quotes"], [])
                segment = segments.get(source_index, "")
                if not segment:
                    continue
                for quote in quotes:
                    if quote not in segment:
                        errors.append(
                            f"Source {source_index} includes a quote not in context."
                        )
        return errors

    def build_sources_catalog(self, source_entries: List[SourceEntry]):
        """Build a stable catalog of metadata for output sources."""
        catalog = []
        for entry in source_entries:
            catalog.append(
                {
                    self._json_keys["source_index"]: entry.source_index,
                    self._source_metadata_keys["seminar_title"]: entry.seminar_title,
                    self._source_metadata_keys["seminar_id"]: entry.seminar_id,
                    self._source_metadata_keys["lesson_label"]: entry.lesson_label,
                    self._source_metadata_keys["chunk_id"]: entry.chunk_id,
                    self._source_metadata_keys["chunk_index"]: entry.chunk_index,
                    self._source_metadata_keys["pages"]: entry.pages or [],
                }
            )
        return catalog

    @staticmethod
    def _get_text(source: Dict[str, Any], key: str) -> str:
        value = source.get(key)
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _extract_json_text(response_text: str) -> str:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return response_text
        return response_text[start:end + 1]

    @staticmethod
    def _basic_clean_json_text(response_text: str) -> str:
        cleaned = ResponseProcessor._extract_json_text(response_text)
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip())

    def _repair_json_text(self, response_text: str) -> str:
        cleaned = self._basic_clean_json_text(response_text)
        cleaned = cleaned.replace("“", '"').replace("”", '"')
        cleaned = cleaned.replace("‘", "'").replace("’", "'")
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return cleaned

    @staticmethod
    def _contains_english_tokens(text: Any) -> bool:
        if not isinstance(text, str):
            return False
        tokens = re.findall(r"[A-Za-z]+", text.lower())
        if not tokens:
            return False
        english_markers = {
            "the", "and", "with", "from", "this", "that", "which", "into",
            "when", "where", "because", "there", "their", "them", "also",
            "however", "therefore", "while", "whose", "what", "who", "how",
        }
        return any(token in english_markers for token in tokens)

    @staticmethod
    def _normalize_char_for_alignment(ch: str) -> str:
        if ch in ("’", "‘", "'"):
            return "'"
        if ch in ("“", "”", "«", "»", '"'):
            return '"'
        if ch in ("–", "—", "-"):
            return "-"
        return ch.lower()

    def _normalize_text_no_ws(self, text: Any):
        if not isinstance(text, str):
            return "", []
        normalized = []
        mapping = []
        for idx, ch in enumerate(text):
            if ch.isspace():
                continue
            normalized.append(self._normalize_char_for_alignment(ch))
            mapping.append(idx)
        return "".join(normalized), mapping

    def _snap_quote_to_context(self, quote: str, segment: str) -> str:
        if not quote or not segment:
            return quote
        if quote in segment:
            return quote
        normalized_segment, mapping = self._normalize_text_no_ws(segment)
        normalized_quote, _ = self._normalize_text_no_ws(quote)
        if normalized_quote:
            start = normalized_segment.find(normalized_quote)
            if start != -1:
                end = start + len(normalized_quote) - 1
                if end < len(mapping):
                    return segment[mapping[start]:mapping[end] + 1]
        return quote

    @staticmethod
    def _parse_context_segments(context_text: str):
        segments = {}
        for block in context_text.split("\n\n"):
            match = re.match(r"^\[(\d+)\]\s*(.*)$", block, re.S)
            if match:
                segments[int(match.group(1))] = match.group(2)
        return segments

    @staticmethod
    def _extract_default_quotes(segment: str, max_quotes: int = 2):
        if not segment:
            return []
        candidates = re.split(r"(?<=[.!?])\s+", segment.strip())
        quotes = [item.strip() for item in candidates if item.strip()]
        return quotes[:max_quotes] if quotes else []


class AuditLogger:
    """Persist context and responses for debugging."""

    def __init__(self, audit_dir: Path):
        self._audit_dir = audit_dir

    def save_context(
        self,
        question: str,
        question_fr: str,
        language_code: str,
        sources: List[SourceEntry],
        context_text: str,
        *,
        source_id: Optional[str] = None,
    ):
        """Persist the query context and sources for auditing."""
        self._audit_dir.mkdir(exist_ok=True)
        payload = {
            "language": language_code,
            "question": question,
            "question_french": question_fr,
            "sources": [source.__dict__ for source in sources],
        }
        if source_id:
            payload["source_id"] = source_id
        (self._audit_dir / "rag_context_last.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self._audit_dir / "rag_context_last.txt").write_text(
            context_text,
            encoding="utf-8",
        )

    def save_response(self, response_text: str, response_json: Optional[Dict[str, Any]]):
        """Persist the raw and normalized response for auditing."""
        self._audit_dir.mkdir(exist_ok=True)
        (self._audit_dir / "rag_response_last.txt").write_text(
            response_text,
            encoding="utf-8",
        )
        if response_json is None:
            return
        (self._audit_dir / "rag_response_last.json").write_text(
            json.dumps(response_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
