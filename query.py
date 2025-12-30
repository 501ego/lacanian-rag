import json
import os
import re
from pathlib import Path
from openai import OpenAI
from retriever import search_similar_chunks
from dotenv import load_dotenv
from lecon_labels import LECON_LABELS
from lacan_canon import LACAN_CANON

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)
GPT_MODEL = "gpt-4o"
TRANSLATION_MODEL = "gpt-4o-mini"
TOP_K = 5
MAX_SOURCES = 4
CHUNKS_DIR = Path("text_chunks_json")
LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}
UNKNOWN_LECON = {"en": "Unknown lesson", "es": "Lección desconocida"}
UI_TEXT = {
    "ask_question": "Ask a question about Lacan",
    "searching": "Searching relevant fragments...",
    "generating": "Generating response with GPT...",
    "context_error": "[ERROR] Could not build the context. Check chunks or metadata.",
    "file_error": "[!] Error opening file {path}: {error}",
    "missing_chunk": "[!] No matching chunk found in {seminar} for id {chunk_id}",
    "translation_error": "[!] Translation failed, using the original question: {error}",
    "validation_error": "[ERROR] The model output did not follow the JSON schema.",
}
JSON_KEYS = {
    "language": "language",
    "sources": "sources",
    "source_index": "source_index",
    "french_quotes": "french_quotes",
    "translations": "translations",
    "translation_critique": "translation_critique",
    "context": "context",
    "lacanian_development": "lacanian_development",
    "comparative_trajectory": "comparative_trajectory",
    "source_metadata": "source_metadata",
}
SOURCE_METADATA_KEYS = {
    "seminar_title": "seminar_title",
    "seminar_id": "seminar_id",
    "lesson_label": "lesson_label",
    "lesson_raw": "lesson_raw",
    "chunk_id": "chunk_id",
    "chunk_index": "chunk_index",
    "pages": "pages",
}
STRUCTURE_HINT = (
    "Output must be valid JSON only, with no extra text.\n"
    "Schema:\n"
    "{\n"
    '  "language": "en|es",\n'
    '  "sources": [\n'
    "    {\n"
    '      "source_index": 1,\n'
    '      "french_quotes": ["..."],\n'
    '      "translations": ["..."],\n'
    '      "translation_critique": "...",\n'
    '      "context": "...",\n'
    '      "lacanian_development": "..."\n'
    "    }\n"
    "  ],\n"
    '  "comparative_trajectory": "..." \n'
    "}\n"
    "Use the exact keys shown above.\n"
    "Use \\n\\n inside strings to separate paragraphs."
)
RETRY_PROMPT = (
    "Your last answer did not follow the JSON schema. "
    "Return JSON only, use the exact keys, and ensure all quotes are copied "
    "exactly from the context. Set the language field correctly."
)
AUDIT_DIR = Path("rag_audit")


def choose_language():
    while True:
        choice = input("Choose language (en/es): ").strip().lower()
        if choice in LANGUAGE_NAMES:
            return choice
        print("Please choose 'en' or 'es'.")


def translate_text(text, source_language, target_language):
    response = client.chat.completions.create(
        model=TRANSLATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translation engine. "
                    f"Translate from {source_language} to {target_language}. "
                    "Output only the translated text without quotes or extra text."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def translate_to_french(text, source_language):
    return translate_text(text, source_language, "French")


def get_lecon_label(lecon_raw, language_code):
    if not lecon_raw:
        return UNKNOWN_LECON.get(language_code, UNKNOWN_LECON["en"])
    labels = LECON_LABELS.get(language_code, LECON_LABELS.get("en", {}))
    return labels.get(lecon_raw, lecon_raw)


def save_audit(question, question_fr, language_code, sources, context_text):
    AUDIT_DIR.mkdir(exist_ok=True)
    payload = {
        "language": language_code,
        "question": question,
        "question_french": question_fr,
        "sources": sources,
    }
    (AUDIT_DIR / "rag_context_last.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (AUDIT_DIR / "rag_context_last.txt").write_text(
        context_text,
        encoding="utf-8",
    )


def save_response_audit(response_text, response_json):
    AUDIT_DIR.mkdir(exist_ok=True)
    (AUDIT_DIR / "rag_response_last.txt").write_text(
        response_text,
        encoding="utf-8",
    )
    if response_json is None:
        return
    (AUDIT_DIR / "rag_response_last.json").write_text(
        json.dumps(response_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_context_segments(context_text):
    segments = {}
    for block in context_text.split("\n\n"):
        match = re.match(r"^\[(\d+)\]\s*(.*)$", block, re.S)
        if match:
            segments[int(match.group(1))] = match.group(2)
    return segments


def extract_json_text(response_text):
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return response_text
    return response_text[start:end + 1]


def parse_model_json(response_text):
    cleaned = extract_json_text(response_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def has_two_paragraphs(text):
    return isinstance(text, str) and "\n\n" in text.strip()


def contains_english_tokens(text):
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


def collect_validation_errors(response_json, total_sources, language_code):
    errors = []
    if not isinstance(response_json, dict):
        return ["Response is not a JSON object."]
    language = response_json.get(JSON_KEYS["language"])
    if language not in ("en", "es"):
        errors.append('Missing or invalid "language" field (must be "en" or "es").')
    sources = response_json.get(JSON_KEYS["sources"])
    if not isinstance(sources, list):
        return errors + ['"sources" must be an array.']
    if len(sources) != total_sources:
        errors.append(f'"sources" must contain exactly {total_sources} objects.')
    for idx, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            errors.append(f"Source {idx} is not an object.")
            continue
        if source.get(JSON_KEYS["source_index"]) != idx:
            errors.append(f"Source {idx} has incorrect source_index.")
        quotes = source.get(JSON_KEYS["french_quotes"])
        translations = source.get(JSON_KEYS["translations"])
        if not isinstance(quotes, list) or not quotes:
            errors.append(f"Source {idx} must include french_quotes array with at least one item.")
        if not isinstance(translations, list) or len(translations) != len(quotes or []):
            errors.append(f"Source {idx} must include translations matching french_quotes length.")
        if any(not isinstance(item, str) or not item.strip() for item in quotes or []):
            errors.append(f"Source {idx} has empty quote strings.")
        if any(not isinstance(item, str) or not item.strip() for item in translations or []):
            errors.append(f"Source {idx} has empty translation strings.")
        for key in (
            JSON_KEYS["translation_critique"],
            JSON_KEYS["context"],
            JSON_KEYS["lacanian_development"],
        ):
            value = source.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"Source {idx} missing required field: {key}.")
        if not has_two_paragraphs(source.get(JSON_KEYS["context"])):
            errors.append(f"Source {idx} context must have at least two paragraphs.")
        if not has_two_paragraphs(source.get(JSON_KEYS["lacanian_development"])):
            errors.append(f"Source {idx} lacanian_development must have at least two paragraphs.")
        if language_code == "es":
            for key in (
                JSON_KEYS["translation_critique"],
                JSON_KEYS["context"],
                JSON_KEYS["lacanian_development"],
            ):
                if contains_english_tokens(source.get(key, "")):
                    errors.append(f"Source {idx} contains English in {key}.")
    trajectory = response_json.get(JSON_KEYS["comparative_trajectory"])
    if not isinstance(trajectory, str) or not has_two_paragraphs(trajectory):
        errors.append("comparative_trajectory must have at least two paragraphs.")
    if language_code == "es" and contains_english_tokens(trajectory):
        errors.append("comparative_trajectory contains English.")
    return errors


def validate_json_output(response_json, total_sources, language_code):
    return not collect_validation_errors(response_json, total_sources, language_code)


def quotes_match_sources(response_json, context_text):
    segments = parse_context_segments(context_text)
    sources = response_json.get(JSON_KEYS["sources"], [])
    for source in sources:
        source_index = source.get(JSON_KEYS["source_index"])
        quotes = source.get(JSON_KEYS["french_quotes"], [])
        segment = segments.get(source_index, "")
        for quote in quotes:
            if quote not in segment:
                return False
    return True


def build_prompt(question, retrieved_chunks, language_code):
    context_parts = []
    seminar_cache = {}
    canon = LACAN_CANON.get(language_code, {})
    ui = UI_TEXT
    sources = []
    source_index = 1

    if not retrieved_chunks:
        return None, [], ""

    for i, chunk in enumerate(retrieved_chunks):
        seminar = chunk.get("seminar")
        if not seminar:
            continue

        if seminar not in seminar_cache:
            path = CHUNKS_DIR / f"{seminar}.json"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    all_chunks = json.load(f)
                by_id = {c["chunk_id"]: c for c in all_chunks}
                by_index = {c["chunk_index"]: c for c in all_chunks}
                seminar_cache[seminar] = (by_id, by_index)
            except Exception as e:
                print(ui["file_error"].format(path=path, error=e))
                seminar_cache[seminar] = (None, None)

        by_id, by_index = seminar_cache.get(seminar, (None, None))
        if not by_id:
            continue

        raw_id = chunk.get("id", "")
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

        if matching_chunk:
            context_parts.append(f"[{source_index}] {matching_chunk['text']}")
            seminario = canon.get(seminar, {}).get("titulo", seminar)
            lecon_raw = matching_chunk.get("lecon") or chunk.get("lecon")
            lecon = get_lecon_label(lecon_raw, language_code)
            pages = matching_chunk.get("pages") or chunk.get("pages") or []
            if isinstance(pages, list):
                pages = sorted(set(pages))
            source_entry = {
                "source_index": source_index,
                "seminar_id": seminar,
                "seminar_title": seminario,
                "lesson_label": lecon,
                "lesson_raw": lecon_raw,
                "chunk_id": matching_chunk.get("chunk_id"),
                "chunk_index": matching_chunk.get("chunk_index"),
                "full_id": chunk.get("id"),
                "pages": pages,
            }
            sources.append(source_entry)
            source_index += 1
        else:
            print(ui["missing_chunk"].format(seminar=seminar, chunk_id=raw_id))

    if not context_parts:
        return None, [], ""

    context_text = "\n\n".join(context_parts)
    language_name = LANGUAGE_NAMES[language_code]
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
- Provide 1-2 French quotes per source and a translation for each quote
- Set "language" to "{language_code}"
- Each "context" and "lacanian_development" value must include at least two paragraphs separated by "\\n\\n"
- The "comparative_trajectory" value must include at least two paragraphs separated by "\\n\\n"
- The "translation_critique" must include at least one alternative rendering
- Write the "lacanian_development" in a Lacanian voice while avoiding facts not in the context; mark speculation explicitly
- If "language" is "es", avoid English words in string values (except French quotes and [n] citations)
- Cite claims with numbered references like [1], [2], etc. inside the relevant string values
- {STRUCTURE_HINT}

Begin your answer below:
"""
    return prompt, sources, context_text


def query_lacan(question, language_code):
    ui = UI_TEXT
    print(ui["searching"])
    language_name = LANGUAGE_NAMES[language_code]
    try:
        question_fr = translate_to_french(question, language_name)
    except Exception as e:
        print(ui["translation_error"].format(error=e))
        question_fr = question
    chunks = search_similar_chunks(question_fr, top_k=TOP_K)
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk.get("id"), chunk.get("seminar"))
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    selected_chunks = unique_chunks[:MAX_SOURCES]
    prompt, source_entries, context_text = build_prompt(
        question, selected_chunks, language_code
    )
    save_audit(question, question_fr, language_code,
               source_entries, context_text)

    if not prompt:
        print(f"\n{ui['context_error']}")
        return

    print(f"\n{ui['generating']}\n")
    messages = [
        {"role": "system", "content": "You are a helpful assistant on Lacanian psychoanalysis."},
        {"role": "user", "content": prompt},
    ]
    response_text = ""
    parsed_response = None
    for attempt in range(3):
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2600,
            response_format={"type": "json_object"},
        )
        response_text = response.choices[0].message.content.strip()
        parsed_response = parse_model_json(response_text)
        if (
            parsed_response
            and validate_json_output(parsed_response, len(source_entries), language_code)
            and parsed_response.get(JSON_KEYS["language"]) == language_code
            and quotes_match_sources(parsed_response, context_text)
        ):
            break
        errors = collect_validation_errors(parsed_response or {}, len(source_entries), language_code)
        error_block = "\n".join(f"- {error}" for error in errors) or "- Unknown validation error."
        messages.append(
            {
                "role": "user",
                "content": f"{RETRY_PROMPT}\nValidation errors:\n{error_block}",
            }
        )

    if (
        not parsed_response
        or not validate_json_output(parsed_response, len(source_entries), language_code)
        or parsed_response.get(JSON_KEYS["language"]) != language_code
    ):
        print(f"\n{ui['validation_error']}")
        print(response_text)
        save_response_audit(response_text, None)
        return

    enriched_sources = []
    for idx, source in enumerate(parsed_response[JSON_KEYS["sources"]]):
        meta = source_entries[idx]
        metadata = {
            SOURCE_METADATA_KEYS["seminar_title"]: meta["seminar_title"],
            SOURCE_METADATA_KEYS["seminar_id"]: meta["seminar_id"],
            SOURCE_METADATA_KEYS["lesson_label"]: meta["lesson_label"],
            SOURCE_METADATA_KEYS["lesson_raw"]: meta["lesson_raw"],
            SOURCE_METADATA_KEYS["chunk_id"]: meta["chunk_id"],
            SOURCE_METADATA_KEYS["chunk_index"]: meta["chunk_index"],
            SOURCE_METADATA_KEYS["pages"]: meta["pages"],
        }
        enriched = dict(source)
        enriched[JSON_KEYS["source_metadata"]] = metadata
        enriched_sources.append(enriched)

    output_payload = dict(parsed_response)
    output_payload[JSON_KEYS["language"]] = language_code
    output_payload[JSON_KEYS["sources"]] = enriched_sources

    save_response_audit(response_text, output_payload)
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    language_code = choose_language()
    ui = UI_TEXT
    question = input(f"{ui['ask_question']}: ")
    query_lacan(question, language_code)
