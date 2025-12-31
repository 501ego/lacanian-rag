"""CLI entry point for Lacanian RAG answers."""

import json
import sys

from ..application.rag_query import run_rag_query
from ..core.config import LANGUAGE_NAMES, UI_TEXT


def choose_language() -> str:
    """Prompt until a supported language code is selected."""
    while True:
        choice = input("Choose language (en/es): ").strip().lower()
        if choice in LANGUAGE_NAMES:
            return choice
        print("Please choose 'en' or 'es'.")


def query_lacan(question: str, language_code: str) -> None:
    """Run a full RAG query and print JSON output."""
    ui_text = UI_TEXT
    print(ui_text["searching"])

    def _notify_generate() -> None:
        print(f"\n{ui_text['generating']}\n")

    try:
        payload, warnings = run_rag_query(
            question,
            language_code,
            on_generate=_notify_generate,
        )
    except ValueError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return

    if warnings.get("translation_error"):
        print(warnings["translation_error"], file=sys.stderr)
    validation_errors = warnings.get("validation_errors", [])
    if validation_errors:
        print(f"\n{ui_text['validation_error']}", file=sys.stderr)
        for error in validation_errors:
            print(f"- {error}", file=sys.stderr)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    selected_language = choose_language()
    question_text = input(f"{UI_TEXT['ask_question']}: ")
    query_lacan(question_text, selected_language)
