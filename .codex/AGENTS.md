# Prompt for Improving an Existing RAG Project (Python, OOP, Clean Code)

## Context

You are working with an already implemented RAG project in Python.

Project facts:

- The embedding model already in use is: text-embedding-3-large.
- The project already contains persisted artifacts:
  - An index file that stores embeddings (for example numpy arrays, FAISS index, or similar).
  - A metadata file that maps one-to-one with embeddings (for example doc_id, source, page or section, chunk_id, original text, hash).
- You must reuse the existing index and metadata. Do not regenerate embeddings and do not rebuild the ingestion pipeline from scratch unless explicitly asked.

Primary goal:

- Improve, refactor, and harden the existing RAG project as much as possible within scope, applying Python OOP and Clean Code best practices.
- The outcome should be a cleaner, more maintainable, more extensible RAG query layer built on top of what already exists.

## Mandatory first step

At the very beginning of the conversation:

1. Inspect and reason about the current project structure and existing code.
2. Infer how the index and metadata are currently loaded and queried.
3. Identify weaknesses, tight coupling, missing abstractions, or code smells.
4. Use that analysis as context for all design and refactor decisions that follow.

Do not skip this step. Assume the user will paste or describe the current structure and code.

## Project architecture (authoritative)

Read this first. It is the fast map of how the repo is organized and how changes must be applied.

Pipeline map:
`app/core/config.py` -> `app/infrastructure/openai_client.py` -> `app/infrastructure/retriever.py`
-> `app/application/rag_services.py` -> `app/application/rag_query.py` -> `app/interfaces/api.py` or `app/interfaces/cli.py`
Ingestion: `app/infrastructure/text_extractor.py` -> `app/infrastructure/embed_chunks.py`
Indexing use case: `app/application/index_text.py`

Ownership rules (do not cross):
- `app/core/config.py`: all constants (paths, limits, labels, JSON keys, UI text). No ad-hoc globals.
- `app/domain/*.py`: Lacan canon and lesson labels.
- `app/infrastructure/openai_client.py`: only OpenAI SDK entry point (chat/translate/embed).
- `app/infrastructure/retriever.py`: index loading + vector search (`IndexStore`, `OpenAIEmbedder`, `Retriever`).
- `app/application/rag_services.py`: prompt building + response processing + audit logging.
- `app/application/rag_query.py`: RAG use case orchestration (input -> retrieval -> prompt -> response -> output).
- `app/application/index_text.py`: raw text chunking + index append orchestration.
- `app/infrastructure/text_extractor.py`: PDF chunking (`PDFChunker`, `TextChunker`).
- `app/infrastructure/embed_chunks.py`: embedding + FAISS build (`ChunkLoader`, `FaissIndexBuilder`, `FaissIndexUpdater`).
- `app/interfaces/api.py`: FastAPI wiring and HTTP concerns.
- `app/interfaces/cli.py`: CLI entry point for RAG queries.
- `app/interfaces/verify.py`: CLI utility for verifying PDF coverage.

If you move responsibilities, update this section to match.

## Design principles (enforced)

- SOLID: single responsibility per class, explicit dependencies, no hidden globals, clear contracts.
- KISS: prefer the simplest design that meets requirements; avoid over-abstraction.
- Clean code naming: use explicit, descriptive names; avoid single-letter names except small loop indices.
- Minimal surface area: only expose what is needed; keep helpers private.
- Error handling: avoid broad exception catches; prefer specific exceptions.
- Imports: remove unused imports and variables.

## Processing instructions

1. Assume reasonable defaults when information is missing, but explicitly support these variants:
   - Index formats:
     - Numpy (.npy or .npz) with shape (n, d)
     - FAISS index (for example faiss_index.bin)
   - Metadata formats:
     - JSON
     - JSONL
2. Design a minimal but robust architecture using OOP:
   - Single responsibility per class
   - Clear boundaries and contracts
   - Low coupling, high cohesion
   - Explicit dependencies (no hidden globals)
   - Full type hints
3. Implement a clean RAG query flow:
   - Load existing index and metadata
   - Embed the user query using text-embedding-3-large
   - Retrieve top-k chunks by similarity
   - Build a bounded context (character or token limit)
   - Generate an answer that cites sources from metadata
4. Focus strictly on improving the current RAG logic and code quality.
   - Do not include observability, monitoring, tracing, metrics, dashboards, or unrelated infrastructure.
5. Treat this as production-quality refactoring guidance, not a tutorial.

## Expected output (code only)

Return a single, self-contained Python code block that includes:

- Project structure described as comments (folders and files).
- Concrete implementations of at least these components:
  - Settings (configuration loading)
  - IndexStore (load and search existing index)
  - MetadataStore (load and access metadata)
  - Retriever (top-k retrieval orchestration)
  - ContextBuilder (context assembly with limits)
  - CitationFormatter (source formatting)
  - RAGService (main facade coordinating everything)
- A CLI entry point (query.py or main.py) that allows:
  python query.py "my question" --top_k 5
- Two retrieval backends:
  A) Numpy cosine similarity (baseline)
  B) FAISS, used automatically if available
- Support for both JSON and JSONL metadata
- Answers that include inline references [1], [2], etc., and a final "Sources" section mapping each reference to metadata entries.
- KISS

## Output rules

- Output only Python code.
- No explanations.
- No Markdown formatting outside the code block.
- No commentary or analysis text.
- The code must be copy-pasteable and runnable with minimal adaptation.
