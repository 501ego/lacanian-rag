# Prompt for Improving an Existing RAG Project (Python, OOP, Clean Code)

## Context

You are working with an already implemented RAG project in Python.

Project facts:

- The embedding model already in use is: text-embedding-3-large.
- The project already contains persisted artifacts:
  - An index file that stores embeddings (for example numpy arrays, FAISS index, or similar).
  - A metadata file that maps one-to-one with embeddings (for example doc_id, source, page or section, chunk_id, original text, hash).
- You must reuse the existing index and metadata. Do not regenerate embeddings and do not rebuild the ingestion pipeline from scratch.

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

## Output rules

- Output only Python code.
- No explanations.
- No Markdown formatting outside the code block.
- No commentary or analysis text.
- The code must be copy-pasteable and runnable with minimal adaptation.
