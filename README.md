# Lacanian RAG API

RAG pipeline for Lacanian texts. It extracts and chunks PDFs, builds a FAISS
vector index with OpenAI embeddings, and exposes both a FastAPI service and a CLI
to run structured RAG queries with citations.

## Features

- PDF text extraction and chunking with page metadata
- FAISS index build/append with OpenAI embeddings
- RAG query flow with structured JSON responses
- FastAPI endpoints for query + incremental indexing
- CLI for interactive queries
- Audit logging of prompts and responses

## Project layout

- `app/application`: use cases (RAG, indexing)
- `app/infrastructure`: chunker, embeddings, FAISS index, OpenAI client, retriever
- `app/interfaces`: API, CLI, logging, verification tooling
- `data_pdfs/`: input PDFs (ignored in git)
- `text_chunks_json/`: chunked output (ignored in git)
- `vector_index/`: FAISS index and metadata (ignored in git)
- `rag_audit/`: audit logs (ignored in git)

## Requirements

- Python 3.10+
- OpenAI API key

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pydantic python-dotenv openai PyPDF2 faiss-cpu numpy tqdm
```

```bash
pip install -r requirements.txt
```

## Testing

Run the unit tests:

```bash
pytest -q
```

Run coverage (requires `pytest-cov`):

```bash
pip install pytest-cov
pytest --cov=app --cov-report=term-missing --cov-fail-under=90
```

## Configure

Set these environment variables (for example in `.env`):

- `OPENAI_API_KEY` (required)
- `OPENAI_GPT_MODEL` (default: `gpt-4o`)
- `OPENAI_TRANSLATION_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-large`)

Optional:

- `LOG_LEVEL` (default: `INFO`)
- `NO_COLOR` (disable colored logs when set)

## Build or update the index

You need a vector index before running queries. Use either approach:

### Batch build from PDFs

1. Place PDFs in `data_pdfs/` (filename becomes the seminar id).
2. Chunk PDFs:

```bash
python -m app.infrastructure.text_extractor
```

3. Build the FAISS index:

```bash
python -m app.infrastructure.embed_chunks
```

### Incremental indexing via API

The API can ingest a PDF, create chunks, and append embeddings:

```bash
curl -X POST http://127.0.0.1:8000/rag/index-text \
  -F "file=@data_pdfs/your.pdf" \
  -F "doc_id=seminar_xi"
```

## Run the API

The ASGI entrypoint is `api.py`.

```bash
uvicorn api:app --reload
# or
fastapi run api.py
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

## API

Base URL: `http://127.0.0.1:8000`

### POST /rag/query

Run a RAG query against the existing FAISS index.

Request body:

```json
{
  "question": "What is the function of the gaze?",
  "language": "en",
  "detail": "concise",
  "stream": false,
  "top_k": 5,
  "max_sources": 4,
  "source_id": null
}
```

Response shape:

```json
{
  "result": {
    "language": "en",
    "sources": [
      {
        "source_index": 1,
        "french_quotes": ["..."],
        "translations": ["..."],
        "translation_critique": "...",
        "context": "...",
        "lacanian_development": "...",
        "source_metadata": {
          "seminar_title": "Seminar XI",
          "seminar_id": "Seminar_XI",
          "lesson_label": "Lesson 1",
          "lesson_raw": "Lecon_1",
          "chunk_id": "chunk_001",
          "chunk_index": 1,
          "pages": [12, 13]
        }
      }
    ],
    "comparative_trajectory": "...",
    "retrieval_catalog": [
      {
        "rank": 1,
        "score": 0.123,
        "seminar_id": "Seminar_XI",
        "seminar_title": "Seminar XI",
        "lesson_label": "Lesson 1",
        "lesson_raw": "Lecon_1",
        "chunk_id": "chunk_001",
        "chunk_index": 1,
        "full_id": "Seminar_XI_chunk_001",
        "pages": [12, 13]
      }
    ],
    "sources_catalog": [
      {
        "source_index": 1,
        "seminar_title": "Seminar XI",
        "seminar_id": "Seminar_XI",
        "lesson_label": "Lesson 1",
        "chunk_id": "chunk_001",
        "chunk_index": 1,
        "pages": [12, 13]
      }
    ]
  },
  "warnings": {
    "translation_error": null,
    "validation_errors": []
  }
}
```

Example curl:

```bash
curl -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the function of the gaze?","language":"en"}'
```

Streaming (SSE):

```bash
curl -N -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the function of the gaze?","language":"en","stream":true}'
```

### POST /rag/expand

Generate a deep critical synthesis from a previous retrieval_catalog.

```bash
curl -X POST http://127.0.0.1:8000/rag/expand \
  -H "Content-Type: application/json" \
  -d '{"language":"en","retrieval_catalog":[{"rank":1,"seminar_id":"Seminar_XI","full_id":"Seminar_XI_chunk_001"}]}'
```

Set `stream: true` to receive SSE events (`meta`, `delta`, `done`, `error`).

### POST /rag/index-text

Upload a PDF, chunk it, and append embeddings to the FAISS index.

```bash
curl -X POST http://127.0.0.1:8000/rag/index-text \
  -F "file=@data_pdfs/your.pdf" \
  -F "doc_id=seminar_xi"
```

## RAG sequence flow

```mermaid
sequenceDiagram
    participant Client
    participant API as /rag/query
    participant Rag as run_rag_query
    participant OpenAI as OpenAIClient
    participant Ret as Retriever
    participant Builder as PromptBuilder/ChunkRepository
    participant Proc as ResponseProcessor
    participant Audit as AuditLogger

    Client->>API: POST {question, language, top_k?, max_sources?, source_id?}
    API->>Rag: run_rag_query(...)
    Rag->>Rag: validate inputs + resolve detail level
    Rag->>OpenAI: translate(question -> French)
    alt translation fails
        Rag->>Rag: fallback to original question
    end
    Rag->>Ret: search_similar_chunks(question_fr, top_k, source_id?)
    alt source_id provided and no chunks
        Rag-->>API: error (no chunks for source_id)
    end
    Rag->>Rag: build retrieval_catalog (labels + chunk ids)
    Rag->>Rag: dedupe + cap max_sources
    Rag->>Builder: build prompt + context + source_entries
    alt audit enabled
        Rag->>Audit: save_context(...)
    end
    Rag->>OpenAI: chat_completion(prompt)
    OpenAI-->>Rag: JSON response text
    Rag->>Proc: parse + normalize + align quotes + ensure translations + validate
    Rag->>Rag: enrich sources + build sources_catalog
    alt audit enabled
        Rag->>Audit: save_response(...)
    end
    Rag-->>API: payload + warnings
    API-->>Client: response
```

## CLI

```bash
python -m app.interfaces.cli
```

Follow the prompts for language selection and question text.

## Verify processed PDFs

```bash
python -m app.interfaces.verify
```

## Notes

- Index files live at `vector_index/lacan.index` and `vector_index/metadata.json`.
- Audit logs are stored in `rag_audit/` when audit is enabled (default).
- If the index or metadata files are missing, RAG queries will fail until the
  index is built or appended.
