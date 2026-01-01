---
name: pytest-autotest-90
description: Generate or update pytest suites to reach at least 90% coverage in Python repos without modifying non-test code. Use when asked to create tests, raise coverage, or build deterministic pytest suites for the app/ package and tests/ structure.
---

# Pytest Autotest 90

## Overview

Generate deterministic pytest suites under `tests/` that mirror `app/`, then iterate on failures and coverage gaps until reaching 90% line coverage without touching non-test code.

## Workflow

1. Read `references/rules.md` for hard constraints, then read `references/prompt.md` for the detailed end-to-end instructions.
2. Discover Python modules under `app/` by walking the tree and parsing AST without importing modules.
3. Create or update `tests/` to mirror the `app/` layout, mapping each module to `tests/<path>/test_<module>.py`.
4. Author tests using `assets/templates/test_module.py.j2` as a base and extend with unit tests, edge cases, and error handling.
5. Mock external APIs and side effects (network, filesystem, env vars, time, randomness) using pytest fixtures, `monkeypatch`, and `tmp_path`.
6. Run quick tests: `pytest -q --maxfail=1` (or prefix with `PYTHONPATH=.` if imports fail), fix failures by editing tests only.
7. Run coverage: `pytest --cov=app --cov-report=term-missing --cov-fail-under=90` (or prefix with `PYTHONPATH=.` if needed).
8. Add targeted tests for missing lines, then repeat steps 6-7 until tests pass and coverage >= 90% or until blockers must be reported.

## Repository-specific guidance

- Treat `app/` as the coverage target and only place tests under `tests/`.
- Mirror paths, for example `app/application/rag_query.py` -> `tests/application/test_rag_query.py`.
- Mock modules that touch external services or filesystems, including `app/infrastructure/openai_client.py`, `app/infrastructure/retriever.py`, and `app/infrastructure/text_extractor.py`.
