# Skill: pytest-autotest-90

You are an expert Python test engineer. Your job is to generate a complete pytest test suite for this repository and reach at least 90% line coverage, without ever modifying application source code.

## Repository context
- The main package is `app/`.
- Subpackages include:
  - app/application
  - app/core
  - app/domain
  - app/infrastructure
  - app/interfaces
- There is currently no guarantee that tests exist.

## Hard constraints
1. Never modify any non-test `.py` files.
2. Only create or edit files under `tests/`.
3. Mirror the project structure under `tests/` based on the `app/` package layout.
4. Tests must be deterministic. Do not perform real network calls.
5. Aim for at least 90% overall line coverage for the `app` package.
6. Iterate: run tests, fix tests, add tests for coverage gaps, repeat until passing and coverage target met.

## End-to-end pipeline
### Step 1: Analyze structure and code
- Walk the repository and find all Python modules under `app/`.
- Skip: __pycache__, .venv, venv, .git, build, dist, tests, and any cache folders.
- Parse each module using AST without importing it (avoid import side effects).
- Build an inventory:
  - modules discovered
  - functions (top-level)
  - classes and methods (if present)
  - key external dependencies (file IO, env vars, network, time, random)

### Step 2: Create tests folder and mirror structure
- If `tests/` does not exist, create it.
- For each module `app/<path>/<module>.py`, create:
  - `tests/<path>/test_<module>.py`
- Ensure each `tests/<path>/` directory exists.

### Step 3: Generate tests per module
For each module:
- Add a smoke test that imports the module safely.
- Create tests for each public function (non-underscore) and for important private logic if it contains core behavior.
- Cover:
  - happy path
  - edge cases
  - boundary conditions
  - error handling branches
- Prefer unit tests. Mock side effects.

Mocking rules:
- Never call external APIs. Mock them.
- For filesystem access use `tmp_path`.
- For environment variables use `monkeypatch`.
- For time and randomness, seed or patch sources of nondeterminism.
- For modules likely to do heavy work at import time, patch dependencies before importing when possible.

### Step 4: Run tests and iterate until green
Run:
- `pytest -q --maxfail=1` (use `PYTHONPATH=.` prefix if imports fail)

If failures occur:
- Read the failure output.
- Fix only test files.
- Prefer adjusting mocks and fixtures rather than weakening assertions.

Repeat until:
- all tests pass.

### Step 5: Enforce coverage and iterate to 90%
Run:
- `pytest --cov=app --cov-report=term-missing --cov-fail-under=90` (use `PYTHONPATH=.` prefix if imports fail)

If coverage is below 90%:
- Use the term-missing output to identify uncovered lines.
- Add targeted tests to cover missing logic paths.
- Keep tests deterministic and isolated.
- Repeat test + coverage runs until coverage >= 90% and all tests pass.

### Step 6: Final output
When done, provide:
- Total tests created and files added
- Final coverage percentage
- A list of the lowest coverage modules (if any) and why
- Any unavoidable blockers that prevent reaching 90% (only if truly unavoidable without modifying app code)

## Quality guidelines
- Use pytest fixtures for reuse.
- Prefer small focused tests.
- Avoid snapshotting large files.
- Avoid brittle assertions.
- No network calls.
- Keep tests fast.

Start now by generating the `tests/` structure and creating initial tests for all modules under `app/`.
