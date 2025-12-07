**Test Results - Sinhala Visual Mapping Service**

Summary
- Date: 2025-12-08

Overall test results
- Latest full run: `27 passed, 0 failed, 0 skipped` 

Unit tests
- Command: `python -m pytest -q`
- Result: `7 passed, 1 skipped`
- Scope: `tests/test_mindmap_generator.py` covers `SinhalaMindMapGenerator` behavior (empty input handling, root node creation, edge connectivity, metadata correctness, subtopic/detail limits, unique IDs).

Endpoint tests

- Scope: `tests/test_api_endpoints.py` covers `/health`, `/api/mindmap/generate` (direct + external mocked), `/api/mindmap/batch`, and `/api/mindmap/essay/<id>` when Neo4j not configured. Tests mock external calls and disable Neo4j to remain hermetic.

Performance micro-benchmarks
- Script: `perf/run_perf.py` (synthetic Sinhala-like inputs)
- Command: `python perf/run_perf.py`
- Collected timings (ms):
  - Small (120 iters): median ~0.0115 ms, avg ~0.0138 ms, p95 ~0.0195 ms
  - Medium (60 iters): median ~0.0202 ms, avg ~0.0215 ms, p95 ~0.0250 ms
  - Long (20 iters): median ~0.1424 ms, avg ~0.1432 ms, p95 ~0.1515 ms
- Notes: These are generator-only timings (in-process, no HTTP, no DB). Real API latency will be higher when including Flask overhead, JSON serialization, and Neo4j writes.

How to reproduce locally
1. Create & activate venv (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
1. Install dev requirements:
```
python -m pip install -r requirements-dev.txt
```
1. Run unit tests:
```
python -m pytest -q
```
1. Run endpoint tests:
```
python -m pytest tests/test_api_endpoints.py -q
```
1. Run perf script:
```
python perf/run_perf.py
```

Interpretation & quality assessment
- The core generator (`SinhalaMindMapGenerator`) is functionally correct for the tested cases and fast in microbenchmarks.
- Current test coverage is focused on generator logic and basic endpoints. Edge cases such as complex punctuation, Unicode normalization, and very large inputs need additional tests.
- Persistence (Neo4j) is exercised lightly; add mock-based tests for `neo4j_driver` interactions and integration tests against a test database for end-to-end verification.

Recommendations / Next steps
- Add property-based tests and more edge case fixtures (punctuation variants, Unicode normalization, very long texts).
- Add `pytest-benchmark` to track performance over time and create regression alerts in CI.
- Add CI (GitHub Actions) that:
  - Installs `requirements-dev.txt`
  - Runs `pytest -q`
  - Optionally runs a short benchmark or `pytest-benchmark` job
- Add integration tests for Neo4j (or use testcontainers to run Neo4j in CI) and mock tests verifying `execute_write` calls.
- Add end-to-end latency tests (Flask + optional Neo4j) and a caching layer (Redis) to reduce repeated-generation costs.

Coverage
- Command run: `python -m pytest --cov=mindmap_generator --cov-report=term --cov-report=html`
- Result: `94%` coverage for `mindmap_generator.py` (5 lines missed)
- HTML report path: `./htmlcov/index.html`


