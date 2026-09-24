# Maintenance log

> This file is appended automatically by a scheduled maintenance task, and the
> commits that touch it carry `[skip ci]`. Every entry corresponds to one real
> change that passed the repository's own checks before it was pushed.
>
> The entries are generated rather than hand-written, and they record
> maintenance work only — they are not a measure of manual development effort.

## 2026-09-22 — pin the unknown-vs-free invariants of cost_for()

- **Change**: new tests/test_pricing.py (15 tests) covering: a None token count is unknown rather than 0; a measured zero is known and free; a null rate is unknown; PRICING_PATH resolved before the lru_cache; the shipped config/pricing.yaml still parses
- **Verification**: pytest tests: 240 passed; ruff: all checks passed

## 2026-09-23 — document the None-means-undefined contract of spearman()

- **Change**: eval/judge_agreement.py: add a docstring to the public spearman() helper stating that it returns None when rho is undefined (fewer than two paired samples, or a constant side) rather than 0, and that callers must not read None as 'no agreement'
- **Verification**: pytest tests: 240 passed; ruff: all checks passed

## 2026-09-24 — Document the top-vs-nested allow-list switch in _clean()

- **Change**: Added a docstring to app/observability/redaction.py::_clean explaining that the top flag selects ALLOWED_TOP_LEVEL for the root dict and ALLOWED_KEYS for every nested dict, and that the two sets differ. Docstring only; no code path changed.
- **Verification**: pytest tests -q -> 240 passed; ruff check -> All checks passed

