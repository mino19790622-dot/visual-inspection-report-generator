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

