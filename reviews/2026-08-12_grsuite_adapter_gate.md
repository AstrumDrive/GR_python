# GR_python → grsuite-contracts gate — 2026-08-12

- Local adapter tests: **4 passed**.
- Local deterministic verifier: **8/8 checks**, **0 failures**.
- ASTRA/ASTRUM remote verifier: **PASS**.
- ASTRA job: `astrum_20260812_141013_nelson_abdc12`.

The adapter translates existing metric, curvature, matter and ADM result
dictionaries. It does not recompute any of those quantities. Missing optional
phases remain absent or conditional, preserving the distinction between “not
requested” and “computed”.
