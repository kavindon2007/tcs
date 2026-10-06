# Evaluation Report

Evaluated on `dataset/sample_claims.csv` (20 rows, 29 images).


## Per-Field Accuracy

| Field | Correct | Total | Accuracy |
|---|---|---|---|
| evidence_standard_met | 0 | 20 | 0.0% |
| risk_flags | 0 | 20 | 0.0% |
| issue_type | 0 | 20 | 0.0% |
| object_part | 0 | 20 | 0.0% |
| claim_status | 0 | 20 | 0.0% |
| valid_image | 0 | 20 | 0.0% |
| severity | 0 | 20 | 0.0% |

## claim_status Confusion Matrix

Rows: predicted. Columns: expected.

| | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 0 | 0 | 0 |
| contradicted | 0 | 0 | 0 |
| not_enough_information | 0 | 0 | 0 |

## Operational Analysis

| Metric | Sample set | Test set (claims.csv) |
|---|---|---|
| Rows processed | 20 | 44 |
| Images processed | 29 | ~63 |
| Approx LLM calls | 98 | ~215 |
| Elapsed time | 421.1s | ~926s estimated |
| Avg latency per claim | 21.1s | — |

### Model pricing assumptions (Gemini 2.5 Flash, Jun 2026)
- Input tokens:  $0.075 / 1M tokens
- Output tokens: $0.30 / 1M tokens
- Image tokens:  ~800 tokens per image

### Approximate cost for full test set (44 claims, ~2 images each)
- Stage 1 (text):   44 calls × ~1K tokens  ≈ $0.004
- Stage 2 (image):  ~88 calls × ~900 tokens ≈ $0.006
- Stage 3 (image):  ~80 calls × ~2K tokens  ≈ $0.012
- Stage 5 (text):   44 calls × ~3K tokens   ≈ $0.010
- **Total estimated cost: < $0.05 for the full test set**

### Rate limits and mitigation
- `MAX_WORKERS=4` limits concurrent requests to respect RPM quotas.
- Exponential backoff (`RETRY_BASE_DELAY=2.0s`, `MAX_RETRIES=3`) handles transient 429s.
- Disk cache keyed by SHA-256(model + prompt + image_path) avoids re-spending
  quota on pipeline re-runs or retries of the same claim.
- Stage 2 (cheap VLM) gates Stage 3 (expensive VLM) — no_match images skip Stage 3.
- Stage 4 and 6 are fully deterministic and consume zero API quota.