# Damage-Claim Verification System

A staged, multimodal pipeline that verifies damage claims using submitted
images, a short claim conversation, user claim history, and minimum
evidence requirements.

## Architecture

The pipeline runs 7 stages per claim. LLM/VLM calls are used only where
genuine judgment is required; all other logic is deterministic Python.

```
Stage 1  ClaimExtraction      gemini-2.5-flash       text-only LLM
Stage 2  ObjectValidator      gemini-2.5-flash-lite   lightweight VLM, per-image
Stage 3  DamageInspector      gemini-2.5-flash        detailed VLM, per-image (4-step CoT)
Stage 4  EvidenceSufficiency  [deterministic]         hard gate, no LLM
Stage 5  FinalDecision        gemini-2.5-flash        text-only LLM, structured findings
Stage 6  RiskOverlay          [deterministic]         cannot modify claim_status
Stage 7  OutputValidation     [deterministic]         schema + cross-field enforcement
```

Every intermediate result (per-image reasoning, evidence gate decision,
decision reasoning) is stored in typed dataclasses and traceable to the
specific observations and rules that produced it.

## Setup

### 1. Install dependencies

```bash
cd code
pip install -r requirements.txt
```

Or with a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in the repo root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Or export it directly:

```bash
export GOOGLE_API_KEY=your_gemini_api_key_here
```

## Running

### Produce predictions for `claims.csv`

```bash
python code/main.py
```

This reads `dataset/claims.csv` and writes `dataset/output.csv`.

Optional flags:
```
--claims PATH    Alternative claims CSV path
--output PATH    Alternative output CSV path
--no-cache       Disable LLM response caching (forces fresh API calls)
```

### Run the evaluation harness

```bash
python code/evaluation/main.py
```

This runs the full pipeline on `dataset/sample_claims.csv` (which has
expected outputs) and produces:
- `evaluation/sample_comparison.csv` — row-by-row predicted vs expected
- `evaluation/evaluation_report.md` — per-field accuracy, confusion matrix,
  and operational cost/latency analysis

## File structure

```
code/
├── main.py                        # Entry point → dataset/output.csv
├── requirements.txt
├── agent/
│   ├── config.py                  # Models, thresholds, allowed values
│   ├── models.py                  # Dataclasses for all stage I/O
│   ├── prompts.py                 # LLM prompt templates
│   ├── llm_client.py              # Gemini API wrapper with retry + cache
│   └── stages/
│       ├── s1_claim_extraction.py
│       ├── s2_object_validator.py
│       ├── s3_damage_inspector.py
│       ├── s4_evidence_gate.py
│       ├── s5_final_decision.py
│       ├── s6_risk_overlay.py
│       └── s7_output_validator.py
└── evaluation/
    └── main.py                    # Eval harness → evaluation_report.md
```

## Key design decisions

**Stage ordering prevents bias.** Stage 1 extracts the claim from the
conversation before any images are seen. This prevents the model from
letting visible damage influence its reading of what the customer claimed.

**Uncertain images are preserved.** Stage 2 (ObjectValidator) returns
`match`, `uncertain`, or `no_match`. Uncertain images pass to Stage 3
with a `manual_review_required` flag rather than being silently dropped.

**No image is silently discarded.** Every submitted image has a
corresponding `ObjectValidatorResult` and `ImageFinding` (placeholder for
no_match) in the pipeline output — full auditability.

**User history never determines truth.** Stage 5 (FinalDecision) is
structurally isolated from user history data. History signals are applied
only in Stage 6 as risk flags and cannot change `claim_status`.

**Deterministic gates use no LLM quota.** Stages 4, 6, and 7 are pure
Python — no API calls. Stage 2 gates Stage 3, so no_match images never
consume expensive reasoning calls.

**Everything is cacheable.** The LLM client caches all responses to
`.llm_cache/` by SHA-256(model + prompt + image_path). Re-running the
pipeline on the same data costs zero additional API quota.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key |
