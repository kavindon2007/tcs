"""
code/evaluation/main.py — Evaluation harness for the damage-claim pipeline.

Runs the full pipeline on dataset/sample_claims.csv (which has expected
outputs) and computes per-field accuracy metrics.  Also produces
evaluation/evaluation_report.md with an operational cost/latency analysis.

Usage:
    python code/evaluation/main.py [--no-cache]
"""

import csv
import sys
import time
from pathlib import Path

# Allow running from repo root: python code/evaluation/main.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from agent import config
from agent.llm_client import LLMClient
from main import load_csv, process_claim

import argparse


EVAL_FIELDS = [
    "evidence_standard_met",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "valid_image",
    "severity",
]


def normalise(value: str) -> str:
    """Normalise a field value for comparison (lowercase, stripped)."""
    return str(value).strip().lower()


def flags_match(predicted: str, expected: str) -> bool:
    """Compare semicolon-separated flag sets order-independently."""
    p_set = set(f.strip() for f in predicted.split(";") if f.strip())
    e_set = set(f.strip() for f in expected.split(";") if f.strip())
    return p_set == e_set


def evaluate_row(predicted: dict, expected: dict) -> dict[str, bool]:
    """Return per-field match booleans for one row."""
    results = {}
    for field in EVAL_FIELDS:
        p = normalise(predicted.get(field, ""))
        e = normalise(expected.get(field, ""))
        if field == "risk_flags":
            results[field] = flags_match(p, e)
        else:
            results[field] = (p == e)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline evaluation harness")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    sample_csv   = config.DATASET_DIR / "sample_claims.csv"
    history_csv  = config.DATASET_DIR / "user_history.csv"
    reqs_csv     = config.DATASET_DIR / "evidence_requirements.csv"

    sample_rows  = load_csv(sample_csv)
    history_rows = load_csv(history_csv)
    reqs         = load_csv(reqs_csv)

    history_map  = {row["user_id"]: row for row in history_rows}
    client       = LLMClient(cache=not args.no_cache)

    total_field_hits:  dict[str, int] = {f: 0 for f in EVAL_FIELDS}
    total_rows = len(sample_rows)
    predicted_rows = []

    # Track operational stats
    start_time   = time.time()
    total_images = sum(
        len(r["image_paths"].split(";")) for r in sample_rows
    )

    for i, row in enumerate(sample_rows):
        print(f"\nEval {i+1}/{total_rows}: {row['user_id']}")

        # Strip expected-output columns so process_claim only sees inputs
        input_row = {
            "user_id":      row["user_id"],
            "image_paths":  row["image_paths"],
            "user_claim":   row["user_claim"],
            "claim_object": row["claim_object"],
        }

        try:
            output = process_claim(
                row=input_row,
                user_history_map=history_map,
                evidence_requirements=reqs,
                client=client,
                dataset_dir=config.DATASET_DIR,
            )
            pred_dict = output.to_dict()
        except Exception as exc:  # noqa: BLE001
            print(f"  [EVAL ERROR] {exc}")
            pred_dict = {f: "error" for f in EVAL_FIELDS}

        # Compare predicted vs expected
        field_matches = evaluate_row(pred_dict, row)
        for field, match in field_matches.items():
            if match:
                total_field_hits[field] += 1

        predicted_rows.append({
            "user_id": row["user_id"],
            **pred_dict,
            **{f"expected_{f}": row.get(f, "") for f in EVAL_FIELDS},
            **{f"match_{f}": str(field_matches.get(f, False)).lower() for f in EVAL_FIELDS},
        })

    elapsed = time.time() - start_time

    # Write per-row comparison CSV
    eval_dir = config.REPO_ROOT / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    comparison_path = eval_dir / "sample_comparison.csv"
    if predicted_rows:
        with open(comparison_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=predicted_rows[0].keys())
            writer.writeheader()
            writer.writerows(predicted_rows)

    # Compute claim_status confusion matrix
    claim_status_values = ["supported", "contradicted", "not_enough_information"]
    confusion: dict[str, dict[str, int]] = {
        a: {b: 0 for b in claim_status_values} for a in claim_status_values
    }
    for prow, srow in zip(predicted_rows, sample_rows):
        pred_cs = normalise(prow.get("claim_status", ""))
        exp_cs  = normalise(srow.get("claim_status", ""))
        if pred_cs in confusion and exp_cs in claim_status_values:
            confusion[pred_cs][exp_cs] += 1

    # Write evaluation_report.md
    report_path = eval_dir / "evaluation_report.md"
    _write_report(
        report_path=report_path,
        total_rows=total_rows,
        total_images=total_images,
        elapsed=elapsed,
        total_field_hits=total_field_hits,
        confusion=confusion,
    )

    print(f"\n{'='*60}")
    print("Evaluation complete.")
    print(f"  Rows:   {total_rows}")
    print(f"  Images: {total_images}")
    print(f"  Time:   {elapsed:.1f}s")
    print(f"\nPer-field accuracy:")
    for field in EVAL_FIELDS:
        acc = total_field_hits[field] / total_rows * 100
        print(f"  {field:<30} {acc:.1f}%")
    print(f"\nReports written to {eval_dir}/")


def _write_report(
    report_path: Path,
    total_rows: int,
    total_images: int,
    elapsed: float,
    total_field_hits: dict[str, int],
    confusion: dict,
) -> None:
    """Write evaluation_report.md with accuracy metrics and operational analysis."""
    lines = [
        "# Evaluation Report\n",
        f"Evaluated on `dataset/sample_claims.csv` ({total_rows} rows, {total_images} images).\n",
        "\n## Per-Field Accuracy\n",
        "| Field | Correct | Total | Accuracy |",
        "|---|---|---|---|",
    ]
    for field, hits in total_field_hits.items():
        acc = hits / total_rows * 100 if total_rows else 0
        lines.append(f"| {field} | {hits} | {total_rows} | {acc:.1f}% |")

    lines += [
        "\n## claim_status Confusion Matrix\n",
        "Rows: predicted. Columns: expected.\n",
        "| | supported | contradicted | not_enough_information |",
        "|---|---|---|---|",
    ]
    for pred in ["supported", "contradicted", "not_enough_information"]:
        row_vals = [str(confusion[pred][e]) for e in ["supported", "contradicted", "not_enough_information"]]
        lines.append(f"| {pred} | {' | '.join(row_vals)} |")

    # Operational analysis
    # Approximate LLM calls per claim:
    # Stage 1: 1 text call
    # Stage 2: N image calls (N = images per claim, avg ~2)
    # Stage 3: up to N image calls (match+uncertain only)
    # Stage 5: 1 text call
    # Approx total: ~1 + 2*N + 1 = 2 + 2*N per claim
    avg_images = total_images / total_rows if total_rows else 1
    approx_calls_per_claim = 2 + 2 * avg_images
    total_calls_sample = int(approx_calls_per_claim * total_rows)
    test_rows = 44  # claims.csv row count
    total_calls_test = int(approx_calls_per_claim * test_rows)
    avg_latency = elapsed / total_rows if total_rows else 0

    lines += [
        "\n## Operational Analysis\n",
        f"| Metric | Sample set | Test set (claims.csv) |",
        f"|---|---|---|",
        f"| Rows processed | {total_rows} | 44 |",
        f"| Images processed | {total_images} | ~{int(avg_images * test_rows)} |",
        f"| Approx LLM calls | {total_calls_sample} | ~{total_calls_test} |",
        f"| Elapsed time | {elapsed:.1f}s | ~{avg_latency * test_rows:.0f}s estimated |",
        f"| Avg latency per claim | {avg_latency:.1f}s | — |",
        "",
        "### Model pricing assumptions (Gemini 2.5 Flash, Jun 2026)",
        "- Input tokens:  $0.075 / 1M tokens",
        "- Output tokens: $0.30 / 1M tokens",
        "- Image tokens:  ~800 tokens per image",
        "",
        "### Approximate cost for full test set (44 claims, ~2 images each)",
        "- Stage 1 (text):   44 calls × ~1K tokens  ≈ $0.004",
        "- Stage 2 (image):  ~88 calls × ~900 tokens ≈ $0.006",
        "- Stage 3 (image):  ~80 calls × ~2K tokens  ≈ $0.012",
        "- Stage 5 (text):   44 calls × ~3K tokens   ≈ $0.010",
        "- **Total estimated cost: < $0.05 for the full test set**",
        "",
        "### Rate limits and mitigation",
        "- `MAX_WORKERS=4` limits concurrent requests to respect RPM quotas.",
        "- Exponential backoff (`RETRY_BASE_DELAY=2.0s`, `MAX_RETRIES=3`) handles transient 429s.",
        "- Disk cache keyed by SHA-256(model + prompt + image_path) avoids re-spending",
        "  quota on pipeline re-runs or retries of the same claim.",
        "- Stage 2 (cheap VLM) gates Stage 3 (expensive VLM) — no_match images skip Stage 3.",
        "- Stage 4 and 6 are fully deterministic and consume zero API quota.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluation report written to {report_path}")


if __name__ == "__main__":
    main()
