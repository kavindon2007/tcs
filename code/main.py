"""
code/main.py — Entry point for the damage-claim verification pipeline.

Reads dataset/claims.csv, runs all 7 pipeline stages per claim, and
writes dataset/output.csv.

Usage:
    python code/main.py [--claims PATH] [--output PATH] [--no-cache]

Environment:
    GOOGLE_API_KEY  — required; Gemini API key
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Allow running from repo root: python code/main.py
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from agent import config
from agent.llm_client import LLMClient
from agent.models import OutputRow
from agent.stages.s1_claim_extraction import extract_claim
from agent.stages.s2_object_validator import validate_object
from agent.stages.s3_damage_inspector import inspect_image, make_no_match_placeholder
from agent.stages.s4_evidence_gate import check_evidence_sufficiency, find_requirement
from agent.stages.s5_final_decision import make_final_decision
from agent.stages.s6_risk_overlay import apply_risk_overlay
from agent.stages.s7_output_validator import build_output_row


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_id_from_path(image_path: str) -> str:
    """Extract the image ID (filename stem) from a relative image path.

    e.g. "images/test/case_001/img_1.jpg" → "img_1"
    """
    return Path(image_path).stem


def resolve_image_path(image_path: str, dataset_dir: Path) -> Path:
    """Resolve a relative image path from the CSV to an absolute path."""
    return dataset_dir / image_path


def process_claim(
    row: dict,
    user_history_map: dict[str, dict],
    evidence_requirements: list[dict],
    client: LLMClient,
    dataset_dir: Path,
) -> OutputRow:
    """Run all 7 pipeline stages for one claims.csv row.

    Args:
        row:                  One row from claims.csv.
        user_history_map:     user_id → history dict from user_history.csv.
        evidence_requirements: All rows from evidence_requirements.csv.
        client:               Shared LLMClient instance.
        dataset_dir:          Absolute path to the dataset/ directory.

    Returns:
        Validated OutputRow ready to write to output.csv.
    """
    user_id      = row["user_id"]
    user_claim   = row["user_claim"]
    claim_object = row["claim_object"]
    raw_paths    = row["image_paths"]

    image_paths_rel = [p.strip() for p in raw_paths.split(";") if p.strip()]
    image_ids       = [image_id_from_path(p) for p in image_paths_rel]
    image_abs_paths = [resolve_image_path(p, dataset_dir) for p in image_paths_rel]

    print(f"\n[{user_id}] Processing {len(image_ids)} image(s) | object={claim_object}")

    # ── Stage 1: ClaimExtraction ──────────────────────────────────────
    print(f"  [S1] Extracting claim…")
    claim = extract_claim(user_claim, claim_object, client)
    print(f"  [S1] parts={claim.claimed_parts} | ambiguous={claim.ambiguous_or_vague}")

    # ── Stage 2: ObjectValidator (sequential — free-tier RPM) ───────────────
    print(f"  [S2] Validating object identity…")
    validator_results = [
        validate_object(image_abs_paths[i], image_ids[i], claim_object, client)
        for i in range(len(image_ids))
    ]
    # Key by image_id
    validator_map = {vr.image_id: vr for vr in validator_results}
    for vr in validator_results:
        print(f"  [S2]   {vr.image_id}: {vr.match_class} (conf={vr.confidence:.2f})")

    # ── Stage 3: DamageInspector (sequential, match+uncertain only) ─────
    print(f"  [S3] Inspecting damage…")
    stage3_results = [
        inspect_image(image_abs_paths[i], image_ids[i], claim_object, claim, client)
        for i in range(len(image_ids))
        if validator_map[image_ids[i]].match_class in ("match", "uncertain")
    ]
    findings_map = {f.image_id: f for f in stage3_results}

    # Add placeholders for no_match images (auditability)
    all_findings = []
    for img_id in image_ids:
        if img_id in findings_map:
            all_findings.append(findings_map[img_id])
        else:
            all_findings.append(make_no_match_placeholder(img_id))
            print(f"  [S3]   {img_id}: skipped (no_match placeholder)")

    for f in stage3_results:
        print(f"  [S3]   {f.image_id}: part={f.object_part} | issue={f.issue_type} | visible={f.issue_visible}")

    # ── Stage 4: Evidence gate (deterministic) ────────────────────────
    print(f"  [S4] Evidence gate…")
    # Use the consensus issue_type from findings for requirement lookup
    consensus_issue = _consensus_issue(all_findings)
    requirement = find_requirement(claim_object, consensus_issue, evidence_requirements)
    evidence_met, evidence_reason = check_evidence_sufficiency(
        claim=claim,
        findings=all_findings,
        object_validator_classes={vr.image_id: vr.match_class for vr in validator_results},
        requirement=requirement,
    )
    print(f"  [S4] evidence_standard_met={evidence_met}: {evidence_reason}")

    # ── Stage 5: FinalDecision (LLM, text-only) ───────────────────────
    print(f"  [S5] Final decision…")
    decision = make_final_decision(
        claim=claim,
        claim_object=claim_object,
        findings=all_findings,
        evidence_standard_met=evidence_met,
        evidence_standard_met_reason=evidence_reason,
        client=client,
    )
    print(f"  [S5] claim_status={decision.claim_status} | severity={decision.severity}")

    # ── Stage 6: Risk overlay (deterministic) ─────────────────────────
    user_history = user_history_map.get(user_id)
    risk = apply_risk_overlay(
        validator_results=validator_results,
        findings=all_findings,
        evidence_standard_met=evidence_met,
        claim=claim,
        decision=decision,
        user_history=user_history,
    )
    print(f"  [S6] risk_flags={risk.risk_flags} | valid_image={risk.valid_image}")

    # ── Stage 7: Output validation ────────────────────────────────────
    output_row = build_output_row(
        user_id=user_id,
        image_paths=raw_paths,
        user_claim=user_claim,
        claim_object=claim_object,
        decision=decision,
        risk=risk,
    )

    return output_row


def _consensus_issue(findings: list) -> str:
    """Pick the most common non-unknown issue_type from Stage 3 findings."""
    from collections import Counter
    counts = Counter(
        f.issue_type for f in findings
        if not f.skipped and f.issue_type not in ("unknown", "none")
    )
    if counts:
        return counts.most_common(1)[0][0]
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Damage-claim verification pipeline")
    parser.add_argument(
        "--claims",
        default=str(config.DATASET_DIR / "claims.csv"),
        help="Path to claims CSV (default: dataset/claims.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(config.DATASET_DIR / "output.csv"),
        help="Path to write output CSV (default: dataset/output.csv)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N claims (useful for single-claim testing)",
    )
    args = parser.parse_args()

    claims_path = Path(args.claims)
    output_path = Path(args.output)
    dataset_dir = claims_path.parent  # images are relative to dataset/

    print(f"Loading claims from {claims_path}")
    claims     = load_csv(claims_path)
    history    = load_csv(config.DATASET_DIR / "user_history.csv")
    reqs       = load_csv(config.DATASET_DIR / "evidence_requirements.csv")

    history_map = {row["user_id"]: row for row in history}

    client = LLMClient(cache=not args.no_cache)

    if args.limit:
        claims = claims[: args.limit]
        print(f"[--limit] Processing first {args.limit} claim(s) only.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.OUTPUT_COLUMNS)
        writer.writeheader()
        f.flush()

        output_rows: list[OutputRow] = []
        for i, row in enumerate(claims):
            print(f"\n{'='*60}")
            print(f"Claim {i+1}/{len(claims)}: user_id={row['user_id']}")
            try:
                output_row = process_claim(
                    row=row,
                    user_history_map=history_map,
                    evidence_requirements=reqs,
                    client=client,
                    dataset_dir=dataset_dir,
                )
                output_rows.append(output_row)
                writer.writerow(output_row.to_dict())
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Claim {row['user_id']} failed: {exc}. Writing error row.")
                err_row = _error_row(row)
                output_rows.append(err_row)
                writer.writerow(err_row.to_dict())
            
            # Force write to disk immediately after each claim
            f.flush()
            os.fsync(f.fileno())

        print(f"\n{'='*60}")
        print(f"Done. {len(output_rows)} rows written to {output_path}")


def _error_row(row: dict) -> OutputRow:
    """Fallback row when claim processing fails entirely."""
    return OutputRow(
        user_id=row.get("user_id", "unknown"),
        image_paths=row.get("image_paths", ""),
        user_claim=row.get("user_claim", ""),
        claim_object=row.get("claim_object", "unknown"),
        evidence_standard_met="false",
        evidence_standard_met_reason="pipeline error",
        risk_flags="manual_review_required",
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification="pipeline processing error",
        supporting_image_ids="none",
        valid_image="false",
        severity="unknown",
    )


if __name__ == "__main__":
    main()
