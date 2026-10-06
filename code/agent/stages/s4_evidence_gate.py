"""
agent/stages/s4_evidence_gate.py — Stage 4: Evidence Sufficiency Gate

Entirely deterministic — no LLM calls.  Checks whether the submitted
image set meets the minimum evidence standard before allowing the expensive
Stage 5 LLM call to proceed.

This is a hard gate: its result is passed as a fact to Stage 5.  Stage 5
cannot override it, only reason about it.
"""

import re

from agent.models import ClaimExtraction, ImageFinding


def check_evidence_sufficiency(
    claim: ClaimExtraction,
    findings: list[ImageFinding],
    object_validator_classes: dict[str, str],  # image_id → "match"/"uncertain"/"no_match"
    requirement: dict | None,
) -> tuple[bool, str]:
    """Hard evidence sufficiency gate.

    Args:
        claim:                    ClaimExtraction from Stage 1.
        findings:                 All ImageFindings (including placeholders).
        object_validator_classes: Stage 2 match_class per image_id.
        requirement:              Matching row from evidence_requirements.csv,
                                  or None if no row matched.

    Returns:
        (evidence_standard_met: bool, reason: str)

    A finding is considered "usable" only when:
      - Stage 2 classified the image as "match" (not uncertain or no_match)
      - Stage 3 found image_quality_ok=True
    Uncertain images are passed to Stage 3 for completeness, but they are
    treated conservatively in the evidence gate.
    """
    if requirement is None:
        return False, "no matching evidence requirement found for this claim_object/issue combination"

    # Usable = Stage 2 match AND Stage 3 quality OK
    usable: list[ImageFinding] = [
        f for f in findings
        if object_validator_classes.get(f.image_id) == "match"
        and f.image_quality_ok
        and not f.skipped
    ]

    # Minimum image count from free-text requirement
    min_required = parse_min_image_count(requirement.get("minimum_image_evidence", ""))
    if len(usable) < min_required:
        return False, (
            f"only {len(usable)} usable image(s) meet quality and object-match "
            f"requirements; {min_required} required "
            f"(after excluding wrong-object or low-quality images)"
        )

    # The claimed part must be visible in at least one usable image
    if claim.claimed_parts:
        part_visible = any(
            f.object_part in claim.claimed_parts
            for f in usable
        )
        if not part_visible:
            return False, (
                f"claimed part(s) {claim.claimed_parts} not clearly visible "
                f"in any usable image"
            )

    return True, "minimum image count and claimed-part visibility requirements met"


def parse_min_image_count(minimum_image_evidence: str) -> int:
    """Derive a numeric minimum image count from the free-text evidence requirement.

    Default is 1.  Bumps to 2 only when the text explicitly signals that
    multiple images are required (e.g. "at least two", "two or more",
    "multiple images required").

    The current evidence_requirements.csv does not trigger the bump —
    every row resolves to 1 — but this function is correct for future
    additions to that file.
    """
    text = minimum_image_evidence.lower()
    if re.search(
        r"\bat least (two|2)\b|two or more|2\s*\+|multiple images",
        text,
    ):
        return 2
    return 1


def find_requirement(
    claim_object: str,
    issue_type: str,
    evidence_requirements: list[dict],
) -> dict | None:
    """Find the most specific evidence requirement for this claim.

    Matching strategy:
    1. Look for a row whose claim_object matches (or is "all") AND whose
       applies_to text contains keywords from the issue family.
    2. Fall back to the generic "general claim review" row for this object.
    3. Fall back to the "all" / "general claim review" row.
    4. Return None if nothing found.
    """
    from agent.config import ISSUE_TO_REQUIREMENT_FAMILY  # avoid circular at module level

    target_family = ISSUE_TO_REQUIREMENT_FAMILY.get(issue_type, "general claim review")

    # Object-specific rows first, then "all" rows
    object_rows = [
        r for r in evidence_requirements
        if r.get("claim_object") == claim_object
    ]
    all_rows = [
        r for r in evidence_requirements
        if r.get("claim_object") == "all"
    ]

    for rows in (object_rows, all_rows):
        for row in rows:
            applies_to = row.get("applies_to", "").lower()
            if any(kw in applies_to for kw in target_family.lower().split(" or ")):
                return row

    # Fallback: generic object row
    for rows in (object_rows, all_rows):
        for row in rows:
            if "general" in row.get("applies_to", "").lower():
                return row

    return None
