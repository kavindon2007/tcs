"""
agent/stages/s6_risk_overlay.py — Stage 6: Risk Overlay

Entirely deterministic — no LLM calls.  Assembles risk_flags and
valid_image from four independent sources:

  1. Stage 2 (ObjectValidator) results
  2. Stage 3 (DamageInspector) findings
  3. Stage 4 (evidence gate) output
  4. User history row from user_history.csv

CRITICAL invariant: this stage cannot modify claim_status.  That field
is set in Stage 5 and is structurally absent from this stage's outputs.
The separation is enforced by the return type (RiskOverlayResult) which
contains only risk_flags and valid_image.
"""

from agent import config
from agent.models import (
    ClaimExtraction,
    DecisionResult,
    ImageFinding,
    ObjectValidatorResult,
    RiskOverlayResult,
)


def apply_risk_overlay(
    validator_results: list[ObjectValidatorResult],
    findings: list[ImageFinding],
    evidence_standard_met: bool,
    claim: ClaimExtraction,
    decision: DecisionResult,
    user_history: dict | None,
) -> RiskOverlayResult:
    """Assemble risk_flags and valid_image from all pipeline signals.

    Args:
        validator_results:    All Stage 2 results (one per image).
        findings:             All Stage 3 ImageFindings (incl. placeholders).
        evidence_standard_met: Stage 4 output.
        claim:                ClaimExtraction from Stage 1.
        decision:             DecisionResult from Stage 5.
        user_history:         Row from user_history.csv for this user_id,
                              or None if not found.

    Returns:
        RiskOverlayResult with risk_flags list and valid_image bool.
    """
    flags: set[str] = set()

    # ------------------------------------------------------------------
    # Source 1: Stage 2 — ObjectValidator
    # ------------------------------------------------------------------
    for vr in validator_results:
        if vr.match_class == "no_match":
            flags.add("wrong_object")
        if vr.contains_instruction_text:
            flags.add("text_instruction_present")
        if vr.match_class == "uncertain":
            flags.add("manual_review_required")

    # ------------------------------------------------------------------
    # Source 2: Stage 3 — DamageInspector findings
    # ------------------------------------------------------------------
    for f in findings:
        if f.skipped:
            continue
        if f.possible_manipulation:
            flags.add("possible_manipulation")
        if f.contains_instruction_text:
            flags.add("text_instruction_present")
        for qflag in f.quality_risk_flags:
            # Only pass through flags that are in the allowed set
            if qflag in config.ALLOWED_RISK_FLAGS:
                flags.add(qflag)
        if not f.matches_claim_object and not f.skipped:
            flags.add("wrong_object")
        if f.image_quality_ok is False and not f.skipped:
            # generic quality degradation already covered by quality_risk_flags,
            # but ensure at least one flag is present if quality is bad
            if not any(
                qf in flags
                for qf in {"blurry_image", "cropped_or_obstructed",
                           "low_light_or_glare", "wrong_angle"}
            ):
                flags.add("damage_not_visible")

    # ------------------------------------------------------------------
    # Source 3: Stage 4 — evidence gate
    # ------------------------------------------------------------------
    if not evidence_standard_met:
        # Part not visible → claim mismatch signal
        if claim.claimed_parts and not any(
            f.object_part in claim.claimed_parts
            for f in findings
            if not f.skipped and f.image_quality_ok
        ):
            flags.add("claim_mismatch")

    # ------------------------------------------------------------------
    # Source 4: User history
    # ------------------------------------------------------------------
    if user_history:
        _apply_history_flags(flags, user_history)

    # ------------------------------------------------------------------
    # valid_image: at least one image is quality-ok and matches the object
    # ------------------------------------------------------------------
    valid_image = any(
        f.image_quality_ok and f.matches_claim_object and not f.skipped
        for f in findings
    )

    # Normalise: remove "none" if any real flag is present; add if empty
    flags.discard("none")
    if not flags:
        flags.add("none")

    # Final filter: only emit allowed risk flags
    flags = {f for f in flags if f in config.ALLOWED_RISK_FLAGS}
    if not flags:
        flags.add("none")

    return RiskOverlayResult(
        risk_flags=sorted(flags),
        valid_image=valid_image,
    )


def _apply_history_flags(flags: set[str], history: dict) -> None:
    """Add user-history-derived risk flags.

    Adds user_history_risk when the user has a problematic history pattern.
    Adds manual_review_required when prior claims needed manual review.
    Never modifies claim_status.
    """
    try:
        rejected       = int(history.get("rejected_claim", 0))
        manual_review  = int(history.get("manual_review_claim", 0))
        last_90        = int(history.get("last_90_days_claim_count", 0))
        history_flags  = str(history.get("history_flags", "")).lower()
    except (ValueError, TypeError):
        return

    if rejected > 0 or last_90 >= 3 or "high_risk" in history_flags or "fraud" in history_flags:
        flags.add("user_history_risk")

    if manual_review > 0:
        flags.add("manual_review_required")
