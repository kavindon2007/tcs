"""
agent/stages/s7_output_validator.py — Stage 7: Output Validation

Strict schema enforcement before anything is written to output.csv.
Invalid values are corrected to safe defaults — the pipeline never crashes
or emits garbage because a model returned an unexpected string.

Also enforces cross-field consistency:
  - "supported" / "contradicted" → supporting_image_ids must be non-empty
  - severity != "none"           → issue_type must not be "none"
  - valid_image == False         → evidence_standard_met should be False
  - issue_type compatible with object_part (via PART_ISSUE_COMPATIBILITY)
"""

from agent import config
from agent.models import DecisionResult, OutputRow, RiskOverlayResult
from agent.stages.s3_damage_inspector import validate_issue_for_part


def build_output_row(
    user_id: str,
    image_paths: str,
    user_claim: str,
    claim_object: str,
    decision: DecisionResult,
    risk: RiskOverlayResult,
) -> OutputRow:
    """Validate all fields and assemble the final output row.

    Args:
        user_id, image_paths, user_claim, claim_object: Pass-through input fields.
        decision:  DecisionResult from Stage 5.
        risk:      RiskOverlayResult from Stage 6.

    Returns:
        Validated OutputRow ready to be written to output.csv.
    """
    # --- Per-field validation with safe defaults ---

    claim_status = decision.claim_status
    if claim_status not in config.ALLOWED_CLAIM_STATUS:
        claim_status = "not_enough_information"

    severity = decision.severity
    if severity not in config.ALLOWED_SEVERITY:
        severity = "unknown"

    issue_type = decision.issue_type
    if issue_type not in config.ALLOWED_ISSUE_TYPES:
        issue_type = "unknown"

    object_part = decision.object_part
    if object_part not in config.ALLOWED_PARTS.get(claim_object, {"unknown"}):
        object_part = "unknown"

    # Part ↔ issue compatibility
    issue_type = validate_issue_for_part(issue_type, object_part)

    # Risk flags
    risk_flags = [
        f for f in risk.risk_flags
        if f in config.ALLOWED_RISK_FLAGS
    ]
    if not risk_flags:
        risk_flags = ["none"]

    # Supporting image IDs
    supporting_ids = list(decision.supporting_image_ids)

    # --- Cross-field consistency checks ---

    # Rule: supported/contradicted → must have at least one supporting image
    if claim_status in {"supported", "contradicted"} and not supporting_ids:
        # Demote to not_enough_information rather than emit invalid output
        claim_status = "not_enough_information"

    # Rule: severity != "none" → issue must not be "none"
    if severity != "unknown" and severity != "none" and issue_type == "none":
        issue_type = "unknown"

    # Rule: valid_image=False → evidence_standard_met should be False
    evidence_standard_met = decision.evidence_standard_met
    valid_image = risk.valid_image
    if not valid_image:
        evidence_standard_met = False

    evidence_standard_met_reason = decision.evidence_standard_met_reason or ""

    return OutputRow(
        user_id=user_id,
        image_paths=image_paths,
        user_claim=user_claim,
        claim_object=claim_object,
        evidence_standard_met=str(evidence_standard_met).lower(),
        evidence_standard_met_reason=evidence_standard_met_reason,
        risk_flags=";".join(sorted(set(risk_flags))),
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=decision.claim_status_justification or "",
        supporting_image_ids=";".join(supporting_ids) if supporting_ids else "none",
        valid_image=str(valid_image).lower(),
        severity=severity,
    )
