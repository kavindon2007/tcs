"""
agent/stages/s5_final_decision.py — Stage 5: FinalDecision

The LLM claim decision layer.  Receives structured per-image findings
(serialised JSON) and the evidence gate result — NOT raw images.

Intentional design constraints:
- User history is structurally absent from inputs to this stage.
- The model must independently reason about support AND contradiction
  before choosing a verdict.
- ambiguous_or_vague from Stage 1 is passed in to calibrate confidence.
"""

import json

from agent import config
from agent.llm_client import LLMClient
from agent.models import ClaimExtraction, DecisionResult, ImageFinding
from agent.prompts import FINAL_DECISION_PROMPT


def make_final_decision(
    claim: ClaimExtraction,
    claim_object: str,
    findings: list[ImageFinding],
    evidence_standard_met: bool,
    evidence_standard_met_reason: str,
    client: LLMClient,
) -> DecisionResult:
    """Run Stage 5: LLM claim decision on structured findings.

    Args:
        claim:                     ClaimExtraction from Stage 1.
        claim_object:              One of "car", "laptop", "package".
        findings:                  All ImageFindings (including placeholders).
        evidence_standard_met:     Result from Stage 4 evidence gate.
        evidence_standard_met_reason: Explanation from Stage 4.
        client:                    Shared LLMClient instance.

    Returns:
        DecisionResult.  Falls back to not_enough_information on error.
    """
    # Serialise findings to JSON — the model receives structured text, not images.
    # Exclude full reasoning fields to keep the prompt focused; include key fields.
    findings_payload = [
        {
            "image_id":             f.image_id,
            "object_part":          f.object_part,
            "issue_type":           f.issue_type,
            "issue_visible":        f.issue_visible,
            "image_quality_ok":     f.image_quality_ok,
            "matches_claim_object": f.matches_claim_object,
            "possible_manipulation":f.possible_manipulation,
            "location_reasoning":   f.location_reasoning,
            "observation_reasoning":f.observation_reasoning,
            "challenge_reasoning":  f.challenge_reasoning,
            "skipped":              f.skipped,
        }
        for f in findings
    ]

    prompt = FINAL_DECISION_PROMPT.format(
        claim_object=claim_object,
        claimed_issue_description=claim.claimed_issue_description,
        claimed_parts=", ".join(claim.claimed_parts) if claim.claimed_parts else "unspecified",
        ambiguous_or_vague=str(claim.ambiguous_or_vague).lower(),
        evidence_standard_met=str(evidence_standard_met).lower(),
        evidence_standard_met_reason=evidence_standard_met_reason,
        image_findings_json=json.dumps(findings_payload, indent=2),
    )

    try:
        raw = client.call_text(model=config.STAGE5_MODEL, prompt=prompt)
        data = client.parse_json(raw)

        claim_status = str(data.get("claim_status", "not_enough_information"))
        if claim_status not in config.ALLOWED_CLAIM_STATUS:
            claim_status = "not_enough_information"

        severity = str(data.get("severity", "unknown"))
        if severity not in config.ALLOWED_SEVERITY:
            severity = "unknown"

        object_part = str(data.get("object_part", "unknown"))
        if object_part not in config.ALLOWED_PARTS.get(claim_object, {"unknown"}):
            object_part = "unknown"

        issue_type = str(data.get("issue_type", "unknown"))
        if issue_type not in config.ALLOWED_ISSUE_TYPES:
            issue_type = "unknown"

        supporting_ids = list(data.get("supporting_image_ids", []))

        # Code-level tie-breaking fallback for compound claims.
        # If the model returned a part not in claimed_parts (or "unknown"),
        # resolve the controlling part deterministically from findings.
        object_part, issue_type = _resolve_controlling_part(
            model_part=object_part,
            model_issue=issue_type,
            claim_status=claim_status,
            claim=claim,
            findings=findings,
        )

        return DecisionResult(
            support_reasoning=str(data.get("support_reasoning", "")),
            contradiction_reasoning=str(data.get("contradiction_reasoning", "")),
            evidence_standard_weighing=str(data.get("evidence_standard_weighing", "")),
            evidence_standard_met=bool(data.get("evidence_standard_met", evidence_standard_met)),
            evidence_standard_met_reason=str(
                data.get("evidence_standard_met_reason", evidence_standard_met_reason)
            ),
            issue_type=issue_type,
            object_part=object_part,
            claim_status=claim_status,
            claim_status_justification=str(data.get("claim_status_justification", "")),
            supporting_image_ids=supporting_ids,
            severity=severity,
        )

    except Exception as exc:  # noqa: BLE001
        print(f"[Stage5] Decision call failed: {exc}. Defaulting to not_enough_information.")
        return DecisionResult(
            support_reasoning="",
            contradiction_reasoning="",
            evidence_standard_weighing="",
            evidence_standard_met=evidence_standard_met,
            evidence_standard_met_reason=evidence_standard_met_reason,
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification=f"Stage 5 failed: {exc}",
            supporting_image_ids=[],
            severity="unknown",
        )


def _resolve_controlling_part(
    model_part: str,
    model_issue: str,
    claim_status: str,
    claim: ClaimExtraction,
    findings: list[ImageFinding],
) -> tuple[str, str]:
    """Code-level tie-breaking fallback for compound claims.

    The model is instructed via the prompt to pick the controlling part, but
    if it returns a part that is NOT in claimed_parts (or returns "unknown"),
    this function deterministically resolves the correct part from findings.

    For single-part claims (claimed_parts has 0 or 1 entries) the model's
    output is always in the claimed set, so this function is a no-op and
    the original model output passes through unchanged.

    Resolution priority:
      contradicted  → first usable finding where part ∈ claimed_parts AND
                      issue_visible=False (part visible, no damage = contradiction)
      supported     → first usable finding where part ∈ claimed_parts AND
                      issue_visible=True  (part visible, damage present = support)
      fallback      → first usable finding where part ∈ claimed_parts (any)
      final         → model's original output unchanged
    """
    # If the model already returned a claimed part, trust it — no intervention.
    if model_part in claim.claimed_parts:
        return model_part, model_issue

    # No claimed_parts specified → nothing to resolve against; pass through.
    if not claim.claimed_parts:
        return model_part, model_issue

    # Usable findings: quality ok, matches object, not a placeholder.
    usable = [
        f for f in findings
        if f.image_quality_ok and f.matches_claim_object and not f.skipped
    ]

    if claim_status == "contradicted":
        # Controlling part = visible but showing no issue (the contradiction evidence)
        for f in usable:
            if f.object_part in claim.claimed_parts and not f.issue_visible:
                return f.object_part, f.issue_type

    elif claim_status == "supported":
        # Controlling part = visible and showing the issue (the support evidence)
        for f in usable:
            if f.object_part in claim.claimed_parts and f.issue_visible:
                return f.object_part, f.issue_type

    # Fallback: first usable finding that mentions any claimed part
    for f in usable:
        if f.object_part in claim.claimed_parts:
            return f.object_part, f.issue_type

    # Nothing matched — return model's output unchanged
    return model_part, model_issue

