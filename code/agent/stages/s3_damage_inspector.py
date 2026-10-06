"""
agent/stages/s3_damage_inspector.py — Stage 3: DamageInspector

Core vision reasoning stage.  For every image that passed Stage 2 (match
or uncertain), the model follows a 4-step chain-of-thought:
  LOCATE → OBSERVE → CHALLENGE → CONCLUDE

All intermediate reasoning fields are stored in the ImageFinding dataclass
so they are available for debugging, evaluation, and traceability.

Images that did NOT pass Stage 2 (no_match) receive a placeholder finding
so every submitted image remains represented in downstream stages.
"""

from pathlib import Path

from agent import config
from agent.llm_client import LLMClient
from agent.models import ClaimExtraction, ImageFinding
from agent.prompts import DAMAGE_INSPECTOR_PROMPT


def inspect_image(
    image_path: str | Path,
    image_id: str,
    claim_object: str,
    claim: ClaimExtraction,
    client: LLMClient,
) -> ImageFinding:
    """Run Stage 3 for a single image (match or uncertain only).

    Args:
        image_path:   Absolute path to the image file.
        image_id:     Filename stem (e.g. "img_1").
        claim_object: One of "car", "laptop", "package".
        claim:        ClaimExtraction from Stage 1.
        client:       Shared LLMClient instance.

    Returns:
        ImageFinding with all CoT reasoning fields populated.
    """
    allowed_parts = sorted(config.ALLOWED_PARTS.get(claim_object, {"unknown"}))
    allowed_issue_types = sorted(config.ALLOWED_ISSUE_TYPES)

    prompt = DAMAGE_INSPECTOR_PROMPT.format(
        claim_object=claim_object,
        claimed_issue_description=claim.claimed_issue_description,
        claimed_parts=", ".join(claim.claimed_parts) if claim.claimed_parts else "unspecified",
        allowed_parts=", ".join(allowed_parts),
        allowed_issue_types=", ".join(allowed_issue_types),
    )

    try:
        raw = client.call_with_image(
            model=config.STAGE3_MODEL,
            prompt=prompt,
            image_path=image_path,
        )
        data = client.parse_json(raw)

        object_part = str(data.get("object_part", "unknown"))
        issue_type  = str(data.get("issue_type", "unknown"))

        # Validate part is in the allowed set
        if object_part not in config.ALLOWED_PARTS.get(claim_object, {"unknown"}):
            object_part = "unknown"

        # Enforce part ↔ issue compatibility
        issue_type = validate_issue_for_part(issue_type, object_part)

        return ImageFinding(
            image_id=image_id,
            object_part=object_part,
            issue_type=issue_type,
            issue_visible=bool(data.get("issue_visible", False)),
            image_quality_ok=bool(data.get("image_quality_ok", True)),
            quality_risk_flags=list(data.get("quality_risk_flags", [])),
            matches_claim_object=bool(data.get("matches_claim_object", True)),
            possible_manipulation=bool(data.get("possible_manipulation", False)),
            contains_instruction_text=bool(data.get("contains_instruction_text", False)),
            location_reasoning=str(data.get("location_reasoning", "")),
            observation_reasoning=str(data.get("observation_reasoning", "")),
            challenge_reasoning=str(data.get("challenge_reasoning", "")),
            raw_notes=str(data.get("raw_notes", "")),
            skipped=False,
        )

    except Exception as exc:  # noqa: BLE001
        print(f"[Stage3] Failed for {image_id}: {exc}. Returning degraded finding.")
        return _degraded_finding(image_id, reason=str(exc))


def make_no_match_placeholder(image_id: str) -> ImageFinding:
    """Create a placeholder finding for an image that failed Stage 2 (no_match).

    The image is still represented in all downstream stages for full
    auditability — it just does not contribute positive evidence.
    """
    return ImageFinding(
        image_id=image_id,
        object_part="unknown",
        issue_type="unknown",
        issue_visible=False,
        image_quality_ok=False,
        quality_risk_flags=["wrong_object"],
        matches_claim_object=False,
        possible_manipulation=False,
        contains_instruction_text=False,
        location_reasoning="Image skipped — Stage 2 classified as no_match.",
        observation_reasoning="",
        challenge_reasoning="",
        raw_notes="no_match placeholder; Stage 3 was not run on this image.",
        skipped=True,
    )


def validate_issue_for_part(issue_type: str, object_part: str) -> str:
    """Enforce that issue_type is physically plausible for the located part.

    If the model returns an incompatible pair, fall back to "unknown"
    rather than writing a nonsensical combination to output.csv.
    """
    allowed = config.PART_ISSUE_COMPATIBILITY.get(object_part, {"unknown", "none"})
    if issue_type not in allowed:
        return "unknown"
    return issue_type


def _degraded_finding(image_id: str, reason: str) -> ImageFinding:
    return ImageFinding(
        image_id=image_id,
        object_part="unknown",
        issue_type="unknown",
        issue_visible=False,
        image_quality_ok=False,
        quality_risk_flags=["damage_not_visible"],
        matches_claim_object=False,
        possible_manipulation=False,
        contains_instruction_text=False,
        location_reasoning=f"Stage 3 failed: {reason}",
        observation_reasoning="",
        challenge_reasoning="",
        raw_notes=f"error: {reason}",
        skipped=True,
    )
