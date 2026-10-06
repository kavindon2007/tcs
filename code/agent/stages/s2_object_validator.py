"""
agent/stages/s2_object_validator.py — Stage 2: ObjectValidator

Lightweight per-image pre-check: does this image plausibly contain the
claimed object?  The model returns a confidence float; this module converts
it to match / uncertain / no_match using config thresholds.  No image is
silently dropped — even no_match results are recorded for full auditability.
"""

from pathlib import Path

from agent import config
from agent.llm_client import LLMClient
from agent.models import ObjectValidatorResult
from agent.prompts import OBJECT_VALIDATOR_PROMPT


def validate_object(
    image_path: str | Path,
    image_id: str,
    claim_object: str,
    client: LLMClient,
) -> ObjectValidatorResult:
    """Run Stage 2 for a single image.

    Args:
        image_path:   Absolute path to the image file.
        image_id:     Filename stem (e.g. "img_1") — used as a stable key.
        claim_object: One of "car", "laptop", "package".
        client:       Shared LLMClient instance.

    Returns:
        ObjectValidatorResult with match_class derived from confidence.
    """
    prompt = OBJECT_VALIDATOR_PROMPT.format(claim_object=claim_object)

    try:
        raw = client.call_with_image(
            model=config.STAGE2_MODEL,
            prompt=prompt,
            image_path=image_path,
        )
        data = client.parse_json(raw)

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        return ObjectValidatorResult(
            image_id=image_id,
            confidence=confidence,
            match_class=classify_object_match(confidence),
            contains_instruction_text=bool(data.get("contains_instruction_text", False)),
            notes=str(data.get("notes", "")),
        )

    except Exception as exc:  # noqa: BLE001
        print(f"[Stage2] Failed for {image_id}: {exc}. Marking uncertain.")
        return ObjectValidatorResult(
            image_id=image_id,
            confidence=0.5,
            match_class="uncertain",
            contains_instruction_text=False,
            notes=f"stage2 error: {exc}",
        )


def classify_object_match(confidence: float) -> str:
    """Convert a raw confidence score to a categorical match class.

    Uses thresholds defined in config so they can be tuned without
    touching this function.

    Returns:
        "match"     if confidence >= OBJECT_VALIDATOR_MATCH_THRESHOLD
        "no_match"  if confidence <= OBJECT_VALIDATOR_NOMATCH_THRESHOLD
        "uncertain" otherwise — passes to Stage 3 with uncertainty flag
    """
    if confidence >= config.OBJECT_VALIDATOR_MATCH_THRESHOLD:
        return "match"
    elif confidence <= config.OBJECT_VALIDATOR_NOMATCH_THRESHOLD:
        return "no_match"
    return "uncertain"
