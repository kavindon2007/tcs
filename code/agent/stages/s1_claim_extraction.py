"""
agent/stages/s1_claim_extraction.py — Stage 1: ClaimExtraction

Converts a raw customer/support conversation into a structured claim object
before any images are seen. This deliberate ordering ensures the model
cannot let visual evidence contaminate its reading of what the customer
actually claimed.
"""

from agent import config
from agent.llm_client import LLMClient
from agent.models import ClaimExtraction
from agent.prompts import CLAIM_EXTRACTION_PROMPT


def extract_claim(
    user_claim: str,
    claim_object: str,
    client: LLMClient,
) -> ClaimExtraction:
    """Run Stage 1: extract structured claim from the conversation.

    Args:
        user_claim:    Raw conversation transcript from claims.csv.
        claim_object:  One of "car", "laptop", "package".
        client:        Shared LLMClient instance.

    Returns:
        ClaimExtraction dataclass.  Falls back to a safe default on parse
        failure so the pipeline can continue with ambiguous_or_vague=True.
    """
    allowed_parts_list = sorted(config.ALLOWED_PARTS.get(claim_object, {"unknown"}))

    prompt = CLAIM_EXTRACTION_PROMPT.format(
        claim_object=claim_object,
        allowed_parts_list=", ".join(allowed_parts_list),
        user_claim=user_claim,
    )

    raw = client.call_text(model=config.STAGE1_MODEL, prompt=prompt)

    try:
        data = client.parse_json(raw)
        return ClaimExtraction(
            claimed_issue_description=str(data.get("claimed_issue_description", "no specific issue described")),
            claimed_parts=_filter_parts(data.get("claimed_parts", []), claim_object),
            claimed_issue_keywords=list(data.get("claimed_issue_keywords", [])),
            ambiguous_or_vague=bool(data.get("ambiguous_or_vague", False)),
        )
    except (ValueError, KeyError, TypeError) as exc:
        print(f"[Stage1] JSON parse failed: {exc}. Returning ambiguous fallback.")
        return ClaimExtraction(
            claimed_issue_description="claim extraction failed; treating as ambiguous",
            claimed_parts=[],
            claimed_issue_keywords=[],
            ambiguous_or_vague=True,
        )


def _filter_parts(raw_parts: list, claim_object: str) -> list[str]:
    """Keep only parts that are in the allowed set for this claim_object."""
    allowed = config.ALLOWED_PARTS.get(claim_object, {"unknown"})
    return [p for p in raw_parts if p in allowed]
