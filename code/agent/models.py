"""
agent/models.py — Dataclasses for every pipeline stage's input/output.
These are the only structures passed between stages; no stage receives raw
dicts from a sibling stage.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Stage 1 output
# ---------------------------------------------------------------------------
@dataclass
class ClaimExtraction:
    """Structured representation of what the customer is claiming.

    Produced by Stage 1 (text-only LLM). Consumed by Stages 3, 4, 5.
    ambiguous_or_vague is an internal signal — it influences Stage 5's
    willingness to return not_enough_information but is never written to
    output.csv directly.
    """
    claimed_issue_description: str
    claimed_parts: list[str] = field(default_factory=list)
    claimed_issue_keywords: list[str] = field(default_factory=list)
    ambiguous_or_vague: bool = False


# ---------------------------------------------------------------------------
# Stage 2 output (per image)
# ---------------------------------------------------------------------------
@dataclass
class ObjectValidatorResult:
    """Object-identity pre-check for a single image.

    Produced by Stage 2 (lightweight VLM). Never silently discarded —
    even no_match images are recorded for auditability.
    match_class is derived in code from confidence using config thresholds;
    the model only returns the raw float.
    """
    image_id: str
    confidence: float
    match_class: str          # "match" | "uncertain" | "no_match"
    contains_instruction_text: bool
    notes: str


# ---------------------------------------------------------------------------
# Stage 3 output (per image)
# ---------------------------------------------------------------------------
@dataclass
class ImageFinding:
    """Detailed per-image damage inspection result.

    Produced by Stage 3 (VLM, 4-step CoT). Only run on match + uncertain
    images. For no_match images a placeholder finding is created so every
    image remains traceable.
    All intermediate reasoning fields are stored — not discarded — so they
    can be surfaced in evaluation and debugging.
    """
    image_id: str
    object_part: str
    issue_type: str
    issue_visible: bool
    image_quality_ok: bool
    quality_risk_flags: list[str] = field(default_factory=list)
    matches_claim_object: bool = True
    possible_manipulation: bool = False
    contains_instruction_text: bool = False
    location_reasoning: str = ""
    observation_reasoning: str = ""
    challenge_reasoning: str = ""
    raw_notes: str = ""
    # Set to True for no_match placeholder findings (skipped Stage 3)
    skipped: bool = False


# ---------------------------------------------------------------------------
# Stage 4 output (deterministic)
# ---------------------------------------------------------------------------
@dataclass
class EvidenceGateResult:
    """Hard-gate evidence sufficiency check.

    Produced entirely in code (no LLM). Passed as context to Stage 5.
    Stage 5 cannot override this — it receives the result as a fact.
    """
    evidence_standard_met: bool
    reason: str


# ---------------------------------------------------------------------------
# Stage 5 output
# ---------------------------------------------------------------------------
@dataclass
class DecisionResult:
    """Final claim decision from the LLM decision layer.

    Produced by Stage 5 (text-only LLM; receives serialised ImageFinding
    list, NOT raw images). User history is structurally absent from Stage 5
    inputs — it is applied only in Stage 6.
    """
    support_reasoning: str
    contradiction_reasoning: str
    evidence_standard_weighing: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    issue_type: str
    object_part: str
    claim_status: str                        # supported | contradicted | not_enough_information
    claim_status_justification: str
    supporting_image_ids: list[str] = field(default_factory=list)
    severity: str = "unknown"


# ---------------------------------------------------------------------------
# Stage 6 output (deterministic)
# ---------------------------------------------------------------------------
@dataclass
class RiskOverlayResult:
    """Risk context added after the claim decision is made.

    Produced entirely in code. Cannot modify claim_status — that is
    structurally enforced by never passing claim_status as an output field
    of this stage.
    """
    risk_flags: list[str] = field(default_factory=list)
    valid_image: bool = False


# ---------------------------------------------------------------------------
# Final assembled output row (after Stage 7 validation)
# ---------------------------------------------------------------------------
@dataclass
class OutputRow:
    """One row of output.csv. Field order matches OUTPUT_COLUMNS in config."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str      # "true" / "false"
    evidence_standard_met_reason: str
    risk_flags: str                 # semicolon-separated or "none"
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str       # semicolon-separated or "none"
    valid_image: str                # "true" / "false"
    severity: str

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
            "evidence_standard_met": self.evidence_standard_met,
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": self.risk_flags,
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": self.supporting_image_ids,
            "valid_image": self.valid_image,
            "severity": self.severity,
        }

# ---------------------------------------------------------------------------
# FNOL — Auto Insurance models
# ---------------------------------------------------------------------------
from typing import Optional


@dataclass
class FNOLField:
    """A single FNOL field with full provenance."""
    value: Optional[str]
    source: str           # "narrative" | "image" | "user_correction"
    confidence: float     # 0.0-1.0
    status: str           # "confirmed" | "uncertain" | "missing"
    evidence_ref: str     # exact user quote or image_id


@dataclass
class FNOLRecord:
    """Complete FNOL record. Every field carries provenance."""
    policy_number:            FNOLField
    insured_name:             FNOLField
    reporter_name:            FNOLField
    reporter_relationship:    FNOLField
    accident_date:            FNOLField
    accident_time:            FNOLField
    accident_location:        FNOLField
    vehicle_make:             FNOLField
    vehicle_model:            FNOLField
    vehicle_year:             FNOLField
    other_party_name:         FNOLField
    other_party_vehicle:      FNOLField
    other_party_insurance:    FNOLField
    damage_description:       FNOLField
    damage_severity:          FNOLField
    injury_indicator:         FNOLField
    injury_description:       FNOLField
    police_report_indicator:  FNOLField
    police_report_number:     FNOLField
    towing_required:          FNOLField
    towing_company:           FNOLField
    raw_narrative: str


@dataclass
class ConsistencyFlag:
    """A factual discrepancy between narrative and image — phrased as a neutral question."""
    field: str
    narrative_says: str
    image_shows: str
    question: str


@dataclass
class ClarificationState:
    """Tracks the interactive clarification loop state."""
    fnol: FNOLRecord
    turn_number: int
    session_id: str
    answered_fields: list
    pending_flags: list
    image_findings: list
    confirmed: bool = False


@dataclass
class FNOLDraft:
    """Final assembled output before PDF generation."""
    fnol: FNOLRecord
    consistency_flags: list
    neutral_summary: str
    chronology: list
    document_checklist: list
    image_findings: list