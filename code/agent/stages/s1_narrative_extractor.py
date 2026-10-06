"""
agent/stages/s1_narrative_extractor.py — Stage 1 (FNOL)
Extracts structured FNOL fields from a raw accident narrative.
Original s1_claim_extraction.py is unchanged and still used by main.py.
"""

from agent import config
from agent.llm_client import LLMClient
from agent.models import FNOLField, FNOLRecord

PROMPT = """You are an insurance intake specialist. Extract FNOL fields from this accident narrative.

Narrative:
{narrative}

Return ONLY a JSON object with these exact keys. For each key, return a string value or null if not mentioned:
policy_number, insured_name, reporter_name, reporter_relationship, accident_date, accident_time,
accident_location, vehicle_make, vehicle_model, vehicle_year, other_party_name, other_party_vehicle,
other_party_insurance, damage_description, damage_severity, injury_indicator, injury_description,
police_report_indicator, police_report_number, towing_required, towing_company

Rules:
- injury_indicator: only "yes", "no", or null. NEVER default to "no" unless the narrative explicitly states no injuries.
- police_report_indicator: only "yes", "no", or null
- damage_severity: only "minor", "moderate", "severe", or null
- reporter_relationship: only "policyholder", "spouse", "witness", "attorney", "other", or null
- towing_required: only "yes", "no", or null
- Return null for any field not clearly mentioned in the narrative
- Return ONLY the JSON object. No explanation, no markdown, no prose."""


def _make_field(value, evidence_ref: str = "") -> FNOLField:
    if value is None:
        return FNOLField(
            value=None, source="narrative",
            confidence=0.0, status="missing", evidence_ref=""
        )
    confidence = 0.85
    status = "confirmed" if confidence >= config.FNOL_UNCERTAIN_THRESHOLD else "uncertain"
    return FNOLField(
        value=str(value), source="narrative",
        confidence=confidence, status=status,
        evidence_ref=evidence_ref[:120]
    )


def extract_fnol(narrative: str, client: LLMClient) -> FNOLRecord:
    """Run Stage 1 FNOL: extract all fields from narrative text."""
    prompt = PROMPT.format(narrative=narrative)
    raw = client.call_text(model=config.FNOL_TEXT_MODEL, prompt=prompt)

    try:
        data = client.parse_json(raw)
    except ValueError:
        data = {}

    ref = narrative[:120]

    def f(key: str) -> FNOLField:
        return _make_field(data.get(key), ref)

    return FNOLRecord(
        policy_number=f("policy_number"),
        insured_name=f("insured_name"),
        reporter_name=f("reporter_name"),
        reporter_relationship=f("reporter_relationship"),
        accident_date=f("accident_date"),
        accident_time=f("accident_time"),
        accident_location=f("accident_location"),
        vehicle_make=f("vehicle_make"),
        vehicle_model=f("vehicle_model"),
        vehicle_year=f("vehicle_year"),
        other_party_name=f("other_party_name"),
        other_party_vehicle=f("other_party_vehicle"),
        other_party_insurance=f("other_party_insurance"),
        damage_description=f("damage_description"),
        damage_severity=f("damage_severity"),
        injury_indicator=f("injury_indicator"),
        injury_description=f("injury_description"),
        police_report_indicator=f("police_report_indicator"),
        police_report_number=f("police_report_number"),
        towing_required=f("towing_required"),
        towing_company=f("towing_company"),
        raw_narrative=narrative,
    )