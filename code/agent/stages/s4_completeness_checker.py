"""
agent/stages/s4_completeness_checker.py — Stage 4 (FNOL)
Deterministic completeness check. No LLM calls.
Original s4_evidence_gate.py is unchanged and still used by main.py.
"""

from agent import config
from agent.models import FNOLRecord

FIELD_QUESTIONS = {
    "policy_number":           "What is your policy number?",
    "injury_indicator":        "Were there any injuries? (yes / no / unknown)",
    "police_report_indicator": "Was a police report filed? (yes / no / unknown)",
    "accident_date":           "What was the date of the accident?",
    "accident_time":           "Approximately what time did the accident occur?",
    "accident_location":       "Where did the accident happen? (street, city, or landmark)",
    "vehicle_make":            "What is the make of your vehicle? (e.g. Toyota, Ford)",
    "vehicle_model":           "What is the model of your vehicle? (e.g. Camry, F-150)",
}


def check_completeness(fnol: FNOLRecord) -> list[str]:
    """Return list of missing mandatory field names in priority order. No LLM."""
    missing = []
    for field_name in config.FNOL_MANDATORY_FIELDS:
        field = getattr(fnol, field_name)
        if field.status == "missing" or field.value is None:
            missing.append(field_name)
    return missing


def question_for(field_name: str) -> str:
    """Return the clarification question for a given field name."""
    return FIELD_QUESTIONS.get(
        field_name,
        f"Could you provide details about: {field_name.replace('_', ' ')}?"
    )