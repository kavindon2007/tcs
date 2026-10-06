"""
agent/config.py — Central configuration for the damage-claim verification pipeline.
All thresholds, model names, allowed value sets, and compatibility rules live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives at code/agent/config.py → repo root is three levels up
REPO_ROOT = Path(__file__).parent.parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / ".llm_cache"

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# Models — Local Ollama (zero cost, no rate limits)
# llama3.2:latest (2GB) — FASTEST available: 3.8s/call vs mistral-nemo's ~40s
# Handles both text and vision. 10x speed improvement over mistral-nemo.
STAGE1_MODEL = "llama3.2:latest"   # ClaimExtraction (text-only, structured JSON)
STAGE2_MODEL = "llama3.2:latest"   # ObjectValidator (VLM — object identity)
STAGE3_MODEL = "llama3.2:latest"   # DamageInspector (VLM — 4-step CoT)
STAGE5_MODEL = "llama3.2:latest"   # FinalDecision (text-only, synthesis)

# ---------------------------------------------------------------------------
# Stage 2 thresholds
# ---------------------------------------------------------------------------
OBJECT_VALIDATOR_MATCH_THRESHOLD   = 0.7   # ≥ 0.7  → "match"
OBJECT_VALIDATOR_NOMATCH_THRESHOLD = 0.3   # ≤ 0.3  → "no_match"
                                            # middle → "uncertain" (passes to Stage 3)

# ---------------------------------------------------------------------------
# Retry / rate-limit config
# ---------------------------------------------------------------------------
MAX_RETRIES     = 3
RETRY_BASE_DELAY = 2.0   # seconds; doubled on each retry (exponential backoff)

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
# Rate limiting — LOCAL Ollama has NO rate limits
# Set to 0: no inter-call delay needed when running on local hardware
INTER_CALL_DELAY_SECONDS = 0.0   # 0 = disabled; local Ollama has no RPM cap

# ---------------------------------------------------------------------------
# Part ↔ issue compatibility
# Incompatible pairs returned by Stage 3 are corrected to "unknown" before
# being written to any output or passed to Stage 5.
# ---------------------------------------------------------------------------
PART_ISSUE_COMPATIBILITY: dict[str, set[str]] = {
    # --- car ---
    "front_bumper":  {"dent", "scratch", "crack", "broken_part", "missing_part", "none", "unknown"},
    "rear_bumper":   {"dent", "scratch", "crack", "broken_part", "missing_part", "none", "unknown"},
    "door":          {"dent", "scratch", "crack", "broken_part", "none", "unknown"},
    "hood":          {"dent", "scratch", "crack", "none", "unknown"},
    "windshield":    {"crack", "glass_shatter", "none", "unknown"},
    "side_mirror":   {"crack", "broken_part", "missing_part", "none", "unknown"},
    "headlight":     {"crack", "glass_shatter", "broken_part", "none", "unknown"},
    "taillight":     {"crack", "glass_shatter", "broken_part", "none", "unknown"},
    "fender":        {"dent", "scratch", "crack", "none", "unknown"},
    "quarter_panel": {"dent", "scratch", "crack", "none", "unknown"},
    "body":          {"dent", "scratch", "crack", "none", "unknown"},
    # --- laptop ---
    "screen":        {"crack", "glass_shatter", "none", "unknown"},
    "keyboard":      {"missing_part", "broken_part", "water_damage", "stain", "none", "unknown"},
    "trackpad":      {"broken_part", "scratch", "none", "unknown"},
    "hinge":         {"broken_part", "crack", "none", "unknown"},
    "lid":           {"dent", "scratch", "crack", "none", "unknown"},
    "corner":        {"dent", "crack", "broken_part", "none", "unknown"},
    "port":          {"broken_part", "missing_part", "none", "unknown"},
    "base":          {"crack", "scratch", "water_damage", "none", "unknown"},
    # --- package ---
    "box":            {"torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"},
    "package_corner": {"crushed_packaging", "torn_packaging", "none", "unknown"},
    "package_side":   {"torn_packaging", "crushed_packaging", "stain", "none", "unknown"},
    "seal":           {"broken_part", "missing_part", "torn_packaging", "none", "unknown"},
    "label":          {"missing_part", "stain", "none", "unknown"},
    "contents":       {"broken_part", "water_damage", "missing_part", "none", "unknown"},
    "item":           {"broken_part", "crack", "water_damage", "none", "unknown"},
    "unknown":        {"unknown", "none"},
}

# ---------------------------------------------------------------------------
# Allowed output values (from problem_statement.md)
# ---------------------------------------------------------------------------
ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
}

ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

ALLOWED_PARTS: dict[str, set[str]] = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    },
}

# ---------------------------------------------------------------------------
# Issue-type → evidence requirement family mapping
# Used in Stage 4 to find the best matching evidence_requirements.csv row.
# ---------------------------------------------------------------------------
ISSUE_TO_REQUIREMENT_FAMILY: dict[str, str] = {
    "dent":              "dent or scratch",
    "scratch":           "dent or scratch",
    "crack":             "crack, broken, or missing part",
    "glass_shatter":     "crack, broken, or missing part",
    "broken_part":       "crack, broken, or missing part",
    "missing_part":      "crack, broken, or missing part",
    "torn_packaging":    "crushed, torn, or seal damage",
    "crushed_packaging": "crushed, torn, or seal damage",
    "water_damage":      "water, stain, or label damage",
    "stain":             "water, stain, or label damage",
    "none":              "general claim review",
    "unknown":           "general claim review",
}

# ---------------------------------------------------------------------------
# Output CSV column order (must match problem_statement.md exactly)
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# ---------------------------------------------------------------------------
# FNOL — Auto Insurance configuration
# ---------------------------------------------------------------------------
import os

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
FNOL_TEXT_MODEL   = os.getenv("FNOL_TEXT_MODEL", "mistral-nemo:latest")  # structured JSON extraction
FNOL_VISION_MODEL = os.getenv("FNOL_VISION_MODEL", "llava:7b")           # vision understanding
INTER_CALL_DELAY  = 0.0   # no rate limit locally

FNOL_MANDATORY_FIELDS = [
    "policy_number",
    "injury_indicator",
    "police_report_indicator",
    "accident_date",
    "accident_time",
    "accident_location",
    "vehicle_make",
    "vehicle_model",
]

FNOL_UNCERTAIN_THRESHOLD = 0.6

OUTPUT_DIR = REPO_ROOT / "output"

PROHIBITED_SUMMARY_TERMS = [
    "at fault", "liable", "negligent", "fraud", "fraudulent",
    "coverage denied", "coverage approved", "settle", "reserve",
    "legal action", "lawsuit", "guilty", "innocent",
]