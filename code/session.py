"""
code/session.py — FNOL Auto Insurance chatbot CLI.

Usage:
    python code/session.py                         # interactive mode
    python code/session.py --demo 1                # load narrative #1
    python code/session.py --demo 3 --images ./img1.jpg ./img2.jpg
    python code/session.py --demo 8 --no-clarify   # skip clarification loop
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import config
from agent.llm_client import LLMClient
from agent.models import (
    ClaimExtraction, FNOLField,
    ClarificationState, FNOLDraft, ConsistencyFlag,
)
from agent.stages.s1_narrative_extractor import extract_fnol
from agent.stages.s4_completeness_checker import check_completeness, question_for
from agent.stages.s2_object_validator import validate_object
from agent.stages.s3_damage_inspector import inspect_image, make_no_match_placeholder

NARRATIVES_PATH = (
    Path(__file__).parent / "data" / "narratives" / "synthetic_narratives.json"
)


# ---------------------------------------------------------------------------
# Narrative loader
# ---------------------------------------------------------------------------

def load_demo_narrative(number: int) -> str:
    if not NARRATIVES_PATH.exists():
        print(f"[ERROR] Narratives file not found: {NARRATIVES_PATH}")
        sys.exit(1)
    with open(NARRATIVES_PATH, encoding="utf-8") as f:
        narratives = json.load(f)
    key = str(number)
    if key not in narratives:
        print(f"[ERROR] Narrative {number} not found. Available: {sorted(narratives.keys(), key=int)}")
        sys.exit(1)
    return narratives[key]["narrative"]


# ---------------------------------------------------------------------------
# Clarification loop
# ---------------------------------------------------------------------------

def run_clarification_loop(state: ClarificationState, client: LLMClient) -> object:
    missing = check_completeness(state.fnol)
    total = len(missing)
    answered = 0

    while missing:
        field_name = missing[0]
        answered += 1
        print(f"\n{'─' * 50}")
        print(f"Question {answered} of {total}")
        print(f"{'─' * 50}")
        print(question_for(field_name))
        print("(Type 'skip' to mark as unknown)\n")

        user_answer = input("> ").strip()

        if user_answer.lower() == "skip":
            value, confidence = "unknown", 0.5
        else:
            value, confidence = user_answer, 0.9

        setattr(state.fnol, field_name, FNOLField(
            value=value,
            source="user_correction",
            confidence=confidence,
            status="confirmed",
            evidence_ref=user_answer,
        ))
        state.answered_fields.append(field_name)
        state.turn_number += 1
        missing = check_completeness(state.fnol)

    return state.fnol


# ---------------------------------------------------------------------------
# Image branch
# ---------------------------------------------------------------------------

def run_image_branch(
    image_paths: list[str],
    fnol,
    client: LLMClient,
) -> tuple[list, list]:
    findings, flags = [], []

    claim = ClaimExtraction(
        claimed_issue_description=fnol.damage_description.value or "auto damage",
        claimed_parts=[],
        claimed_issue_keywords=[],
        ambiguous_or_vague=False,
    )

    for img_path_str in image_paths:
        p = Path(img_path_str)
        if not p.exists():
            print(f"  [WARN] Image not found: {img_path_str}")
            continue

        vr = validate_object(p, p.stem, "car", client)
        print(f"  [S2] {p.stem}: {vr.match_class} (conf={vr.confidence:.2f})")

        if vr.match_class in ("match", "uncertain"):
            finding = inspect_image(p, p.stem, "car", claim, client)
            findings.append(finding)
            print(f"  [S3] {p.stem}: part={finding.object_part} | issue={finding.issue_type}")

            if fnol.damage_description.value:
                narrative_part = fnol.damage_description.value.lower()
                image_part = finding.object_part.replace("_", " ")
                if image_part and image_part not in narrative_part:
                    flags.append(ConsistencyFlag(
                        field="damage_description",
                        narrative_says=fnol.damage_description.value,
                        image_shows=finding.object_part,
                        question=(
                            f"Your narrative describes '{fnol.damage_description.value}', "
                            f"but the image appears to show damage to the {image_part}. "
                            f"Could you clarify which area was damaged?"
                        ),
                    ))
        else:
            findings.append(make_no_match_placeholder(p.stem))

    return findings, flags


# ---------------------------------------------------------------------------
# FNOL assembly
# ---------------------------------------------------------------------------

def assemble_draft(fnol, findings: list, flags: list) -> FNOLDraft:
    checklist = [
        "Photos of damage",
        "Repair estimate",
        "Policy declaration page",
    ]
    if fnol.injury_indicator.value == "yes":
        checklist.append("Medical report or urgent care records")
    if fnol.police_report_indicator.value == "yes":
        checklist.append(
            f"Police report ({fnol.police_report_number.value or 'number TBD'})"
        )
    if fnol.other_party_name.value:
        checklist.append("Other party insurance documentation")

    date  = fnol.accident_date.value     or "unknown date"
    time_ = fnol.accident_time.value     or "unknown time"
    loc   = fnol.accident_location.value or "unknown location"
    who   = fnol.insured_name.value      or "the insured"
    year  = fnol.vehicle_year.value      or ""
    make  = fnol.vehicle_make.value      or ""
    model = fnol.vehicle_model.value     or ""
    vehicle = f"{year} {make} {model}".strip()
    damage  = fnol.damage_description.value or "damage not described"
    other   = fnol.other_party_name.value

    chronology = [
        f"Accident occurred on {date} at {time_} near {loc}.",
        f"{who} was driving a {vehicle}.",
        f"Damage reported: {damage}.",
    ]
    if other:
        chronology.append(f"Other party involved: {other}.")
    if fnol.police_report_indicator.value == "yes":
        report = fnol.police_report_number.value or "pending"
        chronology.append(f"Police responded. Report number: {report}.")
    if fnol.towing_required.value == "yes":
        company = fnol.towing_company.value or "unknown company"
        chronology.append(f"Vehicle towed by {company}.")

    summary = (
        f"On {date} at approximately {time_}, {who} reported an auto accident near {loc}. "
        f"The insured vehicle, a {vehicle}, sustained {damage}. "
        f"Injury status: {fnol.injury_indicator.value or 'unknown'}. "
        f"Police report filed: {fnol.police_report_indicator.value or 'unknown'}. "
        f"This record is for intake purposes only and does not determine liability or coverage."
    )

    return FNOLDraft(
        fnol=fnol,
        consistency_flags=flags,
        neutral_summary=summary,
        chronology=chronology,
        document_checklist=checklist,
        image_findings=findings,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_fnol_table(fnol) -> None:
    fields = [
        ("Policy Number",         fnol.policy_number),
        ("Insured Name",          fnol.insured_name),
        ("Reporter Name",         fnol.reporter_name),
        ("Reporter Relationship", fnol.reporter_relationship),
        ("Accident Date",         fnol.accident_date),
        ("Accident Time",         fnol.accident_time),
        ("Accident Location",     fnol.accident_location),
        ("Vehicle Make",          fnol.vehicle_make),
        ("Vehicle Model",         fnol.vehicle_model),
        ("Vehicle Year",          fnol.vehicle_year),
        ("Other Party Name",      fnol.other_party_name),
        ("Other Party Vehicle",   fnol.other_party_vehicle),
        ("Other Party Insurance", fnol.other_party_insurance),
        ("Damage Description",    fnol.damage_description),
        ("Damage Severity",       fnol.damage_severity),
        ("Injury Indicator",      fnol.injury_indicator),
        ("Injury Description",    fnol.injury_description),
        ("Police Report Filed",   fnol.police_report_indicator),
        ("Police Report Number",  fnol.police_report_number),
        ("Towing Required",       fnol.towing_required),
        ("Towing Company",        fnol.towing_company),
    ]
    print(f"\n{'─' * 65}")
    print("  FNOL DRAFT — Please review")
    print(f"{'─' * 65}")
    for label, field in fields:
        val    = field.value or "—"
        status = field.status
        print(f"  {label:<28} {val:<28} [{status}]")
    print(f"{'─' * 65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FNOL Auto Insurance Chatbot")
    parser.add_argument("--demo", type=int, default=None,
                        help="Load synthetic narrative by number (1-20)")
    parser.add_argument("--images", nargs="*", default=[],
                        help="Image file paths for damage analysis")
    parser.add_argument("--no-clarify", action="store_true",
                        help="Skip the clarification loop (batch/testing mode)")
    args = parser.parse_args()

    client     = LLMClient(cache=True)
    session_id = str(uuid.uuid4())[:8]

    print("\n" + "═" * 55)
    print("  FNOL Auto Insurance — First Notice of Loss")
    print("═" * 55)

    # -- Narrative input --------------------------------------------------
    if args.demo is not None:
        print(f"\n[Demo Mode] Loading narrative #{args.demo}...")
        narrative = load_demo_narrative(args.demo)
        print(f"\nNarrative:\n{narrative}\n")
    else:
        print("\nDescribe the accident in detail.")
        print("Press Enter twice when done.\n")
        lines = []
        while True:
            line = input("> " if not lines else "  ")
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        narrative = "\n".join(lines).strip()

    # -- S1: Extract ------------------------------------------------------
    print("\nExtracting information from your narrative...")
    fnol    = extract_fnol(narrative, client)
    missing = check_completeness(fnol)
    print(f"✓ Extracted {21 - len(missing)} of 21 fields from narrative.")

    # -- S4 + Clarification loop ------------------------------------------
    if missing and not args.no_clarify:
        print(f"⚠  {len(missing)} mandatory field(s) need clarification.")
        state = ClarificationState(
            fnol=fnol, turn_number=0, session_id=session_id,
            answered_fields=[], pending_flags=[], image_findings=[],
        )
        fnol = run_clarification_loop(state, client)
    elif missing:
        print(f"⚠  Skipping clarification (--no-clarify). "
              f"{len(missing)} field(s) remain missing.")

    # -- Image branch -----------------------------------------------------
    findings, flags = [], []
    if args.images:
        print(f"\nAnalysing {len(args.images)} image(s)...")
        findings, flags = run_image_branch(args.images, fnol, client)

    # -- Consistency flags ------------------------------------------------
    if flags:
        print(f"\n⚠  Consistency note(s):")
        for flag in flags:
            print(f"   -> {flag.question}")

    # -- Display FNOL table -----------------------------------------------
    print_fnol_table(fnol)

    # -- Assemble draft ---------------------------------------------------
    draft = assemble_draft(fnol, findings, flags)

    # -- PDF generation ---------------------------------------------------
    confirm = input("Generate PDF? (yes/no): ").strip().lower()
    if confirm in ("yes", "y", ""):
        sys.path.insert(0, str(Path(__file__).parent))
        from pdf_generator import generate_pdf
        out_path = generate_pdf(draft, session_id)
        print(f"\n✓ PDF saved: {out_path}")
    else:
        print("\nSession ended. No PDF generated.")


if __name__ == "__main__":
    main()
