"""
code/pdf_generator.py — Generates the FNOL PDF report using reportlab.
Called by session.py after the user confirms the FNOL draft.
"""

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from agent import config


def generate_pdf(draft, session_id: str) -> Path:
    """Build the FNOL PDF and return its output path."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / f"{session_id}_fnol.pdf"

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    styles  = getSampleStyleSheet()
    BLUE    = colors.HexColor("#1a3c5e")
    LIGHT   = colors.HexColor("#f0f4f8")

    title_style = ParagraphStyle(
        "FNOLTitle", parent=styles["Heading1"],
        fontSize=16, alignment=TA_CENTER, spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "FNOLSub", parent=styles["Normal"],
        fontSize=9, alignment=TA_CENTER, spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "FNOLH2", parent=styles["Heading2"],
        fontSize=11, spaceBefore=12, spaceAfter=4,
        textColor=BLUE,
    )
    body  = styles["Normal"]
    small = ParagraphStyle("FNOLSmall", parent=styles["Normal"], fontSize=9)
    foot  = ParagraphStyle(
        "FNOLFoot", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey, alignment=TA_CENTER,
    )

    story = []

    # ── Header ───────────────────────────────────────────────────────
    story.append(Paragraph("First Notice of Loss — Auto Insurance", title_style))
    story.append(Paragraph(
        f"Session ID: {session_id} &nbsp;&nbsp;&nbsp; "
        f"Filed: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE))
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 1: Structured FNOL Record ────────────────────────────
    story.append(Paragraph("1. FNOL Structured Record", h2))

    fnol = draft.fnol
    field_labels = [
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

    STATUS_COLORS = {
        "confirmed": "#2e7d32",
        "uncertain": "#e65100",
        "missing":   "#c62828",
    }

    table_data = [[
        Paragraph("<b>Field</b>", small),
        Paragraph("<b>Value</b>", small),
        Paragraph("<b>Source</b>", small),
        Paragraph("<b>Status</b>", small),
    ]]
    for label, field in field_labels:
        val   = field.value or "Unknown"
        color = STATUS_COLORS.get(field.status, "#000000")
        table_data.append([
            Paragraph(label, small),
            Paragraph(val, small),
            Paragraph(field.source, small),
            Paragraph(f'<font color="{color}">{field.status}</font>', small),
        ])

    t = Table(table_data, colWidths=[4.5 * cm, 6.5 * cm, 3 * cm, 2.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 2: Neutral Claim Summary ─────────────────────────────
    story.append(Paragraph("2. Neutral Claim Summary", h2))
    story.append(Paragraph(draft.neutral_summary, body))
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 3: Event Chronology ───────────────────────────────────
    story.append(Paragraph("3. Event Chronology", h2))
    for i, event in enumerate(draft.chronology, 1):
        story.append(Paragraph(f"{i}.&nbsp; {event}", body))
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 4: Supporting Document Checklist ──────────────────────
    story.append(Paragraph("4. Supporting Document Checklist", h2))
    for item in draft.document_checklist:
        story.append(Paragraph(f"&#9744;&nbsp; {item}", body))
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 5: Consistency Flags ─────────────────────────────────
    if draft.consistency_flags:
        story.append(Paragraph("5. Consistency Flags — Adjuster Review Required", h2))
        for flag in draft.consistency_flags:
            story.append(Paragraph(f"<b>Field:</b> {flag.field}", body))
            story.append(Paragraph(f"<b>Narrative says:</b> {flag.narrative_says}", body))
            story.append(Paragraph(f"<b>Image shows:</b> {flag.image_shows}", body))
            story.append(Paragraph(f"<b>Question:</b> {flag.question}", body))
            story.append(Spacer(1, 0.2 * cm))

    # ── Section 6: Image Evidence ─────────────────────────────────────
    if draft.image_findings:
        story.append(Paragraph("6. Image Evidence", h2))
        img_data = [[
            Paragraph("<b>Image ID</b>", small),
            Paragraph("<b>Part</b>", small),
            Paragraph("<b>Issue</b>", small),
            Paragraph("<b>Visible</b>", small),
            Paragraph("<b>Quality OK</b>", small),
        ]]
        for f in draft.image_findings:
            img_data.append([
                Paragraph(f.image_id, small),
                Paragraph(f.object_part, small),
                Paragraph(f.issue_type, small),
                Paragraph("Yes" if f.issue_visible else "No", small),
                Paragraph("Yes" if f.image_quality_ok else "No", small),
            ])
        it = Table(img_data, colWidths=[3 * cm, 3.5 * cm, 3.5 * cm, 2 * cm, 2.5 * cm])
        it.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ]))
        story.append(it)

    # ── Footer ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "This document is for intake purposes only. "
        "It does not determine liability, fault, coverage, or settlement.",
        foot,
    ))

    doc.build(story)
    return out_path
