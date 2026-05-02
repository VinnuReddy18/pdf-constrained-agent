from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SS_DIR = ROOT / "ss"
OUTPUT = ROOT / "PDF_Constrained_Agent_Evaluation_Report.pdf"


SCREENSHOTS = [
    {
        "title": "Multilingual Query 1 - Telugu",
        "kind": "Additional multilingual support",
        "path": SS_DIR / "Screenshot 2026-05-02 202159.png",
        "note": "The assistant answers a Telugu user query while grounding the response in the sample PDF and returning citations.",
    },
    {
        "title": "Multilingual Query 2 - Hinglish/Hindi",
        "kind": "Additional multilingual support",
        "path": SS_DIR / "Screenshot 2026-05-02 202214.png",
        "note": "The assistant handles Hindi/Hinglish follow-up style questions and keeps citations attached to the PDF evidence.",
    },
    {
        "title": "Valid Query 1 - Definition of AI Engineering",
        "kind": "Valid PDF-grounded query",
        "path": SS_DIR / "Screenshot 2026-05-03 021416.png",
        "note": "Expected behavior: answer from the PDF with page/section citations.",
    },
    {
        "title": "Valid Query 2 - Core Responsibilities",
        "kind": "Valid PDF-grounded query",
        "path": SS_DIR / "Screenshot 2026-05-03 021433.png",
        "note": "Expected behavior: list responsibilities using only the document context.",
    },
    {
        "title": "Valid Query 3 - ML Engineering vs AI Engineering",
        "kind": "Valid PDF-grounded query",
        "path": SS_DIR / "Screenshot 2026-05-03 021452.png",
        "note": "Expected behavior: compare the concepts using cited PDF evidence.",
    },
    {
        "title": "Valid Query 4 - Decoder-Only Data Flow",
        "kind": "Valid PDF-grounded query",
        "path": SS_DIR / "Screenshot 2026-05-03 021539.png",
        "note": "Expected behavior: extract the decoder-only model data flow and cite source pages.",
    },
    {
        "title": "Valid Query 5 - Causal Masking",
        "kind": "Valid PDF-grounded query",
        "path": SS_DIR / "Screenshot 2026-05-03 021554.png",
        "note": "Expected behavior: answer the specific concept question with citations.",
    },
    {
        "title": "Invalid Query 1 - FIFA World Cup",
        "kind": "Out-of-scope query",
        "path": SS_DIR / "Screenshot 2026-05-03 021717.png",
        "note": "Expected behavior: refuse because the answer is not available in the PDF.",
    },
    {
        "title": "Invalid Query 2 - Weather",
        "kind": "Out-of-scope query",
        "path": SS_DIR / "Screenshot 2026-05-03 021739.png",
        "note": "Expected behavior: refuse current-weather information because it is outside the PDF.",
    },
    {
        "title": "Invalid Query 3 - Python Code",
        "kind": "Out-of-scope query",
        "path": SS_DIR / "Screenshot 2026-05-03 021752.png",
        "note": "Expected behavior: refuse code-generation requests unrelated to the PDF.",
    },
]


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(doc.pagesize[0] - 0.45 * inch, 0.28 * inch, f"Page {doc.page}")
    canvas.restoreState()


def image_flowable(path: Path, max_width: float, max_height: float) -> ReportImage:
    with Image.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    return ReportImage(str(path), width=width * scale, height=height * scale)


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.42 * inch,
        bottomMargin=0.42 * inch,
        title="PDF-Constrained Conversational Agent Evaluation Report",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#111827"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#111827"),
            spaceBefore=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#374151"),
        )
    )

    story = []
    story.append(Paragraph("PDF-Constrained Conversational Agent", styles["ReportTitle"]))
    story.append(Paragraph("Evaluation Report with Valid, Invalid, and Multilingual Query Evidence", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "This report documents the behavior of a PDF-grounded conversational agent. "
            "The agent accepts a PDF, answers only from retrieved PDF evidence, provides citations, "
            "and refuses questions whose answers are not present in the source document.",
            styles["BodyTextCustom"],
        )
    )

    summary_data = [
        ["Item", "Details"],
        ["Sample PDF", "ai_engineering_revision_notes.pdf"],
        ["Valid queries", "5 PDF-grounded questions answered with citations"],
        ["Invalid queries", "3 out-of-scope questions refused"],
        ["Additional support", "Multilingual Telugu and Hindi/Hinglish examples"],
        ["Citation behavior", "Responses include page number and detected section/topic reference where available"],
        ["Refusal behavior", "Unavailable or unrelated information is refused instead of hallucinated"],
    ]
    table = Table(summary_data, colWidths=[2.0 * inch, 8.4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph("Evaluation Checklist", styles["SectionTitle"]))
    checklist_data = [
        ["Requirement", "Evidence in Screenshots"],
        ["Accept any PDF as input", "Sample PDF is selected and indexed in the application."],
        ["Conversational querying", "The chat UI accepts natural-language queries and follow-up style multilingual queries."],
        ["Answer only from PDF", "Valid answers are grounded in the AI Engineering revision notes PDF."],
        ["Refuse out-of-scope queries", "FIFA, weather, and unrelated coding questions are refused."],
        ["Citations", "Valid responses show citations such as page numbers and topic references."],
        ["Testability", "The report includes 5 valid and 3 invalid examples."],
    ]
    checklist = Table(checklist_data, colWidths=[3.1 * inch, 7.3 * inch])
    checklist.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFFFFF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(checklist)

    for item in SCREENSHOTS:
        story.append(PageBreak())
        story.append(Paragraph(item["title"], styles["SectionTitle"]))
        story.append(Paragraph(f"<b>Category:</b> {item['kind']}", styles["BodyTextCustom"]))
        story.append(Paragraph(item["note"], styles["SmallNote"]))
        story.append(Spacer(1, 0.12 * inch))
        story.append(image_flowable(item["path"], max_width=10.6 * inch, max_height=5.15 * inch))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUTPUT)


if __name__ == "__main__":
    main()
