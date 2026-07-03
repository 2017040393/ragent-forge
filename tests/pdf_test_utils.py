from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle


def write_text_pdf(
    path: Path,
    *,
    first_page_text: str = "RAGentForge PDF text layer.",
    second_page_blank: bool = False,
) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    pdf.drawString(72, height - 72, first_page_text)
    pdf.showPage()
    if second_page_blank:
        pdf.showPage()
    pdf.save()


def write_table_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    pdf.drawString(72, height - 72, "Retrieval evaluation table")
    table = Table(
        [
            ["Method", "Hit@1", "MRR"],
            ["lexical", "0.67", "0.78"],
            ["hybrid", "1.00", "1.00"],
        ]
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    table.wrapOn(pdf, width, height)
    table.drawOn(pdf, 72, height - 170)
    pdf.showPage()
    pdf.save()


def write_two_column_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    left_lines = ["Left column alpha", "Left column beta", "Left column gamma"]
    right_lines = ["Right column one", "Right column two", "Right column three"]
    y = height - 72
    for left, right in zip(left_lines, right_lines, strict=True):
        pdf.drawString(72, y, left)
        pdf.drawString(width / 2 + 24, y, right)
        y -= 18
    pdf.showPage()
    pdf.save()


def write_captioned_table_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    pdf.drawString(72, height - 72, "Benchmark notes before the table.")
    pdf.drawString(72, height - 116, "Table 2: Retrieval Evaluation Results")
    table = Table(
        [
            ["Method", "Hit@1", "MRR"],
            ["lexical", "0.67", "0.78"],
            ["hybrid", "1.00", "1.00"],
        ]
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    table.wrapOn(pdf, width, height)
    table.drawOn(pdf, 72, height - 210)
    pdf.drawString(72, height - 250, "Benchmark notes after the table.")
    pdf.showPage()
    pdf.save()


def write_formula_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    lines = [
        "Formula section",
        "RRF(d) = SUM 1 / (k + rank_i(d))",
        "Recall@5 = hits / relevant",
        "Plain explanatory prose remains available.",
    ]
    y = height - 72
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 18
    pdf.showPage()
    pdf.save()


def write_header_footer_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    for page_number in range(1, 4):
        pdf.drawString(72, height - 36, "RAGentForge Technical Report")
        pdf.drawString(72, height - 90, f"Unique body text page {page_number}")
        pdf.drawString(72, 36, "Confidential Draft")
        pdf.showPage()
    pdf.save()
