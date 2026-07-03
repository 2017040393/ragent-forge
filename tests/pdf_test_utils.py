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
