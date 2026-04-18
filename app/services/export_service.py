import asyncio
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from app.models import DeckDraft, GenerationRequest
from app.services.deck_service import PALETTES


async def build_pdf_export(request: GenerationRequest, deck: DeckDraft, file_stem: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{file_stem}.pdf"
    await asyncio.to_thread(_render_pdf, request, deck, pdf_path)
    return pdf_path


def _render_pdf(request: GenerationRequest, deck: DeckDraft, pdf_path: Path) -> None:
    width, height = landscape((1280, 720))
    palette = PALETTES.get(request.design_number, PALETTES[2])
    c = canvas.Canvas(str(pdf_path), pagesize=(width, height))

    _cover_page(c, width, height, palette, deck, request)
    c.showPage()

    for index, slide in enumerate(deck.slides, start=1):
        _content_page(c, width, height, palette, index, len(deck.slides), slide.title, slide.content)
        c.showPage()

    c.save()


def _cover_page(c, width, height, palette, deck, request) -> None:
    c.setFillColor(HexColor(f"#{palette['bg']}"))
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColor(HexColor(f"#{palette['accent']}"))
    c.roundRect(52, height - 72, 180, 18, 8, fill=1, stroke=0)
    c.setFillColor(HexColor(f"#{palette['bg']}"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, height - 61, f"{request.audience.title()} Plan")

    c.setFillColor(HexColor(f"#{palette['text']}"))
    c.setFont("Helvetica-Bold", 28)
    for idx, line in enumerate(simpleSplit(deck.title, "Helvetica-Bold", 28, 430)):
        c.drawString(60, height - 130 - (idx * 34), line)

    c.setFillColor(HexColor(f"#{palette['muted']}"))
    c.setFont("Helvetica", 15)
    for idx, line in enumerate(simpleSplit(deck.subtitle, "Helvetica", 15, 430)):
        c.drawString(62, height - 230 - (idx * 22), line)

    c.setFillColor(HexColor(f"#{palette['panel']}"))
    c.roundRect(width - 320, 110, 250, 340, 18, fill=1, stroke=0)
    c.setFillColor(HexColor(f"#{palette['accent']}"))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(width - 290, 420, "DECK SNAPSHOT")
    metrics = [
        ("Slides", str(len(deck.slides))),
        ("Tone", request.tone.title()),
        ("Format", "PPTX + PDF" if request.export_pdf else "PPTX"),
        ("Preview", "Included"),
    ]
    y = 385
    for label, value in metrics:
        c.setFillColor(HexColor(f"#{palette['muted']}"))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(width - 290, y, label.upper())
        c.setFillColor(HexColor(f"#{palette['text']}"))
        c.setFont("Helvetica-Bold", 20)
        c.drawString(width - 290, y - 24, value)
        y -= 74


def _content_page(c, width, height, palette, index, total, title, content) -> None:
    c.setFillColor(HexColor(f"#{palette['bg']}"))
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColor(HexColor(f"#{palette['panel']}"))
    c.roundRect(40, 42, width - 80, height - 84, 24, fill=1, stroke=0)
    c.setFillColor(HexColor(f"#{palette['accent']}"))
    c.rect(72, height - 152, 10, 80, fill=1, stroke=0)

    c.setFillColor(HexColor(f"#{palette['text']}"))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(102, height - 102, title)

    c.setFont("Helvetica", 16)
    body_lines = simpleSplit(content, "Helvetica", 16, 420)
    y = height - 180
    for line in body_lines:
        c.drawString(104, y, line)
        y -= 24

    c.setFillColor(HexColor(f"#{palette['bg']}"))
    c.roundRect(width - 340, 120, 250, 310, 20, fill=1, stroke=0)
    c.setFillColor(HexColor(f"#{palette['text']}"))
    c.setFont("Helvetica-Bold", 16)
    quote = "Professional structure, clearer messaging, and stronger visuals turn generated decks into something teams can actually ship."
    y = 382
    for line in simpleSplit(quote, "Helvetica-Bold", 16, 180):
        c.drawString(width - 310, y, line)
        y -= 24

    c.setFillColor(HexColor(f"#{palette['accent']}"))
    c.roundRect(width - 310, 185, 150, 28, 12, fill=1, stroke=0)
    c.setFillColor(HexColor(f"#{palette['bg']}"))
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(width - 235, 194, f"Slide {index:02d}")

    c.setFillColor(HexColor(f"#{palette['muted']}"))
    c.setFont("Helvetica", 10)
    c.drawRightString(width - 70, 58, f"{index}/{total}")
