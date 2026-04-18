import re
from typing import Iterable

import requests

from app.config import settings
from app.models import DeckDraft, GenerationRequest, SlideDraft


PROMPT_TEMPLATE = """Create a professional PowerPoint outline.
Return only in this exact format:

#Title: PRESENTATION TITLE
#Subtitle: ONE-LINE POSITIONING

#Slide: 1
#Header: TITLE
#Content: 2-4 concise sentences for the slide body.
#Image: short image description or none

Repeat until the final slide, then end with:
#Slide: END

Rules:
- Build a persuasive, professional deck.
- Keep titles short.
- Keep each body under 320 characters.
- Include a table of contents near the start.
- Include a conclusion slide at the end.
- Match the requested audience and tone.
"""


def _perplexity_chat(messages: list[dict[str, str]], max_tokens: int = 1800) -> str:
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.pplx_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.pplx_model,
            "messages": messages,
            "temperature": 0.35,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "return_images": False,
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content if isinstance(content, str) else str(content)


def build_deck_draft(request: GenerationRequest) -> DeckDraft:
    ai_output = ""
    if settings.pplx_api_key:
        try:
            ai_output = _perplexity_chat(
                [
                    {"role": "system", "content": PROMPT_TEMPLATE},
                    {
                        "role": "user",
                        "content": (
                            f"Topic: {request.topic}\n"
                            f"Audience: {request.audience}\n"
                            f"Objective: {request.objective or 'Create a high-conviction presentation.'}\n"
                            f"Tone: {request.tone}\n"
                            f"Slides requested: {request.slide_count}\n"
                        ),
                    },
                ]
            )
        except Exception:
            ai_output = ""

    if ai_output:
        parsed = parse_deck(ai_output)
        if parsed:
            return parsed

    return build_local_fallback(request)


def parse_deck(raw_text: str) -> DeckDraft | None:
    title = "Professional Presentation"
    subtitle = "Generated narrative ready for PowerPoint."
    slides: list[SlideDraft] = []
    current: dict[str, str | int | None] | None = None

    for line in [item.strip() for item in raw_text.splitlines() if item.strip()]:
        if line.startswith("#Title:"):
            title = line.replace("#Title:", "", 1).strip() or title
        elif line.startswith("#Subtitle:"):
            subtitle = line.replace("#Subtitle:", "", 1).strip() or subtitle
        elif line.startswith("#Slide:"):
            marker = line.replace("#Slide:", "", 1).strip()
            if marker.upper() == "END":
                break
            if current:
                slides.append(_slide_from_state(current))
            current = {"number": int(marker), "title": "", "content": "", "image_prompt": None}
        elif line.startswith("#Header:") and current is not None:
            current["title"] = line.replace("#Header:", "", 1).strip()
        elif line.startswith("#Content:") and current is not None:
            current["content"] = line.replace("#Content:", "", 1).strip()
        elif line.startswith("#Image:") and current is not None:
            image_prompt = line.replace("#Image:", "", 1).strip()
            current["image_prompt"] = None if image_prompt.lower() == "none" else image_prompt
        elif current is not None and current.get("content"):
            current["content"] = f"{current['content']} {line}".strip()

    if current:
        slides.append(_slide_from_state(current))

    if not slides:
        return None

    return DeckDraft(title=title, subtitle=subtitle, slides=slides, fallback_used=False)


def _slide_from_state(state: dict[str, str | int | None]) -> SlideDraft:
    return SlideDraft(
        number=int(state["number"] or 1),
        title=str(state["title"] or "Untitled"),
        content=str(state["content"] or ""),
        image_prompt=state["image_prompt"] if isinstance(state["image_prompt"], str) else None,
    )


def build_local_fallback(request: GenerationRequest) -> DeckDraft:
    topic = request.topic.strip()
    title = _title_case(topic)
    objective = request.objective or f"Explain why {topic} matters and what action to take next."
    sections = [
        ("Executive Brief", f"{title} is presented for a {request.audience} audience with a {request.tone} tone. This deck is built to clarify the decision, create alignment, and move from ideas to execution."),
        ("Agenda", "1. Context  2. Opportunity  3. Approach  4. Proof  5. Rollout  6. Close"),
        ("Why Now", f"The timing around {topic.lower()} is favorable because teams want faster content creation, stronger polish, and fewer manual editing cycles. Speed only matters when the output still looks premium."),
        ("Audience Problem", "Most teams either build slides manually and lose time, or use generic AI tools that produce decks needing heavy cleanup before they can be shared."),
        ("Our Position", f"We turn {topic.lower()} into polished PowerPoint narratives with structured storytelling, premium layouts, and export-ready slides that still feel custom."),
        ("Differentiation", "The product emphasizes subscription value over credits, tighter PowerPoint quality, and a cleaner experience for real work instead of novelty generation."),
        ("Operating Model", "Prompt input, narrative generation, slide composition, design preset application, and instant PPTX delivery create a simple flow that is easy to explain and easy to monetize."),
        ("Revenue Model", "Subscription tiers keep pricing predictable for users while increasing retention, upsell potential, and team expansion over time."),
        ("Conclusion", f"{objective} The next step is to position the product as a premium, reliable alternative for people who need presentable decks on the first export."),
    ]

    slides = [
        SlideDraft(
            number=index,
            title=header,
            content=content,
            image_prompt=_image_prompt(header, topic),
        )
        for index, (header, content) in enumerate(sections[: request.slide_count], start=1)
    ]
    return DeckDraft(
        title=title,
        subtitle=f"Subscription-first presentation generation for {request.audience} teams.",
        slides=slides,
        fallback_used=True,
    )


def summarize_deck(deck: DeckDraft) -> str:
    slide_titles = ", ".join(slide.title for slide in deck.slides[:4])
    return f"{deck.title} includes {len(deck.slides)} slides covering {slide_titles}."


def _title_case(value: str) -> str:
    clean = re.sub(r"\s+", " ", value).strip()
    return clean.title() if clean else "Presentation"


def _image_prompt(header: str, topic: str) -> str:
    lower = header.lower()
    if "agenda" in lower:
        return "minimal editorial agenda graphic with layered cards"
    if "revenue" in lower or "proof" in lower:
        return f"premium dashboard style illustration about {topic}"
    if "conclusion" in lower:
        return "confident executive team closing a successful presentation"
    return f"professional editorial illustration for {topic} focused on {header}"
