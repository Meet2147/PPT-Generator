from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.services.sonar import sonar_generate_json


# ============================================================
# TYPES
# ============================================================

Difficulty = Literal["easy", "medium", "hard"]
Mode = Literal["practice", "exam"]


# ============================================================
# QUIZ SCHEMAS
# ============================================================

class QuizGenerateRequest(BaseModel):
    mode: Mode = "practice"

    total_questions: int = Field(default=25, ge=5, le=100)
    difficulty: Difficulty = "easy"
    include_explanations: bool = True

    # Exam mode settings
    time_limit_minutes: Optional[int] = Field(default=None, ge=5, le=240)
    scenario_ratio: float = Field(default=0.7, ge=0.0, le=1.0)

    # Controls quality
    min_questions: int = Field(default=10, ge=1, le=100)
    max_retries: int = Field(default=2, ge=0, le=5)

    # Optional override: domain weights (should sum ~1.0)
    domain_distribution: Optional[Dict[str, float]] = None


class QuizOption(BaseModel):
    key: str
    text: str


class QuizQuestion(BaseModel):
    id: str
    domain: str
    topic: str
    question: str
    options: List[QuizOption]
    correct_key: str
    explanation: Optional[str] = None


class QuizResponse(BaseModel):
    exam_id: str
    exam_name: str
    vendor: str
    source: str
    disclaimer: str
    mode: str
    time_limit_minutes: Optional[int] = None
    questions: List[QuizQuestion]


# ============================================================
# FLASHCARDS SCHEMAS  ✅ ADD THESE (fixes your ImportError)
# ============================================================

class FlashcardItem(BaseModel):
    id: str
    exam_id: str
    topic: str
    domain: Optional[str] = None
    front: str
    back: str


class FlashcardsGenerateRequest(BaseModel):
    """
    Your app will send quiz questions JSON (same structure as QuizQuestion)
    and the API returns flashcards.
    """
    questions: List[Dict[str, Any]] = Field(default_factory=list)
    use_ai: bool = True
    max_cards: int = Field(default=40, ge=5, le=200)


class FlashcardsResponse(BaseModel):
    exam_id: str
    disclaimer: str
    flashcards: List[FlashcardItem]


# ============================================================
# FLASHCARDS BUILDER (service logic)
# ============================================================

FLASHCARDS_SCHEMA = {
    "type": "object",
    "properties": {
        "flashcards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "front": {"type": "string"},
                    "back": {"type": "string"},
                    "domain": {"type": "string"},
                },
                "required": ["topic", "front", "back", "domain"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["flashcards"],
    "additionalProperties": False,
}

PROMPT = """
You are converting certification quiz questions into flashcards.

Exam: {exam_name}
Vendor: {vendor}

RULES:
- Flashcards must be 100% relevant to {exam_name}.
- The "front" is a short question or concept prompt.
- The "back" is a concise answer + key detail (1-3 bullets max).
- Keep it exam-like and practical.
- Do NOT mention other cloud vendors.
- Do NOT hallucinate services outside the vendor.
- Use the topic/domain info when available.

Input Questions (JSON):
{questions_json}

Return ONLY JSON matching the schema.
"""


def _fid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _best_answer_text(q: Dict[str, Any]) -> str:
    """
    Supports both snake_case and camelCase payloads coming from different clients.
    """
    correct_key = q.get("correct_key") or q.get("correctKey") or ""
    opts = q.get("options") or []

    for o in opts:
        if (o.get("key") or "").upper() == str(correct_key).upper():
            return (o.get("text") or "").strip()

    return ""


async def build_flashcards_from_quiz_questions(
    exam_id: str,
    exam_name: str,
    vendor: str,
    questions: List[Dict[str, Any]],
    use_ai: bool,
    max_cards: int,
) -> List[Dict[str, Any]]:
    """
    Returns: [{id, exam_id, topic, domain, front, back}]
    """

    # Hard limit
    questions = (questions or [])[: max_cards]

    # ----------------------------------------
    # NON-AI MODE (free, deterministic)
    # ----------------------------------------
    if not use_ai:
        cards: List[Dict[str, Any]] = []

        for q in questions:
            topic = (q.get("topic") or "General").strip()
            domain = (q.get("domain") or "").strip() or None
            front = (q.get("question") or "").strip()

            if not front:
                continue

            back_answer = _best_answer_text(q)
            exp = (q.get("explanation") or "").strip()

            back = back_answer.strip()
            if exp:
                if back:
                    back = f"{back}\n\nKey point: {exp}"
                else:
                    back = f"Key point: {exp}"

            if not back:
                back = "Review this concept from the official exam guide."

            cards.append(
                {
                    "id": _fid(front + back),
                    "exam_id": exam_id,
                    "topic": topic,
                    "domain": domain,
                    "front": front,
                    "back": back.strip(),
                }
            )

        return cards

    # ----------------------------------------
    # AI MODE (Sonar)
    # ----------------------------------------
    prompt = PROMPT.format(
        exam_name=exam_name,
        vendor=vendor,
        questions_json=json.dumps(questions, ensure_ascii=False),
    )

    obj = await sonar_generate_json(prompt, schema=FLASHCARDS_SCHEMA)

    cards: List[Dict[str, Any]] = []
    for fc in obj.get("flashcards", []):
        front = (fc.get("front") or "").strip()
        back = (fc.get("back") or "").strip()
        topic = (fc.get("topic") or "General").strip()
        domain = (fc.get("domain") or "").strip() or None

        if not front or not back:
            continue

        cards.append(
            {
                "id": _fid(front + back),
                "exam_id": exam_id,
                "topic": topic,
                "domain": domain,
                "front": front,
                "back": back,
            }
        )

    return cards
