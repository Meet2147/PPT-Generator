import hashlib
from typing import Any, Dict, List

from app.services.sonar import sonar_generate_json

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
    # Quiz schema from your Swift models uses:
    # options: [{key,text}], correct_key in JSON (decoded to correctKey in Swift)
    correct_key = q.get("correct_key") or q.get("correctKey") or ""
    opts = q.get("options") or []
    for o in opts:
        if (o.get("key") or "").upper() == correct_key.upper():
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
    # Hard limit for safety
    questions = questions[: max_cards]

    # If AI off -> deterministic conversion (still useful, zero cost)
    if not use_ai:
        cards: List[Dict[str, Any]] = []
        for q in questions:
            topic = q.get("topic") or "General"
            domain = q.get("domain") or ""
            front = (q.get("question") or "").strip()
            back_answer = _best_answer_text(q)
            exp = (q.get("explanation") or "").strip()

            back = back_answer
            if exp:
                back = f"{back_answer}\n\nKey point: {exp}"

            card = {
                "id": _fid(front + back),
                "exam_id": exam_id,
                "topic": topic,
                "domain": domain or None,
                "front": front,
                "back": back.strip(),
            }
            cards.append(card)
        return cards

    # AI mode: ask Sonar to produce true flashcard “front/back”
    import json
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
        domain = (fc.get("domain") or "").strip()

        if not front or not back:
            continue

        cards.append(
            {
                "id": _fid(front + back),
                "exam_id": exam_id,
                "topic": topic,
                "domain": domain or None,
                "front": front,
                "back": back,
            }
        )

    return cards
