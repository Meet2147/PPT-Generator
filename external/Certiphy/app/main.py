# app/main.py

import hashlib
import json
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from app.services.flashcards_builder import build_flashcards_from_quiz_questions  
from app.schemas import QuizGenerateRequest, QuizResponse
from app.registry import EXAMS, list_exams
from app.schemas import FlashcardsGenerateRequest, FlashcardsResponse
from app.services.sources.microsoft_learn import MicrosoftLearnSource
from app.services.sources.aws_exam_guides_html import AwsExamGuidesHtmlSource
from app.services.sources.google_cloud_html import GoogleCloudHtmlSource

from app.services.quiz_builder import build_exam_like_quiz
from app.services.cache import init_cache, get_cached, set_cached

load_dotenv()

app = FastAPI(title="Certification Quiz API", version="0.4.0")

DISCLAIMER = (
    "Practice questions are generated using AI APIs and may be inaccurate. "
    "They are for study assistance only, not a guarantee of exam results. "
    "Questions are original and not sourced from exam dumps."
)

SOURCES = {
    "microsoft_learn": MicrosoftLearnSource(),
    "aws_exam_guides_html": AwsExamGuidesHtmlSource(),
    "google_cloud_html": GoogleCloudHtmlSource(),
}

# Cache settings (tune as you like)
CACHE_TTL_SECONDS = int(os.getenv("QUIZ_CACHE_TTL_SECONDS", str(30 * 24 * 3600)))  # 30 days default
CACHE_ENABLED = os.getenv("QUIZ_CACHE_ENABLED", "true").lower() == "true"

def _make_cache_key(exam_id: str, req: QuizGenerateRequest) -> str:
    """
    Cache only the knobs that affect quiz content.
    NOTE: We intentionally ignore include_explanations in the cache key,
    because we can strip explanations at response time.
    """
    raw = {
        "exam_id": exam_id,
        "mode": req.mode,
        "difficulty": req.difficulty,
        "total_questions": req.total_questions,
        "scenario_ratio": req.scenario_ratio,
        "domain_distribution": req.domain_distribution,
        "time_limit_minutes": req.time_limit_minutes if req.mode == "exam" else None,
    }
    s = json.dumps(raw, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

@app.on_event("startup")
def _startup():
    if CACHE_ENABLED:
        init_cache()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/exams")
def exams():
    return {"exams": list_exams()}

@app.get("/exams/{exam_id}/blueprint")
async def blueprint(exam_id: str):
    exam_id = exam_id.upper().strip()
    cfg = EXAMS.get(exam_id)
    if not cfg:
        raise HTTPException(status_code=404, detail="Unknown exam_id. Use GET /exams")

    src = SOURCES.get(cfg["source_type"])
    if not src:
        raise HTTPException(status_code=500, detail=f"No source handler for {cfg['source_type']}")

    sections = await src.fetch_sections(cfg["blueprint_url"])
    return {
        "exam_id": exam_id,
        "name": cfg["name"],
        "vendor": cfg["vendor"],
        "source": cfg["blueprint_url"],
        "sections": [{"index": i, "title": s["title"]} for i, s in enumerate(sections)],
    }

@app.post("/exams/{exam_id}/generate-quiz", response_model=QuizResponse)
async def generate_quiz(exam_id: str, req: QuizGenerateRequest):
    exam_id = exam_id.upper().strip()
    cfg = EXAMS.get(exam_id)
    if not cfg:
        raise HTTPException(status_code=404, detail="Unknown exam_id. Use GET /exams")

    src = SOURCES.get(cfg["source_type"])
    if not src:
        raise HTTPException(status_code=500, detail=f"No source handler for {cfg['source_type']}")

    difficulty = (req.difficulty or cfg.get("difficulty_default", "easy")).strip().lower()

    # ---------- CACHE HIT ----------
    cache_key = _make_cache_key(exam_id, req)
    if CACHE_ENABLED:
        cached = get_cached(cache_key, ttl_seconds=CACHE_TTL_SECONDS)
        if cached:
            # If client doesn't want explanations, strip them
            if not req.include_explanations and "questions" in cached:
                for q in cached["questions"]:
                    q["explanation"] = None
            return cached

    # ---------- GENERATE ----------
    sections = await src.fetch_sections(cfg["blueprint_url"])

    questions = await build_exam_like_quiz(
        exam_name=cfg["name"],
        vendor=cfg["vendor"],
        sections=sections,
        domains=cfg.get("domains") or {},
        keywords=cfg.get("keywords") or [],
        mode=req.mode,
        total_questions=req.total_questions,
        difficulty=difficulty,
        include_explanations=req.include_explanations,
        scenario_ratio=req.scenario_ratio,
        min_questions=req.min_questions,
        max_retries=req.max_retries,
        domain_distribution_override=req.domain_distribution,
    )

    if len(questions) < req.min_questions:
        raise HTTPException(
            status_code=502,
            detail=f"Generated only {len(questions)} relevant questions for {exam_id}. "
                   f"Try increasing max_retries or total_questions."
        )

    response_obj = QuizResponse(
        exam_id=exam_id,
        exam_name=cfg["name"],
        vendor=cfg["vendor"],
        source=cfg["blueprint_url"],
        disclaimer=DISCLAIMER,
        mode=req.mode,
        time_limit_minutes=req.time_limit_minutes if req.mode == "exam" else None,
        questions=questions,
    )

    # ---------- CACHE SET ----------
    if CACHE_ENABLED:
        set_cached(cache_key, response_obj.model_dump())

    return response_obj

@app.post("/exams/{exam_id}/generate-flashcards", response_model=FlashcardsResponse)
async def generate_flashcards(exam_id: str, req: FlashcardsGenerateRequest):
    exam_id = exam_id.upper().strip()
    cfg = EXAMS.get(exam_id)
    if not cfg:
        raise HTTPException(status_code=404, detail="Unknown exam_id. Use GET /exams")

    try:
        flashcards = await build_flashcards_from_quiz_questions(
            exam_id=exam_id,
            exam_name=cfg["name"],
            vendor=cfg["vendor"],
            questions=req.questions,
            use_ai=req.use_ai,
            max_cards=req.max_cards,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FlashcardsResponse(
        exam_id=exam_id,
        disclaimer=DISCLAIMER,
        flashcards=flashcards,
    )
