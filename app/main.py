from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from app.config import settings
from app.models import (
    IdentifyRequest, IdentifyResponse,
    PortionRequest, PortionResponse, PortionEstimate,
    NutrientsRequest, NutrientsResponse,
    AnalyzeTextRequest, AnalyzeResponse,
)
from app.clients.gemini import GeminiClient
from app.clients.perplexity import PerplexityClient
from app.utils import extract_json_object, ModelJSONError

app = FastAPI(
    title="Food Nutrients API",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

gemini = GeminiClient(api_key=settings.gemini_api_key)
pplx = PerplexityClient(api_key=settings.pplx_api_key)

# ---------------- Global JSON error handler (prevents jq explosions) ----------------

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    # Always return JSON so jq never breaks
    return ORJSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": str(exc)[:2000]},
    )

# ---------------- Prompts ----------------

IDENTIFY_SYSTEM = "Return ONLY valid JSON. No markdown. No extra text."
PORTION_SYSTEM = "Return ONLY valid JSON. No markdown. No extra text."
NUTRIENTS_SYSTEM = (
    "Return ONLY valid JSON (no markdown, no commentary). "
    "Do NOT include citations, search_results, links, or extra fields."
)

def identify_user_prompt_text(text: str, hints: Optional[List[str]]) -> str:
    return f"""
Return JSON:
{{
  "candidates":[{{"name":str,"confidence":0-1,"normalized_name":str,"cuisine":str|null,"is_packaged":bool|null,"notes":str|null}}],
  "chosen": {{"name":str,"confidence":0-1,"normalized_name":str,"cuisine":str|null,"is_packaged":bool|null,"notes":str|null}}
}}
User text: {text}
Hints: {hints or []}
Rules:
- candidates max 3
- chosen MUST be an object (not string)
""".strip()

def identify_user_prompt_image(hints: Optional[List[str]]) -> str:
    return f"""
Return JSON:
{{
  "candidates":[{{"name":str,"confidence":0-1,"normalized_name":str,"cuisine":str|null,"is_packaged":bool|null,"notes":str|null}}],
  "chosen": {{"name":str,"confidence":0-1,"normalized_name":str,"cuisine":str|null,"is_packaged":bool|null,"notes":str|null}}
}}
Hints: {hints or []}
Rules:
- candidates max 3
- chosen MUST be an object (not string)
""".strip()

def portion_prompt_text(food_name: str, ctx: Optional[str]) -> str:
    return f"""
Food: {food_name}
Context: {ctx or ""}

Return ONLY JSON:
{{
  "servings": <float>,
  "grams_total": <float>,
  "items_count": <float|null>,
  "household": <string|null>,
  "confidence": <0-1>,
  "assumptions": ["max 3 items, each under 80 chars"]
}}
""".strip()

def portion_prompt_image(food_name: str, ctx: Optional[str]) -> str:
    return f"""
Food: {food_name}
Context: {ctx or ""}

Return ONLY JSON:
{{
  "servings": <float>,
  "grams_total": <float>,
  "items_count": <float|null>,
  "household": <string|null>,
  "confidence": <0-1>,
  "assumptions": ["max 3 items, each under 80 chars"]
}}
""".strip()

def nutrients_prompt(req: NutrientsRequest) -> str:
    p = req.portion
    return f"""
Return ONLY JSON with this exact schema keys:
food_name, portion, calories_kcal, macros, micros, vitamins, minerals, ingredients_guess, allergens_guess, data_sources, notes

Food: {req.food_name}
Region: {req.region}
Brand: {req.brand or ""}

Portion:
servings={p.servings}
grams_total={p.grams_total}
items_count={p.items_count}
household={p.household}

Rules:
- Use units g, mg, kcal
- macros/micros/vitamins/minerals values MUST be objects with:
  {{name, amount, unit, per_100g_amount, daily_value_percent}}
- ingredients_guess/allergens_guess/notes/data_sources MUST be arrays (lists)
- No citations, no links, no search_results.
""".strip()

# ---------------- Helpers ----------------

def model_json_or_400(text: str) -> Any:
    try:
        return extract_json_object(text)
    except ModelJSONError as e:
        raise HTTPException(status_code=400, detail=str(e))

async def force_json_with_gemini(schema_hint: str, text: str) -> Dict[str, Any]:
    repair_system = "You repair JSON. Return ONLY valid JSON. No markdown. No extra text."
    repair_prompt = f"""
Fix the following invalid/truncated JSON into valid JSON.

Schema hint:
{schema_hint}

Input:
{text}

Return ONLY JSON. If impossible, return {{}}.
""".strip()
    repaired = await gemini.generate_text(
        model=settings.gemini_portion_model,
        system=repair_system,
        prompt=repair_prompt,
        temperature=0.0,
        max_output_tokens=1400,
    )
    obj = model_json_or_400(repaired)
    return obj if isinstance(obj, dict) else {}

def normalize_identify_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"candidates": [], "chosen": {"name": "unknown", "confidence": 0.0, "normalized_name": "unknown"}}

    candidates = obj.get("candidates") or []
    chosen = obj.get("chosen")

    # handle chosen sometimes being string
    if isinstance(chosen, str):
        chosen_str = chosen
        chosen = None
        for c in candidates:
            if isinstance(c, dict) and (c.get("name") == chosen_str or c.get("normalized_name") == chosen_str):
                chosen = c
                break
        if chosen is None:
            chosen = {"name": chosen_str, "confidence": 0.5, "normalized_name": chosen_str}

    if not isinstance(chosen, dict):
        # fall back to best candidate
        best = None
        best_conf = -1
        for c in candidates:
            if isinstance(c, dict):
                conf = float(c.get("confidence") or 0.0)
                if conf > best_conf:
                    best_conf = conf
                    best = c
        chosen = best or {"name": "unknown", "confidence": 0.0, "normalized_name": "unknown"}

    # ensure candidates list of dicts
    clean_candidates = [c for c in candidates if isinstance(c, dict)]
    if not clean_candidates:
        clean_candidates = [chosen]

    obj["candidates"] = clean_candidates[:3]
    obj["chosen"] = chosen
    return obj

def normalize_portion_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}

    obj.setdefault("servings", 1.0)
    obj.setdefault("grams_total", 0.0)
    obj.setdefault("items_count", None)
    obj.setdefault("household", None)
    obj.setdefault("confidence", 0.6)
    obj.setdefault("assumptions", [])

    # household must be string or None
    h = obj.get("household")
    if h is not None and not isinstance(h, str):
        obj["household"] = str(h)

    if not isinstance(obj.get("assumptions"), list):
        obj["assumptions"] = [str(obj["assumptions"])]

    try:
        obj["servings"] = float(obj["servings"] or 1.0)
    except Exception:
        obj["servings"] = 1.0

    try:
        obj["grams_total"] = float(obj["grams_total"] or 0.0)
    except Exception:
        obj["grams_total"] = 0.0

    ic = obj.get("items_count")
    if ic is None:
        obj["items_count"] = None
    else:
        try:
            obj["items_count"] = float(ic)
        except Exception:
            obj["items_count"] = None

    try:
        obj["confidence"] = float(obj["confidence"])
    except Exception:
        obj["confidence"] = 0.6

    # clamp
    obj["confidence"] = max(0.0, min(1.0, obj["confidence"]))
    return obj

def normalize_nutrients_obj(obj: Any, req: NutrientsRequest) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}

    # force required top-level keys
    obj.setdefault("food_name", req.food_name)
    if "portion" not in obj or not isinstance(obj["portion"], dict):
        # use request portion as source of truth
        p = req.portion
        obj["portion"] = {
            "servings": p.servings,
            "grams_total": p.grams_total,
            "items_count": p.items_count,
            "household": p.household,
            "confidence": getattr(p, "confidence", 0.6) or 0.6,
            "assumptions": getattr(p, "assumptions", []) or [],
        }

    obj.setdefault("calories_kcal", 0.0)
    obj.setdefault("macros", {})
    obj.setdefault("micros", {})
    obj.setdefault("vitamins", {})
    obj.setdefault("minerals", {})
    obj.setdefault("ingredients_guess", [])
    obj.setdefault("allergens_guess", [])
    obj.setdefault("data_sources", [])
    obj.setdefault("notes", [])

    # lists must be lists
    for k in ("ingredients_guess", "allergens_guess", "data_sources", "notes"):
        if not isinstance(obj.get(k), list):
            obj[k] = [str(obj.get(k))]

    # If user doesn't want per 100g
    if not req.include_per_100g:
        for grp in ("macros", "micros", "vitamins", "minerals"):
            if isinstance(obj.get(grp), dict):
                for _, item in obj[grp].items():
                    if isinstance(item, dict):
                        item["per_100g_amount"] = None

    return obj

# ---------------- Routes ----------------

@app.get("/health")
async def health():
    return {"ok": True, "env": settings.app_env}

@app.get("/v1/meta")
async def meta():
    return {
        "version": "1.0.0",
        "models": {
            "gemini_classifier": settings.gemini_classifier_model,
            "gemini_portion": settings.gemini_portion_model,
            "perplexity_sonar": settings.pplx_sonar_model,
        },
    }

@app.post("/v1/food/identify", response_model=IdentifyResponse)
async def identify_food(req: IdentifyRequest):
    if req.mode != "text" or not req.text:
        raise HTTPException(status_code=422, detail="Use mode=text with text, or use /v1/food/identify-image.")

    out = await gemini.generate_text(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_text(req.text, req.hints),
        temperature=0.2,
        max_output_tokens=800,
    )

    raw = model_json_or_400(out)
    obj = normalize_identify_obj(raw)
    return IdentifyResponse.model_validate(obj)

@app.post("/v1/food/identify-image", response_model=IdentifyResponse)
async def identify_food_image(file: UploadFile = File(...), hints: Optional[str] = None):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Empty upload (image_bytes is empty).")

    hints_list = [h.strip() for h in (hints or "").split(",") if h.strip()] or None

    out = await gemini.generate_with_image(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_image(hints_list),
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        temperature=0.2,
        max_output_tokens=900,
    )

    raw = model_json_or_400(out)
    obj = normalize_identify_obj(raw)
    return IdentifyResponse.model_validate(obj)

@app.post("/v1/food/portion", response_model=PortionResponse)
async def portion_text(req: PortionRequest):
    if req.mode != "text":
        raise HTTPException(status_code=422, detail="Use /v1/food/portion-image for images.")

    out = await gemini.generate_text(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_text(req.food_name, req.text_context),
        temperature=0.2,
        max_output_tokens=700,
    )

    try:
        raw = model_json_or_400(out)
    except HTTPException:
        raw = await force_json_with_gemini(
            '{"servings":number,"grams_total":number,"items_count":number|null,"household":string|null,"confidence":number,"assumptions":string[]}',
            out,
        )

    raw = normalize_portion_obj(raw)
    est = PortionEstimate.model_validate(raw)

    if est.grams_total <= 0:
        est = PortionEstimate(
            servings=max(1.0, est.servings),
            grams_total=100.0,
            items_count=None,
            household=req.household_measure or "1 serving (default 100g)",
            confidence=0.3,
            assumptions=["Defaulted to 100g because model output was uncertain"],
        )

    return PortionResponse(food_name=req.food_name, portion=est)

@app.post("/v1/food/portion-image", response_model=PortionResponse)
async def portion_image(food_name: str, file: UploadFile = File(...), text_context: Optional[str] = None):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Empty upload (image_bytes is empty).")

    out = await gemini.generate_with_image(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_image(food_name, text_context),
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        temperature=0.2,
        max_output_tokens=900,
    )

    try:
        raw = model_json_or_400(out)
    except HTTPException:
        raw = await force_json_with_gemini(
            '{"servings":number,"grams_total":number,"items_count":number|null,"household":string|null,"confidence":number,"assumptions":string[]}',
            out,
        )

    raw = normalize_portion_obj(raw)
    est = PortionEstimate.model_validate(raw)

    if est.grams_total <= 0:
        est = PortionEstimate(
            servings=max(1.0, est.servings),
            grams_total=100.0,
            items_count=None,
            household="1 serving (default 100g)",
            confidence=0.3,
            assumptions=["Defaulted to 100g because model output was uncertain"],
        )

    return PortionResponse(food_name=food_name, portion=est)

@app.post("/v1/food/nutrients", response_model=NutrientsResponse)
async def nutrients(req: NutrientsRequest):
    if req.portion is None:
        raise HTTPException(status_code=422, detail="portion is required. Call /v1/food/portion first.")

    out = await pplx.chat(
        model=settings.pplx_sonar_model,
        system=NUTRIENTS_SYSTEM,
        user=nutrients_prompt(req),
        temperature=0.2,
        max_tokens=900,
    )

    # Parse/repair
    try:
        raw = model_json_or_400(out)
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="Nutrients output not a JSON object.")
    except HTTPException:
        raw = await force_json_with_gemini("NutrientsResponse schema JSON object", out)

    raw = normalize_nutrients_obj(raw, req)
    return NutrientsResponse.model_validate(raw)

@app.post("/v1/food/analyze", response_model=AnalyzeResponse)
async def analyze_text(req: AnalyzeTextRequest):
    # 1) Identify
    identify_out = await gemini.generate_text(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_text(req.text, req.hints),
        temperature=0.2,
        max_output_tokens=800,
    )
    identify_raw = model_json_or_400(identify_out)
    identify_obj = normalize_identify_obj(identify_raw)
    identify_res = IdentifyResponse.model_validate(identify_obj)
    chosen = identify_res.chosen.normalized_name or identify_res.chosen.name

    # 2) Portion (text)
    portion_out = await gemini.generate_text(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_text(chosen, req.text),
        temperature=0.2,
        max_output_tokens=700,
    )
    try:
        portion_raw = model_json_or_400(portion_out)
    except HTTPException:
        portion_raw = await force_json_with_gemini(
            '{"servings":number,"grams_total":number,"items_count":number|null,"household":string|null,"confidence":number,"assumptions":string[]}',
            portion_out,
        )

    portion_raw = normalize_portion_obj(portion_raw)
    portion_est = PortionEstimate.model_validate(portion_raw)

    if portion_est.grams_total <= 0:
        portion_est = PortionEstimate(
            servings=1.0,
            grams_total=100.0,
            items_count=None,
            household="1 serving (default 100g)",
            confidence=0.3,
            assumptions=["Defaulted to 100g because model output was uncertain"],
        )

    portion_res = PortionResponse(food_name=chosen, portion=portion_est)

    # 3) Nutrients
    nreq = NutrientsRequest(
        food_name=chosen,
        portion=portion_est,
        region=req.region,
        include_per_100g=req.include_per_100g,
    )

    nutrients_out = await pplx.chat(
        model=settings.pplx_sonar_model,
        system=NUTRIENTS_SYSTEM,
        user=nutrients_prompt(nreq),
        temperature=0.2,
        max_tokens=900,
    )

    try:
        nutrients_raw = model_json_or_400(nutrients_out)
        if not isinstance(nutrients_raw, dict):
            raise HTTPException(status_code=400, detail="Nutrients output not an object")
    except HTTPException:
        nutrients_raw = await force_json_with_gemini("NutrientsResponse schema JSON object", nutrients_out)

    nutrients_raw = normalize_nutrients_obj(nutrients_raw, nreq)
    nutrients_res = NutrientsResponse.model_validate(nutrients_raw)

    return AnalyzeResponse(
        identify=identify_res,
        portion=portion_res,
        nutrients=nutrients_res,
        cost_tier={"identify": "$", "portion": "$$", "nutrients": "$$$$"},
    )
