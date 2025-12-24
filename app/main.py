from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from app.config import settings
from app.models import (
    IdentifyRequest, IdentifyResponse, FoodCandidate,
    PortionRequest, PortionResponse, PortionEstimate,
    NutrientsRequest, NutrientsResponse, NutrientItem,
    AnalyzeTextRequest, AnalyzeResponse,
)
from app.clients.gemini import GeminiClient
from app.clients.perplexity import PerplexityClient
from app.utils import extract_json_object, ModelJSONError

app = FastAPI(
    title="Dashovia Food + PPT API",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

gemini = GeminiClient(api_key=settings.gemini_api_key)       # keys in .env via settings
pplx = PerplexityClient(api_key=settings.pplx_api_key)

# ---------------------------
# Prompts (STRICT JSON)
# ---------------------------
IDENTIFY_SYSTEM = (
    "You are a food identification classifier.\n"
    "Return ONLY a single valid JSON object.\n"
    "No markdown. No extra text.\n"
)

PORTION_SYSTEM = (
    "You estimate portion sizes.\n"
    "Return ONLY a single valid JSON object.\n"
    "No markdown. No extra text.\n"
)

NUTRIENTS_SYSTEM = (
    "You are a nutrition analysis engine.\n"
    "Return ONLY a single valid JSON object.\n"
    "No markdown. No commentary.\n"
)

def identify_user_prompt_text(text: str, hints: Optional[List[str]]) -> str:
    return f"""
Return ONLY JSON of this form:
{{
  "candidates":[
    {{
      "name":"<as user said>",
      "confidence":0-1,
      "normalized_name":"<canonical>",
      "cuisine":null,
      "is_packaged":null,
      "notes":null
    }}
  ],
  "chosen": <one of the candidates objects>
}}
Rules:
- candidates must be a JSON array
- chosen must be an object from candidates (NOT a string)
- If uncertain, lower confidence
User text: {text}
Hints: {hints or []}
""".strip()

def identify_user_prompt_image(hints: Optional[List[str]]) -> str:
    return f"""
Look at the image and identify food item(s).
Return ONLY JSON:
{{
  "candidates":[
    {{
      "name":"<short>",
      "confidence":0-1,
      "normalized_name":"<canonical>",
      "cuisine":null,
      "is_packaged":null,
      "notes":null
    }}
  ],
  "chosen": <one of the candidates objects>
}}
Hints: {hints or []}
""".strip()

def portion_prompt_text(food_name: str, servings: float, household: Optional[str], ctx: Optional[str]) -> str:
    return f"""
Estimate portion for: {food_name}
Context: {ctx or ""}
User provided: servings={servings}, household_measure={household or ""}

Return ONLY JSON:
{{
  "servings": <number>,
  "grams_total": <number>,
  "items_count": <number or null>,
  "household": <string or null>,
  "confidence": <number 0-1>,
  "assumptions": ["..."]
}}

IMPORTANT:
- servings/grams_total/items_count/confidence MUST be numbers (not strings like "40g")
- household MUST be a string or null (never a number)
- assumptions MUST be an array of strings
""".strip()

def portion_prompt_image(food_name: str, ctx: Optional[str]) -> str:
    return f"""
Estimate portion from the image for: {food_name}
Extra context: {ctx or ""}

Return ONLY JSON:
{{
  "servings": <number>,
  "grams_total": <number>,
  "items_count": <number or null>,
  "household": <string or null>,
  "confidence": <number 0-1>,
  "assumptions": ["..."]
}}

IMPORTANT:
- numeric fields MUST be numbers (no "40g" strings)
- household MUST be string/null
""".strip()

def nutrients_prompt(req: NutrientsRequest) -> str:
    p = req.portion
    # Keep schema smaller + stable; you can expand later
    return f"""
Return ONLY JSON matching exactly:

{{
  "food_name": "{req.food_name}",
  "portion": {{
    "servings": {p.servings},
    "grams_total": {p.grams_total},
    "items_count": {p.items_count if p.items_count is not None else None},
    "household": {f'"{p.household}"' if p.household else None},
    "confidence": {p.confidence},
    "assumptions": {p.assumptions}
  }},
  "calories_kcal": <number>,
  "macros": {{
    "protein": {{"name":"protein","amount":<number>,"unit":"g","per_100g_amount":<number or null>,"daily_value_percent":<number or null>}},
    "carbohydrate": {{"name":"carbohydrate","amount":<number>,"unit":"g","per_100g_amount":<number or null>,"daily_value_percent":<number or null>}},
    "fat": {{"name":"fat","amount":<number>,"unit":"g","per_100g_amount":<number or null>,"daily_value_percent":<number or null>}},
    "fiber": {{"name":"fiber","amount":<number>,"unit":"g","per_100g_amount":<number or null>,"daily_value_percent":<number or null>}},
    "sugar": {{"name":"sugar","amount":<number>,"unit":"g","per_100g_amount":<number or null>,"daily_value_percent":<number or null>}}
  }},
  "micros": {{
    "sodium": {{"name":"sodium","amount":<number>,"unit":"mg","per_100g_amount":<number or null>,"daily_value_percent":<number or null>}},
    "potassium": {{"name":"potassium","amount":<number>,"unit":"mg","per_100g_amount":<number or null>,"daily_value_percent":<number or null>}}
  }},
  "vitamins": {{}},
  "minerals": {{}},
  "ingredients_guess": ["..."],
  "allergens_guess": ["..."],
  "data_sources": ["..."],
  "notes": ["..."]
}}

Rules:
- ONLY JSON.
- amount must be numeric (no "2.7g").
- If include_per_100g is false, set every per_100g_amount = null.
- If unknown, use 0.
""".strip()

# ---------------------------
# JSON helpers + Normalizers
# ---------------------------
_NUM_RE = re.compile(r"-?\d+(\.\d+)?")

def _to_float(v: Any, default: float) -> float:
    if v is None:
        return default
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUM_RE.search(v.replace(",", ""))
        return float(m.group(0)) if m else default
    return default

def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUM_RE.search(v.replace(",", ""))
        return float(m.group(0)) if m else None
    return None

def model_json_or_400(text: str) -> Dict[str, Any] | List[Any]:
    try:
        return extract_json_object(text)
    except ModelJSONError as e:
        # SUPER useful during debugging on Render logs
        print("MODEL_RAW_OUTPUT:\n", (text or "")[:2000])
        raise HTTPException(status_code=400, detail=str(e))

async def force_json_with_gemini(schema_hint: str, text: str) -> dict:
    repair_system = (
        "You are a JSON repair tool. "
        "Return ONLY valid JSON. No markdown, no extra text."
    )
    repair_prompt = f"""
The following text was supposed to be JSON but is invalid/truncated.

Goal schema:
{schema_hint}

Input:
{text}

Return ONLY the corrected JSON. If impossible, return {{}}.
""".strip()

    repaired = await gemini.generate_text(
        model=settings.gemini_portion_model,  # gemini flash is fine
        system=repair_system,
        prompt=repair_prompt,
        temperature=0.0,
        max_output_tokens=1200,
    )

    obj = extract_json_object(repaired)
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail="Repair model did not return a JSON object")
    return obj

def normalize_identify_obj(raw: Any) -> Dict[str, Any]:
    """
    Handles model returning:
    - Correct schema
    - A list of detections [{label:..}, ...]
    - A single detection {label:..}
    - chosen as string
    """
    # Case: list of detections
    if isinstance(raw, list):
        # pick first label-like thing
        label = None
        for it in raw:
            if isinstance(it, dict):
                label = it.get("label") or it.get("name") or it.get("food")
                if label:
                    break
        label = label or "unknown"
        cand = {
            "name": str(label),
            "confidence": 0.6,
            "normalized_name": str(label),
            "cuisine": None,
            "is_packaged": None,
            "notes": "normalized from detection list",
        }
        return {"candidates": [cand], "chosen": cand}

    # Case: single dict but not your schema (detection object)
    if isinstance(raw, dict) and ("candidates" not in raw or "chosen" not in raw):
        label = raw.get("label") or raw.get("name") or raw.get("food") or "unknown"
        cand = {
            "name": str(label),
            "confidence": 0.6,
            "normalized_name": str(label),
            "cuisine": None,
            "is_packaged": None,
            "notes": "normalized from detection dict",
        }
        return {"candidates": [cand], "chosen": cand}

    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="Invalid identify output type")

    obj = dict(raw)
    candidates = obj.get("candidates") or []
    chosen = obj.get("chosen")

    # chosen may be string
    if isinstance(chosen, str):
        chosen_lower = chosen.strip().lower()
        match = None
        for c in candidates:
            if isinstance(c, dict):
                name = str(c.get("name", "")).lower()
                norm = str(c.get("normalized_name", "")).lower()
                if chosen_lower in (name, norm):
                    match = c
                    break
        if match is None:
            match = {
                "name": chosen,
                "confidence": 0.5,
                "normalized_name": chosen,
                "cuisine": None,
                "is_packaged": None,
                "notes": "chosen returned as string; synthesized",
            }
            candidates.append(match)
        obj["chosen"] = match
        obj["candidates"] = candidates

    return obj

def normalize_portion_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = dict(obj or {})
    obj.setdefault("servings", 1.0)
    obj.setdefault("grams_total", 0.0)
    obj.setdefault("items_count", None)
    obj.setdefault("household", None)
    obj.setdefault("confidence", 0.6)
    obj.setdefault("assumptions", [])

    # household must be str|None
    h = obj.get("household")
    if h is None:
        obj["household"] = None
    elif not isinstance(h, str):
        obj["household"] = str(h)

    # assumptions list[str]
    a = obj.get("assumptions")
    if a is None:
        obj["assumptions"] = []
    elif isinstance(a, list):
        obj["assumptions"] = [str(x) for x in a]
    elif isinstance(a, str):
        obj["assumptions"] = [x.strip() for x in a.split(",") if x.strip()]
    else:
        obj["assumptions"] = [str(a)]

    obj["servings"] = _to_float(obj.get("servings"), 1.0)
    obj["grams_total"] = _to_float(obj.get("grams_total"), 0.0)
    obj["items_count"] = _to_float_or_none(obj.get("items_count"))

    conf = _to_float(obj.get("confidence"), 0.6)
    obj["confidence"] = max(0.0, min(1.0, conf))
    return obj

def _parse_amount_unit(v: Any, default_unit: str) -> Tuple[float, str]:
    """
    Converts "2.7g" -> (2.7,"g"), "36 mg" -> (36,"mg"), 12 -> (12,default_unit)
    """
    if v is None:
        return 0.0, default_unit
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v), default_unit
    s = str(v).strip().lower()

    amount = _to_float(s, 0.0)
    unit = default_unit
    if "mg" in s:
        unit = "mg"
    elif "kcal" in s:
        unit = "kcal"
    elif "g" in s:
        unit = "g"
    return amount, unit

def normalize_nutrients_obj(obj: Dict[str, Any], req: NutrientsRequest) -> Dict[str, Any]:
    """
    Makes Sonar/Gemini output conform to NutrientsResponse.
    Handles cases where model returns:
    - portion as string
    - nutrient items as strings like "2.7g"
    - ingredients/allergens/notes as comma-separated strings
    """
    obj = dict(obj or {})

    # food_name alias
    if "food_name" not in obj:
        obj["food_name"] = obj.get("food") or obj.get("name") or req.food_name

    # portion must be object
    portion = obj.get("portion")
    if portion is None or isinstance(portion, str):
        # if string, keep as household note but use req.portion as base
        obj["portion"] = req.portion.model_dump()
        if isinstance(portion, str):
            obj["portion"]["household"] = portion
    elif isinstance(portion, dict):
        portion = normalize_portion_obj(portion)
        # ensure required keys exist
        portion.setdefault("confidence", 0.6)
        portion.setdefault("assumptions", [])
        obj["portion"] = portion
    else:
        obj["portion"] = req.portion.model_dump()

    obj.setdefault("calories_kcal", 0.0)
    obj["calories_kcal"] = _to_float(obj.get("calories_kcal"), 0.0)

    def norm_group(group_name: str, default_unit: str) -> Dict[str, Any]:
        grp = obj.get(group_name) or {}
        out: Dict[str, Any] = {}
        if isinstance(grp, dict):
            for k, v in grp.items():
                if isinstance(v, dict):
                    # coerce numeric fields if they arrived as strings
                    amount, unit = _parse_amount_unit(v.get("amount"), default_unit)
                    out[k] = {
                        "name": v.get("name") or k,
                        "amount": amount,
                        "unit": v.get("unit") or unit,
                        "per_100g_amount": _to_float_or_none(v.get("per_100g_amount")),
                        "daily_value_percent": _to_float_or_none(v.get("daily_value_percent")),
                    }
                else:
                    # v is "2.7g" etc
                    amount, unit = _parse_amount_unit(v, default_unit)
                    out[k] = {
                        "name": k,
                        "amount": amount,
                        "unit": unit,
                        "per_100g_amount": None,
                        "daily_value_percent": None,
                    }
        return out

    obj["macros"] = norm_group("macros", "g")
    obj["micros"] = norm_group("micros", "mg")
    obj["vitamins"] = norm_group("vitamins", "mg")
    obj["minerals"] = norm_group("minerals", "mg")

    # normalize list fields
    for field in ("ingredients_guess", "allergens_guess", "notes", "data_sources"):
        v = obj.get(field)
        if v is None:
            continue
        if isinstance(v, list):
            obj[field] = [str(x) for x in v]
        elif isinstance(v, str):
            # split by comma if looks like csv, else wrap
            if "," in v:
                obj[field] = [x.strip() for x in v.split(",") if x.strip()]
            else:
                obj[field] = [v.strip()]
        else:
            obj[field] = [str(v)]

    # enforce include_per_100g toggle
    if not req.include_per_100g:
        for grp in ("macros", "micros", "vitamins", "minerals"):
            if isinstance(obj.get(grp), dict):
                for _, item in obj[grp].items():
                    if isinstance(item, dict):
                        item["per_100g_amount"] = None

    return obj

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
async def health():
    return {"ok": True, "env": getattr(settings, "app_env", "prod")}

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
    if req.mode != "text":
        raise HTTPException(status_code=422, detail="For mode=image use /v1/food/identify-image")
    if not req.text:
        raise HTTPException(status_code=422, detail="text is required for mode=text")

    out = await gemini.generate_text(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_text(req.text, req.hints),
        temperature=0.2,
        max_output_tokens=700,
    )
    raw = model_json_or_400(out)
    obj = normalize_identify_obj(raw)
    return IdentifyResponse.model_validate(obj)

@app.post("/v1/food/identify-image", response_model=IdentifyResponse)
async def identify_food_image(
    file: UploadFile = File(...),
    hints: Optional[str] = Form(default=None),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    hints_list = [h.strip() for h in (hints or "").split(",") if h.strip()] or None

    out = await gemini.generate_with_image(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_image(hints_list),
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        temperature=0.2,
        max_output_tokens=700,
    )

    raw = model_json_or_400(out)
    obj = normalize_identify_obj(raw)
    return IdentifyResponse.model_validate(obj)

@app.post("/v1/food/portion", response_model=PortionResponse)
async def estimate_portion(req: PortionRequest):
    if req.mode != "text":
        raise HTTPException(status_code=422, detail="For mode=image use /v1/food/portion-image")

    out = await gemini.generate_text(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_text(
            food_name=req.food_name,
            servings=req.assumed_servings or 1.0,
            household=req.household_measure,
            ctx=req.text_context,
        ),
        temperature=0.2,
        max_output_tokens=700,
    )

    raw = model_json_or_400(out)
    raw = normalize_portion_obj(raw if isinstance(raw, dict) else {})
    portion = PortionEstimate.model_validate(raw)

    # safety net (never break pipeline)
    if portion.grams_total <= 0:
        portion = PortionEstimate(
            servings=max(1.0, portion.servings),
            grams_total=100.0,
            household=req.household_measure or "1 serving (default 100g)",
            confidence=0.3,
            assumptions=["Defaulted to 100g because model output was uncertain"]
        )

    return PortionResponse(food_name=req.food_name, portion=portion)

@app.post("/v1/food/portion-image", response_model=PortionResponse)
async def estimate_portion_image(
    food_name: str = Form(...),
    file: UploadFile = File(...),
    text_context: Optional[str] = Form(default=None),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    out = await gemini.generate_with_image(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_image(food_name=food_name, ctx=text_context),
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        temperature=0.2,
        max_output_tokens=700,
    )

    raw = model_json_or_400(out)
    raw = normalize_portion_obj(raw if isinstance(raw, dict) else {})
    portion = PortionEstimate.model_validate(raw)

    if portion.grams_total <= 0:
        portion = PortionEstimate(
            servings=max(1.0, portion.servings),
            grams_total=100.0,
            household="1 serving (default 100g)",
            confidence=0.3,
            assumptions=["Defaulted to 100g because model output was uncertain"]
        )

    return PortionResponse(food_name=food_name, portion=portion)

@app.post("/v1/food/nutrients", response_model=NutrientsResponse)
async def nutrients(req: NutrientsRequest):
    try:
        if req.portion is None:
            raise HTTPException(
                status_code=422,
                detail="portion is required. Call /v1/food/portion first."
            )

        out = await pplx.chat(
            model=settings.pplx_sonar_model,
            system=NUTRIENTS_SYSTEM,
            user=nutrients_prompt(req),
            temperature=0.2,
            max_tokens=1600,
        )

        try:
            raw = model_json_or_400(out)
        except HTTPException:
            raw = await force_json_with_gemini(
                "NutrientsResponse schema",
                out
            )

        raw = normalize_nutrients_obj(raw, req)
        return NutrientsResponse.model_validate(raw)

    except HTTPException:
        raise

    except Exception as e:
        # 🔥 FINAL SAFETY NET
        return ORJSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": str(e),
            },
        )

@app.post("/v1/food/analyze", response_model=AnalyzeResponse)
async def analyze_text(req: AnalyzeTextRequest):
    # 1) Identify
    identify_out = await gemini.generate_text(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_text(req.text, req.hints),
        temperature=0.2,
        max_output_tokens=700,
    )
    identify_raw = model_json_or_400(identify_out)
    identify_obj = normalize_identify_obj(identify_raw)
    identify_res = IdentifyResponse.model_validate(identify_obj)
    chosen = identify_res.chosen.normalized_name or identify_res.chosen.name

    # 2) Portion (text)
    portion_out = await gemini.generate_text(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_text(
            food_name=chosen,
            servings=1.0,
            household=None,
            ctx=req.text,
        ),
        temperature=0.2,
        max_output_tokens=700,
    )
    portion_raw = model_json_or_400(portion_out)
    portion_raw = normalize_portion_obj(portion_raw if isinstance(portion_raw, dict) else {})
    portion_est = PortionEstimate.model_validate(portion_raw)

    if portion_est.grams_total <= 0:
        portion_est = PortionEstimate(
            servings=1.0,
            grams_total=100.0,
            household="1 serving (default 100g)",
            confidence=0.3,
            assumptions=["Defaulted to 100g because model output was uncertain"]
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
        max_tokens=900
    )

    try:
        nutrients_raw = model_json_or_400(nutrients_out)
        if not isinstance(nutrients_raw, dict):
            raise HTTPException(status_code=400, detail="Nutrients output not an object")
    except HTTPException:
        nutrients_raw = await force_json_with_gemini("NutrientsResponse schema", nutrients_out)

    nutrients_raw = normalize_nutrients_obj(nutrients_raw, nreq)
    nutrients_res = NutrientsResponse.model_validate(nutrients_raw)

    return AnalyzeResponse(
        identify=identify_res,
        portion=portion_res,
        nutrients=nutrients_res,
        cost_tier={"identify": "$", "portion": "$$", "nutrients": "$$$$"},
    )

@app.post("/v1/food/analyze-image", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    hints: Optional[str] = Form(default=None),
    region: str = Form(default="IN"),
    include_per_100g: bool = Form(default=True),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    hints_list = [h.strip() for h in (hints or "").split(",") if h.strip()] or None

    # 1) Identify image
    identify_out = await gemini.generate_with_image(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_image(hints_list),
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        temperature=0.2,
        max_output_tokens=700,
    )
    identify_raw = model_json_or_400(identify_out)
    identify_obj = normalize_identify_obj(identify_raw)
    identify_res = IdentifyResponse.model_validate(identify_obj)
    chosen = identify_res.chosen.normalized_name or identify_res.chosen.name

    # 2) Portion image
    portion_out = await gemini.generate_with_image(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_image(food_name=chosen, ctx=None),
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        temperature=0.2,
        max_output_tokens=700,
    )
    portion_raw = model_json_or_400(portion_out)
    portion_raw = normalize_portion_obj(portion_raw if isinstance(portion_raw, dict) else {})
    portion_est = PortionEstimate.model_validate(portion_raw)

    if portion_est.grams_total <= 0:
        portion_est = PortionEstimate(
            servings=1.0,
            grams_total=100.0,
            household="1 serving (default 100g)",
            confidence=0.3,
            assumptions=["Defaulted to 100g because model output was uncertain"]
        )
    portion_res = PortionResponse(food_name=chosen, portion=portion_est)

    # 3) Nutrients
    nreq = NutrientsRequest(
        food_name=chosen,
        portion=portion_est,
        region=region,
        include_per_100g=include_per_100g,
    )

    nutrients_out = await pplx.chat(
        model=settings.pplx_sonar_model,
        system=NUTRIENTS_SYSTEM,
        user=nutrients_prompt(nreq),
        temperature=0.2,
        max_tokens=1600,
        search_recency_filter="month",
    )

    try:
        nutrients_raw = model_json_or_400(nutrients_out)
        if not isinstance(nutrients_raw, dict):
            raise HTTPException(status_code=400, detail="Nutrients output not an object")
    except HTTPException:
        nutrients_raw = await force_json_with_gemini("NutrientsResponse schema", nutrients_out)

    nutrients_raw = normalize_nutrients_obj(nutrients_raw, nreq)
    nutrients_res = NutrientsResponse.model_validate(nutrients_raw)

    return AnalyzeResponse(
        identify=identify_res,
        portion=portion_res,
        nutrients=nutrients_res,
        cost_tier={"identify": "$$", "portion": "$$", "nutrients": "$$$$"},
    )
