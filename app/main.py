import os
import re
import uuid
import random
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import re
from typing import Any, Dict
import requests
from pptx import Presentation
from pptx.util import Inches

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import ORJSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.models import (
    IdentifyRequest, IdentifyResponse,
    PortionRequest, PortionEstimate, PortionResponse,
    NutrientsRequest, NutrientsResponse,
    AnalyzeResponse,
    PPTGenerateRequest, PPTGenerateResponse,
)
from app.clients.gemini import GeminiClient
from app.clients.perplexity import PerplexityClient
from app.utils import extract_json_object, ModelJSONError

# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI(
    title="Food + PPT API",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini = GeminiClient(api_key=settings.gemini_api_key)
pplx = PerplexityClient(api_key=settings.pplx_api_key)

# --------------------------------------------------
# Storage dirs (Render disk recommended if you want persistence)
# --------------------------------------------------

GENERATED_DIR = Path(os.getenv("GENERATED_DIR", "GeneratedPresentations"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "Cache"))
DESIGNS_DIR = Path(os.getenv("DESIGNS_DIR", "Designs"))

GENERATED_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
DESIGNS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Helpers: JSON parsing
# --------------------------------------------------

def model_json_or_400(text: str) -> Union[Dict[str, Any], List[Any]]:
    try:
        return extract_json_object(text)
    except ModelJSONError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --------------------------------------------------
# Food prompts
# --------------------------------------------------

IDENTIFY_SYSTEM = """You are a food identification classifier.
Return ONLY valid JSON. No markdown. No extra text.
Return EXACT schema with keys: candidates, chosen.
"""

def identify_user_prompt_text(text: str, hints: Optional[List[str]]) -> str:
    return f"""
Return ONE JSON object:
{{
  "candidates":[
    {{
      "name":"<as user said>",
      "confidence":0.0,
      "normalized_name":"<canonical food name>",
      "cuisine":null,
      "is_packaged":null,
      "notes":null
    }}
  ],
  "chosen": {{
    "name":"...",
    "confidence":0.0,
    "normalized_name":"...",
    "cuisine":null,
    "is_packaged":null,
    "notes":null
  }}
}}
Rules:
- If multiple foods, include up to 3 candidates.
- chosen MUST be an object (NOT a string).
User text: {text}
Hints: {hints or []}
""".strip()

def identify_user_prompt_image(hints: Optional[List[str]]) -> str:
    return f"""
Look at the image and identify the FOOD ITEM(S).
Return ONLY ONE JSON object. If multiple foods, include them in candidates array.

Schema:
{{
  "candidates":[
    {{
      "name":"<short food name>",
      "confidence":0.0,
      "normalized_name":"<canonical food name>",
      "cuisine":null,
      "is_packaged":null,
      "notes":null
    }}
  ],
  "chosen": {{
    "name":"...",
    "confidence":0.0,
    "normalized_name":"...",
    "cuisine":null,
    "is_packaged":null,
    "notes":null
  }}
}}
Hints: {hints or []}
""".strip()

IDENTIFY_REPAIR_SYSTEM = """You are a food identifier.
Return ONLY ONE valid JSON object in IdentifyResponse schema (candidates + chosen).
No markdown. No extra text.
"""

def identify_repair_prompt(detections: list, hints: Optional[List[str]]) -> str:
    return f"""
You received a JSON array from an image model (could be detections like plate/bowl/food labels).
Decide the actual FOOD item(s) present and output IdentifyResponse schema.

Detections JSON array:
{detections}

Hints: {hints or []}

Rules:
- Ignore non-food objects unless they help identify the dish.
- Output candidates (up to 3) and chosen (best).
- normalized_name should be usable for nutrition lookup.
""".strip()

PORTION_SYSTEM = """You estimate portion sizes.
Return ONLY valid JSON. No markdown. No extra text.
"""

def portion_prompt_text(food_name: str, servings: float, household: Optional[str], ctx: Optional[str]) -> str:
    return f"""
Estimate portion for: {food_name}
Context: {ctx or ""}
User provided: servings={servings}, household_measure={household or ""}

Return JSON:
{{
  "servings": <float>,
  "grams_total": <float>,
  "items_count": <float or null>,
  "household": "<string or null>",
  "confidence": 0.0,
  "assumptions": ["..."]
}}
Rules:
- household MUST be string or null.
""".strip()

def portion_prompt_image(food_name: str, ctx: Optional[str]) -> str:
    return f"""
Estimate portion size from the image for: {food_name}
Extra context: {ctx or ""}

Return JSON:
{{
  "servings": <float>,
  "grams_total": <float>,
  "items_count": <float or null>,
  "household": "<string or null>",
  "confidence": 0.0,
  "assumptions": ["..."]
}}
Rules:
- household MUST be string or null.
""".strip()

NUTRIENTS_SYSTEM = """You are a nutrition engine.
Return ONLY valid JSON. No markdown. No extra text.
Output MUST match NutrientsResponse schema exactly.
All nutrient values must be objects (not strings like "2.7g").
"""

def nutrients_prompt(req: NutrientsRequest) -> str:
    p = req.portion
    return f"""
Return ONE JSON object with EXACT keys:
food_name, portion, calories_kcal, macros, micros, vitamins, minerals,
ingredients_guess, allergens_guess, data_sources, notes.

food_name: "{req.food_name}"

portion MUST be an object with:
servings, grams_total, items_count, household, confidence, assumptions

For ALL nutrients, each entry MUST be:
{{"name":"...","amount":<float>,"unit":"g|mg|mcg|kcal","per_100g_amount":<float or null>,"daily_value_percent":<float or null>}}

ingredients_guess/allergens_guess/notes MUST be arrays.

Portion reference:
servings={p.servings}
grams_total={p.grams_total}
items_count={p.items_count}
household={p.household}
confidence={p.confidence}
assumptions={p.assumptions}
""".strip()

# --------------------------------------------------
# Normalizers
# --------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(\.\d+)?")

def _to_float(v: Any, default: float) -> float:
    if v is None:
        return default
    if isinstance(v, bool):
        # avoid bool becoming 0/1 silently
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUM_RE.search(v.replace(",", ""))
        return float(m.group(0)) if m else default
    return default

def _to_float_or_none(v: Any) -> float | None:
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

def normalize_portion_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = dict(obj or {})

    obj.setdefault("servings", 1.0)
    obj.setdefault("grams_total", 0.0)
    obj.setdefault("items_count", None)
    obj.setdefault("household", None)
    obj.setdefault("confidence", 0.6)
    obj.setdefault("assumptions", [])

    # household must be string or None
    h = obj.get("household")
    if h is None:
        obj["household"] = None
    elif not isinstance(h, str):
        obj["household"] = str(h)

    # assumptions must be list[str]
    a = obj.get("assumptions")
    if a is None:
        obj["assumptions"] = []
    elif isinstance(a, list):
        obj["assumptions"] = [str(x) for x in a]
    elif isinstance(a, str):
        obj["assumptions"] = [x.strip() for x in a.split(",") if x.strip()]
    else:
        obj["assumptions"] = [str(a)]

    # Coerce numerics safely (handles "40g", "about 80 grams", "2 rotis")
    obj["servings"] = _to_float(obj.get("servings"), 1.0)
    obj["grams_total"] = _to_float(obj.get("grams_total"), 0.0)
    obj["items_count"] = _to_float_or_none(obj.get("items_count"))

    conf = _to_float(obj.get("confidence"), 0.6)
    obj["confidence"] = max(0.0, min(1.0, conf))

    return obj

def _parse_amount_unit(s: str):
    m = re.search(r"([-+]?\d*\.?\d+)\s*([a-zA-Zµ]+)?", (s or "").strip())
    if not m:
        return 0.0, ""
    return float(m.group(1)), (m.group(2) or "").strip()

def _normalize_nutrient_item(name: str, v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        v.setdefault("name", name)
        v.setdefault("per_100g_amount", None)
        v.setdefault("daily_value_percent", None)
        try:
            v["amount"] = float(v.get("amount", 0) or 0)
        except Exception:
            v["amount"] = 0.0
        if not isinstance(v.get("unit"), str) or not v["unit"]:
            v["unit"] = "g"
        return v

    if isinstance(v, str):
        amt, unit = _parse_amount_unit(v)
        return {
            "name": name,
            "amount": float(amt),
            "unit": unit or "g",
            "per_100g_amount": None,
            "daily_value_percent": None,
        }

    return {"name": name, "amount": 0.0, "unit": "g", "per_100g_amount": None, "daily_value_percent": None}

def normalize_nutrients_obj(obj: Dict[str, Any], req: NutrientsRequest) -> Dict[str, Any]:
    obj["food_name"] = obj.get("food_name") or obj.get("food") or req.food_name

    # Portion must be dict and must include confidence/assumptions
    if isinstance(obj.get("portion"), dict):
        obj["portion"] = normalize_portion_obj(obj["portion"])
    else:
        obj["portion"] = normalize_portion_obj(req.portion.model_dump())

    cal = obj.get("calories_kcal") or obj.get("calories")
    if isinstance(cal, str):
        amt, _ = _parse_amount_unit(cal)
        obj["calories_kcal"] = float(amt)
    else:
        try:
            obj["calories_kcal"] = float(cal or 0.0)
        except Exception:
            obj["calories_kcal"] = 0.0

    for section in ("macros", "micros", "vitamins", "minerals"):
        raw = obj.get(section)
        if not isinstance(raw, dict):
            raw = {}
        fixed = {}
        for k, v in raw.items():
            key = str(k).strip().lower().replace(" ", "_")
            fixed[key] = _normalize_nutrient_item(key, v)
        obj[section] = fixed

    for lf in ("ingredients_guess", "allergens_guess", "notes"):
        v = obj.get(lf)
        if isinstance(v, str):
            obj[lf] = [x.strip() for x in v.split(",") if x.strip()]
        elif isinstance(v, list):
            obj[lf] = [str(x) for x in v]
        else:
            obj[lf] = []

    ds = obj.get("data_sources")
    if isinstance(ds, str):
        obj["data_sources"] = [ds]
    elif isinstance(ds, list):
        obj["data_sources"] = [str(x) for x in ds]
    else:
        obj["data_sources"] = []

    return obj

async def identify_repair_if_list(raw: Any, hints: Optional[List[str]]) -> Dict[str, Any]:
    if isinstance(raw, list):
        repair_out = await gemini.generate_text(
            model=settings.gemini_classifier_model,
            system=IDENTIFY_REPAIR_SYSTEM,
            prompt=identify_repair_prompt(raw, hints),
            temperature=0.2,
            max_output_tokens=700,
        )
        repaired = model_json_or_400(repair_out)
        if not isinstance(repaired, dict):
            raise HTTPException(400, "Identify repair did not return a JSON object")
        return repaired

    if isinstance(raw, dict):
        return raw

    raise HTTPException(400, "Invalid identify output type")

def normalize_identify_dict(obj: Dict[str, Any]) -> IdentifyResponse:
    candidates = obj.get("candidates") or []
    chosen = obj.get("chosen")

    if isinstance(chosen, str):
        chosen_lower = chosen.strip().lower()
        match = None
        for c in candidates:
            if not isinstance(c, dict):
                continue
            if chosen_lower in (
                str(c.get("name", "")).lower(),
                str(c.get("normalized_name", "")).lower()
            ):
                match = c
                break
        if match is None:
            match = {
                "name": chosen,
                "confidence": 0.5,
                "normalized_name": chosen,
                "cuisine": None,
                "is_packaged": None,
                "notes": "chosen string -> synthesized",
            }
            candidates.append(match)
        obj["chosen"] = match
        obj["candidates"] = candidates

    if isinstance(obj.get("chosen"), dict) and not obj.get("candidates"):
        obj["candidates"] = [obj["chosen"]]

    return IdentifyResponse.model_validate(obj)

# --------------------------------------------------
# Analyze text model
# --------------------------------------------------

class AnalyzeTextRequest(BaseModel):
    text: str
    hints: Optional[List[str]] = None
    region: str = "IN"
    include_per_100g: bool = True

# --------------------------------------------------
# PPT Generator (Sonar text → PPTX)
# --------------------------------------------------

PROMPT_TEMPLATE = """Write a presentation/powerpoint about the user's topic. You only answer with the presentation. Follow the structure of the example.

Notice:
- You do all the presentation text for the user.
- You write the texts no longer than 250 characters!
- You make very short titles!
- You make the presentation easy to understand.
- The presentation has a table of contents.
- The presentation has a summary.
- At least 7 slides.
- For each slide, after the #Content: line, add an #Image: line describing a relevant image that could visually represent the slide's topic.
- If no image is relevant, write #Image: none.

Example! - Stick to this formatting exactly!
#Title: TITLE OF THE PRESENTATION

#Slide: 1
#Header: table of contents
#Content: 1. CONTENT OF THIS POWERPOINT
2. CONTENTS OF THIS POWERPOINT
3. CONTENT OF THIS POWERPOINT
#Image: a 3D illustration of a table of contents in a book

#Slide: 2
#Header: TITLE OF SLIDE
#Content: CONTENT OF THE SLIDE
#Image: relevant illustration description here

#Slide: 3
#Header: TITLE OF SLIDE
#Content: CONTENT OF THE SLIDE
#Image: relevant illustration description here

#Slide: 4
#Header: TITLE OF SLIDE
#Content: CONTENT OF THE SLIDE
#Image: relevant illustration description here

#Slide: 5
#Header: TITLE OF SLIDE
#Content: CONTENT OF THE SLIDE
#Image: relevant illustration description here

#Slide: 6
#Header: TITLE OF SLIDE
#Content: CONTENT OF THE SLIDE
#Image: relevant illustration description here

#Slide: 7
#Header: summary
#Content: CONTENT OF THE SUMMARY
#Image: an illustration of a person reviewing a summary report

#Slide: END"""

def _local_fallback_presentation(topic: str) -> str:
    t = (topic or "Your Topic").title()
    return f"""#Title: {t}

#Slide: 1
#Header: table of contents
#Content: 1. Intro
2. Why it matters
3. Key points
4. Examples
5. Tips
6. Pitfalls
7. Summary
#Image: a 3D illustration of a table of contents in a book

#Slide: 2
#Header: intro
#Content: Brief overview of {topic}. Scope and goal.
#Image: relevant illustration

#Slide: 3
#Header: why it matters
#Content: Impact, use-cases, and benefits in simple terms.
#Image: simple infographic

#Slide: 4
#Header: key points
#Content: 3–5 core ideas about {topic}.
#Image: icons representing key ideas

#Slide: 5
#Header: examples
#Content: 2–3 quick examples.
#Image: storyboard-like illustration

#Slide: 6
#Header: tips & pitfalls
#Content: Do's and don'ts.
#Image: checklist illustration

#Slide: 7
#Header: summary
#Content: Short recap and next steps.
#Image: an illustration of a person reviewing a summary report

#Slide: END
"""

async def generate_presentation_text_async(topic: str) -> str:
    # IMPORTANT: uses YOUR PerplexityClient (same service instance)
    system = PROMPT_TEMPLATE
    user = f"The user wants a presentation about {topic}"

    try:
        content = await pplx.chat(
            model=settings.pplx_sonar_model,
            system=system,
            user=user,
            temperature=0.35,
            max_tokens=1400,
            search_recency_filter="month",
        )
        if "#Title:" not in content or "#Slide:" not in content:
            return ""
        return content
    except Exception:
        return ""

async def create_presentation_async(text_content: str, design_number: int, presentation_name: str) -> str:
    template_path = DESIGNS_DIR / f"Design-{design_number}.pptx"
    if not template_path.exists():
        template_path = DESIGNS_DIR / "Design-1.pptx"

    if template_path.exists():
        try:
            prs = Presentation(str(template_path))
        except Exception:
            prs = Presentation()
    else:
        prs = Presentation()

    # Layout helpers
    def _safe_layout(idx: int) -> int:
        if len(prs.slide_layouts) == 0:
            return 0
        return max(0, min(idx, len(prs.slide_layouts) - 1))

    slide_layout_index = _safe_layout(1 if len(prs.slide_layouts) > 1 else 0)
    slide_placeholder_index = 1 if len(prs.slide_layouts) > 1 else 0

    slide_title = ""
    slide_content = ""
    slide_image_prompt = None
    slide_count = 0
    last_layout = -1
    first_time = True

    lines = [ln.rstrip("\n") for ln in text_content.splitlines()]
    i = 0

    async def commit_slide():
        nonlocal slide_title, slide_content, slide_image_prompt, slide_count
        if slide_count > 0 and (slide_title or slide_content):
            slide = prs.slides.add_slide(prs.slide_layouts[_safe_layout(slide_layout_index)])

            # title
            try:
                slide.shapes.title.text = slide_title
            except Exception:
                try:
                    slide.placeholders[0].text = slide_title
                except Exception:
                    pass

            # body
            try:
                body = slide.shapes.placeholders[slide_placeholder_index]
                if hasattr(body, "text_frame"):
                    body.text_frame.text = slide_content
                else:
                    body.text = slide_content
            except Exception:
                pass

            # NOTE: Image generation removed for Render stability (g4f is flaky & heavy).
            # You can add it back later behind a feature flag.

    while i < len(lines):
        line = lines[i]

        if line.startswith("#Title:"):
            title = line.replace("#Title:", "").strip()
            slide = prs.slides.add_slide(prs.slide_layouts[_safe_layout(0)])
            try:
                slide.shapes.title.text = title
            except Exception:
                pass
            i += 1
            continue

        if line.startswith("#Slide:"):
            await commit_slide()
            slide_count += 1
            slide_title = ""
            slide_content = ""
            slide_image_prompt = None

            # rotate layouts a bit
            layouts = [1, 7, 8] if len(prs.slide_layouts) >= 9 else ([1] if len(prs.slide_layouts) > 1 else [0])
            if first_time:
                slide_layout_index = _safe_layout(1 if len(prs.slide_layouts) > 1 else 0)
                slide_placeholder_index = 1 if len(prs.slide_layouts) > 1 else 0
                first_time = False
            else:
                nxt = last_layout
                tries = 0
                while nxt == last_layout and tries < 10:
                    nxt = random.choice(layouts)
                    tries += 1
                slide_layout_index = _safe_layout(nxt)
                slide_placeholder_index = 2 if slide_layout_index == 8 else 1

            last_layout = slide_layout_index
            i += 1
            continue

        if line.startswith("#Header:"):
            slide_title = line.replace("#Header:", "").strip()
            i += 1
            continue

        if line.startswith("#Content:"):
            slide_content = line.replace("#Content:", "").strip()
            i += 1
            while i < len(lines) and not lines[i].startswith("#"):
                slide_content += "\n" + lines[i]
                i += 1
            continue

        if line.startswith("#Image:"):
            slide_image_prompt = line.replace("#Image:", "").strip()
            i += 1
            continue

        i += 1

    await commit_slide()

    out_path = GENERATED_DIR / f"{presentation_name}.pptx"
    await asyncio.to_thread(prs.save, str(out_path))
    return str(out_path)

# --------------------------------------------------
# Routes (Meta)
# --------------------------------------------------

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

# --------------------------------------------------
# Food Routes (identify / portion / nutrients / analyze)
# --------------------------------------------------

@app.post("/v1/food/identify", response_model=IdentifyResponse)
async def identify_food(req: IdentifyRequest):
    if req.mode != "text":
        raise HTTPException(422, "Use /v1/food/identify-image for images")
    if not req.text:
        raise HTTPException(422, "text is required")

    out = await gemini.generate_text(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_text(req.text, req.hints),
        temperature=0.2,
        max_output_tokens=900,
    )
    raw = model_json_or_400(out)
    raw = await identify_repair_if_list(raw, req.hints)
    return normalize_identify_dict(raw)

@app.post("/v1/food/identify-image", response_model=IdentifyResponse)
async def identify_food_image(file: UploadFile = File(...), hints: Optional[str] = None):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded image is empty")

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
    raw = await identify_repair_if_list(raw, hints_list)
    return normalize_identify_dict(raw)

@app.post("/v1/food/portion", response_model=PortionResponse)
async def estimate_portion(req: PortionRequest):
    if req.mode != "text":
        raise HTTPException(422, "Use /v1/food/portion-image for images")

    out = await gemini.generate_text(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_text(req.food_name, req.assumed_servings, req.household_measure, req.text_context),
        temperature=0.2,
        max_output_tokens=900,
    )
    raw = model_json_or_400(out)
    if not isinstance(raw, dict):
        raise HTTPException(400, "Portion model returned invalid JSON")
    raw = normalize_portion_obj(raw)
    portion = PortionEstimate.model_validate(raw)
    return PortionResponse(food_name=req.food_name, portion=portion)

@app.post("/v1/food/portion-image", response_model=PortionResponse)
async def estimate_portion_image(
    food_name: str = Query(...),
    file: UploadFile = File(...),
    text_context: Optional[str] = None,
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded image is empty")

    out = await gemini.generate_with_image(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_image(food_name, ctx=text_context),
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        temperature=0.2,
        max_output_tokens=900,
    )
    raw = model_json_or_400(out)
    if not isinstance(raw, dict):
        raise HTTPException(400, "Portion model returned invalid JSON")
    raw = normalize_portion_obj(raw)
    portion = PortionEstimate.model_validate(raw)
    return PortionResponse(food_name=food_name, portion=portion)

@app.post("/v1/food/nutrients", response_model=NutrientsResponse)
async def nutrients(req: NutrientsRequest):
    out = await pplx.chat(
        model=settings.pplx_sonar_model,
        system=NUTRIENTS_SYSTEM,
        user=nutrients_prompt(req),
        temperature=0.2,
        max_tokens=1800,
        search_recency_filter="month",
    )
    raw = model_json_or_400(out)
    if not isinstance(raw, dict):
        raise HTTPException(400, "Nutrients model returned invalid JSON")

    raw = normalize_nutrients_obj(raw, req)

    if not req.include_per_100g:
        for section in ("macros", "micros", "vitamins", "minerals"):
            for _, item in raw.get(section, {}).items():
                if isinstance(item, dict):
                    item["per_100g_amount"] = None

    return NutrientsResponse.model_validate(raw)

class AnalyzeTextRequest(BaseModel):
    text: str
    hints: Optional[List[str]] = None
    region: str = "IN"
    include_per_100g: bool = True

@app.post("/v1/food/analyze", response_model=AnalyzeResponse)
async def analyze_text(req: AnalyzeTextRequest):
    # identify
    identify_out = await gemini.generate_text(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_text(req.text, req.hints),
        temperature=0.2,
        max_output_tokens=900,
    )
    raw_ident = model_json_or_400(identify_out)
    raw_ident = await identify_repair_if_list(raw_ident, req.hints)
    identify = normalize_identify_dict(raw_ident)
    food_name = identify.chosen.normalized_name or identify.chosen.name

    # portion
    portion_out = await gemini.generate_text(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_text(food_name, 1.0, None, req.text),
        temperature=0.2,
        max_output_tokens=900,
    )
    raw_portion = model_json_or_400(portion_out)
    if not isinstance(raw_portion, dict):
        raise HTTPException(400, "Portion model returned invalid JSON")
    raw_portion = normalize_portion_obj(raw_portion)
    portion = PortionEstimate.model_validate(raw_portion)

    # nutrients
    nreq = NutrientsRequest(food_name=food_name, portion=portion, region=req.region, include_per_100g=req.include_per_100g)
    nutrients_res = await nutrients(nreq)

    return AnalyzeResponse(
        identify=identify,
        portion=PortionResponse(food_name=food_name, portion=portion),
        nutrients=nutrients_res,
        cost_tier={"identify": "$", "portion": "$$", "nutrients": "$$$$"},
    )

@app.post("/v1/food/analyze-image", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    hints: Optional[str] = None,
    region: str = "IN",
    include_per_100g: bool = True,
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded image is empty")

    hints_list = [h.strip() for h in (hints or "").split(",") if h.strip()] or None

    # identify (image)
    identify_out = await gemini.generate_with_image(
        model=settings.gemini_classifier_model,
        system=IDENTIFY_SYSTEM,
        prompt=identify_user_prompt_image(hints_list),
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        temperature=0.2,
        max_output_tokens=900,
    )
    raw_ident = model_json_or_400(identify_out)
    raw_ident = await identify_repair_if_list(raw_ident, hints_list)
    identify = normalize_identify_dict(raw_ident)
    food_name = identify.chosen.normalized_name or identify.chosen.name

    # portion (image)
    portion_out = await gemini.generate_with_image(
        model=settings.gemini_portion_model,
        system=PORTION_SYSTEM,
        prompt=portion_prompt_image(food_name, ctx=None),
        image_bytes=image_bytes,
        mime_type=file.content_type or "image/jpeg",
        temperature=0.2,
        max_output_tokens=900,
    )
    raw_portion = model_json_or_400(portion_out)
    if not isinstance(raw_portion, dict):
        raise HTTPException(400, "Portion model returned invalid JSON")
    raw_portion = normalize_portion_obj(raw_portion)
    portion = PortionEstimate.model_validate(raw_portion)

    # nutrients
    nreq = NutrientsRequest(food_name=food_name, portion=portion, region=region, include_per_100g=include_per_100g)
    nutrients_res = await nutrients(nreq)

    return AnalyzeResponse(
        identify=identify,
        portion=PortionResponse(food_name=food_name, portion=portion),
        nutrients=nutrients_res,
        cost_tier={"identify": "$", "portion": "$$", "nutrients": "$$$$"},
    )

# --------------------------------------------------
# PPT Routes (single instance)
# --------------------------------------------------

@app.post("/v1/ppt/generate", response_model=PPTGenerateResponse)
async def generate_ppt(req: PPTGenerateRequest):
    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(422, "topic is required")

    design_number = req.design_number
    if design_number < 1 or design_number > 7:
        design_number = 1

    safe_topic = re.sub(r"[^\w\s\.\-\(\)]", "", topic).replace("\n", "").strip()
    filename = f"{safe_topic[:40]}_{uuid.uuid4().hex}"

    deck_text = await generate_presentation_text_async(safe_topic)
    fallback_used = False
    if not deck_text:
        deck_text = _local_fallback_presentation(safe_topic)
        fallback_used = True

    ppt_path = await create_presentation_async(deck_text, design_number, filename)

    base_url = (settings.public_base_url or "").rstrip("/")
    if not base_url:
        # Render provides your onrender URL; easiest is to set PUBLIC_BASE_URL env var in Render
        # Example: https://your-service.onrender.com
        base_url = "https://ppt-generator-mtu0.onrender.com"

    download_url = f"{base_url}/v1/ppt/download/{Path(ppt_path).name}"

    return PPTGenerateResponse(
        status="success",
        filename=Path(ppt_path).name,
        # download_url=download_url,
        fallback_used=fallback_used,
    )

@app.get("/v1/ppt/download/{filename}")
async def download_ppt(filename: str):
    file_path = GENERATED_DIR / filename
    if not file_path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=filename,
    )
