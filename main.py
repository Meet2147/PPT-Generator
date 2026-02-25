

import os
import re
import uuid
import random
import asyncio
from io import BytesIO
from pathlib import Path

import requests
import g4f
from g4f.client import Client
from pptx import Presentation
from pptx.util import Inches

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------------------
# Prompt Template
# ---------------------------
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

# ---------------------------
# Clients + App Setup
# ---------------------------
# Keep g4f only for images (you asked to move text gen to Sonar)
g4f_client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("GeneratedPresentations").mkdir(exist_ok=True)
Path("Cache").mkdir(exist_ok=True)
Path("Designs").mkdir(exist_ok=True)

# ---------------------------
# Perplexity (Sonar) minimal client
# ---------------------------
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_URL = "https://api.perplexity.ai/chat/completions"
PPLX_MODEL = "sonar-pro"  # tokens-only (no *-online); switch to "sonar" if you prefer

def _perplexity_chat(messages, model=PPLX_MODEL, max_tokens=1200, temperature=0.4):
    if not PPLX_API_KEY:
        raise RuntimeError(
            "Missing PPLX_API_KEY. Set it in your environment before running the server."
        )
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        # ensure token-only text generation (no images or search)
        "return_images": False,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(PPLX_URL, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Perplexity API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Perplexity follows OpenAI-like schema
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not isinstance(content, str):
        content = str(content or "")
    return content

# ---------------------------
# Helpers
# ---------------------------
def _local_fallback_presentation(topic: str) -> str:
    topic_title = (topic or "Your Topic").title()
    return f"""#Title: {topic_title}

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
#Content: Brief overview of {topic}. Scope and goal of this talk.
#Image: relevant illustration of the topic

#Slide: 3
#Header: why it matters
#Content: Impact, use-cases, and benefits in simple terms.
#Image: simple infographic with benefits

#Slide: 4
#Header: key points
#Content: 3–5 core ideas about {topic}, kept concise.
#Image: icons representing each key idea

#Slide: 5
#Header: examples
#Content: 2–3 quick examples showing the concept in action.
#Image: storyboard-like illustration

#Slide: 6
#Header: tips & pitfalls
#Content: Do's and don'ts for better results.
#Image: checklist illustration

#Slide: 7
#Header: summary
#Content: Short recap and practical next steps.
#Image: an illustration of a person reviewing a summary report

#Slide: END
"""

async def generate_image_async(prompt: str) -> str | None:
    if not prompt or prompt.strip().lower() == "none":
        return None
    try:
        resp = await asyncio.to_thread(
            g4f_client.images.generate,
            model="flux",             # keep if working for you; otherwise stub this out
            prompt=prompt,
            response_format="url"
        )
        if hasattr(resp, "data") and resp.data:
            return resp.data[0].url
    except Exception as e:
        print(f"[generate_image_async] Error generating image: {e}")
    return None

async def generate_presentation_text_async(user_input: str) -> str:
    """Use Perplexity Sonar (no web/online) to generate strict deck content."""
    system_msg = PROMPT_TEMPLATE
    user_msg = f"The user wants a presentation about {user_input}"

    # Attempt 1
    try:
        content = await asyncio.to_thread(
            _perplexity_chat,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            PPLX_MODEL,
            1400,   # allow enough tokens for at least 7 slides
            0.35,
        )
        if "#Slide:" not in content or "#Title:" not in content:
            raise ValueError("Model response missing required markers (#Title:/#Slide:).")
        return content
    except Exception as e:
        print(f"[generate_presentation_text_async] Sonar first attempt failed: {e}")

    # Attempt 2 with stricter instruction
    try:
        strict_prompt = "ONLY output in the exact template below. Do NOT add extra text.\n\n" + PROMPT_TEMPLATE
        content = await asyncio.to_thread(
            _perplexity_chat,
            [
                {"role": "system", "content": strict_prompt},
                {"role": "user", "content": user_msg},
            ],
            PPLX_MODEL,
            1400,
            0.2,
        )
        if "#Slide:" not in content or "#Title:" not in content:
            raise ValueError("Model response missing required markers on retry.")
        return content
    except Exception as e:
        print(f"[generate_presentation_text_async] Sonar retry failed: {e}")
        return ""

async def create_presentation_async(text_content: str, design_number: int, presentation_name: str):
    # Open template (fallback to Design-1, else blank)
    template_path = Path(f"Designs/Design-{design_number}.pptx")
    if not template_path.exists():
        template_path = Path("Designs/Design-1.pptx")

    if template_path.exists():
        try:
            presentation = Presentation(str(template_path))
        except Exception as e:
            print(f"[create_presentation_async] Could not open template, using blank. Error: {e}")
            presentation = Presentation()
    else:
        presentation = Presentation()

    # State
    slide_count = 0
    slide_title = ""
    slide_content = ""
    slide_image_prompt = None
    last_slide_layout_index = -1
    first_time = True

    lines = [ln.rstrip("\n") for ln in text_content.splitlines()]
    i = 0

    # Layout helpers
    def _safe_layout(idx: int) -> int:
        if len(presentation.slide_layouts) == 0:
            return 0
        return max(0, min(idx, len(presentation.slide_layouts) - 1))

    slide_layout_index = _safe_layout(1 if len(presentation.slide_layouts) > 1 else 0)
    slide_placeholder_index = 2 if slide_layout_index == 8 else (1 if len(presentation.slide_layouts) > 1 else 0)

    # Commit helper
    async def commit_slide():
        nonlocal slide_title, slide_content, slide_image_prompt, slide_count
        if slide_count > 0 and (slide_title or slide_content):
            slide = presentation.slides.add_slide(presentation.slide_layouts[_safe_layout(slide_layout_index)])
            # Title
            try:
                slide.shapes.title.text = slide_title
            except Exception:
                try:
                    slide.placeholders[0].text = slide_title
                except Exception:
                    pass
            # Body
            try:
                body_shape = slide.shapes.placeholders[slide_placeholder_index]
                if hasattr(body_shape, "text_frame"):
                    body_shape.text_frame.text = slide_content
                else:
                    body_shape.text = slide_content
            except Exception:
                pass

            # Image (if any)
            if slide_image_prompt:
                img_url = await generate_image_async(slide_image_prompt)
                if img_url:
                    try:
                        resp = await asyncio.to_thread(requests.get, img_url, timeout=20)
                        resp.raise_for_status()
                        image_stream = BytesIO(resp.content)
                        slide.shapes.add_picture(image_stream, Inches(5), Inches(1.5), width=Inches(4))
                    except Exception as e:
                        print(f"[create_presentation_async] Could not add image: {e}")

    # Parse
    while i < len(lines):
        line = lines[i]

        if line.startswith("#Title:"):
            slide_title_text = line.replace("#Title:", "").strip()
            title_layout = _safe_layout(0)
            slide = presentation.slides.add_slide(presentation.slide_layouts[title_layout])
            try:
                slide.shapes.title.text = slide_title_text
            except Exception:
                try:
                    slide.placeholders[0].text = slide_title_text
                except Exception:
                    pass
            i += 1
            continue

        if line.startswith("#Slide:"):
            await commit_slide()
            slide_content = ""
            slide_image_prompt = None
            slide_title = ""
            slide_count += 1

            if len(presentation.slide_layouts) >= 9:
                layout_choices = [1, 7, 8]
            else:
                layout_choices = [1] if len(presentation.slide_layouts) > 1 else [0]

            if first_time:
                slide_layout_index = _safe_layout(1 if len(presentation.slide_layouts) > 1 else 0)
                slide_placeholder_index = 2 if slide_layout_index == 8 else (1 if len(presentation.slide_layouts) > 1 else 0)
                first_time = False
            else:
                next_idx = last_slide_layout_index
                tries = 0
                while next_idx == last_slide_layout_index and tries < 10:
                    next_idx = random.choice(layout_choices)
                    tries += 1
                slide_layout_index = _safe_layout(next_idx)
                slide_placeholder_index = 2 if slide_layout_index == 8 else (1 if len(presentation.slide_layouts) > 1 else 0)

            last_slide_layout_index = slide_layout_index
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

    # Final commit and save
    await commit_slide()

    out_dir = Path("GeneratedPresentations")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{presentation_name}.pptx"
    await asyncio.to_thread(presentation.save, str(file_path))
    return str(file_path)

# ---------------------------
# Routes
# ---------------------------
@app.get("/auto-download/{filename}")
async def download_file(filename: str):
    file_path = Path("GeneratedPresentations") / filename
    if file_path.is_file():
        return FileResponse(
            file_path,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=filename
        )
    else:
        return Response(content="File not found", status_code=404)

@app.post("/generate-ppt/")
async def generate_ppt(user_input: str):
    """
    POST /generate-ppt/?user_input=Your topic here
    """
    try:
        user_text = user_input or ""
        last_char = user_text[-1] if user_text else ""
        input_string = re.sub(r"[^\w\s\.\-\(\)]", "", user_text).replace("\n", "")
        design_number = 2

        # "topic 3" => design 3
        if last_char.isdigit():
            try:
                design_number = int(last_char)
                input_string = (
                    user_text[:-2].strip()
                    if len(user_text) >= 2 and user_text[-2].isspace()
                    else user_text[:-1].strip()
                )
                input_string = re.sub(r"[^\w\s\.\-\(\)]", "", input_string).replace("\n", "")
            except Exception:
                design_number = 2

        if design_number > 7 or design_number == 0:
            design_number = 1

        filename = f"{input_string}_{uuid.uuid4().hex}"

        # Generate with Sonar → strict format
        presentation_text = await generate_presentation_text_async(input_string)

        used_fallback = False
        if not presentation_text:
            print("[/generate-ppt] Sonar did not return valid content; using local fallback.")
            presentation_text = _local_fallback_presentation(input_string)
            used_fallback = True

        file_path = await create_presentation_async(presentation_text, design_number, filename)

        # TODO: change to your deployment host if different
        auto_download_url = f"https://ppt-generator-mtu0.onrender.com/auto-download/{Path(file_path).name}"

        return {
            "status": "success",
            "url": auto_download_url,
            "fallback_used": used_fallback
        }

    except Exception as e:
        return {"status": "error", "message": f"Server exception: {e}"}


