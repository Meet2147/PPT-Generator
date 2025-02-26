from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import openai
import os
import random
import aiofiles
from pptx import Presentation
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure API Key is set
if not openai_api_key:
    raise ValueError("Missing OpenAI API Key! Please set OPENAI_API_KEY in environment variables.")

openai.api_key = openai_api_key  # Set API key
app = FastAPI()

# Create necessary directories
os.makedirs("GeneratedPresentations", exist_ok=True)
os.makedirs("Cache", exist_ok=True)

# OpenAI Prompt for PowerPoint Content
Prompt = """Write a presentation/powerpoint about the user's topic. You only answer with the presentation. Follow the structure of the example.
- Keep texts under 500 characters.
- Use very short titles.
- The presentation should have:
    - Table of contents.
    - Summary.
    - At least 8 slides.

Example format:
#Title: TITLE OF THE PRESENTATION
#Slide: 1
#Header: Table of Contents
#Content: 1. CONTENT OF THIS POWERPOINT
2. CONTENT OF THIS POWERPOINT
3. CONTENT OF THIS POWERPOINT
#Slide: 2
#Header: TITLE OF SLIDE
#Content: CONTENT OF THE SLIDE
#Slide: END
"""

async def generate_ppt_text(topic: str):
    """Generate PowerPoint content using OpenAI"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": Prompt},
            {"role": "user", "content": f"The user wants a presentation about {topic}."}
        ],
        temperature=0.5,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

async def create_ppt(text_file: str, design_number: int, ppt_name: str):
    """Creates a PowerPoint presentation from a text file."""
    design_template = Path(f"Designs/Design-{design_number}.pptx")
    if not design_template.exists():
        design_template = Path("Designs/Design-1.pptx")  # Default fallback

    prs = Presentation(design_template)
    slide_count = 0
    header = ""
    content = ""
    last_slide_layout_index = -1
    firsttime = True

    async with aiofiles.open(text_file, "r", encoding="utf-8") as f:
        lines = await f.readlines()

    for line in lines:
        if line.startswith("#Title:"):
            header = line.replace("#Title:", "").strip()
            slide = prs.slides.add_slide(prs.slide_layouts[0])
            slide.shapes.title.text = header
            continue

        elif line.startswith("#Slide:"):
            if slide_count > 0:
                slide = prs.slides.add_slide(prs.slide_layouts[slide_layout_index])
                slide.shapes.title.text = header
                body_shape = slide.shapes.placeholders[slide_placeholder_index]
                body_shape.text = content
            content = ""
            slide_count += 1
            slide_layout_index = last_slide_layout_index
            layout_indices = [1, 7, 8]

            while slide_layout_index == last_slide_layout_index:
                if firsttime:
                    slide_layout_index = 1
                    slide_placeholder_index = 1
                    firsttime = False
                    break
                slide_layout_index = random.choice(layout_indices)
                slide_placeholder_index = 2 if slide_layout_index == 8 else 1

            last_slide_layout_index = slide_layout_index
            continue

        elif line.startswith("#Header:"):
            header = line.replace("#Header:", "").strip()
            continue

        elif line.startswith("#Content:"):
            content = line.replace("#Content:", "").strip()
            continue

    ppt_path = f"GeneratedPresentations/{ppt_name}.pptx"
    prs.save(ppt_path)
    return ppt_path

@app.get("/generate-ppt/")
async def generate_ppt(request: Request, topic: str = Query(..., description="Enter the topic for the presentation"), design: int = 1):
    """Generates a PowerPoint and returns an auto-download link."""
    if design > 7 or design <= 0:
        design = 1  # Default design if an invalid one is chosen

    # Generate filename using OpenAI API
    filename_prompt = f"Generate a short, descriptive filename based on this topic: \"{topic}\". Just return the filename."
    filename_response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": filename_prompt}],
        temperature=0.5,
        max_tokens=200,
    )

    filename = filename_response.choices[0].message.content.strip().replace(" ", "_")
    text_file_path = f"Cache/{filename}.txt"

    # Generate and save text
    ppt_content = await generate_ppt_text(topic)
    async with aiofiles.open(text_file_path, "w", encoding="utf-8") as f:
        await f.write(ppt_content)

    # Create PPT
    ppt_path = await create_ppt(text_file_path, design, filename)

    # **Updated Auto-Download Link**
    auto_download_url = f"https://ppt-generator-umkg.onrender.com/auto-download/{filename}.pptx"
    
    return JSONResponse({"auto_download_link": auto_download_url})

@app.get("/download-ppt/{ppt_filename}")
async def download_ppt(ppt_filename: str):
    """Serves the generated PowerPoint file for direct download."""
    ppt_path = f"GeneratedPresentations/{ppt_filename}"
    if not os.path.exists(ppt_path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(ppt_path, filename=ppt_filename, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

@app.get("/auto-download/{ppt_filename}", response_class=HTMLResponse)
async def auto_download_ppt(ppt_filename: str):
    """Returns an HTML page that auto-triggers the download of the PowerPoint file."""
    ppt_path = f"GeneratedPresentations/{ppt_filename}"

    if not os.path.exists(ppt_path):
        return HTMLResponse("<h2>File not found</h2>", status_code=404)

    download_url = f"/download-ppt/{ppt_filename}"
    html_content = f"""
    <html>
        <head>
            <title>Downloading...</title>
            <script>
                window.onload = function() {{
                    window.location.href = "{download_url}";
                }};
            </script>
        </head>
        <body>
            <h2>Your download should start automatically. If not, <a href="{download_url}">click here</a>.</h2>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
