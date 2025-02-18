from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import pptx
from pptx.util import Inches, Pt
import tempfile
import fitz  # PyMuPDF for extracting text from PDFs
import openai
import os
import logging
import requests
import json
from dotenv import load_dotenv
import shutil

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

# Access environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Logging configuration
logging.basicConfig(level=logging.INFO)

# PDF content extraction
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

# OpenAI data generation
def generate_presentation_data(pdf_content, topic):
    prompt = f"""
    Create a PowerPoint presentation for the topic '{topic}'.
    The content is: {pdf_content[:1000]}.

    Return the response **strictly in JSON format** with the following structure:

    {{
        "slides": [
            {{
                "title": "Slide Title",
                "content": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
                "image_query": "query"
            }},
            ...
        ]
    }}

    Ensure it is valid JSON without extra text.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    try:
        logging.info(f"Raw OpenAI response: {response}")
        content = response.choices[0].message.content.strip()
        slides_data = json.loads(content)
        return slides_data
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from OpenAI response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to parse OpenAI response. Ensure valid JSON format.")

# Main API endpoint
@app.get("/generate_presentation/")
async def generate_presentation(
    topic: str = Query(..., description="Topic of the presentation"),
    pdf_path: str = Query(..., description="Path to the PDF file")
):
    try:
        pdf_content = extract_text_from_pdf(pdf_path)
        slides_data = generate_presentation_data(pdf_content, topic)

        output_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx").name
        create_presentation(slides_data["slides"], output_file_path)

        response_file_path = f"/tmp/{topic}_Presentation.pptx"
        shutil.move(output_file_path, response_file_path)

        return FileResponse(
            path=response_file_path,
            filename=f"{topic}_Presentation.pptx",
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    except Exception as e:
        logging.error(f"Error generating presentation: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the presentation.")

# Pexels image search
def search_image(search_query, file_path):
    base_url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": search_query, "per_page": 1}

    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json()
        if results["photos"]:
            image_url = results["photos"][0]["src"]["medium"]
            img_data = requests.get(image_url).content
            with open(file_path, "wb") as f:
                f.write(img_data)
            return file_path
    return None

# Create PowerPoint presentation
def create_presentation(slides_data, output_path):
    prs = pptx.Presentation()
    
    for slide_data in slides_data:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(8), Inches(1))
        title_box.text = slide_data["title"]

        left_column = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(4), Inches(5))
        for bullet in slide_data["content"]:
            p = left_column.text_frame.add_paragraph()
            p.text = bullet

        image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        search_query = slide_data["image_query"]
        image_file = search_image(search_query, image_path)
        if image_file:
            slide.shapes.add_picture(image_file, Inches(5), Inches(1.5), Inches(4), Inches(3))

    prs.save(output_path)
    return output_path
