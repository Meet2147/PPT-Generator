from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
import pptx
from pptx.util import Inches, Pt
import tempfile
import requests
import os
from dotenv import load_dotenv
import logging
import fitz  # PyMuPDF for extracting text from PDFs
import openai
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Constants for layout
LEFT_COLUMN_WIDTH = Inches(4)
RIGHT_COLUMN_WIDTH = Inches(4)
TITLE_FONT_SIZE = Pt(28)
BODY_FONT_SIZE = Pt(18)
FONT_NAME = "Calibri"

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from an uploaded PDF.
    """
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

def generate_presentation_data(pdf_content, topic):
    """
    Generate slide titles, content, and image queries from a single OpenAI API call.
    """
    prompt = f"""
    You are tasked with creating a professional PowerPoint presentation based on the following content and topic:
    
    Topic: {topic}
    Content: {pdf_content[:1000]}  # Limit content sent to OpenAI for efficiency.
    
    Generate the following:
    1. Seven slide titles.
    2. Three to five bullet points for each slide title.
    3. A descriptive image query for each slide title.
    
    Output the data in the following JSON format:
    {{
        "slides": [
            {{
                "title": "Slide Title 1",
                "content": ["Bullet point 1", "Bullet point 2", "Bullet point 3"],
                "image_query": "Image query 1"
            }},
            ...
        ]
    }}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a presentation design expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response['choices'][0]['message']['content']

def search_image(search_query, Pexels_api_key, file_path):
    """
    Search for and download an image using the Pexels API.
    """
    base_url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": Pexels_api_key}
    params = {"query": search_query, "per_page": 1}

    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json()
        if results["photos"]:
            image_url = results["photos"][0]["src"]["original"]
            # Download the image
            img_data = requests.get(image_url).content
            with open(file_path, 'wb') as handler:
                handler.write(img_data)
            return file_path
        else:
            return None
    else:
        raise HTTPException(status_code=500, detail="Error fetching image from Pexels API")

def clear_template_slides(prs):
    """
    Clear all existing slides in the presentation template.
    """
    xml_slides = prs.slides._sldIdLst  # Access the slide list in the XML structure
    slides_to_remove = list(xml_slides)  # Create a list of slides to remove
    for slide in slides_to_remove:
        xml_slides.remove(slide)  # Remove each slide

def create_presentation(template_path, slides_data, output_path):
    """
    Create a PowerPoint presentation using the uploaded template and generated data.
    """
    prs = pptx.Presentation(template_path)

    # Clear existing slides in the template
    clear_template_slides(prs)

    # Add new slides with generated content
    for slide_data in slides_data:
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout

        # Add text to the left column
        left_column = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), LEFT_COLUMN_WIDTH, Inches(5))
        text_frame = left_column.text_frame
        text_frame.clear()
        title_paragraph = text_frame.add_paragraph()
        title_paragraph.text = slide_data["title"]
        title_paragraph.font.size = TITLE_FONT_SIZE
        title_paragraph.font.name = FONT_NAME

        for bullet in slide_data["content"]:
            paragraph = text_frame.add_paragraph()
            paragraph.text = bullet.strip()
            paragraph.font.size = BODY_FONT_SIZE
            paragraph.font.name = FONT_NAME

        # Fetch and add an image to the right column
        image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        try:
            image_file = search_image(slide_data["image_query"], PEXELS_API_KEY, image_path)
            if image_file:
                slide.shapes.add_picture(image_file, Inches(5.5), Inches(0.5), RIGHT_COLUMN_WIDTH, Inches(5))
        except Exception as e:
            logging.error(f"Failed to fetch or add image for slide '{slide_data['title']}': {str(e)}")

    prs.save(output_path)
    return output_path

@app.post("/generate_presentation/")
async def generate_presentation(
    topic: str = Form(...),
    pdf: UploadFile = File(...),
    template: UploadFile = File(...)
):
    try:
        # Save uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name
            tmp.write(pdf.file.read())

        # Save uploaded template
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            template_path = tmp.name
            tmp.write(template.file.read())

        # Extract text from PDF
        pdf_content = extract_text_from_pdf(pdf_path)

        # Generate presentation data
        presentation_data = generate_presentation_data(pdf_content, topic)
        slides_data = eval(presentation_data)["slides"]

        # Generate the presentation
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            output_path = create_presentation(template_path, slides_data, tmp.name)

        return FileResponse(output_path, filename=f"{topic}_Presentation.pptx", media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation')
    except Exception as e:
        logging.error(f"Error generating presentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
