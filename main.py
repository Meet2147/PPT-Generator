from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import FileResponse
import openai
import pptx
from pptx.util import Pt
import os
from dotenv import load_dotenv
import tempfile
import requests
import fitz  # PyMuPDF

app = FastAPI()

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")  # replace with your actual API key

# Custom formatting options
TITLE_FONT_SIZE = Pt(30)
SLIDE_FONT_SIZE = Pt(16)
FONT_NAME = "Arial"
TEMPLATE_URL = "https://github.com/Meet2147/PPT-Generator/blob/main/template.pptx?raw=true"

class PresentationRequest(BaseModel):
    topic: str

def download_template(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            tmp.write(response.content)
            return tmp.name
    else:
        raise HTTPException(status_code=500, detail="Failed to download template")

def generate_slide_titles(topic):
    prompt = f"Generate 10 slide titles for the given topic '{topic}'."
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip().split("\n")

def generate_slide_content(slide_title):
    prompt = f"Generate content for the slide: '{slide_title}'."
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content']

def apply_uniform_font(text_frame, font_size):
    for paragraph in text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = font_size
            run.font.name = FONT_NAME

def create_presentation(topic, slide_titles, slide_contents, template_path, output_path):
    prs = pptx.Presentation(template_path)

    # Remove all existing slides from the template
    xml_slides = prs.slides._sldIdLst  
    slide_id_list = [slide for slide in xml_slides]
    for slide in slide_id_list:
        xml_slides.remove(slide)

    slide_layout = prs.slide_layouts[1]
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = topic
    apply_uniform_font(title_slide.shapes.title.text_frame, TITLE_FONT_SIZE)

    for slide_title, slide_content in zip(slide_titles, slide_contents):
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = slide_title
        apply_uniform_font(slide.shapes.title.text_frame, TITLE_FONT_SIZE)

        body_shape = slide.shapes.placeholders[1]
        text_frame = body_shape.text_frame
        text_frame.clear()  # Clear any existing paragraphs
        p = text_frame.add_paragraph()
        p.text = slide_content
        apply_uniform_font(text_frame, SLIDE_FONT_SIZE)

    prs.save(output_path)
    return output_path

def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

@app.post("/generate_presentation/")
def generate_presentation(request: PresentationRequest):
    try:
        topic = request.topic
        slide_titles = generate_slide_titles(topic)
        slide_contents = [generate_slide_content(title) for title in slide_titles]

        template_path = download_template(TEMPLATE_URL)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            presentation_path = create_presentation(topic, slide_titles, slide_contents, template_path, tmp.name)
        
        return FileResponse(presentation_path, filename=f"{topic}_presentation.pptx")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_presentation_from_pdf/")
async def generate_presentation_from_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            pdf_path = tmp.name

        text = extract_text_from_pdf(pdf_path)

        # Generate titles and content using the extracted text
        slide_titles_prompt = f"Generate 10 slide titles from the following text:\n\n{text}"
        slide_titles_response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": slide_titles_prompt}
            ],
            max_tokens=200
        )
        slide_titles = slide_titles_response['choices'][0]['message']['content'].strip().split("\n")

        slide_contents = []
        for title in slide_titles:
            slide_content_prompt = f"Generate content for the slide titled '{title}' from the following text:\n\n{text}"
            slide_content_response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": slide_content_prompt}
                ],
                max_tokens=200
            )
            slide_contents.append(slide_content_response['choices'][0]['message']['content'].strip())

        template_path = download_template(TEMPLATE_URL)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            presentation_path = create_presentation("Presentation from PDF", slide_titles, slide_contents, template_path, tmp.name)
        
        return FileResponse(presentation_path, filename="presentation_from_pdf.pptx")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
