from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import openai
import pptx
from pptx.util import Inches, Pt
import os
from dotenv import load_dotenv
import tempfile

TEMPLATE_URL = "https://github.com/Meet2147/PPT-Generator/blob/main/template.pptx?raw=true"
app = FastAPI()

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY') #replace with your actual API key

# Custom formatting options
TITLE_FONT_SIZE = Pt(30)
SLIDE_FONT_SIZE = Pt(16)
FONT_NAME = "Arial"

class PresentationRequest(BaseModel):
    topic: str

def generate_slide_titles(topic):
    prompt = f"Generate 5 slide titles for the given topic '{topic}'."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip().split("\n")

def apply_uniform_font(text_frame, font_size):
    for paragraph in text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = font_size
            run.font.name = FONT_NAME

def create_presentation(topic, slide_titles, slide_contents, file_path):
    prs = pptx.Presentation("https://github.com/Meet2147/PPT-Generator/blob/main/template.pptx")

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
        body_shape.text = slide_content
        apply_uniform_font(body_shape.text_frame, SLIDE_FONT_SIZE)

    prs.save(file_path)
    return file_path

@app.post("/generate_presentation/")
def generate_presentation(request: PresentationRequest):
    try:
        topic = request.topic
        slide_titles = generate_slide_titles(topic)
        slide_contents = [generate_slide_content(title) for title in slide_titles]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            presentation_path = create_presentation(topic, slide_titles, slide_contents, tmp.name)
        
        return FileResponse(presentation_path, filename=f"{topic}_presentation.pptx")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
