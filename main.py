from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import openai
import pptx
from pptx.util import Inches, Pt
import os
from dotenv import load_dotenv
import tempfile
import requests
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")  # Add your Unsplash API key here

# Constants for formatting
TITLE_FONT_SIZE = Pt(28)
SLIDE_FONT_SIZE = Pt(18)
FONT_NAME = "Calibri"

class PresentationRequest(BaseModel):
    topic: str

def generate_slide_titles(topic):
    """
    Generate 7 concise, engaging slide titles for the presentation.
    """
    prompt = f"""
    Generate 7 concise, engaging slide titles for a professional PowerPoint presentation on the topic: '{topic}'.
    Ensure the titles are clear and aligned with key subtopics.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a presentation design expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return [title.strip() for title in response['choices'][0]['message']['content'].strip().split("\n") if title.strip()]

def generate_slide_content(slide_title, topic):
    """
    Generate 3-5 concise bullet points for a slide based on the title and topic.
    """
    prompt = f"""
    Write 3-5 concise bullet points for a PowerPoint slide titled: '{slide_title}', which is part of a presentation on the topic '{topic}'.
    Focus on key insights and avoid lengthy paragraphs.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a presentation content expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

def generate_image_query(slide_title, topic):
    """
    Generate a descriptive query for fetching relevant images.
    """
    prompt = f"""
    Provide a short descriptive query for an image related to the PowerPoint slide titled '{slide_title}'.
    The slide is part of a presentation on the topic '{topic}'.
    Make the query concise and specific to fetch a relevant image.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in visual content design."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    return response['choices'][0]['message']['content'].strip()

def fetch_image_from_unsplash(query):
    """
    Fetch an image from Unsplash based on a given query.
    """
    url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_ACCESS_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        image_url = response.json().get('urls', {}).get('regular')
        logging.info(f"Image URL for '{query}': {image_url}")
        return image_url
    logging.error(f"Unsplash API failed for query: {query} | Status Code: {response.status_code}")
    return None

def download_image(url):
    """
    Download an image from a given URL and save it temporarily.
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(response.content)
                logging.info(f"Image downloaded successfully: {tmp.name}")
                return tmp.name
        logging.error(f"Failed to download image from URL: {url} | Status Code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error downloading image: {str(e)}")
    return None

def apply_uniform_font(text_frame, font_size):
    """
    Apply uniform font to text in a text frame.
    """
    for paragraph in text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = font_size
            run.font.name = FONT_NAME

def create_presentation(topic, slide_titles, slide_contents, output_path):
    """
    Create a PowerPoint presentation with slides, content, and relevant images.
    """
    prs = pptx.Presentation()

    # Add title slide
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = topic
    apply_uniform_font(title_slide.shapes.title.text_frame, TITLE_FONT_SIZE)

    # Add content slides
    for slide_title, slide_content in zip(slide_titles, slide_contents):
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout
        title_shape = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(8), Inches(1))
        title_shape.text = slide_title
        apply_uniform_font(title_shape.text_frame, TITLE_FONT_SIZE)

        content_shape = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(8), Inches(3))
        text_frame = content_shape.text_frame
        text_frame.clear()
        for bullet in slide_content.split("\n"):
            paragraph = text_frame.add_paragraph()
            paragraph.text = bullet.strip()
            apply_uniform_font(text_frame, SLIDE_FONT_SIZE)

        # Fetch and add relevant image
        image_query = generate_image_query(slide_title, topic)
        image_url = fetch_image_from_unsplash(image_query)
        if image_url:
            image_path = download_image(image_url)
            if image_path:
                try:
                    slide.shapes.add_picture(image_path, Inches(5), Inches(1.5), Inches(3), Inches(2))
                    logging.info(f"Image added to slide: {slide_title}")
                except Exception as e:
                    logging.error(f"Failed to add image to slide '{slide_title}': {str(e)}")

    prs.save(output_path)
    return output_path

@app.post("/generate_presentation/")
def generate_presentation(request: PresentationRequest):
    try:
        topic = request.topic
        slide_titles = generate_slide_titles(topic)
        slide_contents = [generate_slide_content(title, topic) for title in slide_titles]

        # Generate presentation
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            presentation_path = create_presentation(topic, slide_titles, slide_contents, tmp.name)

        return FileResponse(presentation_path, filename=f"{topic}_presentation.pptx", media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation')
    except Exception as e:
        logging.error(f"Error generating presentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
