from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
import pptx
from pptx.util import Inches, Pt
import tempfile
import fitz  # PyMuPDF for extracting text from PDFs
import openai
import os
import logging
import requests
from dotenv import load_dotenv
import json

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

# Access environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")  # Default fallback
openai.api_key = os.getenv("OPENAI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Authentication setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Fake user database
fake_users_db = {
    "user1": {"username": "user1", "hashed_password": pwd_context.hash("password1")},
    "user2": {"username": "user2", "hashed_password": pwd_context.hash("password2")},
}

# Authentication utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Token endpoint
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# PDF content extraction
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

# OpenAI data generation
def generate_presentation_data(pdf_content, topic):
    """
    Generate slide titles, content, and image queries from a single OpenAI API call.
    """
    prompt = f"""
    Create a PowerPoint presentation for the topic '{topic}' with content:
    {pdf_content[:1000]}.
    Generate:
    - 10 slide titles
    - 5 bullet points per slide
    - Image queries for each slide.

    Output in JSON:
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
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    try:
        # Log raw response for debugging
        logging.info(f"Raw OpenAI response: {response}")

        # Parse the content as JSON
        content = response.choices[0].message.content
        slides_data = json.loads(content)  # Safely parse the JSON
        return slides_data
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from OpenAI response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to parse OpenAI response. Ensure valid JSON format.")

# Main API endpoint
@app.post("/generate_presentation/")
async def generate_presentation(
    topic: str = Form(...),
    pdf: UploadFile = File(...),
    template: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    try:
        # Save the uploaded files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_file:
            pdf_path = pdf_file.name
            pdf_file.write(await pdf.read())

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as template_file:
            template_path = template_file.name
            template_file.write(await template.read())

        # Extract text from PDF
        pdf_content = extract_text_from_pdf(pdf_path)

        # Generate slide data
        slides_data = generate_presentation_data(pdf_content, topic)

        # Generate presentation
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as output_file:
            output_path = create_presentation(template_path, slides_data["slides"], output_file.name)

        return FileResponse(output_path, filename=f"{topic}_Presentation.pptx")
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
def create_presentation(template_path, slides_data, output_path):
    prs = pptx.Presentation(template_path)
    prs.slides._sldIdLst.clear()

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

