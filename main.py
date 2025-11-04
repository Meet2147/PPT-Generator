# # from fastapi import FastAPI, Request, Query
# # from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
# # import openai
# # import os
# # # import random
# # import aiofiles
# # from pptx import Presentation
# # from pathlib import Path
# # from dotenv import load_dotenv

# # # Load environment variables
# # load_dotenv()
# # openai_api_key = os.getenv("OPENAI_API_KEY")

# # # Ensure API Key is set
# # if not openai_api_key:
# #     raise ValueError("Missing OpenAI API Key! Please set OPENAI_API_KEY in environment variables.")

# # openai.api_key = openai_api_key  # Set API key
# # app = FastAPI()

# # # Create necessary directories
# # os.makedirs("GeneratedPresentations", exist_ok=True)
# # os.makedirs("Cache", exist_ok=True)

# # # OpenAI Prompt for PowerPoint Content
# # Prompt = """Write a presentation/powerpoint about the user's topic. You only answer with the presentation. Follow the structure of the example.
# # - Keep texts under 500 characters.
# # - Use very short titles.
# # - The presentation should have:
# #     - Table of contents.
# #     - Summary.
# #     - At least 8 slides.

# # Example format:
# # #Title: TITLE OF THE PRESENTATION
# # #Slide: 1
# # #Header: Table of Contents
# # #Content: 1. CONTENT OF THIS POWERPOINT
# # 2. CONTENT OF THIS POWERPOINT
# # 3. CONTENT OF THIS POWERPOINT
# # #Slide: 2
# # #Header: TITLE OF SLIDE
# # #Content: CONTENT OF THE SLIDE
# # #Slide: END
# # """

# # async def generate_ppt_text(topic: str):
# #     """Generate PowerPoint content using OpenAI"""
# #     response = openai.ChatCompletion.create(
# #         model="gpt-4o",
# #         messages=[
# #             {"role": "system", "content": Prompt},
# #             {"role": "user", "content": f"The user wants a presentation about {topic}."}
# #         ],
# #         temperature=0.5,
# #         max_tokens=1000
# #     )
# #     return response.choices[0].message.content.strip()

# # async def create_ppt(text_file: str, design_number: int, ppt_name: str):
# #     """Creates a PowerPoint presentation from a text file."""
# #     design_template = Path(f"Designs/Design-{design_number}.pptx")
# #     if not design_template.exists():
# #         design_template = Path("Designs/Design-1.pptx")  # Default fallback

# #     prs = Presentation(design_template)
# #     slide_count = 0
# #     header = ""
# #     content = ""
# #     last_slide_layout_index = -1
# #     firsttime = True

# #     async with aiofiles.open(text_file, "r", encoding="utf-8") as f:
# #         lines = await f.readlines()

# #     for line in lines:
# #         if line.startswith("#Title:"):
# #             header = line.replace("#Title:", "").strip()
# #             slide = prs.slides.add_slide(prs.slide_layouts[0])
# #             slide.shapes.title.text = header
# #             continue

# #         elif line.startswith("#Slide:"):
# #             if slide_count > 0:
# #                 slide = prs.slides.add_slide(prs.slide_layouts[slide_layout_index])
# #                 slide.shapes.title.text = header
# #                 body_shape = slide.shapes.placeholders[slide_placeholder_index]
# #                 body_shape.text = content
# #             content = ""
# #             slide_count += 1
# #             slide_layout_index = last_slide_layout_index
# #             layout_indices = [1, 7, 8]

# #             while slide_layout_index == last_slide_layout_index:
# #                 if firsttime:
# #                     slide_layout_index = 1
# #                     slide_placeholder_index = 1
# #                     firsttime = False
# #                     break
# #                 slide_layout_index = random.choice(layout_indices)
# #                 slide_placeholder_index = 2 if slide_layout_index == 8 else 1

# #             last_slide_layout_index = slide_layout_index
# #             continue

# #         elif line.startswith("#Header:"):
# #             header = line.replace("#Header:", "").strip()
# #             continue

# #         elif line.startswith("#Content:"):
# #             content = line.replace("#Content:", "").strip()
# #             continue

# #     ppt_path = f"GeneratedPresentations/{ppt_name}.pptx"
# #     prs.save(ppt_path)
# #     return ppt_path

# # @app.get("/generate-ppt/")
# # async def generate_ppt(request: Request, topic: str = Query(..., description="Enter the topic for the presentation"), design: int = 1):
# #     """Generates a PowerPoint and returns an auto-download link."""
# #     if design > 7 or design <= 0:
# #         design = 1  # Default design if an invalid one is chosen

# #     # Generate filename using OpenAI API
# #     filename_prompt = f"Generate a short, descriptive filename based on this topic: \"{topic}\". Just return the filename."
# #     filename_response = openai.ChatCompletion.create(
# #         model="gpt-4o",
# #         messages=[{"role": "system", "content": filename_prompt}],
# #         temperature=0.5,
# #         max_tokens=200,
# #     )

# #     filename = filename_response.choices[0].message.content.strip().replace(" ", "_")
# #     text_file_path = f"Cache/{filename}.txt"

# #     # Generate and save text
# #     ppt_content = await generate_ppt_text(topic)
# #     async with aiofiles.open(text_file_path, "w", encoding="utf-8") as f:
# #         await f.write(ppt_content)

# #     # Create PPT
# #     ppt_path = await create_ppt(text_file_path, design, filename)

# #     # **Updated Auto-Download Link**
# #     auto_download_url = f"https://ppt-generator-mtu0.onrender.com/auto-download/{filename}.pptx"
    
# #     return JSONResponse({"auto_download_link": auto_download_url})

# # @app.get("/download-ppt/{ppt_filename}")
# # async def download_ppt(ppt_filename: str):
# #     """Serves the generated PowerPoint file for direct download."""
# #     ppt_path = f"GeneratedPresentations/{ppt_filename}"
# #     if not os.path.exists(ppt_path):
# #         return JSONResponse({"error": "File not found"}, status_code=404)

# #     return FileResponse(ppt_path, filename=ppt_filename, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")

# # @app.get("/auto-download/{ppt_filename}", response_class=HTMLResponse)
# # async def auto_download_ppt(ppt_filename: str):
# #     """Returns an HTML page that auto-triggers the download of the PowerPoint file."""
# #     ppt_path = f"GeneratedPresentations/{ppt_filename}"

# #     if not os.path.exists(ppt_path):
# #         return HTMLResponse("<h2>File not found</h2>", status_code=404)

# #     download_url = f"/download-ppt/{ppt_filename}"
# #     html_content = f"""
# #     <html>
# #         <head>
# #             <title>Downloading...</title>
# #             <script>
# #                 window.onload = function() {{
# #                     window.location.href = "{download_url}";
# #                 }};
# #             </script>
# #         </head>
# #         <body>
# #             <h2>Your download should start automatically. If not, <a href="{download_url}">click here</a>.</h2>
# #         </body>
# #     </html>
# #     """
# #     return HTMLResponse(content=html_content)

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)

# # import g4f
# # from pptx import Presentation
# # import random
# # import re
# # import os
# # from pathlib import Path
# # import requests
# # from io import BytesIO
# # from g4f.client import Client
# # from pptx.util import Inches
# # from fastapi import FastAPI, Response
# # from fastapi.responses import FileResponse
# # import shutil
# # import uvicorn
# # import uuid
# # import asyncio

# # # ------------------------- Prompt Template ------------------------------
# # PROMPT_TEMPLATE = """Write a presentation/powerpoint about the user's topic. 
# # You only answer with the presentation. Follow the structure of the example.

# # Notice:
# # - You do all the presentation text for the user.
# # - You write the texts no longer than 250 characters!
# # - You make very short titles!
# # - You make the presentation easy to understand.
# # - The presentation has a table of contents.
# # - The presentation has a summary.
# # - At least 7 slides.
# # - For each slide, after the #Content: line, add an #Image: line describing a relevant image that could visually represent the slide's topic. 
# # - If no image is relevant, write #Image: none.

# # Example! - Stick to this formatting exactly!
# # #Title: TITLE OF THE PRESENTATION

# # #Slide: 1
# # #Header: table of contents
# # #Content: 1. CONTENT OF THIS POWERPOINT
# # 2. CONTENTS OF THIS POWERPOINT
# # 3. CONTENT OF THIS POWERPOINT
# # #Image: a 3D illustration of a table of contents in a book

# # #Slide: 2
# # #Header: TITLE OF SLIDE
# # #Content: CONTENT OF THE SLIDE
# # #Image: relevant illustration description here

# # #Slide: END"""

# # client = Client()

# # app = FastAPI()

# # # Ensure the required directories exist
# # Path("GeneratedPresentations").mkdir(exist_ok=True)
# # Path("Cache").mkdir(exist_ok=True)

# # # Placeholder for design templates; you'll need to have these files in a "Designs" directory.
# # # For this example, we assume `Design-2.pptx` exists.
# # # For production, you'd want to handle this more robustly.

# # async def generate_image_async(prompt: str) -> str:
# #     """Generate an image using g4f and return its URL asynchronously."""
# #     if not prompt or prompt.lower() == "none":
# #         return None
# #     try:
# #         resp = await asyncio.to_thread(
# #             client.images.generate,
# #             model="flux",
# #             prompt=prompt,
# #             response_format="url"
# #         )
# #         if hasattr(resp, "data") and resp.data:
# #             return resp.data[0].url
# #     except Exception as e:
# #         print(f"Error generating image: {e}")
# #     return None

# # async def generate_presentation_text_async(user_input: str) -> str:
# #     """Return raw presentation text in the exact marker format asynchronously."""
# #     try:
# #         resp = await asyncio.to_thread(
# #             client.chat.completions.create,
# #             model="gpt-4o-mini",
# #             messages=[
# #                 {"role": "system", "content": PROMPT_TEMPLATE},
# #                 {"role": "user", "content": f"The user wants a presentation about {user_input}"},
# #             ],
# #             web_search=False,
# #         )
# #         content = getattr(resp.choices[0].message, "content", "")
# #         if not isinstance(content, str):
# #             content = str(content or "")

# #         if "#Slide:" not in content and "#Title:" not in content:
# #             raise ValueError("Model response did not include slide markers.")
# #         return content
# #     except Exception as e:
# #         print(f"Error generating presentation text: {e}")
# #         return ""

# # async def create_presentation_async(text_content: str, design_number: int, presentation_name: str):
# #     """Asynchronously create a PowerPoint presentation from generated text."""
# #     presentation = Presentation(f"Designs/Design-{design_number}.pptx")
# #     slide_count = 0
# #     slide_title = ""
# #     slide_content = ""
# #     slide_image_prompt = None
# #     last_slide_layout_index = -1
# #     first_time = True

# #     lines = text_content.splitlines()
# #     line_iter = iter(lines)

# #     for line in line_iter:
# #         line = line.rstrip("\n")

# #         if line.startswith('#Title:'):
# #             slide_title = line.replace('#Title:', '').strip()
# #             slide = presentation.slides.add_slide(presentation.slide_layouts[0])
# #             slide.shapes.title.text = slide_title
# #             continue

# #         elif line.startswith('#Slide:'):
# #             if slide_count > 0:
# #                 slide = presentation.slides.add_slide(presentation.slide_layouts[slide_layout_index])
# #                 slide.shapes.title.text = slide_title
# #                 body_shape = slide.shapes.placeholders[slide_placeholder_index]
# #                 body_shape.text_frame.text = slide_content
# #                 if slide_image_prompt:
# #                     img_url = await generate_image_async(slide_image_prompt)
# #                     if img_url:
# #                         try:
# #                             img_data = await asyncio.to_thread(requests.get, img_url)
# #                             image_stream = BytesIO(img_data.content)
# #                             slide.shapes.add_picture(image_stream, Inches(5), Inches(1.5), width=Inches(4))
# #                         except Exception as e:
# #                             print(f"Could not add image: {e}")
# #             slide_content = ""
# #             slide_image_prompt = None
# #             slide_count += 1
# #             slide_layout_index = last_slide_layout_index
# #             layout_indices = [1, 7, 8]

# #             while slide_layout_index == last_slide_layout_index:
# #                 if first_time:
# #                     slide_layout_index = 1
# #                     slide_placeholder_index = 1
# #                     first_time = False
# #                     break
# #                 slide_layout_index = random.choice(layout_indices)
# #                 slide_placeholder_index = 2 if slide_layout_index == 8 else 1

# #             last_slide_layout_index = slide_layout_index
# #             continue

# #         elif line.startswith('#Header:'):
# #             slide_title = line.replace('#Header:', '').strip()
# #             continue

# #         elif line.startswith('#Content:'):
# #             slide_content = line.replace('#Content:', '').strip()
# #             # Read subsequent lines until the next # marker
# #             try:
# #                 next_line = next(line_iter).strip()
# #                 while next_line and not next_line.startswith('#'):
# #                     slide_content += '\n' + next_line
# #                     next_line = next(line_iter).strip()
# #                 # Push the next line back to the iterator if it's a marker
# #                 if next_line.startswith('#'):
# #                     line_iter = iter([next_line] + list(line_iter))
# #             except StopIteration:
# #                 pass # End of file

# #             continue

# #         elif line.startswith('#Image:'):
# #             slide_image_prompt = line.replace('#Image:', '').strip()
# #             continue

# #     # Final slide commit
# #     if slide_count > 0 and (slide_title or slide_content):
# #         slide = presentation.slides.add_slide(presentation.slide_layouts[slide_layout_index])
# #         slide.shapes.title.text = slide_title
# #         body_shape = slide.shapes.placeholders[slide_placeholder_index]
# #         body_shape.text_frame.text = slide_content
# #         if slide_image_prompt:
# #             img_url = await generate_image_async(slide_image_prompt)
# #             if img_url:
# #                 try:
# #                     img_data = await asyncio.to_thread(requests.get, img_url)
# #                     image_stream = BytesIO(img_data.content)
# #                     slide.shapes.add_picture(image_stream, Inches(5), Inches(1.5), width=Inches(4))
# #                 except Exception as e:
# #                     print(f"Could not add image: {e}")

# #     file_path = f"GeneratedPresentations/{presentation_name}.pptx"
# #     await asyncio.to_thread(presentation.save, file_path)
# #     return file_path

# # @app.get("/auto-download/{filename}")
# # async def download_file(filename: str):
# #     """
# #     Endpoint to download a generated PPTX file.
# #     """
# #     file_path = Path("GeneratedPresentations") / filename
# #     if file_path.is_file():
# #         return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation", filename=filename)
# #     else:
# #         return Response(content="File not found", status_code=404)

# # @app.post("/generate-ppt/")
# # async def generate_ppt(user_input: str):
# #     """
# #     API endpoint to generate a PowerPoint presentation and return a download URL.
# #     """
# #     try:
# #         # Extract design number if provided
# #         user_text = user_input
# #         last_char = user_text[-1] if user_text else ""
# #         input_string = re.sub(r"[^\w\s\.\-\(\)]", "", user_text).replace("\n", "")
# #         design_number = 2

# #         if last_char.isdigit():
# #             design_number = int(last_char)
# #             input_string = user_text[:-2].strip() if len(user_text) >= 2 and user_text[-2].isspace() else user_text[:-1].strip()

# #         if design_number > 7 or design_number == 0:
# #             design_number = 1

# #         # Generate unique filename
# #         filename = f"{input_string}_{uuid.uuid4().hex}"

# #         # Asynchronously generate presentation text and create the PPTX file
# #         presentation_text = await generate_presentation_text_async(input_string)
# #         if not presentation_text:
# #             return {"error": "Failed to generate presentation content."}
            
# #         file_path = await create_presentation_async(presentation_text, design_number, filename)
        
# #         # Construct the auto-download URL
# #         # NOTE: The base URL `https://ppt-generator-mtu0.onrender.com` should be replaced with your actual deployment URL
# #         auto_download_url = f"http://127.0.0.1:8000/auto-download/{Path(file_path).name}"

# #         return {"status": "success", "url": auto_download_url}
    
# #     except Exception as e:
# #         return {"status": "error", "message": str(e)}

# # # To run the app, you'd use a command like:
# # # uvicorn your_file_name:app --host 0.0.0.0 --port 8000


# import g4f
# from pptx import Presentation
# import random
# import re
# import os
# from pathlib import Path
# import requests
# from io import BytesIO
# from g4f.client import Client
# from pptx.util import Inches
# from fastapi import FastAPI, Response
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# import shutil
# import uvicorn
# import uuid
# import asyncio


# PROMPT_TEMPLATE = """Write a presentation/powerpoint about the user's topic. You only answer with the presentation. Follow the structure of the example.

# Notice:
# - You do all the presentation text for the user.
# - You write the texts no longer than 250 characters!
# - You make very short titles!
# - You make the presentation easy to understand.
# - The presentation has a table of contents.
# - The presentation has a summary.
# - At least 7 slides.
# - For each slide, after the #Content: line, add an #Image: line describing a relevant image that could visually represent the slide's topic.
# - If no image is relevant, write #Image: none.

# Example! - Stick to this formatting exactly!
# #Title: TITLE OF THE PRESENTATION

# #Slide: 1
# #Header: table of contents
# #Content: 1. CONTENT OF THIS POWERPOINT
# 2. CONTENTS OF THIS POWERPOINT
# 3. CONTENT OF THIS POWERPOINT
# #Image: a 3D illustration of a table of contents in a book

# #Slide: 2
# #Header: TITLE OF SLIDE
# #Content: CONTENT OF THE SLIDE
# #Image: relevant illustration description here

# #Slide: 3
# #Header: TITLE OF SLIDE
# #Content: CONTENT OF THE SLIDE
# #Image: relevant illustration description here

# #Slide: 4
# #Header: TITLE OF SLIDE
# #Content: CONTENT OF THE SLIDE
# #Image: relevant illustration description here

# #Slide: 5
# #Header: TITLE OF SLIDE
# #Content: CONTENT OF THE SLIDE
# #Image: relevant illustration description here

# #Slide: 6
# #Header: TITLE OF SLIDE
# #Content: CONTENT OF THE SLIDE
# #Image: relevant illustration description here

# #Slide: 7
# #Header: summary
# #Content: CONTENT OF THE SUMMARY
# #Image: an illustration of a person reviewing a summary report

# #Slide: END"""


# client = Client()

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # tighten for prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Path("GeneratedPresentations").mkdir(exist_ok=True)
# Path("Cache").mkdir(exist_ok=True)

# async def generate_image_async(prompt: str) -> str:
#     if not prompt or prompt.lower() == "none":
#         return None
#     try:
#         resp = await asyncio.to_thread(
#             client.images.generate,
#             model="flux",
#             prompt=prompt,
#             response_format="url"
#         )
#         if hasattr(resp, "data") and resp.data:
#             return resp.data[0].url
#     except Exception as e:
#         print(f"Error generating image: {e}")
#     return None

# async def generate_presentation_text_async(user_input: str) -> str:
#     try:
#         resp = await asyncio.to_thread(
#             client.chat.completions.create,
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": PROMPT_TEMPLATE},
#                 {"role": "user", "content": f"The user wants a presentation about {user_input}"},
#             ],
#             web_search=False,
#         )
#         content = getattr(resp.choices[0].message, "content", "")
#         if not isinstance(content, str):
#             content = str(content or "")

#         if "#Slide:" not in content and "#Title:" not in content:
#             raise ValueError("Model response did not include slide markers.")
#         return content
#     except Exception as e:
#         print(f"Error generating presentation text: {e}")
#         return ""

# async def create_presentation_async(text_content: str, design_number: int, presentation_name: str):
#     # Open template (fallback to Design-1, else blank)
#     template_path = Path(f"Designs/Design-{design_number}.pptx")
#     if not template_path.exists():
#         template_path = Path("Designs/Design-1.pptx")
#     if template_path.exists():
#         presentation = Presentation(str(template_path))
#     else:
#         presentation = Presentation()

#     # State
#     slide_count = 0
#     slide_title = ""
#     slide_content = ""
#     slide_image_prompt = None
#     last_slide_layout_index = -1
#     first_time = True

#     # Prepare lines & indices
#     lines = [ln.rstrip("\n") for ln in text_content.splitlines()]
#     i = 0

#     # Safe defaults based on available layouts
#     def _safe_layout(idx: int) -> int:
#         if len(presentation.slide_layouts) == 0:
#             return 0
#         return max(0, min(idx, len(presentation.slide_layouts) - 1))

#     slide_layout_index = _safe_layout(1 if len(presentation.slide_layouts) > 1 else 0)
#     slide_placeholder_index = 2 if slide_layout_index == 8 else (1 if len(presentation.slide_layouts) > 1 else 0)

#     # Commit helper
#     async def commit_slide():
#         nonlocal slide_title, slide_content, slide_image_prompt, slide_count
#         if slide_count > 0 and (slide_title or slide_content):
#             slide = presentation.slides.add_slide(presentation.slide_layouts[_safe_layout(slide_layout_index)])
#             # Title
#             try:
#                 slide.shapes.title.text = slide_title
#             except Exception:
#                 try:
#                     slide.placeholders[0].text = slide_title
#                 except Exception:
#                     pass
#             # Body
#             try:
#                 body_shape = slide.shapes.placeholders[slide_placeholder_index]
#                 if hasattr(body_shape, "text_frame"):
#                     body_shape.text_frame.text = slide_content
#                 else:
#                     body_shape.text = slide_content
#             except Exception:
#                 # placeholder may not exist in some layouts
#                 pass

#             # Image (if any)
#             if slide_image_prompt:
#                 img_url = await generate_image_async(slide_image_prompt)
#                 if img_url:
#                     try:
#                         resp = await asyncio.to_thread(requests.get, img_url, timeout=20)
#                         image_stream = BytesIO(resp.content)
#                         # position/size — tweak to match your template
#                         slide.shapes.add_picture(image_stream, Inches(5), Inches(1.5), width=Inches(4))
#                     except Exception as e:
#                         print(f"Could not add image: {e}")

#     # Parse
#     while i < len(lines):
#         line = lines[i]

#         if line.startswith("#Title:"):
#             slide_title = line.replace("#Title:", "").strip()
#             # Title slide (layout 0 when available)
#             title_layout = _safe_layout(0)
#             slide = presentation.slides.add_slide(presentation.slide_layouts[title_layout])
#             try:
#                 slide.shapes.title.text = slide_title
#             except Exception:
#                 try:
#                     slide.placeholders[0].text = slide_title
#                 except Exception:
#                     pass
#             i += 1
#             continue

#         if line.startswith("#Slide:"):
#             # Commit previous content slide (if any)
#             await commit_slide()

#             # Reset accumulators for the new slide
#             slide_content = ""
#             slide_image_prompt = None
#             slide_count += 1

#             # Choose a layout (avoid repeating the previous)
#             if len(presentation.slide_layouts) >= 9:
#                 layout_choices = [1, 7, 8]
#             else:
#                 layout_choices = [1] if len(presentation.slide_layouts) > 1 else [0]

#             if first_time:
#                 slide_layout_index = _safe_layout(1 if len(presentation.slide_layouts) > 1 else 0)
#                 slide_placeholder_index = 2 if slide_layout_index == 8 else (1 if len(presentation.slide_layouts) > 1 else 0)
#                 first_time = False
#             else:
#                 next_idx = last_slide_layout_index
#                 # ensure a different layout if possible
#                 tries = 0
#                 while next_idx == last_slide_layout_index and tries < 10:
#                     next_idx = random.choice(layout_choices)
#                     tries += 1
#                 slide_layout_index = _safe_layout(next_idx)
#                 slide_placeholder_index = 2 if slide_layout_index == 8 else (1 if len(presentation.slide_layouts) > 1 else 0)

#             last_slide_layout_index = slide_layout_index
#             i += 1
#             continue

#         if line.startswith("#Header:"):
#             slide_title = line.replace("#Header:", "").strip()
#             i += 1
#             continue

#         if line.startswith("#Content:"):
#             # Capture all following non-marker lines as content
#             slide_content = line.replace("#Content:", "").strip()
#             i += 1
#             while i < len(lines) and not lines[i].startswith("#"):
#                 slide_content += "\n" + lines[i]
#                 i += 1
#             continue

#         if line.startswith("#Image:"):
#             slide_image_prompt = line.replace("#Image:", "").strip()
#             i += 1
#             continue

#         # Ignore stray lines
#         i += 1

#     # Final commit
#     await commit_slide()

#     # Save and return
#     out_dir = Path("GeneratedPresentations")
#     out_dir.mkdir(parents=True, exist_ok=True)
#     file_path = out_dir / f"{presentation_name}.pptx"
#     await asyncio.to_thread(presentation.save, str(file_path))
#     return str(file_path)

# @app.get("/auto-download/{filename}")
# async def download_file(filename: str):
#     file_path = Path("GeneratedPresentations") / filename
#     if file_path.is_file():
#         return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation", filename=filename)
#     else:
#         return Response(content="File not found", status_code=404)

# @app.post("/generate-ppt/")
# async def generate_ppt(user_input: str):
#     try:
#         user_text = user_input
#         last_char = user_text[-1] if user_text else ""
#         input_string = re.sub(r"[^\w\s\.\-\(\)]", "", user_text).replace("\n", "")
#         design_number = 2

#         if last_char.isdigit():
#             design_number = int(last_char)
#             input_string = user_text[:-2].strip() if len(user_text) >= 2 and user_text[-2].isspace() else user_text[:-1].strip()

#         if design_number > 7 or design_number == 0:
#             design_number = 1

#         filename = f"{input_string}_{uuid.uuid4().hex}"

#         presentation_text = await generate_presentation_text_async(input_string)
#         if not presentation_text:
#             return {"error": "Failed to generate presentation content."}
            
#         file_path = await create_presentation_async(presentation_text, design_number, filename)
        
#         auto_download_url = f"https://ppt-generator-mtu0.onrender.com/auto-download/{Path(file_path).name}"

#         return {"status": "success", "url": auto_download_url}
    
#     except Exception as e:
#         return {"status": "error", "message": str(e)}


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


