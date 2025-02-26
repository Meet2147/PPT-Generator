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
