from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import openai
import asyncio
import json
import re
## from supabase import create_client, Client
import PyPDF2
import io
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Resume Scorer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"




def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


async def call_nvidia_api(prompt: str) -> str:
    client = openai.OpenAI(
        base_url=NVIDIA_API_BASE,
        api_key=NVIDIA_API_KEY
    )
    loop = asyncio.get_running_loop()
    def sync_create():
        return client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert resume reviewer and ATS specialist. "
                        "Always respond ONLY with valid JSON. No markdown, no explanation outside JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.3,
        )
    response = await loop.run_in_executor(None, sync_create)
    return response.choices[0].message.content


def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


@app.get("/")
def root():
    return {"message": "Resume Scorer API is running 🚀", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/score-resume")
async def score_resume(
    file: UploadFile = File(...),
    job_title: str = Form(None),
    job_description: str = Form(None),
):
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 5MB.")

    # Extract text
    try:
        resume_text = extract_text_from_pdf(file_bytes)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    if len(resume_text) < 100:
        raise HTTPException(status_code=422, detail="Resume appears to be empty or unreadable.")



    # Build scoring prompt
    job_context = ""
    if job_title and job_description:
        job_context = f"""
Also evaluate how well this resume matches the following job:
Job Title: {job_title}
Job Description: {job_description[:2000]}
Include a "job_match" object with: match_score (0-100), matched_keywords (list), missing_keywords (list).
"""

    prompt = f"""
Analyze this resume and return a JSON object with the following structure:
{{
  "overall_score": <0-100>,
  "ats_score": <0-100>,
  "skills_score": <0-100>,
  "experience_score": <0-100>,
  "formatting_score": <0-100>,
  "grammar_score": <0-100>,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "suggestions": ["...", "..."],
  "feedback": {{
    "summary": "...",
    "ats": "...",
    "skills": "...",
    "experience": "...",
    "formatting": "...",
    "grammar": "..."
  }}{",\n  \"job_match\": {{\"match_score\": 0, \"matched_keywords\": [], \"missing_keywords\": []}}" if job_context else ""}
}}

Resume Text:
{resume_text[:5000]}
{job_context}

Return ONLY the JSON object. No other text.
"""

    # Call NVIDIA API
    try:
        raw_response = await call_nvidia_api(prompt)
        score_data = parse_json_response(raw_response)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI scoring failed: {str(e)}")





    return JSONResponse({
        "success": True,
        "filename": file.filename,
        "scores": {
            "overall": score_data.get("overall_score"),
            "ats": score_data.get("ats_score"),
            "skills": score_data.get("skills_score"),
            "experience": score_data.get("experience_score"),
            "formatting": score_data.get("formatting_score"),
            "grammar": score_data.get("grammar_score"),
        },
        "strengths": score_data.get("strengths", []),
        "weaknesses": score_data.get("weaknesses", []),
        "suggestions": score_data.get("suggestions", []),
        "feedback": score_data.get("feedback", {}),
        "job_match": score_data.get("job_match"),
    })


