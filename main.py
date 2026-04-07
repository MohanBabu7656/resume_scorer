from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import httpx
import json
import re
from supabase import create_client, Client
import PyPDF2
import io

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
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


async def call_nvidia_api(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert resume reviewer and ATS specialist. "
                    "Always respond ONLY with valid JSON. No markdown, no explanation outside JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(NVIDIA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


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

    # Save resume to Supabase
    resume_id = str(uuid.uuid4())
    supabase.table("resumes").insert({
        "id": resume_id,
        "filename": file.filename,
        "raw_text": resume_text[:10000],
    }).execute()

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

    # Save score to Supabase
    score_id = str(uuid.uuid4())
    supabase.table("scores").insert({
        "id": score_id,
        "resume_id": resume_id,
        "overall_score": score_data.get("overall_score"),
        "ats_score": score_data.get("ats_score"),
        "skills_score": score_data.get("skills_score"),
        "experience_score": score_data.get("experience_score"),
        "formatting_score": score_data.get("formatting_score"),
        "grammar_score": score_data.get("grammar_score"),
        "feedback": score_data.get("feedback"),
        "strengths": score_data.get("strengths", []),
        "weaknesses": score_data.get("weaknesses", []),
        "suggestions": score_data.get("suggestions", []),
        "model_used": NVIDIA_MODEL,
    }).execute()

    # Save job match if present
    if job_context and "job_match" in score_data:
        jm = score_data["job_match"]
        supabase.table("job_matches").insert({
            "resume_id": resume_id,
            "job_title": job_title,
            "job_description": (job_description or "")[:2000],
            "match_score": jm.get("match_score"),
            "matched_keywords": jm.get("matched_keywords", []),
            "missing_keywords": jm.get("missing_keywords", []),
        }).execute()

    return JSONResponse({
        "success": True,
        "resume_id": resume_id,
        "score_id": score_id,
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


@app.get("/api/results/{resume_id}")
async def get_results(resume_id: str):
    resume = supabase.table("resumes").select("*").eq("id", resume_id).single().execute()
    if not resume.data:
        raise HTTPException(status_code=404, detail="Resume not found.")

    scores = supabase.table("scores").select("*").eq("resume_id", resume_id).order("scored_at", desc=True).limit(1).execute()
    job_match = supabase.table("job_matches").select("*").eq("resume_id", resume_id).limit(1).execute()

    return {
        "resume": resume.data,
        "scores": scores.data[0] if scores.data else None,
        "job_match": job_match.data[0] if job_match.data else None,
    }
