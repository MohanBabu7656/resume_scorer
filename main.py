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
AI_MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1




def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


async def call_nvidia_api(prompt: str, system_content: str = None, model: str = None) -> str:
    """Call the NVIDIA/OpenAI-compatible API. Accepts a custom system prompt and model.

    Returns the assistant message content as a string.
    """
    if model is None:
        model = NVIDIA_MODEL
    if system_content is None:
        system_content = (
            "You are an expert resume reviewer and ATS specialist. "
            "Always respond ONLY with valid JSON. No markdown, no explanation outside JSON."
        )

    client = openai.OpenAI(
        base_url=NVIDIA_API_BASE,
        api_key=NVIDIA_API_KEY
    )
    loop = asyncio.get_running_loop()

    def sync_create():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
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


def validate_score_schema(obj: dict) -> (bool, str):
    if not isinstance(obj, dict):
        return False, "score response is not an object"
    # expected numeric scores
    numeric_keys = [
        "overall_score",
        "ats_score",
        "skills_score",
        "experience_score",
        "formatting_score",
        "grammar_score",
    ]
    for k in numeric_keys:
        v = obj.get(k)
        if v is None:
            return False, f"missing '{k}'"
        if not isinstance(v, (int, float)):
            return False, f"'{k}' is not numeric"
        if not (0 <= v <= 100):
            return False, f"'{k}' out of range 0-100"
    # feedback should be an object
    if "feedback" not in obj or not isinstance(obj["feedback"], dict):
        return False, "missing or invalid 'feedback'"
    # lists
    for list_key in ("strengths", "weaknesses", "suggestions"):
        if list_key in obj and not isinstance(obj[list_key], list):
            return False, f"'{list_key}' must be a list"
    return True, ""


async def ai_call_with_retries(prompt: str, system_content: str = None, model: str = None, schema_validator=None, retries: int = AI_MAX_RETRIES):
    last_exc = None
    backoff = RETRY_BASE_SECONDS
    for attempt in range(1, retries + 1):
        try:
            raw = await call_nvidia_api(prompt, system_content=system_content, model=model)
            parsed = parse_json_response(raw)
            if schema_validator:
                ok, msg = schema_validator(parsed)
                if not ok:
                    raise ValueError(f"Schema validation failed: {msg}")
            return parsed
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            # exhausted
            raise last_exc


async def validate_inputs(resume_text: str, job_title: str = None, job_description: str = None) -> dict:
    """Validate resume and job inputs via meta/llama-guard-4-12b filter."""
    snippet = resume_text[:5000]
    jd = (job_description or "")[:2000]

    content_to_check = f"Resume:\n{snippet}\n\nJob Title: {job_title or ''}\n\nJob Description:\n{jd}"

    client = openai.OpenAI(
        base_url=NVIDIA_API_BASE,
        api_key=NVIDIA_API_KEY
    )
    loop = asyncio.get_running_loop()

    def sync_create():
        return client.chat.completions.create(
            model="meta/llama-guard-4-12b",
            messages=[{"role": "user", "content": content_to_check}],
            max_tokens=20,
            temperature=0.2,
        )

    try:
        response = await loop.run_in_executor(None, sync_create)
        result = response.choices[0].message.content.strip()

        if result.startswith("safe"):
            return {"safe": True, "reasons": [], "categories": {}}
        else:
            # Result format is usually "unsafe\n[Category]" (e.g. S10)
            parts = result.split("\n")
            category = parts[1].strip() if len(parts) > 1 else "Unknown"
            return {
                "safe": False, 
                "reasons": [f"Flagged by Llama Guard: {category}"], 
                "categories": {category: True}
            }
    except Exception as e:
        return {"safe": False, "reasons": [f"Validation call failed: {str(e)}"], "categories": {}}


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
    # 1. Validate Job Context logic before doing any heavy I/O or AI processing
    if job_description and not job_title:
        raise HTTPException(status_code=400, detail="Job title is required when a job description is provided.")
    if job_title and not job_description:
        raise HTTPException(status_code=400, detail="Job description is required when a job title is provided.")

    # 2. Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 3. Validate file size
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

    # Validate inputs using AI safety validator before proceeding
    validation = await validate_inputs(resume_text, job_title, job_description)
    if not validation.get("safe", False):
        categories = validation.get("categories", {})
        flagged_cats = set(categories.keys())

        # Llama Guard 4 Categories allowed for Resume Parsing:
        # S7: Privacy (Resumes have PII like emails/phones)
        # S10: Hate (Job Descriptions might be discriminatory like "only 18+ female", we still want to parse)
        # S12: Sexual Content (Might be mistakenly triggered by age requirements like "18+")
        # Unknown: Fallback just in case format changes
        allowed_categories = {"S7", "S10", "S12", "Unknown"}

        if flagged_cats and flagged_cats.issubset(allowed_categories):
            pass  # allow through despite Llama Guard warning
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Input not safe to process",
                    "reasons": validation.get("reasons", []),
                    "categories": categories,
                },
            )



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

    # Call NVIDIA API with retries and schema validation
    try:
        score_data = await ai_call_with_retries(prompt, schema_validator=validate_score_schema, model=NVIDIA_MODEL)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "AI scoring failed after multiple attempts.",
                "suggestion": "Please refresh the page and re-upload the file to try again.",
                "error": str(e),
            },
        )





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


