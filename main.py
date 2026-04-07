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


def validate_validator_schema(obj: dict) -> (bool, str):
    if not isinstance(obj, dict):
        return False, "validator response is not an object"
    if "safe" not in obj or not isinstance(obj["safe"], bool):
        return False, "missing or invalid 'safe' boolean"
    if "reasons" not in obj or not isinstance(obj["reasons"], list):
        return False, "missing or invalid 'reasons' list"
    if "categories" not in obj or not isinstance(obj["categories"], dict):
        return False, "missing or invalid 'categories' object"
    # basic category value checks
    for k, v in obj["categories"].items():
        if not isinstance(v, bool):
            return False, f"category '{k}' is not boolean"
    return True, ""


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
    """Validate resume and job inputs via the AI safety filter.

    The model must return ONLY valid JSON with the following shape:
    {
      "safe": true|false,
      "reasons": ["..."],
      "categories": {"pii": bool, "illegal": bool, "sexual": bool, "hate": bool, "malware": bool}
    }
    """
    snippet = resume_text[:5000]
    jd = (job_description or "")[:2000]

    prompt = (
        "Evaluate whether the following resume (and optional job description) is safe to process for an automated resume scoring service. "
        "Return ONLY a JSON object exactly matching the schema described. Be strict: if any sensitive personal data, instructions for wrongdoing, malware, explicit sexual content, or hateful content exists, mark as not safe and list concise reasons and categories.\n\n"
        f"Resume:\n{snippet}\n\n"
        f"Job Title: {job_title or ''}\n\n"
        f"Job Description:\n{jd}\n\n"
        "JSON schema:\n{"
        '  "safe": true|false,\n'
        '  "reasons": ["..."],\n'
        '  "categories": {"pii": true|false, "illegal": true|false, "sexual": true|false, "hate": true|false, "malware": true|false}\n'
        '}'
    )

    system_content = (
        "You are a strict content safety filter. Return ONLY valid JSON as specified. "
        "Do not include any explanatory text or markdown."
    )

    try:
        parsed = await ai_call_with_retries(prompt, system_content=system_content, model=NVIDIA_MODEL, schema_validator=validate_validator_schema)
        return parsed
    except Exception:
        return {"safe": False, "reasons": ["Validation call failed or returned invalid JSON after multiple attempts"], "categories": {}}


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

    # Validate inputs using AI safety validator before proceeding
    validation = await validate_inputs(resume_text, job_title, job_description)
    if not validation.get("safe", False):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Input not safe to process",
                "reasons": validation.get("reasons", []),
                "categories": validation.get("categories", {}),
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


