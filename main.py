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
import fitz  # PyMuPDF
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


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        num_pages = len(doc)
        for page in doc:
            extracted = page.get_text()
            if extracted:
                text += extracted + "\n"
    return text.strip(), num_pages


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
    for list_key in ("strengths", "weaknesses"):
        if list_key in obj and not isinstance(obj[list_key], list):
            return False, f"'{list_key}' must be a list"
    
    if "suggestions" in obj:
        if not isinstance(obj["suggestions"], list):
            return False, "'suggestions' must be a list"
        for item in obj["suggestions"]:
            if not isinstance(item, dict) or "current" not in item or "suggested" not in item or "reason" not in item:
                return False, "'suggestions' items must be objects with 'current', 'suggested', and 'reason' keys"

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


async def check_is_resume_llm(resume_text: str) -> tuple[bool, str]:
    """Use an LLM to strictly classify if the text is a resume or not."""
    snippet = resume_text[:2000]
    prompt = f"""Analyze the following text and determine if it represents a resume (curriculum vitae). 
A valid resume MUST contain professional work history/experience, education details, or a detailed list of professional skills.
If the document is ONLY a certificate of completion, an award, a cover letter, a job description, an ID card, a recipe, a code snippet, or a random article, it is NOT a resume and you must return false.

Text:
{snippet}

Respond ONLY with a JSON object in this exact format:
{{
  "is_resume": true or false,
  "reason": "<brief explanation>"
}}"""
    
    try:
        raw = await call_nvidia_api(
            prompt=prompt,
            system_content="You are a strict document classifier. You must only output valid JSON.",
            model=NVIDIA_MODEL
        )
        parsed = parse_json_response(raw)
        return parsed.get("is_resume", True), parsed.get("reason", "Unknown reason.")
    except Exception as e:
        # Fallback to True if the LLM classification itself errors out to not block users
        return True, ""


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


def get_letter_grade(score: float) -> str:
    """Convert a numeric score (0-100) to a letter grade."""
    if score >= 95: return "A+"
    elif score >= 90: return "A"
    elif score >= 85: return "B+"
    elif score >= 80: return "B"
    elif score >= 75: return "C+"
    elif score >= 70: return "C"
    elif score >= 60: return "D"
    else: return "F"


@app.get("/")
def root():
    return {"message": "Resume Scorer API is running 🚀", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/privacy-policy")
def privacy_policy():
    return {
        "title": "Data Privacy and Usage Policy",
        "data_collection": "We temporarily process the resume PDF, job title, and job description you provide.",
        "data_usage": "Your data is used exclusively to score your resume and evaluate its match against the provided job context using our AI models.",
        "data_retention": "Uploaded files and extracted texts are processed in memory for the duration of the request and are not permanently stored on our servers.",
        "third_party_sharing": "Your data is sent securely to our AI providers (like NVIDIA APIs) strictly for immediate processing. It is not used to train their foundational models, nor is it sold to third parties.",
        "security": "We use advanced AI safety filters (Llama Guard) and secure API endpoints to ensure your data is processed safely and securely."
    }


async def process_resume_scoring(
    file: UploadFile,
    job_title: str = None,
    job_description: str = None,
):
    # 1. Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 2. Validate file size
    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 5MB.")

    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF document.")

    # 3. Extract text
    try:
        resume_text, num_pages = extract_text_from_pdf(file_bytes)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF. The file may be corrupted or password-protected.")

    if len(resume_text) < 100:
        raise HTTPException(status_code=422, detail="Resume appears to be empty or unreadable. Please ensure the PDF contains selectable text, not just images.")

    if len(resume_text) > 20000:
        raise HTTPException(status_code=422, detail="Resume text is too long. Please ensure your resume is concise (max ~5-6 pages of text).")

    # 4. Word count and stats calculation
    words = len(resume_text.split())
    
    if words < 40:
        raise HTTPException(status_code=422, detail="The uploaded document is too short to be a valid resume. A resume must contain detailed work experience, education history, or professional skills.")

    # General rule of thumb: ~200-800 words per page is a healthy range
    words_per_page = words / max(1, num_pages)
    word_count_optimal = (200 <= words_per_page <= 800) and (200 <= words <= 2000)

    # 5 & 6. Run LLM Guard and AI safety validation concurrently to save time
    check_task = check_is_resume_llm(resume_text)
    validate_task = validate_inputs(resume_text, job_title, job_description)
    
    (is_resume, reason), validation = await asyncio.gather(check_task, validate_task)

    if not is_resume:
        raise HTTPException(status_code=422, detail=f"The uploaded document is not a valid resume. AI Guard explanation: {reason}")

    if not validation.get("safe", False):
        categories = validation.get("categories", {})
        flagged_cats = set(categories.keys())
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

    # 7. Build scoring prompt
    job_context = ""
    if job_title and job_description:
        job_context = f"""
Also evaluate how well this resume matches the following job:
Job Title: {job_title}
Job Description: {job_description[:2000]}
Include a "job_match" object with: match_score (0-100), matched_keywords (list), missing_keywords (list).
"""

    prompt = f"""
You are an expert ATS specialist, technical recruiter, and executive resume writer. Analyze this resume and return a JSON object with the following structure:
{{
  "overall_score": <0-100>,
  "ats_score": <0-100>,
  "skills_score": <0-100>,
  "experience_score": <0-100>,
  "formatting_score": <0-100>,
  "grammar_score": <0-100>,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "suggestions": [
    {{
      "section": "Experience",
      "current": "Exact quote of the original flawed sentence",
      "suggested": "Complete, polished rewrite ready to be copy-pasted",
      "reason": "Clear explanation of the exact difference, why it is better, and any technical corrections made"
    }}
  ],
  "feedback": {{
    "summary": "...",
    "ats": "...",
    "skills": "...",
    "experience": "...",
    "formatting": "...",
    "grammar": "..."
  }}{",\n  \"job_match\": {{\"match_score\": 0, \"matched_keywords\": [], \"missing_keywords\": []}}" if job_context else ""}
}}

CRITICAL INSTRUCTIONS FOR SUGGESTIONS:
1. ONLY High-Value Improvements: Focus on the most critical weak points, especially technical inaccuracies, poor phrasing, or weak impact.
2. ACTIONABLE & SPECIFIC REWRITES: Never give vague advice. You MUST provide the fully rewritten, professional sentence in the "suggested" field. 
3. EXACT QUOTES: "current" MUST be a verbatim copy-paste from the resume text so the user knows exactly what you are changing.
4. EXPLAIN THE DIFFERENCE: In the "reason" field, explicitly state what changed between "current" and "suggested" and why the new version is stronger.
5. TECHNICAL ACCURACY & CONTEXT: Identify and correct technical mistakes or incorrect terminology based on the candidate's profile (e.g., fixing "Java Script" to "JavaScript", correcting misused tools, or adding relevant technical context that aligns with their overall experience). 
6. METRICS & IMPACT: Transform task-based bullets into achievement-based bullets. Use action verbs and insert placeholders (e.g., "[Metric]%") if specific numbers are missing but logically belong.
7. ATS OPTIMIZATION: Naturally weave in missing industry keywords if a job description is provided, or standard industry terms if not.
8. IGNORE PARSING ARTIFACTS: Disregard bad spacing, missing spaces, or weird symbols. Fix them in your rewrite silently.
9. Limit suggestions to the top 3-5 most impactful rewrites.

Resume Text:
{resume_text[:5000]}
{job_context}

Return ONLY the valid JSON object. Do not include markdown formatting like ```json or any conversational text.
"""

    # 8. Call NVIDIA API with retries and schema validation
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

    overall_score = score_data.get("overall_score", 0)

    return JSONResponse({
        "success": True,
        "filename": file.filename,
        "grade": get_letter_grade(overall_score),
        "stats": {
            "pages": num_pages,
            "word_count": words,
            "word_count_optimal": word_count_optimal
        },
        "scores": {
            "overall": overall_score,
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

@app.post("/api/score-resume")
async def score_resume(file: UploadFile = File(...)):
    """Score only the resume without a job description."""
    return await process_resume_scoring(file)

@app.post("/api/score-job-match")
async def score_job_match(
    file: UploadFile = File(...),
    job_title: str = Form(...),
    job_description: str = Form(...),
):
    """Score the resume against a specific job title and job description."""
    if not job_title or not job_title.strip():
        raise HTTPException(status_code=400, detail="Job title is required for this endpoint.")
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is required for this endpoint.")
        
    return await process_resume_scoring(file, job_title, job_description)
