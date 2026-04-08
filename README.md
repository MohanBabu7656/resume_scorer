# Resume Scorer API (FastAPI + NVIDIA NIM)

A serverless FastAPI backend for scoring resumes using NVIDIA's NIM LLM API. No database required—results are returned instantly to the user.

## Features
- **AI Safety Filter:** Uses `meta/llama-guard-4-12b` to validate the safety of uploaded resumes and job descriptions before processing.
- **Detailed Scoring:** Upload a PDF resume and (optionally) a job description. The resume is scored for ATS compatibility, skills, experience, formatting, grammar, and more.
- **Letter Grades:** Automatically calculates a letter grade (A+, A, B, etc.) based on the overall resume score.
- **Early Input Validation:** Immediately validates that if a job title is provided, a description is also provided (and vice versa) to save compute time.
- **Privacy Policy Endpoint:** Built-in endpoint to serve data privacy and usage information to frontend clients.
- **Job Match Analysis:** Compares the resume against the provided job info (if supplied) to determine how well the candidate fits the role.
- **Stateless:** Results are returned entirely in the API response (not stored).
- **Deployable:** Easily deployable to Vercel or run locally.

## Requirements
- Python 3.10+ (Recommended 3.12+)
- NVIDIA NIM API key (for LLM scoring and safety filtering)

## Setup
1. **Clone the repository**
2. **Install dependencies**
	 ```bash
	 pip install -r requirements.txt
	 # or, if using pyproject.toml
	 pip install .
	 ```
3. **Create a `.env` file** in the project root with:
	 ```env
	 NVIDIA_API_KEY=your_nvidia_api_key
	 ```
	 *(If you want to use Supabase or other features, add those keys as well, but they are not required for stateless mode.)*

4. **Run locally**
	 ```bash
	 uvicorn main:app --reload
	 ```
	 The API will be available at http://localhost:8000

## API Usage

### `POST /api/score-resume`
Scores a given PDF resume and returns a detailed JSON breakdown.
- **Form fields:**
	- `file`: PDF resume file (required, max 5MB)
	- `job_title`: (optional, but required if `job_description` is provided)
	- `job_description`: (optional, but required if `job_title` is provided)

**Example Response:**
```json
{
	"success": true,
	"filename": "resume.pdf",
	"grade": "B+",
	"scores": {
		"overall": 85,
		"ats": 90,
		"skills": 80,
		"experience": 75,
		"formatting": 95,
		"grammar": 88
	},
	"strengths": ["Strong technical skills", "Clear formatting"],
	"weaknesses": ["Limited leadership experience"],
	"suggestions": ["Add more leadership examples"],
	"feedback": {
		"summary": "Great resume!",
		"ats": "...",
		"skills": "..."
	},
	"job_match": {
		"match_score": 78,
		"matched_keywords": ["Python", "FastAPI"],
		"missing_keywords": ["Docker"]
	}
}
```

### `GET /api/privacy-policy`
Returns the application's data privacy and usage policy.

**Example Response:**
```json
{
	"title": "Data Privacy and Usage Policy",
	"data_collection": "We temporarily process the resume PDF, job title, and job description you provide.",
	"data_usage": "Your data is used exclusively to score your resume and evaluate its match against the provided job context using our AI models.",
	"data_retention": "Uploaded files and extracted texts are processed in memory for the duration of the request and are not permanently stored on our servers.",
	"third_party_sharing": "Your data is sent securely to our AI providers (like NVIDIA APIs) strictly for immediate processing. It is not used to train their foundational models, nor is it sold to third parties.",
	"security": "We use advanced AI safety filters (Llama Guard) and secure API endpoints to ensure your data is processed safely and securely."
}
```

## Deployment
- Deploy to Vercel using the [Vercel Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python)
- Set your `NVIDIA_API_KEY` as an environment variable in the Vercel dashboard.

---
