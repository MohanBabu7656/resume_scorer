
# Resume Scorer API (FastAPI + NVIDIA NIM)

A serverless FastAPI backend for scoring resumes using NVIDIA's NIM LLM API. No database required—results are returned instantly to the user.

## Features
- Upload a PDF resume and (optionally) a job description
- Resume is scored for ATS, skills, experience, formatting, grammar, and more
- Optionally, job match analysis if job info is provided
- Results are returned in the API response (not stored)
- Deployable to Vercel or run locally

## Requirements
- Python 3.14+
- NVIDIA NIM API key (for LLM scoring)

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
	 (If you want to use Supabase or other features, add those keys as well, but they are not required for stateless mode.)

4. **Run locally**
	 ```bash
	 uvicorn main:app --reload
	 ```
	 The API will be available at http://localhost:8000

## API Usage
- **POST** `/api/score-resume`
	- Form fields:
		- `file`: PDF resume file (required)
		- `job_title`: (optional)
		- `job_description`: (optional)
	- Returns: JSON with scores, feedback, and suggestions

## Deployment
- Deploy to Vercel using the [Vercel Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python)
- Set your NVIDIA_API_KEY as an environment variable in Vercel

## Example Response
```json
{
	"success": true,
	"filename": "resume.pdf",
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
	"feedback": {"summary": "Great resume!", ...},
	"job_match": {"match_score": 78, ...}
}
```

---
MIT License
