from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import json
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import chromadb
from groq import Groq
from pydantic import BaseModel
import PyPDF2
from dotenv import load_dotenv

# 1. SETUP API AND APP (SECURE)

load_dotenv() # Loads the hidden .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from the .env file!")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Resume Optimizer API")
# Serve the static UI files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


# 2. START LOCAL VECTOR DATABASE

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="resumes")


# 3. DEFINE STRICT OUTPUT STRUCTURE

class BulletImprovement(BaseModel):
    original_bullet: str
    improved_bullet: str
    reasoning: str

class ResumeAnalysis(BaseModel):
    ats_score: int
    missing_keywords: list[str]
    grammar_issues: list[str]
    bullet_improvements: list[BulletImprovement]
    quantification_suggestions: list[str]


# 4. RAG HELPER FUNCTIONS

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# 5. THE API ROUTE

@app.post("/api/v1/optimize", response_model=ResumeAnalysis)
async def optimize_resume(
    target_job_title: str = Form(...),
    job_description: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Must be a PDF.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        resume_text = extract_text_from_pdf(tmp_path)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="PDF is empty or unreadable.")

        chunks = chunk_text(resume_text)
        doc_ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            metadatas=[{"source": file.filename}] * len(chunks),
            ids=doc_ids
        )

        print(f"Analyzing {file.filename} with Groq Llama-3...")

        prompt = f"""
You are an elite Tech Recruiter. Analyze this Resume against the Target Job.
Job Title: {target_job_title}
Job Description: {job_description}

Resume: {resume_text}

Provide a brutally honest ATS score (0-100), missing keywords, grammar issues,
rewrite 3 weak bullet points, and suggest where to add metrics.
You MUST return ONLY a valid JSON object matching this exact structure:
{{"ats_score": int, "missing_keywords": [str], "grammar_issues": [str], "bullet_improvements": [{{"original_bullet": str, "improved_bullet": str, "reasoning": str}}], "quantification_suggestions": [str]}}
Do not include any markdown formatting, backticks, or conversational text. Return raw JSON only.
"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        raw_json_string = response.choices[0].message.content
        return json.loads(raw_json_string)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("\n=== ERROR TRACEBACK ===")
        print(error_details)
        print("=======================\n")
        raise HTTPException(status_code=500, detail=f"CRASH: {str(e)} | TRACE: {error_details}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)