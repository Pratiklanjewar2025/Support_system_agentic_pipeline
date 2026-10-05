from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import os

from evaluation.schemas import EvaluationRequest, EvaluationResponse
from evaluation.evaluator import run_evaluation

app = FastAPI(title="LLM Evaluation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/evaluation/run", response_model=EvaluationResponse)
def api_run_evaluation(req: EvaluationRequest):
    query = req.query or "Why is my database memory getting exhausted?"
    # Runs the evaluation for the single selected model
    result = run_evaluation(query=query, model=req.model, provider=req.provider or "gemini")
    return result

@app.get("/")
def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception as e:
        return HTMLResponse(f"<h1>Dashboard not found</h1><p>{e}</p>")

if __name__ == "__main__":
    print("Starting LLM Evaluation Server on http://localhost:8000")
    uvicorn.run("evaluation.server:app", host="0.0.0.0", port=8000, reload=False)
