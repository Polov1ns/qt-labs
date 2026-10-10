import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.llm import ask_support_bot

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

def load_context_rules() -> str:
    context_path = os.path.join(os.path.dirname(__file__), "..", "context.md")
    if os.path.exists(context_path):
        with open(context_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Правила не знайдено."

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Шаблон index.html не знайдено."

@app.post("/api/ask")
def handle_ask(payload: QuestionRequest):
    user_message = payload.question
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Питання не може бути порожнім.")
    
    rules = load_context_rules()
    result = ask_support_bot(user_message, rules)
    
    return {
        "answer": result["answer"],
        "model": result["model"],
        "elapsed": result["duration"]
    }