import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.llm import ask, LLMError

app = FastAPI(title="Помічник служби підтримки — ПР3")


class QuestionRequest(BaseModel):
    question: str


def load_context_rules() -> str:
    """
    Зчитує правила магазину з файлу context.md.
    Заміна context.md змінює правила без перезапуску чи зміни коду програми.
    """
    context_path = os.path.join(os.path.dirname(__file__), "..", "context.md")
    if not os.path.exists(context_path):
        raise HTTPException(
            status_code=500,
            detail="Файл правил context.md не знайдено на сервері."
        )
    try:
        with open(context_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Не вдалося прочитати файл правил: {exc}"
        )


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """
    Віддає головну сторінку веб-інтерфейсу.
    """
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="Шаблон index.html не знайдено.")


@app.post("/api/ask")
def api_ask(payload: QuestionRequest):
    """
    Обробляє запит користувача:
    - валідує вхідні дані;
    - завантажує актуальний контекст;
    - звертається до моделі;
    - обробляє помилки із поверненням відповідних HTTP-статусів.
    """
    user_message = payload.question.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Питання не може бути порожнім.")

    rules = load_context_rules()

    try:
        result = ask(user_message, rules)
        return {
            "answer": result["answer"],
            "model": result["model"],
            "elapsed": result["elapsed"]
        }
    except LLMError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Внутрішня помилка сервера при обробці звернення."
        )