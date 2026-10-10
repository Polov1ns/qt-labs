"""Веб-рівень застосунку: сторінка діалогу і JSON-ендпоінт.

Цей файл не має знати ані про провайдера моделі, ані про те, як
складається запит, ані про те, як виглядає схема відповіді, — усе це
лишається в `app/llm.py` і `app/schema.py`. Тут вирішується інше: що
застосунок приймає від сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr4:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import llm

app = FastAPI(title="Помічник служби підтримки — ПР4")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Turn(BaseModel):
    """Одна репліка розмови: `user` — клієнт, `assistant` — помічник."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Те, що надсилає сторінка: нове повідомлення й розмову до нього.

    Історію сторінка веде сама і надсилає з кожним запитом — це
    найпростіший спосіб, за якого сервер не зберігає стану. Чи довіряти
    такій історії, чи вести власну на сервері — рішення з розділу 2
    практичної роботи. Каркас лише передає її далі.
    """

    message: str
    history: list[Turn] = []


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель.

    Контекст — це дані застосунку, а не знання моделі. Він живе окремим
    файлом і передається в запит разом з історією та зверненням.
    """
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку діалогу."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    """Повернути структуровану відповідь помічника у форматі JSON.

    Сторінка очікує обʼєкт із полями `result` (перевірена відповідь моделі;
    текст для клієнта — у `result.reply`), `model`, `elapsed` і `usage`.
    Якщо назвете поля інакше — змініть сторінку, вона ваша.

    Збої тут не оброблено. Перенесіть обробку з ПР3 (таймаут, ліміт,
    недоступність, невірний ключ, порожнє повідомлення) і додайте новий
    вид збою: модель відповіла, але відповідь не пройшла перевірку за
    схемою. Який HTTP-статус йому відповідає і що побачить клієнт —
    вирішуєте ви.
    """
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Повідомлення не може бути порожнім")

    history = [turn.model_dump() for turn in payload.history]
    try:
        return llm.ask(payload.message, history, load_context())
    except llm.LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))