"""Модуль роботи з мовною моделлю: єдине місце застосунку, яке знає про API.

Тут живуть налаштування доступу, системна інструкція з прикладами,
збирання запиту з частин і правило, за яким історія вміщується в бюджет
токенів. Веб-рівень (`app/main.py`) отримує звідси перевірений результат
і нічого не знає ані про провайдера, ані про склад повідомлень. Схема
відповіді та її перевірка — в `app/schema.py`.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі
рішення з розділу 2 практичної роботи:

* що входить до системної інструкції: роль, обмеження, формат, приклади
  межових випадків — і які саме приклади;
* де в запиті стоять правила, де історія, де поточне звернення;
* що класти в історію з боку помічника: увесь JSON чи лише текст для
  клієнта;
* чим рахувати токени й що відкидати першим, коли бюджет вичерпано;
* чи передавати схему провайдеру через `response_format`, чи просити JSON
  текстом — і що робити з відповіддю, яка не пройшла перевірку;
* що саме зараховувати до виміряного часу — з повторами чи без.

Обробку збоїв із ПР3 (таймаут, ліміт, недоступність, невірний ключ)
перенесіть сюди. Налаштування, як і раніше, читаються з `.env`; ключ
доступу — секрет.
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from . import schema

load_dotenv()

# Доступ до сервісу. Значень тут немає навмисно — вони у вашому `.env`.
BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")

# Параметри генерації.
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

# Скільки токенів може займати весь запит без відповіді: інструкція,
# правила, історія, звернення. Значення — відправна точка; чим його
# рахувати і що скорочувати, коли він вичерпаний, — ваше рішення.
TOKEN_BUDGET = int(os.getenv("LLM_TOKEN_BUDGET", "3000"))


class LLMError(Exception):
    """Помилка роботи з моделлю, зрозуміла веб-рівню.

    Заготовка. У ПР3 ви вже вирішили, чи розрізняти збої за типами, —
    перенесіть те рішення сюди. Тут додається ще один вид збою: модель
    відповіла, але відповідь не пройшла перевірку за схемою. Він не
    схожий на решту: сервіс працює, ключ дійсний, а результату все одно
    немає.
    """


_client = None


def get_client():
    """Повернути готовий до роботи клієнт сервісу.

    Як у ПР3: створюється один раз, а не на кожен запит; адреса сервісу
    береться з `BASE_URL`, ключ — з `API_KEY`.
    """
    global _client
    if not API_KEY:
        raise LLMError("API ключ відсутній у змінних середовища (.env)")
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def estimate_tokens(text: str) -> int:
    """Оцінити, скільки токенів займе текст.

    Точну кількість знає лише токенізатор провайдера; для рішення «чи
    вміщується запит у бюджет» досить оцінки. Наскільки вона розходиться
    зі справжньою, покаже поле `usage` у відповіді — порівняйте й
    відкалібруйте.
    """
    if not text:
        return 0
    return max(1, len(text) // 3)


def fit_budget(history: list[dict], budget: int) -> list[dict]:
    """Повернути ту частину історії, яка вміщується в бюджет.

    Бюджет — на весь запит, а історія — лише одна його частина:
    інструкція, правила й поточне звернення теж займають місце, і резерв
    на відповідь теж. Що відкидати першим — найстаріші репліки, середину,
    усе крім останніх — і що не можна відкинути ніколи, вирішуєте ви.
    Наслідки цього рішення видно на діалогах із `compare/`: у них
    потрібний факт названо на початку.
    """
    current = 0
    trimmed = []
    for msg in reversed(history):
        tokens = estimate_tokens(msg.get("content", ""))
        if current + tokens > budget:
            break
        current += tokens
        trimmed.insert(0, msg)
    return trimmed


def build_messages(message: str, history: list[dict], context: str) -> list[dict]:
    """Скласти список повідомлень для моделі.

    Частини запиту лишаються окремими: системна інструкція з обмеженнями,
    форматом і прикладами; правила з `context.md`; історія розмови як
    повідомлення `user` і `assistant`; поточне звернення. Історія перед
    цим проходить через `fit_budget`.

    Приклади межових випадків — частина інструкції, а не історії: модель
    не має плутати їх зі справжньою розмовою.
    """
    system_instruction = (
        "Ти помічник служби підтримки інтернет-магазину Сузір'я.\n"
        "Відповідай українською мовою на підставі наданих правил магазину.\n\n"
        "Правила магазину:\n"
        f"{context}\n\n"
        "Обмеження:\n"
        "1. Не вигадуй інформацію, якої немає в правилах. Якщо відповіді немає, встанови rule_matched false та transfer_to_operator true.\n"
        "2. Якщо клієнт називав номер замовлення (6 цифр), обов'язково зафіксуй його у полі order_number.\n"
        "3. Тема звернення (topic) обирається з переліку: Доставка, Оплата, Повернення, Гарантія, Підтримка, Інше.\n"
        "4. Відповідь обов'язково повертай у форматі JSON з полями: reply, topic, rule_matched, needs_clarification, transfer_to_operator, order_number.\n\n"
        "Приклади:\n"
        "1. Запит: Скільки триває доставка. Відповідь: rule_matched true, needs_clarification false, transfer_to_operator false.\n"
        "2. Запит: Чи можна оплатити криптовалютою. Відповідь: rule_matched false, transfer_to_operator true.\n"
        "3. Запит: Хочу повернути товар. Відповідь: needs_clarification true.\n"
    )

    messages = [{"role": "system", "content": system_instruction}]

    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": message})
    return messages


def ask(message: str, history: list[dict], context: str) -> dict:
    """Поставити моделі питання й повернути перевірений результат.

    Повертає щонайменше перевірену за схемою відповідь (`app/schema.py`),
    назву моделі, час виконання і `usage` — кількість токенів запиту й
    відповіді. Точний склад полів — ваше рішення; сторінка каркаса очікує
    `result`, `model`, `elapsed`, `usage`.

    Тут же вирішується, що робити з відповіддю, яка не пройшла перевірку:
    повторити запит із текстом помилки, підставити безпечну відповідь чи
    підняти помилку — і скільки разів повторювати.
    """
    client = get_client()

    used = estimate_tokens(context) + estimate_tokens(message) + 500
    budget = max(200, TOKEN_BUDGET - used)
    trimmed_history = fit_budget(history, budget)

    messages = build_messages(message, trimmed_history, context)

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=TIMEOUT,
        )
    except Exception as e:
        raise LLMError(f"Помилка звернення до моделі: {e}")

    elapsed = round(time.time() - start_time, 2)
    raw_content = response.choices[0].message.content or ""
    usage_obj = response.usage

    try:
        result = schema.validate(raw_content)
    except Exception as e:
        raise LLMError(f"Невалідна відповідь моделі за схемою: {e}")

    return {
        "result": result,
        "model": MODEL,
        "elapsed": elapsed,
        "usage": {
            "prompt_tokens": usage_obj.prompt_tokens if usage_obj else 0,
            "completion_tokens": usage_obj.completion_tokens if usage_obj else 0,
        },
    }