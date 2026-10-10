import os
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
)

# Читаємо правильну назву змінної з .env
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-3.8-flash")

def ask_support_bot(user_message: str, rules_context: str) -> dict:
    system_content = (
        "Ти — помічник служби підтримки інтернет-магазину. "
        "Відповідай клієнту українською мовою, спираючись ВИКЛЮЧНО на надані правила.\n\n"
        f"Офіційні правила магазину:\n{rules_context}\n\n"
        "Якщо відповіді немає в правилах, прямо скажи про це і нічого не вигадуй. "
        "Якщо питання незрозуміле, уточни деталі."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message}
    ]

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.2)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 500)),
            timeout=10.0
        )
        
        duration = round(time.time() - start_time, 2)
        answer_text = response.choices[0].message.content

        return {
            "answer": answer_text,
            "model": MODEL_NAME,
            "duration": duration,
            "error": None
        }

    except RateLimitError:
        return {
            "answer": "Сервер перевантажений (перевищено ліміт запитів). Будь ласка, спробуйте пізніше.",
            "model": MODEL_NAME,
            "duration": round(time.time() - start_time, 2),
            "error": "rate_limit_exceeded"
        }
    except APITimeoutError:
        return {
            "answer": "Час очікування відповіді від сервера минув. Спробуйте ще раз.",
            "model": MODEL_NAME,
            "duration": round(time.time() - start_time, 2),
            "error": "timeout"
        }
    except APIError as e:
        print(f"Помилка OpenAI API: {e}")
        return {
            "answer": f"Виникла технічна помилка при зверненні до служби підтримки: {e.message if hasattr(e, 'message') else str(e)}",
            "model": MODEL_NAME,
            "duration": round(time.time() - start_time, 2),
            "error": str(e)
        }