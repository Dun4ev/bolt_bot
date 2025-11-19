import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import time
import random
import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

DEFAULT_CHAT_MODEL = "gpt-oss-20b"
DEFAULT_CARD_MODEL = "google/gemma-3n-e4b:2"
DEFAULT_ENDPOINT = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_BACKEND = "lm_studio"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_TIMEOUT = 60
DEFAULT_LM_STUDIO_TIMEOUT = 30

POEM_PROMPT = """
Ты — талантливый поэт, пишущий на русском языке. По заданному слову или фразе создай короткое стихотворение из 4–8 строк.

Требования:
- используй литературный русский язык и ритм;
- допускается любая рифма, но избегай повторов;
- обязательно упомяни исходное слово или фразу;
- добавь эмоциональный оттенок (лирический или вдохновляющий).

Вот исходная тема:
{product_text}
"""
POEM_HINT = "Напишите слово или фразу, и я составлю стихотворение."
POEM_BUTTON = "📝 Стихотворение"
BACKEND_BUTTON = "⚙️ Выбрать backend"


def load_env_file(env_file: str = ".env") -> None:
    """Загружает переменные окружения из локального .env, если он существует."""
    env_path = Path(env_file)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


load_env_file()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
LM_STUDIO_ENDPOINT = os.getenv("LM_STUDIO_ENDPOINT", DEFAULT_ENDPOINT)
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", DEFAULT_CHAT_MODEL)
LM_STUDIO_CARD_MODEL = os.getenv("LM_STUDIO_CARD_MODEL", DEFAULT_CARD_MODEL)
LLM_BACKEND = os.getenv("LLM_BACKEND", DEFAULT_BACKEND).lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

# Timeouts
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", DEFAULT_GEMINI_TIMEOUT))
LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", DEFAULT_LM_STUDIO_TIMEOUT))

GEMINI_ENABLED = bool(GEMINI_API_KEY)

if LLM_BACKEND == "gemini" and not GEMINI_ENABLED:
    raise RuntimeError("Для backend=gemini необходимо указать GEMINI_API_KEY в окружении")

CURRENT_BACKEND = LLM_BACKEND

# Функция для отправки промпта к выбранному backend
def ask_lmstudio(prompt):
    messages = [
        {"role": "system", "content": "Ты helpful ассистент."},
        {"role": "user", "content": prompt}
    ]
    return call_chat_backend(messages, LM_STUDIO_MODEL, temperature=0.7, max_tokens=1024)

# -----------------------------
# Пример функции и промпта для генерации товарной карточки с помощью LLM

def ask_llm(product_text):
    """Генерирует стихотворение по входной теме с помощью шаблона POEM_PROMPT."""
    prompt = POEM_PROMPT.format(product_text=product_text)
    messages = [
        {"role": "system", "content": "Ты профессионал по написанию стихов на русском языке."},
        {"role": "user", "content": prompt}
    ]
    return call_chat_backend(messages, LM_STUDIO_CARD_MODEL, temperature=0.4)


def call_chat_backend(
    messages: List[Dict[str, str]],
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Отправляет запрос в выбранный LLM backend и возвращает текст ответа."""
    backend = CURRENT_BACKEND
    target_model = model_name
    if backend == "gemini":
        target_model = GEMINI_MODEL or model_name
        return call_gemini_backend(messages, target_model, temperature, max_tokens)
    return call_lm_studio_backend(messages, target_model, temperature, max_tokens)


def call_lm_studio_backend(
    messages: List[Dict[str, str]],
    model_name: str,
    temperature: float,
    max_tokens: int | None,
) -> str:
    payload: Dict[str, Any] = {
        "model": model_name or LM_STUDIO_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    resp = requests.post(LM_STUDIO_ENDPOINT, json=payload, timeout=LM_STUDIO_TIMEOUT)
    resp.raise_for_status()
    content = resp.json()
    return content["choices"][0]["message"]["content"]


def call_gemini_backend(
    messages: List[Dict[str, str]],
    model_name: str,
    temperature: float,
    max_tokens: int | None,
) -> str:
    system_parts = [msg["content"] for msg in messages if msg.get("role") == "system"]
    conversation = [msg for msg in messages if msg.get("role") != "system"]
    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": ("user" if msg.get("role") != "assistant" else "model"),
                "parts": [{"text": msg.get("content", "")}],
            }
            for msg in conversation
        ]
    }
    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }
    gen_config: Dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = max_tokens
    payload["generationConfig"] = gen_config
    endpoint = GEMINI_ENDPOINT_TEMPLATE.format(model=(model_name or GEMINI_MODEL))
    params = {"key": GEMINI_API_KEY}
    
    for attempt in range(1, 6):
        try:
            resp = requests.post(endpoint, params=params, json=payload, timeout=GEMINI_TIMEOUT)
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt == 5:
                    raise
                sleep_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                logging.warning(f"Gemini 429 Rate Limit. Retrying in {sleep_time:.2f}s (Attempt {attempt}/5)")
                time.sleep(sleep_time)
                continue
            raise

    content = resp.json()
    candidates = content.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini API не вернул кандидатов")
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError("Gemini API не вернул текстовый ответ")
    return parts[0].get("text", "")


def main_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[POEM_BUTTON], [BACKEND_BUTTON]], resize_keyboard=True
    )


def backend_options_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton("LM Studio (локально)", callback_data="backend:lm_studio")
    ]
    gemini_label = "Gemini (Google AI Studio)"
    if not GEMINI_ENABLED:
        gemini_label += " ❌"
    buttons.append(InlineKeyboardButton(gemini_label, callback_data="backend:gemini"))
    keyboard = InlineKeyboardMarkup.from_row(buttons)
    return keyboard


async def backend_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Выберите источник ответов:", reply_markup=backend_options_keyboard()
    )


async def backend_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, backend = query.data.split(":", 1)
    global CURRENT_BACKEND
    if backend == "gemini" and not GEMINI_ENABLED:
        await query.edit_message_text(
            "Gemini API недоступен: укажите GEMINI_API_KEY в .env и перезапустите бота."
        )
        return
    CURRENT_BACKEND = backend
    await query.edit_message_text(
        f"Backend переключён на {describe_backend(backend)}."
    )


def describe_backend(backend: str) -> str:
    if backend == "gemini":
        return "Gemini (Google AI Studio)"
    return "LM Studio"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Жду твой промпт.", reply_markup=main_menu_keyboard()
    )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Меню доступных действий:", reply_markup=main_menu_keyboard()
    )

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = (update.message.text or "").strip()
    if question == POEM_BUTTON:
        context.user_data["mode"] = "poem"
        await update.message.reply_text(POEM_HINT)
        return
    if question == BACKEND_BUTTON:
        await backend_command(update, context)
        return

    await update.message.reply_text("Думаю...")
    
    try:
        if context.user_data.get("mode") == "poem":
            answer = ask_llm(question)
            # Reset mode after answering
            context.user_data["mode"] = None
        else:
            answer = ask_lmstudio(question)
        
        await update.message.reply_text(answer)
    except Exception as e:
        backend_name = describe_backend(CURRENT_BACKEND)
        await update.message.reply_text(
            f"Ошибка при общении с {backend_name}: {str(e)}"
        )

if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN не задан: добавьте его в .env или окружение")
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("backend", backend_command))
    app.add_handler(CallbackQueryHandler(backend_callback, pattern=r"^backend:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    app.run_polling()
