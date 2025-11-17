import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

TELEGRAM_TOKEN = 'ВАШ_ТОКЕН_БОТА'                   # замени этим свой токен от BotFather
LM_STUDIO_ENDPOINT = 'http://127.0.0.1:1234/v1/chat/completions'  # сюда ставь URL из LM Studio

# Функция для отправки промпта к LM Studio (OpenAI-совместимый API)
def ask_lmstudio(prompt):
    payload = {
        "model": "gpt-oss-20b",    # подставь своё имя модели
        "messages": [
            {"role": "system", "content": "Ты helpful ассистент."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    resp = requests.post(LM_STUDIO_ENDPOINT, json=payload)
    resp.raise_for_status()
    content = resp.json()
    return content["choices"][0]["message"]["content"]

# -----------------------------
# Пример функции и промпта для генерации товарной карточки с помощью LLM

def ask_llm(product_text):
    """
    Формирует промпт для создания товарной карточки на маркетплейсы (Wildberries, Ozon, Яндекс.Маркет)
    и отправляет его в LM Studio (или аналогичный endpoint) для генерации результата.
    
    Пример промпта, который отправляется модели:

    Ты — эксперт по созданию карточек товаров для маркетплейсов (Wildberries, Ozon, Яндекс.Маркет).
    По кратким характеристикам товара создай полную товарную карточку.

    Формат ответа:
    Название:
    <краткое, привлекательное название>
    Краткое описание (2–3 предложения):
    <описание>
    Преимущества:
    - пункт 1
    - пункт 2
    - пункт 3
    SEO-блок (под описание):
    <абзац с ключевыми словами>
    Вот характеристики товара:
    {product_text}
    """
    prompt = f"""
Ты — эксперт по созданию карточек товаров для маркетплейсов (Wildberries, Ozon, Яндекс.Маркет).
По кратким характеристикам товара создай полную товарную карточку.

Формат ответа:
Название:
<краткое, привлекательное название>
Краткое описание (2–3 предложения):
<описание>
Преимущества:
- пункт 1
- пункт 2
- пункт 3

SEO-блок (под описание):
<абзац с ключевыми словами>

Вот характеристики товара:  
{product_text}
"""
    payload = {
        "model": "google/gemma-3n-e4b:2",  # примени имя своей модели
        "messages": [
            {"role": "system", "content": "Ты профессионал по генерации товарных карточек."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }
    resp = requests.post(LM_STUDIO_ENDPOINT, json=payload)
    resp.raise_for_status()
    content = resp.json()
    return content["choices"][0]["message"]["content"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Жду твой промпт.")

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    await update.message.reply_text("Думаю...")
    try:
        answer = ask_lmstudio(question)
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"Ошибка при общении с LM Studio: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    app.run_polling()