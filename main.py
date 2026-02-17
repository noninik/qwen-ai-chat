from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)

# Хранилище истории чата (для каждой сессии в памяти)
chat_histories = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "messages": [],
    })

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_message: str = Form(...)):
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-Coder-Next:novita",
            messages=[
                {
                    "role": "system",
                    "content": "Ты полезный AI-ассистент. Отвечай на русском языке. Будь дружелюбным и понятным."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
        )
        bot_reply = response.choices[0].message.content
    except Exception as e:
        bot_reply = f"Ошибка: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_message": user_message,
        "bot_reply": bot_reply,
    })
