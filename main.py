from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
import os
import uuid
import json
import markdown

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)

# Доступные модели
MODELS = {
    "Qwen3 Coder": "Qwen/Qwen3-Coder-Next:novita",
    "Qwen3 235B": "Qwen/Qwen3-235B-A22B",
    "DeepSeek R1": "deepseek-ai/DeepSeek-R1",
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct",
    "Gemma 3 27B": "google/gemma-3-27b-it",
    "Phi-4": "microsoft/Phi-4",
    "Mistral Small": "mistralai/Mistral-Small-24B-Instruct-2501",
}

# Готовые роли
ROLES = {
    "Ассистент": "Ты полезный AI-ассистент. Отвечай на русском языке. Будь дружелюбным и понятным.",
    "Программист": "Ты опытный программист. Пиши чистый, рабочий код с комментариями. Объясняй решения. Отвечай на русском.",
    "Учитель": "Ты терпеливый учитель. Объясняй сложные вещи простыми словами, приводи примеры и аналогии. Отвечай на русском.",
    "Переводчик": "Ты профессиональный переводчик. Переводи текст максимально точно и естественно. Если язык не указан — переводи между русским и английским.",
    "Шутник": "Ты весёлый собеседник. Отвечай с юмором, шутками и мемами, но по делу. Отвечай на русском.",
    "Писатель": "Ты талантливый писатель. Пиши красивые, грамотные тексты. Помогай с сочинениями, статьями, историями. Отвечай на русском.",
    "Аналитик": "Ты аналитик данных. Разбирай информацию, находи закономерности, делай выводы. Структурируй ответы. Отвечай на русском.",
}

# Хранилище сессий
chat_sessions = {}

# Счётчик токенов (примерный)
token_counter = {"total": 0}


def estimate_tokens(text):
    """Примерный подсчёт токенов"""
    return len(text) // 3


def md_to_html(text):
    """Конвертация Markdown в HTML"""
    # Обработка блоков кода с подсветкой
    extensions = ['fenced_code', 'tables', 'nl2br']
    html = markdown.markdown(text, extensions=extensions)
    return html


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        "messages": [],
        "model": "Qwen3 Coder",
        "role": "Ассистент",
    }

    return templates.TemplateResponse("index.html", {
        "request": request,
        "messages": [],
        "session_id": session_id,
        "models": MODELS,
        "roles": ROLES,
        "selected_model": "Qwen3 Coder",
        "selected_role": "Ассистент",
        "token_count": token_counter["total"],
        "chat_list": _get_chat_list(),
    })


@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    user_message: str = Form(...),
    session_id: str = Form(...),
    model_name: str = Form("Qwen3 Coder"),
    role_name: str = Form("Ассистент"),
):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "messages": [],
            "model": model_name,
            "role": role_name,
        }

    session = chat_sessions[session_id]
    session["model"] = model_name
    session["role"] = role_name
    history = session["messages"]

    # Добавляем сообщение пользователя
    history.append({
        "role": "user",
        "content": user_message,
        "html": user_message,
    })

    try:
        # Формируем запрос
        system_prompt = ROLES.get(role_name, ROLES["Ассистент"])
        model_id = MODELS.get(model_name, MODELS["Qwen3 Coder"])

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        response = client.chat.completions.create(
            model=model_id,
            messages=api_messages,
        )

        bot_reply = response.choices[0].message.content
        bot_html = md_to_html(bot_reply)

        # Считаем токены
        tokens_used = estimate_tokens(user_message + bot_reply)
        token_counter["total"] += tokens_used

    except Exception as e:
        bot_reply = f"Ошибка: {str(e)}"
        bot_html = f"<p style='color: #ff6b6b;'>⚠️ {bot_reply}</p>"

    # Добавляем ответ бота
    history.append({
        "role": "assistant",
        "content": bot_reply,
        "html": bot_html,
    })

    # Ограничиваем историю
    if len(history) > 50:
        session["messages"] = history[-50:]

    # Сохраняем название чата (по первому сообщению)
    if "title" not in session:
        session["title"] = user_message[:30] + ("..." if len(user_message) > 30 else "")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "messages": session["messages"],
        "session_id": session_id,
        "models": MODELS,
        "roles": ROLES,
        "selected_model": model_name,
        "selected_role": role_name,
        "token_count": token_counter["total"],
        "chat_list": _get_chat_list(),
        "current_chat_id": session_id,
    })


@app.get("/new", response_class=HTMLResponse)
async def new_chat(request: Request):
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        "messages": [],
        "model": "Qwen3 Coder",
        "role": "Ассистент",
    }

    return templates.TemplateResponse("index.html", {
        "request": request,
        "messages": [],
        "session_id": session_id,
        "models": MODELS,
        "roles": ROLES,
        "selected_model": "Qwen3 Coder",
        "selected_role": "Ассистент",
        "token_count": token_counter["total"],
        "chat_list": _get_chat_list(),
    })


@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def load_chat(request: Request, session_id: str):
    if session_id not in chat_sessions:
        return await new_chat(request)

    session = chat_sessions[session_id]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "messages": session["messages"],
        "session_id": session_id,
        "models": MODELS,
        "roles": ROLES,
        "selected_model": session.get("model", "Qwen3 Coder"),
        "selected_role": session.get("role", "Ассистент"),
        "token_count": token_counter["total"],
        "chat_list": _get_chat_list(),
        "current_chat_id": session_id,
    })


@app.get("/clear/{session_id}", response_class=HTMLResponse)
async def clear_chat(request: Request, session_id: str):
    if session_id in chat_sessions:
        model = chat_sessions[session_id].get("model", "Qwen3 Coder")
        role = chat_sessions[session_id].get("role", "Ассистент")
        chat_sessions[session_id] = {
            "messages": [],
            "model": model,
            "role": role,
        }

    return templates.TemplateResponse("index.html", {
        "request": request,
        "messages": [],
        "session_id": session_id,
        "models": MODELS,
        "roles": ROLES,
        "selected_model": model,
        "selected_role": role,
        "token_count": token_counter["total"],
        "chat_list": _get_chat_list(),
    })


@app.get("/delete/{session_id}")
async def delete_chat(request: Request, session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return await new_chat(request)


@app.get("/export/{session_id}")
async def export_chat(session_id: str):
    if session_id not in chat_sessions:
        return JSONResponse({"error": "Chat not found"}, status_code=404)

    session = chat_sessions[session_id]
    messages = session["messages"]

    # Формируем текстовый файл
    text = "=== AI Chat Export ===\n\n"
    for msg in messages:
        role = "👤 Вы" if msg["role"] == "user" else "🤖 AI"
        text += f"{role}:\n{msg['content']}\n\n{'─' * 40}\n\n"

    return StreamingResponse(
        iter([text]),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=chat_{session_id[:8]}.txt"}
    )


def _get_chat_list():
    """Список всех чатов для сайдбара"""
    chats = []
    for sid, session in chat_sessions.items():
        if session.get("messages"):
            chats.append({
                "id": sid,
                "title": session.get("title", "Новый чат"),
            })
    return chats[-20:]  # Последние 20
