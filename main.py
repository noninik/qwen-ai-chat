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

# ═══════ МОДЕЛИ ═══════
MODELS = {
    "Qwen3 Coder": "Qwen/Qwen3-Coder-Next:novita",
    "Qwen3 235B": "Qwen/Qwen3-235B-A22B",
    "DeepSeek R1": "deepseek-ai/DeepSeek-R1",
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct",
    "Gemma 3 27B": "google/gemma-3-27b-it",
    "Phi-4": "microsoft/Phi-4",
    "Mistral Small": "mistralai/Mistral-Small-24B-Instruct-2501",
}

# ═══════ РОЛИ ═══════
ROLES = {
    "Ассистент": "Ты полезный AI-ассистент. Отвечай на русском языке. Будь дружелюбным и понятным.",
    "Программист": "Ты опытный программист. Пиши чистый, рабочий код с комментариями. Объясняй решения. Отвечай на русском. Если код длинный — пиши полностью, не сокращай.",
    "Учитель": "Ты терпеливый учитель. Объясняй сложные вещи простыми словами, приводи примеры и аналогии. Отвечай на русском.",
    "Переводчик": "Ты профессиональный переводчик. Переводи текст максимально точно и естественно. Если язык не указан — переводи между русским и английским.",
    "Шутник": "Ты весёлый собеседник. Отвечай с юмором, шутками и мемами, но по делу. Отвечай на русском.",
    "Писатель": "Ты талантливый писатель. Пиши красивые, грамотные тексты. Помогай с сочинениями, статьями, историями. Отвечай на русском.",
    "Аналитик": "Ты аналитик данных. Разбирай информацию, находи закономерности, делай выводы. Структурируй ответы. Отвечай на русском.",
}

# ═══════ НАСТРОЙКИ ЛИМИТОВ ═══════
MAX_TOKENS_RESPONSE = 16384       # Максимум токенов в ответе (много кода!)
MAX_MESSAGES_BEFORE_COMPRESS = 20  # После скольки сообщений сжимать
MAX_CONTEXT_TOKENS = 28000        # Лимит контекста (оставляем запас)
TOKENS_PER_CHAR = 0.33            # Примерно 1 токен = 3 символа

# ═══════ ХРАНИЛИЩЕ ═══════
chat_sessions = {}
token_counter = {"total": 0}


def estimate_tokens(text):
    """Примерный подсчёт токенов"""
    if not text:
        return 0
    return int(len(text) * TOKENS_PER_CHAR)


def count_history_tokens(messages):
    """Считаем токены во всей истории"""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
    return total


def md_to_html(text):
    """Конвертация Markdown в HTML"""
    if not text:
        return ""
    extensions = ['fenced_code', 'tables', 'nl2br']
    html = markdown.markdown(text, extensions=extensions)
    return html


def compress_history(session, model_id):
    """
    Сжимает старую историю в краткое содержание.
    Нейронка сама пишет саммари, и мы заменяем старые сообщения на него.
    """
    history = session["messages"]
    
    if len(history) < MAX_MESSAGES_BEFORE_COMPRESS:
        return  # Ещё не пора сжимать
    
    # Берём старые сообщения (первые 2/3 истории)
    split_point = len(history) * 2 // 3
    old_messages = history[:split_point]
    recent_messages = history[split_point:]
    
    # Формируем текст старых сообщений
    old_text = ""
    for msg in old_messages:
        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
        old_text += f"{role}: {msg['content']}\n\n"
    
    try:
        # Просим нейронку сделать краткое содержание
        summary_response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "Сделай краткое содержание диалога. Сохрани ВСЕ важные детали: код, решения, договорённости, имена, числа. Будь максимально информативным. Пиши на русском."
                },
                {
                    "role": "user",
                    "content": f"Вот диалог для сжатия:\n\n{old_text}"
                }
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        
        summary = summary_response.choices[0].message.content
        
        # Сохраняем саммари в сессию
        if "summaries" not in session:
            session["summaries"] = []
        session["summaries"].append(summary)
        
        # Заменяем историю: саммари-сообщение + недавние сообщения
        session["messages"] = recent_messages
        session["compressed"] = True
        
        print(f"✅ История сжата: {len(old_messages)} сообщений → саммари")
        
    except Exception as e:
        print(f"❌ Ошибка сжатия: {e}")
        # Если не удалось сжать — просто обрезаем
        session["messages"] = history[-MAX_MESSAGES_BEFORE_COMPRESS:]


def build_api_messages(session, role_name):
    """
    Собирает сообщения для API с учётом саммари
    """
    system_prompt = ROLES.get(role_name, ROLES["Ассистент"])
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Если есть сжатые саммари — добавляем их
    if session.get("summaries"):
        all_summaries = "\n\n---\n\n".join(session["summaries"])
        messages.append({
            "role": "system",
            "content": f"Краткое содержание предыдущего разговора:\n\n{all_summaries}"
        })
    
    # Добавляем текущие сообщения
    for msg in session["messages"]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"],
        })
    
    # Проверяем что не превышаем лимит
    total_tokens = count_history_tokens(messages)
    
    # Если всё равно слишком много — обрезаем старые сообщения
    while total_tokens > MAX_CONTEXT_TOKENS and len(messages) > 3:
        messages.pop(2)  # Удаляем самое старое (после system)
        total_tokens = count_history_tokens(messages)
    
    return messages


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        "messages": [],
        "model": "Qwen3 Coder",
        "role": "Ассистент",
        "summaries": [],
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
        "context_info": _get_context_info(session_id),
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
            "summaries": [],
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

    model_id = MODELS.get(model_name, MODELS["Qwen3 Coder"])

    # Проверяем — нужно ли сжать историю
    compress_history(session, model_id)

    try:
        # Собираем сообщения для API (с саммари)
        api_messages = build_api_messages(session, role_name)

        response = client.chat.completions.create(
            model=model_id,
            messages=api_messages,
            max_tokens=MAX_TOKENS_RESPONSE,
            temperature=0.7,
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

    # Название чата
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
        "context_info": _get_context_info(session_id),
    })


@app.get("/new", response_class=HTMLResponse)
async def new_chat(request: Request):
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        "messages": [],
        "model": "Qwen3 Coder",
        "role": "Ассистент",
        "summaries": [],
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
        "context_info": _get_context_info(session_id),
    })


@app.get("/continue/{old_session_id}", response_class=HTMLResponse)
async def continue_chat(request: Request, old_session_id: str):
    """
    Создаёт НОВЫЙ чат но с памятью из старого.
    Берёт саммари из старого чата.
    """
    new_session_id = str(uuid.uuid4())
    
    old_session = chat_sessions.get(old_session_id, {})
    old_summaries = old_session.get("summaries", [])
    old_messages = old_session.get("messages", [])
    old_model = old_session.get("model", "Qwen3 Coder")
    old_role = old_session.get("role", "Ассистент")
    old_title = old_session.get("title", "Старый чат")
    
    # Делаем финальное саммари старого чата
    model_id = MODELS.get(old_model, MODELS["Qwen3 Coder"])
    
    final_summary = ""
    if old_messages:
        old_text = ""
        for msg in old_messages:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            old_text += f"{role}: {msg['content']}\n\n"
        
        try:
            summary_response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "Сделай подробное краткое содержание диалога. Сохрани ВСЕ: код, решения, факты, договорённости. Пиши на русском."
                    },
                    {
                        "role": "user",
                        "content": f"Диалог:\n\n{old_text}"
                    }
                ],
                max_tokens=3000,
                temperature=0.3,
            )
            final_summary = summary_response.choices[0].message.content
        except Exception:
            final_summary = old_text[:3000]  # Фолбэк — просто текст
    
    # Собираем все саммари
    all_summaries = old_summaries.copy()
    if final_summary:
        all_summaries.append(final_summary)
    
    # Создаём новую сессию с памятью
    chat_sessions[new_session_id] = {
        "messages": [],
        "model": old_model,
        "role": old_role,
        "summaries": all_summaries,
        "continued_from": old_title,
    }

    return templates.TemplateResponse("index.html", {
        "request": request,
        "messages": [],
        "session_id": new_session_id,
        "models": MODELS,
        "roles": ROLES,
        "selected_model": old_model,
        "selected_role": old_role,
        "token_count": token_counter["total"],
        "chat_list": _get_chat_list(),
        "context_info": _get_context_info(new_session_id),
        "continued_from": old_title,
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
        "context_info": _get_context_info(session_id),
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
            "summaries": [],
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
        "context_info": _get_context_info(session_id),
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

    text = "=== AI Chat Export ===\n"
    text += f"Модель: {session.get('model', '?')}\n"
    text += f"Роль: {session.get('role', '?')}\n"
    text += "=" * 40 + "\n\n"
    
    # Если есть саммари — добавляем
    if session.get("summaries"):
        text += "📝 КОНТЕКСТ ИЗ ПРОШЛЫХ ЧАТОВ:\n"
        for i, s in enumerate(session["summaries"], 1):
            text += f"\n--- Саммари {i} ---\n{s}\n"
        text += "\n" + "=" * 40 + "\n\n"
    
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
        if session.get("messages") or session.get("summaries"):
            title = session.get("title", "Новый чат")
            if session.get("continued_from"):
                title = "🔄 " + title
            chats.append({
                "id": sid,
                "title": title,
                "msg_count": len(session.get("messages", [])),
                "has_memory": bool(session.get("summaries")),
            })
    return chats[-20:]


def _get_context_info(session_id):
    """Информация о контексте для отображения"""
    if session_id not in chat_sessions:
        return {"messages": 0, "tokens": 0, "compressed": False, "percent": 0}
    
    session = chat_sessions[session_id]
    messages = session.get("messages", [])
    tokens = count_history_tokens(messages)
    has_summaries = bool(session.get("summaries"))
    percent = min(100, int(tokens / MAX_CONTEXT_TOKENS * 100))
    
    return {
        "messages": len(messages),
        "tokens": tokens,
        "compressed": has_summaries,
        "percent": percent,
        "summaries_count": len(session.get("summaries", [])),
    }
