from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from openai import OpenAI
import os
import uuid
import markdown

app = FastAPI()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)

MODELS = {
    "Qwen3 Coder": "Qwen/Qwen3-Coder-Next:novita",
    "Qwen3 235B": "Qwen/Qwen3-235B-A22B",
    "DeepSeek R1": "deepseek-ai/DeepSeek-R1",
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct",
    "Gemma 3 27B": "google/gemma-3-27b-it",
    "Phi-4": "microsoft/Phi-4",
    "Mistral Small": "mistralai/Mistral-Small-24B-Instruct-2501",
}

ROLES = {
    "Ассистент": "Ты полезный AI-ассистент. Отвечай на русском языке.",
    "Программист": "Ты опытный программист. Пиши чистый код с комментариями. Если код длинный — пиши полностью. Отвечай на русском.",
    "Учитель": "Ты терпеливый учитель. Объясняй просто, с примерами. Отвечай на русском.",
    "Переводчик": "Ты переводчик. Переводи точно. Если язык не указан — русский/английский.",
    "Шутник": "Ты весёлый собеседник. Отвечай с юмором, но по делу. Отвечай на русском.",
    "Писатель": "Ты талантливый писатель. Пиши красиво и грамотно. Отвечай на русском.",
    "Аналитик": "Ты аналитик. Разбирай информацию, делай выводы. Отвечай на русском.",
}

MAX_TOKENS_RESPONSE = 16384
MAX_MESSAGES_BEFORE_COMPRESS = 20
MAX_CONTEXT_TOKENS = 28000

chat_sessions = {}
token_counter = {"total": 0}


def estimate_tokens(text):
    if not text:
        return 0
    return int(len(text) * 0.33)


def count_history_tokens(messages):
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
    return total


def md_to_html(text):
    if not text:
        return ""
    return markdown.markdown(text, extensions=['fenced_code', 'tables', 'nl2br'])


def compress_history(session, model_id):
    history = session["messages"]
    if len(history) < MAX_MESSAGES_BEFORE_COMPRESS:
        return
    split_point = len(history) * 2 // 3
    old_messages = history[:split_point]
    recent_messages = history[split_point:]
    old_text = ""
    for msg in old_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        old_text += f"{role}: {msg['content']}\n\n"
    try:
        summary_response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "Сделай краткое содержание диалога. Сохрани ВСЕ важные детали: код, решения, факты. Пиши на русском."},
                {"role": "user", "content": f"Диалог:\n\n{old_text}"}
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        summary = summary_response.choices[0].message.content
        if "summaries" not in session:
            session["summaries"] = []
        session["summaries"].append(summary)
        session["messages"] = recent_messages
    except:
        session["messages"] = history[-MAX_MESSAGES_BEFORE_COMPRESS:]


def build_api_messages(session, role_name):
    system_prompt = ROLES.get(role_name, ROLES["Ассистент"])
    messages = [{"role": "system", "content": system_prompt}]
    if session.get("summaries"):
        all_summaries = "\n\n---\n\n".join(session["summaries"])
        messages.append({"role": "system", "content": f"Контекст прошлого разговора:\n\n{all_summaries}"})
    for msg in session["messages"]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    total_tokens = count_history_tokens(messages)
    while total_tokens > MAX_CONTEXT_TOKENS and len(messages) > 3:
        messages.pop(2)
        total_tokens = count_history_tokens(messages)
    return messages


def get_chat_list():
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


def get_context_info(session_id):
    if session_id not in chat_sessions:
        return {"messages": 0, "tokens": 0, "compressed": False, "percent": 0, "summaries_count": 0}
    session = chat_sessions[session_id]
    tokens = count_history_tokens(session.get("messages", []))
    percent = min(100, int(tokens / MAX_CONTEXT_TOKENS * 100))
    return {
        "messages": len(session.get("messages", [])),
        "tokens": tokens,
        "compressed": bool(session.get("summaries")),
        "percent": percent,
        "summaries_count": len(session.get("summaries", [])),
    }


def render_page(session_id, messages, selected_model="Qwen3 Coder", selected_role="Ассистент", current_chat_id="", continued_from=""):
    ctx = get_context_info(session_id)
    chat_list = get_chat_list()

    # Генерируем опции моделей
    model_options = ""
    for name in MODELS:
        sel = "selected" if name == selected_model else ""
        model_options += f'<option value="{name}" {sel}>{name}</option>'

    # Генерируем опции ролей
    role_options = ""
    for name in ROLES:
        sel = "selected" if name == selected_role else ""
        role_options += f'<option value="{name}" {sel}>{name}</option>'

    # Генерируем список чатов
    chat_list_html = ""
    if chat_list:
        for chat in chat_list:
            active = "active" if current_chat_id == chat["id"] else ""
            memory = "🧠 " if chat.get("has_memory") else ""
            chat_list_html += f'''
            <div class="chat-item-wrapper">
                <a href="/chat/{chat["id"]}" class="chat-item {active}">
                    <span class="chat-title">{memory}{chat["title"]}</span>
                    <span class="chat-meta">{chat["msg_count"]} 💬</span>
                </a>
                <div class="chat-actions">
                    <a href="/continue/{chat["id"]}" class="continue-btn" title="Продолжить">🔄</a>
                    <a href="/delete/{chat["id"]}" class="delete-btn" title="Удалить" onclick="return confirm('Удалить?')">🗑️</a>
                </div>
            </div>'''
    else:
        chat_list_html = '<p class="no-chats">Пока нет чатов</p>'

    # Генерируем сообщения
    messages_html = ""
    if messages:
        for msg in messages:
            if msg["role"] == "user":
                messages_html += f'''
                <div class="message user-msg">
                    <div class="avatar">👤</div>
                    <div class="bubble">{msg["content"]}</div>
                </div>'''
            elif msg["role"] == "assistant":
                html_content = msg.get("html", msg["content"])
                messages_html += f'''
                <div class="message bot-msg">
                    <div class="avatar">🤖</div>
                    <div class="bubble">
                        <div class="markdown-content">{html_content}</div>
                        <button class="copy-btn" onclick="copyMessage(this)" title="Скопировать">📋</button>
                    </div>
                </div>'''
    else:
        if continued_from:
            messages_html = f'''
            <div class="welcome">
                <h2>🔄 Продолжаем!</h2>
                <p>Я помню наш прошлый разговор. Можешь продолжать!</p>
                <div class="suggestions">
                    <button onclick="fillQuestion('Напомни что мы обсуждали')">🧠 Что мы обсуждали?</button>
                    <button onclick="fillQuestion('Продолжи писать код')">💻 Продолжи код</button>
                </div>
            </div>'''
        else:
            messages_html = '''
            <div class="welcome">
                <h2>👋 Привет!</h2>
                <p>Выбери модель и роль, затем задай вопрос!</p>
                <div class="suggestions">
                    <button onclick="fillQuestion('Напиши полный код TODO-приложения на Python')">💻 TODO-приложение</button>
                    <button onclick="fillQuestion('Объясни что такое API простыми словами')">📖 Что такое API?</button>
                    <button onclick="fillQuestion('Напиши полный HTML/CSS/JS сайт-портфолио')">🌐 Сайт-портфолио</button>
                    <button onclick="fillQuestion('Придумай идею для стартапа')">🚀 Идея стартапа</button>
                    <button onclick="fillQuestion('Напиши игру змейку на JavaScript')">🎮 Игра змейка</button>
                    <button onclick="fillQuestion('Проанализируй плюсы и минусы удалённой работы')">📊 Аналитика</button>
                </div>
            </div>'''

    # Контекст-бар
    context_html = ""
    if ctx:
        warning_html = ""
        if ctx["percent"] > 70:
            warning_html = f'''
            <div class="context-warning">
                ⚠️ Контекст заполняется.
                <a href="/continue/{session_id}">Продолжить в новом чате с памятью →</a>
            </div>'''
        memory_badge = ""
        if ctx["compressed"]:
            memory_badge = f'<span class="memory-badge">🧠 Память ({ctx["summaries_count"]} саммари)</span>'
        continued_html = ""
        if continued_from:
            continued_html = f'<div class="continued-notice">🔄 Продолжение чата "{continued_from}"</div>'
        context_html = f'''
        <div class="context-bar">
            <div class="context-info">
                <span>💬 {ctx["messages"]} сообщений</span>
                <span>📊 ~{ctx["tokens"]} токенов</span>
                {memory_badge}
            </div>
            <div class="context-progress">
                <div class="context-progress-bar" style="width: {ctx["percent"]}%"></div>
            </div>
            {warning_html}
            {continued_html}
        </div>'''

    return f'''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Чат — Qwen3</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
:root {{
    --bg-primary: #0f0c29; --bg-secondary: #302b63; --bg-tertiary: #24243e;
    --bg-card: rgba(255,255,255,0.05); --bg-input: rgba(255,255,255,0.08);
    --border: rgba(255,255,255,0.1); --border-hover: rgba(255,255,255,0.3);
    --text-primary: #fff; --text-secondary: rgba(255,255,255,0.6);
    --text-muted: rgba(255,255,255,0.4); --accent: #667eea; --accent-2: #764ba2;
    --user-bubble: linear-gradient(135deg,#667eea,#764ba2);
    --bot-bubble: rgba(255,255,255,0.1); --sidebar-bg: rgba(15,12,41,0.95);
}}
[data-theme="light"] {{
    --bg-primary: #f0f2f5; --bg-secondary: #e4e6eb; --bg-tertiary: #fff;
    --bg-card: rgba(0,0,0,0.03); --bg-input: rgba(0,0,0,0.05);
    --border: rgba(0,0,0,0.1); --border-hover: rgba(0,0,0,0.3);
    --text-primary: #1a1a2e; --text-secondary: rgba(0,0,0,0.6);
    --text-muted: rgba(0,0,0,0.4); --bot-bubble: rgba(0,0,0,0.05);
    --sidebar-bg: rgba(240,242,245,0.98);
}}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:linear-gradient(135deg,var(--bg-primary),var(--bg-secondary),var(--bg-tertiary)); min-height:100vh; display:flex; color:var(--text-primary); }}
.sidebar {{ width:280px; height:100vh; background:var(--sidebar-bg); backdrop-filter:blur(20px); border-right:1px solid var(--border); display:flex; flex-direction:column; position:fixed; left:-280px; top:0; z-index:100; transition:left .3s; }}
.sidebar.open {{ left:0; }}
.sidebar-header {{ display:flex; justify-content:space-between; align-items:center; padding:20px; border-bottom:1px solid var(--border); }}
.sidebar-close {{ background:none; border:none; color:var(--text-primary); font-size:1.2rem; cursor:pointer; }}
.new-chat-sidebar-btn {{ display:block; margin:15px; padding:12px; background:var(--bg-input); border:1px dashed var(--border); border-radius:10px; color:var(--text-primary); text-decoration:none; text-align:center; transition:all .3s; }}
.new-chat-sidebar-btn:hover {{ background:var(--accent); border-style:solid; }}
.chat-list {{ flex:1; overflow-y:auto; padding:10px; }}
.chat-item-wrapper {{ display:flex; align-items:center; margin-bottom:5px; border-radius:10px; transition:background .2s; }}
.chat-item-wrapper:hover {{ background:var(--bg-input); }}
.chat-item {{ flex:1; display:flex; justify-content:space-between; align-items:center; padding:10px 12px; color:var(--text-primary); text-decoration:none; border-radius:10px; }}
.chat-item.active {{ background:rgba(102,126,234,0.2); border:1px solid rgba(102,126,234,0.3); }}
.chat-title {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.9rem; }}
.chat-meta {{ font-size:.75rem; color:var(--text-muted); white-space:nowrap; }}
.chat-actions {{ display:flex; gap:2px; opacity:0; transition:opacity .2s; padding-right:8px; }}
.chat-item-wrapper:hover .chat-actions {{ opacity:1; }}
.continue-btn,.delete-btn {{ text-decoration:none; font-size:.8rem; padding:4px 6px; border-radius:6px; transition:background .2s; }}
.continue-btn:hover {{ background:rgba(102,126,234,0.2); }}
.delete-btn:hover {{ background:rgba(244,67,54,0.2); }}
.no-chats {{ text-align:center; color:var(--text-muted); padding:20px; font-size:.9rem; }}
.token-counter {{ padding:15px 20px; border-top:1px solid var(--border); font-size:.85rem; color:var(--text-secondary); }}
.container {{ flex:1; max-width:900px; margin:0 auto; height:100vh; display:flex; flex-direction:column; }}
header {{ display:flex; align-items:center; padding:15px 20px; background:var(--bg-card); backdrop-filter:blur(20px); border-bottom:1px solid var(--border); }}
.menu-btn {{ background:none; border:none; color:var(--text-primary); font-size:1.5rem; cursor:pointer; padding:5px 10px; border-radius:8px; transition:background .2s; }}
.menu-btn:hover {{ background:var(--bg-input); }}
.header-center {{ flex:1; text-align:center; }}
header h1 {{ font-size:1.3rem; }}
.subtitle {{ font-size:.8rem; color:var(--text-muted); margin-top:2px; }}
.header-actions {{ display:flex; gap:8px; }}
.header-actions button,.header-actions a {{ background:var(--bg-input); border:1px solid var(--border); border-radius:8px; color:var(--text-primary); padding:6px 10px; cursor:pointer; text-decoration:none; font-size:1rem; transition:all .2s; }}
.header-actions button:hover,.header-actions a:hover {{ background:var(--border-hover); }}
.settings-bar {{ display:flex; gap:15px; padding:12px 20px; background:var(--bg-card); border-bottom:1px solid var(--border); flex-wrap:wrap; }}
.setting {{ display:flex; align-items:center; gap:8px; flex:1; min-width:200px; }}
.setting label {{ font-size:.85rem; color:var(--text-secondary); white-space:nowrap; }}
.setting select {{ flex:1; padding:8px 12px; background:var(--bg-input); border:1px solid var(--border); border-radius:8px; color:var(--text-primary); font-size:.85rem; cursor:pointer; outline:none; }}
.setting select option {{ background:#1a1a2e; color:#fff; }}
.context-bar {{ padding:8px 20px; background:var(--bg-card); border-bottom:1px solid var(--border); }}
.context-info {{ display:flex; gap:15px; font-size:.8rem; color:var(--text-muted); margin-bottom:5px; flex-wrap:wrap; }}
.memory-badge {{ background:rgba(102,126,234,0.2); padding:2px 8px; border-radius:10px; color:var(--accent); font-size:.75rem; }}
.context-progress {{ height:4px; background:var(--bg-input); border-radius:2px; overflow:hidden; }}
.context-progress-bar {{ height:100%; background:linear-gradient(90deg,#4caf50,#ff9800,#f44336); border-radius:2px; transition:width .5s; }}
.context-warning {{ margin-top:6px; font-size:.8rem; color:#ff9800; padding:6px 10px; background:rgba(255,152,0,0.1); border-radius:8px; border:1px solid rgba(255,152,0,0.2); }}
.context-warning a {{ color:var(--accent); text-decoration:none; font-weight:600; }}
.continued-notice {{ margin-top:6px; font-size:.8rem; color:#4caf50; padding:6px 10px; background:rgba(76,175,80,0.1); border-radius:8px; border:1px solid rgba(76,175,80,0.2); }}
.chat-box {{ flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:15px; }}
.welcome {{ text-align:center; margin:auto; padding:20px; }}
.welcome h2 {{ font-size:1.8rem; margin-bottom:10px; }}
.welcome p {{ color:var(--text-secondary); margin-bottom:25px; }}
.suggestions {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.suggestions button {{ padding:14px 18px; background:var(--bg-input); border:1px solid var(--border); border-radius:12px; color:var(--text-primary); font-size:.9rem; cursor:pointer; transition:all .3s; text-align:left; }}
.suggestions button:hover {{ background:rgba(102,126,234,0.15); border-color:var(--accent); transform:translateY(-2px); }}
.message {{ display:flex; gap:12px; align-items:flex-start; animation:fadeIn .3s ease; }}
@keyframes fadeIn {{ from{{opacity:0;transform:translateY(10px)}} to{{opacity:1;transform:translateY(0)}} }}
.avatar {{ font-size:1.3rem; width:38px; height:38px; display:flex; align-items:center; justify-content:center; border-radius:50%; background:var(--bg-input); flex-shrink:0; }}
.bubble {{ padding:14px 18px; border-radius:16px; max-width:80%; line-height:1.6; font-size:.95rem; position:relative; }}
.user-msg {{ flex-direction:row-reverse; }}
.user-msg .bubble {{ background:var(--user-bubble); color:#fff; border-bottom-right-radius:4px; white-space:pre-wrap; word-wrap:break-word; }}
.bot-msg .bubble {{ background:var(--bot-bubble); border:1px solid var(--border); border-bottom-left-radius:4px; }}
.markdown-content h1,.markdown-content h2,.markdown-content h3 {{ margin:10px 0 5px; }}
.markdown-content p {{ margin:5px 0; }}
.markdown-content ul,.markdown-content ol {{ margin:5px 0 5px 20px; }}
.markdown-content code {{ background:rgba(0,0,0,0.3); padding:2px 6px; border-radius:4px; font-family:'Fira Code',Consolas,monospace; font-size:.85em; }}
.markdown-content pre {{ background:rgba(0,0,0,0.4); border-radius:10px; padding:15px; margin:10px 0; overflow-x:auto; }}
.markdown-content pre code {{ background:none; padding:0; }}
.markdown-content table {{ border-collapse:collapse; margin:10px 0; width:100%; }}
.markdown-content th,.markdown-content td {{ border:1px solid var(--border); padding:8px 12px; text-align:left; }}
.markdown-content th {{ background:var(--bg-input); }}
.markdown-content blockquote {{ border-left:3px solid var(--accent); padding-left:15px; margin:10px 0; color:var(--text-secondary); }}
.copy-btn {{ position:absolute; top:8px; right:8px; background:var(--bg-input); border:1px solid var(--border); border-radius:6px; padding:4px 8px; cursor:pointer; font-size:.8rem; opacity:0; transition:opacity .2s; }}
.bubble:hover .copy-btn {{ opacity:1; }}
.copy-toast {{ position:fixed; bottom:100px; left:50%; transform:translateX(-50%) translateY(20px); background:#4caf50; color:#fff; padding:10px 20px; border-radius:10px; font-size:.9rem; opacity:0; transition:all .3s; z-index:999; pointer-events:none; }}
.copy-toast.show {{ opacity:1; transform:translateX(-50%) translateY(0); }}
.input-form {{ display:flex; gap:10px; padding:15px 20px; background:var(--bg-card); border-top:1px solid var(--border); }}
.input-form input {{ flex:1; padding:14px 20px; border-radius:14px; border:1px solid var(--border); background:var(--bg-input); color:var(--text-primary); font-size:1rem; outline:none; transition:border-color .3s; }}
.input-form input::placeholder {{ color:var(--text-muted); }}
.input-form input:focus {{ border-color:var(--accent); }}
.input-form button {{ width:50px; height:50px; border-radius:14px; border:none; background:linear-gradient(135deg,var(--accent),var(--accent-2)); cursor:pointer; display:flex; align-items:center; justify-content:center; transition:transform .2s; }}
.input-form button:hover {{ transform:scale(1.05); }}
.loading {{ padding:15px 20px; background:var(--bg-card); }}
.loading-dots {{ display:flex; align-items:center; gap:10px; color:var(--text-secondary); font-size:.9rem; }}
.dots {{ display:flex; gap:4px; }}
.dot {{ width:8px; height:8px; border-radius:50%; background:var(--accent); animation:bounce 1.4s infinite ease-in-out; }}
.dot:nth-child(2) {{ animation-delay:.2s; }}
.dot:nth-child(3) {{ animation-delay:.4s; }}
@keyframes bounce {{ 0%,80%,100%{{transform:scale(0);opacity:.5}} 40%{{transform:scale(1);opacity:1}} }}
.sidebar-overlay {{ position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); z-index:99; display:none; }}
.sidebar-overlay.show {{ display:block; }}
.chat-box::-webkit-scrollbar,.chat-list::-webkit-scrollbar {{ width:6px; }}
.chat-box::-webkit-scrollbar-thumb,.chat-list::-webkit-scrollbar-thumb {{ background:rgba(255,255,255,0.2); border-radius:3px; }}
@media(max-width:768px) {{ .suggestions{{grid-template-columns:1fr}} .settings-bar{{flex-direction:column;gap:8px}} .setting{{min-width:unset}} .bubble{{max-width:90%}} }}
    </style>
</head>
<body>
    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <h3>💬 Чаты</h3>
            <button class="sidebar-close" onclick="toggleSidebar()">✕</button>
        </div>
        <a href="/new" class="new-chat-sidebar-btn">+ Новый чат</a>
        <div class="chat-list">{chat_list_html}</div>
        <div class="token-counter"><span>📊 Токены: ~{token_counter["total"]}</span></div>
    </div>

    <div class="container">
        <header>
            <button class="menu-btn" onclick="toggleSidebar()">☰</button>
            <div class="header-center">
                <h1>🤖 AI Чат</h1>
                <p class="subtitle">Powered by Qwen3-Coder</p>
            </div>
            <div class="header-actions">
                <button class="theme-btn" onclick="toggleTheme()" title="Тема">🌙</button>
                <a href="/clear/{session_id}" title="Очистить">🗑️</a>
                <a href="/export/{session_id}" title="Скачать">📥</a>
            </div>
        </header>

        <div class="settings-bar">
            <div class="setting">
                <label>🧠 Модель:</label>
                <select form="chatForm" name="model_name">{model_options}</select>
            </div>
            <div class="setting">
                <label>🎭 Роль:</label>
                <select form="chatForm" name="role_name">{role_options}</select>
            </div>
        </div>

        {context_html}

        <div class="chat-box" id="chatBox">{messages_html}</div>

        <form action="/chat" method="post" class="input-form" id="chatForm" onsubmit="showLoading()">
            <input type="hidden" name="session_id" value="{session_id}">
            <input type="text" name="user_message" id="userInput" placeholder="Написать сообщение..." autocomplete="off" required>
            <button type="submit" id="sendBtn">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M2 21L23 12L2 3V10L17 12L2 14V21Z" fill="white"/></svg>
            </button>
        </form>

        <div class="loading" id="loading" style="display:none">
            <div class="loading-dots">
                <span>🤖 Думаю</span>
                <div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
            </div>
        </div>
    </div>

    <div class="copy-toast" id="copyToast">✅ Скопировано!</div>

    <script>
    window.onload=function(){{scrollToBottom();document.getElementById('userInput').focus();hljs.highlightAll();const t=localStorage.getItem('theme')||'dark';if(t==='light'){{document.documentElement.setAttribute('data-theme','light');document.querySelector('.theme-btn').textContent='☀️'}}}};
    function scrollToBottom(){{const c=document.getElementById('chatBox');c.scrollTop=c.scrollHeight}}
    function showLoading(){{document.getElementById('loading').style.display='block';const b=document.getElementById('sendBtn');b.disabled=true;b.style.opacity='0.5';const c=document.getElementById('chatBox');const w=c.querySelector('.welcome');if(w)w.remove();const i=document.getElementById('userInput');const d=document.createElement('div');d.className='message user-msg';d.innerHTML='<div class="avatar">👤</div><div class="bubble">'+escapeHtml(i.value)+'</div>';c.appendChild(d);scrollToBottom()}}
    function copyMessage(btn){{const b=btn.closest('.bubble');const c=b.querySelector('.markdown-content');const t=c?c.innerText:b.innerText;navigator.clipboard.writeText(t).then(()=>{{const toast=document.getElementById('copyToast');toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),2000)}})}}
    function toggleTheme(){{const h=document.documentElement;const b=document.querySelector('.theme-btn');if(h.getAttribute('data-theme')==='light'){{h.removeAttribute('data-theme');b.textContent='🌙';localStorage.setItem('theme','dark')}}else{{h.setAttribute('data-theme','light');b.textContent='☀️';localStorage.setItem('theme','light')}}}}
    function toggleSidebar(){{const s=document.getElementById('sidebar');let o=document.querySelector('.sidebar-overlay');if(!o){{o=document.createElement('div');o.className='sidebar-overlay';o.onclick=toggleSidebar;document.body.appendChild(o)}};s.classList.toggle('open');o.classList.toggle('show')}}
    function fillQuestion(t){{document.getElementById('userInput').value=t;document.getElementById('userInput').focus()}}
    function escapeHtml(t){{const d=document.createElement('div');d.innerText=t;return d.innerHTML}}
    document.addEventListener('keydown',function(e){{if(e.key==='Enter'&&!e.shiftKey){{const i=document.getElementById('userInput');if(document.activeElement===i&&i.value.trim()){{document.getElementById('chatForm').submit();showLoading()}}}}}});
    </script>
</body>
</html>'''


@app.get("/", response_class=HTMLResponse)
async def home():
    sid = str(uuid.uuid4())
    chat_sessions[sid] = {"messages": [], "model": "Qwen3 Coder", "role": "Ассистент", "summaries": []}
    return HTMLResponse(render_page(sid, []))


@app.post("/chat", response_class=HTMLResponse)
async def chat(user_message: str = Form(...), session_id: str = Form(...), model_name: str = Form("Qwen3 Coder"), role_name: str = Form("Ассистент")):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"messages": [], "model": model_name, "role": role_name, "summaries": []}
    session = chat_sessions[session_id]
    session["model"] = model_name
    session["role"] = role_name
    session["messages"].append({"role": "user", "content": user_message, "html": user_message})
    model_id = MODELS.get(model_name, MODELS["Qwen3 Coder"])
    compress_history(session, model_id)
    try:
        api_messages = build_api_messages(session, role_name)
        response = client.chat.completions.create(model=model_id, messages=api_messages, max_tokens=MAX_TOKENS_RESPONSE, temperature=0.7)
        bot_reply = response.choices[0].message.content
        bot_html = md_to_html(bot_reply)
        token_counter["total"] += estimate_tokens(user_message + bot_reply)
    except Exception as e:
        bot_reply = f"Ошибка: {str(e)}"
        bot_html = f"<p style='color:#ff6b6b'>⚠️ {bot_reply}</p>"
    session["messages"].append({"role": "assistant", "content": bot_reply, "html": bot_html})
    if "title" not in session:
        session["title"] = user_message[:30] + ("..." if len(user_message) > 30 else "")
    return HTMLResponse(render_page(session_id, session["messages"], model_name, role_name, session_id))


@app.get("/new", response_class=HTMLResponse)
async def new_chat():
    sid = str(uuid.uuid4())
    chat_sessions[sid] = {"messages": [], "model": "Qwen3 Coder", "role": "Ассистент", "summaries": []}
    return HTMLResponse(render_page(sid, []))


@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def load_chat(session_id: str):
    if session_id not in chat_sessions:
        return await new_chat()
    s = chat_sessions[session_id]
    return HTMLResponse(render_page(session_id, s["messages"], s.get("model", "Qwen3 Coder"), s.get("role", "Ассистент"), session_id))


@app.get("/clear/{session_id}", response_class=HTMLResponse)
async def clear_chat(session_id: str):
    if session_id in chat_sessions:
        m = chat_sessions[session_id].get("model", "Qwen3 Coder")
        r = chat_sessions[session_id].get("role", "Ассистент")
        chat_sessions[session_id] = {"messages": [], "model": m, "role": r, "summaries": []}
    return HTMLResponse(render_page(session_id, [], m, r))


@app.get("/delete/{session_id}", response_class=HTMLResponse)
async def delete_chat(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return await new_chat()


@app.get("/continue/{old_session_id}", response_class=HTMLResponse)
async def continue_chat(old_session_id: str):
    new_sid = str(uuid.uuid4())
    old = chat_sessions.get(old_session_id, {})
    old_model = old.get("model", "Qwen3 Coder")
    old_role = old.get("role", "Ассистент")
    old_title = old.get("title", "Старый чат")
    model_id = MODELS.get(old_model, MODELS["Qwen3 Coder"])
    old_summaries = old.get("summaries", [])
    final_summary = ""
    if old.get("messages"):
        old_text = ""
        for msg in old["messages"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            old_text += f"{role}: {msg['content']}\n\n"
        try:
            sr = client.chat.completions.create(model=model_id, messages=[{"role": "system", "content": "Сделай подробное краткое содержание. Сохрани ВСЕ детали. Пиши на русском."}, {"role": "user", "content": f"Диалог:\n\n{old_text}"}], max_tokens=3000, temperature=0.3)
            final_summary = sr.choices[0].message.content
        except:
            final_summary = old_text[:3000]
    all_summaries = old_summaries.copy()
    if final_summary:
        all_summaries.append(final_summary)
    chat_sessions[new_sid] = {"messages": [], "model": old_model, "role": old_role, "summaries": all_summaries, "continued_from": old_title}
    return HTMLResponse(render_page(new_sid, [], old_model, old_role, continued_from=old_title))


@app.get("/export/{session_id}")
async def export_chat(session_id: str):
    if session_id not in chat_sessions:
        return JSONResponse({"error": "Not found"}, 404)
    s = chat_sessions[session_id]
    text = f"=== AI Chat Export ===\nМодель: {s.get('model')}\nРоль: {s.get('role')}\n{'='*40}\n\n"
    if s.get("summaries"):
        text += "📝 КОНТЕКСТ:\n"
        for i, sm in enumerate(s["summaries"], 1):
            text += f"\n--- Саммари {i} ---\n{sm}\n"
        text += "\n" + "=" * 40 + "\n\n"
    for msg in s["messages"]:
        r = "👤 Вы" if msg["role"] == "user" else "🤖 AI"
        text += f"{r}:\n{msg['content']}\n\n{'─'*40}\n\n"
    return StreamingResponse(iter([text]), media_type="text/plain", headers={"Content-Disposition": f"attachment; filename=chat_{session_id[:8]}.txt"})
