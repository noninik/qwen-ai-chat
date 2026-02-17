# 🤖 AI Chat — Qwen3-Coder

AI-чат на основе модели Qwen3-Coder от Alibaba.

## Запуск локально

​```bash
pip install -r requirements.txt
export HF_TOKEN="hf_ваш_токен"
uvicorn main:app --reload
​```

Открыть: http://localhost:8000

## Деплой на Render

1. Залить на GitHub
2. Подключить к Render
3. Добавить переменную `HF_TOKEN`
