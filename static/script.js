// ═══════ Автоскролл ═══════
window.onload = function () {
    scrollToBottom();
    document.getElementById('userInput').focus();
    hljs.highlightAll(); // Подсветка кода

    // Загружаем тему
    const theme = localStorage.getItem('theme') || 'dark';
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        document.querySelector('.theme-btn').textContent = '☀️';
    }
};

function scrollToBottom() {
    const chatBox = document.getElementById('chatBox');
    chatBox.scrollTop = chatBox.scrollHeight;
}

// ═══════ Индикатор загрузки ═══════
function showLoading() {
    const loading = document.getElementById('loading');
    const sendBtn = document.getElementById('sendBtn');
    const input = document.getElementById('userInput');

    loading.style.display = 'block';
    sendBtn.disabled = true;
    sendBtn.style.opacity = '0.5';

    // Добавляем сообщение пользователя в чат сразу
    const chatBox = document.getElementById('chatBox');
    const welcome = chatBox.querySelector('.welcome');
    if (welcome) welcome.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-msg';
    msgDiv.innerHTML = `
        <div class="avatar">👤</div>
        <div class="bubble">${escapeHtml(input.value)}</div>
    `;
    chatBox.appendChild(msgDiv);
    scrollToBottom();
}

// ═══════ Копирование ═══════
function copyMessage(btn) {
    const bubble = btn.closest('.bubble');
    const content = bubble.querySelector('.markdown-content');
    const text = content ? content.innerText : bubble.innerText;

    navigator.clipboard.writeText(text).then(() => {
        const toast = document.getElementById('copyToast');
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2000);
    });
}

// ═══════ Тема ═══════
function toggleTheme() {
    const html = document.documentElement;
    const btn = document.querySelector('.theme-btn');

    if (html.getAttribute('data-theme') === 'light') {
        html.removeAttribute('data-theme');
        btn.textContent = '🌙';
        localStorage.setItem('theme', 'dark');
    } else {
        html.setAttribute('data-theme', 'light');
        btn.textContent = '☀️';
        localStorage.setItem('theme', 'light');
    }
}

// ═══════ Сайдбар ═══════
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    let overlay = document.querySelector('.sidebar-overlay');

    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        overlay.onclick = toggleSidebar;
        document.body.appendChild(overlay);
    }

    sidebar.classList.toggle('open');
    overlay.classList.toggle('show');
}

// ═══════ Подсказки ═══════
function fillQuestion(text) {
    document.getElementById('userInput').value = text;
    document.getElementById('userInput').focus();
}

// ═══════ Утилиты ═══════
function escapeHtml(text) {
    const div = document.createElement('div');
    div.innerText = text;
    return div.innerHTML;
}

// ═══════ Enter для отправки ═══════
document.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        const input = document.getElementById('userInput');
        if (document.activeElement === input && input.value.trim()) {
            document.getElementById('chatForm').submit();
            showLoading();
        }
    }
});
