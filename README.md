# EWS — Early Warning System

Система раннего предупреждения академической неуспеваемости студентов.

## Стек
- **База данных**: Supabase (PostgreSQL)
- **AI-анализ**: Ollama (локальный LLM, llama3.2:3b) + детерминированный fallback
- **Бот**: python-telegram-bot v21

---

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Конфигурация
Откройте `config.py` и укажите:
```python
TELEGRAM_TOKEN = "ВАШ_ТОКЕН_ОТ_BOTFATHER"
SUPABASE_URL   = "https://ваш-проект.supabase.co"
SUPABASE_KEY   = "ваш_service_role_key"
```

### 3. Создание таблиц в Supabase
Выполните SQL из файла `schema.sql` в редакторе Supabase.

### 4. Генерация синтетических данных
```bash
python generate_data.py
```
Создаст папку `data/` с 7 CSV-файлами.

### 5. Загрузка данных в Supabase
```bash
python upload_to_supabase.py
```

### 6. Регистрация своего chat_id как куратора
1. Запустите бота: `python bot.py`
2. Напишите `/start` — бот покажет ваш chat_id
3. Выполните в Supabase SQL:
```sql
UPDATE curators SET telegram_chat_id = ВАШ_CHAT_ID WHERE curator_id = 1;
```

### 7. Запуск анализа
```bash
# Вручную (для тестирования)
python run_analysis.py 2024-09-02

# Или через бота командой /analyze
```

### 8. Запуск бота
```bash
python bot.py
```

---

## Команды бота

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие + главное меню с кнопками |
| `/report` | Недельный отчёт куратора (только риск средний/высокий) |
| `/report 2024-09-02` | Отчёт за конкретную неделю |
| `/student 42` | Карточка студента #42 |
| `/top` | Топ-10 студентов высокого риска |
| `/status` | Сводная статистика анализа |
| `/analyze` | Запустить анализ текущей недели (admin) |

Все команды также доступны через **инлайн-кнопки** после `/start`.

---

## Локальный AI (Ollama)

```bash
# Установка Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Загрузка модели (лёгкая, ~2GB)
ollama pull llama3.2:3b

# Запуск сервера
ollama serve
```

Если Ollama недоступна — система автоматически генерирует детерминированное резюме на основе данных (fallback). Бот работает в любом случае.

---

## Архитектура

```
generate_data.py     → data/*.csv
upload_to_supabase.py → Supabase DB
run_analysis.py      → analyzer.py → risk_reports + alert_log
bot.py               → Telegram команды + кнопки
```

### Формула риска (0–10)
| Фактор | Макс. вес |
|--------|-----------|
| Оценки (avg_grade) | 4.5 |
| Посещаемость (absence_hours) | 3.0 |
| LMS активность | 2.5 |
| Платежи (просрочки) | 1.5 |
| Профиль (работа, жильё) | 0.7 |

**Уровни риска:**
- 🟢 Низкий: 0–2.5
- 🟡 Средний: 2.6–5.0  
- 🔴 Высокий: 5.1–10
