<div align="center">

# 🎓 EWS — Early Warning System

**Система раннего предупреждения академической неуспеваемости студентов**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=flat-square&logo=supabase&logoColor=white)](https://supabase.com)
[![Telegram Bot](https://img.shields.io/badge/Telegram-Bot_v21-26A5E4?style=flat-square&logo=telegram&logoColor=white)](https://python-telegram-bot.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF4B4B?style=flat-square)](https://shap.readthedocs.io)

*Проактивная аналитическая система на основе ML для выявления студентов группы риска за 4 недели до критической точки*

[Быстрый старт](#-быстрый-старт) · [Архитектура](#-архитектура) · [ML-модели](#-ml-модели) · [Telegram-бот](#-telegram-бот) · [API](#-структура-проекта)

</div>

---

## 📌 О проекте

**EWS (Early Warning System)** — это полноценный прогностический комплекс, который еженедельно анализирует академические показатели студентов и автоматически уведомляет кураторов о тех, кто входит в группу риска. Система переводит управление успеваемостью от реактивного режима («студент уже не сдал») к проактивному («студент начинает скользить вниз»).

### Ключевые возможности

| Возможность | Описание |
|-------------|----------|
| 🔮 **Предсказание риска** | Ансамбль XGBoost + эвристика, скор 0–10, три уровня: 🔴🟡🟢 |
| 🧠 **37 признаков** | Успеваемость, посещаемость, LMS, платежи, профиль, rolling-window, MoM |
| 🔍 **SHAP-объяснения** | Команда `/explain` — куратор видит *почему* студент в риске |
| 📲 **Telegram-бот** | 8 команд, инлайн-кнопки, пагинация, drill-down по факторам |
| ⚡ **Автоматическая рассылка** | Еженедельные отчёты с задержкой и retry при ошибках |
| 🗄️ **Supabase PostgreSQL** | 9 таблиц, JSONB для факторов, row-level security |
| 🆓 **Открытый стек** | Нулевая стоимость лицензий, готов к пилотному внедрению |

### Почему это важно

По данным ЮНЕСКО, от **20 до 40%** студентов не завершают высшее образование. В Казахстане ежегодно выбывает более **63 тысяч студентов** при контингенте 678 тысяч. Традиционные подходы реагируют *после* факта неуспеваемости. EWS выявляет группу риска **за 4 недели** до критической точки, когда вмешательство куратора ещё эффективно.

---

## 🏗 Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                    ИСТОЧНИКИ ДАННЫХ                              │
│  Оценки (CSV/API) │ LMS-активность │ Платежи │ Профиль студента │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ETL
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              ХРАНИЛИЩЕ  —  Supabase PostgreSQL                   │
│  students │ weekly_grades │ weekly_lms │ payments │ risk_reports │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
┌─────────────────────┐      ┌──────────────────────────────┐
│   ЭВРИСТИКА         │      │   ML-ПАЙПЛАЙН                │
│   analyzer.py       │      │   SMOTETomek → Scaler →      │
│   5 факторов риска  │      │   XGBoost / RF / GB / LR     │
│   Работает с нед. 1 │      │   + SHAP TreeExplainer       │
└──────────┬──────────┘      └──────────────┬───────────────┘
           │                                │
           └──────────── АНСАМБЛЬ ──────────┘
                    0.4 × эвристика
                    0.6 × ML (если USE_ML=true)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TELEGRAM-БОТ  (bot.py)                        │
│  /report │ /student │ /top │ /explain │ /status │ /analyze      │
│  /history │ /groups │ инлайн-кнопки │ пагинация │ drill-down    │
└─────────────────────────────────────────────────────────────────┘
```

### Стек технологий

```
Backend      Python 3.11+
БД           Supabase (PostgreSQL 15) + JSONB
ML           scikit-learn · XGBoost · imbalanced-learn
Объяснения   SHAP (TreeExplainer)
Бот          python-telegram-bot v21 (asyncio)
Данные       pandas · numpy
```

---

## 🚀 Быстрый старт

### Требования

- Python 3.11+
- Аккаунт [Supabase](https://supabase.com) (бесплатный тариф достаточен)
- Telegram-бот от [@BotFather](https://t.me/BotFather)

### 1. Клонирование и установка зависимостей

```bash
git clone https://github.com/your-username/ews-early-warning-system.git
cd ews-early-warning-system

pip install -r requirements.txt
```

### 2. Конфигурация

Откройте `config.py` и заполните обязательные параметры:

```python
# config.py

# ── Supabase ──────────────────────────────────────────────────────
SUPABASE_URL = "https://ВАШ-ПРОЕКТ.supabase.co"
SUPABASE_KEY = "ВАШ_SERVICE_ROLE_KEY"   # Settings → API → service_role

# ── Telegram ──────────────────────────────────────────────────────
TELEGRAM_TOKEN = "ВАШ_ТОКЕН_ОТ_BOTFATHER"

# ── ML-режим (по умолчанию выключен, работает только эвристика) ──
USE_ML = False   # True после python ml_model.py train
```

Или через переменные окружения:

```bash
export SUPABASE_URL="https://..."
export SUPABASE_KEY="..."
export TELEGRAM_TOKEN="..."
export EWS_USE_ML="true"          # включить ML-ансамбль
export EWS_LOG_LEVEL="DEBUG"      # уровень логирования
```

### 3. Создание таблиц в Supabase

Откройте **SQL Editor** в Supabase и выполните:

```sql
-- Академические группы
CREATE TABLE groups (
    group_id   SERIAL PRIMARY KEY,
    group_name VARCHAR(50) NOT NULL UNIQUE
);

-- Кураторы
CREATE TABLE curators (
    curator_id        SERIAL PRIMARY KEY,
    full_name         VARCHAR(200) NOT NULL,
    telegram_chat_id  BIGINT UNIQUE,
    department        VARCHAR(100)
);

-- Связь кураторов с группами (M:N)
CREATE TABLE curator_groups (
    curator_id INTEGER NOT NULL REFERENCES curators(curator_id) ON DELETE CASCADE,
    group_id   INTEGER NOT NULL REFERENCES groups(group_id)    ON DELETE CASCADE,
    PRIMARY KEY (curator_id, group_id)
);

-- Студенты
CREATE TABLE students (
    student_id           SERIAL PRIMARY KEY,
    full_name            VARCHAR(200) NOT NULL,
    group_id             INTEGER NOT NULL REFERENCES groups(group_id),
    exam_type            VARCHAR(20),
    entry_score          INTEGER CHECK (entry_score BETWEEN 0 AND 200),
    school_type          VARCHAR(30),
    language             VARCHAR(20),
    resident             BOOLEAN NOT NULL DEFAULT TRUE,
    specialty            VARCHAR(100),
    registration_region  VARCHAR(100),
    lives_in_almaty      BOOLEAN DEFAULT FALSE,
    accommodation_type   VARCHAR(30),
    works                BOOLEAN NOT NULL DEFAULT FALSE,
    tuition_form         VARCHAR(20) NOT NULL DEFAULT 'платное'
);

-- Еженедельные оценки и посещаемость
CREATE TABLE weekly_grades (
    id            SERIAL PRIMARY KEY,
    student_id    INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    week_start    DATE NOT NULL,
    avg_grade     NUMERIC(5,1) CHECK (avg_grade BETWEEN 0 AND 100),
    absence_hours INTEGER CHECK (absence_hours >= 0),
    UNIQUE (student_id, week_start)
);

-- Еженедельная LMS-активность
CREATE TABLE weekly_lms (
    id                   SERIAL PRIMARY KEY,
    student_id           INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    week_start           DATE NOT NULL,
    logins               INTEGER DEFAULT 0,
    submissions_on_time  INTEGER DEFAULT 0,
    submissions_late     INTEGER DEFAULT 0,
    UNIQUE (student_id, week_start)
);

-- Платежи
CREATE TABLE payments (
    id          INTEGER PRIMARY KEY,
    student_id  INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    due_date    DATE NOT NULL,
    paid_date   DATE,
    amount      NUMERIC(12,2) CHECK (amount > 0)
);

-- Отчёты риска (результаты анализа)
CREATE TABLE risk_reports (
    id          SERIAL PRIMARY KEY,
    student_id  INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    week_start  DATE NOT NULL,
    risk_level  VARCHAR(10) NOT NULL CHECK (risk_level IN ('низкий','средний','высокий')),
    risk_score  NUMERIC(4,2) NOT NULL CHECK (risk_score BETWEEN 0 AND 10),
    ai_summary  TEXT,
    factors     JSONB,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (student_id, week_start)
);

-- Журнал отправленных уведомлений
CREATE TABLE alert_log (
    id          SERIAL PRIMARY KEY,
    student_id  INTEGER NOT NULL REFERENCES students(student_id),
    curator_id  INTEGER NOT NULL REFERENCES curators(curator_id),
    week_start  DATE NOT NULL,
    risk_level  VARCHAR(10) NOT NULL,
    sent_at     TIMESTAMP DEFAULT NOW()
);

-- Индексы для ускорения запросов
CREATE INDEX idx_weekly_grades_student_week ON weekly_grades (student_id, week_start DESC);
CREATE INDEX idx_weekly_lms_student_week    ON weekly_lms    (student_id, week_start DESC);
CREATE INDEX idx_risk_reports_week_level    ON risk_reports  (week_start, risk_level);
CREATE INDEX idx_risk_reports_student_week  ON risk_reports  (student_id, week_start DESC);
CREATE INDEX idx_students_group             ON students      (group_id);
CREATE INDEX idx_curators_chat_id           ON curators      (telegram_chat_id)
    WHERE telegram_chat_id IS NOT NULL;
```

### 4. Генерация синтетических данных

```bash
python generate_data.py
```

Создаст папку `data/` с 7 CSV-файлами:
```
data/
├── groups.csv          # 10 групп
├── curators.csv        # 8 кураторов
├── curator_groups.csv  # связи куратор ↔ группа
├── students.csv        # 300 студентов (5 архетипов)
├── weekly_grades.csv   # 4 800 записей оценок (16 нед.)
├── weekly_lms.csv      # 4 800 записей LMS-активности
└── payments.csv        # ~495 платежей
```

### 5. Загрузка данных в Supabase

```bash
python upload_to_supabase.py
```

### 6. Регистрация куратора

1. Запустите бота: `python bot.py`
2. Напишите `/start` в Telegram — бот покажет ваш `chat_id`
3. Обновите запись в Supabase:

```sql
UPDATE curators SET telegram_chat_id = ВАШ_CHAT_ID WHERE curator_id = 1;
```

### 7. Первый анализ

```bash
# Анализ за конкретную неделю (без отправки в Telegram)
python run_analysis.py 2026-09-02 --no-send

# Анализ + рассылка кураторам
python run_analysis.py 2026-09-02

# Тестовый прогон без реальной отправки
python run_analysis.py 2026-09-02 --dry-run

# С кастомной задержкой между кураторами (сек)
python run_analysis.py 2026-09-02 --delay 10
```

### 8. Запуск бота

```bash
python bot.py
```

---

## 🤖 ML-модели

### Обучение

```bash
# Обучить XGBoost (по умолчанию)
python ml_model.py train

# Обучить конкретный алгоритм
python ml_model.py train random_forest
python ml_model.py train gradient_boosting
python ml_model.py train logistic

# Сравнить все 4 алгоритма (для диссертации / отчёта)
python ml_model.py compare

# Предсказание для первых 5 студентов
python ml_model.py predict

# SHAP-объяснение для студента #42
python ml_model.py explain 42
```

### Результаты сравнения алгоритмов

| Алгоритм | Accuracy | Precision | Recall | **F1** | ROC-AUC | CV F1 |
|----------|----------|-----------|--------|--------|---------|-------|
| Logistic Regression | 0.8700 | 0.8889 | 0.8727 | 0.8807 | 0.9216 | 0.868±0.028 |
| **Random Forest** | **0.9000** | **0.9091** | **0.9091** | **0.9091** | 0.9257 | 0.886±0.050 |
| Gradient Boosting | 0.8600 | 0.8727 | 0.8727 | 0.8727 | 0.9147 | 0.877±0.051 |
| XGBoost | 0.8900 | 0.8929 | 0.9091 | 0.9009 | **0.9345** | 0.883±0.049 |

> **Random Forest** — лучший F1. **XGBoost** — наивысший ROC-AUC и нативная поддержка SHAP → используется в ансамбле по умолчанию.

### Признаковое пространство (37 переменных)

<details>
<summary>Развернуть полный список признаков</summary>

| Группа | Признак | Описание |
|--------|---------|----------|
| **Оценки** | `grade_avg` | Средний балл за период |
| | `grade_min` | Минимальный балл |
| | `grade_max` | Максимальный балл |
| | `grade_std` | Стандартное отклонение (нестабильность) |
| | `grade_trend` | Линейный тренд (МНК) |
| | `grade_last2_avg` | Средний балл за последние 2 недели |
| | `weeks_below_55` | Недель с баллом < 55 |
| | `weeks_below_65` | Недель с баллом < 65 |
| **Посещаемость** | `absence_avg` | Среднее пропусков ч/нед |
| | `absence_total` | Суммарно пропущено часов |
| | `absence_max` | Максимум пропусков за неделю |
| | `absence_trend` | Тренд пропусков |
| | `weeks_high_absent` | Недель с пропусками > 8 ч |
| **LMS** | `lms_logins_avg` | Среднее входов в LMS/нед |
| | `lms_logins_min` | Минимум входов |
| | `lms_logins_trend` | Тренд активности |
| | `lms_late_ratio` | Доля заданий, сданных с опозданием |
| | `lms_zero_weeks` | Недель с нулевой активностью |
| | `lms_silent_exit` | «Тихий уход»: 3+ нед. нулевой активности |
| **Платежи** | `payment_overdue_count` | Количество просрочек |
| | `payment_overdue_amount` | Сумма задолженности (тг) |
| | `payment_late_avg_days` | Средняя задержка платежей (дни) |
| | `payment_total_count` | Всего платежей |
| **Профиль** | `works` | Совмещает с работой |
| | `is_nonresident` | Иногородний студент |
| | `rents_housing` | Снимает жильё |
| | `is_paid` | Платная форма обучения |
| | `entry_score` | Вступительный балл ЕНТ/КТА |
| | `social_stress` | Индекс социальной нагрузки (0–3) |
| **Составные** | `double_hit_weeks` | Недель: балл < 60 И пропуски > 6 ч |
| | `academic_distress` | 2×weeks_below_55 + weeks_high_absent + lms_zero_weeks |
| **Rolling 4 нед.** | `grade_roll4_mean` | Скользящий средний балл |
| | `grade_roll4_std` | Скользящая нестабильность |
| | `absence_roll4_mean` | Скользящие пропуски |
| | `lms_roll4_mean` | Скользящая LMS-активность |
| **MoM** | `grade_mom` | Изменение балла нед./нед. |
| | `absence_mom` | Изменение пропусков нед./нед. |
| | `lms_mom` | Изменение LMS-активности нед./нед. |

</details>

### Включение ML-ансамбля

```bash
# После обучения модели:
export EWS_USE_ML=true
python run_analysis.py

# Или в config.py:
USE_ML = True
ML_WEIGHT = 0.6   # вес ML (0.4 - эвристика)
```

---

## 📲 Telegram-бот

### Команды

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие + главное меню с инлайн-кнопками |
| `/report [YYYY-MM-DD]` | Еженедельный отчёт (только средний/высокий риск) |
| `/student <id>` | Карточка студента с разбивкой по факторам |
| `/top [n]` | Топ-10 студентов по скору риска |
| `/history <id>` | История риска студента по неделям |
| `/groups` | Список групп куратора со статистикой |
| `/factor <id> <factor>` | Детальный разбор конкретного фактора |
| `/explain <id>` | SHAP-объяснение (требует `USE_ML=true`) |
| `/status [YYYY-MM-DD]` | Сводная статистика по уровням риска |
| `/analyze` | Запуск анализа текущей недели (только admin) |

### Пример отчёта

```
📋 Еженедельный отчёт EWS
Куратор: Сейтова Айгерим
Неделя: 2026-09-09

🔴 Высокий риск: 3 | 🟡 Средний риск: 5
────────────────────────────────

🔴 Жақсыбеков Берик  _(работает, иногородний)_
Группа: ГР-103 | ID: 42 | Скор: 7.8/10
Критические факторы:
  • Оценки: Средний балл критический — 38.4/100
  • Посещаемость: Критические пропуски — 14.2 ч/нед
⚡ 🔥 «Двойной удар»: 4 нед. одновременно низкие оценки и высокие пропуски
💡 🔴 Срочный контакт. Рассмотреть индивидуальный план обучения.
└ /student_42
```

### SHAP-объяснение (`/explain 42`)

```
🔍 Почему такой уровень риска?

Факторы, повышающие риск:
  🔴 Балл за последние 2 нед.: +1.012  ▓▓▓▓▓▓▓▓▓▓
  🔴 Минимальный балл:         +0.472  ▓▓▓▓▓
  🔴 Средний балл:             +0.378  ▓▓▓▓
  🔴 LMS (4 нед.):             +0.372  ▓▓▓▓
  🔴 Вступительный балл:       +0.278  ▓▓▓

Факторы, снижающие риск:
  🟢 LMS динамика (MoM):       -0.250  ░░░

Базовый уровень модели: 0.0463
```

---

## 📁 Структура проекта

```
ews/
│
├── 📄 config.py              # Конфигурация, ВСЕ пороги и флаги
├── 📄 db.py                  # Все операции с Supabase
├── 📄 analyzer.py            # Детерминированная эвристика (5 факторов)
│   ├── GradeAnalyzer         # Анализ оценок
│   ├── AttendanceAnalyzer    # Анализ посещаемости
│   └── _detect_early_*       # Ранние сигналы тревоги
│
├── 📄 ml_model.py            # ML-компонент
│   ├── ModelRegistry         # DI-контейнер модели (без глобальных переменных)
│   ├── FeatureCache          # Кэш признаков
│   ├── extract_features()    # 37 признаков
│   ├── train_model()         # Обучение + кросс-валидация
│   ├── predict_student()     # Предсказание + кэш
│   ├── explain_student()     # SHAP TreeExplainer
│   ├── compare_models()      # Сравнение 4 алгоритмов
│   └── ShapExplanation       # Waterfall для Telegram
│
├── 📄 bot.py                 # Telegram-бот (asyncio, python-telegram-bot v21)
├── 📄 run_analysis.py        # Пакетный анализ + рассылка кураторам
├── 📄 report.py              # Форматирование сообщений (Markdown v1)
├── 📄 generate_data.py       # Генерация синтетических данных
├── 📄 upload_to_supabase.py  # ETL: CSV → Supabase
├── 📄 requirements.txt       # Зависимости
│
├── 📁 data/                  # CSV-файлы (создаётся generate_data.py)
├── 📁 models/                # Обученные модели (создаётся ml_model.py train)
│   ├── ews_model.joblib      # Сериализованный pipeline
│   └── ews_meta.json         # Метаданные: метрики, список признаков
└── 📁 logs/                  # Логи (если EWS_LOG_FILE задан)
```

---

## ⚙️ Конфигурация

Все параметры системы централизованы в `config.py`. Магических чисел в коде нет.

<details>
<summary>Полный список параметров</summary>

```python
# ── Пороги риска ──────────────────────────────────────────────────
RISK_LOW_MAX    = 2.5   # скор <= низкий
RISK_MEDIUM_MAX = 5.0   # скор <= средний; > высокий

# ── Пороги оценок ────────────────────────────────────────────────
GRADE_CRITICAL_LOW    = 45   # ниже → критический фактор
GRADE_VERY_LOW        = 55
GRADE_LOW             = 65
GRADE_SATISFACTORY    = 75
SUDDEN_DROP_THRESHOLD = 15   # падение за неделю → сигнал

# ── Пороги посещаемости (ч/нед) ─────────────────────────────────
ABSENCE_CRITICAL_HIGH = 12
ABSENCE_HIGH          = 8
ABSENCE_MEDIUM        = 5

# ── Пороги LMS ──────────────────────────────────────────────────
LMS_ZERO_LOGINS      = 1
LMS_SILENT_EXIT_WEEKS = 3   # нед. подряд без входа → «тихий уход»
LMS_LATE_RATIO_HIGH  = 0.6  # доля просрочек → критично

# ── ML ──────────────────────────────────────────────────────────
USE_ML       = False  # True после обучения модели
ML_WEIGHT    = 0.6    # доля ML в ансамбле
ML_RISK_THRESHOLD = 4.0  # граница при формировании меток

# ── Рассылка ────────────────────────────────────────────────────
CURATOR_DISPATCH_DELAY = 30  # сек между кураторами

# ── Логирование ─────────────────────────────────────────────────
LOG_LEVEL = "INFO"    # DEBUG | INFO | WARNING | ERROR
LOG_FILE  = ""        # путь к файлу или "" для stdout
```

</details>

---

## 🧪 Тестирование

```bash
# Unit-тесты
pytest tests/ -v

# Анализ без отправки (dry-run)
python run_analysis.py 2026-09-02 --dry-run

# Только анализ, без рассылки
python run_analysis.py 2026-09-02 --no-send

# Проверка SHAP для конкретного студента
python ml_model.py explain 1
```

### Структура тестов

```
tests/
├── test_analyzer.py       # GradeAnalyzer, AttendanceAnalyzer
├── test_ml_model.py       # extract_features, predict_student
├── test_report.py         # format_weekly_summary
└── conftest.py            # фикстуры: mock-студент, mock-оценки
```

---

## 📊 Модель данных

```
groups (1) ────────────── (N) students
   └── (N) curator_groups (N) ── (1) curators ── (N) alert_log
                                                        │
students ── (N) weekly_grades                           │
         ── (N) weekly_lms                    (FK curator_id)
         ── (N) payments
         ── (N) risk_reports ── factors: JSONB
         ── (N) alert_log ───── (FK student_id)
```

### Формула риска

| Фактор | Макс. вес | Детали |
|--------|-----------|--------|
| Оценки | 4.5 | avg_grade < 45 → 4.0; тренд, волатильность |
| Посещаемость | 3.5 | absence_avg > 12 → 3.0; нарастание |
| LMS-активность | 2.5 | «тихий уход» → +1.0; late_ratio > 0.6 → +1.0 |
| Платежи | 2.0 | 3+ просрочки → 2.0 |
| Профиль | 1.0 | работа + иногородность + аренда |
| **Итого** | **10.0** | min(сумма, 10.0) |

**Уровни риска:** 🟢 Низкий: 0–2.5 · 🟡 Средний: 2.6–5.0 · 🔴 Высокий: 5.1–10.0

---

## 🔐 Безопасность и этика

- **Row-level security** в Supabase: куратор видит только своих студентов
- **Студент не знает** свой статус риска — уведомления только куратору
- Принцип **«помощь, а не наказание»**: система поддерживает, а не стигматизирует
- Соответствует Закону РК **«О персональных данных»** № 94-V от 21.05.2013
- SHAP-объяснения обеспечивают **алгоритмическую прозрачность**

---

## 📦 Зависимости

```
supabase>=2.3.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
xgboost>=2.0.0
imbalanced-learn>=0.12.0
shap>=0.45.0
python-telegram-bot>=21.0
httpx>=0.27.0
pydantic>=2.0.0
joblib>=1.3.0
```

---

## 🗺 Дорожная карта

- [x] Детерминированная эвристика (5 факторов, 37 признаков)
- [x] ML-пайплайн (XGBoost, RF, GB, LR + SMOTETomek)
- [x] SHAP-интерпретация + команда `/explain`
- [x] Telegram-бот с инлайн-кнопками и пагинацией
- [x] Ансамбль ML + эвристика с настраиваемыми весами
- [x] Pydantic-валидация входных данных
- [x] `logging` вместо `print` во всех модулях
- [ ] Unit-тесты (pytest + fixtures)
- [ ] Веб-дашборд (Streamlit / Apache Superset)
- [ ] REST API (FastAPI) для интеграции с ERP-системами вузов
- [ ] Персональный кабинет студента
- [ ] LSTM / Temporal Fusion Transformer для временных рядов
- [ ] Docker Compose для развёртывания одной командой
- [ ] CI/CD (GitHub Actions)

---

## 🤝 Участие в разработке

Вклад в проект приветствуется!

```bash
# Форкните репозиторий, создайте ветку
git checkout -b feature/your-feature-name

# Внесите изменения, убедитесь что тесты проходят
pytest tests/ -v

# Создайте Pull Request с описанием изменений
```

---

## 📄 Лицензия

MIT License — свободное использование, модификация и распространение при сохранении указания авторства.

---

## 📚 Цитирование

Если вы используете EWS в исследовании:

```bibtex
@mastersthesis{akanova2025ews,
  author    = {Аканова, Гульназ},
  title     = {Разработка аналитической системы на основе больших данных
               для раннего выявления студентов группы риска},
  school    = {[Наименование университета]},
  year      = {2025},
  address   = {Алматы, Казахстан},
  note      = {GitHub: https://github.com/your-username/ews-early-warning-system}
}
```

---

<div align="center">

Разработано в рамках диссертационного исследования · Алматы, 2025

**🎓 Аканова Гульназ**

[⭐ Star this repo](https://github.com/your-username/ews-early-warning-system) · [🐛 Report Bug](https://github.com/your-username/ews-early-warning-system/issues) · [💡 Request Feature](https://github.com/your-username/ews-early-warning-system/issues)

</div>
