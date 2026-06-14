"""
config.py — централизованная конфигурация EWS
Все пороги и флаги вынесены сюда; analyzer.py и ml_model.py
читают их отсюда — магических чисел в коде нет.
"""
import os

# ── Supabase ──────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://bebneiwvvfoaeigsxkts.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_j0aWSXRoHDa-FR0DFEICtA_EUZJJGOt")

# ── Telegram ──────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8790181800:AAGwaf1JdToNKHZRN7ioeglKtytChzNQha0")

# ── Пороги уровней риска (0–10) ───────────────────────────────
RISK_LOW_MAX    = 2.5   # <= низкий
RISK_MEDIUM_MAX = 5.0   # <= средний; > высокий

# ── Пороги оценок ────────────────────────────────────────────
GRADE_CRITICAL_LOW   = 45    # ниже → critical, score 4.0
GRADE_VERY_LOW       = 55    # ниже → critical, score 3.0
GRADE_LOW            = 65    # ниже → warning,  score 2.0
GRADE_SATISFACTORY   = 75    # ниже → warning,  score 0.8
GRADE_BELOW_STREAK   = 55    # порог для подсчёта «недель подряд ниже»
SUDDEN_DROP_THRESHOLD = 15   # падение за одну неделю → сигнал
GRADE_VOLATILITY_HIGH = 18   # std > этого → «хаотичная активность»
GRADE_VOLATILITY_MED  = 10   # std > этого → «нестабильные оценки»
GRADE_RECENT_DECLINE  = 8    # разрыв recent vs overall → предупреждение
GRADE_STREAK_ALARM    = 3    # кол-во недель подряд ниже GRADE_BELOW_STREAK

# ── Пороги посещаемости (часов пропусков в неделю) ───────────
ABSENCE_CRITICAL_HIGH = 12   # > → critical, score 3.0
ABSENCE_HIGH          = 8    # > → critical, score 2.0
ABSENCE_MEDIUM        = 5    # > → warning,  score 1.2
ABSENCE_LOW           = 2    # > → warning,  score 0.4
ABSENCE_STREAK_ALARM  = 3    # недель подряд с высокими пропусками

# ── Пороги LMS ───────────────────────────────────────────────
LMS_ZERO_LOGINS      = 1     # avg < → critical (полная пассивность)
LMS_CRITICAL_LOW     = 3     # avg < → critical
LMS_WARNING_LOW      = 5     # avg < → warning
LMS_MODERATE         = 8     # avg < → ok (умеренный)
LMS_LATE_RATIO_HIGH  = 0.6   # > → critical (сдаёт с опозданием)
LMS_LATE_RATIO_MED   = 0.4   # > → warning
LMS_LATE_RATIO_LOW   = 0.2   # > → info
LMS_SILENT_EXIT_WEEKS = 3    # нулей подряд → «тихий уход»
LMS_DROP_FACTOR      = 0.4   # recent < prev * DROP_FACTOR → резкое падение

# ── Пороги платежей ──────────────────────────────────────────
PAYMENT_OVERDUE_CRITICAL = 3   # >= → critical
PAYMENT_OVERDUE_SERIOUS  = 2   # >= → critical (меньший уровень)
PAYMENT_OVERDUE_ONE      = 1   # → warning
PAYMENT_LATE_DAYS_ALARM  = 14  # средняя задержка > → доп. штраф

# ── Пороги профиля ───────────────────────────────────────────
PROFILE_ENTRY_SCORE_LOW  = 50   # < → фактор риска
PROFILE_ENTRY_SCORE_HIGH = 90   # > → защитный фактор
PROFILE_CRITICAL_SCORE   = 0.7  # суммарный балл профиля >= → critical

# ── Ранние сигналы ───────────────────────────────────────────
SIGNAL_DUAL_HIT_GRADE    = 60   # оценка < для «двойного удара»
SIGNAL_DUAL_HIT_ABSENCE  = 6    # пропуски > для «двойного удара»
SIGNAL_DUAL_HIT_WEEKS    = 3    # минимум недель «двойного удара»
SIGNAL_SILENT_ABSENCE    = 5    # пропуски > при нулевых LMS → «исчезновение»
SIGNAL_BURNOUT_START     = 70   # стартовая оценка для «срыва»
SIGNAL_BURNOUT_END       = 55   # конечная оценка для «срыва»
SIGNAL_VOLATILITY_ALARM  = 20   # std оценок → «хроническая нестабильность»
SIGNAL_OVERDUE_MIN       = 1    # просрочек + оценка < 60 → риск отчисления
SIGNAL_OVERDUE_GRADE_MAX = 60

# ── ML-режим ─────────────────────────────────────────────────
# True  → ансамбль ML + эвристика (нужна обученная модель в models/)
# False → только детерминированная эвристика из analyzer.py
USE_ML = os.environ.get("EWS_USE_ML", "false").lower() == "true"
ML_WEIGHT     = 0.6   # вес ML-скора в ансамбле (1 - ML_WEIGHT → эвристика)
ML_RISK_THRESHOLD = 4.0  # скор >= → «группа риска» при построении датасета

# ── Количество недель для анализа ────────────────────────────
ANALYSIS_WINDOW_WEEKS = 4

# ── Задержка между кураторами при рассылке (сек) ─────────────
CURATOR_DISPATCH_DELAY = 30

# ── Логирование ──────────────────────────────────────────────
LOG_LEVEL  = os.environ.get("EWS_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FILE   = os.environ.get("EWS_LOG_FILE", "")   # пустая строка → только stdout