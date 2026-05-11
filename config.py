"""
config.py — централизованная конфигурация EWS
"""
import os

# ── Supabase ──────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://bebneiwvvfoaeigsxkts.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_j0aWSXRoHDa-FR0DFEICtA_EUZJJGOt")

# ── Telegram ──────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8790181800:AAGwaf1JdToNKHZRN7ioeglKtytChzNQha0")

# ── Ollama (локальный LLM) ────────────────────────────────────
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")  # лёгкая модель

# ── Пороги риска (0–10) ───────────────────────────────────────
RISK_LOW_MAX    = 2.5   # <= низкий
RISK_MEDIUM_MAX = 5.0   # <= средний; > высокий

# ── Количество недель для анализа ────────────────────────────
ANALYSIS_WINDOW_WEEKS = 4
