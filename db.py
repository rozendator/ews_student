"""
db.py — все операции с Supabase
"""
from __future__ import annotations
from functools import lru_cache
from datetime import date, timedelta

from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY, ANALYSIS_WINDOW_WEEKS


@lru_cache(maxsize=1)
def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Кураторы ─────────────────────────────────────────────────

def fetch_curator_by_chat_id(chat_id: int) -> dict | None:
    res = (
        get_client()
        .table("curators")
        .select("*")
        .eq("telegram_chat_id", chat_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def fetch_all_curators() -> list[dict]:
    res = get_client().table("curators").select("*").execute()
    return res.data or []


def fetch_curator_groups(curator_id: int) -> list[dict]:
    res = (
        get_client()
        .table("curator_groups")
        .select("group_id, groups(group_name)")
        .eq("curator_id", curator_id)
        .execute()
    )
    return res.data or []


# ── Студенты ─────────────────────────────────────────────────

def fetch_students_by_group(group_id: int) -> list[dict]:
    res = (
        get_client()
        .table("students")
        .select("*, groups(group_name)")
        .eq("group_id", group_id)
        .execute()
    )
    return res.data or []


def fetch_all_students() -> list[dict]:
    res = (
        get_client()
        .table("students")
        .select("*, groups(group_name)")
        .execute()
    )
    return res.data or []


def fetch_student(student_id: int) -> dict | None:
    res = (
        get_client()
        .table("students")
        .select("*, groups(group_name)")
        .eq("student_id", student_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


# ── Оценки и посещаемость ────────────────────────────────────

def fetch_grades(student_id: int, weeks: int = ANALYSIS_WINDOW_WEEKS) -> list[dict]:
    cutoff = (date.today() - timedelta(weeks=weeks)).isoformat()
    res = (
        get_client()
        .table("weekly_grades")
        .select("*")
        .eq("student_id", student_id)
        .gte("week_start", cutoff)
        .order("week_start")
        .execute()
    )
    return res.data or []


def fetch_grades_by_week(student_id: int, week_start: str, window: int = ANALYSIS_WINDOW_WEEKS) -> list[dict]:
    """Получить данные за N недель до указанной даты включительно."""
    cutoff = (date.fromisoformat(week_start) - timedelta(weeks=window)).isoformat()
    res = (
        get_client()
        .table("weekly_grades")
        .select("*")
        .eq("student_id", student_id)
        .gte("week_start", cutoff)
        .lte("week_start", week_start)
        .order("week_start")
        .execute()
    )
    return res.data or []


# ── LMS активность ───────────────────────────────────────────

def fetch_lms(student_id: int, weeks: int = ANALYSIS_WINDOW_WEEKS) -> list[dict]:
    cutoff = (date.today() - timedelta(weeks=weeks)).isoformat()
    res = (
        get_client()
        .table("weekly_lms")
        .select("*")
        .eq("student_id", student_id)
        .gte("week_start", cutoff)
        .order("week_start")
        .execute()
    )
    return res.data or []


def fetch_lms_by_week(student_id: int, week_start: str, window: int = ANALYSIS_WINDOW_WEEKS) -> list[dict]:
    cutoff = (date.fromisoformat(week_start) - timedelta(weeks=window)).isoformat()
    res = (
        get_client()
        .table("weekly_lms")
        .select("*")
        .eq("student_id", student_id)
        .gte("week_start", cutoff)
        .lte("week_start", week_start)
        .order("week_start")
        .execute()
    )
    return res.data or []


# ── Платежи ──────────────────────────────────────────────────

def fetch_payments(student_id: int) -> list[dict]:
    res = (
        get_client()
        .table("payments")
        .select("*")
        .eq("student_id", student_id)
        .order("due_date")
        .execute()
    )
    return res.data or []


# ── Отчёты риска ─────────────────────────────────────────────

def save_risk_report(report: dict) -> None:
    get_client().table("risk_reports").upsert(
        report,
        on_conflict="student_id,week_start"
    ).execute()


def fetch_risk_report(student_id: int, week_start: str) -> dict | None:
    res = (
        get_client()
        .table("risk_reports")
        .select("*")
        .eq("student_id", student_id)
        .eq("week_start", week_start)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def fetch_high_risk_students(week_start: str, levels: list[str] | None = None) -> list[dict]:
    if levels is None:
        levels = ["высокий", "средний"]
    res = (
        get_client()
        .table("risk_reports")
        .select("*, students(full_name, group_id, groups(group_name))")
        .eq("week_start", week_start)
        .in_("risk_level", levels)
        .order("risk_score", desc=True)
        .execute()
    )
    return res.data or []


def fetch_risk_reports_by_week(week_start: str) -> list[dict]:
    res = (
        get_client()
        .table("risk_reports")
        .select("*, students(full_name, group_id, specialty, tuition_form, works, accommodation_type, groups(group_name))")
        .eq("week_start", week_start)
        .order("risk_score", desc=True)
        .execute()
    )
    return res.data or []


# ── Alert log ────────────────────────────────────────────────

def save_alert_log(entry: dict) -> None:
    get_client().table("alert_log").insert(entry).execute()


def fetch_last_alert(student_id: int, week_start: str) -> dict | None:
    res = (
        get_client()
        .table("alert_log")
        .select("*")
        .eq("student_id", student_id)
        .eq("week_start", week_start)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None
