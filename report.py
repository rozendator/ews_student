"""
report.py — форматирование сообщений для EWS-бота.

Функции:
  format_weekly_summary        — многочастный отчёт куратора
  format_no_risk_message       — «всё хорошо» сообщение
  format_student_alert         — карточка одного студента (краткая)
  format_factor_detail         — детальный разбор фактора
  format_student_history       — история риска по неделям
  format_analysis_complete_notice — уведомление о завершении анализа
"""
from __future__ import annotations
from datetime import date

MAX_PART = 3800  # лимит одной части сообщения


def _escape(text: str) -> str:
    """Экранирует символы Markdown v1."""
    for ch in ("*", "_", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


def _level_emoji(level: str) -> str:
    return {"высокий": "🔴", "средний": "🟡", "низкий": "🟢"}.get(level, "⚪")


def _split_into_parts(lines: list[str], header: str) -> list[str]:
    """Разбивает длинный список строк на части, не превышающие MAX_PART."""
    parts = []
    current = header + "\n"
    for line in lines:
        if len(current) + len(line) + 1 > MAX_PART:
            parts.append(current)
            current = f"_(продолжение)_\n\n{line}\n"
        else:
            current += line + "\n"
    if current.strip():
        parts.append(current)
    return parts if parts else [header]


# ══════════════════════════════════════════════════════════════
#  Недельный отчёт куратора
# ══════════════════════════════════════════════════════════════

def format_weekly_summary(
    curator_name: str,
    week_start: str,
    pairs: list[tuple[dict, dict]],
) -> list[str]:
    """
    Форматирует полный отчёт куратора.
    Возвращает список частей (если отчёт длинный).
    pairs: [(risk_report, student), ...]
    Студенты отсортированы: сначала высокий риск, потом средний; по score убыванию.
    """
    sorted_pairs = sorted(
        pairs,
        key=lambda x: (
            0 if x[0].get("risk_level") == "высокий" else 1,
            -(x[0].get("risk_score") or 0),
        ),
    )

    high_count   = sum(1 for r, _ in sorted_pairs if r.get("risk_level") == "высокий")
    medium_count = sum(1 for r, _ in sorted_pairs if r.get("risk_level") == "средний")

    header = (
        f"📋 *Еженедельный отчёт EWS*\n"
        f"Куратор: *{_escape(curator_name)}*\n"
        f"Неделя: `{week_start}`\n\n"
        f"🔴 Высокий риск: *{high_count}* | 🟡 Средний риск: *{medium_count}*\n"
        f"{'─' * 28}\n"
    )

    student_lines = []
    for report, student in sorted_pairs:
        lvl   = report.get("risk_level", "—")
        score = report.get("risk_score", 0)
        name  = _escape(student.get("full_name", "?"))
        sid   = student.get("student_id", "?")
        emoji = _level_emoji(lvl)

        grp_name = ""
        if isinstance(student.get("groups"), dict):
            grp_name = student["groups"].get("group_name", "")
        elif student.get("group_name"):
            grp_name = student["group_name"]

        # Профиль
        profile_bits = []
        if student.get("works"):
            profile_bits.append("работает")
        if student.get("accommodation_type") == "съём":
            profile_bits.append("снимает жильё")
        if not student.get("resident"):
            profile_bits.append("иногородний")
        profile_str = f" _{', '.join(profile_bits)}_" if profile_bits else ""

        block = [
            f"{emoji} *{name}*{profile_str}",
            f"Группа: {_escape(grp_name)} | ID: {sid} | Скор: {score}/10",
        ]

        # Топ-факторы
        factor_details = (report.get("factors") or {}).get("factor_details", [])
        critical_fds = sorted(
            [fd for fd in factor_details if fd.get("level") == "critical" and fd.get("score", 0) > 0],
            key=lambda x: -x.get("score", 0),
        )
        if critical_fds:
            block.append("Критические факторы:")
            for fd in critical_fds[:3]:
                block.append(f"  • {fd['name']}: {fd['headline']}")

        # Ранние сигналы
        early_sigs = (report.get("factors") or {}).get("early_signals", [])
        if early_sigs:
            block.append(f"⚡ {early_sigs[0]}")

        # AI-резюме (только рекомендация)
        ai_summary = report.get("ai_summary", "")
        if ai_summary:
            # Извлекаем строку с рекомендацией
            for line in ai_summary.split("\n"):
                if "Рекомендация" in line or line.startswith("🔴") or line.startswith("🟡") or line.startswith("🟢"):
                    if "Рекомендация" in line or any(x in line for x in ["Провести", "Встреча", "Плановый", "Срочный"]):
                        block.append(f"💡 {line.strip()}")
                        break

        block.append(f"└ /student\\_{sid}")
        block.append("")

        student_lines.append("\n".join(block))

    return _split_into_parts(student_lines, header)


# ══════════════════════════════════════════════════════════════
#  «Всё хорошо» сообщение
# ══════════════════════════════════════════════════════════════

def format_no_risk_message(curator_name: str, week_start: str) -> str:
    return (
        f"✅ *Отчёт EWS | {week_start}*\n\n"
        f"Куратор: *{_escape(curator_name)}*\n\n"
        f"Студентов высокого или среднего риска в ваших группах не выявлено.\n\n"
        f"Все показатели в норме. Плановый мониторинг продолжается."
    )


# ══════════════════════════════════════════════════════════════
#  Краткая карточка студента (для алертов)
# ══════════════════════════════════════════════════════════════

def format_student_alert(report: dict, student: dict) -> str:
    lvl   = report.get("risk_level", "—")
    score = report.get("risk_score", 0)
    name  = _escape(student.get("full_name", "?"))
    emoji = _level_emoji(lvl)

    lines = [
        f"{emoji} *{name}*",
        f"Уровень риска: *{lvl.upper()}* ({score}/10)",
    ]

    factor_details = (report.get("factors") or {}).get("factor_details", [])
    for fd in factor_details:
        if fd.get("level") in ("critical", "warning") and fd.get("score", 0) > 0:
            fd_emoji = "🔴" if fd["level"] == "critical" else "🟡"
            lines.append(f"{fd_emoji} {fd['name']}: {fd['headline']}")

    early_sigs = (report.get("factors") or {}).get("early_signals", [])
    if early_sigs:
        lines.append(f"⚡ {early_sigs[0]}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Детальный разбор фактора
# ══════════════════════════════════════════════════════════════

def format_factor_detail(factor_dict: dict, student_name: str, week: str) -> str:
    """Форматирует детальное описание одного фактора."""
    name  = factor_dict.get("name", "Фактор")
    level = factor_dict.get("level", "ok")
    score = factor_dict.get("score", 0)
    head  = factor_dict.get("headline", "—")
    obs   = factor_dict.get("observations", [])
    trend = factor_dict.get("trend", "")

    fd_emoji = {"critical": "🔴", "warning": "🟡", "ok": "🟢"}.get(level, "⚪")

    lines = [
        f"🔍 *{name}*",
        f"Студент: *{_escape(student_name)}* | Неделя: `{week}`",
        "",
        f"{fd_emoji} *{_escape(head)}*",
        f"Вклад в скор: *{score:.1f}* | Тренд: *{trend or 'н/д'}*",
        "",
    ]

    if obs:
        lines.append("*Наблюдения:*")
        for o in obs:
            lines.append(f"• {o}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  История риска студента
# ══════════════════════════════════════════════════════════════

def format_student_history(student: dict, history: list[dict]) -> str:
    """Форматирует историю уровней риска по неделям."""
    name = _escape(student.get("full_name", "?"))
    lines = [f"📜 *История риска: {name}*\n"]

    if not history:
        lines.append("История не найдена.")
        return "\n".join(lines)

    for h in history:
        w     = h.get("week_start", "?")
        lvl   = h.get("risk_level", "—")
        score = h.get("risk_score", 0)
        em    = _level_emoji(lvl)
        lines.append(f"`{w}` {em} *{lvl}* ({score}/10)")

    # Тренд по периоду
    if len(history) >= 3:
        scores = [h.get("risk_score", 0) for h in history]
        delta  = scores[-1] - scores[0]
        if delta > 1.5:
            lines.append(f"\n📈 *Тренд*: рост риска (+{delta:.1f}) — ситуация ухудшается.")
        elif delta < -1.5:
            lines.append(f"\n📉 *Тренд*: снижение риска ({delta:.1f}) — положительная динамика.")
        else:
            lines.append(f"\n→ *Тренд*: стабильный.")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Уведомление о завершении анализа
# ══════════════════════════════════════════════════════════════

def format_analysis_complete_notice(
    week: str,
    n_curators: int,
    n_alerts: int,
    delay_sec: int,
) -> str:
    """Уведомление о завершении анализа и начале рассылки."""
    total_min = round((n_curators * delay_sec) / 60, 1)
    return (
        f"✅ *Анализ за {week} завершён*\n\n"
        f"Кураторов к уведомлению: *{n_curators}*\n"
        f"Алертов по студентам: *{n_alerts}*\n\n"
        f"📤 Рассылка начата. Отчёты доставляются поочерёдно "
        f"(пауза {delay_sec} сек между кураторами).\n"
        f"Ориентировочное время завершения: ~{total_min} мин."
    )