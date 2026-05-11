"""
analyzer.py — Глубокий многофакторный анализ риска студента.

Факторы:
  1. Оценки        — абсолютный уровень, тренд, волатильность
  2. Посещаемость  — пропуски, нарастание, критические недели
  3. LMS           — активность, просрочки, сессионная картина
  4. Платежи       — задолженности, история платёжной дисциплины
  5. Профиль       — социальные факторы риска
  6. Ранние сигналы— аномальные падения, «тихий уход», паттерны
"""
from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, field
from config import RISK_LOW_MAX, RISK_MEDIUM_MAX


# ══════════════════════════════════════════════════════════════
#  Dataclass для подробного отчёта по фактору
# ══════════════════════════════════════════════════════════════

@dataclass
class FactorDetail:
    name: str               # Название фактора
    score: float            # Вклад в суммарный скор (0–10)
    level: str              # "ok" | "warning" | "critical"
    headline: str           # Одна строка: главное наблюдение
    observations: list[str] = field(default_factory=list)  # Подробности
    trend: str = "стабильный"  # Тренд показателя
    value: float | None = None  # Числовое значение для отображения


# ══════════════════════════════════════════════════════════════
#  Утилиты статистики
# ══════════════════════════════════════════════════════════════

def _trend_slope(values: list[float]) -> float:
    """Наклон линейной регрессии (МНК)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def _trend_label(slope: float, scale: float = 1.0) -> str:
    s = slope / scale
    if s < -1.5:   return "резкое снижение"
    elif s < -0.5: return "снижение"
    elif s > 1.5:  return "резкий рост"
    elif s > 0.5:  return "рост"
    return "стабильный"


def _volatility(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _last_n(lst: list, n: int) -> list:
    return lst[-n:] if len(lst) >= n else lst


def _detect_sudden_drop(values: list[float], threshold: float) -> tuple[bool, float]:
    """Есть ли резкое падение между двумя соседними неделями."""
    if len(values) < 2:
        return False, 0.0
    drops = [values[i - 1] - values[i] for i in range(1, len(values))]
    max_drop = max(drops)
    return max_drop >= threshold, max_drop


def _detect_silent_exit(logins: list[int], weeks: int = 3) -> bool:
    """'Тихий уход' — нули в последних N неделях."""
    if len(logins) < weeks:
        return False
    return all(l == 0 for l in logins[-weeks:])


def _consecutive_zeros(values: list[float], threshold: float = 0.5) -> int:
    """Число последовательных недель ниже порога в конце."""
    count = 0
    for v in reversed(values):
        if v <= threshold:
            count += 1
        else:
            break
    return count


# ══════════════════════════════════════════════════════════════
#  Анализ отдельных факторов
# ══════════════════════════════════════════════════════════════

def _analyze_grades(grades: list[dict]) -> FactorDetail:
    if not grades:
        return FactorDetail(
            name="Оценки", score=0.5, level="warning",
            headline="Данные об оценках отсутствуют",
            observations=["Нет записей об успеваемости за анализируемый период."],
        )

    vals = [g.get("avg_grade") or 0.0 for g in grades]
    avg  = sum(vals) / len(vals)
    slope = _trend_slope(vals)
    trend = _trend_label(slope, scale=1.0)
    vol  = _volatility(vals)

    # Абсолютный уровень → базовый скор
    if avg < 45:
        base, level = 4.0, "critical"
        h = f"Средний балл критический — {avg:.1f}/100"
    elif avg < 55:
        base, level = 3.0, "critical"
        h = f"Средний балл очень низкий — {avg:.1f}/100"
    elif avg < 65:
        base, level = 2.0, "warning"
        h = f"Средний балл ниже допустимого — {avg:.1f}/100"
    elif avg < 75:
        base, level = 0.8, "warning"
        h = f"Средний балл удовлетворительный — {avg:.1f}/100"
    else:
        base, level = 0.0, "ok"
        h = f"Средний балл хороший — {avg:.1f}/100"

    obs = []

    # Тренд
    if trend in ("резкое снижение", "снижение") and len(vals) >= 3:
        early = sum(vals[:2]) / 2
        late  = sum(vals[-2:]) / 2
        delta = round(early - late, 1)
        obs.append(f"▼ Тренд: {trend} (−{delta} баллов за период).")
        base += 1.0 if trend == "резкое снижение" else 0.5
    elif trend in ("резкий рост", "рост"):
        obs.append(f"▲ Тренд: {trend} — положительная динамика.")
        base = max(0.0, base - 0.3)

    # Резкое одиночное падение
    dropped, drop_val = _detect_sudden_drop(vals, threshold=15)
    if dropped:
        obs.append(f"⚡ Зафиксировано резкое падение на {drop_val:.1f} баллов за одну неделю.")
        base += 0.5

    # Нестабильность
    if vol > 18:
        obs.append(f"📉 Крайне нестабильные оценки (σ={vol:.1f}) — хаотичная активность.")
        base += 0.5
    elif vol > 10:
        obs.append(f"📉 Нестабильные оценки (σ={vol:.1f}).")

    # Последние 2 недели vs весь период
    if len(vals) >= 4:
        recent = sum(vals[-2:]) / 2
        overall = sum(vals[:-2]) / (len(vals) - 2)
        if overall - recent > 8:
            obs.append(f"⚠️ Последние 2 недели: {recent:.1f} vs средний {overall:.1f} — ухудшение.")
        elif recent - overall > 8:
            obs.append(f"✅ Последние 2 недели: {recent:.1f} — заметное улучшение.")

    # Недель подряд ниже 55
    zeros = _consecutive_zeros(vals, threshold=55)
    if zeros >= 3:
        obs.append(f"🚨 {zeros} недели подряд балл ниже 55 — устойчивая проблема.")
        base = min(base + 0.5, 4.5)

    return FactorDetail(
        name="Оценки", score=round(min(base, 4.5), 2),
        level=level, headline=h, observations=obs,
        trend=trend, value=round(avg, 1),
    )


def _analyze_attendance(grades: list[dict]) -> FactorDetail:
    if not grades:
        return FactorDetail(
            name="Посещаемость", score=0.0, level="ok",
            headline="Нет данных о посещаемости",
        )

    vals = [g.get("absence_hours") or 0.0 for g in grades]
    avg_abs = sum(vals) / len(vals)
    total   = sum(vals)
    slope   = _trend_slope(vals)
    trend   = _trend_label(slope, scale=0.5)

    if avg_abs > 12:
        base, level = 3.0, "critical"
        h = f"Критические пропуски — {avg_abs:.1f} ч/нед (итого {total:.0f} ч)"
    elif avg_abs > 8:
        base, level = 2.0, "critical"
        h = f"Высокие пропуски — {avg_abs:.1f} ч/нед"
    elif avg_abs > 5:
        base, level = 1.2, "warning"
        h = f"Умеренные пропуски — {avg_abs:.1f} ч/нед"
    elif avg_abs > 2:
        base, level = 0.4, "warning"
        h = f"Незначительные пропуски — {avg_abs:.1f} ч/нед"
    else:
        base, level = 0.0, "ok"
        h = f"Посещаемость хорошая — {avg_abs:.1f} ч/нед"

    obs = []

    if trend in ("резкий рост", "рост") and avg_abs > 2:
        obs.append(f"▲ Пропуски нарастают ({trend}) — ситуация ухудшается.")
        base += 0.5

    # Пик пропусков
    max_abs = max(vals)
    max_week_idx = vals.index(max_abs)
    if max_abs > avg_abs * 2 and max_abs > 5:
        week_label = grades[max_week_idx].get("week_start", "?")
        obs.append(f"📌 Пик пропусков: {max_abs:.0f} ч на неделе {week_label}.")

    # Последние 2 недели
    if len(vals) >= 3:
        recent_avg = sum(vals[-2:]) / 2
        if recent_avg > avg_abs * 1.5 and recent_avg > 4:
            obs.append(f"⚠️ Последние 2 недели: {recent_avg:.1f} ч/нед — рост пропусков.")

    # 3+ недели подряд с пропусками > порога
    high_streak = _consecutive_zeros([max(0, 8 - v) for v in vals], threshold=0)
    if high_streak >= 3:
        obs.append(f"🚨 {high_streak} недели подряд с высокими пропусками.")

    return FactorDetail(
        name="Посещаемость", score=round(min(base, 3.5), 2),
        level=level, headline=h, observations=obs,
        trend=trend, value=round(avg_abs, 1),
    )


def _analyze_lms(lms: list[dict]) -> FactorDetail:
    if not lms:
        return FactorDetail(
            name="LMS-активность", score=0.5, level="warning",
            headline="Данные LMS отсутствуют",
            observations=["Нет данных об активности в системе дистанционного обучения."],
        )

    logins_vals = [l.get("logins") or 0 for l in lms]
    avg_logins  = sum(logins_vals) / len(logins_vals)
    slope       = _trend_slope(logins_vals)
    trend       = _trend_label(slope, scale=0.5)

    total_on   = sum(l.get("submissions_on_time") or 0 for l in lms)
    total_late = sum(l.get("submissions_late") or 0 for l in lms)
    total_subs = total_on + total_late
    late_ratio = total_late / total_subs if total_subs > 0 else 0.0

    # Базовый скор по логинам
    if avg_logins < 1:
        base, level = 2.0, "critical"
        h = f"Полная пассивность в LMS — {avg_logins:.1f} входов/нед"
    elif avg_logins < 3:
        base, level = 1.5, "critical"
        h = f"Критически низкая LMS-активность — {avg_logins:.1f} входов/нед"
    elif avg_logins < 5:
        base, level = 0.8, "warning"
        h = f"Низкая LMS-активность — {avg_logins:.1f} входов/нед"
    elif avg_logins < 8:
        base, level = 0.2, "ok"
        h = f"Умеренная LMS-активность — {avg_logins:.1f} входов/нед"
    else:
        base, level = 0.0, "ok"
        h = f"Хорошая LMS-активность — {avg_logins:.1f} входов/нед"

    obs = []

    # Просрочки
    if total_subs > 0:
        if late_ratio > 0.6:
            obs.append(f"🕐 {late_ratio:.0%} заданий сдаётся с опозданием ({total_late}/{total_subs}).")
            base += 1.0
        elif late_ratio > 0.4:
            obs.append(f"🕐 Значительная доля просрочек: {late_ratio:.0%} ({total_late}/{total_subs}).")
            base += 0.5
        elif late_ratio > 0.2:
            obs.append(f"Умеренная доля просрочек: {late_ratio:.0%}.")

    # Тихий уход
    if _detect_silent_exit(logins_vals, weeks=3):
        obs.append("🚫 «Тихий уход»: 3+ недели подряд нулевая активность в LMS.")
        base += 1.0
        level = "critical"

    # Тренд
    if trend in ("резкое снижение", "снижение"):
        obs.append(f"▼ Активность в LMS снижается ({trend}).")
        base += 0.3

    # Нестабильность логинов
    vol = _volatility([float(v) for v in logins_vals])
    if vol > 5 and avg_logins < 6:
        obs.append(f"📊 Нерегулярная активность (σ={vol:.1f}) — заходит редко и хаотично.")

    # Резкое падение активности
    if len(logins_vals) >= 4:
        recent = sum(logins_vals[-2:]) / 2
        prev   = sum(logins_vals[-4:-2]) / 2
        if prev > 4 and recent < prev * 0.4:
            obs.append(f"⚡ Резкое падение активности: {prev:.1f} → {recent:.1f} входов/нед.")
            base += 0.3

    return FactorDetail(
        name="LMS-активность", score=round(min(base, 2.5), 2),
        level=level, headline=h, observations=obs,
        trend=trend, value=round(avg_logins, 1),
    )


def _analyze_payments(payments: list[dict]) -> FactorDetail:
    if not payments:
        return FactorDetail(
            name="Платежи", score=0.0, level="ok",
            headline="Платёжных данных нет (возможно, грантовик)",
        )

    overdue  = [p for p in payments if p.get("paid_date") is None]
    paid     = [p for p in payments if p.get("paid_date") is not None]
    total    = len(payments)

    # Анализируем задержки оплаченных
    late_paid = []
    for p in paid:
        try:
            from datetime import date
            due  = date.fromisoformat(p["due_date"])
            paid_d = date.fromisoformat(p["paid_date"])
            delay = (paid_d - due).days
            if delay > 0:
                late_paid.append(delay)
        except Exception:
            pass

    overdue_count = len(overdue)

    if overdue_count >= 3:
        base, level = 2.0, "critical"
        h = f"Критическая задолженность — {overdue_count}/{total} платежей не оплачено"
    elif overdue_count == 2:
        base, level = 1.5, "critical"
        h = f"Серьёзная задолженность — {overdue_count}/{total} платежей не оплачено"
    elif overdue_count == 1:
        base, level = 0.7, "warning"
        h = f"1 просроченный платёж из {total}"
    else:
        base, level = 0.0, "ok"
        h = f"Платёжная дисциплина хорошая — все {total} платежей оплачены"

    obs = []

    if overdue_count > 0:
        overdue_total = sum(p.get("amount", 0) for p in overdue)
        obs.append(f"💸 Сумма задолженности: {overdue_total:,.0f} тг.")

    if late_paid:
        avg_delay = sum(late_paid) / len(late_paid)
        obs.append(f"⏳ Средняя задержка оплаты (по оплаченным): {avg_delay:.0f} дней.")
        if avg_delay > 14:
            base += 0.3

    if overdue_count > 0 and len(paid) > 0:
        obs.append(f"📋 История: {len(paid)} оплачено, {overdue_count} не оплачено.")

    return FactorDetail(
        name="Платежи", score=round(min(base, 2.0), 2),
        level=level, headline=h, observations=obs,
        value=float(overdue_count),
    )


def _analyze_profile(student: dict) -> FactorDetail:
    risk_factors = []
    protective   = []
    base = 0.0

    if student.get("works"):
        risk_factors.append("совмещает учёбу с работой (+0.3)")
        base += 0.3
    if student.get("accommodation_type") == "съём":
        risk_factors.append("снимает жильё (+0.2)")
        base += 0.2
    if not student.get("resident"):
        risk_factors.append("иногородний студент (+0.2)")
        base += 0.2
    if student.get("tuition_form") == "платное":
        risk_factors.append("платное обучение (+0.1)")
        base += 0.1

    entry_score = student.get("entry_score") or 0
    if entry_score < 50:
        risk_factors.append(f"низкий входной балл {entry_score} (+0.2)")
        base += 0.2
    elif entry_score > 90:
        protective.append(f"высокий входной балл {entry_score}")

    school_type = student.get("school_type", "")
    if "государственная" in school_type:
        protective.append("государственная школа")

    obs = []
    if risk_factors:
        obs.append("⚠️ Факторы уязвимости: " + "; ".join(risk_factors) + ".")
    if protective:
        obs.append("✅ Защитные факторы: " + "; ".join(protective) + ".")
    if not risk_factors and not protective:
        obs.append("Профиль студента без выраженных факторов риска.")

    level = "critical" if base >= 0.7 else ("warning" if base >= 0.3 else "ok")
    h = f"{len(risk_factors)} факторов уязвимости" if risk_factors else "Профиль без факторов риска"

    return FactorDetail(
        name="Профиль студента", score=round(min(base, 1.0), 2),
        level=level, headline=h, observations=obs, value=round(base, 2),
    )


# ══════════════════════════════════════════════════════════════
#  Ранние сигналы тревоги (cross-factor patterns)
# ══════════════════════════════════════════════════════════════

def _detect_early_warning_signals(
    grades: list[dict],
    lms: list[dict],
    payments: list[dict],
    student: dict,
) -> list[str]:
    """Обнаруживает паттерны риска, невидимые при изолированном анализе."""
    signals = []

    grade_vals  = [g.get("avg_grade") or 0 for g in grades]
    login_vals  = [l.get("logins") or 0 for l in lms]
    absence_vals = [g.get("absence_hours") or 0 for g in grades]

    # Паттерн "двойной удар": плохие оценки + высокие пропуски одновременно
    if grade_vals and absence_vals and len(grade_vals) == len(absence_vals):
        dual_hit_weeks = sum(
            1 for g, a in zip(grade_vals, absence_vals)
            if g < 60 and a > 6
        )
        if dual_hit_weeks >= 3:
            signals.append(
                f"🔥 «Двойной удар»: {dual_hit_weeks} нед. одновременно низкие оценки (<60) И высокие пропуски (>6ч)."
            )

    # Паттерн "исчезновение": LMS-нули + пропуски растут
    if login_vals and absence_vals:
        recent_logins  = sum(login_vals[-3:]) if len(login_vals) >= 3 else sum(login_vals)
        recent_absence = sum(absence_vals[-3:]) / min(3, len(absence_vals))
        if recent_logins == 0 and recent_absence > 5:
            signals.append("🚨 «Исчезновение»: нулевая LMS-активность + растущие пропуски — студент фактически отключился.")

    # Паттерн "успешный в начале": раннее падение после хорошего старта
    if len(grade_vals) >= 6:
        start = sum(grade_vals[:3]) / 3
        end   = sum(grade_vals[-3:]) / 3
        if start >= 70 and end < 55:
            signals.append(
                f"📉 «Срыв»: начал успешно ({start:.0f}), резкое падение к концу ({end:.0f}) — возможен внешний стрессор."
            )

    # Паттерн "хроническая нестабильность"
    if grade_vals and _volatility(grade_vals) > 20:
        signals.append("🎢 «Хроническая нестабильность»: экстремальные колебания оценок — нерегулярная подготовка или ситуативные проблемы.")

    # Платёж + плохая успеваемость = высокий риск отчисления
    overdue = [p for p in payments if p.get("paid_date") is None]
    avg_grade = sum(grade_vals) / len(grade_vals) if grade_vals else 100
    if len(overdue) >= 1 and avg_grade < 60:
        signals.append("⚠️ Совпадение: долг по оплате + низкая успеваемость = высокий риск отчисления.")

    # Работает + снимает + нет резидента = тройная нагрузка
    stress_count = sum([
        bool(student.get("works")),
        student.get("accommodation_type") == "съём",
        not student.get("resident", True),
    ])
    if stress_count == 3:
        signals.append("🏋️ «Тройная нагрузка»: работает + снимает жильё + иногородний — высокий социальный стресс.")

    return signals


# ══════════════════════════════════════════════════════════════
#  Числовой скоринг (0–10)
# ══════════════════════════════════════════════════════════════

def compute_risk_score(
    grades: list[dict],
    lms: list[dict],
    payments: list[dict],
    student: dict,
) -> tuple[float, dict, list[FactorDetail]]:
    """
    Возвращает (total_score, factors_dict, factor_details).
    factors_dict совместим со старым форматом.
    """
    fd_grades     = _analyze_grades(grades)
    fd_attendance = _analyze_attendance(grades)
    fd_lms        = _analyze_lms(lms)
    fd_payments   = _analyze_payments(payments)
    fd_profile    = _analyze_profile(student)

    details = [fd_grades, fd_attendance, fd_lms, fd_payments, fd_profile]
    factors = {
        "grades":     fd_grades.score,
        "attendance": fd_attendance.score,
        "lms":        fd_lms.score,
        "payments":   fd_payments.score,
        "profile":    fd_profile.score,
    }
    total = round(min(sum(factors.values()), 10.0), 2)
    return total, factors, details


def score_to_level(score: float) -> str:
    if score <= RISK_LOW_MAX:
        return "низкий"
    elif score <= RISK_MEDIUM_MAX:
        return "средний"
    return "высокий"


# ══════════════════════════════════════════════════════════════
#  AI-движок резюме (локальный, детализированный)
# ══════════════════════════════════════════════════════════════

class LocalAI:
    _INTROS = {
        "высокий": [
            "Ситуация требует немедленного вмешательства куратора.",
            "Выявлена критическая совокупность негативных факторов.",
            "Студент находится в зоне критического риска — требуется личный контакт.",
        ],
        "средний": [
            "Выявлен ряд факторов, требующих внимания куратора.",
            "Наблюдаются признаки академической нестабильности.",
            "Ситуация вызывает умеренное беспокойство — необходим мониторинг.",
        ],
        "низкий": [
            "Академическая ситуация в целом удовлетворительная.",
            "Серьёзных отклонений не зафиксировано.",
            "Студент демонстрирует стабильные показатели.",
        ],
    }

    _RECS = {
        "высокий": [
            "🔴 Провести личную встречу в течение 1–2 дней. Выяснить причины и составить план исправления. При необходимости подключить психолога.",
            "🔴 Срочный контакт. Рассмотреть индивидуальный план обучения или академический отпуск.",
            "🔴 Экстренное собеседование. Информировать заведующего кафедрой и деканат.",
        ],
        "средний": [
            "🟡 Профилактическая беседа в течение недели. Усилить мониторинг в ближайшие 2 недели.",
            "🟡 Встреча для выяснения внешних факторов. При ухудшении — перевести в высокий риск.",
            "🟡 Обратить внимание на динамику. Предложить поддержку или тьюторство.",
        ],
        "низкий": [
            "🟢 Плановый мониторинг, особых мер не требуется.",
            "🟢 Продолжайте стандартное наблюдение.",
            "🟢 Ситуация под контролем.",
        ],
    }

    def _pick(self, lst: list[str], seed: int) -> str:
        return lst[seed % len(lst)]

    def generate(
        self,
        student: dict,
        grades: list[dict],
        lms: list[dict],
        payments: list[dict],
        risk_score: float,
        risk_level: str,
        factor_details: list[FactorDetail],
        early_signals: list[str],
    ) -> str:
        full_name = student.get("full_name", "Студент")
        seed = abs(hash(full_name)) % 1000

        level_emoji = {"низкий": "🟢", "средний": "🟡", "высокий": "🔴"}.get(risk_level, "⚪")
        parts = [
            f"{level_emoji} {full_name}: риск {risk_level.upper()}, скор {risk_score}/10.",
            self._pick(self._INTROS[risk_level], seed),
        ]

        # Топ-факторы (критические и warning)
        critical = [d for d in factor_details if d.level == "critical" and d.score > 0]
        warnings = [d for d in factor_details if d.level == "warning" and d.score > 0]

        for fd in sorted(critical, key=lambda x: -x.score)[:2]:
            parts.append(f"• {fd.name}: {fd.headline}.")
            if fd.observations:
                parts.append(f"  {fd.observations[0]}")

        for fd in sorted(warnings, key=lambda x: -x.score)[:1]:
            parts.append(f"• {fd.name}: {fd.headline}.")

        # Ранние сигналы
        if early_signals:
            parts.append(f"Особые сигналы: {early_signals[0]}")

        # Рекомендация
        parts.append(self._pick(self._RECS[risk_level], seed + 1))

        return "\n".join(parts)


_ai = LocalAI()


def generate_ai_summary(
    student: dict,
    grades: list[dict],
    lms: list[dict],
    payments: list[dict],
    risk_score: float,
    risk_level: str,
    factors: dict,
    factor_details: list[FactorDetail] | None = None,
    early_signals: list[str] | None = None,
) -> str:
    return _ai.generate(
        student, grades, lms, payments,
        risk_score, risk_level,
        factor_details or [],
        early_signals or [],
    )


# ══════════════════════════════════════════════════════════════
#  Полный анализ одного студента
# ══════════════════════════════════════════════════════════════

def analyze_student(
    student: dict,
    grades: list[dict],
    lms: list[dict],
    payments: list[dict],
    week_start: str,
) -> dict:
    risk_score, factors, factor_details = compute_risk_score(grades, lms, payments, student)
    risk_level  = score_to_level(risk_score)
    early_sigs  = _detect_early_warning_signals(grades, lms, payments, student)
    ai_summary  = generate_ai_summary(
        student, grades, lms, payments,
        risk_score, risk_level, factors,
        factor_details, early_sigs,
    )

    # Собираем расширенный словарь факторов для хранения
    factors_extended = {
        **factors,
        "factor_details": [
            {
                "name":         fd.name,
                "score":        fd.score,
                "level":        fd.level,
                "headline":     fd.headline,
                "observations": fd.observations,
                "trend":        fd.trend,
                "value":        fd.value,
            }
            for fd in factor_details
        ],
        "early_signals": early_sigs,
    }

    return {
        "student_id":   student["student_id"],
        "week_start":   week_start,
        "risk_level":   risk_level,
        "risk_score":   risk_score,
        "ai_summary":   ai_summary,
        "factors":      factors_extended,
    }