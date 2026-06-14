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

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

import config
from config import (
    RISK_LOW_MAX, RISK_MEDIUM_MAX,
    # Оценки
    GRADE_CRITICAL_LOW, GRADE_VERY_LOW, GRADE_LOW, GRADE_SATISFACTORY,
    GRADE_BELOW_STREAK, SUDDEN_DROP_THRESHOLD,
    GRADE_VOLATILITY_HIGH, GRADE_VOLATILITY_MED, GRADE_RECENT_DECLINE,
    GRADE_STREAK_ALARM,
    # Посещаемость
    ABSENCE_CRITICAL_HIGH, ABSENCE_HIGH, ABSENCE_MEDIUM, ABSENCE_LOW,
    ABSENCE_STREAK_ALARM,
    # LMS
    LMS_ZERO_LOGINS, LMS_CRITICAL_LOW, LMS_WARNING_LOW, LMS_MODERATE,
    LMS_LATE_RATIO_HIGH, LMS_LATE_RATIO_MED, LMS_LATE_RATIO_LOW,
    LMS_SILENT_EXIT_WEEKS, LMS_DROP_FACTOR,
    # Платежи
    PAYMENT_OVERDUE_CRITICAL, PAYMENT_OVERDUE_SERIOUS,
    PAYMENT_OVERDUE_ONE, PAYMENT_LATE_DAYS_ALARM,
    # Профиль
    PROFILE_ENTRY_SCORE_LOW, PROFILE_ENTRY_SCORE_HIGH, PROFILE_CRITICAL_SCORE,
    # Сигналы
    SIGNAL_DUAL_HIT_GRADE, SIGNAL_DUAL_HIT_ABSENCE, SIGNAL_DUAL_HIT_WEEKS,
    SIGNAL_SILENT_ABSENCE, SIGNAL_BURNOUT_START, SIGNAL_BURNOUT_END,
    SIGNAL_VOLATILITY_ALARM, SIGNAL_OVERDUE_MIN, SIGNAL_OVERDUE_GRADE_MAX,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Pydantic-схемы для валидации входных данных
# ══════════════════════════════════════════════════════════════

class GradeRecord(BaseModel):
    student_id: int
    week_start: str
    avg_grade:  Optional[float] = Field(None, ge=0, le=100)
    absence_hours: Optional[float] = Field(None, ge=0, le=40)

    @field_validator("week_start")
    @classmethod
    def valid_date(cls, v: str) -> str:
        date.fromisoformat(v)   # бросает ValueError если формат неверный
        return v


class LMSRecord(BaseModel):
    student_id: int
    week_start: str
    logins: Optional[int] = Field(None, ge=0)
    submissions_on_time: Optional[int] = Field(None, ge=0)
    submissions_late:    Optional[int] = Field(None, ge=0)


class PaymentRecord(BaseModel):
    id: int
    student_id: int
    due_date:   str
    paid_date:  Optional[str] = None
    amount:     Optional[float] = Field(None, ge=0)


class StudentInput(BaseModel):
    student_id:         int
    full_name:          str = ""
    works:              bool = False
    resident:           bool = True
    accommodation_type: str  = ""
    tuition_form:       str  = ""
    entry_score:        Optional[float] = Field(None, ge=0, le=200)
    school_type:        str  = ""
    group_id:           Optional[int] = None
    # Доп. поля — пропускаем неизвестные через model_config
    model_config = {"extra": "allow"}


class FactorDetailModel(BaseModel):
    """Валидированный результат анализа одного фактора."""
    name:         str
    score:        float = Field(ge=0.0, le=10.0)
    level:        str   # "ok" | "warning" | "critical"
    headline:     str
    observations: list[str] = []
    trend:        str  = "стабильный"
    value:        Optional[float] = None

    @field_validator("level")
    @classmethod
    def valid_level(cls, v: str) -> str:
        if v not in ("ok", "warning", "critical"):
            raise ValueError(f"Неверный уровень: {v}")
        return v


# ══════════════════════════════════════════════════════════════
#  Dataclass для удобства внутри модуля (совместим с v1)
# ══════════════════════════════════════════════════════════════

@dataclass
class FactorDetail:
    name: str
    score: float
    level: str
    headline: str
    observations: list[str] = field(default_factory=list)
    trend: str = "стабильный"
    value: float | None = None

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "score":        self.score,
            "level":        self.level,
            "headline":     self.headline,
            "observations": self.observations,
            "trend":        self.trend,
            "value":        self.value,
        }

    def validated(self) -> "FactorDetail":
        """Прогоняет через Pydantic; бросает ValidationError при нарушении контракта."""
        FactorDetailModel(**self.to_dict())
        return self


# ══════════════════════════════════════════════════════════════
#  Утилиты статистики
# ══════════════════════════════════════════════════════════════

def _trend_slope(values: list[float]) -> float:
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
    if len(values) < 2:
        return False, 0.0
    drops = [values[i - 1] - values[i] for i in range(1, len(values))]
    max_drop = max(drops)
    return max_drop >= threshold, max_drop


def _detect_silent_exit(logins: list[int], weeks: int = LMS_SILENT_EXIT_WEEKS) -> bool:
    if len(logins) < weeks:
        return False
    return all(lo == 0 for lo in logins[-weeks:])


def _consecutive_zeros(values: list[float], threshold: float = 0.5) -> int:
    count = 0
    for v in reversed(values):
        if v <= threshold:
            count += 1
        else:
            break
    return count


# ══════════════════════════════════════════════════════════════
#  GradeAnalyzer — анализ оценок (выделен в отдельный класс)
# ══════════════════════════════════════════════════════════════

class GradeAnalyzer:
    """
    Инкапсулирует логику оценки успеваемости.
    Принимает list[GradeRecord] (уже валидированные), возвращает FactorDetail.
    """

    def analyze(self, grades: list[dict]) -> FactorDetail:
        if not grades:
            return FactorDetail(
                name="Оценки", score=0.5, level="warning",
                headline="Данные об оценках отсутствуют",
                observations=["Нет записей об успеваемости за анализируемый период."],
            )

        vals  = [g.get("avg_grade") or 0.0 for g in grades]
        avg   = sum(vals) / len(vals)
        slope = _trend_slope(vals)
        trend = _trend_label(slope, scale=1.0)
        vol   = _volatility(vals)

        base, level, h = self._base_score(avg)
        obs: list[str] = []

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
        dropped, drop_val = _detect_sudden_drop(vals, SUDDEN_DROP_THRESHOLD)
        if dropped:
            obs.append(f"⚡ Резкое падение на {drop_val:.1f} баллов за одну неделю.")
            base += 0.5

        # Нестабильность
        if vol > GRADE_VOLATILITY_HIGH:
            obs.append(f"📉 Крайне нестабильные оценки (σ={vol:.1f}) — хаотичная активность.")
            base += 0.5
        elif vol > GRADE_VOLATILITY_MED:
            obs.append(f"📉 Нестабильные оценки (σ={vol:.1f}).")

        # Последние 2 недели vs весь период
        if len(vals) >= 4:
            recent  = sum(vals[-2:]) / 2
            overall = sum(vals[:-2]) / (len(vals) - 2)
            if overall - recent > GRADE_RECENT_DECLINE:
                obs.append(f"⚠️ Последние 2 недели: {recent:.1f} vs средний {overall:.1f} — ухудшение.")
            elif recent - overall > GRADE_RECENT_DECLINE:
                obs.append(f"✅ Последние 2 недели: {recent:.1f} — заметное улучшение.")

        # Недель подряд ниже порога
        zeros = _consecutive_zeros(vals, threshold=GRADE_BELOW_STREAK)
        if zeros >= GRADE_STREAK_ALARM:
            obs.append(f"🚨 {zeros} недели подряд балл ниже {GRADE_BELOW_STREAK} — устойчивая проблема.")
            base = min(base + 0.5, 4.5)

        logger.debug("GradeAnalyzer: avg=%.1f score=%.2f level=%s", avg, base, level)
        return FactorDetail(
            name="Оценки", score=round(min(base, 4.5), 2),
            level=level, headline=h, observations=obs,
            trend=trend, value=round(avg, 1),
        ).validated()

    @staticmethod
    def _base_score(avg: float) -> tuple[float, str, str]:
        if avg < GRADE_CRITICAL_LOW:
            return 4.0, "critical", f"Средний балл критический — {avg:.1f}/100"
        elif avg < GRADE_VERY_LOW:
            return 3.0, "critical", f"Средний балл очень низкий — {avg:.1f}/100"
        elif avg < GRADE_LOW:
            return 2.0, "warning",  f"Средний балл ниже допустимого — {avg:.1f}/100"
        elif avg < GRADE_SATISFACTORY:
            return 0.8, "warning",  f"Средний балл удовлетворительный — {avg:.1f}/100"
        return 0.0, "ok", f"Средний балл хороший — {avg:.1f}/100"


# ══════════════════════════════════════════════════════════════
#  AttendanceAnalyzer — анализ посещаемости
# ══════════════════════════════════════════════════════════════

class AttendanceAnalyzer:
    """
    Анализирует пропуски (absence_hours) из тех же записей weekly_grades.
    Выделен отдельно от GradeAnalyzer, чтобы каждый класс имел одну ответственность.
    """

    def analyze(self, grades: list[dict]) -> FactorDetail:
        if not grades:
            return FactorDetail(
                name="Посещаемость", score=0.0, level="ok",
                headline="Нет данных о посещаемости",
            )

        vals    = [g.get("absence_hours") or 0.0 for g in grades]
        avg_abs = sum(vals) / len(vals)
        total   = sum(vals)
        slope   = _trend_slope(vals)
        trend   = _trend_label(slope, scale=0.5)

        base, level, h = self._base_score(avg_abs, total)
        obs: list[str] = []

        if trend in ("резкий рост", "рост") and avg_abs > ABSENCE_LOW:
            obs.append(f"▲ Пропуски нарастают ({trend}) — ситуация ухудшается.")
            base += 0.5

        max_abs = max(vals)
        max_week_idx = vals.index(max_abs)
        if max_abs > avg_abs * 2 and max_abs > ABSENCE_MEDIUM:
            week_label = grades[max_week_idx].get("week_start", "?")
            obs.append(f"📌 Пик пропусков: {max_abs:.0f} ч на неделе {week_label}.")

        if len(vals) >= 3:
            recent_avg = sum(vals[-2:]) / 2
            if recent_avg > avg_abs * 1.5 and recent_avg > ABSENCE_MEDIUM - 1:
                obs.append(f"⚠️ Последние 2 недели: {recent_avg:.1f} ч/нед — рост пропусков.")

        # 3+ недель подряд с высокими пропусками
        high_streak = _consecutive_zeros([max(0, ABSENCE_HIGH - v) for v in vals], threshold=0)
        if high_streak >= ABSENCE_STREAK_ALARM:
            obs.append(f"🚨 {high_streak} недели подряд с высокими пропусками.")

        logger.debug("AttendanceAnalyzer: avg=%.1f score=%.2f", avg_abs, base)
        return FactorDetail(
            name="Посещаемость", score=round(min(base, 3.5), 2),
            level=level, headline=h, observations=obs,
            trend=trend, value=round(avg_abs, 1),
        ).validated()

    @staticmethod
    def _base_score(avg_abs: float, total: float) -> tuple[float, str, str]:
        if avg_abs > ABSENCE_CRITICAL_HIGH:
            return 3.0, "critical", f"Критические пропуски — {avg_abs:.1f} ч/нед (итого {total:.0f} ч)"
        elif avg_abs > ABSENCE_HIGH:
            return 2.0, "critical", f"Высокие пропуски — {avg_abs:.1f} ч/нед"
        elif avg_abs > ABSENCE_MEDIUM:
            return 1.2, "warning",  f"Умеренные пропуски — {avg_abs:.1f} ч/нед"
        elif avg_abs > ABSENCE_LOW:
            return 0.4, "warning",  f"Незначительные пропуски — {avg_abs:.1f} ч/нед"
        return 0.0, "ok", f"Посещаемость хорошая — {avg_abs:.1f} ч/нед"


# ══════════════════════════════════════════════════════════════
#  Анализ остальных факторов (функции, как было — простота важнее)
# ══════════════════════════════════════════════════════════════

def _analyze_lms(lms: list[dict]) -> FactorDetail:
    if not lms:
        return FactorDetail(
            name="LMS-активность", score=0.5, level="warning",
            headline="Данные LMS отсутствуют",
            observations=["Нет данных об активности в системе дистанционного обучения."],
        )

    logins_vals = [lo.get("logins") or 0 for lo in lms]
    avg_logins  = sum(logins_vals) / len(logins_vals)
    slope       = _trend_slope(logins_vals)
    trend       = _trend_label(slope, scale=0.5)

    total_on   = sum(lo.get("submissions_on_time") or 0 for lo in lms)
    total_late = sum(lo.get("submissions_late") or 0 for lo in lms)
    total_subs = total_on + total_late
    late_ratio = total_late / total_subs if total_subs > 0 else 0.0

    if avg_logins < LMS_ZERO_LOGINS:
        base, level = 2.0, "critical"
        h = f"Полная пассивность в LMS — {avg_logins:.1f} входов/нед"
    elif avg_logins < LMS_CRITICAL_LOW:
        base, level = 1.5, "critical"
        h = f"Критически низкая LMS-активность — {avg_logins:.1f} входов/нед"
    elif avg_logins < LMS_WARNING_LOW:
        base, level = 0.8, "warning"
        h = f"Низкая LMS-активность — {avg_logins:.1f} входов/нед"
    elif avg_logins < LMS_MODERATE:
        base, level = 0.2, "ok"
        h = f"Умеренная LMS-активность — {avg_logins:.1f} входов/нед"
    else:
        base, level = 0.0, "ok"
        h = f"Хорошая LMS-активность — {avg_logins:.1f} входов/нед"

    obs: list[str] = []

    if total_subs > 0:
        if late_ratio > LMS_LATE_RATIO_HIGH:
            obs.append(f"🕐 {late_ratio:.0%} заданий сдаётся с опозданием ({total_late}/{total_subs}).")
            base += 1.0
        elif late_ratio > LMS_LATE_RATIO_MED:
            obs.append(f"🕐 Значительная доля просрочек: {late_ratio:.0%} ({total_late}/{total_subs}).")
            base += 0.5
        elif late_ratio > LMS_LATE_RATIO_LOW:
            obs.append(f"Умеренная доля просрочек: {late_ratio:.0%}.")

    if _detect_silent_exit(logins_vals):
        obs.append(f"🚫 «Тихий уход»: {LMS_SILENT_EXIT_WEEKS}+ недели подряд нулевая активность в LMS.")
        base += 1.0
        level = "critical"

    if trend in ("резкое снижение", "снижение"):
        obs.append(f"▼ Активность в LMS снижается ({trend}).")
        base += 0.3

    vol = _volatility([float(v) for v in logins_vals])
    if vol > 5 and avg_logins < 6:
        obs.append(f"📊 Нерегулярная активность (σ={vol:.1f}) — заходит редко и хаотично.")

    if len(logins_vals) >= 4:
        recent = sum(logins_vals[-2:]) / 2
        prev   = sum(logins_vals[-4:-2]) / 2
        if prev > 4 and recent < prev * LMS_DROP_FACTOR:
            obs.append(f"⚡ Резкое падение активности: {prev:.1f} → {recent:.1f} входов/нед.")
            base += 0.3

    logger.debug("LMS: avg=%.1f score=%.2f", avg_logins, base)
    return FactorDetail(
        name="LMS-активность", score=round(min(base, 2.5), 2),
        level=level, headline=h, observations=obs,
        trend=trend, value=round(avg_logins, 1),
    ).validated()


def _analyze_payments(payments: list[dict]) -> FactorDetail:
    if not payments:
        return FactorDetail(
            name="Платежи", score=0.0, level="ok",
            headline="Платёжных данных нет (возможно, грантовик)",
        )

    overdue = [p for p in payments if p.get("paid_date") is None]
    paid    = [p for p in payments if p.get("paid_date") is not None]
    total   = len(payments)

    late_paid: list[int] = []
    for p in paid:
        try:
            due    = date.fromisoformat(p["due_date"])
            paid_d = date.fromisoformat(p["paid_date"])
            delay  = (paid_d - due).days
            if delay > 0:
                late_paid.append(delay)
        except Exception:
            pass

    overdue_count = len(overdue)

    if overdue_count >= PAYMENT_OVERDUE_CRITICAL:
        base, level = 2.0, "critical"
        h = f"Критическая задолженность — {overdue_count}/{total} платежей не оплачено"
    elif overdue_count >= PAYMENT_OVERDUE_SERIOUS:
        base, level = 1.5, "critical"
        h = f"Серьёзная задолженность — {overdue_count}/{total} платежей не оплачено"
    elif overdue_count >= PAYMENT_OVERDUE_ONE:
        base, level = 0.7, "warning"
        h = f"1 просроченный платёж из {total}"
    else:
        base, level = 0.0, "ok"
        h = f"Платёжная дисциплина хорошая — все {total} платежей оплачены"

    obs: list[str] = []
    if overdue_count > 0:
        overdue_total = sum(p.get("amount", 0) for p in overdue)
        obs.append(f"💸 Сумма задолженности: {overdue_total:,.0f} тг.")

    if late_paid:
        avg_delay = sum(late_paid) / len(late_paid)
        obs.append(f"⏳ Средняя задержка оплаты (по оплаченным): {avg_delay:.0f} дней.")
        if avg_delay > PAYMENT_LATE_DAYS_ALARM:
            base += 0.3

    if overdue_count > 0 and paid:
        obs.append(f"📋 История: {len(paid)} оплачено, {overdue_count} не оплачено.")

    return FactorDetail(
        name="Платежи", score=round(min(base, 2.0), 2),
        level=level, headline=h, observations=obs,
        value=float(overdue_count),
    ).validated()


def _analyze_profile(student: dict) -> FactorDetail:
    risk_factors: list[str] = []
    protective:   list[str] = []
    base = 0.0

    if student.get("works"):
        risk_factors.append("совмещает учёбу с работой (+0.3)")
        base += 0.3
    if student.get("accommodation_type") == "съём":
        risk_factors.append("снимает жильё (+0.2)")
        base += 0.2
    if not student.get("resident", True):
        risk_factors.append("иногородний студент (+0.2)")
        base += 0.2
    if student.get("tuition_form") == "платное":
        risk_factors.append("платное обучение (+0.1)")
        base += 0.1

    entry_score = student.get("entry_score") or 0
    if entry_score < PROFILE_ENTRY_SCORE_LOW:
        risk_factors.append(f"низкий входной балл {entry_score} (+0.2)")
        base += 0.2
    elif entry_score > PROFILE_ENTRY_SCORE_HIGH:
        protective.append(f"высокий входной балл {entry_score}")

    if "государственная" in (student.get("school_type") or ""):
        protective.append("государственная школа")

    obs: list[str] = []
    if risk_factors:
        obs.append("⚠️ Факторы уязвимости: " + "; ".join(risk_factors) + ".")
    if protective:
        obs.append("✅ Защитные факторы: " + "; ".join(protective) + ".")
    if not risk_factors and not protective:
        obs.append("Профиль студента без выраженных факторов риска.")

    level = "critical" if base >= PROFILE_CRITICAL_SCORE else ("warning" if base >= 0.3 else "ok")
    h = f"{len(risk_factors)} факторов уязвимости" if risk_factors else "Профиль без факторов риска"

    return FactorDetail(
        name="Профиль студента", score=round(min(base, 1.0), 2),
        level=level, headline=h, observations=obs, value=round(base, 2),
    ).validated()


# ══════════════════════════════════════════════════════════════
#  Ранние сигналы тревоги
# ══════════════════════════════════════════════════════════════

def _detect_early_warning_signals(
    grades:   list[dict],
    lms:      list[dict],
    payments: list[dict],
    student:  dict,
) -> list[str]:
    signals: list[str] = []

    grade_vals   = [g.get("avg_grade") or 0 for g in grades]
    login_vals   = [lo.get("logins") or 0 for lo in lms]
    absence_vals = [g.get("absence_hours") or 0 for g in grades]

    # «Двойной удар»
    if grade_vals and absence_vals and len(grade_vals) == len(absence_vals):
        dual_hit = sum(
            1 for g, a in zip(grade_vals, absence_vals)
            if g < SIGNAL_DUAL_HIT_GRADE and a > SIGNAL_DUAL_HIT_ABSENCE
        )
        if dual_hit >= SIGNAL_DUAL_HIT_WEEKS:
            signals.append(
                f"🔥 «Двойной удар»: {dual_hit} нед. одновременно низкие оценки "
                f"(<{SIGNAL_DUAL_HIT_GRADE}) И высокие пропуски (>{SIGNAL_DUAL_HIT_ABSENCE}ч)."
            )

    # «Исчезновение»
    if login_vals and absence_vals:
        recent_logins  = sum(login_vals[-3:]) if len(login_vals) >= 3 else sum(login_vals)
        recent_absence = sum(absence_vals[-3:]) / min(3, len(absence_vals))
        if recent_logins == 0 and recent_absence > SIGNAL_SILENT_ABSENCE:
            signals.append(
                "🚨 «Исчезновение»: нулевая LMS-активность + растущие пропуски — "
                "студент фактически отключился."
            )

    # «Срыв»: хороший старт → резкое падение
    if len(grade_vals) >= 6:
        start = sum(grade_vals[:3]) / 3
        end   = sum(grade_vals[-3:]) / 3
        if start >= SIGNAL_BURNOUT_START and end < SIGNAL_BURNOUT_END:
            signals.append(
                f"📉 «Срыв»: начал успешно ({start:.0f}), резкое падение к концу "
                f"({end:.0f}) — возможен внешний стрессор."
            )

    # «Хроническая нестабильность»
    if grade_vals and _volatility(grade_vals) > SIGNAL_VOLATILITY_ALARM:
        signals.append(
            "🎢 «Хроническая нестабильность»: экстремальные колебания оценок — "
            "нерегулярная подготовка или ситуативные проблемы."
        )

    # Долг + плохая успеваемость
    overdue   = [p for p in payments if p.get("paid_date") is None]
    avg_grade = sum(grade_vals) / len(grade_vals) if grade_vals else 100
    if len(overdue) >= SIGNAL_OVERDUE_MIN and avg_grade < SIGNAL_OVERDUE_GRADE_MAX:
        signals.append(
            "⚠️ Совпадение: долг по оплате + низкая успеваемость = высокий риск отчисления."
        )

    # «Тройная нагрузка»
    stress_count = sum([
        bool(student.get("works")),
        student.get("accommodation_type") == "съём",
        not student.get("resident", True),
    ])
    if stress_count == 3:
        signals.append(
            "🏋️ «Тройная нагрузка»: работает + снимает жильё + иногородний — "
            "высокий социальный стресс."
        )

    return signals


# ══════════════════════════════════════════════════════════════
#  Числовой скоринг (0–10)
# ══════════════════════════════════════════════════════════════

# Синглтоны анализаторов — создаются один раз
_grade_analyzer      = GradeAnalyzer()
_attendance_analyzer = AttendanceAnalyzer()


def compute_risk_score(
    grades:   list[dict],
    lms:      list[dict],
    payments: list[dict],
    student:  dict,
) -> tuple[float, dict, list[FactorDetail]]:
    """
    Возвращает (total_score, factors_dict, factor_details).
    Использует GradeAnalyzer и AttendanceAnalyzer для первых двух факторов.
    """
    fd_grades     = _grade_analyzer.analyze(grades)
    fd_attendance = _attendance_analyzer.analyze(grades)
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
    logger.debug(
        "compute_risk_score: student_id=%s total=%.2f",
        student.get("student_id"), total,
    )
    return total, factors, details


def score_to_level(score: float) -> str:
    if score <= RISK_LOW_MAX:
        return "низкий"
    elif score <= RISK_MEDIUM_MAX:
        return "средний"
    return "высокий"


# ══════════════════════════════════════════════════════════════
#  AI-движок резюме (детерминированный fallback)
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
        grades:  list[dict],
        lms:     list[dict],
        payments: list[dict],
        risk_score:     float,
        risk_level:     str,
        factor_details: list[FactorDetail],
        early_signals:  list[str],
    ) -> str:
        full_name = student.get("full_name", "Студент")
        seed = abs(hash(full_name)) % 1000

        level_emoji = {"низкий": "🟢", "средний": "🟡", "высокий": "🔴"}.get(risk_level, "⚪")
        parts = [
            f"{level_emoji} {full_name}: риск {risk_level.upper()}, скор {risk_score}/10.",
            self._pick(self._INTROS[risk_level], seed),
        ]

        critical = [d for d in factor_details if d.level == "critical" and d.score > 0]
        warnings = [d for d in factor_details if d.level == "warning"  and d.score > 0]

        for fd in sorted(critical, key=lambda x: -x.score)[:2]:
            parts.append(f"• {fd.name}: {fd.headline}.")
            if fd.observations:
                parts.append(f"  {fd.observations[0]}")

        for fd in sorted(warnings, key=lambda x: -x.score)[:1]:
            parts.append(f"• {fd.name}: {fd.headline}.")

        if early_signals:
            parts.append(f"Особые сигналы: {early_signals[0]}")

        parts.append(self._pick(self._RECS[risk_level], seed + 1))
        return "\n".join(parts)


_ai = LocalAI()


def generate_ai_summary(
    student:        dict,
    grades:         list[dict],
    lms:            list[dict],
    payments:       list[dict],
    risk_score:     float,
    risk_level:     str,
    factors:        dict,
    factor_details: list[FactorDetail] | None = None,
    early_signals:  list[str] | None = None,
) -> str:
    return _ai.generate(
        student, grades, lms, payments,
        risk_score, risk_level,
        factor_details or [],
        early_signals  or [],
    )


# ══════════════════════════════════════════════════════════════
#  Полный анализ одного студента
# ══════════════════════════════════════════════════════════════

def analyze_student(
    student:    dict,
    grades:     list[dict],
    lms:        list[dict],
    payments:   list[dict],
    week_start: str,
) -> dict:
    """
    Основная точка входа.
    Если config.USE_ML=True и модель загружена — использует ансамбль ML + эвристика.
    Иначе — только эвристика.
    """
    # Эвристический путь (всегда вычисляется — нужен и для ML-ансамбля)
    risk_score, factors, factor_details = compute_risk_score(grades, lms, payments, student)
    risk_level = score_to_level(risk_score)
    early_sigs = _detect_early_warning_signals(grades, lms, payments, student)

    # ML-путь (опционально)
    if config.USE_ML:
        try:
            from ml_model import load_model, predict_student as ml_predict
            pipeline, feature_names, _ = load_model()
            result = ml_predict(pipeline, student, grades, lms, payments, feature_names)
            risk_score = result.combined_score(config.ML_WEIGHT)
            risk_level = score_to_level(risk_score)
            logger.info(
                "ML ensemble: student_id=%s ml=%.2f heuristic=%.2f combined=%.2f",
                student.get("student_id"),
                result.risk_score_ml,
                result.risk_score_heuristic,
                risk_score,
            )
        except FileNotFoundError:
            logger.warning(
                "USE_ML=True но модель не найдена (models/ews_model.joblib). "
                "Использую только эвристику. Запустите: python ml_model.py train"
            )
        except Exception as exc:
            logger.error("ML predict failed for student %s: %s", student.get("student_id"), exc)

    ai_summary = generate_ai_summary(
        student, grades, lms, payments,
        risk_score, risk_level, factors,
        factor_details, early_sigs,
    )

    factors_extended = {
        **factors,
        "factor_details": [fd.to_dict() for fd in factor_details],
        "early_signals":  early_sigs,
    }

    logger.info(
        "analyze_student: id=%s week=%s level=%s score=%.2f",
        student.get("student_id"), week_start, risk_level, risk_score,
    )

    return {
        "student_id": student["student_id"],
        "week_start": week_start,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "ai_summary": ai_summary,
        "factors":    factors_extended,
    }