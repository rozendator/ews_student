"""
run_analysis.py — анализ студентов + умная рассылка кураторам.

Изменения v2:
  - Рассылка НЕ мгновенная: каждый куратор получает отчёт с задержкой
    (по умолчанию 30 секунд между кураторами, настраивается в config)
  - Перед рассылкой — сводка: «Анализ завершён. Отчёты отправляются поочерёдно.»
  - Логирование времени отправки каждому куратору
  - Retry при ошибке отправки (3 попытки)
  - Поддержка --dry-run для тестирования без реальной отправки

Использование:
    python run_analysis.py [YYYY-MM-DD] [--dry-run] [--delay SEC]
"""
from __future__ import annotations
import sys
import io
import asyncio
import argparse
from datetime import date, timedelta
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import db
from analyzer import analyze_student
from report import (
    format_weekly_summary,
    format_no_risk_message,
    format_analysis_complete_notice,
)
from config import TELEGRAM_TOKEN

try:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.constants import ParseMode
    from telegram.error import TelegramError, RetryAfter
    TG_AVAILABLE = True
except ImportError:
    TG_AVAILABLE = False

# Задержка между кураторами по умолчанию (секунды)
DEFAULT_CURATOR_DELAY = 30


def current_monday() -> str:
    today = date.today()
    return (today - timedelta(days=today.weekday())).isoformat()


# ══════════════════════════════════════════════════════════════
#  Анализ
# ══════════════════════════════════════════════════════════════

def run_analysis(week_start: str) -> dict:
    """Анализирует всех студентов, сохраняет отчёты. Возвращает статистику."""
    students = db.fetch_all_students()
    total = len(students)
    counts = {"низкий": 0, "средний": 0, "высокий": 0, "ошибок": 0}

    print(f"[EWS] Анализ за неделю {week_start} | студентов: {total}")

    for i, student in enumerate(students, 1):
        sid = student["student_id"]
        try:
            grades   = db.fetch_grades_by_week(sid, week_start)
            lms      = db.fetch_lms_by_week(sid, week_start)
            payments = db.fetch_payments(sid)

            report = analyze_student(student, grades, lms, payments, week_start)
            db.save_risk_report(report)

            level = report["risk_level"]
            if level in counts:
                counts[level] += 1

        except Exception as e:
            print(f"  [!] Ошибка для студента #{sid}: {e}")
            counts["ошибок"] += 1
            continue

        if i % 50 == 0 or i == total:
            print(
                f"  [{i}/{total}] "
                f"🔴 ВЫСОКИЙ:{counts['высокий']}  "
                f"🟡 СРЕДНИЙ:{counts['средний']}  "
                f"🟢 НИЗКИЙ:{counts['низкий']}"
            )

    print(f"\n[OK] Анализ завершён! "
          f"Высокий:{counts['высокий']} | Средний:{counts['средний']} | Низкий:{counts['низкий']}")
    return counts


# ══════════════════════════════════════════════════════════════
#  Отправка с ретраем
# ══════════════════════════════════════════════════════════════

async def _send_with_retry(
    bot: "Bot",
    chat_id: int,
    text: str,
    parse_mode: str,
    reply_markup=None,
    max_retries: int = 3,
) -> bool:
    """Отправляет сообщение с повторными попытками при ошибках."""
    for attempt in range(1, max_retries + 1):
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text[:4000],
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            return True
        except RetryAfter as e:
            wait = e.retry_after + 1
            print(f"    [Throttle] Ждём {wait}с (попытка {attempt})...")
            await asyncio.sleep(wait)
        except TelegramError as e:
            if attempt == max_retries:
                print(f"    [ERROR] Не удалось отправить после {max_retries} попыток: {e}")
                return False
            await asyncio.sleep(2 ** attempt)  # exponential backoff
        except Exception as e:
            print(f"    [ERROR] Неожиданная ошибка: {e}")
            return False
    return False


# ══════════════════════════════════════════════════════════════
#  Рассылка кураторам — поочерёдная с задержкой
# ══════════════════════════════════════════════════════════════

async def send_alerts(
    week_start: str,
    dry_run: bool = False,
    curator_delay: int = DEFAULT_CURATOR_DELAY,
) -> None:
    """
    Отправляет отчёты кураторам поочерёдно.
    dry_run=True — только логирует, не отправляет.
    curator_delay — пауза (сек) между кураторами.
    """
    if not TG_AVAILABLE:
        print("[WARN] python-telegram-bot не установлен, отправка пропущена.")
        return
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("[WARN] TELEGRAM_TOKEN не задан, отправка пропущена.")
        return

    bot = Bot(token=TELEGRAM_TOKEN)

    try:
        curators = db.fetch_all_curators()
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить кураторов: {e}")
        return

    # ── Сначала собираем данные для всех кураторов ──────────
    curator_data = []
    for curator in curators:
        if not curator.get("telegram_chat_id"):
            print(f"  [-] {curator.get('full_name', '?')} — нет chat_id, пропуск")
            continue

        try:
            cur_groups = db.fetch_curator_groups(curator["curator_id"])
            group_ids  = [r["group_id"] for r in cur_groups]
        except Exception as e:
            print(f"  [!] Ошибка групп для {curator.get('full_name')}: {e}")
            continue

        if not group_ids:
            continue

        all_students: list[dict] = []
        for gid in group_ids:
            try:
                all_students.extend(db.fetch_students_by_group(gid))
            except Exception as e:
                print(f"  [!] Ошибка студентов группы {gid}: {e}")

        pairs = []
        for student in all_students:
            try:
                report = db.fetch_risk_report(student["student_id"], week_start)
                if report and report.get("risk_level") in ("средний", "высокий"):
                    pairs.append((report, student))
            except Exception as e:
                print(f"  [!] fetch_risk_report #{student.get('student_id')}: {e}")

        curator_data.append({
            "curator":  curator,
            "pairs":    pairs,
            "students": all_students,
        })

    total_curators = len(curator_data)
    total_alerts   = sum(len(d["pairs"]) for d in curator_data)

    print(f"\n[EWS] Подготовлено отчётов: {total_curators} кураторов, {total_alerts} алертов.")
    print(f"[EWS] Режим: {'DRY-RUN (без отправки)' if dry_run else f'ОТПРАВКА с задержкой {curator_delay}с между кураторами'}")

    if not dry_run:
        # ── Глобальное уведомление: анализ завершён ──────────
        try:
            admin_curators = [d for d in curator_data]  # все кураторы получают уведомление
            notice_text = format_analysis_complete_notice(week_start, total_curators, total_alerts, curator_delay)
            # (уведомление отправляется в боте через статус — здесь только лог)
            print(f"[INFO] Уведомление о завершении анализа сформировано.")
        except Exception as e:
            print(f"[WARN] Ошибка формирования уведомления: {e}")

    # ── Поочерёдная отправка ─────────────────────────────────
    sent_ok = 0
    sent_err = 0

    for idx, data in enumerate(curator_data, 1):
        curator = data["curator"]
        pairs   = data["pairs"]
        chat_id = curator["telegram_chat_id"]
        name    = curator.get("full_name", str(chat_id))

        print(f"\n  [{idx}/{total_curators}] Куратор: {name} (chat_id={chat_id})")

        if dry_run:
            print(f"    [DRY-RUN] Было бы отправлено {len(pairs)} алертов.")
            continue

        try:
            if not pairs:
                msg = format_no_risk_message(name, week_start)
                ok = await _send_with_retry(bot, chat_id, msg, ParseMode.MARKDOWN)
            else:
                from report import format_weekly_summary
                messages = format_weekly_summary(name, week_start, pairs)
                ok = True
                for i, msg in enumerate(messages):
                    # Последнему сообщению добавляем кнопку
                    kb = None
                    if i == len(messages) - 1:
                        kb = InlineKeyboardMarkup([[
                            InlineKeyboardButton(
                                "📋 Открыть отчёт",
                                callback_data=f"report:{week_start}",
                            )
                        ]])
                    if not await _send_with_retry(bot, chat_id, msg, ParseMode.MARKDOWN, reply_markup=kb):
                        ok = False
                        break
                    if len(messages) > 1:
                        await asyncio.sleep(0.4)

            if ok:
                sent_ok += 1
                print(f"    [OK] Отправлено. Алертов: {len(pairs)}")

                # Лог алертов в БД
                for r, s in pairs:
                    try:
                        db.save_alert_log({
                            "student_id": s["student_id"],
                            "curator_id": curator["curator_id"],
                            "week_start": week_start,
                            "risk_level": r["risk_level"],
                        })
                    except Exception as e:
                        print(f"    [!] save_alert_log: {e}")
            else:
                sent_err += 1

        except Exception as e:
            print(f"    [ERROR] Ошибка отправки куратору {name}: {e}")
            sent_err += 1

        # ── Пауза перед следующим куратором ──────────────────
        if idx < total_curators:
            print(f"    [WAIT] Пауза {curator_delay}с перед следующим куратором...")
            await asyncio.sleep(curator_delay)

    print(f"\n[EWS] Рассылка завершена: ✅ {sent_ok} успешно, ❌ {sent_err} ошибок.")


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="EWS Analysis Runner")
    parser.add_argument("week", nargs="?", default=None, help="Неделя в формате YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Не отправлять сообщения")
    parser.add_argument("--delay", type=int, default=DEFAULT_CURATOR_DELAY,
                        help=f"Задержка между кураторами (сек, по умолчанию {DEFAULT_CURATOR_DELAY})")
    parser.add_argument("--no-send", action="store_true", help="Только анализ, без рассылки")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    week = args.week or current_monday()

    try:
        date.fromisoformat(week)
    except ValueError:
        print(f"[ERROR] Неверный формат даты: '{week}'. Используйте YYYY-MM-DD.")
        sys.exit(1)

    counts = run_analysis(week)

    if not args.no_send:
        asyncio.run(send_alerts(
            week_start=week,
            dry_run=args.dry_run,
            curator_delay=args.delay,
        ))
    else:
        print("[INFO] Рассылка пропущена (--no-send).")