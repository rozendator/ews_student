"""
bot.py — EWS Telegram бот v2. Максимально интерактивный.

Новое:
  - Детальная карточка студента: факторы → drill-down по каждому
  - Фильтры топа: по группе, по фактору
  - Навигация: пагинация списка, назад/вперёд
  - Инлайн-сравнение: студент vs. группа (если есть данные)
  - /factor <student_id> <factor> — подробности по фактору
  - /history <student_id> — история риска по неделям
  - /groups — список групп куратора со статистикой
  - Callback-меню: каждое действие через кнопки
  - Подробный отчёт: ранние сигналы, наблюдения по каждому фактору
"""
from __future__ import annotations
import asyncio
import logging
import sys
from datetime import date, timedelta
import json

from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest

import db
from report import (
    format_weekly_summary, format_no_risk_message, format_student_alert,
    format_factor_detail, format_student_history,
)
from config import TELEGRAM_TOKEN

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_MSG = 4000
PAGE_SIZE = 5  # студентов на странице в топе


# ══════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════

def current_monday() -> str:
    today = date.today()
    return (today - timedelta(days=today.weekday())).isoformat()


def prev_monday(week: str) -> str:
    return (date.fromisoformat(week) - timedelta(weeks=1)).isoformat()


def next_monday(week: str) -> str:
    return (date.fromisoformat(week) + timedelta(weeks=1)).isoformat()


def is_curator(chat_id: int) -> dict | None:
    try:
        return db.fetch_curator_by_chat_id(chat_id)
    except Exception as e:
        logger.error("is_curator error: %s", e)
        return None


def truncate(text: str, max_len: int = MAX_MSG) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 30] + "\n\n_...текст обрезан_"


def level_emoji(lvl: str) -> str:
    return {"высокий": "🔴", "средний": "🟡", "низкий": "🟢"}.get(lvl, "⚪")


async def safe_reply(
    update: Update,
    text: str,
    reply_markup=None,
    parse_mode: str | None = ParseMode.MARKDOWN,
) -> None:
    msg = update.message or (update.callback_query and update.callback_query.message)
    if not msg:
        return
    try:
        await msg.reply_text(truncate(text), parse_mode=parse_mode, reply_markup=reply_markup)
    except BadRequest as e:
        logger.warning("Markdown error, retrying plain: %s", e)
        plain = text.replace("*", "").replace("`", "").replace("_", "")
        try:
            await msg.reply_text(truncate(plain), reply_markup=reply_markup)
        except Exception as ex:
            logger.error("Failed to send: %s", ex)


async def safe_edit(
    update: Update,
    text: str,
    reply_markup=None,
) -> None:
    query = update.callback_query
    if not query:
        await safe_reply(update, text, reply_markup=reply_markup)
        return
    try:
        await query.edit_message_text(
            truncate(text), parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup,
        )
    except BadRequest as e:
        if "Message is not modified" in str(e):
            return
        logger.warning("edit error: %s", e)
        await safe_reply(update, text, reply_markup=reply_markup)
    except Exception as e:
        logger.error("safe_edit error: %s", e)
        await safe_reply(update, text, reply_markup=reply_markup)


# ══════════════════════════════════════════════════════════════
#  Клавиатуры
# ══════════════════════════════════════════════════════════════

def main_menu_keyboard(is_cur: bool = False) -> InlineKeyboardMarkup:
    if is_cur:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📋 Мой отчёт", callback_data=f"report:{current_monday()}"),
                InlineKeyboardButton("📊 Статус", callback_data="status"),
            ],
            [
                InlineKeyboardButton("🔴 Топ риска", callback_data="top:высокий:0"),
                InlineKeyboardButton("👥 Мои группы", callback_data="groups"),
            ],
            [
                InlineKeyboardButton("📈 История недели", callback_data=f"weekhistory:{current_monday()}"),
            ],
        ])
    else:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Статус анализа", callback_data="status"),
                InlineKeyboardButton("🔴 Топ риска", callback_data="top:высокий:0"),
            ],
            [
                InlineKeyboardButton("⚙️ Запустить анализ", callback_data="analyze"),
            ],
        ])


def week_nav_keyboard(week: str, back_cb: str = "menu") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("◀ Пред.", callback_data=f"report:{prev_monday(week)}"),
            InlineKeyboardButton("▶ След.", callback_data=f"report:{next_monday(week)}"),
        ],
        [InlineKeyboardButton("🏠 Меню", callback_data=back_cb)],
    ])


def student_card_keyboard(student_id: int, week: str, group_id: int | None = None) -> InlineKeyboardMarkup:
    """Клавиатура карточки студента с drill-down по факторам."""
    rows = [
        [
            InlineKeyboardButton("📊 Оценки",       callback_data=f"factor:{student_id}:grades:{week}"),
            InlineKeyboardButton("📅 Посещаемость", callback_data=f"factor:{student_id}:attendance:{week}"),
        ],
        [
            InlineKeyboardButton("💻 LMS",           callback_data=f"factor:{student_id}:lms:{week}"),
            InlineKeyboardButton("💰 Платежи",       callback_data=f"factor:{student_id}:payments:{week}"),
        ],
        [
            InlineKeyboardButton("📜 История риска", callback_data=f"history:{student_id}"),
        ],
    ]
    if group_id:
        rows.append([
            InlineKeyboardButton("👥 Отчёт группы", callback_data=f"report:{week}"),
        ])
    rows.append([InlineKeyboardButton("🏠 Меню", callback_data="menu")])
    return InlineKeyboardMarkup(rows)


def top_keyboard(level: str, page: int, week: str, total: int) -> InlineKeyboardMarkup:
    """Пагинация + фильтры уровня риска для топ-листа."""
    rows = []

    # Фильтр по уровню
    level_btns = []
    for lvl, emoji in [("высокий", "🔴"), ("средний", "🟡"), ("низкий", "🟢")]:
        mark = "●" if lvl == level else "○"
        level_btns.append(
            InlineKeyboardButton(f"{mark}{emoji}", callback_data=f"top:{lvl}:0")
        )
    rows.append(level_btns)

    # Пагинация
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀", callback_data=f"top:{level}:{page - 1}"))
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
    if (page + 1) * PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("▶", callback_data=f"top:{level}:{page + 1}"))
    if nav:
        rows.append(nav)

    rows.append([
        InlineKeyboardButton("📊 Статус", callback_data="status"),
        InlineKeyboardButton("🏠 Меню",   callback_data="menu"),
    ])
    return InlineKeyboardMarkup(rows)


def factor_detail_keyboard(student_id: int, week: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("← Карточка студента", callback_data=f"student:{student_id}:{week}"),
        ],
        [InlineKeyboardButton("🏠 Меню", callback_data="menu")],
    ])


def history_keyboard(student_id: int) -> InlineKeyboardMarkup:
    week = current_monday()
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("← Карточка", callback_data=f"student:{student_id}:{week}")],
        [InlineKeyboardButton("🏠 Меню", callback_data="menu")],
    ])


def groups_keyboard(back: str = "menu") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Меню", callback_data=back)],
    ])


# ══════════════════════════════════════════════════════════════
#  /start
# ══════════════════════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id  = update.effective_chat.id
    curator  = is_curator(chat_id)

    if curator:
        try:
            groups = db.fetch_curator_groups(curator["curator_id"])
            gnames = ", ".join(
                r["groups"]["group_name"] for r in groups if r.get("groups")
            ) or "—"
            n_groups = len(groups)
        except Exception:
            gnames, n_groups = "—", 0

        # Быстрая статистика текущей недели
        week = current_monday()
        try:
            high   = db.fetch_high_risk_students(week, ["высокий"])
            medium = db.fetch_high_risk_students(week, ["средний"])
            # Отфильтруем только студентов куратора
            gids = {r["group_id"] for r in db.fetch_curator_groups(curator["curator_id"])}
            my_high   = [r for r in high   if r.get("students", {}).get("group_id") in gids]
            my_medium = [r for r in medium if r.get("students", {}).get("group_id") in gids]
            stats_line = f"🔴 {len(my_high)} | 🟡 {len(my_medium)} в ваших группах\n\n"
        except Exception:
            stats_line = ""

        text = (
            f"👋 Привет, *{curator['full_name']}*!\n\n"
            f"Ваши группы ({n_groups}): *{gnames}*\n"
            f"{stats_line}"
            f"Неделя: `{week}`\n\n"
            "Выберите действие:"
        )
        await safe_reply(update, text, reply_markup=main_menu_keyboard(is_cur=True))
    else:
        text = (
            f"👋 Добро пожаловать в *EWS Bot*!\n\n"
            f"Ваш chat\\_id: `{chat_id}`\n\n"
            "Вы не зарегистрированы как куратор.\n"
            "Сообщите администратору этот chat\\_id для подключения."
        )
        await safe_reply(update, text, reply_markup=main_menu_keyboard(is_cur=False))


# ══════════════════════════════════════════════════════════════
#  /analyze
# ══════════════════════════════════════════════════════════════

async def _run_analysis(week: str) -> tuple[bool, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "run_analysis.py", week, "--no-send",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        if proc.returncode == 0:
            out = stdout.decode("utf-8", errors="replace")
            return True, out[-800:] if len(out) > 800 else out
        else:
            err = stderr.decode("utf-8", errors="replace")
            return False, err[-800:] if len(err) > 800 else err
    except asyncio.TimeoutError:
        return False, "Анализ занял слишком много времени (>10 мин)."
    except FileNotFoundError:
        return False, "Файл run_analysis.py не найден."
    except Exception as e:
        return False, f"Неожиданная ошибка: {e}"


async def cmd_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    week = ctx.args[0] if ctx.args else current_monday()
    try:
        date.fromisoformat(week)
    except ValueError:
        await safe_reply(update, f"❌ Неверный формат даты: `{week}`.")
        return

    await safe_reply(update, f"⏳ Запускаю анализ за *{week}*...\nЭто займёт несколько минут.")
    success, output = await _run_analysis(week)

    if success:
        # Статистика после анализа
        try:
            high   = db.fetch_high_risk_students(week, ["высокий"])
            medium = db.fetch_high_risk_students(week, ["средний"])
            low    = db.fetch_high_risk_students(week, ["низкий"])
            stats = (
                f"\n\n📊 *Итого:* {len(high) + len(medium) + len(low)} студентов\n"
                f"🔴 Высокий: {len(high)} | 🟡 Средний: {len(medium)} | 🟢 Низкий: {len(low)}"
            )
        except Exception:
            stats = ""

        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Статус",    callback_data="status"),
                InlineKeyboardButton("🔴 Топ риска", callback_data="top:высокий:0"),
            ],
            [InlineKeyboardButton("📤 Разослать кураторам", callback_data=f"send_alerts:{week}")],
        ])
        await safe_reply(
            update,
            f"✅ *Анализ завершён!*{stats}\n\n```\n{output}\n```",
            reply_markup=kb,
        )
    else:
        await safe_reply(update, f"❌ Ошибка анализа:\n```\n{output}\n```")


# ══════════════════════════════════════════════════════════════
#  Отчёт куратора
# ══════════════════════════════════════════════════════════════

async def _do_report(update: Update, curator: dict, week: str) -> None:
    try:
        cur_groups = db.fetch_curator_groups(curator["curator_id"])
        group_ids  = [r["group_id"] for r in cur_groups]
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка получения групп: {e}")
        return

    keyboard = week_nav_keyboard(week)

    if not group_ids:
        await safe_edit(update, "У вас нет прикреплённых групп.", reply_markup=keyboard)
        return

    try:
        all_students: list[dict] = []
        for gid in group_ids:
            all_students.extend(db.fetch_students_by_group(gid))
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка получения студентов: {e}", reply_markup=keyboard)
        return

    if not all_students:
        await safe_edit(update, "В ваших группах нет студентов.", reply_markup=keyboard)
        return

    pairs = []
    for student in all_students:
        try:
            report = db.fetch_risk_report(student["student_id"], week)
            if report and report.get("risk_level") in ("средний", "высокий"):
                pairs.append((report, student))
        except Exception as e:
            logger.warning("fetch_risk_report error: %s", e)

    if not pairs:
        try:
            has_any = db.fetch_high_risk_students(week, ["низкий", "средний", "высокий"])
        except Exception:
            has_any = []
        msg = (
            f"📭 Нет данных за *{week}*. Запустите `/analyze {week}`"
            if not has_any
            else format_no_risk_message(curator["full_name"], week)
        )
        await safe_edit(update, msg, reply_markup=keyboard)
        return

    # Статистика по отчёту
    high_cnt   = sum(1 for r, _ in pairs if r.get("risk_level") == "высокий")
    medium_cnt = sum(1 for r, _ in pairs if r.get("risk_level") == "средний")

    summary_header = (
        f"📋 *Отчёт куратора {curator['full_name']}*\n"
        f"Неделя: `{week}`\n\n"
        f"🔴 Высокий риск: *{high_cnt}* | 🟡 Средний: *{medium_cnt}*\n"
        f"Всего студентов в мониторинге: *{len(all_students)}*\n\n"
    )

    try:
        messages = format_weekly_summary(curator["full_name"], week, pairs)
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка формирования отчёта: {e}", reply_markup=keyboard)
        return

    # Первое сообщение — редактируем, остальные — новые
    first_msg = summary_header + messages[0] if messages else summary_header
    first_msg = first_msg[:MAX_MSG]

    for i, msg in enumerate(([first_msg] + messages[1:]) if messages else [summary_header]):
        kb = keyboard if i == len(messages) - 1 else None
        if i == 0:
            await safe_edit(update, msg, reply_markup=kb)
        else:
            tg_msg = (
                update.callback_query.message
                if update.callback_query else update.message
            )
            if tg_msg:
                try:
                    await tg_msg.reply_text(
                        truncate(msg), parse_mode=ParseMode.MARKDOWN, reply_markup=kb,
                    )
                except BadRequest as e:
                    plain = msg.replace("*", "").replace("`", "").replace("_", "")
                    await tg_msg.reply_text(truncate(plain), reply_markup=kb)
                await asyncio.sleep(0.3)


async def cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id  = update.effective_chat.id
    curator  = is_curator(chat_id)
    if not curator:
        await safe_reply(update, "⛔ Вы не зарегистрированы как куратор.")
        return
    week = ctx.args[0] if ctx.args else current_monday()
    try:
        date.fromisoformat(week)
    except ValueError:
        await safe_reply(update, f"❌ Неверный формат даты: `{week}`.")
        return
    await _do_report(update, curator, week)


# ══════════════════════════════════════════════════════════════
#  Карточка студента с детальным анализом
# ══════════════════════════════════════════════════════════════

async def _do_student_card(update: Update, curator: dict, student_id: int, week: str) -> None:
    """Показывает полную карточку студента."""
    try:
        student = db.fetch_student(student_id)
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка: {e}")
        return

    if not student:
        await safe_edit(update, f"Студент #{student_id} не найден.")
        return

    # Проверка доступа
    try:
        cur_groups   = db.fetch_curator_groups(curator["curator_id"])
        allowed_gids = {r["group_id"] for r in cur_groups}
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка прав доступа: {e}")
        return

    if student.get("group_id") not in allowed_gids:
        await safe_edit(update, "⛔ Этот студент не в ваших группах.")
        return

    try:
        report = db.fetch_risk_report(student_id, week)
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка получения отчёта: {e}")
        return

    if not report:
        await safe_edit(
            update,
            f"Нет отчёта за {week} для студента #{student_id}.\nЗапустите `/analyze`",
        )
        return

    # Загружаем данные для последних 4 недель
    try:
        grades   = db.fetch_grades(student_id, weeks=6)
        lms_data = db.fetch_lms(student_id, weeks=6)
        pays     = db.fetch_payments(student_id)
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка загрузки данных: {e}")
        return

    # Строим карточку
    lvl   = report.get("risk_level", "—")
    score = report.get("risk_score", 0)
    emoji = level_emoji(lvl)

    grp_name = ""
    try:
        grp = db.fetch_group(student.get("group_id"))
        grp_name = grp.get("group_name", "") if grp else ""
    except Exception:
        pass

    lines = [
        f"{emoji} *{student.get('full_name', '?')}*",
        f"Группа: {grp_name} | Специальность: {student.get('specialty', '—')}",
        f"Форма обучения: {student.get('tuition_form', '—')} | Язык: {student.get('language', '—')}",
        f"",
        f"*Уровень риска:* {lvl.upper()} ({score}/10)",
        f"*Неделя:* `{week}`",
        f"",
    ]

    # Факторы из отчёта
    factors = report.get("factors", {})
    factor_details = factors.get("factor_details", [])
    if factor_details:
        lines.append("*Факторы риска:*")
        for fd in factor_details:
            fd_emoji = {"critical": "🔴", "warning": "🟡", "ok": "🟢"}.get(fd.get("level", "ok"), "⚪")
            lines.append(f"{fd_emoji} {fd['name']}: {fd['headline']} ({fd['score']:.1f})")
        lines.append("")

    # Ранние сигналы
    early_signals = factors.get("early_signals", [])
    if early_signals:
        lines.append("*⚡ Ранние сигналы тревоги:*")
        for sig in early_signals:
            lines.append(f"• {sig}")
        lines.append("")

    # AI-резюме
    ai_summary = report.get("ai_summary", "")
    if ai_summary:
        lines.append("*📝 AI-резюме:*")
        lines.append(ai_summary)
        lines.append("")

    # Последние 3 недели оценок
    if grades:
        lines.append("*📈 Последние недели:*")
        for g in grades[-3:]:
            w = g.get("week_start", "?")
            ag = g.get("avg_grade", "—")
            ah = g.get("absence_hours", 0)
            lines.append(f"`{w}`: балл *{ag}* | пропуски *{ah}* ч")
        lines.append("")

    # Задолженность
    overdue = [p for p in pays if p.get("paid_date") is None]
    if overdue:
        total_debt = sum(p.get("amount", 0) for p in overdue)
        lines.append(f"💸 *Задолженность:* {len(overdue)} платежей ({total_debt:,.0f} тг)")

    text = "\n".join(lines)
    kb   = student_card_keyboard(student_id, week, student.get("group_id"))
    await safe_edit(update, text, reply_markup=kb)


async def cmd_student(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    curator = is_curator(chat_id)
    if not curator:
        await safe_reply(update, "⛔ Только для кураторов.")
        return
    if not ctx.args:
        await safe_reply(update, "Использование: `/student <student_id>`")
        return
    try:
        sid = int(ctx.args[0])
    except ValueError:
        await safe_reply(update, "❌ ID должен быть числом.")
        return
    await _do_student_card(update, curator, sid, current_monday())


# ══════════════════════════════════════════════════════════════
#  /status
# ══════════════════════════════════════════════════════════════

async def _do_status(update: Update, week: str) -> None:
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔴 Высокий риск", callback_data="top:высокий:0"),
            InlineKeyboardButton("🟡 Средний",      callback_data="top:средний:0"),
        ],
        [
            InlineKeyboardButton("📤 Разослать",    callback_data=f"send_alerts:{week}"),
            InlineKeyboardButton("🏠 Меню",          callback_data="menu"),
        ],
    ])
    try:
        high   = db.fetch_high_risk_students(week, ["высокий"])
        medium = db.fetch_high_risk_students(week, ["средний"])
        low    = db.fetch_high_risk_students(week, ["низкий"])
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка: {e}", reply_markup=kb)
        return

    total = len(high) + len(medium) + len(low)
    if total == 0:
        await safe_edit(
            update,
            f"📭 Нет данных за *{week}*.\nЗапустите `/analyze`",
            reply_markup=kb,
        )
        return

    # Распределение по факторам (из первых 50 отчётов высокого риска)
    factor_counts: dict[str, int] = {}
    for r in high[:50]:
        fdetails = (r.get("factors") or {}).get("factor_details", [])
        for fd in fdetails:
            if fd.get("level") == "critical":
                fname = fd.get("name", "?")
                factor_counts[fname] = factor_counts.get(fname, 0) + 1

    top_factors = sorted(factor_counts.items(), key=lambda x: -x[1])[:3]
    factors_line = ""
    if top_factors:
        fstr = " | ".join(f"{n}: {c}" for n, c in top_factors)
        factors_line = f"\n\n*Топ критических факторов:*\n{fstr}"

    text = (
        f"📊 *Статус EWS | {week}*\n\n"
        f"Всего: *{total}* студентов\n\n"
        f"🔴 Высокий риск: *{len(high)}*\n"
        f"🟡 Средний риск: *{len(medium)}*\n"
        f"🟢 Низкий риск:  *{len(low)}*"
        f"{factors_line}"
    )
    await safe_edit(update, text, reply_markup=kb)


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    week = ctx.args[0] if ctx.args else current_monday()
    try:
        date.fromisoformat(week)
    except ValueError:
        await safe_reply(update, f"❌ Неверный формат даты: `{week}`.")
        return
    await _do_status(update, week)


# ══════════════════════════════════════════════════════════════
#  /top — пагинированный список с фильтрами
# ══════════════════════════════════════════════════════════════

async def _do_top(update: Update, week: str, level: str = "высокий", page: int = 0) -> None:
    try:
        students_list = db.fetch_high_risk_students(week, [level])
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка: {e}")
        return

    total = len(students_list)
    kb    = top_keyboard(level, page, week, total)

    if not students_list:
        await safe_edit(
            update,
            f"{level_emoji(level)} Студентов с уровнем риска *{level}* за *{week}* нет.",
            reply_markup=kb,
        )
        return

    start = page * PAGE_SIZE
    end   = min(start + PAGE_SIZE, total)
    page_items = students_list[start:end]

    emoji = level_emoji(level)
    lines = [f"{emoji} *Топ — {level} риск | {week}*"]
    lines.append(f"Показано {start + 1}–{end} из {total}\n")

    for i, r in enumerate(page_items, start + 1):
        s     = r.get("students") or {}
        grp   = (s.get("groups") or {}).get("group_name", "?") if isinstance(s.get("groups"), dict) else "?"
        score = r.get("risk_score", 0)
        name  = s.get("full_name", "—")
        sid   = s.get("student_id") or r.get("student_id", "?")

        # Главный критический фактор
        factor_hint = ""
        fdetails = (r.get("factors") or {}).get("factor_details", [])
        critical_factors = [fd for fd in fdetails if fd.get("level") == "critical"]
        if critical_factors:
            top_f = max(critical_factors, key=lambda x: x.get("score", 0))
            factor_hint = f" ⚑ {top_f['name']}"

        grp_safe = grp.replace("(", "\\(").replace(")", "\\)")
        lines.append(
            f"{i}. *{name}* ({grp_safe}) — {score}/10{factor_hint}\n"
            f"   /student\\_{sid}"
        )

    if total > end:
        lines.append(f"\n_...ещё {total - end} студентов_")

    await safe_edit(update, "\n".join(lines), reply_markup=kb)


async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    week = ctx.args[0] if ctx.args else current_monday()
    await _do_top(update, week)


# ══════════════════════════════════════════════════════════════
#  /groups — группы куратора со статистикой
# ══════════════════════════════════════════════════════════════

async def _do_groups(update: Update, curator: dict) -> None:
    week = current_monday()
    try:
        cur_groups = db.fetch_curator_groups(curator["curator_id"])
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка: {e}")
        return

    if not cur_groups:
        await safe_edit(update, "У вас нет прикреплённых групп.", reply_markup=groups_keyboard())
        return

    lines = [f"👥 *Мои группы | {week}*\n"]
    for cg in cur_groups:
        grp = cg.get("groups") or {}
        gid = cg.get("group_id")
        gname = grp.get("group_name", f"#{gid}")

        try:
            students = db.fetch_students_by_group(gid)
            n_students = len(students)
            high_cnt = medium_cnt = 0
            for stu in students:
                r = db.fetch_risk_report(stu["student_id"], week)
                if r:
                    if r.get("risk_level") == "высокий":
                        high_cnt += 1
                    elif r.get("risk_level") == "средний":
                        medium_cnt += 1
        except Exception:
            n_students = high_cnt = medium_cnt = 0

        lines.append(
            f"*{gname}* — {n_students} студентов\n"
            f"  🔴 {high_cnt} | 🟡 {medium_cnt} | 🟢 {n_students - high_cnt - medium_cnt}\n"
        )

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Отчёт", callback_data=f"report:{week}")],
        [InlineKeyboardButton("🏠 Меню", callback_data="menu")],
    ])
    await safe_edit(update, "\n".join(lines), reply_markup=kb)


async def cmd_groups(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    curator = is_curator(chat_id)
    if not curator:
        await safe_reply(update, "⛔ Только для кураторов.")
        return
    await _do_groups(update, curator)


# ══════════════════════════════════════════════════════════════
#  /history — история риска студента
# ══════════════════════════════════════════════════════════════

async def _do_history(update: Update, curator: dict, student_id: int) -> None:
    try:
        student = db.fetch_student(student_id)
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка: {e}")
        return

    if not student:
        await safe_edit(update, f"Студент #{student_id} не найден.")
        return

    # Проверка доступа
    try:
        cur_groups   = db.fetch_curator_groups(curator["curator_id"])
        allowed_gids = {r["group_id"] for r in cur_groups}
    except Exception:
        allowed_gids = set()

    if student.get("group_id") not in allowed_gids:
        await safe_edit(update, "⛔ Нет доступа к этому студенту.")
        return

    try:
        history = db.fetch_student_risk_history(student_id, weeks=8)
    except Exception as e:
        # Fallback: пробуем через fetch_risk_reports
        history = []

    name = student.get("full_name", "?")
    lines = [f"📜 *История риска: {name}*\n"]

    if not history:
        lines.append("История рисков не найдена.")
    else:
        for h in history:
            w     = h.get("week_start", "?")
            lvl   = h.get("risk_level", "—")
            score = h.get("risk_score", 0)
            em    = level_emoji(lvl)
            lines.append(f"`{w}` {em} {lvl} ({score}/10)")

        # Тренд
        if len(history) >= 3:
            scores = [h.get("risk_score", 0) for h in history]
            if scores[-1] > scores[0] + 1:
                lines.append("\n📈 Тренд: *рост риска* — ситуация ухудшается.")
            elif scores[-1] < scores[0] - 1:
                lines.append("\n📉 Тренд: *снижение риска* — положительная динамика.")
            else:
                lines.append("\n→ Тренд: *стабильный*.")

    kb = history_keyboard(student_id)
    await safe_edit(update, "\n".join(lines), reply_markup=kb)


async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    curator = is_curator(chat_id)
    if not curator:
        await safe_reply(update, "⛔ Только для кураторов.")
        return
    if not ctx.args:
        await safe_reply(update, "Использование: `/history <student_id>`")
        return
    try:
        sid = int(ctx.args[0])
    except ValueError:
        await safe_reply(update, "❌ ID должен быть числом.")
        return
    await _do_history(update, curator, sid)


# ══════════════════════════════════════════════════════════════
#  Drill-down по фактору
# ══════════════════════════════════════════════════════════════

async def _do_factor_detail(
    update: Update,
    curator: dict,
    student_id: int,
    factor_key: str,
    week: str,
) -> None:
    """Детальный разбор конкретного фактора студента."""
    factor_names = {
        "grades":     "Оценки",
        "attendance": "Посещаемость",
        "lms":        "LMS-активность",
        "payments":   "Платежи",
        "profile":    "Профиль студента",
    }

    try:
        student = db.fetch_student(student_id)
        report  = db.fetch_risk_report(student_id, week)
    except Exception as e:
        await safe_edit(update, f"❌ Ошибка: {e}")
        return

    if not student or not report:
        await safe_edit(update, "Данные не найдены.")
        return

    fname = factor_names.get(factor_key, factor_key)
    factors_data = report.get("factors", {})
    factor_details = factors_data.get("factor_details", [])

    # Найдём нужный фактор
    fd = next((f for f in factor_details if f.get("name") == fname), None)

    lines = [
        f"🔍 *Детали: {fname}*",
        f"Студент: *{student.get('full_name', '?')}*",
        f"Неделя: `{week}`\n",
    ]

    if fd:
        fd_emoji = {"critical": "🔴", "warning": "🟡", "ok": "🟢"}.get(fd.get("level", "ok"), "⚪")
        lines.append(f"{fd_emoji} *{fd.get('headline', '—')}*")
        lines.append(f"Вклад в скор: *{fd.get('score', 0):.1f}* из 10")
        if fd.get("trend"):
            lines.append(f"Тренд: *{fd['trend']}*")
        lines.append("")

        obs = fd.get("observations", [])
        if obs:
            lines.append("*Наблюдения:*")
            for o in obs:
                lines.append(f"• {o}")
    else:
        lines.append("Детализация по данному фактору недоступна.")

    # Дополнительные данные по фактору
    try:
        if factor_key in ("grades", "attendance"):
            raw_grades = db.fetch_grades(student_id, weeks=8)
            if raw_grades:
                lines.append("\n*📊 Данные по неделям:*")
                for g in raw_grades[-6:]:
                    w = g.get("week_start", "?")
                    ag = g.get("avg_grade", "—")
                    ah = g.get("absence_hours", 0)
                    if factor_key == "grades":
                        bar = "█" * int((ag or 0) / 10) if ag else "░░░░░░░░░░"
                        lines.append(f"`{w}`: {ag:.1f} {bar}")
                    else:
                        lines.append(f"`{w}`: {ah} ч пропусков")

        elif factor_key == "lms":
            raw_lms = db.fetch_lms(student_id, weeks=8)
            if raw_lms:
                lines.append("\n*💻 LMS по неделям:*")
                for l in raw_lms[-6:]:
                    w   = l.get("week_start", "?")
                    log = l.get("logins", 0)
                    on  = l.get("submissions_on_time", 0)
                    late = l.get("submissions_late", 0)
                    lines.append(f"`{w}`: входов {log} | вовремя {on} | просрочено {late}")

        elif factor_key == "payments":
            pays = db.fetch_payments(student_id)
            if pays:
                lines.append("\n*💰 Платежи:*")
                for p in pays:
                    due  = p.get("due_date", "?")
                    paid = p.get("paid_date")
                    amt  = p.get("amount", 0)
                    status = f"✅ оплачено {paid}" if paid else "❌ не оплачено"
                    lines.append(f"`{due}`: {amt:,.0f} тг — {status}")
    except Exception as e:
        logger.warning("factor detail data error: %s", e)

    kb = factor_detail_keyboard(student_id, week)
    await safe_edit(update, "\n".join(lines), reply_markup=kb)


# ══════════════════════════════════════════════════════════════
#  Callback handler — центральный диспетчер
# ══════════════════════════════════════════════════════════════

async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()

    data    = query.data or ""
    chat_id = update.effective_chat.id
    curator = is_curator(chat_id)

    try:
        # ── Меню ───────────────────────────────────────────────
        if data == "menu":
            is_cur = curator is not None
            greeting = (
                f"👋 *{curator['full_name']}*, главное меню:"
                if curator else "👋 *Главное меню EWS*"
            )
            await safe_edit(update, greeting, reply_markup=main_menu_keyboard(is_cur=is_cur))

        elif data == "noop":
            pass  # кнопка-счётчик страниц, ничего не делаем

        # ── Статус ─────────────────────────────────────────────
        elif data == "status":
            await _do_status(update, current_monday())

        # ── Топ с фильтром и пагинацией ────────────────────────
        elif data.startswith("top:"):
            parts = data.split(":")
            level = parts[1] if len(parts) > 1 else "высокий"
            page  = int(parts[2]) if len(parts) > 2 else 0
            await _do_top(update, current_monday(), level, page)

        # ── Отчёт куратора ─────────────────────────────────────
        elif data.startswith("report:"):
            if not curator:
                await safe_edit(update, "⛔ Только для кураторов.")
                return
            week = data.split(":", 1)[1]
            try:
                date.fromisoformat(week)
            except ValueError:
                await safe_edit(update, f"❌ Неверная дата: {week}")
                return
            await _do_report(update, curator, week)

        # ── Карточка студента ──────────────────────────────────
        elif data.startswith("student:"):
            if not curator:
                await safe_edit(update, "⛔ Только для кураторов.")
                return
            parts = data.split(":")
            try:
                sid  = int(parts[1])
                week = parts[2] if len(parts) > 2 else current_monday()
            except (IndexError, ValueError):
                await safe_edit(update, "❌ Неверный формат данных.")
                return
            await _do_student_card(update, curator, sid, week)

        # ── Drill-down по фактору ──────────────────────────────
        elif data.startswith("factor:"):
            if not curator:
                await safe_edit(update, "⛔ Только для кураторов.")
                return
            # format: factor:{student_id}:{factor_key}:{week}
            parts = data.split(":")
            try:
                sid        = int(parts[1])
                factor_key = parts[2]
                week       = parts[3] if len(parts) > 3 else current_monday()
            except (IndexError, ValueError):
                await safe_edit(update, "❌ Неверный формат данных.")
                return
            await _do_factor_detail(update, curator, sid, factor_key, week)

        # ── История риска студента ─────────────────────────────
        elif data.startswith("history:"):
            if not curator:
                await safe_edit(update, "⛔ Только для кураторов.")
                return
            try:
                sid = int(data.split(":")[1])
            except (IndexError, ValueError):
                await safe_edit(update, "❌ Неверный ID.")
                return
            await _do_history(update, curator, sid)

        # ── Группы куратора ────────────────────────────────────
        elif data == "groups":
            if not curator:
                await safe_edit(update, "⛔ Только для кураторов.")
                return
            await _do_groups(update, curator)

        # ── История недели ─────────────────────────────────────
        elif data.startswith("weekhistory:"):
            week = data.split(":", 1)[1]
            await _do_status(update, week)

        # ── Запуск анализа ─────────────────────────────────────
        elif data == "analyze":
            week = current_monday()
            await safe_edit(
                update,
                f"⏳ Запускаю анализ за *{week}*...\nЭто займёт несколько минут.",
            )
            success, output = await _run_analysis(week)
            kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📊 Статус",    callback_data="status"),
                    InlineKeyboardButton("🔴 Топ риска", callback_data="top:высокий:0"),
                ],
                [InlineKeyboardButton("📤 Разослать кураторам", callback_data=f"send_alerts:{week}")],
            ])
            msg = (
                f"✅ *Анализ завершён!*\n\n```\n{output}\n```"
                if success
                else f"❌ *Ошибка анализа:*\n```\n{output}\n```"
            )
            try:
                await query.message.reply_text(
                    truncate(msg), parse_mode=ParseMode.MARKDOWN, reply_markup=kb,
                )
            except Exception as e:
                logger.error("analyze reply error: %s", e)

        # ── Рассылка кураторам ─────────────────────────────────
        elif data.startswith("send_alerts:"):
            week = data.split(":", 1)[1]
            await safe_edit(
                update,
                f"📤 Запускаю рассылку за *{week}*...\n"
                f"Отчёты будут доставляться поочерёдно с паузой 30 сек.",
            )
            # Запускаем рассылку в фоне
            asyncio.create_task(_background_send(week, query.message))

        else:
            logger.warning("Unknown callback: %s", data)

    except Exception as e:
        logger.error("button_handler error for data=%s: %s", data, e, exc_info=True)
        try:
            await query.message.reply_text(f"❌ Внутренняя ошибка: {e}")
        except Exception:
            pass


async def _background_send(week: str, message) -> None:
    """Фоновая рассылка — не блокирует бот."""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "run_analysis.py", week, "--no-send",  # анализ уже выполнен
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Запускаем рассылку отдельным процессом
        proc2 = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            f"import asyncio; from run_analysis import send_alerts; asyncio.run(send_alerts('{week}', curator_delay=30))",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc2.communicate(), timeout=3600)
        if proc2.returncode == 0:
            try:
                await message.reply_text("✅ Рассылка завершена!", parse_mode=ParseMode.MARKDOWN)
            except Exception:
                pass
        else:
            err = stderr.decode("utf-8", errors="replace")[-500:]
            try:
                await message.reply_text(f"❌ Ошибка рассылки:\n```{err}```", parse_mode=ParseMode.MARKDOWN)
            except Exception:
                pass
    except Exception as e:
        logger.error("background send error: %s", e)


# ══════════════════════════════════════════════════════════════
#  Регистрация команд
# ══════════════════════════════════════════════════════════════

async def set_commands(app: Application) -> None:
    await app.bot.set_my_commands([
        BotCommand("start",   "Начало работы / главное меню"),
        BotCommand("report",  "Отчёт куратора за неделю"),
        BotCommand("student", "Карточка студента /student <id>"),
        BotCommand("top",     "Топ студентов по уровню риска"),
        BotCommand("groups",  "Мои группы со статистикой"),
        BotCommand("history", "История риска студента /history <id>"),
        BotCommand("analyze", "Запустить анализ (admin)"),
        BotCommand("status",  "Статистика последнего анализа"),
    ])


def main() -> None:
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Укажите TELEGRAM_TOKEN.")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).post_init(set_commands).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("report",  cmd_report))
    app.add_handler(CommandHandler("student", cmd_student))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("top",     cmd_top))
    app.add_handler(CommandHandler("groups",  cmd_groups))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CallbackQueryHandler(button_handler))

    logger.info("[EWS Bot v2] Запущен.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()