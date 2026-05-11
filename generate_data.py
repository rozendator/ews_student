"""
generate_data.py — генерация синтетических данных EWS
Сохраняет CSV-файлы в папку data/
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

np.random.seed(42)
OUT = Path("data")
OUT.mkdir(exist_ok=True)

N_GROUPS       = 10
N_CURATORS     = 8
N_STUDENTS     = 300
N_WEEKS        = 16
SEMESTER_START = date(2024, 9, 2)

REGIONS     = ["Алматы","Астана","Шымкент","Актобе","Карагандинская обл.",
               "ВКО","ЗКО","Павлодарская обл.","СКО","Жамбылская обл."]
SPECIALTIES = ["Информационные системы","Финансы","Юриспруденция",
               "Педагогика","Менеджмент","Экономика","Программная инженерия"]
SCHOOLS     = ["государственная","частная","НИШ/БИЛ"]
LANGS       = ["казахский","русский","английский"]
DEPTS       = ["Факультет ИТ","Факультет экономики","Факультет права","Факультет педагогики"]

# 1. groups
groups = pd.DataFrame({
    "group_id":   range(1, N_GROUPS + 1),
    "group_name": [f"ГР-{100 + i}" for i in range(N_GROUPS)]
})
groups.to_csv(OUT / "groups.csv", index=False)
print(f"groups: {len(groups)}")

# 2. curators  (ВАЖНО: используем фиктивные chat_id — замените на реальные перед тестом)
first_names = ["Айгерим","Данияр","Сауле","Бауыржан","Зарина","Нурлан","Гульнара","Арман"]
last_names  = ["Сейтова","Ахметов","Нурова","Касымов","Бекова","Жаксыбеков","Омарова","Темиров"]

curators = pd.DataFrame({
    "curator_id":       range(1, N_CURATORS + 1),
    "full_name":        [f"{last_names[i]} {first_names[i]}" for i in range(N_CURATORS)],
    "telegram_chat_id": [100000001 + i * 7 for i in range(N_CURATORS)],
    "department":       np.random.choice(DEPTS, N_CURATORS)
})
curators.to_csv(OUT / "curators.csv", index=False)
print(f"curators: {len(curators)}")

# 3. curator_groups
cg_rows = []
group_ids_shuffled = list(range(1, N_GROUPS + 1))
np.random.shuffle(group_ids_shuffled)
for i, gid in enumerate(group_ids_shuffled):
    cg_rows.append({"curator_id": (i % N_CURATORS) + 1, "group_id": gid})
curator_groups = pd.DataFrame(cg_rows)
curator_groups.to_csv(OUT / "curator_groups.csv", index=False)
print(f"curator_groups: {len(curator_groups)}")

# 4. students
s_first = ["Айдана","Берик","Камила","Нурсултан","Алия","Дамир","Жанар",
           "Ерлан","Сания","Асель","Рустем","Мадина","Тимур","Аида",
           "Самал","Дастан","Гульназ","Алибек","Назгуль","Серик"]
s_last  = ["Сейтқали","Нұрланов","Әбдіқалық","Жақсыбеков","Тоқаева",
           "Мамытов","Серікова","Бегалин","Қасымова","Темірова",
           "Оразов","Бектенова","Жаңабеков","Нұрова","Сатыбалдина",
           "Қожахмет","Байғазина","Қабылов","Мұсабекова","Ізтілеуов"]

stu_rows = []
for sid in range(1, N_STUDENTS + 1):
    region = np.random.choice(REGIONS)
    stu_rows.append({
        "student_id":          sid,
        "full_name":           f"{np.random.choice(s_last)} {np.random.choice(s_first)}",
        "exam_type":           np.random.choice(["ЕНТ","КТА","Международный"], p=[0.80,0.15,0.05]),
        "entry_score":         int(np.clip(np.random.normal(72, 15), 30, 140)),
        "school_type":         np.random.choice(SCHOOLS, p=[0.70,0.20,0.10]),
        "language":            np.random.choice(LANGS, p=[0.50,0.40,0.10]),
        "resident":            bool(np.random.choice([True, False], p=[0.85, 0.15])),
        "specialty":           np.random.choice(SPECIALTIES),
        "registration_region": region,
        "lives_in_almaty":     bool(region == "Алматы"),
        "accommodation_type":  np.random.choice(["общежитие","съём","с родителями"], p=[0.35,0.40,0.25]),
        "works":               bool(np.random.choice([True, False], p=[0.30, 0.70])),
        "tuition_form":        np.random.choice(["грант","платное"], p=[0.45, 0.55]),
        "group_id":            np.random.randint(1, N_GROUPS + 1),
    })
students = pd.DataFrame(stu_rows)
students.to_csv(OUT / "students.csv", index=False)
print(f"students: {len(students)}")

# 5-6. Архетипы для реалистичной динамики
ARCHETYPES = {
    "отличник":   {"p": 0.20, "attend_base": 92, "lms_base": 9, "grade_base": 86, "drift": -0.1},
    "хорошист":   {"p": 0.30, "attend_base": 78, "lms_base": 6, "grade_base": 70, "drift":  0.0},
    "работающий": {"p": 0.20, "attend_base": 62, "lms_base": 4, "grade_base": 60, "drift":  0.3},
    "в_риске":    {"p": 0.20, "attend_base": 48, "lms_base": 2, "grade_base": 44, "drift":  0.8},
    "критичный":  {"p": 0.10, "attend_base": 30, "lms_base": 1, "grade_base": 30, "drift":  1.5},
}
arch_names = list(ARCHETYPES.keys())
arch_probs = [ARCHETYPES[a]["p"] for a in arch_names]
student_arch = np.random.choice(arch_names, size=N_STUDENTS, p=arch_probs)
DEADLINE_WEEKS = {4, 8, 12, 15}

lms_rows = []
for sid in range(1, N_STUDENTS + 1):
    arch = ARCHETYPES[student_arch[sid - 1]]
    for w in range(N_WEEKS):
        week_start = SEMESTER_START + timedelta(weeks=w)
        decay      = arch["drift"] * (w / N_WEEKS)
        dl_boost   = 1.5 if (w + 1) in DEADLINE_WEEKS else 1.0
        lms_rows.append({
            "student_id":          sid,
            "week_start":          week_start.isoformat(),
            "logins":              max(0, int(np.random.normal(arch["lms_base"] * dl_boost - decay * 2, 1.5))),
            "submissions_on_time": max(0, int(np.random.normal(2 - decay, 0.8))),
            "submissions_late":    max(0, int(np.random.normal(0.5 + decay, 0.5))),
        })
weekly_lms = pd.DataFrame(lms_rows)
weekly_lms.to_csv(OUT / "weekly_lms.csv", index=False)
print(f"weekly_lms: {len(weekly_lms)}")

grade_rows = []
for sid in range(1, N_STUDENTS + 1):
    arch = ARCHETYPES[student_arch[sid - 1]]
    for w in range(N_WEEKS):
        week_start = SEMESTER_START + timedelta(weeks=w)
        decay      = arch["drift"] * (w / N_WEEKS)
        grade_rows.append({
            "student_id":    sid,
            "week_start":    week_start.isoformat(),
            "avg_grade":     round(float(np.clip(np.random.normal(arch["grade_base"] - decay * 15, 8), 0, 100)), 1),
            "absence_hours": int(np.clip(np.random.normal(40 * (1 - arch["attend_base"] / 100 + decay * 0.1), 1.5), 0, 40)),
        })
weekly_grades = pd.DataFrame(grade_rows)
weekly_grades.to_csv(OUT / "weekly_grades.csv", index=False)
print(f"weekly_grades: {len(weekly_grades)}")

# 7. payments
pay_rows = []
pid = 1
for sid in range(1, N_STUDENTS + 1):
    if students.loc[students["student_id"] == sid, "tuition_form"].values[0] == "грант":
        continue
    arch = student_arch[sid - 1]
    for installment in range(1, 4):
        due    = SEMESTER_START + timedelta(weeks=(installment - 1) * 5)
        amount = round(np.random.choice([350000, 400000, 450000, 500000]), -3)
        delay  = 0
        if arch in ("в_риске", "критичный"):
            delay = int(np.random.choice([0, 7, 14, 30], p=[0.3, 0.3, 0.25, 0.15]))
        elif arch == "работающий":
            delay = int(np.random.choice([0, 7, 14], p=[0.5, 0.3, 0.2]))
        paid_date = (due + timedelta(days=delay)).isoformat() if delay < 30 else None
        pay_rows.append({"id": pid, "student_id": sid, "due_date": due.isoformat(),
                         "paid_date": paid_date, "amount": amount})
        pid += 1
payments = pd.DataFrame(pay_rows)
payments.to_csv(OUT / "payments.csv", index=False)
print(f"payments: {len(payments)}")
print("\n✅ Все файлы сохранены в data/")
