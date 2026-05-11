"""
upload_to_supabase.py — загружает CSV-файлы из data/ в Supabase
Порядок соблюдён с учётом FK-зависимостей.

Использование:
    python upload_to_supabase.py
Или с переменными окружения:
    SUPABASE_URL=... SUPABASE_KEY=... python upload_to_supabase.py
"""
import os
import sys
import pandas as pd
from pathlib import Path
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Укажите SUPABASE_URL и SUPABASE_KEY в config.py или переменных окружения.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
DATA  = Path("data")
CHUNK = 500


def upsert_table(table: str, csv_file: str, conflict_col: str = None):
    path = DATA / csv_file
    if not path.exists():
        print(f"⚠️  {csv_file} не найден, пропускаю.")
        return

    df = pd.read_csv(path)
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    total   = len(records)

    for i in range(0, total, CHUNK):
        chunk = records[i: i + CHUNK]
        try:
            if conflict_col:
                supabase.table(table).upsert(chunk, on_conflict=conflict_col).execute()
            else:
                supabase.table(table).upsert(chunk).execute()
            print(f"  {table}: {min(i + CHUNK, total)}/{total}")
        except Exception as e:
            print(f"  ❌ Ошибка при загрузке {table} (строки {i}-{i+CHUNK}): {e}")
            raise

    print(f"✅  {table} — готово ({total} строк)\n")


print("=== Загрузка данных в Supabase ===\n")

upsert_table("groups",         "groups.csv",         "group_id")
upsert_table("curators",       "curators.csv",        "curator_id")
upsert_table("curator_groups", "curator_groups.csv")
upsert_table("students",       "students.csv",        "student_id")
upsert_table("weekly_lms",     "weekly_lms.csv")
upsert_table("weekly_grades",  "weekly_grades.csv")
upsert_table("payments",       "payments.csv",        "id")

print("🎉  Все данные успешно загружены!")
