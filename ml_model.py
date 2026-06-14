"""
ml_model.py — ML-модуль EWS v2.

Улучшения:
  - Dependency injection вместо глобальных переменных (ModelRegistry)
  - Кэширование признаков (FeatureCache) — не пересчитываем для одного студента дважды
  - SHAP-объяснения: почему студент попал в риск
  - Полный train/predict pipeline: XGBoost + RF + GradBoost + LR
  - Ensemble: взвешенная комбинация ML + эвристика
  - /explain команда: топ-5 факторов на русском языке
  - compare_models() для сравнения алгоритмов (диссертация)
  - Все пороги из config.py
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

import config
from config import ML_RISK_THRESHOLD, ML_WEIGHT
from analyzer import compute_risk_score, score_to_level

logger = logging.getLogger(__name__)

MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "ews_model.joblib"
META_PATH  = MODEL_DIR / "ews_meta.json"

# Человекочитаемые имена признаков для SHAP / /explain
FEATURE_LABELS: dict[str, str] = {
    "grade_avg":           "Средний балл",
    "grade_min":           "Минимальный балл",
    "grade_max":           "Максимальный балл",
    "grade_std":           "Нестабильность оценок (σ)",
    "grade_trend":         "Тренд оценок",
    "grade_last2_avg":     "Средний балл (последние 2 нед.)",
    "weeks_below_55":      "Недель с баллом < 55",
    "weeks_below_65":      "Недель с баллом < 65",
    "absence_avg":         "Среднее пропусков ч/нед",
    "absence_total":       "Всего пропущено часов",
    "absence_max":         "Максимум пропусков за неделю",
    "absence_trend":       "Тренд пропусков",
    "weeks_high_absent":   "Недель с высокими пропусками",
    "lms_logins_avg":      "Среднее входов в LMS/нед",
    "lms_logins_min":      "Минимум входов в LMS",
    "lms_logins_trend":    "Тренд активности LMS",
    "lms_late_ratio":      "Доля просроченных сдач",
    "lms_zero_weeks":      "Недель без входа в LMS",
    "lms_silent_exit":     "«Тихий уход» из LMS",
    "payment_overdue_count":  "Количество просроченных платежей",
    "payment_overdue_amount": "Сумма задолженности (тг)",
    "payment_late_avg_days":  "Средняя задержка платежей (дни)",
    "payment_total_count":    "Всего платежей",
    "works":               "Работает параллельно с учёбой",
    "is_nonresident":      "Иногородний студент",
    "rents_housing":       "Снимает жильё",
    "is_paid":             "Платное обучение",
    "entry_score":         "Входной балл ЕНТ/КТА",
    "social_stress":       "Индекс социальной нагрузки",
    "double_hit_weeks":    "Недель «двойного удара»",
    "academic_distress":   "Индекс академического дистресса",
    # Rolling-window признаки
    "grade_roll4_mean":    "Скользящий средний балл (4 нед.)",
    "grade_roll4_std":     "Скользящая нестабильность (4 нед.)",
    "absence_roll4_mean":  "Скользящие пропуски (4 нед.)",
    "lms_roll4_mean":      "Скользящие LMS-логины (4 нед.)",
    "grade_mom":           "Изменение балла нед./нед. (MoM)",
    "absence_mom":         "Изменение пропусков нед./нед. (MoM)",
    "lms_mom":             "Изменение LMS-активности нед./нед.",
}


# ═══════════════════════════════════════════════════════════════
#  Dataclass результатов
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelMetrics:
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    roc_auc:   float
    confusion_matrix: list[list[int]]
    classification_report: str
    cv_f1_mean: float
    cv_f1_std:  float
    n_train: int
    n_test:  int
    n_features: int

    def to_dict(self) -> dict:
        return {
            "accuracy":   round(self.accuracy,  4),
            "precision":  round(self.precision, 4),
            "recall":     round(self.recall,    4),
            "f1":         round(self.f1,        4),
            "roc_auc":    round(self.roc_auc,   4),
            "cv_f1_mean": round(self.cv_f1_mean, 4),
            "cv_f1_std":  round(self.cv_f1_std,  4),
            "n_train":    self.n_train,
            "n_test":     self.n_test,
            "n_features": self.n_features,
            "confusion_matrix": self.confusion_matrix,
        }

    def summary(self) -> str:
        cm = self.confusion_matrix
        return (
            f"{'─'*40}\n"
            f"  Accuracy:  {self.accuracy:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall:    {self.recall:.4f}\n"
            f"  F1-score:  {self.f1:.4f}\n"
            f"  ROC-AUC:   {self.roc_auc:.4f}\n"
            f"  CV F1:     {self.cv_f1_mean:.4f} ± {self.cv_f1_std:.4f}\n"
            f"{'─'*40}\n"
            f"  Confusion matrix:\n"
            f"    TN={cm[0][0]}  FP={cm[0][1]}\n"
            f"    FN={cm[1][0]}  TP={cm[1][1]}\n"
            f"{'─'*40}\n"
            f"  Train: {self.n_train} | Test: {self.n_test} | Features: {self.n_features}\n"
        )


@dataclass
class FeatureImportance:
    features:    list[str]
    importances: list[float]
    model_type:  str

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        pairs = sorted(zip(self.features, self.importances), key=lambda x: -x[1])
        return pairs[:n]

    def summary(self, n: int = 10) -> str:
        lines = [f"Top-{n} признаков ({self.model_type}):"]
        for i, (name, imp) in enumerate(self.top(n), 1):
            label = FEATURE_LABELS.get(name, name)
            bar   = "█" * int(imp * 40)
            lines.append(f"  {i:>2}. {label:<38} {imp:.4f}  {bar}")
        return "\n".join(lines)


@dataclass
class ShapExplanation:
    """Результат SHAP-объяснения для одного студента."""
    student_id:   int
    base_value:   float                      # среднее предсказание модели
    shap_values:  dict[str, float]           # feature → shap value
    top_risk:     list[tuple[str, float]]    # топ факторов, увеличивающих риск
    top_protect:  list[tuple[str, float]]    # топ факторов, снижающих риск

    def telegram_text(self, n: int = 5) -> str:
        """Форматирует объяснение для Telegram (/explain команда)."""
        lines = ["🔍 *Почему такой уровень риска?*\n"]

        if self.top_risk:
            lines.append("*Факторы, повышающие риск:*")
            for name, val in self.top_risk[:n]:
                label = FEATURE_LABELS.get(name, name)
                bar = "▓" * min(int(abs(val) * 20), 10)
                lines.append(f"  🔴 {label}: +{val:.3f}  {bar}")

        if self.top_protect:
            lines.append("\n*Факторы, снижающие риск:*")
            for name, val in self.top_protect[:n]:
                label = FEATURE_LABELS.get(name, name)
                bar = "░" * min(int(abs(val) * 20), 10)
                lines.append(f"  🟢 {label}: {val:.3f}  {bar}")

        lines.append(f"\n_Базовый уровень модели: {self.base_value:.3f}_")
        return "\n".join(lines)


@dataclass
class PredictionResult:
    student_id:           int
    risk_probability:     float
    risk_score_ml:        float
    risk_level_ml:        str
    risk_score_heuristic: float
    risk_level_heuristic: str
    features_used:        dict = field(default_factory=dict)
    shap_explanation:     Optional[ShapExplanation] = None

    def combined_score(self, ml_weight: float = ML_WEIGHT) -> float:
        h_weight = 1.0 - ml_weight
        return round(
            self.risk_score_ml * ml_weight + self.risk_score_heuristic * h_weight, 2
        )

    def combined_level(self, ml_weight: float = ML_WEIGHT) -> str:
        return score_to_level(self.combined_score(ml_weight))


# ═══════════════════════════════════════════════════════════════
#  FeatureCache — кэширование признаков (DI-friendly)
# ═══════════════════════════════════════════════════════════════

class FeatureCache:
    """
    Простой in-memory кэш признаков.
    Ключ: (student_id, week_start). Позволяет не пересчитывать признаки
    при вызове predict + explain для одного студента.
    """
    def __init__(self) -> None:
        self._cache: dict[tuple, dict] = {}

    def get(self, student_id: int, week_start: str) -> Optional[dict]:
        return self._cache.get((student_id, week_start))

    def set(self, student_id: int, week_start: str, features: dict) -> None:
        self._cache[(student_id, week_start)] = features

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# ═══════════════════════════════════════════════════════════════
#  ModelRegistry — DI-контейнер для pipeline + meta
# ═══════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Хранит загруженный pipeline и метаданные.
    Передаётся явно в функции (dependency injection) — никаких глобальных переменных.

    Использование:
        registry = ModelRegistry()
        registry.load()           # загружает из models/
        result = predict_student(registry, student, grades, lms, pays)
    """
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        meta_path:  Path = META_PATH,
        cache:      Optional[FeatureCache] = None,
    ) -> None:
        self.model_path    = model_path
        self.meta_path     = meta_path
        self.pipeline:     Optional[ImbPipeline] = None
        self.feature_names: list[str] = []
        self.meta:         dict = {}
        self.cache         = cache or FeatureCache()
        self._loaded       = False

    def load(self) -> "ModelRegistry":
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {self.model_path}. "
                "Запустите: python ml_model.py train"
            )
        self.pipeline      = joblib.load(self.model_path)
        self.meta          = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.feature_names = self.meta["feature_names"]
        self._loaded       = True
        logger.info(
            "ModelRegistry: модель загружена (%s, %d признаков)",
            self.meta.get("model_type", "?"), len(self.feature_names),
        )
        return self

    def is_loaded(self) -> bool:
        return self._loaded

    def save(self, metrics: ModelMetrics, model_type: str) -> None:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        meta = {
            "model_type":    model_type,
            "feature_names": self.feature_names,
            "metrics":       metrics.to_dict(),
        }
        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("ModelRegistry: модель сохранена → %s", self.model_path)


# ═══════════════════════════════════════════════════════════════
#  Инженерия признаков
# ═══════════════════════════════════════════════════════════════

def _safe_mean(lst: list, key: str, default: float = 0.0) -> float:
    vals = [r.get(key) for r in lst if r.get(key) is not None]
    return float(np.mean(vals)) if vals else default


def _safe_sum(lst: list, key: str) -> float:
    return float(sum(r.get(key) or 0 for r in lst))


def _trend_slope(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(vals) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def extract_features(
    student:  dict,
    grades:   list[dict],
    lms:      list[dict],
    payments: list[dict],
) -> dict:
    """
    Формирует вектор из 37 числовых признаков на одного студента.
    Включает базовые (30) + rolling-window + MoM-изменения (7).
    """
    f: dict = {}

    grade_vals   = [g.get("avg_grade") or 0.0 for g in grades]
    absence_vals = [g.get("absence_hours") or 0.0 for g in grades]
    login_vals   = [lo.get("logins") or 0 for lo in lms]

    # ── 1. Академические признаки ─────────────────────────────
    f["grade_avg"]       = _safe_mean(grades, "avg_grade", 0.0)
    f["grade_min"]       = float(min(grade_vals)) if grade_vals else 0.0
    f["grade_max"]       = float(max(grade_vals)) if grade_vals else 0.0
    f["grade_std"]       = float(np.std(grade_vals)) if len(grade_vals) >= 2 else 0.0
    f["grade_trend"]     = _trend_slope(grade_vals)
    f["grade_last2_avg"] = float(np.mean(grade_vals[-2:])) if len(grade_vals) >= 2 else f["grade_avg"]
    f["weeks_below_55"]  = sum(1 for v in grade_vals if v < 55)
    f["weeks_below_65"]  = sum(1 for v in grade_vals if v < 65)

    # ── 2. Посещаемость ───────────────────────────────────────
    f["absence_avg"]       = _safe_mean(grades, "absence_hours", 0.0)
    f["absence_total"]     = _safe_sum(grades, "absence_hours")
    f["absence_max"]       = float(max(absence_vals)) if absence_vals else 0.0
    f["absence_trend"]     = _trend_slope(absence_vals)
    f["weeks_high_absent"] = sum(1 for v in absence_vals if v > 8)

    # ── 3. LMS-активность ─────────────────────────────────────
    on_time   = _safe_sum(lms, "submissions_on_time")
    late_subs = _safe_sum(lms, "submissions_late")
    total_subs = on_time + late_subs

    f["lms_logins_avg"]   = _safe_mean(lms, "logins", 0.0)
    f["lms_logins_min"]   = float(min(login_vals)) if login_vals else 0.0
    f["lms_logins_trend"] = _trend_slope([float(v) for v in login_vals])
    f["lms_late_ratio"]   = late_subs / total_subs if total_subs > 0 else 0.0
    f["lms_zero_weeks"]   = sum(1 for v in login_vals if v == 0)
    f["lms_silent_exit"]  = float(
        len(login_vals) >= 3 and all(v == 0 for v in login_vals[-3:])
    )

    # ── 4. Платежи ────────────────────────────────────────────
    overdue   = [p for p in payments if p.get("paid_date") is None]
    paid_late = []
    for p in payments:
        if p.get("paid_date") and p.get("due_date"):
            try:
                from datetime import date as _date
                delay = (
                    _date.fromisoformat(p["paid_date"]) -
                    _date.fromisoformat(p["due_date"])
                ).days
                if delay > 0:
                    paid_late.append(delay)
            except Exception:
                pass

    f["payment_overdue_count"]  = float(len(overdue))
    f["payment_overdue_amount"] = float(sum(p.get("amount", 0) for p in overdue))
    f["payment_late_avg_days"]  = float(np.mean(paid_late)) if paid_late else 0.0
    f["payment_total_count"]    = float(len(payments))

    # ── 5. Профиль студента ───────────────────────────────────
    f["works"]          = float(bool(student.get("works")))
    f["is_nonresident"] = float(not bool(student.get("resident", True)))
    f["rents_housing"]  = float(student.get("accommodation_type") == "съём")
    f["is_paid"]        = float(student.get("tuition_form") == "платное")
    f["entry_score"]    = float(student.get("entry_score") or 70)
    f["social_stress"]  = f["works"] + f["is_nonresident"] + f["rents_housing"]

    # ── 6. Составные признаки ─────────────────────────────────
    f["double_hit_weeks"] = float(sum(
        1 for g in grades
        if (g.get("avg_grade") or 100) < 60 and (g.get("absence_hours") or 0) > 6
    )) if grades else 0.0

    f["academic_distress"] = (
        f["weeks_below_55"] * 2 +
        f["weeks_high_absent"] +
        f["lms_zero_weeks"]
    )

    # ── 7. Rolling-window признаки (последние 4 недели) ───────
    n = 4
    grade_r4   = grade_vals[-n:]   if len(grade_vals)   >= n else grade_vals
    absence_r4 = absence_vals[-n:] if len(absence_vals) >= n else absence_vals
    login_r4   = login_vals[-n:]   if len(login_vals)   >= n else login_vals

    f["grade_roll4_mean"]   = float(np.mean(grade_r4))   if grade_r4   else 0.0
    f["grade_roll4_std"]    = float(np.std(grade_r4))    if len(grade_r4) >= 2 else 0.0
    f["absence_roll4_mean"] = float(np.mean(absence_r4)) if absence_r4 else 0.0
    f["lms_roll4_mean"]     = float(np.mean(login_r4))   if login_r4   else 0.0

    # ── 8. Month-over-Month изменения ─────────────────────────
    # Сравниваем последние 2 недели с предыдущими 2 неделями
    def _mom(vals: list) -> float:
        if len(vals) < 4:
            return 0.0
        prev   = float(np.mean(vals[-4:-2]))
        recent = float(np.mean(vals[-2:]))
        return recent - prev

    f["grade_mom"]   = _mom(grade_vals)
    f["absence_mom"] = _mom(absence_vals)
    f["lms_mom"]     = _mom([float(v) for v in login_vals])

    return f


# ═══════════════════════════════════════════════════════════════
#  SHAP-объяснения
# ═══════════════════════════════════════════════════════════════

def explain_student(
    registry: ModelRegistry,
    student:  dict,
    grades:   list[dict],
    lms:      list[dict],
    payments: list[dict],
) -> Optional[ShapExplanation]:
    """
    Вычисляет SHAP-объяснение для одного студента.
    Возвращает None если shap не установлен или модель не поддерживает SHAP.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap не установлен. pip install shap")
        return None

    if not registry.is_loaded():
        logger.warning("explain_student: модель не загружена")
        return None

    sid = student.get("student_id", 0)

    # Берём из кэша или вычисляем
    features = registry.cache.get(sid, "explain")
    if features is None:
        features = extract_features(student, grades, lms, payments)
        registry.cache.set(sid, "explain", features)

    feature_names = registry.feature_names
    X_row = pd.DataFrame([{fn: features.get(fn, 0.0) for fn in feature_names}])

    clf = registry.pipeline.named_steps["clf"]

    try:
        # TreeExplainer работает с XGBoost/RF/GBT
        explainer   = shap.TreeExplainer(clf)
        # Трансформируем через scaler (без clf)
        scaler      = registry.pipeline.named_steps.get("scaler")
        X_scaled    = scaler.transform(X_row) if scaler else X_row.values
        shap_vals   = explainer.shap_values(X_scaled)

        # Для бинарной классификации shap_values может быть списком [class0, class1]
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]   # класс «риск»
        else:
            sv = shap_vals[0]

        base_value = float(explainer.expected_value[1]
                           if isinstance(explainer.expected_value, (list, np.ndarray))
                           else explainer.expected_value)

        shap_dict = {fn: float(sv[i]) for i, fn in enumerate(feature_names)}

        sorted_shap = sorted(shap_dict.items(), key=lambda x: -x[1])
        top_risk    = [(n, v) for n, v in sorted_shap if v > 0][:5]
        top_protect = [(n, v) for n, v in sorted_shap if v < 0][:5]

        return ShapExplanation(
            student_id=sid,
            base_value=base_value,
            shap_values=shap_dict,
            top_risk=top_risk,
            top_protect=top_protect,
        )
    except Exception as exc:
        logger.error("SHAP explain failed for student %s: %s", sid, exc)
        return None


# ═══════════════════════════════════════════════════════════════
#  Построение датасета
# ═══════════════════════════════════════════════════════════════

def build_dataset(
    week_start: Optional[str] = None,
    risk_threshold: float = ML_RISK_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Собирает датасет из БД по всем студентам.
    y: 1 = «группа риска», 0 = «норма».
    Метки формируются через эвристический скор из analyzer.py.
    """
    import db

    students = db.fetch_all_students()
    logger.info("[ML] Загружено студентов: %d", len(students))

    rows:   list[dict] = []
    labels: list[int]  = []

    for student in students:
        sid = student["student_id"]
        try:
            if week_start:
                grades   = db.fetch_grades_by_week(sid, week_start)
                lms_data = db.fetch_lms_by_week(sid, week_start)
            else:
                grades   = db.fetch_grades(sid)
                lms_data = db.fetch_lms(sid)
            payments = db.fetch_payments(sid)

            if not grades and not lms_data:
                continue

            features = extract_features(student, grades, lms_data, payments)
            rows.append(features)

            risk_score, _, _ = compute_risk_score(grades, lms_data, payments, student)
            labels.append(1 if risk_score >= risk_threshold else 0)

        except Exception as exc:
            logger.warning("  Ошибка студента #%s: %s", sid, exc)

    if not rows:
        raise ValueError("Датасет пуст — нет данных в БД.")

    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(labels, name="target")

    n_risk   = int(y.sum())
    n_normal = len(y) - n_risk
    logger.info(
        "[ML] Датасет: %d записей | Норма: %d (%.1f%%) | Риск: %d (%.1f%%)",
        len(X), n_normal, n_normal/len(y)*100, n_risk, n_risk/len(y)*100,
    )
    return X, y, list(X.columns)


# ═══════════════════════════════════════════════════════════════
#  Обучение модели
# ═══════════════════════════════════════════════════════════════

def _build_pipeline(model_type: str = "xgboost") -> ImbPipeline:
    if model_type == "xgboost":
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
    elif model_type == "logistic":
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    else:
        raise ValueError(f"Неизвестный model_type: {model_type}")

    return ImbPipeline([
        ("smote",  SMOTETomek(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    test_size:  float = 0.2,
    cv_folds:   int   = 5,
) -> tuple[ImbPipeline, ModelMetrics]:
    feature_names = list(X.columns)
    n_features    = len(feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info(
        "[ML] Обучение %s | Train: %d | Test: %d | Признаков: %d",
        model_type, len(X_train), len(X_test), n_features,
    )

    pipeline = _build_pipeline(model_type)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
    )
    logger.info("[ML] CV F1: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    pipeline.fit(X_train, y_train)

    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = ModelMetrics(
        accuracy   = accuracy_score(y_test, y_pred),
        precision  = precision_score(y_test, y_pred, zero_division=0),
        recall     = recall_score(y_test, y_pred, zero_division=0),
        f1         = f1_score(y_test, y_pred, zero_division=0),
        roc_auc    = roc_auc_score(y_test, y_pred_prob),
        confusion_matrix      = cm,
        classification_report = classification_report(y_test, y_pred, target_names=["Норма", "Риск"]),
        cv_f1_mean = float(cv_scores.mean()),
        cv_f1_std  = float(cv_scores.std()),
        n_train    = len(X_train),
        n_test     = len(X_test),
        n_features = n_features,
    )

    logger.info("[ML]\n%s", metrics.summary())
    return pipeline, metrics


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, ModelMetrics]:
    """Сравнивает все 4 алгоритма — для раздела диссертации."""
    results: dict[str, ModelMetrics] = {}
    for mtype in ["xgboost", "random_forest", "gradient_boosting", "logistic"]:
        logger.info("\n[ML] ── Обучение: %s ──", mtype)
        try:
            _, metrics = train_model(X, y, model_type=mtype)
            results[mtype] = metrics
        except Exception as exc:
            logger.error("Ошибка %s: %s", mtype, exc)
    return results


# ═══════════════════════════════════════════════════════════════
#  Feature Importance
# ═══════════════════════════════════════════════════════════════

def get_feature_importance(
    pipeline:      ImbPipeline,
    feature_names: list[str],
) -> FeatureImportance:
    clf        = pipeline.named_steps["clf"]
    model_type = type(clf).__name__

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_.tolist()
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0]).tolist()
    else:
        importances = [0.0] * len(feature_names)

    return FeatureImportance(
        features    = feature_names,
        importances = importances,
        model_type  = model_type,
    )


# ═══════════════════════════════════════════════════════════════
#  Предсказание (использует ModelRegistry + FeatureCache)
# ═══════════════════════════════════════════════════════════════

def predict_student(
    registry: ModelRegistry,
    student:  dict,
    grades:   list[dict],
    lms:      list[dict],
    payments: list[dict],
    with_shap: bool = False,
) -> PredictionResult:
    """
    ML-предсказание + эвристика для одного студента.
    Кэшируются признаки: повторный вызов с тем же student_id не пересчитывает фичи.
    """
    sid        = student.get("student_id", 0)
    week_key   = "predict"

    features = registry.cache.get(sid, week_key)
    if features is None:
        features = extract_features(student, grades, lms, payments)
        registry.cache.set(sid, week_key, features)

    feature_names = registry.feature_names
    X_row = pd.DataFrame([{fn: features.get(fn, 0.0) for fn in feature_names}])

    prob      = float(registry.pipeline.predict_proba(X_row)[0][1])
    score_ml  = round(prob * 10, 2)
    level_ml  = score_to_level(score_ml)

    h_score, _, _ = compute_risk_score(grades, lms, payments, student)
    h_level       = score_to_level(h_score)

    shap_exp: Optional[ShapExplanation] = None
    if with_shap:
        shap_exp = explain_student(registry, student, grades, lms, payments)

    return PredictionResult(
        student_id           = sid,
        risk_probability     = round(prob, 4),
        risk_score_ml        = score_ml,
        risk_level_ml        = level_ml,
        risk_score_heuristic = h_score,
        risk_level_heuristic = h_level,
        features_used        = features,
        shap_explanation     = shap_exp,
    )


# ═══════════════════════════════════════════════════════════════
#  Сохранение / загрузка (обёртки над ModelRegistry)
# ═══════════════════════════════════════════════════════════════

def save_model(
    pipeline:      ImbPipeline,
    metrics:       ModelMetrics,
    feature_names: list[str],
    model_type:    str,
) -> ModelRegistry:
    """Сохраняет модель и возвращает заполненный ModelRegistry."""
    registry = ModelRegistry()
    registry.pipeline      = pipeline
    registry.feature_names = feature_names
    registry.meta          = {"model_type": model_type, "feature_names": feature_names}
    registry._loaded       = True
    registry.save(metrics, model_type)
    return registry


def load_model() -> tuple[ImbPipeline, list[str], dict]:
    """Обратная совместимость с v1 API."""
    registry = ModelRegistry().load()
    return registry.pipeline, registry.feature_names, registry.meta


# ═══════════════════════════════════════════════════════════════
#  CLI: python ml_model.py [train|predict|compare|explain]
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import logging as _logging
    _logging.basicConfig(
        format=config.LOG_FORMAT,
        level=getattr(_logging, config.LOG_LEVEL, _logging.INFO),
    )

    import db as _db

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "train":
        model_type = sys.argv[2] if len(sys.argv) > 2 else "xgboost"
        print(f"\n=== EWS ML: обучение ({model_type}) ===\n")
        X, y, feature_names = build_dataset()
        pipeline, metrics   = train_model(X, y, model_type=model_type)
        registry = save_model(pipeline, metrics, feature_names, model_type)
        fi = get_feature_importance(pipeline, feature_names)
        print("\n" + fi.summary())
        print("\n" + metrics.classification_report)

    elif cmd == "compare":
        print("\n=== EWS ML: сравнение алгоритмов ===\n")
        X, y, _ = build_dataset()
        results = compare_models(X, y)
        print(f"\n{'Модель':<25} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>8} {'ROC-AUC':>9}")
        print("─" * 65)
        for mtype, m in results.items():
            print(
                f"{mtype:<25} {m.accuracy:>9.4f} {m.precision:>9.4f} "
                f"{m.recall:>9.4f} {m.f1:>8.4f} {m.roc_auc:>9.4f}"
            )

    elif cmd == "predict":
        print("\n=== EWS ML: предсказание (первые 5 студентов) ===\n")
        registry = ModelRegistry().load()
        students = _db.fetch_all_students()[:5]
        for student in students:
            sid    = student["student_id"]
            grades = _db.fetch_grades(sid)
            lms    = _db.fetch_lms(sid)
            pays   = _db.fetch_payments(sid)
            result = predict_student(registry, student, grades, lms, pays)
            combined = result.combined_score()
            print(
                f"  #{sid} {student.get('full_name','?'):<30} "
                f"ML: {result.risk_score_ml:.1f}/10 [{result.risk_level_ml:^7}] | "
                f"Эвр: {result.risk_score_heuristic:.1f}/10 [{result.risk_level_heuristic:^7}] | "
                f"Итого: {combined:.1f}/10 [{result.combined_level():^7}]"
            )

    elif cmd == "explain":
        sid = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        print(f"\n=== EWS ML: SHAP объяснение для студента #{sid} ===\n")
        registry = ModelRegistry().load()
        student  = _db.fetch_student(sid)
        if not student:
            print(f"Студент #{sid} не найден.")
            sys.exit(1)
        grades = _db.fetch_grades(sid)
        lms    = _db.fetch_lms(sid)
        pays   = _db.fetch_payments(sid)
        result = predict_student(registry, student, grades, lms, pays, with_shap=True)
        print(f"Комбинированный скор: {result.combined_score():.2f}/10 [{result.combined_level()}]")
        if result.shap_explanation:
            print(result.shap_explanation.telegram_text())
        else:
            print("SHAP недоступен (установите: pip install shap)")

    else:
        print(f"Неизвестная команда '{cmd}'. Доступно: train, compare, predict, explain")
        sys.exit(1)