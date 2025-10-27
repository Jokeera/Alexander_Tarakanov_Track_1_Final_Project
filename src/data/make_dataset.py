"""Data preparation script for credit scoring dataset (strict & reproducible)"""

from pathlib import Path
import json

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# --- Expected raw schema (до приведения к snake_case/переименований) ---
EXPECTED_RAW_COLS = {
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    # в сырье бывает PAY_1 вместо PAY_0
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default.payment.next.month",
}
# Допускаем, что в сырье есть либо PAY_0, либо PAY_1
ALT_COL_A = "PAY_0"
ALT_COL_B = "PAY_1"


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def assert_required_columns_raw(df_raw: pd.DataFrame):
    """Жёсткая проверка схемы сырого датасета (до любых переименований)."""
    cols = set(df_raw.columns)

    # Проверяем вариант с PAY_0/ PAY_1
    has_alt = (ALT_COL_A in cols) or (ALT_COL_B in cols)

    # Все остальные обязательные колонки должны присутствовать
    required_wo_alts = EXPECTED_RAW_COLS - {ALT_COL_A, ALT_COL_B}
    missing = sorted(required_wo_alts - cols)

    if not has_alt or missing:
        raise ValueError(
            "Raw data columns mismatch. "
            f"Missing (without PAY_0/PAY_1 alternative): {missing}; "
            f"found PAY_0={ALT_COL_A in cols}, PAY_1={ALT_COL_B in cols}"
        )


def clean_column_names(df):
    """Convert column names to snake_case (нижний регистр и точки -> подчёркивание)."""
    df.columns = df.columns.str.lower().str.replace(".", "_", regex=False)
    return df


def clean_data(df):
    """Clean and preprocess raw data (после snake_case)."""
    df = df.copy()

    # Target -> 'target'
    if "default_payment_next_month" in df.columns:
        df = df.rename(columns={"default_payment_next_month": "target"})

    # Обрабатываем PAY_0 vs PAY_1
    if "pay_1" in df.columns and "pay_0" not in df.columns:
        df = df.rename(columns={"pay_1": "pay_0"})

    # EDUCATION: 0,5,6 -> 4 (другое)
    if "education" in df.columns:
        df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: 0 -> 3 (другое)
    if "marriage" in df.columns:
        df["marriage"] = df["marriage"].replace({0: 3})

    # SEX: только 1,2
    if "sex" in df.columns:
        df = df[df["sex"].isin([1, 2])].copy()

    # AGE: [18, 100]
    if "age" in df.columns:
        df = df[(df["age"] >= 18) & (df["age"] <= 100)].copy()

    # PAY_*: клип к [-2, 9]
    pay_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
    for col in pay_cols:
        if col in df.columns:
            df[col] = df[col].clip(-2, 9)

    # Убираем ID, если есть
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Явное приведение типов для стабильности пайплайна."""
    df = df.copy()

    # Целочисленные небольшие признаки
    int_small = ["sex", "education", "marriage", "age"] + [
        c for c in ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"] if c in df.columns
    ]
    for c in int_small:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise").astype("int16")

    # Денежные/непрерывные
    money_like = ["limit_bal"] + \
                 [f"bill_amt{i}" for i in range(1, 7) if f"bill_amt{i}" in df.columns] + \
                 [f"pay_amt{i}" for i in range(1, 7) if f"pay_amt{i}" in df.columns]
    for c in money_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")

    # Целевая
    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="raise").astype("int8")

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Фиксированный порядок колонок (target в конце)."""
    base = [
        "limit_bal", "sex", "education", "marriage", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
    ]
    cols = [c for c in base if c in df.columns]
    if "target" in df.columns:
        cols = cols + ["target"]
    return df.loc[:, cols]


def main():
    """Main function to prepare dataset"""
    # 1) Параметры
    params = load_params()

    # 2) Каталоги
    Path(params["data"]["processed_path"]).mkdir(parents=True, exist_ok=True)

    # 3) Читаем сырьё и проверяем схему (до переименований)
    print("Loading raw data...")
    df_raw = pd.read_csv(params["data"]["raw_path"])
    print(f"Raw data shape: {df_raw.shape}")
    assert_required_columns_raw(df_raw)

    # 4) Приводим имена к snake_case, чистим данные
    df = clean_column_names(df_raw)
    print("Cleaning data...")
    df = clean_data(df)

    # 5) Явные типы и стабильный порядок колонок
    df = cast_dtypes(df)
    df = reorder_columns(df)
    print(f"Cleaned data shape: {df.shape}")

    # 6) Трейн/тест
    print("Splitting data...")
    train_df, test_df = train_test_split(
        df,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=df["target"],
    )

    # 7) Сохраняем
    train_path = params["data"]["train_path"]
    test_path = params["data"]["test_path"]
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 8) Мини-отчёт для прозрачности и CI-логов
    report = {
        "raw_shape": list(df_raw.shape),
        "clean_shape": list(df.shape),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "train_target_balance": {
            str(k): float(v)
            for k, v in train_df["target"].value_counts(normalize=True).round(6).to_dict().items()
        },
    }
    with open("data/processed/prepare_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Train data saved to {train_path}: {train_df.shape}")
    print(f"Test data saved to {test_path}: {test_df.shape}")
    print("Target distribution in train:")
    print(train_df["target"].value_counts(normalize=True))

    # 9) (Опционально) запустить GE-валидацию, если включена в params.yaml
    try:
        from src.data.validation import validate_dataframe  # твоя функция (Great Expectations/Pandera)
        if params.get("validation", {}).get("enable", False):
            validate_dataframe(train_df, dataset_name="train_clean")
            validate_dataframe(test_df, dataset_name="test_clean")
            print("Great Expectations validation passed.")
    except Exception as e:
        # Не роняем стадию prepare, чтобы пайплайн оставался воспроизводимым
        print(f"[make_dataset] GE validation skipped or failed softly: {e}")


if __name__ == "__main__":
    main()
