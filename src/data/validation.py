"""Data validation with Great Expectations (fallback to pandas checks).

Если GE установлен (есть в requirements), используем совместимый API PandasDataset.
Если что-то не так или GE недоступен, делаем строгие проверки pandas и бросаем AssertionError.
"""

from typing import Optional, List
import pandas as pd

# Пытаемся использовать Great Expectations, но не привязываемся к DataContext
try:
    from great_expectations.dataset import PandasDataset  # совместимый слой в GX 0.18.x (deprecated, но удобный)
    _HAS_GE = True
except Exception:
    _HAS_GE = False


def _ge_validate(df: pd.DataFrame) -> None:
    """Валидация с Great Expectations (через PandasDataset). Бросает AssertionError при провале."""
    ds = PandasDataset(df.copy())

    required_cols = [
        "limit_bal", "sex", "education", "marriage", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
        "target",
    ]
    ds.expect_table_columns_to_match_set(required_cols)

    # Не должно быть пропусков в ключевых колонках
    for c in required_cols:
        ds.expect_column_values_to_not_be_null(c)

    # Категориальные / дискретные
    ds.expect_column_values_to_be_in_set("sex", [1, 2])
    ds.expect_column_values_to_be_in_set("education", [1, 2, 3, 4])
    ds.expect_column_values_to_be_in_set("marriage", [1, 2, 3])
    ds.expect_column_values_to_be_in_set("target", [0, 1])

    # Диапазоны
    ds.expect_column_values_to_be_between("age", min_value=18, max_value=100)
    for c in ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]:
        ds.expect_column_values_to_be_between(c, min_value=-2, max_value=9)

    # Деньги/счета — допускаем >= 0 (в датасете возможны нули)
    money_like = ["limit_bal"] + [f"bill_amt{i}" for i in range(1, 7)] + [f"pay_amt{i}" for i in range(1, 7)]
    for c in money_like:
        ds.expect_column_values_to_be_between(c, min_value=0, max_value=None, allow_cross_type_comparisons=True)

    # Сводка и жёсткая остановка при провалах
    res = ds.validate(result_format="SUMMARY")
    if not res["success"]:
        # Собираем короткий отчёт о провалившихся ожиданиях
        failed = []
        for r in res["results"]:
            if r.get("success") is False:
                failed.append(r.get("expectation_config", {}).get("expectation_type", "unknown_expectation"))
        raise AssertionError(f"Data validation failed (Great Expectations). Failed expectations: {sorted(set(failed))}")


def _pandas_validate(df: pd.DataFrame) -> None:
    """Запасной строгий валидатор на pandas, если GE недоступен. Бросает AssertionError при провале."""
    required_cols = {
        "limit_bal", "sex", "education", "marriage", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
        "target",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"

    assert df[required_cols].isna().sum().sum() == 0, "There are missing values in required columns"

    assert set(df["sex"].unique()).issubset({1, 2}), "Unexpected values in 'sex'"
    assert set(df["education"].unique()).issubset({1, 2, 3, 4}), "Unexpected values in 'education'"
    assert set(df["marriage"].unique()).issubset({1, 2, 3}), "Unexpected values in 'marriage'"
    assert set(df["target"].unique()).issubset({0, 1}), "Unexpected values in 'target'"

    assert (df["age"].between(18, 100)).all(), "Age out of range [18,100] detected"

    for c in ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]:
        assert (df[c].between(-2, 9)).all(), f"{c} out of range [-2,9] detected"

    money_like = ["limit_bal"] + [f"bill_amt{i}" for i in range(1, 7)] + [f"pay_amt{i}" for i in range(1, 7)]
    for c in money_like:
        assert (df[c] >= 0).all(), f"{c} has negative values"


def validate_dataframe(df: pd.DataFrame, dataset_name: str = "dataset") -> None:
    """
    Главная функция, которую вызывает пайплайн.
    Если GE доступен — валидируем через GE. Иначе — строгие проверки pandas.
    """
    if _HAS_GE:
        _ge_validate(df)
        print(f"[validation] Great Expectations passed for: {dataset_name}")
    else:
        _pandas_validate(df)
        print(f"[validation] Fallback pandas validation passed for: {dataset_name}")
