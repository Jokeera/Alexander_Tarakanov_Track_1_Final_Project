"""Stage: FEATURES
Читает clean train/test CSV (после PREPARE) и добавляет инженерные признаки.
Входы:  params.data.train_raw_path / test_raw_path
Выходы: params.data.train_features_path / test_features_path
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def load_params() -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


REQ_BASE_COLS = [
    "limit_bal", "sex", "education", "marriage", "age",
    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
    "target",
]
BILL_COLS = ["bill_amt1","bill_amt2","bill_amt3","bill_amt4","bill_amt5","bill_amt6"]
PAY_AMT_COLS = ["pay_amt1","pay_amt2","pay_amt3","pay_amt4","pay_amt5","pay_amt6"]
PAY_STATUS_COLS = ["pay_0","pay_2","pay_3","pay_4","pay_5","pay_6"]


def _assert_required(df: pd.DataFrame, where: str) -> None:
    missing = [c for c in REQ_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[features] Missing columns in {where}: {missing}")


def create_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Добавляет признаки. НЕ трогает target, не делает лики."""
    out = df.copy()

    # 1) Возраст в категории
    age_bins = params["features"]["age_bins"]
    age_labels = params["features"]["age_labels"]
    out["age_bin"] = pd.cut(out["age"], bins=age_bins,
                            labels=age_labels, include_lowest=True)

    # 2) Utilization last month (bill_amt1 / limit_bal)
    #    Безопасное деление, клип на [0, 2]
    denom = np.where(out["limit_bal"] > 0, out["limit_bal"], np.nan)
    out["utilization_last"] = (out["bill_amt1"] / denom).fillna(0.0).clip(0, 2)

    # 3) Сумма задержек (сколько месяцев с просрочкой > 0)
    out["pay_delay_sum"] = (out[PAY_STATUS_COLS].values > 0).sum(axis=1).astype(np.int16)

    # 4) Максимальная задержка
    out["pay_delay_max"] = out[PAY_STATUS_COLS].max(axis=1)

    # 5) Тренд задолженности за 3 месяца: (bill1 - bill3) / (|bill3| + 1)
    out["bill_trend"] = ((out["bill_amt1"] - out["bill_amt3"]) / (out["bill_amt3"].abs() + 1.0)).clip(-5, 5)

    # 6) Тренд платежей за 3 месяца: (pay1 - pay3) / (pay3 + 1)
    out["pay_trend"] = ((out["pay_amt1"] - out["pay_amt3"]) / (out["pay_amt3"] + 1.0)).clip(-5, 5)

    # 7) Средние значения
    out["bill_avg"] = out[BILL_COLS].mean(axis=1)
    out["pay_amt_avg"] = out[PAY_AMT_COLS].mean(axis=1)

    # 8) Отношение платежей к среднему долгу, клип [0,5]
    out["pay_to_bill_ratio"] = np.where(out["bill_avg"] > 0,
                                        out["pay_amt_avg"] / out["bill_avg"], 0.0)
    out["pay_to_bill_ratio"] = out["pay_to_bill_ratio"].clip(0, 5)

    # Стабильный порядок: базовые → новые признаки → target в конце
    new_cols = [
        "age_bin", "utilization_last", "pay_delay_sum", "pay_delay_max",
        "bill_trend", "pay_trend", "bill_avg", "pay_amt_avg", "pay_to_bill_ratio",
    ]
    base_cols_no_target = [c for c in REQ_BASE_COLS if c != "target" and c in out.columns]
    cols = base_cols_no_target + new_cols + (["target"] if "target" in out.columns else [])
    return out.loc[:, cols]


def main() -> None:
    P = load_params()

    raw_train = P["data"]["train_raw_path"]
    raw_test  = P["data"]["test_raw_path"]
    feat_train = P["data"]["train_features_path"]
    feat_test  = P["data"]["test_features_path"]

    Path(feat_train).parent.mkdir(parents=True, exist_ok=True)

    print("[features] Loading train:", raw_train)
    train_df = pd.read_csv(raw_train)
    print("[features] Loading test :", raw_test)
    test_df = pd.read_csv(raw_test)

    _assert_required(train_df, "train")
    _assert_required(test_df, "test")

    print("[features] Creating features (train)...")
    train_features = create_features(train_df, P)
    print("[features] Creating features (test)...")
    test_features = create_features(test_df, P)

    train_features.to_csv(feat_train, index=False)
    test_features.to_csv(feat_test, index=False)

    print(f"[features] Saved: {feat_train} {train_features.shape}")
    print(f"[features] Saved: {feat_test} {test_features.shape}")
    print("[features] New feature columns:",
          ["age_bin","utilization_last","pay_delay_sum","pay_delay_max",
           "bill_trend","pay_trend","bill_avg","pay_amt_avg","pay_to_bill_ratio"])


if __name__ == "__main__":
    main()
