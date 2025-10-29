"""
Stage: PREPARE
Сырой CSV -> строгая очистка -> data/processed/train.csv, test.csv
(без feature engineering)
"""
from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

EXPECTED_RAW_COLS = {
    "ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
    "PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",
    "default.payment.next.month",
}
ALT_COL_A = "PAY_0"   # часто встречается
ALT_COL_B = "PAY_1"   # альтернативное имя

def load_params() -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def assert_required_columns_raw(df: pd.DataFrame):
    cols = set(df.columns)
    has_alt = (ALT_COL_A in cols) or (ALT_COL_B in cols)
    missing = sorted((EXPECTED_RAW_COLS - {ALT_COL_A, ALT_COL_B}) - cols)
    if missing or not has_alt:
        raise ValueError(
            f"Missing columns: {missing}; PAY_0 exists={ALT_COL_A in cols}; PAY_1 exists={ALT_COL_B in cols}"
        )

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.lower().str.replace(".", "_", regex=False)
    return out

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # target rename
    if "default_payment_next_month" in out.columns:
        out = out.rename(columns={"default_payment_next_month": "target"})

    # PAY_1 -> PAY_0 (если нужно)
    if "pay_1" in out.columns and "pay_0" not in out.columns:
        out = out.rename(columns={"pay_1": "pay_0"})

    # нормализация категорий
    if "education" in out.columns:
        out["education"] = out["education"].replace({0: 4, 5: 4, 6: 4})
    if "marriage" in out.columns:
        out["marriage"] = out["marriage"].replace({0: 3})

    # фильтры по здравому смыслу
    if "sex" in out.columns:
        out = out[out["sex"].isin([1, 2])]
    if "age" in out.columns:
        out = out[(out["age"] >= 18) & (out["age"] <= 100)]

    # клип статусов просрочек
    for c in ["pay_0","pay_2","pay_3","pay_4","pay_5","pay_6"]:
        if c in out.columns:
            out[c] = out[c].clip(-2, 9)

    # drop ID
    if "id" in out.columns:
        out = out.drop(columns=["id"])

    if out.empty:
        raise ValueError("After cleaning the dataframe is empty — check filters.")
    if "target" not in out.columns:
        raise ValueError("Column 'target' is missing after renaming.")

    return out

def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["sex","education","marriage","age","pay_0","pay_2","pay_3","pay_4","pay_5","pay_6"]:
        if c in out.columns:
            out[c] = out[c].astype("int16")
    for c in ["limit_bal", *[f"bill_amt{i}" for i in range(1,7)], *[f"pay_amt{i}" for i in range(1,7)]]:
        if c in out.columns:
            out[c] = out[c].astype("float64")
    if "target" in out.columns:
        out["target"] = out["target"].astype("int8")
    return out

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = [
        "limit_bal","sex","education","marriage","age",
        "pay_0","pay_2","pay_3","pay_4","pay_5","pay_6",
        "bill_amt1","bill_amt2","bill_amt3","bill_amt4","bill_amt5","bill_amt6",
        "pay_amt1","pay_amt2","pay_amt3","pay_amt4","pay_amt5","pay_amt6",
    ]
    cols = [c for c in base if c in df.columns]
    if "target" in df.columns:
        cols.append("target")
    return df.loc[:, cols]

def main():
    params = load_params()
    raw_path     = params["data"]["raw_path"]
    train_path   = params["data"]["train_raw_path"]
    test_path    = params["data"]["test_raw_path"]
    processed_dir = Path(params["data"]["processed_path"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data…")
    df_raw = pd.read_csv(raw_path)
    assert_required_columns_raw(df_raw)

    print("Cleaning…")
    df = clean_column_names(df_raw)
    df = clean_data(df)
    df = cast_dtypes(df)
    df = reorder_columns(df)
    print("Cleaned shape:", df.shape)

    print("Splitting…")
    train_df, test_df = train_test_split(
        df,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=df["target"],
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"✅ Saved: {train_path} {train_df.shape}")
    print(f"✅ Saved: {test_path} {test_df.shape}")
    print("Train target balance:", train_df["target"].value_counts(normalize=True).to_dict())

if __name__ == "__main__":
    main()
