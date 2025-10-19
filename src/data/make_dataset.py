"""Data preparation script for credit scoring dataset"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def clean_column_names(df):
    """Convert column names to snake_case"""
    df.columns = df.columns.str.lower().str.replace(".", "_", regex=False)
    return df


def clean_data(df):
    """Clean and preprocess raw data"""
    df = df.copy()
    
    # Rename target variable
    if "default_payment_next_month" in df.columns:
        df = df.rename(columns={"default_payment_next_month": "target"})
    
    # Handle PAY_0 vs PAY_1 naming inconsistency
    if "pay_1" in df.columns and "pay_0" not in df.columns:
        df = df.rename(columns={"pay_1": "pay_0"})
    
    # Clean EDUCATION: map 0,5,6 to 4 (others)
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    
    # Clean MARRIAGE: map 0 to 3 (others)
    df["marriage"] = df["marriage"].replace({0: 3})
    
    # Clean SEX: keep only 1,2
    df = df[df["sex"].isin([1, 2])].copy()
    
    # Clean AGE: limit to [18, 100]
    df = df[(df["age"] >= 18) & (df["age"] <= 100)].copy()
    
    # Clean PAY variables: clip to [-2, 9]
    pay_cols = [f"pay_{i}" for i in range(7)]
    for col in pay_cols:
        if col in df.columns:
            df[col] = df[col].clip(-2, 9)
    
    # Drop ID column if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    
    return df


def main():
    """Main function to prepare dataset"""
    # Load parameters
    params = load_params()
    
    # Create directories
    Path(params["data"]["processed_path"]).mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    print("Loading raw data...")
    df = pd.read_csv(params["data"]["raw_path"])
    print(f"Raw data shape: {df.shape}")
    
    # Clean column names
    df = clean_column_names(df)
    
    # Clean data
    print("Cleaning data...")
    df = clean_data(df)
    print(f"Cleaned data shape: {df.shape}")
    
    # Split data
    print("Splitting data...")
    train_df, test_df = train_test_split(
        df,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=df["target"]
    )
    
    # Save processed data
    train_path = params["data"]["train_path"]
    test_path = params["data"]["test_path"]
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train data saved to {train_path}: {train_df.shape}")
    print(f"Test data saved to {test_path}: {test_df.shape}")
    print(f"Target distribution in train:\n{train_df['target'].value_counts(normalize=True)}")


if __name__ == "__main__":
    main()