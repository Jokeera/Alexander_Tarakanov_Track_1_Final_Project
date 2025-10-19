"""Feature engineering for credit scoring model"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def create_features(df, params):
    """Create engineered features"""
    df = df.copy()

    # Age binning
    age_bins = params["features"]["age_bins"]
    age_labels = params["features"]["age_labels"]
    df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, include_lowest=True)

    # Utilization rate (last month)
    df["utilization_last"] = np.where(df["limit_bal"] > 0, df["bill_amt1"] / df["limit_bal"], 0)
    df["utilization_last"] = df["utilization_last"].clip(0, 2)

    # Payment delay sum (total delayed months)
    pay_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
    df["pay_delay_sum"] = df[pay_cols].apply(lambda x: (x > 0).sum(), axis=1)

    # Maximum payment delay
    df["pay_delay_max"] = df[pay_cols].max(axis=1)

    # Bill amount trend (last 3 months)
    df["bill_trend"] = ((df["bill_amt1"] - df["bill_amt3"]) / (df["bill_amt3"].abs() + 1)).clip(
        -5, 5
    )

    # Payment amount trend (last 3 months)
    df["pay_trend"] = ((df["pay_amt1"] - df["pay_amt3"]) / (df["pay_amt3"] + 1)).clip(-5, 5)

    # Average bill amount
    bill_cols = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
    df["bill_avg"] = df[bill_cols].mean(axis=1)

    # Average payment amount
    pay_amt_cols = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]
    df["pay_amt_avg"] = df[pay_amt_cols].mean(axis=1)

    # Payment to bill ratio
    df["pay_to_bill_ratio"] = np.where(df["bill_avg"] > 0, df["pay_amt_avg"] / df["bill_avg"], 0)
    df["pay_to_bill_ratio"] = df["pay_to_bill_ratio"].clip(0, 5)

    return df


def main():
    """Main function for feature engineering"""
    # Load parameters
    params = load_params()

    # Load train data
    print("Loading training data...")
    train_df = pd.read_csv(params["data"]["train_path"])
    print(f"Train data shape: {train_df.shape}")

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(params["data"]["test_path"])
    print(f"Test data shape: {test_df.shape}")

    # Create features
    print("Creating features for training data...")
    train_features = create_features(train_df, params)

    print("Creating features for test data...")
    test_features = create_features(test_df, params)

    # Save feature-engineered data
    train_features_path = "data/processed/train_features.csv"
    test_features_path = "data/processed/test_features.csv"

    train_features.to_csv(train_features_path, index=False)
    test_features.to_csv(test_features_path, index=False)

    print(f"Train features saved to {train_features_path}: {train_features.shape}")
    print(f"Test features saved to {test_features_path}: {test_features.shape}")

    print("\nNew features created:")
    new_features = [
        "age_bin",
        "utilization_last",
        "pay_delay_sum",
        "pay_delay_max",
        "bill_trend",
        "pay_trend",
        "bill_avg",
        "pay_amt_avg",
        "pay_to_bill_ratio",
    ]
    for feat in new_features:
        print(f"  - {feat}")


if __name__ == "__main__":
    main()
