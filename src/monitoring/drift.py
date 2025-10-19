"""Data drift monitoring using Population Stability Index (PSI)"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI)

    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate change, monitor
    PSI >= 0.25: Significant change, action required
    """

    # Handle edge cases
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    if len(breakpoints) < 2:
        # If all values are the same, return 0
        return 0.0

    # Calculate distributions
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)

    return psi


def detect_drift(train_data, test_data, features_to_monitor, threshold=0.25):
    """
    Detect drift in specified features

    Args:
        train_data: Training/reference data (DataFrame)
        test_data: New/test data (DataFrame)
        features_to_monitor: List of feature names to monitor
        threshold: PSI threshold for drift detection (default: 0.25)

    Returns:
        dict: Drift report with PSI values and drift status
    """

    drift_report = {"psi_threshold": threshold, "features": {}, "drift_detected": False}

    for feature in features_to_monitor:
        if feature not in train_data.columns or feature not in test_data.columns:
            print(f"Warning: Feature '{feature}' not found in data")
            continue

        # Calculate PSI
        psi = calculate_psi(train_data[feature].values, test_data[feature].values)

        # Determine drift status
        if pd.isna(psi):
            status = "error"
            drift = False
        elif psi < 0.1:
            status = "stable"
            drift = False
        elif psi < threshold:
            status = "moderate"
            drift = False
        else:
            status = "drift_detected"
            drift = True

        drift_report["features"][feature] = {
            "psi": float(psi) if not pd.isna(psi) else None,
            "status": status,
            "drift": drift,
        }

        # Update overall drift status
        if drift:
            drift_report["drift_detected"] = True

    return drift_report


def generate_drift_report(
    train_path, test_path, model_path, output_path="models/drift_report.json"
):
    """Generate drift report comparing train and test data with model predictions"""

    # Load parameters
    params = load_params()

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Load model and make predictions if model exists
    try:
        import joblib

        model = joblib.load(model_path)

        # Prepare features (drop target if exists)
        train_features = train_df.drop(columns=["target"], errors="ignore")
        test_features = test_df.drop(columns=["target"], errors="ignore")

        # Make predictions
        train_proba = model.predict_proba(train_features)[:, 1]
        test_proba = model.predict_proba(test_features)[:, 1]

        # Add predictions to dataframes
        train_df["predicted_proba"] = train_proba
        test_df["predicted_proba"] = test_proba

        print("Predictions added to data")
    except Exception as e:
        print(f"Warning: Could not load model or make predictions: {e}")

    # Features to monitor
    features_to_monitor = ["predicted_proba"] + params["monitoring"]["features_to_monitor"]
    threshold = params["monitoring"]["psi_threshold"]

    # Detect drift
    print(f"\nMonitoring drift for features: {features_to_monitor}")
    drift_report = detect_drift(train_df, test_df, features_to_monitor, threshold)

    # Print results
    print(f"\n{'='*60}")
    print("DRIFT MONITORING REPORT")
    print(f"{'='*60}")
    print(f"PSI Threshold: {threshold}")
    print(f"\nOverall Drift Detected: {'YES ⚠️' if drift_report['drift_detected'] else 'NO ✓'}")
    print(f"\n{'Feature':<25} {'PSI':<10} {'Status':<15}")
    print("-" * 60)

    for feature, result in drift_report["features"].items():
        psi_str = f"{result['psi']:.4f}" if result["psi"] is not None else "N/A"
        status = result["status"]

        # Add emoji indicators
        if status == "stable":
            status_display = "✓ Stable"
        elif status == "moderate":
            status_display = "⚡ Moderate"
        elif status == "drift_detected":
            status_display = "⚠️  DRIFT"
        else:
            status_display = "❌ Error"

        print(f"{feature:<25} {psi_str:<10} {status_display:<15}")

    print("=" * 60)

    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    print(f"\nDrift report saved to {output_path}")

    return drift_report


def main():
    """Main drift monitoring function"""

    # Paths
    train_path = "data/processed/train_features.csv"
    test_path = "data/processed/test_features.csv"
    model_path = "models/model.pkl"

    # Generate drift report
    drift_report = generate_drift_report(train_path, test_path, model_path)

    # Exit with error code if drift detected
    if drift_report["drift_detected"]:
        print("\n⚠️  WARNING: Data drift detected! Review the drift report.")
        return 1
    else:
        print("\n✅ No significant drift detected.")
        return 0


if __name__ == "__main__":
    exit(main())
