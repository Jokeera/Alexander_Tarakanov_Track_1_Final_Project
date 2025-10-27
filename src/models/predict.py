"""
Offline prediction script using trained pipeline

Examples:
  python -m src.models.predict --input data.csv --output preds.csv
  python -m src.models.predict --single '{"limit_bal":20000, "sex":2, ...}'
"""

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import yaml


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str):
    """Load trained pipeline."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def main():
    parser = argparse.ArgumentParser(description="Run offline credit scoring predictions")
    parser.add_argument("--input", type=str, help="Path to CSV with raw features")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--single", type=str, help="Single input row as JSON")
    args = parser.parse_args()

    # Load model
    model_path = "models/model.pkl"
    model = load_model(model_path)

    if args.single:
        # Single row JSON
        data = json.loads(args.single)
        df = pd.DataFrame([data])

        proba = model.predict_proba(df)[0, 1]
        pred = int(model.predict(df)[0])

        print(json.dumps({
            "prediction": pred,
            "probability": float(proba)
        }, indent=2))
        return 0

    if args.input:
        # Batch predictions on raw CSV
        df = pd.read_csv(args.input)

        proba = model.predict_proba(df)[:, 1]
        pred = model.predict(df)

        df_out = df.copy()
        df_out["probability"] = proba
        df_out["prediction"] = pred

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output, index=False)

        print(f"✅ Predictions saved to: {args.output}")
        print(df_out.head())
        return 0

    print("❌ No input provided. Use --input or --single")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
