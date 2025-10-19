"""Tests for feature engineering"""

import pytest
import pandas as pd
import numpy as np
from src.features.build_features import create_features


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    params = {
        "features": {
            "age_bins": [18, 30, 45, 60, 100],
            "age_labels": ["young", "middle", "senior", "elderly"]
        }
    }
    
    df = pd.DataFrame({
        "limit_bal": [20000, 30000, 40000],
        "age": [25, 35, 55],
        "pay_0": [2, 0, -1],
        "pay_2": [2, 1, 0],
        "pay_3": [1, 0, -1],
        "pay_4": [0, -1, -2],
        "pay_5": [-1, -2, -2],
        "pay_6": [-2, -2, -2],
        "bill_amt1": [10000, 15000, 20000],
        "bill_amt2": [9000, 14000, 19000],
        "bill_amt3": [8000, 13000, 18000],
        "bill_amt4": [7000, 12000, 17000],
        "bill_amt5": [6000, 11000, 16000],
        "bill_amt6": [5000, 10000, 15000],
        "pay_amt1": [1000, 2000, 3000],
        "pay_amt2": [1100, 2100, 3100],
        "pay_amt3": [1200, 2200, 3200],
        "pay_amt4": [1300, 2300, 3300],
        "pay_amt5": [1400, 2400, 3400],
        "pay_amt6": [1500, 2500, 3500],
        "target": [1, 0, 0]
    })
    
    return df, params


def test_age_bin_creation(sample_data):
    """Test age binning feature"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    assert "age_bin" in df_features.columns
    assert df_features["age_bin"].iloc[0] == "young"  # age 25
    assert df_features["age_bin"].iloc[1] == "middle"  # age 35
    assert df_features["age_bin"].iloc[2] == "senior"  # age 55


def test_utilization_last(sample_data):
    """Test utilization_last feature"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    assert "utilization_last" in df_features.columns
    # Check calculation: bill_amt1 / limit_bal
    expected_util_0 = 10000 / 20000
    assert abs(df_features["utilization_last"].iloc[0] - expected_util_0) < 0.001


def test_pay_delay_sum(sample_data):
    """Test pay_delay_sum feature"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    assert "pay_delay_sum" in df_features.columns
    # First row: pay_0=2, pay_2=2, pay_3=1 (all > 0) = 3 delays
    assert df_features["pay_delay_sum"].iloc[0] == 3
    # Second row: pay_2=1 (only one > 0) = 1 delay
    assert df_features["pay_delay_sum"].iloc[1] == 1


def test_pay_delay_max(sample_data):
    """Test pay_delay_max feature"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    assert "pay_delay_max" in df_features.columns
    # First row: max of [2, 2, 1, 0, -1, -2] = 2
    assert df_features["pay_delay_max"].iloc[0] == 2


def test_bill_trend(sample_data):
    """Test bill_trend feature"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    assert "bill_trend" in df_features.columns
    # Check that values are clipped to [-5, 5]
    assert all(df_features["bill_trend"] >= -5)
    assert all(df_features["bill_trend"] <= 5)


def test_pay_trend(sample_data):
    """Test pay_trend feature"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    assert "pay_trend" in df_features.columns
    # Check that values are clipped to [-5, 5]
    assert all(df_features["pay_trend"] >= -5)
    assert all(df_features["pay_trend"] <= 5)


def test_bill_avg(sample_data):
    """Test bill_avg feature"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    assert "bill_avg" in df_features.columns
    # First row: mean of [10000, 9000, 8000, 7000, 6000, 5000]
    expected_avg = (10000 + 9000 + 8000 + 7000 + 6000 + 5000) / 6
    assert abs(df_features["bill_avg"].iloc[0] - expected_avg) < 0.01


def test_pay_amt_avg(sample_data):
    """Test pay_amt_avg feature"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    assert "pay_amt_avg" in df_features.columns
    # First row: mean of [1000, 1100, 1200, 1300, 1400, 1500]
    expected_avg = (1000 + 1100 + 1200 + 1300 + 1400 + 1500) / 6
    assert abs(df_features["pay_amt_avg"].iloc[0] - expected_avg) < 0.01


def test_pay_to_bill_ratio(sample_data):
    """Test pay_to_bill_ratio feature"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    assert "pay_to_bill_ratio" in df_features.columns
    # Check that values are clipped to [0, 5]
    assert all(df_features["pay_to_bill_ratio"] >= 0)
    assert all(df_features["pay_to_bill_ratio"] <= 5)


def test_all_features_created(sample_data):
    """Test that all expected features are created"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    expected_features = [
        "age_bin", "utilization_last", "pay_delay_sum", "pay_delay_max",
        "bill_trend", "pay_trend", "bill_avg", "pay_amt_avg", "pay_to_bill_ratio"
    ]
    
    for feature in expected_features:
        assert feature in df_features.columns, f"Feature {feature} not found"


def test_no_nan_in_features(sample_data):
    """Test that no NaN values are introduced"""
    df, params = sample_data
    
    df_features = create_features(df, params)
    
    # Check numerical features for NaN
    numerical_features = [
        "utilization_last", "pay_delay_sum", "pay_delay_max",
        "bill_trend", "pay_trend", "bill_avg", "pay_amt_avg", "pay_to_bill_ratio"
    ]
    
    for feature in numerical_features:
        assert not df_features[feature].isna().any(), f"NaN found in {feature}"