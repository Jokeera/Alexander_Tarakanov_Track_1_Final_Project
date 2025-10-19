"""Tests for model metrics and predictions"""

import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from src.models.train import calculate_metrics
from src.monitoring.drift import calculate_psi


def test_calculate_metrics_perfect():
    """Test metrics calculation with perfect predictions"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "pr_auc" in metrics
    
    # Perfect predictions should have metrics close to 1
    assert metrics["f1_score"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_calculate_metrics_random():
    """Test metrics calculation with random predictions"""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_proba = np.random.random(100)
    y_pred = (y_proba > 0.5).astype(int)
    
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Check that all metrics are computed
    assert all(key in metrics for key in ["roc_auc", "f1_score", "precision", "recall", "pr_auc"])
    
    # Check that metrics are in valid range [0, 1]
    for key, value in metrics.items():
        assert 0 <= value <= 1, f"{key} = {value} is out of range"


def test_psi_no_drift():
    """Test PSI calculation with no drift"""
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)
    
    psi = calculate_psi(expected, actual)
    
    # PSI should be low (< 0.1) for similar distributions
    assert psi < 0.15


def test_psi_with_drift():
    """Test PSI calculation with drift"""
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(2, 1, 1000)  # Shifted distribution
    
    psi = calculate_psi(expected, actual)
    
    # PSI should be high (> 0.25) for different distributions
    assert psi > 0.25


def test_psi_same_data():
    """Test PSI calculation with identical data"""
    data = np.random.normal(0, 1, 1000)
    
    psi = calculate_psi(data, data)
    
    # PSI should be very close to 0 for identical data
    assert psi < 0.01


def test_psi_edge_cases():
    """Test PSI calculation with edge cases"""
    # Empty arrays
    psi_empty = calculate_psi(np.array([]), np.array([1, 2, 3]))
    assert np.isnan(psi_empty)
    
    # Constant values
    psi_constant = calculate_psi(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]))
    assert psi_constant == 0.0


def test_metrics_binary_classification():
    """Test that metrics work correctly for binary classification"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.9])
    
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Manual calculation for verification
    expected_precision = 2 / 3  # 2 true positives, 1 false positive
    expected_recall = 2 / 3  # 2 true positives, 1 false negative
    
    assert abs(metrics["precision"] - expected_precision) < 0.01
    assert abs(metrics["recall"] - expected_recall) < 0.01


def test_roc_auc_threshold():
    """Test that ROC-AUC is above random baseline"""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    auc = roc_auc_score(y_true, y_proba)
    
    # AUC should be well above random (0.5)
    assert auc > 0.8


def test_metrics_imbalanced_data():
    """Test metrics on imbalanced dataset"""
    # 90% class 0, 10% class 1
    y_true = np.array([0] * 90 + [1] * 10)
    # Model predicts mostly class 0
    y_pred = np.array([0] * 85 + [1] * 5 + [0] * 10)
    y_proba = np.concatenate([
        np.random.uniform(0, 0.3, 85),
        np.random.uniform(0.6, 0.9, 5),
        np.random.uniform(0, 0.4, 10)
    ])
    
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # All metrics should be computed without errors
    assert all(0 <= metrics[k] <= 1 for k in metrics.keys())
    
    # Recall should be low due to many false negatives
    assert metrics["recall"] < 0.6