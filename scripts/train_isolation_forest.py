#!/usr/bin/env python3
"""
Train Isolation Forest model on engineered features
"""

import numpy as np
import joblib
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
import time

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.iforest import IForestModel

def evaluate_anomaly_detection(y_true, scores, threshold=None):
    """
    Evaluate anomaly detection performance.
    
    Args:
        y_true: True labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)
        threshold: Score threshold for classification
        
    Returns:
        Dictionary with evaluation metrics
    """
    if threshold is None:
        # Use median score on normal samples as threshold
        normal_scores = scores[y_true == 0]
        threshold = np.percentile(normal_scores, 95)  # 95th percentile of normal scores
    
    # Convert scores to binary predictions
    y_pred = (scores > threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    # ROC-AUC (if we have both classes)
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else None
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    }

def train_isolation_forest(X_train, contamination=0.1, random_state=42):
    """Train Isolation Forest model."""
    print(f"🌲 Training Isolation Forest (contamination={contamination})...")
    
    # Create and train model
    model = IForestModel(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1
    )
    
    start_time = time.time()
    model.fit(X_train)
    training_time = time.time() - start_time
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    return model, training_time

def main():
    """Main training pipeline."""
    print("🚀 Starting Isolation Forest Training...")
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load engineered features (standardized version for IF)
    print("📥 Loading engineered features...")
    
    train_data = np.load(data_dir / "train_standardized.npz")
    X_train, y_train = train_data['X'], train_data['y']
    
    val_data = np.load(data_dir / "val_standardized.npz")
    X_val, y_val = val_data['X'], val_data['y']
    
    test_data = np.load(data_dir / "test_standardized.npz")
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"   Train: {X_train.shape}, labels: {np.unique(y_train)}")
    print(f"   Val: {X_val.shape}, labels: {np.unique(y_val)}")
    print(f"   Test: {X_test.shape}, labels: {np.unique(y_test)}")
    
    # Train model
    model, training_time = train_isolation_forest(X_train, contamination=0.1)
    
    # Evaluate on training set (to understand model behavior)
    print("📊 Evaluating on training set...")
    train_scores = model.score(X_train)
    train_metrics = evaluate_anomaly_detection(y_train, train_scores)
    print(f"   Training threshold: {train_metrics['threshold']:.4f}")
    print(f"   Training scores - min: {train_scores.min():.4f}, max: {train_scores.max():.4f}")
    
    # Evaluate on validation set (all normal)
    print("📊 Evaluating on validation set...")
    val_scores = model.score(X_val)
    val_metrics = evaluate_anomaly_detection(y_val, val_scores, threshold=train_metrics['threshold'])
    print(f"   Validation anomaly rate: {val_metrics['recall']:.2%} (should be low for normal data)")
    print(f"   Validation scores - min: {val_scores.min():.4f}, max: {val_scores.max():.4f}")
    
    # Evaluate on test set (mixed normal + attacks)
    print("📊 Evaluating on test set...")
    test_scores = model.score(X_test)
    test_metrics = evaluate_anomaly_detection(y_test, test_scores, threshold=train_metrics['threshold'])
    
    print(f"   Test Results:")
    print(f"   📈 ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   🎯 Precision: {test_metrics['precision']:.4f}")
    print(f"   🔍 Recall: {test_metrics['recall']:.4f}")
    print(f"   ⚖️ F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"   ✅ Accuracy: {test_metrics['accuracy']:.4f}")
    
    print(f"   Confusion Matrix:")
    print(f"   TN: {test_metrics['true_negatives']:,}, FP: {test_metrics['false_positives']:,}")
    print(f"   FN: {test_metrics['false_negatives']:,}, TP: {test_metrics['true_positives']:,}")
    
    # Save model and results
    model_path = models_dir / "isolation_forest.pkl"
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save training results
    results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'model_type': 'IsolationForest',
        'model_path': str(model_path),
        'training_time_seconds': training_time,
        'data_shapes': {
            'train': X_train.shape,
            'val': X_val.shape,
            'test': X_test.shape
        },
        'hyperparameters': {
            'contamination': 0.1,
            'n_estimators': 100,
            'random_state': 42
        },
        'evaluation': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        },
        'score_statistics': {
            'train': {
                'min': float(train_scores.min()),
                'max': float(train_scores.max()),
                'mean': float(train_scores.mean()),
                'std': float(train_scores.std())
            },
            'val': {
                'min': float(val_scores.min()),
                'max': float(val_scores.max()),
                'mean': float(val_scores.mean()),
                'std': float(val_scores.std())
            },
            'test': {
                'min': float(test_scores.min()),
                'max': float(test_scores.max()),
                'mean': float(test_scores.mean()),
                'std': float(test_scores.std())
            }
        }
    }
    
    results_path = models_dir / "isolation_forest_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📋 Results saved: {results_path}")
    
    # Sanity checks
    print("\n🔍 Sanity Checks:")
    print(f"   ✓ Model can score data: {len(test_scores) == len(y_test)}")
    print(f"   ✓ ROC-AUC > 0.5: {test_metrics['roc_auc'] > 0.5}")
    print(f"   ✓ Detection rate reasonable: {0.05 < test_metrics['recall'] < 0.95}")
    print(f"   ✓ Some attacks detected: {test_metrics['true_positives'] > 0}")
    
    if test_metrics['roc_auc'] > 0.7:
        print("🎉 Good performance! ROC-AUC > 0.7")
    elif test_metrics['roc_auc'] > 0.6:
        print("👍 Reasonable performance. ROC-AUC > 0.6")
    else:
        print("⚠️ Low performance. May need hyperparameter tuning.")
    
    print(f"\n✅ Isolation Forest training completed!")
    return model, results

if __name__ == "__main__":
    main()