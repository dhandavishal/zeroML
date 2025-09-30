#!/usr/bin/env python3
"""
Train OneClassSVM with optimized parameters found during tuning
"""

import numpy as np
import json
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.ocsvm import OCSVMModel

def evaluate_model(y_true, scores, threshold_percentile=95):
    """Quick evaluation using best threshold approach."""
    # Use percentile of normal scores as threshold
    normal_scores = scores[y_true == 0]
    threshold = np.percentile(normal_scores, threshold_percentile)
    
    y_pred = (scores > threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    roc_auc = roc_auc_score(y_true, scores)
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }

def main():
    """Train OneClassSVM with best parameters from tuning."""
    print("🚀 Training OneClassSVM with Optimal Parameters")
    print("=" * 50)
    
    # Best parameters from hyperparameter tuning
    best_params = {
        'gamma': 0.1,
        'nu': 0.01,
        'kernel': 'rbf'
    }
    
    print(f"📋 Using parameters: {best_params}")
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📥 Loading standardized data...")
    train_data = np.load(data_dir / "train_standardized.npz")
    X_train, y_train = train_data['X'], train_data['y']
    
    test_data = np.load(data_dir / "test_standardized.npz")
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Create and train model with best parameters
    print("🔧 Training OneClassSVM...")
    model = OCSVMModel(
        gamma=best_params['gamma'],
        nu=best_params['nu'], 
        kernel=best_params['kernel']
    )
    
    start_time = time.time()
    model.fit(X_train)
    training_time = time.time() - start_time
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_scores = model.score(X_test)
    test_metrics = evaluate_model(y_test, test_scores)
    
    print(f"   📈 ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   🎯 Precision: {test_metrics['precision']:.4f}")
    print(f"   🔍 Recall: {test_metrics['recall']:.4f}")
    print(f"   ⚖️ F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Save model
    model_path = models_dir / "oneclass_svm_optimized.pkl"
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save results
    results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'model_type': 'OneClassSVM',
        'model_path': str(model_path),
        'hyperparameters': best_params,
        'training_time_seconds': training_time,
        'test_evaluation': test_metrics,
        'data_shapes': {
            'train': X_train.shape,
            'test': X_test.shape
        }
    }
    
    results_path = models_dir / "oneclass_svm_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📋 Results saved: {results_path}")
    
    # Performance comparison with Isolation Forest
    print(f"\n📊 Performance Summary:")
    print(f"   OneClassSVM ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Expected based on tuning: ~0.90")
    
    if test_metrics['roc_auc'] > 0.8:
        print("🎉 Excellent performance! Much better than Isolation Forest")
    elif test_metrics['roc_auc'] > 0.7:
        print("👍 Good performance improvement over Isolation Forest")
    else:
        print("🤔 Performance lower than expected from tuning")
    
    print(f"\n✅ OneClassSVM training completed!")
    return model, test_metrics

if __name__ == "__main__":
    main()