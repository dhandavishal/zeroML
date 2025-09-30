#!/usr/bin/env python3
"""
Compare Isolation Forest vs OneClassSVM performance
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import matplotlib.pyplot as plt

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.iforest import IForestModel
from zeroml.models.ocsvm import OCSVMModel

def evaluate_with_multiple_thresholds(y_true, scores, model_name):
    """Evaluate model with different threshold strategies."""
    print(f"\n📊 Evaluating {model_name}")
    print("-" * 40)
    
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    
    print(f"Score Statistics:")
    print(f"   Normal: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
    print(f"   Anomaly: mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}")
    
    roc_auc = roc_auc_score(y_true, scores)
    print(f"   📈 ROC-AUC: {roc_auc:.4f}")
    
    # Test different thresholds
    thresholds = [
        ("95th percentile", np.percentile(normal_scores, 95)),
        ("99th percentile", np.percentile(normal_scores, 99)),
        ("Mean + 2 std", normal_scores.mean() + 2 * normal_scores.std())
    ]
    
    best_f1 = 0
    best_metrics = None
    
    for name, threshold in thresholds:
        y_pred = (scores > threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   {name:15} P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
    
    return {
        'roc_auc': roc_auc,
        'best_metrics': best_metrics,
        'score_stats': {
            'normal_mean': float(normal_scores.mean()),
            'normal_std': float(normal_scores.std()),
            'anomaly_mean': float(anomaly_scores.mean()),
            'anomaly_std': float(anomaly_scores.std())
        }
    }

def main():
    """Compare Isolation Forest vs OneClassSVM."""
    print("🔍 Model Comparison: Isolation Forest vs OneClassSVM")
    print("=" * 60)
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    
    # Load test data
    print("📥 Loading test data...")
    test_data = np.load(data_dir / "test_standardized.npz")
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"   Test data: {X_test.shape}")
    print(f"   Labels: {np.bincount(y_test)} (benign=0, attack=1)")
    
    # Load models
    print("\n📂 Loading trained models...")
    
    # Isolation Forest
    if_model = IForestModel.load(models_dir / "isolation_forest_tuned.pkl")
    print("   ✅ Isolation Forest loaded")
    
    # OneClassSVM
    ocsvm_model = OCSVMModel.load(models_dir / "oneclass_svm_optimized.pkl")
    print("   ✅ OneClassSVM loaded")
    
    # Get predictions
    print("\n🔮 Getting model predictions...")
    if_scores = if_model.score(X_test)
    ocsvm_scores = ocsvm_model.score(X_test)
    
    # Evaluate models
    if_results = evaluate_with_multiple_thresholds(y_test, if_scores, "Isolation Forest")
    ocsvm_results = evaluate_with_multiple_thresholds(y_test, ocsvm_scores, "OneClassSVM")
    
    # Comparison summary
    print(f"\n🏆 COMPARISON SUMMARY")
    print("=" * 40)
    
    models = [
        ("Isolation Forest", if_results),
        ("OneClassSVM", ocsvm_results)
    ]
    
    best_model = None
    best_auc = 0
    
    for name, results in models:
        auc = results['roc_auc']
        f1 = results['best_metrics']['f1_score']
        precision = results['best_metrics']['precision']
        recall = results['best_metrics']['recall']
        
        print(f"\n{name}:")
        print(f"   📈 ROC-AUC: {auc:.4f}")
        print(f"   🎯 Best F1: {f1:.4f}")
        print(f"   🔍 Precision: {precision:.4f}")
        print(f"   📊 Recall: {recall:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = name
    
    print(f"\n🏆 Winner: {best_model} (ROC-AUC: {best_auc:.4f})")
    
    # Performance improvement
    improvement = ocsvm_results['roc_auc'] - if_results['roc_auc']
    print(f"📈 OneClassSVM improvement over IF: {improvement:.4f} ({improvement/if_results['roc_auc']*100:.1f}%)")
    
    # Save comparison results
    comparison_results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'test_data_shape': X_test.shape,
        'label_distribution': {
            'benign': int(np.sum(y_test == 0)),
            'attack': int(np.sum(y_test == 1))
        },
        'isolation_forest': if_results,
        'oneclass_svm': ocsvm_results,
        'comparison': {
            'best_model': best_model,
            'best_roc_auc': best_auc,
            'improvement': improvement,
            'improvement_percent': improvement/if_results['roc_auc']*100
        }
    }
    
    results_path = models_dir / "model_comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    print(f"📋 Comparison results saved: {results_path}")
    
    print(f"\n✅ Model comparison completed!")
    return comparison_results

if __name__ == "__main__":
    main()