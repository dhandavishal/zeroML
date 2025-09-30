#!/usr/bin/env python3
"""
End-to-end pipeline testing for ZeroML
Tests both standardized and PCA features on the full dataset
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

def detailed_evaluation(y_true, scores, model_name="Model"):
    """Comprehensive evaluation of anomaly detection performance."""
    print(f"\n📊 Detailed Evaluation: {model_name}")
    print("=" * 50)
    
    # Basic score statistics
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    
    print(f"Score Statistics:")
    print(f"   Normal samples: {len(normal_scores):,}")
    print(f"   Anomaly samples: {len(anomaly_scores):,}")
    print(f"   Normal scores - mean: {normal_scores.mean():.4f}, std: {normal_scores.std():.4f}")
    print(f"   Anomaly scores - mean: {anomaly_scores.mean():.4f}, std: {anomaly_scores.std():.4f}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, scores)
    print(f"   📈 ROC-AUC: {roc_auc:.4f}")
    
    # Try different thresholds
    thresholds = [
        ("50th percentile normal", np.percentile(normal_scores, 50)),
        ("90th percentile normal", np.percentile(normal_scores, 90)),
        ("95th percentile normal", np.percentile(normal_scores, 95)),
        ("99th percentile normal", np.percentile(normal_scores, 99)),
        ("Mean normal", normal_scores.mean()),
        ("Mean + 1 std", normal_scores.mean() + normal_scores.std()),
        ("Mean + 2 std", normal_scores.mean() + 2 * normal_scores.std())
    ]
    
    best_f1 = 0
    best_threshold = None
    best_metrics = None
    
    print(f"\nThreshold Analysis:")
    for name, threshold in thresholds:
        y_pred = (scores > threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"   {name:25} (th={threshold:.4f}): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, FPR={fpr:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'fpr': fpr,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
    
    print(f"\n🏆 Best threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
    
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

def test_feature_representations():
    """Test both standardized and PCA feature representations."""
    print("🚀 Testing Different Feature Representations")
    print("=" * 60)
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    results = {}
    
    # Test configurations
    configs = [
        ("Standardized Features", "standardized", "isolation_forest_tuned.pkl"),
        ("PCA Features", "pca", None)  # Will train new model for PCA
    ]
    
    for config_name, feature_type, model_path in configs:
        print(f"\n🔍 Testing: {config_name}")
        print("-" * 40)
        
        # Load data
        train_data = np.load(data_dir / f"train_{feature_type}.npz")
        X_train, y_train = train_data['X'], train_data['y']
        
        test_data = np.load(data_dir / f"test_{feature_type}.npz")
        X_test, y_test = test_data['X'], test_data['y']
        
        print(f"   📊 Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Load or train model
        if model_path and (models_dir / model_path).exists():
            print(f"   📂 Loading existing model: {model_path}")
            model = IForestModel.load(models_dir / model_path)
        else:
            print(f"   🌲 Training new model for {feature_type} features...")
            model = IForestModel(
                contamination=0.05,  # Best from tuning
                n_estimators=100,
                max_samples='auto',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train)
            
            # Save model
            model_save_path = models_dir / f"isolation_forest_{feature_type}.pkl"
            model.save(model_save_path)
            print(f"   💾 Model saved: {model_save_path}")
        
        # Evaluate
        test_scores = model.score(X_test)
        evaluation = detailed_evaluation(y_test, test_scores, config_name)
        
        results[feature_type] = {
            'config_name': config_name,
            'data_shape': X_test.shape,
            'evaluation': evaluation
        }
    
    return results

def compare_results(results):
    """Compare results across different feature representations."""
    print(f"\n🏆 COMPARISON SUMMARY")
    print("=" * 60)
    
    best_auc = 0
    best_config = None
    
    for feature_type, result in results.items():
        config_name = result['config_name']
        eval_data = result['evaluation']
        
        roc_auc = eval_data['roc_auc']
        best_f1 = eval_data['best_metrics']['f1_score']
        best_recall = eval_data['best_metrics']['recall']
        best_precision = eval_data['best_metrics']['precision']
        
        print(f"\n{config_name}:")
        print(f"   📈 ROC-AUC: {roc_auc:.4f}")
        print(f"   🎯 Best F1: {best_f1:.4f}")
        print(f"   🔍 Best Recall: {best_recall:.4f}")
        print(f"   🎯 Best Precision: {best_precision:.4f}")
        print(f"   📊 Features: {result['data_shape'][1]}")
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_config = config_name
    
    print(f"\n🏆 Winner: {best_config} (ROC-AUC: {best_auc:.4f})")
    
    # Performance assessment
    if best_auc > 0.8:
        print("🎉 Excellent performance!")
    elif best_auc > 0.7:
        print("👍 Good performance")
    elif best_auc > 0.6:
        print("🤔 Moderate performance - consider other algorithms")
    else:
        print("⚠️ Low performance - this dataset may be challenging for unsupervised methods")
    
    return best_config, best_auc

def save_test_results(results, output_path):
    """Save comprehensive test results."""
    test_summary = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'pipeline_test': 'end_to_end_evaluation',
        'configurations_tested': len(results),
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(test_summary, f, indent=2, default=str)
    
    print(f"📋 Test results saved: {output_path}")

def main():
    """Main testing pipeline."""
    print("🎯 ZeroML End-to-End Pipeline Testing")
    print("=" * 50)
    
    # Test different feature representations
    results = test_feature_representations()
    
    # Compare and analyze results
    best_config, best_auc = compare_results(results)
    
    # Save results
    results_path = Path("models/end_to_end_test_results.json")
    save_test_results(results, results_path)
    
    # Final pipeline validation
    print(f"\n✅ PIPELINE VALIDATION")
    print("=" * 30)
    
    success_criteria = [
        ("Models can be trained", True),
        ("Models can score new data", True), 
        ("ROC-AUC > 0.5 (better than random)", best_auc > 0.5),
        ("Some attacks detected", any(r['evaluation']['best_metrics']['tp'] > 0 for r in results.values())),
        ("Not all samples flagged as anomalies", any(r['evaluation']['best_metrics']['tn'] > 0 for r in results.values()))
    ]
    
    all_passed = True
    for criterion, passed in success_criteria:
        status = "✅" if passed else "❌"
        print(f"   {status} {criterion}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 Pipeline validation PASSED!")
        print(f"📊 Best configuration: {best_config}")
        print(f"📈 Best ROC-AUC: {best_auc:.4f}")
    else:
        print(f"\n⚠️ Pipeline validation FAILED!")
        print("   Consider trying different algorithms or feature engineering approaches.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)