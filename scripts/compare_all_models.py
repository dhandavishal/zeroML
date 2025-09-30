#!/usr/bin/env python3
"""
Compare all three models: Isolation Forest, OneClassSVM, and Autoencoder
"""

import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.iforest import IForestModel
from zeroml.models.ocsvm import OCSVMModel

def evaluate_model(y_true, scores, model_name, threshold_percentile=95):
    """Comprehensive evaluation of a single model."""
    print(f"\n📊 {model_name} Evaluation")
    print("-" * 40)
    
    # Score statistics
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    
    print(f"Score Statistics:")
    print(f"   Normal (n={len(normal_scores):,}): mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}")
    print(f"   Attack (n={len(anomaly_scores):,}): mean={anomaly_scores.mean():.6f}, std={anomaly_scores.std():.6f}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, scores)
    print(f"   📈 ROC-AUC: {roc_auc:.4f}")
    
    # Test different thresholds
    thresholds = [
        ("90th percentile", np.percentile(normal_scores, 90)),
        ("95th percentile", np.percentile(normal_scores, 95)),
        ("99th percentile", np.percentile(normal_scores, 99)),
        ("Mean + 2 std", normal_scores.mean() + 2 * normal_scores.std())
    ]
    
    best_f1 = 0
    best_metrics = None
    
    print(f"Threshold Analysis:")
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
        
        print(f"   {name:15} P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, FPR={fpr:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'fpr': fpr,
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

def load_autoencoder_with_normalization(model_path, results_path):
    """Load autoencoder and its normalization parameters."""
    # Load normalization parameters
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    X_min = results['normalization_params']['X_min']
    X_max = results['normalization_params']['X_max']
    
    # Load model
    autoencoder_model = tf.keras.models.load_model(model_path)
    
    return autoencoder_model, X_min, X_max

def main():
    """Compare all three trained models."""
    print("🏆 Three-Model Comparison: IF vs OC-SVM vs Autoencoder")
    print("=" * 60)
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    
    # Load test data for each model type
    print("📥 Loading test data...")
    
    # Standardized features for IF and OC-SVM
    std_data = np.load(data_dir / "test_standardized.npz")
    X_test_std, y_test = std_data['X'], std_data['y']
    
    # PCA features for Autoencoder
    pca_data = np.load(data_dir / "test_pca.npz")
    X_test_pca, _ = pca_data['X'], pca_data['y']
    
    print(f"   Test data: {len(y_test):,} samples")
    print(f"   Labels: Benign={np.sum(y_test==0):,}, Attack={np.sum(y_test==1):,}")
    
    # Load models and get predictions
    print(f"\n🔮 Loading models and generating predictions...")
    
    results = {}
    
    # 1. Isolation Forest
    print("   Loading Isolation Forest...")
    if_model = IForestModel.load(models_dir / "isolation_forest_tuned.pkl")
    if_scores = if_model.score(X_test_std)
    results['isolation_forest'] = evaluate_model(y_test, if_scores, "Isolation Forest")
    
    # 2. OneClassSVM
    print("   Loading OneClassSVM...")
    ocsvm_model = OCSVMModel.load(models_dir / "oneclass_svm_optimized.pkl")
    ocsvm_scores = ocsvm_model.score(X_test_std)
    results['oneclass_svm'] = evaluate_model(y_test, ocsvm_scores, "OneClassSVM")
    
    # 3. Autoencoder
    print("   Loading Autoencoder...")
    ae_model, X_min, X_max = load_autoencoder_with_normalization(
        models_dir / "autoencoder_model.keras",
        models_dir / "autoencoder_results.json"
    )
    
    # Normalize PCA features for autoencoder
    X_test_pca_norm = (X_test_pca - X_min) / (X_max - X_min + 1e-8)
    ae_predictions = ae_model.predict(X_test_pca_norm, batch_size=1024, verbose=0)
    ae_scores = ((X_test_pca_norm - ae_predictions)**2).mean(axis=1)
    results['autoencoder'] = evaluate_model(y_test, ae_scores, "Autoencoder")
    
    # Overall comparison
    print(f"\n🏆 FINAL COMPARISON")
    print("=" * 50)
    
    models = [
        ("Isolation Forest", results['isolation_forest']),
        ("OneClassSVM", results['oneclass_svm']),
        ("Autoencoder", results['autoencoder'])
    ]
    
    best_model = None
    best_auc = 0
    
    print(f"{'Model':<20} {'ROC-AUC':<10} {'Best F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    
    for name, result in models:
        auc = result['roc_auc']
        f1 = result['best_metrics']['f1_score']
        precision = result['best_metrics']['precision']
        recall = result['best_metrics']['recall']
        
        print(f"{name:<20} {auc:<10.4f} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = name
    
    print(f"\n🏆 Winner: {best_model} (ROC-AUC: {best_auc:.4f})")
    
    # Performance rankings
    model_aucs = [(name, results[key]['roc_auc']) for key, name in [
        ('isolation_forest', 'Isolation Forest'),
        ('oneclass_svm', 'OneClassSVM'),
        ('autoencoder', 'Autoencoder')
    ]]
    model_aucs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Performance Rankings:")
    for i, (name, auc) in enumerate(model_aucs):
        medal = ["🥇", "🥈", "🥉"][i]
        performance = "Excellent" if auc > 0.8 else "Good" if auc > 0.7 else "Moderate" if auc > 0.6 else "Poor"
        print(f"   {medal} {name}: {auc:.4f} ({performance})")
    
    # Save comprehensive results
    comparison_results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'test_data': {
            'total_samples': int(len(y_test)),
            'benign_samples': int(np.sum(y_test == 0)),
            'attack_samples': int(np.sum(y_test == 1)),
            'standardized_features': X_test_std.shape[1], 
            'pca_features': X_test_pca.shape[1]
        },
        'model_results': results,
        'rankings': [{'rank': i+1, 'model': name, 'roc_auc': auc} for i, (name, auc) in enumerate(model_aucs)],
        'best_model': best_model,
        'best_roc_auc': best_auc
    }
    
    results_path = models_dir / "three_model_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    print(f"📋 Comprehensive results saved: {results_path}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if best_auc > 0.8:
        print(f"   🎉 {best_model} shows excellent performance for deployment")
        print(f"   🔗 Consider ensemble methods for even better results")
    elif best_auc > 0.6:
        print(f"   👍 {best_model} shows good performance")
        print(f"   🔗 Ensemble methods could improve results significantly")
    else:
        print(f"   ⚠️ All models show limited performance on this dataset")
        print(f"   🔧 Consider feature engineering or semi-supervised approaches")
    
    print(f"\n✅ Three-model comparison completed!")
    return comparison_results

if __name__ == "__main__":
    main()