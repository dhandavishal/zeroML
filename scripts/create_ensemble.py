#!/usr/bin/env python3
"""
Create ensemble predictions combining all three models
"""

import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.iforest import IForestModel
from zeroml.models.ocsvm import OCSVMModel

def load_autoencoder_improved(model_path, results_path):
    """Load improved autoencoder model."""
    autoencoder = tf.keras.models.load_model(model_path)
    return autoencoder

def evaluate_ensemble(y_true, scores, method_name):
    """Comprehensive evaluation of ensemble predictions."""
    print(f"\n📊 {method_name} Evaluation")
    print("-" * 40)
    
    # Score statistics
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    
    print(f"Score Statistics:")
    print(f"   Normal: mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}")
    print(f"   Attack: mean={anomaly_scores.mean():.6f}, std={anomaly_scores.std():.6f}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, scores)
    print(f"   📈 ROC-AUC: {roc_auc:.4f}")
    
    # Test different thresholds
    thresholds = [
        ("90th percentile", np.percentile(normal_scores, 90)),
        ("95th percentile", np.percentile(normal_scores, 95)),
        ("99th percentile", np.percentile(normal_scores, 99)),
        ("Optimal F1", None)  # Will find optimal F1 threshold
    ]
    
    best_f1 = 0
    best_metrics = None
    
    # Find optimal F1 threshold
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else scores.max()
    thresholds[3] = ("Optimal F1", optimal_threshold)
    
    print(f"Threshold Analysis:")
    for name, threshold in thresholds:
        if threshold is None:
            continue
            
        y_pred = (scores > threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"   {name:15} P={precision_val:.3f}, R={recall_val:.3f}, F1={f1_val:.3f}, FPR={fpr:.3f}")
        
        if f1_val > best_f1:
            best_f1 = f1_val
            best_metrics = {
                'threshold': threshold,
                'precision': precision_val,
                'recall': recall_val,
                'f1_score': f1_val,
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

def normalize_scores(scores):
    """Normalize scores to [0, 1] range."""
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

def main():
    """Create and evaluate ensemble models."""
    print("🎭 Ensemble Model Creation and Evaluation")
    print("=" * 50)
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    
    # Load test data
    print("📥 Loading test data...")
    std_data = np.load(data_dir / "test_standardized.npz")
    X_test_std, y_test = std_data['X'], std_data['y']
    
    print(f"   Test data: {len(y_test):,} samples")
    print(f"   Labels: Benign={np.sum(y_test==0):,}, Attack={np.sum(y_test==1):,}")
    
    # Load models and get individual predictions
    print(f"\n🔮 Getting individual model predictions...")
    
    # 1. Isolation Forest
    print("   Loading Isolation Forest...")
    if_model = IForestModel.load(models_dir / "isolation_forest_tuned.pkl")
    if_scores = if_model.score(X_test_std)
    if_scores_norm = normalize_scores(if_scores)
    
    # 2. OneClassSVM
    print("   Loading OneClassSVM...")
    ocsvm_model = OCSVMModel.load(models_dir / "oneclass_svm_optimized.pkl")
    ocsvm_scores = ocsvm_model.score(X_test_std)
    ocsvm_scores_norm = normalize_scores(ocsvm_scores)
    
    # 3. Improved Autoencoder
    print("   Loading Improved Autoencoder...")
    ae_model = tf.keras.models.load_model(models_dir / "autoencoder_improved.keras")
    ae_predictions = ae_model.predict(X_test_std, batch_size=1024, verbose=0)
    ae_scores = np.mean((X_test_std - ae_predictions)**2, axis=1)
    ae_scores_norm = normalize_scores(ae_scores)
    
    print(f"   Individual model scores collected.")
    
    # Create ensemble methods
    print(f"\n🎭 Creating ensemble predictions...")
    
    # Method 1: Simple Average
    simple_avg_scores = (if_scores_norm + ocsvm_scores_norm + ae_scores_norm) / 3
    
    # Method 2: Weighted Average (based on individual ROC-AUC performance)
    # Weights based on individual performance: OneClassSVM (0.9788), AE (0.8228), IF (0.5610)
    weights = np.array([0.5610, 0.9788, 0.8228])
    weights = weights / weights.sum()  # Normalize weights
    weighted_avg_scores = (if_scores_norm * weights[0] + 
                          ocsvm_scores_norm * weights[1] + 
                          ae_scores_norm * weights[2])
    
    # Method 3: Best Two Models (OneClassSVM + Autoencoder)
    best_two_scores = (ocsvm_scores_norm + ae_scores_norm) / 2
    
    # Method 4: Learned Ensemble (Logistic Regression on validation set)
    print("   Training learned ensemble...")
    
    # Use a small portion of benign data for training the ensemble
    val_data = np.load(data_dir / "val_standardized.npz")
    X_val_std, y_val = val_data['X'], val_data['y']
    
    # Get validation predictions
    if_val_scores = normalize_scores(if_model.score(X_val_std))
    ocsvm_val_scores = normalize_scores(ocsvm_model.score(X_val_std))
    ae_val_predictions = ae_model.predict(X_val_std, batch_size=1024, verbose=0)
    ae_val_scores = normalize_scores(np.mean((X_val_std - ae_val_predictions)**2, axis=1))
    
    # Create feature matrix for ensemble learning
    # Since validation set only has benign samples (y_val all 0s), 
    # we'll create synthetic anomalies by using high-scoring validation samples as anomalies
    val_features = np.column_stack([if_val_scores, ocsvm_val_scores, ae_val_scores])
    
    # Create labels: top 10% highest average scores as "anomalies" for training
    avg_val_scores = val_features.mean(axis=1)
    threshold_90 = np.percentile(avg_val_scores, 90)
    y_val_synthetic = (avg_val_scores > threshold_90).astype(int)
    
    # Train logistic regression ensemble
    ensemble_lr = LogisticRegression(random_state=42)
    ensemble_lr.fit(val_features, y_val_synthetic)
    
    # Apply learned ensemble to test set
    test_features = np.column_stack([if_scores_norm, ocsvm_scores_norm, ae_scores_norm])
    learned_scores = ensemble_lr.predict_proba(test_features)[:, 1]
    
    # Evaluate all ensemble methods
    ensemble_results = {}
    
    methods = [
        ("Simple Average", simple_avg_scores),
        ("Weighted Average", weighted_avg_scores),
        ("Best Two Models", best_two_scores),
        ("Learned Ensemble", learned_scores)
    ]
    
    best_ensemble = None
    best_auc = 0
    
    for method_name, scores in methods:
        results = evaluate_ensemble(y_test, scores, method_name)
        ensemble_results[method_name.lower().replace(" ", "_")] = results
        
        if results['roc_auc'] > best_auc:
            best_auc = results['roc_auc']
            best_ensemble = method_name
    
    # Individual model performance for comparison
    print(f"\n📊 Individual Model Comparison:")
    print(f"   Isolation Forest:    ROC-AUC = {roc_auc_score(y_test, if_scores_norm):.4f}")
    print(f"   OneClassSVM:         ROC-AUC = {roc_auc_score(y_test, ocsvm_scores_norm):.4f}")
    print(f"   Improved Autoencoder: ROC-AUC = {roc_auc_score(y_test, ae_scores_norm):.4f}")
    
    # Final comparison
    print(f"\n🏆 ENSEMBLE COMPARISON")
    print("=" * 40)
    
    print(f"{'Ensemble Method':<20} {'ROC-AUC':<10} {'Best F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    
    for method_name, scores in methods:
        key = method_name.lower().replace(" ", "_")
        result = ensemble_results[key]
        auc = result['roc_auc']
        f1 = result['best_metrics']['f1_score']
        precision = result['best_metrics']['precision']
        recall = result['best_metrics']['recall']
        
        print(f"{method_name:<20} {auc:<10.4f} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f}")
    
    print(f"\n🏆 Best Ensemble: {best_ensemble} (ROC-AUC: {best_auc:.4f})")
    
    # Performance improvement analysis
    individual_max = max(
        roc_auc_score(y_test, if_scores_norm),
        roc_auc_score(y_test, ocsvm_scores_norm), 
        roc_auc_score(y_test, ae_scores_norm)
    )
    
    improvement = best_auc - individual_max
    print(f"📈 Improvement over best individual: {improvement:+.4f} ({improvement/individual_max*100:+.1f}%)")
    
    # Save ensemble results
    final_results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'test_data': {
            'total_samples': int(len(y_test)),
            'benign_samples': int(np.sum(y_test == 0)),
            'attack_samples': int(np.sum(y_test == 1))
        },
        'individual_performance': {
            'isolation_forest': float(roc_auc_score(y_test, if_scores_norm)),
            'oneclass_svm': float(roc_auc_score(y_test, ocsvm_scores_norm)),
            'improved_autoencoder': float(roc_auc_score(y_test, ae_scores_norm))
        },
        'ensemble_results': ensemble_results,
        'best_ensemble': {
            'method': best_ensemble,
            'roc_auc': best_auc,
            'improvement_over_best_individual': improvement
        },
        'ensemble_weights': {
            'isolation_forest': float(weights[0]),
            'oneclass_svm': float(weights[1]),
            'autoencoder': float(weights[2])
        }
    }
    
    results_path = models_dir / "ensemble_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"📋 Ensemble results saved: {results_path}")
    
    # Save best ensemble model predictions
    best_method_scores = next(scores for name, scores in methods if name == best_ensemble)
    np.save(models_dir / "best_ensemble_scores.npy", best_method_scores)
    print(f"💾 Best ensemble scores saved: {models_dir / 'best_ensemble_scores.npy'}")
    
    # Final recommendations
    print(f"\n💡 Final Recommendations:")
    if best_auc > 0.95:
        print(f"   🎉 Excellent ensemble performance! Ready for production deployment.")
    elif best_auc > 0.90:
        print(f"   👍 Very good ensemble performance. Consider deployment with monitoring.")
    elif best_auc > 0.80:
        print(f"   🤔 Good ensemble performance. May need further tuning for production.")
    else:
        print(f"   ⚠️ Moderate ensemble performance. Consider additional feature engineering.")
    
    if improvement > 0:
        print(f"   📈 Ensemble provides improvement over individual models.")
    else:
        print(f"   📊 Best individual model (OneClassSVM) may be sufficient.")
    
    print(f"\n✅ Ensemble evaluation completed!")
    return final_results

if __name__ == "__main__":
    main()