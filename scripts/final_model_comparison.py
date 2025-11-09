#!/usr/bin/env python3
"""
Final comprehensive comparison: OneClassSVM vs Improved Autoencoder vs Improved Isolation Forest
Plus ensemble creation
"""

import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.iforest import IForestModel
from zeroml.models.ocsvm import OCSVMModel

def evaluate_model_comprehensive(y_true, scores, model_name):
    """Comprehensive evaluation with multiple thresholds."""
    print(f"\n🔍 {model_name}")
    print("-" * 50)
    
    # Score statistics
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    
    roc_auc = roc_auc_score(y_true, scores)
    print(f"   📈 ROC-AUC: {roc_auc:.4f}")
    print(f"   📊 Score ranges - Normal: [{normal_scores.min():.4f}, {normal_scores.max():.4f}]")
    print(f"   📊 Score ranges - Attack: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    
    # Test different FPR targets
    fpr_targets = [0.01, 0.02, 0.05]
    best_metrics = None
    best_f1 = 0
    
    for target_fpr in fpr_targets:
        threshold = np.percentile(normal_scores, (1-target_fpr)*100)
        y_pred = (scores > threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"   FPR {target_fpr:.1%}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (actual FPR={actual_fpr:.3f})")
        
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'fpr': actual_fpr,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
    
    return {
        'roc_auc': roc_auc,
        'best_metrics': best_metrics,
        'raw_scores': scores
    }

def create_ensemble_predictions(predictions_dict, weights=None):
    """Create ensemble predictions using weighted averaging."""
    if weights is None:
        # Weight by ROC-AUC performance (higher AUC = higher weight)
        aucs = [pred['roc_auc'] for pred in predictions_dict.values()]
        weights = np.array(aucs) / np.sum(aucs)
    
    # Normalize all scores to [0, 1] for fair combination
    normalized_scores = {}
    for name, pred in predictions_dict.items():
        raw_scores = pred['raw_scores']
        norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
        normalized_scores[name] = norm_scores
    
    # Create weighted ensemble
    model_names = list(normalized_scores.keys())
    ensemble_scores = np.zeros_like(list(normalized_scores.values())[0])
    
    for i, name in enumerate(model_names):
        ensemble_scores += weights[i] * normalized_scores[name]
        print(f"   {name}: weight = {weights[i]:.3f}")
    
    return ensemble_scores, weights

def main():
    """Final comprehensive model comparison and ensemble creation."""
    print("🏆 FINAL MODEL COMPARISON & ENSEMBLE")
    print("=" * 60)
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    
    # Load test data
    print("📥 Loading test data...")
    std_data = np.load(data_dir / "test_standardized.npz")
    X_test_std, y_test = std_data['X'], std_data['y']
    
    print(f"   Test samples: {len(y_test):,}")
    print(f"   Benign: {np.sum(y_test==0):,}, Attack: {np.sum(y_test==1):,}")
    
    # Load and evaluate all models
    print(f"\n🎯 MODEL EVALUATIONS")
    print("=" * 40)
    
    predictions = {}
    
    # 1. OneClassSVM (Best performer)
    print("Loading OneClassSVM...")
    ocsvm_model = OCSVMModel.load(models_dir / "oneclass_svm_optimized.pkl")
    ocsvm_scores = ocsvm_model.score(X_test_std)
    predictions['OneClassSVM'] = evaluate_model_comprehensive(y_test, ocsvm_scores, "OneClassSVM")
    
    # 2. Improved Autoencoder
    print("Loading Improved Autoencoder...")
    ae_model = tf.keras.models.load_model(models_dir / "autoencoder_improved.keras")
    ae_predictions = ae_model.predict(X_test_std, batch_size=1024, verbose=0)
    ae_scores = ((X_test_std - ae_predictions)**2).mean(axis=1)
    predictions['Autoencoder'] = evaluate_model_comprehensive(y_test, ae_scores, "Improved Autoencoder")
    
    # 3. Improved Isolation Forest
    print("Loading Improved Isolation Forest...")
    if_model = IForestModel.load(models_dir / "isolation_forest_improved.pkl")
    if_scores = if_model.score(X_test_std)
    predictions['IsolationForest'] = evaluate_model_comprehensive(y_test, if_scores, "Improved Isolation Forest")
    
    # Create ensemble
    print(f"\n🤝 ENSEMBLE CREATION")
    print("=" * 30)
    print(f"Creating weighted ensemble based on ROC-AUC performance...")
    
    ensemble_scores, weights = create_ensemble_predictions(predictions)
    predictions['Ensemble'] = evaluate_model_comprehensive(y_test, ensemble_scores, "Weighted Ensemble")
    
    # Final rankings
    print(f"\n🏆 FINAL RANKINGS")
    print("=" * 30)
    
    # Sort by ROC-AUC
    rankings = [(name, pred['roc_auc'], pred['best_metrics']['f1_score']) 
                for name, pred in predictions.items()]
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<6} {'Model':<20} {'ROC-AUC':<10} {'Best F1':<10} {'Performance'}")
    print("-" * 70)
    
    medals = ["🥇", "🥈", "🥉", "🏅"]
    for i, (name, auc, f1) in enumerate(rankings):
        medal = medals[i] if i < len(medals) else "  "
        performance = ("Excellent" if auc > 0.9 else 
                      "Very Good" if auc > 0.8 else 
                      "Good" if auc > 0.7 else 
                      "Moderate" if auc > 0.6 else "Poor")
        
        print(f"{medal} {i+1:<4} {name:<20} {auc:<10.4f} {f1:<10.4f} {performance}")
    
    # Performance improvements
    print(f"\n📈 PERFORMANCE IMPROVEMENTS")
    print("-" * 40)
    baseline_auc = 0.5610  # Original Isolation Forest
    
    for name, auc, _ in rankings[:-1]:  # Exclude IF from comparison
        if name != "IsolationForest":
            improvement = auc - baseline_auc
            print(f"   {name}: +{improvement:.4f} ({improvement/baseline_auc*100:.1f}% improvement)")
    
    # Save comprehensive results
    final_results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'test_data_info': {
            'total_samples': int(len(y_test)),
            'benign_samples': int(np.sum(y_test == 0)),
            'attack_samples': int(np.sum(y_test == 1)),
            'features': X_test_std.shape[1]
        },
        'model_performances': {
            name: {
                'roc_auc': pred['roc_auc'],
                'best_f1': pred['best_metrics']['f1_score'],
                'best_precision': pred['best_metrics']['precision'],
                'best_recall': pred['best_metrics']['recall'],
                'best_fpr': pred['best_metrics']['fpr']
            }
            for name, pred in predictions.items()
        },
        'ensemble_info': {
            'weights': {name: float(weight) for name, weight in zip(predictions.keys(), weights) if name != 'Ensemble'},
            'method': 'roc_auc_weighted_average'
        },
        'rankings': [{'rank': i+1, 'model': name, 'roc_auc': auc, 'f1_score': f1} 
                    for i, (name, auc, f1) in enumerate(rankings)],
        'best_model': rankings[0][0],
        'best_roc_auc': rankings[0][1]
    }
    
    results_path = models_dir / "final_comprehensive_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"📋 Final results saved: {results_path}")
    
    # Recommendations
    print(f"\n💡 FINAL RECOMMENDATIONS")
    print("-" * 35)
    
    best_model, best_auc, _ = rankings[0]
    
    if best_auc > 0.95:
        print(f"🎉 Outstanding performance! {best_model} is ready for production")
    elif best_auc > 0.9:
        print(f"🎉 Excellent performance! {best_model} shows strong anomaly detection capability")
    elif best_auc > 0.8:
        print(f"👍 Very good performance! {best_model} is suitable for most use cases")
    else:
        print(f"🤔 Performance is acceptable but consider further optimization")
    
    print(f"\n🚀 Deployment Strategy:")
    print(f"   1. Primary Model: {rankings[0][0]} (ROC-AUC: {rankings[0][1]:.4f})")
    print(f"   2. Backup Model: {rankings[1][0]} (ROC-AUC: {rankings[1][1]:.4f})")
    print(f"   3. Ensemble Option: Available with {predictions['Ensemble']['roc_auc']:.4f} ROC-AUC")
    
    print(f"\n✅ Comprehensive model comparison and ensemble creation completed!")
    return final_results

if __name__ == "__main__":
    main()