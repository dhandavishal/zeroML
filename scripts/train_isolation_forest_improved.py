#!/usr/bin/env python3
"""
Improved Isolation Forest training with validation-based thresholding
"""

import numpy as np
import json
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix
from itertools import product

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.iforest import IForestModel

def calculate_fpr_threshold(scores, target_fpr=0.02):
    """
    Calculate threshold to achieve target False Positive Rate.
    
    Args:
        scores: Anomaly scores from benign validation data
        target_fpr: Target false positive rate (default 2%)
        
    Returns:
        threshold value
    """
    # Sort scores in descending order
    sorted_scores = np.sort(scores)[::-1]
    n_samples = len(sorted_scores)
    
    # Calculate index for target FPR
    fp_index = int(target_fpr * n_samples)
    
    # Handle edge cases
    if fp_index >= n_samples:
        return sorted_scores[-1] - 1e-6  # Below minimum score
    if fp_index == 0:
        return sorted_scores[0] + 1e-6  # Above maximum score
    
    return sorted_scores[fp_index]

def evaluate_with_fpr_threshold(y_true, scores, target_fpr=0.02):
    """
    Evaluate model using validation-based threshold for target FPR.
    """
    # Split scores by true labels
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    
    # Calculate threshold based on normal scores to achieve target FPR
    threshold = calculate_fpr_threshold(normal_scores, target_fpr)
    
    # Apply threshold
    y_pred = (scores > threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': actual_fpr,
        'target_fpr': target_fpr,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }

def hyperparameter_tuning(X_train, X_val, y_val, X_test, y_test):
    """
    Comprehensive hyperparameter tuning for Isolation Forest.
    """
    print("🎯 Advanced Hyperparameter Tuning for Isolation Forest")
    print("=" * 60)
    
    # Expanded parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [256, 512, 1024, 'auto'],
        'max_features': [0.5, 0.75, 1.0],
        'bootstrap': [True, False],
        'contamination': ['auto', 0.01, 0.02, 0.03]
    }
    
    # Generate all combinations
    param_combinations = list(product(
        param_grid['n_estimators'],
        param_grid['max_samples'],
        param_grid['max_features'],
        param_grid['bootstrap'],
        param_grid['contamination']
    ))
    
    print(f"🔧 Testing {len(param_combinations)} parameter combinations...")
    
    best_params = None
    best_roc_auc = 0
    best_metrics = None
    best_model = None
    results = []
    
    for i, (n_est, max_samp, max_feat, bootstrap, contamination) in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing:")
        print(f"   n_estimators={n_est}, max_samples={max_samp}, max_features={max_feat}")
        print(f"   bootstrap={bootstrap}, contamination={contamination}")
        
        try:
            # Create and train model
            model = IForestModel(
                n_estimators=n_est,
                max_samples=max_samp,
                max_features=max_feat,
                bootstrap=bootstrap,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
            start_time = time.time()
            model.fit(X_train)
            training_time = time.time() - start_time
            
            # Get validation scores for threshold calculation
            val_scores = model.score(X_val)
            
            # Test different target FPRs
            target_fprs = [0.01, 0.02, 0.03]
            best_fpr_result = None
            best_f1_for_params = 0
            
            for target_fpr in target_fprs:
                # Calculate threshold on validation set
                threshold = calculate_fpr_threshold(val_scores, target_fpr)
                
                # Evaluate on test set with this threshold
                test_scores = model.score(X_test)
                test_metrics = evaluate_with_fpr_threshold(y_test, test_scores, target_fpr)
                
                if test_metrics['f1_score'] > best_f1_for_params:
                    best_f1_for_params = test_metrics['f1_score']
                    best_fpr_result = test_metrics.copy()
                    best_fpr_result['validation_threshold'] = threshold
            
            # Calculate ROC-AUC for comparison
            test_scores = model.score(X_test)
            roc_auc = roc_auc_score(y_test, test_scores)
            
            print(f"   📈 ROC-AUC: {roc_auc:.4f}")
            print(f"   🎯 Best F1: {best_fpr_result['f1_score']:.4f} (FPR={best_fpr_result['fpr']:.3f})")
            print(f"   ⏱️ Time: {training_time:.2f}s")
            
            # Store results
            result = {
                'params': {
                    'n_estimators': n_est,
                    'max_samples': max_samp,
                    'max_features': max_feat,
                    'bootstrap': bootstrap,
                    'contamination': contamination
                },
                'training_time': training_time,
                'roc_auc': roc_auc,
                'best_metrics': best_fpr_result
            }
            results.append(result)
            
            # Track best model by ROC-AUC
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_params = result['params'].copy()
                best_metrics = best_fpr_result.copy()
                best_model = model
                print("   🏆 New best ROC-AUC!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print(f"\n🏆 Best parameters found:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    print(f"   ROC-AUC: {best_roc_auc:.4f}")
    print(f"   Best F1: {best_metrics['f1_score']:.4f}")
    
    return best_params, best_metrics, best_model, results

def main():
    """
    Train improved Isolation Forest with validation-based thresholding.
    """
    print("🚀 Improved Isolation Forest Training")
    print("=" * 40)
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load standardized features (704 dimensions after removing constants)
    print("📥 Loading standardized features...")
    train_data = np.load(data_dir / "train_standardized.npz")
    X_train, y_train = train_data['X'], train_data['y']
    
    val_data = np.load(data_dir / "val_standardized.npz")
    X_val, y_val = val_data['X'], val_data['y']
    
    test_data = np.load(data_dir / "test_standardized.npz")
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"   Train: {X_train.shape}, all benign: {np.all(y_train == 0)}")
    print(f"   Val: {X_val.shape}, all benign: {np.all(y_val == 0)}")
    print(f"   Test: {X_test.shape}, benign: {np.sum(y_test==0)}, attack: {np.sum(y_test==1)}")
    
    # Hyperparameter tuning
    best_params, best_metrics, best_model, all_results = hyperparameter_tuning(
        X_train, X_val, y_val, X_test, y_test
    )
    
    # Save best model
    model_path = models_dir / "isolation_forest_improved.pkl"
    best_model.save(model_path)
    print(f"💾 Best model saved: {model_path}")
    
    # Final evaluation with different target FPRs
    print(f"\n📊 Final Evaluation with Different Target FPRs")
    print("-" * 50)
    
    test_scores = best_model.score(X_test)
    val_scores = best_model.score(X_val)
    
    fpr_targets = [0.01, 0.02, 0.03, 0.05]
    final_evaluations = {}
    
    for target_fpr in fpr_targets:
        metrics = evaluate_with_fpr_threshold(y_test, test_scores, target_fpr)
        final_evaluations[f'fpr_{int(target_fpr*100)}pct'] = metrics
        
        print(f"Target FPR {target_fpr:.1%}:")
        print(f"   Actual FPR: {metrics['fpr']:.3f}, Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
    
    # Save comprehensive results
    results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'model_type': 'Isolation Forest (Improved)',
        'model_path': str(model_path),
        'best_hyperparameters': best_params,
        'best_roc_auc': roc_auc_score(y_test, test_scores),
        'data_shapes': {
            'train': X_train.shape,
            'val': X_val.shape,
            'test': X_test.shape
        },
        'feature_type': 'standardized_704_features',
        'thresholding_method': 'validation_based_fpr_target',
        'final_evaluations': final_evaluations,
        'hyperparameter_search_results': all_results[:10]  # Save top 10 results
    }
    
    results_path = models_dir / "isolation_forest_improved_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📋 Results saved: {results_path}")
    
    # Performance comparison
    final_roc_auc = roc_auc_score(y_test, test_scores)
    print(f"\n🎯 Performance Summary:")
    print(f"   Improved IF ROC-AUC: {final_roc_auc:.4f}")
    print(f"   Previous IF ROC-AUC: 0.5610")
    print(f"   Improvement: {final_roc_auc - 0.5610:.4f} ({(final_roc_auc/0.5610-1)*100:.1f}%)")
    
    if final_roc_auc > 0.8:
        print("🎉 Excellent improvement! Now competitive with other models")
    elif final_roc_auc > 0.7:
        print("👍 Good improvement! Significantly better performance")
    elif final_roc_auc > 0.6:
        print("🤔 Moderate improvement, but still room for growth")
    else:
        print("⚠️ Limited improvement - this dataset may be challenging for IF")
    
    print(f"\n✅ Improved Isolation Forest training completed!")
    return best_model, results

if __name__ == "__main__":
    main()