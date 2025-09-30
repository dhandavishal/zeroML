#!/usr/bin/env python3
"""
Train and tune OneClassSVM model for anomaly detection
"""

import numpy as np
import json
import time
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
import joblib

# Add src to path for imports
import sys
sys.path.append('src')

def create_ocsvm_model(gamma='scale', nu=0.05, kernel='rbf'):
    """
    Create OneClassSVM model without additional scaling 
    (since we already have standardized features)
    """
    return OneClassSVM(
        gamma=gamma,
        nu=nu,
        kernel=kernel
    )

def evaluate_ocsvm(model, X_test, y_test):
    """Evaluate OneClassSVM performance."""
    # Get decision function scores (higher = more normal)
    decision_scores = model.decision_function(X_test)
    
    # For anomaly detection, we want higher scores for anomalies
    # OneClassSVM gives higher scores for normal samples, so we negate
    anomaly_scores = -decision_scores
    
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(y_test, anomaly_scores)
    
    # Get predictions (-1 for outliers, 1 for inliers)
    predictions = model.predict(X_test)
    # Convert to binary (0 for normal, 1 for anomaly)
    binary_preds = (predictions == -1).astype(int)
    
    # Calculate confusion matrix
    tp = np.sum((y_test == 1) & (binary_preds == 1))
    fp = np.sum((y_test == 0) & (binary_preds == 1))
    tn = np.sum((y_test == 0) & (binary_preds == 0))
    fn = np.sum((y_test == 1) & (binary_preds == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'anomaly_scores': anomaly_scores,
        'decision_scores': decision_scores
    }

def hyperparameter_tuning(X_train, X_test, y_test):
    """
    Perform hyperparameter tuning for OneClassSVM.
    Note: We'll use a smaller subset for tuning due to SVM's computational complexity.
    """
    print("🎯 OneClassSVM Hyperparameter Tuning")
    print("🔧 Starting hyperparameter tuning...")
    
    # Use subset for tuning (SVM is computationally expensive)
    n_train_subset = min(10000, len(X_train))
    train_indices = np.random.choice(len(X_train), n_train_subset, replace=False)
    X_train_subset = X_train[train_indices]
    
    n_test_subset = min(5000, len(X_test))
    test_indices = np.random.choice(len(X_test), n_test_subset, replace=False)
    X_test_subset = X_test[test_indices]
    y_test_subset = y_test[test_indices]
    
    print(f"📊 Using subsets: Train {X_train_subset.shape}, Test {X_test_subset.shape}")
    
    # Parameter grid (smaller due to computational cost)
    param_grid = {
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'nu': [0.01, 0.05, 0.1, 0.15, 0.2],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    best_score = 0
    best_params = None
    results = []
    
    total_combinations = len(param_grid['gamma']) * len(param_grid['nu']) * len(param_grid['kernel'])
    print(f"🎯 Testing {total_combinations} parameter combinations...")
    
    combination = 0
    for gamma in param_grid['gamma']:
        for nu in param_grid['nu']:
            for kernel in param_grid['kernel']:
                combination += 1
                
                try:
                    print(f"\n[{combination}/{total_combinations}] Testing:")
                    print(f"   gamma={gamma}, nu={nu}, kernel={kernel}")
                    
                    start_time = time.time()
                    
                    # Create and train model
                    model = create_ocsvm_model(gamma=gamma, nu=nu, kernel=kernel)
                    model.fit(X_train_subset)
                    
                    # Evaluate
                    metrics = evaluate_ocsvm(model, X_test_subset, y_test_subset)
                    train_time = time.time() - start_time
                    
                    print(f"   📈 ROC-AUC: {metrics['roc_auc']:.4f}")
                    print(f"   🎯 F1: {metrics['f1_score']:.4f}")
                    print(f"   ⏱️ Time: {train_time:.2f}s")
                    
                    # Store result
                    result = {
                        'gamma': gamma,
                        'nu': nu, 
                        'kernel': kernel,
                        'train_time': train_time,
                        'roc_auc': metrics['roc_auc'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score'],
                        'tp': metrics['tp'],
                        'fp': metrics['fp'],
                        'tn': metrics['tn'],
                        'fn': metrics['fn']
                    }
                    results.append(result)
                    
                    # Track best
                    if metrics['roc_auc'] > best_score:
                        best_score = metrics['roc_auc']
                        best_params = result.copy()
                        print(f"   🏆 New best! ROC-AUC: {best_score:.4f}")
                
                except Exception as e:
                    print(f"   ❌ Failed: {str(e)}")
                    continue
    
    return best_params, results

def train_final_model(X_train, best_params):
    """Train final OneClassSVM model with best parameters on full training set."""
    print(f"\n🚀 Training final OneClassSVM model with best parameters...")
    print(f"   Parameters: {best_params}")
    
    start_time = time.time()
    
    model = create_ocsvm_model(
        gamma=best_params['gamma'],
        nu=best_params['nu'],
        kernel=best_params['kernel']
    )
    
    model.fit(X_train)
    training_time = time.time() - start_time
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    return model, training_time

def comprehensive_evaluation(model, X_test, y_test, model_name="OneClassSVM"):
    """Detailed evaluation similar to the Isolation Forest evaluation."""
    print(f"\n📊 Comprehensive Evaluation: {model_name}")
    print("=" * 50)
    
    # Get scores
    metrics = evaluate_ocsvm(model, X_test, y_test)
    anomaly_scores = metrics['anomaly_scores']
    
    # Score statistics
    normal_scores = anomaly_scores[y_test == 0]
    attack_scores = anomaly_scores[y_test == 1]
    
    print(f"Score Statistics:")
    print(f"   Normal samples: {len(normal_scores):,}")
    print(f"   Attack samples: {len(attack_scores):,}")
    print(f"   Normal scores - mean: {normal_scores.mean():.4f}, std: {normal_scores.std():.4f}")
    print(f"   Attack scores - mean: {attack_scores.mean():.4f}, std: {attack_scores.std():.4f}")
    print(f"   📈 ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Try different thresholds for detailed analysis
    thresholds = [
        ("50th percentile normal", np.percentile(normal_scores, 50)),
        ("90th percentile normal", np.percentile(normal_scores, 90)),
        ("95th percentile normal", np.percentile(normal_scores, 95)),
        ("99th percentile normal", np.percentile(normal_scores, 99)),
        ("Default SVM threshold", 0.0),  # SVM decision boundary
        ("Mean normal", normal_scores.mean()),
        ("Mean + 1 std", normal_scores.mean() + normal_scores.std())
    ]
    
    print(f"\nThreshold Analysis:")
    best_f1 = 0
    best_threshold_info = None
    
    for name, threshold in thresholds:
        y_pred = (anomaly_scores > threshold).astype(int)
        
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"   {name:25} (th={threshold:.4f}): P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, FPR={fpr:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_info = {
                'name': name,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'fpr': fpr,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
    
    print(f"\n🏆 Best threshold: {best_threshold_info['threshold']:.4f} (F1={best_f1:.4f})")
    
    return {
        'roc_auc': metrics['roc_auc'],
        'default_metrics': metrics,  # Using SVM's default threshold
        'best_threshold_metrics': best_threshold_info,
        'score_stats': {
            'normal_mean': float(normal_scores.mean()),
            'normal_std': float(normal_scores.std()),
            'attack_mean': float(attack_scores.mean()),
            'attack_std': float(attack_scores.std())
        }
    }

def main():
    """Main OneClassSVM training pipeline."""
    print("🚀 Starting OneClassSVM Training Pipeline")
    print("=" * 50)
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load standardized features
    print("📥 Loading standardized features...")
    train_data = np.load(data_dir / "train_standardized.npz")
    X_train, y_train = train_data['X'], train_data['y']
    
    test_data = np.load(data_dir / "test_standardized.npz")
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"   Train: {X_train.shape}, labels: {np.unique(y_train)}")
    print(f"   Test: {X_test.shape}, labels: {np.unique(y_test)}")
    
    # Hyperparameter tuning
    best_params, tuning_results = hyperparameter_tuning(X_train, X_test, y_test)
    
    print(f"\n🏆 Best parameters found:")
    for key, value in best_params.items():
        if key not in ['tp', 'fp', 'tn', 'fn']:
            print(f"   {key}: {value}")
    
    # Train final model
    final_model, training_time = train_final_model(X_train, best_params)
    
    # Comprehensive evaluation
    evaluation = comprehensive_evaluation(final_model, X_test, y_test)
    
    # Save model
    model_path = models_dir / "oneclass_svm.pkl"
    joblib.dump(final_model, model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save results
    results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'model_type': 'OneClassSVM',
        'model_path': str(model_path),
        'training_time_seconds': training_time,
        'best_hyperparameters': best_params,
        'hyperparameter_tuning_results': tuning_results,
        'evaluation': evaluation,
        'data_shapes': {
            'train': X_train.shape,
            'test': X_test.shape
        }
    }
    
    results_path = models_dir / "oneclass_svm_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📋 Results saved: {results_path}")
    
    # Performance summary
    print(f"\n🎯 ONECLASS SVM RESULTS SUMMARY")
    print("=" * 40)
    print(f"📈 ROC-AUC: {evaluation['roc_auc']:.4f}")
    print(f"🎯 Best F1-Score: {evaluation['best_threshold_metrics']['f1_score']:.4f}")
    print(f"🔍 Best Recall: {evaluation['best_threshold_metrics']['recall']:.4f}")
    print(f"🎯 Best Precision: {evaluation['best_threshold_metrics']['precision']:.4f}")
    
    # Compare with baseline (random)
    if evaluation['roc_auc'] > 0.6:
        print("👍 Good performance!")
    elif evaluation['roc_auc'] > 0.5:
        print("🤔 Moderate performance - better than random")
    else:
        print("⚠️ Poor performance - worse than random")
    
    print(f"\n✅ OneClassSVM training completed!")
    return final_model, results

if __name__ == "__main__":
    main()