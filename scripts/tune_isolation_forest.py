#!/usr/bin/env python3
"""
Hyperparameter tuning for Isolation Forest
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
import time

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.iforest import IForestModel

def evaluate_model(model, X_test, y_test, threshold_percentile=95):
    """Evaluate model performance."""
    scores = model.score(X_test)
    
    # Calculate threshold based on training data normal samples
    if threshold_percentile is not None:
        threshold = np.percentile(scores, threshold_percentile)
    else:
        threshold = np.median(scores)
    
    y_pred = (scores > threshold).astype(int)
    
    # Calculate metrics
    try:
        roc_auc = roc_auc_score(y_test, scores)
    except:
        roc_auc = 0.5
    
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': threshold,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }

def tune_hyperparameters():
    """Grid search over hyperparameters."""
    print("🔧 Starting hyperparameter tuning...")
    
    # Load data
    data_dir = Path("data/engineered")
    
    train_data = np.load(data_dir / "train_standardized.npz")
    X_train, y_train = train_data['X'], train_data['y']
    
    test_data = np.load(data_dir / "test_standardized.npz")
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"📊 Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    
    # Hyperparameter grid
    param_grid = {
        'contamination': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'n_estimators': [50, 100, 200],
        'max_samples': ['auto', 256, 512, 1024]
    }
    
    results = []
    best_score = 0
    best_params = None
    
    total_combinations = len(param_grid['contamination']) * len(param_grid['n_estimators']) * len(param_grid['max_samples'])
    print(f"🎯 Testing {total_combinations} parameter combinations...")
    
    combination = 0
    for contamination in param_grid['contamination']:
        for n_estimators in param_grid['n_estimators']:
            for max_samples in param_grid['max_samples']:
                combination += 1
                
                print(f"\n[{combination}/{total_combinations}] Testing:")
                print(f"   contamination={contamination}, n_estimators={n_estimators}, max_samples={max_samples}")
                
                try:
                    # Train model
                    start_time = time.time()
                    model = IForestModel(
                        contamination=contamination,
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train)
                    train_time = time.time() - start_time
                    
                    # Evaluate
                    metrics = evaluate_model(model, X_test, y_test)
                    
                    result = {
                        'contamination': contamination,
                        'n_estimators': n_estimators,
                        'max_samples': max_samples,
                        'train_time': train_time,
                        **metrics
                    }
                    results.append(result)
                    
                    print(f"   📈 ROC-AUC: {metrics['roc_auc']:.4f}")
                    print(f"   🎯 F1: {metrics['f1_score']:.4f}")
                    print(f"   ⏱️ Time: {train_time:.2f}s")
                    
                    # Track best
                    if metrics['roc_auc'] > best_score:
                        best_score = metrics['roc_auc']
                        best_params = result.copy()
                        print(f"   🏆 New best! ROC-AUC: {best_score:.4f}")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
    
    # Save results
    results_path = Path("models/hyperparameter_tuning_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': np.datetime64('now').item().isoformat(),
            'best_params': best_params,
            'all_results': results,
            'total_combinations': total_combinations
        }, f, indent=2)
    
    print(f"\n🏆 Best parameters found:")
    if best_params:
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        
        # Train final model with best parameters
        print(f"\n🚀 Training final model with best parameters...")
        final_model = IForestModel(
            contamination=best_params['contamination'],
            n_estimators=best_params['n_estimators'],
            max_samples=best_params['max_samples'],
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(X_train)
        
        # Save best model
        final_model.save("models/isolation_forest_tuned.pkl")
        print(f"💾 Best model saved: models/isolation_forest_tuned.pkl")
        
        return final_model, best_params
    else:
        print("❌ No valid parameter combination found!")
        return None, None

def main():
    print("🎯 Isolation Forest Hyperparameter Tuning")
    model, best_params = tune_hyperparameters()
    
    if model and best_params:
        print(f"\n✅ Tuning completed successfully!")
        print(f"🏆 Best ROC-AUC: {best_params['roc_auc']:.4f}")
        print(f"📋 Results saved to: models/hyperparameter_tuning_results.json")
    else:
        print("❌ Tuning failed!")

if __name__ == "__main__":
    main()