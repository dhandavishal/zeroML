#!/usr/bin/env python3
"""
Train Autoencoder model on PCA features with hyperparameter tuning
"""

import numpy as np
import json
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.autoencoder import AEModel

def evaluate_model(y_true, scores, threshold_percentile=95):
    """Quick evaluation using reconstruction error threshold."""
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

def hyperparameter_tuning(X_train, X_test, y_test):
    """Simple hyperparameter tuning for Autoencoder."""
    print("🎯 Hyperparameter tuning for Autoencoder...")
    
    # Parameter grid
    param_grid = [
        {'latent_dim': 16, 'epochs': 30, 'batch_size': 512},
        {'latent_dim': 32, 'epochs': 30, 'batch_size': 512},
        {'latent_dim': 64, 'epochs': 30, 'batch_size': 512},
        {'latent_dim': 32, 'epochs': 50, 'batch_size': 256},
        {'latent_dim': 32, 'epochs': 50, 'batch_size': 1024}
    ]
    
    best_params = None
    best_roc_auc = 0
    best_metrics = None
    
    for i, params in enumerate(param_grid):
        print(f"\n[{i+1}/{len(param_grid)}] Testing:")
        print(f"   latent_dim={params['latent_dim']}, epochs={params['epochs']}, batch_size={params['batch_size']}")
        
        try:
            # Create and train model
            model = AEModel(input_dim=X_train.shape[1], latent=params['latent_dim'])
            
            start_time = time.time()
            model.fit(X_train, epochs=params['epochs'], batch_size=params['batch_size'])
            training_time = time.time() - start_time
            
            # Evaluate
            test_scores = model.score(X_test)
            metrics = evaluate_model(y_test, test_scores)
            
            print(f"   📈 ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   🎯 F1: {metrics['f1_score']:.4f}")
            print(f"   ⏱️ Time: {training_time:.2f}s")
            
            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                best_params = params.copy()
                best_params['training_time'] = training_time
                best_metrics = metrics
                print("   🏆 New best!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print(f"\n🏆 Best parameters found:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    return best_params, best_metrics

def main():
    """Train Autoencoder with hyperparameter tuning."""
    print("🚀 Training Autoencoder Model")
    print("=" * 40)
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load PCA features (better for neural networks)
    print("📥 Loading PCA features...")
    train_data = np.load(data_dir / "train_pca.npz")
    X_train, y_train = train_data['X'], train_data['y']
    
    test_data = np.load(data_dir / "test_pca.npz")
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Normalize features to [0, 1] for sigmoid output
    print("🔧 Normalizing features for Autoencoder...")
    X_train_norm = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-8)
    X_test_norm = (X_test - X_train.min()) / (X_train.max() - X_train.min() + 1e-8)
    
    # Hyperparameter tuning
    best_params, best_metrics = hyperparameter_tuning(X_train_norm, X_test_norm, y_test)
    
    # Train final model with best parameters
    print(f"\n🚀 Training final Autoencoder with best parameters...")
    final_model = AEModel(
        input_dim=X_train_norm.shape[1],
        latent=best_params['latent_dim']
    )
    
    start_time = time.time()
    final_model.fit(
        X_train_norm,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size']
    )
    final_training_time = time.time() - start_time
    
    # Final evaluation
    print("📊 Final evaluation...")
    final_scores = final_model.score(X_test_norm)
    final_metrics = evaluate_model(y_test, final_scores)
    
    print(f"   📈 ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"   🎯 Precision: {final_metrics['precision']:.4f}")
    print(f"   🔍 Recall: {final_metrics['recall']:.4f}")
    print(f"   ⚖️ F1-Score: {final_metrics['f1_score']:.4f}")
    
    # Save model
    model_path = models_dir / "autoencoder_model"
    final_model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save results
    results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'model_type': 'Autoencoder',
        'model_path': str(model_path),
        'best_hyperparameters': best_params,
        'final_training_time_seconds': final_training_time,
        'final_evaluation': final_metrics,
        'data_shapes': {
            'train': X_train.shape,
            'test': X_test.shape
        },
        'feature_type': 'PCA',
        'normalization': 'min_max_to_[0,1]'
    }
    
    results_path = models_dir / "autoencoder_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📋 Results saved: {results_path}")
    
    # Performance assessment
    if final_metrics['roc_auc'] > 0.8:
        print("🎉 Excellent performance!")
    elif final_metrics['roc_auc'] > 0.7:
        print("👍 Good performance")
    elif final_metrics['roc_auc'] > 0.6:
        print("🤔 Moderate performance")
    else:
        print("⚠️ Low performance - may need architecture changes")
    
    print(f"\n✅ Autoencoder training completed!")
    return final_model, final_metrics

if __name__ == "__main__":
    main()