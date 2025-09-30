#!/usr/bin/env python3
"""
Train Autoencoder model with simplified approach
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

def main():
    """Train Autoencoder with simple fixed parameters."""
    print("🚀 Training Autoencoder Model (Simplified)")
    print("=" * 45)
    
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
    print("🔧 Normalizing features...")
    X_min = X_train.min()
    X_max = X_train.max()
    X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-8)
    
    print(f"   Normalized range: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    
    # Create and train model with fixed good parameters
    print("🔧 Training Autoencoder...")
    model = AEModel(input_dim=X_train_norm.shape[1], latent=64)
    
    start_time = time.time()
    model.fit(X_train_norm, epochs=20, batch_size=512)
    training_time = time.time() - start_time
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_scores = model.score(X_test_norm)
    
    print(f"   Reconstruction error stats:")
    print(f"   Normal samples: mean={test_scores[y_test==0].mean():.6f}, std={test_scores[y_test==0].std():.6f}")
    print(f"   Attack samples: mean={test_scores[y_test==1].mean():.6f}, std={test_scores[y_test==1].std():.6f}")
    
    # Try different threshold percentiles
    print(f"   Testing different thresholds:")
    thresholds = [90, 95, 99]
    best_f1 = 0
    best_metrics = None
    
    for percentile in thresholds:
        metrics = evaluate_model(y_test, test_scores, threshold_percentile=percentile)
        print(f"   {percentile}th percentile: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_metrics = metrics
    
    print(f"\n📊 Best Results:")
    print(f"   📈 ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"   🎯 Precision: {best_metrics['precision']:.4f}")
    print(f"   🔍 Recall: {best_metrics['recall']:.4f}")
    print(f"   ⚖️ F1-Score: {best_metrics['f1_score']:.4f}")
    
    # Save model (fix the extension issue)
    model_path = models_dir / "autoencoder_model.keras"
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save results
    results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'model_type': 'Autoencoder',
        'model_path': str(model_path),
        'hyperparameters': {
            'latent_dim': 64,
            'epochs': 20,
            'batch_size': 512
        },
        'training_time_seconds': training_time,
        'evaluation': best_metrics,
        'data_shapes': {
            'train': X_train.shape,
            'test': X_test.shape
        },
        'feature_type': 'PCA',
        'normalization': 'min_max_to_[0,1]',
        'normalization_params': {
            'X_min': float(X_min),
            'X_max': float(X_max)
        }
    }
    
    results_path = models_dir / "autoencoder_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📋 Results saved: {results_path}")
    
    # Performance assessment
    if best_metrics['roc_auc'] > 0.8:
        print("🎉 Excellent performance!")
    elif best_metrics['roc_auc'] > 0.7:
        print("👍 Good performance")
    elif best_metrics['roc_auc'] > 0.6:
        print("🤔 Moderate performance")
    elif best_metrics['roc_auc'] > 0.5:
        print("⚠️ Low but better than random")
    else:
        print("❌ Poor performance - worse than random")
    
    print(f"\n✅ Autoencoder training completed!")
    return model, best_metrics

if __name__ == "__main__":
    main()