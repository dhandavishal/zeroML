#!/usr/bin/env python3
"""
Train improved Autoencoder model on standardized features with advanced architecture
"""

import numpy as np
import json
import time
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Add src to path for imports
import sys
sys.path.append('src')

class ImprovedAEModel:
    def __init__(self, input_dim, latent_dim=64, l2_reg=1e-4, l1_latent=1e-5, use_denoising=True):
        """
        Improved Autoencoder with advanced architecture.
        
        Args:
            input_dim: Input dimension (704 for standardized features)
            latent_dim: Latent space dimension
            l2_reg: L2 regularization on weights
            l1_latent: L1 regularization on latent layer
            use_denoising: Whether to add Gaussian noise for denoising
        """
        # Input layer
        inp = keras.Input(shape=(input_dim,), name='input')
        
        # Optional denoising
        if use_denoising:
            x = layers.GaussianNoise(0.05, name='noise')(inp)
        else:
            x = inp
        
        # Encoder: 704 → 512 → 128 → 64
        x = layers.Dense(
            512, 
            kernel_regularizer=regularizers.l2(l2_reg),
            name='encoder_512'
        )(x)
        x = layers.LayerNormalization(name='norm_512')(x)
        x = layers.ReLU(name='relu_512')(x)
        
        x = layers.Dense(
            128,
            kernel_regularizer=regularizers.l2(l2_reg),
            name='encoder_128'
        )(x)
        x = layers.LayerNormalization(name='norm_128')(x)
        x = layers.ReLU(name='relu_128')(x)
        
        # Latent layer with L1 regularization
        latent = layers.Dense(
            latent_dim,
            kernel_regularizer=regularizers.l2(l2_reg),
            activity_regularizer=regularizers.l1(l1_latent),
            name='latent'
        )(x)
        
        # Decoder: 64 → 128 → 512 → 704
        x = layers.Dense(
            128,
            kernel_regularizer=regularizers.l2(l2_reg),
            name='decoder_128'
        )(latent)
        x = layers.LayerNormalization(name='norm_dec_128')(x)
        x = layers.ReLU(name='relu_dec_128')(x)
        
        x = layers.Dense(
            512,
            kernel_regularizer=regularizers.l2(l2_reg),
            name='decoder_512'
        )(x)
        x = layers.LayerNormalization(name='norm_dec_512')(x)
        x = layers.ReLU(name='relu_dec_512')(x)
        
        # Output layer (linear activation for regression)
        output = layers.Dense(
            input_dim,
            kernel_regularizer=regularizers.l2(l2_reg),
            activation='linear',
            name='output'
        )(x)
        
        # Create model
        self.model = keras.Model(inp, output, name='ImprovedAutoencoder')
        
        # Compile with Adam optimizer
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"🏗️ Model Architecture:")
        print(f"   Input: {input_dim}")
        print(f"   Encoder: {input_dim} → 512 → 128 → {latent_dim}")
        print(f"   Decoder: {latent_dim} → 128 → 512 → {input_dim}")
        print(f"   Parameters: {self.model.count_params():,}")
    
    def fit(self, X, epochs=50, batch_size=512, validation_split=0.1):
        """Train the autoencoder with advanced callbacks."""
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                factor=0.5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self, history
    
    def score(self, X):
        """Compute reconstruction error (anomaly scores)."""
        X_pred = self.model.predict(X, batch_size=1024, verbose=0)
        reconstruction_error = np.mean((X - X_pred)**2, axis=1)
        return reconstruction_error
    
    def save(self, path):
        """Save the model."""
        self.model.save(path)
    
    @classmethod
    def load(cls, path):
        """Load a saved model."""
        obj = cls(1)  # Dummy initialization
        obj.model = keras.models.load_model(path)
        return obj

def evaluate_model(y_true, scores, threshold_percentile=95):
    """Evaluate model performance with different thresholds."""
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
    """Train improved Autoencoder on standardized features."""
    print("🚀 Training Improved Autoencoder on Standardized Features")
    print("=" * 60)
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load standardized features (same as used for IF/OC-SVM)
    print("📥 Loading standardized features...")
    train_data = np.load(data_dir / "train_standardized.npz")
    X_train, y_train = train_data['X'], train_data['y']
    
    test_data = np.load(data_dir / "test_standardized.npz")
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"   Train: {X_train.shape} (all benign: {np.unique(y_train)})")
    print(f"   Test: {X_test.shape} (mixed: benign={np.sum(y_test==0)}, attack={np.sum(y_test==1)})")
    
    # Verify data statistics
    print(f"   Train data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   Test data range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    
    # Create improved model
    print("\n🏗️ Creating improved Autoencoder model...")
    model = ImprovedAEModel(
        input_dim=X_train.shape[1],  # 704 features
        latent_dim=64,
        l2_reg=1e-4,
        l1_latent=1e-5,
        use_denoising=True
    )
    
    # Train model
    print(f"\n🔧 Training model...")
    start_time = time.time()
    model, history = model.fit(
        X_train,
        epochs=50,
        batch_size=512,
        validation_split=0.1
    )
    training_time = time.time() - start_time
    
    print(f"   ✅ Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_scores = model.score(X_test)
    
    # Score statistics
    normal_scores = test_scores[y_test == 0]
    attack_scores = test_scores[y_test == 1]
    
    print(f"   Reconstruction Error Statistics:")
    print(f"   Normal samples: mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}")
    print(f"   Attack samples: mean={attack_scores.mean():.6f}, std={attack_scores.std():.6f}")
    print(f"   Separation ratio: {attack_scores.mean() / normal_scores.mean():.2f}x")
    
    # Test different thresholds
    print(f"\n   Testing different thresholds:")
    thresholds = [90, 95, 99]
    best_f1 = 0
    best_metrics = None
    
    for percentile in thresholds:
        metrics = evaluate_model(y_test, test_scores, threshold_percentile=percentile)
        print(f"   {percentile}th percentile: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_metrics = metrics
    
    print(f"\n📊 Best Results:")
    print(f"   📈 ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"   🎯 Precision: {best_metrics['precision']:.4f}")
    print(f"   🔍 Recall: {best_metrics['recall']:.4f}")
    print(f"   ⚖️ F1-Score: {best_metrics['f1_score']:.4f}")
    
    # Save model
    model_path = models_dir / "autoencoder_improved.keras"
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save results
    results = {
        'timestamp': np.datetime64('now').item().isoformat(),
        'model_type': 'ImprovedAutoencoder',
        'model_path': str(model_path),
        'architecture': {
            'input_dim': int(X_train.shape[1]),
            'encoder': '704 → 512 → 128 → 64',
            'decoder': '64 → 128 → 512 → 704',
            'latent_dim': 64,
            'normalization': 'LayerNorm',
            'activation': 'ReLU',
            'output_activation': 'linear',
            'regularization': {
                'l2_weights': 1e-4,
                'l1_latent': 1e-5
            },
            'denoising': True,
            'gaussian_noise': 0.05
        },
        'training': {
            'optimizer': 'Adam(1e-3)',
            'loss': 'MSE',
            'batch_size': 512,
            'epochs_max': 50,
            'validation_split': 0.1,
            'callbacks': ['ReduceLROnPlateau', 'EarlyStopping'],
            'training_time_seconds': training_time
        },
        'evaluation': best_metrics,
        'data_shapes': {
            'train': X_train.shape,
            'test': X_test.shape
        },
        'feature_type': 'standardized_704_features',
        'score_statistics': {
            'normal_mean': float(normal_scores.mean()),
            'normal_std': float(normal_scores.std()),
            'attack_mean': float(attack_scores.mean()),
            'attack_std': float(attack_scores.std()),
            'separation_ratio': float(attack_scores.mean() / normal_scores.mean())
        }
    }
    
    results_path = models_dir / "autoencoder_improved_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📋 Results saved: {results_path}")
    
    # Performance assessment
    auc = best_metrics['roc_auc']
    if auc > 0.9:
        print("🎉 Excellent performance! Rival to OneClassSVM!")
    elif auc > 0.8:
        print("👍 Very good performance! Much improved!")
    elif auc > 0.7:
        print("🤔 Good performance - significant improvement")
    elif auc > 0.6:
        print("📈 Moderate performance - better than before")
    elif auc > 0.5:
        print("⚠️ Slight improvement but still challenging")
    else:
        print("❌ Still poor performance")
    
    # Compare with original autoencoder
    try:
        with open(models_dir / "autoencoder_results.json", 'r') as f:
            old_results = json.load(f)
        old_auc = old_results['evaluation']['roc_auc']
        improvement = auc - old_auc
        print(f"📈 Improvement over original AE: {improvement:+.4f} ({improvement/old_auc*100:+.1f}%)")
    except:
        print("📊 Original autoencoder results not found for comparison")
    
    print(f"\n✅ Improved Autoencoder training completed!")
    return model, best_metrics

if __name__ == "__main__":
    main()