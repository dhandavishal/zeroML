#!/usr/bin/env python3
"""
Feature Engineering Pipeline for ZeroML
Removes constant features and applies preprocessing transformations.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

def identify_constant_features(X, threshold=1e-8):
    """
    Identify features with constant or near-constant values.
    
    Args:
        X: Feature matrix (samples x features)
        threshold: Variance threshold for considering a feature constant
        
    Returns:
        indices of constant features to remove
    """
    variances = np.var(X, axis=0, ddof=1)
    constant_features = np.where(variances <= threshold)[0]
    return constant_features

def remove_constant_features(X_splits, feature_names=None):
    """
    Remove constant features across all splits consistently.
    
    Args:
        X_splits: Dictionary with 'train', 'val', 'test' numpy arrays
        feature_names: Optional list of feature names
        
    Returns:
        cleaned_splits, removed_indices, kept_indices
    """
    # Find union of constant features across all splits
    constant_features_train = identify_constant_features(X_splits['train'])
    constant_features_val = identify_constant_features(X_splits['val'])  
    constant_features_test = identify_constant_features(X_splits['test'])
    
    # Take union to be conservative - remove if constant in ANY split
    all_constant = np.union1d(
        np.union1d(constant_features_train, constant_features_val),
        constant_features_test
    )
    
    print(f"🔍 Constant features found:")
    print(f"   Train: {len(constant_features_train)}")
    print(f"   Val: {len(constant_features_val)}")
    print(f"   Test: {len(constant_features_test)}")
    print(f"   Union (to remove): {len(all_constant)}")
    
    # Keep all features except constant ones
    kept_features = np.setdiff1d(np.arange(X_splits['train'].shape[1]), all_constant)
    
    # Apply removal to all splits
    cleaned_splits = {}
    for split_name, X in X_splits.items():
        cleaned_splits[split_name] = X[:, kept_features]
        print(f"   {split_name}: {X.shape} -> {cleaned_splits[split_name].shape}")
    
    return cleaned_splits, all_constant, kept_features

def apply_standardization(X_splits, scaler_path=None):
    """
    Apply standardization based on training set statistics.
    
    Args:
        X_splits: Dictionary with 'train', 'val', 'test' arrays
        scaler_path: Path to save the fitted scaler
        
    Returns:
        standardized_splits, fitted_scaler
    """
    print("🔧 Applying standardization...")
    
    # Handle missing values (-1) by setting them to 0 before scaling
    # Then restore them after scaling
    scaler = StandardScaler()
    
    # Fit on training data (replace -1 with 0 for fitting)
    X_train_for_fit = X_splits['train'].copy().astype(np.float32)
    X_train_for_fit[X_train_for_fit == -1] = 0
    scaler.fit(X_train_for_fit)
    
    standardized_splits = {}
    for split_name, X in X_splits.items():
        # Convert to float and handle missing values
        X_processed = X.copy().astype(np.float32)
        missing_mask = (X_processed == -1)
        
        # Replace missing with 0 for scaling
        X_processed[missing_mask] = 0
        
        # Apply scaling
        X_standardized = scaler.transform(X_processed)
        
        # Restore missing value indicator (set to -2 to distinguish from 0)
        X_standardized[missing_mask] = -2.0
        
        standardized_splits[split_name] = X_standardized
        print(f"   {split_name}: {X.shape} -> {X_standardized.shape}")
    
    # Save scaler if path provided
    if scaler_path:
        joblib.dump(scaler, scaler_path)
        print(f"   💾 Scaler saved: {scaler_path}")
    
    return standardized_splits, scaler

def apply_pca(X_splits, n_components=0.95, pca_path=None):
    """
    Apply PCA for dimensionality reduction.
    
    Args:
        X_splits: Dictionary with standardized data
        n_components: Number of components or variance ratio to retain
        pca_path: Path to save the fitted PCA
        
    Returns:
        transformed_splits, fitted_pca, explained_variance_ratio
    """
    print(f"🎯 Applying PCA (target: {n_components} variance)...")
    
    # Fit PCA on training data (handle missing values)
    X_train_for_pca = X_splits['train'].copy()
    X_train_for_pca[X_train_for_pca == -2.0] = 0  # Replace missing indicators with 0
    
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train_for_pca)
    
    print(f"   📊 Components: {pca.n_components_}")
    print(f"   📈 Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    transformed_splits = {}
    for split_name, X in X_splits.items():
        # Handle missing values for PCA
        X_for_pca = X.copy()
        X_for_pca[X_for_pca == -2.0] = 0
        
        # Apply PCA transformation
        X_transformed = pca.transform(X_for_pca)
        transformed_splits[split_name] = X_transformed
        print(f"   {split_name}: {X.shape} -> {X_transformed.shape}")
    
    # Save PCA if path provided
    if pca_path:
        joblib.dump(pca, pca_path)
        print(f"   💾 PCA saved: {pca_path}")
    
    return transformed_splits, pca, pca.explained_variance_ratio_

def save_feature_engineering_metadata(metadata, output_path):
    """Save metadata about feature engineering transformations."""
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"📋 Metadata saved: {output_path}")

def main():
    """Main feature engineering pipeline."""
    print("🚀 Starting Feature Engineering Pipeline...")
    
    # Paths
    data_dir = Path("data")
    interim_dir = data_dir / "interim"
    output_dir = data_dir / "engineered"
    output_dir.mkdir(exist_ok=True)
    
    # Load raw splits
    print("📥 Loading dataset splits...")
    X_splits = {}
    y_splits = {}
    
    # Load training and validation (benign data)
    X_splits['train'] = np.load(interim_dir / "benign" / "X_train.npy")
    y_splits['train'] = np.load(interim_dir / "benign" / "y_train.npy")
    X_splits['val'] = np.load(interim_dir / "benign" / "X_val.npy")
    y_splits['val'] = np.load(interim_dir / "benign" / "y_val.npy")
    
    # Load test data (mixed benign + attack)
    X_splits['test'] = np.load(interim_dir / "X_test.npy")
    y_splits['test'] = np.load(interim_dir / "y_test.npy")
    
    for split in ['train', 'val', 'test']:
        print(f"   {split}: X={X_splits[split].shape}, y={y_splits[split].shape}")
    
    # Stage 1: Remove constant features
    print("\n🔥 Stage 1: Removing constant features...")
    cleaned_splits, removed_features, kept_features = remove_constant_features(X_splits)
    
    # Stage 2: Apply standardization
    print("\n⚖️ Stage 2: Standardization...")
    standardized_splits, scaler = apply_standardization(
        cleaned_splits, 
        scaler_path=output_dir / "standard_scaler.pkl"
    )
    
    # Stage 3: Optional PCA (let's start with a reasonable number of components)
    print("\n🎯 Stage 3: PCA (optional)...")
    pca_splits, pca, explained_var = apply_pca(
        standardized_splits,
        n_components=min(200, len(kept_features)),  # Cap at 200 components
        pca_path=output_dir / "pca_transformer.pkl"
    )
    
    # Save engineered features (both standardized and PCA versions)
    print("\n💾 Saving engineered features...")
    
    # Save standardized version (recommended for Isolation Forest and OC-SVM)
    for split in ['train', 'val', 'test']:
        np.savez(
            output_dir / f"{split}_standardized.npz",
            X=standardized_splits[split],
            y=y_splits[split]
        )
    
    # Save PCA version (recommended for Autoencoder)
    for split in ['train', 'val', 'test']:
        np.savez(
            output_dir / f"{split}_pca.npz",
            X=pca_splits[split],
            y=y_splits[split]
        )
    
    # Save metadata
    metadata = {
        "timestamp": np.datetime64('now').item().isoformat(),
        "original_features": X_splits['train'].shape[1],
        "removed_features": len(removed_features),
        "kept_features": len(kept_features),
        "standardized_features": standardized_splits['train'].shape[1],
        "pca_components": pca_splits['train'].shape[1],
        "pca_explained_variance": float(explained_var.sum()),
        "removed_feature_indices": removed_features.tolist(),
        "kept_feature_indices": kept_features.tolist(),
        "scaler_path": str(output_dir / "standard_scaler.pkl"),
        "pca_path": str(output_dir / "pca_transformer.pkl"),
        "data_shapes": {
            split: {
                "original": X_splits[split].shape,
                "standardized": standardized_splits[split].shape,
                "pca": pca_splits[split].shape
            }
            for split in ['train', 'val', 'test']
        }
    }
    
    save_feature_engineering_metadata(metadata, output_dir / "feature_engineering_metadata.json")
    
    print(f"\n✅ Feature engineering completed!")
    print(f"📊 Summary:")
    print(f"   Original features: {metadata['original_features']}")
    print(f"   After removing constants: {metadata['kept_features']} ({metadata['removed_features']} removed)")
    print(f"   PCA components: {metadata['pca_components']} (explaining {metadata['pca_explained_variance']:.2%} variance)")
    print(f"   Output directory: {output_dir}")

if __name__ == "__main__":
    main()