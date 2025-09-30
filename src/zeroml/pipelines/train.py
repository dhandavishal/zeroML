import numpy as np
import joblib
import json
import argparse
from pathlib import Path
from zeroml.models.iforest import IForestModel
from zeroml.models.ocsvm import OCSVMModel
from zeroml.models.autoencoder import AEModel
from zeroml.models.calibrate import threshold_for_fpr

def load_data(data_dir):
    """Load training and validation data"""
    data_dir = Path(data_dir)
    
    X_train = np.load(data_dir / "X_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    return X_train, X_val

def train_model(model_type, X_train, **kwargs):
    """Train the specified model type"""
    print(f"Training {model_type} model...")
    
    if model_type == "iforest":
        model = IForestModel(**kwargs)
    elif model_type == "ocsvm":
        model = OCSVMModel(**kwargs)
    elif model_type == "autoencoder":
        input_dim = X_train.shape[1]
        model = AEModel(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train)
    return model

def calibrate_threshold(model, X_val, target_fpr=0.02):
    """Calibrate decision threshold on validation data"""
    print(f"Calibrating threshold for FPR={target_fpr}...")
    
    scores_val = model.score(X_val)
    threshold = threshold_for_fpr(scores_val, target_fpr)
    
    print(f"Calibrated threshold: {threshold:.6f}")
    return threshold

def save_artifacts(model, threshold, output_dir, model_type):
    """Save trained model and threshold"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{model_type}_model"
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")
    
    # Save threshold
    threshold_path = output_dir / "threshold.json"
    with open(threshold_path, 'w') as f:
        json.dump({"threshold": float(threshold), "target_fpr": 0.02}, f)
    print(f"Threshold saved to: {threshold_path}")
    
    # Save training metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "model_type": model_type,
            "threshold": float(threshold),
            "target_fpr": 0.02
        }, f)
    print(f"Metadata saved to: {metadata_path}")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Train ZeroML anomaly detection model")
    parser.add_argument("--data-dir", type=str, default="data/processed", 
                        help="Directory containing training data")
    parser.add_argument("--output-dir", type=str, default="outputs", 
                        help="Directory to save trained model")
    parser.add_argument("--model-type", type=str, choices=["iforest", "ocsvm", "autoencoder"], 
                        default="iforest", help="Type of model to train")
    parser.add_argument("--target-fpr", type=float, default=0.02, 
                        help="Target false positive rate for threshold calibration")
    
    # Model-specific parameters
    parser.add_argument("--contamination", type=float, default=0.1, 
                        help="Contamination parameter for isolation forest")
    parser.add_argument("--n-estimators", type=int, default=100, 
                        help="Number of estimators for isolation forest")
    parser.add_argument("--nu", type=float, default=0.1, 
                        help="Nu parameter for one-class SVM")
    parser.add_argument("--kernel", type=str, default="rbf", 
                        help="Kernel for one-class SVM")
    parser.add_argument("--latent-dim", type=int, default=32, 
                        help="Latent dimension for autoencoder")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Training epochs for autoencoder")
    
    args = parser.parse_args()
    
    # Load data
    X_train, X_val = load_data(args.data_dir)
    
    # Prepare model kwargs
    model_kwargs = {}
    if args.model_type == "iforest":
        model_kwargs = {
            "contamination": args.contamination,
            "n_estimators": args.n_estimators,
            "random_state": 42
        }
    elif args.model_type == "ocsvm":
        model_kwargs = {
            "nu": args.nu,
            "kernel": args.kernel
        }
    elif args.model_type == "autoencoder":
        model_kwargs = {
            "latent": args.latent_dim
        }
    
    # Train model
    model = train_model(args.model_type, X_train, **model_kwargs)
    
    # Calibrate threshold
    threshold = calibrate_threshold(model, X_val, args.target_fpr)
    
    # Save artifacts
    save_artifacts(model, threshold, args.output_dir, args.model_type)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()