import numpy as np
import joblib
import json
import argparse
from pathlib import Path
from zeroml.models.iforest import IForestModel
from zeroml.models.ocsvm import OCSVMModel
from zeroml.models.autoencoder import AEModel
from zeroml.eval.metrics import prf, rocauc, cm

def load_model_and_threshold(model_dir, model_type):
    """Load trained model and threshold"""
    model_dir = Path(model_dir)
    
    # Load model
    model_path = model_dir / f"{model_type}_model"
    if model_type == "iforest":
        model = IForestModel.load(str(model_path))
    elif model_type == "ocsvm":
        model = OCSVMModel.load(str(model_path))
    elif model_type == "autoencoder":
        model = AEModel.load(str(model_path))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load threshold
    threshold_path = model_dir / "threshold.json"
    with open(threshold_path, 'r') as f:
        threshold_data = json.load(f)
    
    threshold = threshold_data["threshold"]
    print(f"Loaded {model_type} model and threshold: {threshold:.6f}")
    
    return model, threshold

def load_test_data(data_dir):
    """Load test data"""
    data_dir = Path(data_dir)
    
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Test set anomaly rate: {y_test.mean():.3f}")
    
    return X_test, y_test

def run_inference(model, X_test, threshold):
    """Run inference on test data"""
    print("Computing anomaly scores...")
    scores = model.score(X_test)
    
    print("Making predictions...")
    predictions = (scores > threshold).astype(int)
    
    return scores, predictions

def compute_metrics(y_true, y_pred, scores):
    """Compute evaluation metrics"""
    print("Computing evaluation metrics...")
    
    # Precision, Recall, F1-score
    precision, recall, f1, _ = prf(y_true, y_pred)
    
    # ROC-AUC
    roc_auc = rocauc(y_true, scores)
    
    # Confusion Matrix
    confusion_mat = cm(y_true, y_pred)
    tn, fp, fn, tp = confusion_mat.ravel()
    
    # Additional metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": confusion_mat.tolist()
    }
    
    return metrics

def save_results(metrics, scores, predictions, output_dir):
    """Save inference results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save scores and predictions
    np.save(output_dir / "scores.npy", scores)
    np.save(output_dir / "predictions.npy", predictions)
    print(f"Scores and predictions saved to: {output_dir}")

def print_metrics_summary(metrics):
    """Print a summary of metrics"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"FPR:         {metrics['fpr']:.4f}")
    print(f"TPR:         {metrics['tpr']:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
    print(f"FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")
    print("="*50)

def main():
    """Main inference pipeline"""
    parser = argparse.ArgumentParser(description="Run ZeroML inference")
    parser.add_argument("--model-dir", type=str, default="outputs", 
                        help="Directory containing trained model")
    parser.add_argument("--data-dir", type=str, default="data/processed", 
                        help="Directory containing test data")
    parser.add_argument("--output-dir", type=str, default="outputs/inference", 
                        help="Directory to save inference results")
    parser.add_argument("--model-type", type=str, choices=["iforest", "ocsvm", "autoencoder"], 
                        default="iforest", help="Type of model to load")
    
    args = parser.parse_args()
    
    # Load model and threshold
    model, threshold = load_model_and_threshold(args.model_dir, args.model_type)
    
    # Load test data
    X_test, y_test = load_test_data(args.data_dir)
    
    # Run inference
    scores, predictions = run_inference(model, X_test, threshold)
    
    # Compute metrics
    metrics = compute_metrics(y_test, predictions, scores)
    
    # Save results
    save_results(metrics, scores, predictions, args.output_dir)
    
    # Print summary
    print_metrics_summary(metrics)
    
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()