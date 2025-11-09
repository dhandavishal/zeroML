#!/usr/bin/env python3
"""
ZeroML Visualization Dashboard
Create comprehensive graphs and plots of model performance metrics
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import tensorflow as tf
import os

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path for imports
import sys
sys.path.append('src')

from zeroml.models.iforest import IForestModel
from zeroml.models.ocsvm import OCSVMModel

def load_all_model_scores():
    """Load all models and get their scores on test data."""
    print("📊 Loading all models and generating scores...")
    
    # Paths
    data_dir = Path("data/engineered")
    models_dir = Path("models")
    
    # Load test data
    std_data = np.load(data_dir / "test_standardized.npz")
    X_test_std, y_test = std_data['X'], std_data['y']
    
    scores = {}
    
    # OneClassSVM
    print("   Loading OneClassSVM...")
    ocsvm_model = OCSVMModel.load(models_dir / "oneclass_svm_optimized.pkl")
    scores['OneClassSVM'] = ocsvm_model.score(X_test_std)
    
    # Improved Autoencoder
    print("   Loading Improved Autoencoder...")
    ae_model = tf.keras.models.load_model(models_dir / "autoencoder_improved.keras")
    ae_predictions = ae_model.predict(X_test_std, batch_size=1024, verbose=0)
    scores['Autoencoder'] = ((X_test_std - ae_predictions)**2).mean(axis=1)
    
    # Improved Isolation Forest
    print("   Loading Improved Isolation Forest...")
    if_model = IForestModel.load(models_dir / "isolation_forest_improved.pkl")
    scores['Isolation Forest'] = if_model.score(X_test_std)
    
    # Create ensemble (weighted by ROC-AUC)
    print("   Creating ensemble scores...")
    # Normalize scores to [0, 1]
    norm_scores = {}
    for name, score in scores.items():
        norm_scores[name] = (score - score.min()) / (score.max() - score.min() + 1e-8)
    
    # Weights based on ROC-AUC performance
    weights = {'OneClassSVM': 0.411, 'Autoencoder': 0.346, 'Isolation Forest': 0.243}
    ensemble_scores = np.zeros_like(list(norm_scores.values())[0])
    for name, weight in weights.items():
        ensemble_scores += weight * norm_scores[name]
    
    scores['Ensemble'] = ensemble_scores
    
    return scores, y_test, X_test_std

def plot_roc_curves(scores, y_test, save_dir):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    model_aucs = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (name, score) in enumerate(scores.items()):
        fpr, tpr, _ = roc_curve(y_test, score)
        auc = np.trapz(tpr, fpr)
        model_aucs[name] = auc
        
        plt.plot(fpr, tpr, color=colors[i], linewidth=2.5, 
                label=f'{name} (AUC = {auc:.4f})')
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - ZeroML Anomaly Detection Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add performance annotations
    best_model = max(model_aucs.items(), key=lambda x: x[1])
    plt.text(0.6, 0.2, f'🏆 Best Model: {best_model[0]}\nROC-AUC: {best_model[1]:.4f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
             fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ ROC curves saved: {save_dir}/roc_curves.png")

def plot_precision_recall_curves(scores, y_test, save_dir):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    model_aps = {}
    
    for i, (name, score) in enumerate(scores.items()):
        precision, recall, _ = precision_recall_curve(y_test, score)
        ap = np.trapz(precision, recall)
        model_aps[name] = ap
        
        plt.plot(recall, precision, color=colors[i], linewidth=2.5,
                label=f'{name} (AP = {ap:.4f})')
    
    # Baseline (random classifier for imbalanced data)
    baseline = np.sum(y_test) / len(y_test)
    plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                label=f'Baseline (AP = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves - ZeroML Anomaly Detection Models', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Precision-Recall curves saved: {save_dir}/precision_recall_curves.png")

def plot_score_distributions(scores, y_test, save_dir):
    """Plot score distributions for normal vs anomalous samples."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, score) in enumerate(scores.items()):
        ax = axes[i]
        
        # Split scores by class
        normal_scores = score[y_test == 0]
        anomaly_scores = score[y_test == 1]
        
        # Plot histograms
        ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='lightblue', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Attack', color='lightcoral', density=True)
        
        # Add statistics
        ax.axvline(normal_scores.mean(), color='blue', linestyle='--', alpha=0.8, 
                  label=f'Normal μ={normal_scores.mean():.4f}')
        ax.axvline(anomaly_scores.mean(), color='red', linestyle='--', alpha=0.8,
                  label=f'Attack μ={anomaly_scores.mean():.4f}')
        
        ax.set_title(f'{name} Score Distribution', fontweight='bold')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Score Distributions: Normal vs Attack Traffic', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "score_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Score distributions saved: {save_dir}/score_distributions.png")

def plot_performance_comparison(scores, y_test, save_dir):
    """Create comprehensive performance comparison plots."""
    # Calculate metrics for different FPR thresholds
    fpr_thresholds = [0.01, 0.02, 0.05, 0.10]
    metrics_data = []
    
    for name, score in scores.items():
        normal_scores = score[y_test == 0]
        
        for target_fpr in fpr_thresholds:
            threshold = np.percentile(normal_scores, (1-target_fpr)*100)
            y_pred = (score > threshold).astype(int)
            
            tp = np.sum((y_test == 1) & (y_pred == 1))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            tn = np.sum((y_test == 0) & (y_pred == 0))
            fn = np.sum((y_test == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            metrics_data.append({
                'Model': name,
                'Target_FPR': f'{target_fpr:.0%}',
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Actual_FPR': actual_fpr
            })
    
    df = pd.DataFrame(metrics_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precision comparison
    pivot_precision = df.pivot(index='Model', columns='Target_FPR', values='Precision')
    sns.heatmap(pivot_precision, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0,0])
    axes[0,0].set_title('Precision at Different FPR Thresholds', fontweight='bold')
    
    # Recall comparison
    pivot_recall = df.pivot(index='Model', columns='Target_FPR', values='Recall')
    sns.heatmap(pivot_recall, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0,1])
    axes[0,1].set_title('Recall at Different FPR Thresholds', fontweight='bold')
    
    # F1-Score comparison
    pivot_f1 = df.pivot(index='Model', columns='Target_FPR', values='F1_Score')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1,0])
    axes[1,0].set_title('F1-Score at Different FPR Thresholds', fontweight='bold')
    
    # ROC-AUC comparison (bar plot)
    model_aucs = {}
    for name, score in scores.items():
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, score)
        model_aucs[name] = auc
    
    bars = axes[1,1].bar(model_aucs.keys(), model_aucs.values(), 
                        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1,1].set_title('ROC-AUC Comparison', fontweight='bold')
    axes[1,1].set_ylabel('ROC-AUC Score')
    axes[1,1].set_ylim(0, 1)
    
    # Add value annotations on bars
    for bar, value in zip(bars, model_aucs.values()):
        height = bar.get_height()
        axes[1,1].annotate(f'{value:.4f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.suptitle('ZeroML Performance Comparison Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Performance comparison saved: {save_dir}/performance_comparison.png")

def plot_confusion_matrices(scores, y_test, save_dir):
    """Plot confusion matrices for all models at optimal thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, score) in enumerate(scores.items()):
        ax = axes[i]
        
        # Find optimal threshold (best F1-score)
        normal_scores = score[y_test == 0]
        best_f1 = 0
        best_threshold = 0
        
        for target_fpr in [0.01, 0.02, 0.05]:
            threshold = np.percentile(normal_scores, (1-target_fpr)*100)
            y_pred = (score > threshold).astype(int)
            
            tp = np.sum((y_test == 1) & (y_pred == 1))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            tn = np.sum((y_test == 0) & (y_pred == 0))
            fn = np.sum((y_test == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Generate predictions with best threshold
        y_pred = (score > best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        
        ax.set_title(f'{name}\nF1-Score: {best_f1:.4f}', fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    
    plt.suptitle('Confusion Matrices at Optimal Thresholds', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Confusion matrices saved: {save_dir}/confusion_matrices.png")

def create_summary_report(scores, y_test, save_dir):
    """Create a summary report with key statistics."""
    print("📋 Creating summary report...")
    
    summary = {
        'dataset_info': {
            'total_samples': len(y_test),
            'normal_samples': int(np.sum(y_test == 0)),
            'attack_samples': int(np.sum(y_test == 1)),
            'attack_rate': float(np.sum(y_test == 1) / len(y_test))
        },
        'model_performance': {}
    }
    
    for name, score in scores.items():
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, score)
        
        # Calculate metrics at 2% FPR (common operational threshold)
        normal_scores = score[y_test == 0]
        threshold = np.percentile(normal_scores, 98)  # 2% FPR
        y_pred = (score > threshold).astype(int)
        
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        summary['model_performance'][name] = {
            'roc_auc': float(auc),
            'precision_at_2pct_fpr': float(precision),
            'recall_at_2pct_fpr': float(recall),
            'f1_score_at_2pct_fpr': float(f1),
            'score_range': {
                'min': float(score.min()),
                'max': float(score.max()),
                'mean': float(score.mean()),
                'std': float(score.std())
            }
        }
    
    # Save summary
    with open(save_dir / "performance_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create text report
    with open(save_dir / "performance_summary.txt", 'w') as f:
        f.write("ZeroML Anomaly Detection - Performance Summary Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Total Samples: {summary['dataset_info']['total_samples']:,}\n")
        f.write(f"  Normal Samples: {summary['dataset_info']['normal_samples']:,}\n")
        f.write(f"  Attack Samples: {summary['dataset_info']['attack_samples']:,}\n")
        f.write(f"  Attack Rate: {summary['dataset_info']['attack_rate']:.2%}\n\n")
        
        f.write("Model Performance (at 2% FPR threshold):\n")
        f.write("-" * 50 + "\n")
        
        # Sort by ROC-AUC
        sorted_models = sorted(summary['model_performance'].items(), 
                             key=lambda x: x[1]['roc_auc'], reverse=True)
        
        for i, (name, metrics) in enumerate(sorted_models):
            medal = ['🥇', '🥈', '🥉', '🏅'][i] if i < 4 else '  '
            f.write(f"{medal} {name}:\n")
            f.write(f"    ROC-AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"    Precision: {metrics['precision_at_2pct_fpr']:.4f}\n")
            f.write(f"    Recall: {metrics['recall_at_2pct_fpr']:.4f}\n")
            f.write(f"    F1-Score: {metrics['f1_score_at_2pct_fpr']:.4f}\n\n")
    
    print(f"   ✅ Summary report saved: {save_dir}/performance_summary.json")
    print(f"   ✅ Text report saved: {save_dir}/performance_summary.txt")

def main():
    """Main visualization function."""
    print("🎨 ZeroML Visualization Dashboard")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load all model scores
    scores, y_test, X_test = load_all_model_scores()
    
    print(f"\n📊 Generating visualizations...")
    
    # Generate all plots
    plot_roc_curves(scores, y_test, output_dir)
    plot_precision_recall_curves(scores, y_test, output_dir)
    plot_score_distributions(scores, y_test, output_dir)
    plot_performance_comparison(scores, y_test, output_dir)
    plot_confusion_matrices(scores, y_test, output_dir)
    
    # Create summary report
    create_summary_report(scores, y_test, output_dir)
    
    print(f"\n✅ All visualizations completed!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"🖼️ Generated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"   - {file.name}")
    
    print(f"\n🎯 Key Insights:")
    # Get best model
    from sklearn.metrics import roc_auc_score
    best_auc = 0
    best_model = ""
    for name, score in scores.items():
        auc = roc_auc_score(y_test, score)
        if auc > best_auc:
            best_auc = auc
            best_model = name
    
    print(f"   🏆 Best Model: {best_model} (ROC-AUC: {best_auc:.4f})")
    print(f"   📈 Dataset: {len(y_test):,} samples ({np.sum(y_test==1):,} attacks)")
    print(f"   🎨 Visualizations show clear model performance differences")

if __name__ == "__main__":
    main()