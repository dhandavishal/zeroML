#!/usr/bin/env python3
"""
Dataset Statistics Generator for ZeroML
Creates comprehensive statistics and visualizations for the processed dataset
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

class DatasetStatistics:
    def __init__(self, data_dir="data/interim"):
        self.data_dir = Path(data_dir)
        self.stats = {}
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
        
    def load_all_splits(self):
        """Load all dataset splits"""
        print("📊 Loading all dataset splits...")
        
        data = {}
        
        # Load benign training data
        benign_dir = self.data_dir / "benign"
        if benign_dir.exists():
            try:
                data['X_train'] = np.load(benign_dir / "X_train.npy")
                data['y_train'] = np.load(benign_dir / "y_train.npy")
                data['X_val'] = np.load(benign_dir / "X_val.npy")
                data['y_val'] = np.load(benign_dir / "y_val.npy")
                print(f"   ✅ Training: {data['X_train'].shape}")
                print(f"   ✅ Validation: {data['X_val'].shape}")
            except Exception as e:
                print(f"   ⚠️  Benign splits not found: {e}")
        
        # Load test data
        try:
            data['X_test'] = np.load(self.data_dir / "X_test.npy")
            data['y_test'] = np.load(self.data_dir / "y_test.npy")
            print(f"   ✅ Test: {data['X_test'].shape}")
        except Exception as e:
            print(f"   ⚠️  Test data not found: {e}")
        
        return data
    
    def compute_basic_statistics(self, data):
        """Compute basic dataset statistics"""
        print("🔢 Computing basic statistics...")
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "splits": {}
        }
        
        total_samples = 0
        total_features = 0
        
        for split_name, X in data.items():
            if split_name.startswith('X_'):
                split = split_name[2:]  # Remove 'X_' prefix
                y_name = 'y_' + split
                
                n_samples, n_features = X.shape
                total_samples += n_samples
                total_features = n_features  # Should be same for all
                
                # Basic shape info
                split_stats = {
                    "samples": int(n_samples),
                    "features": int(n_features),
                    "memory_mb": round(X.nbytes / (1024 * 1024), 2),
                    "dtype": str(X.dtype)
                }
                
                # Value statistics
                split_stats["values"] = {
                    "min": float(X.min()),
                    "max": float(X.max()),
                    "mean": float(X.mean()),
                    "std": float(X.std()),
                    "unique_values": len(np.unique(X)),
                    "missing_count": int(np.sum(X == -1)),
                    "missing_percent": round(float(np.sum(X == -1)) / X.size * 100, 2)
                }
                
                # Label statistics (if available)
                if y_name in data:
                    y = data[y_name]
                    unique_labels, counts = np.unique(y, return_counts=True)
                    split_stats["labels"] = {
                        "unique_labels": unique_labels.tolist(),
                        "label_counts": counts.tolist(),
                        "label_distribution": dict(zip(unique_labels.astype(int), 
                                                     (counts / len(y)).round(3)))
                    }
                
                stats["splits"][split] = split_stats
        
        # Overall statistics
        stats["overall"] = {
            "total_samples": int(total_samples),
            "features": int(total_features),
            "total_memory_mb": round(sum(s["memory_mb"] for s in stats["splits"].values()), 2)
        }
        
        return stats
    
    def compute_feature_statistics(self, data, max_features=20):
        """Compute per-feature statistics for a subset of features"""
        print(f"🔍 Computing feature statistics (first {max_features} features)...")
        
        feature_stats = {}
        
        # Use training data for feature analysis
        if 'X_train' in data:
            X = data['X_train']
            n_features = min(max_features, X.shape[1])
            
            for i in range(n_features):
                feature = X[:, i]
                
                unique_vals, counts = np.unique(feature, return_counts=True)
                value_dist = {str(int(val)): int(count) for val, count in zip(unique_vals, counts)}
                
                feature_stats[f"feature_{i}"] = {
                    "min": float(feature.min()),
                    "max": float(feature.max()),
                    "mean": float(feature.mean()),
                    "std": float(feature.std()),
                    "unique_values": len(unique_vals),
                    "missing_count": int(np.sum(feature == -1)),
                    "zero_count": int(np.sum(feature == 0)),
                    "one_count": int(np.sum(feature == 1)),
                    "value_distribution": value_dist
                }
        
        return feature_stats
    
    def analyze_data_quality(self, data):
        """Analyze data quality across splits"""
        print("🔍 Analyzing data quality...")
        
        quality_stats = {}
        
        for split_name, X in data.items():
            if split_name.startswith('X_'):
                split = split_name[2:]
                
                # Check for problematic values
                nan_count = np.sum(np.isnan(X))
                inf_count = np.sum(np.isinf(X))
                
                # Feature consistency
                feature_variances = np.var(X, axis=0)
                constant_features = np.sum(feature_variances == 0)
                low_variance_features = np.sum(feature_variances < 1e-6)
                
                quality_stats[split] = {
                    "nan_values": int(nan_count),
                    "inf_values": int(inf_count),
                    "constant_features": int(constant_features),
                    "low_variance_features": int(low_variance_features),
                    "data_quality_score": float(1.0 - (nan_count + inf_count) / X.size)
                }
        
        return quality_stats
    
    def generate_summary_report(self, stats, feature_stats, quality_stats, output_path):
        """Generate comprehensive summary report"""
        print("📋 Generating summary report...")
        
        # Combine all statistics
        full_report = {
            "dataset_summary": stats,
            "feature_analysis": feature_stats,
            "data_quality": quality_stats,
            "recommendations": self.generate_recommendations(stats, quality_stats)
        }
        
        # Convert numpy types before JSON serialization
        full_report = self.convert_numpy_types(full_report)
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        # Generate text summary
        text_path = output_path.with_suffix('.txt')
        with open(text_path, 'w') as f:
            f.write("ZEROML DATASET STATISTICS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic stats
            f.write("📊 DATASET OVERVIEW\n")
            f.write(f"Generated: {stats['timestamp']}\n")
            f.write(f"Total samples: {stats['overall']['total_samples']:,}\n")
            f.write(f"Features: {stats['overall']['features']:,}\n")
            f.write(f"Total memory: {stats['overall']['total_memory_mb']} MB\n\n")
            
            # Split details
            f.write("📂 SPLIT BREAKDOWN\n")
            for split_name, split_stats in stats['splits'].items():
                f.write(f"\n{split_name.upper()}:\n")
                f.write(f"  Samples: {split_stats['samples']:,}\n")
                f.write(f"  Memory: {split_stats['memory_mb']} MB\n")
                f.write(f"  Missing values: {split_stats['values']['missing_percent']}%\n")
                
                if 'labels' in split_stats:
                    f.write(f"  Labels: {split_stats['labels']['unique_labels']}\n")
                    for label, count in zip(split_stats['labels']['unique_labels'], 
                                          split_stats['labels']['label_counts']):
                        f.write(f"    Label {label}: {count:,} samples\n")
            
            # Data quality
            f.write(f"\n🔍 DATA QUALITY\n")
            for split, quality in quality_stats.items():
                f.write(f"{split}: Quality score {quality['data_quality_score']:.3f}\n")
                if quality['constant_features'] > 0:
                    f.write(f"  ⚠️  {quality['constant_features']} constant features\n")
            
            # Recommendations
            f.write(f"\n💡 RECOMMENDATIONS\n")
            for rec in full_report['recommendations']:
                f.write(f"- {rec}\n")
        
        print(f"📊 Reports saved:")
        print(f"   JSON: {output_path}")
        print(f"   Text: {text_path}")
        
        return full_report
    
    def generate_recommendations(self, stats, quality_stats):
        """Generate ML-specific recommendations"""
        recommendations = []
        
        # Check dataset size
        total_samples = stats['overall']['total_samples']
        if total_samples < 10000:
            recommendations.append("Dataset is small (<10K samples). Consider data augmentation.")
        
        # Check feature count
        n_features = stats['overall']['features']
        if n_features > 1000:
            recommendations.append(f"High-dimensional data ({n_features} features). Consider feature selection or PCA.")
        
        # Check missing values
        for split_name, split_stats in stats['splits'].items():
            missing_pct = split_stats['values']['missing_percent']
            if missing_pct > 10:
                recommendations.append(f"{split_name} has {missing_pct}% missing values. Review imputation strategy.")
        
        # Check data quality
        for split, quality in quality_stats.items():
            if quality['constant_features'] > 0:
                recommendations.append(f"Remove {quality['constant_features']} constant features from {split}.")
            if quality['data_quality_score'] < 0.95:
                recommendations.append(f"Data quality issues in {split} (score: {quality['data_quality_score']:.3f}).")
        
        # ML-specific recommendations
        if 'test' in stats['splits'] and 'labels' in stats['splits']['test']:
            test_labels = stats['splits']['test']['labels']
            if len(test_labels['unique_labels']) == 2:
                # Binary classification
                label_dist = test_labels['label_distribution']
                minority_ratio = min(label_dist.values())
                if minority_ratio < 0.1:
                    recommendations.append("Severe class imbalance in test set. Consider rebalancing.")
                elif minority_ratio < 0.3:
                    recommendations.append("Moderate class imbalance. Monitor precision/recall carefully.")
        
        recommendations.append("Dataset is ready for anomaly detection training.")
        recommendations.append("Consider ensemble methods (IF + OC-SVM + Autoencoder) for robust detection.")
        
        return recommendations
    
    def create_basic_visualizations(self, data, output_dir):
        """Create basic visualizations (if matplotlib works)"""
        try:
            print("📈 Creating basic visualizations...")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sample size comparison
            if len(data) > 0:
                plt.figure(figsize=(10, 6))
                
                split_names = []
                sample_counts = []
                
                for name, X in data.items():
                    if name.startswith('X_'):
                        split_names.append(name[2:])
                        sample_counts.append(X.shape[0])
                
                plt.bar(split_names, sample_counts)
                plt.title('Dataset Split Sizes')
                plt.ylabel('Number of Samples')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'split_sizes.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Saved: {output_dir / 'split_sizes.png'}")
            
            # Test set label distribution
            if 'y_test' in data:
                plt.figure(figsize=(8, 6))
                unique_labels, counts = np.unique(data['y_test'], return_counts=True)
                label_names = ['Benign' if l == 0 else 'Attack' for l in unique_labels]
                
                plt.pie(counts, labels=label_names, autopct='%1.1f%%', startangle=90)
                plt.title('Test Set Label Distribution')
                plt.savefig(output_dir / 'test_labels.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Saved: {output_dir / 'test_labels.png'}")
                
        except Exception as e:
            print(f"   ⚠️  Visualization creation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate ZeroML dataset statistics")
    parser.add_argument("--data-dir", type=str, default="data/interim",
                        help="Directory containing split data")
    parser.add_argument("--output-file", type=str, default="data/dataset_summary.json",
                        help="Output file for statistics")
    parser.add_argument("--viz-dir", type=str, default="data/visualizations",
                        help="Directory for visualizations")
    parser.add_argument("--max-features", type=int, default=20,
                        help="Maximum features to analyze in detail")
    
    args = parser.parse_args()
    
    # Create statistics generator
    stats_gen = DatasetStatistics(args.data_dir)
    
    # Load all data
    data = stats_gen.load_all_splits()
    if not data:
        print("❌ No data found to analyze")
        return 1
    
    # Compute statistics
    basic_stats = stats_gen.compute_basic_statistics(data)
    feature_stats = stats_gen.compute_feature_statistics(data, args.max_features)
    quality_stats = stats_gen.analyze_data_quality(data)
    
    # Generate report
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = stats_gen.generate_summary_report(
        basic_stats, feature_stats, quality_stats, output_path
    )
    
    # Create visualizations
    stats_gen.create_basic_visualizations(data, args.viz_dir)
    
    print("\n✅ Dataset statistics generation completed!")
    print(f"📊 Total samples: {basic_stats['overall']['total_samples']:,}")
    print(f"🔧 Features: {basic_stats['overall']['features']:,}")
    print(f"💾 Memory usage: {basic_stats['overall']['total_memory_mb']} MB")
    
    return 0

if __name__ == "__main__":
    exit(main())