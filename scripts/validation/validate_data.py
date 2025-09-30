#!/usr/bin/env python3
"""
Data Quality Validation Script
Validates NPY files for consistency, quality, and ML readiness
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def validate_npy_file(npy_path):
    """Validate a single NPY file"""
    npy_path = Path(npy_path)
    
    if not npy_path.exists():
        return {"error": f"File does not exist: {npy_path}"}
    
    try:
        # Load data
        X = np.load(npy_path)
        
        # Basic shape validation
        if len(X.shape) != 2:
            return {"error": f"Invalid shape: {X.shape} (expected 2D array)"}
        
        n_samples, n_features = X.shape
        
        # Data type validation
        if X.dtype not in [np.int8, np.int16, np.int32, np.float32, np.float64]:
            return {"error": f"Unexpected data type: {X.dtype}"}
        
        # Value range analysis
        unique_values = np.unique(X)
        min_val, max_val = X.min(), X.max()
        
        # Check for common nprint values (0, 1, -1)
        nprint_values = set([-1, 0, 1])
        actual_values = set(unique_values)
        is_nprint_format = actual_values.issubset(nprint_values)
        
        # Missing value analysis (nprint uses -1 for missing)
        missing_count = np.sum(X == -1)
        missing_percent = (missing_count / X.size) * 100
        
        # NaN/Inf check
        nan_count = np.sum(np.isnan(X))
        inf_count = np.sum(np.isinf(X))
        
        # Feature statistics
        feature_stats = []
        for i in range(min(10, n_features)):  # Check first 10 features
            col = X[:, i]
            feature_stats.append({
                "feature_idx": i,
                "unique_values": len(np.unique(col)),
                "missing_count": int(np.sum(col == -1)),
                "zero_count": int(np.sum(col == 0)),
                "one_count": int(np.sum(col == 1)),
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean())
            })
        
        # Memory usage
        memory_mb = X.nbytes / (1024 * 1024)
        
        validation_result = {
            "file_path": str(npy_path),
            "file_size_mb": round(npy_path.stat().st_size / (1024 * 1024), 2),
            "shape": {"samples": int(n_samples), "features": int(n_features)},
            "dtype": str(X.dtype),
            "memory_usage_mb": round(memory_mb, 2),
            "value_range": {"min": float(min_val), "max": float(max_val)},
            "unique_values": len(unique_values),
            "unique_values_sample": unique_values[:20].tolist(),
            "is_nprint_format": bool(is_nprint_format),
            "missing_values": {
                "count": int(missing_count),
                "percentage": round(missing_percent, 2)
            },
            "data_quality": {
                "nan_count": int(nan_count),
                "inf_count": int(inf_count),
                "has_issues": bool(nan_count > 0 or inf_count > 0)
            },
            "feature_stats": feature_stats,
            "validation_passed": True
        }
        
        return validation_result
        
    except Exception as e:
        return {"error": f"Failed to load/analyze file: {str(e)}"}

def compare_files(file_results):
    """Compare multiple NPY files for consistency"""
    if len(file_results) < 2:
        return {"message": "Need at least 2 files for comparison"}
    
    # Extract valid results (no errors)
    valid_results = [r for r in file_results if "error" not in r]
    
    if len(valid_results) < 2:
        return {"error": "Need at least 2 valid files for comparison"}
    
    # Check feature count consistency
    feature_counts = [r["shape"]["features"] for r in valid_results]
    consistent_features = len(set(feature_counts)) == 1
    
    # Check data type consistency
    dtypes = [r["dtype"] for r in valid_results]
    consistent_dtypes = len(set(dtypes)) == 1
    
    # Check nprint format consistency
    nprint_formats = [r["is_nprint_format"] for r in valid_results]
    consistent_nprint = len(set(nprint_formats)) == 1
    
    comparison = {
        "num_files": len(valid_results),
        "feature_consistency": {
            "consistent": consistent_features,
            "feature_counts": feature_counts,
            "expected_features": feature_counts[0] if consistent_features else "INCONSISTENT"
        },
        "dtype_consistency": {
            "consistent": consistent_dtypes,
            "dtypes": dtypes
        },
        "nprint_format_consistency": {
            "consistent": consistent_nprint,
            "all_nprint_format": all(nprint_formats)
        },
        "sample_sizes": [r["shape"]["samples"] for r in valid_results],
        "total_samples": sum(r["shape"]["samples"] for r in valid_results),
        "files": [r["file_path"] for r in valid_results]
    }
    
    return comparison

def generate_data_report(file_results, comparison, output_dir):
    """Generate comprehensive data quality report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    valid_files = [r for r in file_results if "error" not in r]
    failed_files = [r for r in file_results if "error" in r]
    
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_files_checked": len(file_results),
        "valid_files": len(valid_files),
        "failed_files": len(failed_files),
        "total_samples": sum(r["shape"]["samples"] for r in valid_files),
        "total_features": valid_files[0]["shape"]["features"] if valid_files else 0,
        "consistency_check": comparison,
        "file_details": file_results
    }
    
    # Save JSON report
    report_file = output_dir / "data_quality_report.json"
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate text summary
    summary_file = output_dir / "data_quality_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("DATA QUALITY VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"📊 SUMMARY\n")
        f.write(f"Total files: {len(file_results)}\n")
        f.write(f"Valid files: {len(valid_files)}\n")
        f.write(f"Failed files: {len(failed_files)}\n")
        f.write(f"Total samples: {summary['total_samples']:,}\n")
        f.write(f"Features per sample: {summary['total_features']}\n\n")
        
        if valid_files:
            f.write(f"📁 FILE DETAILS\n")
            for result in valid_files:
                f.write(f"- {Path(result['file_path']).name}: "
                       f"{result['shape']['samples']:,} samples, "
                       f"{result['file_size_mb']}MB\n")
            f.write("\n")
        
        if failed_files:
            f.write(f"❌ FAILED FILES\n")
            for result in failed_files:
                f.write(f"- {result.get('file_path', 'Unknown')}: {result['error']}\n")
            f.write("\n")
        
        f.write(f"✅ CONSISTENCY CHECK\n")
        f.write(f"Feature consistency: {comparison['feature_consistency']['consistent']}\n")
        f.write(f"Data type consistency: {comparison['dtype_consistency']['consistent']}\n")
        f.write(f"nPrint format: {comparison['nprint_format_consistency']['all_nprint_format']}\n")
    
    print(f"📊 Reports saved:")
    print(f"   - JSON: {report_file}")
    print(f"   - Summary: {summary_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Validate NPY data files")
    parser.add_argument("--input-dir", type=str, default="data/processed",
                        help="Directory containing NPY files")
    parser.add_argument("--output-dir", type=str, default="data/validation",
                        help="Directory to save validation reports")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Specific NPY files to validate")
    
    args = parser.parse_args()
    
    # Find NPY files
    if args.files:
        npy_files = [Path(f) for f in args.files]
    else:
        input_dir = Path(args.input_dir)
        npy_files = list(input_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"❌ No NPY files found in {args.input_dir}")
        return 1
    
    print(f"🔍 Validating {len(npy_files)} NPY files...")
    
    # Validate each file
    file_results = []
    for npy_file in npy_files:
        print(f"   Checking {npy_file.name}...")
        result = validate_npy_file(npy_file)
        file_results.append(result)
        
        if "error" in result:
            print(f"   ❌ {result['error']}")
        else:
            print(f"   ✅ {result['shape']['samples']:,} samples, "
                  f"{result['shape']['features']} features")
    
    # Compare files for consistency
    print(f"\n🔄 Running consistency checks...")
    comparison = compare_files(file_results)
    
    # Generate report
    print(f"\n📋 Generating validation report...")
    summary = generate_data_report(file_results, comparison, args.output_dir)
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"VALIDATION SUMMARY")
    print(f"="*60)
    print(f"✅ Valid files: {summary['valid_files']}")
    print(f"❌ Failed files: {summary['failed_files']}")
    print(f"📦 Total samples: {summary['total_samples']:,}")
    print(f"🔧 Features: {summary['total_features']}")
    
    if comparison.get('feature_consistency', {}).get('consistent'):
        print(f"✅ Feature consistency: PASSED")
    else:
        print(f"❌ Feature consistency: FAILED")
    
    return 0 if summary['failed_files'] == 0 else 1

if __name__ == "__main__":
    exit(main())