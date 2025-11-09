#!/usr/bin/env python3
"""
Complete pipeline to process Friday attack data and compare with Wednesday results
"""

import numpy as np
import subprocess
import json
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a shell command and report progress."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"✅ Completed in {elapsed:.2f} seconds")
        return True
    else:
        print(f"❌ Error: {result.stderr}")
        print(f"⏱️ Failed after {elapsed:.2f} seconds")
        return False

def main():
    """Run complete pipeline for Friday attack data."""
    
    print("="*70)
    print("🎯 FRIDAY ATTACK DATA PIPELINE")
    print("="*70)
    print("This will:")
    print("  1. Process Friday attack PCAP to features")
    print("  2. Create new dataset splits (Monday + Friday)")
    print("  3. Rerun feature engineering")
    print("  4. Train all three models on new splits")
    print("  5. Compare Friday vs Wednesday attack detection")
    print("="*70)
    
    input("\nPress Enter to continue...")
    
    # Step 1: Process Friday PCAP to nPrint features
    success = run_command(
        "python scripts/processing/pcap_to_npy.py",
        "Step 1: Converting Friday PCAP to nPrint features"
    )
    if not success:
        print("❌ Pipeline stopped due to PCAP processing error")
        return
    
    # Step 2: Create new dataset splits with Friday data
    success = run_command(
        "python scripts/create_splits.py",
        "Step 2: Creating dataset splits (Benign + Friday Attack)"
    )
    if not success:
        print("❌ Pipeline stopped due to split creation error")
        return
    
    # Step 3: Feature engineering
    success = run_command(
        "python scripts/feature_engineering.py",
        "Step 3: Feature engineering (remove constants, standardize, PCA)"
    )
    if not success:
        print("❌ Pipeline stopped due to feature engineering error")
        return
    
    # Step 4: Generate dataset statistics
    run_command(
        "python scripts/generate_stats.py",
        "Step 4: Generating dataset statistics"
    )
    
    # Step 5: Train Isolation Forest (improved)
    success = run_command(
        "python scripts/train_isolation_forest_improved.py",
        "Step 5: Training Improved Isolation Forest"
    )
    
    # Step 6: Train OneClassSVM
    success = run_command(
        "python scripts/train_oneclass_svm_optimized.py",
        "Step 6: Training OneClassSVM"
    )
    
    # Step 7: Train Improved Autoencoder
    success = run_command(
        "python scripts/train_autoencoder_improved.py",
        "Step 7: Training Improved Autoencoder"
    )
    
    # Step 8: Final model comparison
    success = run_command(
        "python scripts/final_model_comparison.py",
        "Step 8: Final Model Comparison and Ensemble"
    )
    
    # Step 9: Generate visualizations
    success = run_command(
        "python scripts/visualize_results.py",
        "Step 9: Generating visualizations and plots"
    )
    
    # Step 10: Create comparison report (Friday vs Wednesday)
    print("\n" + "="*60)
    print("📊 Creating Friday vs Wednesday Comparison Report")
    print("="*60)
    
    # Load results
    models_dir = Path("models")
    
    try:
        with open(models_dir / "final_comprehensive_results.json", 'r') as f:
            friday_results = json.load(f)
        
        print("\n🎯 FRIDAY ATTACK DETECTION RESULTS")
        print("-" * 50)
        
        for model, metrics in friday_results['model_performances'].items():
            print(f"\n{model}:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  Best F1: {metrics['best_f1']:.4f}")
            print(f"  Precision: {metrics['best_precision']:.4f}")
            print(f"  Recall: {metrics['best_recall']:.4f}")
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📁 Results saved to:")
        print(f"  - Models: models/")
        print(f"  - Results: models/*.json")
        print(f"  - Visualizations: results/visualizations/")
        print("\n💡 To compare with Wednesday results, check:")
        print("  - models/final_comprehensive_results.json")
        
    except Exception as e:
        print(f"⚠️ Could not load final results: {e}")
    
    print("\n🎉 Friday attack data pipeline completed!")
    print("="*60)

if __name__ == "__main__":
    main()
