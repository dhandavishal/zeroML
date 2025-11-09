#!/usr/bin/env python3
"""
Compare model performance on Wednesday vs Friday attack data.
"""

import json
from pathlib import Path
import numpy as np

def load_results(filename):
    """Load results from JSON file."""
    filepath = Path("models") / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")

def print_model_comparison(model_name, wed_results, fri_results):
    """Print comparison for a single model."""
    print(f"\n📊 {model_name}")
    print("-" * 70)
    
    # Extract ROC-AUC - handle both direct and nested formats
    wed_auc = wed_results.get('roc_auc', 0)
    if wed_auc == 0 and 'test_evaluation' in wed_results:
        wed_auc = wed_results['test_evaluation'].get('roc_auc', 0)
    if wed_auc == 0 and 'evaluation' in wed_results:
        wed_auc = wed_results['evaluation'].get('roc_auc', 0)
    if wed_auc == 0 and 'best_roc_auc' in wed_results:
        wed_auc = wed_results['best_roc_auc']
    
    fri_auc = fri_results.get('roc_auc', 0)
    if fri_auc == 0 and 'test_evaluation' in fri_results:
        fri_auc = fri_results['test_evaluation'].get('roc_auc', 0)
    if fri_auc == 0 and 'evaluation' in fri_results:
        fri_auc = fri_results['evaluation'].get('roc_auc', 0)
    if fri_auc == 0 and 'best_roc_auc' in fri_results:
        fri_auc = fri_results['best_roc_auc']
    
    # Calculate change
    auc_diff = fri_auc - wed_auc
    auc_pct = (auc_diff / wed_auc * 100) if wed_auc > 0 else 0
    
    print(f"  Wednesday Attack - ROC-AUC: {wed_auc:.4f}")
    print(f"  Friday Attack    - ROC-AUC: {fri_auc:.4f}")
    
    if auc_diff > 0:
        print(f"  📈 Improvement: +{auc_diff:.4f} ({auc_pct:+.2f}%)")
    elif auc_diff < 0:
        print(f"  📉 Decrease: {auc_diff:.4f} ({auc_pct:.2f}%)")
    else:
        print(f"  ➡️ No change")
    
    # Compare at different FPR thresholds
    print(f"\n  Performance at different FPR thresholds:")
    for fpr_val in ['1', '2', '5']:
        key = f'fpr_{fpr_val}_percent'
        if key in wed_results and key in fri_results:
            wed = wed_results[key]
            fri = fri_results[key]
            
            print(f"\n    FPR {fpr_val}%:")
            print(f"      Wednesday: P={wed.get('precision', 0):.3f}, "
                  f"R={wed.get('recall', 0):.3f}, F1={wed.get('f1', 0):.3f}")
            print(f"      Friday:    P={fri.get('precision', 0):.3f}, "
                  f"R={fri.get('recall', 0):.3f}, F1={fri.get('f1', 0):.3f}")

def main():
    """Main comparison function."""
    print_header("🔄 WEDNESDAY vs FRIDAY ATTACK COMPARISON")
    
    print("\n📅 Dataset Information:")
    print("  Wednesday Attack: CICIDS2017 Wednesday dataset")
    print("  Friday Attack:    CICIDS2017 Friday dataset")
    print("  Both datasets:    ~48k attack packets each")
    
    # Store results for reference - will be populated dynamically
    wednesday_results = {
        'oneclass_svm': None,
        'autoencoder': None,
        'isolation_forest': None,
        'ensemble': None
    }
    
    friday_results = {
        'oneclass_svm': None,
        'autoencoder': None,
        'isolation_forest': None,
        'ensemble': None
    }
    
    # Load Friday results (current)
    print("\n📥 Loading Friday attack results...")
    fri_ocsvm = load_results("oneclass_svm_results.json")
    fri_ae = load_results("autoencoder_improved_results.json")
    fri_if = load_results("isolation_forest_improved_results.json")
    fri_final = load_results("final_comprehensive_results.json")
    
    if fri_ocsvm:
        friday_results['oneclass_svm'] = fri_ocsvm
        auc = fri_ocsvm.get('test_evaluation', {}).get('roc_auc', 0)
        if auc == 0:
            auc = fri_ocsvm.get('roc_auc', 0)
        print(f"  ✅ OneClassSVM: ROC-AUC = {auc:.4f}")
    
    if fri_ae:
        friday_results['autoencoder'] = fri_ae
        auc = fri_ae.get('evaluation', {}).get('roc_auc', 0)
        if auc == 0:
            auc = fri_ae.get('test_evaluation', {}).get('roc_auc', 0)
        if auc == 0:
            auc = fri_ae.get('roc_auc', 0)
        print(f"  ✅ Autoencoder: ROC-AUC = {auc:.4f}")
    
    if fri_if:
        friday_results['isolation_forest'] = fri_if
        auc = fri_if.get('best_roc_auc', 0)
        if auc == 0:
            auc = fri_if.get('evaluation', {}).get('roc_auc', 0)
        if auc == 0:
            auc = fri_if.get('test_evaluation', {}).get('roc_auc', 0)
        if auc == 0:
            auc = fri_if.get('roc_auc', 0)
        print(f"  ✅ Isolation Forest: ROC-AUC = {auc:.4f}")
    
    if fri_final and 'Ensemble' in fri_final:
        friday_results['ensemble'] = fri_final['Ensemble']
        print(f"  ✅ Ensemble: ROC-AUC = {fri_final['Ensemble'].get('roc_auc', 0):.4f}")
    
    # For Wednesday results, use known values from conversation history
    print("\n📥 Wednesday attack results (from previous run):")
    wednesday_results['oneclass_svm'] = {
        'roc_auc': 0.9788,
        'precision': 0.9253,
        'recall': 0.9292,
        'f1': 0.9273,
        'fpr_1_percent': {'precision': 0.968, 'recall': 0.704, 'f1': 0.815},
        'fpr_2_percent': {'precision': 0.968, 'recall': 0.781, 'f1': 0.865},
        'fpr_5_percent': {'precision': 0.942, 'recall': 0.891, 'f1': 0.916}
    }
    wednesday_results['autoencoder'] = {
        'roc_auc': 0.8228,
        'fpr_1_percent': {'precision': 0.694, 'recall': 0.030, 'f1': 0.058},
        'fpr_2_percent': {'precision': 0.851, 'recall': 0.180, 'f1': 0.297},
        'fpr_5_percent': {'precision': 0.839, 'recall': 0.389, 'f1': 0.531}
    }
    wednesday_results['isolation_forest'] = {
        'roc_auc': 0.5780,
        'fpr_1_percent': {'precision': 0.654, 'recall': 0.029, 'f1': 0.055},
        'fpr_2_percent': {'precision': 0.600, 'recall': 0.036, 'f1': 0.068},
        'fpr_5_percent': {'precision': 0.454, 'recall': 0.060, 'f1': 0.106}
    }
    wednesday_results['ensemble'] = {
        'roc_auc': 0.8489
    }
    
    print(f"  ✅ OneClassSVM: ROC-AUC = 0.9788")
    print(f"  ✅ Autoencoder: ROC-AUC = 0.8228")
    print(f"  ✅ Isolation Forest: ROC-AUC = 0.5780")
    print(f"  ✅ Ensemble: ROC-AUC = 0.8489")
    
    # Compare each model
    print_header("📊 MODEL-BY-MODEL COMPARISON")
    
    if friday_results['oneclass_svm']:
        print_model_comparison(
            "OneClassSVM",
            wednesday_results['oneclass_svm'],
            friday_results['oneclass_svm']
        )
    
    if friday_results['autoencoder']:
        print_model_comparison(
            "Autoencoder",
            wednesday_results['autoencoder'],
            friday_results['autoencoder']
        )
    
    if friday_results['isolation_forest']:
        print_model_comparison(
            "Isolation Forest",
            wednesday_results['isolation_forest'],
            friday_results['isolation_forest']
        )
    
    if friday_results['ensemble']:
        print_model_comparison(
            "Ensemble",
            wednesday_results['ensemble'],
            friday_results['ensemble']
        )
    
    # Summary
    print_header("🎯 SUMMARY & INSIGHTS")
    
    print("\n✅ Key Findings:")
    
    # Compare OneClassSVM
    if friday_results['oneclass_svm']:
        wed_auc = wednesday_results['oneclass_svm']['roc_auc']
        fri_auc = friday_results['oneclass_svm'].get('test_evaluation', {}).get('roc_auc', 0)
        if fri_auc == 0:
            fri_auc = friday_results['oneclass_svm'].get('roc_auc', 0)
        if fri_auc > wed_auc:
            print(f"\n  🏆 OneClassSVM performs BETTER on Friday attacks")
            print(f"     (+{(fri_auc - wed_auc):.4f} ROC-AUC improvement)")
            print(f"     This suggests Friday attacks may be easier to detect")
        else:
            print(f"\n  📊 OneClassSVM performs slightly worse on Friday attacks")
            print(f"     ({(fri_auc - wed_auc):.4f} ROC-AUC decrease)")
            print(f"     This suggests Friday attacks may be more challenging")
    
    # Compare Autoencoder
    if friday_results['autoencoder']:
        wed_auc = wednesday_results['autoencoder']['roc_auc']
        fri_auc = friday_results['autoencoder'].get('evaluation', {}).get('roc_auc', 0)
        if fri_auc == 0:
            fri_auc = friday_results['autoencoder'].get('test_evaluation', {}).get('roc_auc', 0)
        if fri_auc == 0:
            fri_auc = friday_results['autoencoder'].get('roc_auc', 0)
        diff = fri_auc - wed_auc
        if abs(diff) < 0.01:
            print(f"\n  ⚖️ Autoencoder shows consistent performance across attack days")
            print(f"     (Difference: {diff:.4f} ROC-AUC)")
        elif diff > 0:
            print(f"\n  📈 Autoencoder improves on Friday attacks (+{diff:.4f})")
        else:
            print(f"\n  📉 Autoencoder degrades on Friday attacks ({diff:.4f})")
    
    # Compare Isolation Forest
    if friday_results['isolation_forest']:
        wed_auc = wednesday_results['isolation_forest']['roc_auc']
        fri_auc = friday_results['isolation_forest'].get('best_roc_auc', 0)
        if fri_auc == 0:
            fri_auc = friday_results['isolation_forest'].get('evaluation', {}).get('roc_auc', 0)
        if fri_auc == 0:
            fri_auc = friday_results['isolation_forest'].get('test_evaluation', {}).get('roc_auc', 0)
        if fri_auc == 0:
            fri_auc = friday_results['isolation_forest'].get('roc_auc', 0)
        diff = fri_auc - wed_auc
        if abs(diff) < 0.01:
            print(f"\n  ⚠️ Isolation Forest remains poor on both attack days")
            print(f"     (ROC-AUC ~0.58, barely better than random)")
        elif diff > 0:
            print(f"\n  📈 Isolation Forest slightly better on Friday (+{diff:.4f})")
        else:
            print(f"\n  📉 Isolation Forest worse on Friday ({diff:.4f})")
    
    # Compare Ensemble
    if friday_results['ensemble']:
        wed_auc = wednesday_results['ensemble']['roc_auc']
        fri_auc = friday_results['ensemble'].get('roc_auc', 0)
        if fri_auc == 0 and 'test_evaluation' in friday_results['ensemble']:
            fri_auc = friday_results['ensemble']['test_evaluation'].get('roc_auc', 0)
        diff = fri_auc - wed_auc
        print(f"\n  🤝 Ensemble Performance:")
        print(f"     Wednesday: {wed_auc:.4f} ROC-AUC")
        print(f"     Friday:    {fri_auc:.4f} ROC-AUC")
        print(f"     Change:    {diff:+.4f} ({diff/wed_auc*100:+.2f}%)")
    
    print("\n💡 Recommendations:")
    print("  • OneClassSVM is the best model for both attack days")
    print("  • Performance differences indicate attack characteristics vary by day")
    print("  • Consider retraining periodically to adapt to new attack patterns")
    print("  • Ensemble approach provides good robustness across different attacks")
    
    # Save comparison results
    def safe_get_auc(result_dict):
        """Safely extract ROC-AUC from result dict."""
        if not result_dict:
            return None
        auc = result_dict.get('best_roc_auc', 0)
        if auc == 0:
            auc = result_dict.get('roc_auc', 0)
        if auc == 0 and 'test_evaluation' in result_dict:
            auc = result_dict['test_evaluation'].get('roc_auc', 0)
        if auc == 0 and 'evaluation' in result_dict:
            auc = result_dict['evaluation'].get('roc_auc', 0)
        return auc if auc > 0 else None
    
    comparison_results = {
        'wednesday': wednesday_results,
        'friday': friday_results,
        'summary': {
            'oneclass_svm_improvement': (
                safe_get_auc(friday_results['oneclass_svm']) - 
                wednesday_results['oneclass_svm']['roc_auc']
            ) if friday_results['oneclass_svm'] and safe_get_auc(friday_results['oneclass_svm']) else None,
            'autoencoder_improvement': (
                safe_get_auc(friday_results['autoencoder']) - 
                wednesday_results['autoencoder']['roc_auc']
            ) if friday_results['autoencoder'] and safe_get_auc(friday_results['autoencoder']) else None,
            'isolation_forest_improvement': (
                safe_get_auc(friday_results['isolation_forest']) - 
                wednesday_results['isolation_forest']['roc_auc']
            ) if friday_results['isolation_forest'] and safe_get_auc(friday_results['isolation_forest']) else None,
            'ensemble_improvement': (
                safe_get_auc(friday_results['ensemble']) - 
                wednesday_results['ensemble']['roc_auc']
            ) if friday_results['ensemble'] and safe_get_auc(friday_results['ensemble']) else None
        }
    }
    
    output_file = Path("models") / "wednesday_vs_friday_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n📁 Comparison results saved: {output_file}")
    print("\n✅ Comparison completed!")

if __name__ == "__main__":
    main()
