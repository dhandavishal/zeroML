#!/usr/bin/env python3
"""
Dataset Splitting Script for ZeroML
Creates proper train/validation/test splits for anomaly detection
"""

import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# Add scripts directory to path for manifest manager
sys.path.append(str(Path(__file__).parent))
from create_manifest import ManifestManager

class DatasetSplitter:
    def __init__(self, manifest_path="data/manifest.txt", output_dir="data/interim"):
        self.manifest_manager = ManifestManager(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_benign_train_val_split(self, val_size=0.2, random_state=42):
        """Split benign data into training and validation sets"""
        print("🔄 Creating benign train/validation split...")
        
        # Load all benign data
        X_benign, y_benign = self.manifest_manager.load_data_by_filter(label=0)
        
        if X_benign is None:
            print("❌ No benign data found")
            return False
        
        print(f"📊 Loaded benign data: {X_benign.shape}")
        print(f"   Unique labels: {np.unique(y_benign)}")
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X_benign, y_benign, 
            test_size=val_size, 
            random_state=random_state,
            stratify=None  # No stratification needed for single class
        )
        
        # Create output directories
        benign_dir = self.output_dir / "benign"
        benign_dir.mkdir(exist_ok=True)
        
        # Save splits
        train_path = benign_dir / "X_train.npy"
        val_path = benign_dir / "X_val.npy"
        train_labels_path = benign_dir / "y_train.npy"
        val_labels_path = benign_dir / "y_val.npy"
        
        np.save(train_path, X_train)
        np.save(val_path, X_val)
        np.save(train_labels_path, y_train)
        np.save(val_labels_path, y_val)
        
        # Save split metadata
        split_info = {
            "split_date": pd.Timestamp.now().isoformat(),
            "original_samples": int(X_benign.shape[0]),
            "original_features": int(X_benign.shape[1]),
            "val_size": val_size,
            "random_state": random_state,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "train_file": str(train_path),
            "val_file": str(val_path),
            "data_type": "benign_only"
        }
        
        metadata_path = benign_dir / "split_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"✅ Benign split completed:")
        print(f"   Training: {X_train.shape[0]:,} samples → {train_path}")
        print(f"   Validation: {X_val.shape[0]:,} samples → {val_path}")
        print(f"   Split ratio: {(1-val_size)*100:.0f}% train, {val_size*100:.0f}% val")
        print(f"   Metadata: {metadata_path}")
        
        return split_info
    
    def create_mixed_test_set(self, benign_ratio=0.6, max_samples=50000, random_state=42):
        """Create balanced test set with both benign and attack samples"""
        print("🔄 Creating mixed test set...")
        
        # Load attack data
        X_attack, y_attack = self.manifest_manager.load_data_by_filter(label=1)
        if X_attack is None:
            print("❌ No attack data found")
            return False
        
        # Load benign data (we'll sample from it)
        X_benign, y_benign = self.manifest_manager.load_data_by_filter(label=0)
        if X_benign is None:
            print("❌ No benign data found")
            return False
        
        print(f"📊 Available data:")
        print(f"   Benign: {X_benign.shape[0]:,} samples")
        print(f"   Attack: {X_attack.shape[0]:,} samples")
        
        # Calculate target sample sizes
        target_benign = int(max_samples * benign_ratio)
        target_attack = int(max_samples * (1 - benign_ratio))
        
        # Sample benign data (avoid train/val contamination)
        np.random.seed(random_state)
        if X_benign.shape[0] > target_benign:
            benign_indices = np.random.choice(X_benign.shape[0], target_benign, replace=False)
            X_benign_test = X_benign[benign_indices]
            y_benign_test = y_benign[benign_indices]
        else:
            X_benign_test = X_benign
            y_benign_test = y_benign
        
        # Sample attack data
        if X_attack.shape[0] > target_attack:
            attack_indices = np.random.choice(X_attack.shape[0], target_attack, replace=False)
            X_attack_test = X_attack[attack_indices]
            y_attack_test = y_attack[attack_indices]
        else:
            X_attack_test = X_attack
            y_attack_test = y_attack
        
        # Combine test data
        X_test = np.vstack([X_benign_test, X_attack_test])
        y_test = np.concatenate([y_benign_test, y_attack_test])
        
        # Shuffle the combined test set
        shuffle_indices = np.random.permutation(len(X_test))
        X_test = X_test[shuffle_indices]
        y_test = y_test[shuffle_indices]
        
        # Save test set
        test_path = self.output_dir / "X_test.npy"
        test_labels_path = self.output_dir / "y_test.npy"
        
        np.save(test_path, X_test)
        np.save(test_labels_path, y_test)
        
        # Calculate actual ratios
        actual_benign = np.sum(y_test == 0)
        actual_attack = np.sum(y_test == 1)
        actual_benign_ratio = actual_benign / len(y_test)
        
        # Save test set metadata
        test_info = {
            "split_date": pd.Timestamp.now().isoformat(),
            "target_samples": max_samples,
            "actual_samples": int(len(X_test)),
            "target_benign_ratio": benign_ratio,
            "actual_benign_ratio": round(actual_benign_ratio, 3),
            "benign_samples": int(actual_benign),
            "attack_samples": int(actual_attack),
            "features": int(X_test.shape[1]),
            "test_file": str(test_path),
            "test_labels_file": str(test_labels_path),
            "random_state": random_state,
            "data_type": "mixed_test"
        }
        
        metadata_path = self.output_dir / "test_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(test_info, f, indent=2)
        
        print(f"✅ Mixed test set created:")
        print(f"   Total samples: {len(X_test):,}")
        print(f"   Benign: {actual_benign:,} ({actual_benign_ratio:.1%})")
        print(f"   Attack: {actual_attack:,} ({1-actual_benign_ratio:.1%})")
        print(f"   Features: {X_test.shape[1]}")
        print(f"   Test data: {test_path}")
        print(f"   Test labels: {test_labels_path}")
        print(f"   Metadata: {metadata_path}")
        
        return test_info
    
    def validate_splits(self):
        """Validate all created splits"""
        print("🔍 Validating dataset splits...")
        
        splits_info = []
        issues = []
        
        # Check benign splits
        benign_dir = self.output_dir / "benign"
        if benign_dir.exists():
            try:
                X_train = np.load(benign_dir / "X_train.npy")
                X_val = np.load(benign_dir / "X_val.npy")
                y_train = np.load(benign_dir / "y_train.npy")
                y_val = np.load(benign_dir / "y_val.npy")
                
                # Validate shapes
                if X_train.shape[0] != y_train.shape[0]:
                    issues.append("Train features/labels shape mismatch")
                if X_val.shape[0] != y_val.shape[0]:
                    issues.append("Validation features/labels shape mismatch")
                if X_train.shape[1] != X_val.shape[1]:
                    issues.append("Train/validation feature count mismatch")
                
                splits_info.append({
                    "split": "benign_train",
                    "samples": X_train.shape[0],
                    "features": X_train.shape[1],
                    "labels": np.unique(y_train).tolist()
                })
                splits_info.append({
                    "split": "benign_val",
                    "samples": X_val.shape[0],
                    "features": X_val.shape[1],
                    "labels": np.unique(y_val).tolist()
                })
                
            except Exception as e:
                issues.append(f"Failed to load benign splits: {e}")
        
        # Check test split
        try:
            X_test = np.load(self.output_dir / "X_test.npy")
            y_test = np.load(self.output_dir / "y_test.npy")
            
            if X_test.shape[0] != y_test.shape[0]:
                issues.append("Test features/labels shape mismatch")
            
            splits_info.append({
                "split": "test",
                "samples": X_test.shape[0],
                "features": X_test.shape[1],
                "labels": np.unique(y_test).tolist(),
                "label_counts": np.unique(y_test, return_counts=True)[1].tolist()
            })
            
        except Exception as e:
            issues.append(f"Failed to load test split: {e}")
        
        if issues:
            print(f"❌ Validation failed with {len(issues)} issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print(f"✅ All splits validated successfully")
            for split in splits_info:
                print(f"   - {split['split']}: {split['samples']:,} samples, "
                      f"{split['features']} features, labels={split['labels']}")
            return True
    
    def print_dataset_summary(self):
        """Print comprehensive dataset summary"""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        total_samples = 0
        
        # Benign splits
        benign_dir = self.output_dir / "benign"
        if benign_dir.exists():
            try:
                X_train = np.load(benign_dir / "X_train.npy")
                X_val = np.load(benign_dir / "X_val.npy")
                
                print(f"🟢 BENIGN DATA (Training):")
                print(f"   Training: {X_train.shape[0]:,} samples")
                print(f"   Validation: {X_val.shape[0]:,} samples")
                print(f"   Total benign: {X_train.shape[0] + X_val.shape[0]:,} samples")
                print(f"   Features: {X_train.shape[1]}")
                
                total_samples += X_train.shape[0] + X_val.shape[0]
                
            except:
                print(f"⚠️  Benign splits not found")
        
        # Test set
        try:
            X_test = np.load(self.output_dir / "X_test.npy")
            y_test = np.load(self.output_dir / "y_test.npy")
            
            benign_test = np.sum(y_test == 0)
            attack_test = np.sum(y_test == 1)
            
            print(f"\n🔵 TEST DATA (Mixed):")
            print(f"   Total test: {len(y_test):,} samples")
            print(f"   Benign: {benign_test:,} samples ({benign_test/len(y_test):.1%})")
            print(f"   Attack: {attack_test:,} samples ({attack_test/len(y_test):.1%})")
            print(f"   Features: {X_test.shape[1]}")
            
            total_samples += len(y_test)
            
        except:
            print(f"⚠️  Test set not found")
        
        print(f"\n📊 OVERALL:")
        print(f"   Total samples processed: {total_samples:,}")
        print(f"   Ready for anomaly detection training")

def main():
    parser = argparse.ArgumentParser(description="Create dataset splits for ZeroML")
    parser.add_argument("--manifest", type=str, default="data/manifest.txt",
                        help="Path to manifest file")
    parser.add_argument("--output-dir", type=str, default="data/interim",
                        help="Output directory for splits")
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Validation set size (0.0-1.0)")
    parser.add_argument("--test-benign-ratio", type=float, default=0.6,
                        help="Ratio of benign samples in test set")
    parser.add_argument("--max-test-samples", type=int, default=50000,
                        help="Maximum samples in test set")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--action", type=str, 
                        choices=["splits", "validate", "summary"],
                        default="splits", help="Action to perform")
    
    args = parser.parse_args()
    
    splitter = DatasetSplitter(args.manifest, args.output_dir)
    
    if args.action == "splits":
        print("🚀 Creating dataset splits...")
        
        # Create benign train/val split
        benign_info = splitter.create_benign_train_val_split(
            val_size=args.val_size,
            random_state=args.random_state
        )
        
        if not benign_info:
            return 1
        
        # Create mixed test set
        test_info = splitter.create_mixed_test_set(
            benign_ratio=args.test_benign_ratio,
            max_samples=args.max_test_samples,
            random_state=args.random_state
        )
        
        if not test_info:
            return 1
        
        # Validate all splits
        if splitter.validate_splits():
            splitter.print_dataset_summary()
            print("\n✅ Dataset splitting completed successfully!")
        else:
            return 1
    
    elif args.action == "validate":
        return 0 if splitter.validate_splits() else 1
    
    elif args.action == "summary":
        splitter.print_dataset_summary()
    
    return 0

if __name__ == "__main__":
    # Need pandas for timestamp
    import pandas as pd
    exit(main())