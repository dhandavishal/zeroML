#!/usr/bin/env python3
"""
Create dataset splits with choice of attack data (Wednesday or Friday)
"""

import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_splits_with_attack_choice(attack_day='friday', val_size=0.2, max_test_samples=50000, random_state=42):
    """
    Create train/val/test splits with specified attack day.
    
    Args:
        attack_day: 'wednesday' or 'friday'
        val_size: Validation split ratio
        max_test_samples: Maximum test set size
        random_state: Random seed
    """
    
    print(f"🚀 Creating dataset splits with {attack_day.upper()} attack data")
    print("=" * 60)
    
    data_dir = Path("data/processed")
    output_dir = Path("data/interim")
    output_dir.mkdir(exist_ok=True)
    
    # Load benign data (Monday)
    print("📥 Loading benign data...")
    monday_benign = np.load(data_dir / "monday_benign.npy")
    y_benign = np.zeros(len(monday_benign), dtype=np.int8)
    print(f"   Monday benign: {monday_benign.shape[0]:,} packets")
    
    # Load selected attack data
    print(f"📥 Loading {attack_day} attack data...")
    attack_file = data_dir / f"{attack_day}_attack.npy"
    
    if not attack_file.exists():
        print(f"❌ Error: {attack_file} not found!")
        print(f"   Available files: {list(data_dir.glob('*.npy'))}")
        return False
    
    attack_data = np.load(attack_file)
    y_attack = np.ones(len(attack_data), dtype=np.int8)
    print(f"   {attack_day.capitalize()} attack: {attack_data.shape[0]:,} packets")
    
    # Step 1: Split benign data into train/val
    print(f"\n🔄 Creating benign train/validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        monday_benign, y_benign,
        test_size=val_size,
        random_state=random_state
    )
    
    print(f"   Training: {len(X_train):,} samples (80%)")
    print(f"   Validation: {len(X_val):,} samples (20%)")
    
    # Save benign splits
    benign_dir = output_dir / "benign"
    benign_dir.mkdir(exist_ok=True)
    
    np.save(benign_dir / "X_train.npy", X_train)
    np.save(benign_dir / "y_train.npy", y_train)
    np.save(benign_dir / "X_val.npy", X_val)
    np.save(benign_dir / "y_val.npy", y_val)
    
    # Step 2: Create mixed test set
    print(f"\n🔄 Creating mixed test set...")
    
    # Sample for balanced test set (60% benign, 40% attack)
    target_benign = int(max_test_samples * 0.6)
    target_attack = int(max_test_samples * 0.4)
    
    # Use remaining benign data (not used in train/val)
    # Or sample from all benign data
    benign_test_size = min(target_benign, len(monday_benign))
    attack_test_size = min(target_attack, len(attack_data))
    
    np.random.seed(random_state)
    benign_test_idx = np.random.choice(len(monday_benign), benign_test_size, replace=False)
    attack_test_idx = np.random.choice(len(attack_data), attack_test_size, replace=False)
    
    X_test_benign = monday_benign[benign_test_idx]
    y_test_benign = y_benign[benign_test_idx]
    
    X_test_attack = attack_data[attack_test_idx]
    y_test_attack = y_attack[attack_test_idx]
    
    # Combine and shuffle
    X_test = np.vstack([X_test_benign, X_test_attack])
    y_test = np.concatenate([y_test_benign, y_test_attack])
    
    shuffle_idx = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_idx]
    y_test = y_test[shuffle_idx]
    
    print(f"   Test set: {len(X_test):,} samples")
    print(f"   - Benign: {np.sum(y_test==0):,} ({np.sum(y_test==0)/len(y_test)*100:.1f}%)")
    print(f"   - Attack: {np.sum(y_test==1):,} ({np.sum(y_test==1)/len(y_test)*100:.1f}%)")
    
    # Save test set
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    
    # Save metadata
    metadata = {
        'attack_day': attack_day,
        'creation_date': str(np.datetime64('now')),
        'splits': {
            'train': {
                'samples': int(len(X_train)),
                'label_distribution': {'benign': int(len(X_train))}
            },
            'val': {
                'samples': int(len(X_val)),
                'label_distribution': {'benign': int(len(X_val))}
            },
            'test': {
                'samples': int(len(X_test)),
                'label_distribution': {
                    'benign': int(np.sum(y_test==0)),
                    'attack': int(np.sum(y_test==1))
                }
            }
        },
        'data_sources': {
            'benign': 'monday_benign.npy',
            'attack': f'{attack_day}_attack.npy'
        }
    }
    
    metadata_path = output_dir / f"splits_metadata_{attack_day}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset splits created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Metadata saved: {metadata_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Create dataset splits with attack day choice')
    parser.add_argument('--attack-day', type=str, default='friday', 
                       choices=['wednesday', 'friday'],
                       help='Which attack day to use for test set (default: friday)')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Validation set size ratio (default: 0.2)')
    parser.add_argument('--max-test', type=int, default=50000,
                       help='Maximum test set size (default: 50000)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    success = create_splits_with_attack_choice(
        attack_day=args.attack_day,
        val_size=args.val_size,
        max_test_samples=args.max_test,
        random_state=args.random_seed
    )
    
    if success:
        print("\n🎉 Ready for feature engineering and model training!")
    else:
        print("\n❌ Split creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
