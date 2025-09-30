#!/usr/bin/env python3
"""
Manifest Management System for ZeroML
Creates, validates, and manages data catalogs for ML training
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

class ManifestManager:
    def __init__(self, manifest_path="data/manifest.txt", data_dir="data/processed"):
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir)
        self.columns = [
            'filename', 'label', 'attack_type', 'packet_count', 
            'file_size_mb', 'source_dataset', 'date_processed', 'split_purpose'
        ]
    
    def create_manifest_from_directory(self):
        """Auto-generate manifest from NPY files in directory"""
        print(f"🔍 Scanning {self.data_dir} for NPY files...")
        
        npy_files = list(self.data_dir.glob("*.npy"))
        if not npy_files:
            print(f"❌ No NPY files found in {self.data_dir}")
            return False
        
        manifest_entries = []
        
        for npy_file in npy_files:
            print(f"   Analyzing {npy_file.name}...")
            
            try:
                # Load file to get stats
                X = np.load(npy_file)
                packet_count = X.shape[0]
                file_size_mb = round(npy_file.stat().st_size / (1024 * 1024), 1)
                
                # Infer label and attack type from filename
                filename_lower = npy_file.name.lower()
                if 'benign' in filename_lower or 'normal' in filename_lower:
                    label = 0
                    attack_type = 'benign'
                    split_purpose = 'train_val'
                elif 'attack' in filename_lower or 'malicious' in filename_lower:
                    label = 1
                    attack_type = 'mixed_attacks'
                    split_purpose = 'test'
                else:
                    # Default assumption
                    label = 0
                    attack_type = 'unknown'
                    split_purpose = 'unknown'
                
                # Infer source dataset from filename
                if 'monday' in filename_lower:
                    source_dataset = 'CICIDS2017_Monday'
                elif 'tuesday' in filename_lower:
                    source_dataset = 'CICIDS2017_Tuesday'
                elif 'wednesday' in filename_lower:
                    source_dataset = 'CICIDS2017_Wednesday'
                elif 'thursday' in filename_lower:
                    source_dataset = 'CICIDS2017_Thursday'
                elif 'friday' in filename_lower:
                    source_dataset = 'CICIDS2017_Friday'
                else:
                    source_dataset = 'Unknown'
                
                entry = {
                    'filename': npy_file.name,
                    'label': label,
                    'attack_type': attack_type,
                    'packet_count': packet_count,
                    'file_size_mb': file_size_mb,
                    'source_dataset': source_dataset,
                    'date_processed': datetime.now().strftime('%Y-%m-%d'),
                    'split_purpose': split_purpose
                }
                
                manifest_entries.append(entry)
                print(f"     ✅ {packet_count:,} packets, {file_size_mb}MB, label={label}")
                
            except Exception as e:
                print(f"     ❌ Error processing {npy_file.name}: {e}")
                continue
        
        if not manifest_entries:
            print("❌ No valid NPY files processed")
            return False
        
        # Save manifest
        self.save_manifest(manifest_entries)
        return True
    
    def save_manifest(self, entries):
        """Save manifest entries to file"""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.manifest_path, 'w') as f:
            # Write header
            f.write("# ZeroML Dataset Manifest\n")
            f.write("# Format: filename,label,attack_type,packet_count,file_size_mb,source_dataset,date_processed,split_purpose\n")
            f.write("# Labels: benign=0, attack=1\n")
            f.write("# Attack types: benign, dos, ddos, web_attack, infiltration, botnet, brute_force, heartbleed, sql_injection, xss\n\n")
            
            # Write entries
            for entry in entries:
                line = ",".join([
                    entry['filename'],
                    str(entry['label']),
                    entry['attack_type'],
                    str(entry['packet_count']),
                    str(entry['file_size_mb']),
                    entry['source_dataset'],
                    entry['date_processed'],
                    entry['split_purpose']
                ])
                f.write(line + "\n")
        
        print(f"📝 Manifest saved to {self.manifest_path}")
    
    def load_manifest(self):
        """Load manifest file as DataFrame"""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        # Read manifest, skipping comment lines
        lines = []
        with open(self.manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    lines.append(line.split(','))
        
        if not lines:
            raise ValueError("No data entries found in manifest file")
        
        df = pd.DataFrame(lines, columns=self.columns)
        
        # Convert numeric columns
        df['label'] = df['label'].astype(int)
        df['packet_count'] = df['packet_count'].astype(int)
        df['file_size_mb'] = df['file_size_mb'].astype(float)
        
        return df
    
    def validate_manifest(self):
        """Validate manifest against actual files"""
        print("🔍 Validating manifest...")
        
        try:
            df = self.load_manifest()
        except Exception as e:
            print(f"❌ Failed to load manifest: {e}")
            return False
        
        validation_errors = []
        
        for _, row in df.iterrows():
            file_path = self.data_dir / row['filename']
            
            # Check file exists
            if not file_path.exists():
                validation_errors.append(f"File not found: {row['filename']}")
                continue
            
            try:
                # Check file can be loaded
                X = np.load(file_path)
                
                # Validate packet count
                actual_packets = X.shape[0]
                expected_packets = row['packet_count']
                if actual_packets != expected_packets:
                    validation_errors.append(
                        f"{row['filename']}: Packet count mismatch. "
                        f"Expected {expected_packets}, got {actual_packets}"
                    )
                
                # Validate file size (allow 10% tolerance)
                actual_size = file_path.stat().st_size / (1024 * 1024)
                expected_size = row['file_size_mb']
                size_diff_pct = abs(actual_size - expected_size) / expected_size * 100
                if size_diff_pct > 10:
                    validation_errors.append(
                        f"{row['filename']}: File size mismatch. "
                        f"Expected {expected_size}MB, got {actual_size:.1f}MB"
                    )
                
            except Exception as e:
                validation_errors.append(f"{row['filename']}: Failed to load - {e}")
        
        if validation_errors:
            print(f"❌ Validation failed with {len(validation_errors)} errors:")
            for error in validation_errors:
                print(f"   - {error}")
            return False
        else:
            print(f"✅ Manifest validation passed for {len(df)} files")
            return True
    
    def get_files_by_split(self, split_purpose):
        """Get files filtered by split purpose"""
        df = self.load_manifest()
        filtered = df[df['split_purpose'] == split_purpose]
        return filtered
    
    def get_files_by_label(self, label):
        """Get files filtered by label (0=benign, 1=attack)"""
        df = self.load_manifest()
        filtered = df[df['label'] == label]
        return filtered
    
    def load_data_by_filter(self, **filters):
        """Load NPY data based on manifest filters"""
        df = self.load_manifest()
        
        # Apply filters
        for key, value in filters.items():
            if key in df.columns:
                df = df[df[key] == value]
        
        if df.empty:
            print(f"⚠️  No files match the filters: {filters}")
            return None, None
        
        print(f"📊 Loading {len(df)} files matching filters...")
        
        # Load and concatenate data
        X_list = []
        y_list = []
        
        for _, row in df.iterrows():
            file_path = self.data_dir / row['filename']
            print(f"   Loading {row['filename']}... ({row['packet_count']:,} packets)")
            
            X = np.load(file_path)
            y = np.full(X.shape[0], row['label'])
            
            X_list.append(X)
            y_list.append(y)
        
        # Concatenate all data
        X_combined = np.vstack(X_list)
        y_combined = np.concatenate(y_list)
        
        print(f"✅ Loaded combined data: {X_combined.shape}")
        return X_combined, y_combined
    
    def print_summary(self):
        """Print manifest summary"""
        try:
            df = self.load_manifest()
        except Exception as e:
            print(f"❌ Cannot load manifest: {e}")
            return
        
        print("\n" + "="*60)
        print("MANIFEST SUMMARY")
        print("="*60)
        
        print(f"📁 Total files: {len(df)}")
        print(f"📦 Total packets: {df['packet_count'].sum():,}")
        print(f"💾 Total size: {df['file_size_mb'].sum():.1f}MB")
        
        print(f"\n🏷️  By Label:")
        label_counts = df.groupby('label').agg({
            'filename': 'count',
            'packet_count': 'sum',
            'file_size_mb': 'sum'
        })
        for label, row in label_counts.iterrows():
            label_name = "Benign" if label == 0 else "Attack"
            print(f"   {label_name}: {row['filename']} files, {row['packet_count']:,} packets, {row['file_size_mb']:.1f}MB")
        
        print(f"\n📋 By Split Purpose:")
        split_counts = df.groupby('split_purpose').agg({
            'filename': 'count',
            'packet_count': 'sum'
        })
        for split, row in split_counts.iterrows():
            print(f"   {split}: {row['filename']} files, {row['packet_count']:,} packets")
        
        print(f"\n📂 Files:")
        for _, row in df.iterrows():
            print(f"   - {row['filename']}: {row['attack_type']}, {row['packet_count']:,} packets")

def main():
    parser = argparse.ArgumentParser(description="Manage ZeroML data manifest")
    parser.add_argument("--manifest", type=str, default="data/manifest.txt",
                        help="Path to manifest file")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing NPY files")
    parser.add_argument("--action", type=str, 
                        choices=["create", "validate", "summary", "load-test"],
                        default="create", help="Action to perform")
    parser.add_argument("--split", type=str, help="Filter by split purpose")
    parser.add_argument("--label", type=int, help="Filter by label (0=benign, 1=attack)")
    
    args = parser.parse_args()
    
    manager = ManifestManager(args.manifest, args.data_dir)
    
    if args.action == "create":
        print("🚀 Creating manifest from directory...")
        success = manager.create_manifest_from_directory()
        if success:
            manager.validate_manifest()
            manager.print_summary()
    
    elif args.action == "validate":
        success = manager.validate_manifest()
        return 0 if success else 1
    
    elif args.action == "summary":
        manager.print_summary()
    
    elif args.action == "load-test":
        filters = {}
        if args.split:
            filters['split_purpose'] = args.split
        if args.label is not None:
            filters['label'] = args.label
        
        X, y = manager.load_data_by_filter(**filters)
        if X is not None:
            print(f"✅ Successfully loaded: X={X.shape}, y={y.shape}")
            print(f"   Label distribution: {np.unique(y, return_counts=True)}")
    
    return 0

if __name__ == "__main__":
    exit(main())