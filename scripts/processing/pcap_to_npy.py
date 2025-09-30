#!/usr/bin/env python3
"""
PCAP to NPY Conversion Script with Error Handling
Converts PCAP files to CSV using nprint, then to NPY arrays for ML training
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import time

# Add src to Python path for importing our modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from zeroml.nprint_io import run_nprint, csv_to_numpy

class PCAPProcessor:
    def __init__(self, nprint_path="./nprint_bin/nprint"):
        self.nprint_path = nprint_path
        self.stats = {
            'processed_files': [],
            'failed_files': [],
            'total_packets': 0,
            'processing_time': 0,
            'start_time': None,
            'end_time': None
        }
    
    def validate_nprint(self):
        """Check if nprint binary exists and is executable"""
        if not os.path.exists(self.nprint_path):
            raise FileNotFoundError(f"nprint binary not found at {self.nprint_path}")
        
        if not os.access(self.nprint_path, os.X_OK):
            raise PermissionError(f"nprint binary at {self.nprint_path} is not executable")
        
        try:
            result = subprocess.run([self.nprint_path, "--version"], 
                                 capture_output=True, text=True, timeout=10)
            print(f"✅ nprint validation successful")
            return True
        except Exception as e:
            raise RuntimeError(f"nprint binary test failed: {e}")
    
    def process_single_pcap(self, pcap_path, output_dir, headers="-4 -t -u -i", 
                           count=50000, exclude_regex=None):
        """Process a single PCAP file to NPY"""
        pcap_path = Path(pcap_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filenames
        base_name = pcap_path.stem
        csv_path = output_dir / f"{base_name}.csv"
        npy_path = output_dir / f"{base_name}.npy"
        
        print(f"\n🔄 Processing {pcap_path.name}...")
        print(f"   Headers: {headers}")
        print(f"   Count limit: {count}")
        
        start_time = time.time()
        
        try:
            # Step 1: PCAP to CSV using nprint
            print(f"   📊 Converting PCAP to CSV...")
            cmd = [self.nprint_path, "-P", str(pcap_path)] + headers.split() + \
                  ["-c", str(count), "-W", str(csv_path), "-S"]
            
            if exclude_regex:
                cmd += ["-x", exclude_regex]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"nprint failed: {result.stderr}")
            
            # Check CSV was created and has content
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                raise RuntimeError("CSV file was not created or is empty")
            
            # Step 2: CSV to NPY
            print(f"   🔄 Converting CSV to NPY...")
            df = pd.read_csv(csv_path)
            
            # Remove the first column (src_ip) which is non-numeric
            if 'src_ip' in df.columns:
                df = df.drop('src_ip', axis=1)
            
            # Convert to numpy array
            X = df.to_numpy(dtype=np.int8)
            
            # Save NPY file
            np.save(npy_path, X)
            
            # Collect statistics
            processing_time = time.time() - start_time
            num_packets = X.shape[0]
            num_features = X.shape[1]
            file_size_mb = npy_path.stat().st_size / (1024 * 1024)
            
            file_stats = {
                'pcap_file': str(pcap_path),
                'csv_file': str(csv_path),
                'npy_file': str(npy_path),
                'num_packets': int(num_packets),
                'num_features': int(num_features),
                'file_size_mb': round(file_size_mb, 2),
                'processing_time_sec': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            self.stats['processed_files'].append(file_stats)
            self.stats['total_packets'] += num_packets
            
            print(f"   ✅ Success! {num_packets} packets, {num_features} features")
            print(f"   💾 NPY file: {file_size_mb:.1f}MB")
            print(f"   ⏱️  Processing time: {processing_time:.1f}s")
            
            return file_stats
            
        except subprocess.TimeoutExpired:
            error_msg = f"nprint processing timeout (>300s)"
            print(f"   ❌ {error_msg}")
            self.stats['failed_files'].append({
                'pcap_file': str(pcap_path),
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return None
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(f"   ❌ {error_msg}")
            self.stats['failed_files'].append({
                'pcap_file': str(pcap_path),
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def process_batch(self, pcap_files, output_dir, **kwargs):
        """Process multiple PCAP files"""
        self.stats['start_time'] = datetime.now().isoformat()
        print(f"🚀 Starting batch processing of {len(pcap_files)} PCAP files")
        
        for i, pcap_file in enumerate(pcap_files, 1):
            print(f"\n[{i}/{len(pcap_files)}] Processing {Path(pcap_file).name}")
            self.process_single_pcap(pcap_file, output_dir, **kwargs)
        
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['processing_time'] = sum(f.get('processing_time_sec', 0) 
                                          for f in self.stats['processed_files'])
        
        self.print_summary()
        return self.stats
    
    def print_summary(self):
        """Print processing statistics"""
        print(f"\n" + "="*60)
        print(f"📊 PROCESSING SUMMARY")
        print(f"="*60)
        print(f"✅ Successfully processed: {len(self.stats['processed_files'])} files")
        print(f"❌ Failed: {len(self.stats['failed_files'])} files")
        print(f"📦 Total packets processed: {self.stats['total_packets']:,}")
        print(f"⏱️  Total processing time: {self.stats['processing_time']:.1f}s")
        
        if self.stats['failed_files']:
            print(f"\n❌ FAILED FILES:")
            for failed in self.stats['failed_files']:
                print(f"   - {Path(failed['pcap_file']).name}: {failed['error']}")
        
        print(f"\n📁 OUTPUT FILES:")
        for processed in self.stats['processed_files']:
            print(f"   - {Path(processed['npy_file']).name}: "
                  f"{processed['num_packets']:,} packets, "
                  f"{processed['file_size_mb']}MB")

def main():
    parser = argparse.ArgumentParser(description="Convert PCAP files to NPY arrays")
    parser.add_argument("--input-dir", type=str, default="data/raw",
                        help="Directory containing PCAP files")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Directory to save NPY files")
    parser.add_argument("--nprint-path", type=str, default="./nprint_bin/nprint",
                        help="Path to nprint binary")
    parser.add_argument("--headers", type=str, default="-4 -t -u -i",
                        help="nprint header flags")
    parser.add_argument("--count", type=int, default=50000,
                        help="Maximum packets to process per file")
    parser.add_argument("--exclude-regex", type=str, default=None,
                        help="Regex pattern to exclude from nprint output")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Specific PCAP files to process")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PCAPProcessor(args.nprint_path)
    
    try:
        # Validate nprint
        processor.validate_nprint()
        
        # Find PCAP files
        if args.files:
            pcap_files = args.files
        else:
            input_dir = Path(args.input_dir)
            pcap_files = list(input_dir.glob("*.pcap"))
            
        if not pcap_files:
            print(f"❌ No PCAP files found in {args.input_dir}")
            return 1
        
        print(f"📁 Found {len(pcap_files)} PCAP files:")
        for f in pcap_files:
            size_mb = Path(f).stat().st_size / (1024 * 1024)
            print(f"   - {Path(f).name}: {size_mb:.1f}MB")
        
        # Process files
        stats = processor.process_batch(
            pcap_files, 
            args.output_dir,
            headers=args.headers,
            count=args.count,
            exclude_regex=args.exclude_regex
        )
        
        # Save statistics
        stats_file = Path(args.output_dir) / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n📊 Statistics saved to: {stats_file}")
        
        # Return appropriate exit code
        return 0 if not stats['failed_files'] else 1
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())