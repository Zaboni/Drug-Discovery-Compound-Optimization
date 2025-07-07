#!/usr/bin/env python3
"""
Manual Tox21 Data Download Script
Downloads toxicity data and saves to current directory to avoid permission issues.
"""

import pandas as pd
import logging
from pathlib import Path
import requests
import io


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_tox21_data(dataset='tox21_train', max_records=None):
    """Download Tox21 toxicity data from DeepChem repository."""
    
    # DeepChem URLs for Tox21 datasets
    urls = {
        'tox21_train': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
        'tox21_full': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21_full.csv.gz'
    }
    
    if dataset not in urls:
        print(f"❌ Unknown dataset: {dataset}")
        print(f"Available datasets: {list(urls.keys())}")
        return None
    
    url = urls[dataset]
    print(f"🔍 Downloading {dataset} from DeepChem repository...")
    print(f"URL: {url}")
    
    try:
        # Download compressed CSV
        print("  Downloading compressed data...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        print(f"  Downloaded {len(response.content)} bytes")
        
        # Read compressed data
        print("  Reading compressed CSV...")
        df = pd.read_csv(io.BytesIO(response.content), compression='gzip')
        
        # Limit records if specified
        if max_records and len(df) > max_records:
            df = df.head(max_records)
            print(f"  Limited to {max_records} records")
        
        print(f"✅ Successfully loaded {len(df)} records")
        
        # Show basic info about the dataset
        print(f"\n📊 Dataset Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        
        # Identify assay columns (toxicity endpoints)
        assay_columns = [col for col in df.columns if any(x in col for x in ['NR-', 'SR-'])]
        print(f"   Toxicity assays: {len(assay_columns)}")
        
        if assay_columns:
            print(f"   Assay examples: {assay_columns[:5]}")
        
        # Check for SMILES column
        smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
        if smiles_cols:
            print(f"   SMILES column: {smiles_cols[0]}")
            valid_smiles = int(df[smiles_cols[0]].notna().sum())
            print(f"   Valid SMILES: {valid_smiles}/{len(df)} ({valid_smiles/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading {dataset}: {e}")
        return None

def analyze_toxicity_data(df):
    """Analyze toxicity data and show summary statistics."""
    
    if df is None or df.empty:
        return
    
    print(f"\n🧪 Toxicity Analysis:")
    
    # Find assay columns
    assay_columns = [col for col in df.columns if any(x in col for x in ['NR-', 'SR-'])]
    
    if not assay_columns:
        print("   No toxicity assay columns found")
        return
    
    # Analyze each assay
    assay_stats = []
    for col in assay_columns[:10]:  # Show first 10 assays
        if col in df.columns:
            # Count non-null values
            tested = int(df[col].notna().sum())
            if tested > 0:
                # Count active (assuming 1 = active, 0 = inactive)
                active = int(df[col].eq(1).sum())
                inactive = int(df[col].eq(0).sum())
                activity_rate = active / tested * 100 if tested > 0 else 0
                
                assay_stats.append({
                    'assay': col,
                    'tested': tested,
                    'active': active,
                    'inactive': inactive,
                    'activity_rate': activity_rate,
                    'coverage': tested / len(df) * 100
                })
    
    if assay_stats:
        stats_df = pd.DataFrame(assay_stats)
        print(f"   Top assays by coverage:")
        for _, row in stats_df.head(5).iterrows():
            print(f"     {row['assay']}: {row['tested']} tested, {row['active']} active ({row['activity_rate']:.1f}%)")

def save_data(df, filename):
    """Save data to file with fallback options."""
    
    if df is None or df.empty:
        print("❌ No data to save")
        return False
    
    # Try current directory first
    try:
        df.to_csv(filename, index=False)
        print(f"📁 Data saved to: {Path(filename).absolute()}")
        return True
    except Exception as e:
        print(f"❌ Error saving to current directory: {e}")
        
        # Try Downloads folder
        try:
            downloads_path = Path.home() / "Downloads" / filename
            df.to_csv(downloads_path, index=False)
            print(f"📁 Data saved to Downloads: {downloads_path}")
            return True
        except Exception as e2:
            print(f"❌ Also failed to save to Downloads: {e2}")
            
            # Last resort - show sample data
            print(f"\n📋 Sample data (first 5 rows):")
            print(df.head().to_string())
            return False

if __name__ == "__main__":
    print("🧪 Manual Tox21 Download Script")
    print("=" * 40)
    
    # Download training dataset
    print("\n1️⃣ Downloading Tox21 training dataset...")
    df_train = download_tox21_data('tox21_train', max_records=1000)
    
    if df_train is not None:
        analyze_toxicity_data(df_train)
        save_data(df_train, "tox21_train_sample.csv")
    

    print(f"\n🎉 Manual download complete!")
    print("You can move this file to your data/raw directory manually if needed.")
    print("\nFile created:")
    print("- tox21_train_sample.csv: Training data with toxicity endpoints")