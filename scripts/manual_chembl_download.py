#!/usr/bin/env python3
"""
Manual ChEMBL Data Download Script
Downloads data and saves to current directory to avoid permission issues.
"""

import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    logger.error("ChEMBL client not available. Install with: pip install chembl_webresource_client")

def download_chembl_data(target_id="CHEMBL279", max_records=100):
    """Download ChEMBL data manually."""
    
    if not CHEMBL_AVAILABLE:
        print("❌ ChEMBL client not available")
        return None
    
    print(f"🔍 Downloading data for target {target_id}...")
    
    # Get activities
    activities = []
    activity_types = ['IC50', 'EC50', 'Ki', 'Kd']
    
    for activity_type in activity_types:
        try:
            print(f"  Downloading {activity_type} activities...")
            
            # Query ChEMBL
            activity_data = new_client.activity.filter(
                target_chembl_id=target_id,
                standard_type=activity_type,
                standard_value__isnull=False
            ).only([
                'molecule_chembl_id',
                'canonical_smiles', 
                'standard_type',
                'standard_value',
                'standard_units',
                'pchembl_value',
                'assay_chembl_id',
                'target_chembl_id'
            ])
            
            # Convert to list
            activity_list = list(activity_data)[:max_records//4]  # Split across activity types
            activities.extend(activity_list)
            
            print(f"    Found {len(activity_list)} {activity_type} activities")
            
        except Exception as e:
            print(f"    Error with {activity_type}: {e}")
    
    if activities:
        df = pd.DataFrame(activities)
        print(f"✅ Total activities downloaded: {len(df)}")
        
        # Save to current directory (should avoid permission issues)
        filename = f"chembl_{target_id}_activities.csv"
        try:
            df.to_csv(filename, index=False)
            print(f"📁 Data saved to: {Path(filename).absolute()}")
            
            # Show basic stats
            print(f"\n📊 Data Summary:")
            print(f"   Records: {len(df)}")
            print(f"   Unique compounds: {df['molecule_chembl_id'].nunique()}")
            print(f"   Activity types: {df['standard_type'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            print("Trying alternative save location...")
            
            # Try saving to Downloads folder
            downloads_path = Path.home() / "Downloads" / filename
            try:
                df.to_csv(downloads_path, index=False)
                print(f"📁 Data saved to Downloads: {downloads_path}")
                return df
            except Exception as e2:
                print(f"❌ Also failed to save to Downloads: {e2}")
                
                # Last resort - print first few rows
                print("\n📋 First 10 rows of data:")
                print(df.head(10).to_string())
                return df
    else:
        print("❌ No data downloaded")
        return None

if __name__ == "__main__":
    # Download data
    df = download_chembl_data("CHEMBL279", max_records=100)
    
    if df is not None:
        print(f"\n🎉 Download complete! You now have {len(df)} bioactivity records.")
        print("You can move this file to your data/raw directory manually if needed.")
    else:
        print("\n❌ Download failed. Check your internet connection and ChEMBL client installation.")