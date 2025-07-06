#!/usr/bin/env python3
"""
Manual PubChem Data Download Script
Downloads data and saves to current directory to avoid permission issues.
"""

import pandas as pd
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pubchempy as pcp
    PUBCHEMPY_AVAILABLE = True
except ImportError:
    pcp = None
    PUBCHEMPY_AVAILABLE = False
    logger.error("PubChemPy not available. Install with: pip install pubchempy")

def download_compounds_by_name(compound_names, max_compounds=50):
    """Download compound data by names."""
    
    if not PUBCHEMPY_AVAILABLE:
        print("❌ PubChemPy not available")
        return None
    
    print(f"🔍 Downloading data for {len(compound_names)} compounds...")
    
    compounds_data = []
    for i, name in enumerate(compound_names[:max_compounds]):
        try:
            print(f"  [{i+1}/{len(compound_names)}] Downloading: {name}")
            
            # Get compound by name
            compounds = pcp.get_compounds(name, 'name')
            
            if compounds:
                compound = compounds[0]  # Take first match
                
                compound_data = {
                    'name': name,
                    'cid': compound.cid,
                    'canonical_smiles': compound.canonical_smiles,
                    'molecular_formula': compound.molecular_formula,
                    'molecular_weight': compound.molecular_weight,
                    'xlogp': compound.xlogp,
                    'tpsa': compound.tpsa,
                    'hbd': compound.h_bond_donor_count,
                    'hba': compound.h_bond_acceptor_count,
                    'rotatable_bonds': compound.rotatable_bond_count,
                    'heavy_atom_count': compound.heavy_atom_count,
                    'complexity': compound.complexity
                }
                
                compounds_data.append(compound_data)
                print(f"    ✅ Found CID: {compound.cid}")
                
            else:
                print(f"    ❌ No compound found for: {name}")
                
            time.sleep(0.2)  # Be nice to the API
            
        except Exception as e:
            print(f"    ❌ Error downloading {name}: {e}")
    
    if compounds_data:
        df = pd.DataFrame(compounds_data)
        print(f"✅ Downloaded data for {len(df)} compounds")
        return df
    else:
        print("❌ No compound data downloaded")
        return None

def download_compounds_by_cid(cids, max_compounds=50):
    """Download compound data by CIDs."""
    
    if not PUBCHEMPY_AVAILABLE:
        print("❌ PubChemPy not available")
        return None
    
    print(f"🔍 Downloading data for {len(cids)} CIDs...")
    
    compounds_data = []
    for i, cid in enumerate(cids[:max_compounds]):
        try:
            print(f"  [{i+1}/{len(cids)}] Downloading CID: {cid}")
            
            compound = pcp.Compound.from_cid(cid)
            
            compound_data = {
                'cid': compound.cid,
                'canonical_smiles': compound.canonical_smiles,
                'molecular_formula': compound.molecular_formula,
                'molecular_weight': compound.molecular_weight,
                'xlogp': compound.xlogp,
                'tpsa': compound.tpsa,
                'hbd': compound.h_bond_donor_count,
                'hba': compound.h_bond_acceptor_count,
                'rotatable_bonds': compound.rotatable_bond_count,
                'heavy_atom_count': compound.heavy_atom_count,
                'complexity': compound.complexity
            }
            
            compounds_data.append(compound_data)
            print(f"    ✅ Downloaded: {compound.canonical_smiles}")
            
            time.sleep(0.2)  # Be nice to the API
            
        except Exception as e:
            print(f"    ❌ Error downloading CID {cid}: {e}")
    
    if compounds_data:
        df = pd.DataFrame(compounds_data)
        print(f"✅ Downloaded data for {len(df)} compounds")
        return df
    else:
        print("❌ No compound data downloaded")
        return None

def save_data(df, filename):
    """Save data to file with fallback options."""
    
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
            
            # Last resort - show data
            print("\n📋 First 10 rows of data:")
            print(df.head(10).to_string())
            return False

if __name__ == "__main__":
    print("🧪 Manual PubChem Download Script")
    print("=" * 40)
    
    # Example 1: Download by compound names
    print("\n1️⃣ Downloading compounds by name...")
    compound_names = [
        "aspirin", "caffeine", "ibuprofen", "acetaminophen", 
        "morphine", "codeine", "penicillin", "insulin"
    ]
    
    df_names = download_compounds_by_name(compound_names, max_compounds=5)
    if df_names is not None:
        save_data(df_names, "pubchem_compounds_by_name.csv")
        print(f"\n📊 Summary by Names:")
        print(f"   Records: {len(df_names)}")
        try:
            avg_mw = pd.to_numeric(df_names['molecular_weight'], errors='coerce').mean()
            print(f"   Average MW: {avg_mw:.1f}")
        except:
            print(f"   Molecular weights: {df_names['molecular_weight'].tolist()}")

    print("\n" + "="*40)

    # Example 2: Download by CIDs
    print("\n2️⃣ Downloading compounds by CID...")
    cids = [2244, 2519, 3672, 1983, 5288826, 5362129, 6323497, 16129778]

    df_cids = download_compounds_by_cid(cids, max_compounds=5)
    if df_cids is not None:
        save_data(df_cids, "pubchem_compounds_by_cid.csv")
        print(f"\n📊 Summary by CIDs:")
        print(f"   Records: {len(df_cids)}")
        try:
            avg_mw = pd.to_numeric(df_cids['molecular_weight'], errors='coerce').mean()
            print(f"   Average MW: {avg_mw:.1f}")
        except:
            print(f"   Molecular weights: {df_cids['molecular_weight'].tolist()}")
    
    print("\n🎉 Manual download complete!")
    print("You can move these files to your data/raw directory manually if needed.")