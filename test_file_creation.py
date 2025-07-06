#!/usr/bin/env python3
"""Test file creation in data/raw directory."""

import pandas as pd
from pathlib import Path

# Create test data
df = pd.DataFrame({
    'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2'],
    'canonical_smiles': ['CCO', 'CC(=O)O'],
    'standard_value': [100, 200]
})

# Set up paths exactly like ChEMBL script
output_dir = Path('data/raw')
output_path = output_dir / 'test_chembl.csv'
output_path_str = str(output_path.absolute())

print(f"Output directory: {output_dir}")
print(f"Output directory absolute: {output_dir.absolute()}")
print(f"Directory exists: {output_dir.exists()}")
print(f"Saving to: {output_path_str}")

try:
    # Use the same approach as ChEMBL script
    with open(output_path_str, 'w', encoding='utf-8', newline='') as f:
        df.to_csv(f, index=False, lineterminator='\n')
    print("✅ File created successfully!")
    print(f"File exists: {output_path.exists()}")
    print(f"File size: {output_path.stat().st_size} bytes")
except Exception as e:
    print(f"❌ Error: {e}")