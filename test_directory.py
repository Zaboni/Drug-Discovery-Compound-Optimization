#!/usr/bin/env python3
"""Test script to verify directory creation and file saving."""

import os
from pathlib import Path
import pandas as pd

# Create test data
test_data = {
    'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'],
    'activity': [1, 0, 1]
}
df = pd.DataFrame(test_data)

# Create output directory
output_dir = Path('data/raw')
print(f"Creating directory: {output_dir.absolute()}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Directory exists: {output_dir.exists()}")

# Save test file
output_file = output_dir / 'test_file.csv'
print(f"Saving to: {output_file.absolute()}")

try:
    df.to_csv(str(output_file), index=False)
    print(f"✅ File saved successfully!")
    print(f"File exists: {output_file.exists()}")
    print(f"File size: {output_file.stat().st_size} bytes")
except Exception as e:
    print(f"❌ Error saving file: {e}")

# List directory contents
print(f"\nDirectory contents:")
for item in output_dir.iterdir():
    print(f"  {item.name}")