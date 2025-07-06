#!/usr/bin/env python3
"""
Test script to verify the Drug Discovery Compound Optimization installation.
"""

import sys


def test_imports():
    """Test all module imports."""
    print("🧪 Testing module imports...")
    
    # Test core packages
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Pandas/NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    # Test optional packages
    try:
        import rdkit
        print("✅ RDKit imported successfully")
    except ImportError:
        print("⚠️  RDKit not available - some functionality will be limited")
    
    try:
        import deepchem
        print("✅ DeepChem imported successfully")
    except ImportError:
        print("⚠️  DeepChem not available - some functionality will be limited")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError:
        print("⚠️  FastAPI not available - API functionality will be limited")
    
    return True

def test_modules():
    """Test our custom modules."""
    print("\n🔬 Testing custom modules...")
    
    try:
        from src.data_processing import MolecularDataProcessor
        print("✅ Data processing module imported successfully")
    except ImportError as e:
        print(f"❌ Data processing module import failed: {e}")
        return False
    
    try:
        from src.models import PropertyPredictor, RandomForestModel
        print("✅ Models module imported successfully")
    except ImportError as e:
        print(f"❌ Models module import failed: {e}")
        return False
    
    try:
        from src.training import Trainer
        print("✅ Training module imported successfully")
    except ImportError as e:
        print(f"❌ Training module import failed: {e}")
        return False
    
    try:
        from src.utils import validate_smiles, calculate_similarity
        print("✅ Utils module imported successfully")
    except ImportError as e:
        print(f"❌ Utils module import failed: {e}")
        return False
    
    try:
        from src.api import app
        print("✅ API module imported successfully")
    except ImportError as e:
        print(f"❌ API module import failed: {e}")
        return False
    
    return True

def test_functionality():
    """Test basic functionality."""
    print("\n⚙️  Testing basic functionality...")
    
    try:
        # Test data processing
        from src.data_processing import MolecularDataProcessor
        processor = MolecularDataProcessor()
        
        # Test SMILES processing
        test_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
        processed = processor.process_smiles(test_smiles)
        print(f"✅ Processed {len(processed)} SMILES successfully")
        
        # Test feature extraction (if RDKit available)
        try:
            features = processor.extract_features(processed)
            print(f"✅ Extracted features for {len(features)} molecules")
        except Exception:
            print("⚠️  Feature extraction limited (RDKit may not be available)")
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False
    
    try:
        # Test models
        from src.models import RandomForestModel
        import numpy as np
        
        # Create dummy data
        X_dummy = np.random.randn(100, 10)
        y_dummy = np.random.randn(100)
        
        # Test Random Forest
        rf_model = RandomForestModel(task_type="regression", n_estimators=10)
        rf_model.train(X_dummy, y_dummy)
        predictions = rf_model.predict(X_dummy[:5])
        print(f"✅ Random Forest model trained and tested successfully")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    try:
        # Test utilities
        from src.utils import validate_smiles, calculate_molecular_descriptors
        
        # Test SMILES validation
        is_valid = validate_smiles("CCO")
        print(f"✅ SMILES validation test: CCO is {'valid' if is_valid else 'invalid'}")
        
        # Test descriptor calculation (if RDKit available)
        try:
            descriptors = calculate_molecular_descriptors("CCO")
            if descriptors:
                print(f"✅ Calculated {len(descriptors)} molecular descriptors")
            else:
                print("⚠️  Descriptor calculation limited (RDKit may not be available)")
        except Exception:
            print("⚠️  Descriptor calculation limited (RDKit may not be available)")
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("🧬 Drug Discovery Compound Optimization - Installation Test")
    print("=" * 60)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test modules
    if not test_modules():
        success = False
    
    # Test functionality
    if not test_functionality():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Installation is successful.")
        print("\n🚀 You can now:")
        print("   - Run individual modules: python src/data_processing.py")
        print("   - Start the API server: python -m uvicorn src.api:app --reload")
        print("   - Launch Jupyter Lab: jupyter lab")
        print("   - Explore the notebooks in the notebooks/ directory")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("   You may need to install missing dependencies.")
    
    print("\n🧬 Happy drug discovery! 🧬")
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)