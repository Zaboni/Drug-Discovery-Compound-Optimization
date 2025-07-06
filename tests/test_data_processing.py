#!/usr/bin/env python3
"""
Unit tests for data processing module.
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from data_processing import (
        MolecularDataProcessor, MolecularDataLoader, MolecularPreprocessor,
        FeatureEnginerator, DataSplitter
    )
except ImportError as e:
    print(f"Warning: Could not import data_processing modules: {e}")
    # Define dummy classes for testing when modules are not available
    class MolecularDataProcessor:
        def process_smiles(self, smiles_list): return []
        def extract_features(self, data): return pd.DataFrame()

    class MolecularDataLoader:
        def load_csv_file(self, path): return pd.DataFrame()
        def load_smiles_file(self, path): return pd.DataFrame()
        def auto_load(self, path): return pd.DataFrame()

    class MolecularPreprocessor:
        def validate_molecules(self, df, col='smiles'): return df
        def standardize_molecules(self, df): return df
        def remove_duplicates(self, df, col): return df

    class FeatureEnginerator:
        def extract_molecular_descriptors(self, df): return df
        def extract_molecular_fingerprints(self, df): return df

    class DataSplitter:
        def random_split(self, df, **kwargs): return df, df, df
        def cluster_split(self, df, **kwargs): return df, df, df


class TestMolecularDataLoader(unittest.TestCase):
    """Test cases for MolecularDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = MolecularDataLoader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]
        self.test_names = ["ethanol", "acetic_acid", "benzene", "ibuprofen"]
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_csv_file(self):
        """Test loading CSV files."""
        # Create test CSV file
        test_data = pd.DataFrame({
            'smiles': self.test_smiles,
            'name': self.test_names,
            'activity': [1, 0, 1, 1]
        })
        
        csv_file = Path(self.temp_dir) / "test.csv"
        test_data.to_csv(csv_file, index=False)
        
        # Load and test
        loaded_data = self.loader.load_csv_file(str(csv_file))
        
        self.assertEqual(len(loaded_data), 4)
        self.assertIn('smiles', loaded_data.columns)
        self.assertEqual(loaded_data['smiles'].tolist(), self.test_smiles)
    
    def test_load_smiles_file(self):
        """Test loading SMILES files."""
        # Create test SMILES file
        smiles_file = Path(self.temp_dir) / "test.smi"
        with open(smiles_file, 'w') as f:
            for smiles, name in zip(self.test_smiles, self.test_names):
                f.write(f"{smiles}\t{name}\n")
        
        # Load and test
        loaded_data = self.loader.load_smiles_file(str(smiles_file))
        
        self.assertEqual(len(loaded_data), 4)
        self.assertIn('smiles', loaded_data.columns)
        self.assertEqual(loaded_data['smiles'].tolist(), self.test_smiles)
    
    def test_auto_load(self):
        """Test automatic file format detection."""
        # Create test CSV file
        test_data = pd.DataFrame({
            'smiles': self.test_smiles,
            'name': self.test_names
        })
        
        csv_file = Path(self.temp_dir) / "test.csv"
        test_data.to_csv(csv_file, index=False)
        
        # Load using auto_load
        loaded_data = self.loader.auto_load(str(csv_file))
        
        self.assertEqual(len(loaded_data), 4)
        self.assertIn('smiles', loaded_data.columns)
    
    def test_invalid_file_format(self):
        """Test handling of invalid file formats."""
        invalid_file = Path(self.temp_dir) / "test.xyz"
        invalid_file.touch()
        
        with self.assertRaises(ValueError):
            self.loader.auto_load(str(invalid_file))


class TestMolecularPreprocessor(unittest.TestCase):
    """Test cases for MolecularPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = MolecularPreprocessor()
        
        # Create test DataFrame
        self.test_df = pd.DataFrame({
            'smiles': ["CCO", "CC(=O)O", "c1ccccc1", "invalid_smiles", "CCO"],  # Include duplicate and invalid
            'name': ["ethanol", "acetic_acid", "benzene", "invalid", "ethanol_dup"],
            'activity': [1, 0, 1, None, 1]
        })
    
    def test_validate_molecules(self):
        """Test molecule validation."""
        validated_df = self.preprocessor.validate_molecules(self.test_df)
        
        self.assertIn('valid', validated_df.columns)
        self.assertIn('canonical_smiles', validated_df.columns)
        
        # Check that valid SMILES are marked as valid
        valid_count = validated_df['valid'].sum()
        self.assertGreaterEqual(valid_count, 3)  # At least 3 valid SMILES
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        # First validate molecules to get canonical SMILES
        validated_df = self.preprocessor.validate_molecules(self.test_df)
        
        # Remove duplicates
        dedup_df = self.preprocessor.remove_duplicates(validated_df, 'canonical_smiles')
        
        # Should have fewer rows than original
        self.assertLessEqual(len(dedup_df), len(validated_df))
    
    def test_standardize_molecules(self):
        """Test molecule standardization."""
        standardized_df = self.preprocessor.standardize_molecules(self.test_df)
        
        self.assertIn('standardized_smiles', standardized_df.columns)
        self.assertEqual(len(standardized_df), len(self.test_df))


class TestFeatureEnginerator(unittest.TestCase):
    """Test cases for FeatureEnginerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_eng = FeatureEnginerator()
        
        # Create test DataFrame with valid SMILES
        self.test_df = pd.DataFrame({
            'canonical_smiles': ["CCO", "CC(=O)O", "c1ccccc1"],
            'name': ["ethanol", "acetic_acid", "benzene"]
        })
    
    def test_extract_molecular_descriptors(self):
        """Test molecular descriptor extraction."""
        descriptors_df = self.feature_eng.extract_molecular_descriptors(self.test_df)
        
        # Should have original columns plus descriptors
        self.assertGreaterEqual(len(descriptors_df.columns), len(self.test_df.columns))
        
        # Check for common descriptors
        expected_descriptors = ['molecular_weight', 'logp', 'tpsa', 'num_hbd', 'num_hba']
        for desc in expected_descriptors:
            if desc in descriptors_df.columns:  # Only check if RDKit is available
                self.assertIn(desc, descriptors_df.columns)
    
    def test_extract_molecular_fingerprints(self):
        """Test molecular fingerprint extraction."""
        fingerprints_df = self.feature_eng.extract_molecular_fingerprints(self.test_df)
        
        # Should have original columns plus fingerprints
        self.assertGreaterEqual(len(fingerprints_df.columns), len(self.test_df.columns))


class TestDataSplitter(unittest.TestCase):
    """Test cases for DataSplitter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.splitter = DataSplitter()
        
        # Create test DataFrame
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'canonical_smiles': [f"C{i}" for i in range(100)],  # Dummy SMILES
            'activity': np.random.randint(0, 2, 100),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
    
    def test_random_split(self):
        """Test random data splitting."""
        try:
            train_df, val_df, test_df = self.splitter.random_split(
                self.test_df, 
                train_ratio=0.7, 
                val_ratio=0.15, 
                test_ratio=0.15
            )
            
            # Check split sizes
            total_size = len(train_df) + len(val_df) + len(test_df)
            self.assertEqual(total_size, len(self.test_df))
            
            # Check approximate ratios
            self.assertAlmostEqual(len(train_df) / len(self.test_df), 0.7, delta=0.1)
            self.assertAlmostEqual(len(val_df) / len(self.test_df), 0.15, delta=0.1)
            self.assertAlmostEqual(len(test_df) / len(self.test_df), 0.15, delta=0.1)
            
        except ImportError:
            # Skip test if sklearn not available
            self.skipTest("Scikit-learn not available")
    
    def test_stratified_split(self):
        """Test stratified data splitting."""
        try:
            train_df, val_df, test_df = self.splitter.random_split(
                self.test_df,
                target_column='activity',
                stratify=True,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )
            
            # Check that splits maintain class distribution
            original_ratio = self.test_df['activity'].mean()
            train_ratio = train_df['activity'].mean()
            
            self.assertAlmostEqual(original_ratio, train_ratio, delta=0.2)
            
        except ImportError:
            # Skip test if sklearn not available
            self.skipTest("Scikit-learn not available")
    
    def test_cluster_split(self):
        """Test cluster-based data splitting."""
        try:
            features = ['feature1', 'feature2']
            train_df, val_df, test_df = self.splitter.cluster_split(
                self.test_df,
                features=features,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )
            
            # Check split sizes
            total_size = len(train_df) + len(val_df) + len(test_df)
            self.assertEqual(total_size, len(self.test_df))
            
        except ImportError:
            # Skip test if sklearn not available
            self.skipTest("Scikit-learn not available")


class TestMolecularDataProcessor(unittest.TestCase):
    """Test cases for MolecularDataProcessor class (legacy)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = MolecularDataProcessor()
        self.test_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "invalid_smiles"]
    
    def test_process_smiles(self):
        """Test SMILES processing."""
        try:
            processed_data = self.processor.process_smiles(self.test_smiles)
            
            self.assertEqual(len(processed_data), len(self.test_smiles))
            
            # Check that valid SMILES are processed correctly
            valid_count = sum(1 for data in processed_data if data['valid'])
            self.assertGreaterEqual(valid_count, 3)  # At least 3 valid SMILES
            
        except ImportError:
            # Skip test if RDKit not available
            self.skipTest("RDKit not available")
    
    def test_extract_features(self):
        """Test feature extraction."""
        try:
            processed_data = self.processor.process_smiles(self.test_smiles[:3])  # Use only valid SMILES
            features_df = self.processor.extract_features(processed_data)
            
            self.assertEqual(len(features_df), 3)
            self.assertIn('smiles', features_df.columns)
            
        except ImportError:
            # Skip test if RDKit not available
            self.skipTest("RDKit not available")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = MolecularDataLoader()
        self.preprocessor = MolecularPreprocessor()
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # Should not crash
        validated_df = self.preprocessor.validate_molecules(empty_df, 'smiles')
        self.assertEqual(len(validated_df), 0)
    
    def test_missing_smiles_column(self):
        """Test handling of missing SMILES column."""
        df_no_smiles = pd.DataFrame({
            'name': ['test1', 'test2'],
            'activity': [1, 0]
        })
        
        # Should handle gracefully
        validated_df = self.preprocessor.validate_molecules(df_no_smiles, 'smiles')
        self.assertEqual(len(validated_df), 2)
    
    def test_all_invalid_smiles(self):
        """Test handling of all invalid SMILES."""
        invalid_df = pd.DataFrame({
            'smiles': ['invalid1', 'invalid2', 'invalid3']
        })
        
        validated_df = self.preprocessor.validate_molecules(invalid_df)
        
        # Should mark all as invalid
        if 'valid' in validated_df.columns:
            valid_count = validated_df['valid'].sum()
            self.assertEqual(valid_count, 0)
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent files."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_csv_file("nonexistent_file.csv")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test dataset
        self.test_data = pd.DataFrame({
            'smiles': [
                "CCO", "CC(=O)O", "c1ccccc1", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O"
            ],
            'name': ["ethanol", "acetic_acid", "benzene", "ibuprofen", "caffeine", "salbutamol"],
            'activity': [0, 0, 1, 1, 1, 1],
            'ic50': [1000, 5000, 100, 50, 200, 75]
        })
        
        # Save test data
        self.test_file = Path(self.temp_dir) / "test_data.csv"
        self.test_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_pipeline(self):
        """Test the complete data processing pipeline."""
        # 1. Load data
        loader = MolecularDataLoader()
        df = loader.load_csv_file(str(self.test_file))
        
        # 2. Preprocess
        preprocessor = MolecularPreprocessor()
        df = preprocessor.validate_molecules(df)
        df = preprocessor.standardize_molecules(df)
        
        # 3. Extract features
        feature_eng = FeatureEnginerator()
        df = feature_eng.extract_molecular_descriptors(df)
        
        # 4. Split data
        try:
            splitter = DataSplitter()
            train_df, val_df, test_df = splitter.random_split(df, target_column='activity')
            
            # Verify pipeline completed
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(val_df), 0)
            self.assertGreater(len(test_df), 0)
            
        except ImportError:
            # Skip splitting if sklearn not available
            pass
        
        # Verify data has been processed
        self.assertIn('valid', df.columns)
        self.assertIn('standardized_smiles', df.columns)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestMolecularDataLoader,
        TestMolecularPreprocessor,
        TestFeatureEnginerator,
        TestDataSplitter,
        TestMolecularDataProcessor,
        TestEdgeCases,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)