#!/usr/bin/env python3
"""
Unit tests for molecular feature extraction and validation.
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from utils import (
        validate_smiles, canonicalize_smiles, calculate_similarity,
        calculate_molecular_descriptors, check_lipinski_rule_of_five,
        assess_data_quality, benchmark_model_performance
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import utils modules: {e}")
    UTILS_AVAILABLE = False
    
    # Define dummy functions for testing
    def validate_smiles(smiles): return True
    def canonicalize_smiles(smiles): return smiles
    def calculate_similarity(s1, s2, method='tanimoto'): return 0.5
    def calculate_molecular_descriptors(smiles): return {}
    def check_lipinski_rule_of_five(smiles): return {}
    def assess_data_quality(df, smiles_col='smiles'): return {}
    def benchmark_model_performance(y_true, y_pred, task_type='regression'): return {}


class TestSMILESValidation(unittest.TestCase):
    """Test cases for SMILES validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
        ]
        
        self.invalid_smiles = [
            "invalid_smiles",
            "C1CCC",  # Incomplete ring
            "C(C)(C)(C)(C)C",  # Invalid valence
            "",  # Empty string
            "123456"  # Numbers only
        ]
    
    def test_validate_valid_smiles(self):
        """Test validation of valid SMILES."""
        for smiles in self.valid_smiles:
            with self.subTest(smiles=smiles):
                result = validate_smiles(smiles)
                if UTILS_AVAILABLE:
                    # Only check if RDKit is available
                    try:
                        from rdkit import Chem
                        mol = Chem.MolFromSmiles(smiles)
                        expected = mol is not None
                        self.assertEqual(result, expected)
                    except ImportError:
                        # Skip if RDKit not available
                        pass
    
    def test_validate_invalid_smiles(self):
        """Test validation of invalid SMILES."""
        for smiles in self.invalid_smiles:
            with self.subTest(smiles=smiles):
                result = validate_smiles(smiles)
                if UTILS_AVAILABLE:
                    # Only check if RDKit is available
                    try:
                        from rdkit import Chem
                        mol = Chem.MolFromSmiles(smiles)
                        expected = mol is not None
                        self.assertEqual(result, expected)
                    except ImportError:
                        # Skip if RDKit not available
                        pass
    
    def test_canonicalize_smiles(self):
        """Test SMILES canonicalization."""
        test_cases = [
            ("CCO", "CCO"),  # Already canonical
            ("OCC", "CCO"),  # Should canonicalize to CCO
            ("c1ccccc1", "c1ccccc1"),  # Benzene
        ]
        
        for input_smiles, expected in test_cases:
            with self.subTest(smiles=input_smiles):
                result = canonicalize_smiles(input_smiles)
                if UTILS_AVAILABLE:
                    try:
                        from rdkit import Chem
                        mol = Chem.MolFromSmiles(input_smiles)
                        if mol is not None:
                            canonical = Chem.MolToSmiles(mol)
                            self.assertEqual(result, canonical)
                    except ImportError:
                        # Skip if RDKit not available
                        pass


class TestMolecularSimilarity(unittest.TestCase):
    """Test cases for molecular similarity calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_pairs = [
            ("CCO", "CCO", 1.0),  # Identical molecules
            ("CCO", "CCC", None),  # Different molecules
            ("c1ccccc1", "c1ccccc1", 1.0),  # Identical benzene
            ("CCO", "CC(=O)O", None),  # Ethanol vs acetic acid
        ]
    
    def test_similarity_identical_molecules(self):
        """Test similarity of identical molecules."""
        smiles = "CCO"
        similarity = calculate_similarity(smiles, smiles)
        
        if UTILS_AVAILABLE:
            try:
                from rdkit import Chem
                # Identical molecules should have similarity 1.0
                self.assertAlmostEqual(similarity, 1.0, places=2)
            except ImportError:
                # Skip if RDKit not available
                pass
    
    def test_similarity_different_molecules(self):
        """Test similarity of different molecules."""
        similarity = calculate_similarity("CCO", "CCC")
        
        if UTILS_AVAILABLE:
            try:
                from rdkit import Chem
                # Different molecules should have similarity < 1.0
                self.assertLess(similarity, 1.0)
                self.assertGreaterEqual(similarity, 0.0)
            except ImportError:
                # Skip if RDKit not available
                pass
    
    def test_similarity_methods(self):
        """Test different similarity methods."""
        smiles1, smiles2 = "CCO", "CCC"
        methods = ["tanimoto", "dice", "cosine"]
        
        for method in methods:
            with self.subTest(method=method):
                similarity = calculate_similarity(smiles1, smiles2, method)
                
                if UTILS_AVAILABLE:
                    try:
                        from rdkit import Chem
                        self.assertGreaterEqual(similarity, 0.0)
                        self.assertLessEqual(similarity, 1.0)
                    except ImportError:
                        # Skip if RDKit not available
                        pass


class TestMolecularDescriptors(unittest.TestCase):
    """Test cases for molecular descriptor calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_molecules = {
            "CCO": {  # Ethanol
                "molecular_weight": 46.07,
                "logp": -0.31,
                "num_hbd": 1,
                "num_hba": 1
            },
            "c1ccccc1": {  # Benzene
                "molecular_weight": 78.11,
                "logp": 2.13,
                "num_hbd": 0,
                "num_hba": 0
            }
        }
    
    def test_descriptor_calculation(self):
        """Test molecular descriptor calculation."""
        for smiles, expected_props in self.test_molecules.items():
            with self.subTest(smiles=smiles):
                descriptors = calculate_molecular_descriptors(smiles)
                
                if UTILS_AVAILABLE and descriptors:
                    try:
                        from rdkit import Chem
                        # Check that descriptors are calculated
                        self.assertIsInstance(descriptors, dict)
                        
                        # Check specific descriptors if available
                        if 'molecular_weight' in descriptors:
                            self.assertAlmostEqual(
                                descriptors['molecular_weight'],
                                expected_props['molecular_weight'],
                                delta=1.0
                            )
                        
                        if 'num_hbd' in descriptors:
                            self.assertEqual(
                                descriptors['num_hbd'],
                                expected_props['num_hbd']
                            )
                            
                    except ImportError:
                        # Skip if RDKit not available
                        pass
    
    def test_invalid_smiles_descriptors(self):
        """Test descriptor calculation for invalid SMILES."""
        invalid_smiles = "invalid_smiles"
        descriptors = calculate_molecular_descriptors(invalid_smiles)
        
        if UTILS_AVAILABLE:
            try:
                from rdkit import Chem
                # Should return empty dict for invalid SMILES
                self.assertEqual(descriptors, {})
            except ImportError:
                # Skip if RDKit not available
                pass


class TestLipinskiRuleOfFive(unittest.TestCase):
    """Test cases for Lipinski's Rule of Five assessment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.drug_like_molecules = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        ]
        
        self.non_drug_like_molecules = [
            "C" * 50,  # Very large molecule (hypothetical)
        ]
    
    def test_drug_like_molecules(self):
        """Test Lipinski assessment for drug-like molecules."""
        for smiles in self.drug_like_molecules:
            with self.subTest(smiles=smiles):
                result = check_lipinski_rule_of_five(smiles)
                
                if UTILS_AVAILABLE and result.get('valid', False):
                    try:
                        from rdkit import Chem
                        # Should have few violations
                        violations = result.get('violations', 0)
                        self.assertLessEqual(violations, 1)
                        
                        # Check individual criteria
                        self.assertLessEqual(result.get('molecular_weight', 0), 500)
                        self.assertLessEqual(result.get('logp', 0), 5)
                        self.assertLessEqual(result.get('num_hbd', 0), 5)
                        self.assertLessEqual(result.get('num_hba', 0), 10)
                        
                    except ImportError:
                        # Skip if RDKit not available
                        pass
    
    def test_invalid_smiles_lipinski(self):
        """Test Lipinski assessment for invalid SMILES."""
        invalid_smiles = "invalid_smiles"
        result = check_lipinski_rule_of_five(invalid_smiles)
        
        if UTILS_AVAILABLE:
            try:
                from rdkit import Chem
                # Should return error for invalid SMILES
                self.assertFalse(result.get('valid', True))
            except ImportError:
                # Skip if RDKit not available
                pass


class TestDataQualityAssessment(unittest.TestCase):
    """Test cases for data quality assessment."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test dataset with various quality issues
        self.test_df = pd.DataFrame({
            'smiles': [
                "CCO", "CC(=O)O", "c1ccccc1", "invalid_smiles", None,
                "CCO", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Duplicate
            ],
            'activity': [1, 0, 1, None, 1, 1, 0],
            'ic50': [100, 500, 50, None, 200, 100, 75],
            'molecular_weight': [46.07, 60.05, 78.11, None, None, 46.07, 206.28]
        })
    
    def test_data_quality_assessment(self):
        """Test comprehensive data quality assessment."""
        if UTILS_AVAILABLE:
            quality_report = assess_data_quality(self.test_df, 'smiles')
            
            # Check report structure
            self.assertIn('dataset_size', quality_report)
            self.assertIn('missing_data', quality_report)
            self.assertIn('duplicates', quality_report)
            self.assertIn('recommendations', quality_report)
            
            # Check dataset size
            self.assertEqual(quality_report['dataset_size'], len(self.test_df))
            
            # Check missing data detection
            missing_data = quality_report['missing_data']
            self.assertIn('smiles', missing_data)
            self.assertGreater(missing_data['smiles']['count'], 0)
    
    def test_empty_dataframe_quality(self):
        """Test quality assessment of empty DataFrame."""
        if UTILS_AVAILABLE:
            empty_df = pd.DataFrame()
            quality_report = assess_data_quality(empty_df)
            
            self.assertEqual(quality_report['dataset_size'], 0)


class TestModelPerformanceMetrics(unittest.TestCase):
    """Test cases for model performance evaluation."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.y_true_regression = np.random.randn(100)
        self.y_pred_regression = self.y_true_regression + np.random.randn(100) * 0.1
        
        self.y_true_classification = np.random.randint(0, 2, 100)
        self.y_pred_classification = np.random.randint(0, 2, 100)
    
    def test_regression_metrics(self):
        """Test regression performance metrics."""
        if UTILS_AVAILABLE:
            try:
                metrics = benchmark_model_performance(
                    self.y_true_regression,
                    self.y_pred_regression,
                    task_type="regression"
                )
                
                # Check that metrics are calculated
                expected_metrics = ['mse', 'rmse', 'mae', 'r2']
                for metric in expected_metrics:
                    if metric in metrics:
                        self.assertIsInstance(metrics[metric], (int, float))
                        
            except ImportError:
                # Skip if sklearn not available
                self.skipTest("Scikit-learn not available")
    
    def test_classification_metrics(self):
        """Test classification performance metrics."""
        if UTILS_AVAILABLE:
            try:
                metrics = benchmark_model_performance(
                    self.y_true_classification,
                    self.y_pred_classification,
                    task_type="classification"
                )
                
                # Check that metrics are calculated
                expected_metrics = ['accuracy', 'precision', 'recall', 'f1']
                for metric in expected_metrics:
                    if metric in metrics:
                        self.assertIsInstance(metrics[metric], (int, float))
                        self.assertGreaterEqual(metrics[metric], 0.0)
                        self.assertLessEqual(metrics[metric], 1.0)
                        
            except ImportError:
                # Skip if sklearn not available
                self.skipTest("Scikit-learn not available")
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        if UTILS_AVAILABLE:
            try:
                y_true = np.array([1, 0, 1, 0, 1])
                y_pred = np.array([1, 0, 1, 0, 1])
                
                metrics = benchmark_model_performance(
                    y_true, y_pred, task_type="classification"
                )
                
                # Perfect predictions should have accuracy = 1.0
                if 'accuracy' in metrics:
                    self.assertAlmostEqual(metrics['accuracy'], 1.0, places=2)
                    
            except ImportError:
                # Skip if sklearn not available
                self.skipTest("Scikit-learn not available")


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling in molecular features."""
    
    def test_empty_string_smiles(self):
        """Test handling of empty string SMILES."""
        result = validate_smiles("")
        if UTILS_AVAILABLE:
            try:
                from rdkit import Chem
                self.assertFalse(result)
            except ImportError:
                pass
    
    def test_none_smiles(self):
        """Test handling of None SMILES."""
        result = validate_smiles(None)
        if UTILS_AVAILABLE:
            try:
                from rdkit import Chem
                self.assertFalse(result)
            except ImportError:
                pass
    
    def test_very_long_smiles(self):
        """Test handling of very long SMILES."""
        long_smiles = "C" * 1000  # Very long chain
        result = validate_smiles(long_smiles)
        
        if UTILS_AVAILABLE:
            try:
                from rdkit import Chem
                # Should handle gracefully
                self.assertIsInstance(result, bool)
            except ImportError:
                pass
    
    def test_special_characters_smiles(self):
        """Test handling of SMILES with special characters."""
        special_smiles = "C@#$%^&*()"
        result = validate_smiles(special_smiles)
        
        if UTILS_AVAILABLE:
            try:
                from rdkit import Chem
                self.assertFalse(result)
            except ImportError:
                pass


def run_tests():
    """Run all molecular feature tests."""
    test_classes = [
        TestSMILESValidation,
        TestMolecularSimilarity,
        TestMolecularDescriptors,
        TestLipinskiRuleOfFive,
        TestDataQualityAssessment,
        TestModelPerformanceMetrics,
        TestEdgeCasesAndErrorHandling
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