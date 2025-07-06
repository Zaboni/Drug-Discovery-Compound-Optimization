"""
Data Processing Package for Drug Discovery Compound Optimization

This package handles molecular data preprocessing, feature extraction,
and data preparation for machine learning models.
"""

# Import core functionality
from .core import (
    RDKIT_AVAILABLE, SKLEARN_AVAILABLE, MORGAN_GENERATOR_AVAILABLE,
    logger
)

# Import main classes
from .processor import MolecularDataProcessor
from .loader import MolecularDataLoader  
from .preprocessor import MolecularPreprocessor
from .feature_engineering import FeatureEnginerator
from .data_splitting import DataSplitter
from .splitting_strategies import AdvancedSplittingStrategies
from .advanced_features import AdvancedFeatureExtractor

__all__ = [
    # Core constants and utilities
    'RDKIT_AVAILABLE',
    'SKLEARN_AVAILABLE', 
    'MORGAN_GENERATOR_AVAILABLE',
    'logger',
    
    # Main classes
    'MolecularDataProcessor',
    'MolecularDataLoader',
    'MolecularPreprocessor',
    'FeatureEnginerator',
    'DataSplitter',
    'AdvancedSplittingStrategies',
    'AdvancedFeatureExtractor'
]