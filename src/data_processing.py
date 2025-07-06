"""
Data Processing Module for Drug Discovery Compound Optimization

This module provides backward compatibility by importing from the new
data_processing package structure.
"""

# Import from the new package structure for backward compatibility
from .data_processing import (
    # Core constants and utilities
    RDKIT_AVAILABLE,
    SKLEARN_AVAILABLE,
    MORGAN_GENERATOR_AVAILABLE,
    logger,

    # Main classes
    MolecularDataProcessor,
    MolecularDataLoader,
    MolecularPreprocessor,
    FeatureEnginerator,
    DataSplitter
)

# For backward compatibility, also make available at module level
__all__ = [
    'RDKIT_AVAILABLE',
    'SKLEARN_AVAILABLE',
    'MORGAN_GENERATOR_AVAILABLE',
    'logger',
    'MolecularDataProcessor',
    'MolecularDataLoader',
    'MolecularPreprocessor',
    'FeatureEnginerator',
    'DataSplitter'
]


def main():
    """Example usage - imports from new package structure."""
    from .data_processing.processor import main as processor_main
    processor_main()


if __name__ == "__main__":
    main()